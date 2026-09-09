"""
Execute a Gap-Scan RB experiment on a local simulator or on real IBM hardware.

Same code path for both, so the hardware run is exercised end to end offline before it
ever consumes free QPU minutes:

    # local, noiseless
    .venv\\Scripts\\python.exe -m hardware.run_gapscan --sim

    # local, with correlated (quasi-static) dephasing injected
    .venv\\Scripts\\python.exe -m hardware.run_gapscan --sim --noise correlated

    # dry run against a real backend: builds and transpiles, submits NOTHING
    .venv\\Scripts\\python.exe -m hardware.run_gapscan --backend ibm_brisbane --dry-run

    # the real thing
    .venv\\Scripts\\python.exe -m hardware.run_gapscan --backend ibm_brisbane --shots 1024

Outputs (under data/):
    <run>_survivals.npz   per-sequence survival probabilities, indexed [delay][dd][length]
    <run>_meta.json       full configuration, realized delays, and calibration snapshot

The saved arrays are per-sequence, not averaged, because the estimator's bootstrap
resamples whole sequences; averaging at save time would throw away the information the
finite-sample confidence intervals are built from.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing hardware.backend installs the OS trust store (see its module docstring);
# do it before any IBM SDK call so TLS verification uses the right anchors.
import hardware.backend as _hw_backend  # noqa: E402,F401
from hardware.circuits import build_circuit_set, quantize_delay  # noqa: E402

DEFAULT_LENGTHS = [2, 6, 10, 14, 18, 22, 26, 30]
DEFAULT_DELAYS_NS = [40, 80, 160, 320, 640, 1280]


def _survival_from_counts(counts: dict, shots: int) -> float:
    """P(measure 0) — the ideal outcome, since the circuit composes to the identity."""
    zeros = 0
    for k, v in counts.items():
        key = k.replace(" ", "")
        if key and key[-1] == "0":     # little-endian: qubit 0 is the last character
            zeros += v
    return zeros / shots



def _load_fake_backend(name: str):
    """Load a fake IBM backend by short name (e.g. 'brisbane', 'sherbrooke', 'torino')."""
    from qiskit_ibm_runtime import fake_provider as fp

    short = name.lower().replace("ibm_", "").replace("fake", "").strip("_")
    for attr in dir(fp):
        if attr.startswith("Fake") and attr.lower() == f"fake{short}":
            return getattr(fp, attr)()
    # Fall back to any Fake* backend whose name contains the request.
    for attr in dir(fp):
        if attr.startswith("Fake") and short in attr.lower():
            return getattr(fp, attr)()
    available = sorted(a[4:].lower() for a in dir(fp) if a.startswith("Fake"))
    raise ValueError(f"unknown fake backend {name!r}; available: {available}")


def build_noise_model(kind: str):
    """Local noise model for validating the pipeline offline.

    'markovian'  memoryless dephasing during the idle -> expect nu ~ 1
    'correlated' a quasi-static Z rotation, constant within a circuit -> expect nu ~ 2

    NOTE: Aer applies a fixed channel per instruction, so 'correlated' here is a coherent
    per-run detuning rather than true run-to-run stochastic memory. It is enough to prove
    the plumbing and the delta-scaling readout; the physics-grade correlated model lives in
    sim/noise.py, and the excess-variance statistic W_var is what distinguishes the two.
    """
    from qiskit_aer.noise import NoiseModel, coherent_unitary_error, phase_damping_error

    nm = NoiseModel()
    if kind == "none":
        return None
    if kind == "markovian":
        # Dephasing attached to the delay instruction: error grows with idle time.
        nm.add_all_qubit_quantum_error(phase_damping_error(0.02), ["delay"])
        return nm
    if kind == "correlated":
        theta = 0.06
        u = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
        nm.add_all_qubit_quantum_error(coherent_unitary_error(u), ["delay"])
        return nm
    raise ValueError("noise must be none|markovian|correlated")



def _counts_from_job_result(res) -> list:
    """Normalize a SamplerV2 result into a list of counts dicts (API varies by version)."""
    out = []
    for r in res:
        got = None
        try:
            data = r.data
            creg = None
            if hasattr(data, "values"):
                try:
                    creg = next(iter(data.values()))
                except Exception:
                    creg = None
            if creg is None and hasattr(data, "__dict__"):
                creg = next(iter(data.__dict__.values()))
            got = creg.get_counts()
        except Exception:
            got = None
        if got is None:
            got = r.join_data().get_counts()
        out.append(got)
    return out


def _report_witness(survivals, lengths, meta_out, meta_path, seed=0):
    """Run the witness on the 'none' DD arm and persist the readout beside the data."""
    try:
        from estimator.witness import classify, gap_scan_witness

        from estimator.witness import excess_variance_repeats

        datasets = []
        rep_stats = []
        for key in sorted(survivals.keys()):
            dd, dstr = key.split("|")
            if dd != "none":
                continue
            arr = survivals[key]
            flat = arr.reshape(arr.shape[0], -1) if arr.ndim == 3 else arr
            datasets.append({"lengths": np.array(lengths), "survivals": flat,
                             "delta": float(dstr) * 1e6, "tau_gate": 0.05})
            if arr.ndim == 3 and arr.shape[2] >= 2:
                rep_stats.append(excess_variance_repeats(
                    arr, meta_out.get("shots", 1024))["w_var_rep"])
        if rep_stats:
            wv = float(np.nanmedian(rep_stats))
            print(f"\nW_var^rep (same-sequence temporal repeats) = {wv:.2f}")
            print("  ~1 means no run-to-run memory beyond shot noise; >>1 means real "
                  "temporal correlation")
            meta_out["w_var_rep"] = wv
        if len(datasets) >= 2:
            w = gap_scan_witness(datasets, n_boot=200,
                                 rng=np.random.default_rng(seed + 1))
            c = classify(w)
            print(f"\nwitness: nu = {w['nu']:.4f} "
                  f"CI [{w['nu_ci'][0]:.4f}, {w['nu_ci'][1]:.4f}], "
                  f"W_var = {w['w_var']:.2f}")
            print(f"classification: {c['label']}")
            for r in c["reasons"]:
                print(f"  - {r}")
            meta_out["witness"] = {
                "nu": w["nu"], "nu_ci": list(w["nu_ci"]), "w_var": w["w_var"],
                "label": c["label"], "epcs": [float(x) for x in w["epcs"]],
            }
        else:
            print(f"\n[note] only {len(datasets)} delay point(s) in the 'none' arm; "
                  "need >= 2 for a nu fit")
    except Exception as e:
        print(f"[warn] witness failed: {e}")
    meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")


def fetch(job_id: str, outdir: str = "data") -> dict:
    """Retrieve a previously submitted job by id and build the dataset from its counts."""
    import hardware.backend as hb

    pend = None
    pend_file = None
    for f in sorted(Path(outdir).glob("*_pending.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("job_id") == job_id:
            pend, pend_file = d, f
            break
    if pend is None:
        raise FileNotFoundError(
            f"no *_pending.json in {outdir}/ records job_id {job_id}; without it the "
            "mapping from circuit order back to (delay, dd, length, sequence) is lost"
        )

    svc = hb.get_service()
    job = svc.job(job_id)
    st = str(job.status())
    print(f"job {job_id}: status = {st}")
    if "DONE" not in st.upper():
        print("not finished yet - try again later.")
        return {"job_id": job_id, "status": st, "done": False}

    counts_list = _counts_from_job_result(job.result())
    lengths = pend["lengths"]
    n_sequences = pend["n_sequences"]
    n_repeats = pend.get("n_repeats", 1)
    shots = pend["shots"]

    survivals = {}
    for mt, counts in zip(pend["circuit_meta"], counts_list):
        key = f"{mt['dd']}|{mt['delay']:.12g}"
        if key not in survivals:
            survivals[key] = np.full((len(lengths), n_sequences, n_repeats),
                                     np.nan, dtype=float)
        survivals[key][lengths.index(mt["m"]), mt["seq_index"],
                       mt.get("repeat_index", 0)] = _survival_from_counts(counts, shots)

    out = Path(outdir)
    run_id = pend_file.name.replace("_pending.json", "")
    np.savez_compressed(out / f"{run_id}_survivals.npz",
                        lengths=np.array(lengths), **survivals)
    meta_out = {k: v for k, v in pend.items() if k != "circuit_meta"}
    meta_out["keys"] = sorted(survivals.keys())
    meta_out["fetched"] = True
    _report_witness(survivals, lengths, meta_out, out / f"{run_id}_meta.json",
                    seed=pend.get("seed", 0))
    print(f"saved {run_id}_survivals.npz")
    return meta_out


def run(
    lengths=DEFAULT_LENGTHS,
    delays_ns=DEFAULT_DELAYS_NS,
    n_sequences: int = 30,
    n_repeats: int = 1,
    shots: int = 1024,
    dd_conditions=("none",),
    qubit: int = 0,
    seed: int = 20260729,
    backend_name: str | None = None,
    simulator: bool = True,
    noise: str = "none",
    fake: str | None = None,
    dry_run: bool = False,
    no_wait: bool = False,
    optimization_level: int = 1,
    outdir: str = "data",
    tag: str = "",
) -> dict:
    """Build, execute, and save one gap-scan experiment."""
    from qiskit import transpile

    rng = np.random.default_rng(seed)
    lengths = list(lengths)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ---------------------------------------------------------------- backend
    calibration = None
    if fake:
        # Fake IBM backends carry real dt, T1/T2 and duration-dependent thermal
        # relaxation, so this exercises the true hardware code path offline AND gives a
        # physically meaningful null: memoryless relaxation must yield nu ~ 1.
        from qiskit_aer import AerSimulator

        from hardware.backend import backend_dt, backend_timing, calibration_snapshot

        fake_backend = _load_fake_backend(fake)
        backend = AerSimulator.from_backend(fake_backend)
        dt = backend_dt(fake_backend)
        timing = backend_timing(fake_backend)
        granularity = timing["granularity"]
        calibration = calibration_snapshot(fake_backend, qubit)
        label = f"fake_{getattr(fake_backend, 'name', fake)}"
        print(f"fake backend: {label} (dt={dt}) | "
              f"T1={calibration.get('T1_s')} T2={calibration.get('T2_s')}")
    elif simulator:
        from qiskit_aer import AerSimulator

        nm = build_noise_model(noise)
        backend = AerSimulator(noise_model=nm)
        dt = None
        granularity = 1
        label = f"aer_{noise}"
    else:
        from hardware.backend import (
            append_calibration_log,
            backend_dt,
            backend_timing,
            calibration_snapshot,
            get_service,
            pick_backend,
        )

        service = get_service()
        backend = pick_backend(service, name=backend_name)
        dt = backend_dt(backend)
        timing = backend_timing(backend)
        granularity = timing["granularity"]
        calibration = calibration_snapshot(backend, qubit)
        print(f"timing constraints: {timing}")
        append_calibration_log(calibration, Path(outdir) / "calibration_log.jsonl")
        label = backend.name
        print(f"backend: {label} (dt={dt}) | calibration logged")

    # ------------------------------------------------- delays -> real hardware grid
    realized = {}
    delays_s = []
    for ns in delays_ns:
        req = ns * 1e-9
        _, actual = quantize_delay(req, dt, granularity=granularity)
        realized[ns] = actual
        delays_s.append(actual)
    print("requested vs realized idle gaps (ns): " + ", ".join(
        f"{ns}->{realized[ns]*1e9:.2f}" for ns in delays_ns))
    n_distinct = len(set(round(v, 15) for v in realized.values()))
    if n_distinct < len(delays_ns):
        print(f"  !! WARNING: only {n_distinct} DISTINCT delays out of {len(delays_ns)} "
              f"requested. Duplicated x-points bias the log-log nu fit -- choose gaps that "
              f"are multiples of granularity*dt = {granularity * (dt or 0) * 1e9:.1f} ns.")

    # ---------------------------------------------------------------- circuits
    circuits, meta = build_circuit_set(
        lengths, delays_s, n_sequences, rng,
        dd_conditions=tuple(dd_conditions), unit="s", qubit=qubit, dt=dt,
        granularity=granularity, n_repeats=n_repeats,
    )
    print(f"built {len(circuits)} circuits "
          f"({len(lengths)} lengths x {len(delays_s)} delays x "
          f"{len(dd_conditions)} DD x {n_sequences} sequences x {n_repeats} repeats)")

    tqc = transpile(circuits, backend=backend, optimization_level=optimization_level,
                    initial_layout=[qubit] if (not simulator or fake) else None)
    n_delays = [c.count_ops().get("delay", 0) for c in tqc]
    print(f"after transpile: delay instructions per circuit min={min(n_delays)}, "
          f"max={max(n_delays)} (must be >= sequence length)")

    if dry_run:
        print("DRY RUN — nothing submitted.")
        return {"dry_run": True, "n_circuits": len(circuits), "backend": label,
                "realized_delays_s": realized}

    # ---------------------------------------------------------------- execute
    print(f"executing {len(tqc)} circuits x {shots} shots ...")
    if simulator:
        result = backend.run(tqc, shots=shots).result()
        counts_list = [result.get_counts(i) for i in range(len(tqc))]
    else:
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(mode=backend)
        job = sampler.run(tqc, shots=shots)
        jid = job.job_id()
        print(f"job id: {jid}  (status: {job.status()})")

        # Persist everything needed to reassemble the dataset later. Real devices queue
        # for hours, so submitting and fetching must be separable -- otherwise a dropped
        # connection loses a job that has already consumed quota.
        pend = Path(outdir)
        pend.mkdir(parents=True, exist_ok=True)
        run_id_pending = f"gapscan_{label}_{stamp}" + (f"_{tag}" if tag else "")
        pending_path = pend / f"{run_id_pending}_pending.json"
        pending_path.write_text(json.dumps({
            "job_id": jid, "backend": label, "lengths": lengths, "shots": shots,
            "n_sequences": n_sequences, "n_repeats": n_repeats,
            "delays_realized_s": {str(k): v for k, v in realized.items()},
            "dd_conditions": list(dd_conditions), "qubit": qubit, "seed": seed,
            "dt_s": dt, "calibration": calibration, "timestamp_utc": stamp,
            "circuit_meta": meta,
        }, indent=2), encoding="utf-8")
        print(f"pending job saved -> {pending_path.name}")

        if no_wait:
            print("\nSubmitted and NOT waiting. Fetch results later with:")
            print(f"  .venv\\Scripts\\python.exe -m hardware.run_gapscan --fetch {jid}")
            return {"submitted": True, "job_id": jid,
                    "pending_file": str(pending_path),
                    "backend": label, "n_circuits": len(tqc)}

        print("waiting for results (safe to interrupt: the job keeps running and can be "
              "fetched by id) ...")
        counts_list = _counts_from_job_result(job.result())

    # ------------------------------------------------- reshape into survivals
    survivals: dict[str, np.ndarray] = {}
    for mt, counts in zip(meta, counts_list):
        key = f"{mt['dd']}|{mt['delay']:.12g}"
        arr = survivals.setdefault(
            key, np.full((len(lengths), n_sequences, n_repeats), np.nan, dtype=float)
        )
        li = lengths.index(mt["m"])
        arr[li, mt["seq_index"], mt.get("repeat_index", 0)] = _survival_from_counts(
            counts, shots)

    # ---------------------------------------------------------------- save
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"gapscan_{label}_{stamp}{('_' + tag) if tag else ''}"
    npz_path = out / f"{run_id}_survivals.npz"
    np.savez_compressed(npz_path, lengths=np.array(lengths), **survivals)

    meta_out = {
        "run_id": run_id,
        "backend": label,
        "simulator": simulator,
        "noise_model": noise if simulator else None,
        "lengths": lengths,
        "delays_requested_ns": list(delays_ns),
        "delays_realized_s": {str(k): v for k, v in realized.items()},
        "dd_conditions": list(dd_conditions),
        "n_sequences": n_sequences,
        "n_repeats": n_repeats,
        "shots": shots,
        "qubit": qubit,
        "seed": seed,
        "optimization_level": optimization_level,
        "dt_s": dt,
        "calibration": calibration,
        "keys": sorted(survivals.keys()),
        "timestamp_utc": stamp,
    }
    meta_path = out / f"{run_id}_meta.json"
    meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"saved {npz_path.name} and {meta_path.name}")

    # ------------------------------------------------- immediate witness readout
    # Delegate to the shared reporter so submitted and fetched runs report identically
    # (and so the 3-D repeat axis is flattened in exactly one place).
    _report_witness(survivals, lengths, meta_out, meta_path, seed=seed)

    return meta_out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a Gap-Scan RB experiment")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="run on the local Aer simulator")
    g.add_argument("--backend", metavar="NAME", help="run on IBM hardware (name or auto)")
    g.add_argument("--fake", metavar="NAME", help="run on a fake IBM backend (realistic dt/T1/T2, offline)")
    ap.add_argument("--noise", default="none", choices=["none", "markovian", "correlated"],
                    help="simulator noise model")
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    ap.add_argument("--delays-ns", type=float, nargs="+", default=DEFAULT_DELAYS_NS)
    ap.add_argument("--sequences", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats of each identical circuit (>=2 enables W_var^rep)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--dd", nargs="+", default=["none"],
                    choices=["none", "hahn", "cpmg", "xy4"])
    ap.add_argument("--qubit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--dry-run", action="store_true",
                    help="build+transpile, submit nothing")
    ap.add_argument("--no-wait", action="store_true",
                    help="submit and exit; fetch later with --fetch JOB_ID")
    ap.add_argument("--fetch", metavar="JOB_ID",
                    help="retrieve a previously submitted job by id")
    ap.add_argument("--optimization-level", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.fetch:
        fetch(args.fetch)
        return 0

    simulator = not args.backend
    if not simulator and not args.dry_run:
        from hardware.backend import credentials_available

        if not credentials_available():
            print("No IBM Quantum credentials. Set them up first:")
            print("  1) free account: https://quantum.cloud.ibm.com/")
            print("  2) .venv\\Scripts\\python.exe -m hardware.backend --save-token YOUR_TOKEN")
            return 1

    run(
        lengths=args.lengths,
        delays_ns=args.delays_ns,
        n_sequences=args.sequences,
        n_repeats=args.repeats,
        shots=args.shots,
        dd_conditions=tuple(args.dd),
        qubit=args.qubit,
        seed=args.seed,
        backend_name=args.backend if args.backend and args.backend != "auto" else None,
        simulator=simulator,
        noise=args.noise,
        fake=args.fake,
        dry_run=args.dry_run,
        no_wait=args.no_wait,
        optimization_level=args.optimization_level,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
