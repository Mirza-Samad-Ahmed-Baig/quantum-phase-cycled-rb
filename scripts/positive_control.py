"""
Injected-fluctuator POSITIVE CONTROL for the gap-scan witness, on real hardware.

Why this experiment exists
--------------------------
The multi-qubit survey returned a null: every qubit measured on ibm_fez came back
`memoryless`, with the offset-aware exponent consistent with nu = 1. A null is only worth
publishing if the instrument that produced it is demonstrably able to produce a non-null,
and nothing in the survey establishes that ON HARDWARE -- every positive demonstration of
the witness so far lives in simulation. A referee's first question is therefore unanswerable
as things stand: is this device Markovian, or is this protocol blind?

This script answers it by MANUFACTURING the signal. A synthetic fluctuator with a known
correlation time tau_c is injected as virtual rz frame changes inside each idle window
(`hardware.circuits.injected_slot_phases`, drawing from the same `sim.noise` generator the
simulation layer is validated against). Because rz is virtual on IBM hardware, the injection
costs no time and leaves the gap geometry untouched: the injected and un-injected arms differ
ONLY by the phases, so they are directly comparable.

The three arms span the physics the witness is supposed to separate:

    arm     tau_c            regime            predicted nu
    ----    -------------    --------------    ------------
    qs      >> largest gap   quasi-static      2
    mid     ~ mid gap        crossover         between 1 and 2
    fast    << smallest gap  Markovian         1

plus `none`, the untouched device, which is the null arm and the subtraction baseline.

The analysis is a PAIRED SUBTRACTION. Both arms run in the same job, on the same qubit, with
the same Clifford sequences and the same gaps, so the device's own error cancels:

    EPC_excess(delta) = EPC_injected(delta) - EPC_none(delta)

and the exponent is fitted to that excess. This matters because the device's intrinsic error
is NOT delta-independent -- it scales roughly as delta^1 -- so it cannot be absorbed into the
additive eps_0 of the offset-aware fit. Subtracting the paired null arm removes it exactly,
leaving the injected physics alone. The measured excess exponent is then compared against the
closed form the injection was drawn from; agreement is the calibration this paper needs.

Usage
    # size the injection and inspect the design, submitting nothing
    .venv\\Scripts\\python.exe scripts\\positive_control.py --dry-run

    # submit (spends QPU minutes)
    .venv\\Scripts\\python.exe scripts\\positive_control.py --submit --backend ibm_fez --qubit 46

    # analyse once DONE
    .venv\\Scripts\\python.exe scripts\\positive_control.py --fetch JOB_ID
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hardware.backend as hb  # noqa: E402
from estimator.witness import (  # noqa: E402
    dd_control_validity,
    epc_from_p,
    excess_variance_repeats,
    fit_decay_p,
    scaling_exponent_offset,
)
from hardware.circuits import (  # noqa: E402
    build_circuit_set,
    predicted_injected_epc,
    quantize_delay,
)
from hardware.run_gapscan import _counts_from_job_result, _survival_from_counts  # noqa: E402
from sim.noise import f_stable  # noqa: E402

DEFAULT_LENGTHS = [2, 6, 12, 20]
# Gap grid. Concentrated on the decade where a quasi-static (nu = 2) excess is actually
# resolvable: its signal falls as delta^2, so gaps below ~40 ns contribute noise rather than
# leverage at any injection amplitude that keeps |theta| inside the 2pi/3 trap boundary.
DEFAULT_DELAYS_NS = [40, 80, 160, 320, 640]
# Target excess EPC at the reference (largest) gap. Chosen so the injected signal clearly
# exceeds the device's own error there (~1.3e-2 on ibm_fez q46 at 640 ns) while the typical
# phase stays far inside |theta| < 2pi/3, the window where p = (1+2cos theta)/3 is still
# positive and strong classical noise cannot counterfeit the quantum-memory signature.
TARGET_EXCESS_EPC = 0.06
# Keep a gap only where the paired excess is resolved by this many sigma. See the SNR cut
# in fetch_and_analyse for why an unresolved point is worse than no point at all.
SNR_MIN = 2.0


def sigma_for_target(tau_c_s: float, delta_ref_s: float, target_epc: float) -> float:
    """Injection amplitude sigma (rad/s) giving `target_epc` of excess EPC at `delta_ref`.

    Inverts EPC = (1 - e^{-v/2})/3 for the phase variance v, then inverts
    v = 2 sigma^2 tau_c^2 f(delta/tau_c). Sizing every arm to the SAME excess EPC at the
    reference gap equalises signal-to-noise across arms, so a difference in the recovered
    exponent reflects the correlation time rather than one arm simply having been driven
    harder than another.
    """
    if not 0 < target_epc < 1.0 / 3.0:
        raise ValueError("target EPC must lie in (0, 1/3); EPC saturates at 1/3")
    v = -2.0 * np.log(1.0 - 3.0 * target_epc)
    denom = 2.0 * tau_c_s**2 * float(f_stable(delta_ref_s / tau_c_s))
    return float(np.sqrt(v / denom))


def predicted_nu(tau_c_s: float, deltas_s, sigma_rad_s: float) -> float:
    """Exponent the injected physics dictates over the actual gap grid.

    Fitted the same way the data will be (log-log slope of the closed-form excess EPC), so
    the prediction and the measurement are compared on identical footing rather than against
    an idealised asymptote the finite gap range never reaches. Two finite-range effects are
    deliberately kept in the prediction rather than idealised away:

      * the gap grid never reaches either asymptote, so a crossover arm sits strictly
        between 1 and 2 by construction;
      * EPC saturates at 1/3, which compresses the log-log slope once the injection is
        strong. This is why the arm's ACTUAL sigma is used and not a nominal 1.0 -- the
        prediction has to carry the same saturation the measurement will.
    """
    d = np.asarray(deltas_s, dtype=float)
    e = np.array([predicted_injected_epc(sigma_rad_s, tau_c_s, x) for x in d])
    ok = e > 0
    return float(np.polyfit(np.log(d[ok]), np.log(e[ok]), 1)[0])


def tau_c_for_nu(delays_s, target_nu: float, target_epc: float,
                 lo: float = 1e-11, hi: float = 1e-3) -> float:
    """Correlation time whose predicted exponent over THIS gap grid equals `target_nu`.

    The crossover arm has to be placed by solving, not by guessing a fraction of the gap
    range. Predicted nu is a compressive function of tau_c over a finite grid: setting
    tau_c to 30% of the largest gap sounds central but predicts nu = 1.81, which is only
    0.18 away from the quasi-static arm and well inside the error bars the measurement will
    have. Bisection on the closed form puts the arm exactly midway between the asymptotes,
    where it can actually discriminate.
    """
    def f(tc):
        return predicted_nu(tc, delays_s, sigma_for_target(tc, float(max(delays_s)),
                                                           target_epc)) - target_nu

    a, b = lo, hi
    fa = f(a)
    for _ in range(200):
        mid = np.sqrt(a * b)               # bisect in log space; tau_c spans decades
        fm = f(mid)
        if abs(fm) < 1e-4:
            return float(mid)
        if (fa < 0) == (fm < 0):
            a, fa = mid, fm
        else:
            b = mid
    return float(np.sqrt(a * b))


def build_arms(delays_s, target_epc: float = TARGET_EXCESS_EPC, which=None):
    """Define the injected arms and size each one against the largest gap."""
    d_ref = float(max(delays_s))
    d_min = float(min(delays_s))
    specs = [
        ("qs", 40.0 * d_ref),    # quasi-static: correlated across the whole sequence
        # Crossover, solved so the closed form predicts nu = 1.5 -- midway between the
        # Markovian and quasi-static asymptotes, hence maximally far from both arms.
        ("mid", tau_c_for_nu(delays_s, 1.5, target_epc)),
        ("fast", 0.05 * d_min),  # Markovian: decorrelates inside the smallest gap
    ]
    arms = [None]
    for name, tau_c in specs:
        if which is not None and name not in which:
            continue
        arms.append({
            "name": name,
            "model": "ou",
            "tau_c_s": float(tau_c),
            "sigma_rad_s": sigma_for_target(tau_c, d_ref, target_epc),
        })
    return arms


def resolve_lengths(args, delays_s, tau_gate, t2_s, epc_hint=None):
    """Either the fixed grid from the command line, or a coherence-matched grid per gap.

    `epc_hint` maps gap (in seconds) to a measured EPC for that gap, normally taken from the
    probe run. Sizing the grid from a guess would defeat the point: too high an EPC estimate
    truncates the sequences and throws away decay signal, too low a one pushes m past the
    depolarised floor and spends shots on noise.
    """
    if not args.adaptive:
        return list(args.lengths)
    from hardware.circuits import adaptive_lengths

    out = {}
    for d in delays_s:
        epc = None
        if epc_hint:
            epc = epc_hint.get(round(float(d), 15))
        if epc is None or not np.isfinite(epc) or epc <= 0:
            # Fall back to a T2-only bound; still far better than one grid for every gap.
            epc = 1e-3
        # PER-ARM grids. Sizing one grid from the strongest arm truncates the null arm's
        # decay; sizing it from the null arm runs the strongest injected arm into the
        # depolarised floor, leaving its log-linear fit two usable points and a bootstrap
        # error of order the value itself. On the first ibm_fez run that is exactly what
        # happened: the quasi-static arm came back 3.4e-2 +- 1.9e-2 and the exponent could
        # not be fitted at all. Each arm now gets a grid matched to its own total error.
        for arm in args._arms:
            name = "none" if arm is None else arm["name"]
            extra = (0.0 if arm is None
                     else predicted_injected_epc(arm["sigma_rad_s"], arm["tau_c_s"], float(d)))
            out[(name, float(d))] = adaptive_lengths(
                float(d), epc + extra, tau_gate, t2_s, n_points=args.n_lengths)
    return out


def _load_epc_hint(results_path: str, delays_s) -> dict:
    """Read measured EPC(gap) from an earlier results file, matched onto this gap grid.

    Gaps are matched by nearest realized value rather than by label so that a probe taken
    with a slightly different timing-grid rounding still lines up; a mismatch of a few dt
    would otherwise silently drop the hint and fall back to the default estimate.
    """
    d = json.loads(Path(results_path).read_text(encoding="utf-8"))
    if isinstance(d, dict) and "delays_ns" in d and "epc_none" in d:
        src = {float(ns) * 1e-9: float(e)
               for ns, e in zip(d["delays_ns"], d["epc_none"]) if e is not None}
    elif isinstance(d, list) and d and "deltas_ns" in d[0]:
        src = {float(ns) * 1e-9: float(e)
               for ns, e in zip(d[0]["deltas_ns"], d[0]["epcs"]) if e is not None}
    else:
        raise ValueError(f"{results_path} does not look like a gap-scan results file")
    out = {}
    for d_target in delays_s:
        near = min(src, key=lambda s: abs(s - float(d_target)))
        if abs(near - float(d_target)) < 0.25 * float(d_target):
            out[round(float(d_target), 15)] = src[near]
    return out


def _report_design(arms, delays_s, lengths, n_seq, n_rep, shots, dd_conditions):
    print(f"\n{'arm':>6} {'tau_c (ns)':>12} {'sigma (rad/s)':>15} {'nu_pred':>8}  "
          + "  ".join(f"{d*1e9:>8.0f}ns" for d in delays_s))
    for arm in arms:
        if arm is None:
            print(f"{'none':>6} {'-':>12} {'-':>15} {'-':>8}  "
                  + "  ".join(f"{'-':>10}" for _ in delays_s))
            continue
        row = [predicted_injected_epc(arm["sigma_rad_s"], arm["tau_c_s"], d)
               for d in delays_s]
        print(f"{arm['name']:>6} {arm['tau_c_s']*1e9:>12.1f} {arm['sigma_rad_s']:>15.3e} "
              f"{predicted_nu(arm['tau_c_s'], delays_s, arm['sigma_rad_s']):>8.3f}  "
              + "  ".join(f"{v:>10.2e}" for v in row))
    # Guard the documented false-positive trap before any QPU time is spent.
    print("\n  phase-magnitude check (|theta| must stay below 2pi/3 = 2.094 rad):")
    for arm in arms:
        if arm is None:
            continue
        v = max(2.0 * arm["sigma_rad_s"]**2 * arm["tau_c_s"]**2
                * float(f_stable(d / arm["tau_c_s"])) for d in delays_s)
        rms = float(np.sqrt(v))
        flag = "OK" if 3.0 * rms < 2.094 else "!! 3-sigma tail crosses 2pi/3"
        print(f"    {arm['name']:>6}: rms|theta| = {rms:.3f} rad, 3-sigma = {3*rms:.3f}  {flag}")
    if isinstance(lengths, dict):
        print("\n  per-arm, gap-adaptive length grids (each matched to that arm's own error):")
        n_len = 0
        for arm in arms:
            name = "none" if arm is None else arm["name"]
            print(f"    arm {name}:")
            for d in delays_s:
                g = lengths[(name, float(d))]
                n_len += len(g)
                print(f"      {d*1e9:>6.0f} ns -> m = {list(map(int, g))}")
        n_circ = len(dd_conditions) * n_len * n_seq * n_rep
        print(f"\n  {n_len} (arm,gap,length) triples x {len(dd_conditions)} DD x "
              f"{n_seq} seq x {n_rep} rep = {n_circ} circuits")
    else:
        n_circ = len(arms) * len(delays_s) * len(dd_conditions) * len(lengths) * n_seq * n_rep
        print(f"\n  {len(arms)} arms x {len(delays_s)} gaps x {len(dd_conditions)} DD x "
              f"{len(lengths)} lengths x {n_seq} seq x {n_rep} rep = {n_circ} circuits")
    print(f"  {n_circ} circuits x {shots} shots = {n_circ * shots:,} shots total")
    return n_circ


def build_and_submit(args):
    from qiskit import transpile

    svc = hb.get_service(args.account)
    backend = svc.backend(args.backend)
    timing = hb.backend_timing(backend)
    dt, gran = timing["dt_s"], timing["granularity"]
    snap = hb.calibration_snapshot(backend, args.qubit)
    print(f"backend {args.backend}  dt={dt}  granularity={gran}  qubit {args.qubit}")
    print(f"  T1={snap.get('T1_s', 0) * 1e6:.1f} us  T2={snap.get('T2_s', 0) * 1e6:.1f} us  "
          f"sx_err={snap.get('gate_error_sx')}  ro_err={snap.get('readout_error')}")

    delays_s = []
    for ns in args.delays_ns:
        _, actual = quantize_delay(ns * 1e-9, dt, granularity=gran)
        delays_s.append(actual)
    if len(set(round(v, 15) for v in delays_s)) < len(delays_s):
        raise SystemExit("gaps collapsed onto the timing grid; duplicated x-points would "
                         "bias the exponent fit -- choose gaps further apart")

    # Dead time between idle windows. The environment keeps evolving during the Cliffords,
    # so leaving this at zero would overstate slot-to-slot correlation in the injected arms.
    sx_dur = snap.get("gate_duration_sx_s") or 0.0
    tau_gate = 2.0 * float(sx_dur)
    print(f"  sx duration = {sx_dur * 1e9:.1f} ns -> slot dead time tau_gate = "
          f"{tau_gate * 1e9:.1f} ns")

    arms = build_arms(delays_s, args.target_epc, which=args.arms)
    args._arms = arms
    epc_hint = _load_epc_hint(args.epc_from, delays_s) if args.epc_from else None
    if epc_hint:
        print("\n  EPC hints from " + args.epc_from + ": "
              + ", ".join(f"{k*1e9:.0f}ns={v:.2e}" for k, v in sorted(epc_hint.items())))
    lengths = resolve_lengths(args, delays_s, tau_gate, snap.get("T2_s"), epc_hint)
    n_circ = _report_design(arms, delays_s, lengths, args.sequences,
                            args.repeats, args.shots, args.dd)

    rng = np.random.default_rng(args.seed)
    circs, meta = build_circuit_set(
        lengths, delays_s, args.sequences, rng,
        dd_conditions=tuple(args.dd), unit="s", qubit=args.qubit,
        dt=dt, granularity=gran, n_repeats=args.repeats,
        inject_arms=tuple(arms), tau_gate_s=tau_gate, inject_seed=args.seed,
    )
    tqc = transpile(circs, backend=backend, optimization_level=1,
                    initial_layout=[args.qubit])
    nd = [c.count_ops().get("delay", 0) for c in tqc]
    nrz = [c.count_ops().get("rz", 0) for c in tqc]
    print(f"\ntranspiled {len(tqc)} circuits; delays/circuit min={min(nd)} max={max(nd)}; "
          f"rz/circuit min={min(nrz)} max={max(nrz)}")
    if min(nd) == 0:
        raise SystemExit("a transpiled circuit lost all its delays -- the gap would not be "
                         "executed; refusing to submit")

    if args.dry_run:
        print("\nDRY RUN - nothing submitted.")
        return None

    from qiskit_ibm_runtime import SamplerV2 as Sampler

    before = svc.usage().get("usage_remaining_seconds")
    job = Sampler(mode=backend).run(tqc, shots=args.shots)
    jid = job.job_id()
    print(f"\njob id: {jid}  status: {job.status()}")
    print(f"QPU seconds remaining before this job: {before}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("data")
    out.mkdir(exist_ok=True)
    pend = out / f"poscontrol_{args.backend}_{stamp}_{args.tag}_pending.json"
    pend.write_text(json.dumps({
        "job_id": jid, "backend": args.backend, "qubit": args.qubit,
        "lengths": ({f"{k[0]}|{k[1]}": list(map(int, v)) for k, v in lengths.items()}
                    if isinstance(lengths, dict) else list(lengths)),
        "adaptive_lengths": bool(isinstance(lengths, dict)),
        "shots": args.shots,
        "n_sequences": args.sequences, "n_repeats": args.repeats,
        "delays_realized_s": delays_s, "dd_conditions": list(args.dd),
        "tau_gate_s": tau_gate, "seed": args.seed,
        "arms": [a for a in arms if a is not None],
        "calibration": snap, "timestamp_utc": stamp,
        "usage_remaining_before_s": before,
        "circuit_meta": meta,
    }, indent=2, default=float), encoding="utf-8")
    hb.append_calibration_log(snap)
    print(f"pending -> {pend.name}")
    print(f"\nfetch with:\n  .venv\\Scripts\\python.exe scripts\\positive_control.py "
          f"--fetch {jid}")
    return jid


def _epc_curve(data, lengths_by_key, keys, n_boot: int = 300):
    """EPC per gap for one (arm, dd) slice, plus a bootstrap sigma over sequences.

    Each key carries its OWN length grid, because with gap-adaptive scheduling the gaps do
    not share one. The bootstrap resamples whole SEQUENCES rather than shots: the Clifford
    sequence is the correlated unit (its transpiled gate count sets its fidelity), so
    resampling shots would treat sequence-to-sequence spread as if it were shot noise and
    report confidence intervals several times too narrow.
    """
    epcs, sig = [], []
    for k in keys:
        a = data.get(k)
        if a is None:
            epcs.append(np.nan)
            sig.append(np.nan)
            continue
        L = np.asarray(lengths_by_key[k], dtype=float)
        flat = a.reshape(a.shape[0], -1)
        epcs.append(epc_from_p(fit_decay_p(L, np.nanmean(flat, axis=1))))
        # Python's built-in hash is process-randomized. New analyses use a stable
        # seed; historical stored bootstrap errors are retained as historical values.
        import hashlib
        seed = int.from_bytes(hashlib.sha256(repr(k).encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        n = flat.shape[1]
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            b = epc_from_p(fit_decay_p(L, np.nanmean(flat[:, idx], axis=1)))
            if np.isfinite(b):
                boot.append(b)
        sig.append(float(np.std(boot, ddof=1)) if len(boot) > 5 else np.nan)
    return np.array(epcs, dtype=float), np.array(sig, dtype=float)


def fetch_and_analyse(job_id: str, account: str | None = None):
    pend = pend_file = None
    for f in sorted(Path("data").glob("poscontrol_*_pending.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("job_id") == job_id:
            pend, pend_file = d, f
            break
    if pend is None:
        raise FileNotFoundError(f"no poscontrol_*_pending.json records job_id {job_id}")

    svc = hb.get_service(account)
    job = svc.job(job_id)
    st = str(job.status())
    print(f"job {job_id}: {st}")
    if "DONE" not in st.upper():
        print("not finished yet.")
        return None
    counts = _counts_from_job_result(job.result())

    n_seq, n_rep, shots = pend["n_sequences"], pend["n_repeats"], pend["shots"]
    # Derive each slice's length grid from the metadata rather than from a single global
    # list: with gap-adaptive scheduling every gap has its own grid.
    lengths_by_key = {}
    for mt in pend["circuit_meta"]:
        key = (mt.get("arm", "none"), mt["dd"], float(mt["delay"]))
        lengths_by_key.setdefault(key, set()).add(int(mt["m"]))
    lengths_by_key = {k: sorted(v) for k, v in lengths_by_key.items()}

    data = {}
    for mt, c in zip(pend["circuit_meta"], counts):
        key = (mt.get("arm", "none"), mt["dd"], float(mt["delay"]))
        L = lengths_by_key[key]
        arr = data.setdefault(key, np.full((len(L), n_seq, n_rep), np.nan))
        arr[L.index(int(mt["m"])), mt["seq_index"], mt.get("repeat_index", 0)] = \
            _survival_from_counts(c, shots)

    delays_s = np.array(sorted({k[2] for k in data}))
    arms = {a["name"]: a for a in pend["arms"]}
    dds = pend["dd_conditions"]

    # ---- Baseline (null arm, no DD): the paired subtraction reference. -------------
    base_keys = [("none", "none", float(d)) for d in delays_s]
    base_epc, base_sig = _epc_curve(data, lengths_by_key, base_keys)
    print("\n" + "=" * 96)
    print("PAIRED SUBTRACTION: excess EPC = EPC(injected) - EPC(none), same qubit, same job")
    print("=" * 96)
    print(f"{'gap (ns)':>10} " + "  ".join(f"{d*1e9:>10.0f}" for d in delays_s))
    print(f"{'none':>10} " + "  ".join(f"{v:>10.3e}" for v in base_epc))

    results = {"job_id": job_id, "backend": pend["backend"], "qubit": pend["qubit"],
               "delays_ns": (delays_s * 1e9).tolist(),
               "lengths_by_gap": {str(round(float(d)*1e9)): lengths_by_key[("none", "none", float(d))]
                                  for d in delays_s if ("none", "none", float(d)) in lengths_by_key},
               "epc_none": base_epc.tolist(), "epc_none_sigma": base_sig.tolist(),
               "arms": []}

    for name, arm in arms.items():
        keys = [(name, "none", float(d)) for d in delays_s]
        if not all(k in data for k in keys):
            continue
        epc, sig = _epc_curve(data, lengths_by_key, keys)
        excess = epc - base_epc
        excess_sig = np.sqrt(sig**2 + base_sig**2)
        print(f"{name:>10} " + "  ".join(f"{v:>10.3e}" for v in epc))
        print(f"{'  excess':>10} " + "  ".join(f"{v:>10.3e}" for v in excess))

        # SNR CUT. A nu = 2 arm's excess spans (delta_max/delta_min)^2 across the grid, so
        # the small-gap end can sit far below the achievable precision no matter how the
        # injection is sized. Those points carry no information about the exponent but do
        # carry noise -- in the probe run the 8 ns excess came out NEGATIVE and dragged the
        # fitted exponent from a predicted 1.99 down to 0.90. Points are kept only where the
        # excess is resolved; how many survived is reported so the cut is never silent.
        resolved = (np.isfinite(excess) & np.isfinite(excess_sig)
                    & (excess > SNR_MIN * excess_sig))
        ok = resolved
        # Fit the exponent on the excess alone: with the device's own error subtracted
        # there is no additive pedestal left, so a pure log-log slope is the right estimator
        # and eps_0 is pinned at 0 rather than fitted against a signal that has none.
        fit = scaling_exponent_offset(delays_s[ok] * 1e9, excess[ok], eps0=0.0,
                                      sigma_epcs=excess_sig[ok]) if ok.sum() >= 3 else {}
        dropped = [f"{delays_s[i]*1e9:.0f}ns" for i in range(len(delays_s)) if not ok[i]]
        if dropped:
            print(f"{'':>10}   gaps dropped below {SNR_MIN}-sigma resolution: "
                  + ", ".join(dropped))
        nu_pred = predicted_nu(arm["tau_c_s"], delays_s, arm["sigma_rad_s"])
        nu_meas = fit.get("nu", np.nan)
        nu_err = fit.get("nu_stderr", np.nan)
        z = ((nu_meas - nu_pred) / nu_err) if np.isfinite(nu_err) and nu_err > 0 else np.nan
        print(f"{'':>10}   tau_c = {arm['tau_c_s']*1e9:>10.1f} ns   "
              f"nu_predicted = {nu_pred:.3f}   nu_measured = {nu_meas:.3f} +- {nu_err:.3f}"
              f"   (z = {z:+.2f})")

        # Historical variability diagnostic; inter-slot dependence does not
        # necessarily produce spread between averaged shot groups.
        wvr = np.nan
        if n_rep >= 2:
            wvr = float(np.nanmedian([
                excess_variance_repeats(data[k], shots)["w_var_rep"] for k in keys]))
        # DD control, reported only where the echo can pay for itself.
        dd_rows = []
        for dd in dds:
            if dd == "none":
                continue
            dkeys = [(name, dd, float(d)) for d in delays_s]
            if not all(k in data for k in dkeys):
                continue
            depc, _ = _epc_curve(data, lengths_by_key, dkeys)
            gate_err = pend["calibration"].get("gate_error_x") or \
                pend["calibration"].get("gate_error_sx") or 0.0
            npulse = {"hahn": 1, "cpmg": 2, "xy4": 4}.get(dd, 1)
            for i, d in enumerate(delays_s):
                val = dd_control_validity(epc[i], npulse, gate_err)
                dd_rows.append({
                    "dd": dd, "delta_ns": float(d * 1e9),
                    "epc_dd": float(depc[i]) if np.isfinite(depc[i]) else None,
                    "epc_none_arm": float(epc[i]) if np.isfinite(epc[i]) else None,
                    "ratio": (float(depc[i] / epc[i])
                              if np.isfinite(depc[i]) and np.isfinite(epc[i]) and epc[i] else None),
                    "valid": val["valid"], "headroom": val["headroom"],
                })
        results["arms"].append({
            "name": name, "tau_c_s": arm["tau_c_s"], "sigma_rad_s": arm["sigma_rad_s"],
            "epc": epc.tolist(), "epc_sigma": sig.tolist(),
            "excess": excess.tolist(), "excess_sigma": excess_sig.tolist(),
            "n_gaps_fitted": int(ok.sum()), "gaps_dropped": dropped,
            "nu_predicted": nu_pred, "nu_measured": float(nu_meas),
            "nu_stderr": float(nu_err) if np.isfinite(nu_err) else None,
            "z_vs_prediction": float(z) if np.isfinite(z) else None,
            "w_var_rep": float(wvr) if np.isfinite(wvr) else None,
            "dd": dd_rows,
        })

    print("\n" + "-" * 96)
    print("CALIBRATION SUMMARY — does the witness return the exponent the injection dictates?")
    print("-" * 96)
    print(f"{'arm':>8} {'tau_c (ns)':>12} {'nu_pred':>9} {'nu_meas':>9} {'+-':>7} "
          f"{'z':>7} {'W^rep':>7}")
    for a in results["arms"]:
        print(f"{a['name']:>8} {a['tau_c_s']*1e9:>12.1f} {a['nu_predicted']:>9.3f} "
              f"{a['nu_measured']:>9.3f} {(a['nu_stderr'] or float('nan')):>7.3f} "
              f"{(a['z_vs_prediction'] if a['z_vs_prediction'] is not None else float('nan')):>7.2f} "
              f"{(a['w_var_rep'] if a['w_var_rep'] is not None else float('nan')):>7.2f}")

    out = Path("data") / pend_file.name.replace("_pending.json", "_results.json")
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {out.name}")
    try:
        print(f"QPU seconds remaining now: {svc.usage().get('usage_remaining_seconds')}")
    except Exception:
        pass
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--submit", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--fetch", metavar="JOB_ID")
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--account", default="coauthor")
    ap.add_argument("--qubit", type=int, default=46)
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    ap.add_argument("--adaptive", action="store_true",
                    help="give each gap its own coherence-matched length grid")
    ap.add_argument("--n-lengths", type=int, default=6,
                    help="points per gap when --adaptive is set")
    ap.add_argument("--epc-from", metavar="RESULTS_JSON", default=None,
                    help="size the adaptive grids from measured EPC in this results file")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="which injected arms to build (qs mid fast); default all")
    ap.add_argument("--delays-ns", type=float, nargs="+", default=DEFAULT_DELAYS_NS)
    ap.add_argument("--sequences", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--dd", nargs="+", default=["none"])
    ap.add_argument("--target-epc", type=float, default=TARGET_EXCESS_EPC)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--tag", default="poscontrol")
    a = ap.parse_args()
    if a.fetch:
        fetch_and_analyse(a.fetch, a.account)
    else:
        build_and_submit(a)
