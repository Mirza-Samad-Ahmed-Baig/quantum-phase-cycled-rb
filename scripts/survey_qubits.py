"""
Multi-qubit Gap-Scan RB survey: hunt for a qubit with CORRELATED idle noise (nu ~ 2).

Qubit selection is physics-driven, not arbitrary. A qubit limited only by relaxation has
T2 = 2*T1; a small ratio T2/(2*T1) means excess PURE dephasing beyond the T1 limit, which
on superconducting hardware usually comes from 1/f flux noise or TLS coupling -- precisely
the low-frequency correlated noise this protocol detects. Controls with a large ratio should
come back Markovian.

All qubits go into ONE job so the survey waits in the queue once rather than N times. Each
qubit's circuits are transpiled to its own physical layout and then concatenated; the qubit
index travels in the per-circuit metadata so results can be split apart on fetch.

Usage
    # select qubits, build, transpile, submit (one job), exit
    .venv\\Scripts\\python.exe scripts\\survey_qubits.py --submit

    # dry run: build and transpile only, submit nothing
    .venv\\Scripts\\python.exe scripts\\survey_qubits.py --dry-run

    # fetch and rank once the job is DONE
    .venv\\Scripts\\python.exe scripts\\survey_qubits.py --fetch JOB_ID
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hardware.backend as hb  # noqa: E402  (installs the OS trust store)
from estimator.witness import (  # noqa: E402
    classify,
    epc_from_p,
    excess_variance_repeats,
    fit_decay_p,
    gap_scan_witness,
    monotonicity_test,
    scaling_exponent_offset,
)
from hardware.circuits import build_circuit_set, quantize_delay  # noqa: E402
from hardware.run_gapscan import _counts_from_job_result, _survival_from_counts  # noqa: E402

BACKEND = "ibm_fez"     # default; override with --backend
ACCOUNT = None          # default Qiskit account; override with --account
LENGTHS = [2, 6, 12, 20]
DELAYS_NS = [40, 160, 640]
N_SEQ = 20            # >= 20 per PLAN.md before any non-monotonicity claim
N_REPEATS = 1         # screening; follow up on hits with repeats for W_var^rep
SHOTS = 256
MAX_READOUT = 0.05


def select_qubits(backend, n_candidates=6, n_controls=2):
    """Rank qubits by T2/(2*T1): lowest = most excess dephasing = best candidates."""
    props = backend.properties()
    rows = []
    for q in range(backend.num_qubits):
        try:
            t1, t2, ro = props.t1(q), props.t2(q), props.readout_error(q)
            if t1 and t2 and ro is not None and ro < MAX_READOUT:
                rows.append({"qubit": q, "T1_s": t1, "T2_s": t2,
                             "ratio": t2 / (2 * t1), "readout_error": ro})
        except Exception:
            continue
    rows.sort(key=lambda r: r["ratio"])
    cands = rows[:n_candidates]
    ctrls = rows[-n_controls:]
    for r in cands:
        r["group"] = "candidate"
    for r in ctrls:
        r["group"] = "control"
    return cands + ctrls


def build_and_submit(dry_run: bool, tag: str = "survey", qubits=None,
                     delays_ns=None, n_seq=None, n_repeats=None, shots=None,
                     dd_conditions=("none",), lengths=None):
    from qiskit import transpile

    svc = hb.get_service(ACCOUNT)
    backend = svc.backend(BACKEND)
    timing = hb.backend_timing(backend)
    dt, gran = timing["dt_s"], timing["granularity"]
    print(f"backend {BACKEND}  dt={dt}  granularity={gran}")

    global N_SEQ, N_REPEATS, SHOTS, LENGTHS
    if n_seq: N_SEQ = n_seq
    if n_repeats: N_REPEATS = n_repeats
    if shots: SHOTS = shots
    if lengths: LENGTHS = list(lengths)
    dns = delays_ns or DELAYS_NS
    if qubits:
        props = backend.properties()
        sel = []
        for q in qubits:
            t1, t2 = props.t1(q), props.t2(q)
            sel.append({'qubit': q, 'T1_s': t1, 'T2_s': t2,
                        'ratio': t2 / (2 * t1), 'readout_error': props.readout_error(q),
                        'group': 'followup'})
    else:
        sel = select_qubits(backend)
    print(f"\n{'qubit':>5} {'group':>10} {'T1(us)':>8} {'T2(us)':>8} {'T2/2T1':>8} {'ro_err':>8}")
    for r in sel:
        print(f"{r['qubit']:>5} {r['group']:>10} {r['T1_s']*1e6:>8.1f} "
              f"{r['T2_s']*1e6:>8.1f} {r['ratio']:>8.3f} {r['readout_error']:>8.4f}")

    realized, delays_s = {}, []
    for ns in dns:
        _, actual = quantize_delay(ns * 1e-9, dt, granularity=gran)
        realized[ns] = actual
        delays_s.append(actual)
    n_distinct = len(set(round(v, 15) for v in realized.values()))
    print(f"\nrealized gaps (ns): "
          + ", ".join(f"{k}->{v*1e9:.0f}" for k, v in realized.items())
          + f"   [{n_distinct} distinct]")
    if n_distinct < len(dns):
        print("  !! WARNING: gaps collapsed; duplicated x-points bias the nu fit")

    all_tqc, all_meta = [], []
    for r in sel:
        q = r["qubit"]
        # Same seed for every qubit: identical Clifford sequences, so cross-qubit
        # comparisons are paired and cannot be explained by different draws.
        rng = np.random.default_rng(4242)
        circs, meta = build_circuit_set(
            LENGTHS, delays_s, N_SEQ, rng, dd_conditions=tuple(dd_conditions), unit="s",
            qubit=q, dt=dt, granularity=gran, n_repeats=N_REPEATS,
        )
        tqc = transpile(circs, backend=backend, optimization_level=1, initial_layout=[q])
        for m in meta:
            m["qubit"] = q
            m["group"] = r["group"]
        all_tqc.extend(tqc)
        all_meta.extend(meta)
    print(f"\nbuilt {len(all_tqc)} circuits total "
          f"({len(sel)} qubits x {len(LENGTHS)} lengths x {len(delays_s)} delays x {N_SEQ} seq)")
    nd = [c.count_ops().get("delay", 0) for c in all_tqc]
    print(f"delay instructions per circuit: min={min(nd)} max={max(nd)}")

    if dry_run:
        print("\nDRY RUN — nothing submitted.")
        return None

    from qiskit_ibm_runtime import SamplerV2 as Sampler

    job = Sampler(mode=backend).run(all_tqc, shots=SHOTS)
    jid = job.job_id()
    print(f"\njob id: {jid}  status: {job.status()}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("data")
    out.mkdir(exist_ok=True)
    pend = out / f"survey_{BACKEND}_{stamp}_{tag}_pending.json"
    pend.write_text(json.dumps({
        "job_id": jid, "backend": BACKEND, "lengths": LENGTHS, "shots": SHOTS,
        "n_sequences": N_SEQ, "n_repeats": N_REPEATS,
        "delays_realized_s": {str(k): v for k, v in realized.items()},
        "qubits": sel, "dd_conditions": list(dd_conditions),
        "timestamp_utc": stamp, "circuit_meta": all_meta,
    }, indent=2), encoding="utf-8")
    print(f"pending -> {pend.name}")
    print(f"\nfetch later with:\n  .venv\\Scripts\\python.exe scripts\\survey_qubits.py "
          f"--fetch {jid}")
    return jid


def fetch_and_rank(job_id: str):
    pend = None
    for f in sorted(Path("data").glob("survey_*_pending.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("job_id") == job_id:
            pend, pend_file = d, f
            break
    if pend is None:
        raise FileNotFoundError(f"no survey_*_pending.json records job_id {job_id}")

    svc = hb.get_service(ACCOUNT)
    job = svc.job(job_id)
    st = str(job.status())
    print(f"job {job_id}: {st}")
    if "DONE" not in st.upper():
        print("not finished yet.")
        return
    counts = _counts_from_job_result(job.result())

    lengths = pend["lengths"]
    n_seq, n_rep, shots = pend["n_sequences"], pend["n_repeats"], pend["shots"]
    # (qubit, dd, delay) -> (L, N_seq, N_rep)
    data = {}
    for mt, c in zip(pend["circuit_meta"], counts):
        key = (mt["qubit"], mt["dd"], float(mt["delay"]))
        arr = data.setdefault(key, np.full((len(lengths), n_seq, n_rep), np.nan))
        arr[lengths.index(mt["m"]), mt["seq_index"], mt.get("repeat_index", 0)] = \
            _survival_from_counts(c, shots)

    qinfo = {r["qubit"]: r for r in pend["qubits"]}
    results = []
    for q in sorted(qinfo):
        keys = sorted([k for k in data if k[0] == q and k[1] == "none"], key=lambda k: k[2])
        if len(keys) < 2:
            continue
        datasets, epcs, dls = [], [], []
        for k in keys:
            a = data[k]
            flat = a.reshape(a.shape[0], -1)
            datasets.append({"lengths": np.array(lengths), "survivals": flat,
                             "delta": k[2] * 1e6, "tau_gate": 0.05})
            epcs.append(epc_from_p(fit_decay_p(np.array(lengths), np.nanmean(flat, axis=1))))
            dls.append(k[2] * 1e9)
        w = gap_scan_witness(datasets, n_boot=300, rng=np.random.default_rng(q + 1))
        # The offset-aware fit is the CORRECT exponent: gate error adds a delta-independent
        # eps_0, and a pure log-log slope through eps_0 + A*delta^nu reads far too low
        # (a true nu=2 with eps_0=1e-3 reads 0.84, i.e. "memoryless").
        off = scaling_exponent_offset(np.array(dls), np.array(epcs))
        # CLASSIFY ON THE OFFSET-AWARE EXPONENT. Classifying on the raw log-log slope while
        # flagging detections on the offset-aware one made the two disagree: q53 was reported
        # `memoryless` from a raw nu = 0.51 in the very run whose offset-aware
        # nu = 1.87 +- 0.22 cleared the detection threshold. The label now comes from the
        # same estimator the decision does, and the raw slope is kept only as a diagnostic.
        w_off = dict(w)
        if off.get("nu") is not None and np.isfinite(off.get("nu", np.nan)):
            err = off.get("nu_stderr") or 0.0
            w_off["nu"] = off["nu"]
            w_off["nu_ci"] = (off["nu"] - 2.0 * err, off["nu"] + 2.0 * err)
        c = classify(w_off)
        # Historical shot-group variability diagnostic. Sequence dependence and
        # drift can contribute; this is not a necessary inter-slot-memory witness.
        wv_rep = np.nanmedian([
            excess_variance_repeats(data[k], shots)["w_var_rep"] for k in keys
        ]) if n_rep >= 2 else np.nan
        # Echo response is an additional control, not a universal memory criterion.
        dd_ratio = np.nan
        dkeys = sorted([k for k in data if k[0] == q and k[1] != "none"], key=lambda k: k[2])
        if dkeys:
            kmax = max(keys, key=lambda k: k[2])
            dmax = max(dkeys, key=lambda k: k[2])
            e_none = epc_from_p(fit_decay_p(np.array(lengths),
                       np.nanmean(data[kmax].reshape(len(lengths), -1), axis=1)))
            e_dd = epc_from_p(fit_decay_p(np.array(lengths),
                     np.nanmean(data[dmax].reshape(len(lengths), -1), axis=1)))
            dd_ratio = e_dd / e_none if e_none else np.nan
        mono_all = all(
            monotonicity_test(np.array(lengths), np.nanmean(data[k].reshape(len(lengths), -1), axis=1),
                              np.nanstd(data[k].reshape(len(lengths), -1), axis=1, ddof=1)
                              / np.sqrt(n_seq * n_rep))["monotonic_decreasing"]
            for k in keys)
        results.append({
            "qubit": q, "group": qinfo[q]["group"], "ratio": qinfo[q]["ratio"],
            "T2_us": qinfo[q]["T2_s"] * 1e6, "nu": w["nu"], "nu_ci": w["nu_ci"],
            "r2": w["nu_r_squared"], "label": c["label"], "nu_raw": w["nu"],
            "classify_reasons": c.get("reasons"), "epcs": epcs, "deltas_ns": dls,
            "monotonic": mono_all,
            "nu_offset": off.get("nu"), "nu_offset_err": off.get("nu_stderr"),
            "eps0_fit": off.get("eps0"), "offset_r2": off.get("r_squared"),
            "w_var_rep": float(wv_rep) if np.isfinite(wv_rep) else None,
            "dd_ratio": float(dd_ratio) if np.isfinite(dd_ratio) else None,
        })

    results.sort(key=lambda r: -(r["nu_offset"] if r.get("nu_offset") and np.isfinite(r["nu_offset"]) else -9))
    print("\n" + "=" * 92)
    print("SURVEY RESULTS — ranked by nu (higher = more correlated; nu~1 Markovian, nu~2 correlated)")
    print("=" * 92)
    print(f"{'qubit':>5} {'group':>9} {'T2/2T1':>7} {'nu_raw':>7} "
          f"{'nu_OFFSET':>10} {'+-':>6} {'eps0':>8} {'R2':>6} {'W^rep':>6} {'DD':>6}")
    for r in results:
        no = r.get("nu_offset")
        ne = r.get("nu_offset_err")
        print(f"{r['qubit']:>5} {r['group']:>9} {r['ratio']:>7.3f} {r['nu']:>7.3f} "
              f"{(no if no is not None else float('nan')):>10.3f} "
              f"{(ne if ne is not None else float('nan')):>6.3f} "
              f"{(r.get('eps0_fit') or float('nan')):>8.5f} "
              f"{(r.get('offset_r2') or float('nan')):>6.3f} "
              f"{(r.get('w_var_rep') if r.get('w_var_rep') is not None else float('nan')):>6.2f} "
              f"{(r.get('dd_ratio') if r.get('dd_ratio') is not None else float('nan')):>6.2f}")

    hits = [r for r in results if r.get("nu_offset") and np.isfinite(r["nu_offset"])
        and r["nu_offset"] - 2 * (r.get("nu_offset_err") or 9) > 1.35]
    print("\n" + "-" * 92)
    if hits:
        print(f"EXPONENT SCREENING CANDIDATE on {len(hits)} qubit(s): "
              f"{[h['qubit'] for h in hits]} — nu CI lies entirely above 1.35.")
        print("Follow up with an independent acquisition and matched-marginal controls.")
        print("An exponent or shot-group variability alone does not certify memory.")
    else:
        print("No exponent screening candidate met the historical threshold.")
        print("This does not establish the absence of inter-slot memory or a quantitative exclusion.")
    print("-" * 92)

    out = Path("data") / pend_file.name.replace("_pending.json", "_results.json")
    out.write_text(json.dumps(
        [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
         for r in results], indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {out.name}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--submit", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--fetch", metavar="JOB_ID")
    ap.add_argument('--qubits', type=int, nargs='+')
    ap.add_argument('--delays-ns', type=float, nargs='+')
    ap.add_argument('--sequences', type=int)
    ap.add_argument('--repeats', type=int)
    ap.add_argument('--shots', type=int)
    ap.add_argument('--tag', default='survey')
    ap.add_argument('--dd', nargs='+', default=['none'])
    ap.add_argument('--lengths', type=int, nargs='+')
    ap.add_argument('--backend', default=BACKEND)
    ap.add_argument('--account', default=None)
    a = ap.parse_args()
    BACKEND = a.backend
    ACCOUNT = a.account
    if a.fetch:
        fetch_and_rank(a.fetch)
    else:
        build_and_submit(dry_run=a.dry_run, tag=a.tag, qubits=a.qubits,
                         delays_ns=a.delays_ns, n_seq=a.sequences,
                         n_repeats=a.repeats, shots=a.shots,
                         dd_conditions=tuple(a.dd))
