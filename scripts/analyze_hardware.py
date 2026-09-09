"""
Analyze a Gap-Scan RB dataset measured on real hardware and plot it.

    .venv\\Scripts\\python.exe scripts\\analyze_hardware.py [data/<run>_survivals.npz]

Produces figures/hardware_gapscan.png: the RB decays at each idle gap, and the
EPC-vs-delta log-log scan with nu = 1 / nu = 2 reference slopes.

Includes the W_var CONFOUND DIAGNOSTIC described in PLAN.md: on hardware the raw
between-sequence variance is inflated by genuine Clifford-to-Clifford fidelity differences,
not only by run-to-run memory. The two are separable by their m-dependence -- sequence
identity dominates at SMALL m and washes out as m grows, whereas real run-to-run memory
persists or grows with m. This script reports the trend so the distinction is visible
rather than hidden inside a single number.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimator.witness import (  # noqa: E402
    classify,
    epc_from_p,
    fit_decay_p,
    gap_scan_witness,
    monotonicity_test,
)


def load(path: str | None):
    if path is None:
        cands = sorted(glob.glob("data/*_survivals.npz"))
        cands = [c for c in cands if "fake" not in c and "aer" not in c] or cands
        if not cands:
            raise FileNotFoundError("no data/*_survivals.npz found")
        path = cands[-1]
    d = np.load(path)
    meta_path = path.replace("_survivals.npz", "_meta.json")
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return path, d, meta


def main(argv):
    path, d, meta = load(argv[1] if len(argv) > 1 else None)
    lengths = np.asarray(d["lengths"], dtype=int)
    cal = meta.get("calibration") or {}
    keys = sorted([k for k in d.files if k != "lengths"],
                  key=lambda k: float(k.split("|")[1]))
    none_keys = [k for k in keys if k.split("|")[0] == "none"]

    print(f"file    : {Path(path).name}")
    print(f"backend : {meta.get('backend')} qubit {meta.get('qubit')}")
    if cal:
        print(f"device  : T1={cal.get('T1_s', 0)*1e6:.1f}us  T2={cal.get('T2_s', 0)*1e6:.1f}us  "
              f"readout_err={cal.get('readout_error')}  sx_err={cal.get('gate_error_sx')}")
    print(f"config  : {meta.get('shots')} shots x {meta.get('n_sequences')} sequences, "
          f"m={list(lengths)}\n")

    # ---------------------------------------------------------------- witness
    datasets = []
    for k in none_keys:
        datasets.append({"lengths": lengths, "survivals": d[k],
                         "delta": float(k.split("|")[1]) * 1e6, "tau_gate": 0.05})
    w = gap_scan_witness(datasets, n_boot=400, rng=np.random.default_rng(1))
    c = classify(w)
    print("GAP-SCAN WITNESS")
    print(f"  nu    = {w['nu']:.4f}  CI [{w['nu_ci'][0]:.4f}, {w['nu_ci'][1]:.4f}]  "
          f"(log-log R^2 = {w['nu_r_squared']:.4f})")
    print(f"  W_var = {w['w_var']:.2f}  CI [{w['w_var_ci'][0]:.2f}, {w['w_var_ci'][1]:.2f}]")
    print(f"  label = {c['label']}")
    for r in c["reasons"]:
        print(f"    - {r}")

    # ------------------------------------------------- linearity of EPC in delta
    print("\nEPC SCALING (the primary witness, read directly off the data)")
    dl = np.array([float(k.split("|")[1]) * 1e9 for k in none_keys])
    ep = np.array([epc_from_p(fit_decay_p(lengths, np.nanmean(d[k], axis=1)))
                   for k in none_keys])
    for i in range(len(none_keys)):
        print(f"  delta={dl[i]:>6.0f} ns   EPC={ep[i]:.5f}"
              + (f"   (x{dl[i]/dl[0]:.0f} in delta -> x{ep[i]/ep[0]:.2f} in EPC; "
                 f"nu=1 predicts x{dl[i]/dl[0]:.0f}, nu=2 predicts x{(dl[i]/dl[0])**2:.0f})"
                 if i else ""))

    # ------------------------------------------------- W_var confound diagnostic
    print("\nW_var CONFOUND DIAGNOSTIC (excess variance vs sequence length)")
    print("  Sequence-identity systematics dominate at SMALL m and wash out as m grows;")
    print("  genuine run-to-run memory persists or grows with m.")
    verdicts = []
    for k in none_keys:
        s = d[k]
        pbar = np.nanmean(s, axis=1)
        between = np.nanvar(s, axis=1, ddof=1)
        within = np.maximum(pbar * (1 - pbar) / meta["shots"], 1e-15)
        ratio = between / within
        trend = np.polyfit(np.log(lengths), np.log(np.maximum(ratio, 1e-9)), 1)[0]
        verdicts.append(trend)
        print(f"  delta={float(k.split('|')[1])*1e9:>6.0f} ns  ratio="
              + " ".join(f"{x:6.1f}" for x in ratio)
              + f"   d ln(ratio)/d ln(m) = {trend:+.2f}")
    mean_trend = float(np.mean(verdicts))
    print(f"  mean trend = {mean_trend:+.2f}  -> "
          + ("DECREASING with m: consistent with Clifford-to-Clifford fidelity spread, "
             "NOT run-to-run memory" if mean_trend < 0 else
             "non-decreasing with m: consistent with genuine run-to-run memory"))

    # ---------------------------------------------------------------- monotonicity
    print("\nMONOTONICITY (quantum-memory signature)")
    for k in none_keys:
        s = d[k]
        asf = np.nanmean(s, axis=1)
        se = np.nanstd(s, axis=1, ddof=1) / np.sqrt(s.shape[1])
        mt = monotonicity_test(lengths, asf, se)
        print(f"  delta={float(k.split('|')[1])*1e9:>6.0f} ns  "
              f"monotonic={mt['monotonic_decreasing']}  max rise z={mt['max_rise_z']:+.2f} "
              f"(z_crit={mt['z_critical']:.2f})")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figdir = Path("figures")
        figdir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

        ax = axes[0]
        for k in none_keys:
            s = d[k]
            asf = np.nanmean(s, axis=1)
            se = np.nanstd(s, axis=1, ddof=1) / np.sqrt(s.shape[1])
            ax.errorbar(lengths, asf, yerr=se, fmt="o-", capsize=2,
                        label=rf"$\delta$={float(k.split('|')[1])*1e9:.0f} ns")
        ax.set_xlabel("Clifford sequence length $m$")
        ax.set_ylabel("ASF")
        ax.set_title(f"{meta.get('backend')} q{meta.get('qubit')}: RB decay per idle gap")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.loglog(dl, ep, "o", ms=8, color="tab:blue",
                  label=rf"measured: $\nu$={w['nu']:.2f} "
                        rf"[{w['nu_ci'][0]:.2f}, {w['nu_ci'][1]:.2f}]")
        ax.loglog(dl, ep[0] * (dl / dl[0]) ** 1, ":", color="grey", lw=1.5,
                  label=r"$\nu=1$ (Markovian)")
        ax.loglog(dl, ep[0] * (dl / dl[0]) ** 2, "-.", color="black", lw=1.5,
                  label=r"$\nu=2$ (correlated)")
        ax.set_xlabel(r"idle gap $\delta$ (ns)")
        ax.set_ylabel("fitted EPC")
        ax.set_title("Gap scan on real hardware")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        fig.tight_layout()
        out = figdir / "hardware_gapscan.png"
        fig.savefig(out, dpi=140)
        print(f"\nfigure -> {out}")
    except Exception as e:
        print(f"[warn] figure failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
