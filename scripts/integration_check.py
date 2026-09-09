"""
End-to-end Gap-Scan RB pipeline check: sim -> estimator -> classification.

Runs the full experiment in simulation, where the ground truth is known, and verifies
that the witness recovers the right answer in each regime:

  (a) Markovian / fast noise    -> 'memoryless'
  (b) correlated quasi-static   -> 'classical-memory'   (the RB blind spot)

Also reproduces the two results the paper turns on:
  1. Standard RB is BLIND: correlated noise still fits a single exponential in m.
  2. The gap scan SEES it: nu separates the two regimes under identical fitting code.

Run:  .venv\\Scripts\\python.exe scripts\\integration_check.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimator.witness import classify, gap_scan_witness, monotonicity_test  # noqa: E402
from sim.noise import chi_matrix  # noqa: E402
from sim.rb_classical import (  # noqa: E402
    epc_from_p,
    fit_single_exponential,
    gaussian_det_asf,
    monte_carlo_rb,
    nu_analytic,
)

# --------------------------------------------------------------------------- #
# Experiment configuration (kept small enough to run in a couple of minutes)
# --------------------------------------------------------------------------- #
TAU_GATE = 0.05                                  # us
LENGTHS = np.arange(2, 42, 4)                    # uniform grid (ESPRIT needs uniform)
DELTAS = np.geomspace(0.02, 0.5, 6)              # us
N_SEQ = 150
N_SHOTS = 2048

# Regime A: fast noise, tau_c << delta  -> Markovian-like, nu -> 1
FAST = {"model": "markovian", "sigma": 0.55, "tau_c": 2e-3}
# Regime B: slow noise, tau_c >> delta  -> quasi-static / correlated, nu -> 2
SLOW = {"model": "rtn", "sigma": 0.32, "tau_c": 150.0}


def build_scan(cfg: dict, rng: np.random.Generator) -> list:
    """Run the RB gap scan for one noise configuration."""
    out = []
    for d in DELTAS:
        out.append(
            monte_carlo_rb(
                LENGTHS,
                cfg["model"],
                N_SEQ,
                TAU_GATE + float(d),
                float(d),
                rng,
                sigma=cfg["sigma"],
                tau_c=cfg["tau_c"],
                n_shots=N_SHOTS,
                tau_gate=TAU_GATE,
            )
        )
    return out


def max_slot_phase(cfg: dict, delta: float, n_sigma: float = 4.0) -> float:
    """Conservative bound on the per-slot phase, for the 2pi/3 certificate."""
    chi00 = chi_matrix(1, cfg["sigma"], cfg["tau_c"], TAU_GATE + delta, delta)[0, 0]
    return float(n_sigma * np.sqrt(chi00))


def main() -> int:
    rng = np.random.default_rng(20260729)
    t0 = time.time()
    results = {}

    print("=" * 74)
    print("GAP-SCAN RB — END-TO-END INTEGRATION CHECK")
    print("=" * 74)
    print(f"tau_gate = {TAU_GATE} us | lengths = {LENGTHS.tolist()}")
    print(f"deltas   = {np.array2string(DELTAS, precision=3)} us")
    print(f"{N_SEQ} sequences x {N_SHOTS} shots per point\n")

    # ---------------------------------------------------------------- stage 1
    print("-" * 74)
    print("STAGE 1 — the blind spot: does standard RB see the correlation?")
    print("-" * 74)
    d_mid = float(DELTAS[len(DELTAS) // 2])
    asf_slow = gaussian_det_asf(LENGTHS, SLOW["sigma"], SLOW["tau_c"], TAU_GATE + d_mid, d_mid)
    fit_slow = fit_single_exponential(LENGTHS, asf_slow)
    mono = bool(np.all(np.diff(asf_slow) < 1e-12))
    print(f"  strongly correlated noise (tau_c = {SLOW['tau_c']} us >> delta = {d_mid:.3f} us)")
    print(f"  single-exponential fit: R^2 = {fit_slow['r_squared']:.9f}, "
          f"residual rms = {fit_slow['residual_rms']:.2e}")
    print(f"  ASF monotonically decreasing: {mono}")
    blind = fit_slow["r_squared"] > 0.999 and mono
    print(f"  -> standard RB is BLIND (looks perfectly Markovian): "
          f"{'CONFIRMED' if blind else 'NOT CONFIRMED'}")
    results["blind_spot"] = blind

    # ---------------------------------------------------------------- stage 2
    print("\n" + "-" * 74)
    print("STAGE 2 — gap scan + witness on FAST (Markovian) ground truth")
    print("-" * 74)
    scan_fast = build_scan(FAST, rng)
    w_fast = gap_scan_witness(scan_fast, tau_gate=TAU_GATE, n_boot=200, rng=rng)
    c_fast = classify(w_fast, max_slot_phase=max_slot_phase(FAST, d_mid))
    print(f"  EPC(delta) = {np.array2string(w_fast['epcs'], precision=5)}")
    print(f"  nu    = {w_fast['nu']:.4f}  CI = [{w_fast['nu_ci'][0]:.4f}, {w_fast['nu_ci'][1]:.4f}]"
          f"   (log-log R^2 = {w_fast['nu_r_squared']:.4f})")
    print(f"  W_var = {w_fast['w_var']:.3f}  CI = [{w_fast['w_var_ci'][0]:.2f}, "
          f"{w_fast['w_var_ci'][1]:.2f}]")
    print(f"  label = {c_fast['label']}")
    for r in c_fast["reasons"]:
        print(f"    - {r}")
    ok_fast = c_fast["label"] == "memoryless"
    print(f"  -> expected 'memoryless': {'PASS' if ok_fast else 'FAIL'}")
    results["fast_memoryless"] = ok_fast

    # ---------------------------------------------------------------- stage 3
    print("\n" + "-" * 74)
    print("STAGE 3 — gap scan + witness on SLOW (correlated) ground truth")
    print("-" * 74)
    scan_slow = build_scan(SLOW, rng)
    w_slow = gap_scan_witness(scan_slow, tau_gate=TAU_GATE, n_boot=200, rng=rng)
    c_slow = classify(w_slow, max_slot_phase=max_slot_phase(SLOW, d_mid))
    print(f"  EPC(delta) = {np.array2string(w_slow['epcs'], precision=5)}")
    print(f"  nu    = {w_slow['nu']:.4f}  CI = [{w_slow['nu_ci'][0]:.4f}, {w_slow['nu_ci'][1]:.4f}]"
          f"   (log-log R^2 = {w_slow['nu_r_squared']:.4f})")
    print(f"  W_var = {w_slow['w_var']:.3f}  CI = [{w_slow['w_var_ci'][0]:.2f}, "
          f"{w_slow['w_var_ci'][1]:.2f}]")
    print(f"  label = {c_slow['label']}")
    for r in c_slow["reasons"]:
        print(f"    - {r}")
    ok_slow = c_slow["label"] in ("classical-memory", "static-detuning")
    print(f"  -> expected correlated (classical-memory): {'PASS' if ok_slow else 'FAIL'}")
    results["slow_correlated"] = ok_slow

    # ---------------------------------------------------------------- stage 4
    print("\n" + "-" * 74)
    print("STAGE 4 — separation and comparison against analytic prediction")
    print("-" * 74)
    sep = w_slow["nu"] - w_fast["nu"]
    x_mid = d_mid / SLOW["tau_c"]
    print(f"  nu(correlated) - nu(Markovian) = {sep:.4f}")
    print(f"  analytic nu at delta/tau_c = {x_mid:.2e}  ->  {float(nu_analytic(x_mid)):.4f}")
    print("  NOTE nu -> 1 FROM ABOVE, so a Markovian nu slightly above 1 is the correct")
    print("       finite-delta/tau_c prediction, not an artifact.")
    print("  NOTE finite-m single-exponential fitting compresses EPC at large delta, biasing")
    print("       nu DOWNWARD (toward 1) — the conservative direction, since it makes a false")
    print("       claim of correlation harder rather than easier.")
    separated = sep > 0.4 and w_slow["nu"] > w_fast["nu"]
    print(f"  -> gap scan separates the regimes: {'PASS' if separated else 'FAIL'}")
    results["separation"] = separated

    # ---------------------------------------------------------------- stage 5
    print("\n" + "-" * 74)
    print("STAGE 5 — quantum-memory branch (exact joint system+environment propagation)")
    print("-" * 74)
    from sim.rb_quantum import conditional_phase, quantum_rb, quantum_rb_exact

    lengths_q = np.arange(1, 61)
    # Commuting environment operators -> CCC / Theorem 5 -> must stay monotonic.
    a_ccc = quantum_rb_exact(lengths_q, 0.25, j_coupling=0.45, omega_x=0.0,
                             env_init="superposed")
    rise_ccc = float(np.max(np.diff(a_ccc)))
    # Non-commuting -> genuine quantum memory -> non-monotonic, with the conditional
    # phase held below 2pi/3 so classical sign-alternation is excluded.
    j_q, wx_q, d_q = 1.6, 0.5, 0.5
    theta_q = conditional_phase(j_q, d_q)
    a_qm = quantum_rb_exact(lengths_q, d_q, j_coupling=j_q, omega_x=wx_q, n_env=2)
    rise_qm = float(np.max(np.diff(a_qm)))
    print(f"  commuting (CCC, omega_x=0):     max rise = {rise_ccc:+.3e}  -> monotonic")
    print(f"  non-commuting (quantum memory): max rise = {rise_qm:+.3e}")
    print(f"  conditional phase |theta| = {theta_q:.3f} < 2pi/3 = {2*np.pi/3:.3f} (certified)")

    # Run the witness on sampled quantum-memory data and classify it.
    dq = quantum_rb(lengths_q, 400, d_q, rng, j_coupling=j_q, omega_x=wx_q, n_env=2,
                    n_shots=N_SHOTS, tau_gate=TAU_GATE)
    mono_q = monotonicity_test(dq.lengths, dq.asf, dq.asf_stderr)
    w_q = {"nu": np.nan, "nu_ci": (np.nan, np.nan), "w_var": np.nan,
           "any_nonmonotonic": not mono_q["monotonic_decreasing"]}
    c_q = classify(w_q, max_slot_phase=theta_q)
    print(f"  label on sampled data = {c_q['label']}")
    ok_q = (rise_ccc < 1e-9) and (rise_qm > 1e-6) and c_q["label"] == "quantum-memory"
    print(f"  -> CCC monotonic AND quantum memory detected+classified: "
          f"{'PASS' if ok_q else 'FAIL'}")
    results["quantum_memory"] = ok_q

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figdir = Path(__file__).resolve().parents[1] / "figures"
        figdir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

        ax = axes[0]
        for label, scan, style in (("Markovian", scan_fast, "o-"), ("correlated", scan_slow, "s--")):
            mid = scan[len(scan) // 2]
            ax.errorbar(mid.lengths, mid.asf, yerr=mid.asf_stderr, fmt=style,
                        capsize=2, label=f"{label} (delta={mid.delta:.3f}us)")
        ax.set_xlabel("sequence length m")
        ax.set_ylabel("ASF")
        ax.set_title("Both look like clean exponentials\n(standard RB is blind)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        # Plot against delta, matching how nu is defined and fitted (the stochastic phase
        # accumulates over the idle window only). Plotting slot time here would put a
        # different x-axis under a slope quoted in delta.
        ax.loglog(w_fast["deltas"], w_fast["epcs"], "o-",
                  label=f"Markovian: nu={w_fast['nu']:.2f}")
        ax.loglog(w_slow["deltas"], w_slow["epcs"], "s--",
                  label=f"correlated: nu={w_slow['nu']:.2f}")
        # Reference power laws through the first correlated point.
        d0, e0 = w_slow["deltas"][0], w_slow["epcs"][0]
        ax.loglog(w_slow["deltas"], e0 * (w_slow["deltas"] / d0) ** 1, ":", color="grey",
                  lw=1, label=r"$\nu=1$ (Markovian)")
        ax.loglog(w_slow["deltas"], e0 * (w_slow["deltas"] / d0) ** 2, "-.", color="black",
                  lw=1, label=r"$\nu=2$ (quasi-static)")
        ax.set_xlabel(r"idle gap $\delta$ ($\mu$s)")
        ax.set_ylabel("fitted EPC")
        ax.set_title("Gap scan reveals it\n(slope = scaling exponent nu)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        fig.tight_layout()
        out = figdir / "integration_check.png"
        fig.savefig(out, dpi=140)
        print(f"\n  figure written to {out}")
    except Exception as e:  # pragma: no cover - plotting is optional
        print(f"\n  [warn] figure not written: {e}")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k, v in results.items():
        print(f"  {k:20s}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"\n  runtime: {time.time() - t0:.1f}s")
    print(f"  integration check: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
