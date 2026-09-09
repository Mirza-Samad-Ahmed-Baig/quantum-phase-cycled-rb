"""
ROC calibration of the classifier's decision thresholds on simulated ground truth.

`estimator.witness` ships two decision constants -- NU_MARKOVIAN_MAX (nu below which noise
is called memoryless) and WVAR_STOCHASTIC_MIN (excess variance above which run-to-run
memory is called genuine). Until now both were set by hand from a handful of runs, and the
module comment pointed at this script, which did not exist. A referee is entitled to ask
where 1.35 came from; the honest answer has to be a curve, not a preference.

Method
------
Ground truth is generated with the same Monte-Carlo RB engine the rest of the project uses,
over a grid of correlation times spanning the Markovian and quasi-static regimes:

  * NEGATIVES: memoryless noise (tau_c far below the smallest gap), which the classifier
    must not flag. These fix the false-positive rate.
  * POSITIVES: correlated noise at several tau_c, which it should flag. These fix the
    detection rate, and their spread over tau_c is the point -- a threshold tuned on
    strongly quasi-static noise alone would look far better than it is.

For each trial the full witness is computed exactly as on hardware, including the
sequence-level bootstrap, and the DECISION RULE IS THE ONE USED IN PRODUCTION: the whole
confidence interval must clear the threshold, not merely the point estimate. Sweeping the
threshold then traces a genuine ROC.

The operating point is chosen by a stated criterion rather than by eye: the smallest
threshold whose false-positive rate stays at or below `--target-fpr`. Reporting the
resulting detection rate at that point is what tells a reader the price of the choice.

    .venv\\Scripts\\python.exe scripts\\calibrate_thresholds.py            # quick
    .venv\\Scripts\\python.exe scripts\\calibrate_thresholds.py --trials 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimator.witness import gap_scan_witness  # noqa: E402
from sim.rb_classical import monte_carlo_rb  # noqa: E402

TAU_GATE = 0.05                       # us
LENGTHS = np.arange(2, 42, 4)
DELTAS = np.geomspace(0.02, 0.5, 6)   # us
N_SEQ = 60
N_SHOTS = 1024

# Negative class: correlation time far below the smallest gap -> genuinely memoryless.
NEGATIVE = {"model": "markovian", "sigma": 0.55, "tau_c": 2e-3}
# Positive class: correlated noise across the regime the protocol claims to cover.
POSITIVE_TAU_C = [1.0, 5.0, 30.0, 150.0]      # us; smallest gap is 0.02 us
POSITIVE_SIGMA = 0.32


def _clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Exact upper confidence limit on a binomial rate: k successes in n trials.

    Uses the Beta quantile form, which stays correct at k = 0 where normal-approximation
    intervals collapse to zero width and would claim a false-positive rate of exactly 0
    from a handful of trials.
    """
    from scipy.stats import beta

    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def one_trial(cfg: dict, rng: np.random.Generator) -> dict:
    """One full gap scan through the production witness."""
    datasets = [
        monte_carlo_rb(
            LENGTHS, cfg["model"], N_SEQ, TAU_GATE + float(d), float(d), rng,
            sigma=cfg["sigma"], tau_c=cfg["tau_c"], n_shots=N_SHOTS, tau_gate=TAU_GATE,
        )
        for d in DELTAS
    ]
    w = gap_scan_witness(datasets, n_boot=200, rng=rng)
    lo, hi = w.get("nu_ci", (np.nan, np.nan))
    return {"nu": w.get("nu"), "lo": lo, "hi": hi, "w_var": w.get("w_var")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30,
                    help="trials per class (per tau_c for the positive class)")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", default="data/threshold_calibration.json")
    a = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(a.seed)

    print(f"negatives: memoryless, {a.trials} trials")
    neg = [one_trial(NEGATIVE, rng) for _ in range(a.trials)]

    pos: dict[float, list] = {}
    for tc in POSITIVE_TAU_C:
        print(f"positives: correlated tau_c = {tc} us, {a.trials} trials")
        cfg = {"model": "rtn", "sigma": POSITIVE_SIGMA, "tau_c": tc}
        pos[tc] = [one_trial(cfg, rng) for _ in range(a.trials)]

    # The production rule: flag only when the ENTIRE CI clears the threshold.
    def flagged(rows, thr):
        return np.mean([np.isfinite(r["lo"]) and r["lo"] > thr for r in rows])

    grid = np.round(np.arange(1.00, 2.01, 0.05), 2)
    fpr = np.array([flagged(neg, t) for t in grid])
    tpr = {tc: np.array([flagged(rows, t) for t in grid]) for tc, rows in pos.items()}
    tpr_all = np.array([flagged([r for rows in pos.values() for r in rows], t)
                        for t in grid])

    print("\n" + "=" * 78)
    print("ROC over the nu threshold (decision uses the whole bootstrap CI)")
    print("=" * 78)
    header = f"{'thr':>6} {'FPR':>7}" + "".join(f"{f'TPR@{tc}us':>11}" for tc in POSITIVE_TAU_C)
    print(header + f"{'TPR(all)':>10}")
    for i, t in enumerate(grid):
        row = f"{t:>6.2f} {fpr[i]:>7.3f}"
        for tc in POSITIVE_TAU_C:
            row += f"{tpr[tc][i]:>11.3f}"
        print(row + f"{tpr_all[i]:>10.3f}")

    # Choose on an UPPER CONFIDENCE BOUND for the false-positive rate, not on the point
    # estimate. With n negative trials, observing zero false positives does not mean the
    # rate is zero -- it means it is below roughly 3/n. Selecting on the raw estimate
    # therefore picks the loosest threshold on the grid as soon as n is small, which is
    # exactly the regime where the estimate is least trustworthy. The Clopper-Pearson
    # bound makes the sample size visible in the answer: too few trials and NO threshold
    # qualifies, which is the correct response rather than a confident wrong one.
    n_neg = len(neg)
    k = np.round(fpr * n_neg).astype(int)
    fpr_hi = np.array([_clopper_pearson_upper(int(kk), n_neg, 0.95) for kk in k])
    ok = np.where(fpr_hi <= a.target_fpr)[0]
    if len(ok) == 0:
        need = int(np.ceil(np.log(0.05) / np.log(1 - a.target_fpr)))
        print(f"\nNo threshold reaches a 95% UPPER BOUND on FPR of {a.target_fpr} with "
              f"{n_neg} negative trials.")
        print(f"  With zero observed false positives this needs n >= {need}; "
              f"re-run with --trials {need}.")
        chosen = float(grid[int(np.argmin(fpr_hi))])
    else:
        chosen = float(grid[ok[0]])
    i = int(np.argmin(np.abs(grid - chosen)))
    print(f"\n95% upper bound on FPR at the chosen threshold: {fpr_hi[i]:.3f} "
          f"(point estimate {fpr[i]:.3f}, n = {n_neg})")

    print("\n" + "-" * 78)
    print(f"OPERATING POINT: smallest threshold with 95% upper-bound FPR <= {a.target_fpr}")
    print(f"  nu_threshold = {chosen:.2f}   measured FPR = {fpr[i]:.3f}   "
          f"pooled TPR = {tpr_all[i]:.3f}")
    for tc in POSITIVE_TAU_C:
        print(f"    detection rate at tau_c = {tc:>6} us : {tpr[tc][i]:.3f}")

    # A single "optimal" threshold overstates what this measurement determines. What the
    # ROC actually shows is a PLATEAU of thresholds that simultaneously control the
    # false-positive rate and lose no detections; any value inside it is defensible, and
    # reporting the interval is more honest than reporting its left endpoint. Preferring a
    # point above the left edge is also physically motivated: nu approaches 1 FROM ABOVE at
    # finite delta/tau_c, so a genuinely Markovian qubit can legitimately read a little over
    # 1 and a threshold hard against the edge would eventually clip it.
    good = (fpr_hi <= a.target_fpr) & (tpr_all >= tpr_all.max() - 1e-9)
    from estimator.witness import NU_MARKOVIAN_MAX
    j = int(np.argmin(np.abs(grid - NU_MARKOVIAN_MAX)))
    if good.any():
        lo_t, hi_t = float(grid[good][0]), float(grid[good][-1])
        print(f"\n  ADMISSIBLE PLATEAU: nu_max in [{lo_t:.2f}, {hi_t:.2f}] holds the "
              f"95% upper-bound FPR at or below {a.target_fpr} while keeping the pooled "
              f"detection rate at {tpr_all.max():.3f}")
        inside = lo_t - 1e-9 <= NU_MARKOVIAN_MAX <= hi_t + 1e-9
        print(f"  shipped NU_MARKOVIAN_MAX = {NU_MARKOVIAN_MAX} "
              f"(FPR {fpr[j]:.3f}, TPR {tpr_all[j]:.3f}) -> "
              + ("INSIDE the plateau; no change needed" if inside else
                 "OUTSIDE the plateau; update it in estimator/witness.py"))
    else:
        print(f"\n  No threshold both controls the FPR and preserves detection; "
              f"shipped {NU_MARKOVIAN_MAX} sits at FPR {fpr[j]:.3f}, TPR {tpr_all[j]:.3f}")
        lo_t = hi_t = float("nan")
    print("-" * 78)

    # W_var separation is reported as a distribution rather than an ROC: the negative class
    # here is stochastic-but-memoryless, so W_var's job is to sit near 1, not to discriminate
    # tau_c. A static-detuning arm would be needed for its own ROC.
    wneg = [r["w_var"] for r in neg if r["w_var"] is not None and np.isfinite(r["w_var"])]
    if wneg:
        print(f"\nW_var on memoryless ground truth: median {np.median(wneg):.2f}, "
              f"95th pct {np.percentile(wneg, 95):.2f} "
              f"(threshold {__import__('estimator.witness', fromlist=['x']).WVAR_STOCHASTIC_MIN})")

    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "grid": grid.tolist(),
        "fpr": fpr.tolist(),
        "tpr_pooled": tpr_all.tolist(),
        "tpr_by_tau_c": {str(k): v.tolist() for k, v in tpr.items()},
        "positive_tau_c_us": POSITIVE_TAU_C,
        "target_fpr": a.target_fpr,
        "chosen_threshold": chosen,
        "plateau_lo": lo_t, "plateau_hi": hi_t,
        "shipped_nu_markovian_max": float(NU_MARKOVIAN_MAX),
        "shipped_inside_plateau": bool(lo_t <= NU_MARKOVIAN_MAX <= hi_t)
        if lo_t == lo_t else False,
        "fpr_at_chosen": float(fpr[i]),
        "fpr_upper95_at_chosen": float(fpr_hi[i]),
        "fpr_upper95": fpr_hi.tolist(),
        "tpr_at_chosen": float(tpr_all[i]),
        "trials_per_class": a.trials,
        "n_sequences": N_SEQ, "n_shots": N_SHOTS,
        "deltas_us": DELTAS.tolist(), "lengths": LENGTHS.tolist(),
        "w_var_memoryless_median": float(np.median(wneg)) if wneg else None,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved {out}   ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
