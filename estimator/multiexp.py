"""
Multi-exponential extraction from RB decay curves (ESPRIT / matrix pencil).

RB data has the form y_m = sum_j A_j p_j^m + B. Standard curve fitting assumes a
single exponential; when several decay modes are present, subspace methods recover
them far more reliably. ESPRIT/MUSIC are the established tools for this in the RB
context (Helsen et al., "A general framework for randomized benchmarking",
arXiv:2010.07974).

Handling the constant offset B
------------------------------
Rather than adding a spurious pole at p=1, we FIRST DIFFERENCE the samples:
    d_m = y_{m+1} - y_m = sum_j A_j (p_j - 1) p_j^m
which annihilates B exactly and leaves the poles p_j untouched. Differencing
requires a UNIFORMLY SPACED length grid, which this module enforces.

Purity note: this module never imports `sim`. It consumes plain arrays so it can be
validated against synthetic ground truth independently of the simulator.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "esprit_poles",
    "fit_multiexp",
    "select_model_order",
    "bootstrap_multiexp",
]


def _hankel(x: np.ndarray, n_rows: int) -> np.ndarray:
    n = len(x)
    n_cols = n - n_rows + 1
    if n_cols < 1:
        raise ValueError("n_rows too large for the data length")
    return np.lib.stride_tricks.sliding_window_view(x, n_cols)[:n_rows]


def select_model_order(
    singular_values: np.ndarray, max_order: int, rel_floor: float = 0.02
) -> int:
    """Pick the number of exponentials from the singular-value spectrum.

    Rule (stated explicitly so it is auditable rather than magic): keep every
    singular value exceeding `rel_floor` times the largest, then additionally cut at
    the largest consecutive ratio drop within that set. Conservative by design --
    on genuinely single-exponential data it should return 1, because falsely
    reporting a second exponential would make the shape witness a false-positive
    machine.
    """
    s = np.asarray(singular_values, dtype=float)
    s = s[s > 0]
    if len(s) == 0:
        return 1
    keep = int(np.sum(s / s[0] > rel_floor))
    keep = max(1, min(keep, max_order))
    if keep > 1:
        ratios = s[: keep - 1] / np.maximum(s[1:keep], 1e-300)
        cut = int(np.argmax(ratios)) + 1
        keep = max(1, min(keep, cut))
    return keep


def esprit_poles(y: np.ndarray, order: int | None = None, max_order: int = 4) -> dict:
    """Recover decay bases p_j from uniformly-sampled y via ESPRIT on first differences.

    Returns dict with poles, order, singular_values.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        raise ValueError("need at least 5 samples for ESPRIT")
    d = np.diff(y)
    n_rows = max(2, len(d) // 2)
    h = _hankel(d, n_rows)
    u, s, _ = np.linalg.svd(h, full_matrices=False)
    k = order if order is not None else select_model_order(
        s, max_order=min(max_order, h.shape[0] - 1, h.shape[1] - 1)
    )
    k = max(1, min(k, min(h.shape) - 1))
    us = u[:, :k]
    u1, u2 = us[:-1], us[1:]
    # Shift-invariance: u2 ~ u1 @ psi, poles are eigenvalues of psi.
    psi, *_ = np.linalg.lstsq(u1, u2, rcond=None)
    poles = np.linalg.eigvals(psi)
    return {"poles": poles, "order": k, "singular_values": s}


def fit_multiexp(
    lengths: np.ndarray,
    y: np.ndarray,
    order: int | None = None,
    max_order: int = 4,
    fix_offset: float | None = None,
) -> dict:
    """Full multi-exponential fit: poles via ESPRIT, amplitudes via least squares.

    `lengths` must be uniformly spaced (required by the differencing step).
    If `fix_offset` is given (e.g. 0.5 for ideal single-qubit SPAM) the constant is
    held there; otherwise it is a free parameter.

    Returns dict with poles (real, sorted by |amplitude| descending), amplitudes,
    offset, order, singular_values, fitted, residual_rms, r_squared,
    amplitude_ratio (second-largest / largest, the SHAPE statistic).
    """
    lengths = np.asarray(lengths, dtype=float)
    y = np.asarray(y, dtype=float)
    steps = np.diff(lengths)
    if not np.allclose(steps, steps[0]):
        raise ValueError("ESPRIT requires a uniformly spaced length grid")

    res = esprit_poles(y, order=order, max_order=max_order)
    poles = res["poles"]
    # Keep physically meaningful decays: real part in (0, 1].
    real_poles = np.real(poles)
    real_poles = real_poles[(real_poles > 1e-6) & (real_poles <= 1.0 + 1e-9)]
    if len(real_poles) == 0:
        real_poles = np.array([np.clip(np.real(poles[np.argmax(np.abs(poles))]), 1e-6, 1.0)])
    real_poles = np.unique(np.clip(real_poles, 1e-6, 1.0))

    # Linear least squares for amplitudes (and offset if free).
    cols = [p**lengths for p in real_poles]
    if fix_offset is None:
        cols.append(np.ones_like(lengths))
    design = np.column_stack(cols)
    target = y if fix_offset is None else y - fix_offset
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    if fix_offset is None:
        amps, offset = coef[: len(real_poles)], float(coef[-1])
    else:
        amps, offset = coef, float(fix_offset)

    order_idx = np.argsort(-np.abs(amps))
    amps, real_poles = amps[order_idx], real_poles[order_idx]
    fitted = offset + sum(a * p**lengths for a, p in zip(amps, real_poles))
    resid = y - fitted
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ratio = float(abs(amps[1]) / abs(amps[0])) if len(amps) > 1 and amps[0] != 0 else 0.0
    return {
        "poles": real_poles,
        "amplitudes": amps,
        "offset": offset,
        "order": len(real_poles),
        "singular_values": res["singular_values"],
        "fitted": fitted,
        "residual_rms": float(np.sqrt(np.mean(resid**2))),
        "r_squared": float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan,
        "amplitude_ratio": ratio,
    }


def bootstrap_multiexp(
    lengths: np.ndarray,
    survivals: np.ndarray,
    n_boot: int = 400,
    order: int | None = None,
    fix_offset: float | None = 0.5,
    rng: np.random.Generator | None = None,
    ci: float = 0.95,
) -> dict:
    """Sequence-level nonparametric bootstrap for the dominant pole and shape statistic.

    CRITICAL: `survivals` has shape (L, N) and resampling draws N whole SEQUENCES with
    replacement, reusing the SAME sequence indices across every length. Resampling
    individual points or shots instead would break the correlation between lengths and
    understate the uncertainty -- exactly the error that would invalidate the paper's
    finite-sample claim.
    """
    rng = np.random.default_rng() if rng is None else rng
    survivals = np.atleast_2d(np.asarray(survivals, dtype=float))
    n_seq = survivals.shape[1]
    p_boot, ratio_boot = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seq, size=n_seq)   # same sequences at every length
        y = survivals[:, idx].mean(axis=1)
        try:
            f = fit_multiexp(lengths, y, order=order, fix_offset=fix_offset)
            p_boot.append(f["poles"][0])
            ratio_boot.append(f["amplitude_ratio"])
        except Exception:
            continue
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    point = fit_multiexp(lengths, survivals.mean(axis=1), order=order, fix_offset=fix_offset)
    p_boot = np.array(p_boot)
    ratio_boot = np.array(ratio_boot)
    return {
        "point": point,
        "dominant_pole": float(point["poles"][0]),
        "dominant_pole_ci": (
            (float(np.percentile(p_boot, lo_q)), float(np.percentile(p_boot, hi_q)))
            if len(p_boot) else (np.nan, np.nan)
        ),
        "amplitude_ratio": float(point["amplitude_ratio"]),
        "amplitude_ratio_ci": (
            (float(np.percentile(ratio_boot, lo_q)), float(np.percentile(ratio_boot, hi_q)))
            if len(ratio_boot) else (np.nan, np.nan)
        ),
        "n_boot_ok": int(len(p_boot)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(4242)
    checks = {}
    m = np.arange(1, 41)          # uniform grid

    print("=== 1. recover a KNOWN single exponential ===")
    p_true = 0.985
    y = 0.5 + 0.5 * p_true**m
    f = fit_multiexp(m, y, fix_offset=0.5)
    print(f"    true p={p_true}  recovered p={f['poles'][0]:.9f}  order={f['order']}  "
          f"resid={f['residual_rms']:.2e}")
    checks["single_exp"] = abs(f["poles"][0] - p_true) < 1e-6 and f["order"] == 1

    print("\n=== 2. recover KNOWN two exponentials ===")
    p1, p2, a1, a2 = 0.99, 0.90, 0.3, 0.2
    y2 = 0.5 + a1 * p1**m + a2 * p2**m
    f2 = fit_multiexp(m, y2, order=2, fix_offset=0.5)
    got = np.sort(f2["poles"])[::-1]
    print(f"    true poles=({p1}, {p2})  recovered=({got[0]:.6f}, {got[1]:.6f})")
    print(f"    true amps =({a1}, {a2})  recovered={np.array2string(f2['amplitudes'], precision=5)}")
    checks["two_exp"] = abs(got[0] - p1) < 1e-4 and abs(got[1] - p2) < 1e-4

    print("\n=== 3. with sampling noise ===")
    devs = []
    for trial in range(20):
        yn = 0.5 + 0.5 * p_true**m + rng.normal(0, 0.002, size=len(m))
        fn = fit_multiexp(m, yn, order=1, fix_offset=0.5)
        devs.append(abs(fn["poles"][0] - p_true))
    print(f"    noise sd=0.002 -> mean |p_hat - p| = {np.mean(devs):.2e}, max = {np.max(devs):.2e}")
    checks["noisy_recovery"] = float(np.mean(devs)) < 5e-3

    print("\n=== 4. model order does NOT hallucinate a 2nd exponential on 1-exp data ===")
    orders = []
    for _ in range(40):
        yn = 0.5 + 0.5 * p_true**m + rng.normal(0, 0.003, size=len(m))
        orders.append(fit_multiexp(m, yn, fix_offset=0.5)["order"])
    frac1 = float(np.mean(np.array(orders) == 1))
    print(f"    order selected = 1 in {100*frac1:.0f}% of noisy single-exponential trials")
    checks["no_overfit"] = frac1 > 0.8

    print("\n=== 5. BOOTSTRAP COVERAGE of the dominant pole (the headline claim) ===")
    print("    nominal 95% CI; each replicate = 120 synthetic sequences")
    n_rep, n_seq, covered = 200, 120, 0
    for _ in range(n_rep):
        true_curve = 0.5 + 0.5 * p_true**m
        surv = np.clip(true_curve[:, None] + rng.normal(0, 0.03, size=(len(m), n_seq)), 0, 1)
        bs = bootstrap_multiexp(m, surv, n_boot=120, order=1, rng=rng)
        lo, hi = bs["dominant_pole_ci"]
        if lo <= p_true <= hi:
            covered += 1
    cov = covered / n_rep
    print(f"    measured coverage = {100*cov:.1f}%  (target ~95%)")
    checks["coverage"] = 0.90 <= cov <= 0.99

    print("\n=== 6. bootstrap resamples SEQUENCES, not points ===")
    # Heterogeneous population: half the sequences decay fast, half slowly. Sequence-level
    # resampling varies the fast/slow PROPORTION and so must produce a genuinely wider CI
    # than (incorrect) per-length point resampling, which averages that variation away.
    # Note a flat-outlier test would prove nothing here: adding a constant sequence changes
    # only the amplitude, and ESPRIT recovers poles from first differences, so the pole is
    # invariant by construction.
    n_half = 30
    fast = 0.5 + 0.5 * 0.95**m
    slow = 0.5 + 0.5 * 0.995**m
    surv = np.column_stack([np.repeat(slow[:, None], n_half, axis=1),
                            np.repeat(fast[:, None], n_half, axis=1)])
    seq_bs = bootstrap_multiexp(m, surv, n_boot=400, order=1, rng=rng)
    lo, hi = seq_bs["dominant_pole_ci"]
    w_seq = hi - lo

    # Deliberately WRONG bootstrap: resample independently at each length.
    pt = []
    for _ in range(400):
        y = np.array([
            surv[i, rng.integers(0, surv.shape[1], size=surv.shape[1])].mean()
            for i in range(surv.shape[0])
        ])
        try:
            pt.append(fit_multiexp(m, y, order=1, fix_offset=0.5)["poles"][0])
        except Exception:
            continue
    w_pt = float(np.percentile(pt, 97.5) - np.percentile(pt, 2.5))
    print(f"    sequence-level CI width = {w_seq:.3e}  (correct)")
    print(f"    point-level    CI width = {w_pt:.3e}  (incorrect, must be narrower)")
    print(f"    ratio = {w_seq / w_pt:.1f}x" if w_pt > 1e-12
          else "    point-level CI collapsed to zero width (understates uncertainty)")
    checks["sequence_level"] = w_seq > 1e-4 and w_seq > 2.0 * w_pt

    print("\n=== SUMMARY ===")
    for k, v in checks.items():
        print(f"  {k:20s}: {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    print(f"\nmultiexp.py self-test: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
