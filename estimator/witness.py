"""
Legacy gap-scan screening heuristics, retained for reproduction of earlier results.

The labels below are model-dependent diagnostics, NOT memory certificates.
Symmetric RTN and an independent slot-reset model have identical mean gap scans.
Between-sequence variance also contains sequence-identity effects. Use the
phase-cycle observable in estimator/phase_cycle.py for the new connected test;
see paper/main.tex for its assumptions and the exact reset-equivalence theorem.

Three complementary statistics, each with bootstrap confidence intervals. They are
deliberately non-redundant: each one alone can be fooled, and the combination is what
makes the classification falsifiable.

1. SCALING EXPONENT nu  (the PRIMARY witness)
   Fit EPC = eps0 + A*delta**nu across the gap scan; use delta, not slot duration.
       nu -> 1  Markovian / fast noise
       nu -> 2  correlated / quasi-static noise
   Analytic prediction: nu(x) = x(1-e^-x)/(x-1+e^-x) with x = delta/tau_c, decreasing
   monotonically from 2 to 1. IMPORTANT: nu approaches 1 FROM ABOVE, so a measured nu
   slightly above 1 is the correct Markovian prediction at finite delta/tau_c, not
   evidence of correlation. The decision threshold must sit meaningfully above 1.

2. EXCESS-VARIANCE W_var  (disambiguates stochastic memory from a static detuning)
   W_var = Var_between_sequences[P] / Var_within_sequence[P], evaluated at matched ASF.
   Necessary because a DETERMINISTIC static detuning also yields nu = 2 and an exact
   single exponential, yet carries no run-to-run memory: for it W_var ~ 0, whereas
   genuinely stochastic quasi-static noise inflates it. Without this, nu alone would
   misreport a miscalibrated frequency as temporal correlation.

3. MONOTONICITY of ASF(m)  (the quantum-memory signature)
   Per arXiv:2510.13051 Corollary 4, classical-memory (CCC/CFF) models give a
   MONOTONICALLY DECREASING ASF, so an observed non-monotonicity is incompatible with
   them and points to quantum memory.

   FALSE-POSITIVE TRAP, checked explicitly: the twirled parameter
   p = (1 + 2 cos theta)/3 goes NEGATIVE for |theta| > 2*pi/3 ~ 2.094, which makes even
   purely CLASSICAL noise produce sign-alternating, non-monotonic ASF. Any
   non-monotonicity claim must therefore be accompanied by evidence that the per-slot
   phase stayed below that bound; `classify` reports this caveat rather than silently
   declaring quantum memory.

Purity: this module never imports `sim`; it consumes arrays and plain dicts.
"""
from __future__ import annotations

import numpy as np

from .multiexp import fit_multiexp

__all__ = [
    "epc_from_p",
    "fit_decay_p",
    "scaling_exponent",
    "scaling_exponent_offset",
    "excess_variance",
    "excess_variance_repeats",
    "dd_control_validity",
    "monotonicity_test",
    "gap_scan_witness",
    "classify",
    "NU_MARKOVIAN_MAX",
    "WVAR_STOCHASTIC_MIN",
]

# Decision thresholds. Calibrated on simulated ground truth (see scripts/calibrate_thresholds.py)
# and deliberately exposed as module constants so they are auditable, not buried magic numbers.
NU_MARKOVIAN_MAX = 1.35     # nu at or below this is consistent with fast/Markovian noise
WVAR_STOCHASTIC_MIN = 2.0   # W_var above this indicates genuine run-to-run (stochastic) memory


def epc_from_p(p: float, d: int = 2) -> float:
    """EPC = (1 - p)(1 - 1/d); for a qubit, (1-p)/2."""
    return float((1.0 - p) * (1.0 - 1.0 / d))


def fit_decay_p(lengths, asf, fix_offset: float = 0.5, allow_gain: bool = False) -> float:
    """Dominant decay parameter p from an ASF curve (log-linear, fixed asymptote).

    Used identically for every noise model so that cross-model comparisons of the
    fitted exponent can never be an artifact of differing fit procedures.

    A NON-DECAYING curve returns NaN rather than p > 1. When an ASF is flat to within shot
    noise the log-linear slope can come out positive, and p > 1 propagates to a NEGATIVE
    EPC -- which is not a small numerical blemish but a silently meaningless number that
    keeps flowing through ratios. It is how the ibm_fez confirm run reported
    EPC(hahn)/EPC(none) = -1.87 for a DD arm that had simply been run outside its valid
    regime and measured no decay at all. Refusing to return p > 1 turns that into a visible
    NaN. Pass `allow_gain=True` only for diagnostics that genuinely want the raw slope.
    """
    lengths = np.asarray(lengths, dtype=float)
    asf = np.asarray(asf, dtype=float)
    y = (asf - fix_offset) / (1.0 - fix_offset)
    ok = np.isfinite(y) & (y > 1e-12)
    if ok.sum() < 2:
        return np.nan
    slope = np.polyfit(lengths[ok], np.log(y[ok]), 1)[0]
    p = float(np.exp(slope))
    if p > 1.0 and not allow_gain:
        return np.nan
    return p


def scaling_exponent(deltas, epcs, tau_gate: float = 0.0) -> dict:
    """Log-log slope nu of EPC against (tau_gate + delta).

    IMPORTANT — leave `tau_gate` at 0 for the physical witness. The stochastic phase is
    injected over the IDLE WINDOW only (tau_w = delta), so the predicted power law is in
    delta itself: nu = d ln(EPC) / d ln(delta), running from 2 (quasi-static) to 1
    (Markovian). Gate error enters as a delta-INDEPENDENT offset eps_0 added to EPC, not
    as a shift inside the power law.

    Regressing against the full slot time (tau_gate + delta) instead compresses the
    x-axis (it varies by a smaller factor than delta does) and inflates nu -- e.g. a truly
    linear Markovian scan reads nu ~ 1.43 rather than ~1.0, which would be misclassified
    as correlated noise. The `tau_gate` argument is retained only for diagnostics.

    Returns dict with nu, intercept, r_squared, n_used.
    """
    deltas = np.asarray(deltas, dtype=float)
    epcs = np.asarray(epcs, dtype=float)
    t = tau_gate + deltas
    ok = np.isfinite(epcs) & (epcs > 0) & (t > 0)
    if ok.sum() < 2:
        return {"nu": np.nan, "intercept": np.nan, "r_squared": np.nan, "n_used": int(ok.sum())}
    x, y = np.log(t[ok]), np.log(epcs[ok])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "nu": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - np.sum((y - pred) ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "n_used": int(ok.sum()),
    }


def scaling_exponent_offset(deltas, epcs, eps0: float | None = None,
                            sigma_epcs=None) -> dict:
    """Offset-aware exponent: fit EPC(delta) = eps_0 + A * delta^nu.

    THIS IS THE CORRECT ESTIMATOR, and `scaling_exponent` (a pure log-log slope) is only
    valid when the delta-independent error is negligible. Gate error contributes an additive
    eps_0 that does NOT scale with the idle gap, and fitting a pure power law through
    eps_0 + A*delta^nu flattens the apparent slope.

    Measured impact on the ibm_fez 8-qubit survey: subtracting an estimated eps_0 moved the
    mean exponent from 0.563 to 1.044, and individual qubits shifted by up to +1.20
    (q53: 0.757 -> 1.961). The bias runs TOWARD 'memoryless', so it does not create false
    positives -- it MASKS real correlated noise, which is worse for discovery.

    Pass `eps0` to hold the offset fixed (e.g. measured at the smallest achievable gap, or
    from the device's gate error). Leave it None to fit all three parameters, which needs at
    least 4 delay points -- ideally including one near the minimum gap to pin eps_0 down.
    """
    from scipy.optimize import curve_fit

    d = np.asarray(deltas, dtype=float)
    y = np.asarray(epcs, dtype=float)
    ok = np.isfinite(d) & np.isfinite(y) & (d > 0)
    d, y = d[ok], y[ok]
    w = None
    if sigma_epcs is not None:
        w = np.asarray(sigma_epcs, dtype=float)[ok]

    n_free = 2 if eps0 is not None else 3
    if len(d) < n_free + 1:
        return {"nu": np.nan, "eps0": eps0, "amplitude": np.nan, "r_squared": np.nan,
                "n_used": int(len(d)), "n_free": n_free,
                "note": f"need >= {n_free + 1} delay points to fit {n_free} parameters "
                        "with a degree of freedom to spare"}

    scale = float(np.median(d))
    if eps0 is None:
        def f(x, e0, a, nu):
            return e0 + a * (x / scale) ** nu
        p0 = [max(min(y) * 0.5, 1e-9), max(np.ptp(y), 1e-9), 1.0]
        bounds = ([0.0, 0.0, 0.0], [max(y), 10 * max(y) + 1e-9, 4.0])
    else:
        def f(x, a, nu):
            return eps0 + a * (x / scale) ** nu
        p0 = [max(np.ptp(y), 1e-9), 1.0]
        bounds = ([0.0, 0.0], [10 * max(y) + 1e-9, 4.0])

    try:
        popt, pcov = curve_fit(f, d, y, p0=p0, bounds=bounds, maxfev=40000,
                               sigma=w, absolute_sigma=w is not None)
    except Exception as e:
        return {"nu": np.nan, "eps0": eps0, "amplitude": np.nan, "r_squared": np.nan,
                "n_used": int(len(d)), "n_free": n_free, "note": f"fit failed: {e}"}

    pred = f(d, *popt)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    if eps0 is None:
        e0_fit, amp, nu = popt
        nu_err = float(perr[2])
    else:
        amp, nu = popt
        e0_fit = eps0
        nu_err = float(perr[1])
    return {
        "nu": float(nu),
        "nu_stderr": nu_err,
        "eps0": float(e0_fit),
        "amplitude": float(amp),
        "delta_scale": scale,
        "r_squared": float(1.0 - np.sum((y - pred) ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "n_used": int(len(d)),
        "n_free": n_free,
    }


def dd_control_validity(epc_none: float, n_pulses: int, gate_error: float,
                        margin: float = 5.0) -> dict:
    """Is a DD arm in the regime where it can actually inform anything?

    A DD control only tests the environment if the echo train costs less error than the idle
    dephasing it removes. Otherwise the pulses' own infidelity dominates and, being
    delta-INDEPENDENT, it also flattens the gap scan -- so the control looks like it
    "destroyed the signal" when all it did was add error.

    Measured on ibm_fez the hard way: xy4 (4 pulses at ~1.7e-3 each) against an idle EPC of
    3.1e-2 at delta = 640 ns gave EPC(xy4)/EPC(none) = 15.7 / 2.85 / 1.37 at
    delta = 40/160/640 ns and collapsed the xy4 exponent to nu = -0.035. The rule below is
    that experience written down: report the ratio only where cost * margin <= idle error.

    Returns the pulse cost, the ratio it must beat, and a `valid` flag.
    """
    cost = float(n_pulses) * float(gate_error)
    e = float(epc_none)
    headroom = e / cost if cost > 0 else np.inf
    return {
        "pulse_cost": cost,
        "epc_none": e,
        "headroom": float(headroom),
        "margin_required": float(margin),
        "valid": bool(np.isfinite(headroom) and headroom >= margin),
        "note": ("DD arm is informative: idle error exceeds pulse cost by "
                 f"{headroom:.1f}x" if headroom >= margin else
                 f"DD arm NOT informative: pulse cost {cost:.2e} vs idle EPC {e:.2e} "
                 f"({headroom:.1f}x < {margin}x required)"),
    }


def excess_variance(survivals: np.ndarray, n_shots: int | None = None) -> dict:
    """W_var = between-sequence variance / within-sequence (shot) variance.

    `survivals` is (L, N) per-sequence survival probabilities. The within-sequence
    (binomial) variance is P(1-P)/n_shots; if `n_shots` is None it is estimated from
    the mean survival with a nominal 1024 shots so the ratio stays interpretable.

    A deterministic static detuning gives W_var ~ 1 (no run-to-run spread beyond shot
    noise); stochastic quasi-static noise gives W_var >> 1.
    """
    surv = np.atleast_2d(np.asarray(survivals, dtype=float))
    shots = 1024 if n_shots is None else int(n_shots)
    between = surv.var(axis=1, ddof=1)
    pbar = surv.mean(axis=1)
    within = np.maximum(pbar * (1.0 - pbar) / shots, 1e-15)
    ratios = between / within
    # Evaluate away from the ASF ceiling, where both variances vanish and the ratio is ill-conditioned.
    usable = (pbar < 0.995) & (pbar > 0.505)
    sel = ratios[usable] if usable.any() else ratios
    return {
        "w_var": float(np.median(sel)),
        "per_length": ratios,
        "n_used": int(usable.sum()) if usable.any() else int(len(ratios)),
    }



def excess_variance_repeats(survivals3d, n_shots: int) -> dict:
    """W_var^rep: variance across REPEATS OF THE SAME SEQUENCE, over shot noise.

    `survivals3d` has shape (L, N_sequences, N_repeats).

    This is the hardware-valid form of the excess-variance witness. The between-SEQUENCE
    version (`excess_variance`) is confounded on real devices, where different random
    Clifford sequences genuinely differ in fidelity: on ibm_fez it read 19.1 on a qubit
    whose nu = 1.02 indicated Markovian noise. Holding the circuit fixed and varying only
    time removes that systematic, so what remains is genuine run-to-run variation
    (stochastic memory or drift -- both are temporal correlation).

    Interpretation: W_var^rep ~ 1 means no temporal variation beyond shot noise (a
    deterministic static detuning sits here); W_var^rep >> 1 means real run-to-run memory.
    """
    s = np.asarray(survivals3d, dtype=float)
    if s.ndim != 3:
        raise ValueError(f"expected (L, N_seq, N_rep), got shape {s.shape}")
    if s.shape[2] < 2:
        return {"w_var_rep": np.nan, "n_used": 0,
                "note": "needs >= 2 repeats per sequence"}
    between_rep = np.nanvar(s, axis=2, ddof=1)          # (L, N_seq)
    pbar = np.nanmean(s, axis=2)                        # (L, N_seq)
    within = np.maximum(pbar * (1.0 - pbar) / int(n_shots), 1e-15)
    ratio = between_rep / within
    usable = (pbar < 0.995) & (pbar > 0.505) & np.isfinite(ratio)
    sel = ratio[usable] if usable.any() else ratio[np.isfinite(ratio)]
    return {
        "w_var_rep": float(np.median(sel)) if sel.size else np.nan,
        "per_length": np.nanmedian(ratio, axis=1),
        "n_used": int(usable.sum()),
    }


def monotonicity_test(
    lengths, asf, asf_stderr=None, family_alpha: float = 0.05
) -> dict:
    """Test for a statistically significant ASF INCREASE (the quantum-memory signature).

    Statistic: the maximum NET rise over all ordered pairs (i < j), i.e.
    max_{i<j} (ASF_j - ASF_i) standardized by the combined standard error.

    Two deliberate design choices, both needed to keep the false-positive rate honest:

    * NET rise over pairs, not consecutive differences. A genuine revival is *sustained*
      across several sequence lengths, whereas shot noise produces isolated one-point
      spikes that a pairwise-net statistic largely cancels.
    * BONFERRONI family-wise correction over the L(L-1)/2 pairs actually tested. Testing
      each of ~39 consecutive differences at a fixed 3 sigma inflates the family-wise
      false-positive rate to roughly 5-10%, which would make this witness fire on
      genuinely memoryless data -- the single most damaging failure mode for the paper's
      quantum-memory claim.
    """
    from scipy.stats import norm

    asf = np.asarray(asf, dtype=float)
    n = len(asf)
    d_consecutive = np.diff(asf)
    if n < 2:
        return {"monotonic_decreasing": True, "n_significant_rises": 0,
                "max_rise": 0.0, "max_rise_z": np.nan, "z_critical": np.nan, "n_tests": 0}

    if asf_stderr is None:
        # Without uncertainties, fall back to a bare sign test (documented as weak).
        return {
            "monotonic_decreasing": bool(not np.any(d_consecutive > 0)),
            "n_significant_rises": int(np.sum(d_consecutive > 0)),
            "max_rise": float(np.max(d_consecutive)),
            "max_rise_z": np.nan,
            "z_critical": np.nan,
            "n_tests": 0,
        }

    se = np.asarray(asf_stderr, dtype=float)
    rises = asf[None, :] - asf[:, None]                      # [i, j] = ASF_j - ASF_i
    comb = np.sqrt(se[:, None] ** 2 + se[None, :] ** 2)
    z = rises / np.maximum(comb, 1e-15)
    iu = np.triu_indices(n, k=1)
    z_pairs = z[iu]
    n_tests = int(len(z_pairs))
    z_crit = float(norm.isf(family_alpha / max(n_tests, 1)))
    significant = z_pairs > z_crit
    return {
        "monotonic_decreasing": bool(not np.any(significant)),
        "n_significant_rises": int(np.sum(significant)),
        "max_rise": float(np.max(rises[iu])),
        "max_rise_z": float(np.max(z_pairs)),
        "z_critical": z_crit,
        "n_tests": n_tests,
    }


def gap_scan_witness(
    datasets,
    tau_gate: float | None = None,
    n_boot: int = 300,
    rng: np.random.Generator | None = None,
    ci: float = 0.95,
) -> dict:
    """Compute the full witness from a gap scan, with sequence-level bootstrap CIs.

    `datasets` is a sequence of objects exposing .lengths, .survivals (L, N), .delta,
    .tau_gate (e.g. sim.data.ASFData) or equivalent dicts with those keys.

    Returns dict with deltas, epcs, nu (+CI), w_var (+CI), shape statistics, the
    monotonicity result, and everything `classify` needs.
    """
    rng = np.random.default_rng() if rng is None else rng

    def get(d, name):
        return d[name] if isinstance(d, dict) else getattr(d, name)

    deltas, epcs, shapes, mono_flags, all_surv, lengths_list = [], [], [], [], [], []
    for d in datasets:
        lengths = np.asarray(get(d, "lengths"))
        surv = np.atleast_2d(np.asarray(get(d, "survivals"), dtype=float))
        asf = surv.mean(axis=1)
        n = max(surv.shape[1], 2)
        se = surv.std(axis=1, ddof=1) / np.sqrt(n)
        deltas.append(float(get(d, "delta")))
        epcs.append(epc_from_p(fit_decay_p(lengths, asf)))
        try:
            shapes.append(fit_multiexp(lengths, asf, fix_offset=0.5)["amplitude_ratio"])
        except Exception:
            shapes.append(np.nan)
        mono_flags.append(monotonicity_test(lengths, asf, se))
        all_surv.append(surv)
        lengths_list.append(lengths)

    deltas = np.array(deltas)
    epcs = np.array(epcs)
    tg = float(get(datasets[0], "tau_gate")) if tau_gate is None else float(tau_gate)
    # Regress against delta itself (offset 0), NOT the full slot time: the noise is
    # injected over the idle window only, so the predicted power law is in delta.
    # See scaling_exponent's docstring -- using tau_gate here inflates nu.
    point = scaling_exponent(deltas, epcs, 0.0)

    # Sequence-level bootstrap of the WHOLE scan: resample sequences within each delta,
    # refit every EPC, then refit nu. This propagates finite-sample noise into nu itself.
    nu_boot, wvar_boot = [], []
    for _ in range(n_boot):
        e_b = []
        for surv, lengths in zip(all_surv, lengths_list):
            idx = rng.integers(0, surv.shape[1], size=surv.shape[1])
            e_b.append(epc_from_p(fit_decay_p(lengths, surv[:, idx].mean(axis=1))))
        s = scaling_exponent(deltas, np.array(e_b), 0.0)
        if np.isfinite(s["nu"]):
            nu_boot.append(s["nu"])
        w = [excess_variance(surv[:, rng.integers(0, surv.shape[1], size=surv.shape[1])])["w_var"]
             for surv in all_surv]
        wvar_boot.append(float(np.median(w)))

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    nu_boot = np.array(nu_boot)
    wvar_boot = np.array(wvar_boot)
    w_point = float(np.median([excess_variance(s)["w_var"] for s in all_surv]))

    any_nonmono = any(not m["monotonic_decreasing"] for m in mono_flags)
    return {
        "deltas": deltas,
        "epcs": epcs,
        "nu": point["nu"],
        "nu_ci": ((float(np.percentile(nu_boot, lo_q)), float(np.percentile(nu_boot, hi_q)))
                  if len(nu_boot) else (np.nan, np.nan)),
        "nu_r_squared": point["r_squared"],
        "w_var": w_point,
        "w_var_ci": ((float(np.percentile(wvar_boot, lo_q)), float(np.percentile(wvar_boot, hi_q)))
                     if len(wvar_boot) else (np.nan, np.nan)),
        "shape_amplitude_ratios": np.array(shapes, dtype=float),
        "monotonicity": mono_flags,
        "any_nonmonotonic": bool(any_nonmono),
        "n_boot_ok": int(len(nu_boot)),
    }


def classify(
    witness: dict,
    nu_markovian_max: float = NU_MARKOVIAN_MAX,
    wvar_min: float = WVAR_STOCHASTIC_MIN,
    max_slot_phase: float | None = None,
) -> dict:
    """Falsifiable classification: memoryless | classical-memory | quantum-memory.

    Decision rule (applied in order):
      1. A statistically significant ASF INCREASE at any gap -> 'quantum-memory',
         BUT only reported as such when the per-slot phase is known to satisfy
         |theta| < 2*pi/3; otherwise the verdict is flagged
         'quantum-memory-unconfirmed' because classical noise beyond that bound
         counterfeits non-monotonicity (p = (1+2cos theta)/3 turns negative).
      2. Else if the nu confidence interval lies entirely ABOVE nu_markovian_max
         -> correlated noise. Then W_var separates the sub-cases:
            W_var >= wvar_min  -> 'classical-memory'   (stochastic run-to-run memory)
            W_var <  wvar_min  -> 'static-detuning'    (deterministic; nu=2 but no memory)
      3. Else -> 'memoryless'.

    Requiring the whole CI to clear the threshold (not just the point estimate) is what
    keeps the false-positive rate low on genuinely memoryless data.
    """
    nu = witness.get("nu", np.nan)
    lo, hi = witness.get("nu_ci", (np.nan, np.nan))
    w_var = witness.get("w_var", np.nan)
    reasons = []

    if witness.get("any_nonmonotonic", False):
        phase_ok = (max_slot_phase is not None) and (max_slot_phase < 2 * np.pi / 3)
        if phase_ok:
            label = "quantum-memory"
            reasons.append(
                f"significant ASF increase with max|theta|={max_slot_phase:.3f} < 2pi/3, "
                "so classical sign-alternation is excluded"
            )
        else:
            label = "quantum-memory-unconfirmed"
            reasons.append(
                "significant ASF increase, but per-slot phase not certified below 2pi/3; "
                "strong classical noise can counterfeit non-monotonicity"
            )
        return {"label": label, "reasons": reasons, "nu": nu, "nu_ci": (lo, hi), "w_var": w_var}

    correlated = np.isfinite(lo) and lo > nu_markovian_max
    if correlated:
        reasons.append(f"nu CI [{lo:.3f}, {hi:.3f}] lies entirely above {nu_markovian_max}")
        if np.isfinite(w_var) and w_var >= wvar_min:
            reasons.append(f"W_var = {w_var:.2f} >= {wvar_min}: genuine run-to-run memory")
            label = "classical-memory"
        else:
            reasons.append(
                f"W_var = {w_var:.2f} < {wvar_min}: no run-to-run spread, consistent with a "
                "deterministic static detuning rather than stochastic memory"
            )
            label = "static-detuning"
    else:
        label = "memoryless"
        reasons.append(
            f"nu = {nu:.3f} with CI [{lo:.3f}, {hi:.3f}] not resolved above "
            f"{nu_markovian_max}; consistent with fast/Markovian noise "
            "(note nu -> 1 from above, so nu slightly > 1 is expected)"
        )
    return {"label": label, "reasons": reasons, "nu": nu, "nu_ci": (lo, hi), "w_var": w_var}


if __name__ == "__main__":
    rng = np.random.default_rng(31337)
    checks = {}
    m = np.arange(1, 41)
    tau_gate = 0.05

    def synth(deltas, epc_of_delta, n_seq=140, n_shots=1024, extra_spread=0.0, seed_rng=rng):
        """Build synthetic per-sequence datasets with a prescribed EPC(delta) law."""
        out = []
        for d in deltas:
            p = 1.0 - 2.0 * epc_of_delta(d)
            asf = 0.5 + 0.5 * p**m
            # Between-sequence spread (stochastic memory) + binomial shot noise.
            offs = seed_rng.normal(0.0, extra_spread, size=n_seq)
            curves = np.clip(asf[:, None] * (1.0 + offs[None, :]), 0.0, 1.0)
            surv = seed_rng.binomial(n_shots, curves) / n_shots
            out.append({"lengths": m, "survivals": surv, "delta": float(d), "tau_gate": tau_gate})
        return out

    deltas = np.geomspace(0.05, 0.5, 7)
    # Physically correct synthetic laws: EPC is a power of DELTA (the idle window over
    # which the stochastic phase accumulates), referenced to the largest gap.
    def epc_law(nu_true, epc_max=4e-3):
        return lambda d: epc_max * (d / deltas[-1]) ** nu_true

    print("=== 1. scaling_exponent recovers nu from synthetic power laws ===")
    for nu_true in (1.0, 1.5, 2.0):
        epcs = 1e-3 * deltas ** nu_true
        got = scaling_exponent(deltas, epcs)["nu"]
        print(f"    nu_true={nu_true}  recovered={got:.6f}")
        checks[f"nu_recover_{nu_true}"] = abs(got - nu_true) < 1e-6

    print("\n=== 2. MEMORYLESS ground truth (nu=1) -> classified memoryless ===")
    ds = synth(deltas, epc_law(1.0), extra_spread=0.0)
    w = gap_scan_witness(ds, n_boot=200, rng=rng)
    c = classify(w)
    print(f"    nu = {w['nu']:.3f}  CI = [{w['nu_ci'][0]:.3f}, {w['nu_ci'][1]:.3f}]")
    print(f"    W_var = {w['w_var']:.2f}   label = {c['label']}")
    checks["memoryless"] = c["label"] == "memoryless"

    print("\n=== 3. CLASSICAL-MEMORY ground truth (nu=2, stochastic) -> classical-memory ===")
    ds = synth(deltas, epc_law(2.0), extra_spread=0.02)
    w = gap_scan_witness(ds, n_boot=200, rng=rng)
    c = classify(w)
    print(f"    nu = {w['nu']:.3f}  CI = [{w['nu_ci'][0]:.3f}, {w['nu_ci'][1]:.3f}]")
    print(f"    W_var = {w['w_var']:.2f}   label = {c['label']}")
    checks["classical_memory"] = c["label"] == "classical-memory"

    print("\n=== 4. STATIC DETUNING (nu=2 but NO run-to-run spread) -> static-detuning ===")
    print("    This is why nu alone is insufficient and W_var is required.")
    ds = synth(deltas, epc_law(2.0), extra_spread=0.0)
    w = gap_scan_witness(ds, n_boot=200, rng=rng)
    c = classify(w)
    print(f"    nu = {w['nu']:.3f}  CI = [{w['nu_ci'][0]:.3f}, {w['nu_ci'][1]:.3f}]")
    print(f"    W_var = {w['w_var']:.2f}   label = {c['label']}")
    checks["static_detuning"] = c["label"] == "static-detuning"

    print("\n=== 5. FALSE-POSITIVE RATE on memoryless ground truth (critical) ===")
    print("    A witness that fires on Markovian data is useless; measuring the rate.")
    n_trials, false_pos = 40, 0
    for _ in range(n_trials):
        ds = synth(deltas, epc_law(1.0), n_seq=140, extra_spread=0.0)
        w = gap_scan_witness(ds, n_boot=120, rng=rng)
        if classify(w)["label"] != "memoryless":
            false_pos += 1
    fpr = false_pos / n_trials
    print(f"    false-positive rate = {100*fpr:.1f}%  ({false_pos}/{n_trials})")
    checks["low_false_positive"] = fpr <= 0.10

    print("\n=== 6. monotonicity test: shot noise must NOT trigger it ===")
    asf = 0.5 + 0.5 * 0.98**m
    se = np.full_like(asf, 0.01)
    # Family-wise false-positive rate over many independent noise realizations.
    spurious = 0
    n_mono_trials = 200
    for _ in range(n_mono_trials):
        noisy = asf + rng.normal(0, 0.01, size=len(asf))
        if not monotonicity_test(m, noisy, se)["monotonic_decreasing"]:
            spurious += 1
    mono_fpr = spurious / n_mono_trials
    mt = monotonicity_test(m, asf + rng.normal(0, 0.01, size=len(asf)), se)
    print(f"    monotone curve + noise: family-wise false-positive rate = "
          f"{100*mono_fpr:.1f}% over {n_mono_trials} trials (z_crit={mt['z_critical']:.2f}, "
          f"{mt['n_tests']} pairs tested)")
    # A genuine sustained revival must still be detected.
    rev = asf.copy()
    rev[20:] += 0.15
    mt2 = monotonicity_test(m, rev, se)
    print(f"    injected revival     -> monotonic_decreasing={mt2['monotonic_decreasing']}, "
          f"max net rise z={mt2['max_rise_z']:.2f}")
    checks["monotonicity"] = mono_fpr <= 0.05 and not mt2["monotonic_decreasing"]

    print("\n=== 7. non-monotonicity WITHOUT a phase certificate stays unconfirmed ===")
    w_fake = {"nu": 1.9, "nu_ci": (1.7, 2.1), "w_var": 5.0, "any_nonmonotonic": True}
    unconf = classify(w_fake)
    conf = classify(w_fake, max_slot_phase=0.4)
    print(f"    no certificate      -> {unconf['label']}")
    print(f"    max|theta|=0.4<2pi/3 -> {conf['label']}")
    checks["phase_caveat"] = (
        unconf["label"] == "quantum-memory-unconfirmed" and conf["label"] == "quantum-memory"
    )

    print("\n=== SUMMARY ===")
    for k, v in checks.items():
        print(f"  {k:22s}: {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    print(f"\nwitness.py self-test: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
