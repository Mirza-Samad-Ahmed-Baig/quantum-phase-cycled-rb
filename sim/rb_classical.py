"""
Single-qubit Clifford RB under classically correlated dephasing, with a tunable idle gap.

Three engines, deliberately redundant so they cross-validate each other:

1. `rtn_transfer_asf`  — EXACT, O(m), no Monte Carlo. For N two-state fluctuators the
   Clifford-averaged ASF is obtained from a 2^N transfer matrix:
       A_g = [[i g sigma - gamma_f, gamma_f], [gamma_f, -i g sigma - gamma_f]],  g in {-1,0,+1}
       M   = (1/3) sum_g expm(A_0 (T - tau_w)) @ expm(A_g tau_w)
       Z(m)= 1^T M^m pi,   pi = (1/2,...,1/2),   ASF = 1/2 + Z(m)/2
2. `gaussian_det_asf`  — EXACT weak-phase closed form for Gaussian correlated noise:
       Z(m) = det(I_m + (2/3) chi)^{-1/2}
   computed via slogdet (never by forming the determinant). One expression covers the
   Markovian, crossover and quasi-static regimes.
3. `monte_carlo_rb`    — explicit random Clifford sequences propagated on the Bloch
   sphere with explicit noise trajectories, returning PER-SEQUENCE survivals for the
   estimator's sequence-level bootstrap.

Master identity underpinning 1 and 2
------------------------------------
    ASF(m) = 1/2 + 1/2 * E_noise[ prod_{k=1..m} p_k ],   p_k = (1 + 2 cos theta_k)/3

The Clifford average factorizes EXACTLY (see sim.clifford), so only the NOISE average
remains -- and that is exactly where the temporal correlation lives. The error after the
inverting Clifford is an untwirled Rz, which cannot change <Z>, so only m slots contribute.

TWO RESULTS THAT SHAPE THE PAPER (both verified in __main__)
-----------------------------------------------------------
* EXACT-EXPONENTIALITY THEOREM (one symmetric fluctuator). With S = [[0,1],[1,0]],
  S A_g S = A_{-g}, so S M S = M; the S-symmetric vector pi is an exact eigenvector, hence
  Z(m) = Z(1)^m EXACTLY, at EVERY correlation time. So for a single fluctuator the
  m-shape blind spot is TOTAL -- multi-exponential structure does NOT appear, and only the
  delta-scan can see the correlation. Genuine multi-exponentiality needs N >= 2 fluctuators
  (the S-symmetric subspace becomes 2^(N-1)-dimensional) or Gaussian-amplitude noise.
  => The delta-scaling exponent is the PRIMARY witness; ASF m-shape is secondary and
     model-dependent. Non-exponentiality tracks the SPREAD OF THE MAGNITUDE of the slot
     phase, not the correlation time.
* DELTA-SCALING. EPC(delta) = chi_00/6 = (sigma^2 tau_c/3)(delta - tau_c(1 - e^{-delta/tau_c})),
  giving EPC ~ sigma^2 delta^2/6 (QUADRATIC) for delta << tau_c and
  EPC ~ Gamma_phi delta/3 (LINEAR) for delta >> tau_c. The local exponent
      nu(x) = x(1 - e^-x)/(x - 1 + e^-x),  x = delta/tau_c
  decreases monotonically from 2 to 1. Note nu -> 1 FROM ABOVE, so a measured nu slightly
  above 1 is the correct Markovian prediction at finite delta/tau_c, not an artifact.

Units: microseconds.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .clifford import (
    clifford_so3,
    clifford_unitaries,
    compose_index_table,
    inverse_index_table,
    twirl_depolarizing_parameter,
)
from .data import ASFData
from .noise import chi_matrix, f_stable, gamma_from_tau_c, gamma_phi, sample_slot_phases

__all__ = [
    "rtn_transfer_asf",
    "gaussian_det_asf",
    "asf_from_slot_phases",
    "monte_carlo_rb",
    "fit_single_exponential",
    "epc_from_p",
    "epc_analytic",
    "nu_analytic",
    "gap_scan",
]


# --------------------------------------------------------------------------- #
# Exact engines
# --------------------------------------------------------------------------- #
def rtn_transfer_asf(
    lengths,
    sigma: float | list[float],
    tau_c: float | list[float],
    T: float,
    tau_w: float,
) -> np.ndarray:
    """Exact Clifford-averaged ASF for N symmetric telegraph fluctuators. O(max(m)).

    `sigma` and `tau_c` may be scalars (one fluctuator) or equal-length sequences.
    Returns ASF at each requested length.
    """
    sigmas = np.atleast_1d(np.asarray(sigma, dtype=float))
    taus = np.atleast_1d(np.asarray(tau_c, dtype=float))
    if sigmas.shape != taus.shape:
        raise ValueError("sigma and tau_c must have the same shape")
    lengths = np.atleast_1d(np.asarray(lengths, dtype=int))
    dead = T - tau_w

    def a_mat(s: float, g: int, gf: float) -> np.ndarray:
        return np.array(
            [[1j * g * s - gf, gf], [gf, -1j * g * s - gf]], dtype=complex
        )

    total = 0.0
    for g in (-1, 0, 1):
        kron = np.array([[1.0]], dtype=complex)
        for s, tc in zip(sigmas, taus):
            gf = gamma_from_tau_c(float(tc))
            block = expm(a_mat(float(s), 0, gf) * dead) @ expm(a_mat(float(s), g, gf) * tau_w)
            kron = np.kron(kron, block)
        total = total + kron
    m_op = total / 3.0

    n = 2 ** len(sigmas)
    v = np.full(n, 0.5 ** len(sigmas), dtype=complex)
    want = set(int(k) for k in lengths)
    z = {}
    for k in range(1, int(lengths.max()) + 1):
        v = m_op @ v
        if k in want:
            z[k] = float(v.sum().real)
    return np.array([0.5 + 0.5 * z[int(k)] for k in lengths])


def gaussian_det_asf(
    lengths, sigma: float, tau_c: float, T: float, tau_w: float
) -> np.ndarray:
    """Exact weak-phase ASF for Gaussian exp-correlated noise: Z = det(I + (2/3)chi)^{-1/2}.

    Uses slogdet so that large m stays numerically stable.
    """
    out = []
    for k in np.atleast_1d(np.asarray(lengths, dtype=int)):
        chi = chi_matrix(int(k), sigma, tau_c, T, tau_w)
        _, logdet = np.linalg.slogdet(np.eye(int(k)) + (2.0 / 3.0) * chi)
        out.append(0.5 + 0.5 * np.exp(-0.5 * logdet))
    return np.array(out)


def asf_from_slot_phases(phases: np.ndarray) -> float:
    """Clifford-averaged ASF from sampled slot phases, shape (n_traj, m).

    Applies the exact twirl p_k = (1+2cos theta_k)/3 and averages the product over
    trajectories. Exact in the Clifford average; Monte Carlo only over the noise.
    """
    p = twirl_depolarizing_parameter(phases)
    return float(0.5 + 0.5 * np.mean(np.prod(p, axis=1)))


# --------------------------------------------------------------------------- #
# Explicit-sequence Monte Carlo (per-sequence survivals for bootstrap)
# --------------------------------------------------------------------------- #
def monte_carlo_rb(
    lengths,
    model: str,
    n_sequences: int,
    T: float,
    tau_w: float,
    rng: np.random.Generator,
    sigma: float = 1.0,
    tau_c: float = 1.0,
    n_shots: int | None = None,
    tau_gate: float | None = None,
    **model_kwargs,
) -> ASFData:
    """Explicit random-Clifford RB with correlated dephasing; per-sequence survivals.

    Each sequence gets its OWN noise trajectory (as on real hardware, where every
    shot group sees a fresh stretch of the environment's history). Within a
    sequence the noise stays correlated across slots.

    If `n_shots` is given, survivals are additionally binomial-sampled to emulate
    finite sampling; otherwise exact Bloch survival probabilities are returned.
    """
    lengths = np.atleast_1d(np.asarray(lengths, dtype=int))
    so3 = clifford_so3(clifford_unitaries())
    table = compose_index_table()
    inv = inverse_index_table()
    n_cliff = so3.shape[0]

    surv = np.empty((len(lengths), n_sequences), dtype=float)
    for li, m in enumerate(lengths):
        m = int(m)
        # Random Clifford sequence per RB sequence.
        idx = rng.integers(0, n_cliff, size=(n_sequences, m))
        # Correlated noise phases: one trajectory per sequence.
        phases = sample_slot_phases(
            model, n_sequences, m, T, tau_w, rng, sigma=sigma, tau_c=tau_c, **model_kwargs
        )
        # Propagate Bloch vectors from |0> = (0,0,1).
        v = np.zeros((n_sequences, 3))
        v[:, 2] = 1.0
        cum = np.zeros(n_sequences, dtype=np.int64)  # cumulative Clifford index
        for k in range(m):
            r = so3[idx[:, k]]                            # (N,3,3)
            v = np.einsum("nij,nj->ni", r, v)
            cum = table[idx[:, k], cum]                   # G_k = C_k ... C_1
            c, s = np.cos(phases[:, k]), np.sin(phases[:, k])
            vx, vy = v[:, 0].copy(), v[:, 1].copy()
            v[:, 0] = c * vx - s * vy                     # Rz(theta_k) on the Bloch sphere
            v[:, 1] = s * vx + c * vy
        # Inverting Clifford.
        v = np.einsum("nij,nj->ni", so3[inv[cum]], v)
        p_surv = np.clip(0.5 * (1.0 + v[:, 2]), 0.0, 1.0)
        if n_shots is not None:
            p_surv = rng.binomial(n_shots, p_surv) / n_shots
        surv[li] = p_surv

    delta = tau_w
    tg = (T - tau_w) if tau_gate is None else tau_gate
    return ASFData(
        lengths=lengths,
        survivals=surv,
        delta=float(delta),
        tau_gate=float(tg),
        meta={
            "model": model,
            "sigma": sigma,
            "tau_c": tau_c,
            "T": T,
            "tau_w": tau_w,
            "n_shots": n_shots,
            "n_sequences": n_sequences,
        },
    )


# --------------------------------------------------------------------------- #
# Fitting / analytic references
# --------------------------------------------------------------------------- #
def fit_single_exponential(
    lengths, asf, fix_asymptote: bool = True
) -> dict:
    """Fit ASF(m) = A p^m + B. Returns dict with p, A, B, r_squared, residual_rms.

    With `fix_asymptote` (default) B is pinned to 1/2 and A to 1/2, matching the
    ideal-SPAM single-qubit form, and p is obtained by linear regression of
    log(2*ASF - 1) on m. This is the *same* estimator used for every noise model so
    that cross-model comparisons of the fitted exponent cannot be an artifact of
    using different fitting procedures.
    """
    lengths = np.asarray(lengths, dtype=float)
    asf = np.asarray(asf, dtype=float)
    if fix_asymptote:
        y = 2.0 * asf - 1.0
        ok = y > 1e-12
        if ok.sum() < 2:
            return {"p": np.nan, "A": 0.5, "B": 0.5, "r_squared": np.nan,
                    "residual_rms": np.nan, "n_used": int(ok.sum())}
        slope, intercept = np.polyfit(lengths[ok], np.log(y[ok]), 1)
        p = float(np.exp(slope))
        model = 0.5 + 0.5 * np.exp(intercept) * p ** lengths
        resid = asf - model
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((asf - asf.mean()) ** 2))
        return {
            "p": p,
            "A": float(0.5 * np.exp(intercept)),
            "B": 0.5,
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "residual_rms": float(np.sqrt(np.mean(resid**2))),
            "n_used": int(ok.sum()),
        }
    # Free three-parameter fit.
    from scipy.optimize import curve_fit

    def f(mm, a, pp, b):
        return a * pp**mm + b

    popt, _ = curve_fit(f, lengths, asf, p0=[0.5, 0.99, 0.5], maxfev=20000)
    resid = asf - f(lengths, *popt)
    ss_tot = float(np.sum((asf - asf.mean()) ** 2))
    return {
        "p": float(popt[1]),
        "A": float(popt[0]),
        "B": float(popt[2]),
        "r_squared": float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan,
        "residual_rms": float(np.sqrt(np.mean(resid**2))),
        "n_used": len(lengths),
    }


def epc_from_p(p: float, d: int = 2) -> float:
    """Error per Clifford from the depolarizing parameter: EPC = (1-p)(1 - 1/d)."""
    return float((1.0 - p) * (1.0 - 1.0 / d))


def epc_analytic(delta: float, sigma: float, tau_c: float) -> float:
    """EPC(delta) = chi_00/6 = (sigma^2 tau_c^2/3) f(delta/tau_c), f(x)=x-1+e^-x."""
    return float(sigma**2 * tau_c**2 * f_stable(delta / tau_c) / 3.0)


def nu_analytic(x) -> np.ndarray:
    """Local delta-exponent nu(x) = x(1-e^-x)/(x-1+e^-x), x = delta/tau_c. Runs 2 -> 1."""
    x = np.asarray(x, dtype=float)
    return x * (-np.expm1(-x)) / f_stable(x)


# --------------------------------------------------------------------------- #
# Gap scan
# --------------------------------------------------------------------------- #
def gap_scan(
    deltas,
    lengths,
    tau_gate: float,
    sigma: float,
    tau_c: float,
    engine: str = "gaussian",
    n_fluctuators: int = 1,
) -> dict:
    """Sweep the idle gap and return fitted EPC(delta) plus the scaling exponent nu.

    engine: 'gaussian' (det formula) or 'rtn' (exact transfer matrix).
    Returns dict with deltas, epc, p, r_squared, nu (log-log slope), and nu_pred.
    """
    deltas = np.atleast_1d(np.asarray(deltas, dtype=float))
    lengths = np.atleast_1d(np.asarray(lengths, dtype=int))
    epc, ps, r2 = [], [], []
    for d in deltas:
        T = tau_gate + d
        if engine == "gaussian":
            asf = gaussian_det_asf(lengths, sigma, tau_c, T, d)
        elif engine == "rtn":
            s = [sigma / np.sqrt(n_fluctuators)] * n_fluctuators
            t = [tau_c] * n_fluctuators
            asf = rtn_transfer_asf(lengths, s, t, T, d)
        else:
            raise ValueError("engine must be 'gaussian' or 'rtn'")
        fit = fit_single_exponential(lengths, asf)
        ps.append(fit["p"])
        r2.append(fit["r_squared"])
        epc.append(epc_from_p(fit["p"]))
    epc = np.array(epc)
    good = np.isfinite(epc) & (epc > 0)
    nu = float(np.polyfit(np.log(deltas[good]), np.log(epc[good]), 1)[0]) if good.sum() > 1 else np.nan
    return {
        "deltas": deltas,
        "epc": epc,
        "p": np.array(ps),
        "r_squared": np.array(r2),
        "nu": nu,
        "nu_pred_midpoint": float(nu_analytic(np.median(deltas) / tau_c)),
    }


if __name__ == "__main__":
    checks = {}
    rng = np.random.default_rng(20260729)
    lengths = np.array([1, 2, 5, 10, 20, 35, 50, 75])

    print("=== 1. EXACT-EXPONENTIALITY THEOREM: one symmetric fluctuator ===")
    print("    (Z(m) must equal Z(1)^m exactly, at EVERY correlation time)")
    worst_dev = 0.0
    for tc_over_T in (0.01, 0.1, 1.0, 10.0, 1e6):
        T = tau_w = 0.5
        tc = tc_over_T * T
        asf = rtn_transfer_asf([1, 10, 50], 0.8, tc, T, tau_w)
        z1, z10, z50 = 2 * asf - 1
        dev = max(abs(z10 - z1**10), abs(z50 - z1**50))
        worst_dev = max(worst_dev, dev)
        print(f"    tau_c/T={tc_over_T:<8g} Z(1)={z1:.12f}  Z(10)={z10:.12f}  "
              f"Z(1)^10={z1**10:.12f}  max dev={dev:.2e}")
    checks["exact_exponentiality_N1"] = worst_dev < 1e-10
    print(f"    -> single fluctuator is EXACTLY single-exponential: worst dev {worst_dev:.2e}")

    print("\n=== 2. N=2 fluctuators DO become multi-exponential ===")
    T = tau_w = 0.5
    asf2 = rtn_transfer_asf([1, 50], [0.6, 0.6], [25.0, 25.0], T, tau_w)
    z1, z50 = 2 * asf2 - 1
    ratio = z50 / z1**50
    print(f"    Z(50)/Z(1)^50 = {ratio:.4f}  (must be clearly > 1; spec golden ~5.6)")
    checks["multiexp_N2"] = ratio > 2.0

    print("\n=== 3. transfer matrix vs Gaussian det vs trajectory Monte Carlo ===")
    sigma, tc, T = 0.35, 1.2, 0.5
    asf_det = gaussian_det_asf(lengths, sigma, tc, T, T)
    ph = sample_slot_phases("ou", 4000, int(lengths.max()), T, T, rng, sigma=sigma, tau_c=tc)
    asf_mc = np.array([asf_from_slot_phases(ph[:, :int(k)]) for k in lengths])
    dev = float(np.max(np.abs(asf_det - asf_mc)))
    for k, a, b in zip(lengths, asf_det, asf_mc):
        print(f"    m={int(k):3d}  det={a:.6f}  MC(OU)={b:.6f}  dev={abs(a-b):.4f}")
    print(f"    -> max deviation {dev:.4f}")
    checks["engines_agree"] = dev < 0.03

    print("\n=== 4. THE BLIND SPOT: correlated noise still looks single-exponential in m ===")
    T = 0.5
    asf_corr = rtn_transfer_asf(lengths, 0.8, 50.0, T, T)   # tau_c >> T, strongly correlated
    fit_corr = fit_single_exponential(lengths, asf_corr)
    print(f"    strongly-correlated RTN: single-exp fit R^2 = {fit_corr['r_squared']:.9f}, "
          f"residual rms = {fit_corr['residual_rms']:.3e}")
    mono = bool(np.all(np.diff(asf_corr) < 1e-12))
    print(f"    ASF monotonically decreasing: {mono}")
    print("    -> standard RB reveals NOTHING: the m-shape is a clean exponential")
    checks["blind_spot"] = fit_corr["r_squared"] > 0.9999 and mono

    print("\n=== 5. THE GAP-SCAN SIGNAL: nu ~ 2 (correlated) vs nu ~ 1 (Markovian) ===")
    print("    Same fitting code, same lengths, same statistics for both -> the")
    print("    difference cannot come from the fitting procedure.")
    tau_gate = 0.05
    sigma = 0.30
    deltas = np.geomspace(0.02, 0.5, 7)

    # Correlated: tau_c >> delta over the whole scan -> quasi-static -> nu -> 2
    tc_corr = 200.0
    sc_corr = gap_scan(deltas, lengths, tau_gate, sigma, tc_corr, engine="gaussian")
    # Markovian-like: tau_c << delta over the whole scan -> nu -> 1
    tc_fast = 2e-4
    sigma_fast = sigma * np.sqrt(tc_corr / tc_fast) / 40.0   # keep EPC in a sane band
    sc_fast = gap_scan(deltas, lengths, tau_gate, sigma_fast, tc_fast, engine="gaussian")

    print(f"    correlated (tau_c={tc_corr}us >> delta): nu = {sc_corr['nu']:.4f}   "
          f"(predicted {sc_corr['nu_pred_midpoint']:.4f})")
    print(f"    fast/Markovian (tau_c={tc_fast}us << delta): nu = {sc_fast['nu']:.4f}   "
          f"(predicted {sc_fast['nu_pred_midpoint']:.4f})")
    print(f"    separation = {sc_corr['nu'] - sc_fast['nu']:.4f}")
    checks["gap_scan_separates"] = (
        sc_corr["nu"] > 1.8 and sc_fast["nu"] < 1.15
        and (sc_corr["nu"] - sc_fast["nu"]) > 0.6
    )

    print("\n=== 6. nu(x) golden values (spec regression targets) ===")
    goldens = {1e-4: 1.99997, 0.01: 1.99667, 0.1: 1.96722, 1.0: 1.71828,
               2.149: 1.50002, 10.0: 1.11106, 100.0: 1.01010}
    worst_g = 0.0
    for x, want in goldens.items():
        got = float(nu_analytic(x))
        worst_g = max(worst_g, abs(got - want))
        print(f"    nu({x:<8g}) = {got:.6f}   spec golden {want}")
    checks["nu_goldens"] = worst_g < 2e-4
    print(f"    -> worst deviation from spec goldens: {worst_g:.2e}")

    print("\n=== 7. Monte Carlo with explicit Clifford sequences (per-sequence data) ===")
    data = monte_carlo_rb(
        [1, 5, 10, 20, 40], "rtn", 300, 0.55, 0.5, rng,
        sigma=0.30, tau_c=100.0, n_shots=1024, tau_gate=0.05,
    )
    print(f"    survivals shape = {data.survivals.shape} (lengths x sequences)")
    print(f"    ASF = {np.array2string(data.asf, precision=5)}")
    print(f"    SE  = {np.array2string(data.asf_stderr, precision=5)}")
    fit_mc = fit_single_exponential(data.lengths, data.asf)
    print(f"    fitted p = {fit_mc['p']:.6f}  EPC = {epc_from_p(fit_mc['p']):.6f}")
    checks["monte_carlo"] = (
        data.survivals.shape == (5, 300)
        and np.all(data.asf <= 1.0 + 1e-9)
        and np.all(data.asf >= 0.0)
        and data.asf[0] > data.asf[-1]
    )

    print("\n=== 8. noiseless limit recovers ASF = 1 ===")
    clean = monte_carlo_rb([1, 10, 30], "markovian", 200, 0.55, 0.5, rng,
                           sigma=0.0, tau_c=1.0, tau_gate=0.05)
    print(f"    ASF (sigma=0) = {np.array2string(clean.asf, precision=12)}")
    checks["noiseless"] = bool(np.allclose(clean.asf, 1.0, atol=1e-9))

    print("\n=== SUMMARY ===")
    for k, v in checks.items():
        print(f"  {k:28s}: {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    print(f"\nrb_classical.py self-test: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
