"""
Correlated-dephasing noise models for Gap-Scan RB.

TIME UNIT CONVENTION: microseconds (us) everywhere. Angular noise amplitude
`sigma` therefore has units rad/us, and rates gamma_f have units 1/us.

Slot geometry
-------------
Each RB slot k spans a period T = tau_gate + delta. The stochastic phase is
injected over the IDLE window only (window length tau_w = delta), because during
the idle the only Hamiltonian is the noise itself, so Rz(theta_idle) is *exact* —
no Magnus / time-ordering error. Intrinsic gate error is folded into a separate
delta-independent offset eps_0 downstream.

    slot k occupies [k*T, (k+1)*T);  phase accumulates over [k*T, k*T + tau_w)

Models provided
---------------
* Random telegraph noise (RTN / dichotomous): xi(t) in {+sigma, -sigma}, switching
  rate gamma_f, autocorrelation sigma^2 exp(-2 gamma_f |t|), correlation time
  tau_c = 1/(2 gamma_f), Lorentzian PSD S(w) = 4 gamma_f sigma^2/(4 gamma_f^2 + w^2).
  Trajectories are sampled EXACTLY (exponential waiting times) and the slot phase
  integral of the piecewise-constant process is evaluated in closed form via a
  piecewise-linear cumulative — never a fine-grid Riemann sum.
* Ornstein-Uhlenbeck: Gaussian, exponentially correlated, exact discrete update.
* 1/f: sum of RTN fluctuators with log-spaced switching rates.
* Quasi-static: one random phase per sequence (tau_c -> infinity).
* Markovian null: independent per-slot phases (tau_c -> 0).

Exact slot-phase covariance (Gaussian, exponentially correlated)
---------------------------------------------------------------
For C(t) = sigma^2 exp(-|t|/tau_c) and theta_k the integral of xi over a window
of length tau_w inside slot k:

    diagonal:      chi_kk = 2 sigma^2 tau_c^2 (x - 1 + e^-x),      x = tau_w/tau_c
    off-diagonal:  chi_jk = sigma^2 tau_c^2 (1 - e^-x)^2 e^{-(nT - tau_w)/tau_c},  n=|j-k|>=1

Limits: tau_w << tau_c gives chi_kk -> sigma^2 tau_w^2 (QUADRATIC in the window,
the quasi-static regime); tau_w >> tau_c gives chi_kk -> 2 Gamma_phi (tau_w - tau_c)
(LINEAR), with pure-dephasing rate Gamma_phi = sigma^2 tau_c. That quadratic-vs-linear
contrast is precisely the gap-scan signal.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "f_stable",
    "chi_matrix",
    "rtn_autocorrelation",
    "sample_rtn_slot_phases",
    "sample_ou_slot_phases",
    "sample_one_over_f_slot_phases",
    "sample_quasistatic_slot_phases",
    "sample_markovian_slot_phases",
    "sample_slot_phases",
    "tau_c_from_gamma",
    "gamma_from_tau_c",
    "gamma_phi",
    "MODELS",
]

MODELS = ("rtn", "ou", "one_over_f", "quasistatic", "markovian")


def tau_c_from_gamma(gamma_f: float) -> float:
    """RTN correlation time tau_c = 1/(2 gamma_f)."""
    return 1.0 / (2.0 * gamma_f)


def gamma_from_tau_c(tau_c: float) -> float:
    """RTN switching rate gamma_f = 1/(2 tau_c)."""
    return 1.0 / (2.0 * tau_c)


def gamma_phi(sigma: float, tau_c: float) -> float:
    """Motional-narrowing pure-dephasing rate Gamma_phi = sigma^2 tau_c."""
    return sigma**2 * tau_c


def f_stable(x):
    """f(x) = x - 1 + exp(-x), evaluated without catastrophic cancellation.

    For small x this is x^2/2 - x^3/6 + x^4/24 + O(x^5); the naive expression
    loses all precision there because x - 1 + e^-x is a difference of nearly
    equal numbers.
    """
    x = np.asarray(x, dtype=float)
    small = x < 1e-4
    safe = np.minimum(x, 700.0)
    return np.where(
        small,
        x**2 / 2 - x**3 / 6 + x**4 / 24,
        x + np.expm1(-safe),
    )


def chi_matrix(m: int, sigma: float, tau_c: float, T: float, tau_w: float) -> np.ndarray:
    """Exact slot-phase covariance matrix chi, shape (m, m), for Gaussian exp-correlated noise.

    See module docstring for the closed forms. Uses expm1 for numerical stability.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if not (0.0 < tau_w <= T):
        raise ValueError("require 0 < tau_w <= T")
    x = tau_w / tau_c
    diag = 2.0 * sigma**2 * tau_c**2 * f_stable(x)
    n = np.abs(np.subtract.outer(np.arange(m), np.arange(m))).astype(float)
    lag = np.maximum(n * T - tau_w, 0.0)
    off = (
        sigma**2
        * tau_c**2
        * np.expm1(-min(x, 700.0)) ** 2
        * np.exp(-np.minimum(lag / tau_c, 700.0))
    )
    chi = off.copy()
    np.fill_diagonal(chi, diag)
    return chi


def rtn_autocorrelation(t, sigma: float, gamma_f: float):
    """Analytic RTN autocorrelation sigma^2 exp(-2 gamma_f |t|)."""
    return sigma**2 * np.exp(-2.0 * gamma_f * np.abs(np.asarray(t, dtype=float)))


def _rtn_cumulative_phase(
    total_time: float, sigma: float, gamma_f: float, rng: np.random.Generator
):
    """Sample one exact RTN trajectory; return (breakpoints, cumulative_phase, signs).

    Waiting times are exponential with rate gamma_f (exact, no discretization).
    The cumulative phase Psi(t) = integral_0^t xi(s) ds is piecewise LINEAR, so it
    is fully described by its values at the breakpoints; the phase over any window
    is then an exact difference Psi(b) - Psi(a).
    """
    s0 = 1.0 if rng.random() < 0.5 else -1.0
    # Generate jump times in blocks until the horizon is covered.
    times = [0.0]
    t = 0.0
    while t < total_time:
        n_draw = max(16, int(1.2 * gamma_f * (total_time - t)) + 16)
        waits = rng.exponential(1.0 / gamma_f, size=n_draw)
        for w in waits:
            t += w
            times.append(t)
            if t >= total_time:
                break
    bp = np.array(times, dtype=float)
    signs = s0 * (-1.0) ** np.arange(len(bp) - 1)
    seg = np.diff(bp)
    psi = np.concatenate(([0.0], np.cumsum(sigma * signs * seg)))
    return bp, psi, signs


def _phase_at(bp: np.ndarray, psi: np.ndarray, signs: np.ndarray, sigma: float, t: float) -> float:
    """Evaluate the piecewise-linear cumulative phase Psi(t) exactly, O(log n)."""
    i = int(np.searchsorted(bp, t, side="right")) - 1
    i = min(max(i, 0), len(signs) - 1)
    return float(psi[i] + sigma * signs[i] * (t - bp[i]))


def sample_rtn_slot_phases(
    n_traj: int,
    m: int,
    sigma: float,
    tau_c: float,
    T: float,
    tau_w: float,
    rng: np.random.Generator,
    n_fluctuators: int = 1,
) -> np.ndarray:
    """Exact RTN slot phases, shape (n_traj, m).

    Correlations persist ACROSS slots because each trajectory is a single
    continuous realization spanning the whole sequence — this is the property the
    whole study depends on, so it must never be broken by re-seeding per slot.
    """
    gamma_f = gamma_from_tau_c(tau_c)
    horizon = m * T + tau_w + 1e-12
    out = np.zeros((n_traj, m), dtype=float)
    starts = np.arange(m) * T
    for r in range(n_traj):
        acc = np.zeros(m)
        for _ in range(n_fluctuators):
            bp, psi, signs = _rtn_cumulative_phase(horizon, sigma, gamma_f, rng)
            for k, a in enumerate(starts):
                acc[k] += _phase_at(bp, psi, signs, sigma, a + tau_w) - _phase_at(
                    bp, psi, signs, sigma, a
                )
        out[r] = acc
    return out


def sample_ou_slot_phases(
    n_traj: int,
    m: int,
    sigma: float,
    tau_c: float,
    T: float,
    tau_w: float,
    rng: np.random.Generator,
    n_sub: int = 64,
    n_sub_max: int = 8192,
) -> np.ndarray:
    """Ornstein-Uhlenbeck slot phases, shape (n_traj, m).

    The OU process is advanced with its EXACT discrete update
    x <- x e^{-dt/tau_c} + sigma sqrt(1 - e^{-2 dt/tau_c}) * N(0,1),
    so the marginal and correlation structure are exact at the sub-step level;
    only the within-step phase integral is discretized (trapezoid, n_sub steps).

    THE SUB-STEP MUST RESOLVE tau_c. The trapezoid rule assumes the integrand varies little
    across a step; once dt = tau_w/n_sub exceeds tau_c the process is already decorrelated
    within a single step and the rule badly over-integrates. For dt >> tau_c the trapezoid
    returns Var = sigma^2 tau_w^2 / n_sub instead of the true 2 sigma^2 tau_c tau_w, an
    overestimate of tau_w / (2 tau_c n_sub) that GROWS LINEARLY WITH tau_w -- so it does not
    merely rescale the noise, it tilts the very delta-scaling this project measures. Left at
    a fixed n_sub = 64 it inflated a hardware injection by 2.55x at tau_w = 640 ns,
    tau_c = 2 ns, and pushed the measured exponent from a predicted 0.99 to 1.46.

    n_sub is therefore raised until dt <= tau_c/4, capped at `n_sub_max`; the cap is only
    reached for correlation times far below any gap of interest, where the phase is
    effectively white and the residual bias is bounded by tau_w/(2 tau_c n_sub_max).
    """
    if tau_c > 0:
        n_sub = int(min(max(n_sub, np.ceil(4.0 * tau_w / tau_c)), n_sub_max))
    dt = tau_w / n_sub
    decay = np.exp(-dt / tau_c)
    kick = sigma * np.sqrt(max(1.0 - decay**2, 0.0))
    gap = T - tau_w
    decay_gap = np.exp(-gap / tau_c) if gap > 0 else 1.0
    kick_gap = sigma * np.sqrt(max(1.0 - decay_gap**2, 0.0)) if gap > 0 else 0.0

    out = np.empty((n_traj, m), dtype=float)
    x = rng.normal(0.0, sigma, size=n_traj)  # stationary initial condition
    for k in range(m):
        acc = np.zeros(n_traj)
        prev = x
        for _ in range(n_sub):
            x = x * decay + kick * rng.normal(size=n_traj)
            acc += 0.5 * (prev + x) * dt
            prev = x
        out[:, k] = acc
        if gap > 0:
            x = x * decay_gap + kick_gap * rng.normal(size=n_traj)
    return out


def sample_one_over_f_slot_phases(
    n_traj: int,
    m: int,
    sigma: float,
    T: float,
    tau_w: float,
    rng: np.random.Generator,
    tau_min: float = 0.05,
    tau_max: float = 500.0,
    n_fluct: int = 8,
) -> np.ndarray:
    """1/f noise as a sum of RTN fluctuators with log-spaced correlation times."""
    taus = np.geomspace(tau_min, tau_max, n_fluct)
    s_each = sigma / np.sqrt(n_fluct)
    total = np.zeros((n_traj, m), dtype=float)
    for tc in taus:
        total += sample_rtn_slot_phases(n_traj, m, s_each, float(tc), T, tau_w, rng)
    return total


def sample_quasistatic_slot_phases(
    n_traj: int,
    m: int,
    sigma: float,
    tau_w: float,
    rng: np.random.Generator,
    dichotomous: bool = False,
) -> np.ndarray:
    """Quasi-static limit: ONE random offset per trajectory, identical in every slot.

    With `dichotomous=True` the offset is +/- sigma (the exactly-single-exponential
    case); otherwise it is Gaussian (which gives algebraic, m^-1/2-like decay).
    """
    if dichotomous:
        amp = sigma * np.where(rng.random(n_traj) < 0.5, -1.0, 1.0)
    else:
        amp = rng.normal(0.0, sigma, size=n_traj)
    return np.repeat((amp * tau_w)[:, None], m, axis=1)


def sample_markovian_slot_phases(
    n_traj: int,
    m: int,
    sigma: float,
    tau_c: float,
    tau_w: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Markovian null: independent Gaussian phases per slot, variance 2 Gamma_phi tau_w.

    This is the tau_c -> 0 limit of the exponentially-correlated model, matched so
    that its dephasing rate equals Gamma_phi = sigma^2 tau_c.
    """
    var = 2.0 * gamma_phi(sigma, tau_c) * tau_w
    return rng.normal(0.0, np.sqrt(var), size=(n_traj, m))


def sample_slot_phases(
    model: str,
    n_traj: int,
    m: int,
    T: float,
    tau_w: float,
    rng: np.random.Generator,
    sigma: float = 1.0,
    tau_c: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Dispatch to a named model. Returns slot phases of shape (n_traj, m)."""
    if model == "rtn":
        return sample_rtn_slot_phases(n_traj, m, sigma, tau_c, T, tau_w, rng, **kwargs)
    if model == "ou":
        return sample_ou_slot_phases(n_traj, m, sigma, tau_c, T, tau_w, rng, **kwargs)
    if model == "one_over_f":
        return sample_one_over_f_slot_phases(n_traj, m, sigma, T, tau_w, rng, **kwargs)
    if model == "quasistatic":
        return sample_quasistatic_slot_phases(n_traj, m, sigma, tau_w, rng, **kwargs)
    if model == "markovian":
        return sample_markovian_slot_phases(n_traj, m, sigma, tau_c, tau_w, rng)
    raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")


if __name__ == "__main__":
    rng = np.random.default_rng(20260729)
    checks = {}

    print("=== f_stable: no catastrophic cancellation ===")
    for x in (1e-10, 1e-6, 1e-2, 1.0, 30.0):
        naive = x - 1 + np.exp(-x)
        print(f"  x={x:<8g} f_stable={float(f_stable(x)):.12e}  naive={naive:.12e}")
    checks["f_stable"] = abs(float(f_stable(1e-10)) - 0.5e-20) < 1e-30

    print("\n=== chi_matrix limits ===")
    sig, T = 1.0, 1.0
    qs = chi_matrix(1, sig, 1e6, T, T)[0, 0]
    print(f"  quasi-static (tau_c>>T): chi_00={qs:.9f}   expect sigma^2 tau_w^2={sig**2 * T**2:.9f}")
    tc = 1e-3
    fast = chi_matrix(1, sig, tc, T, T)[0, 0]
    expect_fast = 2 * gamma_phi(sig, tc) * (T - tc)
    print(f"  fast (tau_c<<T):         chi_00={fast:.9e}   expect 2*Gphi*(tau_w-tau_c)={expect_fast:.9e}")
    checks["chi_limits"] = (
        abs(qs - sig**2 * T**2) / (sig**2 * T**2) < 1e-6
        and abs(fast - expect_fast) / expect_fast < 1e-6
    )

    print("\n=== chi off-diagonal vs numerical double integral ===")
    sig, tcc, T, tw = 1.7, 0.9, 0.45, 0.45
    chi = chi_matrix(3, sig, tcc, T, tw)
    grid = np.linspace(0, tw, 2001)
    worst_rel = 0.0
    for n in (0, 1, 2):
        t1 = grid[:, None]
        t2 = (grid + n * T)[None, :]
        integrand = sig**2 * np.exp(-np.abs(t1 - t2) / tcc)
        num = np.trapezoid(np.trapezoid(integrand, grid, axis=1), grid)
        rel = abs(num - chi[0, n]) / abs(chi[0, n])
        worst_rel = max(worst_rel, rel)
        print(f"  n={n}: closed form={chi[0, n]:.9f}  numeric={num:.9f}  rel dev={rel:.2e}")
    checks["chi_vs_numeric"] = worst_rel < 1e-5

    print("\n=== exact RTN phase integral vs brute-force fine grid ===")
    sig, tcc = 1.3, 0.7
    gf = gamma_from_tau_c(tcc)
    bp, psi, signs = _rtn_cumulative_phase(5.0, sig, gf, np.random.default_rng(7))
    a, b = 0.37, 1.83
    exact = _phase_at(bp, psi, signs, sig, b) - _phase_at(bp, psi, signs, sig, a)
    gg = np.linspace(a, b, 400001)
    idx = np.clip(np.searchsorted(bp, gg, side="right") - 1, 0, len(signs) - 1)
    brute = float(np.trapezoid(sig * signs[idx], gg))
    print(f"  exact={exact:.12f}  fine-grid={brute:.12f}  |dev|={abs(exact - brute):.3e}")
    checks["rtn_exact_integral"] = abs(exact - brute) < 1e-4

    print("\n=== RTN sampled autocorrelation vs sigma^2 exp(-2 gamma_f |t|) ===")
    sig, tcc = 1.0, 0.5
    gf = gamma_from_tau_c(tcc)
    n_tr, dt, n_pt = 400, 0.02, 900
    ts = np.arange(n_pt) * dt
    series = np.empty((n_tr, n_pt))
    for r in range(n_tr):
        bp, psi, signs = _rtn_cumulative_phase(ts[-1] + 1.0, sig, gf, rng)
        idx = np.clip(np.searchsorted(bp, ts, side="right") - 1, 0, len(signs) - 1)
        series[r] = sig * signs[idx]
    worst_ac = 0.0
    for lag in (0, 5, 15, 30):
        emp = float(np.mean(series[:, : n_pt - lag] * series[:, lag:]))
        ana = float(rtn_autocorrelation(lag * dt, sig, gf))
        worst_ac = max(worst_ac, abs(emp - ana))
        print(f"  lag={lag * dt:.2f}us  empirical={emp:+.5f}  analytic={ana:+.5f}")
    checks["rtn_autocorr"] = worst_ac < 0.06

    print("\n=== sampled slot-phase covariance vs analytic chi (RTN) ===")
    sig, tcc, T, tw, m = 0.9, 1.4, 0.6, 0.6, 4
    ph = sample_rtn_slot_phases(6000, m, sig, tcc, T, tw, rng)
    emp_cov = np.cov(ph, rowvar=False)
    ana = chi_matrix(m, sig, tcc, T, tw)
    print(f"  empirical var(slot0)={emp_cov[0, 0]:.6f}   analytic chi_00={ana[0, 0]:.6f}")
    print(f"  empirical cov(0,1)  ={emp_cov[0, 1]:.6f}   analytic chi_01={ana[0, 1]:.6f}")
    rel0 = abs(emp_cov[0, 0] - ana[0, 0]) / ana[0, 0]
    rel1 = abs(emp_cov[0, 1] - ana[0, 1]) / ana[0, 1]
    print(f"  rel dev: diag={rel0:.3f}  off-diag={rel1:.3f}")
    checks["rtn_slot_cov"] = rel0 < 0.08 and rel1 < 0.15

    print("\n=== correlation actually persists across slots ===")
    corr01 = emp_cov[0, 1] / np.sqrt(emp_cov[0, 0] * emp_cov[1, 1])
    mk = sample_markovian_slot_phases(6000, m, sig, tcc, tw, rng)
    mk_cov = np.cov(mk, rowvar=False)
    mk_corr01 = mk_cov[0, 1] / np.sqrt(mk_cov[0, 0] * mk_cov[1, 1])
    print(f"  RTN       slot0-slot1 correlation = {corr01:+.4f}  (must be clearly > 0)")
    print(f"  Markovian slot0-slot1 correlation = {mk_corr01:+.4f}  (must be ~ 0)")
    checks["cross_slot_correlation"] = corr01 > 0.15 and abs(mk_corr01) < 0.06

    print("\n=== quasi-static: phase identical in every slot ===")
    qsp = sample_quasistatic_slot_phases(50, 5, 0.4, 0.5, rng)
    checks["quasistatic_constant"] = bool(np.allclose(qsp.std(axis=1), 0.0))
    print(f"  per-trajectory spread across slots = {qsp.std(axis=1).max():.3e} (must be 0)")

    print("\n=== all models runnable ===")
    for mdl in MODELS:
        p = sample_slot_phases(mdl, 40, 4, 0.6, 0.6, rng, sigma=0.5, tau_c=1.0)
        print(f"  {mdl:<12} shape={p.shape}  rms={p.std():.5f}")
    checks["dispatch"] = True

    print("\n=== SUMMARY ===")
    for k, v in checks.items():
        print(f"  {k:26s}: {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    print(f"\nnoise.py self-test: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
