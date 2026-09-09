"""
Historical sensitivity calculation; its original unconditional exclusion was withdrawn.

Default execution now runs the offline nuisance-profile and adequacy audit.
The original simulation is retained behind --legacy for historical reproduction
only; its fixed-nuisance limits are not confidence-certified exclusions.

A null result is only meaningful with a stated sensitivity. This script answers, from data
already measured (no QPU time):

  1. DETECTION POWER: for which (tau_c, correlated fraction) would the protocol actually have
     fired, under the SAME detection rule used on the real data
     (offset-aware nu, detect if nu - 2*sigma_nu > NU_MARKOVIAN_MAX)?
  2. EXCLUSION LIMIT: given the measured curves show no detection, what upper limit does that
     place on a correlated component, as a function of tau_c?

Model. The idle error per Clifford splits into a delta-independent gate term and two
delta-dependent pieces:

    EPC(delta) = eps_0 + A_mark * delta            (Markovian: linear in the gap)
                       + C * f(delta / tau_c),     f(x) = x - 1 + e^-x   (correlated)

f interpolates the exact slot-phase variance: f -> x^2/2 for delta << tau_c (quasi-static,
EPC quadratic in delta) and f -> x for delta >> tau_c (Markovian, linear). At fixed total
idle error we vary only the FRACTION carried by the correlated term, which makes the limit
interpretable without needing to know sigma and tau_c separately.

Why sensitivity is tau_c-dependent: nu -> 2 only when delta << tau_c across the scan. With
gaps of 8-640 ns, a correlated component with tau_c >> 640 ns is quasi-static everywhere and
shows up strongly; one with tau_c << 8 ns is already in its Markovian limit and is
indistinguishable from ordinary dephasing at ANY amplitude. The scan therefore cannot exclude
fast correlations -- and saying so is part of an honest null.

Usage
    .venv\\Scripts\\python.exe scripts\\sensitivity.py [--qubit 46] [--trials 200]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimator.witness import (  # noqa: E402
    NU_MARKOVIAN_MAX,
    epc_from_p,
    fit_decay_p,
    scaling_exponent_offset,
)
from sim.noise import f_stable  # noqa: E402


def load_survey(tag: str = "offset"):
    """Load a survey dataset: returns (pending dict, {(qubit,dd,delay): (L,Nseq,Nrep)})."""
    import hardware.backend as hb
    from hardware.run_gapscan import _counts_from_job_result, _survival_from_counts

    cands = sorted(glob.glob(f"data/survey_*_{tag}_pending.json"))
    if not cands:
        raise FileNotFoundError(f"no data/survey_*_{tag}_pending.json")
    pend = json.loads(Path(cands[-1]).read_text(encoding="utf-8"))
    counts = _counts_from_job_result(hb.get_service().job(pend["job_id"]).result())
    L, NS, NR, SH = pend["lengths"], pend["n_sequences"], pend["n_repeats"], pend["shots"]
    data = {}
    for mt, c in zip(pend["circuit_meta"], counts):
        k = (mt["qubit"], mt["dd"], float(mt["delay"]))
        a = data.setdefault(k, np.full((len(L), NS, NR), np.nan))
        a[L.index(mt["m"]), mt["seq_index"], mt.get("repeat_index", 0)] = \
            _survival_from_counts(c, SH)
    return pend, data


def measured_epc_with_errors(pend, data, qubit: int, n_boot: int = 400, seed: int = 0):
    """EPC per delay with a sequence-level bootstrap uncertainty. Returns (delta_ns, epc, sigma)."""
    rng = np.random.default_rng(seed)
    L = np.asarray(pend["lengths"], int)
    keys = sorted([k for k in data if k[0] == qubit and k[1] == "none"], key=lambda k: k[2])
    dl, ep, sg = [], [], []
    for k in keys:
        arr = data[k]
        flat = arr.reshape(arr.shape[0], -1)
        n = flat.shape[1]
        point = epc_from_p(fit_decay_p(L, np.nanmean(flat, axis=1)))
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)          # resample whole sequences
            boots.append(epc_from_p(fit_decay_p(L, np.nanmean(flat[:, idx], axis=1))))
        boots = np.array([b for b in boots if np.isfinite(b)])
        dl.append(k[2] * 1e9)
        ep.append(point)
        sg.append(float(np.std(boots, ddof=1)) if len(boots) > 2 else np.nan)
    return np.array(dl), np.array(ep), np.array(sg)


def forward_model(delta_ns, eps0, idle_max, frac, tau_c_ns, delta_max):
    """EPC(delta) with a fraction `frac` of the idle error carried by a correlated term."""
    a_mark = (1.0 - frac) * idle_max / delta_max
    fmax = float(f_stable(delta_max / tau_c_ns))
    c = frac * idle_max / fmax if fmax > 0 else 0.0
    return eps0 + a_mark * delta_ns + c * f_stable(delta_ns / tau_c_ns)


def detection_power(delta_ns, sigma_epc, eps0, idle_max, frac, tau_c_ns,
                     trials: int, rng, nu_cut: float = NU_MARKOVIAN_MAX) -> float:
    """Monte Carlo probability that the REAL detection rule fires for this (frac, tau_c)."""
    truth = forward_model(delta_ns, eps0, idle_max, frac, tau_c_ns, delta_ns.max())
    hits = 0
    for _ in range(trials):
        obs = truth + rng.normal(0.0, sigma_epc)
        if np.any(obs <= 0):
            obs = np.clip(obs, 1e-9, None)
        r = scaling_exponent_offset(delta_ns, obs, sigma_epcs=sigma_epc)
        nu, err = r.get("nu"), r.get("nu_stderr")
        if nu is not None and np.isfinite(nu) and err is not None and np.isfinite(err):
            if nu - 2.0 * err > nu_cut:
                hits += 1
    return hits / trials


def exclusion_limit(delta_ns, epc, sigma_epc, eps0, idle_max, tau_c_ns,
                    dchi2: float = 2.71) -> float:
    """Historical fixed-nuisance slice, NOT a profiled or validated confidence limit.

    Retained to reproduce the superseded artifact; see estimator/exclusion.py.
    """
    def chi2(frac):
        pred = forward_model(delta_ns, eps0, idle_max, frac, tau_c_ns, delta_ns.max())
        return float(np.sum(((epc - pred) / sigma_epc) ** 2))

    grid = np.linspace(0.0, 1.0, 201)
    vals = np.array([chi2(f) for f in grid])
    best = float(vals.min())
    allowed = grid[vals <= best + dchi2]
    return float(allowed.max()) if len(allowed) else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="offset")
    ap.add_argument("--qubit", type=int, default=46)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce withdrawn historical calculation; not a valid exclusion")
    a = ap.parse_args(argv)

    if not a.legacy:
        from scripts.audit_exclusion import main as audit_main
        audit_main()
        return 0
    print("LEGACY REPRODUCTION ONLY: ensuing exclusion labels are superseded; see sensitivity_profile_audit.json.")

    pend, data = load_survey(a.tag)
    dl, ep, sg = measured_epc_with_errors(pend, data, a.qubit)
    print(f"MEASURED (ibm_fez q{a.qubit}, {len(dl)} gaps)")
    for d, e, s in zip(dl, ep, sg):
        print(f"  delta={d:>6.0f} ns   EPC={e:.6f} +- {s:.6f}   ({100*s/e:.1f}%)")

    fit = scaling_exponent_offset(dl, ep, sigma_epcs=sg)
    eps0 = fit["eps0"]
    idle_max = max(ep[-1] - eps0, 1e-9)
    print(f"\noffset-aware fit: nu = {fit['nu']:.3f} +- {fit['nu_stderr']:.3f}, "
          f"eps_0 = {eps0:.6f}, R^2 = {fit['r_squared']:.4f}")
    print(f"idle error at the largest gap = {idle_max:.6f}")
    print(f"detection rule: nu - 2*sigma_nu > {NU_MARKOVIAN_MAX}  -> "
          f"{'FIRES' if fit['nu'] - 2*fit['nu_stderr'] > NU_MARKOVIAN_MAX else 'does not fire'}"
          " on the real data")

    rng = np.random.default_rng(7)
    tau_grid = np.geomspace(20.0, 200000.0, 13)        # ns
    frac_grid = np.linspace(0.1, 1.0, 10)

    print(f"\nDETECTION POWER: probability the rule fires ({a.trials} trials per cell)")
    print("rows = correlated fraction of the idle error, cols = tau_c")
    hdr = "  frac |" + "".join(f"{t/1000:>7.2f}" for t in tau_grid)
    print("       |" + "  tau_c (us) ".center(len(hdr) - 8, " "))
    print(hdr)
    power = np.zeros((len(frac_grid), len(tau_grid)))
    for i, fr in enumerate(frac_grid):
        row = []
        for j, tc in enumerate(tau_grid):
            p = detection_power(dl, sg, eps0, idle_max, fr, tc, a.trials, rng)
            power[i, j] = p
            row.append(p)
        print(f"  {fr:4.1f} |" + "".join(f"{100*x:>7.0f}" for x in row))

    print("\nEXCLUSION LIMIT (95% one-sided upper limit on the correlated fraction)")
    print("  tau_c (us)   max correlated fraction allowed by the data")
    limits = []
    for tc in tau_grid:
        ul = exclusion_limit(dl, ep, sg, eps0, idle_max, tc)
        limits.append(ul)
        bar = "#" * int(round(ul * 40))
        print(f"  {tc/1000:>9.2f}   {ul:>5.2f}  {bar}")

    # Summarize where the protocol is genuinely sensitive.
    sens = [(tc, frac_grid[np.argmax(power[:, j] >= 0.95)] if np.any(power[:, j] >= 0.95)
             else np.nan) for j, tc in enumerate(tau_grid)]
    print("\nSMALLEST CORRELATED FRACTION DETECTABLE AT 95% POWER")
    for tc, fr in sens:
        print(f"  tau_c = {tc/1000:>8.2f} us   -> "
              + (f"{fr:.1f}" if np.isfinite(fr) else "not detectable at any fraction <= 1"))

    good = [(tc, fr) for tc, fr in sens if np.isfinite(fr)]
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    if good:
        tmin = min(t for t, _ in good)
        best = min(fr for _, fr in good)
        print(f"The scan (gaps {dl.min():.0f}-{dl.max():.0f} ns) is sensitive to correlated")
        print(f"noise with tau_c >~ {tmin/1000:.2f} us, down to a correlated fraction of ~{best:.1f}")
        print("of the idle error at 95% power. No detection was seen, so within that window")
        print("the data EXCLUDE a correlated component at the limits tabulated above.")
    else:
        print("No (tau_c, fraction) cell reached 95% power: with these uncertainties the scan")
        print("cannot exclude a correlated component anywhere. The null is UNINFORMATIVE and")
        print("needs smaller EPC error bars (more sequences/shots) or a wider gap range.")
    print("\nHONEST LIMITATION: correlations with tau_c well below the smallest gap are already")
    print("in their Markovian limit and are indistinguishable from ordinary dephasing at any")
    print("amplitude. This protocol bounds SLOW correlations only, and the bound is stated as")
    print("a function of tau_c rather than as a single number.")

    # ------------------------------------------------------------------ figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path("figures").mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

        ax = axes[0]
        im = ax.pcolormesh(tau_grid / 1000, frac_grid, 100 * power,
                           shading="nearest", cmap="viridis", vmin=0, vmax=100)
        cs = ax.contour(tau_grid / 1000, frac_grid, 100 * power, levels=[50, 95],
                        colors=["white", "red"], linewidths=1.6)
        ax.clabel(cs, fmt="%d%%")
        ax.set_xscale("log")
        ax.set_xlabel(r"correlation time $\tau_c$ ($\mu$s)")
        ax.set_ylabel("correlated fraction of idle error")
        ax.set_title(f"Detection power, ibm_fez q{a.qubit}\n"
                     rf"(rule: $\nu-2\sigma_\nu>{NU_MARKOVIAN_MAX}$)")
        fig.colorbar(im, ax=ax, label="power (%)")

        ax = axes[1]
        ax.semilogx(tau_grid / 1000, limits, "o-", color="tab:red")
        ax.fill_between(tau_grid / 1000, limits, 1.0, alpha=0.15, color="tab:red",
                        label="excluded (95%)")
        ax.set_xlabel(r"correlation time $\tau_c$ ($\mu$s)")
        ax.set_ylabel("max correlated fraction allowed")
        ax.set_ylim(0, 1.05)
        ax.set_title("Exclusion limit from the measured null")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()
        out = Path("figures") / "sensitivity.png"
        fig.savefig(out, dpi=140)
        print(f"\nfigure -> {out}")
    except Exception as e:
        print(f"[warn] figure failed: {e}")

    Path("data").mkdir(exist_ok=True)
    Path("data/sensitivity.json").write_text(json.dumps({
        "qubit": a.qubit, "delta_ns": dl.tolist(), "epc": ep.tolist(),
        "epc_sigma": sg.tolist(), "eps0": eps0, "nu": fit["nu"],
        "nu_stderr": fit["nu_stderr"], "tau_c_ns": tau_grid.tolist(),
        "frac_grid": frac_grid.tolist(), "power": power.tolist(),
        "exclusion_upper_limit": limits,
    }, indent=2), encoding="utf-8")
    print("saved data/sensitivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
