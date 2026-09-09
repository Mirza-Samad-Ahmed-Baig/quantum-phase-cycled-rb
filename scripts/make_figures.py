"""
Build every figure in the manuscript from the released data files.

Each figure is regenerated from `data/*.json` alone, so the paper's plots and its dataset
cannot drift apart: if a number changes, the figure changes with it on the next run.

    .venv\\Scripts\\python.exe scripts\\make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path("data")
FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

# One palette, colour-blind safe, used consistently across every figure so that an arm keeps
# its identity from one plot to the next.
C = {
    "none": "#4C4C4C",
    "qs": "#D55E00",
    "mid": "#0072B2",
    "fast": "#009E73",
    "pred": "#999999",
    "accent": "#CC79A7",
}
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _gap_ticks(ax, gaps):
    """Tick a log gap axis at the gaps actually measured.

    Matplotlib's default log locator labels every 2x/3x/4x minor decade, which on a scan
    spanning well under two decades produces overlapping labels that are unreadable at
    column width. The measured gaps are the only x-values that mean anything here.
    """
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    g = [float(x) for x in gaps]
    ax.xaxis.set_major_locator(FixedLocator(g))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{x:.0f}" for x in g]))
    ax.xaxis.set_minor_locator(NullLocator())


def _latest(pattern: str):
    files = sorted(DATA.glob(pattern))
    return files[-1] if files else None


def _load(pattern: str):
    f = _latest(pattern)
    if f is None:
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def fig_positive_control(res, out="fig_positive_control.png"):
    """The paper's central hardware figure: injected physics in, measured exponent out."""
    if not res or not res.get("arms"):
        print("  skip positive control (no data)")
        return None
    d = np.array(res["delays_ns"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax = axes[0]
    base = np.array(res["epc_none"], dtype=float)
    bsig = np.array(res["epc_none_sigma"], dtype=float)
    ax.errorbar(d, base, yerr=bsig, fmt="o-", color=C["none"], ms=4, lw=1.2,
                capsize=2, label="device only (null arm)")
    for a in res["arms"]:
        ex = np.array(a["excess"], dtype=float)
        es = np.array(a["excess_sigma"], dtype=float)
        col = C.get(a["name"], C["accent"])
        good = np.isfinite(ex) & (ex > 0)
        ax.errorbar(d[good], ex[good], yerr=es[good], fmt="s-", color=col, ms=4, lw=1.2,
                    capsize=2,
                    label=fr"injected {a['name']}: $\tau_c$={a['tau_c_s']*1e9:.0f} ns")
    for nu, style in ((1.0, ":"), (2.0, "--")):
        ref = base[0] * (d / d[0]) ** nu
        ax.plot(d, ref, style, color=C["pred"], lw=1.0, zorder=0)
        ax.annotate(fr"$\nu={nu:.0f}$", (d[-1], ref[-1]), color=C["pred"],
                    fontsize=7, ha="left", va="center", xytext=(3, 0),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _gap_ticks(ax, d)
    ax.set_xlabel(r"idle gap $\delta$ (ns)")
    ax.set_ylabel("EPC (excess for injected arms)")
    ax.set_title("(a) paired gap scan", loc="left")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    pred = [a["nu_predicted"] for a in res["arms"]]
    meas = [a["nu_measured"] for a in res["arms"]]
    err = [a["nu_stderr"] or 0.0 for a in res["arms"]]
    names = [a["name"] for a in res["arms"]]
    lims = [0.7, 2.3]
    ax.plot(lims, lims, "-", color=C["pred"], lw=1.0, zorder=0)
    for p, m, e, n in zip(pred, meas, err, names):
        ax.errorbar([p], [m], yerr=[e], fmt="o", color=C.get(n, C["accent"]), ms=6,
                    capsize=3, label=n)
        ax.annotate(n, (p, m), fontsize=7, xytext=(5, -8), textcoords="offset points",
                    color=C.get(n, C["accent"]))
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\nu$ predicted by the injected $\tau_c$")
    ax.set_ylabel(r"$\nu$ measured by the witness")
    ax.set_title("(b) calibration", loc="left")

    fig.tight_layout()
    p = FIGS / out
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_null_and_exclusion(sens, survey, out="fig_null_exclusion.png"):
    """Native data and an absolute-adequacy audit; no unsupported exclusion."""
    if not sens:
        print("  skip null/exclusion (no sensitivity.json)")
        return None
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.5))

    ax = axes[0]
    d = np.array(sens["delta_ns"], dtype=float)
    e = np.array(sens["epc"], dtype=float)
    s = np.array(sens["epc_sigma"], dtype=float)
    ax.errorbar(d, e, yerr=s, fmt="o", color=C["none"], ms=4, capsize=2,
                label="ibm_fez q46 (stored scan)")
    # Plot the ACTUAL fits, refitted here from the released points, rather than a curve
    # anchored to one datum. Anchoring the amplitude to the largest gap made the drawn
    # curve miss the middle of the scan and understated how well the nu ~ 1 model fits.
    from estimator.witness import scaling_exponent_offset

    dd = np.geomspace(d.min(), d.max(), 200)
    free = scaling_exponent_offset(d, e, sigma_epcs=s)
    eps0 = free["eps0"]
    nu = free.get("nu", sens["nu"])
    scale = free.get("delta_scale", 1.0)
    ax.plot(dd, eps0 + free["amplitude"] * (dd / scale) ** nu, "-", color=C["fast"], lw=1.4,
            label=fr"fit $\varepsilon_0 + A\delta^\nu$: $\nu={nu:.2f}\pm{free.get('nu_stderr', sens['nu_stderr']):.2f}$")
    # Best-possible nu = 2 curve under the same offset, so the comparison is the fairest
    # one available to the correlated hypothesis rather than a straw man.
    w = 1.0 / np.clip(s, 1e-12, None) ** 2
    x2 = (d / scale) ** 2.0
    amp2 = float(np.sum(w * (e - eps0) * x2) / np.sum(w * x2 * x2))
    ax.plot(dd, eps0 + amp2 * (dd / scale) ** 2.0, "--", color=C["qs"], lw=1.4,
            label=r"best $\nu=2$ (correlated) fit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _gap_ticks(ax, d)
    ax.set_xlabel(r"idle gap $\delta$ (ns)")
    ax.set_ylabel("EPC")
    ax.set_title("(a) native EPC summaries", loc="left")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    audit = _load("sensitivity_profile_audit.json")
    if not audit:
        raise FileNotFoundError("Run scripts/audit_exclusion.py before generating the native-data figure")
    tau=np.array([r["tau_c"] for r in audit["rows"]])
    ax.plot(tau,[r["minimum_chi2"] for r in audit["rows"]],"o-",ms=3,color=C["mid"],
            label="profiled model residual")
    ax.axhline(audit["rows"][0]["gaussian_ball_cutoff"],ls="--",color=C["qs"],
               label=r"95% Gaussian-ball cutoff ($n=6$)")
    ax.legend(fontsize=7,frameon=False,loc="best")
    ax.set_xscale("log")
    ax.set_xlabel(r"fluctuator correlation time $\tau_c$ (ns)")
    ax.set_ylabel(r"minimum $\chi^2$")
    ax.set_ylim(10,21)
    ax.set_title("(b) model adequacy fails", loc="left")

    fig.tight_layout()
    p = FIGS / out
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_survey(panels, confirms=None, out="fig_survey.png"):
    """Both device surveys, with the confirmation re-measurement drawn on the same axis.

    Screening and confirmation belong in one figure: the point of the two-stage design is
    that the largest first-pass exponent moves back towards 1 when re-measured, and that is
    only visible if both values sit on the same row.
    """
    panels = [(name, [r for r in rows if r.get("nu_offset") is not None])
              for name, rows in panels if rows]
    panels = [(n, r) for n, r in panels if r]
    if not panels:
        print("  skip survey (no offset-aware exponents)")
        return None
    confirms = confirms or {}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.1),
                             squeeze=False)
    axes = axes[0]
    for ax, (name, rows) in zip(axes, panels):
        rows = sorted(rows, key=lambda r: r["nu_offset"])
        y = np.arange(len(rows))
        nu = [r["nu_offset"] for r in rows]
        er = [r.get("nu_offset_err") or 0.0 for r in rows]
        col = [C["qs"] if r.get("group") == "control" else C["mid"] for r in rows]
        ax.errorbar(nu, y, xerr=er, fmt="none", ecolor="#999999", capsize=2, lw=1.0)
        ax.scatter(nu, y, c=col, s=26, zorder=3, label="screen")
        for i, r in enumerate(rows):
            c = confirms.get(name, {}).get(r["qubit"])
            if c is None:
                continue
            ax.errorbar([c["nu_offset"]], [i], xerr=[c.get("nu_offset_err") or 0.0],
                        fmt="D", color=C["fast"], ms=5, capsize=2, zorder=4,
                        label="confirm" if i == 0 else None)
            ax.annotate("", xy=(c["nu_offset"], i), xytext=(r["nu_offset"], i),
                        arrowprops=dict(arrowstyle="->", color=C["fast"], lw=1.0,
                                        shrinkA=4, shrinkB=4))
        ax.axvline(1.0, color=C["fast"], ls=":", lw=1.2)
        ax.axvline(2.0, color=C["qs"], ls="--", lw=1.2)
        ax.axvline(1.35, color="#000000", ls="-", lw=0.9, alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"q{r['qubit']}" for r in rows])
        ax.set_xlabel(r"offset-aware exponent $\nu$")
        ax.set_title(name, loc="left")
        # Headroom above the top row so the reference-line labels and the legend never sit
        # on top of a data point.
        ax.set_ylim(-0.8, len(rows) - 0.15 + 0.9)
        top = len(rows) - 0.15 + 0.45
        ax.annotate(r"$\nu=1$", (1.0, top), fontsize=7, color=C["fast"], ha="center")
        ax.annotate(r"$\nu=2$", (2.0, top), fontsize=7, color=C["qs"], ha="center")
        ax.annotate("threshold", (1.35, -0.62), fontsize=7, ha="center", alpha=0.6)
        if confirms.get(name):
            ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    p = FIGS / out
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_controls(res, device_wvar=None, out="fig_controls.png"):
    """The two secondary witnesses, each shown against its own control.

    A single bar for the injected arm would show nothing: the claim is not that
    W_var^rep is 4, it is that the SAME statistic reads 4 with injected memory and ~1
    without, on the same qubit. Both bars have to be on the axis for that to be visible.
    """
    arms = (res or {}).get("arms", [])
    dd_rows = [(a["name"], r) for a in arms for r in a.get("dd", []) if r.get("ratio")]
    wv = [(a["name"], a["w_var_rep"]) for a in arms if a.get("w_var_rep") is not None]
    if not dd_rows and not wv:
        print("  skip controls (no DD / repeat data)")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax = axes[0]
    for nm in sorted({n for n, _ in dd_rows}):
        sub = sorted([r for n, r in dd_rows if n == nm], key=lambda r: r["delta_ns"])
        xs = [r["delta_ns"] for r in sub]
        ys = [r["ratio"] for r in sub]
        col = C.get(nm, C["accent"])
        ax.plot(xs, ys, "-o", color=col, lw=1.4, ms=6, label=f"injected {nm}")
        for r in sub:
            ax.annotate(f"{r['headroom']:.0f}$\\times$ headroom",
                        (r["delta_ns"], r["ratio"]), fontsize=7, color=col,
                        xytext=(0, -14), textcoords="offset points", ha="center")
    ax.axhline(1.0, color="#000000", lw=1.0, alpha=0.5)
    ax.annotate("no suppression", (0.02, 1.0), xycoords=("axes fraction", "data"),
                fontsize=7, va="bottom", alpha=0.6)
    ax.set_ylim(0, 1.25)
    ax.set_xscale("log")
    _gaps = sorted({r["delta_ns"] for _, r in dd_rows})
    # Pad the log axis so the per-point headroom labels are not clipped at the frame.
    ax.set_xlim(_gaps[0] * 0.80, _gaps[-1] * 1.25)
    _gap_ticks(ax, _gaps)
    ax.set_xlabel(r"idle gap $\delta$ (ns)")
    ax.set_ylabel("EPC(Hahn) / EPC(no DD)")
    ax.set_title("(a) DD suppresses injected memory", loc="left")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    labels, vals, cols = [], [], []
    for nm, v in wv:
        labels.append(f"injected\n{nm}")
        vals.append(v)
        cols.append(C.get(nm, C["accent"]))
    if device_wvar is not None:
        labels.append("device\nonly")
        vals.append(device_wvar)
        cols.append(C["none"])
    ax.bar(labels, vals, color=cols, width=0.55)
    ax.axhline(2.0, color="#000000", ls="-", lw=1.0, alpha=0.6)
    ax.annotate("decision threshold", (0.98, 2.0), xycoords=("axes fraction", "data"),
                fontsize=7, ha="right", va="bottom", alpha=0.7)
    ax.axhline(1.0, color="#000000", ls=":", lw=1.0, alpha=0.5)
    ax.annotate("no memory", (0.98, 1.0), xycoords=("axes fraction", "data"),
                fontsize=7, ha="right", va="bottom", alpha=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (i, v), fontsize=8, ha="center",
                    xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel(r"$W_{\rm var}^{\rm rep}$")
    ax.set_title("(b) excess variance responds only to real memory", loc="left")

    fig.tight_layout()
    p = FIGS / out
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def _unused_fig_dd_and_wvar(res, out="fig_controls.png"):
    """The two internal controls: DD suppression and the repeated-sequence variance."""
    arms = (res or {}).get("arms", [])
    dd_rows = [(a["name"], r) for a in arms for r in a.get("dd", []) if r.get("ratio")]
    wv = [(a["name"], a["w_var_rep"]) for a in arms if a.get("w_var_rep") is not None]
    if not dd_rows and not wv:
        print("  skip controls (no DD / repeat data)")
        return None
    n = int(bool(dd_rows)) + int(bool(wv))
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.0), squeeze=False)
    axes = axes[0]
    i = 0
    if dd_rows:
        ax = axes[i]
        i += 1
        for name in sorted({n_ for n_, _ in dd_rows}):
            sub = [r for n_, r in dd_rows if n_ == name]
            sub.sort(key=lambda r: r["delta_ns"])
            xs = [r["delta_ns"] for r in sub]
            ys = [r["ratio"] for r in sub]
            valid = [r["valid"] for r in sub]
            col = C.get(name, C["accent"])
            ax.plot(xs, ys, "-", color=col, lw=1.2, alpha=0.5)
            for x, y_, v in zip(xs, ys, valid):
                ax.plot([x], [y_], "o" if v else "x", color=col, ms=6 if v else 7,
                        mfc=col if v else "none")
            ax.plot([], [], "o-", color=col, label=name)
        ax.axhline(1.0, color="#000000", lw=0.9, alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlabel(r"idle gap $\delta$ (ns)")
        ax.set_ylabel("EPC(DD) / EPC(no DD)")
        ax.set_title("(a) DD control\n(x = outside valid regime)", loc="left")
        ax.legend(frameon=False)
    if wv:
        ax = axes[i]
        names = [n_ for n_, _ in wv]
        vals = [v for _, v in wv]
        ax.bar(names, vals, color=[C.get(n_, C["accent"]) for n_ in names], width=0.6)
        ax.axhline(1.0, color="#000000", lw=0.9, alpha=0.5)
        ax.set_ylabel(r"$W_{\rm var}^{\rm rep}$")
        ax.set_title("(b) repeated-sequence variance", loc="left")
    fig.tight_layout()
    p = FIGS / out
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def main():
    print("building figures from data/ ...")
    pos = _load("poscontrol_*_main*_results.json")
    sens = _load("sensitivity.json")
    # ibm_fez's screening survey predates the offset-aware exponent, so its rows carry
    # nu_offset = None; the follow-up job recomputed it for the qubits it re-ran.
    fez = _load("survey_ibm_fez_*_offset_results.json")
    marr = _load("survey_ibm_marrakesh_*_survey_results.json")
    panels = [("ibm_fez", fez or []), ("ibm_marrakesh", marr or [])]
    confirms = {}
    for dev, pat in (("ibm_fez", "survey_ibm_fez_*_confirm_results.json"),
                     ("ibm_marrakesh", "survey_ibm_marrakesh_*_confirm_results.json")):
        rows = _load(pat) or []
        confirms[dev] = {r["qubit"]: r for r in rows if r.get("nu_offset") is not None}
    fig_positive_control(pos)
    fig_null_and_exclusion(sens, fez)
    fig_survey(panels, confirms)
    ctrl = _load("poscontrol_*_ddrep_results.json")
    # Device-only baseline for the same statistic, from the fez confirmation run on the
    # same qubit; without it panel (b) shows a number instead of a comparison.
    fez_conf = _load("survey_ibm_fez_*_confirm_results.json") or []
    qb = (ctrl or {}).get("qubit")
    base_w = next((r["w_var_rep"] for r in fez_conf
                   if r.get("qubit") == qb and r.get("w_var_rep") is not None), None)
    fig_controls(ctrl, base_w)
    print("done.")


if __name__ == "__main__":
    main()
