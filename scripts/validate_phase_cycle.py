"""Reproduce phase-cycle theory, explicit-Clifford checks and held-out calibration.

Offline; no IBM credentials or QPU use. Simulation is explicitly labelled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.phase_cycle import (rtn_biased_z, rtn_cycle_truth, rtn_connected_closed,
                             explicit_survival, sample_cycle_blocks)
from estimator.phase_cycle import block_components, connected_estimate, phase_cycle_witness


def binomial_ci(k, n, alpha=.05):
    return [float(beta_dist.ppf(alpha/2, k, n-k+1)) if k else 0.,
            float(beta_dist.ppf(1-alpha/2, k+1, n-k)) if k < n else 1.]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--blocks", type=int, default=512)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    if a.figures_only:
        make_figure(json.loads(Path("data/phase_cycle_validation.json").read_text(encoding="utf-8")))
        return
    lengths = np.arange(0, 51)
    max_blind, max_formula = 0., 0.
    for tc in np.geomspace(.01, 1e4, 19):
        for d in (.03, .2, .7):
            normal = rtn_biased_z(lengths, 1., tc, d, .1)
            reset = rtn_biased_z(lengths, 1., tc, d, .1, reset=True)
            max_blind = max(max_blind, float(np.max(np.abs(normal-reset))))
            p = rtn_cycle_truth(1., tc, d, .1)
            obs = connected_estimate(block_components(np.tile(p, (2, 1))))
            max_formula = max(max_formula, abs(obs-rtn_connected_closed(1., tc, d, .1)))
    assert max_blind < 1e-11 and max_formula < 1e-12
    # Full 24x24 Clifford enumeration checks factorization, signs and inverse order.
    idx = np.array([(i,j) for i in range(24) for j in range(24)])
    max_twirl = 0.
    for angles in ((.4,.9), (-1.1,.3), (2.2,-2.7)):
        p = explicit_survival(idx, np.tile(angles,(len(idx),1))).mean()
        theory = (1+np.prod((1+2*np.cos(angles))/3))/2
        max_twirl = max(max_twirl, abs(p-theory))
    assert max_twirl < 1e-13
    # Critical damping is covered separately from the parameter grid.
    p = rtn_cycle_truth(1., .5, .5, .1)
    assert abs(connected_estimate(block_components(np.tile(p,(2,1))))-
               rtn_connected_closed(1., .5, .5, .1)) < 1e-13
    print(f"Exact checks: reset equivalence {max_blind:.2e}, closed form {max_formula:.2e}, "
          f"full Clifford enumeration {max_twirl:.2e}", flush=True)

    # Fixed design and independent seed streams. No threshold search on these trials.
    params = dict(sigma=1.6, tau_c=2., delta=.5, dead=.1, shots=64)
    rows = []
    for ci, control in enumerate(("rtn", "reset", "static", "white")):
        truth = rtn_connected_closed(params["sigma"],params["tau_c"],params["delta"],params["dead"]) if control=="rtn" else 0.
        hits, covered, finite_hits, finite_covered = 0,0,0,0
        estimates = []
        for trial in range(a.trials):
            seed = np.random.SeedSequence([20260905,ci,trial])
            data = sample_cycle_blocks(a.blocks,np.random.default_rng(seed),control=control,**params)
            w = phase_cycle_witness(data,n_boot=a.bootstrap,seed=900000+ci*a.trials+trial)
            estimates.append(w["connected"])
            hits += w["bootstrap_detected"]
            covered += w["bootstrap_ci"][0] <= truth <= w["bootstrap_ci"][1]
            finite_hits += w["finite_sample_detected"]
            finite_covered += w["finite_sample_ci"][0] <= truth <= w["finite_sample_ci"][1]
        row = dict(control=control,truth=truth,trials=a.trials,hits=hits,
                   detection_rate=hits/a.trials,detection_ci=binomial_ci(hits,a.trials),
                   bootstrap_coverage=covered/a.trials,coverage_ci=binomial_ci(covered,a.trials),
                   finite_sample_detection_rate=finite_hits/a.trials,
                   finite_sample_coverage=finite_covered/a.trials,
                   mean_estimate=float(np.mean(estimates)),estimates=estimates)
        rows.append(row)
        print(f"{control}: detection {hits}/{a.trials}, bootstrap coverage {covered}/{a.trials}, "
              f"finite-bound coverage {finite_covered}/{a.trials}",flush=True)
    out = dict(kind="simulation_only",parameters=params,blocks=a.blocks,bootstrap=a.bootstrap,
               seed_protocol="SeedSequence([20260905,control_index,trial]); separate bootstrap seeds",
               decision="two-sided 95% interval excludes zero; fixed before validation",
               max_reset_equivalence_error=max_blind,max_closed_form_error=max_formula,
               max_enumerated_twirl_error=max_twirl,calibration=rows,
               source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                              ("sim/phase_cycle.py","estimator/phase_cycle.py",__file__)})
    Path("data/phase_cycle_validation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    make_figure(out)


def make_figure(out):
    rows=out["calibration"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"axes.titlesize":8,"axes.labelsize":8,
                         "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6})
    fig, ax = plt.subplots(1,3,figsize=(7.1,2.5))
    ds=np.geomspace(.02,.8,60)
    for reset,style,label in ((False,"-","continuous RTN"),(True,"--","independent slot resets")):
        ys=[(1-rtn_biased_z([1],1.6,2.,d,.1,reset=reset)[0])/2 for d in ds]
        ax[0].loglog(ds,ys,style,label=label)
    ax[0].set(xlabel="idle duration (us)",ylabel="unbiased idle EPC",title="(a) Identical gap scans",
              xticks=[.02,.05,.1,.2,.5],xticklabels=[".02",".05",".1",".2",".5"])
    from matplotlib.ticker import NullFormatter
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].legend(fontsize=6)
    hs=np.linspace(0,8,120)
    ax[1].plot(hs,[rtn_connected_closed(1.6,2.,.5,h) for h in hs],label="continuous RTN")
    ax[1].axhline(0,color="tab:orange",ls="--",label="reset / fixed detuning")
    ax[1].set(xlabel="uncoupled separation h (us)",ylabel="connected contrast",title="(b) Exact response")
    ax[1].legend(fontsize=6)
    for i,r in enumerate(rows):
        q=np.quantile(r["estimates"],[.025,.5,.975])
        ax[2].errorbar(i,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt="o",capsize=3)
        ax[2].plot(i,r["truth"],"kx",ms=7)
    ax[2].set(xticks=range(4),xticklabels=[r["control"] for r in rows],ylabel="connected contrast",title="(c) Simulated estimates")
    ax[2].axhline(0,color="gray",lw=.7)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(f"figures/fig_phase_cycle.{ext}",dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
