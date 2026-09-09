"""Reanalyse saved EPC summaries offline; retain the original artifact as history."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from estimator.exclusion import profile_fraction


def main():
    source=Path("data/sensitivity.json")
    d=json.loads(source.read_text(encoding="utf-8"))
    cov=np.diag(np.asarray(d["epc_sigma"])**2)
    rows=[profile_fraction(d["delta_ns"],d["epc"],cov,t) for t in d["tau_c_ns"]]
    compatible=any(r["gaussian_ball_compatible"] for r in rows)
    out=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             data_level="stored EPC summaries; raw per-sequence covariance not available here",
             nuisance_method="nonnegative weighted least squares, refit offset and total idle amplitude at every fraction",
             covariance_assumption="diagonal, using stored bootstrap standard errors",
             publishable_exclusion=False,
             decision="Model fails absolute Gaussian-ball compatibility on tested tau grid; withdraw unconditional exclusion." if not compatible else
                      "Conditional diagnostic only; raw-data covariance and coverage checks required.",
             rows=rows)
    Path("data/sensitivity_profile_audit.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    for r in rows:
        print(f"tau={r['tau_c']:9.1f} ns: chi2={r['minimum_chi2']:.2f}, "
              f"conditional upper={r['conditional_profile_upper']:.3f}, compatible={r['gaussian_ball_compatible']}")
    print(out["decision"])
    # Regression checks: nuisance profiling recovers a noiseless interior model;
    # a completely zero idle component cannot identify a correlated fraction.
    x=np.array([0.,8,40,100,300,640])
    from estimator.exclusion import correlated_shape
    y=.002+.01*(.6*x/x[-1]+.4*correlated_shape(x,2000))
    r=profile_fraction(x,y,np.eye(len(x))*1e-8,2000)
    assert abs(r["best_fraction"]-.4)<1e-8 and r["minimum_chi2"]<1e-15
    z=profile_fraction(x,np.full(len(x),.002),np.eye(len(x))*1e-8,2000)
    assert z["gaussian_ball_upper"]==1.


if __name__=="__main__":main()
