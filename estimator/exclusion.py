"""Profile nonnegative nuisance parameters and check absolute model adequacy.

The Gaussian-ball set has finite coverage only if the supplied covariance is
known and the Gaussian EPC approximation is correct. Here bootstrap-estimated
EPC errors make it a diagnostic. Empty sets mean model incompatibility, never a
zero upper limit or evidence excluding all noise.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls, brentq
from scipy.stats import chi2


def correlated_shape(delta, tau):
    x=np.asarray(delta,float)/tau
    if tau<=0 or np.any(x<0): raise ValueError("invalid correlation time or delay")
    # Stable independently implemented integrated exponential correlation.
    f=np.where(x<1e-4,x*x/2-x**3/6+x**4/24,x+np.expm1(-x))
    return f/f[-1]


def profile_fraction(delta, epc, covariance, tau, alpha=.05):
    delta,epc,cov=np.asarray(delta,float),np.asarray(epc,float),np.asarray(covariance,float)
    if delta.ndim!=1 or len(delta)<4 or epc.shape!=delta.shape or cov.shape!=(len(delta),len(delta)):
        raise ValueError("need >=4 delays, matching EPCs and full covariance")
    if not all(np.all(np.isfinite(x)) for x in (delta,epc,cov)) or np.any(np.diff(delta)<=0) or delta[0]<0:
        raise ValueError("finite inputs and strictly increasing nonnegative delays required")
    if not np.allclose(cov,cov.T): raise ValueError("covariance must be symmetric")
    chol=np.linalg.cholesky(cov)
    white=lambda a:solve_triangular(chol,a,lower=True)
    y=white(epc)
    linear=delta/delta[-1]
    corr=correlated_shape(delta,tau)
    X=np.column_stack((np.ones(len(delta)),linear,corr))
    best,residual=nnls(white(X),y)
    minimum=float(residual**2)
    fhat=float(best[2]/(best[1]+best[2])) if best[1]+best[2]>0 else None
    def fit(frac):
        design=np.column_stack((np.ones(len(delta)),(1-frac)*linear+frac*corr))
        coeff,res=nnls(white(design),y)
        return float(res**2),coeff
    grid=np.linspace(0,1,401)
    values=np.array([fit(f)[0] for f in grid])
    def upper(cutoff):
        allowed=np.flatnonzero(values<=cutoff)
        if not len(allowed):
            # A narrow confidence interval can lie between grid points.
            if fhat is None or fit(fhat)[0]>cutoff: return None
            start=fhat
        else: start=float(grid[allowed[-1]])
        if fit(1.)[0]<=cutoff: return 1.
        return float(brentq(lambda f:fit(f)[0]-cutoff,start,1.))
    delta_cut=float(chi2.ppf(1-2*alpha,1))
    ball_cut=float(chi2.ppf(1-alpha,len(delta)))
    return dict(tau_c=float(tau),best_fraction=fhat,best_offset=float(best[0]),
                best_linear_idle=float(best[1]),best_correlated_idle=float(best[2]),
                minimum_chi2=minimum,approx_gof_p=float(chi2.sf(minimum,len(delta)-3)),
                gaussian_ball_cutoff=ball_cut,gaussian_ball_compatible=minimum<=ball_cut,
                gaussian_ball_upper=upper(ball_cut),
                conditional_profile_upper=upper(minimum+delta_cut),
                profile_delta_cutoff=delta_cut,fraction_grid=grid.tolist(),profile_chi2=values.tolist())
