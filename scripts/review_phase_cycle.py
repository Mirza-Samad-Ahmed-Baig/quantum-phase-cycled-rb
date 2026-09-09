"""Independent review checks and an ideal, resource-matched Ramsey comparator.

The comparator is a coherent two-window Ramsey/echo construction, not a claim
that the RESOLUTE experiment has been reproduced. Both methods replay the same
classical phase pair across settings. Resource matching counts circuit shots
and sensing-window exposure, not physical gate duration or wall-clock time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from scipy.stats import beta as beta_distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.noise import sample_rtn_slot_phases
from sim.phase_cycle import explicit_survival, SETTINGS
from estimator.phase_cycle import block_components, connected_estimate


def even_transfer(q, sigma, segments):
    """Piecewise modulation, independent of the production transfer routine."""
    matrices = []
    for k in (-1, 0, 1):
        value = np.eye(2, dtype=complex)
        for duration, coupling in segments:
            value = expm((q + 1j*k*coupling*np.diag([sigma, -sigma]))*duration) @ value
        matrices.append(value)
    return np.real_if_close(sum(matrices)/3).real


def mathematical_checks():
    rng = np.random.default_rng(2026090601)
    errors = []
    for _ in range(200):
        rate, sigma = np.exp(rng.uniform(-3, 2, 2))
        q = np.array([[-rate, rate], [rate, -rate]])
        pi = np.ones(2)/2
        state, product = pi.copy(), 1.
        for _ in range(rng.integers(2, 17)):
            segments = [(rng.uniform(0, 1), rng.uniform(-2, 2)) for _ in range(3)]
            m = even_transfer(q, sigma, segments)
            state = m @ state
            product *= np.sum(m @ pi)
            errors.append(abs(state.sum()-product))
    # A non-symmetric fluctuator is outside the proposition and can distinguish resets.
    q = np.array([[-.2, 1.1], [.2, -1.1]])
    pi = np.array([1.1, .2])/1.3
    mats = [even_transfer(q, 1.8, [(d, 1.)]) for d in (.4, .8, .3)]
    state, product = pi.copy(), 1.
    for m in mats:
        state = m @ state
        product *= np.sum(m @ pi)
    asymmetric_difference = float(state.sum()-product)
    assert abs(asymmetric_difference) > 1e-5

    covariance_errors = []
    for _ in range(100):
        phi = rng.uniform(-6, 6, (20, 2))
        weights = rng.dirichlet(np.ones(20))
        beta = rng.uniform(-np.pi, np.pi)
        p = lambda x: (1+2*np.cos(x))/3
        joint = np.array([weights @ (p(phi[:, 0]+s*beta)*p(phi[:, 1]+t*beta))
                          for s, t in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
        means = np.array([weights @ ((p(phi[:, k]+beta)-p(phi[:, k]-beta))/2)
                          for k in (0, 1)])
        actual = joint @ np.array([1, -1, -1, 1])/4-np.prod(means)
        sine = np.sin(phi)
        expected = 4/9*np.sin(beta)**2*(weights @ np.prod(sine, axis=1)-np.prod(weights @ sine))
        covariance_errors.append(abs(actual-expected))
    assert max(errors) < 1e-12 and max(covariance_errors) < 1e-12

    # Verify the four Ramsey means using state amplitudes, not trigonometric identities.
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    plus = np.ones(2)/np.sqrt(2)
    rotation = lambda phi: np.diag(np.exp(np.array([-1j, 1j])*phi/2))
    ramsey_errors = []
    for a, b in rng.uniform(-6, 6, (100, 2)):
        states = (rotation(b) @ rotation(a) @ plus,
                  rotation(b) @ x @ rotation(a) @ plus,
                  rotation(a) @ plus, rotation(b) @ plus)
        measured = [float(np.real(np.vdot(s, obs @ s)))
                    for s, obs in zip(states, (x, x, y, y))]
        ramsey_errors.append(float(np.max(np.abs(np.array(measured)-
                                  [np.cos(a+b), np.cos(a-b), np.sin(a), np.sin(b)]))))
    assert max(ramsey_errors) < 1e-12
    return dict(unequal_modulated_schedules=200, max_reset_error=max(errors),
                asymmetric_counterexample_difference=asymmetric_difference,
                arbitrary_joint_laws=100, max_covariance_error=max(covariance_errors),
                max_ramsey_state_amplitude_error=max(ramsey_errors))


def comparator(trials=200, blocks=128, rb_shots=8):
    from sim.phase_cycle import rtn_connected_closed
    sigma, tau, delta, dead = 1.6, 2., .5, .1
    truth = rtn_connected_closed(sigma, tau, delta, dead)*9/4
    rows = []
    for model_index, control in enumerate(("rtn", "reset", "static", "white")):
        estimates = []
        for trial in range(trials):
            rng = np.random.default_rng(np.random.SeedSequence([2026090602, model_index, trial]))
            if control in ("rtn", "reset"):
                phases = sample_rtn_slot_phases(blocks, 2, sigma, tau, delta+dead, delta, rng)
                if control == "reset":
                    phases[:, 1] = sample_rtn_slot_phases(blocks, 1, sigma, tau, delta+dead, delta, rng)[:, 0]
            elif control == "static":
                phases = np.full((blocks, 2), sigma*delta)
            else:
                phases = rng.normal(0, sigma*delta, (blocks, 2))
            seq = rng.integers(0, 24, (blocks, 2))
            rb = [explicit_survival(seq, phases+np.pi/2*np.array(st))
                  for st in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
            for k in (0, 1):
                for s in (1, -1):
                    rb.append(explicit_survival(seq[:, k:k+1], phases[:, k:k+1]+s*np.pi/2))
            rb_p = np.column_stack(rb)
            rb_observed = rng.binomial(rb_shots, rb_p)/rb_shots
            rb_estimate = connected_estimate(block_components(rb_observed))*9/4
            a, b = phases.T
            ramsey_p = (1+np.column_stack((np.cos(a+b), np.cos(a-b), np.sin(a), np.sin(b))))/2
            ramsey_observed = rng.binomial(2*rb_shots, ramsey_p)/(2*rb_shots)
            c = np.column_stack((ramsey_observed[:, 1]-ramsey_observed[:, 0],
                                 2*ramsey_observed[:, 2]-1, 2*ramsey_observed[:, 3]-1))
            estimates.append((rb_estimate, connected_estimate(c)))
        values = np.asarray(estimates)
        target = truth if control == "rtn" else 0.
        rmse = np.sqrt(np.mean((values-target)**2, axis=0))
        # Paired trial bootstrap quantifies Monte Carlo uncertainty in the RMSE ratio.
        rng = np.random.default_rng(2026090610+model_index)
        ratios = []
        for _ in range(2000):
            idx = rng.integers(0, trials, trials)
            errors = np.sqrt(np.mean((values[idx]-target)**2, axis=0))
            ratios.append(errors[0]/errors[1])
        rows.append(dict(model=control, truth=target, estimates=values.tolist(),
                         mean=values.mean(0).tolist(), rmse=rmse.tolist(),
                         rb_to_ramsey_rmse_ratio=float(rmse[0]/rmse[1]),
                         ratio_monte_carlo_interval=np.quantile(ratios, [.025, .975]).tolist()))
    return dict(kind="ideal_simulation_comparison", trials_per_model=trials, blocks=blocks,
                rb_settings=8, rb_shots_per_setting=rb_shots,
                ramsey_settings=4, ramsey_shots_per_setting=2*rb_shots,
                circuit_shots_per_method=blocks*8*rb_shots,
                sensing_windows_per_method=blocks*12*rb_shots,
                target="Cov(sin(phi1),sin(phi2)); RB estimates multiplied by 9/4",
                assumptions="ideal gates and SPAM, replayed classical phase pairs; not matched wall time",
                parameters=dict(sigma=sigma, tau_c=tau, delta=delta, dead=dead), rows=rows)


def hardware_readout_sensitivity():
    rows = []
    for path in sorted(Path("data").glob("phase_cycle_*_counts.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        pending = json.loads(Path(raw["pending_file"].replace("\\", "/")).read_text(encoding="utf-8"))
        results = json.loads(path.with_name(path.name.replace("_counts", "_results")).read_text(encoding="utf-8"))
        design = pending["design"]
        p = np.full((4, design["blocks"], 8), np.nan)
        for counts, mt in zip(raw["counts"], pending["execution_meta"], strict=True):
            if "arm" in mt:
                p[design["arms"].index(mt["arm"]), mt["block"], SETTINGS.index(mt["setting"])]=counts.get("0", 0)/sum(counts.values())
        assert np.isfinite(p).all()
        intervals = []
        for k, n in zip(results["calibration_success"], results["calibration_shots"], strict=True):
            # Two binomial intervals, each at 97.5%, simultaneous >=95% by Bonferroni.
            lo = 0. if k == 0 else beta_distribution.ppf(.0125, k, n-k+1)
            hi = 1. if k == n else beta_distribution.ppf(.9875, k+1, n-k)
            intervals.append((float(lo), float(hi)))
        low = intervals[0][0]-intervals[1][1]
        high = intervals[0][1]-intervals[1][0]
        assert low > 0
        components = [block_components(x) for x in p]
        diff = lambda gain: connected_estimate(components[0]/gain)-connected_estimate(components[1]/gain)
        gains = np.linspace(low, high, 2001)
        values = np.array([diff(g) for g in gains])
        rows.append(dict(job_id=results["job_id"], calibration_probability_intervals=intervals,
                         simultaneous_gain_interval=[low, high], primary_at_measured_gain=diff(results["readout_gain"]),
                         primary_range_over_gain_grid=[float(values.min()), float(values.max())],
                         caveat="calibration-only sensitivity with observed experiment fixed; not a confidence interval for the primary contrast"))
    return rows


def main():
    result = dict(mathematics=mathematical_checks(), comparator=comparator(),
                  hardware_readout_sensitivity=hardware_readout_sensitivity())
    result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), Path("sim/phase_cycle.py"),
                                          Path("sim/noise.py"), Path("estimator/phase_cycle.py"))}
    Path("data/phase_cycle_review.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mathematics"], indent=2))
    for row in result["comparator"]["rows"]:
        print(row["model"], "RMSE RB/Ramsey", row["rmse"], "ratio", row["rb_to_ramsey_rmse_ratio"])
    print(json.dumps(result["hardware_readout_sensitivity"], indent=2))


if __name__ == "__main__":
    main()
