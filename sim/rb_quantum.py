"""
RB on a system qubit coupled to a small QUANTUM environment (exact joint propagation).

This is the quantum-memory counterpart to `rb_classical`. The distinction that matters
for arXiv:2510.13051 is *where the memory lives*:

* CLASSICAL memory (CCC/CFF)  -> a convex mixture of Markovian branches. Theorem 5: if
  H^ES = sum_i H_i^E (x) H_i^S with the ENVIRONMENT operators mutually COMMUTING, the
  induced process is CCC, and if all branch decay parameters coincide it is completely
  RB-blind. Corollary 2/4: such models give a MONOTONICALLY DECREASING ASF (under the
  diamond-norm sufficiency condition).
* QUANTUM memory -> the environment's own dynamics fail to commute with the coupling, so
  the branch decomposition breaks down. Then the ASF can be NON-MONOTONIC, and per
  Corollary 4 an observed non-monotonicity is incompatible with the classical-memory
  models: that is the witness.

We realize both cases in one Hamiltonian (n_env environment qubits):

    H = J sum_e Z_S (x) Z_e   +   sum_e [ omega_x X_e + (eps_z/2) Z_e ]

  omega_x = 0  ->  every environment operator in sight is Z_e, all commuting  -> CCC /
                   blind-spot regime (Theorem 5), expect monotonic ASF.
  omega_x != 0 ->  the coupling sees Z_e while the environment evolves under X_e, and
                   [Z_e, X_e] != 0  -> genuine quantum memory, non-monotonicity possible.

CRITICAL IMPLEMENTATION REQUIREMENT
-----------------------------------
The joint system+environment state is propagated across the ENTIRE sequence and the
environment is traced out only at the very END. Tracing out per slot would destroy
exactly the memory under study and silently reduce the model to a Markovian channel.
The environment IS reset between sequences, because Theorem 1 assumes a product initial
state rho (x) sigma -- physically, the environment re-equilibrates between runs.

Units: microseconds; J, omega_x, eps_z are angular frequencies (rad/us).
"""
from __future__ import annotations

import numpy as np

from .clifford import clifford_unitaries, compose_index_table, inverse_index_table
from .data import ASFData

__all__ = [
    "build_hamiltonian",
    "environment_initial_state",
    "quantum_rb",
    "quantum_rb_exact",
    "conditional_phase",
]

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_all(mats) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def build_hamiltonian(
    j_coupling: float, omega_x: float = 0.0, eps_z: float = 0.0, n_env: int = 1
) -> np.ndarray:
    """Joint Hamiltonian on (system (x) environment), dimension 2 * 2**n_env.

    H = J sum_e Z_S Z_e + sum_e [omega_x X_e + (eps_z/2) Z_e]

    Set omega_x = 0 to land in the commuting / CCC blind-spot regime of Theorem 5.
    """
    dim_env = 2**n_env
    h = np.zeros((2 * dim_env, 2 * dim_env), dtype=complex)
    for e in range(n_env):
        env_ops = [_I2] * n_env
        env_ops[e] = _Z
        h += j_coupling * np.kron(_Z, _kron_all(env_ops))
        if omega_x != 0.0:
            ops = [_I2] * n_env
            ops[e] = _X
            h += omega_x * np.kron(_I2, _kron_all(ops))
        if eps_z != 0.0:
            ops = [_I2] * n_env
            ops[e] = _Z
            h += 0.5 * eps_z * np.kron(_I2, _kron_all(ops))
    return h


def environment_initial_state(n_env: int = 1, kind: str = "ground") -> np.ndarray:
    """Environment initial state vector, dimension 2**n_env.

    'ground'      |0...0>
    'superposed'  |+...+>  -- the theory paper notes a superposed environment state can
                  matter for worst-case error, so it is exposed as a first-class option.
    """
    if kind == "ground":
        v = np.zeros(2**n_env, dtype=complex)
        v[0] = 1.0
        return v
    if kind == "superposed":
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
        return _kron_all([plus] * n_env).reshape(-1)
    raise ValueError("kind must be 'ground' or 'superposed'")


def conditional_phase(j_coupling: float, delta: float) -> float:
    """Conditional phase accumulated per slot, |theta| = 2*J*delta.

    Supplies the 2*pi/3 certificate required by estimator.witness.classify: beyond
    |theta| = 2*pi/3 the twirled parameter (1+2cos theta)/3 turns negative and even
    classical noise produces non-monotonic ASF.
    """
    return float(2.0 * abs(j_coupling) * delta)


def quantum_rb(
    lengths,
    n_sequences: int,
    delta: float,
    rng: np.random.Generator,
    j_coupling: float,
    omega_x: float = 0.0,
    eps_z: float = 0.0,
    n_env: int = 1,
    env_init: str = "ground",
    n_shots: int | None = None,
    tau_gate: float = 0.05,
) -> ASFData:
    """RB with an exactly-propagated quantum environment. Returns per-sequence survivals.

    The environment evolves during the idle gap `delta` of every slot and is NEVER traced
    out mid-sequence; it is reset only between sequences.
    """
    from scipy.linalg import expm

    lengths = np.atleast_1d(np.asarray(lengths, dtype=int))
    dim_env = 2**n_env
    dim = 2 * dim_env

    h = build_hamiltonian(j_coupling, omega_x, eps_z, n_env)
    u_idle = expm(-1j * h * delta)                       # identical for every slot

    cliffs = clifford_unitaries()
    table = compose_index_table(cliffs)
    inv = inverse_index_table(cliffs)
    # Pre-embed each Clifford into the joint space.
    cliff_full = np.array([np.kron(c, np.eye(dim_env, dtype=complex)) for c in cliffs])
    n_cliff = len(cliffs)

    env0 = environment_initial_state(n_env, env_init)
    # System starts in |0>.
    psi0 = np.kron(np.array([1.0, 0.0], dtype=complex), env0)

    # Projector onto system |0> (environment traced over) as a boolean mask.
    sys0_mask = np.zeros(dim, dtype=bool)
    sys0_mask[:dim_env] = True

    surv = np.empty((len(lengths), n_sequences), dtype=float)
    for li, m in enumerate(lengths):
        m = int(m)
        idx = rng.integers(0, n_cliff, size=(n_sequences, m))
        psi = np.tile(psi0, (n_sequences, 1))            # (N, dim)
        cum = np.zeros(n_sequences, dtype=np.int64)
        for k in range(m):
            # Ideal Clifford on the system.
            psi = np.einsum("nij,nj->ni", cliff_full[idx[:, k]], psi)
            # Joint idle evolution: memory accumulates and is NOT traced out.
            psi = psi @ u_idle.T
            cum = table[idx[:, k], cum]
        # Inverting Clifford.
        psi = np.einsum("nij,nj->ni", cliff_full[inv[cum]], psi)
        p_surv = np.sum(np.abs(psi[:, sys0_mask]) ** 2, axis=1)
        p_surv = np.clip(p_surv, 0.0, 1.0)
        if n_shots is not None:
            p_surv = rng.binomial(n_shots, p_surv) / n_shots
        surv[li] = p_surv

    return ASFData(
        lengths=lengths,
        survivals=surv,
        delta=float(delta),
        tau_gate=float(tau_gate),
        meta={
            "model": "quantum_env",
            "j_coupling": j_coupling,
            "omega_x": omega_x,
            "eps_z": eps_z,
            "n_env": n_env,
            "env_init": env_init,
            "n_shots": n_shots,
            "conditional_phase": conditional_phase(j_coupling, delta),
            "commuting_regime": bool(omega_x == 0.0 and eps_z == 0.0),
        },
    )


def quantum_rb_exact(
    lengths,
    delta: float,
    j_coupling: float,
    omega_x: float = 0.0,
    eps_z: float = 0.0,
    n_env: int = 1,
    env_init: str = "ground",
) -> np.ndarray:
    """EXACT Clifford-averaged ASF with a quantum environment. No Monte Carlo error.

    Why an exact average is available even though the environment correlates the slots:
    writing G_k = C_k...C_1, the total map factorizes into twirled slot operators
        M_k = (G_k^dag (x) I) U_idle (G_k (x) I),
    and (C_1..C_m) -> (G_1..G_m) is a bijection on Cliff^m, so the G_k are i.i.d. uniform.
    Because the G_k are INDEPENDENT ACROSS SLOTS, the average of the ordered product of
    superoperators equals the ordered product of the averaged superoperator, even though
    the shared environment prevents the *noise* from factorizing. Hence the exact
    Clifford-averaged joint channel is Lambda^m with

        Lambda(rho) = (1/24) sum_G M_G rho M_G^dag   acting on the JOINT space.

    This is what makes it legitimate to average per slot while still keeping the
    environment's memory: the memory lives in the joint state that Lambda propagates, and
    the environment is traced out only at the end.

    Returns the ASF at each requested length.
    """
    from scipy.linalg import expm

    lengths = np.atleast_1d(np.asarray(lengths, dtype=int))
    dim_env = 2**n_env
    eye_env = np.eye(dim_env, dtype=complex)

    h = build_hamiltonian(j_coupling, omega_x, eps_z, n_env)
    u_idle = expm(-1j * h * delta)
    cliffs = clifford_unitaries()
    ms = np.array([
        np.kron(c.conj().T, eye_env) @ u_idle @ np.kron(c, eye_env) for c in cliffs
    ])

    env0 = environment_initial_state(n_env, env_init)
    psi0 = np.kron(np.array([1.0, 0.0], dtype=complex), env0)
    rho = np.outer(psi0, psi0.conj())

    want = set(int(k) for k in lengths)
    out = {}
    for k in range(1, int(lengths.max()) + 1):
        # rho <- Lambda(rho), the exact Clifford average over the 24 twirled slot maps.
        rho = np.einsum("gij,jk,glk->il", ms, rho, ms.conj()) / len(cliffs)
        if k in want:
            out[k] = float(np.trace(rho[:dim_env, :dim_env]).real)
    return np.array([out[int(k)] for k in lengths])


if __name__ == "__main__":
    import time

    from .rb_classical import fit_single_exponential

    rng = np.random.default_rng(20260729)
    checks = {}
    lengths = np.arange(1, 31)
    t0 = time.time()

    print("=== 1. zero coupling recovers noiseless RB (ASF = 1) ===")
    d = quantum_rb(lengths, 60, 0.2, rng, j_coupling=0.0, omega_x=0.5)
    print(f"    ASF min = {d.asf.min():.12f}  max = {d.asf.max():.12f}")
    checks["zero_coupling"] = bool(np.allclose(d.asf, 1.0, atol=1e-9))

    print("\n=== 2. environment is NOT traced out mid-sequence ===")
    print("    A memoryless surrogate would give a strictly exponential decay; genuine")
    print("    joint propagation must deviate from it at strong coupling.")
    d_strong = quantum_rb(lengths, 400, 0.35, rng, j_coupling=1.5, omega_x=1.1)
    fit = fit_single_exponential(d_strong.lengths, d_strong.asf)
    print(f"    single-exponential fit R^2 = {fit['r_squared']:.6f}, "
          f"residual rms = {fit['residual_rms']:.4f}")
    checks["not_traced"] = fit["residual_rms"] > 1e-3

    print("\n=== 3. weak coupling -> monotonic, RB-like decay ===")
    d_weak = quantum_rb(lengths, 400, 0.1, rng, j_coupling=0.12, omega_x=0.1)
    diffs = np.diff(d_weak.asf)
    frac_down = float(np.mean(diffs < 0))
    fit_w = fit_single_exponential(d_weak.lengths, d_weak.asf)
    print(f"    fraction of decreasing steps = {frac_down:.2f}, single-exp R^2 = "
          f"{fit_w['r_squared']:.5f}, ASF(end) = {d_weak.asf[-1]:.4f}")
    checks["weak_monotonic"] = frac_down > 0.8 and d_weak.asf[-1] < d_weak.asf[0]

    print("\n=== 4. exact engine agrees with the Monte Carlo engine ===")
    ex = quantum_rb_exact(lengths, 0.35, j_coupling=1.5, omega_x=1.1)
    mc = quantum_rb(lengths, 4000, 0.35, np.random.default_rng(11),
                    j_coupling=1.5, omega_x=1.1)
    dev = float(np.max(np.abs(ex - mc.asf)))
    print(f"    max |exact - MC(4000 seq)| = {dev:.4f}")
    checks["exact_matches_mc"] = dev < 0.02

    print("\n=== 5. COMMUTING regime (omega_x = 0): Theorem 5 / CCC blind spot ===")
    print("    Environment operators are all Z_e -> CCC -> Corollary 2 predicts a")
    print("    MONOTONICALLY decreasing ASF. Checked on the exact curve (no MC noise).")
    worst_ccc = -np.inf
    for j in (0.2, 0.45, 0.9):
        for init in ("ground", "superposed"):
            a = quantum_rb_exact(np.arange(1, 61), 0.25, j_coupling=j, omega_x=0.0,
                                 eps_z=0.0, env_init=init)
            rise = float(np.max(np.diff(a)))
            worst_ccc = max(worst_ccc, rise)
            print(f"    J={j:<5} init={init:<11} max rise = {rise:+.3e}")
    print(f"    -> largest rise anywhere = {worst_ccc:+.3e} (must be ~0)")
    checks["commuting_monotonic"] = worst_ccc < 1e-9

    print("\n=== 6. NON-COMMUTING: exact scan for the quantum-memory signature ===")
    print("    Requiring a genuine rise in the EXACT curve, with the conditional phase")
    print("    kept BELOW 2pi/3 so classical sign-alternation cannot explain it.")
    lengths_long = np.arange(1, 61)
    cap = 2 * np.pi / 3
    best = None
    for n_env in (1, 2):
        for j in (0.3, 0.6, 0.9, 1.2, 1.6, 2.0):
            for wx in (0.5, 1.0, 2.0, 4.0, 8.0):
                for dl in (0.15, 0.3, 0.5):
                    theta = conditional_phase(j, dl)
                    if theta >= cap:
                        continue                      # stay inside the certified regime
                    for init in ("ground", "superposed"):
                        a = quantum_rb_exact(lengths_long, dl, j_coupling=j, omega_x=wx,
                                             n_env=n_env, env_init=init)
                        rise = float(np.max(np.diff(a)))
                        if best is None or rise > best[0]:
                            best = (rise, j, wx, dl, n_env, init, theta, a)
    rise, j, wx, dl, n_env, init, theta, a_best = best
    print(f"    strongest rise = {rise:+.3e}")
    print(f"    at J={j}, omega_x={wx}, delta={dl}, n_env={n_env}, env_init={init}")
    print(f"    conditional phase |theta| = {theta:.4f} < 2pi/3 = {cap:.4f}  (certified)")
    print(f"    ASF range: {a_best.min():.6f} .. {a_best.max():.6f}")
    checks["quantum_nonmonotonic"] = rise > 1e-6

    print("\n=== 7. classify() labels a certified rise as quantum-memory ===")
    from estimator.witness import classify  # noqa: E402

    w = {"nu": np.nan, "nu_ci": (np.nan, np.nan), "w_var": np.nan,
         "any_nonmonotonic": rise > 1e-6}
    lab_cert = classify(w, max_slot_phase=theta)["label"]
    lab_uncert = classify(w)["label"]
    print(f"    with phase certificate    -> {lab_cert}")
    print(f"    without phase certificate -> {lab_uncert}")
    checks["classify_quantum"] = (
        lab_cert == "quantum-memory" and lab_uncert == "quantum-memory-unconfirmed"
    )

    print(f"\n=== runtime: {time.time() - t0:.1f}s ===")
    print("\n=== SUMMARY ===")
    for k, v in checks.items():
        print(f"  {k:24s}: {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    print(f"\nrb_quantum.py self-test: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
