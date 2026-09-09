"""Phase-cycled Clifford contrasts that retain the odd phase correlation.

Units are microseconds and radians. All results are for gate-independent idle
dephasing, ideal independent Clifford twirls and ideal SPAM. The hardware gap
scan alone is not an inter-slot memory certificate (see paper/main.tex).
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from scipy.linalg import expm

from .clifford import clifford_so3, clifford_unitaries, compose_index_table, inverse_index_table
from .noise import sample_rtn_slot_phases

SETTINGS = ("++", "+-", "-+", "--", "1+", "1-", "2+", "2-")


@lru_cache(maxsize=1)
def _clifford_tables():
    return clifford_so3(clifford_unitaries()), compose_index_table(), inverse_index_table()


def rtn_phase_transfer(sigma, tau_c, delta, dead=0.0, bias=0.0):
    """Exact weighted two-state transition matrix for one biased slot."""
    if not all(np.isfinite([sigma, tau_c, delta, dead, bias])):
        raise ValueError("parameters must be finite")
    if sigma < 0 or tau_c <= 0 or delta < 0 or dead < 0:
        raise ValueError("require sigma, delta, dead >= 0 and tau_c > 0")
    gamma = 0.5 / tau_c
    q = np.array([[-gamma, gamma], [gamma, -gamma]])
    field = np.diag([sigma, -sigma])
    weighted = sum(np.exp(1j * k * bias) * expm((q + 1j * k * field) * delta)
                   for k in (-1, 0, 1)) / 3.0
    return np.real_if_close(expm(q * dead) @ weighted).real


def rtn_biased_z(lengths, sigma, tau_c, delta, dead=0.0, bias=0.0, reset=False):
    """Normalized contrast Z(m)=2*ASF(m)-1, including m=0.

    reset=True redraws the stationary environment independently at each slot,
    preserving the full single-slot phase law, not just its variance.
    """
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or np.any(lengths < 0) or np.any(lengths != lengths.astype(int)):
        raise ValueError("lengths must be a vector of nonnegative integers")
    mat = rtn_phase_transfer(sigma, tau_c, delta, dead, bias)
    pi = np.ones(2) / 2
    if reset:
        return np.power(np.sum(mat @ pi), lengths)
    return np.array([np.sum(np.linalg.matrix_power(mat, int(m)) @ pi) for m in lengths])


def rtn_cycle_truth(sigma, tau_c, delta, dead=0.0, beta=np.pi / 2):
    """Exact eight ideal survival means for the phase-cycle experiment."""
    pi = np.ones(2) / 2
    mats = {s: rtn_phase_transfer(sigma, tau_c, delta, dead, s * beta) for s in (-1, 1)}
    joint = [np.sum(mats[b] @ mats[a] @ pi) for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    single = [np.sum(mats[s] @ pi) for s in (1, -1, 1, -1)]
    return 0.5 * (1 + np.array(joint + single))


def rtn_connected_closed(sigma, tau_c, delta, dead=0.0, beta=np.pi / 2):
    """(4/9) sin(beta)^2 Cov(sin(phi_1),sin(phi_2)), exact for symmetric RTN.

    Stable in both oscillatory and overdamped regimes and at the critical point.
    Dead time advances the environment with zero coupling to the probe.
    """
    # Reuse validation, independently evaluate the closed form.
    rtn_phase_transfer(sigma, tau_c, delta, dead, beta)
    gamma = 0.5 / tau_c
    k2 = gamma * gamma - sigma * sigma
    if abs(k2) * delta * delta < 1e-10:
        response = sigma * delta * np.exp(-gamma * delta) * (1 + k2 * delta * delta / 6)
    elif k2 > 0:
        k = np.sqrt(k2)
        response = sigma * (np.exp((-gamma + k) * delta) - np.exp((-gamma - k) * delta)) / (2*k)
    else:
        k = np.sqrt(-k2)
        response = sigma * np.exp(-gamma * delta) * np.sin(k * delta) / k
    return float(4 / 9 * np.sin(beta)**2 * np.exp(-dead / tau_c) * response**2)


def explicit_survival(indices, phases):
    """Independent Bloch propagation, without using the twirl identity.

    indices and phases have shape (blocks, slots). The inverse contains only
    the sampled Clifford sequence; controlled biases remain uncompensated.
    """
    indices, phases = np.asarray(indices, int), np.asarray(phases, float)
    if indices.shape != phases.shape or indices.ndim != 2:
        raise ValueError("indices and phases must have equal (blocks, slots) shape")
    so3, table, inv = _clifford_tables()
    v = np.zeros((len(indices), 3))
    v[:, 2] = 1
    cum = np.zeros(len(indices), dtype=int)
    for k in range(indices.shape[1]):
        v = np.einsum("nij,nj->ni", so3[indices[:, k]], v)
        cum = table[indices[:, k], cum]
        c, s = np.cos(phases[:, k]), np.sin(phases[:, k])
        x, y = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0], v[:, 1] = c*x - s*y, s*x + c*y
    v = np.einsum("nij,nj->ni", so3[inv[cum]], v)
    return np.clip((1 + v[:, 2]) / 2, 0, 1)


def sample_cycle_blocks(n_blocks, rng, sigma=1.0, tau_c=2.0, delta=0.5,
                        dead=0.1, beta=np.pi/2, shots=64, control="rtn"):
    """Eight measured survivals per independent paired Clifford/trajectory block.

    All phase settings share a trajectory only in this engineered-noise model.
    This is not a claim that distinct physical hardware shots share a trajectory.
    The reset control has the same single-slot law as RTN and independent slots.
    Static control has deterministic equal phases, an essential false-positive test.
    """
    if control == "static":
        phases = np.full((n_blocks, 2), sigma * delta)
    elif control == "white":
        phases = rng.normal(0, sigma * delta, (n_blocks, 2))
    elif control in ("rtn", "reset"):
        phases = sample_rtn_slot_phases(n_blocks, 2, sigma, tau_c, delta+dead, delta, rng)
        if control == "reset":
            other = sample_rtn_slot_phases(n_blocks, 1, sigma, tau_c, delta+dead, delta, rng)
            phases[:, 1] = other[:, 0]
    else:
        raise ValueError("control must be rtn, reset, static or white")
    seq = rng.integers(0, 24, (n_blocks, 2))
    columns = [explicit_survival(seq, phases + beta*np.array(signs))
               for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    for slot in (0, 1):
        for sign in (1, -1):
            columns.append(explicit_survival(seq[:, slot:slot+1], phases[:, slot:slot+1]+sign*beta))
    prob = np.column_stack(columns)
    return prob if shots is None else rng.binomial(shots, prob) / shots
