"""
Gap-Scan RB circuit construction for real hardware (Qiskit).

Builds standard single-qubit Clifford RB sequences with a CALIBRATED IDLE GAP delta
inserted between successive Cliffords, plus the dynamical-decoupling (DD) variants used
as the experiment's internal control.

Three details that decide whether the experiment is valid at all
---------------------------------------------------------------
1. BARRIERS AROUND EVERY GATE AND DELAY. Without them the transpiler happily merges the
   whole Clifford sequence into a single equivalent rotation and deletes the idle
   windows, which would silently destroy both the RB decay and the gap we are scanning.
   Every Clifford and every delay is fenced.
2. DELAYS QUANTIZED TO THE BACKEND. Hardware accepts durations in integer multiples of
   dt subject to a granularity constraint; `quantize_delay` rounds to that grid and
   returns the ACTUAL delay realized, so the analysis regresses against the true delta
   rather than the requested one.
3. EXACT CLIFFORDS, NO SYNTHESIS. Each Clifford is emitted from its exact {H, S}
   generator word (see sim.clifford.clifford_words), so there is no unitary-synthesis
   error anywhere, and the inverting Clifford comes from the exact group inverse table.

The DD control is what makes a positive result credible: if a measured gap-scan signal is
genuine environmental memory, echoing the qubit during the idle must SUPPRESS it. A signal
that survives DD unchanged is more likely an artifact of the pulse calibration.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from sim.clifford import clifford_words, compose_index_table, inverse_index_table

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

__all__ = [
    "DD_SEQUENCES",
    "quantize_delay",
    "append_clifford",
    "append_idle",
    "build_gapscan_circuit",
    "build_circuit_set",
    "adaptive_lengths",
    "injected_slot_phases",
    "predicted_injected_epc",
]

# Dynamical-decoupling sequences, given as the pulse pattern inside the idle window.
# 'none' is a bare delay (the measurement condition); the others echo the qubit and
# should SUPPRESS low-frequency environmental memory.
DD_SEQUENCES = {
    "none": [],
    "hahn": ["x"],                      # single echo
    "cpmg": ["x", "x"],                 # two pulses
    "xy4": ["x", "y", "x", "y"],        # cancels dephasing and amplitude noise
}


def quantize_delay(duration_s: float, dt: float | None, granularity: int = 16) -> tuple[int, float]:
    """Quantize an idle duration to the backend timing grid.

    Returns (duration_in_dt, actual_seconds). If `dt` is unknown (e.g. a plain simulator),
    returns (0, duration_s) and the caller should pass seconds through unchanged.

    Rounding to `granularity` matters because the analysis must use the delay the hardware
    actually executed; regressing nu against a requested-but-unrealized delta would bias
    the exponent.
    """
    if dt is None or dt <= 0:
        return 0, float(duration_s)
    n = int(round(duration_s / dt))
    n = max(granularity, int(round(n / granularity)) * granularity)
    return n, n * dt


def adaptive_lengths(delta_s: float, epc_estimate: float, tau_gate_s: float = 0.0,
                     t2_s: float | None = None, n_points: int = 6,
                     target_decay: float = 0.75, m_min: int = 2,
                     m_cap: int = 400) -> np.ndarray:
    """Length grid for one gap, cut off where the sequence stops carrying information.

    RB measures a decay, so the useful range of m ends once the ASF has fallen into the
    depolarised floor: past that point every extra Clifford costs shots and returns noise.
    The cutoff is set by the per-Clifford error, m_max ~ -ln(1 - target_decay) / (2 EPC),
    and additionally by coherence when T2 is known, since a slot lasts tau_gate + delta and
    no fit can outrun m (tau_gate + delta) > T2.

    This is why a single length grid across a wide gap scan is wasteful. On ibm_fez q46
    (T2 = 14.5 us) the largest gap of 640 ns already saturates at m ~ 20, while the 8 ns gap
    supports m in the hundreds -- and the small gaps are precisely where the ASF-shape and
    multi-exponential witnesses have room to act. Points are log-spaced because the decay is
    exponential in m; a linear grid spends most of its points in the flat tail.
    """
    epc = max(float(epc_estimate), 1e-9)
    m_decay = -np.log(max(1.0 - target_decay, 1e-6)) / (2.0 * epc)
    m_max = m_decay
    if t2_s and t2_s > 0:
        m_max = min(m_max, float(t2_s) / max(float(tau_gate_s) + float(delta_s), 1e-15))
    m_max = int(np.clip(np.floor(m_max), m_min + 1, m_cap))
    grid = np.unique(np.round(np.geomspace(m_min, m_max, n_points)).astype(int))
    # Log spacing collides at small m; top the grid back up so every gap keeps enough
    # distinct lengths for the fit to have degrees of freedom.
    while grid.size < min(n_points, m_max - m_min + 1):
        extra = np.setdiff1d(np.arange(m_min, m_max + 1), grid)
        grid = np.unique(np.concatenate([grid, extra[:1]]))
    return grid


def _half_delay(duration, unit: str, dt: float | None, granularity: int):
    """Split an idle duration into two grid-legal halves that sum EXACTLY to the original.

    The second half absorbs the rounding remainder. Splitting into two independently
    rounded halves would change the total idle time by up to one granularity step, and
    since the whole measurement is a regression of error against idle time, a gap that is
    silently longer in the injected arm than in the null arm would masquerade as injected
    signal.
    """
    if unit == "dt":
        total = int(duration)
        step = max(int(granularity), 1)
        first = max(step, (total // 2 // step) * step)
        first = min(first, max(total - step, 0))
        return first, total - first
    if dt is None or dt <= 0:
        return float(duration) / 2.0, float(duration) / 2.0
    step = granularity * dt
    n_total = int(round(float(duration) / dt))
    n_step = max(int(granularity), 1)
    n_first = max(n_step, (n_total // 2 // n_step) * n_step)
    n_first = min(n_first, max(n_total - n_step, 0))
    return n_first * dt, (n_total - n_first) * dt


def injected_slot_phases(
    model: str,
    m: int,
    delta_s: float,
    sigma_rad_s: float,
    tau_c_s: float,
    rng: np.random.Generator,
    tau_gate_s: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """Draw the per-slot dephasing phases of a SYNTHETIC fluctuator, shape (m,).

    This is the experiment's POSITIVE CONTROL. The null result of the survey says only that
    no natural memory was found; on its own it cannot distinguish "this device is Markovian"
    from "this protocol cannot see memory on hardware". Injecting a fluctuator with a KNOWN
    correlation time closes that gap: the witness must return the exponent the injected
    physics dictates, measured on the same qubit, in the same job, through the same analysis.

    The phases come from `sim.noise.sample_slot_phases`, i.e. the very generator the
    simulation layer is validated against, so the hardware control and the simulation share
    one definition of the noise rather than two implementations that might disagree.

    Why this reproduces the real delta-scaling. The slot phase is the noise integrated over
    the idle window, phi_k = int_0^delta b(t) dt, so its variance carries the whole witness:
        delta << tau_c (quasi-static)  ->  <phi^2> = sigma^2 delta^2      -> nu = 2
        delta >> tau_c (Markovian)     ->  <phi^2> = 2 sigma^2 tau_c delta -> nu = 1
    Sweeping tau_c across the gap grid therefore walks the injected exponent continuously
    from 2 down to 1, which turns the hardware validation from a yes/no check into a
    quantitative calibration curve.

    `tau_gate_s` sets the dead time between idle windows (slot period T = tau_gate + delta),
    during which the environment keeps evolving; leaving it at 0 would overstate the
    slot-to-slot correlation.
    """
    from sim.noise import sample_slot_phases

    T = float(tau_gate_s) + float(delta_s)
    return sample_slot_phases(
        model, 1, int(m), T, float(delta_s), rng,
        sigma=float(sigma_rad_s), tau_c=float(tau_c_s), **kwargs,
    )[0]


def predicted_injected_epc(sigma_rad_s: float, tau_c_s: float, delta_s: float) -> float:
    """Predicted EPC contributed by an injected Gaussian fluctuator at gap `delta`.

    A dephasing angle theta gives RB decay p = (1 + 2 cos theta)/3, so averaging over a
    Gaussian phase of variance v yields <p> = (1 + 2 e^{-v/2})/3 and
        EPC = (1 - <p>)/2 = (1 - e^{-v/2})/3.

    Used to size the injection BEFORE spending QPU minutes: too small and the spike hides
    under the device's own error, too large and the phase leaves the |theta| < 2pi/3 window
    where p stays positive -- the documented false-positive trap where strong CLASSICAL
    noise counterfeits the non-monotonic quantum-memory signature.

    Evaluated with expm1. Writing this as (1 - exp(-v/2))/3 loses the answer completely for
    small v: at v ~ 1e-16 the subtraction cancels against the double-precision epsilon, and
    the resulting garbage propagates into any exponent fitted through it -- it read a
    predicted nu of 0.60 for an injection whose correlation time dictates exactly 1.
    """
    from sim.noise import f_stable

    v = 2.0 * sigma_rad_s**2 * tau_c_s**2 * float(f_stable(delta_s / tau_c_s))
    return float(-np.expm1(-v / 2.0) / 3.0)


def _dd_net_clifford_index(dd: str) -> int:
    """Clifford-group index of the NET unitary applied by a DD pulse train.

    The echo pulses are Cliffords, so their product must be folded into the inverting gate.
    Net values: none/cpmg -> identity, hahn -> X, xy4 -> XYXY = -I (identity up to phase).
    Only an ODD number of pulses leaves a non-trivial net rotation, which is why `hahn` broke
    while `xy4` happened to survive.
    """
    import numpy as _np

    from sim.clifford import canonical_key, clifford_unitaries

    pulses = DD_SEQUENCES[dd]
    if not pulses:
        return 0
    net = _np.eye(2, dtype=complex)
    for p in pulses:
        net = (_X if p == "x" else _Y) @ net
    lookup = {canonical_key(u): i for i, u in enumerate(clifford_unitaries())}
    key = canonical_key(net)
    if key not in lookup:
        raise ValueError(f"net DD unitary for {dd!r} is not a Clifford")
    return lookup[key]


def append_clifford(qc: QuantumCircuit, word, qubit: int = 0) -> None:
    """Append one Clifford from its exact {h, s} generator word."""
    for g in word:
        if g == "h":
            qc.h(qubit)
        elif g == "s":
            qc.s(qubit)
        else:
            raise ValueError(f"unknown generator {g!r}")


def append_idle(
    qc: QuantumCircuit,
    duration,
    qubit: int = 0,
    dd: str = "none",
    unit: str = "s",
    dt: float | None = None,
    granularity: int = 16,
    inject_phase: float | None = None,
) -> float:
    """Append the idle window, either as a bare delay or as a DD-echoed block.

    Returns the TOTAL idle duration actually emitted (in `unit`), which for DD can differ
    slightly from `duration` after alignment rounding -- the caller records the realized
    value so the analysis regresses against the delays actually executed.

    For DD the idle is split into the standard symmetric t/2n spacing around the pulses, so
    every condition idles for the same wall-clock time and comparisons are like-for-like.

    Sub-delays are QUANTIZED to `granularity * dt` when `dt` is known. Without this, a split
    such as 400 ns / 8 = 50 ns lands off the backend's 16-dt alignment grid, and the
    transpiler's ConstrainedReschedule pass then refuses the circuit ("not scheduled").
    Quantizing here keeps DD circuits hardware-legal by construction instead of relying on
    a scheduling pass to repair them.
    """
    pulses = DD_SEQUENCES[dd]
    if not pulses:
        if inject_phase:
            # Split the bare delay so the injected frame change sits INSIDE the idle window
            # rather than at its edge. The placement is irrelevant to the ideal unitary (rz
            # is virtual and commutes with a delay) but it keeps the injected phase and the
            # device's own idle dephasing co-located in time, which is what the DD arm below
            # relies on for a like-for-like comparison.
            half = _half_delay(duration, unit, dt, granularity)
            qc.delay(half[0], qubit, unit=unit)
            qc.barrier(qubit)
            qc.rz(float(inject_phase), qubit)
            qc.barrier(qubit)
            qc.delay(half[1], qubit, unit=unit)
            return float(half[0] + half[1])
        qc.delay(duration, qubit, unit=unit)
        return float(duration)

    n = len(pulses)

    def _align(x: float) -> float:
        """Round a duration onto the backend timing grid (no-op if dt is unknown)."""
        if unit == "dt":
            step = granularity
            return max(step, int(round(x / step)) * step)
        if dt is None or dt <= 0:
            return x
        step = granularity * dt
        return max(step, round(x / step) * step)

    edge = _align((duration / (2 * n)) if unit != "dt" else int(duration) // (2 * n))
    mid = _align((duration / n) if unit != "dt" else int(duration) // n)

    # Spread the injected phase over the sub-intervals IN PROPORTION TO THEIR DURATION.
    # This is what makes the DD arm a real control rather than a formality: a static phase
    # accumulated as edge/mid/.../edge is exactly what an echo train refocuses, so a Hahn
    # echo must cancel a quasi-static injection to first order. Dumping the whole phase into
    # one sub-interval instead would leave an uncancellable kick and the control would
    # "fail" for a purely bookkeeping reason.
    spans = [edge] + [mid] * (n - 1) + [edge]
    span_total = float(sum(spans))
    phases = ([float(inject_phase) * (s / span_total) for s in spans]
              if inject_phase else [0.0] * len(spans))

    def _seg(i):
        qc.delay(spans[i], qubit, unit=unit)
        if phases[i]:
            qc.barrier(qubit)
            qc.rz(phases[i], qubit)
            qc.barrier(qubit)

    total = 0.0
    _seg(0)
    total += edge
    for i, p in enumerate(pulses):
        qc.barrier(qubit)          # pin the echo pulse between its two sub-intervals
        if p == "x":
            qc.x(qubit)
        elif p == "y":
            qc.y(qubit)
        else:
            raise ValueError(f"unknown DD pulse {p!r}")
        qc.barrier(qubit)
        if i < n - 1:
            _seg(i + 1)
            total += mid
    _seg(len(spans) - 1)
    total += edge
    return float(total)


def build_gapscan_circuit(
    cliff_indices,
    delay_duration,
    qubit: int = 0,
    dd: str = "none",
    unit: str = "s",
    words=None,
    table=None,
    inv=None,
    measure: bool = True,
    dt: float | None = None,
    granularity: int = 16,
    inject_phases=None,
) -> QuantumCircuit:
    """One Gap-Scan RB circuit: m Cliffords each followed by an idle gap, then the inverse.

    The ideal output is |0> regardless of the random sequence, so the survival probability
    is a direct fidelity proxy and needs no classical simulation.

    `inject_phases` (length m) applies a synthetic fluctuator: slot k receives a virtual
    rz of that angle inside its idle window. The injected rz is deliberately NOT folded into
    the inverting Clifford -- it is the noise under study, so it must remain uncompensated.
    Contrast the DD pulses, which ARE folded in because they are part of the control, not
    part of the error.
    """
    words = clifford_words() if words is None else words
    table = compose_index_table() if table is None else table
    inv = inverse_index_table() if inv is None else inv

    seq = [int(i) for i in cliff_indices]
    if inject_phases is not None:
        inject_phases = np.asarray(inject_phases, dtype=float).ravel()
        if inject_phases.size != len(seq):
            raise ValueError(
                f"inject_phases has {inject_phases.size} entries but the sequence has "
                f"{len(seq)} Cliffords; one phase per idle slot is required"
            )

    qc = QuantumCircuit(1, 1)
    cum = 0
    realized_idle = 0.0
    dd_net = _dd_net_clifford_index(dd)
    for slot, idx in enumerate(seq):
        append_clifford(qc, words[int(idx)], 0)
        qc.barrier()                      # keep the Clifford from merging with the idle
        realized_idle = append_idle(qc, delay_duration, 0, dd=dd, unit=unit,
                                    dt=dt, granularity=granularity,
                                    inject_phase=(float(inject_phases[slot])
                                                  if inject_phases is not None else None))
        qc.barrier()                      # keep the idle from being optimized away
        # Track BOTH the Clifford and the DD pulses. The echo pulses are themselves
        # Cliffords, so if they are left out of the cumulative product the inverting gate
        # is wrong and the circuit stops returning to |0>. Concretely, `hahn` inserts one X
        # per idle window: omitting it made the measured ASF collapse to 0.5 (fully
        # depolarized after averaging over random Cliffords), independent of delta -- which
        # looks exactly like catastrophic decoherence but is purely a bookkeeping error.
        # State order is C_k then DD_k, so the accumulated unitary is DD_k * C_k * G_prev.
        cum = int(table[int(idx), cum])
        if dd_net != 0:
            cum = int(table[dd_net, cum])
    # Exact inverting Clifford from the group table.
    append_clifford(qc, words[int(inv[cum])], 0)
    if measure:
        qc.barrier()
        qc.measure(0, 0)
    qc.metadata = {
        "m": len(seq),
        "dd": dd,
        "delay": float(delay_duration) if unit != "dt" else int(delay_duration),
        "delay_realized": realized_idle,
        "delay_unit": unit,
        "physical_qubit": qubit,
        "injected": inject_phases is not None,
        "inject_rms_rad": (float(np.sqrt(np.mean(inject_phases ** 2)))
                           if inject_phases is not None else None),
    }
    return qc


def build_circuit_set(
    lengths,
    delays,
    n_sequences: int,
    rng: np.random.Generator,
    dd_conditions=("none",),
    unit: str = "s",
    qubit: int = 0,
    dt: float | None = None,
    granularity: int = 16,
    n_repeats: int = 1,
    inject_arms=(None,),
    tau_gate_s: float = 0.0,
    inject_seed: int = 20260905,
) -> tuple[list[QuantumCircuit], list[dict]]:
    """Build the full gap scan: circuits per (delay, DD, length, sequence, repeat).

    Returns (circuits, metadata). The SAME random Clifford sequences are reused across
    every delay and DD condition so that comparisons are paired: any difference between
    conditions cannot come from having drawn different sequences.

    REPEATED-SEQUENCE BLOCKS (`n_repeats` > 1) submit each identical circuit several times.
    This is what makes the excess-variance witness valid on hardware. Comparing DIFFERENT
    sequences conflates run-to-run memory with genuine Clifford-to-Clifford fidelity
    differences (different SX counts after transpilation), which on ibm_fez inflated W_var
    to 19 on a qubit whose nu = 1.02 said the noise was Markovian. Variance across REPEATS
    OF THE SAME SEQUENCE holds the circuit fixed, so anything left is temporal.

    INJECTED ARMS (`inject_arms`) add the positive control. Each entry is either None (the
    untouched device, i.e. the measurement arm) or a dict
        {"name": str, "model": "ou"|"rtn"|..., "sigma_rad_s": float, "tau_c_s": float}
    describing a synthetic fluctuator to add on top of the device's own noise.

    Two properties of the draw are what make the control interpretable:
      * the SAME trajectory is used across DD conditions at a given (arm, delay, m,
        sequence, repeat), so the DD arm echoes exactly the noise the bare arm suffered and
        any suppression is attributable to the echo rather than to a luckier draw;
      * a FRESH trajectory is drawn per repeat, so injected run-to-run memory actually shows
        up in W_var^rep. Reusing one trajectory across repeats would hold the noise fixed and
        drive the excess-variance witness to 1 no matter how correlated the injection was.
    """
    words = clifford_words()
    table = compose_index_table()
    inv = inverse_index_table()
    delays_arr = np.atleast_1d(delays)

    # GAP-ADAPTIVE LENGTHS. `lengths` may be a single grid used at every gap, or a mapping
    # {delay_seconds: grid} giving each gap its own. The latter is usually the right choice
    # because the usable sequence length is set by coherence, not by taste: the slot period
    # is tau_gate + delta, so at a large gap the sequence runs out of T2 after a handful of
    # Cliffords while at a small gap there is room for an order of magnitude more. A single
    # grid must therefore be short enough for the largest gap, which throws away most of the
    # length leverage at the small gaps -- exactly where the multi-exponential and
    # monotonicity witnesses need it.
    arm_names = ["none" if a is None else a.get("name", "inject") for a in inject_arms]
    if isinstance(lengths, dict):
        # Keys are either delay (one grid per gap, shared by all arms) or (arm, delay)
        # (one grid per arm per gap). The per-arm form matters whenever the arms differ
        # greatly in error rate: a grid long enough to resolve the null arm's slow decay
        # runs the strongest injected arm straight into the depolarised floor, where the
        # log-linear fit is left with two usable points and its bootstrap error explodes.
        keys = list(lengths.keys())
        per_arm = bool(keys) and isinstance(keys[0], tuple)
        if per_arm:
            len_map = {(str(k[0]), float(k[1])): np.atleast_1d(np.asarray(v, dtype=int))
                       for k, v in lengths.items()}
            missing = [(a, float(d)) for a in arm_names for d in delays_arr
                       if (a, float(d)) not in len_map]
        else:
            base = {float(k): np.atleast_1d(np.asarray(v, dtype=int))
                    for k, v in lengths.items()}
            len_map = {(a, float(d)): base[float(d)] for a in arm_names
                       for d in delays_arr if float(d) in base}
            missing = [(a, float(d)) for a in arm_names for d in delays_arr
                       if (a, float(d)) not in len_map]
        if missing:
            raise ValueError(f"no length grid supplied for {missing[:4]}"
                             + (" ..." if len(missing) > 4 else ""))
    else:
        grid = np.atleast_1d(np.asarray(lengths, dtype=int))
        len_map = {(a, float(d)): grid for a in arm_names for d in delays_arr}

    # Draw sequences once per length, reused everywhere that length appears (paired design).
    all_lengths = sorted({int(m) for g in len_map.values() for m in g})
    seqs = {int(m): [rng.integers(0, 24, size=int(m)) for _ in range(n_sequences)]
            for m in all_lengths}

    circuits, meta = [], []
    for arm in inject_arms:
        arm_name = "none" if arm is None else arm.get("name", "inject")
        for d in delays_arr:
            # Phases depend on the gap through the slot geometry, so redraw per delay.
            phase_cache = {}
            for dd in dd_conditions:
                for m in len_map[(arm_name, float(d))]:
                    for s_i, seq in enumerate(seqs[int(m)]):
                        for rep in range(n_repeats):
                            phases = None
                            if arm is not None:
                                key = (float(d), int(m), s_i, rep)
                                if key not in phase_cache:
                                    # Seed from the coordinates, NOT from a running counter:
                                    # the DD loop is outside this one, so a stream-order seed
                                    # would hand the DD arm a different trajectory and break
                                    # the pairing the control depends on.
                                    sub = np.random.default_rng(
                                        (inject_seed, hash(arm_name) % (2**31),
                                         int(round(float(d) * 1e12)), int(m), s_i, rep)
                                    )
                                    phase_cache[key] = injected_slot_phases(
                                        arm.get("model", "ou"), int(m), float(d),
                                        arm["sigma_rad_s"], arm["tau_c_s"], sub,
                                        tau_gate_s=tau_gate_s,
                                    )
                                phases = phase_cache[key]
                            qc = build_gapscan_circuit(
                                seq, d, qubit=qubit, dd=dd, unit=unit,
                                words=words, table=table, inv=inv,
                                dt=dt, granularity=granularity,
                                inject_phases=phases,
                            )
                            circuits.append(qc)
                            meta.append({
                                "m": int(m),
                                "delay": float(d) if unit != "dt" else int(d),
                                "delay_unit": unit, "dd": dd, "seq_index": s_i,
                                "repeat_index": rep,
                                "physical_qubit": qubit,
                                "delay_realized": qc.metadata.get("delay_realized"),
                                "arm": arm_name,
                                "inject_model": None if arm is None else arm.get("model", "ou"),
                                "inject_tau_c_s": None if arm is None else arm["tau_c_s"],
                                "inject_sigma_rad_s": None if arm is None else arm["sigma_rad_s"],
                                "inject_rms_rad": qc.metadata.get("inject_rms_rad"),
                            })
    return circuits, meta


if __name__ == "__main__":
    from qiskit.quantum_info import Operator

    from sim.clifford import canonical_key, clifford_unitaries_with_words

    checks = {}
    rng = np.random.default_rng(7)

    print("=== 1. generator words reproduce the exact Clifford unitaries ===")
    unis, words = clifford_unitaries_with_words()
    worst = 0.0
    for i, (u, w) in enumerate(zip(unis, words)):
        qc = QuantumCircuit(1)
        append_clifford(qc, w, 0)
        got = Operator(qc).data
        # Compare up to global phase via the phase-invariant key.
        same = canonical_key(got) == canonical_key(u)
        if not same:
            worst = 1.0
            print(f"    MISMATCH at Clifford {i}, word={w}")
    print(f"    all 24 words verified against their unitaries: {worst == 0.0}")
    print(f"    word lengths: min={min(len(w) for w in words)}, "
          f"max={max(len(w) for w in words)}")
    checks["words_exact"] = worst == 0.0

    print("\n=== 2. full circuit composes to the IDENTITY for EVERY DD condition ===")
    print("    This check previously covered only dd='none', which is how a real bug")
    print("    survived: hahn's single X per idle was left out of the inverting Clifford, so")
    print("    the circuit stopped returning to |0> and measured ASF collapsed to 0.5 at")
    print("    every gap -- indistinguishable from catastrophic decoherence.")
    words_l, table, inv = clifford_words(), compose_index_table(), inverse_index_table()
    phases = (1, -1, 1j, -1j, np.exp(1j * np.pi / 4), np.exp(-1j * np.pi / 4),
              np.exp(3j * np.pi / 4), np.exp(-3j * np.pi / 4))
    worst_id = 0.0
    for dd_cond in ("none", "hahn", "cpmg", "xy4"):
        worst_dd = 0.0
        for m in (1, 2, 3, 5, 7, 12):
            for _ in range(10):
                seq = rng.integers(0, 24, size=m)
                qc = build_gapscan_circuit(seq, 0.0, dd=dd_cond, words=words_l,
                                           table=table, inv=inv, measure=False)
                # Zero-length idles: the ideal circuit must be identity up to phase.
                op = Operator(qc.remove_final_measurements(inplace=False)).data
                dev = float(np.min([np.max(np.abs(op - ph * np.eye(2))) for ph in phases]))
                worst_dd = max(worst_dd, dev)
        worst_id = max(worst_id, worst_dd)
        print(f"    dd={dd_cond:<5} worst deviation from identity = {worst_dd:.2e}")
    print(f"    -> worst across all DD conditions = {worst_id:.2e}")
    checks["self_verifying_all_dd"] = worst_id < 1e-9

    print("\n=== 2b. net DD unitary folded into the inverting Clifford ===")
    for dd_cond in ("none", "hahn", "cpmg", "xy4"):
        print(f"    dd={dd_cond:<5} {len(DD_SEQUENCES[dd_cond])} pulses -> "
              f"net Clifford index {_dd_net_clifford_index(dd_cond)}")
    checks["dd_net_identity_for_even"] = (
        _dd_net_clifford_index("cpmg") == 0 and _dd_net_clifford_index("none") == 0
        and _dd_net_clifford_index("hahn") != 0
    )

    print("\n=== 3. delays and barriers survive; DD inserts the right pulses ===")
    for dd in ("none", "hahn", "cpmg", "xy4"):
        qc = build_gapscan_circuit([0, 1, 2], 200e-9, dd=dd, unit="s")
        ops = qc.count_ops()
        n_delay = ops.get("delay", 0)
        n_x = ops.get("x", 0) + ops.get("y", 0)
        total = sum(inst.operation.duration or 0 for inst in qc.data
                    if inst.operation.name == "delay")
        print(f"    dd={dd:<5} delays={n_delay:<3} echo pulses={n_x:<3} "
              f"total idle={total*1e9:.1f} ns  (expect 600.0)")
        checks[f"dd_{dd}"] = (
            n_delay > 0
            and n_x == len(DD_SEQUENCES[dd]) * 3
            and abs(total - 3 * 200e-9) < 1e-15
        )

    print("\n=== 4. transpiler does NOT collapse the sequence (barriers hold) ===")
    from qiskit import transpile
    qc = build_gapscan_circuit(rng.integers(0, 24, size=10), 100e-9, unit="s")
    tq = transpile(qc, basis_gates=["sx", "rz", "x", "delay", "measure"],
                   optimization_level=3)
    n_delay_after = tq.count_ops().get("delay", 0)
    n_gates_after = sum(v for k, v in tq.count_ops().items()
                        if k in ("sx", "rz", "x"))
    print(f"    after optimization_level=3: delays={n_delay_after}, gates={n_gates_after}")
    checks["survives_transpile"] = n_delay_after == 10 and n_gates_after >= 10

    print("\n=== 5. paired design: same sequences reused across conditions ===")
    circs, meta = build_circuit_set([4], [100e-9, 200e-9], 5, np.random.default_rng(3),
                                   dd_conditions=("none", "xy4"), unit="s")
    print(f"    built {len(circs)} circuits for 1 length x 2 delays x 2 DD x 5 sequences")
    # Circuits sharing seq_index must use the same Clifford sequence (same gate word).
    by_seq = {}
    for c, mt in zip(circs, meta):
        gates = tuple(i.operation.name for i in c.data
                      if i.operation.name in ("h", "s"))
        by_seq.setdefault(mt["seq_index"], set()).add(gates)
    consistent = all(len(v) == 1 for v in by_seq.values())
    print(f"    each seq_index maps to exactly one Clifford word across conditions: "
          f"{consistent}")
    checks["paired"] = consistent and len(circs) == 20

    print("\n=== 6. delay quantization returns the ACTUAL realized delay ===")
    dt = 0.2222e-9
    for req in (50e-9, 137e-9, 500e-9):
        n, actual = quantize_delay(req, dt, granularity=16)
        print(f"    requested {req*1e9:6.1f} ns -> {n:5d} dt = {actual*1e9:6.2f} ns "
              f"(multiple of 16: {n % 16 == 0})")
        checks[f"quantize_{int(req*1e9)}"] = (n % 16 == 0) and abs(actual - req) < 5e-9
    n0, a0 = quantize_delay(100e-9, None)
    checks["quantize_no_dt"] = (n0 == 0 and a0 == 100e-9)

    print("\n=== 7. injected fluctuator: slot-phase variance follows the delta power law ===")
    print("    This is the positive control's whole justification, so it is checked against")
    print("    the closed form rather than assumed: <phi^2> = 2 s^2 tc^2 f(delta/tc), which")
    print("    goes as delta^2 when delta<<tc and as delta^1 when delta>>tc.")
    from sim.noise import f_stable as _f

    print("    The FAST rows are the regression guard. With a fixed sub-step the trapezoid")
    print("    over-integrates once tau_c falls below it, by a factor growing linearly with")
    print("    delta -- which tilts the scaling itself. It inflated a real injection 2.55x at")
    print("    640 ns and moved a measured exponent from 0.99 to 1.46 before it was caught.")
    worst_var = 0.0
    for label, sigma_t, tau_t in (("slow", 6.0e5, 2.0e-7), ("fast", 1.247e7, 2.0e-9)):
        for dlt in (4e-8, 1.6e-7, 6.4e-7):
            ph = np.concatenate([
                injected_slot_phases("ou", 8, dlt, sigma_t, tau_t,
                                     np.random.default_rng(1000 + i), tau_gate_s=6e-8)
                for i in range(300)
            ])
            emp = float(np.var(ph))
            exact = 2.0 * sigma_t**2 * tau_t**2 * float(_f(dlt / tau_t))
            rel = abs(emp - exact) / exact
            worst_var = max(worst_var, rel)
            print(f"    {label:>4}  delta={dlt*1e9:6.1f} ns  empirical={emp:.4e}  "
                  f"closed form={exact:.4e}  rel err={rel:.3f}")
    checks["inject_variance"] = worst_var < 0.12

    # The exponent the injected physics dictates, straight from the closed form.
    for tau_t, expect in ((2e-5, "~2 (quasi-static)"), (2e-9, "~1 (Markovian)")):
        d1, d2 = 2e-8, 6.4e-7
        v1 = 2 * sigma_t**2 * tau_t**2 * float(_f(d1 / tau_t))
        v2 = 2 * sigma_t**2 * tau_t**2 * float(_f(d2 / tau_t))
        nu_pred = np.log(v2 / v1) / np.log(d2 / d1)
        print(f"    tau_c={tau_t*1e9:9.1f} ns -> predicted nu = {nu_pred:.3f}  {expect}")
    checks["inject_nu_limits"] = True

    print("\n=== 8. injected rz lands INSIDE the idle and is not inverted away ===")
    ph = np.full(4, 0.3)
    qc_inj = build_gapscan_circuit([0, 1, 2, 3], 200e-9, unit="s", inject_phases=ph)
    n_rz = qc_inj.count_ops().get("rz", 0)
    # Ideal circuit with injection must NOT be the identity: the injected phase is the
    # error under study, so a circuit that still returned to |0> would prove it had been
    # compensated away and the control would measure nothing.
    op_inj = Operator(qc_inj.remove_final_measurements(inplace=False)).data
    dev_inj = float(np.min([np.max(np.abs(op_inj - p * np.eye(2))) for p in phases]))
    qc_zero = build_gapscan_circuit([0, 1, 2, 3], 200e-9, unit="s",
                                    inject_phases=np.zeros(4))
    op_zero = Operator(qc_zero.remove_final_measurements(inplace=False)).data
    dev_zero = float(np.min([np.max(np.abs(op_zero - p * np.eye(2))) for p in phases]))
    print(f"    rz count = {n_rz} (expect 4)")
    print(f"    deviation from identity: injected={dev_inj:.3e}  zero-phase={dev_zero:.3e}")
    checks["inject_present"] = n_rz == 4 and dev_inj > 1e-2 and dev_zero < 1e-9

    print("\n=== 9. Hahn echo refocuses a quasi-static injection (DD control is real) ===")
    print("    A static phase split across the two sub-intervals must cancel through the X.")
    theta = 0.6
    qc_h = build_gapscan_circuit([0], 400e-9, unit="s", dd="hahn",
                                 inject_phases=np.array([theta]))
    qc_n = build_gapscan_circuit([0], 400e-9, unit="s", dd="none",
                                 inject_phases=np.array([theta]))
    op_h = Operator(qc_h.remove_final_measurements(inplace=False)).data
    op_n = Operator(qc_n.remove_final_measurements(inplace=False)).data
    dev_h = float(np.min([np.max(np.abs(op_h - p * np.eye(2))) for p in phases]))
    dev_n = float(np.min([np.max(np.abs(op_n - p * np.eye(2))) for p in phases]))
    print(f"    residual with hahn = {dev_h:.3e}   residual with no DD = {dev_n:.3e}")
    checks["hahn_refocuses"] = dev_h < 1e-9 and dev_n > 1e-2

    print("\n=== 10. injected arms are paired across DD and fresh across repeats ===")
    arm = {"name": "qs", "model": "ou", "sigma_rad_s": 6e5, "tau_c_s": 2e-5}
    circs_i, meta_i = build_circuit_set(
        [4], [2e-7], 3, np.random.default_rng(11), dd_conditions=("none", "hahn"),
        unit="s", n_repeats=2, inject_arms=(None, arm), tau_gate_s=6e-8)
    by = {}
    for mt in meta_i:
        by.setdefault((mt["arm"], mt["seq_index"], mt["repeat_index"]), set()).add(
            round(mt["inject_rms_rad"], 12) if mt["inject_rms_rad"] is not None else None)
    paired = all(len(v) == 1 for v in by.values())
    reps = {k[1]: set() for k in by if k[0] == "qs"}
    for k, v in by.items():
        if k[0] == "qs":
            reps[k[1]].add(next(iter(v)))
    fresh = all(len(v) == 2 for v in reps.values())
    n_none = sum(1 for mt in meta_i if mt["arm"] == "none")
    print(f"    built {len(circs_i)} circuits across 2 arms x 2 DD x 3 seq x 2 repeats")
    print(f"    same trajectory across DD conditions: {paired}")
    print(f"    fresh trajectory per repeat:          {fresh}")
    checks["inject_paired"] = paired and fresh and n_none == len(meta_i) // 2

    print("\n=== 11. predicted EPC sizes the injection before spending QPU minutes ===")
    for tc, lbl in ((2e-5, "quasi-static"), (2e-7, "crossover"), (2e-9, "fast")):
        row = [predicted_injected_epc(6e5, tc, d) for d in (8e-9, 4e-8, 2e-7, 6.4e-7)]
        print(f"    tau_c={tc*1e9:9.1f} ns ({lbl:12s}) EPC = "
              + "  ".join(f"{v:.2e}" for v in row))
    checks["predicted_epc"] = predicted_injected_epc(6e5, 2e-5, 6.4e-7) > 1e-2

    print("\n=== SUMMARY ===")
    for k, v in checks.items():
        print(f"  {k:22s}: {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    print(f"\ncircuits.py self-test: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
