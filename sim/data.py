"""Canonical data container passed from the simulation layer to the estimator."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ASFData"]


@dataclass
class ASFData:
    """Average-sequence-fidelity data for ONE idle-gap value.

    Attributes
    ----------
    lengths : (L,) int array
        Clifford sequence lengths m.
    survivals : (L, N) float array
        PER-SEQUENCE survival probabilities. Sequence-resolved data is mandatory,
        not a convenience: the estimator's bootstrap resamples at the sequence
        level, and collapsing to the mean here would silently understate the
        uncertainty that the paper's finite-sample claim depends on.
    delta : float
        Idle gap (us).
    tau_gate : float
        Gate duration (us). Slot period is T = tau_gate + delta.
    meta : dict
        Provenance: noise model, sigma, tau_c, seed, n_shots, etc.
    """

    lengths: np.ndarray
    survivals: np.ndarray
    delta: float
    tau_gate: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lengths = np.asarray(self.lengths, dtype=int)
        self.survivals = np.atleast_2d(np.asarray(self.survivals, dtype=float))
        if self.survivals.shape[0] != len(self.lengths):
            raise ValueError(
                f"survivals first axis ({self.survivals.shape[0]}) must match "
                f"len(lengths) ({len(self.lengths)})"
            )

    @property
    def slot_period(self) -> float:
        """T = tau_gate + delta (us)."""
        return self.tau_gate + self.delta

    @property
    def n_sequences(self) -> int:
        return int(self.survivals.shape[1])

    @property
    def asf(self) -> np.ndarray:
        """Mean survival per length, shape (L,)."""
        return self.survivals.mean(axis=1)

    @property
    def asf_stderr(self) -> np.ndarray:
        """Standard error of the mean per length, shape (L,)."""
        n = max(self.n_sequences, 2)
        return self.survivals.std(axis=1, ddof=1) / np.sqrt(n)
