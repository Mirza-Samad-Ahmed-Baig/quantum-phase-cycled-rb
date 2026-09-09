# Quantum phase-cycled randomized benchmarking

Code and data for phase-cycled randomized benchmarking of quantum processors:
a connected phase observable for classical memory hidden by the ordinary
Clifford average.

The associated paper is published on arXiv:

```bibtex
@misc{baig2026phasecycledrandomizedbenchmarkingquantum,
      title={Phase-cycled randomized benchmarking of quantum processors: recovering hidden classical noise correlations},
      author={Mirza Samad Ahmed Baig and Syeda Anshrah Gillani and Abdul Akbar Khan and Muhammad Omer Khan},
      year={2026},
      eprint={2609.06448},
      archivePrefix={arXiv},
      primaryClass={cs.ET},
      url={https://arxiv.org/abs/2609.06448},
}
```

## Research contribution

1. **Exact identifiability limit.** A stationary symmetric telegraph fluctuator
   and a process independently reset at slot boundaries give identical mean RB
   curves for every sequence length and idle duration, including unequal slots.
   An idle-gap exponent therefore cannot alone certify memory between slots.
2. **An observable that removes that ambiguity.** Four two-slot phase settings
   and four single-slot marginals give
   `C_beta = (4/9) sin(beta)^2 Cov(sin(phi_1), sin(phi_2))` under ideal
   Clifford twirling and classical idle dephasing. Independent slots and fixed
   detuning give zero. This is not a universal or quantum-memory classifier.
3. **Explicit uncertainty and controls.** An unbiased connected estimator,
   paired block bootstrap, a separate conservative finite-sample confidence
   set, and engineered correlated/reset/static/no-injection hardware controls.

Idle-duration RB and engineered-noise RB calibration are prior art:
[O'Malley et al. (2015)](https://doi.org/10.1103/PhysRevApplied.3.044009) and
[Edmunds et al. (2020)](https://doi.org/10.1103/PhysRevResearch.2.013156).
Sine-phase correlation spectroscopy and phase cycling are also established,
including [Yan et al. (2012)](https://doi.org/10.1103/PhysRevB.85.174521) and
[Zohar et al. (2026)](https://arxiv.org/abs/2603.05650). The proposed distinction
is the explicit Clifford-RB construction and its reset-equivalence connection.

**The method does not outperform the Ramsey comparator.** In 800 additional
paired ideal simulation trials, Ramsey/echo has 1.85-4.22 times lower RMSE at equal
shot and sensing-window budgets. These are simulation results, not a hardware
comparison or a match for gate duration. All trial estimates are archived in
[data/phase_cycle_review.json](data/phase_cycle_review.json).

## Verified offline

- Ordinary continuous/reset contrast agreement: **4.66e-15** maximum discrepancy.
- Exact phase-cycle response versus transfer matrix: **1.39e-16**.
- Full `24 x 24` Clifford enumeration versus twirl: **3.33e-16**.
- Hardware circuit statevectors versus independent Bloch propagation: **6.66e-16**.
- **800 simulation trials**, 200 per noise class, with 512 independent blocks
  and 64 shots per setting. Correlated RTN detection was 200/200; empirical
  bootstrap coverage ranged from **92.5% to 95.5%**. This is not a demonstrated
  universal 95% guarantee. The separate conservative confidence sets covered
  the truth in all 800 trials; correlated-arm detection with that bound was 71%.

These are simulation and mathematical validation results, not new native-device
memory discoveries. Detailed trial results and binomial intervals are in
[data/phase_cycle_validation.json](data/phase_cycle_validation.json).

## Hardware evidence

Two completed acquisitions on `ibm_fez`, qubit 46, used independent random seeds
and the same protocol: 128 blocks per arm, eight phase settings, eight shots per
setting, and 128 readout calibration circuits. Each returned 33,792 circuit shots.

| Primary correlated-minus-reset contrast | Estimate | Empirical 95% interval |
|---|---:|---:|
| First acquisition | 0.2540 | [0.1775, 0.3310] |
| Independent-seed confirmation | 0.2114 | [0.1359, 0.2851] |

Both primary intervals exclude zero; every individual reset, fixed-detuning and
no-injection control interval contains zero. The second shared-sign arm's interval
does not contain its ideal population prediction, so exact quantitative agreement
is not claimed. This is repeatability within one session with engineered phases,
not a native-memory discovery or replication across calibration cycles.

All raw counts, designs, results and three failed attempts are retained in
`data/phase_cycle_*`. The [campaign record](data/phase_cycle_campaign.json)
includes all five attempts: successful jobs were charged 24 QPU seconds in total,
and unsuccessful jobs 45 seconds. Bootstrap intervals assume stable acquisition
and affine readout; their hardware coverage is not a finite-sample guarantee.

## Legacy hardware and the withdrawn exclusion

The earlier idle-gap campaign covered 16 qubits on `ibm_fez` and `ibm_marrakesh`.
No screening candidate replicated under the follow-up rule. The injected
duration-response controls remain useful exploratory calibration evidence.

**The former "25% at 95% confidence over four decades" exclusion is withdrawn.**
It held nuisance quantities fixed, generalized the strongest point to the full
correlation-time range, and omitted an absolute model-fit check. Profiling the
offset and total idle amplitude gives minimum chi-square values 15.81-18.97,
above the six-dimensional Gaussian-ball cutoff 12.59 on every tested time.
The stored EPC error model is therefore inadequate for the unconditional bound.
See [data/sensitivity_profile_audit.json](data/sensitivity_profile_audit.json).

Six legacy raw datasets were recovered and reproduce 101 EPC point estimates.
Four older Fez datasets remain summary-only; the
[inventory](data/legacy_archive_inventory.json) records that limitation. Historical
positive-control bootstrap errors cannot be reproduced bit for bit because the
original seed depended on the Python process. Future analyses use a stable seed.

`estimator/witness.py` retains historical labels for reproducing old analyses.
Those labels are model-dependent screening heuristics, not memory certificates.
`scripts/sensitivity.py` runs the corrected offline audit by default.

## Reproduce

Use the existing Python environment. Do not install `qiskit-aer`: its binary
extension previously crashed this environment, and none of the new work needs it.

```powershell
.\.venv\Scripts\python.exe scripts\validate_phase_cycle.py --trials 200 --blocks 512 --bootstrap 400
.\.venv\Scripts\python.exe scripts\review_phase_cycle.py
.\.venv\Scripts\python.exe scripts\audit_exclusion.py
.\.venv\Scripts\python.exe scripts\audit_legacy_raw.py
.\.venv\Scripts\python.exe scripts\make_numbers.py
.\.venv\Scripts\python.exe scripts\make_figures.py
.\.venv\Scripts\python.exe scripts\report_phase_cycle.py
.\.venv\Scripts\python.exe scripts\report_review.py
```

## Hardware workflow

Preparation and analysis are offline. Submission and fetching use IBM access.
Credentials stay in Qiskit's account store, outside this repository.

```powershell
# Save a concrete randomized design without contacting IBM.
.\.venv\Scripts\python.exe scripts\phase_cycle_hardware.py --prepare --packed --blocks 128 --shots 8 --plan data\phase_cycle_design_next.json

# Verify the actual circuit construction offline.
.\.venv\Scripts\python.exe scripts\phase_cycle_hardware.py --check --plan data\phase_cycle_design_next.json

# Submit only after inspecting its reported circuit/shot/time budget.
.\.venv\Scripts\python.exe scripts\phase_cycle_hardware.py --submit --plan data\phase_cycle_design_next.json --account <your-account>

# Fetch a particular saved acquisition, preserving raw counts or its failure.
.\.venv\Scripts\python.exe scripts\phase_cycle_hardware.py --fetch data\phase_cycle_TIMESTAMP_pending.json --account <your-account>

# Reanalyse saved counts without credentials or QPU time.
.\.venv\Scripts\python.exe scripts\phase_cycle_hardware.py --analyse data\phase_cycle_TIMESTAMP_counts.json
```

The four arms share random Clifford pairs. Packed execution groups matching
Clifford templates with broadcast phase parameters, then randomizes template and
binding order. This reduced Runtime overhead and enabled both successful runs.
The primary comparison is correlated minus reset. Preparation records the
protocol before acquisition; fetching saves counts, job metrics and hashes.
Failed jobs are retained explicitly and are never represented as measurements.

## Layout

| Path | Purpose |
|---|---|
| `sim/phase_cycle.py` | Exact biased RTN transfer, closed response, independent Clifford simulation |
| `estimator/phase_cycle.py` | Connected estimator and separate bootstrap/finite confidence intervals |
| `estimator/exclusion.py` | Nuisance profiling and Gaussian model-fit diagnostics |
| `scripts/validate_phase_cycle.py` | Exact checks, independent trial calibration and theory figure |
| `scripts/phase_cycle_hardware.py` | Design, bounded submission, raw-count archive and offline analysis |
| `scripts/audit_exclusion.py` | Audit of legacy EPC summaries |
| `sim/`, `hardware/`, `estimator/witness.py` | Earlier gap-scan infrastructure and historical reproduction |

Licensing: code under [LICENSE](LICENSE) (custom Quantum RB Attribution License,
crediting the authors and paper); original data/figures under
[LICENSE_DATA.md](LICENSE_DATA.md) (CC BY 4.0). See [CITATION.cff](CITATION.cff)
and [ATTRIBUTION.md](ATTRIBUTION.md) for citation metadata.
