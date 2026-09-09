"""Reconstruct archived legacy EPC point estimates from retrieved raw counts."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimator.witness import fit_decay_p, epc_from_p
from hardware.run_gapscan import _survival_from_counts


def main():
    checks = []
    for raw_path in sorted(Path("data").glob("*_counts.json")):
        if raw_path.name.startswith("phase_cycle_"):
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        pending = json.loads(Path(raw["pending_file"]).read_text(encoding="utf-8"))
        saved = json.loads(raw_path.with_name(raw_path.name.replace("_counts", "_results")).read_text(encoding="utf-8"))
        is_positive = raw_path.name.startswith("poscontrol_")
        key_of = lambda mt: (mt.get("arm", "none") if is_positive else mt["qubit"], mt["dd"], float(mt["delay"]))
        lengths = {}
        for mt in pending["circuit_meta"]:
            lengths.setdefault(key_of(mt), set()).add(int(mt["m"]))
        lengths = {key: sorted(value) for key, value in lengths.items()}
        arrays = {key: np.full((len(value), pending["n_sequences"], pending["n_repeats"]), np.nan)
                  for key, value in lengths.items()}
        for mt, counts in zip(pending["circuit_meta"], raw["counts"], strict=True):
            key = key_of(mt)
            arrays[key][lengths[key].index(int(mt["m"])), mt["seq_index"], mt.get("repeat_index", 0)] = _survival_from_counts(counts, pending["shots"])
        assert all(np.isfinite(value).all() for value in arrays.values())
        epcs = {key: epc_from_p(fit_decay_p(np.array(lengths[key]), value.mean(axis=(1, 2))))
                for key, value in arrays.items()}
        errors = []
        if is_positive:
            rows = [("none", saved["epc_none"])] + [(row["name"], row["epc"]) for row in saved["arms"]]
            for name, expected in rows:
                keys = sorted((key for key in epcs if key[:2] == (name, "none")), key=lambda key: key[2])
                errors.extend(np.abs(np.array([epcs[key] for key in keys])-expected))
        else:
            for row in saved:
                keys = sorted((key for key in epcs if key[:2] == (row["qubit"], "none")), key=lambda key: key[2])
                errors.extend(np.abs(np.array([epcs[key] for key in keys])-row["epcs"]))
        maximum = float(max(errors))
        assert maximum < 1e-10, (raw_path, maximum)
        checks.append(dict(job_id=raw["job_id"], point_estimates_checked=len(errors),
                           maximum_epc_difference=maximum, raw_file=raw_path.as_posix()))
    report = dict(checks=checks,
                  scope="EPC point estimates only; historical bootstrap seeds used process-dependent Python hashes, so stored uncertainty summaries are retained rather than claimed bitwise reproducible")
    Path("data/legacy_raw_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Verified", sum(row["point_estimates_checked"] for row in checks), "EPC point estimates from", len(checks), "raw jobs")


if __name__ == "__main__":
    main()
