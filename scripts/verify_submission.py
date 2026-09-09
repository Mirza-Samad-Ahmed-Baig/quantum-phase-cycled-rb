"""Extract the prepared supplement and reproduce reported analyses offline."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"


def main():
    archive_path = OUT / "supplementary_reproducibility.zip"
    destination = Path(tempfile.mkdtemp(prefix=".tmp-submission-repro-", dir=ROOT))
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.json"))
        assert set(archive.namelist()) == set(manifest["files"]) | {"manifest.json"}
        for name, record in manifest["files"].items():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive path escapes extraction directory")
            assert hashlib.sha256(archive.read(name)).hexdigest() == record["sha256"]
        archive.extractall(destination)

    # Refuse outbound socket connections in each child. These analysis entry
    # points have no reason to contact IBM or any other network service.
    runner = """import runpy, socket, sys
def offline(*args, **kwargs):
    raise RuntimeError('Network connection attempted during offline reproduction')
socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    commands = []
    def run(*args):
        result = subprocess.run([sys.executable, "-c", runner, *args], cwd=destination,
                                capture_output=True, text=True, timeout=240)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        commands.append(list(args))

    jobs = []
    for path in sorted((destination / "data").glob("phase_cycle_*_counts.json")):
        result_path = path.with_name(path.name.replace("_counts", "_results"))
        before = json.loads(result_path.read_text(encoding="utf-8"))
        run("scripts/phase_cycle_hardware.py", "--analyse", path.relative_to(destination).as_posix())
        after = json.loads(result_path.read_text(encoding="utf-8"))
        assert after == before, f"Recomputed result differs: {result_path.name}"
        jobs.append(dict(job_id=after["job_id"], all_result_fields_identical=True,
                         bootstrap=after["bootstrap"], primary=after["primary"]))
    assert len(jobs) == 2
    run("scripts/audit_legacy_raw.py")
    legacy = json.loads((destination / "data/legacy_raw_audit.json").read_text(encoding="utf-8"))
    run("scripts/check_manuscript.py")

    source_checks = []
    for name in ("phase_cycle_validation.json", "phase_cycle_review.json"):
        record = json.loads((destination / "data" / name).read_text(encoding="utf-8"))
        for original, expected in record["source_sha256"].items():
            normalized = original.replace("\\", "/")
            # Historical records may use an absolute path on the original host.
            relative = "/".join(normalized.split("/")[-2:])
            actual = hashlib.sha256((destination / relative).read_bytes()).hexdigest()
            assert actual == expected, f"Simulation source changed: {relative}"
            source_checks.append(dict(record=name, source=relative, sha256=actual))

    with (destination / "data/phase_cycle_observations.csv").open(newline="", encoding="utf-8") as stream:
        observations = list(csv.DictReader(stream))
    assert len(observations) == 8448
    shots = sum(int(row["shots"]) for row in observations)
    assert shots == 67584
    with (destination / "data/phase_cycle_comparator.csv").open(newline="", encoding="utf-8") as stream:
        trials = sum(1 for _ in csv.DictReader(stream))
    assert trials == 800
    report = dict(status="passed", archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                  verified_archive_files=len(manifest["files"]), python=sys.version,
                  extracted_directory=destination.name, outbound_connections_disabled=True,
                  commands=commands, phase_cycle_jobs=jobs,
                  legacy_point_estimates=sum(row["point_estimates_checked"] for row in legacy["checks"]),
                  matching_simulation_source_hashes=source_checks,
                  observation_rows=len(observations), observed_shots=shots, comparator_trials=trials,
                  scope="Archived-count analyses and structural checks rerun on this host; simulation source hashes verified. Not an independent peer review or fresh-environment installation test.")
    (OUT / "reproduction_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "verified_archive_files", "legacy_point_estimates", "observation_rows", "observed_shots", "comparator_trials")}, indent=2))


if __name__ == "__main__":
    main()
