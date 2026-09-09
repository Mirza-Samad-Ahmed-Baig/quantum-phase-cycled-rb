"""Retrieve counts for existing legacy result files; never submit a QPU job."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware.backend import get_service
from hardware.run_gapscan import _counts_from_job_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default="coauthor")
    a = ap.parse_args()
    service = get_service(a.account)
    records = []
    for pending_path in sorted(Path("data").glob("*_pending.json")):
        if pending_path.name.startswith("phase_cycle_"):
            continue
        result_path = pending_path.with_name(pending_path.name.replace("_pending", "_results"))
        if not result_path.exists():
            continue
        p = json.loads(pending_path.read_text(encoding="utf-8"))
        out = pending_path.with_name(pending_path.name.replace("_pending", "_counts"))
        record = dict(job_id=p["job_id"], pending_file=pending_path.as_posix(),
                      results_file=result_path.as_posix())
        try:
            if out.exists():
                raw = json.loads(out.read_text(encoding="utf-8"))
            else:
                job = service.job(p["job_id"])
                if str(job.status()).upper() != "DONE":
                    raise ValueError(f"Archived result's job status is {job.status()}")
                counts = _counts_from_job_result(job.result())
                if len(counts) != len(p["circuit_meta"]):
                    raise ValueError("Raw count length differs from saved circuit metadata")
                if any(sum(row.values()) != p["shots"] for row in counts):
                    raise ValueError("Saved shots differ from returned circuit counts")
                raw = dict(kind="legacy_raw_hardware_counts", job_id=p["job_id"],
                           pending_file=pending_path.as_posix(), counts=counts,
                           metrics=job.metrics(), retrieved_utc=datetime.now(timezone.utc).isoformat(),
                           pending_sha256=hashlib.sha256(pending_path.read_bytes()).hexdigest())
                out.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
            record.update(status="counts_archived", counts_file=out.as_posix(),
                          circuits=len(raw["counts"]), sha256=hashlib.sha256(out.read_bytes()).hexdigest())
        except Exception as error:
            # Error type is safe to archive. Do not serialize provider exception text,
            # which can contain account-specific URLs or request credentials.
            record.update(status="retrieval_failed", error_type=type(error).__name__)
        records.append(record)
        Path("data/legacy_archive_inventory.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(record["job_id"], record["status"], record.get("circuits", record.get("error_type")), flush=True)


if __name__ == "__main__":
    main()
