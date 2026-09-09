"""Package the local review draft and its reproducibility evidence; never publish."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root=Path(__file__).resolve().parents[1]
    selected=set()
    for folder in ("sim","estimator","hardware","scripts"):
        selected.update((root/folder).glob("*.py"))
    for name in ("README.md","PLAN.md","requirements.txt","requirements.lock.txt","requirements-reproduce.txt",
                 "LICENSE","LICENSE_DATA.md","ATTRIBUTION.md","NOTICE","CITATION.cff",
                 "sanity_check.py","paper/main.tex","paper/main.pdf","paper/refs.bib",
                 "paper/numbers.tex","paper/phase_cycle_hardware.tex","paper/phase_cycle_review.tex",
                 "paper/readout_sensitivity.tex","paper/RESEARCH_STATUS.md","paper/CRITICAL_REVIEW.md",
                 "paper/qst_submission.tex","paper/qst_submission.pdf"):
        selected.add(root/name)
    for pattern in ("*.json", "*.jsonl", "*.csv", "*.npz"):
        selected.update((root/"data").glob(pattern))
    for pattern in ("*.md", "*.txt", "*.json", "*.csv", "*.pdf", "*.zip"):
        selected.update((root/"submission").glob(pattern))
        selected.update((root/"arxiv").glob(pattern))
        selected.update((root/"release").glob(pattern))
    selected.update((root/"figures").glob("fig_*.png"))
    selected.update((root/"figures").glob("fig_*.pdf"))
    missing=[str(p.relative_to(root)) for p in selected if not p.is_file()]
    if missing:raise FileNotFoundError(missing)
    # Explicit file allowlist: no account stores, tokens, environment files,
    # third-party environments or executable tools are included.
    records={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,
               sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(selected)}
    manifest=dict(kind="local_review_draft_not_public_release",files=records,
                  hardware_success_files=[str(p.relative_to(root)) for p in sorted((root/"data").glob("phase_cycle_*_results.json"))],
                  pending_submission_items=["author facts and scientific review", "funding and competing-interest declarations",
                                            "approval of current PDF", "named author uploads personally"],
                  data_route="prepared journal supplementary archive; separate public DOI not required before submission")
    mp=root/"paper/reproducibility_manifest.json"
    mp.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    archive=root/"paper/research_review_bundle.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED) as z:
        for p in sorted(selected):z.write(p,p.relative_to(root).as_posix())
        z.write(mp,mp.relative_to(root).as_posix())
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name,record in records.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==record["sha256"]
    print(f"Verified {len(records)} files; {archive.name}: {archive.stat().st_size:,} bytes")


if __name__=="__main__":main()
