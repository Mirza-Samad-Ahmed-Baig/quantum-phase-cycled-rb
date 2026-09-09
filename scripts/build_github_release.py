"""Prepare a clean public repository snapshot, without Git history or credentials."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from build_arxiv import REPO, REPO_OWNER, REPO_URL, CODE_LICENSE, PUBLISHING_SCRIPTS, preprint_body

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release"


def main():
    OUT.mkdir(exist_ok=True)
    with ZipFile(ROOT / "submission/supplementary_reproducibility.zip") as archive:
        entries = {name: archive.read(name) for name in archive.namelist()
                   if name != "manifest.json" and Path(name).name not in PUBLISHING_SCRIPTS}
    entries["README.md"] = (OUT / "github_readme.md").read_bytes()
    entries["paper/main.tex"] = preprint_body((ROOT / "paper/main.tex").read_text(encoding="utf-8")).encode()
    entries["paper/preprint.pdf"] = (ROOT / "arxiv/preprint.pdf").read_bytes()
    entries["paper/README.md"] = (
        "# Manuscript\n\nThe preprint PDF and editable main.tex point to this repository\n"
        "for the data, analysis code and reproduction instructions.\n"
        "Run the number/figure/report scripts from the repository root, then compile\n"
        "main.tex from paper/ with a REVTeX 4-2 toolchain. Journal cover letters and\n"
        "internal author forms are maintained outside this public snapshot.\n").encode()
    entries[".gitignore"] = (
        "__pycache__/\n*.py[cod]\n.venv/\nvenv/\n.env\n.env.*\n.qiskit/\n"
        "*.aux\n*.blg\n*.log\n*.out\n*.xdv\n*.synctex.gz\n*Notes.bib\n"
        ".DS_Store\nThumbs.db\n").encode()
    records = {name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
               for name, value in sorted(entries.items())}
    entries["manifest.json"] = json.dumps(dict(repository=REPO, files=records,
        note="Hashes describe the initial snapshot. Regenerated outputs may differ; original raw counts are preserved."), indent=2).encode()
    target = OUT / REPO
    target.mkdir(exist_ok=True)
    for name, value in entries.items():
        path = (target / name).resolve()
        assert path.is_relative_to(target.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    archive_path = OUT / (REPO+".zip")
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for name, value in sorted(entries.items()):
            archive.writestr(REPO+"/"+name, value)
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name, value in entries.items():
            assert archive.read(REPO+"/"+name) == value
    assert not any(name.startswith(("submission/", ".git/", ".qiskit/")) for name in entries)
    report = dict(repository_name=REPO, prepared_locally=True, published=False,
                  files=len(entries), zip_file=archive_path.name, zip_bytes=archive_path.stat().st_size,
                  zip_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                  license_code=CODE_LICENSE, license_data="CC-BY-4.0",
                  repository_url=REPO_URL, github_owner=REPO_OWNER)
    (OUT / "package_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
