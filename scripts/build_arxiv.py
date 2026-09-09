"""Build a compact arXiv source package and metadata; never upload."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from build_submission import expanded_tex

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "arxiv"
SOURCE = OUT / "source"
TITLE = "Phase-cycled randomized benchmarking of quantum processors: recovering hidden classical noise correlations"
REPO = "quantum-phase-cycled-rb"
REPO_OWNER = "Mirza-Samad-Ahmed-Baig"
REPO_URL = "https://github.com/"+REPO_OWNER+"/"+REPO
CODE_LICENSE = "LicenseRef-Quantum-RB-Attribution-1.0"
PUBLISHING_SCRIPTS = {
    "build_submission.py", "build_research_bundle.py", "verify_submission.py",
    "build_arxiv.py", "build_github_release.py", "audit_references.py",
}


def preprint_body(body):
    data = ("The accompanying repository contains the data, analysis code and\n"
            "reproduction instructions. Its inventory distinguishes observed hardware\n"
            "counts, simulated data, failed acquisitions and legacy summaries; four\n"
            "older datasets remain summary-only. Code uses a custom source-available\n"
            "license requiring attribution in public work substantially using it, and\n"
            "original data and figures use CC BY 4.0. See the repository attribution\n"
            "and citation files for the authors and license scope.\n")
    data += "The companion project repository is \\href{"+REPO_URL+"}{\\texttt{"+REPO+"}}.\n"
    body, n = re.subn(r"\\section\*\{Data and code availability\}.*?(?=\\section\*|\\begin\{acknowledgments\})",
                     lambda _: "\\section*{Data and code availability}\n"+data+"\n", body, flags=re.S)
    assert n == 1
    # Missing journal declarations are kept in the local author form. Do not
    # turn an unknown funding/conflict fact into a declaration of 'none'.
    for heading in ("Author contributions", "Competing interests", "Funding"):
        pattern = r"\\section\*\{"+heading+r"\}.*?(?=\\section\*|\\begin\{acknowledgments\})"
        body = re.sub(pattern, lambda m: "" if "pending confirmation" in m.group(0) else m.group(0), body, flags=re.S)
    body = body.replace("The listed authors must critically review the manuscript, references and\n"
                        "computational evidence and take responsibility for the submitted work.",
                        "Responsibility for the scientific content rests with the authors.")
    return body


def prepare():
    SOURCE.mkdir(parents=True, exist_ok=True)
    body = preprint_body(expanded_tex(ROOT / "paper/main.tex"))
    body = "".join(line for line in body.splitlines(keepends=True)
                   if not line.lstrip().startswith("%"))
    def graphic(match):
        original = (ROOT / "paper" / match.group(2)).resolve()
        relative = "figures/"+original.name
        target = SOURCE / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, target)
        return match.group(1)+"{"+relative+"}"
    body = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}", graphic, body)
    (SOURCE / "main.tex").write_text(body, encoding="utf-8")
    shutil.copyfile(ROOT / "paper/refs.bib", SOURCE / "refs.bib")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", body))
    abstract = re.sub(r"\\(\w+)", lambda m: macros.get(m.group(1), m.group(0)), abstract)
    abstract = " ".join(abstract.replace("$", "").replace(r"\%", "%").replace(r"\xspace", "").split())
    assert not re.search(r"\\[A-Za-z]", abstract)
    assert len(abstract.split()) <= 300
    authors = json.loads((ROOT / "submission/author_declarations.json").read_text(encoding="utf-8"))["authors"]
    metadata = dict(title=TITLE, authors=[a["name"] for a in authors], abstract=abstract,
                    primary_category="quant-ph", license_recommendation="arXiv.org perpetual, non-exclusive license 1.0",
                    journal_reference=None, doi=None, arxiv_identifier=None,
                    repository_name=REPO, repository_url=REPO_URL,
                    comments="Includes 3 figures. Code and data: "+REPO_URL+". Not peer reviewed.")
    (OUT / "submission_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "abstract.txt").write_text(abstract+"\n", encoding="utf-8")
    print("Prepared compact arXiv source; abstract words:", len(abstract.split()))


def package():
    tex, pdf, bbl = (SOURCE / ("main"+suffix) for suffix in (".tex", ".pdf", ".bbl"))
    assert pdf.exists() and bbl.exists(), "Compile with --keep-intermediates first"
    assert pdf.stat().st_mtime >= tex.stat().st_mtime
    assert bbl.stat().st_mtime >= tex.stat().st_mtime
    bibliography = b"".join(line for line in bbl.read_bytes().splitlines(keepends=True)
                            if not line.lstrip().startswith(b"%"))
    entries = {"main.tex": tex.read_bytes(), "main.bbl": bibliography,
               "refs.bib": (SOURCE / "refs.bib").read_bytes()}
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text(encoding="utf-8")):
        entries[name] = (SOURCE / name).read_bytes()
    archive_path = OUT / "arxiv_source.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for name, value in sorted(entries.items()):
            assert all(re.fullmatch(r"[A-Za-z0-9_+.,=-]+", p) for p in Path(name).parts)
            archive.writestr(name, value)
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(entries)
        assert "main.pdf" not in entries
        assert not any(name.startswith("anc/") for name in entries)
        for name, value in entries.items():
            assert archive.read(name) == value
    shutil.copyfile(pdf, OUT / "preprint.pdf")
    report = dict(package_verified=True, not_uploaded=True, source_file="main.tex",
                  supported_arxiv_compiler="xelatex", local_compiler="Tectonic 0.16.9 (XeTeX)",
                  source_archive=archive_path.name, source_archive_bytes=archive_path.stat().st_size,
                  source_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                  preprint_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                  source_files=len(entries), ancillary_files=0,
                  data_code_route="GitHub repository", repository_url=REPO_URL,
                  author_actions_pending=["coauthors review and approve this preprint",
                      "confirm any required funding/conflict acknowledgments before posting",
                      "confirm authors' rights and select the arXiv license"],
                  omitted_local_placeholders=["unconfirmed author contributions", "unconfirmed funding", "unconfirmed competing interests"],
                  files={name: hashlib.sha256(value).hexdigest() for name, value in sorted(entries.items())})
    (OUT / "package_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Verified arXiv upload ZIP:", len(entries), "files;", archive_path.stat().st_size, "bytes")


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--package", action="store_true")
    args=ap.parse_args()
    prepare() if args.prepare else package()
