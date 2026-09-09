"""
Structural checks on the manuscript that do not need a LaTeX installation.

These checks complement the PDF build by identifying an unbalanced environment,
a dangling reference or a citation missing from the bibliography. The optional
submission check also identifies unfinished declarations and missing package approval.

    .venv\\Scripts\\python.exe scripts\\check_manuscript.py
"""
from __future__ import annotations

import re
import sys
import argparse
import hashlib
import json
from pathlib import Path

TEX = Path("paper") / "main.tex"
BIB = Path("paper") / "refs.bib"

ENVIRONMENTS = (
    "document", "abstract", "table", "figure", "itemize", "enumerate",
    "tabular", "ruledtabular", "equation", "acknowledgments",
)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--submission",action="store_true",help="also flag unfinished author facts and package approval")
    args=ap.parse_args()
    if not TEX.exists():
        print(f"{TEX} not found")
        return 1
    visited=set()
    def expand(path):
        path=path.resolve()
        if path in visited:
            raise ValueError(f"repeated or cyclic TeX input: {path}")
        visited.add(path)
        value=path.read_text(encoding="utf-8")
        def replace(match):
            child=path.parent/match.group(1)
            if not child.suffix:child=child.with_suffix(".tex")
            return expand(child)
        return re.sub(r"\\input\{([^}]+)\}",replace,value)
    try:
        raw=expand(TEX)
    except (OSError,ValueError) as error:
        print(f"TeX input error: {error}")
        return 1
    # Strip LaTeX comments before any analysis. A commented-out \begin{...} or \ref{...}
    # is not part of the document, and counting it produced a phantom unbalanced
    # environment from a comment that merely NAMED the environment it was explaining.
    body = re.sub(r"(?<!\\)%.*", "", raw)
    issues: list[str] = []

    if args.submission:
        pending_patterns=(r"author contribution statement is pending",r"declaration is pending",
                          r"funding statement is pending",
                          r"Public archival deposition and a permanent identifier are pending",
                          r"awaiting completion of its hardware acquisition",
                          r"has no completed hardware dataset",r"REPOSITORY URL")
        for pattern in pending_patterns:
            if re.search(pattern,body,re.IGNORECASE):
                issues.append(f"submission requirement unfinished: {pattern}")
        manifest_path = Path("submission/package_manifest.json")
        if not manifest_path.exists():
            issues.append("submission package has not been built")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("ready_to_submit"):
                issues.append("author facts/final-version approval remain incomplete; see submission/package_manifest.json")
            for record in manifest["files"]:
                path = manifest_path.parent / record["file"]
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    issues.append(f"submission package file missing or changed: {record['file']}")

    for env in ENVIRONMENTS:
        opened = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", body))
        closed = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", body))
        if opened != closed:
            issues.append(f"environment {env!r}: {opened} begin vs {closed} end")

    if body.count("{") != body.count("}"):
        issues.append(f"brace imbalance: {body.count('{')} open vs {body.count('}')} close")

    # Every environment used must be balanced AND defined. Checking only a hand-listed set
    # of environments misses the more dangerous case: an environment that is never declared
    # at all. `proposition` was used in the appendix and defined nowhere, which REVTeX would
    # have rejected outright -- balance checks pass happily on a \begin/\end pair for an
    # environment that does not exist.
    used_envs = set(re.findall(r"\\begin\{([A-Za-z*]+)\}", body))
    declared = set(re.findall(r"\\newtheorem\{([^}]+)\}", body))
    declared |= set(re.findall(r"\\newenvironment\{([^}]+)\}", body))
    builtin = {
        "document", "abstract", "table", "table*", "figure", "figure*", "itemize",
        "enumerate", "description", "tabular", "tabular*", "ruledtabular", "equation",
        "equation*", "align", "align*", "gather", "gather*", "eqnarray", "eqnarray*",
        "acknowledgments", "center", "quote", "verbatim", "widetext", "split", "cases",
        "array", "pmatrix", "bmatrix", "matrix", "thebibliography",
    }
    undefined = sorted(used_envs - builtin - declared)
    if undefined:
        issues.append(f"environments used but never defined: {undefined}")
    for env in sorted(used_envs):
        o = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", body))
        c = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", body))
        if o != c:
            issues.append(f"environment {env!r}: {o} begin vs {c} end")

    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", body))
    if refs - labels:
        issues.append(f"undefined \\ref targets: {sorted(refs - labels)}")
    unused = labels - refs
    if unused:
        print(f"  note: labels defined but never referenced: {sorted(unused)}")

    keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", body):
        keys |= {k.strip() for k in group.split(",")}
    bibkeys: set[str] = set()
    if BIB.exists():
        bibkeys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
    if keys - bibkeys:
        issues.append(f"cited but absent from refs.bib: {sorted(keys - bibkeys)}")

    # Figures must exist, or the build silently drops them.
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        p = (TEX.parent / g).resolve()
        if not p.exists():
            issues.append(f"missing figure file: {g}")

    print("manuscript structural check:", "OK" if not issues else "ISSUES FOUND")
    for i in issues:
        print("  -", i)
    print(f"  labels={len(labels)} refs={len(refs)} citations={len(keys)} "
          f"bib entries={len(bibkeys)} TeX files={len(visited)} words~{len(body.split())}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
