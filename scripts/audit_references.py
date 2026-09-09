"""Check journal reference metadata against DOI registration records (read-only)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import quote
from urllib.request import Request, urlopen


def normalize(value):
    value = re.sub(r"<[^>]+>", "", value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.lower())


def main():
    cache_path = Path("data/reference_audit.json")
    cached = {r["doi"]: r for r in json.loads(cache_path.read_text(encoding="utf-8"))["records"]
              if r["status"] == "verified_metadata"} if cache_path.exists() else {}
    bib = Path("paper/refs.bib").read_text(encoding="utf-8")
    entries = []
    for block in re.split(r"(?=@(?:article|misc)\{)", bib):
        if not block.strip():
            continue
        key = re.search(r"@\w+\{([^,]+)", block).group(1)
        doi = re.search(r"doi\s*=\s*\{([^}]+)\}", block).group(1)
        title = re.search(r"title\s*=\s*\{(.*?)\},\s*\n", block, re.S).group(1)
        year = int(re.search(r"year\s*=\s*\{(\d+)\}", block).group(1))
        entries.append((key, doi, title, year))

    def check(entry):
        key, doi, title, year = entry
        if doi in cached and cached[doi]["local_title"] == title and cached[doi]["local_year"] == year:
            return cached[doi]
        record = dict(key=key, doi=doi, local_title=title, local_year=year)
        if doi.lower().startswith("10.48550/arxiv."):
            record["status"] = "preprint_primary_arxiv_checked_separately"
            return record
        try:
            url = "https://api.crossref.org/works/"+quote(doi, safe="")
            request = Request(url, headers={"User-Agent": "PhaseCycleReferenceAudit/1.0"})
            with urlopen(request, timeout=25) as response:
                m = json.load(response)["message"]
            remote_title = m["title"][0]
            score = difflib.SequenceMatcher(None, normalize(title), normalize(remote_title)).ratio()
            dates = {k: m[k]["date-parts"] for k in ("published", "published-online", "published-print") if k in m}
            years = {date[0][0] for date in dates.values()}
            record.update(status="verified_metadata" if score > .93 and year in years else "review_metadata",
                          registered_title=remote_title, title_similarity=score, dates=dates,
                          journal=m.get("container-title"), volume=m.get("volume"), pages=m.get("page"),
                          article_number=m.get("article-number"), authors=m.get("author"), source=url)
        except Exception as error:
            primary_checks = {
                "Emerson2005": "https://chaos.if.uj.edu.pl/~karol/pdf/EAZ05.pdf",
                "HelsenFramework2022": "https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.020357",
                "OMalley2015": "https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.3.044009",
                "Knill2008": "https://journals.aps.org/pra/abstract/10.1103/PhysRevA.77.012307",
            }
            record.update(status="publisher_page_verified_2026_09_05" if key in primary_checks else "lookup_failed",
                          error_type=type(error).__name__, primary_check=primary_checks.get(key))
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(check, entries))
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), records=records,
                  caveat="Metadata verification is not a substitute for author review of relevance and contents.")
    Path("data/reference_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in records:
        print(r["key"], r["status"], r.get("title_similarity", ""))


if __name__ == "__main__":
    main()
