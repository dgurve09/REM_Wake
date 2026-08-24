"""Resolve all DOI references in project Markdown and retain their metadata."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd


DOI_PATTERN = re.compile(r"https://doi\.org/([^\s`<>]+)", re.IGNORECASE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-08-23_reference_doi_audit_v0.1"


def clean_doi(value: str) -> str:
    value = value.rstrip(".,;:")
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1]
    return value


def doi_sources() -> dict[str, list[str]]:
    sources: dict[str, list[str]] = defaultdict(list)
    root = repo_root()
    for path in sorted(root.rglob("*.md")):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        for match in DOI_PATTERN.finditer(text):
            doi = clean_doi(match.group(1)).lower()
            if relative not in sources[doi]:
                sources[doi].append(relative)
    return dict(sources)


def resolve_doi(doi: str) -> dict:
    encoded = urllib.parse.quote(doi, safe="/")
    request = urllib.request.Request(
        f"https://doi.org/{encoded}",
        headers={
            "Accept": "application/vnd.citationstyles.csl+json",
            "User-Agent": "REM-Wake-reference-audit/0.1 (research metadata verification)",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8"))
    returned = str(metadata.get("DOI", doi)).lower()
    issued = metadata.get("issued", {}).get("date-parts", [[None]])
    year = issued[0][0] if issued and issued[0] else None
    return {
        "returned_doi": returned,
        "title": metadata.get("title", ""),
        "type": metadata.get("type", ""),
        "publisher": metadata.get("publisher", ""),
        "issued_year": year,
        "status": "pass" if returned == doi else "fail",
    }


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite changed reviewed output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    sources = doi_sources()
    rows = []
    for index, doi in enumerate(sorted(sources), start=1):
        try:
            metadata = resolve_doi(doi)
            error = "none"
        except Exception as exc:
            metadata = {
                "returned_doi": "",
                "title": "",
                "type": "",
                "publisher": "",
                "issued_year": None,
                "status": "fail",
            }
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "doi": doi,
                **metadata,
                "source_file_count": len(sources[doi]),
                "source_files": ";".join(sources[doi]),
                "error": error,
            }
        )
        print(f"{index:2d}/{len(sources)} {doi}: {metadata['status']}")

    result = pd.DataFrame(rows)
    out_dir = output_dir()
    write_once(
        out_dir / "reference_doi_resolution_v0.1.tsv",
        result.to_csv(sep="\t", index=False, lineterminator="\n"),
    )
    passed = int((result["status"] == "pass").sum())
    readme = f"""# Reference DOI Audit v0.1

**Work date:** 2026-08-23
**Protocol:** `docs/audit/reference_doi_audit_plan_v0.1.md`

| Check | Result |
|---|---:|
| Unique DOI references | {len(result)} |
| DOI identifiers resolved with matching metadata | {passed} |
| Failed identifiers | {len(result) - passed} |

This audit verifies that the recorded DOI identifiers resolve to structured metadata. It does not substitute for reading the cited work or checking every narrative interpretation.
"""
    write_once(out_dir / "README.md", readme)
    failed = result[result["status"] != "pass"]
    if len(failed):
        raise SystemExit(f"DOI failures: {failed[['doi', 'error']].to_dict('records')}")
    print(f"Passed {passed}/{len(result)} DOI resolution checks")


if __name__ == "__main__":
    main()
