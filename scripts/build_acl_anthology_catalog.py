#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_cloud_placeholders import is_cloud_placeholder


DEFAULT_ACL_METADATA = os.environ.get(
    "ACL_METADATA_PATH", "ACL-anthology-corpus/data/metadata.jsonl"
)
DEFAULT_ACL_BIB_URL = "https://aclanthology.org/anthology.bib.gz"
DEFAULT_ACL_BIB_GZ = "data/cache/acl_anthology/anthology.bib.gz"
DEFAULT_OUTPUT_DIR = "data/census/acl_anthology"
DEFAULT_START_YEAR = 2023
DEFAULT_END_YEAR = 2025

CORE_VENUE_EXACT = {
    "emnlp-main",
    "coling-main",
    "lrec-main",
    "eacl-main",
    "aacl-main",
    "ijcnlp-main",
    "tacl-1",
    "cl-1",
    "cl-2",
    "cl-3",
    "cl-4",
}

CORE_VENUE_PREFIXES = (
    "acl-long",
    "acl-short",
    "naacl-long",
    "naacl-short",
    "eacl-long",
    "eacl-short",
    "findings-",
)

FRONT_MATTER_TITLE_PREFIXES = (
    "proceedings of",
    "front matter",
    "preface",
    "organizing committee",
    "program committee",
    "author index",
    "table of contents",
    "index of authors",
    "message from",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metadata_jsonl_is_readable(path: str | Path) -> bool:
    target = Path(path)
    return target.exists() and not is_cloud_placeholder(target)


def download_file(url: str, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(output)
    return output


def clean_bib_value(value: str) -> str:
    value = value.strip().rstrip(",")
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return re.sub(r"\s+", " ", value).strip()


def split_bib_authors(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s+and\s+", clean_bib_value(value)) if part.strip()]


def acl_id_from_bib_id(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if re.match(r"^[A-Z]\d{2}-\d{4}$", value):
        return f"{value[0]}{value[1:3]}-{value[3:]}"
    return value


def acl_id_from_url(value: str) -> str:
    match = re.search(r"aclanthology\.org/([^/#?]+)/?", value or "")
    return match.group(1).rstrip("/") if match else ""


def parse_bib_entry(entry_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_value: list[str] = []
    header = entry_text.split("\n", 1)[0]
    id_match = re.match(r"@\w+\{([^,]+),?", header.strip())
    if id_match:
        fields["ID"] = id_match.group(1).strip()
    for line in entry_text.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped == "}":
            continue
        match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(.*)", stripped)
        if match:
            if current_key:
                fields[current_key] = clean_bib_value(" ".join(current_value))
            current_key = match.group(1).lower()
            current_value = [match.group(2)]
        elif current_key:
            current_value.append(stripped)
    if current_key:
        fields[current_key] = clean_bib_value(" ".join(current_value))
    return fields


def iter_bib_entries(path: str | Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        buffer: list[str] = []
        brace_depth = 0
        in_entry = False
        for line in handle:
            if not in_entry:
                if line.lstrip().startswith("@"):
                    in_entry = True
                    buffer = [line]
                    brace_depth = line.count("{") - line.count("}")
                continue
            buffer.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                yield parse_bib_entry("".join(buffer))
                buffer = []
                in_entry = False


def read_bib_gz(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in iter_bib_entries(path):
        url = str(entry.get("url") or "")
        acl_id = acl_id_from_url(url) or acl_id_from_bib_id(str(entry.get("ID") or ""))
        year = str(entry.get("year") or "")
        rows.append(
            {
                "id": acl_id,
                "title": entry.get("title") or "",
                "authors": split_bib_authors(str(entry.get("author") or "")),
                "year": int(year) if year.isdigit() else year,
                "event_url": "",
                "pdf_url": f"https://aclanthology.org/{acl_id}.pdf" if acl_id else "",
                "url": url or (f"https://aclanthology.org/{acl_id}/" if acl_id else ""),
            }
        )
    return rows


def read_metadata(args: argparse.Namespace) -> list[dict[str, Any]]:
    if metadata_jsonl_is_readable(args.metadata_jsonl):
        return read_jsonl(args.metadata_jsonl)
    bib_path = Path(args.metadata_bib_gz)
    if not bib_path.exists() or is_cloud_placeholder(bib_path):
        if not args.download_bib_if_needed:
            raise FileNotFoundError(
                f"Metadata JSONL is unavailable or a cloud placeholder: {args.metadata_jsonl}. "
                f"Provide --metadata-bib-gz or pass --download-bib-if-needed."
            )
        download_file(args.bib_url, bib_path)
    return read_bib_gz(bib_path)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def venue_from_id(acl_id: str) -> str:
    parts = acl_id.split(".")
    if len(parts) < 2:
        return ""
    return parts[1]


def event_slug(event_url: str) -> str:
    if not event_url:
        return ""
    return event_url.rstrip("/").rsplit("/", 1)[-1]


def is_front_matter(row: dict[str, Any]) -> bool:
    acl_id = str(row.get("id") or "")
    title = str(row.get("title") or "").strip().lower()
    if re.search(r"\.0$", acl_id):
        return True
    return any(title.startswith(prefix) for prefix in FRONT_MATTER_TITLE_PREFIXES)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    acl_id = str(row.get("id") or "").strip()
    year = int(row.get("year")) if str(row.get("year") or "").isdigit() else None
    venue = venue_from_id(acl_id)
    event = event_slug(str(row.get("event_url") or ""))
    return {
        "paper_id": f"ACL:{acl_id}",
        "acl_id": acl_id,
        "title": row.get("title") or "",
        "authors": "; ".join(row.get("authors") or []),
        "year": year,
        "venue_prefix": venue,
        "event": event,
        "event_url": row.get("event_url") or "",
        "url": f"https://aclanthology.org/{acl_id}/" if acl_id else "",
        "pdf_url": row.get("pdf_url") or "",
        "source": "acl_anthology",
        "is_front_matter": is_front_matter(row),
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for raw in rows:
        normalized = normalize_row(raw)
        acl_id = normalized["acl_id"]
        if not acl_id:
            continue
        # Keep the first complete-looking row for each ACL id.
        if acl_id not in by_id:
            by_id[acl_id] = normalized
            continue
        current = by_id[acl_id]
        if len(normalized.get("authors", "")) > len(current.get("authors", "")):
            by_id[acl_id] = normalized
    return sorted(by_id.values(), key=lambda row: (row.get("year") or 0, row.get("acl_id") or ""))


def year_label(start_year: int, end_year: int) -> str:
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    return f"{start_year}_{end_year}"


def filter_scope(rows: list[dict[str, Any]], scope: str, start_year: int, end_year: int) -> list[dict[str, Any]]:
    filtered = [
        row for row in rows
        if row.get("year") is not None
        and start_year <= int(row["year"]) <= end_year
        and not row.get("is_front_matter")
    ]
    if scope == "all":
        return filtered
    if scope == "core":
        return [
            row for row in filtered
            if is_core_venue(str(row.get("venue_prefix") or ""))
        ]
    raise ValueError(f"Unknown scope: {scope}")


def is_core_venue(venue_prefix: str) -> bool:
    return venue_prefix in CORE_VENUE_EXACT or any(
        venue_prefix.startswith(prefix) for prefix in CORE_VENUE_PREFIXES
    )


def summarize(
    raw_rows: list[dict[str, Any]],
    deduped_rows: list[dict[str, Any]],
    scope_rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    return {
        "start_year": start_year,
        "end_year": end_year,
        "raw_rows": len(raw_rows),
        "unique_acl_ids": len(deduped_rows),
        "duplicate_rows_removed": len(raw_rows) - len(deduped_rows),
        "front_matter_unique_rows": sum(1 for row in deduped_rows if row.get("is_front_matter")),
        "scope_rows": len(scope_rows),
        "scope_by_year": dict(sorted(Counter(row.get("year") for row in scope_rows).items())),
        "scope_by_venue_prefix": dict(Counter(row.get("venue_prefix") for row in scope_rows).most_common()),
        "scope_by_event": dict(Counter(row.get("event") for row in scope_rows).most_common()),
    }


def markdown(summary: dict[str, Any], scope: str, outputs: dict[str, str]) -> str:
    lines = [
        "# ACL Anthology Catalog",
        "",
        f"- Scope: `{scope}`",
        f"- Raw rows: {summary['raw_rows']}",
        f"- Unique ACL IDs: {summary['unique_acl_ids']}",
        f"- Duplicate rows removed: {summary['duplicate_rows_removed']}",
        f"- Front-matter rows excluded: {summary['front_matter_unique_rows']}",
        f"- Scope papers: {summary['scope_rows']}",
        "",
        "## Papers By Year",
        "",
        "| Year | Papers |",
        "| --- | ---: |",
    ]
    for year, count in summary["scope_by_year"].items():
        lines.append(f"| {year} | {count} |")
    lines.extend(["", "## Top Venue Prefixes", "", "| Venue | Papers |", "| --- | ---: |"])
    for venue, count in list(summary["scope_by_venue_prefix"].items())[:30]:
        lines.append(f"| {venue} | {count} |")
    lines.extend(["", "## Outputs", ""])
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized ACL Anthology paper catalog from local metadata.")
    parser.add_argument("--metadata-jsonl", default=DEFAULT_ACL_METADATA)
    parser.add_argument("--metadata-bib-gz", default=DEFAULT_ACL_BIB_GZ)
    parser.add_argument("--bib-url", default=DEFAULT_ACL_BIB_URL)
    parser.add_argument("--download-bib-if-needed", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scope", choices=["core", "all"], default="core")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    args = parser.parse_args()

    raw_rows = read_metadata(args)
    deduped_rows = dedupe_rows(raw_rows)
    scope_rows = filter_scope(deduped_rows, args.scope, args.start_year, args.end_year)
    summary = summarize(raw_rows, deduped_rows, scope_rows, args.start_year, args.end_year)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"acl_anthology_{year_label(args.start_year, args.end_year)}_{args.scope}"
    outputs = {
        "summary": str(output_dir / f"{stem}_summary.json"),
        "jsonl": str(output_dir / f"{stem}.jsonl"),
        "csv": str(output_dir / f"{stem}.csv"),
        "markdown": str(output_dir / f"{stem}_summary.md"),
    }
    write_json(outputs["summary"], summary)
    write_jsonl(outputs["jsonl"], scope_rows)
    write_csv(
        outputs["csv"],
        scope_rows,
        [
            "paper_id",
            "acl_id",
            "title",
            "authors",
            "year",
            "venue_prefix",
            "event",
            "url",
            "pdf_url",
            "source",
        ],
    )
    Path(outputs["markdown"]).write_text(markdown(summary, args.scope, outputs), encoding="utf-8")
    print(json.dumps({"summary": summary, "outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
