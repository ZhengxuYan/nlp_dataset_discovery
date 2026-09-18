#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/raw/arxiv_results_2023_2025.csv"
DEFAULT_ACL_JSONL = "data/census/acl_anthology/acl_anthology_2023_2025_all_with_abstracts.jsonl"
DEFAULT_OUTPUT_CSV = "data/processed/arxiv_2023_2025_dedup_no_acl_for_dataset_screening.csv"
DEFAULT_OUTPUT_JSONL = "data/processed/arxiv_2023_2025_dedup_no_acl_for_dataset_screening.jsonl"
DEFAULT_SUMMARY_JSON = "data/processed/arxiv_2023_2025_dedup_no_acl_for_dataset_screening_summary.json"
DEFAULT_START_YEAR = 2023
DEFAULT_END_YEAR = 2025


def year_label(start_year: int, end_year: int) -> str:
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    return f"{start_year}_{end_year}"


def default_paths(start_year: int, end_year: int) -> dict[str, str]:
    label = year_label(start_year, end_year)
    return {
        "input_csv": f"data/raw/arxiv_results_{label}.csv",
        "acl_jsonl": f"data/census/acl_anthology/acl_anthology_{label}_all_with_abstracts.jsonl",
        "output_csv": f"data/processed/arxiv_{label}_dedup_no_acl_for_dataset_screening.csv",
        "output_jsonl": f"data/processed/arxiv_{label}_dedup_no_acl_for_dataset_screening.jsonl",
        "summary_json": f"data/processed/arxiv_{label}_dedup_no_acl_for_dataset_screening_summary.json",
    }


def normalize_title(title: str) -> str:
    text = title.casefold()
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def arxiv_base_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def arxiv_version(arxiv_id: str) -> int:
    match = re.search(r"v(\d+)$", arxiv_id.strip())
    return int(match.group(1)) if match else 0


def parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d")
    except Exception:
        return datetime.min


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = Path(path)
    if not target.exists():
        return rows
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_acl_titles(paths: list[str]) -> set[str]:
    titles: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            title = normalize_title(str(row.get("title") or row.get("Title") or ""))
            if title:
                titles.add(title)
    return titles


def read_arxiv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def row_sort_key(row: dict[str, str]) -> tuple[datetime, datetime, int]:
    return (
        parse_date(row.get("Updated Date", "")),
        parse_date(row.get("Publication Date", "")),
        arxiv_version(row.get("arXiv ID", "")),
    )


def compact_row(row: dict[str, str]) -> dict[str, Any]:
    published = row.get("Publication Date") or ""
    year = None
    try:
        year = int(published[:4])
    except Exception:
        pass
    return {
        "paper_id": row.get("arXiv ID") or "",
        "arxiv_id": row.get("arXiv ID") or "",
        "arxiv_base_id": arxiv_base_id(row.get("arXiv ID") or ""),
        "title": row.get("Title") or "",
        "authors": row.get("Authors") or "",
        "author_affiliations": row.get("Author Affiliations") or "",
        "published_date": published,
        "updated_date": row.get("Updated Date") or "",
        "year": year,
        "abstract": row.get("Abstract") or "",
        "categories": row.get("Categories") or "",
        "primary_category": row.get("Primary Category") or "",
        "comment": row.get("Comment") or "",
        "journal_reference": row.get("Journal Reference") or "",
        "doi": row.get("DOI") or "",
        "arxiv_url": row.get("arXiv URL") or "",
        "pdf_url": row.get("PDF URL") or "",
    }


def write_csv(path: str | Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deduplicated arXiv abstract-screening catalog and remove ACL papers already processed.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--acl-jsonl", action="append", default=None, help="ACL/processed JSONL with title fields. Can be repeated.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    paths = default_paths(args.start_year, args.end_year)
    args.input_csv = args.input_csv or paths["input_csv"]
    args.output_csv = args.output_csv or paths["output_csv"]
    args.output_jsonl = args.output_jsonl or paths["output_jsonl"]
    args.summary_json = args.summary_json or paths["summary_json"]
    if args.acl_jsonl is None:
        args.acl_jsonl = [paths["acl_jsonl"]]

    raw_rows = read_arxiv_rows(args.input_csv)
    fieldnames = list(raw_rows[0].keys()) if raw_rows else []

    by_base: dict[str, dict[str, str]] = {}
    arxiv_version_duplicates = 0
    for row in raw_rows:
        arxiv_id = (row.get("arXiv ID") or "").strip()
        if not arxiv_id:
            continue
        base = arxiv_base_id(arxiv_id)
        previous = by_base.get(base)
        if previous is None or row_sort_key(row) > row_sort_key(previous):
            if previous is not None:
                arxiv_version_duplicates += 1
            by_base[base] = row
        else:
            arxiv_version_duplicates += 1

    title_deduped: dict[str, dict[str, str]] = {}
    duplicate_titles = 0
    for row in by_base.values():
        title = normalize_title(row.get("Title") or "")
        if not title:
            title = f"__missing_title__:{row.get('arXiv ID')}"
        previous = title_deduped.get(title)
        if previous is None or row_sort_key(row) > row_sort_key(previous):
            if previous is not None:
                duplicate_titles += 1
            title_deduped[title] = row
        else:
            duplicate_titles += 1

    acl_titles = read_acl_titles(args.acl_jsonl)
    kept_rows: list[dict[str, str]] = []
    removed_acl_overlap = 0
    removed_acl_examples: list[dict[str, str]] = []
    for title, row in title_deduped.items():
        if title in acl_titles:
            removed_acl_overlap += 1
            if len(removed_acl_examples) < 20:
                removed_acl_examples.append({
                    "arxiv_id": row.get("arXiv ID") or "",
                    "title": row.get("Title") or "",
                })
            continue
        kept_rows.append(row)

    kept_rows.sort(key=lambda row: ((row.get("Publication Date") or ""), row.get("arXiv ID") or ""))
    compact_rows = [compact_row(row) for row in kept_rows]
    write_csv(args.output_csv, kept_rows, fieldnames)
    write_jsonl(args.output_jsonl, compact_rows)

    year_counts = Counter(str(row.get("year") or "unknown") for row in compact_rows)
    primary_counts = Counter(row.get("primary_category") or "unknown" for row in compact_rows)
    summary = {
        "input_csv": args.input_csv,
        "acl_jsonl": args.acl_jsonl,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "raw_rows": len(raw_rows),
        "unique_arxiv_base_ids": len(by_base),
        "removed_arxiv_version_duplicates": arxiv_version_duplicates,
        "unique_titles_after_arxiv_dedup": len(title_deduped),
        "removed_duplicate_titles": duplicate_titles,
        "acl_title_keys": len(acl_titles),
        "removed_acl_title_overlap": removed_acl_overlap,
        "output_rows": len(kept_rows),
        "output_csv": args.output_csv,
        "output_jsonl": args.output_jsonl,
        "year_counts": dict(sorted(year_counts.items())),
        "top_primary_categories": primary_counts.most_common(20),
        "removed_acl_examples": removed_acl_examples,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
