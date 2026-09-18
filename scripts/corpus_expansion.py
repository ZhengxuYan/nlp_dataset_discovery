#!/usr/bin/env python3
"""Utilities for reproducible corpus expansion runs.

This module centralizes year-range validation, output naming, and light-weight
artifact audits so 2023-2025 outputs can coexist with 2020-2025 outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2025


@dataclass(frozen=True)
class YearRange:
    start_year: int = DEFAULT_START_YEAR
    end_year: int = DEFAULT_END_YEAR

    def __post_init__(self) -> None:
        if self.start_year < 1900:
            raise ValueError("start_year must be >= 1900")
        if self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")

    @property
    def label(self) -> str:
        return f"{self.start_year}_{self.end_year}"

    def includes(self, year: int | str | None) -> bool:
        parsed = parse_year(year)
        return parsed is not None and self.start_year <= parsed <= self.end_year

    def filter_records(self, rows: Iterable[Mapping[str, object]]) -> Iterator[Mapping[str, object]]:
        for row in rows:
            if self.includes(extract_record_year(row)):
                yield row


def parse_year(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    for token in (text[:4], text):
        if len(token) >= 4 and token[:4].isdigit():
            year = int(token[:4])
            if 1900 <= year <= 2100:
                return year
    return None


def extract_record_year(row: Mapping[str, object]) -> int | None:
    for key in (
        "year",
        "publication_year",
        "published_year",
        "paper_year",
        "release_year",
    ):
        if key in row:
            year = parse_year(row.get(key))  # type: ignore[arg-type]
            if year is not None:
                return year
    for key in ("published", "published_date", "date", "created", "updated"):
        if key in row:
            year = parse_year(row.get(key))  # type: ignore[arg-type]
            if year is not None:
                return year
    return None


def output_paths(root: Path, year_range: YearRange) -> dict[str, Path]:
    label = year_range.label
    return {
        "raw_catalog": root / "data" / "raw" / f"arxiv_results_{label}.csv",
        "dedup_screening_csv": root / "data" / "processed" / f"arxiv_{label}_dedup_no_acl_for_dataset_screening.csv",
        "dedup_screening_jsonl": root / "data" / "processed" / f"arxiv_{label}_dedup_no_acl_for_dataset_screening.jsonl",
        "llm_screening_jsonl": root / "data" / "processed" / f"arxiv_{label}_dataset_screening_gemini31_flashlite.jsonl",
        "fulltext_inputs_jsonl": root / "data" / "processed" / f"arxiv_{label}_nlp_dataset_intro_for_fulltext.jsonl",
        "fulltext_extractions_jsonl": root / "data" / "census" / f"arxiv_fulltext_dataset_extractions_{label}.jsonl",
        "integrated_dataset_bank_jsonl": root / "data" / "census" / f"integrated_fulltext_dataset_bank_{label}.jsonl",
        "integrated_acu_bank_jsonl": root / "data" / "census" / f"integrated_fulltext_acu_bank_{label}.jsonl",
        "integrated_summary_json": root / "data" / "census" / f"integrated_fulltext_banks_{label}_summary.json",
        "metadata_enriched_jsonl": root / "data" / "census" / f"integrated_fulltext_dataset_bank_{label}_enriched.jsonl",
        "metadata_enriched_summary_json": root / "data" / "census" / f"integrated_fulltext_dataset_bank_{label}_enriched_summary.json",
    }


def is_cloud_placeholder(path: Path) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    return stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if isinstance(value, dict):
                yield value


def iter_csv(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def count_years(path: Path, year_range: YearRange) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if is_cloud_placeholder(path):
        return {"path": str(path), "exists": True, "cloud_placeholder": True}

    rows: Iterable[Mapping[str, object]]
    if path.suffix == ".csv":
        rows = iter_csv(path)
    elif path.suffix in {".jsonl", ".json"}:
        rows = iter_jsonl(path) if path.suffix == ".jsonl" else [json.loads(path.read_text(encoding="utf-8"))]
    else:
        return {"path": str(path), "exists": True, "unsupported": True}

    counts: dict[str, int] = {}
    missing_year = 0
    total = 0
    in_range = 0
    for row in rows:
        total += 1
        year = extract_record_year(row)
        if year is None:
            missing_year += 1
            continue
        counts[str(year)] = counts.get(str(year), 0) + 1
        if year_range.includes(year):
            in_range += 1
    return {
        "path": str(path),
        "exists": True,
        "total_rows": total,
        "in_range_rows": in_range,
        "missing_year_rows": missing_year,
        "year_counts": counts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit corpus expansion output paths and year coverage.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--audit", action="store_true", help="Count available rows by year for expected artifacts.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a compact text report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    year_range = YearRange(args.start_year, args.end_year)
    paths = output_paths(args.root, year_range)
    payload: dict[str, object] = {
        "year_range": asdict(year_range),
        "label": year_range.label,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    if args.audit:
        payload["audit"] = {key: count_years(path, year_range) for key, path in paths.items()}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Corpus expansion label: {year_range.label}")
        for key, path in paths.items():
            marker = "placeholder" if is_cloud_placeholder(path) else ("exists" if path.exists() else "missing")
            print(f"{key}: {path} [{marker}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
