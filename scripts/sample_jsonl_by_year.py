#!/usr/bin/env python3
"""Create deterministic year-stratified JSONL samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def row_year(row: Mapping[str, Any]) -> str:
    value = row.get("year")
    if value:
        return str(value)
    published = str(row.get("published_date") or row.get("publication_date") or "")
    if len(published) >= 4 and published[:4].isdigit():
        return published[:4]
    return "unknown"


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a JSONL catalog by publication year.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--per-year", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    if args.per_year <= 0:
        raise ValueError("--per-year must be positive")

    wanted_years = [str(year) for year in range(args.start_year, args.end_year + 1)]
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_counts: Counter[str] = Counter()

    for row in iter_jsonl(args.input_jsonl):
        year = row_year(row)
        if year not in wanted_years:
            continue
        input_counts[year] += 1
        if len(by_year[year]) < args.per_year:
            by_year[year].append(row)

    missing = [year for year in wanted_years if len(by_year[year]) < args.per_year]
    if missing:
        missing_desc = ", ".join(f"{year}: {len(by_year[year])}/{args.per_year}" for year in missing)
        raise RuntimeError(f"Not enough rows for requested sample: {missing_desc}")

    selected: list[dict[str, Any]] = []
    for year in wanted_years:
        selected.extend(by_year[year])

    write_jsonl(args.output_jsonl, selected)
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "per_year": args.per_year,
        "rows_written": len(selected),
        "input_year_counts": {year: input_counts[year] for year in wanted_years},
        "sample_year_counts": {year: len(by_year[year]) for year in wanted_years},
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
