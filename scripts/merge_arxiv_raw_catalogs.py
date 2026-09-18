#!/usr/bin/env python3
"""Merge arXiv raw CSV exports and deduplicate by arXiv base ID."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION_RE = re.compile(r"^(?P<base>.+?)(?:v(?P<version>\d+))?$")


def arxiv_base_id(arxiv_id: str) -> str:
    match = VERSION_RE.match((arxiv_id or "").strip())
    return match.group("base") if match else arxiv_id.strip()


def arxiv_version(arxiv_id: str) -> int:
    match = VERSION_RE.match((arxiv_id or "").strip())
    if not match or not match.group("version"):
        return 0
    return int(match.group("version"))


def filled_field_count(row: dict[str, str]) -> int:
    return sum(1 for value in row.values() if (value or "").strip())


def row_sort_key(row: dict[str, str]) -> tuple[str, str, int, int]:
    return (
        row.get("Updated Date") or "",
        row.get("Publication Date") or "",
        arxiv_version(row.get("arXiv ID") or ""),
        filled_field_count(row),
    )


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], input_stats: list[dict[str, Any]]) -> dict[str, Any]:
    year_counts = Counter((row.get("Publication Date") or "unknown")[:4] for row in rows)
    primary_counts = Counter(row.get("Primary Category") or "unknown" for row in rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files": input_stats,
        "output_rows": len(rows),
        "year_counts": dict(sorted(year_counts.items())),
        "top_primary_categories": primary_counts.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", action="append", required=True, help="Input raw arXiv CSV. Repeatable.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    fieldnames: list[str] = []
    by_base_id: dict[str, dict[str, str]] = {}
    input_stats: list[dict[str, Any]] = []
    duplicate_rows = 0

    for input_csv in args.input_csv:
        path = Path(input_csv)
        current_fieldnames, rows = read_rows(path)
        if not fieldnames:
            fieldnames = current_fieldnames
        else:
            for field in current_fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)

        kept_from_file = 0
        replaced_existing = 0
        for row in rows:
            arxiv_id = (row.get("arXiv ID") or "").strip()
            if not arxiv_id:
                continue
            base_id = arxiv_base_id(arxiv_id)
            previous = by_base_id.get(base_id)
            if previous is None:
                by_base_id[base_id] = row
                kept_from_file += 1
            elif row_sort_key(row) > row_sort_key(previous):
                by_base_id[base_id] = row
                replaced_existing += 1
                duplicate_rows += 1
            else:
                duplicate_rows += 1

        input_stats.append(
            {
                "path": str(path),
                "rows": len(rows),
                "new_base_ids_seen": kept_from_file,
                "replaced_existing_rows": replaced_existing,
            }
        )

    output_rows = sorted(
        by_base_id.values(),
        key=lambda row: ((row.get("Publication Date") or ""), arxiv_base_id(row.get("arXiv ID") or "")),
    )
    write_csv(Path(args.output_csv), output_rows, fieldnames)

    summary = summarize(output_rows, input_stats)
    summary["duplicate_or_older_rows_removed"] = duplicate_rows
    summary["output_csv"] = args.output_csv
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
