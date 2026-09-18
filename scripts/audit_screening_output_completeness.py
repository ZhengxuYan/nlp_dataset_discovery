#!/usr/bin/env python3
"""Audit and reconcile screening output completeness against its input catalog."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_id(row: Mapping[str, Any]) -> str:
    value = row.get("paper_id") or row.get("arxiv_id") or row.get("id")
    if not value:
        raise ValueError(f"Row missing paper_id/arxiv_id/id: {row}")
    return str(value)


def build_summary(
    catalog_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    *,
    missing_ids: list[str],
    duplicate_ids: list[str],
    extra_ids: list[str],
) -> dict[str, Any]:
    clean_ids = {row_id(row) for row in output_rows if row_id(row) not in extra_ids}
    return {
        "catalog_rows": len(catalog_rows),
        "raw_output_rows": len(output_rows),
        "clean_rows": len(clean_ids),
        "missing_count": len(missing_ids),
        "duplicate_count": len(duplicate_ids),
        "extra_count": len(extra_ids),
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "extra_ids": extra_ids,
        "is_complete": len(missing_ids) == 0 and len(extra_ids) == 0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit screening output completeness against an input catalog.")
    parser.add_argument("--catalog-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--clean-output-jsonl", type=Path)
    parser.add_argument("--missing-catalog-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    catalog_rows = list(iter_jsonl(args.catalog_jsonl))
    output_rows = list(iter_jsonl(args.output_jsonl))
    catalog_by_id = {row_id(row): row for row in catalog_rows}
    catalog_ids = [row_id(row) for row in catalog_rows]
    output_counts = Counter(row_id(row) for row in output_rows)

    duplicate_ids = sorted([paper_id for paper_id, count in output_counts.items() if count > 1])
    extra_ids = sorted([paper_id for paper_id in output_counts if paper_id not in catalog_by_id])
    missing_ids = [paper_id for paper_id in catalog_ids if paper_id not in output_counts]

    if args.clean_output_jsonl:
        seen: set[str] = set()
        output_by_id: dict[str, dict[str, Any]] = {}
        for row in output_rows:
            paper_id = row_id(row)
            if paper_id in seen or paper_id not in catalog_by_id:
                continue
            seen.add(paper_id)
            output_by_id[paper_id] = row
        write_jsonl(args.clean_output_jsonl, (output_by_id[paper_id] for paper_id in catalog_ids if paper_id in output_by_id))

    if args.missing_catalog_jsonl:
        write_jsonl(args.missing_catalog_jsonl, (catalog_by_id[paper_id] for paper_id in missing_ids))

    summary = build_summary(
        catalog_rows,
        output_rows,
        missing_ids=missing_ids,
        duplicate_ids=duplicate_ids,
        extra_ids=extra_ids,
    )
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
