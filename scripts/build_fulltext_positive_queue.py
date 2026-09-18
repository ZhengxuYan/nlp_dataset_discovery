#!/usr/bin/env python3
"""Build a full-text extraction queue from abstract-screening positives."""

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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def is_fulltext_candidate(row: Mapping[str, Any]) -> bool:
    return bool(row.get("is_dataset_introducing") and row.get("datasets"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build full-text extraction queue from screening positives.")
    parser.add_argument("--screening-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = list(iter_jsonl(args.screening_jsonl))
    positives = [row for row in rows if is_fulltext_candidate(row)]
    year_counts = Counter(str(row.get("year") or "unknown") for row in positives)
    dataset_counts = sum(len(row.get("datasets") or []) for row in positives)
    written = write_jsonl(args.output_jsonl, positives)
    summary = {
        "screening_jsonl": str(args.screening_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "input_rows": len(rows),
        "fulltext_candidate_rows": written,
        "introduced_dataset_mentions": dataset_counts,
        "year_counts": dict(sorted(year_counts.items())),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
