#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and sort dataset census JSONL output by paper_id.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--prefer", choices=["first", "last"], default="last")
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            continue
        if paper_id in by_id:
            duplicate_count += 1
            if args.prefer == "last":
                by_id[paper_id] = row
        else:
            by_id[paper_id] = row
    normalized = [by_id[paper_id] for paper_id in sorted(by_id)]
    write_jsonl(args.output_jsonl, normalized)
    print(json.dumps({
        "input_rows": len(rows),
        "output_rows": len(normalized),
        "duplicate_rows_removed": duplicate_count,
        "output_jsonl": args.output_jsonl,
    }, indent=2))


if __name__ == "__main__":
    main()
