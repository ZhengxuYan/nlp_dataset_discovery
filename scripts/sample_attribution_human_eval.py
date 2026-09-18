#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
    parser = argparse.ArgumentParser(description="Sample ACU-level attribution rows for human evidence evaluation.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--condition", default=None, help="Optional condition filter, e.g. oracle or fusion.")
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    if args.condition:
        rows = [row for row in rows if row.get("condition") == args.condition]
    rng = random.Random(args.seed)
    sample = list(rows)
    rng.shuffle(sample)
    sample = sample[: min(args.n, len(sample))]
    write_jsonl(args.output_jsonl, sample)
    print(json.dumps({
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "condition": args.condition,
        "available_rows": len(rows),
        "sampled_rows": len(sample),
        "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
