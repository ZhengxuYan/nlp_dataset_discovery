#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def usage_value(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full-text dataset extraction outputs.")
    parser.add_argument("jsonl")
    parser.add_argument("--input-price-per-1m", type=float, default=0.0)
    parser.add_argument("--output-price-per-1m", type=float, default=0.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.jsonl)
    runtimes = [float(row.get("runtime_seconds") or 0) for row in rows if row.get("runtime_seconds")]
    prompt_tokens = []
    output_tokens = []
    total_tokens = []
    dataset_counts = []
    acu_counts = []
    warning_counts = []

    for row in rows:
        usage = row.get("llm_usage") or {}
        prompt = usage_value(usage, "prompt_token_count", "input_tokens")
        output = usage_value(usage, "candidates_token_count", "output_tokens")
        total = usage_value(usage, "total_token_count", "total_tokens")
        if prompt:
            prompt_tokens.append(prompt)
        if output:
            output_tokens.append(output)
        if total:
            total_tokens.append(total)
        datasets = row.get("datasets") or []
        dataset_counts.append(len(datasets))
        acu_counts.extend(len(dataset.get("acus") or []) for dataset in datasets)
        warning_counts.append(len(row.get("quality_warnings") or []))

    input_cost = sum(prompt_tokens) / 1_000_000 * args.input_price_per_1m
    output_cost = sum(output_tokens) / 1_000_000 * args.output_price_per_1m
    summary = {
        "rows": len(rows),
        "datasets": sum(dataset_counts),
        "mean_datasets_per_paper": statistics.mean(dataset_counts) if dataset_counts else 0,
        "mean_acus_per_dataset": statistics.mean(acu_counts) if acu_counts else 0,
        "hard_validation_error_rows": sum(1 for row in rows if row.get("validation_errors")),
        "quality_warning_rows": sum(1 for count in warning_counts if count),
        "runtime_seconds_total_observed": sum(runtimes),
        "runtime_seconds_mean_per_paper": statistics.mean(runtimes) if runtimes else None,
        "runtime_seconds_median_per_paper": statistics.median(runtimes) if runtimes else None,
        "prompt_tokens_total": sum(prompt_tokens),
        "output_tokens_total": sum(output_tokens),
        "total_tokens_total": sum(total_tokens),
        "prompt_tokens_mean": statistics.mean(prompt_tokens) if prompt_tokens else None,
        "output_tokens_mean": statistics.mean(output_tokens) if output_tokens else None,
        "total_tokens_mean": statistics.mean(total_tokens) if total_tokens else None,
        "estimated_input_cost": input_cost,
        "estimated_output_cost": output_cost,
        "estimated_total_cost": input_cost + output_cost,
        "input_price_per_1m": args.input_price_per_1m,
        "output_price_per_1m": args.output_price_per_1m,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
