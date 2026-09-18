#!/usr/bin/env python3
"""Prepare incremental 2020-2022 queues for a 2020-2025 main analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
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


def row_year(row: Mapping[str, Any]) -> int | None:
    value = row.get("year") or str(row.get("published_date") or "")[:4]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def stable_id(row: Mapping[str, Any]) -> str:
    value = row.get("paper_id") or row.get("acl_id") or row.get("arxiv_id")
    if not value:
        raise ValueError("row is missing paper_id, acl_id, and arxiv_id")
    return str(value)


def filter_years(rows: Iterable[dict[str, Any]], start_year: int, end_year: int) -> list[dict[str, Any]]:
    return [row for row in rows if (year := row_year(row)) is not None and start_year <= year <= end_year]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [stable_id(row) for row in rows]
    years = Counter(str(row_year(row)) for row in rows)
    with_abstract = sum(bool(str(row.get("abstract") or "").strip()) for row in rows)
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "by_year": dict(sorted(years.items())),
        "rows_with_abstract": with_abstract,
        "rows_without_abstract": len(rows) - with_abstract,
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    arxiv = manifest["arxiv_queue"]
    acl = manifest["acl_queue"]
    seed = manifest["arxiv_seed"]
    lines = [
        "# 2020-2025 Main Analysis Expansion",
        "",
        f"- Generated: {manifest['generated_at']}",
        "- Incremental query years: 2020-2022",
        "- Existing 2023-2025 dataset/DCU bank is preserved and reused.",
        "- A pre-2020 prior-only DCU bank is required before scoring 2020 queries with earlier-year evidence.",
        "",
        "## Screening Queues",
        "",
        "| Source | Rows | Existing seed rows | Remaining before new calls | Abstract coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| arXiv 2020-2022 | {arxiv['rows']:,} | {seed['rows']:,} | {manifest['arxiv_remaining_after_seed']:,} | {arxiv['rows_with_abstract'] / arxiv['rows']:.1%} |",
        f"| ACL 2020-2022 | {acl['rows']:,} | 0 | {acl['rows']:,} | {acl['rows_with_abstract'] / acl['rows']:.1%} |",
        "",
        "## Execution Order",
        "",
        "1. Human-audit the stratified screening sample and freeze the classifier prompt/model.",
        "2. Run resumable abstract screening on the two incremental queues.",
        "3. Build full-text queues from dataset-introducing positives and run extraction.",
        "4. Merge 2020-2022 extraction records with the existing 2023-2025 dataset/DCU bank.",
        "5. Build a pre-2020 prior-only DCU bank, then run year-consistent attribution for 2020-2025.",
        "6. Rebuild all corpus tables, trend figures, significance analyses, abstract, results, and limitations.",
        "",
        "## Resumable Screening Commands",
        "",
        "The arXiv output is pre-seeded with completed pilot rows. Both commands skip IDs already present.",
        "",
        "```bash",
        manifest["commands"]["arxiv"],
        "",
        manifest["commands"]["acl"],
        "```",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv-catalog", type=Path, required=True)
    parser.add_argument("--acl-catalog", type=Path, required=True)
    parser.add_argument("--arxiv-pilot", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_year > args.end_year:
        raise ValueError("start year must not exceed end year")

    arxiv_rows = filter_years(iter_jsonl(args.arxiv_catalog), args.start_year, args.end_year)
    acl_rows = filter_years(iter_jsonl(args.acl_catalog), args.start_year, args.end_year)
    arxiv_ids = {stable_id(row) for row in arxiv_rows}
    seed_by_id: dict[str, dict[str, Any]] = {}
    for row in filter_years(iter_jsonl(args.arxiv_pilot), args.start_year, args.end_year):
        identifier = stable_id(row)
        if identifier in arxiv_ids:
            seed_by_id[identifier] = row
    seed_rows = list(seed_by_id.values())

    arxiv_queue = args.output_dir / "arxiv_2020_2022_for_dataset_screening.jsonl"
    acl_queue = args.output_dir / "acl_2020_2022_for_dataset_screening.jsonl"
    arxiv_output = args.output_dir / "arxiv_2020_2022_dataset_screening_gemini31_flashlite.jsonl"
    acl_output = args.output_dir / "acl_2020_2022_dataset_screening_gemini31_flashlite.jsonl"
    write_jsonl(arxiv_queue, arxiv_rows)
    write_jsonl(acl_queue, acl_rows)
    write_jsonl(arxiv_output, seed_rows)

    command_base = "python scripts/run_dataset_census_classifier.py"
    common = "--backend gemini --model gemini-3.1-flash-lite --batch-size 10 --workers 2 --max-retries 3 --sleep 0.2"
    commands = {
        "arxiv": (
            f"{command_base} --catalog {arxiv_queue} --output-jsonl {arxiv_output} "
            f"--error-jsonl {arxiv_output.with_suffix('.errors.jsonl')} {common}"
        ),
        "acl": (
            f"{command_base} --catalog {acl_queue} --output-jsonl {acl_output} "
            f"--error-jsonl {acl_output.with_suffix('.errors.jsonl')} {common}"
        ),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "incremental_years": [args.start_year, args.end_year],
        "arxiv_queue_path": str(arxiv_queue),
        "acl_queue_path": str(acl_queue),
        "arxiv_output_path": str(arxiv_output),
        "acl_output_path": str(acl_output),
        "arxiv_queue": summarize_rows(arxiv_rows),
        "acl_queue": summarize_rows(acl_rows),
        "arxiv_seed": summarize_rows(seed_rows),
        "arxiv_remaining_after_seed": len(arxiv_rows) - len(seed_rows),
        "commands": commands,
        "pre_2020_prior_bank_required": True,
        "safe_to_run_attribution_without_pre_2020_prior_bank": False,
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.artifact_dir / "main_analysis_expansion_2020_2025_manifest.json"
    md_path = args.artifact_dir / "main_analysis_expansion_2020_2025_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
