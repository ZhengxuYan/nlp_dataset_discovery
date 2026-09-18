#!/usr/bin/env python3
"""Plan or execute the 2020-2025 corpus expansion pipeline.

The default mode is safe: it prints the commands and writes a run plan without
launching network/API-heavy jobs. Use --execute after placeholder_count is 0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_cloud_placeholders import is_cloud_placeholder
from scripts.corpus_expansion import YearRange, output_paths


DEFAULT_PLAN_OUTPUT = Path("artifacts/corpus_expansion_2020_2025_run_plan.json")


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]
    outputs: list[str]
    requires_no_placeholders: bool = True
    network_or_llm: bool = False


def placeholder_count(root: Path, paths: list[Path]) -> int:
    total = 0
    for path in paths:
        target = path if path.is_absolute() else root / path
        if target.is_file():
            total += int(is_cloud_placeholder(target))
        elif target.is_dir():
            total += sum(1 for child in target.rglob("*") if child.is_file() and is_cloud_placeholder(child))
    return total


def build_steps(root: Path, year_range: YearRange, mode: str) -> list[PipelineStep]:
    paths = output_paths(root, year_range)
    label = year_range.label
    extraction_limit = "25" if mode == "smoke" else "0"
    enrichment_limit = ["--limit", "25"] if mode == "smoke" else []
    health_args = [] if mode == "smoke" else ["--check-url-health"]
    start_date = date(year_range.start_year, 1, 1).isoformat()
    end_date = date(year_range.end_year, 12, 31).isoformat()
    arxiv_command = [
        "python",
        "scrapers/arxiv_scraper/run_arxiv_intervals.py",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--output-file",
        str(paths["raw_catalog"]),
    ]
    if mode == "smoke":
        arxiv_command.extend(["--dry-run", "--sleep-seconds", "0"])

    fulltext_cmd = [
        "python",
        "scripts/run_fulltext_dataset_extraction.py",
        "--census-jsonl",
        str(paths["fulltext_inputs_jsonl"]),
        "--output-jsonl",
        str(paths["fulltext_extractions_jsonl"]),
    ]
    if mode == "smoke":
        fulltext_cmd.extend(["--limit", extraction_limit, "--dry-run"])

    return [
        PipelineStep(
            name="preflight_placeholder_check",
            command=["python", "scripts/check_cloud_placeholders.py", "scripts", "scv", "scrapers", "data", "--summary-only"],
            outputs=[],
            requires_no_placeholders=False,
        ),
        PipelineStep(
            name="acl_anthology_catalog_all",
            command=[
                "python",
                "scripts/build_acl_anthology_catalog.py",
                "--start-year",
                str(year_range.start_year),
                "--end-year",
                str(year_range.end_year),
                "--scope",
                "all",
                "--download-bib-if-needed",
            ],
            outputs=[
                str(root / "data" / "census" / "acl_anthology" / f"acl_anthology_{label}_all.jsonl"),
                str(root / "data" / "census" / "acl_anthology" / f"acl_anthology_{label}_all_summary.json"),
            ],
        ),
        PipelineStep(
            name="arxiv_interval_scrape",
            command=arxiv_command,
            outputs=[str(paths["raw_catalog"])],
            network_or_llm=True,
        ),
        PipelineStep(
            name="arxiv_screening_catalog",
            command=[
                "python",
                "scripts/prepare_arxiv_screening_catalog.py",
                "--start-year",
                str(year_range.start_year),
                "--end-year",
                str(year_range.end_year),
            ],
            outputs=[str(paths["dedup_screening_csv"]), str(paths["dedup_screening_jsonl"])],
        ),
        PipelineStep(
            name="fulltext_dataset_extraction",
            command=fulltext_cmd,
            outputs=[str(paths["fulltext_extractions_jsonl"])],
            network_or_llm=mode != "smoke",
        ),
        PipelineStep(
            name="integrated_fulltext_banks",
            command=[
                "python",
                "scripts/build_integrated_fulltext_banks.py",
                "--input-jsonl",
                str(paths["fulltext_extractions_jsonl"]),
                "--source-corpus",
                "arxiv_acl",
                "--output-extractions-jsonl",
                str(root / "data" / "census" / f"integrated_fulltext_dataset_extractions_{label}.jsonl"),
                "--output-dataset-bank-jsonl",
                str(paths["integrated_dataset_bank_jsonl"]),
                "--output-acu-bank-jsonl",
                str(paths["integrated_acu_bank_jsonl"]),
                "--summary-json",
                str(paths["integrated_summary_json"]),
            ],
            outputs=[
                str(paths["integrated_dataset_bank_jsonl"]),
                str(paths["integrated_acu_bank_jsonl"]),
                str(paths["integrated_summary_json"]),
            ],
        ),
        PipelineStep(
            name="public_metadata_enrichment",
            command=[
                "python",
                "scripts/enrich_dataset_metadata.py",
                "--input",
                str(paths["integrated_dataset_bank_jsonl"]),
                "--output",
                str(paths["metadata_enriched_jsonl"]),
                "--summary-output",
                str(paths["metadata_enriched_summary_json"]),
                "--cache",
                "data/cache/public_metadata_api_cache.json",
                *enrichment_limit,
                *health_args,
            ],
            outputs=[str(paths["metadata_enriched_jsonl"]), str(paths["metadata_enriched_summary_json"])],
            network_or_llm=True,
        ),
        PipelineStep(
            name="enrichment_stable_id_audit",
            command=[
                "python",
                "scripts/audit_enrichment_stable_ids.py",
                "--input-jsonl",
                str(paths["integrated_dataset_bank_jsonl"]),
                "--enriched-jsonl",
                str(paths["metadata_enriched_jsonl"]),
                "--output-json",
                str(root / "artifacts" / f"enrichment_stable_id_audit_{label}.json"),
                "--output-md",
                str(root / "artifacts" / f"enrichment_stable_id_audit_{label}.md"),
            ],
            outputs=[
                str(root / "artifacts" / f"enrichment_stable_id_audit_{label}.json"),
                str(root / "artifacts" / f"enrichment_stable_id_audit_{label}.md"),
            ],
        ),
        PipelineStep(
            name="metadata_coverage_report",
            command=[
                "python",
                "scripts/summarize_metadata_coverage.py",
                "--input-jsonl",
                str(paths["metadata_enriched_jsonl"]),
                "--output-json",
                str(root / "artifacts" / f"metadata_coverage_{label}.json"),
                "--output-md",
                str(root / "artifacts" / f"metadata_coverage_{label}.md"),
            ],
            outputs=[
                str(root / "artifacts" / f"metadata_coverage_{label}.json"),
                str(root / "artifacts" / f"metadata_coverage_{label}.md"),
            ],
        ),
        PipelineStep(
            name="metadata_schema_audit",
            command=[
                "python",
                "scripts/audit_metadata_schema.py",
                "--input-jsonl",
                str(paths["metadata_enriched_jsonl"]),
                "--output-json",
                str(root / "artifacts" / f"metadata_schema_audit_{label}.json"),
                "--output-md",
                str(root / "artifacts" / f"metadata_schema_audit_{label}.md"),
            ],
            outputs=[
                str(root / "artifacts" / f"metadata_schema_audit_{label}.json"),
                str(root / "artifacts" / f"metadata_schema_audit_{label}.md"),
            ],
        ),
        PipelineStep(
            name="metadata_review_sample",
            command=[
                "python",
                "scripts/build_metadata_review_sample.py",
                "--input-jsonl",
                str(paths["metadata_enriched_jsonl"]),
                "--output-json",
                str(root / "artifacts" / f"metadata_review_sample_{label}.json"),
                "--output-jsonl",
                str(root / "artifacts" / f"metadata_review_sample_{label}.jsonl"),
                "--output-md",
                str(root / "artifacts" / f"metadata_review_sample_{label}.md"),
            ],
            outputs=[
                str(root / "artifacts" / f"metadata_review_sample_{label}.json"),
                str(root / "artifacts" / f"metadata_review_sample_{label}.jsonl"),
                str(root / "artifacts" / f"metadata_review_sample_{label}.md"),
            ],
        ),
        PipelineStep(
            name="expansion_regression_audit",
            command=[
                "python",
                "scripts/audit_expansion_regression.py",
                "--new-start-year",
                str(year_range.start_year),
                "--new-end-year",
                str(year_range.end_year),
                "--output-json",
                str(root / "artifacts" / f"expansion_regression_audit_{label}.json"),
                "--output-md",
                str(root / "artifacts" / f"expansion_regression_audit_{label}.md"),
            ],
            outputs=[
                str(root / "artifacts" / f"expansion_regression_audit_{label}.json"),
                str(root / "artifacts" / f"expansion_regression_audit_{label}.md"),
            ],
        ),
    ]


def plan_payload(root: Path, year_range: YearRange, mode: str, steps: list[PipelineStep], placeholders: int) -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "mode": mode,
        "year_range": asdict(year_range),
        "placeholder_count": placeholders,
        "safe_to_execute_full_pipeline": placeholders == 0,
        "steps": [
            {
                "name": step.name,
                "command": step.command,
                "command_string": " ".join(step.command),
                "outputs": step.outputs,
                "requires_no_placeholders": step.requires_no_placeholders,
                "network_or_llm": step.network_or_llm,
            }
            for step in steps
        ],
    }


def run_steps(steps: list[PipelineStep], root: Path, allow_network_steps: bool) -> None:
    for step in steps:
        if step.network_or_llm and not allow_network_steps:
            print(f"Skipping network/LLM step without --allow-network-steps: {step.name}")
            continue
        print(f"Running {step.name}: {' '.join(step.command)}")
        subprocess.run(step.command, cwd=root, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or execute the 2020-2025 corpus expansion pipeline.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-network-steps", action="store_true")
    parser.add_argument("--ignore-placeholders", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = args.root.resolve()
    year_range = YearRange(args.start_year, args.end_year)
    steps = build_steps(root, year_range, args.mode)
    placeholders = placeholder_count(root, [Path("scripts"), Path("scv"), Path("scrapers"), Path("data")])
    payload = plan_payload(root, year_range, args.mode, steps, placeholders)

    plan_output = args.plan_output if args.plan_output.is_absolute() else root / args.plan_output
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote run plan: {plan_output}")
    for step in steps:
        print(f"- {step.name}: {' '.join(step.command)}")

    if not args.execute:
        return 0
    if placeholders and not args.ignore_placeholders:
        print(f"Refusing to execute because placeholder_count={placeholders}. Hydrate files or pass --ignore-placeholders.")
        return 2
    run_steps(steps, root, args.allow_network_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
