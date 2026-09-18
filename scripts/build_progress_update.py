#!/usr/bin/env python3
"""Build a concise progress update for the 2020-2025 corpus expansion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_cloud_placeholders import is_cloud_placeholder
from scripts.corpus_expansion import YearRange, output_paths


DEFAULT_OUTPUT = Path("artifacts/progress_update_2020_2025.md")


def count_placeholders(paths: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for base in paths:
        count = 0
        if base.is_file():
            count = int(is_cloud_placeholder(base))
        elif base.is_dir():
            for child in base.rglob("*"):
                if child.is_file() and is_cloud_placeholder(child):
                    count += 1
        counts[str(base)] = count
    return counts


def artifact_status(root: Path, year_range: YearRange) -> dict[str, dict[str, object]]:
    statuses: dict[str, dict[str, object]] = {}
    for name, path in output_paths(root, year_range).items():
        statuses[name] = {
            "path": str(path),
            "exists": path.exists(),
            "cloud_placeholder": is_cloud_placeholder(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return statuses


def render_markdown(
    root: Path,
    year_range: YearRange,
    placeholder_counts: dict[str, int],
    statuses: dict[str, dict[str, object]],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total_placeholders = sum(placeholder_counts.values())
    path_lines = "\n".join(
        f"- `{name}`: `{status['path']}`"
        for name, status in statuses.items()
    )
    placeholder_lines = "\n".join(
        f"- `{path}`: {count}"
        for path, count in placeholder_counts.items()
    )
    missing_outputs = [name for name, status in statuses.items() if not status["exists"]]
    missing_lines = "\n".join(f"- `{name}`" for name in missing_outputs) or "- None"

    return f"""# 2020-2025 Dataset Discovery Progress Update

Generated: {generated_at}
Workspace: `{root}`

## Completed
- Added a reusable year-range utility for `2020_2025` corpus output naming and audit checks.
- Added a cloud-placeholder checker so we can detect files that will block pipeline reads before launching long jobs.
- Parameterized real pipeline entrypoints for ACL Anthology catalog building, arXiv interval scraping, arXiv screening catalog preparation, and dataset census defaults.
- Added a safe orchestrator that writes a 2020-2025 run plan and refuses full execution while cloud placeholders remain.
- Added a public metadata enrichment script for paper citations/identifiers and dataset resource metadata.
- Added coverage reporting and professor-facing update generation for citation/download/star metadata progress.
- Added an evidence-backed checklist mapping original plan items to done/ready/blocked statuses.
- Added a local end-to-end smoke run that builds an integrated dataset/ACU bank, applies cached offline public metadata, and reports citation/download/star coverage.
- Added a consolidated repository runbook with safe smoke commands, full-run commands, artifacts, and verification steps.
- Added a hydration manifest and high-priority file list that prioritize remaining cloud placeholders by root, directory, file type, and execution relevance.
- Added a grouped remaining high-priority hydration queue so hydration can proceed by directory instead of a flat file list.
- Added a hydration action guide that turns the high-priority placeholder list into concrete manual steps and verification commands.
- Added a short hydration status update artifact that combines the current blocker, local smoke evidence, and exact next commands.
- Added concise English and Chinese professor update briefs suitable for meeting notes or short messages.
- Added a one-command local refresh script that rebuilds run plans, hydration reports, smoke artifacts, audits, and update summaries.
- Added a consolidated artifact index so all progress/update deliverables can be found from one file.
- Added a one-page status packet that summarizes current evidence, blockers, and next action.
- Added a full-run readiness gate that reports whether placeholders, run-plan steps, local smoke outputs, and refresh status are ready.
- Added a post-hydration sequence runner that gates on readiness, then runs staged 2020-2021 smoke, full 2020-2025 expansion, refresh, and final readiness reporting.
- Added an old-vs-expanded regression audit that checks whether 2020-2025 artifacts preserve the 2023-2025 overlap after the full run.
- Added a metadata schema audit that checks whether enriched rows contain the planned paper/resource metadata field groups and missing-reason counts.
- Added a metadata review sample for manual inspection of high/low citation rows and HF/GitHub-linked datasets.
- Added focused tests for year filtering, output naming, identifier extraction, URL classification, and offline enrichment behavior.

## Current Blocker
- The workspace still has {total_placeholders} cloud placeholder files across the checked pipeline/data directories.
- Full 2020-2025 reruns should wait until this count is `0`; otherwise old scripts/data reads can hang.

Placeholder counts:
{placeholder_lines}

## 2020-2025 Target Artifacts
{path_lines}

Missing target outputs:
{missing_lines}

## Next Commands
```bash
python scripts/check_cloud_placeholders.py scripts scv scrapers data --fail-on-placeholder
python scripts/refresh_expansion_status_2020_2025.py
python scripts/validate_expansion_readiness.py --allow-blocked
python scripts/run_post_hydration_expansion_sequence.py --execute --allow-network-steps
python scripts/build_expansion_artifact_index.py
python scripts/build_expansion_status_packet.py
python scripts/check_cloud_placeholders.py scripts scv scrapers data --summary-only
python scripts/build_hydration_manifest.py
python scripts/build_remaining_hydration_queue.py
python scripts/build_hydration_action_guide.py
python scripts/build_hydration_status_update.py
python scripts/build_professor_update_brief.py
python scripts/check_cloud_placeholders.py --paths-file artifacts/high_priority_hydration_files_2020_2025.txt --summary-only
python scripts/audit_expansion_regression.py
python scripts/audit_enrichment_stable_ids.py \\
  --input-jsonl data/census/integrated_fulltext_dataset_bank_2020_2025.jsonl \\
  --enriched-jsonl data/census/integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl \\
  --output-json artifacts/enrichment_stable_id_audit_2020_2025.json \\
  --output-md artifacts/enrichment_stable_id_audit_2020_2025.md
python scripts/audit_metadata_schema.py \\
  --input-jsonl data/census/integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl \\
  --output-json artifacts/metadata_schema_audit_2020_2025.json \\
  --output-md artifacts/metadata_schema_audit_2020_2025.md
python scripts/build_metadata_review_sample.py \\
  --input-jsonl data/census/integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl \\
  --output-json artifacts/metadata_review_sample_2020_2025.json \\
  --output-jsonl artifacts/metadata_review_sample_2020_2025.jsonl \\
  --output-md artifacts/metadata_review_sample_2020_2025.md
python scripts/run_corpus_expansion_2020_2025.py --mode smoke
python scripts/run_local_smoke_2020_2025.py artifacts/local_smoke_2020_2025
python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps
python scripts/build_professor_update.py
python scripts/build_expansion_checklist.py
python scripts/build_acl_anthology_catalog.py --start-year 2020 --end-year 2025 --scope all
python scrapers/arxiv_scraper/run_arxiv_intervals.py --start-date 2020-01-01 --end-date 2025-12-31
python scripts/corpus_expansion.py --start-year 2020 --end-year 2025 --audit --json
python scripts/prepare_arxiv_screening_catalog.py --start-year 2020 --end-year 2025
python scripts/enrich_dataset_metadata.py \\
  --input data/census/integrated_fulltext_dataset_bank_2020_2025.jsonl \\
  --output data/census/integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl \\
  --summary-output data/census/integrated_fulltext_dataset_bank_2020_2025_enriched_summary.json \\
  --cache data/cache/public_metadata_api_cache.json \\
  --check-url-health
python scripts/summarize_metadata_coverage.py \\
  --input-jsonl data/census/integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl \\
  --output-json artifacts/metadata_coverage_2020_2025.json \\
  --output-md artifacts/metadata_coverage_2020_2025.md
```

## Status For Professor
We have moved from a hardcoded 2023-2025 setup toward a reproducible 2020-2025 pipeline interface. The real ACL/arXiv/census entrypoints now accept year-range parameters, the new output naming preserves old artifacts, deterministic public metadata enrichment is implemented, and an orchestrator now records the exact smoke/full commands while preventing accidental runs before hydration. Paper metadata matching now uses exact DOI/arXiv identifiers first, then scored title fuzzy fallback with low-similarity rejection and provenance fields. Dataset-resource enrichment uses direct URLs first and can fall back to scored dataset-name lookup for Hugging Face downloads and Papers with Code links when URLs are missing. A local end-to-end smoke run demonstrates the integrated bank plus citation/download/star metadata flow without network access and now includes a stable-ID audit to verify that enriched metadata is matched back by paper/dataset record ID rather than row position. `README_EXPANSION_2020_2025.md` documents the handoff, and `artifacts/professor_update_brief_2020_2025.md` plus `artifacts/professor_update_brief_zh_2020_2025.md` give concise update-ready summaries. `artifacts/expansion_artifact_index_2020_2025.md` now provides a single index of deliverables, and `artifacts/expansion_status_packet_2020_2025.md` gives a one-page current-state packet. `scripts/refresh_expansion_status_2020_2025.py` refreshes the local run plan, hydration reports, smoke artifacts, audits, and update summaries in one command, while `scripts/validate_expansion_readiness.py` produces a clear ready/blocked gate and `scripts/run_post_hydration_expansion_sequence.py` provides the guarded post-hydration execution path. The hydration manifest, grouped remaining queue, `artifacts/hydration_action_guide_2020_2025.md`, `artifacts/hydration_status_2020_2025.md`, and `artifacts/high_priority_hydration_files_2020_2025.txt` make the remaining blocker actionable. `scripts/audit_enrichment_stable_ids.py` is ready to audit ID-safe enrichment joins, `scripts/audit_metadata_schema.py` is ready to audit field completeness and missing reasons, `scripts/build_metadata_review_sample.py` is ready for manual enrichment sample checks, and `scripts/audit_expansion_regression.py` is ready to verify that the expanded 2020-2025 artifacts do not shrink the 2023-2025 overlap. The remaining operational blocker is local file hydration, not method design.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Markdown progress update for the expansion work.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--placeholder-path",
        action="append",
        type=Path,
        dest="placeholder_paths",
        default=None,
        help="Path to include in placeholder counts; can be passed multiple times.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = args.root
    year_range = YearRange(args.start_year, args.end_year)
    placeholder_paths = args.placeholder_paths or [Path("scripts"), Path("scv"), Path("scrapers"), Path("data")]
    placeholder_counts = count_placeholders([root / path for path in placeholder_paths])
    statuses = artifact_status(root, year_range)
    markdown = render_markdown(root, year_range, placeholder_counts, statuses)

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")

    if args.json_output:
        json_output = args.json_output if args.json_output.is_absolute() else root / args.json_output
        json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "year_range": {"start_year": year_range.start_year, "end_year": year_range.end_year},
            "placeholder_counts": placeholder_counts,
            "artifact_status": statuses,
            "markdown_output": str(output),
        }
        json_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
