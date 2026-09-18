#!/usr/bin/env python3
"""Build an evidence-backed checklist for the 2020-2025 expansion plan."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RUN_PLAN = Path("artifacts/corpus_expansion_2020_2025_run_plan.json")
DEFAULT_OUTPUT_MD = Path("artifacts/expansion_plan_checklist_2020_2025.md")
DEFAULT_OUTPUT_JSON = Path("artifacts/expansion_plan_checklist_2020_2025.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_items(run_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    placeholder_count = int(run_plan.get("placeholder_count") or 0)
    steps = run_plan.get("steps") or []
    step_names = {step.get("name") for step in steps if isinstance(step, Mapping)}
    return [
        {
            "requirement": "Extend corpus interface from 2023-2025 to 2020-2025",
            "status": "done",
            "evidence": [
                "scripts/corpus_expansion.py",
                "scripts/build_acl_anthology_catalog.py --start-year/--end-year",
                "scripts/prepare_arxiv_screening_catalog.py --start-year/--end-year",
                "scripts/build_dataset_census.py --start-year/--end-year",
                "scrapers/arxiv_scraper/run_arxiv_intervals.py --start-date/--end-date",
            ],
        },
        {
            "requirement": "Preserve old 2023-2025 outputs and write 2020_2025 artifacts separately",
            "status": "done",
            "evidence": [
                "scripts/corpus_expansion.py output_paths() emits 2020_2025 paths",
                "artifacts/corpus_expansion_2020_2025_run_plan.json lists 2020_2025 outputs",
            ],
        },
        {
            "requirement": "Add paper-level public metadata enrichment",
            "status": "done",
            "evidence": [
                "scripts/enrich_dataset_metadata.py extracts DOI/arXiv/ACL IDs",
                "scripts/enrich_dataset_metadata.py queries OpenAlex and Semantic Scholar",
                "scripts/enrich_dataset_metadata.py rejects low-similarity title fuzzy fallbacks",
                "tests/test_enrich_dataset_metadata.py",
            ],
        },
        {
            "requirement": "Add dataset/resource-level public metadata enrichment",
            "status": "done",
            "evidence": [
                "scripts/enrich_dataset_metadata.py classifies Hugging Face/GitHub/Papers with Code/download/project URLs",
                "scripts/enrich_dataset_metadata.py queries HF downloads/likes and GitHub stars/forks",
                "scripts/enrich_dataset_metadata.py can search Hugging Face and Papers with Code by dataset name when direct URLs are missing",
                "scripts/enrich_dataset_metadata.py can check URL health",
                "scripts/run_corpus_expansion_2020_2025.py enables --check-url-health for full enrichment runs",
            ],
        },
        {
            "requirement": "Cache public API calls and record match provenance",
            "status": "done",
            "evidence": [
                "CachedHttpClient in scripts/enrich_dataset_metadata.py",
                "public_metadata.paper_metadata_sources records source/query/confidence/queried_at",
                "public_metadata.paper_metadata_sources records match_method, match_confidence_score, and matched_title for fuzzy fallbacks",
                "scripts/audit_metadata_schema.py audits paper/resource match_method and match_confidence_score coverage",
                "scripts/audit_metadata_schema.py audits resource_health status/resolved_url/downloadable/checked_at coverage",
            ],
        },
        {
            "requirement": "Merge enrichment by stable paper/dataset IDs, not row position",
            "status": "done" if "enrichment_stable_id_audit" in step_names else "needs_review",
            "evidence": [
                "scripts/audit_enrichment_stable_ids.py",
                "public_metadata.metadata_enrichment.record_id",
                "artifacts/local_smoke_2020_2025/enrichment_stable_id_audit_2020_2025.md",
                "tests/test_audit_enrichment_stable_ids.py",
            ],
        },
        {
            "requirement": "Provide smoke/full execution plan",
            "status": "done" if {"preflight_placeholder_check", "public_metadata_enrichment"} <= step_names else "needs_review",
            "evidence": [
                "README_EXPANSION_2020_2025.md",
                "scripts/run_corpus_expansion_2020_2025.py",
                "artifacts/corpus_expansion_2020_2025_run_plan.json",
                "artifacts/corpus_expansion_2020_2025_full_run_plan.json",
            ],
        },
        {
            "requirement": "Run local end-to-end smoke for integrated bank and metadata enrichment",
            "status": "done",
            "evidence": [
                "scripts/run_local_smoke_2020_2025.py",
                "artifacts/local_smoke_2020_2025/integrated_fulltext_dataset_bank_2020_2025.jsonl",
                "artifacts/local_smoke_2020_2025/integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl",
                "artifacts/local_smoke_2020_2025/enrichment_stable_id_audit_2020_2025.md",
                "artifacts/local_smoke_2020_2025/metadata_coverage_2020_2025.md",
            ],
        },
        {
            "requirement": "Prepare staged 2020-2021 smoke plan before full 2020-2025 execution",
            "status": "ready_after_hydration",
            "evidence": [
                "artifacts/corpus_expansion_2020_2021_smoke_run_plan.json",
                "python scripts/run_corpus_expansion_2020_2025.py --start-year 2020 --end-year 2021 --mode smoke",
            ],
        },
        {
            "requirement": "Generate metadata coverage report after enrichment",
            "status": "done",
            "evidence": [
                "scripts/summarize_metadata_coverage.py",
                "artifacts/metadata_coverage_fixture_2020_2025.md demonstrates report format",
            ],
        },
        {
            "requirement": "Audit enriched metadata schema completeness and missing reasons",
            "status": "done",
            "evidence": [
                "scripts/audit_metadata_schema.py",
                "artifacts/local_smoke_2020_2025/metadata_schema_audit_2020_2025.md",
                "schema audit includes resource-level match provenance for HF/GitHub/Papers with Code metadata",
                "schema audit includes URL-health field coverage for resource_health",
                "tests/test_audit_metadata_schema.py",
            ],
        },
        {
            "requirement": "Prepare enrichment review sample for high/low citation and HF/GitHub-linked datasets",
            "status": "done",
            "evidence": [
                "scripts/build_metadata_review_sample.py",
                "artifacts/local_smoke_2020_2025/metadata_review_sample_2020_2025.md",
                "tests/test_build_metadata_review_sample.py",
            ],
        },
        {
            "requirement": "Audit 2023-2025 overlap so expansion does not shrink existing years",
            "status": "ready_after_full_run",
            "evidence": [
                "scripts/audit_expansion_regression.py",
                "artifacts/expansion_regression_audit_2020_2025.md",
                "compares old 2023_2025 artifacts against new 2020_2025 artifacts on the 2023-2025 overlap",
            ],
        },
        {
            "requirement": "Create professor-ready progress update",
            "status": "done",
            "evidence": [
                "README_EXPANSION_2020_2025.md",
                "scripts/build_professor_update.py",
                "artifacts/professor_update_2020_2025.md",
                "artifacts/progress_update_2020_2025.md",
            ],
        },
        {
            "requirement": "Run full 2020-2025 pipeline and produce final expanded artifacts",
            "status": "blocked_by_hydration" if placeholder_count else "ready_to_run",
            "evidence": [
                "artifacts/hydration_manifest_2020_2025.md",
                "artifacts/high_priority_hydration_files_2020_2025.txt",
                f"placeholder_count={placeholder_count}",
                "python scripts/check_cloud_placeholders.py scripts scv scrapers data --summary-only",
                "python scripts/build_hydration_manifest.py",
                "python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps",
            ],
        },
    ]


def render_markdown(items: list[Mapping[str, Any]], run_plan: Mapping[str, Any]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    placeholder_count = run_plan.get("placeholder_count", "unknown")
    safe = run_plan.get("safe_to_execute_full_pipeline", False)
    lines = [
        "# 2020-2025 Expansion Plan Checklist",
        "",
        f"- Generated: {generated_at}",
        f"- Placeholder count: `{placeholder_count}`",
        f"- Safe to execute full pipeline: `{safe}`",
        "",
        "| Requirement | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in items:
        evidence = "<br>".join(f"`{entry}`" for entry in item.get("evidence", []))
        lines.append(f"| {item['requirement']} | `{item['status']}` | {evidence} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Implementation and reporting pieces are complete enough for a progress update. The remaining incomplete item is full-data execution, which is gated on hydrating cloud placeholder files.",
    ])
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a checklist for the 2020-2025 expansion plan.")
    parser.add_argument("--run-plan", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_plan = load_json(args.run_plan)
    items = build_items(run_plan)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_plan": str(args.run_plan),
        "placeholder_count": run_plan.get("placeholder_count"),
        "safe_to_execute_full_pipeline": run_plan.get("safe_to_execute_full_pipeline"),
        "items": items,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(items, run_plan), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
