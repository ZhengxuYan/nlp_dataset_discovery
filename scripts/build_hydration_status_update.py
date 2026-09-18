#!/usr/bin/env python3
"""Build a concise hydration status update for the 2020-2025 expansion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MANIFEST = Path("artifacts/hydration_manifest_2020_2025.json")
DEFAULT_RUN_PLAN = Path("artifacts/corpus_expansion_2020_2025_run_plan.json")
DEFAULT_LOCAL_SMOKE_COVERAGE = Path("artifacts/local_smoke_2020_2025/metadata_coverage_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/hydration_status_2020_2025.md")
DEFAULT_OUTPUT_JSON = Path("artifacts/hydration_status_2020_2025.json")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def rate(rates: Mapping[str, Any], preferred: str, fallback: str) -> Any:
    return rates.get(preferred, rates.get(fallback))


def build_status(
    manifest: Mapping[str, Any],
    run_plan: Mapping[str, Any],
    local_smoke_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    by_priority = dict(manifest.get("by_priority") or {})
    placeholder_count = int(manifest.get("placeholder_count") or run_plan.get("placeholder_count") or 0)
    high_priority_count = int(by_priority.get("high") or 0)
    medium_priority_count = int(by_priority.get("medium") or 0)
    low_priority_count = int(by_priority.get("low") or 0)
    safe_to_execute = bool(run_plan.get("safe_to_execute_full_pipeline")) and placeholder_count == 0
    coverage_rates = dict(local_smoke_coverage.get("coverage_rates") or {})

    if placeholder_count:
        next_gate = "hydrate_high_priority_files"
    elif not safe_to_execute:
        next_gate = "refresh_run_plan"
    else:
        next_gate = "run_full_pipeline"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "placeholder_count": placeholder_count,
        "by_priority": {
            "high": high_priority_count,
            "medium": medium_priority_count,
            "low": low_priority_count,
        },
        "safe_to_execute_full_pipeline": safe_to_execute,
        "next_gate": next_gate,
        "local_smoke_rows": int(local_smoke_coverage.get("rows") or 0),
        "local_smoke_coverage_rates": coverage_rates,
        "professor_update_status": (
            "Implementation and offline smoke are ready; full run is gated on local file hydration."
            if placeholder_count
            else "Hydration gate is clear; run the full 2020-2025 pipeline."
        ),
    }


def render_markdown(status: Mapping[str, Any]) -> str:
    by_priority = status.get("by_priority") or {}
    rates = status.get("local_smoke_coverage_rates") or {}
    safe = status.get("safe_to_execute_full_pipeline")
    lines = [
        "# Hydration Status: 2020-2025 Expansion",
        "",
        f"- Generated: {status.get('generated_at')}",
        f"- Placeholder files remaining: `{status.get('placeholder_count')}`",
        f"- High-priority placeholders: `{by_priority.get('high', 0)}`",
        f"- Medium-priority placeholders: `{by_priority.get('medium', 0)}`",
        f"- Low-priority placeholders: `{by_priority.get('low', 0)}`",
        f"- Safe to run full pipeline now: `{safe}`",
        f"- Next gate: `{status.get('next_gate')}`",
        "",
        "## Progress Update",
        "",
        status.get("professor_update_status", ""),
        "",
        "## Evidence Already Available",
        "",
        "- Year-range parameterization and 2020_2025 output naming are implemented.",
        "- Public metadata enrichment is implemented for paper citations/identifiers and dataset-resource signals.",
        "- Cached/offline local smoke validated the integrated bank -> enrichment -> coverage-report path.",
        f"- Local smoke rows: `{status.get('local_smoke_rows')}`",
        f"- Local smoke citation coverage: `{pct(rates.get('citation_count_pct'))}`",
        f"- Local smoke OpenAlex ID coverage: `{pct(rates.get('openalex_id_pct'))}`",
        f"- Local smoke Semantic Scholar ID coverage: `{pct(rates.get('semantic_scholar_id_pct'))}`",
        f"- Local smoke HF download coverage per HF resource: `{pct(rate(rates, 'hf_download_count_per_hf_resource_pct', 'hf_download_count_per_hf_link_pct'))}`",
        f"- Local smoke GitHub star coverage per GitHub resource: `{pct(rate(rates, 'github_star_count_per_github_resource_pct', 'github_star_count_per_github_link_pct'))}`",
        f"- Local smoke healthy URL coverage: `{pct(rates.get('healthy_url_pct'))}`",
        "",
        "## Immediate Commands",
        "",
        "```bash",
        "python scripts/check_cloud_placeholders.py scripts scv scrapers data --summary-only",
        "python scripts/build_hydration_manifest.py",
        "python scripts/build_hydration_status_update.py",
        "python scripts/check_cloud_placeholders.py --paths-file artifacts/high_priority_hydration_files_2020_2025.txt --summary-only",
        "```",
        "",
        "After placeholder count reaches zero:",
        "",
        "```bash",
        "python scripts/run_corpus_expansion_2020_2025.py --mode smoke",
        "python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps",
        "python scripts/audit_expansion_regression.py",
        "```",
        "",
        "Hydrate this file list first: `artifacts/high_priority_hydration_files_2020_2025.txt`",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a short hydration status update.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-plan", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--local-smoke-coverage", type=Path, default=DEFAULT_LOCAL_SMOKE_COVERAGE)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    status = build_status(
        load_json(args.manifest),
        load_json(args.run_plan),
        load_json(args.local_smoke_coverage),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(status), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
