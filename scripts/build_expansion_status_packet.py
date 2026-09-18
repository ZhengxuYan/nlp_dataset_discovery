#!/usr/bin/env python3
"""Build a one-page status packet for the 2020-2025 expansion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RUN_PLAN = Path("artifacts/corpus_expansion_2020_2025_run_plan.json")
DEFAULT_READINESS = Path("artifacts/expansion_readiness_2020_2025.json")
DEFAULT_HYDRATION = Path("artifacts/hydration_status_2020_2025.json")
DEFAULT_COVERAGE = Path("artifacts/local_smoke_2020_2025/metadata_coverage_2020_2025.json")
DEFAULT_ARTIFACT_INDEX = Path("artifacts/expansion_artifact_index_2020_2025.json")
DEFAULT_OUTPUT_JSON = Path("artifacts/expansion_status_packet_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/expansion_status_packet_2020_2025.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def rate(rates: Mapping[str, Any], preferred: str, fallback: str) -> Any:
    return rates.get(preferred, rates.get(fallback))


def build_packet(
    run_plan: Mapping[str, Any],
    readiness: Mapping[str, Any],
    hydration: Mapping[str, Any],
    coverage: Mapping[str, Any],
    artifact_index: Mapping[str, Any],
) -> dict[str, Any]:
    rates = coverage.get("coverage_rates") or hydration.get("local_smoke_coverage_rates") or {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "blocked_by_hydration" if not readiness.get("ready_for_full_pipeline") else "ready_for_full_pipeline",
        "run_plan_steps": len(run_plan.get("steps") or []),
        "readiness_blockers": readiness.get("blockers") or [],
        "placeholder_count": hydration.get("placeholder_count", readiness.get("full_placeholder_count")),
        "high_priority_placeholder_count": (hydration.get("by_priority") or {}).get("high", readiness.get("high_priority_placeholder_count")),
        "local_smoke_rows": hydration.get("local_smoke_rows", coverage.get("rows")),
        "coverage_rates": rates,
        "artifact_count": artifact_index.get("artifact_count"),
        "missing_artifact_count": artifact_index.get("missing_count"),
        "primary_artifacts": {
            "chinese_brief": "artifacts/professor_update_brief_zh_2020_2025.md",
            "chinese_update_draft": "artifacts/professor_update_draft_zh_2020_2025.md",
            "meeting_packet": "artifacts/professor_meeting_packet_2020_2025.md",
            "artifact_index": "artifacts/expansion_artifact_index_2020_2025.md",
            "completion_audit": "artifacts/expansion_completion_audit_2020_2025.md",
            "full_run_plan": "artifacts/corpus_expansion_2020_2025_full_run_plan.json",
            "readiness": "artifacts/expansion_readiness_2020_2025.md",
            "hydration_queue": "artifacts/remaining_high_priority_hydration_queue_2020_2025.md",
            "hydration_helper_plan": "artifacts/hydration_helper_plan_2020_2025.md",
            "next_action_handoff": "artifacts/next_action_handoff_2020_2025.md",
            "runbook": "README_EXPANSION_2020_2025.md",
        },
    }


def render_markdown(packet: Mapping[str, Any]) -> str:
    rates = packet.get("coverage_rates") or {}
    blockers = packet.get("readiness_blockers") or []
    lines = [
        "# 2020-2025 Expansion Status Packet",
        "",
        f"- Generated: {packet.get('generated_at')}",
        f"- Status: `{packet.get('status')}`",
        f"- Run plan steps: `{packet.get('run_plan_steps')}`",
        f"- Total placeholders: `{packet.get('placeholder_count')}`",
        f"- High-priority placeholders: `{packet.get('high_priority_placeholder_count')}`",
        f"- Artifact index: `{packet.get('artifact_count')}` tracked, `{packet.get('missing_artifact_count')}` missing",
        "",
        "## Completed Evidence",
        "",
        "- 2020-2025 run plan is generated and preserves separate `2020_2025` output naming.",
        "- Public paper metadata enrichment uses exact DOI/arXiv matches first and scored title fuzzy fallback with low-similarity rejection.",
        "- Dataset-resource enrichment uses direct URLs first and can fall back to scored dataset-name lookup for Hugging Face and Papers with Code.",
        "- Full metadata enrichment is configured to check URL health, and local fixture smoke validates cached status/resolved/downloadable URL-health fields.",
        "- Local fixture smoke covers integrated dataset/ACU bank creation, public metadata enrichment, stable-ID preservation, coverage reporting, schema/provenance audit, and review sample.",
        "- Post-run gates are defined for metadata coverage, schema completeness, manual review sample, and 2023-2025 overlap regression.",
        "",
        "## Local Smoke Metadata Coverage",
        "",
        f"- Citation count: `{pct(rates.get('citation_count_pct'))}`",
        f"- OpenAlex ID: `{pct(rates.get('openalex_id_pct'))}`",
        f"- Semantic Scholar ID: `{pct(rates.get('semantic_scholar_id_pct'))}`",
        f"- HF downloads per HF resource: `{pct(rate(rates, 'hf_download_count_per_hf_resource_pct', 'hf_download_count_per_hf_link_pct'))}`",
        f"- GitHub stars per GitHub resource: `{pct(rate(rates, 'github_star_count_per_github_resource_pct', 'github_star_count_per_github_link_pct'))}`",
        f"- Healthy URLs: `{pct(rates.get('healthy_url_pct'))}`",
        "",
        "## Current Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend([
        "",
        "## Next Action",
        "",
        "Hydrate the remaining high-priority files, then refresh and rerun readiness:",
        "",
        "```bash",
        "python scripts/check_cloud_placeholders.py --paths-file artifacts/remaining_high_priority_hydration_files_2020_2025.txt --summary-only",
        "python scripts/refresh_expansion_status_2020_2025.py",
        "python scripts/validate_expansion_readiness.py --allow-blocked",
        "```",
        "",
        "## Primary Artifacts",
        "",
    ])
    for label, path in (packet.get("primary_artifacts") or {}).items():
        lines.append(f"- `{label}`: `{path}`")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a one-page expansion status packet.")
    parser.add_argument("--run-plan", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--hydration", type=Path, default=DEFAULT_HYDRATION)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--artifact-index", type=Path, default=DEFAULT_ARTIFACT_INDEX)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    packet = build_packet(
        load_json(args.run_plan),
        load_json(args.readiness),
        load_json(args.hydration),
        load_json(args.coverage),
        load_json(args.artifact_index),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(packet), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
