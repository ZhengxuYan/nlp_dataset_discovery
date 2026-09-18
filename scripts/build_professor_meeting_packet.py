#!/usr/bin/env python3
"""Build a meeting-ready professor update packet for the 2020-2025 expansion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_STATUS = Path("artifacts/expansion_status_packet_2020_2025.json")
DEFAULT_COVERAGE = Path("artifacts/local_smoke_2020_2025/metadata_coverage_2020_2025.json")
DEFAULT_SCHEMA = Path("artifacts/local_smoke_2020_2025/metadata_schema_audit_2020_2025.json")
DEFAULT_REVIEW = Path("artifacts/local_smoke_2020_2025/metadata_review_sample_2020_2025.json")
DEFAULT_STABLE_ID = Path("artifacts/local_smoke_2020_2025/enrichment_stable_id_audit_2020_2025.json")
DEFAULT_INTEGRATED = Path("artifacts/local_smoke_2020_2025/integrated_fulltext_banks_2020_2025_summary.json")
DEFAULT_FULL_PLAN = Path("artifacts/corpus_expansion_2020_2025_full_run_plan.json")
DEFAULT_COMPLETION_AUDIT = Path("artifacts/expansion_completion_audit_2020_2025.md")
DEFAULT_OUTPUT_JSON = Path("artifacts/professor_meeting_packet_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/professor_meeting_packet_2020_2025.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def review_buckets(review: Mapping[str, Any]) -> list[str]:
    buckets = {
        item.get("bucket")
        for item in review.get("samples") or []
        if isinstance(item, Mapping) and item.get("bucket")
    }
    return sorted(str(bucket) for bucket in buckets)


def full_plan_health_enabled(full_plan: Mapping[str, Any]) -> bool:
    for step in full_plan.get("steps") or []:
        if not isinstance(step, Mapping) or step.get("name") != "public_metadata_enrichment":
            continue
        return "--check-url-health" in (step.get("command") or [])
    return False


def build_packet(
    *,
    status: Mapping[str, Any],
    coverage: Mapping[str, Any],
    schema: Mapping[str, Any],
    review: Mapping[str, Any],
    stable_id: Mapping[str, Any],
    integrated: Mapping[str, Any],
    full_plan: Mapping[str, Any],
) -> dict[str, Any]:
    rates = coverage.get("coverage_rates") or {}
    schema_present = schema.get("present_values") or {}
    buckets = review_buckets(review)
    done_summary = {
        "run_plan_steps": status.get("run_plan_steps"),
        "artifact_count": status.get("artifact_count"),
        "missing_artifact_count": status.get("missing_artifact_count"),
        "local_smoke_papers": integrated.get("papers"),
        "local_smoke_datasets": integrated.get("datasets"),
        "local_smoke_acus": integrated.get("acus"),
        "local_smoke_rows": coverage.get("rows"),
        "review_sample_count": review.get("sample_count"),
        "stable_id_audit_status": stable_id.get("status"),
        "full_plan_url_health_enabled": full_plan_health_enabled(full_plan),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status.get("status"),
        "readiness_blockers": status.get("readiness_blockers") or [],
        "placeholder_count": status.get("placeholder_count"),
        "high_priority_placeholder_count": status.get("high_priority_placeholder_count"),
        "done_summary": done_summary,
        "coverage_rates": {
            "citation_count_pct": rates.get("citation_count_pct"),
            "openalex_id_pct": rates.get("openalex_id_pct"),
            "semantic_scholar_id_pct": rates.get("semantic_scholar_id_pct"),
            "hf_download_count_per_hf_resource_pct": rates.get("hf_download_count_per_hf_resource_pct", rates.get("hf_download_count_per_hf_link_pct")),
            "github_star_count_per_github_resource_pct": rates.get("github_star_count_per_github_resource_pct", rates.get("github_star_count_per_github_link_pct")),
            "healthy_url_pct": rates.get("healthy_url_pct"),
        },
        "schema_provenance_counts": {
            "paper_source_confidence_scores": schema_present.get("paper_metadata_sources.match_confidence_score"),
            "hf_match_confidence_scores": schema_present.get("hf_metadata.match_confidence_score"),
            "pwc_match_confidence_scores": schema_present.get("pwc_metadata.match_confidence_score"),
            "resource_health_status": schema_present.get("resource_health.status"),
            "resource_health_downloadable": schema_present.get("resource_health.downloadable"),
        },
        "review_buckets": buckets,
        "review_bucket_counts": dict(Counter(
            item.get("bucket")
            for item in review.get("samples") or []
            if isinstance(item, Mapping) and item.get("bucket")
        )),
        "artifacts": {
            "status_packet": str(DEFAULT_STATUS),
            "full_run_plan": str(DEFAULT_FULL_PLAN),
            "coverage": str(DEFAULT_COVERAGE),
            "schema_provenance_audit": str(DEFAULT_SCHEMA),
            "review_sample": str(DEFAULT_REVIEW),
            "stable_id_audit": str(DEFAULT_STABLE_ID),
            "completion_audit": str(DEFAULT_COMPLETION_AUDIT),
            "hydration_queue": "artifacts/remaining_high_priority_hydration_queue_2020_2025.md",
        },
    }


def render_markdown(packet: Mapping[str, Any]) -> str:
    done = packet.get("done_summary") or {}
    rates = packet.get("coverage_rates") or {}
    schema = packet.get("schema_provenance_counts") or {}
    artifacts = packet.get("artifacts") or {}
    buckets = packet.get("review_buckets") or []
    lines = [
        "# 教授 Meeting Packet: 2020-2025 Dataset Discovery Expansion",
        "",
        f"- Generated: {packet.get('generated_at')}",
        f"- Current status: `{packet.get('status')}`",
        f"- Full pipeline blocker: `{packet.get('placeholder_count')}` cloud placeholders, `{packet.get('high_priority_placeholder_count')}` high-priority",
        "",
        "## 30-Second Update",
        "",
        (
            "已经把 2023-2025 workflow 扩展成可复现的 2020-2025 pipeline，并加入 public metadata enrichment。"
            "本地 smoke 现在覆盖 2 篇 paper、3 个 dataset、3 个 ACU，验证 citation counts、OpenAlex/Semantic Scholar IDs、"
            "HF downloads、GitHub stars、Papers with Code links、URL health、stable-ID join audit、schema/provenance audit 和 manual review sample。"
            "剩余 blocker 不是方法设计，而是本地 cloud placeholder hydration。"
        ),
        "",
        "## Evidence Snapshot",
        "",
        "| Item | Evidence |",
        "| --- | ---: |",
        f"| Run plan steps | `{done.get('run_plan_steps')}` |",
        f"| Artifact index | `{done.get('artifact_count')}` tracked, `{done.get('missing_artifact_count')}` missing |",
        f"| Local smoke papers/datasets/ACUs | `{done.get('local_smoke_papers')}` / `{done.get('local_smoke_datasets')}` / `{done.get('local_smoke_acus')}` |",
        f"| Stable-ID audit | `{done.get('stable_id_audit_status')}` |",
        f"| Full run URL-health enabled | `{done.get('full_plan_url_health_enabled')}` |",
        f"| Review sample rows | `{done.get('review_sample_count')}` |",
        "",
        "## Metadata Coverage From Local Smoke",
        "",
        f"- Citation count: `{pct(rates.get('citation_count_pct'))}`",
        f"- OpenAlex ID: `{pct(rates.get('openalex_id_pct'))}`",
        f"- Semantic Scholar ID: `{pct(rates.get('semantic_scholar_id_pct'))}`",
        f"- HF downloads per HF resource: `{pct(rates.get('hf_download_count_per_hf_resource_pct'))}`",
        f"- GitHub stars per GitHub resource: `{pct(rates.get('github_star_count_per_github_resource_pct'))}`",
        f"- Healthy URLs: `{pct(rates.get('healthy_url_pct'))}`",
        "",
        "## Provenance Checks",
        "",
        f"- Paper source confidence scores: `{schema.get('paper_source_confidence_scores')}`",
        f"- HF match confidence scores: `{schema.get('hf_match_confidence_scores')}`",
        f"- PWC match confidence scores: `{schema.get('pwc_match_confidence_scores')}`",
        f"- Resource health status fields: `{schema.get('resource_health_status')}`",
        f"- Resource downloadable fields: `{schema.get('resource_health_downloadable')}`",
        "",
        "## Manual Review Buckets",
        "",
        ", ".join(f"`{bucket}`" for bucket in buckets) if buckets else "None",
        "",
        "## Next Action",
        "",
        "```bash",
        "python scripts/check_cloud_placeholders.py --paths-file artifacts/remaining_high_priority_hydration_files_2020_2025.txt --summary-only",
        "python scripts/refresh_expansion_status_2020_2025.py",
        "python scripts/validate_expansion_readiness.py --allow-blocked",
        "python scripts/run_post_hydration_expansion_sequence.py --execute --allow-network-steps",
        "```",
        "",
        "## Key Artifacts",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a meeting-ready professor update packet.")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--stable-id", type=Path, default=DEFAULT_STABLE_ID)
    parser.add_argument("--integrated", type=Path, default=DEFAULT_INTEGRATED)
    parser.add_argument("--full-plan", type=Path, default=DEFAULT_FULL_PLAN)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    packet = build_packet(
        status=load_json(args.status),
        coverage=load_json(args.coverage),
        schema=load_json(args.schema),
        review=load_json(args.review),
        stable_id=load_json(args.stable_id),
        integrated=load_json(args.integrated),
        full_plan=load_json(args.full_plan),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(packet), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
