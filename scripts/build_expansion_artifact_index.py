#!/usr/bin/env python3
"""Build a consolidated index of 2020-2025 expansion deliverables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_OUTPUT_JSON = Path("artifacts/expansion_artifact_index_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/expansion_artifact_index_2020_2025.md")


ARTIFACTS = [
    ("professor_brief_zh", "Chinese short update for professor/meeting notes", Path("artifacts/professor_update_brief_zh_2020_2025.md")),
    ("professor_update_draft_zh", "Send-ready Chinese professor update draft", Path("artifacts/professor_update_draft_zh_2020_2025.md")),
    ("professor_meeting_packet", "Meeting-ready Chinese packet with current evidence and next actions", Path("artifacts/professor_meeting_packet_2020_2025.md")),
    ("professor_brief_en", "English short update for professor/meeting notes", Path("artifacts/professor_update_brief_2020_2025.md")),
    ("professor_update_full", "Longer professor-facing progress update", Path("artifacts/professor_update_2020_2025.md")),
    ("runbook", "Expansion runbook and command handoff", Path("README_EXPANSION_2020_2025.md")),
    ("readiness_gate", "Ready/blocked full-run readiness report", Path("artifacts/expansion_readiness_2020_2025.md")),
    ("smoke_run_plan", "2020-2025 smoke pipeline run plan", Path("artifacts/corpus_expansion_2020_2025_run_plan.json")),
    ("full_run_plan", "2020-2025 full pipeline run plan with network/resource-health steps", Path("artifacts/corpus_expansion_2020_2025_full_run_plan.json")),
    ("staged_smoke_plan", "2020-2021 staged smoke plan before full run", Path("artifacts/corpus_expansion_2020_2021_smoke_run_plan.json")),
    ("local_smoke_manifest", "Local fixture smoke output manifest", Path("artifacts/local_smoke_2020_2025/manifest.json")),
    ("stable_id_audit_smoke", "Local smoke enrichment stable-ID audit", Path("artifacts/local_smoke_2020_2025/enrichment_stable_id_audit_2020_2025.md")),
    ("metadata_coverage_smoke", "Local smoke citation/download/star coverage report", Path("artifacts/local_smoke_2020_2025/metadata_coverage_2020_2025.md")),
    ("metadata_schema_smoke", "Local smoke metadata schema audit", Path("artifacts/local_smoke_2020_2025/metadata_schema_audit_2020_2025.md")),
    ("metadata_review_smoke", "Local smoke metadata manual review sample", Path("artifacts/local_smoke_2020_2025/metadata_review_sample_2020_2025.md")),
    ("hydration_manifest", "Full placeholder manifest by priority/root/type", Path("artifacts/hydration_manifest_2020_2025.md")),
    ("hydration_action_guide", "Manual hydration instructions and verification commands", Path("artifacts/hydration_action_guide_2020_2025.md")),
    ("hydration_helper_plan", "macOS dry-run plan for triggering cloud placeholder downloads", Path("artifacts/hydration_helper_plan_2020_2025.md")),
    ("remaining_hydration_queue", "Grouped remaining high-priority hydration queue", Path("artifacts/remaining_high_priority_hydration_queue_2020_2025.md")),
    ("high_priority_hydration_files", "Flat high-priority hydration file list", Path("artifacts/high_priority_hydration_files_2020_2025.txt")),
    ("next_action_handoff", "Immediate next-action gate, commands, and success condition", Path("artifacts/next_action_handoff_2020_2025.md")),
    ("regression_audit", "Old 2023-2025 vs expanded overlap regression audit", Path("artifacts/expansion_regression_audit_2020_2025.md")),
    ("post_hydration_sequence", "Guarded post-hydration execution summary", Path("artifacts/post_hydration_expansion_sequence_2020_2025.json")),
    ("refresh_summary", "One-command local refresh summary", Path("artifacts/refresh_expansion_status_2020_2025.json")),
    ("requirement_checklist", "Plan requirement checklist with evidence", Path("artifacts/expansion_plan_checklist_2020_2025.md")),
    ("completion_audit", "Requirement-by-requirement completion audit with proven/pending/blocked classes", Path("artifacts/expansion_completion_audit_2020_2025.md")),
    ("technical_progress", "Technical progress report", Path("artifacts/progress_update_2020_2025.md")),
    ("status_packet", "One-page status packet with current evidence, blockers, and next action", Path("artifacts/expansion_status_packet_2020_2025.md")),
]


def artifact_record(root: Path, key: str, description: str, path: Path) -> dict[str, Any]:
    target = path if path.is_absolute() else root / path
    return {
        "key": key,
        "description": description,
        "path": str(path),
        "exists": target.exists(),
        "size_bytes": target.stat().st_size if target.exists() else None,
    }


def build_index(root: Path) -> dict[str, Any]:
    records = [artifact_record(root, key, description, path) for key, description, path in ARTIFACTS]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "artifact_count": len(records),
        "missing_count": sum(1 for record in records if not record["exists"]),
        "artifacts": records,
    }


def render_markdown(index: Mapping[str, Any]) -> str:
    lines = [
        "# 2020-2025 Expansion Artifact Index",
        "",
        f"- Generated: {index.get('generated_at')}",
        f"- Artifacts tracked: `{index.get('artifact_count')}`",
        f"- Missing artifacts: `{index.get('missing_count')}`",
        "",
        "| Key | Purpose | Exists | Path |",
        "| --- | --- | --- | --- |",
    ]
    for record in index.get("artifacts") or []:
        if not isinstance(record, Mapping):
            continue
        lines.append(
            "| `{}` | {} | `{}` | `{}` |".format(
                record.get("key"),
                str(record.get("description", "")).replace("|", "\\|"),
                record.get("exists"),
                record.get("path"),
            )
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an index of expansion artifacts.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    index = build_index(args.root)
    output_json = args.output_json if args.output_json.is_absolute() else args.root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else args.root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(index), encoding="utf-8")
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
