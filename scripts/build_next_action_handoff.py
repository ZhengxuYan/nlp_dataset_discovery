#!/usr/bin/env python3
"""Build the immediate next-action handoff for the 2020-2025 expansion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_READINESS = Path("artifacts/expansion_readiness_2020_2025.json")
DEFAULT_STATUS = Path("artifacts/expansion_status_packet_2020_2025.json")
DEFAULT_COMPLETION_AUDIT = Path("artifacts/expansion_completion_audit_2020_2025.json")
DEFAULT_POST_HYDRATION_SEQUENCE = Path("artifacts/post_hydration_expansion_sequence_2020_2025.json")
DEFAULT_OUTPUT_JSON = Path("artifacts/next_action_handoff_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/next_action_handoff_2020_2025.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def gate_from_inputs(readiness: Mapping[str, Any], completion: Mapping[str, Any]) -> str:
    if completion.get("complete"):
        return "complete"
    if readiness.get("ready_for_full_pipeline"):
        return "run_full_pipeline"
    blockers = set(readiness.get("blockers") or completion.get("blockers") or [])
    if "high_priority_placeholders_remaining" in blockers or "full_placeholder_scan_not_clear" in blockers:
        return "hydrate_cloud_placeholders"
    return "resolve_readiness_blockers"


def build_handoff(
    readiness: Mapping[str, Any],
    status_packet: Mapping[str, Any],
    completion: Mapping[str, Any],
    post_hydration_sequence: Mapping[str, Any],
) -> dict[str, Any]:
    current_gate = gate_from_inputs(readiness, completion)
    next_commands = {
        "hydrate_cloud_placeholders": [
            "python scripts/hydrate_cloud_placeholders_macos.py --paths-file artifacts/remaining_high_priority_hydration_files_2020_2025.txt",
            "python scripts/check_cloud_placeholders.py --paths-file artifacts/remaining_high_priority_hydration_files_2020_2025.txt --summary-only",
            "python scripts/refresh_expansion_status_2020_2025.py",
            "python scripts/validate_expansion_readiness.py --allow-blocked",
        ],
        "run_full_pipeline": [
            "python scripts/run_post_hydration_expansion_sequence.py --execute --allow-network-steps",
            "python scripts/refresh_expansion_status_2020_2025.py",
            "python scripts/build_expansion_completion_audit.py",
        ],
        "resolve_readiness_blockers": [
            "python scripts/refresh_expansion_status_2020_2025.py",
            "python scripts/validate_expansion_readiness.py --allow-blocked",
        ],
        "complete": [
            "python scripts/build_expansion_completion_audit.py",
        ],
    }
    handoff = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_gate": current_gate,
        "status": status_packet.get("status", completion.get("status")),
        "ready_for_full_pipeline": bool(readiness.get("ready_for_full_pipeline")),
        "completion_audit_complete": bool(completion.get("complete")),
        "blockers": readiness.get("blockers") or completion.get("blockers") or [],
        "placeholder_count": status_packet.get("placeholder_count", readiness.get("full_placeholder_count")),
        "high_priority_placeholder_count": status_packet.get(
            "high_priority_placeholder_count",
            readiness.get("high_priority_placeholder_count"),
        ),
        "artifact_count": status_packet.get("artifact_count"),
        "missing_artifact_count": status_packet.get("missing_artifact_count"),
        "run_plan_steps": status_packet.get("run_plan_steps"),
        "post_hydration_step_count": len(post_hydration_sequence.get("steps") or []),
        "next_commands": next_commands[current_gate],
        "success_condition": success_condition(current_gate),
        "handoff_artifacts": {
            "hydration_queue": "artifacts/remaining_high_priority_hydration_queue_2020_2025.md",
            "hydration_action_guide": "artifacts/hydration_action_guide_2020_2025.md",
            "hydration_helper_plan": "artifacts/hydration_helper_plan_2020_2025.md",
            "readiness_gate": "artifacts/expansion_readiness_2020_2025.md",
            "completion_audit": "artifacts/expansion_completion_audit_2020_2025.md",
            "professor_meeting_packet": "artifacts/professor_meeting_packet_2020_2025.md",
            "post_hydration_sequence": "artifacts/post_hydration_expansion_sequence_2020_2025.json",
        },
    }
    return handoff


def success_condition(current_gate: str) -> str:
    if current_gate == "hydrate_cloud_placeholders":
        return "Readiness changes to ready_for_full_pipeline with no high-priority placeholder blocker."
    if current_gate == "run_full_pipeline":
        return "Full 2020-2025 artifacts exist and completion audit has no pending full-run requirement."
    if current_gate == "complete":
        return "Completion audit remains complete after refresh."
    return "Readiness report has no unresolved blockers."


def render_markdown(handoff: Mapping[str, Any]) -> str:
    lines = [
        "# 2020-2025 Expansion Next Action Handoff",
        "",
        f"- Generated: {handoff.get('generated_at')}",
        f"- Current gate: `{handoff.get('current_gate')}`",
        f"- Status: `{handoff.get('status')}`",
        f"- Ready for full pipeline: `{handoff.get('ready_for_full_pipeline')}`",
        f"- Completion audit complete: `{handoff.get('completion_audit_complete')}`",
        f"- Total placeholders: `{handoff.get('placeholder_count')}`",
        f"- High-priority placeholders: `{handoff.get('high_priority_placeholder_count')}`",
        f"- Artifact index: `{handoff.get('artifact_count')}` tracked, `{handoff.get('missing_artifact_count')}` missing",
        f"- Run plan steps: `{handoff.get('run_plan_steps')}`",
        "",
        "## Blocking Evidence",
        "",
    ]
    blockers = handoff.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Run These Next",
            "",
            "```bash",
            *handoff.get("next_commands", []),
            "```",
            "",
            "## Success Condition",
            "",
            str(handoff.get("success_condition")),
            "",
            "## Handoff Artifacts",
            "",
        ]
    )
    for label, path in (handoff.get("handoff_artifacts") or {}).items():
        lines.append(f"- `{label}`: `{path}`")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the immediate next-action handoff.")
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--status-packet", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--completion-audit", type=Path, default=DEFAULT_COMPLETION_AUDIT)
    parser.add_argument("--post-hydration-sequence", type=Path, default=DEFAULT_POST_HYDRATION_SEQUENCE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    handoff = build_handoff(
        load_json(args.readiness),
        load_json(args.status_packet),
        load_json(args.completion_audit),
        load_json(args.post_hydration_sequence),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(handoff, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(handoff), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
