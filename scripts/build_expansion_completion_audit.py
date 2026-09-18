#!/usr/bin/env python3
"""Build a requirement-by-requirement completion audit for the expansion plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CHECKLIST = Path("artifacts/expansion_plan_checklist_2020_2025.json")
DEFAULT_READINESS = Path("artifacts/expansion_readiness_2020_2025.json")
DEFAULT_STATUS = Path("artifacts/expansion_status_packet_2020_2025.json")
DEFAULT_OUTPUT_JSON = Path("artifacts/expansion_completion_audit_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/expansion_completion_audit_2020_2025.md")

STATUS_CLASS = {
    "done": "proved",
    "ready_after_hydration": "pending_after_hydration",
    "ready_after_full_run": "pending_after_full_run",
    "blocked_by_hydration": "blocked_by_hydration",
    "ready_to_run": "pending_execution",
    "needs_review": "needs_review",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def classify_status(status: str) -> str:
    return STATUS_CLASS.get(status, "unknown")


def build_audit(
    checklist: Mapping[str, Any],
    readiness: Mapping[str, Any],
    status_packet: Mapping[str, Any],
) -> dict[str, Any]:
    requirements = []
    for index, item in enumerate(checklist.get("items") or [], 1):
        if not isinstance(item, Mapping):
            continue
        raw_status = str(item.get("status") or "unknown")
        requirements.append(
            {
                "index": index,
                "requirement": item.get("requirement"),
                "status": raw_status,
                "completion_class": classify_status(raw_status),
                "evidence": item.get("evidence") or [],
            }
        )
    counts = Counter(req["completion_class"] for req in requirements)
    blockers = readiness.get("blockers") or status_packet.get("readiness_blockers") or []
    complete = bool(requirements) and counts.get("proved", 0) == len(requirements)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": complete,
        "ready_for_full_pipeline": bool(readiness.get("ready_for_full_pipeline")),
        "status": status_packet.get("status"),
        "placeholder_count": status_packet.get("placeholder_count", readiness.get("full_placeholder_count")),
        "high_priority_placeholder_count": status_packet.get(
            "high_priority_placeholder_count",
            readiness.get("high_priority_placeholder_count"),
        ),
        "blockers": blockers,
        "requirement_count": len(requirements),
        "completion_counts": dict(sorted(counts.items())),
        "requirements": requirements,
        "interpretation": (
            "Implementation and local evidence are strong enough for a progress update, "
            "but the full objective is not complete until hydrated inputs allow staged and full 2020-2025 execution."
        ),
    }


def render_markdown(audit: Mapping[str, Any]) -> str:
    counts = audit.get("completion_counts") or {}
    lines = [
        "# 2020-2025 Expansion Completion Audit",
        "",
        f"- Generated: {audit.get('generated_at')}",
        f"- Complete: `{audit.get('complete')}`",
        f"- Ready for full pipeline: `{audit.get('ready_for_full_pipeline')}`",
        f"- Status: `{audit.get('status')}`",
        f"- Total placeholders: `{audit.get('placeholder_count')}`",
        f"- High-priority placeholders: `{audit.get('high_priority_placeholder_count')}`",
        "",
        "## Completion Counts",
        "",
        "| Class | Count |",
        "| --- | ---: |",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend([
        "",
        "## Requirement Audit",
        "",
        "| # | Requirement | Status | Class | Evidence |",
        "| ---: | --- | --- | --- | --- |",
    ])
    for item in audit.get("requirements") or []:
        if not isinstance(item, Mapping):
            continue
        evidence = "<br>".join(f"`{entry}`" for entry in item.get("evidence") or [])
        lines.append(
            "| {} | {} | `{}` | `{}` | {} |".format(
                item.get("index"),
                str(item.get("requirement", "")).replace("|", "\\|"),
                item.get("status"),
                item.get("completion_class"),
                evidence,
            )
        )
    blockers = audit.get("blockers") or []
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend([
        "",
        "## Interpretation",
        "",
        str(audit.get("interpretation") or ""),
    ])
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a completion audit for the 2020-2025 expansion.")
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    audit = build_audit(
        checklist=load_json(args.checklist),
        readiness=load_json(args.readiness),
        status_packet=load_json(args.status),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
