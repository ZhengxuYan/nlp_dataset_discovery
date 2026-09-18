#!/usr/bin/env python3
"""Refresh local status artifacts for the 2020-2025 expansion work."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DEFAULT_SUMMARY = Path("artifacts/refresh_expansion_status_2020_2025.json")


@dataclass(frozen=True)
class RefreshStep:
    name: str
    command: list[str]


def build_steps(include_local_smoke: bool = True) -> list[RefreshStep]:
    steps = [
        RefreshStep("smoke_run_plan", ["python", "scripts/run_corpus_expansion_2020_2025.py", "--mode", "smoke"]),
        RefreshStep(
            "full_run_plan",
            [
                "python",
                "scripts/run_corpus_expansion_2020_2025.py",
                "--mode",
                "full",
                "--plan-output",
                "artifacts/corpus_expansion_2020_2025_full_run_plan.json",
            ],
        ),
        RefreshStep(
            "staged_2020_2021_smoke_plan",
            [
                "python",
                "scripts/run_corpus_expansion_2020_2025.py",
                "--start-year",
                "2020",
                "--end-year",
                "2021",
                "--mode",
                "smoke",
                "--plan-output",
                "artifacts/corpus_expansion_2020_2021_smoke_run_plan.json",
            ],
        ),
        RefreshStep("hydration_manifest", ["python", "scripts/build_hydration_manifest.py"]),
        RefreshStep("remaining_hydration_queue", ["python", "scripts/build_remaining_hydration_queue.py"]),
        RefreshStep("hydration_helper_plan", ["python", "scripts/hydrate_cloud_placeholders_macos.py"]),
        RefreshStep("hydration_action_guide", ["python", "scripts/build_hydration_action_guide.py"]),
        RefreshStep("hydration_status", ["python", "scripts/build_hydration_status_update.py"]),
        RefreshStep("expansion_regression_audit", ["python", "scripts/audit_expansion_regression.py"]),
        RefreshStep("expansion_checklist", ["python", "scripts/build_expansion_checklist.py"]),
        RefreshStep("professor_update", ["python", "scripts/build_professor_update.py"]),
        RefreshStep("professor_update_brief", ["python", "scripts/build_professor_update_brief.py"]),
        RefreshStep("readiness", ["python", "scripts/validate_expansion_readiness.py", "--allow-blocked"]),
        RefreshStep(
            "progress_update",
            [
                "python",
                "scripts/build_progress_update.py",
                "--json-output",
                "artifacts/progress_update_2020_2025.json",
            ],
        ),
        RefreshStep("professor_meeting_packet_seed", ["python", "scripts/build_professor_meeting_packet.py"]),
        RefreshStep("artifact_index_seed", ["python", "scripts/build_expansion_artifact_index.py"]),
        RefreshStep("status_packet_seed", ["python", "scripts/build_expansion_status_packet.py"]),
        RefreshStep("completion_audit", ["python", "scripts/build_expansion_completion_audit.py"]),
        RefreshStep("next_action_handoff_seed", ["python", "scripts/build_next_action_handoff.py"]),
        RefreshStep("professor_update_draft_seed", ["python", "scripts/build_professor_update_draft.py"]),
        RefreshStep("artifact_index", ["python", "scripts/build_expansion_artifact_index.py"]),
        RefreshStep("status_packet", ["python", "scripts/build_expansion_status_packet.py"]),
        RefreshStep("next_action_handoff", ["python", "scripts/build_next_action_handoff.py"]),
        RefreshStep("professor_meeting_packet", ["python", "scripts/build_professor_meeting_packet.py"]),
        RefreshStep("professor_update_draft", ["python", "scripts/build_professor_update_draft.py"]),
    ]
    if include_local_smoke:
        steps.insert(
            2,
            RefreshStep(
                "local_smoke",
                ["python", "scripts/run_local_smoke_2020_2025.py", "artifacts/local_smoke_2020_2025"],
            ),
        )
    return steps


def run_steps(steps: list[RefreshStep], root: Path, dry_run: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for step in steps:
        payload = {"name": step.name, "command": step.command, "command_string": " ".join(step.command)}
        if dry_run:
            payload["status"] = "planned"
        else:
            completed = subprocess.run(step.command, cwd=root, check=False)
            payload["status"] = "passed" if completed.returncode == 0 else "failed"
            payload["returncode"] = completed.returncode
            if completed.returncode != 0:
                results.append(payload)
                break
        results.append(payload)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh non-network status artifacts for the 2020-2025 expansion.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--skip-local-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = args.root.resolve()
    steps = build_steps(include_local_smoke=not args.skip_local_smoke)
    results = run_steps(steps, root, dry_run=args.dry_run)
    failed = [result for result in results if result.get("status") == "failed"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "dry_run": args.dry_run,
        "steps": [asdict(step) for step in steps],
        "results": results,
        "ok": not failed,
    }
    output = args.summary_output if args.summary_output.is_absolute() else root / args.summary_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
