#!/usr/bin/env python3
"""Run the post-hydration staged and full 2020-2025 expansion sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DEFAULT_OUTPUT = Path("artifacts/post_hydration_expansion_sequence_2020_2025.json")


@dataclass(frozen=True)
class SequenceStep:
    name: str
    command: list[str]
    requires_execute: bool = True


def build_steps(allow_network_steps: bool = False) -> list[SequenceStep]:
    full_command = ["python", "scripts/run_corpus_expansion_2020_2025.py", "--mode", "full", "--execute"]
    if allow_network_steps:
        full_command.append("--allow-network-steps")
    return [
        SequenceStep("refresh_status", ["python", "scripts/refresh_expansion_status_2020_2025.py"], requires_execute=False),
        SequenceStep("readiness_gate", ["python", "scripts/validate_expansion_readiness.py"], requires_execute=False),
        SequenceStep(
            "staged_2020_2021_smoke",
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
        SequenceStep("full_2020_2025_expansion", full_command),
        SequenceStep("refresh_after_full", ["python", "scripts/refresh_expansion_status_2020_2025.py"]),
        SequenceStep("final_readiness_report", ["python", "scripts/validate_expansion_readiness.py", "--allow-blocked"]),
    ]


def run_sequence(steps: list[SequenceStep], root: Path, execute: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for step in steps:
        payload = {"name": step.name, "command": step.command, "command_string": " ".join(step.command)}
        if step.requires_execute and not execute:
            payload["status"] = "planned"
            results.append(payload)
            continue
        completed = subprocess.run(step.command, cwd=root, check=False)
        payload["status"] = "passed" if completed.returncode == 0 else "failed"
        payload["returncode"] = completed.returncode
        results.append(payload)
        if completed.returncode != 0:
            break
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the post-hydration expansion sequence.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="Actually run staged/full execution steps.")
    parser.add_argument("--allow-network-steps", action="store_true", help="Allow network/LLM steps in the full run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = args.root.resolve()
    steps = build_steps(allow_network_steps=args.allow_network_steps)
    results = run_sequence(steps, root, execute=args.execute)
    failed = [result for result in results if result.get("status") == "failed"]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "execute": args.execute,
        "allow_network_steps": args.allow_network_steps,
        "steps": [asdict(step) for step in steps],
        "results": results,
        "ok": not failed,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
