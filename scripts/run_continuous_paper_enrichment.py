#!/usr/bin/env python3
"""Continuously run resumable Semantic Scholar paper-enrichment shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def count_jsonl_rows(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def load_state(path: Path, *, input_path: Path, start_offset: int, total_rows: int) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        recorded_input = Path(str(state.get("input", ""))).resolve()
        if recorded_input != input_path.resolve():
            raise ValueError(f"State input mismatch: {recorded_input} != {input_path.resolve()}")
        return state
    return {
        "input": str(input_path.resolve()),
        "start_offset": start_offset,
        "next_offset": start_offset,
        "total_rows": total_rows,
        "completed_shards": 0,
        "consecutive_failures": 0,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "status": "ready",
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    total = int(state["total_rows"])
    next_offset = min(int(state["next_offset"]), total)
    return {
        "status": state.get("status"),
        "processed_rows": next_offset,
        "remaining_rows": max(total - next_offset, 0),
        "total_rows": total,
        "percent_complete": round(100 * next_offset / total, 2) if total else 100.0,
        "completed_shards": state.get("completed_shards", 0),
        "consecutive_failures": state.get("consecutive_failures", 0),
        "last_completed_shard": state.get("last_completed_shard"),
        "last_error": state.get("last_error"),
        "updated_at": state.get("updated_at"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--end-offset", type=int)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--semantic-batch-size", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=15.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-wait-seconds", type=float, default=900.0)
    parser.add_argument("--heartbeat-hours", type=float, default=24.0)
    parser.add_argument("--status-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_offset < 0 or args.shard_size <= 0 or args.heartbeat_hours <= 0:
        raise ValueError("Offsets must be non-negative and sizes/heartbeat must be positive")
    total_rows = count_jsonl_rows(args.input)
    end_offset = min(args.end_offset if args.end_offset is not None else total_rows, total_rows)
    state = load_state(args.state, input_path=args.input, start_offset=args.start_offset, total_rows=end_offset)
    if args.status_only:
        print(json.dumps(progress_snapshot(state), indent=2, sort_keys=True))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    next_heartbeat = time.monotonic() + args.heartbeat_hours * 3600

    while int(state["next_offset"]) < end_offset:
        start = int(state["next_offset"])
        limit = min(args.shard_size, end_offset - start)
        finish = start + limit
        stem = f"arxiv_2020_2025_paper_metadata_s2_batch_enriched_{start:06d}_{finish:06d}"
        output_path = args.output_dir / f"{stem}.jsonl"
        summary_path = args.output_dir / f"{stem}_summary.json"
        log_path = args.log_dir / f"arxiv_paper_metadata_s2_batch_{start:06d}_{finish:06d}.log"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/enrich_dataset_metadata.py"),
            "--input", str(args.input),
            "--output", str(output_path),
            "--summary-output", str(summary_path),
            "--cache", str(args.cache),
            "--offset", str(start),
            "--limit", str(limit),
            "--paper-only",
            "--openalex-mode", "off",
            "--semantic-batch-only",
            "--semantic-batch-size", str(args.semantic_batch_size),
            "--processing-chunk-size", str(args.semantic_batch_size),
            "--sleep-seconds", str(args.sleep_seconds),
            "--max-retries", str(args.max_retries),
            "--save-every", str(args.semantic_batch_size),
            "--progress-every", str(args.semantic_batch_size),
        ]
        state.update({"status": "running", "current_shard": [start, finish], "updated_at": utc_now()})
        save_state(args.state, state)
        print(json.dumps({"event": "shard_started", "offset": start, "finish": finish, "timestamp": utc_now()}), flush=True)
        with log_path.open("a", encoding="utf-8") as log_fh:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                check=False,
            )

        rows_written = count_jsonl_rows(output_path) if output_path.exists() else 0
        if completed.returncode == 0 and rows_written == limit:
            state.update(
                {
                    "status": "running",
                    "next_offset": finish,
                    "completed_shards": int(state.get("completed_shards", 0)) + 1,
                    "consecutive_failures": 0,
                    "last_completed_shard": [start, finish],
                    "last_completed_at": utc_now(),
                    "last_error": None,
                    "updated_at": utc_now(),
                }
            )
            save_state(args.state, state)
            print(json.dumps({"event": "shard_completed", **progress_snapshot(state)}), flush=True)
        else:
            state.update(
                {
                    "status": "retry_wait",
                    "consecutive_failures": int(state.get("consecutive_failures", 0)) + 1,
                    "last_error": f"exit={completed.returncode}, rows={rows_written}, expected={limit}",
                    "updated_at": utc_now(),
                }
            )
            save_state(args.state, state)
            print(json.dumps({"event": "shard_failed", **progress_snapshot(state)}), flush=True)
            time.sleep(args.retry_wait_seconds)

        if time.monotonic() >= next_heartbeat:
            print(json.dumps({"event": "24h_status", **progress_snapshot(state)}), flush=True)
            next_heartbeat = time.monotonic() + args.heartbeat_hours * 3600

    state.update({"status": "complete", "next_offset": end_offset, "updated_at": utc_now(), "completed_at": utc_now()})
    save_state(args.state, state)
    print(json.dumps({"event": "complete", **progress_snapshot(state)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
