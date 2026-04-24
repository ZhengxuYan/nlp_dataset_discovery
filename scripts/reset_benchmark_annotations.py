#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scv.benchmark_models import BenchmarkDraftRecord
from scv.benchmark_store import (
    DEFAULT_BENCHMARK_DRAFTS_PATH,
    load_benchmark_drafts,
    now_iso,
    summarize_query_status,
    write_jsonl,
)


FINAL_ANNOTATION_CLEAR_PATCH = {
    "gold_prior_paper_ids": [],
    "gold_prior_dataset_names": [],
    "hard_negative_ids": [],
    "soft_negative_ids": [],
    "gold_prior_support_acus": [],
    "gold_added_information_label": None,
    "annotation_notes": "",
}


def backup_path(path: Path) -> Path:
    stamp = now_iso().replace(":", "").replace("+", "Z")
    return path.with_suffix(path.suffix + f".bak.{stamp}")


def reset_annotations(path: Path = DEFAULT_BENCHMARK_DRAFTS_PATH) -> tuple[int, Path | None]:
    rows = load_benchmark_drafts(path)
    if not rows:
        return 0, None

    backup = backup_path(path)
    shutil.copy2(path, backup)

    reset_rows = []
    for row in rows:
        data = row.model_dump()
        data.update(FINAL_ANNOTATION_CLEAR_PATCH)
        data["draft_status"] = "needs_annotation"
        reset_row = BenchmarkDraftRecord(**data)
        reset_rows.append(reset_row)

    write_jsonl(path, reset_rows)

    # Derive upstream-aware status after the annotation fields have been cleared.
    derived_rows = []
    for row in load_benchmark_drafts(path):
        data = row.model_dump()
        data["draft_status"] = summarize_query_status(row.query_paper_id)
        derived_rows.append(BenchmarkDraftRecord(**data))
    write_jsonl(path, derived_rows)
    return len(derived_rows), backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear final manual benchmark annotations while keeping candidates, papers, processed ACUs, and suggestions.")
    parser.add_argument("--drafts", type=Path, default=DEFAULT_BENCHMARK_DRAFTS_PATH)
    args = parser.parse_args()

    count, backup = reset_annotations(args.drafts)
    if count == 0:
        print(f"No benchmark drafts found at {args.drafts}. Nothing reset.")
        return
    print(f"Reset final annotations for {count} benchmark drafts.")
    print(f"Backup written to {backup}.")


if __name__ == "__main__":
    main()
