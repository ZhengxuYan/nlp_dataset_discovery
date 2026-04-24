#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scv.benchmark_store import (
    DEFAULT_BENCHMARK_DRAFTS_PATH,
    DEFAULT_PREVIOUS_WORK_PATH,
    load_benchmark_drafts,
    load_previous_work_candidates,
    now_iso,
    summarize_query_status,
    write_jsonl,
)


def backup_path(path: Path) -> Path:
    stamp = now_iso().replace(":", "").replace("+", "Z")
    return path.with_suffix(path.suffix + f".bak.{stamp}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear previous-work candidates for selected query paper IDs.")
    parser.add_argument("--paper-ids", nargs="+", required=True)
    args = parser.parse_args()

    selected = set(args.paper_ids)
    previous_backup = backup_path(DEFAULT_PREVIOUS_WORK_PATH)
    draft_backup = backup_path(DEFAULT_BENCHMARK_DRAFTS_PATH)
    shutil.copy2(DEFAULT_PREVIOUS_WORK_PATH, previous_backup)
    shutil.copy2(DEFAULT_BENCHMARK_DRAFTS_PATH, draft_backup)

    candidates = load_previous_work_candidates()
    removed_ids = {
        candidate.candidate_id
        for candidate in candidates
        if candidate.query_paper_id in selected
    }
    kept_candidates = [
        candidate
        for candidate in candidates
        if candidate.query_paper_id not in selected
    ]
    write_jsonl(DEFAULT_PREVIOUS_WORK_PATH, kept_candidates)

    updated_drafts = []
    for draft in load_benchmark_drafts():
        if draft.query_paper_id not in selected:
            updated_drafts.append(draft)
            continue
        draft.candidate_ids = []
        draft.gold_prior_paper_ids = []
        draft.gold_prior_dataset_names = []
        draft.hard_negative_ids = []
        draft.soft_negative_ids = []
        draft.suggested_gold_prior_paper_ids = []
        draft.suggested_hard_negative_ids = []
        draft.suggested_soft_negative_ids = []
        draft.gold_prior_support_acus = []
        draft.gold_added_information_label = None
        draft.annotation_notes = ""
        draft.linked_prior_papers = []
        draft.draft_status = summarize_query_status(draft.query_paper_id)
        updated_drafts.append(draft)
    write_jsonl(DEFAULT_BENCHMARK_DRAFTS_PATH, updated_drafts)

    print(f"Removed {len(removed_ids)} candidates for {len(selected)} query papers.")
    print(f"Backup written to {previous_backup}.")
    print(f"Backup written to {draft_backup}.")


if __name__ == "__main__":
    main()
