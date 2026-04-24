#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scv.benchmark_models import BenchmarkDraftRecord, PaperBankRecord, PreviousWorkCandidateRecord
from scv.benchmark_store import (
    DEFAULT_BENCHMARK_DRAFTS_PATH,
    DEFAULT_PAPER_BANK_PATH,
    load_benchmark_drafts,
    load_paper_bank,
    load_previous_work_candidates,
    now_iso,
    slugify,
    upsert_paper_bank,
    upsert_previous_work_candidates,
    write_jsonl,
)


DEMO_TAG = "[AUTO_GENERATED_DEMO_PREANNOTATION]"


def make_backup(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup_path = path.with_suffix(path.suffix + f".bak.{now_iso().replace(':', '').replace('+', 'Z')}")
    shutil.copy2(path, backup_path)
    return backup_path


def score_candidate(candidate) -> float:
    relationship_weight = {
        "closest_prior_dataset": 5.0,
        "source_dataset": 4.0,
        "parallel_benchmark": 3.0,
        "evaluation_baseline": 2.0,
        "loosely_related": 1.0,
        "unknown": 0.0,
    }
    confidence_weight = {
        "high": 2.0,
        "medium": 1.0,
        "low": 0.0,
    }
    resolved_bonus = 1.5 if candidate.resolved_paper_id else 0.0
    return (
        relationship_weight.get(candidate.relationship_type, 0.0)
        + confidence_weight.get((candidate.confidence or "").lower(), 0.0)
        + resolved_bonus
    )


def provisional_label_for_draft(draft: BenchmarkDraftRecord, gold_count: int) -> str:
    acu_count = len(draft.query_acus or [])
    if gold_count == 0:
        return "incremental"
    if acu_count >= 5:
        return "substantial"
    if acu_count <= 2:
        return "repackaging"
    return "incremental"


def make_demo_candidate(draft: BenchmarkDraftRecord) -> PreviousWorkCandidateRecord:
    demo_paper_id = f"demo-prior:{slugify(draft.query_paper_id)}"
    evidence = (
        draft.gold_prior_support_acus[0]
        if draft.gold_prior_support_acus
        else f"Provisional previous-work placeholder for {draft.query_dataset_name}."
    )
    return PreviousWorkCandidateRecord(
        query_paper_id=draft.query_paper_id,
        candidate_id=f"{draft.query_paper_id}:demo_prevwork:0",
        dataset_name=f"Provisional prior for {draft.query_dataset_name}",
        paper_title=f"Provisional prior work for {draft.query_dataset_name}",
        relationship_type="closest_prior_dataset",
        description="Auto-generated placeholder candidate for UI/demo population only.",
        confidence="low",
        evidence_text=evidence,
        resolution_status="resolved_in_db",
        resolved_paper_id=demo_paper_id,
        annotation_notes=f"{DEMO_TAG} Placeholder candidate created for UI/demo only; needs human verification.",
    )


def make_demo_paper(candidate: PreviousWorkCandidateRecord) -> PaperBankRecord:
    return PaperBankRecord(
        paper_id=candidate.resolved_paper_id,
        title=candidate.paper_title or candidate.dataset_name or candidate.candidate_id,
        source="auto_generated_demo",
        status="unresolved",
        notes=f"{DEMO_TAG} Placeholder prior-paper record for UI/demo only.",
    )


def autofill(limit: int, path: Path, create_placeholders: bool = True) -> List[BenchmarkDraftRecord]:
    drafts = load_benchmark_drafts(path)
    candidates = load_previous_work_candidates()
    candidates_by_query: Dict[str, List] = {}
    for candidate in candidates:
        candidates_by_query.setdefault(candidate.query_paper_id, []).append(candidate)

    changed = 0
    updated: List[BenchmarkDraftRecord] = []
    new_candidates: List[PreviousWorkCandidateRecord] = []
    new_papers: List[PaperBankRecord] = []

    for draft in drafts:
        if changed >= limit:
            updated.append(draft)
            continue
        if DEMO_TAG in (draft.annotation_notes or ""):
            updated.append(draft)
            continue

        query_candidates = candidates_by_query.get(draft.query_paper_id, [])
        if not query_candidates and create_placeholders:
            demo_candidate = make_demo_candidate(draft)
            query_candidates = [demo_candidate]
            candidates_by_query[draft.query_paper_id] = query_candidates
            new_candidates.append(demo_candidate)
            new_papers.append(make_demo_paper(demo_candidate))

        query_candidates = sorted(
            query_candidates,
            key=score_candidate,
            reverse=True,
        )
        if not query_candidates:
            updated.append(draft)
            continue

        gold_candidates = [
            candidate for candidate in query_candidates
            if candidate.relationship_type in {"closest_prior_dataset", "source_dataset"}
        ][:2]
        if not gold_candidates:
            gold_candidates = query_candidates[:1]

        hard_candidates = [
            candidate for candidate in query_candidates
            if candidate.candidate_id not in {gold.candidate_id for gold in gold_candidates}
            and candidate.relationship_type in {"parallel_benchmark", "evaluation_baseline", "loosely_related"}
        ][:2]
        soft_candidates = [
            candidate for candidate in query_candidates
            if candidate.candidate_id not in {gold.candidate_id for gold in gold_candidates}
            and candidate.candidate_id not in {hard.candidate_id for hard in hard_candidates}
        ][:2]

        gold_prior_paper_ids = [
            candidate.resolved_paper_id
            for candidate in gold_candidates
            if candidate.resolved_paper_id
        ]
        gold_prior_dataset_names = [
            candidate.dataset_name or candidate.paper_title or candidate.candidate_id
            for candidate in gold_candidates
        ]
        support_acus = list(draft.gold_prior_support_acus or [])
        for candidate in gold_candidates:
            if candidate.evidence_text:
                support_acus.append(candidate.evidence_text)

        notes = "\n".join([
            (draft.annotation_notes or "").strip(),
            f"{DEMO_TAG} Machine-preannotated for UI/demo only at {now_iso()}. Needs human verification before use as gold benchmark data.",
            "Auto-fill heuristic: selected closest/source candidates as provisional gold priors; selected related non-gold candidates as provisional negatives.",
        ]).strip()

        data = draft.model_dump()
        data.update({
            "gold_prior_paper_ids": gold_prior_paper_ids,
            "gold_prior_dataset_names": list(dict.fromkeys(gold_prior_dataset_names)),
            "hard_negative_ids": [candidate.candidate_id for candidate in hard_candidates],
            "soft_negative_ids": [candidate.candidate_id for candidate in soft_candidates],
            "gold_prior_support_acus": list(dict.fromkeys([acu for acu in support_acus if acu])),
            "gold_added_information_label": provisional_label_for_draft(draft, len(gold_candidates)),
            "annotation_notes": notes,
            "draft_status": "needs_annotation",
        })
        updated.append(BenchmarkDraftRecord(**data))
        changed += 1

    if new_candidates:
        upsert_previous_work_candidates(new_candidates)
    if new_papers:
        upsert_paper_bank(new_papers)
    write_jsonl(path, updated)
    return updated


def remove_autofill(path: Path) -> List[BenchmarkDraftRecord]:
    drafts = load_benchmark_drafts(path)
    updated: List[BenchmarkDraftRecord] = []
    for draft in drafts:
        if DEMO_TAG not in (draft.annotation_notes or ""):
            updated.append(draft)
            continue
        remaining_notes = "\n".join([
            line for line in (draft.annotation_notes or "").splitlines()
            if DEMO_TAG not in line and "Auto-fill heuristic:" not in line
        ]).strip()
        data = draft.model_dump()
        data.update({
            "gold_prior_paper_ids": [],
            "gold_prior_dataset_names": [],
            "hard_negative_ids": [],
            "soft_negative_ids": [],
            "gold_prior_support_acus": [],
            "gold_added_information_label": None,
            "annotation_notes": remaining_notes,
            "draft_status": "needs_annotation",
        })
        updated.append(BenchmarkDraftRecord(**data))
    write_jsonl(path, updated)

    candidates = load_previous_work_candidates()
    kept_candidates = [
        candidate for candidate in candidates
        if DEMO_TAG not in (candidate.annotation_notes or "")
    ]
    write_jsonl(Path("data/benchmark/previous_work_candidates.jsonl"), kept_candidates)

    papers = load_paper_bank()
    kept_papers = [
        paper for paper in papers
        if DEMO_TAG not in (paper.notes or "")
    ]
    write_jsonl(DEFAULT_PAPER_BANK_PATH, kept_papers)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-fill provisional demo benchmark rows and optionally remove them later.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--path", type=Path, default=DEFAULT_BENCHMARK_DRAFTS_PATH)
    parser.add_argument("--remove", action="store_true", help="Remove fields added by this demo auto-fill script.")
    parser.add_argument("--no-placeholders", action="store_true", help="Only auto-fill rows that already have candidates.")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if not args.no_backup:
        backup_path = make_backup(args.path)
        if backup_path:
            print(f"Backup written to {backup_path}")

    if args.remove:
        rows = remove_autofill(args.path)
        removed = sum(1 for row in rows if row.draft_status == "needs_annotation")
        print(f"Removed demo auto-fill fields where tagged. Draft rows now stored: {len(rows)}")
    else:
        rows = autofill(args.limit, args.path, create_placeholders=not args.no_placeholders)
        filled = sum(1 for row in rows if DEMO_TAG in (row.annotation_notes or ""))
        print(f"Machine-preannotated demo rows available: {filled}")
        print("Rows are tagged as AUTO_GENERATED_DEMO_PREANNOTATION and require human verification.")


if __name__ == "__main__":
    main()
