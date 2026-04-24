from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .benchmark_models import (
    BenchmarkDraftRecord,
    JobRecord,
    PaperBankRecord,
    PreviousWorkCandidateRecord,
    ProcessedBankRecord,
)
from .utils import ensure_dir


T = TypeVar("T", bound=BaseModel)


DEFAULT_BENCHMARK_DIR = Path("data/benchmark")
DEFAULT_PDF_DIR = DEFAULT_BENCHMARK_DIR / "pdfs"
DEFAULT_PROCESSED_DIR = DEFAULT_BENCHMARK_DIR / "processed"
DEFAULT_PAPER_BANK_PATH = DEFAULT_BENCHMARK_DIR / "paper_bank.jsonl"
DEFAULT_PROCESSED_BANK_PATH = DEFAULT_BENCHMARK_DIR / "processed_bank.jsonl"
DEFAULT_PREVIOUS_WORK_PATH = DEFAULT_BENCHMARK_DIR / "previous_work_candidates.jsonl"
DEFAULT_BENCHMARK_DRAFTS_PATH = DEFAULT_BENCHMARK_DIR / "benchmark_drafts.jsonl"
DEFAULT_JOB_QUEUE_PATH = DEFAULT_BENCHMARK_DIR / "job_queue.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return normalized or "paper"


def canonical_paper_id(arxiv_id: Optional[str], title: str) -> str:
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    return f"title:{slugify(title)}"


def candidate_id(query_paper_id: str, index: int) -> str:
    return f"{query_paper_id}:prevwork:{index}"


def ensure_benchmark_dirs(base_dir: Path = DEFAULT_BENCHMARK_DIR) -> None:
    ensure_dir(str(base_dir))
    ensure_dir(str(base_dir / "pdfs"))
    ensure_dir(str(base_dir / "processed"))


def read_jsonl(path: Path, model_cls: Type[T]) -> List[T]:
    if not path.exists():
        return []
    rows: List[T] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(model_cls(**json.loads(line)))
    return rows


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    ensure_benchmark_dirs(path.parent if path.parent != Path("") else DEFAULT_BENCHMARK_DIR)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(), ensure_ascii=True) + "\n")


def upsert_records(path: Path, rows: Iterable[T], model_cls: Type[T], key: str) -> List[T]:
    existing = read_jsonl(path, model_cls)
    index: Dict[str, T] = {getattr(item, key): item for item in existing}
    for row in rows:
        index[getattr(row, key)] = row
    ordered = list(index.values())
    write_jsonl(path, ordered)
    return ordered


def replace_records(
    path: Path,
    rows: Iterable[T],
    model_cls: Type[T],
    key: str,
    *,
    remove_predicate,
) -> List[T]:
    existing = read_jsonl(path, model_cls)
    filtered_existing = [item for item in existing if not remove_predicate(item)]
    index: Dict[str, T] = {getattr(item, key): item for item in filtered_existing}
    for row in rows:
        index[getattr(row, key)] = row
    ordered = list(index.values())
    write_jsonl(path, ordered)
    return ordered


def load_paper_bank(path: Path = DEFAULT_PAPER_BANK_PATH) -> List[PaperBankRecord]:
    return read_jsonl(path, PaperBankRecord)


def load_processed_bank(path: Path = DEFAULT_PROCESSED_BANK_PATH) -> List[ProcessedBankRecord]:
    return read_jsonl(path, ProcessedBankRecord)


def load_previous_work_candidates(path: Path = DEFAULT_PREVIOUS_WORK_PATH) -> List[PreviousWorkCandidateRecord]:
    return read_jsonl(path, PreviousWorkCandidateRecord)


def load_benchmark_drafts(path: Path = DEFAULT_BENCHMARK_DRAFTS_PATH) -> List[BenchmarkDraftRecord]:
    return read_jsonl(path, BenchmarkDraftRecord)


def load_job_queue(path: Path = DEFAULT_JOB_QUEUE_PATH) -> List[JobRecord]:
    return read_jsonl(path, JobRecord)


def upsert_paper_bank(rows: Iterable[PaperBankRecord], path: Path = DEFAULT_PAPER_BANK_PATH) -> List[PaperBankRecord]:
    return upsert_records(path, rows, PaperBankRecord, "paper_id")


def upsert_processed_bank(rows: Iterable[ProcessedBankRecord], path: Path = DEFAULT_PROCESSED_BANK_PATH) -> List[ProcessedBankRecord]:
    return upsert_records(path, rows, ProcessedBankRecord, "paper_id")


def upsert_previous_work_candidates(
    rows: Iterable[PreviousWorkCandidateRecord],
    path: Path = DEFAULT_PREVIOUS_WORK_PATH,
) -> List[PreviousWorkCandidateRecord]:
    return upsert_records(path, rows, PreviousWorkCandidateRecord, "candidate_id")


def upsert_benchmark_drafts(
    rows: Iterable[BenchmarkDraftRecord],
    path: Path = DEFAULT_BENCHMARK_DRAFTS_PATH,
) -> List[BenchmarkDraftRecord]:
    return upsert_records(path, rows, BenchmarkDraftRecord, "query_paper_id")


def upsert_job_queue(rows: Iterable[JobRecord], path: Path = DEFAULT_JOB_QUEUE_PATH) -> List[JobRecord]:
    return upsert_records(path, rows, JobRecord, "job_id")


def is_human_complete_draft(draft: BenchmarkDraftRecord) -> bool:
    return bool(draft.gold_prior_paper_ids) and bool(draft.gold_added_information_label)


def replace_previous_work_candidates_for_queries(
    query_paper_ids: Iterable[str],
    rows: Iterable[PreviousWorkCandidateRecord],
    path: Path = DEFAULT_PREVIOUS_WORK_PATH,
) -> List[PreviousWorkCandidateRecord]:
    query_ids = set(query_paper_ids)
    return replace_records(
        path,
        rows,
        PreviousWorkCandidateRecord,
        "candidate_id",
        remove_predicate=lambda item: item.query_paper_id in query_ids,
    )


def replace_benchmark_drafts_for_queries(
    query_paper_ids: Iterable[str],
    rows: Iterable[BenchmarkDraftRecord],
    path: Path = DEFAULT_BENCHMARK_DRAFTS_PATH,
) -> List[BenchmarkDraftRecord]:
    query_ids = set(query_paper_ids)
    return replace_records(
        path,
        rows,
        BenchmarkDraftRecord,
        "query_paper_id",
        remove_predicate=lambda item: item.query_paper_id in query_ids,
    )


def patch_record(path: Path, model_cls: Type[T], key_name: str, key_value: str, patch: Dict) -> T:
    rows = read_jsonl(path, model_cls)
    patched_row: Optional[T] = None
    updated_rows: List[T] = []
    for row in rows:
        if getattr(row, key_name) == key_value:
            data = row.model_dump()
            data.update(patch)
            patched_row = model_cls(**data)
            updated_rows.append(patched_row)
        else:
            updated_rows.append(row)
    if patched_row is None:
        raise KeyError(f"{model_cls.__name__} with {key_name}={key_value} not found")
    write_jsonl(path, updated_rows)
    return patched_row


def load_processed_query_papers(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_query_status(query_paper_id: str) -> str:
    candidates = [row for row in load_previous_work_candidates() if row.query_paper_id == query_paper_id]
    drafts = {row.query_paper_id: row for row in load_benchmark_drafts()}
    paper_bank = {row.paper_id: row for row in load_paper_bank()}
    processed_bank = {row.paper_id: row for row in load_processed_bank()}

    draft = drafts.get(query_paper_id)
    if draft and draft.draft_status == "complete" and is_human_complete_draft(draft):
        return "complete"
    if not candidates:
        return "needs_extraction"
    if any(candidate.resolution_status in {"needs_resolution", "ambiguous", "unresolved"} for candidate in candidates):
        return "needs_resolution"
    resolved_prior_ids = [candidate.resolved_paper_id for candidate in candidates if candidate.resolved_paper_id]
    if resolved_prior_ids and any(not paper_bank.get(paper_id) or paper_bank[paper_id].status in {"resolved_metadata", "unresolved", "fetch_failed"} for paper_id in resolved_prior_ids):
        return "needs_fetch"
    if resolved_prior_ids and any(paper_id not in processed_bank for paper_id in resolved_prior_ids if paper_bank.get(paper_id) and paper_bank[paper_id].status == "fetched"):
        return "needs_processing"
    if draft:
        return "needs_annotation" if draft.draft_status == "complete" else draft.draft_status
    return "needs_annotation"
