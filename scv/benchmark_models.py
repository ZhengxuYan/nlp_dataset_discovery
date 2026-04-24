from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


RelationshipType = Literal[
    "closest_prior_dataset",
    "source_dataset",
    "parallel_benchmark",
    "evaluation_baseline",
    "loosely_related",
    "unknown",
]

ResolutionStatus = Literal[
    "needs_resolution",
    "resolved_in_db",
    "resolved_arxiv",
    "unresolved",
    "ambiguous",
]

PaperStatus = Literal[
    "query_seed",
    "resolved_metadata",
    "fetched",
    "fetch_failed",
    "processing",
    "processed",
    "processing_failed",
    "unresolved",
]

DraftStatus = Literal[
    "needs_extraction",
    "needs_resolution",
    "needs_fetch",
    "needs_processing",
    "needs_annotation",
    "complete",
]

JobStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "stale",
]

JobType = Literal[
    "extract_candidates",
    "resolve_candidates",
    "fetch_papers",
    "process_papers",
    "build_drafts",
    "run_pipeline_for_query",
    "bulk_extract_missing",
    "bulk_resolve_all",
    "bulk_fetch_all",
    "bulk_process_all",
    "bulk_build_all",
]


class PaperBankRecord(BaseModel):
    paper_id: str
    arxiv_id: Optional[str] = None
    title: str
    url: Optional[str] = None
    pdf_path: Optional[str] = None
    source: str = "unknown"
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    status: PaperStatus = "query_seed"
    is_query_seed: bool = False
    query_source_path: Optional[str] = None
    notes: Optional[str] = None


class ProcessedBankRecord(BaseModel):
    paper_id: str
    processed_json_path: str
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    processed_at: Optional[str] = None
    processing_status: Literal["processed", "processing_failed"] = "processed"
    error_message: Optional[str] = None


class PreviousWorkCandidateRecord(BaseModel):
    query_paper_id: str
    candidate_id: str
    dataset_name: Optional[str] = None
    paper_title: Optional[str] = None
    citation_key: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    citation_marker: Optional[str] = None
    reference_title: Optional[str] = None
    reference_authors: List[str] = Field(default_factory=list)
    reference_year: Optional[int] = None
    reference_url: Optional[str] = None
    reference_doi: Optional[str] = None
    reference_arxiv_id: Optional[str] = None
    resolution_source: Optional[str] = None
    relationship_type: RelationshipType = "unknown"
    description: str
    confidence: str = "medium"
    evidence_text: str
    resolution_status: ResolutionStatus = "needs_resolution"
    resolved_paper_id: Optional[str] = None
    resolved_arxiv_id: Optional[str] = None
    resolved_url: Optional[str] = None
    resolution_candidates: List[str] = Field(default_factory=list)
    annotation_notes: Optional[str] = None


class PriorPaperSummary(BaseModel):
    paper_id: str
    title: str
    arxiv_id: Optional[str] = None
    processed_json_path: Optional[str] = None
    dataset_names: List[str] = Field(default_factory=list)
    acus: List[str] = Field(default_factory=list)


class BenchmarkDraftRecord(BaseModel):
    query_paper_id: str
    query_dataset_name: str
    query_acus: List[str] = Field(default_factory=list)
    candidate_ids: List[str] = Field(default_factory=list)
    gold_prior_paper_ids: List[str] = Field(default_factory=list)
    gold_prior_dataset_names: List[str] = Field(default_factory=list)
    hard_negative_ids: List[str] = Field(default_factory=list)
    soft_negative_ids: List[str] = Field(default_factory=list)
    suggested_gold_prior_paper_ids: List[str] = Field(default_factory=list)
    suggested_hard_negative_ids: List[str] = Field(default_factory=list)
    suggested_soft_negative_ids: List[str] = Field(default_factory=list)
    gold_prior_support_acus: List[str] = Field(default_factory=list)
    gold_added_information_label: Optional[str] = None
    annotation_notes: str = ""
    draft_status: DraftStatus = "needs_extraction"
    linked_prior_papers: List[PriorPaperSummary] = Field(default_factory=list)
    canonical: bool = True


class JobRecord(BaseModel):
    job_id: str
    job_type: JobType
    query_paper_id: Optional[str] = None
    paper_ids: List[str] = Field(default_factory=list)
    status: JobStatus = "queued"
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error_message: Optional[str] = None
