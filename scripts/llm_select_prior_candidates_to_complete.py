#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from bespokelabs import curator

from scv.benchmark_store import (
    DEFAULT_BENCHMARK_DIR,
    load_benchmark_drafts,
    load_previous_work_candidates,
)


TASK_OUTPUT_PATH = DEFAULT_BENCHMARK_DIR / "prior_completion_tasks.jsonl"
PREFERRED_RELATIONSHIPS = {
    "closest_prior_dataset",
    "source_dataset",
    "evaluation_baseline",
    "parallel_benchmark",
}


class CandidateSelectionOutput(BaseModel):
    selected_candidate_ids: List[str] = Field(
        description="Candidate IDs to prioritize for completion. Select 1-3 candidates only from the provided list."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence that selected candidates are correct prior-support candidates worth completing."
    )
    rationale: str = Field(
        description="Short reason for selecting these candidates."
    )
    skip_row: bool = Field(
        description="True if none of the provided candidates is worth completing."
    )


TASK_PROMPT = """You are helping build a benchmark of dataset added information.

For one query dataset, choose which unresolved/unprocessed previous-work candidates should be completed first.

Goal:
- Select candidates that are likely to be true prior-support datasets, source datasets, benchmarks, corpora, or shared tasks.
- Prefer closest_prior_dataset and source_dataset.
- Select at most 3 candidates.
- Do not select generic method-only related work.
- If no candidate is strong enough to justify PDF/manual completion, set skip_row=true and select no candidates.

Query dataset:
Name: {query_dataset_name}

Query ACUs:
{query_acus}

Candidates:
{candidates}

Return structured output only. Use only provided candidate IDs.
"""


class PriorCompletionSelector(curator.LLM):
    response_format = CandidateSelectionOutput

    def prompt(self, input: dict) -> str:
        return TASK_PROMPT.format(
            query_dataset_name=input["query_dataset_name"],
            query_acus="\n".join(f"- {acu}" for acu in input.get("query_acus", [])) or "- None",
            candidates=_format_candidates(input.get("candidates", [])),
        )


def _format_candidates(candidates: list[dict]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(
            "\n".join(
                [
                    "---",
                    f"candidate_id: {candidate['candidate_id']}",
                    f"name: {candidate.get('name') or 'None'}",
                    f"relationship_type: {candidate.get('relationship_type')}",
                    f"resolution_status: {candidate.get('resolution_status')}",
                    f"reference_title: {candidate.get('reference_title') or 'None'}",
                    f"reference_url: {candidate.get('reference_url') or 'None'}",
                    f"resolved_paper_id: {candidate.get('resolved_paper_id') or 'None'}",
                    f"description: {candidate.get('description') or ''}",
                    f"evidence_text: {candidate.get('evidence_text') or ''}",
                ]
            )
        )
    return "\n".join(blocks) or "- None"


def _action_needed(candidate) -> str:
    if candidate.resolved_paper_id:
        return "process_or_review_existing_resolution"
    if candidate.reference_arxiv_id:
        return "resolve_by_reference_arxiv"
    if candidate.reference_url:
        return "resolve_or_download_from_reference_url"
    if candidate.reference_title or candidate.paper_title:
        return "resolve_by_title"
    return "manual_find_pdf"


def _build_inputs(limit: int, paper_ids: Optional[set[str]]) -> tuple[list, list[dict]]:
    drafts = load_benchmark_drafts()
    candidates = load_previous_work_candidates()
    candidates_by_query: dict[str, list] = {}
    for candidate in candidates:
        candidates_by_query.setdefault(candidate.query_paper_id, []).append(candidate)

    selected_drafts = []
    inputs = []
    for draft in drafts:
        if paper_ids and draft.query_paper_id not in paper_ids:
            continue
        if draft.draft_status == "complete":
            continue
        query_candidates = []
        for candidate in candidates_by_query.get(draft.query_paper_id, []):
            if candidate.relationship_type not in PREFERRED_RELATIONSHIPS:
                continue
            if candidate.resolution_status in {"resolved_arxiv", "resolved_in_db"} and candidate.resolved_paper_id:
                continue
            query_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "name": candidate.dataset_name or candidate.reference_title or candidate.paper_title,
                    "relationship_type": candidate.relationship_type,
                    "resolution_status": candidate.resolution_status,
                    "reference_title": candidate.reference_title,
                    "reference_url": candidate.reference_url,
                    "resolved_paper_id": candidate.resolved_paper_id,
                    "description": candidate.description,
                    "evidence_text": candidate.evidence_text,
                }
            )
        if not query_candidates:
            continue
        selected_drafts.append(draft)
        inputs.append(
            {
                "query_paper_id": draft.query_paper_id,
                "query_dataset_name": draft.query_dataset_name,
                "query_acus": draft.query_acus,
                "candidates": query_candidates,
            }
        )
        if limit and len(inputs) >= limit:
            break
    return selected_drafts, inputs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Use an LLM to select unresolved prior candidates worth completing.")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--paper-ids", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=TASK_OUTPUT_PATH)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    drafts, inputs = _build_inputs(args.limit, set(args.paper_ids or []))
    if args.list_only:
        print(f"Eligible incomplete rows with unresolved preferred candidates: {len(inputs)}")
        for draft, review_input in zip(drafts, inputs):
            print(f"{draft.query_paper_id}\t{draft.query_dataset_name}\tcandidates={len(review_input['candidates'])}")
        return
    if not inputs:
        print("No eligible rows found.")
        return

    selector = PriorCompletionSelector(model_name=args.model, backend=args.backend)
    print(f"Selecting completion candidates for {len(inputs)} rows with {args.model}.")
    responses = selector(inputs)
    outputs = responses.dataset if hasattr(responses, "dataset") else responses

    candidate_index = {
        candidate.candidate_id: candidate
        for candidate in load_previous_work_candidates()
    }
    task_rows: list[dict] = []
    for draft, review_input, raw_output in zip(drafts, inputs, outputs):
        output = CandidateSelectionOutput(**raw_output) if isinstance(raw_output, dict) else raw_output
        valid_ids = {candidate["candidate_id"] for candidate in review_input["candidates"]}
        selected_ids = [candidate_id for candidate_id in output.selected_candidate_ids if candidate_id in valid_ids]
        if output.skip_row or not selected_ids:
            continue
        for rank, candidate_id in enumerate(selected_ids, start=1):
            candidate = candidate_index[candidate_id]
            task_rows.append(
                {
                    "query_paper_id": draft.query_paper_id,
                    "query_dataset_name": draft.query_dataset_name,
                    "candidate_id": candidate.candidate_id,
                    "rank": rank,
                    "candidate_name": candidate.dataset_name or candidate.reference_title or candidate.paper_title,
                    "relationship_type": candidate.relationship_type,
                    "resolution_status": candidate.resolution_status,
                    "reference_title": candidate.reference_title,
                    "reference_url": candidate.reference_url,
                    "reference_arxiv_id": candidate.reference_arxiv_id,
                    "resolved_paper_id": candidate.resolved_paper_id,
                    "action_needed": _action_needed(candidate),
                    "confidence": output.confidence,
                    "rationale": output.rationale,
                }
            )

    write_jsonl(args.output, task_rows)
    print(f"Wrote {len(task_rows)} completion tasks to {args.output}.")


if __name__ == "__main__":
    main()
