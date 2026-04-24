#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from bespokelabs import curator

from scv.benchmark_models import BenchmarkDraftRecord
from scv.benchmark_store import (
    load_benchmark_drafts,
    load_previous_work_candidates,
    load_processed_bank,
    upsert_benchmark_drafts,
)


STRONG_RELATIONSHIPS = {"closest_prior_dataset", "source_dataset"}
USABLE_RELATIONSHIPS = {"closest_prior_dataset", "source_dataset", "evaluation_baseline", "parallel_benchmark"}


class ReviewedDraftOutput(BaseModel):
    gold_prior_paper_ids: List[str] = Field(
        description="Resolved paper IDs from the provided strong candidates that are true gold prior-support papers."
    )
    gold_prior_dataset_names: List[str] = Field(
        description="Dataset names associated with the selected gold prior papers."
    )
    hard_negative_ids: List[str] = Field(
        description="Candidate IDs from the provided candidates that are related but not true prior support."
    )
    soft_negative_ids: List[str] = Field(
        description="Candidate IDs from the provided candidates that are weakly related."
    )
    gold_prior_support_acus: List[str] = Field(
        description="Atomic factual statements about the selected prior datasets/papers that support comparison."
    )
    gold_added_information_label: Literal["repackaging", "incremental", "substantial"] = Field(
        description="Ordinal estimate of added information in the query dataset relative to selected gold prior(s)."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence that the selected gold prior set and label are correct."
    )
    annotation_notes: str = Field(
        description="Concise rationale for the selected gold priors and added-information label."
    )


REVIEW_PROMPT = """You are helping construct a human-reviewable benchmark for dataset added-information estimation.

Your task is to review one query dataset and choose gold prior-support papers from a restricted candidate set.

Important constraints:
- Use ONLY provided candidate IDs and resolved paper IDs.
- Do NOT invent new candidates, papers, datasets, or IDs.
- Prefer candidates whose relationship is closest_prior_dataset or source_dataset.
- A gold prior should be a real prior dataset, benchmark, corpus, source dataset, or shared task that materially supports comparison with the query dataset.
- Do not select method-only related work unless it clearly introduced or supplied a dataset used by the query dataset.
- If multiple strong candidates are complementary ancestors/source datasets, select multiple.
- If a candidate is related but not direct prior support, put it in hard_negative_ids or soft_negative_ids.
- Choose the added-information label relative to the selected gold prior(s):
  - repackaging: mostly repackages or lightly reformats existing data.
  - incremental: adds a modest but real extension, annotation layer, scale/domain/language/task variation, or combination.
  - substantial: introduces a clearly new dataset construction, capability, task setting, modality, annotation structure, or large new resource.

Query dataset:
Name: {query_dataset_name}

Query ACUs:
{query_acus}

Strong candidate prior papers:
{strong_candidates}

Other usable candidates:
{other_candidates}

Return structured output only. Again: do not invent IDs.
"""


class StrongRowReviewer(curator.LLM):
    response_format = ReviewedDraftOutput

    def prompt(self, input: dict) -> str:
        return REVIEW_PROMPT.format(
            query_dataset_name=input["query_dataset_name"],
            query_acus="\n".join(f"- {acu}" for acu in input.get("query_acus", [])) or "- None",
            strong_candidates=_format_candidates(input.get("strong_candidates", [])),
            other_candidates=_format_candidates(input.get("other_candidates", [])),
        )


def _load_processed_payload(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


def _processed_summary(processed_path: str, max_acus: int = 12) -> tuple[List[str], List[str]]:
    payload = _load_processed_payload(processed_path)
    dataset_names: List[str] = []
    acus: List[str] = []
    for dataset in payload.get("datasets", []):
        name = dataset.get("name")
        if name:
            dataset_names.append(name)
        for acu in dataset.get("acus", []):
            if acu and acu not in acus:
                acus.append(acu)
    return dataset_names[:8], acus[:max_acus]


def _format_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return "- None"
    blocks = []
    for candidate in candidates:
        prior_acus = "\n".join(f"  - {acu}" for acu in candidate.get("prior_acus", [])[:10]) or "  - None"
        dataset_names = ", ".join(candidate.get("prior_dataset_names", [])[:8]) or "None"
        blocks.append(
            "\n".join(
                [
                    "---",
                    f"candidate_id: {candidate['candidate_id']}",
                    f"resolved_paper_id: {candidate.get('resolved_paper_id')}",
                    f"relationship_type: {candidate.get('relationship_type')}",
                    f"candidate_dataset_name: {candidate.get('dataset_name') or 'None'}",
                    f"reference_title: {candidate.get('reference_title') or 'None'}",
                    f"resolved_title: {candidate.get('resolved_title') or 'None'}",
                    f"prior_dataset_names: {dataset_names}",
                    f"description: {candidate.get('description') or ''}",
                    f"evidence_text: {candidate.get('evidence_text') or ''}",
                    "prior ACUs:",
                    prior_acus,
                ]
            )
        )
    return "\n".join(blocks)


def _build_review_inputs(limit: int, paper_ids: Optional[set[str]]) -> tuple[list[BenchmarkDraftRecord], list[dict]]:
    drafts = load_benchmark_drafts()
    candidates = load_previous_work_candidates()
    processed_by_paper = {row.paper_id: row for row in load_processed_bank() if row.processing_status == "processed"}
    candidates_by_query: dict[str, list] = {}
    for candidate in candidates:
        candidates_by_query.setdefault(candidate.query_paper_id, []).append(candidate)

    selected_drafts: list[BenchmarkDraftRecord] = []
    inputs: list[dict] = []
    for draft in drafts:
        if paper_ids and draft.query_paper_id not in paper_ids:
            continue
        if draft.draft_status == "complete":
            continue
        query_candidates = candidates_by_query.get(draft.query_paper_id, [])
        enriched = []
        for candidate in query_candidates:
            if not candidate.resolved_paper_id:
                continue
            processed = processed_by_paper.get(candidate.resolved_paper_id)
            if not processed:
                continue
            prior_dataset_names, prior_acus = _processed_summary(processed.processed_json_path)
            enriched.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "resolved_paper_id": candidate.resolved_paper_id,
                    "relationship_type": candidate.relationship_type,
                    "dataset_name": candidate.dataset_name,
                    "reference_title": candidate.reference_title,
                    "resolved_title": next(
                        (paper.title for paper in draft.linked_prior_papers if paper.paper_id == candidate.resolved_paper_id),
                        None,
                    ),
                    "description": candidate.description,
                    "evidence_text": candidate.evidence_text,
                    "prior_dataset_names": prior_dataset_names,
                    "prior_acus": prior_acus,
                }
            )
        strong_candidates = [c for c in enriched if c["relationship_type"] in STRONG_RELATIONSHIPS]
        if not strong_candidates:
            continue
        other_candidates = [c for c in enriched if c["relationship_type"] in USABLE_RELATIONSHIPS and c["relationship_type"] not in STRONG_RELATIONSHIPS]
        selected_drafts.append(draft)
        inputs.append(
            {
                "query_paper_id": draft.query_paper_id,
                "query_dataset_name": draft.query_dataset_name,
                "query_acus": draft.query_acus,
                "strong_candidates": strong_candidates,
                "other_candidates": other_candidates,
            }
        )
        if limit and len(inputs) >= limit:
            break
    return selected_drafts, inputs


def _validate_ids(output: ReviewedDraftOutput, review_input: dict) -> ReviewedDraftOutput:
    candidate_ids = {candidate["candidate_id"] for candidate in review_input["strong_candidates"] + review_input["other_candidates"]}
    strong_paper_ids = {candidate["resolved_paper_id"] for candidate in review_input["strong_candidates"]}
    output.gold_prior_paper_ids = [paper_id for paper_id in output.gold_prior_paper_ids if paper_id in strong_paper_ids]
    output.hard_negative_ids = [candidate_id for candidate_id in output.hard_negative_ids if candidate_id in candidate_ids]
    output.soft_negative_ids = [candidate_id for candidate_id in output.soft_negative_ids if candidate_id in candidate_ids]
    return output


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Use an LLM to review rows with strong processed prior candidates.")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--paper-ids", nargs="*", default=None)
    parser.add_argument("--list-only", action="store_true", help="List eligible rows without calling the LLM.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mark-complete", action="store_true", help="Mark rows complete when the LLM selects at least one valid gold prior.")
    args = parser.parse_args()

    selected_drafts, review_inputs = _build_review_inputs(args.limit, set(args.paper_ids or []))
    if not review_inputs:
        print("No eligible rows with strong processed candidates found.")
        return

    if args.list_only:
        print(f"Eligible rows: {len(review_inputs)}")
        for draft, review_input in zip(selected_drafts, review_inputs):
            print(
                f"{draft.query_paper_id}\t{draft.query_dataset_name}\t"
                f"strong={len(review_input['strong_candidates'])}\t"
                f"other={len(review_input['other_candidates'])}"
            )
        return

    print(f"Reviewing {len(review_inputs)} rows with {args.model}.")
    reviewer = StrongRowReviewer(model_name=args.model, backend=args.backend)
    responses = reviewer(review_inputs)
    outputs = responses.dataset if hasattr(responses, "dataset") else responses

    updated_drafts: list[BenchmarkDraftRecord] = []
    for draft, review_input, raw_output in zip(selected_drafts, review_inputs, outputs):
        output = ReviewedDraftOutput(**raw_output) if isinstance(raw_output, dict) else raw_output
        output = _validate_ids(output, review_input)
        if not output.gold_prior_paper_ids:
            print(f"Skipped {draft.query_paper_id}: LLM did not select a valid strong gold prior.")
            continue
        draft.gold_prior_paper_ids = output.gold_prior_paper_ids
        draft.gold_prior_dataset_names = output.gold_prior_dataset_names
        draft.hard_negative_ids = output.hard_negative_ids
        draft.soft_negative_ids = output.soft_negative_ids
        draft.gold_prior_support_acus = output.gold_prior_support_acus
        draft.gold_added_information_label = output.gold_added_information_label
        draft.annotation_notes = (
            "LLM_ASSISTED_REVIEW: "
            f"confidence={output.confidence}. "
            f"{output.annotation_notes}"
        )
        draft.draft_status = "complete" if args.mark_complete else "needs_annotation"
        updated_drafts.append(draft)
        print(
            f"{draft.query_paper_id}: selected {len(output.gold_prior_paper_ids)} gold prior(s), "
            f"label={output.gold_added_information_label}, confidence={output.confidence}, "
            f"status={draft.draft_status}"
        )

    if args.dry_run:
        print(f"Dry run: would update {len(updated_drafts)} drafts.")
        return
    if updated_drafts:
        upsert_benchmark_drafts(updated_drafts)
    print(f"Updated {len(updated_drafts)} drafts.")


if __name__ == "__main__":
    main()
