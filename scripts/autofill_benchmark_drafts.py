#!/usr/bin/env python3
import argparse
import sys
import os
import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from scv.benchmark_store import (
    load_benchmark_drafts,
    load_previous_work_candidates,
    upsert_benchmark_drafts,
)
from bespokelabs import curator

from scv.benchmark_models import PreviousWorkCandidateRecord

class GeneratedCandidate(BaseModel):
    dataset_name: str
    paper_title: str
    relationship_type: Literal["closest_prior_dataset", "source_dataset", "parallel_benchmark", "evaluation_baseline", "loosely_related", "unknown"]
    description: str
    evidence_text: str

class AutofillDraftOutput(BaseModel):
    generated_candidates: List[GeneratedCandidate] = Field(
        description="If the input candidates are fewer than 6, generate plausible additional negative candidates about this domain so the total number of candidates is at least 6."
    )
    gold_prior_candidate_ids: List[str] = Field(
        description="List of candidate_ids that represent the gold prior paper (the previous dataset/benchmark that provides the most direct comparison basis)."
    )
    gold_prior_dataset_names: List[str] = Field(
        description="Names of the gold prior datasets from the gold prior candidates."
    )
    hard_negative_ids: List[str] = Field(
        description="List of candidate_ids that are highly related but not the actual gold prior (e.g. parallel benchmarks, fundamentally different tasks in same domain)."
    )
    soft_negative_ids: List[str] = Field(
        description="List of candidate_ids that are only loosely related to the queried dataset."
    )
    gold_prior_support_acus: List[str] = Field(
        description="List of factual ACUs (Atomic Content Units) about the gold prior datasets that highlight what they lacked or covered."
    )
    gold_added_information_label: Literal["repackaging", "incremental", "substantial"] = Field(
        description="How much added information does the new query dataset yield compared to the gold prior(s)?"
    )
    annotation_notes: str = Field(
        description="Short reasoning for the chosen candidates and label."
    )


AUTOFILL_PROMPT = """You are an expert annotator completing a benchmark draft about a newly introduced dataset in a query paper.

We have extracted a list of prior-work candidates that the query paper might be comparing itself against. 
Your job is to identify:
1. Which candidates are the "gold prior papers" (the most direct predecessors or baselines the query dataset is meant to replace/improve).
2. Which candidates are "hard negatives" (highly relevant but not a direct predecessor, e.g. a parallel benchmark).
3. Which candidates are "soft negatives" (only loosely related).

Additionally, extract a few "support ACUs" (atomic factual statements) describing what the gold prior dataset contained (to act as the baseline comparison), assess the "added information" of the query dataset relative to the gold prior, and provide short annotation notes justifying your decision.

CRITICAL INSTRUCTION: There must be at least 6 total candidates. If the provided 'Candidates' section has fewer than 6 items, you MUST hallucinate and create plausible prior-work candidates inside `generated_candidates` (setting their titles, dataset names, description, and relationship_type appropriately) so that the sum of provided candidates and generated candidates is >= 6. The gold prior should be chosen from the real provided candidates.

IMPORTANT FOR GENERATED CANDIDATES: DO NOT use generic platforms or corpora like "arXiv", "PubMed", "Wikipedia", or "Common Crawl" as the dataset names. Hallucinate specific, realistic, academically-sounding dataset names and paper titles (e.g., "BioMed-QA: A Large-scale Biomedical Question Answering Dataset", "WikiText-103", "PubMedNLI", etc.) that sound like real NLP datasets within the corresponding domain.

Query Dataset Name: {query_dataset_name}
Query ACUs:
{query_acus}

Candidates:
{candidates_text}

Provide your structured output according to the schema.
Ensure you only use candidate_ids that are listed in the Candidates section. Do not hallucinate IDs.
"""

class BenchmarkAutofiller(curator.LLM):
    response_format = AutofillDraftOutput

    def prompt(self, input: dict) -> str:
        candidates_text = ""
        for cand in input.get("candidates", []):
            candidates_text += f"---\nCandidate ID: {cand.get('candidate_id')}\n"
            candidates_text += f"Paper Title: {cand.get('paper_title')}\n"
            candidates_text += f"Dataset Name: {cand.get('dataset_name')}\n"
            candidates_text += f"Description: {cand.get('description')}\n"
            candidates_text += f"Evidence Text: {cand.get('evidence_text')}\n"
            candidates_text += f"Relationship: {cand.get('relationship_type')}\n"
        
        query_acus_str = "\n".join([f"- {acu}" for acu in input.get("query_acus", [])]) or "- None"
        
        return AUTOFILL_PROMPT.format(
            query_dataset_name=input.get("query_dataset_name", "Unknown"),
            query_acus=query_acus_str,
            candidates_text=candidates_text or "No candidates.",
        )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Autofill benchmark drafts using an LLM")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of drafts to process")
    parser.add_argument("--min-candidates", type=int, default=5, help="Minimum candidates required")
    args = parser.parse_args()

    drafts = load_benchmark_drafts()
    candidates = load_previous_work_candidates()

    # Create mapping from query_paper_id to its candidates
    candidates_by_query = {}
    for cand in candidates:
        if cand.query_paper_id not in candidates_by_query:
            candidates_by_query[cand.query_paper_id] = []
        candidates_by_query[cand.query_paper_id].append(cand)

    fill_inputs = []
    drafts_to_process = []
    for draft in drafts:
        cand_list = candidates_by_query.get(draft.query_paper_id, [])
        if len(cand_list) >= args.min_candidates and draft.draft_status != "complete":
            if args.limit > 0 and len(drafts_to_process) >= args.limit:
                break
            drafts_to_process.append(draft)
            
            cand_dicts = []
            for c in cand_list:
                cand_dicts.append({
                    "candidate_id": c.candidate_id,
                    "paper_title": c.paper_title or "",
                    "dataset_name": c.dataset_name or "",
                    "description": c.description or "",
                    "evidence_text": c.evidence_text or "",
                    "relationship_type": c.relationship_type,
                    "resolved_paper_id": c.resolved_paper_id,
                })
            
            fill_inputs.append({
                "query_paper_id": draft.query_paper_id,
                "query_dataset_name": draft.query_dataset_name,
                "query_acus": draft.query_acus,
                "candidates": cand_dicts,
            })

    if not fill_inputs:
        print("No drafts with > 5 candidates found that need annotation.")
        return

    print(f"Processing {len(fill_inputs)} drafts...")
    autofiller = BenchmarkAutofiller(model_name=args.model, backend=args.backend)
    
    # Process requests
    responses = autofiller(fill_inputs)
    
    # Unpack Dataset outputs. Usually it returns object or object with .dataset
    dataset = responses.dataset if hasattr(responses, 'dataset') else responses
    
    # Update drafts
    updated_drafts = []
    
    # Create an index mapping candidate_id to its candidate object, so we can fetch resolved_paper_id
    cand_id_to_obj = {c.candidate_id: c for c in candidates}
    
    new_candidates_to_upsert = []

    for draft, out in zip(drafts_to_process, dataset):
        output = AutofillDraftOutput(**out) if isinstance(out, dict) else out
        
        for i, gcand in enumerate(output.generated_candidates):
            new_cand = PreviousWorkCandidateRecord(
                query_paper_id=draft.query_paper_id,
                candidate_id=f"{draft.query_paper_id}:llmsynth:{draft.query_dataset_name.replace(' ', '')[:10]}_{i}",
                dataset_name=gcand.dataset_name,
                paper_title=gcand.paper_title,
                relationship_type=gcand.relationship_type,
                description=gcand.description,
                evidence_text=gcand.evidence_text,
                resolution_status="unresolved",
                confidence="low"
            )
            new_candidates_to_upsert.append(new_cand)
            if gcand.relationship_type == "loosely_related":
                output.soft_negative_ids.append(new_cand.candidate_id)
            else:
                output.hard_negative_ids.append(new_cand.candidate_id)
                
        # Translate candidate ids to resolved_paper_ids where applicable for gold priors
        gold_prior_paper_ids = []
        for cid in output.gold_prior_candidate_ids:
            if cid in cand_id_to_obj and cand_id_to_obj[cid].resolved_paper_id:
                gold_prior_paper_ids.append(cand_id_to_obj[cid].resolved_paper_id)
        
        draft.gold_prior_paper_ids = gold_prior_paper_ids
        draft.gold_prior_dataset_names = output.gold_prior_dataset_names
        draft.hard_negative_ids = output.hard_negative_ids
        draft.soft_negative_ids = output.soft_negative_ids
        draft.gold_prior_support_acus = output.gold_prior_support_acus
        draft.gold_added_information_label = output.gold_added_information_label
        draft.annotation_notes = output.annotation_notes
        
        draft.draft_status = "complete"
        updated_drafts.append(draft)

    if new_candidates_to_upsert:
        from scv.benchmark_store import upsert_previous_work_candidates
        upsert_previous_work_candidates(new_candidates_to_upsert)

    upsert_benchmark_drafts(updated_drafts)
    print(f"Successfully updated {len(updated_drafts)} drafts.")

if __name__ == '__main__':
    main()
