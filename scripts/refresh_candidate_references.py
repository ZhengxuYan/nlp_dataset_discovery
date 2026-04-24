#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scv.benchmark_store import (
    DEFAULT_PREVIOUS_WORK_PATH,
    canonical_paper_id,
    load_previous_work_candidates,
    load_processed_query_papers,
    write_jsonl,
)
from scv.prior_work_builder import (
    PreviousWorkCandidate,
    _find_reference_for_candidate,
    parse_reference_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh bibliography-derived fields on existing previous-work candidates.")
    parser.add_argument("--input", type=str, default="data/processed/final_scv_200.jsonl")
    parser.add_argument("--paper-ids", nargs="*", default=None)
    args = parser.parse_args()

    query_rows = load_processed_query_papers(args.input)
    arxiv_by_paper_id = {
        canonical_paper_id(row.get("arxiv_id"), row.get("title", "")): row.get("arxiv_id")
        for row in query_rows
    }
    selected = set(args.paper_ids or [])
    candidates = load_previous_work_candidates()
    updated = []
    changed = 0

    for candidate in candidates:
        if selected and candidate.query_paper_id not in selected:
            updated.append(candidate)
            continue
        arxiv_id = arxiv_by_paper_id.get(candidate.query_paper_id)
        source_dir = Path("data/extracted_papers") / arxiv_id if arxiv_id else None
        references = parse_reference_map(source_dir) if source_dir and source_dir.exists() else {}
        reference = _find_reference_for_candidate(
            PreviousWorkCandidate(
                dataset_name=candidate.dataset_name,
                paper_title=candidate.paper_title,
                citation_key=candidate.citation_key,
                authors=candidate.authors,
                year=candidate.year,
                citation_marker=candidate.citation_marker,
                relationship_type=candidate.relationship_type,
                description=candidate.description,
                confidence=candidate.confidence,
                evidence_text=candidate.evidence_text,
            ),
            references,
        )
        if reference:
            before = candidate.model_dump()
            candidate.reference_title = reference.title or None
            candidate.reference_authors = reference.authors
            candidate.reference_year = reference.year
            candidate.reference_url = reference.url
            candidate.reference_doi = reference.doi
            candidate.reference_arxiv_id = reference.arxiv_id
            candidate.resolution_source = "reference_arxiv" if reference.arxiv_id else "reference_title" if reference.title else candidate.resolution_source
            if before != candidate.model_dump():
                changed += 1
        updated.append(candidate)

    write_jsonl(DEFAULT_PREVIOUS_WORK_PATH, updated)
    print(f"Refreshed bibliography fields for {changed} candidates.")


if __name__ == "__main__":
    main()
