#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import evaluate_global_claim_level_dcu_retrieval as ev


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Claim-Level DCU Hard Negatives",
        "",
        f"- Rows: {summary['rows']}",
        f"- Query DCUs: {summary['query_dcus']}",
        f"- Seed method: {summary['seed_method']}",
        f"- Candidate depth: {summary['candidate_depth']}",
        f"- Negatives per query: {summary['negatives_per_query']}",
        "",
        "## Negative Types",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted((summary.get("negative_type_counts") or {}).items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Delta Types", "", "| Delta type | Count |", "| --- | ---: |"])
    for key, value in sorted((summary.get("delta_type_counts") or {}).items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def build_indexes(candidate_ids: list[str], candidates: list[dict[str, Any]], papers: dict[str, dict[str, Any]]) -> dict[str, ev.TextIndex]:
    paper_ids = sorted(papers)
    return {
        "paper_abstract": ev.TextIndex(paper_ids, [papers[paper_id].get("paper_abstract_text") or "" for paper_id in paper_ids]),
        "paper_summary": ev.TextIndex(paper_ids, [papers[paper_id].get("paper_summary_text") or "" for paper_id in paper_ids]),
        "paper_full": ev.TextIndex(paper_ids, [papers[paper_id].get("paper_full_text") or papers[paper_id].get("paper_summary_text") or "" for paper_id in paper_ids]),
        "acu": ev.TextIndex(candidate_ids, [ev.serialize(candidate, "acu") for candidate in candidates]),
        "typed_acu": ev.TextIndex(candidate_ids, [ev.serialize(candidate, "typed_acu") for candidate in candidates]),
        "dcu": ev.TextIndex(candidate_ids, [ev.serialize(candidate, "dcu") for candidate in candidates]),
        "contextual_dcu": ev.TextIndex(candidate_ids, [ev.serialize(candidate, "contextual_dcu") for candidate in candidates]),
    }


def token_set(text: str) -> set[str]:
    return set(ev.tokenize_list(text))


def overlap_features(query_dcu: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    query_tokens = token_set(query_dcu.get("text") or "")
    candidate_tokens = token_set(candidate.get("text") or "")
    query_meta = query_dcu.get("metadata") or {}
    candidate_meta = candidate.get("metadata") or {}
    metadata_overlap = max(
        ev.jaccard(ev.list_tokens(query_meta.get("tasks")), ev.list_tokens(candidate_meta.get("tasks"))),
        ev.jaccard(ev.list_tokens(query_meta.get("domains")), ev.list_tokens(candidate_meta.get("domains"))),
        ev.jaccard(ev.list_tokens(query_meta.get("languages")), ev.list_tokens(candidate_meta.get("languages"))),
        ev.jaccard(ev.list_tokens(query_meta.get("modalities")), ev.list_tokens(candidate_meta.get("modalities"))),
    )
    source_overlap = max(
        ev.jaccard(ev.list_tokens(query_meta.get("source_datasets")), ev.list_tokens(candidate_meta.get("source_datasets"))),
        ev.jaccard(token_set(str(query_meta.get("source_data_origin") or "")), token_set(str(candidate_meta.get("source_data_origin") or ""))),
    )
    return {
        "lexical_jaccard": ev.jaccard(query_tokens, candidate_tokens),
        "lexical_containment": ev.containment(query_tokens, candidate_tokens),
        "metadata_overlap": metadata_overlap,
        "source_overlap": source_overlap,
    }


def negative_types(
    *,
    query_dcu: dict[str, Any],
    candidate: dict[str, Any],
    features: dict[str, float],
    negative_rank: int,
    best_gold_rank: int | None,
) -> list[str]:
    types = []
    if best_gold_rank is not None and negative_rank < best_gold_rank:
        types.append("ranked_before_gold")
    if query_dcu.get("type") and query_dcu.get("type") == candidate.get("type"):
        types.append("same_contribution_type")
    if features["lexical_jaccard"] >= 0.12 or features["lexical_containment"] >= 0.25:
        types.append("high_lexical_overlap")
    if features["metadata_overlap"] >= 0.20:
        types.append("metadata_overlap")
    if features["source_overlap"] >= 0.20:
        types.append("source_overlap")
    return types or ["surface_similar_non_gold"]


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "paper": candidate.get("prior_paper_title"),
        "dataset": candidate.get("prior_dataset_name"),
        "type": candidate.get("type"),
        "text": candidate.get("text"),
    }


def build_hard_negatives(
    *,
    claim_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    papers: dict[str, dict[str, Any]],
    seed_method: str,
    candidate_depth: int,
    negatives_per_query: int,
    only_before_gold: bool,
    negative_source: str,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    candidate_ids = [candidate["id"] for candidate in candidates]
    indexes = build_indexes(candidate_ids, candidates, papers)
    query_items, missing_gold = ev.prepare_queries(claim_rows, set(candidate_by_id), limit)
    rows: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    delta_counts: Counter[str] = Counter()
    ranked_before_gold = 0

    for index, item in enumerate(query_items, start=1):
        query_dcu = item["query_dcu"]
        gold_ids = set(item["gold_ids"])
        scored = ev.score_retrieval_method(seed_method, query_dcu, candidates, papers, indexes)
        ranking = ev.stable_rank(scored, candidate_by_id)
        rank_lookup = {candidate_id: rank for rank, candidate_id in enumerate(ranking, start=1)}
        gold_ranks = [rank_lookup[gold_id] for gold_id in gold_ids if gold_id in rank_lookup]
        best_gold_rank = min(gold_ranks) if gold_ranks else None
        if best_gold_rank is None:
            continue
        scored_negatives = []
        if negative_source == "same_gold_paper":
            gold_paper_ids = {
                candidate_by_id[gold_id].get("prior_paper_id")
                for gold_id in gold_ids
                if gold_id in candidate_by_id
            }
            candidate_pool = [
                candidate["id"]
                for candidate in candidates
                if candidate["id"] not in gold_ids and candidate.get("prior_paper_id") in gold_paper_ids
            ]
        elif negative_source == "mixed":
            gold_paper_ids = {
                candidate_by_id[gold_id].get("prior_paper_id")
                for gold_id in gold_ids
                if gold_id in candidate_by_id
            }
            same_paper_pool = [
                candidate["id"]
                for candidate in candidates
                if candidate["id"] not in gold_ids and candidate.get("prior_paper_id") in gold_paper_ids
            ]
            seed_pool = [candidate_id for candidate_id in ranking[:candidate_depth] if candidate_id not in gold_ids]
            candidate_pool = ev.unique(same_paper_pool + seed_pool)
        else:
            candidate_pool = [candidate_id for candidate_id in ranking[:candidate_depth] if candidate_id not in gold_ids]
        for candidate_id in candidate_pool:
            if candidate_id in gold_ids:
                continue
            negative_rank = rank_lookup.get(candidate_id, len(ranking) + 1)
            if only_before_gold and negative_rank >= best_gold_rank:
                continue
            candidate = candidate_by_id[candidate_id]
            features = overlap_features(query_dcu, candidate)
            types = negative_types(
                query_dcu=query_dcu,
                candidate=candidate,
                features=features,
                negative_rank=negative_rank,
                best_gold_rank=best_gold_rank,
            )
            if candidate.get("prior_paper_id") in {
                candidate_by_id[gold_id].get("prior_paper_id")
                for gold_id in gold_ids
                if gold_id in candidate_by_id
            }:
                types = ev.unique(["same_gold_prior_paper"] + types)
            hardness = (
                int("same_gold_prior_paper" in types),
                int("ranked_before_gold" in types),
                int("same_contribution_type" in types),
                int("source_overlap" in types),
                int("metadata_overlap" in types),
                features["lexical_containment"],
                features["lexical_jaccard"],
                -negative_rank,
            )
            scored_negatives.append((hardness, candidate_id, negative_rank, types, features))
        scored_negatives.sort(reverse=True)
        for _, candidate_id, negative_rank, types, features in scored_negatives[:negatives_per_query]:
            if negative_rank < best_gold_rank:
                ranked_before_gold += 1
            type_counts.update(types)
            delta_type = item["label"].get("delta_type") or query_dcu.get("type") or "other"
            delta_counts[delta_type] += 1
            rows.append({
                "benchmark_id": item["row"].get("benchmark_id"),
                "query_dcu_id": query_dcu["id"],
                "query_dataset_name": item["row"].get("query_dataset_name"),
                "query_title": item["row"].get("query_title"),
                "query_text": query_dcu.get("text"),
                "query_type": query_dcu.get("type"),
                "delta_type": delta_type,
                "gold_prior_dcu_ids": list(gold_ids),
                "gold_prior_dcus": [summarize_candidate(candidate_by_id[gold_id]) for gold_id in gold_ids],
                "best_gold_rank": best_gold_rank,
                "hard_negative_dcu_id": candidate_id,
                "hard_negative_rank": negative_rank,
                "hard_negative_dcu": summarize_candidate(candidate_by_id[candidate_id]),
                "negative_types": types,
                "features": features,
                "seed_method": seed_method,
                "candidate_depth": candidate_depth,
                "negative_source": negative_source,
                "ranked_before_gold": negative_rank < best_gold_rank,
            })
        if index % 25 == 0:
            print(json.dumps({"progress": index, "total": len(query_items), "hard_negatives": len(rows)}, ensure_ascii=False), flush=True)

    summary = {
        "rows": len(rows),
        "query_dcus": len(query_items),
        "missing_gold_labels": missing_gold,
        "global_prior_dcus": len(candidates),
        "seed_method": seed_method,
        "candidate_depth": candidate_depth,
        "negatives_per_query": negatives_per_query,
        "only_before_gold": only_before_gold,
        "negative_source": negative_source,
        "ranked_before_gold_count": ranked_before_gold,
        "ranked_before_gold_rate": ranked_before_gold / len(rows) if rows else 0.0,
        "negative_type_counts": dict(type_counts),
        "delta_type_counts": dict(delta_counts),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build claim-level DCU hard negatives from global prior DCU rankings.")
    parser.add_argument("--claim-level-jsonl", required=True)
    parser.add_argument("--prior-extractions-jsonl", required=True)
    parser.add_argument("--seed-method", default="hybrid_acu")
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--negatives-per-query", type=int, default=3)
    parser.add_argument("--negative-source", choices=["seed_ranking", "same_gold_paper", "mixed"], default="seed_ranking")
    parser.add_argument("--only-before-gold", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    if args.seed_method not in ev.RETRIEVAL_METHODS | ev.HEURISTIC_METHODS | ev.RANDOM_METHODS:
        raise SystemExit(f"Unknown seed method: {args.seed_method}")
    candidates, papers = ev.flatten_prior_acus(ev.read_jsonl(args.prior_extractions_jsonl))
    rows, summary = build_hard_negatives(
        claim_rows=ev.read_jsonl(args.claim_level_jsonl),
        candidates=candidates,
        papers=papers,
        seed_method=args.seed_method,
        candidate_depth=args.candidate_depth,
        negatives_per_query=args.negatives_per_query,
        only_before_gold=args.only_before_gold,
        negative_source=args.negative_source,
        limit=args.limit,
    )
    write_jsonl(args.output_jsonl, rows)
    write_json(args.summary_json, summary)
    if args.output_markdown:
        Path(args.output_markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_markdown).write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({
        "output_jsonl": args.output_jsonl,
        "summary_json": args.summary_json,
        "output_markdown": args.output_markdown,
        **summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
