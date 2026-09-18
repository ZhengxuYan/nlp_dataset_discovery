#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import evaluate_global_claim_level_dcu_retrieval as ev
from build_claim_level_dcu_hard_negative_benchmark import build_indexes


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def compare_ranks(ranking: list[str], gold_ids: list[str], negative_id: str) -> tuple[str, int | None, int | None]:
    lookup = {candidate_id: index for index, candidate_id in enumerate(ranking, start=1)}
    gold_ranks = [lookup[gold_id] for gold_id in gold_ids if gold_id in lookup]
    best_gold_rank = min(gold_ranks) if gold_ranks else None
    negative_rank = lookup.get(negative_id)
    if best_gold_rank is None and negative_rank is None:
        return "unresolved_both_missing", best_gold_rank, negative_rank
    if best_gold_rank is None:
        return "negative_ahead", best_gold_rank, negative_rank
    if negative_rank is None:
        return "gold_ahead", best_gold_rank, negative_rank
    if best_gold_rank < negative_rank:
        return "gold_ahead", best_gold_rank, negative_rank
    if negative_rank < best_gold_rank:
        return "negative_ahead", best_gold_rank, negative_rank
    return "tie", best_gold_rank, negative_rank


def summarize_counts(counts: Counter[str]) -> dict[str, Any]:
    evaluable = counts["gold_ahead"] + counts["negative_ahead"] + counts["tie"]
    return {
        "n": counts["n"],
        "evaluable": evaluable,
        "gold_ahead": counts["gold_ahead"],
        "negative_ahead": counts["negative_ahead"],
        "tie": counts["tie"],
        "unresolved_both_missing": counts["unresolved_both_missing"],
        "hard_negative_rejection_rate": counts["gold_ahead"] / evaluable if evaluable else 0.0,
        "hard_negative_error_rate": counts["negative_ahead"] / evaluable if evaluable else 0.0,
    }


def evaluate(
    *,
    hard_negative_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    methods: list[str],
    model: str,
    rerank_depth: int,
    cache_dir: str,
) -> dict[str, Any]:
    candidates, papers = ev.flatten_prior_acus(prior_rows)
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    candidate_ids = [candidate["id"] for candidate in candidates]
    indexes = build_indexes(candidate_ids, candidates, papers)
    cache = ev.JsonCache(cache_dir)

    by_method = {}
    by_negative_type: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    by_delta_type: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for method in methods:
        counts: Counter[str] = Counter()
        ranking_cache: dict[str, list[str]] = {}
        for row in hard_negative_rows:
            query_dcu = {
                "id": row["query_dcu_id"],
                "text": row.get("query_text") or "",
                "type": row.get("query_type") or row.get("delta_type") or "",
                "query_title": row.get("query_title") or "",
                "prior_dataset_name": row.get("query_dataset_name") or "",
                "metadata": {"dataset_name": row.get("query_dataset_name") or ""},
            }
            cache_key = query_dcu["id"]
            if cache_key not in ranking_cache:
                actual_method = "hybrid_dcu" if method == "hybrid_dcu_support_rerank" else method
                scored = ev.score_retrieval_method(actual_method, query_dcu, candidates, papers, indexes)
                ranking = ev.stable_rank(scored, candidate_by_id)
                if method == "hybrid_dcu_support_rerank":
                    pool = ranking[:rerank_depth]
                    reranked = ev.support_rerank(query_dcu, pool, candidate_by_id, model=model, cache=cache)
                    ranking = ev.unique(reranked + ranking)
                ranking_cache[cache_key] = ranking
            ranking = ranking_cache[cache_key]
            outcome, best_gold_rank, negative_rank = compare_ranks(
                ranking,
                row.get("gold_prior_dcu_ids") or [],
                row.get("hard_negative_dcu_id") or "",
            )
            counts["n"] += 1
            counts[outcome] += 1
            for negative_type in row.get("negative_types") or ["unknown"]:
                by_negative_type[method][negative_type]["n"] += 1
                by_negative_type[method][negative_type][outcome] += 1
            delta_type = row.get("delta_type") or "other"
            by_delta_type[method][delta_type]["n"] += 1
            by_delta_type[method][delta_type][outcome] += 1
            if outcome == "negative_ahead" and len(examples[method]) < 10:
                examples[method].append({
                    "query_dcu_id": row.get("query_dcu_id"),
                    "query_dataset_name": row.get("query_dataset_name"),
                    "query_text": row.get("query_text"),
                    "gold_prior_dcu_ids": row.get("gold_prior_dcu_ids") or [],
                    "hard_negative_dcu_id": row.get("hard_negative_dcu_id"),
                    "best_gold_rank": best_gold_rank,
                    "negative_rank": negative_rank,
                    "negative_types": row.get("negative_types") or [],
                    "hard_negative_dcu": row.get("hard_negative_dcu"),
                })
        by_method[method] = summarize_counts(counts)

    return {
        "rows": len(hard_negative_rows),
        "global_prior_dcus": len(candidates),
        "methods": methods,
        "model": model,
        "rerank_depth": rerank_depth,
        "by_method": by_method,
        "by_negative_type": {
            method: {key: summarize_counts(value) for key, value in type_rows.items()}
            for method, type_rows in by_negative_type.items()
        },
        "by_delta_type": {
            method: {key: summarize_counts(value) for key, value in type_rows.items()}
            for method, type_rows in by_delta_type.items()
        },
        "negative_ahead_examples": dict(examples),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Claim-Level DCU Hard-Negative Rejection",
        "",
        f"- Rows: {report['rows']}",
        f"- Global prior DCUs: {report['global_prior_dcus']}",
        "",
        "## Overall",
        "",
        "| Method | N | Evaluable | Rejection Rate | Error Rate | Gold Ahead | Negative Ahead | Tie | Unresolved |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in report["by_method"].items():
        lines.append(
            f"| {method} | {row['n']} | {row['evaluable']} | "
            f"{row['hard_negative_rejection_rate']:.3f} | {row['hard_negative_error_rate']:.3f} | "
            f"{row['gold_ahead']} | {row['negative_ahead']} | {row['tie']} | {row['unresolved_both_missing']} |"
        )
    lines.extend(["", "## By Negative Type", ""])
    for method, rows in report.get("by_negative_type", {}).items():
        lines.extend([f"### {method}", "", "| Type | N | Rejection Rate | Error Rate |", "| --- | ---: | ---: | ---: |"])
        for negative_type, row in sorted(rows.items()):
            lines.append(
                f"| {negative_type} | {row['n']} | "
                f"{row['hard_negative_rejection_rate']:.3f} | {row['hard_negative_error_rate']:.3f} |"
            )
        lines.append("")
    lines.extend(["", "## By Delta Type", ""])
    for method, rows in report.get("by_delta_type", {}).items():
        lines.extend([f"### {method}", "", "| Delta type | N | Rejection Rate | Error Rate |", "| --- | ---: | ---: | ---: |"])
        for delta_type, row in sorted(rows.items()):
            lines.append(
                f"| {delta_type} | {row['n']} | "
                f"{row['hard_negative_rejection_rate']:.3f} | {row['hard_negative_error_rate']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate claim-level DCU hard-negative rejection.")
    parser.add_argument("--hard-negative-jsonl", required=True)
    parser.add_argument("--prior-extractions-jsonl", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--cache-dir", default="data/benchmark/retrieval_cache/global_claim_level_dcu_rerank_cache")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    unknown = sorted(set(args.methods) - ev.HEURISTIC_METHODS - ev.RETRIEVAL_METHODS - ev.RANDOM_METHODS)
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    report = evaluate(
        hard_negative_rows=read_jsonl(args.hard_negative_jsonl),
        prior_rows=ev.read_jsonl(args.prior_extractions_jsonl),
        methods=args.methods,
        model=args.model,
        rerank_depth=args.rerank_depth,
        cache_dir=args.cache_dir,
    )
    report.update({
        "hard_negative_jsonl": args.hard_negative_jsonl,
        "prior_extractions_jsonl": args.prior_extractions_jsonl,
        "cache_dir": args.cache_dir,
    })
    write_json(args.output_json, report)
    write_markdown(args.output_markdown, markdown_report(report))
    print(json.dumps({
        "output_json": args.output_json,
        "output_markdown": args.output_markdown,
        "rows": report["rows"],
        "by_method": report["by_method"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
