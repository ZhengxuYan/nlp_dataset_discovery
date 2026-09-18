#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()


def load_evaluator_module():
    script_path = Path(__file__).resolve().parent / "evaluate_completed_benchmark_rows.py"
    spec = importlib.util.spec_from_file_location("eval_completed_rows", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_completed_rows"] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def best_gold_rank(ranking: list[str], gold_ids: set[str]) -> int | None:
    for rank, paper_id in enumerate(ranking, start=1):
        if paper_id in gold_ids:
            return rank
    return None


def overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def classify_negative(ev, draft, candidate, candidate_dcus) -> tuple[list[str], dict[str, Any]]:
    query_dcus = ev.query_dcus_for_draft(draft)
    query_tokens = set(ev.tokenize(ev.query_text_for_draft(draft)))
    candidate_tokens = set(ev.tokenize(candidate.summary_text))
    query_types = {dcu.acu_type for dcu in query_dcus if dcu.acu_type}
    candidate_types = {dcu.acu_type for dcu in candidate_dcus if dcu.acu_type}
    query_tasks = set()
    query_domains = set()
    query_languages = set()
    query_modalities = set()
    for dcu in query_dcus:
        query_tasks.update(ev.tokenize(" ".join(dcu.tasks)))
        query_domains.update(ev.tokenize(" ".join(dcu.domains)))
        query_languages.update(ev.tokenize(" ".join(dcu.languages)))
        query_modalities.update(ev.tokenize(" ".join(dcu.modalities)))
    candidate_tasks = set()
    candidate_domains = set()
    candidate_languages = set()
    candidate_modalities = set()
    candidate_sources = set()
    for dcu in candidate_dcus:
        candidate_tasks.update(ev.tokenize(" ".join(dcu.tasks)))
        candidate_domains.update(ev.tokenize(" ".join(dcu.domains)))
        candidate_languages.update(ev.tokenize(" ".join(dcu.languages)))
        candidate_modalities.update(ev.tokenize(" ".join(dcu.modalities)))
        candidate_sources.update(ev.tokenize(" ".join(dcu.source_datasets)))

    query_source_terms = set()
    for acu in draft.query_acus:
        query_source_terms.update(ev.tokenize(acu))

    features = {
        "lexical_overlap": overlap(query_tokens, candidate_tokens),
        "acu_type_overlap": overlap(query_types, candidate_types),
        "task_overlap": overlap(query_tasks, candidate_tasks),
        "domain_overlap": overlap(query_domains, candidate_domains),
        "language_overlap": overlap(query_languages, candidate_languages),
        "modality_overlap": overlap(query_modalities, candidate_modalities),
        "source_term_overlap": overlap(query_source_terms, candidate_sources),
    }
    labels: list[str] = []
    if features["task_overlap"] >= 0.10 or features["domain_overlap"] >= 0.10:
        labels.append("same_task_or_domain")
    if features["language_overlap"] >= 0.10 or features["modality_overlap"] >= 0.10:
        labels.append("same_language_or_modality")
    if features["source_term_overlap"] >= 0.05:
        labels.append("source_or_lineage_overlap")
    if features["acu_type_overlap"] >= 0.20:
        labels.append("same_contribution_type")
    if features["lexical_overlap"] >= 0.08:
        labels.append("high_lexical_overlap")
    if not labels:
        labels.append("surface_similar_non_gold")
    return labels, features


def top_candidate_dcus(ev, draft, paper_id: str, payload: dict[str, Any], max_dcus: int) -> list[dict[str, Any]]:
    query_dcus = ev.query_dcus_for_draft(draft)
    scored = []
    for dcu in ev.prior_dcus_for_paper(paper_id, payload):
        score = max((ev.routed_dcu_pair_score(query, dcu) for query in query_dcus), default=0.0)
        scored.append((score, dcu))
    rows = []
    for score, dcu in sorted(scored, key=lambda item: item[0], reverse=True)[:max_dcus]:
        rows.append({
            "dcu_id": dcu.dcu_id,
            "score": round(float(score), 4),
            "dataset_name": dcu.dataset_name,
            "acu_type": dcu.acu_type,
            "text": dcu.text,
            "tasks": list(dcu.tasks),
            "domains": list(dcu.domains),
            "languages": list(dcu.languages),
            "modalities": list(dcu.modalities),
            "source_datasets": list(dcu.source_datasets),
        })
    return rows


def build_hard_negatives(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ev = load_evaluator_module()
    drafts = ev.complete_acl_drafts(args.acl_benchmark_jsonl)
    if args.limit is not None:
        drafts = drafts[: args.limit]
    payloads = ev.fulltext_extraction_payloads(args.prior_extractions_jsonl)
    ctx = ev.RetrievalContext(
        drafts=drafts,
        payloads=payloads,
        cited_by_query={},
        cache=ev.JsonCache(args.cache_dir),
        top_k=args.top_k,
        allow_model_download=args.allow_model_download,
        allow_fallback_methods=args.allow_fallback_methods,
        rerank_depth=args.rerank_depth,
        gpt_model=args.model,
    )
    candidates_by_id = {
        candidate.candidate_id: candidate
        for candidate in [ev.processed_to_candidate(pid, payload) for pid, payload in payloads.items()]
    }

    output_rows: list[dict[str, Any]] = []
    skipped = Counter()
    method_counts = Counter()
    label_counts = Counter()
    before_gold_counts = Counter()
    for method in args.methods:
        retriever = ev.build_retriever(method, ctx)
        for draft in drafts:
            gold_ids = set(draft.gold_prior_paper_ids)
            ranking = retriever.rank(draft, max(args.top_k, args.ranking_depth))
            gold_rank = best_gold_rank(ranking, gold_ids)
            negatives_added = 0
            for rank, paper_id in enumerate(ranking[: args.ranking_depth], start=1):
                if paper_id in gold_ids:
                    continue
                candidate = candidates_by_id.get(paper_id)
                payload = payloads.get(paper_id) or {}
                if not candidate or not payload:
                    skipped["missing_candidate_or_payload"] += 1
                    continue
                candidate_dcus = ev.prior_dcus_for_paper(paper_id, payload)
                negative_types, features = classify_negative(ev, draft, candidate, candidate_dcus)
                ranked_before_gold = gold_rank is None or rank < gold_rank
                if args.only_before_gold and not ranked_before_gold:
                    continue
                hard_score = (
                    1.0 * ranked_before_gold
                    + 0.5 * features["lexical_overlap"]
                    + 0.4 * features["acu_type_overlap"]
                    + 0.3 * max(features["task_overlap"], features["domain_overlap"])
                    + 0.2 * max(features["language_overlap"], features["modality_overlap"])
                )
                output_rows.append({
                    "query_paper_id": draft.query_paper_id,
                    "query_dataset_name": draft.query_dataset_name,
                    "query_acus": [
                        {"id": dcu.dcu_id, "type": dcu.acu_type, "text": dcu.text}
                        for dcu in ev.query_dcus_for_draft(draft)
                    ],
                    "method": method,
                    "candidate_rank": rank,
                    "gold_best_rank": gold_rank,
                    "ranked_before_gold": ranked_before_gold,
                    "hard_negative_score": round(float(hard_score), 4),
                    "negative_types": negative_types,
                    "features": {key: round(float(value), 4) for key, value in features.items()},
                    "candidate_paper_id": paper_id,
                    "candidate_title": payload.get("title") or candidate.name,
                    "candidate_year": payload.get("year"),
                    "candidate_summary": candidate.summary_text[:1000],
                    "candidate_top_dcus": top_candidate_dcus(ev, draft, paper_id, payload, args.max_candidate_dcus),
                    "gold_prior_paper_ids": list(gold_ids),
                    "human_label": None,
                    "human_notes": "",
                })
                negatives_added += 1
                method_counts[method] += 1
                before_gold_counts[method] += int(ranked_before_gold)
                label_counts.update(negative_types)
                if negatives_added >= args.negatives_per_query_method:
                    break
    output_rows.sort(
        key=lambda row: (
            row["method"],
            row["query_paper_id"],
            -float(row["hard_negative_score"]),
            int(row["candidate_rank"]),
        )
    )
    summary = {
        "rows": len(output_rows),
        "query_rows": len(drafts),
        "methods": args.methods,
        "ranking_depth": args.ranking_depth,
        "negatives_per_query_method": args.negatives_per_query_method,
        "only_before_gold": args.only_before_gold,
        "method_counts": dict(method_counts),
        "ranked_before_gold_counts": dict(before_gold_counts),
        "negative_type_counts": dict(label_counts),
        "skipped": dict(skipped),
    }
    return output_rows, summary


def write_markdown(path: str | Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Hard-Negative Prior-Support Benchmark",
        "",
        f"- Rows: {summary['rows']}",
        f"- Query rows: {summary['query_rows']}",
        f"- Methods: {', '.join(summary['methods'])}",
        f"- Ranking depth: {summary['ranking_depth']}",
        f"- Only before gold: {summary['only_before_gold']}",
        "",
        "## Method Counts",
        "",
        "| Method | Hard negatives | Ranked before gold |",
        "| --- | ---: | ---: |",
    ]
    for method, count in summary["method_counts"].items():
        lines.append(f"| {method} | {count} | {summary['ranked_before_gold_counts'].get(method, 0)} |")
    lines.extend(["", "## Negative Types", "", "| Type | Count |", "| --- | ---: |"])
    for label, count in sorted(summary["negative_type_counts"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Examples", ""])
    for row in rows[:20]:
        lines.extend([
            f"### {row['query_dataset_name']} / {row['method']} rank {row['candidate_rank']}",
            "",
            f"- Candidate: `{row['candidate_paper_id']}` {row['candidate_title']}",
            f"- Types: {', '.join(row['negative_types'])}",
            f"- Ranked before gold: {row['ranked_before_gold']} (gold best rank: {row['gold_best_rank']})",
            f"- Top DCU: {row['candidate_top_dcus'][0]['text'] if row['candidate_top_dcus'] else 'none'}",
            "",
        ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard-negative examples for dataset prior-support retrieval.")
    parser.add_argument("--acl-benchmark-jsonl", required=True)
    parser.add_argument("--prior-extractions-jsonl", required=True)
    parser.add_argument("--methods", nargs="+", default=["fusion", "gpt_5_4_listwise_rerank"])
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--ranking-depth", type=int, default=20)
    parser.add_argument("--negatives-per-query-method", type=int, default=3)
    parser.add_argument("--max-candidate-dcus", type=int, default=3)
    parser.add_argument("--only-before-gold", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dir", default="data/benchmark/retrieval_cache")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--allow-fallback-methods", action="store_true")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    rows, summary = build_hard_negatives(args)
    write_jsonl(args.output_jsonl, rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        write_markdown(args.output_markdown, summary, rows)
    print(json.dumps({
        "output_jsonl": args.output_jsonl,
        "summary_json": args.summary_json,
        "output_markdown": args.output_markdown,
        **summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
