#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


class OfflineCacheMiss(RuntimeError):
    pass


class NoApiOpenAIClient:
    class Responses:
        @staticmethod
        def create(*args, **kwargs):
            raise OfflineCacheMiss("Refusing to call OpenAI API in cache-only hard-negative evaluation.")

    responses = Responses()


def load_eval_module():
    module_path = Path(__file__).with_name("evaluate_completed_benchmark_rows.py")
    spec = importlib.util.spec_from_file_location("evaluate_completed_benchmark_rows_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class CacheOnlyJsonCache:
    def __init__(self, base_cache, *, allow_api_cache_misses: bool = False):
        self.base_cache = base_cache
        self.allow_api_cache_misses = allow_api_cache_misses
        self.missing: Counter[str] = Counter()

    def path(self, namespace: str, key_payload):
        return self.base_cache.path(namespace, key_payload)

    def get(self, namespace: str, key_payload):
        value = self.base_cache.get(namespace, key_payload)
        if value is None and namespace.startswith("gpt_") and not self.allow_api_cache_misses:
            self.missing[namespace] += 1
            raise OfflineCacheMiss(f"Missing cached {namespace} entry; refusing to call API.")
        return value

    def set(self, namespace: str, key_payload, value):
        if namespace.startswith("gpt_") and not self.allow_api_cache_misses:
            raise OfflineCacheMiss(f"Refusing to write uncached {namespace} result in cache-only mode.")
        return self.base_cache.set(namespace, key_payload, value)

    def stats(self) -> dict:
        stats = self.base_cache.stats()
        stats["offline_cache_misses"] = dict(self.missing)
        return stats


def rank_lookup(ranking: Sequence[str]) -> Dict[str, int]:
    return {paper_id: index for index, paper_id in enumerate(ranking, start=1)}


def evaluate_rows(rows: Sequence[dict], rankings_by_method: Dict[str, Dict[str, List[str]]]) -> dict:
    by_method = {}
    by_type: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    examples: Dict[str, List[dict]] = defaultdict(list)

    for method, rankings in rankings_by_method.items():
        counts = Counter()
        for row in rows:
            query_id = row["query_paper_id"]
            ranking = rankings.get(query_id) or []
            lookup = rank_lookup(ranking)
            negative_id = row["candidate_paper_id"]
            gold_ids = [paper_id for paper_id in row.get("gold_prior_paper_ids") or [] if paper_id]
            negative_rank = lookup.get(negative_id)
            gold_ranks = [lookup[paper_id] for paper_id in gold_ids if paper_id in lookup]
            best_gold_rank = min(gold_ranks) if gold_ranks else None

            if best_gold_rank is None and negative_rank is None:
                outcome = "unresolved_both_missing"
            elif best_gold_rank is None:
                outcome = "negative_ahead"
            elif negative_rank is None:
                outcome = "gold_ahead"
            elif best_gold_rank < negative_rank:
                outcome = "gold_ahead"
            elif negative_rank < best_gold_rank:
                outcome = "negative_ahead"
            else:
                outcome = "tie"

            counts["n"] += 1
            counts[outcome] += 1
            if outcome != "unresolved_both_missing":
                counts["evaluable"] += 1
            if bool(row.get("ranked_before_gold")):
                counts["fusion_seed_ranked_before_gold"] += 1

            for negative_type in row.get("negative_types") or ["unknown"]:
                by_type[method][negative_type]["n"] += 1
                by_type[method][negative_type][outcome] += 1
                if outcome != "unresolved_both_missing":
                    by_type[method][negative_type]["evaluable"] += 1

            if outcome == "negative_ahead" and len(examples[method]) < 10:
                examples[method].append({
                    "query_paper_id": query_id,
                    "query_dataset_name": row.get("query_dataset_name"),
                    "candidate_paper_id": negative_id,
                    "candidate_title": row.get("candidate_title"),
                    "negative_rank": negative_rank,
                    "best_gold_rank": best_gold_rank,
                    "gold_prior_paper_ids": gold_ids,
                    "negative_types": row.get("negative_types") or [],
                })

        evaluable = counts["evaluable"]
        by_method[method] = {
            "n": counts["n"],
            "evaluable": evaluable,
            "gold_ahead": counts["gold_ahead"],
            "negative_ahead": counts["negative_ahead"],
            "tie": counts["tie"],
            "unresolved_both_missing": counts["unresolved_both_missing"],
            "hard_negative_rejection_rate": counts["gold_ahead"] / evaluable if evaluable else 0.0,
            "hard_negative_error_rate": counts["negative_ahead"] / evaluable if evaluable else 0.0,
            "fusion_seed_ranked_before_gold_rate": counts["fusion_seed_ranked_before_gold"] / counts["n"] if counts["n"] else 0.0,
        }

    by_type_summary = {}
    for method, type_counts in by_type.items():
        by_type_summary[method] = {}
        for negative_type, counts in type_counts.items():
            evaluable = counts["evaluable"]
            by_type_summary[method][negative_type] = {
                "n": counts["n"],
                "evaluable": evaluable,
                "hard_negative_rejection_rate": counts["gold_ahead"] / evaluable if evaluable else 0.0,
                "hard_negative_error_rate": counts["negative_ahead"] / evaluable if evaluable else 0.0,
            }

    return {
        "by_method": by_method,
        "by_negative_type": by_type_summary,
        "negative_ahead_examples": examples,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Hard-Negative Rejection",
        "",
        "## Overall",
        "",
        "| Method | N | Evaluable | Rejection Rate | Error Rate | Gold Ahead | Negative Ahead | Unresolved |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in report["by_method"].items():
        lines.append(
            "| {method} | {n} | {evaluable} | {rej:.3f} | {err:.3f} | {gold} | {neg} | {unres} |".format(
                method=method,
                n=row["n"],
                evaluable=row["evaluable"],
                rej=row["hard_negative_rejection_rate"],
                err=row["hard_negative_error_rate"],
                gold=row["gold_ahead"],
                neg=row["negative_ahead"],
                unres=row["unresolved_both_missing"],
            )
        )
    lines.extend(["", "## By Negative Type", ""])
    for method, type_rows in report.get("by_negative_type", {}).items():
        lines.extend([
            f"### {method}",
            "",
            "| Type | N | Evaluable | Rejection Rate | Error Rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for negative_type, row in sorted(type_rows.items()):
            lines.append(
                "| {typ} | {n} | {evaluable} | {rej:.3f} | {err:.3f} |".format(
                    typ=negative_type,
                    n=row["n"],
                    evaluable=row["evaluable"],
                    rej=row["hard_negative_rejection_rate"],
                    err=row["hard_negative_error_rate"],
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether retrieval methods rank gold prior papers above hard negatives.")
    parser.add_argument("--hard-negative-jsonl", required=True)
    parser.add_argument("--acl-benchmark-jsonl", required=True)
    parser.add_argument("--prior-extractions-jsonl", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--cache-dir", default="data/benchmark/retrieval_cache")
    parser.add_argument("--ranking-depth", type=int, default=20)
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--allow-api-cache-misses", action="store_true", help="Allow GPT cache misses to call the API. Default is cache-only.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    eval_module = load_eval_module()
    if not args.allow_api_cache_misses:
        eval_module.ensure_openai_client = lambda: NoApiOpenAIClient()
    hard_negative_rows = read_jsonl(args.hard_negative_jsonl)
    query_ids = sorted({row["query_paper_id"] for row in hard_negative_rows})

    all_drafts = eval_module.complete_acl_drafts(args.acl_benchmark_jsonl)
    drafts_by_id = {draft.query_paper_id: draft for draft in all_drafts}
    drafts = [drafts_by_id[query_id] for query_id in query_ids if query_id in drafts_by_id]
    missing_queries = [query_id for query_id in query_ids if query_id not in drafts_by_id]
    payloads = eval_module.fulltext_extraction_payloads(args.prior_extractions_jsonl)

    base_cache = eval_module.JsonCache(args.cache_dir)
    cache = CacheOnlyJsonCache(base_cache, allow_api_cache_misses=args.allow_api_cache_misses)
    ctx = eval_module.RetrievalContext(
        drafts=drafts,
        payloads=payloads,
        cited_by_query={},
        cache=cache,
        top_k=args.ranking_depth,
        allow_model_download=False,
        allow_fallback_methods=False,
        rerank_depth=args.rerank_depth,
        gpt_model=args.model,
    )

    rankings_by_method: Dict[str, Dict[str, List[str]]] = {}
    method_errors: Dict[str, List[dict]] = defaultdict(list)
    for method in args.methods:
        rankings_by_method[method] = {}
        try:
            retriever = eval_module.build_retriever(method, ctx)
        except Exception as exc:
            method_errors[method].append({"stage": "init", "error": str(exc)})
            continue
        for draft in drafts:
            try:
                rankings_by_method[method][draft.query_paper_id] = retriever.rank(draft, args.ranking_depth)
            except OfflineCacheMiss as exc:
                method_errors[method].append({
                    "stage": "rank",
                    "query_paper_id": draft.query_paper_id,
                    "error": str(exc),
                })
            except Exception as exc:
                method_errors[method].append({
                    "stage": "rank",
                    "query_paper_id": draft.query_paper_id,
                    "error": str(exc),
                })

    report = evaluate_rows(hard_negative_rows, rankings_by_method)
    report.update({
        "hard_negative_jsonl": args.hard_negative_jsonl,
        "rows": len(hard_negative_rows),
        "query_rows": len(query_ids),
        "evaluated_query_rows": len(drafts),
        "missing_queries": missing_queries,
        "methods": args.methods,
        "ranking_depth": args.ranking_depth,
        "rerank_depth": args.rerank_depth,
        "model": args.model,
        "cache_only": not args.allow_api_cache_misses,
        "cache_stats": cache.stats(),
        "method_errors": method_errors,
    })

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        output_markdown = Path(args.output_markdown)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "output_markdown": args.output_markdown,
        "rows": len(hard_negative_rows),
        "query_rows": len(query_ids),
        "cache_only": not args.allow_api_cache_misses,
        "method_errors": {method: len(errors) for method, errors in method_errors.items()},
        "by_method": report["by_method"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
