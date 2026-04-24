import argparse
import json
import os
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv

from scv.analysis import AddedInformationAnalyzer, load_models
from scv.benchmarking import (
    CandidateRecord,
    HybridSupportRetriever,
    ORDINAL_TO_INT,
    evaluate_retrieval_run,
    ordinal_from_score,
    safe_mean,
    spearman_correlation,
    summarize_metric_runs,
)


def load_benchmark_rows(input_file: str) -> List[Dict]:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Benchmark file {input_file} not found.")
    rows: List[Dict] = []
    with open(input_file, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def candidate_from_dict(candidate: Dict) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate["candidate_id"],
        name=candidate.get("name", candidate["candidate_id"]),
        acus=list(candidate.get("acus", [])),
        summary_text=" ".join([
            candidate.get("name", ""),
            candidate.get("paper_title", ""),
            candidate.get("domain", ""),
            candidate.get("role", ""),
            candidate.get("source_dataset", ""),
            " ".join(candidate.get("acus", [])),
        ]),
        domain=candidate.get("domain", ""),
        role=candidate.get("role", ""),
        source_dataset=candidate.get("source_dataset", ""),
        is_cited=bool(candidate.get("is_cited", False)),
    )


def adapt_legacy_row(row: Dict, row_idx: int) -> Dict:
    """Allow the old synthetic benchmark rows to run through the new evaluator."""
    candidates = []
    for key, suffix in [("true_ancestor", "ancestor"), ("hard_negative", "hard_negative"), ("soft_negative", "soft_negative")]:
        if key not in row:
            continue
        item = row[key]
        candidates.append({
            "candidate_id": f"{row_idx}:{suffix}",
            "name": item["name"],
            "acus": item.get("acus", []),
            "domain": row.get("domain", ""),
            "role": "Benchmark Candidate",
            "source_dataset": "",
            "paper_title": "",
            "paper_id": "",
            "is_cited": key == "true_ancestor",
        })

    query = row.get("query_dataset", {})
    label = ""
    if "incremental_descendant" in row:
        label = "incremental"
    return {
        "example_id": f"legacy-{row_idx}",
        "query_dataset": {
            "name": query.get("name", ""),
            "acus": query.get("acus", []),
            "domain": row.get("domain", ""),
            "role": "Main Contribution",
            "source_dataset": "",
        },
        "gold_prior_support_ids": [f"{row_idx}:ancestor"] if "true_ancestor" in row else [],
        "gold_prior_support_acus": row.get("true_ancestor", {}).get("acus", []),
        "gold_added_information_label": label,
        "gold_added_information_rationale": "Legacy synthetic benchmark row adapted into the real-benchmark schema.",
        "candidates": candidates,
    }


def normalize_rows(rows: Iterable[Dict]) -> List[Dict]:
    normalized = []
    for idx, row in enumerate(rows):
        if "query_dataset" in row and "candidates" in row:
            normalized.append(row)
        else:
            normalized.append(adapt_legacy_row(row, idx))
    return normalized


def filter_candidates(row: Dict, pool_mode: str) -> List[Dict]:
    candidates = list(row.get("candidates", []))
    if pool_mode == "cited":
        cited = [candidate for candidate in candidates if candidate.get("is_cited")]
        return cited or candidates
    return candidates


def get_gold_support_acus(row: Dict, candidate_lookup: Dict[str, CandidateRecord]) -> List[str]:
    support_acus = list(row.get("gold_prior_support_acus", []))
    for support_id in row.get("gold_prior_support_ids", []):
        candidate = candidate_lookup.get(support_id)
        if candidate:
            support_acus.extend(candidate.acus)
    return list(dict.fromkeys([acu for acu in support_acus if acu]))


def ordinal_metrics(gold_labels: List[str], predicted_scores: List[float]) -> Dict[str, float]:
    predicted_labels = [ordinal_from_score(score) for score in predicted_scores]
    gold_numeric = [ORDINAL_TO_INT[label] for label in gold_labels]
    predicted_numeric = [ORDINAL_TO_INT[label] for label in predicted_labels]
    accuracy = safe_mean([1.0 if g == p else 0.0 for g, p in zip(gold_labels, predicted_labels)])
    mae = safe_mean([abs(g - p) for g, p in zip(gold_numeric, predicted_numeric)])
    rho = spearman_correlation(gold_numeric, predicted_numeric)
    return {
        "ordinal_accuracy": accuracy,
        "ordinal_mae": mae,
        "spearman": rho,
    }


def print_metric_block(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 42)
    print(title)
    print("=" * 42)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def run_retrieval_eval(rows: List[Dict], pool_mode: str, top_k: int) -> Dict[str, Dict[str, float]]:
    methods = ["dense", "lexical", "splade", "colbert", "rank_fusion", "fusion", "hybrid_rerank"]
    HybridSupportRetriever.reset_debug_counters()
    all_results: Dict[str, List[Dict[str, float]]] = {method: [] for method in methods}

    for row in rows:
        candidates = filter_candidates(row, pool_mode)
        if not candidates or not row.get("gold_prior_support_ids"):
            continue
        retriever = HybridSupportRetriever([candidate_from_dict(candidate) for candidate in candidates])
        query = row["query_dataset"]
        for method in methods:
            ranked = retriever.rank(
                query_name=query.get("name", ""),
                query_acus=query.get("acus", []),
                query_metadata=query,
                top_k=max(top_k, 5),
                method=method,
            )
            metrics = evaluate_retrieval_run(
                [candidate_id for candidate_id, _, _ in ranked],
                row.get("gold_prior_support_ids", []),
            )
            all_results[method].append(metrics)

    return {method: summarize_metric_runs(rows) for method, rows in all_results.items() if rows}


def print_retrieval_debug_info() -> None:
    counters = HybridSupportRetriever.get_debug_counters()
    print("\n" + "=" * 42)
    print("Retrieval Debug Counters")
    print("=" * 42)
    for key, value in counters.items():
        print(f"{key}: {value}")


def run_added_information_eval(
    rows: List[Dict],
    pool_mode: str,
    top_k: int,
    llm_model_name: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    load_models()
    analyzer_nli = AddedInformationAnalyzer()
    analyzer_llm = AddedInformationAnalyzer(llm_model_name=llm_model_name) if llm_model_name else None

    oracle_gold_labels: List[str] = []
    oracle_nli_scores: List[float] = []
    oracle_llm_scores: List[float] = []
    end_to_end_gold_labels: List[str] = []
    end_to_end_nli_scores: List[float] = []
    end_to_end_llm_scores: List[float] = []

    for row in rows:
        gold_label = row.get("gold_added_information_label")
        query = row.get("query_dataset", {})
        query_acus = query.get("acus", [])
        if not gold_label or not query_acus:
            continue

        candidates = filter_candidates(row, pool_mode)
        candidate_records = [candidate_from_dict(candidate) for candidate in candidates]
        candidate_lookup = {candidate.candidate_id: candidate for candidate in candidate_records}
        gold_support_acus = get_gold_support_acus(row, candidate_lookup)
        if gold_support_acus:
            oracle_gold_labels.append(gold_label)
            oracle_nli_scores.append(
                analyzer_nli.calculate_added_information_score_nli(query_acus, forced_context=gold_support_acus)
            )
            if analyzer_llm and analyzer_llm.llm_evaluator:
                oracle_llm_scores.append(
                    analyzer_llm.calculate_added_information_score_llm(query_acus, forced_context=gold_support_acus)
                )

        if candidate_records:
            retriever = HybridSupportRetriever(candidate_records)
            ranked = retriever.rank(
                query_name=query.get("name", ""),
                query_acus=query_acus,
                query_metadata=query,
                top_k=top_k,
                method="hybrid_rerank",
            )
            retrieved_support_acus: List[str] = []
            for candidate_id, _, _ in ranked:
                candidate = candidate_lookup.get(candidate_id)
                if candidate:
                    retrieved_support_acus.extend(candidate.acus)
            if retrieved_support_acus:
                end_to_end_gold_labels.append(gold_label)
                end_to_end_nli_scores.append(
                    analyzer_nli.calculate_added_information_score_nli(query_acus, forced_context=retrieved_support_acus)
                )
                if analyzer_llm and analyzer_llm.llm_evaluator:
                    end_to_end_llm_scores.append(
                        analyzer_llm.calculate_added_information_score_llm(query_acus, forced_context=retrieved_support_acus)
                    )

    results: Dict[str, Dict[str, float]] = {}
    if oracle_gold_labels and oracle_nli_scores:
        results["oracle_nli"] = ordinal_metrics(oracle_gold_labels, oracle_nli_scores)
    if oracle_gold_labels and oracle_llm_scores:
        results["oracle_llm"] = ordinal_metrics(oracle_gold_labels[: len(oracle_llm_scores)], oracle_llm_scores)
    if end_to_end_gold_labels and end_to_end_nli_scores:
        results["end_to_end_nli"] = ordinal_metrics(end_to_end_gold_labels, end_to_end_nli_scores)
    if end_to_end_gold_labels and end_to_end_llm_scores:
        results["end_to_end_llm"] = ordinal_metrics(end_to_end_gold_labels[: len(end_to_end_llm_scores)], end_to_end_llm_scores)
    return results


def print_error_analysis(rows: List[Dict], pool_mode: str) -> None:
    categories = {
        "missing_true_prior_from_pool": 0,
        "multiple_gold_support_sets": 0,
        "coarse_or_long_acus": 0,
    }
    for row in rows:
        candidates = filter_candidates(row, pool_mode)
        candidate_ids = {candidate.get("candidate_id") for candidate in candidates}
        gold_ids = set(row.get("gold_prior_support_ids", []))
        if gold_ids and not gold_ids.intersection(candidate_ids):
            categories["missing_true_prior_from_pool"] += 1
        if len(gold_ids) > 1:
            categories["multiple_gold_support_sets"] += 1
        query_acus = row.get("query_dataset", {}).get("acus", [])
        if any(len(acu.split()) > 25 for acu in query_acus):
            categories["coarse_or_long_acus"] += 1
    print_metric_block("Error Analysis Signals", {key: float(value) for key, value in categories.items()})


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run retrieval and added-information evaluation for dataset-support benchmarks.")
    parser.add_argument("--input", type=str, default="data/benchmark/real_added_information_benchmark_template.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool", choices=["all", "cited"], default="all")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-added-information", action="store_true")
    parser.add_argument("--llm-model", type=str, default="gpt-5.4")
    args = parser.parse_args()

    rows = normalize_rows(load_benchmark_rows(args.input))
    print(f"Loaded {len(rows)} benchmark rows from {args.input}")

    if not args.skip_retrieval:
        retrieval_results = run_retrieval_eval(rows, pool_mode=args.pool, top_k=args.top_k)
        for method, metrics in retrieval_results.items():
            print_metric_block(f"Retrieval: {method}", metrics)
        print_retrieval_debug_info()

    if not args.skip_added_information:
        llm_model = args.llm_model if os.environ.get("OPENAI_API_KEY") else None
        if not llm_model:
            print("\nOPENAI_API_KEY not found. Skipping LLM added-information evaluation.")
        added_info_results = run_added_information_eval(
            rows,
            pool_mode=args.pool,
            top_k=args.top_k,
            llm_model_name=llm_model,
        )
        for setup, metrics in added_info_results.items():
            print_metric_block(f"Added Information: {setup}", metrics)

    print_error_analysis(rows, pool_mode=args.pool)


if __name__ == "__main__":
    main()
