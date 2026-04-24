#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "--allow-model-download" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from scv.benchmark_models import BenchmarkDraftRecord
from scv.benchmark_store import (
    load_benchmark_drafts,
    load_previous_work_candidates,
    load_processed_bank,
)
from scv.benchmarking import (
    CandidateRecord,
    HybridSupportRetriever,
    ORDINAL_LABELS,
    ORDINAL_TO_INT,
    evaluate_retrieval_run,
    ordinal_from_score,
    safe_mean,
    spearman_correlation,
    summarize_metric_runs,
    tokenize,
)


try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

try:
    from bespokelabs import curator
except ImportError:
    curator = None  # type: ignore


DEFAULT_DRAFTS = "data/benchmark/benchmark_drafts.jsonl"
DEFAULT_PROCESSED_BANK = "data/benchmark/processed_bank.jsonl"
DEFAULT_PREVIOUS_WORK = "data/benchmark/previous_work_candidates.jsonl"


class AddedInformationLabelOutput(BaseModel):
    label: Literal["repackaging", "incremental", "substantial"] = Field(
        description="Ordinal added-information label."
    )
    score: float = Field(
        description="Continuous score from 0.0 fully supported/repackaging to 1.0 substantial added information."
    )
    rationale: str = Field(description="Brief rationale grounded in the supplied ACUs.")


ADDED_INFORMATION_PROMPT = """You are evaluating how much added information a query dataset contributes relative to prior-support ACUs.

Labels:
- repackaging: mostly reformats, filters, or lightly repackages the prior support.
- incremental: adds a modest but real extension, annotation layer, scale/domain/language/task variation, or combination.
- substantial: introduces a clearly new construction, capability, task setting, modality, annotation structure, or large new resource.

Query dataset: {query_dataset_name}

Query ACUs:
{query_acus}

Prior-support ACUs:
{prior_acus}

Return structured output only.
"""


if curator is not None:
    class AddedInformationLabeler(curator.LLM):
        response_format = AddedInformationLabelOutput

        def prompt(self, input: dict) -> str:
            return ADDED_INFORMATION_PROMPT.format(
                query_dataset_name=input["query_dataset_name"],
                query_acus="\n".join(f"- {acu}" for acu in input.get("query_acus", [])) or "- None",
                prior_acus="\n".join(f"- {acu}" for acu in input.get("prior_acus", [])) or "- None",
            )
else:
    AddedInformationLabeler = None  # type: ignore


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def complete_drafts(drafts_path: str) -> List[BenchmarkDraftRecord]:
    return [
        draft for draft in load_benchmark_drafts(Path(drafts_path))
        if draft.draft_status == "complete"
        and draft.gold_prior_paper_ids
        and draft.gold_added_information_label
    ]


def processed_payloads(processed_bank_path: str) -> Dict[str, dict]:
    payloads: Dict[str, dict] = {}
    for row in load_processed_bank(Path(processed_bank_path)):
        if row.processing_status != "processed":
            continue
        path = Path(row.processed_json_path)
        if not path.exists():
            continue
        try:
            payloads[row.paper_id] = load_json(path)
        except Exception:
            continue
    return payloads


def processed_to_candidate(paper_id: str, payload: dict, *, is_cited: bool = False) -> CandidateRecord:
    dataset_names: List[str] = []
    acus: List[str] = []
    summary_parts: List[str] = []
    domains: List[str] = []
    roles: List[str] = []
    sources: List[str] = []

    for dataset in payload.get("datasets") or []:
        name = dataset.get("name") or ""
        if name:
            dataset_names.append(name)
        acus.extend([acu for acu in dataset.get("acus") or [] if acu])
        for key in ("usage_description", "added_information_summary", "novelty_summary"):
            if dataset.get(key):
                summary_parts.append(str(dataset[key]))
        if dataset.get("domain"):
            domains.append(str(dataset["domain"]))
        if dataset.get("role"):
            roles.append(str(dataset["role"]))
        if dataset.get("source_dataset"):
            sources.append(str(dataset["source_dataset"]))

    title = payload.get("title") or paper_id
    summary_text = " ".join([
        title,
        payload.get("contribution_summary") or payload.get("paper_contribution_summary") or "",
        " ".join(dataset_names),
        " ".join(acus),
        " ".join(summary_parts),
    ])
    return CandidateRecord(
        candidate_id=paper_id,
        name="; ".join(dataset_names) or title,
        acus=unique(acus),
        summary_text=summary_text,
        domain=domains[0] if domains else "",
        role=roles[0] if roles else "",
        source_dataset=sources[0] if sources else "",
        is_cited=is_cited,
    )


def collect_acus_for_papers(paper_ids: Sequence[str], payloads: Dict[str, dict], max_acus: int = 40) -> List[str]:
    acus: List[str] = []
    for paper_id in paper_ids:
        payload = payloads.get(paper_id) or {}
        for dataset in payload.get("datasets") or []:
            acus.extend([acu for acu in dataset.get("acus") or [] if acu])
    return unique(acus)[:max_acus]


def candidate_papers_by_query(previous_work_path: str) -> Dict[str, set]:
    by_query: Dict[str, set] = defaultdict(set)
    for candidate in load_previous_work_candidates(Path(previous_work_path)):
        if candidate.resolved_paper_id:
            by_query[candidate.query_paper_id].add(candidate.resolved_paper_id)
    return by_query


def lexical_rank(
    query_name: str,
    query_acus: Sequence[str],
    candidates: Sequence[CandidateRecord],
    top_k: int,
) -> List[str]:
    if BM25Okapi is None:
        return []
    corpus_tokens = [tokenize(candidate.summary_text) for candidate in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    query_text = " ".join([query_name, *query_acus])
    scores = bm25.get_scores(tokenize(query_text))
    return [
        candidate.candidate_id
        for candidate, _ in sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[:top_k]
    ]


def run_retrieval_eval(
    drafts: Sequence[BenchmarkDraftRecord],
    payloads: Dict[str, dict],
    cited_by_query: Dict[str, set],
    methods: Sequence[str],
    top_k: int,
) -> tuple[Dict[str, dict], Dict[str, List[dict]], Dict[str, List[str]]]:
    metric_rows: Dict[str, List[Dict[str, float]]] = {method: [] for method in methods}
    misses: Dict[str, List[dict]] = {method: [] for method in methods}
    skipped: Dict[str, List[str]] = defaultdict(list)

    for draft in drafts:
        gold = [paper_id for paper_id in draft.gold_prior_paper_ids if paper_id in payloads]
        if not gold:
            skipped["gold_not_processed"].append(draft.query_paper_id)
            continue

        candidates = [
            processed_to_candidate(
                paper_id,
                payload,
                is_cited=paper_id in cited_by_query.get(draft.query_paper_id, set()),
            )
            for paper_id, payload in payloads.items()
            if paper_id != draft.query_paper_id
        ]

        retriever = None
        for method in methods:
            try:
                if method == "lexical":
                    ranked_ids = lexical_rank(draft.query_dataset_name, draft.query_acus, candidates, top_k)
                else:
                    if retriever is None:
                        retriever = HybridSupportRetriever(candidates)
                    ranked_ids = [
                        candidate_id
                        for candidate_id, _, _ in retriever.rank(
                            draft.query_dataset_name,
                            draft.query_acus,
                            top_k=top_k,
                            method=method,
                        )
                    ]
                metrics = evaluate_retrieval_run(ranked_ids, gold)
                if top_k >= 10:
                    metrics["recall@10"] = 1.0 if any(candidate_id in gold for candidate_id in ranked_ids[:10]) else 0.0
                metric_rows[method].append(metrics)
                if metrics.get("recall@5", 0.0) == 0.0 and len(misses[method]) < 10:
                    misses[method].append({
                        "query_paper_id": draft.query_paper_id,
                        "query_dataset_name": draft.query_dataset_name,
                        "gold_prior_paper_ids": gold,
                        "top_retrieved": ranked_ids[:5],
                    })
            except Exception as exc:
                skipped[f"{method}_failed"].append(f"{draft.query_paper_id}: {exc}")
                break

    summaries = {
        method: summarize_metric_runs(rows)
        for method, rows in metric_rows.items()
        if rows
    }
    return summaries, misses, skipped


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    f1s = []
    for label in ORDINAL_LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return safe_mean(f1s)


def added_information_summary(gold: Sequence[str], pred: Sequence[str]) -> dict:
    gold_int = [ORDINAL_TO_INT[label] for label in gold]
    pred_int = [ORDINAL_TO_INT[label] for label in pred]
    confusion = {
        gold_label: {pred_label: 0 for pred_label in ORDINAL_LABELS}
        for gold_label in ORDINAL_LABELS
    }
    for gold_label, pred_label in zip(gold, pred):
        confusion[gold_label][pred_label] += 1
    return {
        "n": len(gold),
        "accuracy": safe_mean([1.0 if g == p else 0.0 for g, p in zip(gold, pred)]),
        "macro_f1": macro_f1(gold, pred),
        "spearman": spearman_correlation(gold_int, pred_int),
        "confusion": confusion,
    }


def heuristic_added_information_label(query_acus: Sequence[str], prior_acus: Sequence[str]) -> str:
    if not query_acus:
        return "repackaging"
    prior_tokens = set(token for acu in prior_acus for token in tokenize(acu))
    if not prior_tokens:
        return "substantial"
    unsupported = []
    for acu in query_acus:
        tokens = set(tokenize(acu))
        overlap = len(tokens & prior_tokens) / max(len(tokens), 1)
        unsupported.append(1.0 - overlap)
    return ordinal_from_score(safe_mean(unsupported))


def parse_curator_row(row) -> AddedInformationLabelOutput:
    if hasattr(row, "label"):
        return row
    if isinstance(row, dict) and "parsed_response_message" in row:
        parsed = row["parsed_response_message"]
        if isinstance(parsed, AddedInformationLabelOutput):
            return parsed
        return AddedInformationLabelOutput(**parsed)
    if isinstance(row, dict):
        return AddedInformationLabelOutput(**row)
    raise TypeError(f"Unsupported LLM response row: {type(row)}")


def run_added_information_eval(
    drafts: Sequence[BenchmarkDraftRecord],
    payloads: Dict[str, dict],
    retrieval_rankings: Dict[str, List[str]],
    *,
    mode: str,
    method: str,
    model_name: str,
    top_k: int,
) -> dict:
    examples = []
    gold_labels = []
    pred_labels = []

    for draft in drafts:
        if mode == "oracle":
            prior_acus = list(draft.gold_prior_support_acus)
        else:
            prior_acus = collect_acus_for_papers(retrieval_rankings.get(draft.query_paper_id, [])[:top_k], payloads)
        if not prior_acus:
            continue
        examples.append({
            "query_paper_id": draft.query_paper_id,
            "query_dataset_name": draft.query_dataset_name,
            "query_acus": list(draft.query_acus),
            "prior_acus": prior_acus,
            "gold_label": draft.gold_added_information_label,
        })

    if not examples:
        return {"n": 0, "error": "No examples with prior ACUs."}

    gold_labels = [example["gold_label"] for example in examples]
    if method == "heuristic":
        pred_labels = [
            heuristic_added_information_label(example["query_acus"], example["prior_acus"])
            for example in examples
        ]
    elif method == "llm":
        if curator is None or AddedInformationLabeler is None:
            return {"n": len(examples), "error": "bespokelabs.curator is unavailable."}
        labeler = AddedInformationLabeler(model_name=model_name)
        responses = labeler(examples)
        pred_labels = [parse_curator_row(row).label for row in responses.dataset]
    else:
        raise ValueError(f"Unknown added-information method: {method}")

    result = added_information_summary(gold_labels, pred_labels)
    result["examples"] = [
        {
            "query_paper_id": example["query_paper_id"],
            "gold": gold,
            "pred": pred,
        }
        for example, gold, pred in zip(examples[:10], gold_labels[:10], pred_labels[:10])
    ]
    return result


def build_rankings_for_method(
    drafts: Sequence[BenchmarkDraftRecord],
    payloads: Dict[str, dict],
    cited_by_query: Dict[str, set],
    method: str,
    top_k: int,
) -> Dict[str, List[str]]:
    rankings = {}
    for draft in drafts:
        candidates = [
            processed_to_candidate(
                paper_id,
                payload,
                is_cited=paper_id in cited_by_query.get(draft.query_paper_id, set()),
            )
            for paper_id, payload in payloads.items()
            if paper_id != draft.query_paper_id
        ]
        if method == "lexical":
            ranked = lexical_rank(draft.query_dataset_name, draft.query_acus, candidates, top_k)
        else:
            retriever = HybridSupportRetriever(candidates)
            ranked = [
                candidate_id
                for candidate_id, _, _ in retriever.rank(
                    draft.query_dataset_name,
                    draft.query_acus,
                    top_k=top_k,
                    method=method,
                )
            ]
        rankings[draft.query_paper_id] = ranked
    return rankings


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate complete benchmark rows for retrieval and added-information estimation.")
    parser.add_argument("--drafts", default=DEFAULT_DRAFTS)
    parser.add_argument("--processed-bank", default=DEFAULT_PROCESSED_BANK)
    parser.add_argument("--previous-work", default=DEFAULT_PREVIOUS_WORK)
    parser.add_argument("--model", default=os.environ.get("BENCHMARK_BUILDER_MODEL", "gpt-5.4"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=["lexical", "dense", "fusion", "rank_fusion", "hybrid_rerank"])
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--added-info-method", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--end-to-end-retrieval-method", default="hybrid_rerank")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow sentence-transformers/HuggingFace to download missing dense models.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if not args.allow_model_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    drafts = complete_drafts(args.drafts)
    if not drafts:
        raise SystemExit("No complete benchmark rows found.")
    payloads = processed_payloads(args.processed_bank)
    cited_by_query = candidate_papers_by_query(args.previous_work)

    retrieval, misses, skipped = run_retrieval_eval(
        drafts,
        payloads,
        cited_by_query,
        methods=args.methods,
        top_k=args.top_k,
    )

    report = {
        "row_counts": {
            "complete": len(drafts),
            "processed_corpus": len(payloads),
        },
        "label_distribution": dict(Counter(draft.gold_added_information_label for draft in drafts)),
        "retrieval": retrieval,
        "miss_examples": misses,
        "skipped": dict(skipped),
        "debug_counters": HybridSupportRetriever.get_debug_counters(),
    }

    if not args.retrieval_only:
        report["added_information"] = {}
        report["added_information"]["oracle"] = run_added_information_eval(
            drafts,
            payloads,
            {},
            mode="oracle",
            method=args.added_info_method,
            model_name=args.model,
            top_k=args.top_k,
        )
        try:
            rankings = build_rankings_for_method(
                drafts,
                payloads,
                cited_by_query,
                args.end_to_end_retrieval_method,
                args.top_k,
            )
        except Exception as exc:
            if args.end_to_end_retrieval_method != "lexical":
                try:
                    rankings = build_rankings_for_method(
                        drafts,
                        payloads,
                        cited_by_query,
                        "lexical",
                        args.top_k,
                    )
                except Exception as lexical_exc:
                    report["added_information"]["end_to_end"] = {
                        "error": str(exc),
                        "lexical_fallback_error": str(lexical_exc),
                    }
                else:
                    report["added_information"]["end_to_end"] = run_added_information_eval(
                        drafts,
                        payloads,
                        rankings,
                        mode="end_to_end",
                        method=args.added_info_method,
                        model_name=args.model,
                        top_k=args.top_k,
                    )
                    report["added_information"]["end_to_end"]["retrieval_method"] = "lexical_fallback"
                    report["added_information"]["end_to_end"]["fallback_reason"] = str(exc)
            else:
                report["added_information"]["end_to_end"] = {"error": str(exc)}
        else:
            report["added_information"]["end_to_end"] = run_added_information_eval(
                drafts,
                payloads,
                rankings,
                mode="end_to_end",
                method=args.added_info_method,
                model_name=args.model,
                top_k=args.top_k,
            )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
