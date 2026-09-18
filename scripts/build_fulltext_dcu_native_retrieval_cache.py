#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_fulltext_prior_candidate_queue as queue_builder  # noqa: E402
import run_fulltext_dcu_native_attribution as native  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def existing_ids(path: str | Path) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    return {
        str(row.get("query_bank_id") or "")
        for row in read_jsonl(target)
        if row.get("query_bank_id")
    }


def shard_rows(rows: list[dict[str, Any]], *, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if num_shards == 1:
        return rows
    return [
        row
        for index, row in enumerate(rows)
        if index % num_shards == shard_index
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute DCU-native top-k prior DCU retrievals.")
    parser.add_argument("--dataset-bank-jsonl", required=True)
    parser.add_argument("--acu-bank-jsonl", required=True)
    parser.add_argument("--query-year", type=int, default=None)
    parser.add_argument("--year-mode", choices=["earlier", "earlier_or_same", "any"], default="earlier")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["first", "random", "stratified_year"], default="stratified_year")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--fast-dense-pool-size", type=int, default=1000, help="Use dense top-N as a prefilter before BM25 hybrid scoring. Set 0 for full hybrid.")
    parser.add_argument("--fast-sparse-pool-size", type=int, default=0, help="Optional full-BM25 lexical safety pool. Slower; default disabled.")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--hybrid-dcu-embedding-cache", default="data/census/integrated_fulltext_acu_bank_2023_2025_hybrid_dcu_minilm_embeddings.npz")
    parser.add_argument("--hybrid-dcu-dense-batch-size", type=int, default=128)
    parser.add_argument("--hybrid-dcu-max-text-chars", type=int, default=1600)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite:
        Path(args.output_jsonl).unlink(missing_ok=True)
        Path(args.summary_json).unlink(missing_ok=True)

    dataset_rows = native.load_query_rows(
        args.dataset_bank_jsonl,
        query_year=args.query_year,
        limit=args.limit,
        sample_mode=args.sample_mode,
    )
    rows = native.select_rows(dataset_rows, limit=args.limit, sample_mode=args.sample_mode, sample_seed=args.sample_seed)
    unsharded_rows = len(rows)
    rows = shard_rows(rows, num_shards=args.num_shards, shard_index=args.shard_index)
    done = existing_ids(args.output_jsonl)
    rows = [row for row in rows if str(row.get("bank_id") or "") not in done]
    acu_rows = read_jsonl(args.acu_bank_jsonl)
    all_dataset_rows = read_jsonl(args.dataset_bank_jsonl)
    dataset_by_bank_id = {
        str(row.get("bank_id") or ""): row
        for row in all_dataset_rows
        if row.get("bank_id")
    }
    hybrid_index = queue_builder.HybridDcuIndex(
        acu_rows,
        dataset_by_bank_id,
        embedding_cache=args.hybrid_dcu_embedding_cache,
        dense_batch_size=args.hybrid_dcu_dense_batch_size,
        max_text_chars=args.hybrid_dcu_max_text_chars,
    )
    print(json.dumps({
        "remaining_rows": len(rows),
        "unsharded_rows": unsharded_rows,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "acu_corpus_rows": len(acu_rows),
        "candidate_depth": args.candidate_depth,
        "fast_dense_pool_size": args.fast_dense_pool_size,
        "fast_sparse_pool_size": args.fast_sparse_pool_size,
        "year_mode": args.year_mode,
        "output_jsonl": args.output_jsonl,
    }, ensure_ascii=False, indent=2), flush=True)

    started = time.monotonic()
    processed = 0
    total_query_acus = 0
    total_prior_dcus = 0
    output_rows = read_jsonl(args.output_jsonl) if Path(args.output_jsonl).exists() else []
    for row in rows:
        q_acus = native.query_acus(row)
        retrievals = {}
        for query_acu in q_acus:
            candidates = native.retrieve_prior_dcus(
                query=row,
                query_acu=query_acu,
                acu_rows=acu_rows,
                dataset_by_bank_id=dataset_by_bank_id,
                hybrid_index=hybrid_index,
                year_mode=args.year_mode,
                candidate_depth=args.candidate_depth,
                min_score=args.min_score,
                fast_dense_pool_size=args.fast_dense_pool_size,
                fast_sparse_pool_size=args.fast_sparse_pool_size,
            )
            retrievals[query_acu["id"]] = {
                "method": "hybrid_dcu",
                "candidate_depth": args.candidate_depth,
                "fast_dense_pool_size": args.fast_dense_pool_size,
                "fast_sparse_pool_size": args.fast_sparse_pool_size,
                "prior_dcus": native.compact_prior_dcus(candidates),
            }
            total_prior_dcus += len(candidates)
        total_query_acus += len(q_acus)
        out = {
            "query_bank_id": row.get("bank_id") or "",
            "query_paper_id": row.get("paper_id") or "",
            "query_dataset_id": row.get("dataset_id") or "",
            "query_dataset_name": row.get("dataset_name") or "",
            "query_title": row.get("title") or "",
            "query_year": row.get("year"),
            "source_corpus": row.get("source_corpus") or "",
            "query_acus": q_acus,
            "retrieval": {
                "method": "hybrid_dcu",
                "year_mode": args.year_mode,
                "candidate_depth": args.candidate_depth,
                "min_score": args.min_score,
            },
            "query_acu_retrievals": retrievals,
        }
        append_jsonl(args.output_jsonl, out)
        output_rows.append(out)
        processed += 1
        elapsed = time.monotonic() - started
        print(json.dumps({
            "processed": processed,
            "total": len(rows),
            "query_acus": total_query_acus,
            "elapsed_seconds": round(elapsed, 1),
            "avg_seconds_per_row": round(elapsed / processed, 2),
        }, ensure_ascii=False), flush=True)

    summary = {
        "rows": len(output_rows),
        "new_rows": processed,
        "query_acus": sum(len(row.get("query_acus") or []) for row in output_rows),
        "mean_prior_dcus_per_query_acu": (
            sum(
                len((payload or {}).get("prior_dcus") or [])
                for row in output_rows
                for payload in (row.get("query_acu_retrievals") or {}).values()
            )
            / max(1, sum(len(row.get("query_acus") or []) for row in output_rows))
        ),
        "query_year": args.query_year,
        "year_mode": args.year_mode,
        "candidate_depth": args.candidate_depth,
        "fast_dense_pool_size": args.fast_dense_pool_size,
        "fast_sparse_pool_size": args.fast_sparse_pool_size,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "unsharded_rows": unsharded_rows,
        "dataset_bank_jsonl": args.dataset_bank_jsonl,
        "acu_bank_jsonl": args.acu_bank_jsonl,
        "output_jsonl": args.output_jsonl,
    }
    write_json(args.summary_json, summary)
    print(json.dumps({"summary_json": args.summary_json, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
