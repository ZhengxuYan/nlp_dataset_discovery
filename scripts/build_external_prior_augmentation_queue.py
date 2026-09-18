#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


UNCLEAR_VALUES = {"", "unclear", "unknown", "none", "n/a", "na", "null"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm(text: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    return re.sub(r"\s+", " ", text)


def useful(text: Any) -> bool:
    value = norm(text)
    return bool(value) and value not in UNCLEAR_VALUES and len(value) > 1


def stable_id(key: str) -> str:
    return "externalprior:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def choose_title(name: str, cited_title: str | None) -> tuple[str, str]:
    if useful(cited_title):
        return str(cited_title).strip(), "cited_paper_title"
    return str(name).strip(), "dataset_or_resource_name"


def iter_query_bank_ids(rows: list[dict[str, Any]]) -> set[str]:
    ids = set()
    for row in rows:
        for key in ("query_bank_id", "bank_id"):
            if row.get(key):
                ids.add(str(row[key]))
    return ids


def mention_rows(dataset_row: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for mention in dataset_row.get("prior_dataset_mentions") or []:
        if not isinstance(mention, dict) or not useful(mention.get("name")):
            continue
        title, title_source = choose_title(str(mention.get("name") or ""), mention.get("cited_paper_title"))
        output.append({
            "matched_title": title,
            "title_source": title_source,
            "candidate_dataset_name": mention.get("name") or "",
            "relationship_type": mention.get("relationship_type") or "prior_dataset_mentions",
            "evidence": mention.get("evidence") or "",
            "matched_raw_reference": " ".join(
                str(x) for x in [
                    mention.get("cited_paper_title") or "",
                    mention.get("name") or "",
                    mention.get("evidence") or "",
                ] if x
            ),
        })
    for source in dataset_row.get("source_datasets") or []:
        if not isinstance(source, dict) or not useful(source.get("name")):
            continue
        title, title_source = choose_title(str(source.get("name") or ""), None)
        output.append({
            "matched_title": title,
            "title_source": title_source,
            "candidate_dataset_name": source.get("name") or "",
            "relationship_type": source.get("relationship") or "source_dataset",
            "evidence": source.get("evidence") or "",
            "matched_raw_reference": " ".join(
                str(x) for x in [
                    source.get("name") or "",
                    source.get("evidence") or "",
                ] if x
            ),
        })
    return output


def compact_query_mention(dataset_row: dict[str, Any], mention: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_id": dataset_row.get("bank_id"),
        "query_paper_id": dataset_row.get("paper_id"),
        "query_acl_id": dataset_row.get("acl_id"),
        "query_title": dataset_row.get("title"),
        "query_year": dataset_row.get("year"),
        "query_dataset_id": dataset_row.get("dataset_id"),
        "query_dataset_name": dataset_row.get("dataset_name"),
        "candidate_dataset_name": mention.get("candidate_dataset_name"),
        "candidate_paper_title": mention.get("matched_title") if mention.get("title_source") == "cited_paper_title" else "",
        "relationship_type": mention.get("relationship_type"),
        "evidence": mention.get("evidence"),
        "title_source": mention.get("title_source"),
    }


def build_queue(
    selected_rows: list[dict[str, Any]],
    dataset_bank_rows: list[dict[str, Any]],
    *,
    include_name_only: bool,
    max_refs_per_query: int,
) -> list[dict[str, Any]]:
    selected_ids = iter_query_bank_ids(selected_rows)
    dataset_by_bank_id = {
        str(row.get("bank_id")): row
        for row in dataset_bank_rows
        if row.get("bank_id")
    }
    grouped: dict[str, dict[str, Any]] = {}
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    for bank_id in sorted(selected_ids):
        dataset_row = dataset_by_bank_id.get(bank_id)
        if not dataset_row:
            continue
        mentions = mention_rows(dataset_row)
        if not include_name_only:
            mentions = [m for m in mentions if m.get("title_source") == "cited_paper_title"]
        for mention in mentions[:max_refs_per_query]:
            title = mention.get("matched_title") or ""
            if not useful(title):
                continue
            key = f"title:{norm(title)}"
            item = grouped.setdefault(key, {
                "prior_ref_id": stable_id(key),
                "dedupe_key": key,
                "matched_title": title,
                "matched_authors": [],
                "matched_year": "",
                "matched_url": "",
                "matched_doi": "",
                "matched_arxiv_id": "",
                "matched_raw_reference": mention.get("matched_raw_reference") or title,
                "reference_ids": [],
                "reference_match_confidences": {"external_augmentation": 1},
                "candidate_dataset_names": collections.Counter(),
                "query_dataset_mentions": [],
                "created_at": now,
            })
            if mention.get("candidate_dataset_name"):
                item["candidate_dataset_names"][mention["candidate_dataset_name"]] += 1
            item["query_dataset_mentions"].append(compact_query_mention(dataset_row, mention))

    output = []
    for item in grouped.values():
        item["candidate_dataset_names"] = dict(item["candidate_dataset_names"])
        item["n_query_dataset_mentions"] = len(item["query_dataset_mentions"])
        item["n_unique_query_papers"] = len({m.get("query_paper_id") for m in item["query_dataset_mentions"]})
        output.append(item)
    return sorted(output, key=lambda row: (-row["n_query_dataset_mentions"], row["matched_title"].lower()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external prior augmentation reference queue from selected query datasets.")
    parser.add_argument("--selected-query-jsonl", required=True, help="Attribution output or candidate queue JSONL containing query_bank_id.")
    parser.add_argument("--dataset-bank-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--include-name-only", action="store_true", help="Use dataset/resource names when cited paper titles are unavailable.")
    parser.add_argument("--max-refs-per-query", type=int, default=20)
    args = parser.parse_args()

    selected_rows = read_jsonl(args.selected_query_jsonl)
    dataset_bank_rows = read_jsonl(args.dataset_bank_jsonl)
    queue = build_queue(
        selected_rows,
        dataset_bank_rows,
        include_name_only=args.include_name_only,
        max_refs_per_query=args.max_refs_per_query,
    )
    write_jsonl(args.output_jsonl, queue)
    title_sources = collections.Counter()
    relationship_types = collections.Counter()
    for row in queue:
        for mention in row.get("query_dataset_mentions") or []:
            title_sources[mention.get("title_source") or "unknown"] += 1
            relationship_types[mention.get("relationship_type") or "unknown"] += 1
    summary = {
        "selected_query_rows": len(selected_rows),
        "selected_query_ids": len(iter_query_bank_ids(selected_rows)),
        "external_prior_refs": len(queue),
        "total_query_dataset_mentions": sum(row["n_query_dataset_mentions"] for row in queue),
        "title_sources": dict(title_sources),
        "relationship_types": dict(relationship_types.most_common()),
        "include_name_only": args.include_name_only,
        "max_refs_per_query": args.max_refs_per_query,
    }
    write_json(args.summary_json, summary)
    print(json.dumps({
        "output_jsonl": args.output_jsonl,
        "summary_json": args.summary_json,
        **summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
