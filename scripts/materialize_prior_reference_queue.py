#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def extract_arxiv_id(text: str) -> str:
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"\barxiv[:\s]+([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1)
    return ""


def extract_doi(text: str) -> str:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
    return match.group(0).rstrip(".,);") if match else ""


def extract_url(text: str) -> str:
    match = re.search(r"https?://[^\s)>\"]+", text)
    return match.group(0).rstrip(".,);") if match else ""


def dedupe_key(match: dict[str, Any]) -> str:
    raw = str(match.get("matched_raw_reference") or "")
    arxiv_id = str(match.get("matched_arxiv_id") or extract_arxiv_id(raw)).lower()
    doi = str(match.get("matched_doi") or extract_doi(raw)).lower()
    url = str(match.get("matched_url") or extract_url(raw)).lower()
    title = normalize_text(match.get("matched_title") or raw)
    year = str(match.get("matched_year") or "")
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if doi:
        return f"doi:{doi}"
    if url and "aclanthology.org" in url:
        return f"url:{url}"
    return f"title:{title}|year:{year}"


def stable_id(key: str) -> str:
    return "priorref:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def compact_mention(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    match = candidate.get("reference_match") or {}
    return {
        "benchmark_id": row.get("benchmark_id"),
        "query_paper_id": row.get("query_paper_id"),
        "query_acl_id": row.get("query_acl_id"),
        "query_title": row.get("query_title"),
        "query_year": row.get("query_year"),
        "query_dataset_id": row.get("query_dataset_id"),
        "query_dataset_name": row.get("query_dataset_name"),
        "candidate_dataset_name": candidate.get("dataset_name"),
        "candidate_paper_title": candidate.get("paper_title"),
        "relationship_type": candidate.get("relationship_type"),
        "second_pass_confidence": candidate.get("second_pass_confidence"),
        "reference_match_confidence": match.get("confidence"),
        "reference_match_rationale": match.get("rationale"),
    }


def build_queue(rows: list[dict[str, Any]], *, min_confidence: str) -> list[dict[str, Any]]:
    min_rank = CONFIDENCE_ORDER[min_confidence]
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        for candidate in row.get("reference_matched_candidates") or []:
            match = candidate.get("reference_match") or {}
            if match.get("match_label") != "matched_reference":
                continue
            confidence = str(match.get("confidence") or "low").lower()
            if CONFIDENCE_ORDER.get(confidence, -1) < min_rank:
                continue

            raw = str(match.get("matched_raw_reference") or "")
            key = dedupe_key(match)
            item = grouped.setdefault(
                key,
                {
                    "prior_ref_id": stable_id(key),
                    "dedupe_key": key,
                    "matched_title": match.get("matched_title") or "",
                    "matched_authors": match.get("matched_authors") or [],
                    "matched_year": match.get("matched_year") or "",
                    "matched_url": match.get("matched_url") or extract_url(raw),
                    "matched_doi": match.get("matched_doi") or extract_doi(raw),
                    "matched_arxiv_id": match.get("matched_arxiv_id") or extract_arxiv_id(raw),
                    "matched_raw_reference": raw,
                    "reference_ids": [],
                    "reference_match_confidences": collections.Counter(),
                    "candidate_dataset_names": collections.Counter(),
                    "query_dataset_mentions": [],
                    "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
            )
            ref_id = match.get("reference_id")
            if ref_id and ref_id not in item["reference_ids"]:
                item["reference_ids"].append(ref_id)
            item["reference_match_confidences"][confidence] += 1
            if candidate.get("dataset_name"):
                item["candidate_dataset_names"][candidate.get("dataset_name")] += 1
            item["query_dataset_mentions"].append(compact_mention(row, candidate))

    output = []
    for item in grouped.values():
        item["reference_match_confidences"] = dict(item["reference_match_confidences"])
        item["candidate_dataset_names"] = dict(item["candidate_dataset_names"])
        item["n_query_dataset_mentions"] = len(item["query_dataset_mentions"])
        item["n_unique_query_papers"] = len({m.get("query_paper_id") for m in item["query_dataset_mentions"]})
        output.append(item)
    return sorted(output, key=lambda x: (-x["n_query_dataset_mentions"], str(x.get("matched_year") or ""), x["matched_title"]))


def write_summary(path: str | Path, rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> None:
    id_counts = collections.Counter()
    years = collections.Counter()
    for row in rows:
        if row.get("matched_arxiv_id"):
            id_counts["arxiv_id"] += 1
        if row.get("matched_doi"):
            id_counts["doi"] += 1
        if row.get("matched_url"):
            id_counts["url"] += 1
        if row.get("matched_year"):
            years[str(row.get("matched_year"))] += 1
    summary = {
        "source_rows": len(source_rows),
        "prior_references": len(rows),
        "total_query_dataset_mentions": sum(r["n_query_dataset_mentions"] for r in rows),
        "identifier_coverage": dict(id_counts),
        "top_years": dict(years.most_common(20)),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate citation-grounded prior references into a fetch/process queue.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--min-confidence", choices=sorted(CONFIDENCE_ORDER), default="medium")
    args = parser.parse_args()

    source_rows = read_jsonl(args.input_jsonl)
    queue = build_queue(source_rows, min_confidence=args.min_confidence)
    write_jsonl(args.output_jsonl, queue)
    if args.summary_json:
        write_summary(args.summary_json, queue, source_rows)
    print(json.dumps({
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "summary_json": args.summary_json,
        "source_rows": len(source_rows),
        "prior_references": len(queue),
        "total_query_dataset_mentions": sum(r["n_query_dataset_mentions"] for r in queue),
        "min_confidence": args.min_confidence,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
