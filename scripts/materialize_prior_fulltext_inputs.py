#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


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


def safe_id(text: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()[:80]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{base}-{digest}" if base else digest


def acl_pdf_url_from_doi(doi: str) -> str:
    doi = str(doi or "").strip()
    prefix = "10.18653/v1/"
    if not doi.lower().startswith(prefix):
        return ""
    acl_id = doi[len(prefix) :]
    if re.match(r"^[a-z]\d{2}-\d+$", acl_id):
        acl_id = acl_id[:1].upper() + acl_id[1:]
    return f"https://aclanthology.org/{acl_id}.pdf"


def build_rows(source_rows: list[dict[str, Any]], *, require_pdf_url: bool) -> list[dict[str, Any]]:
    output = []
    for row in source_rows:
        resolution = row.get("resolution") or {}
        pdf_url = str(resolution.get("pdf_url") or acl_pdf_url_from_doi(resolution.get("doi") or ""))
        if require_pdf_url and not pdf_url:
            continue
        title = str(resolution.get("title") or row.get("matched_title") or "")
        if not title:
            continue
        paper_id = f"PRIOR:{safe_id(row.get('prior_ref_id') or title)}"
        output.append({
            "paper_id": paper_id,
            "prior_ref_id": row.get("prior_ref_id"),
            "title": title,
            "year": resolution.get("year") or row.get("matched_year"),
            "abstract": "",
            "venue_prefix": "prior_reference",
            "event": "",
            "booktitle": "",
            "anthology_url": resolution.get("url") or row.get("matched_url") or "",
            "pdf_url": pdf_url,
            "is_dataset_introducing": True,
            "datasets": [
                {
                    "name": next(iter((row.get("candidate_dataset_names") or {}).keys()), title),
                    "source": "prior_reference_queue",
                }
            ],
            "resolution": resolution,
            "reference_queue_item": {
                "matched_title": row.get("matched_title"),
                "matched_year": row.get("matched_year"),
                "candidate_dataset_names": row.get("candidate_dataset_names") or {},
                "n_query_dataset_mentions": row.get("n_query_dataset_mentions"),
            },
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize resolved prior references into full-text extraction input rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--include-without-pdf", action="store_true")
    args = parser.parse_args()

    source_rows = read_jsonl(args.input_jsonl)
    rows = build_rows(source_rows, require_pdf_url=not args.include_without_pdf)
    write_jsonl(args.output_jsonl, rows)
    print(json.dumps({
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "source_rows": len(source_rows),
        "fulltext_input_rows": len(rows),
        "require_pdf_url": not args.include_without_pdf,
    }, indent=2))


if __name__ == "__main__":
    main()
