#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import requests


S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clear_jsonl(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def existing_ids(path: str | Path) -> set[str]:
    if not Path(path).exists():
        return set()
    return {str(row.get("prior_ref_id") or "") for row in read_jsonl(path)}


def normalize_title(text: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    return re.sub(r"\s+", " ", text)


def token_set(text: str) -> set[str]:
    stop = {"a", "an", "the", "of", "for", "and", "to", "in", "on", "with", "as", "by", "is", "are"}
    return {t for t in normalize_title(text).split() if len(t) > 1 and t not in stop}


def title_overlap_score(query_title: str, candidate_title: str) -> float:
    query_tokens = token_set(query_title)
    candidate_tokens = token_set(candidate_title)
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)


def doi_url(doi: str) -> str:
    return f"https://doi.org/{doi}" if doi else ""


def arxiv_abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""


def semantic_scholar_search(title: str, *, timeout: int) -> dict[str, Any] | None:
    if not title:
        return None
    params = {
        "query": title,
        "limit": 5,
        "fields": "title,year,authors,url,externalIds,openAccessPdf,venue,publicationVenue",
    }
    response = requests.get(S2_SEARCH_URL, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    candidates = data.get("data") or []
    if not candidates:
        return None
    scored = []
    for candidate in candidates:
        score = title_overlap_score(title, candidate.get("title") or "")
        scored.append((score, candidate))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    if best_score < 0.45:
        return None
    best["_title_overlap_score"] = round(best_score, 3)
    return best


def arxiv_title_search(title: str, *, timeout: int) -> dict[str, Any] | None:
    if not title:
        return None
    query = f'ti:"{title}"'
    params = {"search_query": query, "start": 0, "max_results": 3}
    response = requests.get(ARXIV_QUERY_URL, params=params, timeout=timeout)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    scored = []
    for entry in entries:
        found_title = "".join(entry.findtext("atom:title", default="", namespaces=ns).split())
        found_title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ns)).strip()
        score = title_overlap_score(title, found_title)
        scored.append((score, entry, found_title))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, entry, found_title = scored[0]
    if best_score < 0.45:
        return None
    entry_id = entry.findtext("atom:id", default="", namespaces=ns)
    arxiv_match = re.search(r"arxiv\.org/abs/([^/?#]+)", entry_id)
    arxiv_id = arxiv_match.group(1) if arxiv_match else ""
    year = ""
    published = entry.findtext("atom:published", default="", namespaces=ns)
    if published:
        year = published[:4]
    return {
        "title": found_title,
        "year": year,
        "arxiv_id": arxiv_id,
        "url": arxiv_abs_url(arxiv_id),
        "pdf_url": arxiv_pdf_url(arxiv_id),
        "_title_overlap_score": round(best_score, 3),
    }


def resolve_row(row: dict[str, Any], *, timeout: int, use_s2: bool, use_arxiv_search: bool) -> dict[str, Any]:
    title = str(row.get("matched_title") or "")
    arxiv_id = str(row.get("matched_arxiv_id") or "")
    doi = str(row.get("matched_doi") or "")
    url = str(row.get("matched_url") or "")

    resolution: dict[str, Any] = {
        "status": "unresolved",
        "source": "",
        "title": title,
        "year": row.get("matched_year") or "",
        "url": url,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "pdf_url": "",
        "semantic_scholar": {},
        "title_overlap_score": None,
        "resolver_warnings": [],
        "resolved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if arxiv_id:
        resolution.update({
            "status": "resolved_arxiv",
            "source": "bibliography_arxiv_id",
            "url": arxiv_abs_url(arxiv_id),
            "pdf_url": arxiv_pdf_url(arxiv_id),
        })
        return resolution

    if doi:
        resolution.update({
            "status": "resolved_doi",
            "source": "bibliography_doi",
            "url": url or doi_url(doi),
        })
        return resolution

    if use_s2:
        try:
            s2 = semantic_scholar_search(title, timeout=timeout)
        except requests.HTTPError as exc:
            resolution["resolver_warnings"].append(f"semantic_scholar_http_error: {exc}")
            s2 = None
        except requests.RequestException as exc:
            resolution["resolver_warnings"].append(f"semantic_scholar_request_error: {exc}")
            s2 = None
        if s2:
            external = s2.get("externalIds") or {}
            open_pdf = s2.get("openAccessPdf") or {}
            s2_arxiv = external.get("ArXiv") or ""
            s2_doi = external.get("DOI") or ""
            resolution.update({
                "status": "resolved_semantic_scholar",
                "source": "semantic_scholar_title_search",
                "title": s2.get("title") or title,
                "year": s2.get("year") or resolution["year"],
                "url": s2.get("url") or resolution["url"],
                "doi": s2_doi or resolution["doi"],
                "arxiv_id": s2_arxiv or resolution["arxiv_id"],
                "pdf_url": open_pdf.get("url") or (arxiv_pdf_url(s2_arxiv) if s2_arxiv else resolution["pdf_url"]),
                "semantic_scholar": {
                    "paperId": s2.get("paperId"),
                    "venue": s2.get("venue"),
                    "authors": [a.get("name") for a in (s2.get("authors") or [])[:10]],
                },
                "title_overlap_score": s2.get("_title_overlap_score"),
            })
            return resolution

    if use_arxiv_search:
        try:
            arxiv = arxiv_title_search(title, timeout=timeout)
        except requests.HTTPError as exc:
            resolution["resolver_warnings"].append(f"arxiv_http_error: {exc}")
            arxiv = None
        except requests.RequestException as exc:
            resolution["resolver_warnings"].append(f"arxiv_request_error: {exc}")
            arxiv = None
        if arxiv:
            resolution.update({
                "status": "resolved_arxiv_title_search",
                "source": "arxiv_title_search",
                "title": arxiv.get("title") or title,
                "year": arxiv.get("year") or resolution["year"],
                "url": arxiv.get("url") or resolution["url"],
                "arxiv_id": arxiv.get("arxiv_id") or "",
                "pdf_url": arxiv.get("pdf_url") or "",
                "title_overlap_score": arxiv.get("_title_overlap_score"),
            })
            return resolution

    return resolution


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve prior reference queue items to fetchable paper metadata.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--error-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--no-semantic-scholar", action="store_true")
    parser.add_argument("--no-arxiv-search", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    error_path = args.error_jsonl or str(Path(args.output_jsonl).with_suffix(".errors.jsonl"))
    if args.overwrite:
        clear_jsonl(args.output_jsonl)
        clear_jsonl(error_path)

    rows = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not args.overwrite:
        done = existing_ids(args.output_jsonl)
        rows = [r for r in rows if str(r.get("prior_ref_id") or "") not in done]

    print(json.dumps({
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "remaining_rows": len(rows),
        "use_semantic_scholar": not args.no_semantic_scholar,
        "use_arxiv_search": not args.no_arxiv_search,
    }, indent=2))

    processed = 0
    failed = 0
    statuses: dict[str, int] = {}
    for row in rows:
        try:
            output = dict(row)
            resolution = resolve_row(
                row,
                timeout=args.timeout,
                use_s2=not args.no_semantic_scholar,
                use_arxiv_search=not args.no_arxiv_search,
            )
            output["resolution"] = resolution
            append_jsonl(args.output_jsonl, [output])
            processed += 1
            statuses[resolution["status"]] = statuses.get(resolution["status"], 0) + 1
        except Exception as exc:  # noqa: BLE001
            append_jsonl(error_path, [{
                "prior_ref_id": row.get("prior_ref_id"),
                "matched_title": row.get("matched_title"),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "failed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }])
            failed += 1
        print(json.dumps({
            "processed": processed,
            "failed": failed,
            "last": row.get("matched_title"),
            "statuses": statuses,
        }, ensure_ascii=False))
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
