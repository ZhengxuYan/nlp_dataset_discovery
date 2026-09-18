#!/usr/bin/env python3
"""Enrich dataset census rows with free public paper and resource metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence


URL_RE = re.compile(r"https?://[^\s)>\"]+")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_RE = re.compile(r"\b(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
ACL_RE = re.compile(r"\b(?:[A-Z]\d{2}-\d{4}|20\d{2}\.[a-z0-9-]+\.\d+)\b", re.IGNORECASE)
MIN_TITLE_FUZZY_SCORE = 0.78
MIN_DATASET_FUZZY_SCORE = 0.72


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().rstrip(".,;")
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    match = DOI_RE.search(value)
    return match.group(0).lower() if match else None


def normalize_url(url: str) -> str:
    url = url.strip().rstrip(".,;)")
    while url.endswith("]") and url.count("]") > url.count("["):
        url = url[:-1]
    return url


def normalize_title_for_match(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(value.split())


def title_similarity(query_title: str | None, candidate_title: str | None) -> float | None:
    query = normalize_title_for_match(query_title)
    candidate = normalize_title_for_match(candidate_title)
    if not query or not candidate:
        return None
    return round(SequenceMatcher(None, query, candidate).ratio(), 3)


def fuzzy_confidence_label(score: float | None) -> str:
    if score is None:
        return "fuzzy_unknown"
    if score >= 0.92:
        return "fuzzy_high"
    if score >= MIN_TITLE_FUZZY_SCORE:
        return "fuzzy_medium"
    return "fuzzy_low"


def dataset_fuzzy_confidence_label(score: float | None) -> str:
    if score is None:
        return "fuzzy_unknown"
    if score >= 0.90:
        return "fuzzy_high"
    if score >= MIN_DATASET_FUZZY_SCORE:
        return "fuzzy_medium"
    return "fuzzy_low"


def stable_record_id(row: Mapping[str, Any]) -> str:
    for key in ("record_id", "dataset_id", "id", "paper_id", "source_id"):
        value = row.get(key)
        if value:
            return str(value)
    basis = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def walk_values(value: Any) -> Iterator[Any]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child)
    else:
        yield value


def first_string(row: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def collect_dataset_names(row: Mapping[str, Any]) -> list[str]:
    candidates: list[Any] = [
        row.get("dataset_name"),
        row.get("dataset_id"),
        row.get("acronym"),
    ]
    aliases = row.get("aliases")
    if isinstance(aliases, list):
        candidates.extend(aliases)
    identity = row.get("dataset_identity")
    if isinstance(identity, Mapping):
        candidates.extend([identity.get("canonical_name"), identity.get("acronym")])
        identity_aliases = identity.get("aliases")
        if isinstance(identity_aliases, list):
            candidates.extend(identity_aliases)
    names: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        text = candidate.strip()
        if not text or text.lower() in {"unclear", "unknown", "none"}:
            continue
        if text.isdigit():
            continue
        key = normalize_title_for_match(text)
        if key and key not in seen:
            seen.add(key)
            names.append(text)
    return names


def collect_urls(row: Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for value in walk_values(row):
        if not isinstance(value, str):
            continue
        for match in URL_RE.finditer(value):
            url = normalize_url(match.group(0))
            try:
                urllib.parse.urlparse(url)
            except ValueError:
                continue
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def extract_arxiv_id(row: Mapping[str, Any], urls: list[str]) -> str | None:
    for key in ("arxiv_id", "arxiv", "paper_arxiv_id"):
        value = first_string(row, [key])
        if value:
            match = ARXIV_RE.search(value)
            if match:
                return match.group(1)
    for url in urls:
        if "arxiv.org" not in url:
            continue
        match = ARXIV_RE.search(url)
        if match:
            return match.group(1)
    return None


def extract_acl_id(row: Mapping[str, Any], urls: list[str]) -> str | None:
    for key in ("acl_id", "anthology_id", "acl_anthology_id"):
        value = first_string(row, [key])
        if value:
            return value
    for url in urls:
        if "aclanthology.org" not in url:
            continue
        parts = [part for part in urllib.parse.urlparse(url).path.split("/") if part]
        if parts:
            return parts[-1].removesuffix(".pdf")
    return None


def extract_doi(row: Mapping[str, Any], urls: list[str]) -> str | None:
    for key in ("doi", "paper_doi"):
        doi = normalize_doi(first_string(row, [key]))
        if doi:
            return doi
    for value in walk_values(row):
        if isinstance(value, str):
            doi = normalize_doi(value)
            if doi:
                return doi
    for url in urls:
        doi = normalize_doi(url)
        if doi:
            return doi
    return None


def classify_dataset_urls(urls: list[str]) -> dict[str, Any]:
    hf_urls: list[str] = []
    github_urls: list[str] = []
    pwc_urls: list[str] = []
    project_urls: list[str] = []
    download_urls: list[str] = []
    for url in urls:
        host = urllib.parse.urlparse(url).netloc.lower()
        path = urllib.parse.urlparse(url).path.lower()
        if "huggingface.co" in host and "/datasets/" in path:
            hf_urls.append(url)
        elif host == "github.com" or host.endswith(".github.com"):
            github_urls.append(url)
        elif "paperswithcode.com" in host:
            pwc_urls.append(url)
        elif any(path.endswith(ext) for ext in (".zip", ".tar.gz", ".tgz", ".jsonl", ".csv", ".parquet")):
            download_urls.append(url)
        else:
            project_urls.append(url)
    return {
        "all": urls,
        "huggingface": hf_urls,
        "github": github_urls,
        "paperswithcode": pwc_urls,
        "project_pages": project_urls,
        "downloads": download_urls,
    }


def hf_repo_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0] == "datasets":
        return "/".join(parts[1:3])
    return None


def github_repo(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return None


@dataclass
class ApiResult:
    ok: bool
    status: int | None
    data: Any
    url: str
    error: str | None = None


TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


def semantic_scholar_headers() -> dict[str, str]:
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    return {"x-api-key": api_key} if api_key else {}


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return min(float(value), 120.0)
    except ValueError:
        return None


def _backoff_seconds(attempt: int, sleep_seconds: float) -> float:
    return max(sleep_seconds, min(2.0 * (2 ** attempt), 60.0))


def is_transient_result(payload: Mapping[str, Any]) -> bool:
    status = payload.get("status")
    if status in TRANSIENT_STATUSES:
        return True
    return status is None and bool(payload.get("error"))


class CachedHttpClient:
    def __init__(
        self,
        cache_path: Path,
        offline: bool = False,
        sleep_seconds: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self.cache_path = cache_path
        self.offline = offline
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            self.cache: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}

    def save(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8")

    def get_json(self, url: str, headers: Mapping[str, str] | None = None) -> ApiResult:
        key = "json:" + url
        if key in self.cache:
            payload = self.cache[key]
            if self.offline or not is_transient_result(payload):
                return ApiResult(payload.get("ok", False), payload.get("status"), payload.get("data"), url, payload.get("error"))
        if self.offline:
            return ApiResult(False, None, None, url, "offline cache miss")
        request_headers = {
            "User-Agent": "nlp-dataset-discovery/metadata-enrichment (mailto:metadata-enrichment@example.com)",
            **dict(headers or {}),
        }
        result: dict[str, Any] = {"ok": False, "status": None, "data": None, "error": "not attempted", "queried_at": utc_now()}
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, headers=request_headers)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    result = {
                        "ok": True,
                        "status": response.status,
                        "data": data,
                        "queried_at": utc_now(),
                        "attempts": attempt + 1,
                    }
                    break
            except urllib.error.HTTPError as exc:
                retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                result = {
                    "ok": False,
                    "status": exc.code,
                    "data": None,
                    "error": str(exc),
                    "queried_at": utc_now(),
                    "attempts": attempt + 1,
                }
                if exc.code not in TRANSIENT_STATUSES or attempt >= self.max_retries:
                    break
                time.sleep(retry_after if retry_after is not None else _backoff_seconds(attempt, self.sleep_seconds))
            except Exception as exc:  # noqa: BLE001 - persisted as audit detail
                result = {
                    "ok": False,
                    "status": None,
                    "data": None,
                    "error": str(exc),
                    "queried_at": utc_now(),
                    "attempts": attempt + 1,
                }
                if attempt >= self.max_retries:
                    break
                time.sleep(_backoff_seconds(attempt, self.sleep_seconds))
        if result.get("ok") or not is_transient_result(result):
            self.cache[key] = result
        else:
            self.cache.pop(key, None)
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return ApiResult(result["ok"], result["status"], result["data"], url, result.get("error"))

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> ApiResult:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        key = "json-post:" + url + ":" + hashlib.sha1(body).hexdigest()
        if key in self.cache:
            cached = self.cache[key]
            if self.offline or not is_transient_result(cached):
                return ApiResult(cached.get("ok", False), cached.get("status"), cached.get("data"), url, cached.get("error"))
        if self.offline:
            return ApiResult(False, None, None, url, "offline cache miss")
        request_headers = {
            "Content-Type": "application/json",
            "User-Agent": "nlp-dataset-discovery/metadata-enrichment (mailto:metadata-enrichment@example.com)",
            **dict(headers or {}),
        }
        result: dict[str, Any] = {"ok": False, "status": None, "data": None, "error": "not attempted", "queried_at": utc_now()}
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    result = {
                        "ok": True,
                        "status": response.status,
                        "data": data,
                        "queried_at": utc_now(),
                        "attempts": attempt + 1,
                    }
                    break
            except urllib.error.HTTPError as exc:
                retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                result = {
                    "ok": False,
                    "status": exc.code,
                    "data": None,
                    "error": str(exc),
                    "queried_at": utc_now(),
                    "attempts": attempt + 1,
                }
                if exc.code not in TRANSIENT_STATUSES or attempt >= self.max_retries:
                    break
                time.sleep(retry_after if retry_after is not None else _backoff_seconds(attempt, self.sleep_seconds))
            except Exception as exc:  # noqa: BLE001 - persisted as audit detail
                result = {
                    "ok": False,
                    "status": None,
                    "data": None,
                    "error": str(exc),
                    "queried_at": utc_now(),
                    "attempts": attempt + 1,
                }
                if attempt >= self.max_retries:
                    break
                time.sleep(_backoff_seconds(attempt, self.sleep_seconds))
        if result.get("ok") or not is_transient_result(result):
            self.cache[key] = result
        else:
            self.cache.pop(key, None)
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return ApiResult(result["ok"], result["status"], result["data"], url, result.get("error"))

    def check_url(self, url: str) -> dict[str, Any]:
        key = "health:" + url
        if key in self.cache:
            return dict(self.cache[key])
        if self.offline:
            return {"url": url, "ok": False, "error": "offline cache miss", "checked_at": utc_now()}
        for method in ("HEAD", "GET"):
            request = urllib.request.Request(url, method=method)
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    payload = {
                        "url": url,
                        "ok": 200 <= response.status < 400,
                        "status": response.status,
                        "resolved_url": response.url,
                        "downloadable": _looks_downloadable(url, response.headers.get("content-type")),
                        "checked_at": utc_now(),
                    }
                    self.cache[key] = payload
                    return payload
            except urllib.error.HTTPError as exc:
                if method == "HEAD" and exc.code in {403, 405}:
                    continue
                payload = {"url": url, "ok": False, "status": exc.code, "error": str(exc), "checked_at": utc_now()}
                self.cache[key] = payload
                return payload
            except Exception as exc:  # noqa: BLE001
                if method == "HEAD":
                    continue
                payload = {"url": url, "ok": False, "error": str(exc), "checked_at": utc_now()}
                self.cache[key] = payload
                return payload
        payload = {"url": url, "ok": False, "error": "unreachable", "checked_at": utc_now()}
        self.cache[key] = payload
        return payload


def _looks_downloadable(url: str, content_type: str | None) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in (".zip", ".tar.gz", ".tgz", ".jsonl", ".csv", ".parquet")):
        return True
    return bool(content_type and any(token in content_type for token in ("application/", "text/csv")))


def query_paper_metadata(
    row: Mapping[str, Any],
    urls: list[str],
    client: CachedHttpClient,
    semantic_override: Mapping[str, Any] | None = None,
    openalex_mode: str = "full",
) -> dict[str, Any]:
    title = first_string(row, ["title", "paper_title", "name"])
    doi = extract_doi(row, urls)
    arxiv_id = extract_arxiv_id(row, urls)
    acl_id = extract_acl_id(row, urls)
    identifiers: dict[str, Any] = {"doi": doi, "arxiv_id": arxiv_id, "acl_anthology_id": acl_id}
    sources: list[dict[str, Any]] = []
    if openalex_mode == "full":
        openalex = _query_openalex(doi, title, client)
    elif openalex_mode == "doi-only":
        openalex = _query_openalex(doi, None, client)
    elif openalex_mode == "off":
        openalex = {}
    else:
        raise ValueError(f"Unknown OpenAlex mode: {openalex_mode}")
    semantic = dict(semantic_override) if semantic_override is not None else _query_semantic_scholar(doi, arxiv_id, title, client)

    if openalex.get("id"):
        identifiers["openalex_work_id"] = openalex.get("id")
        sources.append(_metadata_source("openalex", openalex))
    if semantic.get("paperId"):
        identifiers["semantic_scholar_paper_id"] = semantic.get("paperId")
        sources.append(_metadata_source("semantic_scholar", semantic))

    return {
        "paper_identifiers": identifiers,
        "paper_metrics": {
            "citation_count": semantic.get("citationCount", openalex.get("cited_by_count")),
            "influential_citation_count": semantic.get("influentialCitationCount"),
            "reference_count": semantic.get("referenceCount", openalex.get("referenced_works_count")),
            "publication_year": semantic.get("year", openalex.get("publication_year")),
            "venue": semantic.get("venue") or openalex.get("venue"),
            "authors": semantic.get("authors") or openalex.get("authors"),
        },
        "paper_metadata_sources": [{**source, "queried_at": utc_now()} for source in sources],
    }


def paper_identifiers_only(row: Mapping[str, Any], urls: list[str]) -> dict[str, Any]:
    return {
        "paper_identifiers": {
            "doi": extract_doi(row, urls),
            "arxiv_id": extract_arxiv_id(row, urls),
            "acl_anthology_id": extract_acl_id(row, urls),
        },
        "paper_metrics": {
            "citation_count": None,
            "influential_citation_count": None,
            "reference_count": None,
            "publication_year": row.get("year"),
            "venue": row.get("venue") or row.get("event") or row.get("venue_prefix"),
            "authors": row.get("authors"),
        },
        "paper_metadata_sources": [],
    }


def _metadata_source(source: str, result: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "source": source,
        "query_key": result.get("query_key"),
        "match_confidence": result.get("match_confidence"),
        "match_method": result.get("match_method"),
        "match_confidence_score": result.get("match_confidence_score"),
    }
    if result.get("matched_title"):
        payload["matched_title"] = result.get("matched_title")
    return payload


def _query_openalex(doi: str | None, title: str | None, client: CachedHttpClient) -> dict[str, Any]:
    if doi:
        url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}"
        result = client.get_json(url, headers=semantic_scholar_headers())
        if result.ok and isinstance(result.data, dict):
            return _openalex_work(result.data, f"doi:{doi}", "exact", "doi", 1.0)
    if title:
        url = "https://api.openalex.org/works?search=" + urllib.parse.quote(title) + "&per-page=1"
        result = client.get_json(url, headers=semantic_scholar_headers())
        if result.ok and isinstance(result.data, dict) and result.data.get("results"):
            work = result.data["results"][0]
            matched_title = first_string(work, ["display_name", "title"])
            score = title_similarity(title, matched_title)
            if score is not None and score < MIN_TITLE_FUZZY_SCORE:
                return {}
            return _openalex_work(work, f"title:{title}", fuzzy_confidence_label(score), "title_fuzzy", score)
    return {}


def _openalex_work(
    work: Mapping[str, Any],
    query_key: str,
    confidence: str,
    match_method: str,
    confidence_score: float | None,
) -> dict[str, Any]:
    authors = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") if isinstance(authorship, Mapping) else None
        if isinstance(author, Mapping) and author.get("display_name"):
            authors.append(author["display_name"])
    venue = None
    primary_location = work.get("primary_location")
    if isinstance(primary_location, Mapping):
        source = primary_location.get("source")
        if isinstance(source, Mapping):
            venue = source.get("display_name")
    return {
        "id": work.get("id"),
        "cited_by_count": work.get("cited_by_count"),
        "referenced_works_count": len(work.get("referenced_works") or []),
        "publication_year": work.get("publication_year"),
        "venue": venue,
        "authors": authors,
        "query_key": query_key,
        "match_confidence": confidence,
        "match_method": match_method,
        "match_confidence_score": confidence_score,
        "matched_title": first_string(work, ["display_name", "title"]),
    }


def _query_semantic_scholar(doi: str | None, arxiv_id: str | None, title: str | None, client: CachedHttpClient) -> dict[str, Any]:
    fields = "paperId,title,year,venue,citationCount,influentialCitationCount,referenceCount,authors"
    candidates: list[tuple[str, str, str]] = []
    if doi:
        candidates.append((f"DOI:{doi}", f"doi:{doi}", "exact"))
    if arxiv_id:
        candidates.append((f"ARXIV:{arxiv_id}", f"arxiv:{arxiv_id}", "exact"))
    for paper_key, query_key, confidence in candidates:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(paper_key, safe='')}?fields={fields}"
        result = client.get_json(url)
        if result.ok and isinstance(result.data, dict):
            return _semantic_paper(result.data, query_key, confidence, confidence.removeprefix("exact_") if confidence.startswith("exact_") else query_key.split(":", 1)[0], 1.0)
    if title:
        url = "https://api.semanticscholar.org/graph/v1/paper/search?limit=1&fields=" + fields + "&query=" + urllib.parse.quote(title)
        result = client.get_json(url)
        if result.ok and isinstance(result.data, dict) and result.data.get("data"):
            paper = result.data["data"][0]
            matched_title = first_string(paper, ["title"])
            score = title_similarity(title, matched_title)
            if score is not None and score < MIN_TITLE_FUZZY_SCORE:
                return {}
            return _semantic_paper(paper, f"title:{title}", fuzzy_confidence_label(score), "title_fuzzy", score)
    return {}


SEMANTIC_FIELDS = "paperId,title,year,venue,citationCount,influentialCitationCount,referenceCount,authors"


def semantic_batch_key(row: Mapping[str, Any], urls: list[str]) -> tuple[str, str, str] | None:
    doi = extract_doi(row, urls)
    if doi:
        return f"DOI:{doi}", f"doi:{doi}", "doi"
    arxiv_id = extract_arxiv_id(row, urls)
    if arxiv_id:
        return f"ARXIV:{arxiv_id}", f"arxiv:{arxiv_id}", "arxiv"
    return None


def prefetch_semantic_scholar_exact(
    rows: Sequence[Mapping[str, Any]],
    client: CachedHttpClient,
    *,
    batch_size: int = 100,
) -> dict[str, dict[str, Any]]:
    results, _ = prefetch_semantic_scholar_exact_with_stats(rows, client, batch_size=batch_size)
    return results


def prefetch_semantic_scholar_exact_with_stats(
    rows: Sequence[Mapping[str, Any]],
    client: CachedHttpClient,
    *,
    batch_size: int = 100,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    keyed_rows: list[tuple[str, str, str, str]] = []
    for row in rows:
        urls = collect_urls(row)
        key = semantic_batch_key(row, urls)
        if key is None:
            continue
        paper_key, query_key, match_method = key
        keyed_rows.append((stable_record_id(row), paper_key, query_key, match_method))

    results: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "requested_ids": len(keyed_rows),
        "batches": 0,
        "failed_batches": 0,
        "failed_batch_errors": [],
        "null_items": 0,
    }
    endpoint = f"https://api.semanticscholar.org/graph/v1/paper/batch?fields={SEMANTIC_FIELDS}"
    for start in range(0, len(keyed_rows), batch_size):
        batch = keyed_rows[start:start + batch_size]
        stats["batches"] += 1
        result = client.post_json(endpoint, {"ids": [paper_key for _, paper_key, _, _ in batch]}, headers=semantic_scholar_headers())
        if not (result.ok and isinstance(result.data, list)):
            stats["failed_batches"] += 1
            stats["failed_batch_errors"].append(
                {
                    "batch_start": start,
                    "batch_size": len(batch),
                    "status": result.status,
                    "error": result.error,
                }
            )
            continue
        for item, (record_id, _, query_key, match_method) in zip(result.data, batch):
            if not isinstance(item, Mapping) or not item.get("paperId"):
                stats["null_items"] += 1
                continue
            results[record_id] = _semantic_paper(item, query_key, "exact", match_method, 1.0)
    stats["matches"] = len(results)
    return results, stats


def _semantic_paper(
    paper: Mapping[str, Any],
    query_key: str,
    confidence: str,
    match_method: str,
    confidence_score: float | None,
) -> dict[str, Any]:
    authors = [a.get("name") for a in paper.get("authors") or [] if isinstance(a, Mapping) and a.get("name")]
    return {
        "paperId": paper.get("paperId"),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "citationCount": paper.get("citationCount"),
        "influentialCitationCount": paper.get("influentialCitationCount"),
        "referenceCount": paper.get("referenceCount"),
        "authors": authors,
        "query_key": query_key,
        "match_confidence": confidence,
        "match_method": match_method,
        "match_confidence_score": confidence_score,
        "matched_title": first_string(paper, ["title"]),
    }


def query_resource_metadata(
    dataset_urls: Mapping[str, list[str]],
    client: CachedHttpClient,
    check_health: bool,
    dataset_names: list[str] | None = None,
    allow_name_fallback: bool = True,
) -> dict[str, Any]:
    dataset_names = dataset_names or []
    hf_metadata = [_query_hf_dataset(url, client) for url in dataset_urls.get("huggingface", [])]
    if allow_name_fallback and not hf_metadata:
        hf_metadata = _first_confident_name_match(dataset_names, lambda name: _query_hf_dataset_by_name(name, client))
    github_metadata = [_query_github_repo(url, client) for url in dataset_urls.get("github", [])]
    pwc_metadata = [_query_pwc_dataset(url, client) for url in dataset_urls.get("paperswithcode", [])]
    if allow_name_fallback and not pwc_metadata:
        pwc_metadata = _first_confident_name_match(dataset_names, lambda name: _query_pwc_dataset_by_name(name, client))
    health = [client.check_url(url) for url in dataset_urls.get("all", [])] if check_health else []
    return {
        "hf_metadata": [item for item in hf_metadata if item],
        "github_metadata": [item for item in github_metadata if item],
        "pwc_metadata": [item for item in pwc_metadata if item],
        "resource_health": health,
    }


def _first_confident_name_match(names: list[str], lookup: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for name in names:
        item = lookup(name)
        if not item:
            continue
        if item.get("error"):
            errors.append(item)
            continue
        return [item]
    return errors[:1]


def _query_hf_dataset(url: str, client: CachedHttpClient) -> dict[str, Any] | None:
    repo = hf_repo_id(url)
    if not repo:
        return None
    api_url = "https://huggingface.co/api/datasets/" + urllib.parse.quote(repo, safe="/")
    result = client.get_json(api_url)
    payload = {
        "url": url,
        "repo_id": repo,
        "source": "huggingface",
        "status": result.status,
        "match_method": "url_exact",
        "match_confidence": "exact",
        "match_confidence_score": 1.0,
    }
    if result.ok and isinstance(result.data, Mapping):
        payload.update(_hf_dataset_fields(result.data))
    else:
        payload["error"] = result.error
    return payload


def _query_hf_dataset_by_name(name: str, client: CachedHttpClient) -> dict[str, Any] | None:
    api_url = "https://huggingface.co/api/datasets?search=" + urllib.parse.quote(name) + "&limit=5"
    result = client.get_json(api_url)
    payload = {
        "source": "huggingface",
        "query_name": name,
        "status": result.status,
        "match_method": "dataset_name_fuzzy",
        "match_confidence": "fuzzy_unknown",
        "match_confidence_score": None,
    }
    if not (result.ok and isinstance(result.data, list)):
        payload["error"] = result.error
        return payload
    best: tuple[float, Mapping[str, Any]] | None = None
    for candidate in result.data:
        if not isinstance(candidate, Mapping):
            continue
        repo_id = candidate.get("id") or candidate.get("repo_id")
        repo_name = str(repo_id).split("/")[-1] if repo_id else None
        candidate_name = first_string(candidate, ["pretty_name", "name"]) or repo_name
        score = max(
            value
            for value in [
                title_similarity(name, str(repo_id) if repo_id else None),
                title_similarity(name, candidate_name),
                title_similarity(name, repo_name),
            ]
            if value is not None
        ) if any(
            value is not None
            for value in [
                title_similarity(name, str(repo_id) if repo_id else None),
                title_similarity(name, candidate_name),
                title_similarity(name, repo_name),
            ]
        ) else None
        if score is None:
            continue
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None or best[0] < MIN_DATASET_FUZZY_SCORE:
        payload.update({"match_confidence": "fuzzy_low", "match_confidence_score": best[0] if best else None, "error": "no confident dataset name match"})
        return payload
    score, candidate = best
    repo_id = candidate.get("id") or candidate.get("repo_id")
    payload.update({
        "url": f"https://huggingface.co/datasets/{repo_id}" if repo_id else None,
        "repo_id": repo_id,
        "matched_name": first_string(candidate, ["pretty_name", "name"]) or repo_id,
        "match_confidence": dataset_fuzzy_confidence_label(score),
        "match_confidence_score": score,
    })
    payload.update(_hf_dataset_fields(candidate))
    return payload


def _hf_dataset_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "downloads": data.get("downloads"),
        "likes": data.get("likes"),
        "last_modified": data.get("lastModified"),
        "tags": data.get("tags"),
        "license": _extract_hf_license(data),
        "card_exists": bool(data.get("cardData")),
    }


def _extract_hf_license(data: Mapping[str, Any]) -> Any:
    card_data = data.get("cardData")
    if isinstance(card_data, Mapping) and card_data.get("license"):
        return card_data.get("license")
    tags = data.get("tags") or []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _query_github_repo(url: str, client: CachedHttpClient) -> dict[str, Any] | None:
    repo = github_repo(url)
    if not repo:
        return None
    api_url = "https://api.github.com/repos/" + repo
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    result = client.get_json(api_url, headers=headers)
    payload = {
        "url": url,
        "repo": repo,
        "source": "github",
        "status": result.status,
        "match_method": "url_exact",
        "match_confidence": "exact",
        "match_confidence_score": 1.0,
    }
    if result.ok and isinstance(result.data, Mapping):
        license_info = result.data.get("license")
        payload.update({
            "stars": result.data.get("stargazers_count"),
            "forks": result.data.get("forks_count"),
            "watchers": result.data.get("watchers_count"),
            "open_issues": result.data.get("open_issues_count"),
            "latest_push": result.data.get("pushed_at"),
            "license": license_info.get("spdx_id") if isinstance(license_info, Mapping) else None,
        })
    else:
        payload["error"] = result.error
    return payload


def _query_pwc_dataset(url: str, client: CachedHttpClient) -> dict[str, Any] | None:
    parsed = urllib.parse.urlparse(url)
    slug = [part for part in parsed.path.split("/") if part][-1:] or [None]
    if not slug[0]:
        return None
    api_url = "https://paperswithcode.com/api/v1/datasets/?q=" + urllib.parse.quote(slug[0])
    result = client.get_json(api_url)
    payload = {
        "url": url,
        "slug": slug[0],
        "source": "paperswithcode",
        "status": result.status,
        "match_method": "url_exact",
        "match_confidence": "exact",
        "match_confidence_score": 1.0,
    }
    if result.ok and isinstance(result.data, Mapping):
        payload["results"] = result.data.get("results", [])[:3]
    else:
        payload["error"] = result.error
    return payload


def _query_pwc_dataset_by_name(name: str, client: CachedHttpClient) -> dict[str, Any] | None:
    api_url = "https://paperswithcode.com/api/v1/datasets/?q=" + urllib.parse.quote(name)
    result = client.get_json(api_url)
    payload = {
        "source": "paperswithcode",
        "query_name": name,
        "status": result.status,
        "match_method": "dataset_name_fuzzy",
        "match_confidence": "fuzzy_unknown",
        "match_confidence_score": None,
    }
    if not (result.ok and isinstance(result.data, Mapping)):
        payload["error"] = result.error
        return payload
    results = result.data.get("results") or []
    best: tuple[float, Mapping[str, Any]] | None = None
    for candidate in results:
        if not isinstance(candidate, Mapping):
            continue
        candidate_name = first_string(candidate, ["name", "full_name"])
        score = title_similarity(name, candidate_name)
        if score is None:
            continue
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None or best[0] < MIN_DATASET_FUZZY_SCORE:
        payload.update({"match_confidence": "fuzzy_low", "match_confidence_score": best[0] if best else None, "error": "no confident dataset name match"})
        return payload
    score, candidate = best
    payload.update({
        "matched_name": first_string(candidate, ["name", "full_name"]),
        "url": candidate.get("url"),
        "slug": candidate.get("slug"),
        "match_confidence": dataset_fuzzy_confidence_label(score),
        "match_confidence_score": score,
        "results": [candidate],
    })
    return payload


def enrich_row(
    row: Mapping[str, Any],
    client: CachedHttpClient,
    check_health: bool = False,
    paper_only: bool = False,
    resource_only: bool = False,
    resource_name_fallback: bool = True,
    semantic_override: Mapping[str, Any] | None = None,
    openalex_mode: str = "full",
) -> dict[str, Any]:
    urls = collect_urls(row)
    dataset_urls = classify_dataset_urls(urls)
    dataset_names = collect_dataset_names(row)
    enriched = dict(row)
    resource_metadata = (
        {"hf_metadata": [], "github_metadata": [], "pwc_metadata": [], "resource_health": []}
        if paper_only
        else query_resource_metadata(dataset_urls, client, check_health, dataset_names, allow_name_fallback=resource_name_fallback)
    )
    paper_metadata = (
        paper_identifiers_only(row, urls)
        if resource_only
        else query_paper_metadata(row, urls, client, semantic_override=semantic_override, openalex_mode=openalex_mode)
    )
    metadata = {
        **paper_metadata,
        "dataset_urls": dataset_urls,
        "dataset_name_candidates": dataset_names,
        **resource_metadata,
        "metadata_enrichment": {
            "record_id": stable_record_id(row),
            "queried_at": utc_now(),
            "match_policy": "exact identifiers and URLs first; title/name search is fuzzy fallback",
            "paper_only": paper_only,
            "resource_only": resource_only,
            "resource_name_fallback": resource_name_fallback,
        },
    }
    enriched["public_metadata"] = metadata
    return enriched


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    offset: int = 0,
    limit: int | None,
    sample_strategy: str,
) -> Iterator[dict[str, Any]]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if sample_strategy == "first":
        yielded = 0
        for idx, row in enumerate(rows):
            if idx < offset:
                continue
            if limit is not None and yielded >= limit:
                break
            yielded += 1
            yield row
        return
    if sample_strategy != "stratified-year":
        raise ValueError(f"Unknown sample strategy: {sample_strategy}")
    if limit is None:
        yield from rows
        return
    by_year: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        year = str(row.get("year") or row.get("published_date", "")[:4] or "unknown")
        by_year.setdefault(year, []).append(row)
    years = sorted(by_year)
    if not years:
        return
    per_year = max(limit // len(years), 1)
    remainder = limit - per_year * len(years)
    selected: list[dict[str, Any]] = []
    for idx, year in enumerate(years):
        take = per_year + (1 if idx < remainder else 0)
        selected.extend(by_year[year][:take])
    for row in selected[offset:limit + offset if limit is not None else None]:
        yield row


def row_matches_resource_filter(row: Mapping[str, Any], resource_link_filter: str) -> bool:
    if resource_link_filter == "all":
        return True
    dataset_urls = classify_dataset_urls(collect_urls(row))
    if resource_link_filter == "any":
        return bool(dataset_urls.get("huggingface") or dataset_urls.get("github") or dataset_urls.get("paperswithcode"))
    if resource_link_filter == "huggingface":
        return bool(dataset_urls.get("huggingface"))
    if resource_link_filter == "github":
        return bool(dataset_urls.get("github"))
    if resource_link_filter == "paperswithcode":
        return bool(dataset_urls.get("paperswithcode"))
    raise ValueError(f"Unknown resource link filter: {resource_link_filter}")


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {
        "rows": 0,
        "paper_openalex_matches": 0,
        "paper_semantic_scholar_matches": 0,
        "hf_dataset_links": 0,
        "github_links": 0,
        "pwc_links": 0,
        "healthy_urls": 0,
        "year_counts": {},
    }
    for row in rows:
        summary["rows"] += 1
        metadata = row.get("public_metadata") if isinstance(row, Mapping) else None
        if not isinstance(metadata, Mapping):
            continue
        identifiers = metadata.get("paper_identifiers")
        if isinstance(identifiers, Mapping):
            if identifiers.get("openalex_work_id"):
                summary["paper_openalex_matches"] += 1
            if identifiers.get("semantic_scholar_paper_id"):
                summary["paper_semantic_scholar_matches"] += 1
        dataset_urls = metadata.get("dataset_urls")
        if isinstance(dataset_urls, Mapping):
            summary["hf_dataset_links"] += len(dataset_urls.get("huggingface") or [])
            summary["github_links"] += len(dataset_urls.get("github") or [])
            summary["pwc_links"] += len(dataset_urls.get("paperswithcode") or [])
        for health in metadata.get("resource_health") or []:
            if isinstance(health, Mapping) and health.get("ok"):
                summary["healthy_urls"] += 1
        metrics = metadata.get("paper_metrics")
        year = metrics.get("publication_year") if isinstance(metrics, Mapping) else None
        if year:
            year_counts = summary["year_counts"]
            year_counts[str(year)] = year_counts.get(str(year), 0) + 1
    summary["generated_at"] = utc_now()
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich dataset-bank JSONL rows with public metadata.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/public_metadata_api_cache.json"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-strategy", choices=["first", "stratified-year"], default="first")
    parser.add_argument("--offline", action="store_true", help="Only use cached API responses.")
    parser.add_argument("--check-url-health", action="store_true", help="HEAD/GET resource URLs and cache status.")
    parser.add_argument("--paper-only", action="store_true", help="Only query paper-level metadata; skip HF/GitHub/PWC/resource health.")
    parser.add_argument("--resource-only", action="store_true", help="Only query resource-level metadata; skip paper citation APIs.")
    parser.add_argument(
        "--resource-link-filter",
        choices=["all", "any", "huggingface", "github", "paperswithcode"],
        default="all",
        help="Restrict selected rows to records with specific explicit resource links.",
    )
    parser.add_argument(
        "--resource-name-fallback",
        action="store_true",
        help="Search HF/Papers with Code by dataset name when explicit resource links are missing.",
    )
    parser.add_argument(
        "--openalex-mode",
        choices=["full", "doi-only", "off"],
        default="full",
        help="Control OpenAlex lookups. Use off or doi-only for large first-pass citation enrichment.",
    )
    parser.add_argument("--no-semantic-batch", action="store_true", help="Disable Semantic Scholar exact-match batch prefetch.")
    parser.add_argument("--semantic-batch-only", action="store_true", help="Do not fall back to per-row Semantic Scholar calls when batch prefetch misses.")
    parser.add_argument(
        "--allow-semantic-batch-failures",
        action="store_true",
        help="Continue after failed Semantic Scholar batch requests. By default, batch failures abort the run so coverage is not understated.",
    )
    parser.add_argument("--semantic-batch-size", type=int, default=100)
    parser.add_argument(
        "--processing-chunk-size",
        type=int,
        default=1000,
        help="Number of selected rows to prefetch and write at a time.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.processing_chunk_size <= 0:
        raise ValueError("--processing-chunk-size must be positive")
    if args.paper_only and args.resource_only:
        raise ValueError("--paper-only and --resource-only cannot both be set")
    client = CachedHttpClient(args.cache, offline=args.offline, sleep_seconds=args.sleep_seconds, max_retries=args.max_retries)
    candidate_rows = (
        row
        for row in iter_jsonl(args.input)
        if row_matches_resource_filter(row, args.resource_link_filter)
    )
    selected_rows = list(select_rows(candidate_rows, offset=args.offset, limit=args.limit, sample_strategy=args.sample_strategy))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched_rows: list[Mapping[str, Any]] = []
    count = 0
    semantic_prefetch_matches = 0
    semantic_batch_stats: list[dict[str, Any]] = []
    try:
        with args.output.open("w", encoding="utf-8") as fh:
            for chunk_start in range(0, len(selected_rows), args.processing_chunk_size):
                chunk = selected_rows[chunk_start:chunk_start + args.processing_chunk_size]
                if args.progress_every:
                    print(
                        json.dumps(
                            {
                                "chunk_start": chunk_start,
                                "chunk_rows": len(chunk),
                                "selected_rows": len(selected_rows),
                                "timestamp": utc_now(),
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                if args.no_semantic_batch:
                    semantic_prefetch = {}
                    chunk_batch_stats = {"requested_ids": 0, "batches": 0, "failed_batches": 0, "failed_batch_errors": [], "null_items": 0, "matches": 0}
                else:
                    semantic_prefetch, chunk_batch_stats = prefetch_semantic_scholar_exact_with_stats(
                        chunk,
                        client,
                        batch_size=args.semantic_batch_size,
                    )
                chunk_batch_stats = {"chunk_start": chunk_start, **chunk_batch_stats}
                semantic_batch_stats.append(chunk_batch_stats)
                if chunk_batch_stats.get("failed_batches") and not args.allow_semantic_batch_failures:
                    client.save()
                    raise RuntimeError(
                        "Semantic Scholar batch prefetch failed; rerun will retry transient failures. "
                        f"Details: {json.dumps(chunk_batch_stats, sort_keys=True)}"
                    )
                semantic_prefetch_matches += len(semantic_prefetch)
                for row in chunk:
                    semantic_override = semantic_prefetch.get(stable_record_id(row))
                    if args.semantic_batch_only and semantic_override is None:
                        semantic_override = {}
                    enriched = enrich_row(
                        row,
                        client,
                        check_health=args.check_url_health,
                        paper_only=args.paper_only,
                        resource_only=args.resource_only,
                        resource_name_fallback=args.resource_name_fallback,
                        semantic_override=semantic_override,
                        openalex_mode=args.openalex_mode,
                    )
                    enriched_rows.append(enriched)
                    fh.write(json.dumps(enriched, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
                    fh.flush()
                    if args.save_every and count % args.save_every == 0:
                        client.save()
                    if args.progress_every and count % args.progress_every == 0:
                        print(json.dumps({"processed": count, "timestamp": utc_now()}), file=sys.stderr, flush=True)
                client.save()
    except KeyboardInterrupt:
        client.save()
        raise
    summary = summarize(enriched_rows)
    summary["input"] = str(args.input)
    summary["output"] = str(args.output)
    summary["rows_written"] = count
    summary["offset"] = args.offset
    summary["sample_strategy"] = args.sample_strategy
    summary["semantic_batch_prefetch_matches"] = semantic_prefetch_matches
    summary["semantic_batch_stats"] = semantic_batch_stats
    summary["semantic_batch_only"] = args.semantic_batch_only
    summary["openalex_mode"] = args.openalex_mode
    summary["processing_chunk_size"] = args.processing_chunk_size
    summary["resource_only"] = args.resource_only
    summary["resource_link_filter"] = args.resource_link_filter
    summary["resource_name_fallback"] = args.resource_name_fallback
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    client.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
