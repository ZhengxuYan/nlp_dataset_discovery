#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")


REPRESENTATION_VERSION = "global_claim_level_dcu_v2"
RERANK_PROMPT_VERSION = "support_rerank_v1"
HEURISTIC_METHODS = {"acu_text", "acu_type_text", "dcu_metadata"}
RANDOM_METHODS = {"random"}
RETRIEVAL_METHODS = {
    "random",
    "paper_bm25",
    "paper_dense",
    "paper_hybrid",
    "paper_abstract_bm25",
    "paper_abstract_dense",
    "paper_abstract_hybrid",
    "paper_summary_bm25",
    "paper_summary_dense",
    "paper_summary_hybrid",
    "paper_full_bm25",
    "paper_full_dense",
    "paper_full_hybrid",
    "bm25_acu",
    "dense_acu",
    "hybrid_acu",
    "bm25_typed_acu",
    "dense_typed_acu",
    "hybrid_typed_acu",
    "bm25_dcu",
    "dense_dcu",
    "hybrid_dcu",
    "contextual_dcu",
    "hybrid_dcu_support_rerank",
}
DEFAULT_METHODS = [
    "acu_text",
    "acu_type_text",
    "dcu_metadata",
    "paper_abstract_bm25",
    "paper_abstract_dense",
    "paper_abstract_hybrid",
    "paper_summary_bm25",
    "paper_summary_dense",
    "paper_summary_hybrid",
    "bm25_acu",
    "dense_acu",
    "hybrid_acu",
    "bm25_typed_acu",
    "dense_typed_acu",
    "hybrid_typed_acu",
    "bm25_dcu",
    "dense_dcu",
    "hybrid_dcu",
    "contextual_dcu",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokenize_list(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def tokenize(text: str) -> set[str]:
    return set(tokenize_list(text))


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def stable_acu_key(paper_id: str, dataset_id: str, acu_type: str, text: str) -> str:
    payload = "\t".join([clean_text(paper_id), clean_text(dataset_id), clean_text(acu_type), clean_text(text)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def stable_cache_key(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def strip_invalid_unicode(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [strip_invalid_unicode(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_invalid_unicode(item) for key, item in value.items()}
    return value


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def containment(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def list_tokens(values: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(values, list):
        for value in values:
            if isinstance(value, dict):
                value = value.get("name") or value.get("dataset_name") or json.dumps(value, sort_keys=True)
            output.update(tokenize(str(value)))
    elif values:
        output.update(tokenize(str(values)))
    return output


def nested_list(row: dict[str, Any], *path: str) -> list[Any]:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return []
        value = value.get(key)
    return value if isinstance(value, list) else []


def dataset_name(dataset: dict[str, Any]) -> str:
    identity = dataset.get("dataset_identity") or {}
    return str(identity.get("canonical_name") or dataset.get("dataset_name") or dataset.get("dataset_id") or "")


def dataset_metadata(dataset: dict[str, Any]) -> dict[str, Any]:
    construction = dataset.get("construction") or {}
    synthetic = construction.get("synthetic_generation") or {}
    scale = dataset.get("scale") or {}
    evaluation = dataset.get("evaluation") or {}
    availability = dataset.get("availability") or {}
    return {
        "dataset_name": dataset_name(dataset),
        "role": dataset.get("role") or "",
        "resource_type": dataset.get("resource_type") or "",
        "primary_use": dataset.get("primary_use") or "",
        "tasks": nested_list(dataset, "coverage", "tasks"),
        "domains": nested_list(dataset, "coverage", "domains"),
        "languages": nested_list(dataset, "coverage", "languages"),
        "modalities": nested_list(dataset, "coverage", "modality"),
        "source_data_origin": construction.get("source_data_origin") or "",
        "collection_method": construction.get("collection_method") or "",
        "source_datasets": [
            item.get("name") if isinstance(item, dict) else item
            for item in construction.get("source_datasets") or []
        ],
        "transformation_types": construction.get("transformation_types") or [],
        "annotation_protocol": construction.get("annotation_protocol") or "",
        "annotator_type": construction.get("annotator_type") or "",
        "quality_control": construction.get("quality_control") or "",
        "uses_llm": bool(synthetic.get("uses_llm")),
        "llm_models": synthetic.get("model_names") or [],
        "scale": " ".join(
            str(value)
            for value in [
                scale.get("size_text"),
                scale.get("num_instances"),
                scale.get("num_documents"),
                scale.get("num_tokens"),
                scale.get("num_languages"),
                scale.get("num_domains"),
            ]
            if value not in (None, "", "unclear")
        ),
        "used_for_training": evaluation.get("used_for_training"),
        "used_for_evaluation": evaluation.get("used_for_evaluation"),
        "metrics": evaluation.get("benchmark_metrics") or [],
        "release_status": availability.get("release_status") or "",
        "license": availability.get("license") or "",
    }


def paper_text(record: dict[str, Any]) -> str:
    return paper_summary_text(record)


def paper_abstract_text(record: dict[str, Any]) -> str:
    return "\n".join(
        str(part).strip()
        for part in [
            record.get("title"),
            record.get("abstract"),
        ]
        if str(part or "").strip()
    )


def paper_summary_text(record: dict[str, Any]) -> str:
    return "\n".join(
        str(part).strip()
        for part in [
            record.get("title"),
            record.get("abstract"),
            record.get("paper_contribution_summary"),
        ]
        if str(part or "").strip()
    )


def format_list(values: Any, max_items: int = 6) -> str:
    if not isinstance(values, list):
        return str(values or "")
    cleaned = [str(value).strip() for value in values if str(value or "").strip()]
    return ", ".join(cleaned[:max_items])


def acu_text(dcu: dict[str, Any]) -> str:
    return str(dcu.get("text") or "")


def typed_acu_text(dcu: dict[str, Any]) -> str:
    return f"[type={dcu.get('type') or 'unknown'}] {acu_text(dcu)}"


def dcu_text(dcu: dict[str, Any]) -> str:
    meta = dcu.get("metadata") or {}
    fields = [
        f"paper title: {dcu.get('prior_paper_title') or dcu.get('query_title') or ''}",
        f"dataset: {meta.get('dataset_name') or dcu.get('prior_dataset_name') or ''}",
        f"unit type: {dcu.get('type') or ''}",
        f"role: {meta.get('role') or ''}",
        f"resource type: {meta.get('resource_type') or ''}",
        f"primary use: {meta.get('primary_use') or ''}",
        f"tasks: {format_list(meta.get('tasks'))}",
        f"domains: {format_list(meta.get('domains'))}",
        f"languages: {format_list(meta.get('languages'))}",
        f"modalities: {format_list(meta.get('modalities'))}",
        f"source origin: {meta.get('source_data_origin') or ''}",
        f"source datasets: {format_list(meta.get('source_datasets'))}",
        f"collection method: {meta.get('collection_method') or ''}",
        f"annotation protocol: {meta.get('annotation_protocol') or ''}",
        f"quality control: {meta.get('quality_control') or ''}",
        f"scale: {meta.get('scale') or ''}",
        f"metrics: {format_list(meta.get('metrics'))}",
        f"release: {meta.get('release_status') or ''}",
        f"license: {meta.get('license') or ''}",
        f"claim: {dcu.get('text') or ''}",
    ]
    return "\n".join(field for field in fields if not field.endswith(": "))


def contextual_dcu_text(dcu: dict[str, Any]) -> str:
    meta = dcu.get("metadata") or {}
    context = (
        f"{meta.get('dataset_name') or dcu.get('prior_dataset_name') or ''} is a "
        f"{meta.get('resource_type') or 'dataset'} for {format_list(meta.get('tasks'), 4)} "
        f"in {format_list(meta.get('domains'), 4)}. "
        f"It uses {meta.get('source_data_origin') or format_list(meta.get('source_datasets'), 4)} "
        f"and is constructed via {meta.get('collection_method') or meta.get('annotation_protocol') or 'unspecified methods'}."
    )
    return "\n".join([
        context,
        f"[type={dcu.get('type') or 'unknown'}]",
        f"claim: {dcu.get('text') or ''}",
        f"scale: {meta.get('scale') or ''}",
    ])


def query_dcu_from_row(row: dict[str, Any], query_acu: dict[str, Any]) -> dict[str, Any]:
    meta = dict(row.get("query_metadata") or {})
    meta.setdefault("dataset_name", row.get("query_dataset_name") or "")
    return {
        "id": f"{row.get('benchmark_id') or row.get('query_paper_id')}::{query_acu.get('id')}",
        "text": query_acu.get("text") or "",
        "type": query_acu.get("type") or "",
        "query_title": row.get("query_title") or meta.get("paper_title") or "",
        "prior_dataset_name": meta.get("dataset_name") or row.get("query_dataset_name") or "",
        "metadata": meta,
    }


def flatten_prior_acus(prior_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    papers: dict[str, dict[str, Any]] = {}
    for paper in prior_rows:
        paper_id = str(paper.get("paper_id") or "")
        papers[paper_id] = {
            "paper_id": paper_id,
            "title": paper.get("title") or "",
            "abstract": paper.get("abstract") or "",
            "paper_contribution_summary": paper.get("paper_contribution_summary") or "",
            "pdf_url": paper.get("pdf_url") or "",
            "paper_abstract_text": paper_abstract_text(paper),
            "paper_summary_text": paper_summary_text(paper),
            "text": paper_summary_text(paper),
        }
        for dataset_index, dataset in enumerate(paper.get("datasets") or []):
            dataset_id = str(dataset.get("dataset_id") or dataset_index)
            meta = dataset_metadata(dataset)
            for acu_index, acu in enumerate(dataset.get("acus") or []):
                text = str(acu.get("text") or "").strip()
                if not text:
                    continue
                acu_type = str(acu.get("type") or "")
                global_id = stable_acu_key(paper_id, dataset_id, acu_type, text)
                output.append({
                    "id": global_id,
                    "text": text,
                    "type": acu_type,
                    "importance": acu.get("importance") or "",
                    "prior_paper_id": paper_id,
                    "prior_paper_title": paper.get("title") or "",
                    "prior_dataset_id": dataset_id,
                    "prior_dataset_name": meta.get("dataset_name") or "",
                    "source_acu_id": acu.get("id") or f"a{acu_index}",
                    "evidence": acu.get("evidence") or "",
                    "section": acu.get("section") or "",
                    "metadata": meta,
                })
    return output, papers


def metadata_score(query_meta: dict[str, Any], prior_meta: dict[str, Any]) -> float:
    score = 0.0
    score += 0.18 * jaccard(list_tokens(query_meta.get("tasks")), list_tokens(prior_meta.get("tasks")))
    score += 0.12 * jaccard(list_tokens(query_meta.get("domains")), list_tokens(prior_meta.get("domains")))
    score += 0.10 * jaccard(list_tokens(query_meta.get("languages")), list_tokens(prior_meta.get("languages")))
    score += 0.10 * jaccard(list_tokens(query_meta.get("modalities")), list_tokens(prior_meta.get("modalities")))
    score += 0.17 * jaccard(list_tokens(query_meta.get("source_datasets")), list_tokens(prior_meta.get("source_datasets")))
    score += 0.09 * jaccard(tokenize(str(query_meta.get("source_data_origin") or "")), tokenize(str(prior_meta.get("source_data_origin") or "")))
    score += 0.08 * jaccard(tokenize(str(query_meta.get("annotation_protocol") or "")), tokenize(str(prior_meta.get("annotation_protocol") or "")))
    score += 0.06 * jaccard(tokenize(str(query_meta.get("collection_method") or "")), tokenize(str(prior_meta.get("collection_method") or "")))
    score += 0.05 * jaccard(tokenize(str(query_meta.get("quality_control") or "")), tokenize(str(prior_meta.get("quality_control") or "")))
    score += 0.05 * jaccard(tokenize(str(query_meta.get("dataset_name") or "")), tokenize(str(prior_meta.get("dataset_name") or "")))
    return score


def heuristic_score(query_acu: dict[str, Any], query_meta: dict[str, Any], prior_acu: dict[str, Any], method: str) -> float:
    q_tokens = tokenize(query_acu.get("text") or "")
    p_tokens = tokenize(prior_acu.get("text") or "")
    text_score = 0.65 * containment(q_tokens, p_tokens) + 0.35 * jaccard(q_tokens, p_tokens)
    type_bonus = 0.18 if query_acu.get("type") and query_acu.get("type") == prior_acu.get("type") else 0.0
    if method == "acu_text":
        return text_score
    if method == "acu_type_text":
        return text_score + type_bonus
    if method == "dcu_metadata":
        return 0.48 * text_score + type_bonus + 0.52 * metadata_score(query_meta, prior_acu.get("metadata") or {})
    raise ValueError(f"Unknown heuristic method: {method}")


class DenseModel:
    _model = None

    @classmethod
    def get(cls):
        if cls._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("sentence-transformers is required for dense retrieval methods.") from exc
            local = os.environ.get("BENCHMARK_DENSE_MODEL_PATH") or find_local_minilm_snapshot()
            cls._model = SentenceTransformer(local or "sentence-transformers/all-MiniLM-L6-v2")
        return cls._model


def find_local_minilm_snapshot() -> str | None:
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    if not snapshots.exists():
        return None
    candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
    return str(candidates[-1]) if candidates else None


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe[:180] or "paper"


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required to extract PDF text.") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def condense_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head_chars = int(max_chars * 0.72)
    tail_chars = max_chars - head_chars
    return f"{text[:head_chars]}\n\n[... middle truncated ...]\n\n{text[-tail_chars:]}", True


def download_pdf_to_path(url: str, pdf_path: Path, timeout: int) -> None:
    if requests is None:
        raise RuntimeError("requests is required to download PDF full text.")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with pdf_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)


def load_cached_full_paper_text(
    paper: dict[str, Any],
    *,
    cache_dir: Path,
    timeout: int,
    max_chars: int,
) -> tuple[str, dict[str, Any]]:
    paper_id = str(paper.get("paper_id") or "")
    url = str(paper.get("pdf_url") or "")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{safe_filename(paper_id)}.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return str(payload.get("text") or ""), payload
        except json.JSONDecodeError:
            cache_path.unlink()
    if not url:
        payload = {
            "paper_id": paper_id,
            "pdf_url": url,
            "text": "",
            "error": "missing_pdf_url",
        }
        cache_path.write_text(json.dumps(strip_invalid_unicode(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return "", payload
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "paper.pdf"
            download_pdf_to_path(url, pdf_path, timeout)
            raw_text = pdf_to_text(pdf_path)
        text, truncated = condense_text(raw_text, max_chars)
        payload = {
            "paper_id": paper_id,
            "pdf_url": url,
            "text": text,
            "char_count": len(raw_text),
            "truncated": truncated,
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "paper_id": paper_id,
            "pdf_url": url,
            "text": "",
            "error": str(exc),
        }
        return "", payload
    payload = strip_invalid_unicode(payload)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(payload.get("text") or ""), payload


def hydrate_full_paper_texts(
    papers: dict[str, dict[str, Any]],
    *,
    cache_dir: str | Path,
    timeout: int,
    max_chars: int,
) -> dict[str, Any]:
    summary = {"attempted": 0, "loaded": 0, "failed": 0, "cache_dir": str(cache_dir)}
    for paper in papers.values():
        summary["attempted"] += 1
        text, payload = load_cached_full_paper_text(
            paper,
            cache_dir=Path(cache_dir),
            timeout=timeout,
            max_chars=max_chars,
        )
        if text:
            summary["loaded"] += 1
            paper["paper_full_text"] = text
        else:
            summary["failed"] += 1
            paper["paper_full_text"] = paper.get("paper_summary_text") or paper.get("text") or ""
            paper["paper_full_text_error"] = payload.get("error") or "empty_text"
    return summary


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_v = min(scores)
    max_v = max(scores)
    if max_v - min_v < 1e-8:
        return [1.0 for _ in scores]
    return [(score - min_v) / (max_v - min_v) for score in scores]


def bm25_scores(query_text: str, candidate_texts: list[str]) -> list[float]:
    if BM25Okapi is None:
        raise RuntimeError("rank_bm25 is required for BM25 retrieval methods.")
    bm25 = BM25Okapi([tokenize_list(text) for text in candidate_texts])
    return [float(score) for score in bm25.get_scores(tokenize_list(query_text))]


def dense_scores(query_text: str, candidate_texts: list[str]) -> list[float]:
    if np is None:
        raise RuntimeError("numpy is required for dense retrieval methods.")
    model = DenseModel.get()
    doc_embeddings = model.encode(
        candidate_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return [float(score) for score in (doc_embeddings @ query_embedding)]


class TextIndex:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self.texts = texts
        self._bm25 = None
        self._doc_embeddings = None

    @property
    def bm25(self):
        if self._bm25 is None:
            if BM25Okapi is None:
                raise RuntimeError("rank_bm25 is required for BM25 retrieval methods.")
            self._bm25 = BM25Okapi([tokenize_list(text) for text in self.texts])
        return self._bm25

    @property
    def doc_embeddings(self):
        if self._doc_embeddings is None:
            if np is None:
                raise RuntimeError("numpy is required for dense retrieval methods.")
            model = DenseModel.get()
            self._doc_embeddings = model.encode(
                self.texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return self._doc_embeddings

    def bm25_scores(self, query_text: str) -> list[float]:
        return [float(score) for score in self.bm25.get_scores(tokenize_list(query_text))]

    def dense_scores(self, query_text: str) -> list[float]:
        model = DenseModel.get()
        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return [float(score) for score in (self.doc_embeddings @ query_embedding)]

    def score(self, query_text: str, backend: str) -> list[tuple[str, float]]:
        if backend == "bm25":
            scores = self.bm25_scores(query_text)
        elif backend == "dense":
            scores = self.dense_scores(query_text)
        elif backend == "hybrid":
            bm25 = normalize_scores(self.bm25_scores(query_text))
            dense = normalize_scores(self.dense_scores(query_text))
            scores = [0.5 * left + 0.5 * right for left, right in zip(bm25, dense)]
        else:
            raise ValueError(f"Unknown backend: {backend}")
        return list(zip(self.ids, scores))


def representation_for_method(method: str) -> str:
    if "typed_acu" in method:
        return "typed_acu"
    if method.endswith("_acu") or method in {"bm25_acu", "dense_acu", "hybrid_acu"}:
        return "acu"
    if method in {"bm25_dcu", "dense_dcu", "hybrid_dcu", "hybrid_dcu_support_rerank"}:
        return "dcu"
    if method == "contextual_dcu":
        return "contextual_dcu"
    if method.startswith("paper_"):
        return "paper"
    return "dcu"


def serialize(dcu: dict[str, Any], representation: str) -> str:
    if representation == "acu":
        return acu_text(dcu)
    if representation == "typed_acu":
        return typed_acu_text(dcu)
    if representation == "dcu":
        return dcu_text(dcu)
    if representation == "contextual_dcu":
        return contextual_dcu_text(dcu)
    raise ValueError(f"Unsupported representation: {representation}")


def score_retrieval_method(
    method: str,
    query_dcu: dict[str, Any],
    candidates: list[dict[str, Any]],
    papers: dict[str, dict[str, Any]],
    indexes: dict[str, TextIndex],
) -> list[tuple[str, float]]:
    if method in RANDOM_METHODS:
        seed_text = f"{query_dcu.get('id') or query_dcu.get('text') or ''}::{REPRESENTATION_VERSION}"
        rng = random.Random(stable_cache_key(seed_text))
        return [(candidate["id"], rng.random()) for candidate in candidates]

    if method in HEURISTIC_METHODS:
        return [
            (candidate["id"], heuristic_score(query_dcu, query_dcu.get("metadata") or {}, candidate, method))
            for candidate in candidates
        ]

    if method.startswith("paper_"):
        query_text = "\n".join([
            str(query_dcu.get("query_title") or ""),
            str((query_dcu.get("metadata") or {}).get("paper_title") or ""),
            str((query_dcu.get("metadata") or {}).get("dataset_name") or ""),
            str(query_dcu.get("text") or ""),
        ])
        paper_method_alias = {
            "paper_bm25": "paper_summary_bm25",
            "paper_dense": "paper_summary_dense",
            "paper_hybrid": "paper_summary_hybrid",
        }.get(method, method)
        if "_abstract_" in paper_method_alias:
            index_name = "paper_abstract"
        elif "_full_" in paper_method_alias:
            index_name = "paper_full"
        else:
            index_name = "paper_summary"

        if paper_method_alias.endswith("_bm25"):
            scored = indexes[index_name].score(query_text, "bm25")
        elif paper_method_alias.endswith("_dense"):
            scored = indexes[index_name].score(query_text, "dense")
        elif paper_method_alias.endswith("_hybrid"):
            scored = indexes[index_name].score(query_text, "hybrid")
        else:
            raise ValueError(f"Unknown paper method: {method}")
        paper_score = dict(scored)
        return [
            (candidate["id"], paper_score.get(candidate.get("prior_paper_id"), 0.0))
            for candidate in candidates
        ]

    representation = representation_for_method(method)
    query_text = serialize(query_dcu, representation)
    if method.startswith("bm25_") or method == "contextual_dcu":
        return indexes[representation].score(query_text, "bm25")
    elif method.startswith("dense_"):
        return indexes[representation].score(query_text, "dense")
    elif method.startswith("hybrid_"):
        return indexes[representation].score(query_text, "hybrid")
    else:
        raise ValueError(f"Unknown retrieval method: {method}")


def stable_rank(scored: list[tuple[str, float]], candidate_by_id: dict[str, dict[str, Any]]) -> list[str]:
    return [
        candidate_id
        for candidate_id, _ in sorted(
            scored,
            key=lambda item: (
                -item[1],
                candidate_by_id[item[0]].get("prior_paper_id") or "",
                candidate_by_id[item[0]].get("prior_dataset_name") or "",
                candidate_by_id[item[0]].get("text") or "",
                item[0],
            ),
        )
    ]


def local_gold_to_global_ids(row: dict[str, Any], label: dict[str, Any]) -> list[str]:
    local_by_id = {acu.get("id"): acu for acu in row.get("prior_acu_bank") or []}
    output: list[str] = []
    for local_id in label.get("selected_prior_acu_ids") or []:
        acu = local_by_id.get(local_id)
        if not acu:
            continue
        output.append(stable_acu_key(
            str(acu.get("prior_paper_id") or ""),
            str(acu.get("prior_dataset_id") or ""),
            str(acu.get("type") or ""),
            str(acu.get("text") or ""),
        ))
    return output


class JsonCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, namespace: str, key: Any) -> Path:
        return self.cache_dir / namespace / f"{stable_cache_key(key)}.json"

    def get(self, namespace: str, key: Any) -> dict[str, Any] | None:
        path = self.path_for(namespace, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, namespace: str, key: Any, payload: dict[str, Any]) -> dict[str, Any]:
        path = self.path_for(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def ensure_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for hybrid_dcu_support_rerank.") from exc
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ensure_gemini_client():
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-genai is required for Gemini support reranking.") from exc
    return genai.Client(), genai_types


def call_json_llm(prompt: str, *, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if model.startswith("gemini"):
        client, genai_types = ensure_gemini_client()
        response = client.models.generate_content(
            model=model.removeprefix("gemini/"),
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        usage_dict: dict[str, Any] = {}
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()
            elif hasattr(usage, "dict"):
                usage_dict = usage.dict()
        return parse_json_object(response.text or ""), usage_dict

    client = ensure_openai_client()
    response = client.responses.create(model=model.removeprefix("openai/"), input=prompt)
    usage = getattr(response, "usage", None)
    usage_dict: dict[str, Any] = {}
    if usage is not None:
        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif isinstance(usage, dict):
            usage_dict = usage
    return parse_json_object(response.output_text), usage_dict


def dcu_brief(dcu: dict[str, Any]) -> str:
    meta = dcu.get("metadata") or {}
    return (
        f"DCU ID: {dcu['id']}\n"
        f"Paper: {dcu.get('prior_paper_title') or dcu.get('query_title') or ''}\n"
        f"Dataset: {meta.get('dataset_name') or dcu.get('prior_dataset_name') or ''}\n"
        f"Type: {dcu.get('type') or ''}\n"
        f"Task/domain: {format_list(meta.get('tasks'), 4)} / {format_list(meta.get('domains'), 4)}\n"
        f"Source/annotation: {meta.get('source_data_origin') or ''} {meta.get('annotation_protocol') or ''}\n"
        f"Scale: {meta.get('scale') or ''}\n"
        f"Claim: {dcu.get('text') or ''}"
    )


def support_rerank(
    query_dcu: dict[str, Any],
    candidate_ids: list[str],
    candidate_by_id: dict[str, dict[str, Any]],
    *,
    model: str,
    cache: JsonCache,
) -> list[str]:
    key = {
        "model": model,
        "representation_version": REPRESENTATION_VERSION,
        "prompt_version": RERANK_PROMPT_VERSION,
        "query_id": query_dcu.get("id"),
        "query_text": query_dcu.get("text"),
        "query_type": query_dcu.get("type"),
        "candidate_ids": candidate_ids,
    }
    cached = cache.get("support_rerank", key)
    if cached is not None:
        return [candidate_id for candidate_id in cached.get("ranking", []) if candidate_id in candidate_ids]

    candidate_block = "\n\n".join(dcu_brief(candidate_by_id[candidate_id]) for candidate_id in candidate_ids)
    prompt = (
        "Rank the candidate prior Dataset Contribution Units (DCUs) for the query DCU.\n"
        "This is prior-support retrieval, not ordinary topical relevance. Rank a candidate higher when it directly or partially supports the query claim, "
        "or provides close prior evidence for the same dataset contribution dimension. Surface similarity alone is insufficient.\n"
        "Return JSON only with schema: {\"ranking\": [\"dcu_id\", ...]}.\n\n"
        f"Query DCU:\n{dcu_brief(query_dcu)}\n\n"
        f"Candidate prior DCUs:\n{candidate_block}"
    )
    parsed, usage = call_json_llm(prompt, model=model)
    ranking = [candidate_id for candidate_id in parsed.get("ranking", []) if candidate_id in candidate_ids]
    ranking = unique(ranking + candidate_ids)
    return cache.set("support_rerank", key, {"ranking": ranking, "usage": usage})["ranking"]


def unique(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def dcg_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    score = 0.0
    for index, candidate_id in enumerate(ranking[:k], start=1):
        if candidate_id in gold:
            score += 1.0 / math.log2(index + 1)
    return score


def query_metrics(ranking: list[str], gold_ids: list[str], top_ks: list[int], ndcg_ks: list[int]) -> dict[str, float]:
    gold = set(gold_ids)
    first_rank = None
    for index, candidate_id in enumerate(ranking, start=1):
        if candidate_id in gold:
            first_rank = index
            break
    metrics = {
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "gold_count": len(gold),
    }
    for k in top_ks:
        hits = sum(1 for candidate_id in ranking[:k] if candidate_id in gold)
        metrics[f"hit@{k}"] = 1.0 if hits else 0.0
        metrics[f"recall@{k}"] = metrics[f"hit@{k}"]
        metrics[f"set_recall@{k}"] = hits / len(gold) if gold else 0.0
        metrics[f"precision@{k}"] = hits / k if k else 0.0
    for ndcg_k in ndcg_ks:
        ideal_hits = min(len(gold), ndcg_k)
        ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
        metrics[f"ndcg@{ndcg_k}"] = dcg_at_k(ranking, gold, ndcg_k) / ideal if ideal else 0.0
    return metrics


def summarize(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {"n": 0}
    keys = sorted({key for item in items for key in item})
    return {"n": len(items), **{key: sum(float(item.get(key, 0.0)) for item in items) / len(items) for key in keys}}


def prepare_queries(claim_rows: list[dict[str, Any]], candidate_ids: set[str], limit: int | None) -> tuple[list[dict[str, Any]], int]:
    queries: list[dict[str, Any]] = []
    missing_gold = 0
    for row in claim_rows:
        query_by_id = row.get("query_acu_by_id") or {acu["id"]: acu for acu in row.get("query_acus") or []}
        for label in row.get("labels") or []:
            if not label.get("evaluate"):
                continue
            query_acu = query_by_id.get(label.get("query_acu_id"))
            if not query_acu:
                continue
            gold = [gid for gid in local_gold_to_global_ids(row, label) if gid in candidate_ids]
            if not gold:
                missing_gold += 1
                continue
            queries.append({
                "row": row,
                "query_dcu": query_dcu_from_row(row, query_acu),
                "label": label,
                "gold_ids": gold,
            })
            if limit is not None and len(queries) >= limit:
                return queries, missing_gold
    return queries, missing_gold


def evaluate(
    claim_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    papers: dict[str, dict[str, Any]],
    methods: list[str],
    top_ks: list[int],
    ndcg_ks: list[int],
    *,
    limit: int | None,
    rerank_depth: int,
    model: str,
    cache_dir: str,
    full_paper_cache_dir: str,
    download_timeout: int,
    full_paper_max_chars: int,
) -> dict[str, Any]:
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    queries, missing_gold = prepare_queries(claim_rows, set(candidate_by_id), limit)
    needs_full_paper = any(method.startswith("paper_full_") for method in methods)
    full_paper_summary = None
    if needs_full_paper:
        full_paper_summary = hydrate_full_paper_texts(
            papers,
            cache_dir=full_paper_cache_dir,
            timeout=download_timeout,
            max_chars=full_paper_max_chars,
        )
    candidate_ids = [candidate["id"] for candidate in candidates]
    paper_ids = sorted(papers)
    indexes = {
        "paper_abstract": TextIndex(paper_ids, [papers[paper_id].get("paper_abstract_text") or "" for paper_id in paper_ids]),
        "paper_summary": TextIndex(paper_ids, [papers[paper_id].get("paper_summary_text") or "" for paper_id in paper_ids]),
        "paper_full": TextIndex(paper_ids, [papers[paper_id].get("paper_full_text") or papers[paper_id].get("paper_summary_text") or "" for paper_id in paper_ids]),
        "acu": TextIndex(candidate_ids, [serialize(candidate, "acu") for candidate in candidates]),
        "typed_acu": TextIndex(candidate_ids, [serialize(candidate, "typed_acu") for candidate in candidates]),
        "dcu": TextIndex(candidate_ids, [serialize(candidate, "dcu") for candidate in candidates]),
        "contextual_dcu": TextIndex(candidate_ids, [serialize(candidate, "contextual_dcu") for candidate in candidates]),
    }
    metric_rows = {method: [] for method in methods}
    by_delta = {method: defaultdict(list) for method in methods}
    examples = {method: [] for method in methods}
    cache = JsonCache(cache_dir)

    for index, item in enumerate(queries, start=1):
        query_dcu = item["query_dcu"]
        label = item["label"]
        gold = item["gold_ids"]
        base_rankings: dict[str, list[str]] = {}
        for method in methods:
            actual_method = "hybrid_dcu" if method == "hybrid_dcu_support_rerank" else method
            if actual_method not in base_rankings:
                scored = score_retrieval_method(actual_method, query_dcu, candidates, papers, indexes)
                base_rankings[actual_method] = stable_rank(scored, candidate_by_id)
            ranking = base_rankings[actual_method]
            if method == "hybrid_dcu_support_rerank":
                pool = ranking[:rerank_depth]
                reranked = support_rerank(query_dcu, pool, candidate_by_id, model=model, cache=cache)
                ranking = unique(reranked + ranking)

            metrics = query_metrics(ranking, gold, top_ks, ndcg_ks)
            metric_rows[method].append(metrics)
            by_delta[method][label.get("delta_type") or "other"].append(metrics)
            first_rank = 1.0 / metrics["mrr"] if metrics["mrr"] else None
            if (not first_rank or first_rank > 20) and len(examples[method]) < 10:
                examples[method].append({
                    "benchmark_id": item["row"].get("benchmark_id"),
                    "query_dataset_name": item["row"].get("query_dataset_name"),
                    "query_acu": query_dcu,
                    "label": label,
                    "gold_global_ids": gold,
                    "first_gold_rank": first_rank,
                    "top10": [
                        {
                            "id": candidate_id,
                            "paper": candidate_by_id[candidate_id].get("prior_paper_title"),
                            "dataset": candidate_by_id[candidate_id].get("prior_dataset_name"),
                            "type": candidate_by_id[candidate_id].get("type"),
                            "text": candidate_by_id[candidate_id].get("text"),
                        }
                        for candidate_id in ranking[:10]
                    ],
                    "gold_prior_dcus": [
                        {
                            "id": gold_id,
                            "paper": candidate_by_id[gold_id].get("prior_paper_title"),
                            "dataset": candidate_by_id[gold_id].get("prior_dataset_name"),
                            "type": candidate_by_id[gold_id].get("type"),
                            "text": candidate_by_id[gold_id].get("text"),
                        }
                        for gold_id in gold
                    ],
                })
        if index % 25 == 0:
            print(json.dumps({"progress": index, "total": len(queries)}, ensure_ascii=False), flush=True)

    return {
        "claim_labels": len(queries),
        "missing_gold_labels": missing_gold,
        "global_prior_acus": len(candidates),
        "global_prior_papers": len(papers),
        "full_paper_cache": full_paper_summary,
        "by_method": {method: summarize(items) for method, items in metric_rows.items()},
        "by_delta_type": {
            method: {delta: summarize(items) for delta, items in delta_rows.items()}
            for method, delta_rows in by_delta.items()
        },
        "miss_examples": examples,
    }


def method_category(method: str) -> str:
    if method in RANDOM_METHODS:
        return "random"
    if method.startswith("paper_"):
        return "paper"
    if method in HEURISTIC_METHODS:
        return "heuristic"
    if "typed_acu" in method:
        return "typed_acu"
    if method.endswith("_acu"):
        return "acu"
    if "dcu" in method:
        return "dcu"
    return "other"


def markdown_report(report: dict[str, Any], top_ks: list[int], ndcg_ks: list[int]) -> str:
    metric_columns = ["MRR"] + [f"nDCG@{k}" for k in ndcg_ks] + [f"Hit@{k}" for k in top_ks] + [f"SetR@{k}" for k in top_ks if k in {5, 10, 20, 50}]
    lines = [
        "# Global Claim-Level DCU Retrieval",
        "",
        f"- Claim labels: {report['claim_labels']}",
        f"- Missing gold labels: {report['missing_gold_labels']}",
        f"- Global prior ACUs: {report['global_prior_acus']}",
        f"- Global prior papers: {report.get('global_prior_papers', 0)}",
        "",
        "| Category | Method | N | " + " | ".join(metric_columns) + " |",
        "| --- | --- | ---: | " + " | ".join("---:" for _ in metric_columns) + " |",
    ]
    full_cache = report.get("full_paper_cache")
    if full_cache:
        lines.insert(6, f"- Full paper text loaded: {full_cache.get('loaded', 0)}/{full_cache.get('attempted', 0)}")
    for method, row in report["by_method"].items():
        values = [
            f"{row.get('mrr', 0.0):.3f}",
            *[f"{row.get(f'ndcg@{k}', 0.0):.3f}" for k in ndcg_ks],
            *[f"{row.get(f'hit@{k}', row.get(f'recall@{k}', 0.0)):.3f}" for k in top_ks],
            *[f"{row.get(f'set_recall@{k}', 0.0):.3f}" for k in top_ks if k in {5, 10, 20, 50}],
        ]
        lines.append(f"| {method_category(method)} | {method} | {row.get('n', 0)} | " + " | ".join(values) + " |")

    lines.extend(["", "## Per Delta Type", ""])
    for method, deltas in report.get("by_delta_type", {}).items():
        lines.append(f"### {method}")
        lines.append("| Delta type | N | MRR | Hit@10 | Hit@20 | SetR@10 | SetR@20 | nDCG@10 | nDCG@50 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for delta, row in sorted(deltas.items()):
            lines.append(
                f"| {delta} | {row.get('n', 0)} | {row.get('mrr', 0.0):.3f} | "
                f"{row.get('hit@10', row.get('recall@10', 0.0)):.3f} | "
                f"{row.get('hit@20', row.get('recall@20', 0.0)):.3f} | "
                f"{row.get('set_recall@10', 0.0):.3f} | "
                f"{row.get('set_recall@20', 0.0):.3f} | "
                f"{row.get('ndcg@10', 0.0):.3f} | "
                f"{row.get('ndcg@50', 0.0):.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate global claim-level DCU retrieval against LLM-selected prior ACU labels.")
    parser.add_argument("--claim-level-jsonl", required=True)
    parser.add_argument("--prior-extractions-jsonl", required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--ndcg-ks", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--cache-dir", default="data/benchmark/retrieval_cache/global_claim_level_dcu_rerank_cache")
    parser.add_argument("--full-paper-cache-dir", default="data/benchmark/retrieval_cache/full_paper_text_cache")
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--full-paper-max-chars", type=int, default=120000)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    unknown = sorted(set(args.methods) - HEURISTIC_METHODS - RETRIEVAL_METHODS - RANDOM_METHODS)
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    if any("bm25" in method or "hybrid" in method or method == "contextual_dcu" for method in args.methods):
        if BM25Okapi is None:
            raise SystemExit("rank_bm25 is required for BM25/hybrid methods.")
    if any(method.startswith("paper_full_") for method in args.methods) and requests is None:
        raise SystemExit("requests is required for paper_full_* methods.")

    claim_rows = read_jsonl(args.claim_level_jsonl)
    prior_rows = read_jsonl(args.prior_extractions_jsonl)
    candidates, papers = flatten_prior_acus(prior_rows)
    report = evaluate(
        claim_rows,
        candidates,
        papers,
        args.methods,
        args.top_ks,
        args.ndcg_ks,
        limit=args.limit,
        rerank_depth=args.rerank_depth,
        model=args.model,
        cache_dir=args.cache_dir,
        full_paper_cache_dir=args.full_paper_cache_dir,
        download_timeout=args.download_timeout,
        full_paper_max_chars=args.full_paper_max_chars,
    )
    report.update({
        "claim_level_jsonl": args.claim_level_jsonl,
        "prior_extractions_jsonl": args.prior_extractions_jsonl,
        "methods": args.methods,
        "top_ks": args.top_ks,
        "ndcg_ks": args.ndcg_ks,
        "limit": args.limit,
        "model": args.model,
        "rerank_depth": args.rerank_depth,
        "cache_dir": args.cache_dir,
        "full_paper_cache_dir": args.full_paper_cache_dir,
        "download_timeout": args.download_timeout,
        "full_paper_max_chars": args.full_paper_max_chars,
        "representation_version": REPRESENTATION_VERSION,
    })

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        output_markdown = Path(args.output_markdown)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(markdown_report(report, args.top_ks, args.ndcg_ks), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "output_markdown": args.output_markdown,
        "claim_labels": report["claim_labels"],
        "missing_gold_labels": report["missing_gold_labels"],
        "global_prior_acus": report["global_prior_acus"],
        "global_prior_papers": report["global_prior_papers"],
        "by_method": report["by_method"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
