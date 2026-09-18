#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required for this script.") from exc

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required for this script.") from exc

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def year_int(row: dict[str, Any]) -> int | None:
    value = row.get("year")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_allowed_prior(query: dict[str, Any], acu: dict[str, Any], mode: str) -> bool:
    if query.get("paper_id") == acu.get("paper_id"):
        return False
    query_year = year_int(query)
    prior_year = year_int(acu)
    if mode == "any":
        return True
    if query_year is None or prior_year is None:
        return False
    if mode == "earlier":
        return prior_year < query_year
    if mode == "earlier_or_same":
        return prior_year <= query_year
    raise ValueError(f"Unsupported year mode: {mode}")


def query_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("dataset_name"),
        row.get("title"),
        row.get("role"),
        row.get("resource_type"),
        " ".join(row.get("tasks") or []),
        " ".join(row.get("domains") or []),
        " ".join(row.get("languages") or []),
        " ".join(row.get("modalities") or []),
        row.get("usage_description"),
        row.get("added_information_summary"),
        row.get("search_text"),
    ]
    return "\n".join(str(part).strip() for part in parts if str(part or "").strip())


def candidate_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("dataset_name"),
        row.get("title"),
        row.get("dataset_role"),
        row.get("resource_type"),
        row.get("acu_type"),
        row.get("acu_text"),
        row.get("evidence"),
        row.get("search_text"),
    ]
    return "\n".join(str(part).strip() for part in parts if str(part or "").strip())


def format_list(values: Any, max_items: int = 8) -> str:
    if not isinstance(values, list):
        return str(values or "")
    cleaned = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("dataset_name") or value.get("relationship") or json.dumps(value, sort_keys=True)
        text = str(value or "").strip()
        if text and text != "unclear":
            cleaned.append(text)
    return ", ".join(cleaned[:max_items])


def scale_text(row: dict[str, Any]) -> str:
    scale = row.get("scale") or {}
    if not isinstance(scale, dict):
        return str(scale or "")
    return " ".join(
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
    )


def query_dcu_text(row: dict[str, Any], acu: dict[str, Any]) -> str:
    parts = [
        f"paper title: {row.get('title') or ''}",
        f"year: {row.get('year') or ''}",
        f"dataset: {row.get('dataset_name') or ''}",
        f"unit type: {acu.get('type') or acu.get('acu_type') or ''}",
        f"role: {row.get('role') or ''}",
        f"resource type: {row.get('resource_type') or ''}",
        f"primary use: {row.get('primary_use') or ''}",
        f"tasks: {format_list(row.get('tasks'))}",
        f"domains: {format_list(row.get('domains'))}",
        f"languages: {format_list(row.get('languages'))}",
        f"modalities: {format_list(row.get('modalities'))}",
        f"source origin: {row.get('source_data_origin') or ''}",
        f"source datasets: {format_list(row.get('source_datasets'))}",
        f"collection method: {row.get('collection_method') or ''}",
        f"annotation protocol: {row.get('annotation_protocol') or ''}",
        f"quality control: {row.get('quality_control') or ''}",
        f"scale: {scale_text(row)}",
        f"release: {row.get('release_status') or ''}",
        f"license: {row.get('license') or ''}",
        f"claim: {acu.get('text') or acu.get('acu_text') or ''}",
        f"evidence: {acu.get('evidence') or ''}",
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))


def prior_dcu_text(acu: dict[str, Any], dataset: dict[str, Any] | None) -> str:
    dataset = dataset or {}
    parts = [
        f"paper title: {dataset.get('title') or acu.get('title') or ''}",
        f"year: {dataset.get('year') or acu.get('year') or ''}",
        f"dataset: {dataset.get('dataset_name') or acu.get('dataset_name') or ''}",
        f"unit type: {acu.get('acu_type') or ''}",
        f"role: {dataset.get('role') or acu.get('dataset_role') or ''}",
        f"resource type: {dataset.get('resource_type') or acu.get('resource_type') or ''}",
        f"primary use: {dataset.get('primary_use') or ''}",
        f"tasks: {format_list(dataset.get('tasks'))}",
        f"domains: {format_list(dataset.get('domains'))}",
        f"languages: {format_list(dataset.get('languages'))}",
        f"modalities: {format_list(dataset.get('modalities'))}",
        f"source origin: {dataset.get('source_data_origin') or ''}",
        f"source datasets: {format_list(dataset.get('source_datasets'))}",
        f"collection method: {dataset.get('collection_method') or ''}",
        f"annotation protocol: {dataset.get('annotation_protocol') or ''}",
        f"quality control: {dataset.get('quality_control') or ''}",
        f"scale: {scale_text(dataset)}",
        f"release: {dataset.get('release_status') or ''}",
        f"license: {dataset.get('license') or ''}",
        f"claim: {acu.get('acu_text') or ''}",
        f"evidence: {acu.get('evidence') or ''}",
        f"section: {acu.get('section') or ''}",
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]


def tokenize_list(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_v = float(scores.min())
    max_v = float(scores.max())
    if max_v - min_v < 1e-8:
        return np.ones_like(scores, dtype=float)
    return (scores - min_v) / (max_v - min_v)


def find_local_minilm_snapshot() -> str | None:
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    if not snapshots.exists():
        return None
    candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
    return str(candidates[-1]) if candidates else None


class DenseModel:
    _model = None

    @classmethod
    def get(cls):
        if cls._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("sentence-transformers is required for hybrid_dcu retrieval.") from exc
            local = os.environ.get("BENCHMARK_DENSE_MODEL_PATH") or find_local_minilm_snapshot()
            cls._model = SentenceTransformer(local or "sentence-transformers/all-MiniLM-L6-v2")
        return cls._model


class HybridDcuIndex:
    def __init__(
        self,
        acu_rows: list[dict[str, Any]],
        dataset_by_bank_id: dict[str, dict[str, Any]],
        *,
        embedding_cache: str | Path | None,
        dense_batch_size: int,
        max_text_chars: int,
    ):
        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 is required for hybrid_dcu retrieval.")
        self.acu_rows = acu_rows
        self.texts = [
            prior_dcu_text(acu, dataset_by_bank_id.get(str(acu.get("bank_id") or "")))[:max_text_chars]
            for acu in acu_rows
        ]
        self.bm25 = BM25Okapi([tokenize_list(text) for text in self.texts])
        self.dense_batch_size = dense_batch_size
        self.max_text_chars = max_text_chars
        self.doc_embeddings = self.load_or_encode_embeddings(embedding_cache)
        self.tokenized_texts = [tokenize_list(text) for text in self.texts]

    def cache_key(self) -> str:
        digest = hashlib.sha1()
        for row, text in zip(self.acu_rows, self.texts):
            digest.update(str(row.get("acu_global_id") or "").encode("utf-8"))
            digest.update(b"\0")
            digest.update(text.encode("utf-8", "replace"))
            digest.update(b"\0")
        payload = {
            "version": "hybrid_dcu_embeddings_v1",
            "model": os.environ.get("BENCHMARK_DENSE_MODEL_PATH") or "sentence-transformers/all-MiniLM-L6-v2",
            "max_text_chars": self.max_text_chars,
            "corpus_sha1": digest.hexdigest(),
            "n": len(self.texts),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def load_or_encode_embeddings(self, embedding_cache: str | Path | None) -> np.ndarray:
        cache_path = Path(embedding_cache) if embedding_cache else None
        key = self.cache_key()
        if cache_path and cache_path.exists():
            payload = np.load(cache_path, allow_pickle=False)
            cached_key = str(payload["cache_key"].item()) if "cache_key" in payload else ""
            if cached_key == key:
                print(json.dumps({
                    "hybrid_dcu_embedding_cache": str(cache_path),
                    "cache_hit": True,
                    "embeddings": int(payload["embeddings"].shape[0]),
                }, ensure_ascii=False), flush=True)
                return payload["embeddings"]
        model = DenseModel.get()
        embeddings = model.encode(
            self.texts,
            batch_size=self.dense_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, cache_key=np.array(key), embeddings=embeddings)
            print(json.dumps({
                "hybrid_dcu_embedding_cache": str(cache_path),
                "cache_hit": False,
                "embeddings": int(embeddings.shape[0]),
            }, ensure_ascii=False), flush=True)
        return embeddings

    def scores(self, query: str) -> np.ndarray:
        query = query[:self.max_text_chars]
        bm25_scores = normalize_scores(np.asarray(self.bm25.get_scores(tokenize_list(query)), dtype=float))
        model = DenseModel.get()
        query_embedding = model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        dense_scores = normalize_scores(np.asarray(self.doc_embeddings @ query_embedding, dtype=float))
        return 0.5 * bm25_scores + 0.5 * dense_scores

    def dense_scores(self, query: str) -> np.ndarray:
        query = query[:self.max_text_chars]
        model = DenseModel.get()
        query_embedding = model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(self.doc_embeddings @ query_embedding, dtype=float)

    @staticmethod
    def bm25_subset_scores(query_tokens: list[str], tokenized_docs: list[list[str]]) -> np.ndarray:
        if not tokenized_docs:
            return np.asarray([], dtype=float)
        subset_bm25 = BM25Okapi(tokenized_docs)
        return np.asarray(subset_bm25.get_scores(query_tokens), dtype=float)

    def fast_hybrid_scores(
        self,
        query: str,
        *,
        dense_pool_size: int,
        sparse_pool_size: int = 0,
    ) -> list[tuple[int, float]]:
        query = query[:self.max_text_chars]
        query_tokens = tokenize_list(query)
        dense_raw = self.dense_scores(query)
        pool: set[int] = set(int(index) for index in top_indices(dense_raw, min(dense_pool_size, dense_raw.shape[0])))
        if sparse_pool_size > 0:
            # Optional lexical safety net. This is slower than dense-only
            # prefiltering but still cheaper than full hybrid when small.
            sparse_raw = self.bm25.get_scores(query_tokens)
            pool.update(int(index) for index in top_indices(np.asarray(sparse_raw, dtype=float), min(sparse_pool_size, len(sparse_raw))))
        pool_indices = np.asarray(sorted(pool), dtype=int)
        if pool_indices.size == 0:
            return []
        dense_subset = normalize_scores(dense_raw[pool_indices])
        tokenized_subset = [self.tokenized_texts[int(index)] for index in pool_indices]
        bm25_subset = normalize_scores(self.bm25_subset_scores(query_tokens, tokenized_subset))
        hybrid = 0.5 * bm25_subset + 0.5 * dense_subset
        order = np.argsort(-hybrid)
        return [(int(pool_indices[int(pos)]), float(hybrid[int(pos)])) for pos in order]


def normalize_name(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_tokens(value: str) -> set[str]:
    stop = {
        "a", "an", "and", "benchmark", "benchmarks", "corpus", "data", "dataset",
        "datasets", "evaluation", "for", "of", "resource", "resources", "set",
        "the", "to", "with",
    }
    return {tok for tok in normalize_name(value).split() if len(tok) > 1 and tok not in stop}


def dataset_names(row: dict[str, Any]) -> list[str]:
    names = [
        row.get("dataset_name"),
        row.get("dataset_id"),
        row.get("acronym"),
        *(row.get("aliases") or []),
    ]
    output = []
    seen = set()
    for name in names:
        norm = normalize_name(name)
        if not norm or norm == "unclear" or norm in seen:
            continue
        seen.add(norm)
        output.append(str(name))
    return output


def query_explicit_names(row: dict[str, Any]) -> list[dict[str, str]]:
    names = []
    for mention in row.get("prior_dataset_mentions") or []:
        if isinstance(mention, dict) and normalize_name(mention.get("name")):
            names.append({
                "name": str(mention.get("name")),
                "source": f"prior_dataset_mentions:{mention.get('relationship_type') or 'unknown'}",
            })
    for source in row.get("source_datasets") or []:
        if isinstance(source, dict) and normalize_name(source.get("name")):
            names.append({
                "name": str(source.get("name")),
                "source": f"source_datasets:{source.get('relationship') or 'unknown'}",
            })
    seen = set()
    output = []
    for item in names:
        key = (normalize_name(item["name"]), item["source"])
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def build_bank_indexes(dataset_rows: list[dict[str, Any]], acu_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bank_id = {str(row.get("bank_id") or ""): row for row in dataset_rows if row.get("bank_id")}
    acus_by_bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for acu in acu_rows:
        bank_id = str(acu.get("bank_id") or "")
        if bank_id:
            acus_by_bank[bank_id].append(acu)

    exact_name_to_bank_ids: dict[str, set[str]] = defaultdict(set)
    token_to_bank_ids: dict[str, set[str]] = defaultdict(set)
    bank_names: dict[str, list[str]] = {}
    for row in dataset_rows:
        bank_id = str(row.get("bank_id") or "")
        if not bank_id:
            continue
        names = dataset_names(row)
        bank_names[bank_id] = names
        for name in names:
            norm = normalize_name(name)
            if not norm:
                continue
            exact_name_to_bank_ids[norm].add(bank_id)
            for token in name_tokens(norm):
                token_to_bank_ids[token].add(bank_id)
    return {
        "by_bank_id": by_bank_id,
        "acus_by_bank": acus_by_bank,
        "exact_name_to_bank_ids": exact_name_to_bank_ids,
        "token_to_bank_ids": token_to_bank_ids,
        "bank_names": bank_names,
    }


def name_match_score(query_name: str, candidate_name: str) -> float:
    q_norm = normalize_name(query_name)
    c_norm = normalize_name(candidate_name)
    if not q_norm or not c_norm:
        return 0.0
    if q_norm == c_norm:
        return 1.0
    if len(q_norm) >= 5 and len(c_norm) >= 5 and (q_norm in c_norm or c_norm in q_norm):
        return 0.92
    q_tokens = name_tokens(q_norm)
    c_tokens = name_tokens(c_norm)
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    containment = overlap / min(len(q_tokens), len(c_tokens))
    jaccard = overlap / len(q_tokens | c_tokens)
    if containment >= 0.8 and overlap >= 2:
        return 0.85
    if jaccard >= 0.6 and overlap >= 2:
        return 0.75
    return 0.0


def add_candidate_acu(
    item: dict[str, Any],
    acu: dict[str, Any],
    *,
    score: float,
    max_prior_acus_per_dataset: int,
    retrieval_source: str | None = None,
) -> None:
    if len(item["matched_acus"]) >= max_prior_acus_per_dataset:
        return
    seen = {entry.get("acu_global_id") for entry in item["matched_acus"]}
    acu_global_id = acu.get("acu_global_id") or ""
    if acu_global_id and acu_global_id in seen:
        return
    row = {
        "acu_global_id": acu_global_id,
        "acu_id": acu.get("acu_id") or "",
        "acu_text": acu.get("acu_text") or "",
        "acu_type": acu.get("acu_type") or "",
        "importance": acu.get("importance") or "",
        "evidence": acu.get("evidence") or "",
        "section": acu.get("section") or "",
        "score": score,
    }
    if retrieval_source:
        row["retrieval_source"] = retrieval_source
    item["matched_acus"].append(row)


def ensure_candidate_item(
    grouped: dict[str, dict[str, Any]],
    bank_id: str,
    *,
    candidate_dataset: dict[str, Any] | None,
    candidate_acu: dict[str, Any] | None,
    score: float,
    retrieval_source: str,
) -> dict[str, Any]:
    dataset = candidate_dataset or {}
    acu = candidate_acu or {}
    item = grouped.setdefault(bank_id, {
        "prior_bank_id": bank_id,
        "prior_paper_id": dataset.get("paper_id") or acu.get("paper_id") or "",
        "prior_dataset_id": dataset.get("dataset_id") or acu.get("dataset_id") or "",
        "prior_dataset_name": dataset.get("dataset_name") or acu.get("dataset_name") or "",
        "prior_title": dataset.get("title") or acu.get("title") or "",
        "prior_year": dataset.get("year") or acu.get("year"),
        "score": score,
        "retrieval_sources": [],
        "matched_names": [],
        "matched_acus": [],
    })
    item["score"] = max(float(item["score"]), score)
    if retrieval_source not in item["retrieval_sources"]:
        item["retrieval_sources"].append(retrieval_source)
    return item


def add_explicit_matches(
    grouped: dict[str, dict[str, Any]],
    query: dict[str, Any],
    indexes: dict[str, Any],
    *,
    year_mode: str,
    max_prior_acus_per_dataset: int,
    max_explicit_matches_per_name: int,
) -> int:
    matches = 0
    for explicit in query_explicit_names(query):
        explicit_name = explicit["name"]
        norm = normalize_name(explicit_name)
        if not norm:
            continue
        candidate_bank_ids = set(indexes["exact_name_to_bank_ids"].get(norm, set()))
        tokens = name_tokens(norm)
        if tokens:
            # Prefer rarer tokens to keep fuzzy matching precise and cheap.
            token_buckets = sorted(
                (indexes["token_to_bank_ids"].get(token, set()) for token in tokens),
                key=len,
            )
            for bucket in token_buckets[:3]:
                candidate_bank_ids.update(bucket)
        scored = []
        for bank_id in candidate_bank_ids:
            candidate_dataset = indexes["by_bank_id"].get(bank_id)
            if not candidate_dataset:
                continue
            # Check year/self constraints against a representative ACU when possible.
            candidate_acus = indexes["acus_by_bank"].get(bank_id) or []
            representative = candidate_acus[0] if candidate_acus else candidate_dataset
            if not is_allowed_prior(query, representative, year_mode):
                continue
            score = max(name_match_score(explicit_name, name) for name in indexes["bank_names"].get(bank_id, []))
            if score <= 0:
                continue
            scored.append((score, bank_id, candidate_dataset, candidate_acus))
        scored.sort(key=lambda item: item[0], reverse=True)
        for score, bank_id, candidate_dataset, candidate_acus in scored[:max_explicit_matches_per_name]:
            item = ensure_candidate_item(
                grouped,
                bank_id,
                candidate_dataset=candidate_dataset,
                candidate_acu=candidate_acus[0] if candidate_acus else None,
                score=max(1.0, score),
                retrieval_source=explicit["source"],
            )
            item["matched_names"].append({
                "query_name": explicit_name,
                "match_score": score,
                "source": explicit["source"],
            })
            for acu in candidate_acus:
                add_candidate_acu(
                    item,
                    acu,
                    score=max(1.0, score),
                    max_prior_acus_per_dataset=max_prior_acus_per_dataset,
                )
            matches += 1
    return matches


def build_queue(
    dataset_rows: list[dict[str, Any]],
    acu_rows: list[dict[str, Any]],
    *,
    top_k: int,
    candidate_pool_size: int,
    batch_size: int,
    year_mode: str,
    max_prior_acus_per_dataset: int,
    min_score: float,
    limit: int | None,
    use_explicit_matches: bool,
    max_explicit_matches_per_name: int,
    query_year: int | None,
    retrieval_method: str,
    hybrid_dcu_embedding_cache: str | None,
    hybrid_dcu_dense_batch_size: int,
    hybrid_dcu_max_text_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_dataset_rows = list(dataset_rows)
    if query_year is not None:
        dataset_rows = [row for row in dataset_rows if row.get("year") == query_year]
    if limit is not None:
        dataset_rows = dataset_rows[:limit]

    indexes = build_bank_indexes(all_dataset_rows, acu_rows)
    if retrieval_method == "tfidf":
        corpus_texts = [candidate_text(row) for row in acu_rows]
        query_texts = [query_text(row) for row in dataset_rows]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9_\-]+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            norm="l2",
        )
        vectorizer.fit(corpus_texts + query_texts)
        corpus_matrix = vectorizer.transform(corpus_texts)
        hybrid_dcu_index = None
    elif retrieval_method == "hybrid_dcu":
        vectorizer = None
        corpus_matrix = None
        hybrid_dcu_index = HybridDcuIndex(
            acu_rows,
            indexes["by_bank_id"],
            embedding_cache=hybrid_dcu_embedding_cache,
            dense_batch_size=hybrid_dcu_dense_batch_size,
            max_text_chars=hybrid_dcu_max_text_chars,
        )
    else:
        raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

    rows: list[dict[str, Any]] = []
    total_candidates = 0
    queries_with_candidates = 0
    queries_with_explicit_candidates = 0
    total_explicit_candidates = 0
    n_batches = math.ceil(len(dataset_rows) / batch_size) if batch_size else 0

    for batch_index, start in enumerate(range(0, len(dataset_rows), batch_size), start=1):
        batch = dataset_rows[start:start + batch_size]
        if retrieval_method == "tfidf":
            assert vectorizer is not None and corpus_matrix is not None
            query_matrix = vectorizer.transform([query_text(row) for row in batch])
            score_matrix = (query_matrix @ corpus_matrix.T).toarray()
        else:
            score_matrix = None
        for row_offset, query in enumerate(batch):
            pool_size = min(candidate_pool_size, len(acu_rows))
            grouped: dict[str, dict[str, Any]] = {}
            explicit_count = 0
            if use_explicit_matches:
                explicit_count = add_explicit_matches(
                    grouped,
                    query,
                    indexes,
                    year_mode=year_mode,
                    max_prior_acus_per_dataset=max_prior_acus_per_dataset,
                    max_explicit_matches_per_name=max_explicit_matches_per_name,
                )
                if explicit_count:
                    queries_with_explicit_candidates += 1
                    total_explicit_candidates += explicit_count
            if retrieval_method == "tfidf":
                assert score_matrix is not None
                scores = score_matrix[row_offset]
                for acu_index in top_indices(scores, pool_size):
                    score = float(scores[acu_index])
                    if score < min_score:
                        continue
                    acu = acu_rows[int(acu_index)]
                    if not is_allowed_prior(query, acu, year_mode):
                        continue
                    bank_id = str(acu.get("bank_id") or "")
                    if not bank_id:
                        continue
                    item = ensure_candidate_item(
                        grouped,
                        bank_id,
                        candidate_dataset=indexes["by_bank_id"].get(bank_id),
                        candidate_acu=acu,
                        score=score,
                        retrieval_source="tfidf",
                    )
                    add_candidate_acu(
                        item,
                        acu,
                        score=score,
                        max_prior_acus_per_dataset=max_prior_acus_per_dataset,
                    )
            else:
                assert hybrid_dcu_index is not None
                for query_acu in query.get("acus") or []:
                    query_text_value = query_dcu_text(query, query_acu)
                    if not query_text_value.strip():
                        continue
                    scores = hybrid_dcu_index.scores(query_text_value)
                    for acu_index in top_indices(scores, pool_size):
                        score = float(scores[acu_index])
                        if score < min_score:
                            continue
                        acu = acu_rows[int(acu_index)]
                        if not is_allowed_prior(query, acu, year_mode):
                            continue
                        bank_id = str(acu.get("bank_id") or "")
                        if not bank_id:
                            continue
                        item = ensure_candidate_item(
                            grouped,
                            bank_id,
                            candidate_dataset=indexes["by_bank_id"].get(bank_id),
                            candidate_acu=acu,
                            score=score,
                            retrieval_source="hybrid_dcu",
                        )
                        add_candidate_acu(
                            item,
                            acu,
                            score=score,
                            max_prior_acus_per_dataset=max_prior_acus_per_dataset,
                            retrieval_source="hybrid_dcu",
                        )

            candidates = sorted(
                grouped.values(),
                key=lambda item: (
                    0 if item.get("retrieval_sources") and item["retrieval_sources"][0] != "tfidf" else 1,
                    -float(item["score"]),
                ),
            )[:top_k]
            if candidates:
                queries_with_candidates += 1
                total_candidates += len(candidates)
            rows.append({
                "query_bank_id": query.get("bank_id") or "",
                "query_paper_id": query.get("paper_id") or "",
                "query_dataset_id": query.get("dataset_id") or "",
                "query_dataset_name": query.get("dataset_name") or "",
                "query_title": query.get("title") or "",
                "query_year": query.get("year"),
                "query_role": query.get("role") or "",
                "query_resource_type": query.get("resource_type") or "",
                "query_tasks": query.get("tasks") or [],
                "query_domains": query.get("domains") or [],
                "query_languages": query.get("languages") or [],
                "query_acus": query.get("acus") or [],
                "candidate_generation": {
                    "method": "tfidf_acu_to_dataset_grouped" if retrieval_method == "tfidf" else "hybrid_dcu_to_dataset_grouped",
                    "retrieval_method": retrieval_method,
                    "year_mode": year_mode,
                    "top_k": top_k,
                    "candidate_pool_size": candidate_pool_size,
                    "max_prior_acus_per_dataset": max_prior_acus_per_dataset,
                    "min_score": min_score,
                    "use_explicit_matches": use_explicit_matches,
                    "max_explicit_matches_per_name": max_explicit_matches_per_name,
                    "explicit_candidate_count": explicit_count,
                },
                "prior_candidates": candidates,
            })
        print(json.dumps({
            "batch": batch_index,
            "batches": n_batches,
            "processed_queries": min(start + len(batch), len(dataset_rows)),
            "queries_with_candidates": queries_with_candidates,
            "queries_with_explicit_candidates": queries_with_explicit_candidates,
        }, ensure_ascii=False), flush=True)

    summary = {
        "query_rows": len(dataset_rows),
        "acu_corpus_rows": len(acu_rows),
        "queries_with_candidates": queries_with_candidates,
        "queries_without_candidates": len(dataset_rows) - queries_with_candidates,
        "total_prior_candidates": total_candidates,
        "queries_with_explicit_candidates": queries_with_explicit_candidates,
        "total_explicit_candidates": total_explicit_candidates,
        "mean_prior_candidates_per_query": total_candidates / len(dataset_rows) if dataset_rows else 0.0,
        "top_k": top_k,
        "candidate_pool_size": candidate_pool_size,
        "batch_size": batch_size,
        "year_mode": year_mode,
        "max_prior_acus_per_dataset": max_prior_acus_per_dataset,
        "min_score": min_score,
        "use_explicit_matches": use_explicit_matches,
        "max_explicit_matches_per_name": max_explicit_matches_per_name,
        "query_year": query_year,
        "retrieval_method": retrieval_method,
        "candidate_generation_method": "tfidf_acu_to_dataset_grouped" if retrieval_method == "tfidf" else "hybrid_dcu_to_dataset_grouped",
        "hybrid_dcu_embedding_cache": hybrid_dcu_embedding_cache,
        "hybrid_dcu_dense_batch_size": hybrid_dcu_dense_batch_size,
        "hybrid_dcu_max_text_chars": hybrid_dcu_max_text_chars,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-corpus prior candidate queue from dataset and ACU banks.")
    parser.add_argument("--dataset-bank-jsonl", required=True)
    parser.add_argument("--acu-bank-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--retrieval-method", choices=["tfidf", "hybrid_dcu"], default="tfidf")
    parser.add_argument("--hybrid-dcu-embedding-cache", default="data/census/fulltext_acu_bank_hybrid_dcu_minilm_embeddings.npz")
    parser.add_argument("--hybrid-dcu-dense-batch-size", type=int, default=128)
    parser.add_argument("--hybrid-dcu-max-text-chars", type=int, default=1600)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-pool-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--year-mode", choices=["earlier", "earlier_or_same", "any"], default="earlier_or_same")
    parser.add_argument("--max-prior-acus-per-dataset", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--disable-explicit-matches", action="store_true")
    parser.add_argument("--max-explicit-matches-per-name", type=int, default=5)
    parser.add_argument("--query-year", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_rows = read_jsonl(args.dataset_bank_jsonl)
    acu_rows = read_jsonl(args.acu_bank_jsonl)
    rows, summary = build_queue(
        dataset_rows,
        acu_rows,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        batch_size=args.batch_size,
        year_mode=args.year_mode,
        max_prior_acus_per_dataset=args.max_prior_acus_per_dataset,
        min_score=args.min_score,
        limit=args.limit,
        use_explicit_matches=not args.disable_explicit_matches,
        max_explicit_matches_per_name=args.max_explicit_matches_per_name,
        query_year=args.query_year,
        retrieval_method=args.retrieval_method,
        hybrid_dcu_embedding_cache=args.hybrid_dcu_embedding_cache,
        hybrid_dcu_dense_batch_size=args.hybrid_dcu_dense_batch_size,
        hybrid_dcu_max_text_chars=args.hybrid_dcu_max_text_chars,
    )
    write_jsonl(args.output_jsonl, rows)
    write_json(args.summary_json, summary)
    print(json.dumps({
        "output_jsonl": args.output_jsonl,
        "summary_json": args.summary_json,
        **summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
