import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import norm
try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .analysis import SENTENCE_MODEL

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SparseEncoder
    HAS_SPARSE_ENCODER = True
except ImportError:
    SparseEncoder = None  # type: ignore
    HAS_SPARSE_ENCODER = False

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


ORDINAL_LABELS = ["repackaging", "incremental", "substantial"]
ORDINAL_TO_INT = {label: idx for idx, label in enumerate(ORDINAL_LABELS)}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def reciprocal_rank(rank: Optional[int]) -> float:
    if not rank or rank <= 0:
        return 0.0
    return 1.0 / rank


def ordinal_from_score(score: float) -> str:
    if score < 0.2:
        return "repackaging"
    if score < 0.55:
        return "incremental"
    return "substantial"


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = safe_mean(left_ranks)
    right_mean = safe_mean(right_ranks)
    num = sum((l - left_mean) * (r - right_mean) for l, r in zip(left_ranks, right_ranks))
    den_left = math.sqrt(sum((l - left_mean) ** 2 for l in left_ranks))
    den_right = math.sqrt(sum((r - right_mean) ** 2 for r in right_ranks))
    denom = den_left * den_right
    return float(num / denom) if denom else 0.0


@dataclass
class CandidateRecord:
    candidate_id: str
    name: str
    acus: List[str]
    summary_text: str
    domain: str = ""
    role: str = ""
    source_dataset: str = ""
    is_cited: bool = False


class HybridSupportRetriever:
    """Hybrid retriever over dataset candidates with a corpus-aware genericness penalty."""
    _GLOBAL_SPLADE_MODEL = None
    _GLOBAL_SENTENCE_MODEL = None
    DEBUG_COUNTERS: Dict[str, int] = {
        "splade_real": 0,
        "splade_fallback_lexical": 0,
        "splade_unavailable_no_sparse_encoder": 0,
        "splade_unavailable_model_init_failed": 0,
        "splade_unavailable_no_doc_embeddings": 0,
        "splade_exception_runtime": 0,
        "colbert_real": 0,
        "colbert_fallback_dense": 0,
        "colbert_early_empty_query_or_corpus": 0,
    }
    _SPLADE_ERROR_PRINTED = False
    _SPLADE_INIT_ERROR_PRINTED = False
    _SENTENCE_MODEL_ERROR_PRINTED = False

    def __init__(self, candidates: Sequence[CandidateRecord]):
        self.candidates = list(candidates)
        self.acu_to_candidate: List[str] = []
        self.acu_texts: List[str] = []
        self.candidate_lookup = {candidate.candidate_id: candidate for candidate in self.candidates}
        self._splade_model = None
        self._splade_doc_embeddings = None

        for candidate in self.candidates:
            if candidate.acus:
                for acu in candidate.acus:
                    self.acu_texts.append(acu)
                    self.acu_to_candidate.append(candidate.candidate_id)
            else:
                self.acu_texts.append(candidate.summary_text)
                self.acu_to_candidate.append(candidate.candidate_id)

        self.acu_embeddings = None
        sentence_model = self._get_sentence_model()
        if self.acu_texts and sentence_model is not None:
            self.acu_embeddings = sentence_model.encode(self.acu_texts, convert_to_numpy=True)

        self.tokenized_corpus = [tokenize(candidate.summary_text) for candidate in self.candidates]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if HAS_BM25 and self.tokenized_corpus else None

        token_document_frequency: Counter = Counter()
        for tokens in self.tokenized_corpus:
            for token in set(tokens):
                token_document_frequency[token] += 1
        corpus_size = max(len(self.candidates), 1)
        self.genericness: Dict[str, float] = {}
        for candidate, tokens in zip(self.candidates, self.tokenized_corpus):
            if not tokens:
                self.genericness[candidate.candidate_id] = 0.0
                continue
            avg_df = safe_mean([token_document_frequency[token] / corpus_size for token in set(tokens)])
            self.genericness[candidate.candidate_id] = avg_df

    def _ensure_splade(self) -> None:
        if not HAS_SPARSE_ENCODER:
            HybridSupportRetriever.DEBUG_COUNTERS["splade_unavailable_no_sparse_encoder"] += 1
            return
        if HybridSupportRetriever._GLOBAL_SPLADE_MODEL is None:
            try:
                HybridSupportRetriever._GLOBAL_SPLADE_MODEL = SparseEncoder("naver/splade-cocondenser-ensembledistil")
            except Exception as exc:
                HybridSupportRetriever._GLOBAL_SPLADE_MODEL = None
                HybridSupportRetriever.DEBUG_COUNTERS["splade_unavailable_model_init_failed"] += 1
                if not HybridSupportRetriever._SPLADE_INIT_ERROR_PRINTED:
                    print(f"[SPLADE init failed] {exc}")
                    HybridSupportRetriever._SPLADE_INIT_ERROR_PRINTED = True
                return
        try:
            self._splade_model = HybridSupportRetriever._GLOBAL_SPLADE_MODEL
            if self.candidates:
                self._splade_doc_embeddings = self._splade_model.encode_document(
                    [candidate.summary_text for candidate in self.candidates]
                )
        except Exception:
            self._splade_model = None
            self._splade_doc_embeddings = None
            HybridSupportRetriever.DEBUG_COUNTERS["splade_unavailable_no_doc_embeddings"] += 1

    def _get_sentence_model(self):
        if SENTENCE_MODEL is not None:
            return SENTENCE_MODEL
        if HybridSupportRetriever._GLOBAL_SENTENCE_MODEL is not None:
            return HybridSupportRetriever._GLOBAL_SENTENCE_MODEL
        if SentenceTransformer is None:
            return None
        try:
            HybridSupportRetriever._GLOBAL_SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            return HybridSupportRetriever._GLOBAL_SENTENCE_MODEL
        except Exception as exc:
            if not HybridSupportRetriever._SENTENCE_MODEL_ERROR_PRINTED:
                print(f"[Dense/ColBERT disabled] Failed to load fallback sentence model: {exc}")
                HybridSupportRetriever._SENTENCE_MODEL_ERROR_PRINTED = True
            return None

    @staticmethod
    def _to_numpy(array_like) -> np.ndarray:
        if torch is not None and isinstance(array_like, torch.Tensor):
            return array_like.detach().cpu().numpy()
        if hasattr(array_like, "toarray"):
            return array_like.toarray()
        return np.asarray(array_like)

    def _dense_scores(self, query_acus: Sequence[str], top_k: int = 20) -> Dict[str, float]:
        sentence_model = self._get_sentence_model()
        if not query_acus or self.acu_embeddings is None or sentence_model is None:
            return {}
        from sentence_transformers import util

        query_embeddings = sentence_model.encode(list(query_acus), convert_to_numpy=True)
        hits = util.semantic_search(query_embeddings, self.acu_embeddings, top_k=min(top_k, len(self.acu_texts)))
        scores: Dict[str, float] = defaultdict(float)
        for query_hits in hits:
            for hit in query_hits:
                candidate_id = self.acu_to_candidate[hit["corpus_id"]]
                scores[candidate_id] += float(hit["score"])
        return dict(scores)

    def _lexical_scores(self, query_text: str) -> Dict[str, float]:
        if not self.bm25:
            return {}
        tokenized_query = tokenize(query_text)
        if not tokenized_query:
            return {}
        raw_scores = self.bm25.get_scores(tokenized_query)
        return {
            candidate.candidate_id: float(score)
            for candidate, score in zip(self.candidates, raw_scores)
            if score > 0
        }

    def _splade_scores(self, query_text: str) -> Dict[str, float]:
        if not query_text.strip():
            return {}
        self._ensure_splade()
        if self._splade_model is None or self._splade_doc_embeddings is None:
            HybridSupportRetriever.DEBUG_COUNTERS["splade_unavailable_no_doc_embeddings"] += 1
            HybridSupportRetriever.DEBUG_COUNTERS["splade_fallback_lexical"] += 1
            return self._lexical_scores(query_text)
        if self._splade_model is not None and self._splade_doc_embeddings is not None:
            try:
                query_embedding = self._splade_model.encode_query([query_text])
                # Use SparseEncoder similarity API to handle sparse tensor internals safely.
                sim = self._splade_model.similarity(query_embedding, self._splade_doc_embeddings)
                scores = self._to_numpy(sim).reshape(-1)
                HybridSupportRetriever.DEBUG_COUNTERS["splade_real"] += 1
                return {
                    candidate.candidate_id: float(score)
                    for candidate, score in zip(self.candidates, scores)
                    if score > 0
                }
            except Exception as exc:
                HybridSupportRetriever.DEBUG_COUNTERS["splade_exception_runtime"] += 1
                if not HybridSupportRetriever._SPLADE_ERROR_PRINTED:
                    print(f"[SPLADE fallback] Real SPLADE scoring failed once with: {exc}")
                    HybridSupportRetriever._SPLADE_ERROR_PRINTED = True
                pass
        # Fallback: use lexical scores as a sparse baseline when SPLADE is unavailable.
        HybridSupportRetriever.DEBUG_COUNTERS["splade_fallback_lexical"] += 1
        return self._lexical_scores(query_text)

    def _colbert_scores(self, query_acus: Sequence[str]) -> Dict[str, float]:
        sentence_model = self._get_sentence_model()
        if not query_acus or sentence_model is None or not self.acu_texts:
            HybridSupportRetriever.DEBUG_COUNTERS["colbert_early_empty_query_or_corpus"] += 1
            return {}
        try:
            query_tokens = sentence_model.encode(
                list(query_acus),
                output_value="token_embeddings",
                convert_to_numpy=True,
            )
            doc_tokens = sentence_model.encode(
                self.acu_texts,
                output_value="token_embeddings",
                convert_to_numpy=True,
            )
        except Exception:
            # Fallback: dense retriever when token-level embeddings are unavailable.
            HybridSupportRetriever.DEBUG_COUNTERS["colbert_fallback_dense"] += 1
            return self._dense_scores(query_acus)

        acu_scores: Dict[int, float] = defaultdict(float)
        for query_matrix in query_tokens:
            q = self._to_numpy(query_matrix).astype(float, copy=False)
            if q.size == 0:
                continue
            q_norm = norm(q, axis=1, keepdims=True)
            q_norm[q_norm == 0] = 1.0
            q = q / q_norm

            for idx, doc_matrix in enumerate(doc_tokens):
                d = self._to_numpy(doc_matrix).astype(float, copy=False)
                if d.size == 0:
                    continue
                d_norm = norm(d, axis=1, keepdims=True)
                d_norm[d_norm == 0] = 1.0
                d = d / d_norm
                # ColBERT-style MaxSim: sum over query tokens of max cosine similarity to doc tokens.
                maxsim = (q @ d.T).max(axis=1)
                acu_scores[idx] += float(np.sum(maxsim))

        scores: Dict[str, float] = defaultdict(float)
        for acu_idx, score in acu_scores.items():
            candidate_id = self.acu_to_candidate[acu_idx]
            scores[candidate_id] += score
        HybridSupportRetriever.DEBUG_COUNTERS["colbert_real"] += 1
        return dict(scores)

    @classmethod
    def get_debug_counters(cls) -> Dict[str, int]:
        return dict(cls.DEBUG_COUNTERS)

    @classmethod
    def reset_debug_counters(cls) -> None:
        for key in cls.DEBUG_COUNTERS:
            cls.DEBUG_COUNTERS[key] = 0

    def _rrf_fusion(self, ranked_lists: Sequence[List[str]], k: int = 60) -> Dict[str, float]:
        fused: Dict[str, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, candidate_id in enumerate(ranked, start=1):
                fused[candidate_id] += 1.0 / (k + rank)
        return dict(fused)

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = np.array(list(scores.values()), dtype=float)
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        if max_v - min_v < 1e-8:
            return {key: 1.0 for key in scores}
        return {key: (value - min_v) / (max_v - min_v) for key, value in scores.items()}

    def rank(
        self,
        query_name: str,
        query_acus: Sequence[str],
        query_metadata: Optional[Dict] = None,
        top_k: int = 5,
        method: str = "hybrid_rerank",
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        query_metadata = query_metadata or {}
        query_text = " ".join([query_name] + list(query_acus) + [
            query_metadata.get("domain", ""),
            query_metadata.get("role", ""),
            query_metadata.get("source_dataset", ""),
        ])
        query_acus_for_semantic = list(query_acus)
        if not query_acus_for_semantic and query_name:
            # Legacy benchmark rows can have empty ACUs; keep semantic retrievers active.
            query_acus_for_semantic = [query_name]

        dense = self._normalize(self._dense_scores(query_acus_for_semantic))
        lexical = self._normalize(self._lexical_scores(query_text))
        splade = self._normalize(self._splade_scores(query_text))
        colbert = self._normalize(self._colbert_scores(query_acus_for_semantic))
        dense_ranked = [candidate_id for candidate_id, _ in sorted(dense.items(), key=lambda item: item[1], reverse=True)]
        lexical_ranked = [candidate_id for candidate_id, _ in sorted(lexical.items(), key=lambda item: item[1], reverse=True)]
        splade_ranked = [candidate_id for candidate_id, _ in sorted(splade.items(), key=lambda item: item[1], reverse=True)]
        colbert_ranked = [candidate_id for candidate_id, _ in sorted(colbert.items(), key=lambda item: item[1], reverse=True)]
        rank_fusion = self._normalize(self._rrf_fusion([dense_ranked, lexical_ranked, splade_ranked, colbert_ranked]))

        query_domain = (query_metadata.get("domain") or "").strip().lower()
        query_role = (query_metadata.get("role") or "").strip().lower()
        query_source_dataset = (query_metadata.get("source_dataset") or "").strip().lower()

        merged_scores: Dict[str, Dict[str, float]] = {}
        for candidate in self.candidates:
            dense_score = dense.get(candidate.candidate_id, 0.0)
            lexical_score = lexical.get(candidate.candidate_id, 0.0)
            splade_score = splade.get(candidate.candidate_id, 0.0)
            colbert_score = colbert.get(candidate.candidate_id, 0.0)
            rank_fusion_score = rank_fusion.get(candidate.candidate_id, 0.0)
            metadata_bonus = 0.0
            if query_domain and query_domain == candidate.domain.strip().lower():
                metadata_bonus += 0.12
            if query_role and query_role == candidate.role.strip().lower():
                metadata_bonus += 0.08
            if query_source_dataset and query_source_dataset == candidate.name.strip().lower():
                metadata_bonus += 0.18
            citation_bonus = 0.1 if candidate.is_cited else 0.0
            genericness_penalty = 0.0
            if method == "hybrid_rerank":
                genericness_penalty = 0.25 * self.genericness.get(candidate.candidate_id, 0.0)

            if method == "dense":
                total = dense_score
            elif method == "lexical":
                total = lexical_score
            elif method == "splade":
                total = splade_score
            elif method == "colbert":
                total = colbert_score
            elif method == "rank_fusion":
                total = rank_fusion_score
            elif method == "fusion":
                total = 0.6 * dense_score + 0.4 * lexical_score
            else:
                total = (
                    (0.3 * dense_score)
                    + (0.2 * lexical_score)
                    + (0.2 * splade_score)
                    + (0.2 * colbert_score)
                    + (0.1 * rank_fusion_score)
                    + metadata_bonus
                    + citation_bonus
                    - genericness_penalty
                )

            merged_scores[candidate.candidate_id] = {
                "score": total,
                "dense": dense_score,
                "lexical": lexical_score,
                "splade": splade_score,
                "colbert": colbert_score,
                "rank_fusion": rank_fusion_score,
                "metadata_bonus": metadata_bonus,
                "citation_bonus": citation_bonus,
                "genericness_penalty": genericness_penalty,
            }

        ranked = sorted(
            merged_scores.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )
        return [(candidate_id, parts["score"], parts) for candidate_id, parts in ranked[:top_k]]


def evaluate_retrieval_run(
    ranked_candidate_ids: Sequence[str],
    gold_support_ids: Sequence[str],
) -> Dict[str, float]:
    gold = list(dict.fromkeys(gold_support_ids))
    ranked = list(ranked_candidate_ids)
    if not gold:
        return {"mrr": 0.0, "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0}
    first_hit_rank = None
    for rank, candidate_id in enumerate(ranked, start=1):
        if candidate_id in gold:
            first_hit_rank = rank
            break
    return {
        "mrr": reciprocal_rank(first_hit_rank),
        "recall@1": 1.0 if any(candidate_id in gold for candidate_id in ranked[:1]) else 0.0,
        "recall@3": 1.0 if any(candidate_id in gold for candidate_id in ranked[:3]) else 0.0,
        "recall@5": 1.0 if any(candidate_id in gold for candidate_id in ranked[:5]) else 0.0,
    }


def summarize_metric_runs(metric_rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(metric_rows)
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: safe_mean([row.get(key, 0.0) for row in rows]) for key in keys}
