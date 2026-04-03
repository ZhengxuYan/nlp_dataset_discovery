import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .analysis import SENTENCE_MODEL, load_models

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


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

    def __init__(self, candidates: Sequence[CandidateRecord]):
        load_models()
        self.candidates = list(candidates)
        self.acu_to_candidate: List[str] = []
        self.acu_texts: List[str] = []
        self.candidate_lookup = {candidate.candidate_id: candidate for candidate in self.candidates}

        for candidate in self.candidates:
            if candidate.acus:
                for acu in candidate.acus:
                    self.acu_texts.append(acu)
                    self.acu_to_candidate.append(candidate.candidate_id)
            else:
                self.acu_texts.append(candidate.summary_text)
                self.acu_to_candidate.append(candidate.candidate_id)

        self.acu_embeddings = None
        if self.acu_texts and SENTENCE_MODEL is not None:
            self.acu_embeddings = SENTENCE_MODEL.encode(self.acu_texts, convert_to_numpy=True)

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

    def _dense_scores(self, query_acus: Sequence[str], top_k: int = 20) -> Dict[str, float]:
        if not query_acus or self.acu_embeddings is None or SENTENCE_MODEL is None:
            return {}
        from sentence_transformers import util

        query_embeddings = SENTENCE_MODEL.encode(list(query_acus), convert_to_numpy=True)
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

        dense = self._normalize(self._dense_scores(query_acus))
        lexical = self._normalize(self._lexical_scores(query_text))

        query_domain = (query_metadata.get("domain") or "").strip().lower()
        query_role = (query_metadata.get("role") or "").strip().lower()
        query_source_dataset = (query_metadata.get("source_dataset") or "").strip().lower()

        merged_scores: Dict[str, Dict[str, float]] = {}
        for candidate in self.candidates:
            dense_score = dense.get(candidate.candidate_id, 0.0)
            lexical_score = lexical.get(candidate.candidate_id, 0.0)
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
            elif method == "fusion":
                total = 0.6 * dense_score + 0.4 * lexical_score
            else:
                total = (0.5 * dense_score) + (0.25 * lexical_score) + metadata_bonus + citation_bonus - genericness_penalty

            merged_scores[candidate.candidate_id] = {
                "score": total,
                "dense": dense_score,
                "lexical": lexical_score,
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
