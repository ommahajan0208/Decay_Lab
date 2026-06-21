from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import Memory
from .retrieval import Retriever, RetrievalResult
from .memory_store import MemoryStore


@dataclass
class EvalExample:
    query: str
    relevant_ids: List[str]


@dataclass
class EvalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    map_at_k: float


class Evaluator:
    """
    Benchmark harness supporting Precision@K, Recall@K, MRR@K, MAP@K,
    and automatic parameter weight optimization.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    @staticmethod
    def _score_at_k(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k: int,
    ) -> Tuple[float, float]:
        topk = results[:k]
        retrieved_ids = {r.memory.id for r in topk}

        rel = set(relevant_ids)
        if not rel:
            return 0.0, 0.0

        tp = len(retrieved_ids.intersection(rel))
        precision = tp / k if k else 0.0
        recall = tp / len(rel)
        return precision, recall

    @staticmethod
    def _rr_at_k(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        topk = results[:k]
        rel = set(relevant_ids)
        for idx, r in enumerate(topk):
            if r.memory.id in rel:
                return 1.0 / (idx + 1)
        return 0.0

    @staticmethod
    def _ap_at_k(
        results: List[RetrievalResult],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        topk = results[:k]
        rel = set(relevant_ids)
        if not rel:
            return 0.0

        ap = 0.0
        hits = 0
        for idx, r in enumerate(topk):
            if r.memory.id in rel:
                hits += 1
                precision_at_idx = hits / (idx + 1)
                ap += precision_at_idx

        return ap / len(rel)

    def evaluate(
        self,
        examples: List[EvalExample],
        k: int = 5,
    ) -> EvalMetrics:
        precisions: List[float] = []
        recalls: List[float] = []
        mrrs: List[float] = []
        maps: List[float] = []

        for ex in examples:
            results = self.retriever.rank(ex.query, limit=k)
            p, r = self._score_at_k(results, ex.relevant_ids, k=k)
            rr = self._rr_at_k(results, ex.relevant_ids, k=k)
            ap = self._ap_at_k(results, ex.relevant_ids, k=k)

            precisions.append(p)
            recalls.append(r)
            mrrs.append(rr)
            maps.append(ap)

        if not precisions:
            return EvalMetrics(precision_at_k=0.0, recall_at_k=0.0, mrr_at_k=0.0, map_at_k=0.0)

        return EvalMetrics(
            precision_at_k=sum(precisions) / len(precisions),
            recall_at_k=sum(recalls) / len(recalls),
            mrr_at_k=sum(mrrs) / len(mrrs),
            map_at_k=sum(maps) / len(maps),
        )

    def optimize_weights(
        self,
        examples: List[EvalExample],
        k: int = 5,
        metric: str = "map_at_k",
    ) -> Tuple[Dict[str, float], float]:
        """
        Grid search over semantic_alpha values to find the retriever configuration
        that maximizes the given metric.
        """
        best_score = -1.0
        best_config: Dict[str, float] = {}

        orig_alpha = self.retriever.semantic_alpha

        for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            self.retriever.semantic_alpha = alpha
            metrics = self.evaluate(examples, k=k)
            score = getattr(metrics, metric)
            if score > best_score:
                best_score = score
                best_config = {"semantic_alpha": alpha}

        self.retriever.semantic_alpha = orig_alpha
        return best_config, best_score
