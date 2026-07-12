from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from .decay_engine import DecayEngine
from .models import Memory
from .memory_store import MemoryStore

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from sentence_transformers.util import cos_sim
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class RetrievalResult:
    memory: Memory
    score: float
    relevance: float
    strength: float


class Retriever:
    """
    Ranks memories using decayed strength and a 2-stage Neural Retrieval Pipeline.
    Stage 1: Bi-Encoder (SBERT) + Lexical for top-K candidates.
    Stage 2: Cross-Encoder (BERT) for precise re-ranking.
    """

    def __init__(self, store: MemoryStore, decay_engine: DecayEngine | None = None, use_semantic: bool = True, semantic_alpha: float = 0.7):
        self.store = store
        self.decay_engine = decay_engine or DecayEngine()
        self.use_semantic = use_semantic and HAS_SENTENCE_TRANSFORMERS
        self.semantic_alpha = semantic_alpha
        
        self._bi_encoder = None
        self._cross_encoder = None
        
        if self.use_semantic:
            logging.info("Loading SBERT Bi-Encoder...")
            self._bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logging.info("Loading BERT Cross-Encoder...")
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.strip().lower() for t in text.replace("\n", " ").split(" ") if t.strip()]

    def _lexical_relevance(self, query: str, memory: Memory) -> float:
        q_tokens = set(self._tokenize(query))
        if not q_tokens:
            return 0.0

        m_tokens = set(self._tokenize(memory.content))
        inter = len(q_tokens.intersection(m_tokens))
        union = len(q_tokens.union(m_tokens))
        return inter / union if union else 0.0

    def rank(
        self,
        query: str,
        limit: int = 5,
        now: Optional[float] = None,
        min_relevance: float = 0.0,
        query_for_bandit: str = "",
    ) -> List[RetrievalResult]:
        memories = self.store.list_all()
        if not memories:
            return []
            
        # --- STAGE 1: Bi-Encoder Retrieval ---
        semantic_scores = {}
        if self.use_semantic and self._bi_encoder is not None:
            q_emb = self._bi_encoder.encode(query, convert_to_tensor=True)
            contents = [m.content for m in memories]
            m_embs = self._bi_encoder.encode(contents, convert_to_tensor=True)
            sims = cos_sim(q_emb, m_embs)[0]
            for idx, m in enumerate(memories):
                semantic_scores[m.id] = max(0.0, float(sims[idx]))

        candidates: List[RetrievalResult] = []
        for m in memories:
            strength = self.decay_engine.effective_strength(m, now=now, query=query_for_bandit)
            lexical_rel = self._lexical_relevance(query, m)
            
            if self.use_semantic and self._bi_encoder is not None:
                sem_rel = semantic_scores.get(m.id, 0.0)
                rel = self.semantic_alpha * sem_rel + (1.0 - self.semantic_alpha) * lexical_rel
            else:
                rel = lexical_rel
                
            if rel < min_relevance:
                continue
            score = rel * strength
            candidates.append(RetrievalResult(memory=m, score=score, relevance=rel, strength=strength))

        # Sort candidates and keep top-20 for re-ranking
        candidates.sort(key=lambda r: r.score, reverse=True)
        top_k_candidates = candidates[:20]

        # --- STAGE 2: Cross-Encoder Re-Ranking ---
        if self.use_semantic and self._cross_encoder is not None and top_k_candidates:
            pairs = [[query, c.memory.content] for c in top_k_candidates]
            cross_scores = self._cross_encoder.predict(pairs)
            
            for i, c in enumerate(top_k_candidates):
                import math
                # Convert logits to a 0-1 probability score using sigmoid
                sigmoid_score = 1 / (1 + math.exp(-cross_scores[i]))
                c.relevance = sigmoid_score
                c.score = c.relevance * c.strength
            
            # Final sort based on precise cross-encoder scores
            top_k_candidates.sort(key=lambda r: r.score, reverse=True)

        return top_k_candidates[: max(0, limit)]

    def retrieve_and_touch(
        self,
        query: str,
        limit: int = 5,
        now: Optional[float] = None,
        min_relevance: float = 0.0,
        prune_keep_ids: Optional[List[str]] = None,
        query_for_bandit: str = "",
    ) -> Tuple[List[RetrievalResult], int]:
        import time

        results = self.rank(
            query,
            limit=limit,
            now=now,
            min_relevance=min_relevance,
            query_for_bandit=query_for_bandit,
        )
        t = now if now is not None else time.time()

        for r in results:
            touched = r.memory
            touched.last_accessed_at = t
            recall_count = int(touched.metadata.get("recall_count", 0))
            touched.metadata["recall_count"] = recall_count + 1
            self.store.upsert(touched)

        pruned_count = 0
        if prune_keep_ids is not None:
            pruned_count = self.store.prune_inplace(prune_keep_ids)

        return results, pruned_count
