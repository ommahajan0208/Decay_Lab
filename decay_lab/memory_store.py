"""
Updated garbage_collect returns pruned memories with cause-of-death metadata.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from .models import Memory


class MemoryStore:
    """
    Simple JSON-file backed memory store.

    File format:
    {
      "memories": [ {Memory...}, ... ]
    }
    """

    def __init__(self, path: str = "data/memories.json"):
        self.path = path
        self._ensure_file()

    def _ensure_file(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"memories": []}, f, indent=2)

    def _load(self) -> List[Memory]:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        memories = data.get("memories", [])
        return [Memory.from_dict(m) for m in memories]

    def _save(self, memories: List[Memory]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"memories": [m.to_dict() for m in memories]}, f, indent=2)

    def list_all(self) -> List[Memory]:
        return self._load()

    def get(self, memory_id: str) -> Optional[Memory]:
        for m in self._load():
            if m.id == memory_id:
                return m
        return None

    def upsert(self, memory: Memory) -> None:
        memories = self._load()
        replaced = False
        for i, m in enumerate(memories):
            if m.id == memory.id:
                memories[i] = memory
                replaced = True
                break
        if not replaced:
            memories.append(memory)
        self._save(memories)

    def delete(self, memory_id: str) -> bool:
        memories = self._load()
        before = len(memories)
        memories = [m for m in memories if m.id != memory_id]
        self._save(memories)
        return len(memories) != before

    def prune_inplace(self, keep_ids: List[str]) -> int:
        keep = set(keep_ids)
        memories = self._load()
        new_memories = [m for m in memories if m.id in keep]
        self._save(new_memories)
        return len(memories) - len(new_memories)

    def garbage_collect(self, decay_engine, now: Optional[float] = None) -> Tuple[int, List[Dict]]:
        """
        Returns (pruned_count, list_of_pruned_memory_dicts).
        Each pruned dict includes id, content, final_strength, threshold, recall_count.
        """
        memories = self._load()
        threshold = decay_engine.profile.prune_threshold
        t = now if now is not None else time.time()

        new_memories = []
        pruned_list = []
        for m in memories:
            strength = decay_engine.effective_strength(m, now=t)
            if strength >= threshold:
                new_memories.append(m)
            else:
                pruned_list.append({
                    "id": m.id,
                    "content": m.content,
                    "final_strength": round(strength, 4),
                    "threshold": threshold,
                    "recall_count": int(m.metadata.get("recall_count", 0)),
                    "profile": decay_engine.profile.name,
                    "died_at": t,
                })

        self._save(new_memories)
        return len(pruned_list), pruned_list

    def apply_sleep_consolidation(self, decay_engine, now: Optional[float] = None) -> Dict:
        """
        Sleep cycle: +8h time-jump, consolidate recalled memories, prune weak ones.
        Memories with recall_count >= 2 get a +0.1 strength bonus.
        Returns summary dict.
        """
        memories = self._load()
        consolidated = []
        for m in memories:
            if int(m.metadata.get("recall_count", 0)) >= 2:
                m.strength = min(1.0, m.strength + 0.10)
                consolidated.append(m.id)
            # save back
            self.upsert(m)

        pruned_count, pruned_list = self.garbage_collect(decay_engine, now=now)
        return {
            "consolidated": consolidated,
            "pruned_count": pruned_count,
            "pruned": pruned_list,
        }
