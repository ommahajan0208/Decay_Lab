from __future__ import annotations

import time
from typing import Dict, List, Tuple

from .decay_engine import DecayEngine
from .models import Memory
from .memory_store import MemoryStore


def compute_strength_series(
    store: MemoryStore,
    decay_engine: DecayEngine,
    query: str,
    steps: int = 10,
    horizon_seconds: float = 3600.0,
    base_now: float = None,
) -> List[Tuple[float, Dict[str, float]]]:
    """
    Returns a time series of decayed effective strengths.
    base_now allows simulated time offset to be injected.

    Output:
      [
        (t_offset_seconds, {memory_id: effective_strength, ...}),
        ...
      ]
    """
    memories = store.list_all()
    if base_now is None:
        base_now = time.time()
    series: List[Tuple[float, Dict[str, float]]] = []

    for i in range(steps + 1):
        t = base_now + (i / steps) * horizon_seconds
        strengths: Dict[str, float] = {}
        for m in memories:
            strengths[m.id] = decay_engine.effective_strength(m, now=t)
        series.append(((i / steps) * horizon_seconds, strengths))

    return series



def pretty_print_top(
    store: MemoryStore,
    decay_engine: DecayEngine,
    k: int = 5,
) -> None:
    memories = store.list_all()
    now = time.time()
    scored = [(decay_engine.effective_strength(m, now=now), m) for m in memories]
    scored.sort(reverse=True, key=lambda x: x[0])

    print(f"Top {min(k, len(scored))} effective memories (now):")
    for s, m in scored[:k]:
        last = m.last_accessed_at if m.last_accessed_at is not None else m.created_at
        print(f"- {m.id} | score={s:.4f} | last={last:.0f} | content={m.content[:60]!r}")
