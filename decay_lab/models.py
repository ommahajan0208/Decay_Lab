from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import time


@dataclass
class Memory:
    """
    A single memory item with a decaying importance score.

    Attributes:
        id: Stable unique identifier.
        content: Arbitrary text/body of the memory.
        tags: Lightweight categorization.
        created_at: Unix seconds when created.
        last_accessed_at: Unix seconds when last retrieved/used.
        strength: Initial importance (decays over time).
        metadata: Optional extra fields.
    """
    id: str
    content: str
    tags: Dict[str, Any]
    created_at: float
    last_accessed_at: Optional[float]
    strength: float
    metadata: Dict[str, Any]

    @staticmethod
    def now() -> float:
        return time.time()

    @classmethod
    def new(
        cls,
        id: str,
        content: str,
        tags: Optional[Dict[str, Any]] = None,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
    ) -> "Memory":
        created = created_at if created_at is not None else cls.now()
        return cls(
            id=id,
            content=content,
            tags=tags or {},
            created_at=created,
            last_accessed_at=None,
            strength=strength,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Memory":
        return cls(
            id=d["id"],
            content=d["content"],
            tags=d.get("tags", {}),
            created_at=d.get("created_at", cls.now()),
            last_accessed_at=d.get("last_accessed_at"),
            strength=float(d.get("strength", 1.0)),
            metadata=d.get("metadata", {}),
        )
