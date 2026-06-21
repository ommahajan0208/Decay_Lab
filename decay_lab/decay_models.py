from __future__ import annotations

from abc import ABC, abstractmethod
import math
import time
from typing import Iterable, Optional, Tuple

from .models import Memory


class DecayModel(ABC):
    @abstractmethod
    def score(self, memory: Memory, now: Optional[float] = None) -> float:
        """Return this model's effective strength for a memory."""


class ExponentialDecay(DecayModel):
    def __init__(self, lambda_decay: float = 1e-4):
        if lambda_decay < 0:
            raise ValueError("lambda_decay must be >= 0")
        self.lambda_decay = lambda_decay

    def score(self, memory: Memory, now: Optional[float] = None) -> float:
        if self.lambda_decay == 0:
            return memory.strength
        age = _age_seconds(memory, now=now)
        return memory.strength * math.exp(-self.lambda_decay * age)


class HLRDecay(DecayModel):
    """
    Half-Life Regression (Settles & Meeder 2016).
    Predicts a half-life `h` based on memory history, and decays using 2^(-t/h).
    """
    def __init__(self, base_half_life: float = 3600.0, recall_boost: float = 2.0):
        self.base_half_life = base_half_life
        self.recall_boost = recall_boost
        
        # A simple linear simulation of the learned HLR weights.
        # In a real scenario, this would be updated via SGD using spacing logs.
        self.theta = {
            "bias": math.log2(max(1.0, self.base_half_life)),
            "recall": math.log2(max(1.0, self.recall_boost))
        }

    def _predict_half_life(self, memory: Memory) -> float:
        recall_count = float(memory.metadata.get("recall_count", 0))
        log_h = self.theta["bias"] + self.theta["recall"] * math.log1p(recall_count)
        return 2 ** log_h

    def score(self, memory: Memory, now: Optional[float] = None) -> float:
        age = _age_seconds(memory, now=now)
        h = self._predict_half_life(memory)
        p = 2 ** (-age / max(h, 1.0))
        return memory.strength * p


class PowerLawDecay(DecayModel):
    def __init__(self, alpha: float = 0.35, scale_seconds: float = 3600.0):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if scale_seconds <= 0:
            raise ValueError("scale_seconds must be > 0")
        self.alpha = alpha
        self.scale_seconds = scale_seconds

    def score(self, memory: Memory, now: Optional[float] = None) -> float:
        age = _age_seconds(memory, now=now)
        return memory.strength * (1.0 + (age / self.scale_seconds)) ** (-self.alpha)


class ReinforcementDecay(DecayModel):
    def __init__(self, gamma: float = 0.08):
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = gamma

    def score(self, memory: Memory, now: Optional[float] = None) -> float:
        recall_count = float(memory.metadata.get("recall_count", 0))
        return memory.strength * (1.0 + self.gamma * math.log1p(max(0.0, recall_count)))


class WeightedEnsembleDecay(DecayModel):
    def __init__(self, models: Iterable[Tuple[float, DecayModel]]):
        self.models = list(models)
        if not self.models:
            raise ValueError("models must not be empty")
        if any(weight < 0 for weight, _ in self.models):
            raise ValueError("weights must be >= 0")

        total_weight = sum(weight for weight, _ in self.models)
        if total_weight <= 0:
            raise ValueError("at least one weight must be > 0")

        self.models = [(weight / total_weight, model) for weight, model in self.models]

    def score(self, memory: Memory, now: Optional[float] = None) -> float:
        return sum(weight * model.score(memory, now=now) for weight, model in self.models)


def _age_seconds(memory: Memory, now: Optional[float] = None) -> float:
    t = now if now is not None else time.time()
    anchor = memory.last_accessed_at if memory.last_accessed_at is not None else memory.created_at
    return max(0.0, t - anchor)
