from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

from .decay_models import (
    DecayModel,
    ExponentialDecay,
    PowerLawDecay,
    ReinforcementDecay,
    HLRDecay,
    WeightedEnsembleDecay,
)
from .models import Memory
from .bandit import ContextualBandit

@dataclass
class BrainProfile:
    name: str
    hlr_base_half_life: float
    hlr_recall_boost: float
    power_law_alpha: float
    reinforcement_gamma: float
    prune_threshold: float
    weight_hlr: float
    weight_power: float
    weight_reinforcement: float


SMART_BRAIN = BrainProfile(
    name="smart",
    hlr_base_half_life=86400.0,    # Base half-life: 24 hours
    hlr_recall_boost=4.0,          # High boost for recall
    power_law_alpha=0.15,          # Very gentle power-law tail
    reinforcement_gamma=0.20,      # Strong baseline reinforcement memory
    prune_threshold=0.01,          # Rarely deletes anything
    weight_hlr=0.40,
    weight_power=0.40,
    weight_reinforcement=0.20,
)

DUMB_BRAIN = BrainProfile(
    name="dumb",
    hlr_base_half_life=1800.0,     # Base half-life: 30 minutes
    hlr_recall_boost=1.5,          # Low boost for recall
    power_law_alpha=0.60,          # Aggressive power-law tail
    reinforcement_gamma=0.05,      # Weak reinforcement
    prune_threshold=0.15,          # Aggressively prunes memories for storage efficiency
    weight_hlr=0.70,
    weight_power=0.20,
    weight_reinforcement=0.10,
)

ADAPTIVE_BRAIN = BrainProfile(
    name="adaptive",
    hlr_base_half_life=3600.0,
    hlr_recall_boost=2.0,
    power_law_alpha=0.35,
    reinforcement_gamma=0.15,
    prune_threshold=0.05,
    weight_hlr=0.0, # Weights are dynamically set by the Bandit
    weight_power=0.0,
    weight_reinforcement=0.0,
)

class DecayEngine:
    def __init__(self, profile: BrainProfile = SMART_BRAIN):
        self.bandit = ContextualBandit()
        self.profile = profile
        self.set_profile(profile)

    def set_profile(self, profile: BrainProfile):
        self.profile = profile
        self._update_model()

    def _update_model(self):
        if self.profile.name == "adaptive":
            weights = self.bandit.get_weights()
            w_hlr, w_power, w_reinf = weights[0], weights[1], weights[2]
        else:
            w_hlr = self.profile.weight_hlr
            w_power = self.profile.weight_power
            w_reinf = self.profile.weight_reinforcement

        self.model = WeightedEnsembleDecay(
            [
                (w_hlr, HLRDecay(base_half_life=self.profile.hlr_base_half_life, recall_boost=self.profile.hlr_recall_boost)),
                (w_power, PowerLawDecay(alpha=self.profile.power_law_alpha)),
                (w_reinf, ReinforcementDecay(gamma=self.profile.reinforcement_gamma)),
            ]
        )

    def age_seconds(self, memory: Memory, now: Optional[float] = None) -> float:
        t = now if now is not None else time.time()
        anchor = memory.last_accessed_at if memory.last_accessed_at is not None else memory.created_at
        return max(0.0, t - anchor)

    def effective_strength(self, memory: Memory, now: Optional[float] = None) -> float:
        # If adaptive, ensure weights are continuously sampled/updated
        if self.profile.name == "adaptive":
            self._update_model()
        return self.model.score(memory, now=now)
