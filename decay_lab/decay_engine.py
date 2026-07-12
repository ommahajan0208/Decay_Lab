from __future__ import annotations

import os
from dataclasses import dataclass
import time
from typing import List, Optional

from .decay_models import (
    DecayModel,
    HLRDecay,
    PowerLawDecay,
    ReinforcementDecay,
    WeightedEnsembleDecay,
)
from .models import Memory
from .bandit import LinUCBBandit, build_context


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
    # Weights are irrelevant in adaptive mode - the bandit selects a single arm
    weight_hlr=0.0,
    weight_power=0.0,
    weight_reinforcement=0.0,
)


class DecayEngine:
    def __init__(self, profile: BrainProfile = SMART_BRAIN):
        # Persist bandit state alongside the data directory
        _data_dir = os.path.join(os.path.dirname(__file__), "data")
        _bandit_path = os.path.join(_data_dir, "bandit_state.json")
        self.bandit = LinUCBBandit(alpha=1.0, persist_path=_bandit_path)
        self.profile = profile

        # Build the three individual models - used directly in adaptive arm selection
        self._hlr_model: Optional[HLRDecay] = None
        self._power_model: Optional[PowerLawDecay] = None
        self._reinf_model: Optional[ReinforcementDecay] = None

        # Blended ensemble model (used by smart/dumb profiles)
        self.model: Optional[WeightedEnsembleDecay] = None

        self.set_profile(profile)

    def set_profile(self, profile: BrainProfile):
        self.profile = profile
        self._rebuild_models()

    def _rebuild_models(self):
        """Rebuild all three individual models and the blended ensemble."""
        self._hlr_model = HLRDecay(
            base_half_life=self.profile.hlr_base_half_life,
            recall_boost=self.profile.hlr_recall_boost,
        )
        self._power_model = PowerLawDecay(alpha=self.profile.power_law_alpha)
        self._reinf_model = ReinforcementDecay(gamma=self.profile.reinforcement_gamma)

        if self.profile.name != "adaptive":
            self.model = WeightedEnsembleDecay(
                [
                    (self.profile.weight_hlr, self._hlr_model),
                    (self.profile.weight_power, self._power_model),
                    (self.profile.weight_reinforcement, self._reinf_model),
                ]
            )

    def age_seconds(self, memory: Memory, now: Optional[float] = None) -> float:
        t = now if now is not None else time.time()
        anchor = memory.last_accessed_at if memory.last_accessed_at is not None else memory.created_at
        return max(0.0, t - anchor)

    def effective_strength(
        self,
        memory: Memory,
        now: Optional[float] = None,
        query: str = "",
    ) -> float:
        """
        Compute the effective memory strength at time `now`.

        For adaptive profile: the LinUCB bandit selects one arm based on the
        context vector built from (memory, query, now). Only that arm's model
        is used to compute strength.

        For smart/dumb profiles: the fixed-weight ensemble is used, unchanged.

        Args:
            memory: The memory to score.
            now:    Simulated current timestamp (defaults to real time.time()).
            query:  The search query string. Passed through to the bandit context
                    vector so query complexity is a feature. Empty string is safe.
        """
        t = now if now is not None else time.time()

        if self.profile.name != "adaptive":
            return self.model.score(memory, now=t)

        # -- Adaptive mode: LinUCB arm selection --
        recall_count = int(memory.metadata.get("recall_count", 0))
        ctx = build_context(
            memory_strength=memory.strength,
            last_accessed_at=memory.last_accessed_at,
            recall_count=recall_count,
            created_at=memory.created_at,
            query=query,
            now=t,
        )
        arm = self.bandit.select(ctx)

        arm_models = [self._hlr_model, self._power_model, self._reinf_model]
        return arm_models[arm].score(memory, now=t)
