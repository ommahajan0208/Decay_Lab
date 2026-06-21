"""
simulations/student.py - "The Student" Spaced Repetition Simulation

Simulates two weeks of study sessions compressed into milliseconds.
10 exam topics are created on Day 0.
Some topics are recalled at Day 1 and Day 3 (simulated study sessions).
On Day 7 (exam day), we compute how many topics each brain still retains
above a threshold of 0.20 (i.e., "remembers").

Smart Brain: high half-life → remembers more on exam day.
Dumb Brain: low half-life → forgets quickly between sessions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

from decay_lab.models import Memory
from decay_lab.decay_engine import DecayEngine, SMART_BRAIN, DUMB_BRAIN, BrainProfile

DAY = 86400.0        # seconds in a day
THRESHOLD = 0.20     # memory must be above 20% to count as "remembered"

EXAM_TOPICS = [
    ("topic_01", "Quantum entanglement and Bell's theorem"),
    ("topic_02", "The Krebs cycle and ATP synthesis"),
    ("topic_03", "French Revolution causes and outcomes"),
    ("topic_04", "Gradient descent and backpropagation"),
    ("topic_05", "Keynesian vs Austrian economics"),
    ("topic_06", "DNA replication and proofreading"),
    ("topic_07", "Fourier transform and frequency domain"),
    ("topic_08", "Tectonic plate boundaries and seismology"),
    ("topic_09", "Classical conditioning - Pavlov experiments"),
    ("topic_10", "Bayes theorem and conditional probability"),
]

# Which topics are studied (recalled) on each day
STUDY_SCHEDULE: Dict[int, List[str]] = {
    1: ["topic_01", "topic_02", "topic_03", "topic_04", "topic_05"],   # Day 1: study first half
    3: ["topic_01", "topic_03", "topic_06", "topic_07", "topic_08"],   # Day 3: revisit some, add new
}


@dataclass
class MemorySnapshot:
    id: str
    content: str
    strength_at_exam: float
    recalled_on_days: List[int]
    remembered: bool          # above threshold on exam day


@dataclass 
class SimulationResult:
    profile_name: str
    snapshots: List[MemorySnapshot]
    remembered_count: int
    total: int
    timeline: List[Dict]      # day-by-day strength for charting


def _make_memories(t0: float) -> Dict[str, Memory]:
    """Create 10 fresh exam topic memories at t0 (Day 0)."""
    return {
        mid: Memory.new(
            id=mid,
            content=content,
            strength=1.0,
            created_at=t0,
        )
        for mid, content in EXAM_TOPICS
    }


def _simulate_recall(memories: Dict[str, Memory], topic_ids: List[str], t: float) -> None:
    """Simulate recalling specific topics at time t - updates last_accessed_at and recall_count."""
    for mid in topic_ids:
        if mid in memories:
            m = memories[mid]
            m.last_accessed_at = t
            m.metadata["recall_count"] = int(m.metadata.get("recall_count", 0)) + 1


def run_simulation(profile: BrainProfile, num_days: int = 7) -> SimulationResult:
    """
    Run the student simulation for a given BrainProfile.
    Time is fully compressed - we set `now=` timestamps to simulate days passing.
    """
    engine = DecayEngine(profile=profile)
    t0 = time.time()
    memories = _make_memories(t0)

    # Track which topics were recalled on which days
    recalled_map: Dict[str, List[int]] = {mid: [] for mid, _ in EXAM_TOPICS}

    # Replay study sessions using time-shifted `now=` calls
    for day, topic_ids in sorted(STUDY_SCHEDULE.items()):
        t_study = t0 + day * DAY
        _simulate_recall(memories, topic_ids, t=t_study)
        for mid in topic_ids:
            if mid in recalled_map:
                recalled_map[mid].append(day)

    # Day-by-day timeline for charting (Days 0 → exam day)
    timeline = []
    for day in range(num_days + 1):
        t_now = t0 + day * DAY
        day_entry = {"day": day, "strengths": {}}
        for mid, m in memories.items():
            day_entry["strengths"][mid] = round(engine.effective_strength(m, now=t_now), 4)
        timeline.append(day_entry)

    # Compute final strengths on Exam Day
    t_exam = t0 + num_days * DAY
    snapshots = []
    for mid, m in memories.items():
        s = engine.effective_strength(m, now=t_exam)
        snapshots.append(MemorySnapshot(
            id=mid,
            content=m.content,
            strength_at_exam=round(s, 4),
            recalled_on_days=recalled_map[mid],
            remembered=s >= THRESHOLD,
        ))

    remembered = sum(1 for s in snapshots if s.remembered)

    return SimulationResult(
        profile_name=profile.name,
        snapshots=snapshots,
        remembered_count=remembered,
        total=len(snapshots),
        timeline=timeline,
    )


def run_both() -> Tuple[SimulationResult, SimulationResult]:
    """Run simulation for both Smart and Dumb brains, return results."""
    smart = run_simulation(SMART_BRAIN)
    dumb  = run_simulation(DUMB_BRAIN)
    return smart, dumb
