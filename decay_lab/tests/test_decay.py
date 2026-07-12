from __future__ import annotations

import unittest

from decay_lab.decay_engine import DecayEngine
from decay_lab.decay_models import ExponentialDecay, PowerLawDecay, ReinforcementDecay, WeightedEnsembleDecay
from decay_lab.models import Memory
from decay_lab.retrieval import Retriever
from decay_lab.evaluator import Evaluator, EvalExample


class TestDecayEngine(unittest.TestCase):
    def test_decay_engine_monotonic_decrease(self):
        engine = DecayEngine()

        m = Memory(
            id="t1",
            content="hello",
            tags={},
            created_at=0.0,
            last_accessed_at=0.0,
            strength=1.0,
            metadata={},
        )

        s0 = engine.effective_strength(m, now=0.0)
        s1 = engine.effective_strength(m, now=10.0)
        s2 = engine.effective_strength(m, now=100.0)

        self.assertGreaterEqual(s0, s1)
        self.assertGreaterEqual(s1, s2)
        self.assertGreaterEqual(s2, 0.0)

    def test_age_seconds_uses_last_accessed_if_present(self):
        engine = DecayEngine()

        m = Memory(
            id="t2",
            content="hello",
            tags={},
            created_at=0.0,
            last_accessed_at=50.0,
            strength=1.0,
            metadata={},
        )

        # now=100 => age should be 50 (anchor last_accessed_at)
        age = engine.age_seconds(m, now=100.0)
        self.assertEqual(age, 50.0)


class TestDecayModels(unittest.TestCase):
    def test_exponential_decay_decreases_with_age(self):
        model = ExponentialDecay(lambda_decay=0.1)
        memory = Memory("m", "hello", {}, 0.0, 0.0, 1.0, {})

        self.assertGreater(model.score(memory, now=1.0), model.score(memory, now=10.0))

    def test_power_law_decay_decreases_with_age(self):
        model = PowerLawDecay(alpha=0.5, scale_seconds=1.0)
        memory = Memory("m", "hello", {}, 0.0, 0.0, 1.0, {})

        self.assertGreater(model.score(memory, now=1.0), model.score(memory, now=10.0))

    def test_reinforcement_decay_boosts_recalled_memories(self):
        model = ReinforcementDecay(gamma=0.1)
        weak = Memory("weak", "hello", {}, 0.0, 0.0, 1.0, {})
        reinforced = Memory("strong", "hello", {}, 0.0, 0.0, 1.0, {"recall_count": 10})

        self.assertGreater(model.score(reinforced, now=10.0), model.score(weak, now=10.0))

    def test_weighted_ensemble_normalizes_weights(self):
        memory = Memory("m", "hello", {}, 0.0, 0.0, 1.0, {})
        ensemble = WeightedEnsembleDecay(
            [
                (2.0, ExponentialDecay(lambda_decay=0.0)),
                (2.0, PowerLawDecay(alpha=0.0)),
            ]
        )

        self.assertEqual(ensemble.score(memory, now=100.0), 1.0)


class InMemoryStore:
    def __init__(self, memories):
        self.memories = list(memories)

    def list_all(self):
        return list(self.memories)

    def upsert(self, memory):
        for i, existing in enumerate(self.memories):
            if existing.id == memory.id:
                self.memories[i] = memory
                return
        self.memories.append(memory)

    def prune_inplace(self, keep_ids):
        keep = set(keep_ids)
        before = len(self.memories)
        self.memories = [m for m in self.memories if m.id in keep]
        return before - len(self.memories)


class TestRetriever(unittest.TestCase):
    def test_min_relevance_filters_unrelated_memories(self):
        store = InMemoryStore(
            [
                Memory("study", "Study notes: retrieval ranking and pruning", {}, 0.0, None, 1.0, {}),
                Memory("pizza", "I love pizza and italian food", {}, 0.0, None, 10.0, {}),
            ]
        )
        retriever = Retriever(store=store, decay_engine=DecayEngine(), use_semantic=False)

        results = retriever.rank("study", limit=5, now=0.0, min_relevance=0.01)

        self.assertEqual([r.memory.id for r in results], ["study"])

    def test_rank_multiplies_relevance_by_ensemble_strength(self):
        store = InMemoryStore(
            [
                Memory("low", "study", {}, 0.0, None, 1.0, {}),
                Memory("high", "study", {}, 0.0, None, 2.0, {}),
            ]
        )
        retriever = Retriever(store=store, decay_engine=DecayEngine(), use_semantic=False)

        results = retriever.rank("study", limit=5, now=0.0, min_relevance=0.01)

        self.assertEqual([r.memory.id for r in results], ["high", "low"])
        self.assertAlmostEqual(results[0].score, results[0].relevance * results[0].strength)

    def test_retrieve_and_touch_increments_recall_count(self):
        memory = Memory("study", "study notes", {}, 0.0, None, 1.0, {})
        store = InMemoryStore([memory])
        retriever = Retriever(store=store, decay_engine=DecayEngine(), use_semantic=False)

        results, _ = retriever.retrieve_and_touch("study", now=10.0, min_relevance=0.01)

        self.assertEqual([r.memory.id for r in results], ["study"])
        self.assertEqual(store.memories[0].metadata["recall_count"], 1)
        self.assertEqual(store.memories[0].last_accessed_at, 10.0)


class TestRetrieverExpanded(unittest.TestCase):
    def test_lexical_relevance(self):
        store = InMemoryStore(
            [
                Memory("m1", "pizza italian food", {}, 0.0, None, 1.0, {}),
                Memory("m2", "sunny weather warm", {}, 0.0, None, 1.0, {}),
            ]
        )
        retriever = Retriever(store=store, decay_engine=DecayEngine(), use_semantic=False)

        results = retriever.rank("italian pizza", limit=5, now=0.0, min_relevance=0.01)
        self.assertEqual([r.memory.id for r in results], ["m1"])
        self.assertGreater(results[0].relevance, 0.0)

    def test_rank_returns_all_matching_memories(self):
        store = InMemoryStore(
            [
                Memory("m1", "weekly meeting", {"topic": "work"}, 0.0, None, 1.0, {}),
                Memory("m2", "weekly status update", {"topic": "ml"}, 0.0, None, 1.0, {}),
            ]
        )
        retriever = Retriever(store=store, decay_engine=DecayEngine(), use_semantic=False)

        results = retriever.rank("weekly", limit=5, now=0.0)
        self.assertEqual(len(results), 2)


class TestEvaluatorExpanded(unittest.TestCase):
    def test_evaluator_metrics_mrr_and_map(self):
        store = InMemoryStore(
            [
                Memory("m1", "pizza italian", {}, 0.0, None, 1.0, {}),
                Memory("m2", "sunny weather", {}, 0.0, None, 1.0, {}),
            ]
        )
        retriever = Retriever(store=store, decay_engine=DecayEngine(), use_semantic=False)
        evaluator = Evaluator(retriever)

        examples = [
            EvalExample(query="italian", relevant_ids=["m1"]),
            EvalExample(query="weather", relevant_ids=["m2"]),
        ]

        metrics = evaluator.evaluate(examples, k=2)
        self.assertEqual(metrics.precision_at_k, 0.5)  # top 2 contains 1 target out of 2 items
        self.assertEqual(metrics.recall_at_k, 1.0)
        self.assertEqual(metrics.mrr_at_k, 1.0)
        self.assertEqual(metrics.map_at_k, 1.0)

    def test_evaluator_grid_search_optimize(self):
        store = InMemoryStore(
            [
                Memory("m1", "pizza italian", {}, 0.0, None, 1.0, {}),
                Memory("m2", "sunny weather", {}, 0.0, None, 1.0, {}),
            ]
        )
        retriever = Retriever(store=store, decay_engine=DecayEngine(), use_semantic=False)
        evaluator = Evaluator(retriever)

        examples = [
            EvalExample(query="pizza", relevant_ids=["m1"]),
        ]

        best_config, best_score = evaluator.optimize_weights(examples, k=2, metric="map_at_k")
        self.assertIn("semantic_alpha", best_config)
        self.assertGreater(best_score, 0.0)


from decay_lab.bandit import LinUCBBandit, build_context, N_ARMS, N_FEATURES


class TestLinUCBBandit(unittest.TestCase):
    def _make_bandit(self, alpha: float = 1.0) -> LinUCBBandit:
        return LinUCBBandit(alpha=alpha, persist_path=None)

    def _uniform_context(self) -> list:
        return [0.5] * N_FEATURES

    def test_select_returns_valid_arm_index(self):
        bandit = self._make_bandit()
        ctx = self._uniform_context()
        arm = bandit.select(ctx)
        self.assertIn(arm, range(N_ARMS))

    def test_select_stores_last_arm_and_context(self):
        bandit = self._make_bandit()
        ctx = self._uniform_context()
        arm = bandit.select(ctx)
        self.assertEqual(bandit.last_arm, arm)
        self.assertEqual(bandit.last_context, ctx)

    def test_update_only_modifies_chosen_arm(self):
        bandit = self._make_bandit()
        ctx = self._uniform_context()

        # Snapshot A matrices before update
        import copy
        A_before = copy.deepcopy(bandit.A)
        b_before = copy.deepcopy(bandit.b)

        chosen = bandit.select(ctx)
        bandit.update(reward=1.0)

        for arm in range(N_ARMS):
            if arm == chosen:
                # Chosen arm's matrices must have changed
                self.assertNotEqual(bandit.A[arm], A_before[arm], f"Arm {arm} A should change")
                self.assertNotEqual(bandit.b[arm], b_before[arm], f"Arm {arm} b should change")
            else:
                # Unchosen arms must be identical
                self.assertEqual(bandit.A[arm], A_before[arm], f"Arm {arm} A should not change")
                self.assertEqual(bandit.b[arm], b_before[arm], f"Arm {arm} b should not change")

    def test_update_appends_to_history(self):
        bandit = self._make_bandit()
        bandit.select(self._uniform_context())
        bandit.update(reward=1.0)
        self.assertEqual(len(bandit.history), 1)
        self.assertIn("arm", bandit.history[0])
        self.assertIn("reward", bandit.history[0])
        self.assertIn("context", bandit.history[0])

    def test_update_without_select_does_nothing(self):
        bandit = self._make_bandit()
        # No select() has been called - update should be a no-op
        bandit.update(reward=1.0)
        self.assertEqual(len(bandit.history), 0)

    def test_ucb_exploration_bonus_present_with_high_alpha(self):
        """With high alpha and a fresh bandit, UCB bonus should dominate."""
        bandit_low  = self._make_bandit(alpha=0.01)
        bandit_high = self._make_bandit(alpha=10.0)
        ctx = self._uniform_context()

        scores_low  = bandit_low.get_ucb_scores(ctx)
        scores_high = bandit_high.get_ucb_scores(ctx)

        # High alpha should produce strictly higher (or equal) UCB scores
        for lo, hi in zip(scores_low, scores_high):
            self.assertLessEqual(lo, hi)

    def test_build_context_returns_normalized_floats(self):
        import time
        now = time.time()
        ctx = build_context(
            memory_strength=0.9,
            last_accessed_at=now - 3600 * 48,  # 2 days ago
            recall_count=5,
            created_at=now - 3600 * 24 * 7,   # 7 days old
            query="what is machine learning",
            now=now,
        )
        self.assertEqual(len(ctx), N_FEATURES)
        for val in ctx:
            self.assertGreaterEqual(val, 0.0, "Context values must be >= 0")
            self.assertLessEqual(val, 1.0, "Context values must be <= 1 (normalized)")

    def test_get_theta_returns_correct_length(self):
        bandit = self._make_bandit()
        for arm in range(N_ARMS):
            theta = bandit.get_theta(arm)
            self.assertEqual(len(theta), N_FEATURES)


if __name__ == "__main__":
    unittest.main()
