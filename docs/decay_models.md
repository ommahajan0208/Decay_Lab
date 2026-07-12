# Decay Models

All decay model classes live in `decay_lab/decay_models.py`.
Each implements the `DecayModel` abstract base class with a single method:
`score(memory, now=None) -> float`.

---

## HLRDecay

**Half-Life Regression** (Settles & Meeder, 2016). The primary model used by
all Brain Profiles.

```text
score = strength * 2^(-age / h)
```

Where `h` is a predicted half-life computed from recall history:

```text
h = 2^(bias + recall_weight * log(1 + recall_count))
```

A memory that has been recalled many times gets a longer half-life and decays
more slowly. This is the most cognitively realistic model in the ensemble.

**Parameters:**

| Parameter | Smart | Adaptive | Dumb |
|---|---|---|---|
| `base_half_life` | 86,400s (24h) | 3,600s (1h) | 1,800s (30min) |
| `recall_boost` | 4.0 | 2.0 | 1.5 |

**Benchmark result (Phase 1):** HLR Smart beats the naive baseline on all three
metrics. See [benchmarks/README.md](../benchmarks/README.md) for the full table.

---

## ExponentialDecay

Classic Ebbinghaus-style forgetting. Used as the v0 baseline; superseded by HLR.

```text
score = strength * exp(-lambda * age)
```

Drops quickly over time. Useful for temporary facts or recent chat details.

---

## PowerLawDecay

Long-tail forgetting suitable for durable knowledge (skills, preferences).

```text
score = strength * (1 + age / scale)^(-alpha)
```

Decays more slowly than exponential. At high `alpha` values the score converges
toward a stable floor rather than zero.

**Parameters:**

| Parameter | Smart | Adaptive | Dumb |
|---|---|---|---|
| `alpha` | 0.15 | 0.35 | 0.60 |

**Benchmark note:** PowerLaw Smart beats the baseline on all metrics. Higher alpha
variants maintain good AUC (correct ranking) but become too pessimistic in absolute
probability (low mean prediction vs actual recall rate).

---

## ReinforcementDecay

Recall-based strengthening. Rewards memories that have been retrieved frequently.

```text
score = strength * (1 + gamma * log(1 + recall_count))
```

The recall count is stored in `memory.metadata["recall_count"]` and is incremented
on every retrieval.

**Parameters:**

| Parameter | Smart | Adaptive | Dumb |
|---|---|---|---|
| `gamma` | 0.20 | 0.15 | 0.05 |

**Important:** This model has no elapsed-time term. It always returns a value >= 1.0
for any recalled memory, making it unsuitable as a standalone predictor. It is
designed as an ensemble booster: contributing 10-20% weight alongside time-decay
models. The Phase 1 benchmark confirmed it should never be the bandit's sole arm.

---

## WeightedEnsembleDecay

Combines multiple decay models using normalized weights.

```python
WeightedEnsembleDecay([
    (0.40, HLRDecay(...)),
    (0.40, PowerLawDecay(...)),
    (0.20, ReinforcementDecay(...)),
])
```

Weights are normalized automatically. These two are equivalent:

```python
[(0.5, model_a), (0.5, model_b)]
[(5,   model_a), (5,   model_b)]
```

The Smart Brain profile ensemble:

```text
ensemble_strength =
  0.40 * hlr_decay
  0.40 * power_law_decay
  0.20 * reinforcement_decay
```

The Dumb Brain profile ensemble:

```text
ensemble_strength =
  0.70 * hlr_decay
  0.20 * power_law_decay
  0.10 * reinforcement_decay
```

In Adaptive mode, `WeightedEnsembleDecay` is not used. The LinUCB bandit selects
exactly one model per query.

---
