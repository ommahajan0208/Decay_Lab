# Architecture Guide

This document covers the file-by-file architecture of Decay Lab, the retrieval
pipeline, the scoring formula, and the REST API surface.

---

## File Reference

### decay_lab/models.py

Defines the `Memory` dataclass - the core data object passed through every layer.

| Field | Type | Purpose |
|---|---|---|
| `id` | str | Stable memory identifier |
| `content` | str | Memory text |
| `tags` | dict | Lightweight metadata (topic, type, etc.) |
| `created_at` | float | Unix timestamp of creation |
| `last_accessed_at` | float | Unix timestamp of last retrieval |
| `strength` | float | Base importance (0.0 - 1.0) |
| `metadata` | dict | Dynamic fields; currently includes `recall_count` |

---

### decay_lab/memory_store.py

JSON-backed persistence layer. Loads and saves `decay_lab/data/memories.json`.

Operations: `get_all`, `upsert`, `delete`, `prune` (keep only selected IDs),
`garbage_collect` (delete memories below a strength threshold).

The file is human-readable and can be inspected or reset directly.

---

### decay_lab/decay_models.py

All decay model classes. See [decay_models.md](decay_models.md) for formulas.

Classes: `DecayModel` (ABC), `ExponentialDecay`, `HLRDecay`, `PowerLawDecay`,
`ReinforcementDecay`, `WeightedEnsembleDecay`.

---

### decay_lab/decay_engine.py

Wraps the decay model system behind one stable API:

```python
engine.effective_strength(memory, now=None, query="")
```

The engine selects model weights based on the active `BrainProfile`:

```text
SMART_BRAIN:    40% HLRDecay / 40% PowerLawDecay / 20% ReinforcementDecay
DUMB_BRAIN:     70% HLRDecay / 20% PowerLawDecay / 10% ReinforcementDecay
ADAPTIVE_BRAIN: LinUCB bandit selects one arm per query
```

In Adaptive mode, `build_context()` produces a 5-feature vector from the memory
and query state, then `bandit.select(context)` chooses one arm (HLR, PowerLaw,
or Reinforcement). Only that arm's score is returned; the other two are not evaluated.

Caller code never needs to know which arm was selected.

---

### decay_lab/bandit.py

Implements the `LinUCBBandit` class (LinUCB-Disjoint algorithm).

**Context vector** - 5 normalized features:

| Feature | Description |
|---|---|
| days since last access | Normalized elapsed time |
| log recall count | log(1 + recall_count), normalized |
| raw strength | Memory's current strength value |
| memory age | Days since creation, normalized |
| log query length | log(1 + word count), normalized |

**Per-arm state:** One 5x5 matrix `A` (initialized to identity) and one 5-vector
`b` (initialized to zeros).

**Selection:** For each arm `i`, compute `theta_i = A_i^-1 @ b_i`, then
UCB score = `theta_i @ x + alpha * sqrt(x @ A_i^-1 @ x)`. Select arm with
highest UCB.

**Update:** On feedback (reward +1 or -1), only the selected arm's matrices
are updated: `A[arm] += x @ x^T`, `b[arm] += reward * x`.

Matrix inversion uses pure-Python Gauss-Jordan (no numpy dependency).
Arm matrices persist to `decay_lab/data/bandit_state.json` between restarts.

---

### decay_lab/retrieval.py

2-stage neural retrieval pipeline.

**Stage 1 - Bi-Encoder:**
1. Encode query and all memories with Sentence-BERT (`all-MiniLM-L6-v2`).
2. Compute cosine similarity (semantic relevance).
3. Blend: `rel = 0.7 * semantic + 0.3 * jaccard`.
4. Filter by `min_relevance` threshold.
5. Score: `candidate_score = relevance * ensemble_strength`.
6. Keep top 20 candidates.

**Stage 2 - Cross-Encoder re-ranking:**
7. Score each `(query, memory)` pair with `ms-marco-MiniLM-L-6-v2`.
8. Convert logit to probability via sigmoid.
9. Recompute: `final_score = cross_encoder_relevance * strength`.
10. Return top `limit` results sorted by final score.

After ranking, returned memories have `last_accessed_at` updated and
`recall_count` incremented.

Degrades gracefully to pure Jaccard retrieval if `sentence-transformers` is absent.

---

### decay_lab/evaluator.py

Retrieval quality measurement and optimizer.

- `EvalExample` - a `(query, relevant_ids)` ground-truth pair.
- `Evaluator.evaluate(examples, k)` - runs all examples and returns `EvalMetrics`
  with Precision@K, Recall@K, MRR@K, MAP@K.
- `Evaluator.optimize_weights(examples, k, metric)` - grid-searches `semantic_alpha`
  from 0.0 to 1.0 in 0.2 steps to maximize a chosen metric.

---

### decay_lab/visualization.py

Helpers for inspecting memory strength over time (not query-specific ranking):

- `compute_strength_series(memory, days)` - returns a time series of strength values.
- `print_top_memories(store, engine)` - prints memories ranked by current strength.

---

### decay_lab/server.py

Flask REST API (~390 lines, 15+ endpoints).

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/memories` | GET | List all memories with current strength |
| `/api/memories/add` | POST | Insert a new memory |
| `/api/search` | POST | Retrieve memories for a query |
| `/api/time/advance` | POST | Advance simulated clock |
| `/api/sleep` | POST | Run sleep consolidation cycle |
| `/api/profile` | POST | Switch Brain Profile |
| `/api/feedback` | POST | Apply reward to last selected bandit arm |
| `/api/bandit` | GET | Arm names, UCB scores, theta, context, history |
| `/api/bandit/tune` | POST | Adjust exploration parameter alpha |
| `/api/series` | GET | Decay strength series for charting |
| `/api/brain_race` | GET | Strength of all memories under all 3 profiles |
| `/api/simulation/student` | POST | Run "The Student" scenario |
| `/api/graveyard` | GET | Memories deleted by garbage collector |

---

### decay_lab/simulations/

Contains the "Student" simulation engine (`student.py`).

Simulates 10 exam topics created on Day 0, studied on Days 1 and 3, with an exam
on Day 7. Uses the `now=` parameter throughout the decay engine to compress 7 days
into milliseconds of real time.

---

### decay_lab/app.py

Command-line demo entry point.

Seeds demo memories if the store is empty, then enters an interactive query loop:
prints score, relevance, strength, timestamp, and content for each result.

```powershell
python -m decay_lab.app
```

---

## Why Multiplicative Scoring

Earlier versions used additive scoring:

```text
score = 0.7 * strength + 0.3 * relevance
```

This allowed unrelated but high-strength memories to appear for any query.

The fix is multiplicative:

```text
score = relevance * strength
```

If relevance is zero, the final score is zero regardless of how strong the memory is.

---

## Brain Profile Weights

| Profile | HLR | PowerLaw | Reinforcement | Base half-life | Prune threshold |
|---|---|---|---|---|---|
| Smart | 40% | 40% | 20% | 86,400s (24h) | 0.01 |
| Dumb | 70% | 20% | 10% | 1,800s (30min) | 0.15 |
| Adaptive | bandit | - | - | profile-selected | profile-selected |
