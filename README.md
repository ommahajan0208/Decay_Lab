# Decay Lab Notes

Decay Lab is an experimental framework for simulating and designing human-inspired AI memory retrieval systems. 

> **Project Origin:** The initial idea for Decay Lab was conceived and prototype-developed during the [Build Day: Agent Harness & Memory](https://www.buildclub.com.au/) hackathon (hosted by **Build Club x Mem0** on Saturday, June 13, 2026). Since the event, it has been further developed and expanded personally.

## Core Philosophy & Ideology

In traditional LLM agent architectures, memory is treated as a static database: facts are stored indefinitely, retrieved via raw vector similarity, and only pruned when the context window is exceeded. This leads to information retrieval noise, memory overload, and high token costs.

**Decay Lab** is built on the premise that **agent memory should mirror the human brain**. Our cognitive architecture is governed by three fundamental laws:

1. **Forgetting is a Feature, Not a Bug:** Not all memories are created equal. Temporary facts (e.g., "The weather is currently rainy") should decay rapidly to free up cognitive space, while core facts (e.g., user preferences or system commands) should persist long-term.
2. **Retention is an Active Dynamic:** Human memory doesn't just decay; it gets reinforced. Every time a memory is successfully recalled, its half-life increases, mimicking the spacing effect found in cognitive psychology.
3. **Relevance is the Master Gate:** A memory might have near-infinite strength, but if it is completely irrelevant to the user's active query, its retrieval probability should drop to zero. 

### The Mathematical Model

To turn this ideology into code, we model memory retrieval score ($S$) as a multiplicative interaction of **Contextual Relevance** ($R$) and **Ensemble Memory Strength** ($M_s$):

$$S = R \times M_s$$

Where the **Ensemble Memory Strength** ($M_s$) is a weighted combination of three distinct decay models:
* **Exponential Decay:** Simulates short-term, rapid memory fading.
* **Power-Law Decay:** Models long-term, slow-decaying knowledge retention.
* **Reinforcement/Spacing Decay:** Boosts strength logarithmically based on recall frequency and timing.

The current scoring flow is:

```text
query
  -> lexical relevance
  -> ensemble memory strength
  -> final retrieval score
```

Final retrieval score:

```text
score = relevance * ensemble_strength
```

This matters because if relevance is zero, the final score is zero. Strong but unrelated memories should not appear for a query.

## Version History

### v0: The Lexical Baseline
- Started with a single exponential decay function assuming every memory forgets at the same rate.
- Retrieval was purely based on simple lexical token overlapping (Jaccard similarity).
- Score was an additive combination of relevance and strength, which incorrectly surfaced strong but completely unrelated memories.

### v1: Weighted Ensemble & Web Dashboard
- Replaced single exponential decay with a **Weighted Ensemble Decay** model combining:
  - Exponential Decay (short-term fading)
  - Power-Law Decay (long-term tail)
  - Reinforcement Decay (spaced-repetition boosts)
- Fixed the scoring bug by changing the formula to multiplicative (`score = relevance * strength`), ensuring unrelated items never surface.
- Built a Flask Web Dashboard with Chart.js to visualize multi-hour memory decay trajectories.

### v2: State-of-the-Art Neural Cognitive Engine
- **Neural Semantic Retrieval (Bi-Encoder):** Replaced Bag-of-Words with Sentence-BERT (`all-MiniLM-L6-v2`) dense embeddings for deep semantic matching.
- **Cross-Encoder Re-Ranking:** Introduced a 2-stage retrieval pipeline using BERT (`ms-marco-MiniLM-L-6-v2`) to jointly score query-memory pairs for high-precision final ranking.
- **Learned Decay (Half-Life Regression):** Replaced static decay rates with an `HLRDecay` module (Settles & Meeder, 2016) that predicts dynamic memory half-lives based on recall history.
- **Cognitive Brain Profiles:** Introduced simulated retention and storage strategies, controllable via the web dashboard:
  - *Smart Brain:* High retention, massive reinforcement boosts, high storage cost.
  - *Dumb Brain:* Fast decay with an active Garbage Collector that permanently deletes weak memories from the database to ensure storage efficiency.

### v3: Reinforcement Learning and Contextual Bandits
- **Adaptive Brain Profile:** Added an `ADAPTIVE_BRAIN` profile. Instead of using a fixed-weight ensemble, it selects exactly one decay model per query using a bandit.
- **LinUCB Contextual Bandit:** Implemented a LinUCB-Disjoint bandit (`decay_lab/bandit.py`). For each query the bandit receives a 5-feature context vector (days since last access, log recall count, raw strength, memory age, log query length), computes an Upper Confidence Bound score per arm, and selects the arm with the highest UCB score. Only the selected arm scores the memory; the other two arms are untouched.
- **True Bandit Update:** On thumbs-up or thumbs-down feedback, the reward is applied exclusively to the arm that was chosen for that query (`A[chosen] += x @ x^T`, `b[chosen] += reward * x`). Arm matrices persist to disk between sessions.
- **User Feedback Loop:** Thumbs-up or thumbs-down on retrieved memories in the UI sends a +1 or -1 reward to `/api/feedback`, which forwards it to the bandit's `update()` method.

### v4: Realistic Spaced Repetition Simulation: "The Student"
- **New Simulations Tab:** The web dashboard now has a dedicated Simulations tab.
- **"The Student" Scenario:** 10 exam topics are created on Day 0. Study sessions happen at Day 1 and Day 3 (only some topics recalled). Exam occurs on Day 7.
- **Time-Compressed Replay:** The simulation uses the `now=` parameter throughout the entire decay engine to replay 7 days worth of forgetting and reinforcement instantly.
- **Smart vs Dumb Benchmark:** Both Brain Profiles run against the identical study schedule and compete head-to-head. Smart Brain remembers topics it never recalled by leveraging its long half-life. Dumb Brain catastrophically forgets any topic it didn't study multiple times.
- **Rich Visualization:** Full day-by-day decay curves for all 10 topics (Smart vs Dumb overlaid), plus a detailed per-topic table with strength bars and pass/fail badges.

## What We Changed

We started with a single exponential decay function. That was simple, but it assumed every memory forgets in the same way.

Now the project uses a weighted ensemble of three decay models, with weights determined by the active Brain Profile. The default (Smart Brain) ensemble is:

```text
ensemble_strength =
  0.40 * hlr_decay          (Half-Life Regression)
  0.40 * power_law_decay
  0.20 * reinforcement_decay
```

The Dumb Brain profile uses a more aggressive configuration:

```text
ensemble_strength =
  0.70 * hlr_decay
  0.20 * power_law_decay
  0.10 * reinforcement_decay
```

Retrieval combines that ensemble strength with query relevance. When `sentence-transformers` is installed, relevance is a blend of semantic and lexical signals:

```text
relevance = 0.70 * semantic_score + 0.30 * lexical_score
final_score = relevance * ensemble_strength
```

Without `sentence-transformers`, the system falls back to pure lexical (Jaccard) relevance. A small relevance threshold filters completely unrelated memories from the output.

## Decay Models

All decay models live in:

```text
decay_lab/decay_models.py
```

They were originally split into multiple files, but we collapsed them into one file to keep the project easier to understand.

### HLRDecay

Half-Life Regression (Settles and Meeder, 2016). The primary model used by all Brain Profiles.

```text
strength * 2 ^ (-age / h)
```

Where `h` is a predicted half-life computed from recall history:

```text
h = 2 ^ (bias + recall_weight * log(1 + recall_count))
```

This means a memory that has been recalled many times gets a longer half-life, so it decays more slowly. It is the most cognitively realistic model in the ensemble.

### ExponentialDecay

Classic time-based forgetting:

```text
strength * exp(-lambda * age)
```

This drops quickly over time. It is useful for temporary facts, recent chat details, or memories that should fade unless reinforced. In earlier versions of the project this was the primary decay model; it has since been superseded by HLRDecay.

### PowerLawDecay

Long-tail forgetting:

```text
strength * (1 + age / scale) ^ -alpha
```

This also decays over time, but more gently. It is useful for things that tend to stick around longer, like skills, preferences, or repeated concepts.

### ReinforcementDecay

Recall-based strengthening:

```text
strength * (1 + gamma * log(1 + recall_count))
```

This rewards memories that have been retrieved before. If a memory keeps being useful, it becomes harder to forget.

The recall count is stored in:

```text
memory.metadata["recall_count"]
```

### WeightedEnsembleDecay

Combines multiple decay models using weights. The weights are normalized automatically, so these are equivalent:

```python
[(0.5, model_a), (0.5, model_b)]
[(5, model_a), (5, model_b)]
```

Both mean 50 percent model A and 50 percent model B.

## File Guide

### decay_lab/app.py

Command-line demo entry point.

Responsibilities:

- Seeds demo memories if the JSON store is empty.
- Prompts the user for queries.
- Retrieves relevant memories.
- Prints score, relevance, strength, timestamp, and content.
- Shows the top effective memories after each query.

Run it from the workspace root:

```powershell
python -m decay_lab.app
```

### decay_lab/models.py

Defines the `Memory` data object.

Important fields:

- `id`: stable memory identifier.
- `content`: the memory text.
- `tags`: lightweight metadata like topic.
- `created_at`: when the memory was created.
- `last_accessed_at`: when it was last retrieved.
- `strength`: base importance.
- `metadata`: extra dynamic fields, currently including `recall_count`.

### decay_lab/memory_store.py

JSON-backed memory storage.

Responsibilities:

- Load memories from `decay_lab/data/memories.json`.
- Save memories back to disk.
- Insert or update memories.
- Delete memories.
- Prune memories by keeping only selected IDs.

This keeps the app dependency-free and easy to inspect.

### decay_lab/decay_models.py

Contains all model classes used to estimate memory strength:

- `DecayModel`
- `ExponentialDecay`
- `PowerLawDecay`
- `ReinforcementDecay`
- `WeightedEnsembleDecay`

This is the research core of the project.

### decay_lab/decay_engine.py

Wraps the decay model system behind one stable API:

```python
effective_strength(memory, now=None, query="")
```

The engine selects weights and decay parameters based on the active `BrainProfile`. Three built-in profiles are defined:

```text
SMART_BRAIN:    40% HLRDecay / 40% PowerLawDecay / 20% ReinforcementDecay
DUMB_BRAIN:     70% HLRDecay / 20% PowerLawDecay / 10% ReinforcementDecay
ADAPTIVE_BRAIN: LinUCB bandit selects one arm per query; no blending
```

In adaptive mode, the engine passes the query string and memory state to `build_context()` to produce a 5-feature vector, calls `bandit.select(context)` to pick one arm (HLR, Power-Law, or Reinforcement), and returns only that arm's score. The other two arms are not evaluated.

Other code does not need to know which arm was chosen. It just calls `effective_strength` and receives a float.

### decay_lab/retrieval.py

Ranks memories for a query using a 2-stage neural retrieval pipeline.

Stage 1 - Bi-Encoder retrieval:

1. Encode the query and all memories using Sentence-BERT (`all-MiniLM-L6-v2`).
2. Compute cosine similarity scores (semantic relevance).
3. Blend with Jaccard lexical relevance: `rel = 0.7 * semantic + 0.3 * lexical`.
4. Skip memories below `min_relevance`.
5. Compute ensemble strength with `DecayEngine`.
6. Compute candidate score: `score = relevance * strength`.
7. Sort and keep top 20 candidates.

Stage 2 - Cross-Encoder re-ranking:

8. Score each `(query, memory)` pair with a BERT Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
9. Convert logit to a 0-1 probability via sigmoid.
10. Recompute final score: `score = cross_encoder_relevance * strength`.
11. Sort by final score and return top `limit` results.

After ranking, touch returned memories by updating `last_accessed_at` and incrementing `metadata["recall_count"]`.

When `sentence-transformers` is not installed, the pipeline degrades gracefully to pure lexical (Jaccard) relevance with no re-ranking.

### decay_lab/visualization.py

Simple helpers for inspecting memory strength over time.

Current responsibilities:

- Compute future strength series for memories.
- Print the top effective memories according to decay strength.

Important note: this is not query-specific ranking. It shows strongest memories overall.

### decay_lab/evaluator.py

Benchmark harness for measuring and optimizing retrieval quality.

It supports:

- `EvalExample`: a (query, relevant_ids) pair used as ground truth.
- `Evaluator.evaluate(examples, k)`: runs all examples through the retriever and returns `EvalMetrics` containing Precision@K, Recall@K, MRR@K, and MAP@K.
- `Evaluator.optimize_weights(examples, k, metric)`: grid-searches over `semantic_alpha` values to find the retriever configuration that maximizes a chosen metric.

### decay_lab/bandit.py

Implements the `LinUCBBandit` class and context vector helpers.

Key components:

- `LinUCBBandit` - LinUCB-Disjoint algorithm. Maintains one `A` matrix and `b` vector per arm. All matrix operations are pure Python (Gauss-Jordan inversion, no numpy).
- `build_context(memory_strength, last_accessed_at, recall_count, created_at, query, now)` - converts raw memory state and query into a normalized 5-element float vector.
- `FEATURE_NAMES` and `ARM_NAMES` - exported constants used by the dashboard.
- `persist_path` - optional path to a JSON file for saving and loading arm matrices between server restarts.

### decay_lab/tests/test_decay.py

Unit tests for the core behavior.

It currently checks:

- Engine produces monotonically decreasing strength over time.
- Age calculation uses `last_accessed_at` as anchor when present.
- Exponential decay decreases over time.
- Power-law decay decreases over time.
- Reinforcement boosts recalled memories.
- Ensemble weights normalize correctly.
- Unrelated memories are filtered by minimum relevance.
- Retrieval ranks by `relevance * ensemble_strength`.
- Retrieval increments `recall_count` and updates `last_accessed_at`.
- Lexical relevance returns only matching memories.
- Retrieval returns all matching memories up to the limit.
- `Evaluator.evaluate` computes correct Precision@K, Recall@K, MRR@K, and MAP@K.
- `Evaluator.optimize_weights` returns a valid `semantic_alpha` configuration.
- `LinUCBBandit.select` returns a valid arm index (0, 1, or 2).
- `select` stores `last_arm` and `last_context` for the feedback call.
- `update` modifies only the chosen arm's `A` and `b` matrices.
- `update` appends a correct entry to `history`.
- `update` without a prior `select` is a no-op.
- UCB scores are strictly higher with a larger `alpha` (exploration bonus confirmed).
- `build_context` returns a 5-element vector with all values in `[0, 1]`.
- `get_theta` returns a vector of length `N_FEATURES`.

Run tests from the workspace root:

```powershell
python -m unittest -v decay_lab.tests.test_decay
```

### decay_lab/data/memories.json

Local JSON memory database used by the demo.

This file changes when memories are touched because `last_accessed_at` and `metadata["recall_count"]` can be updated during retrieval.

### decay_lab/docs/quickstart.md

Short setup and usage guide.

Use this when you just want to run the app or tests without reading the architecture notes.

## Why The New Score Is Better

The old retrieval score mixed strength and relevance additively:

```text
score = 0.7 * strength + 0.3 * relevance
```

That allowed unrelated but strong memories to appear.

Example:

```text
query = "study"
```

A memory about running or pizza could still show up if its strength was high enough.

The new score fixes that:

```text
score = relevance * strength
```

If relevance is zero:

```text
score = 0 * strength = 0
```

So unrelated memories do not appear.

## Current Project Shape

```text
Decay_Lab/
  decay_lab/
    app.py
    bandit.py          (LinUCBBandit, build_context, Gauss-Jordan inverse)
    dashboard.html
    decay_engine.py
    decay_models.py
    evaluator.py
    memory_store.py
    models.py
    retrieval.py
    server.py
    visualization.py
    data/
      memories.json
      bandit_state.json
    docs/
      quickstart.md
    simulations/
      __init__.py
      student.py
    tests/
      test_decay.py
    web/
      index.html
      script.js
      style.css
  benchmark_runner.py   (Phase 1 decay benchmark - see Benchmarks section)
  datasets/             (place real HLR/FSRS CSV files here)
  results/
    benchmark_results.json
    tables/
      decay_metrics.csv
    figures/
      forgetting_curves.png
      roc_curve.png
      calibration.png
    logs/
      benchmark.log
  requirements.txt
  README.md
```

## Benchmarks

Decay Lab includes a benchmark suite for evaluating the core decay models against
spaced-repetition review data.

### Phase 1 - Decay Model Benchmark

**Dataset** - Calibrated synthetic SRS data generated from published HLR statistics
(Settles & Meeder 2016, Ebbinghaus forgetting curve, FSRS Ye 2022). The synthetic
oracle uses the exact HLR formula as ground truth so every metric is interpretable.
To use the real Duolingo dataset instead, download it from Harvard Dataverse and
pass `--csv-path`.

**Models evaluated** - 9 variants across all three decay families:

| Family | Variants |
|---|---|
| HLR | smart (hl=86400s, boost=4.0), adaptive (hl=3600s, boost=2.0), dumb (hl=1800s, boost=1.5) |
| Power-Law | smart (alpha=0.15), adaptive (alpha=0.35), dumb (alpha=0.60) |
| Reinforcement | smart (gamma=0.20), adaptive (gamma=0.15), dumb (gamma=0.05) |

**Metrics** - Log-Loss, RMSE, ROC-AUC vs a naive mean-predict baseline.

**Results (n=50,000 synthetic events, seed=42)**

| Model | Log-Loss | RMSE | AUC |
|---|---|---|---|
| HLR smart (hl=86400s boost=4.0) | **0.4573** | **0.3830** | **0.8618** |
| PowerLaw smart (alpha=0.15) | 0.5746 | 0.4397 | 0.8523 |
| PowerLaw adaptive (alpha=0.35) | 0.9291 | 0.5932 | 0.8523 |
| PowerLaw dumb (alpha=0.60) | 1.5071 | 0.7105 | 0.8523 |
| Reinforcement (all variants) | 6.95 | 0.579 | ~0.50 |
| Baseline (mean predict) | 0.6380 | 0.4722 | 0.5000 |

Key findings:

- HLR Smart beats the naive baseline on all three metrics (28% log-loss improvement, AUC 0.86).
- Power-Law Smart also beats the baseline. Its AUC is almost identical to HLR's, which means the
  shape is correct - only the scale differs.
- Reinforcement models output a constant prediction near 1.0 because the formula has no elapsed-time
  term. They are designed as ensemble boosters, not standalone predictors. The benchmark
  confirms they should never be selected alone by the bandit.
- HLR Adaptive/Dumb half-lives (1h/30min) are too short for the ~9 day mean review interval,
  producing near-zero predictions.

**Run**

```powershell
# Install matplotlib for figure generation (optional)
pip install matplotlib

# Default: 50 000 synthetic events
python benchmark_runner.py

# Larger run with calibration tables
python benchmark_runner.py --n 200000 --calibration

# Real Duolingo/HLR dataset (download from Harvard Dataverse first)
python benchmark_runner.py --csv-path datasets/settles.acl16.data.tsv.gz

# Skip figures
python benchmark_runner.py --no-plots
```

**Outputs**

```text
results/
  benchmark_results.json        full metrics report
  tables/
    decay_metrics.csv           paper-format table sorted by log-loss
  figures/
    forgetting_curves.png       9 model decay curves over 0-30 days
    roc_curve.png               ROC curves with AUC in legend
    calibration.png             reliability diagram (predicted vs actual)
  logs/
    benchmark.log               timestamped run log
```

---

## Good Next Steps

- Train LinUCB theta vectors from real user interaction logs rather than starting from identity matrices.
- Add a contextual mixture-of-experts mode that keeps the blending behavior but makes weights context-dependent (`w(x) = softmax(W @ x + b)`).
- Replace JSON flat-file storage with a vector database (FAISS or Chroma) to support semantic ANN search at scale.
- Add memory types such as `preference`, `temporary`, `project`, or `habit`, with per-type decay profile overrides.
- Add better tokenization so punctuation does not reduce lexical match scores.
- Implement a per-query interference suppression mechanism that penalizes all memories similar to the top result, not just on insert.
- Add CLI commands to inspect or reset `recall_count` and to dump per-arm theta vectors.
- Phase 2 benchmark: retrieval evaluation against MS MARCO Dev-Small (P@1, MRR@10, NDCG@10).
- Phase 3 benchmark: bandit regret comparison of LinUCB vs Greedy vs Epsilon-Greedy.

