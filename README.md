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

### v3: Reinforcement Learning & Contextual Bandits
- **Adaptive Brain Profile:** Added an `ADAPTIVE_BRAIN` profile that replaces static ensemble weights with dynamic weights.
- **Contextual Bandit:** Integrated a Thompson Sampling inspired Softmax bandit (`decay_lab/bandit.py`) to continuously learn and tune the exact decay blend (HLR vs Power vs Reinforcement).
- **User Feedback Loop:** Users can click like (thumbs up) or dislike (thumbs down) on retrieved memories in the UI. This reward signal is fed into the backend bandit, which updates its weight distribution in real-time.

### v4: Realistic Spaced Repetition Simulation: "The Student"
- **New Simulations Tab:** The web dashboard now has a dedicated Simulations tab.
- **"The Student" Scenario:** 10 exam topics are created on Day 0. Study sessions happen at Day 1 and Day 3 (only some topics recalled). Exam occurs on Day 7.
- **Time-Compressed Replay:** The simulation uses the `now=` parameter throughout the entire decay engine to replay 7 days worth of forgetting and reinforcement instantly.
- **Smart vs Dumb Benchmark:** Both Brain Profiles run against the identical study schedule and compete head-to-head. Smart Brain remembers topics it never recalled by leveraging its long half-life. Dumb Brain catastrophically forgets any topic it didn't study multiple times.
- **Rich Visualization:** Full day-by-day decay curves for all 10 topics (Smart vs Dumb overlaid), plus a detailed per-topic table with strength bars and pass/fail badges.

## What We Changed

We started with a single exponential decay function. That was simple, but it assumed every memory forgets in the same way.

Now the project uses a weighted ensemble of decay models:

```text
ensemble_strength =
  0.50 * exponential_decay
  0.30 * power_law_decay
  0.20 * reinforcement_decay
```

Then retrieval combines that ensemble strength with query relevance:

```text
final_score = lexical_relevance * ensemble_strength
```

We also added a small relevance threshold in the app so totally unrelated memories are hidden from the CLI output.

## Decay Models

All decay models live in:

```text
decay_lab/decay_models.py
```

They were originally split into multiple files, but we collapsed them into one file to keep the project easier to understand.

### ExponentialDecay

Classic time-based forgetting:

```text
strength * exp(-lambda * age)
```

This drops quickly over time. It is useful for temporary facts, recent chat details, or memories that should fade unless reinforced.

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
effective_strength(memory, now=None)
```

By default it creates this ensemble:

```text
50% ExponentialDecay
30% PowerLawDecay
20% ReinforcementDecay
```

Other code does not need to know the details of the ensemble. It just asks the engine for effective strength.

### decay_lab/retrieval.py

Ranks memories for a query.

Current retrieval behavior:

1. Tokenize the query.
2. Tokenize each memory.
3. Compute lexical relevance using Jaccard overlap.
4. Skip memories below `min_relevance`.
5. Compute ensemble strength with `DecayEngine`.
6. Compute final score:

```text
score = relevance * strength
```

7. Sort by score.
8. Touch returned memories by updating `last_accessed_at`.
9. Increment `metadata["recall_count"]`.

This is where relevance and memory strength meet.

### decay_lab/visualization.py

Simple helpers for inspecting memory strength over time.

Current responsibilities:

- Compute future strength series for memories.
- Print the top effective memories according to decay strength.

Important note: this is not query-specific ranking. It shows strongest memories overall.

### decay_lab/evaluator.py

Evaluation helper module.

This file is for measuring retrieval behavior and comparing outputs. It is useful as the project grows from a demo into experiments.

### decay_lab/tests/test_decay.py

Unit tests for the core behavior.

It currently checks:

- Exponential decay decreases over time.
- Power-law decay decreases over time.
- Reinforcement boosts recalled memories.
- Ensemble weights normalize correctly.
- Unrelated memories are filtered by minimum relevance.
- Retrieval ranks by `relevance * ensemble_strength`.
- Retrieval increments `recall_count`.

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

The project intentionally stays small:

```text
Decay_Lab/
  decay_lab/
    app.py
    bandit.py
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
    docs/
      quickstart.md
    simulations/
    tests/
      test_decay.py
  requirements.txt
  README.md
```

The ensemble idea is still there, but without a large file tree.

## Good Next Steps

Useful future improvements:

- Add interference decay for similar competing memories.
- Add memory types, such as `preference`, `temporary`, `project`, or `habit`.
- Give each memory type different ensemble weights.
- Add better tokenization so punctuation does not reduce matches.
- Add a small evaluator dataset to compare scoring formulas.
- Add CLI commands to inspect or reset `recall_count`.

