# Decay Lab

An experimental framework that models AI agent memory as a biologically-inspired,
time-decaying, LinUCB-adaptive retrieval system.

> **Project Origin:** Conceived and prototype-developed during the
> [Build Day: Agent Harness & Memory](https://www.buildclub.com.au/) hackathon
> (Build Club x Mem0, June 13, 2026). Continued in personal development since.

---

## Core Philosophy

In traditional LLM agent architectures, memory is treated as a static database:
facts are stored indefinitely, retrieved via raw vector similarity, and only pruned
when the context window is exceeded. This leads to retrieval noise, memory overload,
and high token costs.

**Decay Lab** is built on three principles:

1. **Forgetting is a feature.** Temporary facts should decay rapidly. Core facts
   should persist long-term.
2. **Retention is active.** Every successful recall increases a memory's half-life,
   mimicking the spacing effect in cognitive psychology.
3. **Relevance is the master gate.** A memory with near-infinite strength should
   score zero if it is unrelated to the current query.

### Mathematical Model

$$S = R \times M_s$$

Where $R$ is contextual relevance and $M_s$ is ensemble memory strength:

```text
score = relevance * ensemble_strength
```

If relevance is zero, the final score is zero regardless of strength.

---

## Version History

### v0: Lexical Baseline
- Single exponential decay at a fixed rate for all memories.
- Purely lexical (Jaccard) retrieval with additive scoring.
- Bug: strong but unrelated memories appeared in results.

### v1: Weighted Ensemble and Web Dashboard
- Replaced single decay with a weighted ensemble of three models (HLR, Power-Law, Reinforcement).
- Fixed scoring to multiplicative (`score = relevance * strength`).
- Added a Flask web dashboard with Chart.js decay visualization.

### v2: Neural Retrieval and Brain Profiles
- Replaced Bag-of-Words with Sentence-BERT bi-encoder dense retrieval.
- Added BERT cross-encoder re-ranking (2-stage pipeline).
- Introduced HLR (Half-Life Regression) for dynamic half-life prediction.
- Added Smart Brain and Dumb Brain profiles with different retention behaviors.

### v3: LinUCB Contextual Bandit
- Added Adaptive Brain profile: a LinUCB-Disjoint bandit selects one decay model per query.
- Bandit receives a 5-feature context vector; only the chosen arm is scored.
- User thumbs-up/down feedback updates the selected arm's matrices exclusively.
- Arm matrices persist to disk between sessions.

### v4: "The Student" Simulation
- Added Simulations tab to the web dashboard.
- Simulates 10 exam topics over a compressed 7-day study/exam schedule.
- Smart Brain vs Dumb Brain compete on the same study plan.
- Full day-by-day decay curves and per-topic pass/fail table.

### v5: Phase 1 Decay Model Benchmark
- Added a reproducible benchmark suite (`benchmark_runner.py`) for all decay models.
- Supports both calibrated synthetic SRS data (zero download) and the real Duolingo HLR dataset.
- Generates paper-format metrics (Log-Loss, RMSE, AUC), figures, and timestamped logs.

---

## Project Structure

```text
Decay_Lab/
  decay_lab/          core library
    decay_models.py   HLR, PowerLaw, Reinforcement, WeightedEnsemble
    decay_engine.py   BrainProfile orchestrator + LinUCB integration
    bandit.py         LinUCB-Disjoint bandit (pure Python)
    retrieval.py      2-stage Bi-Encoder + Cross-Encoder pipeline
    evaluator.py      Precision@K, Recall@K, MRR@K, MAP@K
    memory_store.py   JSON-backed persistence + garbage collector
    models.py         Memory dataclass
    server.py         Flask REST API (15+ endpoints)
    simulations/      "The Student" scenario
    tests/            Unit test suite (22+ cases)
    web/              Dashboard HTML/JS/CSS
  benchmark_runner.py Phase 1 decay benchmark
  datasets/           Place real SRS CSV files here
  results/            Benchmark outputs (auto-created on first run)
  docs/               Extended documentation
  requirements.txt
```

---

## Quick Start

```powershell
# Install dependencies
pip install flask sentence-transformers matplotlib

# Run the CLI demo
python -m decay_lab.app

# Start the web dashboard (http://localhost:5000)
python decay_lab/server.py

# Run unit tests
python -m unittest -v decay_lab.tests.test_decay

# Run the decay model benchmark
python benchmark_runner.py
```

---

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | File-by-file architecture guide, retrieval pipeline, API reference |
| [docs/decay_models.md](docs/decay_models.md) | All decay model formulas, parameters, and behavior |
| [docs/testing.md](docs/testing.md) | Full test case descriptions and how to run them |
| [benchmarks/README.md](benchmarks/README.md) | Benchmark methodology, results, figures, and dataset guide |
| [decay_lab/docs/quickstart.md](decay_lab/docs/quickstart.md) | Minimal setup guide |


