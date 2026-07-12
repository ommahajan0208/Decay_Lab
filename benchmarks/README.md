# Benchmarks

This directory contains benchmark methodology, results, and dataset documentation
for Decay Lab.

---

## Phase 1 - Decay Model Benchmark

Evaluates all three decay model families (HLR, Power-Law, Reinforcement) across
Smart, Adaptive, and Dumb parameter profiles - 9 model variants total - against
spaced-repetition review data.

### Dataset

Two modes are supported:

**Synthetic (default):** A calibrated synthetic SRS dataset generated entirely
in-process. The ground-truth recall probability uses the published HLR oracle
formula from Settles & Meeder (2016):

```text
half_life = 2^(bias + theta_s * log(1 + recall_count))
p_true    = 2^(-delta / half_life)
correct   ~ Bernoulli(p_true)
```

Interval distribution is LogNormal calibrated to real Duolingo statistics
(median ~3 days, sigma=1.5). No download required. Fully reproducible via `--seed`.

**Real HLR dataset:** The Duolingo Half-Life Regression dataset (Settles & Meeder
2016, 13 million learning traces). Download from Harvard Dataverse:

```
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME
```

Place the `.tsv.gz` file in the `datasets/` directory at the repo root, then pass
`--csv-path datasets/<filename>`.

Expected columns: `p_recall`, `delta`, `history_seen`, `history_correct`, `timestamp`,
`user_id`, `lexeme_id`.

### Models Evaluated

| Family | Variant | Key Parameters |
|---|---|---|
| HLR | smart | base_half_life=86400s, recall_boost=4.0 |
| HLR | adaptive | base_half_life=3600s, recall_boost=2.0 |
| HLR | dumb | base_half_life=1800s, recall_boost=1.5 |
| PowerLaw | smart | alpha=0.15 |
| PowerLaw | adaptive | alpha=0.35 |
| PowerLaw | dumb | alpha=0.60 |
| Reinforcement | smart | gamma=0.20 |
| Reinforcement | adaptive | gamma=0.15 |
| Reinforcement | dumb | gamma=0.05 |

### Metrics

- **Log-Loss** (lower is better) - measures calibration of predicted recall probability.
- **RMSE** (lower is better) - root mean squared error vs binary recall outcome.
- **ROC-AUC** (higher is better) - ranking quality: probability that a forgotten
  memory is ranked below a remembered one.

All metrics are compared against a naive baseline that always predicts the empirical
mean recall rate.

### Results

Latest results are stored in `results/benchmark_results.json` and
`results/tables/decay_metrics.csv`. Re-generate at any time by running the benchmark.

Run with default settings to reproduce:

```powershell
python benchmark_runner.py --seed 42 --n 50000
```

### Figures

Three figures are generated in `results/figures/`:

- **forgetting_curves.png** - predicted recall probability over 0-30 days for each
  model at recall_count=3. Shows the shape of each family's decay function.
- **roc_curve.png** - ROC curves with AUC values in the legend. Summarizes
  discriminative power per model.
- **calibration.png** - reliability diagram. X-axis: mean predicted probability.
  Y-axis: actual recall rate. Perfect calibration = diagonal.

### Run

```powershell
# Install matplotlib for figures (optional)
pip install matplotlib

# Default: 50 000 synthetic events
python benchmark_runner.py

# Larger run with calibration tables printed
python benchmark_runner.py --n 200000 --calibration

# Real Duolingo/HLR dataset
python benchmark_runner.py --csv-path datasets/settles.acl16.data.tsv.gz

# Skip figure generation
python benchmark_runner.py --no-plots

# Different seed
python benchmark_runner.py --seed 123
```

### Outputs

```text
results/
  benchmark_results.json      full metrics JSON report
  tables/
    decay_metrics.csv         paper-format table sorted by log-loss
  figures/
    forgetting_curves.png
    roc_curve.png
    calibration.png
  logs/
    benchmark.log             timestamped run log (appended on each run)
datasets/
  (place real dataset files here)
```

---

## Phase 2 - Retrieval Benchmark (Planned)

Will evaluate the 2-stage retrieval pipeline against standard IR baselines on
MS MARCO Dev-Small.

Planned metrics: P@1, MRR@10, Recall@10, NDCG@10, MAP.

Planned baselines: BM25, dense embeddings only, hybrid (bi-encoder + lexical), full
2-stage pipeline (bi-encoder + cross-encoder).

---

## Phase 3 - Bandit Benchmark (Planned)

Will simulate a contextual bandit environment to compare arm-selection algorithms
on cumulative reward and regret over a fixed trial horizon.

Planned algorithms: LinUCB, Greedy, Epsilon-Greedy, Thompson Sampling, Random.

Planned metrics: cumulative reward, cumulative regret, arm selection distribution,
learning curve (reward vs trial).
