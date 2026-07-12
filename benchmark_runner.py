"""
benchmark_runner.py - SRS Forgetting Curve Benchmark for Decay Lab
===================================================================

Evaluates HLR, Power-Law, and Reinforcement decay models against
spaced-repetition review data in the Duolingo HLR format.

DATA SOURCE - two modes (automatically selected):

  Mode A - Real HLR dataset (recommended for publication-quality results):
    Download the Duolingo/HLR dataset from Harvard Dataverse:
      https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME
    Unzip and place the .gz or .tsv file anywhere, then pass its path:
      python benchmark_runner.py --csv-path /path/to/settles.acl16.data.gz

    The CSV columns expected (Settles & Meeder 2016 format):
      p_recall, timestamp, delta, user_id, learning_language,
      ui_language, lexeme_id, lexeme_string, history_seen,
      history_correct, session_seen, session_correct

  Mode B - Calibrated synthetic dataset (default when no --csv-path given):
    Generated entirely in-process from real SRS statistics published in:
      - Settles & Meeder (2016) - mean delta, p_recall, history_seen distributions
      - Ebbinghaus forgetting curve (1885/2015 replication - Murre & Dros)
      - FSRS paper (Ye 2022) - retention rate calibration
    This is a zero-download, fully reproducible baseline.

METRICS:
  - Log-Loss  (lower is better)
  - RMSE      (lower is better)
  - ROC-AUC   (higher is better)
  - Calibration table per model (--calibration flag)

USAGE:
  python benchmark_runner.py                          # synthetic, 50 000 events
  python benchmark_runner.py --n 200000               # synthetic, 200 000 events
  python benchmark_runner.py --csv-path data.tsv.gz   # real HLR dataset
  python benchmark_runner.py --calibration            # + calibration tables
  python benchmark_runner.py --seed 99                # different random seed
  python benchmark_runner.py --no-plots               # skip figure generation
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from decay_lab.decay_models import HLRDecay, PowerLawDecay, ReinforcementDecay
from decay_lab.models import Memory

# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------
RESULTS_DIR  = ROOT / "results"
TABLES_DIR   = RESULTS_DIR / "tables"
FIGURES_DIR  = RESULTS_DIR / "figures"
LOGS_DIR     = RESULTS_DIR / "logs"
DATASETS_DIR = ROOT / "datasets"

for _d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR, DATASETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"
METRICS_CSV  = TABLES_DIR  / "decay_metrics.csv"
LOG_FILE     = LOGS_DIR    / "benchmark.log"

# ---------------------------------------------------------------------------
# Models - every decay family with the three profile parameter sets
# ---------------------------------------------------------------------------
MODELS: Dict[str, object] = {
    "HLR  smart    (hl=86400s  boost=4.0)":    HLRDecay(base_half_life=86400.0, recall_boost=4.0),
    "HLR  adaptive (hl=3600s   boost=2.0)":    HLRDecay(base_half_life=3600.0,  recall_boost=2.0),
    "HLR  dumb     (hl=1800s   boost=1.5)":    HLRDecay(base_half_life=1800.0,  recall_boost=1.5),
    "PowerLaw  smart    (alpha=0.15)":          PowerLawDecay(alpha=0.15),
    "PowerLaw  adaptive (alpha=0.35)":          PowerLawDecay(alpha=0.35),
    "PowerLaw  dumb     (alpha=0.60)":          PowerLawDecay(alpha=0.60),
    "Reinforcement  smart    (gamma=0.20)":     ReinforcementDecay(gamma=0.20),
    "Reinforcement  adaptive (gamma=0.15)":     ReinforcementDecay(gamma=0.15),
    "Reinforcement  dumb     (gamma=0.05)":     ReinforcementDecay(gamma=0.05),
}

# ---------------------------------------------------------------------------
# Review event data structure
# ---------------------------------------------------------------------------

class ReviewEvent(NamedTuple):
    elapsed_seconds: float  # time since last review of this item
    recall_count: int        # number of previous reviews (0 = first review)
    correct: int             # 1 = recalled correctly, 0 = forgot


# ===========================================================================
# MODE A  -  Real HLR CSV loader
# ===========================================================================

_MIN_DELTA = 300.0  # skip reviews with < 5-minute gap (same-session repeats)


def _open_csv(path: Path):
    """Return a file-like text stream - handles .gz and plain files."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def load_hlr_csv(path: Path, max_rows: Optional[int]) -> List[ReviewEvent]:
    """
    Parse Duolingo HLR format.
    Columns: p_recall, timestamp, delta, ..., history_seen, history_correct, ...
    delta is in seconds since last review of that lexeme for that user.
    """
    events: List[ReviewEvent] = []
    rows_read = 0

    with _open_csv(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            # Try comma delimiter
            f.seek(0)
            reader = csv.DictReader(f, delimiter=",")

        required = {"p_recall", "delta", "history_seen"}
        if reader.fieldnames:
            found = set(reader.fieldnames)
            missing = required - found
            if missing:
                raise ValueError(
                    f"CSV missing required columns: {missing}\n"
                    f"Columns found: {sorted(found)}"
                )
        print(f"    Columns: {', '.join(list(reader.fieldnames or [])[:8])}...")

        for row in reader:
            if max_rows and rows_read >= max_rows:
                break
            rows_read += 1
            try:
                seen  = int(float(row["history_seen"]))
                delta = float(row["delta"])
                p     = float(row["p_recall"])
            except (KeyError, ValueError):
                continue

            if seen == 0 or delta < _MIN_DELTA:
                continue

            correct = 1 if p >= 0.5 else 0
            events.append(ReviewEvent(
                elapsed_seconds=delta,
                recall_count=seen,
                correct=correct,
            ))

    print(f"    Rows read  : {rows_read:,}")
    print(f"    Valid events: {len(events):,}")
    return events


# ===========================================================================
# MODE B  -  Calibrated synthetic dataset
# ===========================================================================
#
# All parameters are grounded in published SRS research:
#
# Recall rate vs interval:
#   Ebbinghaus (1885) + Murre & Dros (2015 replication):
#     R(t) = e^(-t/S)   where S is stability in days
#   Mean baseline stability from HLR paper: ~3-4 days for new words
#
# Half-life distribution (from Settles & Meeder 2016, Table 2):
#   Median half-life: ~4 days; range 0.5h - 30 days
#
# Elapsed interval distribution (from HLR dataset statistics):
#   Median delta: ~3 days; right-skewed (many short, some very long)
#   Modelled as LogNormal(mu=log(3*86400), sigma=1.5)
#
# Recall count distribution:
#   Most cards have 1-5 prior reviews; modelled as Geometric(p=0.35)
#
# True recall probability given (delta, recall_count):
#   Using HLR ground truth formula:
#     half_life = 2^(bias + theta_s * log1p(seen))
#     bias = log2(86400 * 3.0)   ~ 3 day baseline half-life
#     theta_s = 1.2              ~ strong recall boost per repetition
#     p_true = 2^(-delta / half_life)
#   Then inject Bernoulli noise: correct ~ Bernoulli(p_true)
#
# These numbers replicate the distributions visible in the HLR paper
# appendix and the FSRS calibration data (Ye 2022).
# ===========================================================================

_HLR_BIAS     = math.log2(86400.0 * 3.0)  # 3-day baseline half-life
_HLR_THETA_S  = 1.2                        # recall count coefficient
_DELTA_MU     = math.log(3.0 * 86400.0)   # median ~3 days in seconds
_DELTA_SIGMA  = 1.5                        # spread of interval distribution
_RECALL_GEOM  = 0.35                       # p for geometric recall count dist


def _sample_geometric(p: float, rng: random.Random) -> int:
    """Sample from Geometric(p) - minimum value 0."""
    k = 0
    while rng.random() > p and k < 50:
        k += 1
    return k


def _lognormal(mu: float, sigma: float, rng: random.Random) -> float:
    """Sample from LogNormal(mu, sigma) using Box-Muller."""
    u1 = rng.random()
    u2 = rng.random()
    z  = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
    return math.exp(mu + sigma * z)


def _true_recall_prob(elapsed_seconds: float, recall_count: int) -> float:
    """
    Ground-truth recall probability using the HLR formula from
    Settles & Meeder (2016). This is what the models are trying to predict.
    """
    log_h = _HLR_BIAS + _HLR_THETA_S * math.log1p(recall_count)
    half_life = 2.0 ** log_h
    p = 2.0 ** (-elapsed_seconds / max(half_life, 1.0))
    return max(0.0, min(1.0, p))


def generate_synthetic_events(n: int, seed: int) -> List[ReviewEvent]:
    """
    Generate n statistically calibrated synthetic review events.
    The ground-truth p_recall uses the published HLR formula; binary
    correct/incorrect is sampled from Bernoulli(p_true).
    """
    rng = random.Random(seed)
    events: List[ReviewEvent] = []

    for _ in range(n):
        # Sample recall count (0 = first review - no interval; skip)
        recall_count = _sample_geometric(_RECALL_GEOM, rng) + 1  # ensure >= 1

        # Sample elapsed interval from LogNormal
        delta = _lognormal(_DELTA_MU, _DELTA_SIGMA, rng)
        delta = max(_MIN_DELTA, delta)

        # Compute ground-truth recall probability (HLR model as oracle)
        p_true = _true_recall_prob(delta, recall_count)

        # Bernoulli draw
        correct = 1 if rng.random() < p_true else 0

        events.append(ReviewEvent(
            elapsed_seconds=delta,
            recall_count=recall_count,
            correct=correct,
        ))

    return events


# ===========================================================================
# Memory construction
# ===========================================================================
_NOW = time.time()


def _event_to_memory(ev: ReviewEvent) -> Tuple[Memory, float]:
    """
    Map a ReviewEvent to a (Memory, now) pair such that:
      _age_seconds(memory, now) == ev.elapsed_seconds

    Both created_at and last_accessed_at are set to (now - elapsed) so
    any decay model's age calculation gives the exact real elapsed time.
    """
    anchor = _NOW - ev.elapsed_seconds
    m = Memory(
        id="__bm__",
        content="",
        tags={},
        created_at=anchor,
        last_accessed_at=anchor,
        strength=1.0,
        metadata={"recall_count": ev.recall_count},
    )
    return m, _NOW


# ===========================================================================
# Metrics
# ===========================================================================
_EPS = 1e-9


def _clamp(p: float) -> float:
    return max(_EPS, min(1.0 - _EPS, p))


def _roc_auc(scores: List[float], labels: List[int]) -> float:
    """ROC-AUC via Wilcoxon rank-sum (stdlib only)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    for rank_1, (_, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum += rank_1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def score_model(model, events: List[ReviewEvent]) -> dict:
    log_losses: List[float] = []
    sq_errors:  List[float] = []
    preds:      List[float] = []
    labels:     List[int]   = []

    for ev in events:
        m, now = _event_to_memory(ev)
        p = _clamp(model.score(m, now=now))
        y = float(ev.correct)

        log_losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
        sq_errors.append((p - y) ** 2)
        preds.append(p)
        labels.append(ev.correct)

    n = len(events)
    return {
        "log_loss":      round(sum(log_losses) / n, 6),
        "rmse":          round(math.sqrt(sum(sq_errors) / n), 6),
        "auc":           round(_roc_auc(preds, labels), 6),
        "mean_pred":     round(sum(preds) / n, 4),
        "actual_recall": round(sum(labels) / n, 4),
        "n":             n,
    }


def baseline_stats(events: List[ReviewEvent]) -> dict:
    """Naive baseline: always predict the empirical mean recall rate."""
    n = len(events)
    actual = sum(e.correct for e in events) / n
    p = _clamp(actual)
    ll = -(actual * math.log(p) + (1 - actual) * math.log(1 - p))
    return {
        "log_loss":      round(ll, 6),
        "rmse":          round(math.sqrt(actual * (1 - actual)), 6),
        "auc":           0.5,
        "mean_pred":     round(actual, 4),
        "actual_recall": round(actual, 4),
        "n":             n,
    }


# ===========================================================================
# Calibration
# ===========================================================================

def calibration_table(model, events: List[ReviewEvent], n_bins: int = 10) -> List[dict]:
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for ev in events:
        m, now = _event_to_memory(ev)
        p = _clamp(model.score(m, now=now))
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, ev.correct))

    rows = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        lo, hi = i / n_bins, (i + 1) / n_bins
        mp   = sum(x[0] for x in bucket) / len(bucket)
        ar   = sum(x[1] for x in bucket) / len(bucket)
        rows.append({
            "range":         f"{lo:.1f}-{hi:.1f}",
            "n":             len(bucket),
            "mean_pred":     round(mp, 3),
            "actual_recall": round(ar, 3),
            "gap":           round(ar - mp, 3),
        })
    return rows


# ===========================================================================
# Display
# ===========================================================================
COL_W = 46


def _bar(value: float, width: int = 14) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_header(n_events: int, actual_recall: float, source: str) -> None:
    sep = "=" * 96
    print()
    print(sep)
    print("  Decay Lab - SRS Forgetting Curve Benchmark")
    print(sep)
    print(f"  Source : {source}")
    print(f"  Events : {n_events:,}  |  Actual recall rate: {actual_recall:.1%}")
    print(f"  Metric : Lower log-loss/RMSE = better  |  Higher AUC = better")
    print(sep)


def print_main_table(results: Dict[str, dict], base: dict) -> None:
    thin = "-" * 96
    print(thin)
    print(f"  {'Model':<{COL_W}}  {'Log-Loss':>9}  {'RMSE':>8}  {'AUC':>7}  {'Mean pred':>9}  Status")
    print(thin)

    sorted_models = sorted(results.items(), key=lambda kv: kv[1]["log_loss"])

    for name, r in sorted_models:
        beat = (
            r["log_loss"] < base["log_loss"],
            r["rmse"]     < base["rmse"],
            r["auc"]      > base["auc"],
        )
        tags = "".join(
            ["LL " if beat[0] else "", "RMSE " if beat[1] else "", "AUC" if beat[2] else ""]
        ).strip()

        if all(beat):
            status = "BEATS BASELINE"
        elif any(beat):
            status = f"partial ({tags})"
        else:
            status = "below baseline"

        print(
            f"  {name:<{COL_W}}  "
            f"{r['log_loss']:>9.4f}  "
            f"{r['rmse']:>8.4f}  "
            f"{r['auc']:>7.4f}  "
            f"{r['mean_pred']:>9.4f}  "
            f"{status}"
        )

    thin2 = "-" * 96
    print(thin2)
    print(
        f"  {'Baseline (always predict mean recall rate)':<{COL_W}}  "
        f"{base['log_loss']:>9.4f}  "
        f"{base['rmse']:>8.4f}  "
        f"{base['auc']:>7.4f}  "
        f"{base['mean_pred']:>9.4f}  reference"
    )
    print("=" * 96)


def print_winner_summary(results: Dict[str, dict], base: dict) -> None:
    best_ll   = min(results.items(), key=lambda kv: kv[1]["log_loss"])
    best_rmse = min(results.items(), key=lambda kv: kv[1]["rmse"])
    best_auc  = max(results.items(), key=lambda kv: kv[1]["auc"])

    print()
    print("  Winner Summary")
    print("  " + "-" * 68)
    print(f"  Best Log-Loss : {best_ll[0].strip()}")
    print(f"                  {best_ll[1]['log_loss']:.4f}  (baseline: {base['log_loss']:.4f})")
    print(f"  Best RMSE     : {best_rmse[0].strip()}")
    print(f"                  {best_rmse[1]['rmse']:.4f}  (baseline: {base['rmse']:.4f})")
    print(f"  Best AUC      : {best_auc[0].strip()}")
    print(f"                  {best_auc[1]['auc']:.4f}  (baseline: 0.5000)")
    print()


def print_calibration_section(model, name: str, events: List[ReviewEvent]) -> None:
    rows = calibration_table(model, events)
    print(f"  Calibration: {name.strip()}")
    print(f"  {'Range':<10}  {'N':>7}  {'Predicted':>9}  {'Actual':>9}  {'Gap':>7}  Bar")
    print("  " + "-" * 70)
    for row in rows:
        gs = "+" if row["gap"] >= 0 else ""
        bar = _bar(row["mean_pred"])
        print(
            f"  {row['range']:<10}  {row['n']:>7}  "
            f"{row['mean_pred']:>9.3f}  {row['actual_recall']:>9.3f}  "
            f"{gs}{row['gap']:>6.3f}  {bar}"
        )
    print()


# ===========================================================================
# Figure generation  (requires matplotlib - skipped gracefully if absent)
# ===========================================================================

def _model_color(name: str) -> str:
    """Return a consistent color per decay family."""
    n = name.lower()
    if "hlr" in n:
        idx = 0 if "smart" in n else (1 if "adaptive" in n else 2)
        return ["#1f77b4", "#4ea8de", "#aec7e8"][idx]
    if "powerlaw" in n or "power" in n:
        idx = 0 if "smart" in n else (1 if "adaptive" in n else 2)
        return ["#2ca02c", "#66c266", "#b5d9b5"][idx]
    # Reinforcement
    idx = 0 if "smart" in n else (1 if "adaptive" in n else 2)
    return ["#d62728", "#e87a7a", "#f5b8b8"][idx]


def _model_linestyle(name: str) -> str:
    if "smart" in name.lower():
        return "-"
    if "adaptive" in name.lower():
        return "--"
    return ":"


def plot_forgetting_curves(figures_dir: Path) -> Path:
    """
    Plot predicted recall probability vs days elapsed (0-30 days)
    for recall_count=3, strength=1.0.
    One line per model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = [i * 0.5 for i in range(61)]  # 0 to 30 in 0.5-day steps
    ref  = time.time()
    recall_count = 3

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, model in MODELS.items():
        ys = []
        for d in days:
            anchor = ref - d * 86400.0
            m = Memory(
                id="__plot__", content="", tags={},
                created_at=anchor, last_accessed_at=anchor,
                strength=1.0, metadata={"recall_count": recall_count},
            )
            raw = model.score(m, now=ref)
            ys.append(min(max(raw, 0.0), 1.0))

        ax.plot(
            days, ys,
            label=name.strip(),
            color=_model_color(name),
            linestyle=_model_linestyle(name),
            linewidth=1.8,
        )

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="-.", label="50% recall")
    ax.set_xlabel("Days since last review", fontsize=12)
    ax.set_ylabel("Predicted recall probability", fontsize=12)
    ax.set_title("Forgetting Curves by Model (recall\_count=3)", fontsize=13)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = figures_dir / "forgetting_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _compute_roc(scores: List[float], labels: List[int]):
    """Return (fpr_list, tpr_list) for plotting."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0, 1], [0, 1]
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    fprs, tprs = [0.0], [0.0]
    tp = fp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)
    return fprs, tprs


def plot_roc_curves(
    model_preds: Dict[str, List[float]],
    labels: List[int],
    figures_dir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.50)")

    for name, preds in model_preds.items():
        fprs, tprs = _compute_roc(preds, labels)
        auc = _roc_auc(preds, labels)
        ax.plot(
            fprs, tprs,
            label=f"{name.strip()} ({auc:.3f})",
            color=_model_color(name),
            linestyle=_model_linestyle(name),
            linewidth=1.8,
        )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Decay Model Comparison", fontsize=13)
    ax.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = figures_dir / "roc_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_calibration(
    model_cal: Dict[str, List[dict]],
    figures_dir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")

    for name, rows in model_cal.items():
        xs = [r["mean_pred"]     for r in rows]
        ys = [r["actual_recall"] for r in rows]
        ax.plot(
            xs, ys,
            marker="o", markersize=4,
            label=name.strip(),
            color=_model_color(name),
            linestyle=_model_linestyle(name),
            linewidth=1.6,
        )

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Actual recall rate", fontsize=12)
    ax.set_title("Calibration (Reliability Diagram)", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = figures_dir / "calibration.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_figures(
    events: List[ReviewEvent],
    results: Dict[str, dict],
    figures_dir: Path,
) -> List[Path]:
    """Generate all three figures. Returns list of saved paths."""
    saved: List[Path] = []

    # 1 - Forgetting curves (no event data needed)
    try:
        p = plot_forgetting_curves(figures_dir)
        saved.append(p)
        print(f"    forgetting_curves.png")
    except Exception as exc:
        print(f"    forgetting_curves.png  SKIPPED ({exc})")

    # 2 - ROC curves (need per-model predictions)
    model_preds: Dict[str, List[float]] = {}
    labels: List[int] = [ev.correct for ev in events]
    for name, model in MODELS.items():
        preds = []
        for ev in events:
            m, now = _event_to_memory(ev)
            preds.append(_clamp(model.score(m, now=now)))
        model_preds[name] = preds

    try:
        p = plot_roc_curves(model_preds, labels, figures_dir)
        saved.append(p)
        print(f"    roc_curve.png")
    except Exception as exc:
        print(f"    roc_curve.png  SKIPPED ({exc})")

    # 3 - Calibration
    model_cal: Dict[str, List[dict]] = {}
    for name, model in MODELS.items():
        model_cal[name] = calibration_table(model, events)

    try:
        p = plot_calibration(model_cal, figures_dir)
        saved.append(p)
        print(f"    calibration.png")
    except Exception as exc:
        print(f"    calibration.png  SKIPPED ({exc})")

    return saved


# ===========================================================================
# Main
# ===========================================================================

def _setup_logging(log_path: Path) -> logging.Logger:
    log = logging.getLogger("benchmark")
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)
    return log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SRS forgetting curve benchmark for Decay Lab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_runner.py                    # synthetic, 50 000 events
  python benchmark_runner.py --n 200000         # synthetic, 200 000 events
  python benchmark_runner.py --csv-path file    # real HLR dataset
  python benchmark_runner.py --calibration      # add calibration tables
  python benchmark_runner.py --no-plots         # skip matplotlib figures
        """
    )
    parser.add_argument(
        "--csv-path", type=str, default=None,
        help=(
            "Path to a Duolingo HLR format CSV/TSV (optionally gzipped). "
            "If omitted, a calibrated synthetic dataset is generated. "
            "Download from: https://dataverse.harvard.edu/dataset.xhtml"
            "?persistentId=doi:10.7910/DVN/N8XJME"
        )
    )
    parser.add_argument(
        "--n", type=int, default=50_000,
        help="Number of synthetic events to generate (default: 50 000)."
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Max CSV rows to read (0 = all). Only used with --csv-path."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic data generation (default: 42)."
    )
    parser.add_argument(
        "--calibration", action="store_true",
        help="Print per-model calibration tables."
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation (useful if matplotlib is not installed)."
    )
    args = parser.parse_args()

    # ---- Logging -----------------------------------------------------------
    log = _setup_logging(LOG_FILE)
    run_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.info("=" * 60)
    log.info("Decay Lab Benchmark  started  %s", run_ts)
    log.info("Args: %s", vars(args))

    print()
    print("=" * 96)
    print("  Decay Lab - SRS Forgetting Curve Benchmark Runner")
    print("=" * 96)
    print(f"  Output dirs:  results/  tables/  figures/  logs/  datasets/")

    # ---- Load or generate events -------------------------------------------
    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"\nERROR: File not found: {csv_path}")
            sys.exit(1)
        max_rows = args.max_rows if args.max_rows > 0 else None
        print(f"\n[1/3] Loading real HLR dataset: {csv_path.name}")
        events = load_hlr_csv(csv_path, max_rows=max_rows)
        source = f"Real HLR data: {csv_path.name} ({len(events):,} events)"
    else:
        print(f"\n[1/3] Generating calibrated synthetic dataset  (n={args.n:,}, seed={args.seed})")
        print("      Based on Settles & Meeder (2016) HLR statistics + Ebbinghaus curve")
        print("      To use the real Duolingo dataset: python benchmark_runner.py --csv-path <file>")
        t0 = time.perf_counter()
        events = generate_synthetic_events(args.n, args.seed)
        print(f"      Generated {len(events):,} events in {time.perf_counter()-t0:.2f}s")
        source = f"Synthetic (n={len(events):,}, seed={args.seed}, HLR oracle formula)"

    if len(events) < 100:
        print(f"\nERROR: Only {len(events)} events. Check your dataset / increase --n.")
        sys.exit(1)

    actual_rate       = sum(e.correct for e in events) / len(events)
    mean_elapsed_days = sum(e.elapsed_seconds for e in events) / len(events) / 86400.0
    max_recalls       = max(e.recall_count for e in events)
    print(f"      Actual recall rate    : {actual_rate:.1%}")
    print(f"      Mean elapsed interval : {mean_elapsed_days:.2f} days")
    print(f"      Max recall_count seen : {max_recalls}")
    log.info("Dataset: %s | n=%d | recall=%.3f | mean_delta_days=%.2f",
             source, len(events), actual_rate, mean_elapsed_days)

    # ---- Score all models --------------------------------------------------
    print(f"\n[2/3] Scoring {len(MODELS)} models on {len(events):,} events...")
    results: Dict[str, dict] = {}
    for name, model in MODELS.items():
        print(f"  {name.strip()}...", end=" ", flush=True)
        t0 = time.perf_counter()
        results[name] = score_model(model, events)
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.2f}s)")
        log.info("  %-46s  ll=%.4f  rmse=%.4f  auc=%.4f",
                 name.strip(),
                 results[name]["log_loss"],
                 results[name]["rmse"],
                 results[name]["auc"])

    base = baseline_stats(events)
    log.info("  Baseline                                            ll=%.4f  rmse=%.4f  auc=%.4f",
             base["log_loss"], base["rmse"], base["auc"])

    # ---- Console display ---------------------------------------------------
    print_header(len(events), actual_recall=actual_rate, source=source)
    print_main_table(results, base)
    print_winner_summary(results, base)

    if args.calibration:
        print("  Calibration Tables")
        print("  gap > 0 = model under-predicts (too pessimistic)")
        print("  gap < 0 = model over-predicts (too optimistic)")
        print()
        for name, model in MODELS.items():
            print_calibration_section(model, name, events)

    # ---- Save JSON ---------------------------------------------------------
    print(f"\n[3/3] Saving results...")
    report = {
        "source":             source,
        "n_events":           len(events),
        "actual_recall_rate": round(actual_rate, 4),
        "mean_elapsed_days":  round(mean_elapsed_days, 2),
        "baseline":           base,
        "models":             results,
        "generated_at":       run_ts,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"    results/benchmark_results.json")
    log.info("Saved JSON: %s", RESULTS_JSON)

    # ---- Save CSV table ----------------------------------------------------
    sorted_models = sorted(results.items(), key=lambda kv: kv[1]["log_loss"])
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "LogLoss", "RMSE", "AUC", "MeanPred", "ActualRecall"])
        for name, r in sorted_models:
            writer.writerow([
                name.strip(),
                f"{r['log_loss']:.4f}",
                f"{r['rmse']:.4f}",
                f"{r['auc']:.4f}",
                f"{r['mean_pred']:.4f}",
                f"{r['actual_recall']:.4f}",
            ])
        # Baseline row
        writer.writerow([
            "Baseline (mean predict)",
            f"{base['log_loss']:.4f}",
            f"{base['rmse']:.4f}",
            f"{base['auc']:.4f}",
            f"{base['mean_pred']:.4f}",
            f"{base['actual_recall']:.4f}",
        ])
    print(f"    results/tables/decay_metrics.csv")
    log.info("Saved CSV: %s", METRICS_CSV)

    # ---- Figures -----------------------------------------------------------
    if args.no_plots:
        print("    figures/  (skipped - --no-plots)")
    else:
        try:
            import matplotlib  # noqa: F401  - just probe availability
            print(f"    Generating figures...")
            saved_figs = generate_figures(events, results, FIGURES_DIR)
            for p in saved_figs:
                log.info("Saved figure: %s", p)
            if not saved_figs:
                print("    figures/  (none generated - check matplotlib)")
        except ImportError:
            print("    figures/  (skipped - install matplotlib: pip install matplotlib)")
            log.warning("matplotlib not available - figures skipped")

    # ---- Summary -----------------------------------------------------------
    log.info("Benchmark complete")
    print()
    print("  Output tree:")
    print(f"    results/")
    print(f"      benchmark_results.json")
    print(f"      tables/")
    print(f"        decay_metrics.csv")
    print(f"      figures/")
    for fig in ["forgetting_curves.png", "roc_curve.png", "calibration.png"]:
        exists = (FIGURES_DIR / fig).exists()
        mark = "ok" if exists else "--"
        print(f"        {fig}  [{mark}]")
    print(f"      logs/")
    print(f"        benchmark.log")
    print(f"    datasets/  (place real datasets here)")
    print()


if __name__ == "__main__":
    main()
