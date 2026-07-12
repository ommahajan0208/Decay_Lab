from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Tuple

# Feature index constants for the 5-element context vector
# [0] days_since_last_access
# [1] recall_count (log-scaled)
# [2] current_raw_strength
# [3] memory_age_days
# [4] query_length (token count, log-scaled)

N_ARMS = 3
N_FEATURES = 5
ARM_NAMES = ["HLR", "Power-Law", "Reinforcement"]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mat_vec(M: List[List[float]], v: List[float]) -> List[float]:
    """Multiply (n x n) matrix M by vector v, return length-n vector."""
    n = len(v)
    return [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]


def _identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _inverse_3x3(M: List[List[float]]) -> List[List[float]]:
    """Exact inverse for a 3x3 matrix via cofactors. Used only when N_FEATURES == 3."""
    raise NotImplementedError("Use _inverse_nxn instead.")


def _inverse_nxn(M: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan inversion for an (n x n) matrix.
    Returns the inverse. Raises ValueError if singular.
    """
    n = len(M)
    # Augmented matrix [M | I]
    aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        # Find pivot
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12:
            raise ValueError("Matrix is singular; cannot invert.")
        scale = aug[col][col]
        aug[col] = [x / scale for x in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]


class LinUCBBandit:
    """
    LinUCB-Disjoint contextual bandit with N_ARMS arms and N_FEATURES context features.

    Each arm a maintains:
        A_a  -- (N_FEATURES x N_FEATURES) regularization matrix, initialized to I
        b_a  -- (N_FEATURES,) reward accumulation vector, initialized to 0

    On select(context):
        theta_a = A_a^-1 @ b_a
        ucb_a   = theta_a^T x + alpha * sqrt(x^T A_a^-1 x)
        chosen  = argmax(ucb_a)

    On update(reward):
        A[chosen] += x @ x^T
        b[chosen] += reward * x

    Context vector (5 features per query-memory pair):
        [0] days_since_last_access  -- float, 0 if never accessed
        [1] log1p(recall_count)     -- float
        [2] raw_strength            -- float in [0, 1+]
        [3] memory_age_days         -- float
        [4] log1p(query_length)     -- float (token count proxy)
    """

    def __init__(self, alpha: float = 1.0, persist_path: Optional[str] = None):
        self.alpha = alpha
        self.persist_path = persist_path

        # Per-arm parameters
        self.A: List[List[List[float]]] = [_identity(N_FEATURES) for _ in range(N_ARMS)]
        self.b: List[List[float]] = [[0.0] * N_FEATURES for _ in range(N_ARMS)]

        # State from the most recent select() call - needed by update()
        self.last_context: Optional[List[float]] = None
        self.last_arm: Optional[int] = None

        # Human-readable label for the last selected arm
        self.last_arm_name: Optional[str] = None

        # Running history of (arm_name, reward) pairs for the dashboard
        self.history: List[dict] = []

        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    # ---- Public API ---------------------------------------------------------

    def select(self, context: List[float]) -> int:
        """
        Given a context vector, return the arm index with the highest UCB score.
        Stores context and chosen arm for the subsequent update() call.
        """
        ucb_scores = []
        for a in range(N_ARMS):
            A_inv = _inverse_nxn(self.A[a])
            theta = _mat_vec(A_inv, self.b[a])
            A_inv_x = _mat_vec(A_inv, context)
            mean = _dot(theta, context)
            variance = _dot(context, A_inv_x)
            bonus = self.alpha * math.sqrt(max(0.0, variance))
            ucb_scores.append(mean + bonus)

        chosen = ucb_scores.index(max(ucb_scores))
        self.last_context = list(context)
        self.last_arm = chosen
        self.last_arm_name = ARM_NAMES[chosen]
        return chosen

    def update(self, reward: float) -> None:
        """
        Update the matrices for the last selected arm using the observed reward.
        Must be called after select().
        """
        if self.last_context is None or self.last_arm is None:
            return

        a = self.last_arm
        x = self.last_context
        n = N_FEATURES

        # A[a] += x @ x^T
        for i in range(n):
            for j in range(n):
                self.A[a][i][j] += x[i] * x[j]

        # b[a] += reward * x
        for i in range(n):
            self.b[a][i] += reward * x[i]

        self.history.append({
            "arm": ARM_NAMES[a],
            "reward": round(reward, 3),
            "context": [round(v, 4) for v in x],
        })

        if self.persist_path:
            self._save(self.persist_path)

    def get_theta(self, arm: int) -> List[float]:
        """Return the learned weight vector theta for a given arm."""
        A_inv = _inverse_nxn(self.A[arm])
        return _mat_vec(A_inv, self.b[arm])

    def get_ucb_scores(self, context: List[float]) -> List[float]:
        """Return UCB scores for all arms without storing state."""
        scores = []
        for a in range(N_ARMS):
            A_inv = _inverse_nxn(self.A[a])
            theta = _mat_vec(A_inv, self.b[a])
            A_inv_x = _mat_vec(A_inv, context)
            mean = _dot(theta, context)
            variance = _dot(context, A_inv_x)
            bonus = self.alpha * math.sqrt(max(0.0, variance))
            scores.append(round(mean + bonus, 4))
        return scores

    # ---- Persistence --------------------------------------------------------

    def _save(self, path: str) -> None:
        data = {
            "alpha": self.alpha,
            "A": self.A,
            "b": self.b,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.alpha = float(data.get("alpha", self.alpha))
        if "A" in data:
            self.A = data["A"]
        if "b" in data:
            self.b = data["b"]


# ---- Context vector builder -------------------------------------------------

def build_context(
    memory_strength: float,
    last_accessed_at: Optional[float],
    recall_count: int,
    created_at: float,
    query: str,
    now: float,
) -> List[float]:
    """
    Build a normalized 5-element context vector from memory state and the query.

    Features:
        [0] days_since_last_access  -- capped at 30
        [1] log1p(recall_count)
        [2] raw strength            -- capped at 2.0
        [3] memory_age_days         -- capped at 365
        [4] log1p(query_token_count)
    """
    seconds_per_day = 86400.0

    anchor = last_accessed_at if last_accessed_at is not None else created_at
    days_since_access = max(0.0, (now - anchor) / seconds_per_day)
    days_since_access = min(days_since_access, 30.0) / 30.0  # normalize [0, 1]

    log_recall = math.log1p(max(0, recall_count)) / math.log1p(50)  # normalize against cap of 50

    strength = min(max(0.0, memory_strength), 2.0) / 2.0  # normalize [0, 1]

    memory_age_days = max(0.0, (now - created_at) / seconds_per_day)
    memory_age_days = min(memory_age_days, 365.0) / 365.0  # normalize [0, 1]

    token_count = len(query.split()) if query else 0
    log_query_len = math.log1p(token_count) / math.log1p(100)  # normalize against cap of 100 tokens

    return [days_since_access, log_recall, strength, memory_age_days, log_query_len]


FEATURE_NAMES = [
    "days_since_access (norm)",
    "log_recall_count (norm)",
    "raw_strength (norm)",
    "memory_age_days (norm)",
    "log_query_length (norm)",
]
