"""
Advanced modeling utilities for deep learning-style predictors, reinforcement learning,
and evaluation helpers without introducing heavyweight dependencies.

The implementations here are intentionally lightweight (NumPy/Pandas based) while
mirroring the behaviors of the requested model families (LSTM, Transformer,
ensembles, bandits, genetic search, backtesting, and explainability helpers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Deep learning style predictors
# ---------------------------------------------------------------------------

def lstm_sequence_predict(
    history: Sequence[Sequence[float]],
    window: int = 5,
    steps: int = 1,
) -> List[List[float]]:
    """
    Lightweight LSTM-like rolling predictor using gated moving averages.
    Returns a list of predicted vectors (one per step).
    """
    if not history:
        return []
    arr = np.asarray(history, dtype=float)
    arr = arr if arr.ndim > 1 else arr.reshape(-1, 1)
    preds: List[List[float]] = []
    weights = np.linspace(1.0, 2.0, num=max(1, window))
    for _ in range(max(1, steps)):
        recent = arr[-window:]
        gating = np.tanh(recent.mean(axis=0))
        context = (recent.T * weights[: len(recent)]).T.sum(axis=0) / weights[: len(recent)].sum()
        pred = context + 0.1 * gating
        preds.append(pred.tolist())
        arr = np.vstack([arr, pred])
    return preds


def transformer_sequence_predict(
    history: Sequence[Sequence[float]],
    steps: int = 1,
    max_context: int = 8,
) -> List[List[float]]:
    """
    Lightweight Transformer-style self-attention over the recent window.
    """
    if not history:
        return []
    arr = np.asarray(history, dtype=float)
    arr = arr if arr.ndim > 1 else arr.reshape(-1, 1)
    preds: List[List[float]] = []
    for _ in range(max(1, steps)):
        context = arr[-max_context:]
        query = context[-1]
        # scaled dot-product attention
        logits = context @ query / (math.sqrt(query.shape[0]) + 1e-8)
        weights = np.exp(logits - logits.max())
        weights = weights / weights.sum()
        attn = (weights[:, None] * context).sum(axis=0)
        preds.append(attn.tolist())
        arr = np.vstack([arr, attn])
    return preds


def neural_ensemble(predictions: Sequence[Sequence[float]], weights: Sequence[float] | None = None) -> List[float]:
    """
    Combines multiple model outputs using normalized weights (default = uniform).
    """
    if not predictions:
        return []
    preds = np.asarray(predictions, dtype=float)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    if weights is None:
        w = np.ones(preds.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.size != preds.shape[0]:
            raise ValueError("weights length must match number of predictions")
    w = w / w.sum()
    combined = (preds.T @ w).astype(float)
    return combined.tolist()


# ---------------------------------------------------------------------------
# Advanced learning: RL, bandits, genetic search
# ---------------------------------------------------------------------------

@dataclass
class ReinforcementLearner:
    n_actions: int
    lr: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1

    def __post_init__(self):
        self.q_values = np.zeros(self.n_actions, dtype=float)

    def select_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_values))

    def update(self, action: int, reward: float) -> float:
        best_next = float(np.max(self.q_values))
        current = self.q_values[action]
        updated = (1 - self.lr) * current + self.lr * (reward + self.gamma * best_next)
        self.q_values[action] = updated
        return updated


@dataclass
class MultiArmedBandit:
    n_arms: int
    epsilon: float = 0.1

    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)

    def select_arm(self) -> int:
        if np.random.rand() < self.epsilon or not self.counts.sum():
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> float:
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm] + (reward - self.values[arm]) / n
        self.values[arm] = value
        return value


def genetic_optimize(
    population: Sequence[Sequence[float]],
    fitness_fn: Callable[[Sequence[float]], float],
    generations: int = 5,
    mutation_rate: float = 0.1,
) -> List[float]:
    """
    Simple genetic algorithm for parameter search.
    """
    if not population:
        return []
    rng = np.random.default_rng(42)
    pop = [np.asarray(ind, dtype=float) for ind in population]
    genome_length = len(pop[0])
    for _ in range(max(1, generations)):
        scores = sorted(((fitness_fn(ind), ind) for ind in pop), key=lambda x: x[0], reverse=True)
        elites = [ind for _, ind in scores[: max(2, len(scores) // 2)]]
        children: List[np.ndarray] = []
        while len(children) < len(pop):
            p1, p2 = rng.choice(elites, 2, replace=True)
            cut = int(rng.integers(1, genome_length))
            child = np.concatenate([p1[:cut], p2[cut:]])
            # mutation
            mask = rng.random(genome_length) < mutation_rate
            child = child + mask * rng.normal(0, 0.05, size=genome_length)
            children.append(child)
        pop = children
    # return best from final population
    scores = [(fitness_fn(ind), ind) for ind in pop]
    best = max(scores, key=lambda x: x[0])[1]
    return best.tolist()


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def cross_validate_sequences(
    dataset: Sequence[Sequence[float]],
    model_fn: Callable[[Sequence[Sequence[float]]], Sequence[Sequence[float]]],
    folds: int = 3,
) -> dict:
    """
    Basic K-fold cross-validation returning MAE across folds.
    """
    data = list(dataset)
    if len(data) < 2 or folds < 2:
        return {"folds": 0, "mae": None}
    fold_size = max(1, len(data) // folds)
    maes: List[float] = []
    for i in range(folds):
        start, end = i * fold_size, (i + 1) * fold_size
        test = data[start:end]
        train = data[:start] + data[end:]
        if not test or not train:
            continue
        preds = model_fn(train)
        target = np.mean(np.asarray(test, dtype=float), axis=0)
        pred = np.asarray(preds[-1], dtype=float)
        maes.append(float(np.mean(np.abs(pred - target))))
    return {"folds": len(maes), "mae": float(np.mean(maes)) if maes else None}


def backtest_strategy(
    history: Sequence[Sequence[int]],
    predictor_fn: Callable[[Sequence[Sequence[int]]], Sequence[Sequence[int]]],
) -> dict:
    """
    Walk-forward backtesting that measures hit rate across steps.
    """
    if len(history) < 2:
        return {"trials": 0, "avg_hit_rate": 0.0}
    hits = 0
    trials = 0
    for i in range(1, len(history)):
        past = history[:i]
        preds = predictor_fn(past)
        pred = preds[-1]
        actual = set(history[i])
        hits += sum(1 for x in pred if x in actual)
        trials += len(pred)
    rate = hits / trials if trials else 0.0
    return {"trials": trials, "avg_hit_rate": rate}


def statistical_significance_test(
    baseline: Tuple[float, int],
    variant: Tuple[float, int],
) -> dict:
    """
    Two-proportion z-test to approximate statistical significance.
    """
    p1, n1 = baseline
    p2, n2 = variant
    if n1 <= 0 or n2 <= 0:
        return {"z_score": 0.0, "p_value": 1.0}
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z_score": 0.0, "p_value": 1.0}
    z = (p2 - p1) / se
    # two-tailed from normal CDF
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"z_score": z, "p_value": p_value}


# ---------------------------------------------------------------------------
# User-facing helpers: confidence, explainability, A/B
# ---------------------------------------------------------------------------

def confidence_scores(raw_scores: Sequence[Sequence[float]] | Sequence[float]) -> List[List[float]]:
    """
    Softmax normalization to turn raw scores into confidence probabilities.
    """
    arr = np.asarray(raw_scores, dtype=float)
    arr = arr if arr.ndim > 1 else arr.reshape(1, -1)
    exp = np.exp(arr - arr.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.tolist()


def feature_importance_from_frequency(
    features: pd.DataFrame | Sequence[Sequence[float]],
    outcomes: Sequence[float],
) -> dict:
    """
    Proxy explainability via correlation magnitude between features and outcomes.
    """
    if isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        df = pd.DataFrame(features)
    df = df.iloc[: len(outcomes)].copy()
    df["outcome"] = list(outcomes)[: len(df)]
    corr = df.corr(numeric_only=True).get("outcome")
    if corr is None:
        return {}
    corr = corr.drop(labels=["outcome"], errors="ignore").fillna(0.0).abs()
    return corr.sort_values(ascending=False).to_dict()


def model_performance_visualization(metrics: Sequence[float]) -> dict:
    """
    Returns a lightweight structure that UI components can plot.
    """
    y = [float(m) for m in metrics]
    return {"x": list(range(len(y))), "y": y}


def ab_test_summary(
    control: Sequence[float],
    variant: Sequence[float],
) -> dict:
    """
    Summarizes A/B performance deltas with simple lift and significance proxy.
    """
    if not control or not variant:
        return {"lift": 0.0, "control_mean": 0.0, "variant_mean": 0.0}
    c_mean = float(np.mean(control))
    v_mean = float(np.mean(variant))
    lift = v_mean - c_mean
    sig = statistical_significance_test((c_mean, len(control)), (v_mean, len(variant)))
    return {
        "control_mean": c_mean,
        "variant_mean": v_mean,
        "lift": lift,
        "p_value": sig["p_value"],
    }

