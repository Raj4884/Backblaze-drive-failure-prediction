"""Metrics: PR-AUC, ROC-AUC, precision/recall at an alert budget, Brier, reliability, and a
bootstrap that resamples whole drives (snapshots of one drive are not independent)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

NAN = float("nan")


def _top_k_counts(y: np.ndarray, s: np.ndarray, frac: float) -> tuple[float, float]:
    """(expected true positives among flagged, number flagged) when flagging the top `frac` of
    snapshots. Ties at the cut-off are split proportionally instead of broken arbitrarily."""
    n = len(y)
    k = frac * n
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    kk = int(math.floor(k))
    if kk >= n:
        return float(ys.sum()), float(n)
    cut = ss[kk] if kk < n else ss[-1]
    above = ss > cut
    tied = ss == cut
    n_above = int(above.sum())
    need = k - n_above
    tied_pos = float(ys[tied].sum())
    n_tied = int(tied.sum())
    tp = (
        float(ys[above].sum()) + (need / n_tied) * tied_pos
        if n_tied and need > 0
        else float(ys[above].sum())
    )
    return tp, float(k)


def precision_at(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    tp, flagged = _top_k_counts(y, s, frac)
    return tp / flagged if flagged else NAN


def recall_at(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    total = float(y.sum())
    if total == 0:
        return NAN
    tp, _ = _top_k_counts(y, s, frac)
    return tp / total


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() > 0 else NAN


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else NAN


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(p, 0, 1)))


def brier_reference(y: np.ndarray) -> float:
    """Brier score of always predicting the base rate."""
    r = float(y.mean())
    return r * (1 - r)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Observed failure rate vs mean predicted probability in quantile bins."""
    df = pd.DataFrame({"y": y, "p": p})
    try:
        df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    except ValueError:
        df["bin"] = 0
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    return out.reset_index(drop=True)


def cluster_bootstrap(
    groups: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 300,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile CI where each resample draws whole drives (groups) with replacement.
    `stat` receives an array of row indices."""
    codes, uniques = pd.factorize(groups)
    members = [np.flatnonzero(codes == i) for i in range(len(uniques))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        v = stat(idx)
        if not math.isnan(v):
            vals.append(v)
    if len(vals) < max(10, n_boot // 5):
        return NAN, NAN
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_scores(
    y: np.ndarray,
    s: np.ndarray,
    groups: np.ndarray,
    budget: float = 0.01,
    n_boot: int = 300,
    seed: int = 0,
) -> dict:
    out = {
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
        f"recall@{budget:.0%}": recall_at(y, s, budget),
        f"precision@{budget:.0%}": precision_at(y, s, budget),
    }
    out["pr_auc_ci"] = cluster_bootstrap(groups, lambda i: pr_auc(y[i], s[i]), n_boot, seed)
    out["recall_ci"] = cluster_bootstrap(
        groups, lambda i: recall_at(y[i], s[i], budget), n_boot, seed
    )
    return out
