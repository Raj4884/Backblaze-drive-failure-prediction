import math

import numpy as np
import pandas as pd

from drive_failure.evaluate import (
    brier_reference,
    cluster_bootstrap,
    precision_at,
    recall_at,
    reliability,
)
from drive_failure.models import rule_score
from drive_failure.pipeline import format_report, run


def test_recall_precision_at_budget():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert recall_at(y, s, 0.2) == 0.5
    assert precision_at(y, s, 0.2) == 0.5
    assert recall_at(y, s, 0.3) == 1.0


def test_ties_are_split_not_broken_arbitrarily():
    y = np.array([1, 0, 0, 0])
    same = np.zeros(4)
    # Flagging 50% of a fully tied set should catch half the positives on average.
    assert recall_at(y, same, 0.5) == 0.5
    assert recall_at(y[::-1].copy(), same, 0.5) == 0.5


def test_recall_nan_without_positives():
    assert math.isnan(recall_at(np.zeros(5), np.arange(5.0), 0.2))


def test_cluster_bootstrap_is_deterministic_and_brackets_value():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 4)
    y = rng.integers(0, 2, 200)
    stat = lambda i: float(y[i].mean())  # noqa: E731
    a = cluster_bootstrap(groups, stat, 200, seed=3)
    b = cluster_bootstrap(groups, stat, 200, seed=3)
    assert a == b and a[0] <= y.mean() <= a[1]


def test_reliability_and_brier_reference():
    y = np.array([0, 0, 0, 1] * 25)
    p = np.linspace(0, 1, 100)
    r = reliability(y, p, bins=5)
    assert len(r) == 5 and r["n"].sum() == 100
    assert abs(brier_reference(y) - 0.1875) < 1e-9


def test_rule_score_counts_warning_counters():
    X = pd.DataFrame(
        {
            "smart_5_raw": [0, 3, 1],
            "smart_187_raw": [0, 0, 2],
            "smart_197_raw": [None, 0, 1],
            "smart_198_raw": [0, 0, 0],
        }
    )
    assert rule_score(X).tolist() == [0.0, 1.0, 3.0]


def test_pipeline_runs_and_report_has_sections(synth_snaps):
    report, bundle = run(synth_snaps, n_boot=30)
    text = format_report(report, "> SYNTHETIC")
    for h in ("## Splits", "## Test results", "## Calibration", "## Caveats", "SYNTHETIC"):
        assert h in text
    assert set(bundle) >= {"model", "calibrator", "features", "alert_threshold"}
    # On this synthetic data a learned model must beat chance by a clear margin.
    boosted = next(v for k, v in report["results"].items() if k.startswith("Boosted"))
    assert boosted["roc_auc"] > 0.6


def test_few_positives_triggers_warning(synth_snaps):
    report, _ = run(synth_snaps, n_boot=20)
    if report["splits"][2]["positives"] < 30:
        assert report["warnings"]
