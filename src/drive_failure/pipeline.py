"""Train, calibrate and evaluate on a chronological split; build the report and model bundle."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from drive_failure.evaluate import (
    brier,
    brier_reference,
    evaluate_scores,
    reliability,
)
from drive_failure.features import time_split
from drive_failure.models import (
    fit_boosted,
    fit_calibrator,
    make_logistic,
    proba,
    rule_score,
)
from drive_failure.schema import FEATURES

MIN_TEST_POSITIVES = 30


def split_summary(parts: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for name, df in parts.items():
        rows.append(
            {
                "split": name,
                "from": df["date"].min().date().isoformat(),
                "to": df["date"].max().date().isoformat(),
                "snapshots": len(df),
                "positives": int(df["label"].sum()),
                "failing_drives": int(df.loc[df["label"] == 1, "serial_number"].nunique()),
                "base_rate": float(df["label"].mean()),
            }
        )
    return rows


def run(
    snaps: pd.DataFrame, horizon: int = 30, budget: float = 0.01, seed: int = 0, n_boot: int = 300
) -> tuple[dict, dict]:
    train, val, test = time_split(snaps, horizon)
    parts = {"train": train, "validation": val, "test": test}

    logit = make_logistic().fit(train[FEATURES], train["label"])
    boosted, boosted_name = fit_boosted(train, val, seed)
    raw_val = proba(boosted, val)
    calib = fit_calibrator(raw_val, val["label"].to_numpy())

    y = test["label"].to_numpy()
    groups = test["serial_number"].to_numpy()
    scores = {
        "SMART warning rule": rule_score(test),
        "Logistic regression": proba(logit, test),
        f"Boosted trees ({boosted_name})": proba(boosted, test),
    }
    results = {
        name: evaluate_scores(y, s, groups, budget, n_boot, seed) for name, s in scores.items()
    }
    boosted_key = f"Boosted trees ({boosted_name})"
    raw_test = scores[boosted_key]
    cal_test = calib.predict(raw_test)
    calibration = {
        "brier_raw": brier(y, np.clip(raw_test, 0, 1)),
        "brier_calibrated": brier(y, cal_test),
        "brier_base_rate": brier_reference(y),
        "reliability": reliability(y, cal_test),
    }
    importances = None
    if hasattr(boosted, "feature_importances_"):
        imp = pd.Series(boosted.feature_importances_, index=FEATURES).sort_values(ascending=False)
        importances = imp.head(10)

    alert_threshold = float(np.quantile(calib.predict(raw_val), 1 - budget))
    report = {
        "horizon": horizon,
        "budget": budget,
        "splits": split_summary(parts),
        "results": results,
        "calibration": calibration,
        "importances": importances,
        "boosted_name": boosted_name,
        "warnings": [],
    }
    if int(y.sum()) < MIN_TEST_POSITIVES:
        report["warnings"].append(
            f"Only {int(y.sum())} positive test snapshots (< {MIN_TEST_POSITIVES}); intervals will "
            "be wide and rankings between models unreliable."
        )
    bundle = {
        "model": boosted,
        "calibrator": calib,
        "features": FEATURES,
        "horizon": horizon,
        "model_name": boosted_name,
        "alert_threshold": alert_threshold,
        "alert_budget": budget,
    }
    return report, bundle


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `drive-failure train` first.")
    return joblib.load(path)


def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _ci(v, ci):
    if math.isnan(v):
        return "n/a"
    if math.isnan(ci[0]):
        return f"{v:.3f}"
    return f"{v:.3f} ({ci[0]:.3f} to {ci[1]:.3f})"


def format_report(report: dict, data_note: str = "") -> str:
    b = report["budget"]
    lines = ["# Drive failure prediction report", ""]
    if data_note:
        lines += [data_note, ""]
    lines += [
        f"Task: predict whether a drive fails within {report['horizon']} days of a snapshot. "
        "Evaluation is chronological with a gap equal to the horizon between train, validation "
        "and test. Intervals are 95% bootstrap intervals that resample whole drives.",
        "",
    ]
    for w in report["warnings"]:
        lines += [f"> Warning: {w}", ""]
    lines += [
        "## Splits",
        "",
        "| Split | From | To | Snapshots | Positives | Failing drives | Base rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["splits"]:
        lines.append(
            f"| {r['split']} | {r['from']} | {r['to']} | {r['snapshots']} | {r['positives']} "
            f"| {r['failing_drives']} | {r['base_rate']:.4f} |"
        )
    lines += [
        "",
        "## Test results",
        "",
        f"Alert budget: flag the top {b:.0%} of snapshots by score.",
        "",
        f"| Model | PR-AUC (95% CI) | ROC-AUC | Recall@{b:.0%} (95% CI) | Precision@{b:.0%} |",
        "|---|---|---|---|---|",
    ]
    for name, r in report["results"].items():
        lines.append(
            f"| {name} | {_ci(r['pr_auc'], r['pr_auc_ci'])} | {_f(r['roc_auc'])} "
            f"| {_ci(r[f'recall@{b:.0%}'], r['recall_ci'])} | {_f(r[f'precision@{b:.0%}'])} |"
        )
    c = report["calibration"]
    lines += [
        "",
        "## Calibration (boosted model, test period)",
        "",
        f"Brier score: raw {_f(c['brier_raw'], 5)}, after isotonic calibration "
        f"{_f(c['brier_calibrated'], 5)}, always-predict-base-rate {_f(c['brier_base_rate'], 5)}.",
        "",
        "| Bin | Snapshots | Mean predicted | Observed rate |",
        "|---|---|---|---|",
    ]
    for i, r in c["reliability"].iterrows():
        lines.append(f"| {i + 1} | {int(r['n'])} | {r['mean_pred']:.5f} | {r['observed']:.5f} |")
    if report["importances"] is not None:
        lines += ["", "## Top features (model importance, not causal)", ""]
        for k, v in report["importances"].items():
            lines.append(f"- {k}: {v:.3f}")
    lines += [
        "",
        "## Caveats",
        "",
        "- Failures are rare; read PR-AUC against the base rate, not against 0.5.",
        "- Many real failures show no SMART warning, so recall is capped well below 100%.",
        "- The same drives can appear in train and test (different dates). The gap prevents label "
        "leakage but not drive-level similarity.",
        "- Drives that leave the fleet without failing are treated as healthy.",
    ]
    return "\n".join(lines) + "\n"
