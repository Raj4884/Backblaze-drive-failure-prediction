"""Models: SMART warning rule, logistic regression, gradient boosting (XGBoost if installed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from drive_failure.schema import FEATURES, WARNING_IDS


def rule_score(X: pd.DataFrame) -> np.ndarray:
    """Baseline: number of warning SMART counters (5, 187, 197, 198) that are above zero."""
    cols = [f"smart_{i}_raw" for i in WARNING_IDS]
    return (X[cols].fillna(0) > 0).sum(axis=1).to_numpy(dtype=float)


def _signed_log1p(a):
    return np.sign(a) * np.log1p(np.abs(a))


def make_logistic():
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        FunctionTransformer(_signed_log1p),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def make_boosted(scale_pos_weight: float = 1.0, seed: int = 0):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=2,
        ), "xgboost"
    except ImportError:
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=seed
        ), "sklearn-hist-gb"


def fit_boosted(train: pd.DataFrame, val: pd.DataFrame, seed: int = 0):
    pos = max(int(train["label"].sum()), 1)
    neg = max(len(train) - pos, 1)
    model, name = make_boosted(neg / pos, seed)
    X, y = train[FEATURES], train["label"]
    if name == "xgboost":
        model.set_params(early_stopping_rounds=30)
        model.fit(X, y, eval_set=[(val[FEATURES], val["label"])], verbose=False)
    else:
        model.fit(X, y)
    return model, name


def proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURES])[:, 1]


def fit_calibrator(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Isotonic map from raw model scores to probabilities, fitted on the validation period."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, y)
    return iso
