"""Turn daily SMART records into labelled snapshots.

A snapshot is one drive on one grid date t. Features use only data up to t. The label is 1 if the
drive fails in the window (t, t + horizon]. Snapshots whose window runs past the end of the data
are dropped, because their label is unknown. Drives that leave the fleet without failing count as
negatives (standard for this dataset; it slightly understates risk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from drive_failure.schema import (
    DELTA_COLS,
    FEATURES,
    LAG_DAYS,
    MFRS,
    RAW_COLS,
    REQUIRED_DAILY,
)


def manufacturer(model: str) -> str:
    m = str(model).upper()
    if m.startswith("ST"):
        return "seagate"
    if m.startswith(("HGST", "HUH", "HMS", "HDS", "HUS")):
        return "hgst"
    if m.startswith(("WDC", "WD")):
        return "wd"
    if m.startswith(("TOSHIBA", "MG", "MD", "MQ")):
        return "toshiba"
    return "other"


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["capacity_tb"] = df["capacity_bytes"] / 1e12
    mfr = df["model"].map(manufacturer)
    for m in MFRS:
        df[f"mfr_{m}"] = (mfr == m).astype(float)
    return df


def build_snapshots(daily: pd.DataFrame, horizon: int = 30, step: int = 5) -> pd.DataFrame:
    """Labelled snapshots on the absolute grid ordinal(date) % step == 0."""
    missing = [c for c in REQUIRED_DAILY if c not in daily.columns]
    if missing:
        raise ValueError(f"daily table is missing columns {missing}")
    if 30 % step:
        raise ValueError("step must divide 30")
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = np.nan
    data_end = df["date"].max()

    fail_date = df.loc[df["failure"] == 1].groupby("serial_number")["date"].min()
    df["fail_date"] = fail_date.reindex(df["serial_number"]).to_numpy()

    on_grid = df["date"].map(pd.Timestamp.toordinal) % step == 0
    alive = df["fail_date"].isna() | (df["date"] < df["fail_date"])
    snaps = df[on_grid & alive].copy()
    snaps = snaps[snaps["date"] <= data_end - pd.Timedelta(days=horizon)]
    if snaps.empty:
        raise ValueError("no usable snapshots: data span is shorter than the horizon")

    # 30-day deltas: value now minus the latest reading at or before t - 30 days (<= 7 days stale).
    lag_src = df[["serial_number", "date"] + RAW_COLS].rename(
        columns={c: f"prev_{c}" for c in RAW_COLS}
    )
    lag_src = lag_src.sort_values("date")
    snaps = snaps.assign(lag_date=snaps["date"] - pd.Timedelta(days=LAG_DAYS)).sort_values(
        "lag_date"
    )
    merged = pd.merge_asof(
        snaps,
        lag_src.rename(columns={"date": "lag_date"}),
        on="lag_date",
        by="serial_number",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    for col in DELTA_COLS:
        base = col.replace("d30_", "")
        merged[col] = merged[base] - merged[f"prev_{base}"]

    merged = add_static_features(merged)
    window_end = merged["date"] + pd.Timedelta(days=horizon)
    merged["label"] = (merged["fail_date"] > merged["date"]) & (merged["fail_date"] <= window_end)
    merged["label"] = merged["label"].astype(int)
    keep = ["serial_number", "date", "model", "label"] + FEATURES
    return merged[keep].sort_values(["date", "serial_number"]).reset_index(drop=True)


def time_split(
    snaps: pd.DataFrame, horizon: int = 30, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split with a `horizon`-day gap between parts.

    A snapshot's label looks `horizon` days ahead, so without the gap a training label could be
    decided by an event that also falls inside the validation or test period. Drives may appear in
    more than one part (the same fleet is observed over time); the gap, not drive identity, is what
    prevents label leakage, and the README states this limit.
    """
    dates = np.sort(snaps["date"].unique())
    c1 = dates[int(len(dates) * train_frac) - 1]
    c2 = dates[int(len(dates) * (train_frac + val_frac)) - 1]
    gap = np.timedelta64(horizon, "D")
    train = snaps[snaps["date"] <= c1]
    val = snaps[(snaps["date"] > c1 + gap) & (snaps["date"] <= c2)]
    test = snaps[snaps["date"] > c2 + gap]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("data span too short for a train/validation/test split with gaps")
    return train, val, test
