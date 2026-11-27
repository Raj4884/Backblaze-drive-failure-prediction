"""Loading Backblaze Drive Stats daily files.

Backblaze publishes one CSV per day (zipped per quarter) at
https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data. Columns include
date, serial_number, model, capacity_bytes, failure and smart_<id>_raw / smart_<id>_normalized.
Failure is 1 on a drive's last day. Backblaze asks users to cite it as the source and does not
allow selling the data itself.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from drive_failure.schema import RAW_COLS, REQUIRED_DAILY, SMART_IDS

KEEP = set(REQUIRED_DAILY) | set(RAW_COLS)


def _read_one(handle, step: int, sample_frac: float) -> pd.DataFrame:
    df = pd.read_csv(handle, usecols=lambda c: c in KEEP, parse_dates=["date"])
    missing = [c for c in REQUIRED_DAILY if c not in df.columns]
    if missing:
        raise ValueError(f"file is missing columns {missing}; Backblaze schema may have changed")
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    if sample_frac < 1.0:
        h = pd.util.hash_array(df["serial_number"].astype(str).to_numpy(dtype=object))
        df = df[(h % 10_000) < int(sample_frac * 10_000)]
    on_grid = df["date"].map(pd.Timestamp.toordinal) % step == 0
    # Failure rows are kept even when off-grid so the failure date is always known.
    return df[on_grid | (df["failure"] == 1)]


def iter_csv_sources(paths: Iterable[Path]):
    for path in sorted(paths):
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith(".csv") and not name.startswith("__MACOSX"):
                        yield name, io.BytesIO(zf.read(name))
        elif path.suffix == ".csv":
            yield path.name, path
        elif path.is_dir():
            yield from iter_csv_sources(list(path.rglob("*.csv")) + list(path.rglob("*.zip")))


def load_backblaze(paths: Iterable[Path], step: int = 5, sample_frac: float = 1.0) -> pd.DataFrame:
    """Read daily files, keep needed columns, keep every `step`-th day and optionally a
    deterministic fraction of drives (hashed on serial number)."""
    if 30 % step:
        raise ValueError("step must divide 30 so the 30-day lag lands on the same grid")
    frames = [_read_one(src, step, sample_frac) for _, src in iter_csv_sources(paths)]
    if not frames:
        raise FileNotFoundError("no .csv or .zip files found; see README for the download link")
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.drop_duplicates(["serial_number", "date"])
    return daily.sort_values(["serial_number", "date"]).reset_index(drop=True)


def smart_cols_present(df: pd.DataFrame) -> list[str]:
    return [f"smart_{i}_raw" for i in SMART_IDS if f"smart_{i}_raw" in df.columns]
