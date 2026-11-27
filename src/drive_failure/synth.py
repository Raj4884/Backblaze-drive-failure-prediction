"""Synthetic Backblaze-shaped data for tests and the demo. NOT real: the signal strength and
failure rate are chosen by this generator, so metrics on it say nothing about real drives."""

from __future__ import annotations

import numpy as np
import pandas as pd

MODELS = [
    ("ST4000DM000", 4e12),
    ("HGST HMS5C4040BLE640", 4e12),
    ("WDC WD40EFRX", 4e12),
    ("TOSHIBA MG07ACA14TA", 14e12),
]


def make_daily(
    n_drives: int = 600,
    days: int = 400,
    fail_frac: float = 0.12,
    signal_frac: float = 0.6,
    seed: int = 0,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """Daily records. A fraction `fail_frac` of drives fail at a random day; `signal_frac` of
    those show rising SMART 5/187/197/198 in the 45 days before failure (the rest fail silently,
    as many real drives do). A few healthy drives show harmless small non-zero values."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days)
    rows = []
    for k in range(n_drives):
        model, cap = MODELS[rng.integers(len(MODELS))]
        serial = f"SYN{k:05d}"
        first = int(rng.integers(0, days // 3))
        fails = rng.random() < fail_frac
        last = int(rng.integers(first + 60, days)) if fails else days - 1
        if last <= first:
            last = min(days - 1, first + 60)
        has_signal = fails and rng.random() < signal_frac
        noisy = (not fails) and rng.random() < 0.05
        poh0 = float(rng.integers(1000, 30000))
        acc = {5: 0.0, 187: 0.0, 197: 0.0, 198: 0.0}  # SMART error counters only ever grow
        rate = {5: 3.0, 187: 0.6, 197: 2.0, 198: 1.2}
        for d in range(first, last + 1):
            ttf = last - d
            ramp = max(0.0, (45 - ttf) / 45) if has_signal else 0.0
            for i in acc:
                acc[i] += float(rng.poisson(rate[i] * ramp * ramp))
            if noisy and d == first + 100:
                acc[5] += 1.0
            rows.append(
                {
                    "date": dates[d],
                    "serial_number": serial,
                    "model": model,
                    "capacity_bytes": cap,
                    "failure": int(fails and d == last),
                    "smart_5_raw": acc[5],
                    "smart_9_raw": poh0 + 24 * (d - first),
                    "smart_187_raw": acc[187],
                    "smart_188_raw": float(rng.poisson(0.2 + 3 * ramp)),
                    "smart_194_raw": float(rng.normal(32 + 6 * ramp, 3)),
                    "smart_197_raw": acc[197],
                    "smart_198_raw": acc[198],
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["serial_number", "date"]).reset_index(drop=True)
