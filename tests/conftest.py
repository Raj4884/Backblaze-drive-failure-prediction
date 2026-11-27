import pandas as pd
import pytest

from drive_failure.features import build_snapshots
from drive_failure.synth import make_daily


def daily_for(serial, start, days, fail_on=None, smart5=None, model="ST4000DM000"):
    """Hand-built daily rows for one drive. fail_on = day offset of the failure row."""
    rows = []
    end = days if fail_on is None else fail_on + 1
    for d in range(end):
        rows.append(
            {
                "date": pd.Timestamp(start) + pd.Timedelta(days=d),
                "serial_number": serial,
                "model": model,
                "capacity_bytes": 4e12,
                "failure": int(fail_on is not None and d == fail_on),
                "smart_5_raw": smart5(d) if smart5 else 0.0,
            }
        )
    return rows


@pytest.fixture(scope="session")
def synth_daily():
    return make_daily(n_drives=400, days=420, seed=1)


@pytest.fixture(scope="session")
def synth_snaps(synth_daily):
    return build_snapshots(synth_daily)
