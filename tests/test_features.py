import pandas as pd
import pytest

from drive_failure.features import build_snapshots, manufacturer, time_split
from tests.conftest import daily_for

START = "2024-01-01"


def frame(*row_lists):
    return pd.DataFrame([r for rows in row_lists for r in rows])


def test_labels_follow_the_failure_window():
    fail_on = 100
    daily = frame(
        daily_for("A", START, 200, fail_on=fail_on),
        daily_for("B", START, 200),  # healthy, keeps data_end late
    )
    snaps = build_snapshots(daily, horizon=30, step=5)
    a = snaps[snaps["serial_number"] == "A"]
    fail_date = pd.Timestamp(START) + pd.Timedelta(days=fail_on)
    assert (a["date"] < fail_date).all()  # no snapshot on or after the failure day
    pos = a[a["label"] == 1]["date"]
    assert not pos.empty
    assert ((fail_date - pos).dt.days <= 30).all() and ((fail_date - pos).dt.days > 0).all()
    early = a[a["date"] < fail_date - pd.Timedelta(days=30)]
    assert (early["label"] == 0).all()


def test_snapshots_near_end_of_data_are_dropped():
    daily = frame(daily_for("A", START, 120))
    snaps = build_snapshots(daily, horizon=30, step=5)
    assert snaps["date"].max() <= pd.Timestamp(START) + pd.Timedelta(days=119 - 30)


def test_delta_uses_only_past_readings():
    daily = frame(daily_for("A", START, 150, smart5=lambda d: float(d)))
    snaps = build_snapshots(daily, horizon=30, step=5)
    ok = snaps.dropna(subset=["d30_smart_5_raw"])
    assert len(ok) > 0
    assert (ok["d30_smart_5_raw"] == 30).all()  # value(t) - value(t - 30 days)
    # A reading in the future must not leak into the current row.
    first = snaps.iloc[0]
    assert first["smart_5_raw"] == (first["date"] - pd.Timestamp(START)).days


def test_requires_columns_and_valid_step():
    with pytest.raises(ValueError, match="missing columns"):
        build_snapshots(pd.DataFrame({"date": []}))
    daily = frame(daily_for("A", START, 100))
    with pytest.raises(ValueError, match="divide 30"):
        build_snapshots(daily, step=7)


def test_too_short_data_raises():
    with pytest.raises(ValueError, match="shorter than the horizon"):
        build_snapshots(frame(daily_for("A", START, 20)), horizon=30)


def test_time_split_has_gaps_and_order(synth_snaps):
    tr, va, te = time_split(synth_snaps, horizon=30)
    assert tr["date"].max() + pd.Timedelta(days=30) < va["date"].min()
    assert va["date"].max() + pd.Timedelta(days=30) < te["date"].min()


@pytest.mark.parametrize(
    "model,expected",
    [
        ("ST4000DM000", "seagate"),
        ("HGST HMS5C4040BLE640", "hgst"),
        ("WDC WD40EFRX", "wd"),
        ("TOSHIBA MG07ACA14TA", "toshiba"),
        ("Samsung SSD", "other"),
    ],
)
def test_manufacturer(model, expected):
    assert manufacturer(model) == expected
