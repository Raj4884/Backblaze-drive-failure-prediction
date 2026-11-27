import zipfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from drive_failure.api import create_app
from drive_failure.data import load_backblaze
from drive_failure.pipeline import run, save_bundle


def write_day(path, date, rows):
    cols = [
        "date",
        "serial_number",
        "model",
        "capacity_bytes",
        "failure",
        "smart_5_raw",
        "smart_9_raw",
        "extra_col",
    ]
    pd.DataFrame(
        [
            {
                "date": date,
                "serial_number": s,
                "model": "ST4000DM000",
                "capacity_bytes": 4e12,
                "failure": f,
                "smart_5_raw": 1,
                "smart_9_raw": 10,
                "extra_col": 1,
            }
            for s, f in rows
        ]
    )[cols].to_csv(path, index=False)


def test_loader_keeps_grid_days_and_failure_rows(tmp_path):
    # ordinal % 5 == 0 days are on the grid; find one and an off-grid neighbour.
    base = pd.Timestamp("2024-03-01")
    on = next(
        base + pd.Timedelta(days=i)
        for i in range(5)
        if (base + pd.Timedelta(days=i)).toordinal() % 5 == 0
    )
    off = on + pd.Timedelta(days=1)
    write_day(tmp_path / "a.csv", on.date(), [("S1", 0)])
    write_day(tmp_path / "b.csv", off.date(), [("S1", 0), ("S2", 1)])
    df = load_backblaze([tmp_path], step=5)
    assert set(df["date"]) == {on, off}
    assert (df[df["date"] == off]["serial_number"] == "S2").all()  # only the failure row survives
    assert "extra_col" not in df.columns and "smart_187_raw" in df.columns


def test_loader_reads_zip_and_validates(tmp_path):
    csv = tmp_path / "d.csv"
    write_day(csv, "2024-03-05", [("S1", 0)])
    z = tmp_path / "q.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(csv, "2024-03-05.csv")
    csv.unlink()
    assert len(load_backblaze([z], step=5)) in (0, 1)
    with pytest.raises(ValueError, match="divide 30"):
        load_backblaze([z], step=7)
    with pytest.raises(FileNotFoundError):
        load_backblaze([tmp_path / "empty"], step=5)


def test_loader_rejects_changed_schema(tmp_path):
    pd.DataFrame({"date": ["2024-01-01"], "serial": ["x"]}).to_csv(
        tmp_path / "bad.csv", index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        load_backblaze([tmp_path / "bad.csv"], step=5)


def test_sample_frac_is_deterministic(tmp_path):
    d = pd.Timestamp("2024-03-01")
    d = next(
        d + pd.Timedelta(days=i)
        for i in range(5)
        if (d + pd.Timedelta(days=i)).toordinal() % 5 == 0
    )
    write_day(tmp_path / "a.csv", d.date(), [(f"S{i}", 0) for i in range(300)])
    a = load_backblaze([tmp_path], step=5, sample_frac=0.3)
    b = load_backblaze([tmp_path], step=5, sample_frac=0.3)
    assert a.equals(b) and 40 < len(a) < 140


def test_api_predict_and_missing_model(tmp_path, synth_snaps):
    client = TestClient(create_app(str(tmp_path / "none.joblib")))
    assert client.post("/predict", json={"drives": [{}]}).status_code == 503
    _, bundle = run(synth_snaps, n_boot=10)
    path = tmp_path / "m.joblib"
    save_bundle(bundle, path)
    client = TestClient(create_app(str(path)))
    assert client.get("/health").json()["status"] == "ok"
    healthy = {"serial_number": "H", "readings": {"smart_5_raw": 0, "smart_9_raw": 5000}}
    sick = {
        "serial_number": "S",
        "readings": {
            "smart_5_raw": 200,
            "smart_197_raw": 80,
            "smart_198_raw": 50,
            "d30_smart_5_raw": 150,
            "d30_smart_197_raw": 60,
            "d30_smart_198_raw": 40,
        },
    }
    r = client.post(
        "/predict",
        json={
            "drives": [healthy, sick],
        },
    ).json()
    p = {x["serial_number"]: x["failure_probability"] for x in r["predictions"]}
    assert 0 <= p["H"] <= 1 and p["S"] > p["H"]
    assert client.post("/predict", json={"drives": []}).status_code == 422
    r2 = client.post("/predict", json={"drives": [{"readings": {"bogus": 1}}]}).json()
    assert r2["ignored_fields"] == ["bogus"]
