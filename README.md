# drive-failure

Predict whether a hard drive will fail within 30 days, using SMART readings from the public
[Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
data. The point of the project is the evaluation, not the model: chronological splits with a
leakage gap, comparison against a simple SMART rule, calibrated probabilities, alert-budget
metrics, and confidence intervals that resample whole drives.

## Data

Backblaze publishes one CSV per day, zipped per quarter (`data_Q#_YYYY.zip`). Each row is one
drive on one day with SMART raw and normalized values and a `failure` flag on its last day.
Backblaze asks that you cite it as the source and does not allow selling the data itself.
Download a few consecutive quarters into `data/raw/`. Quarters are large (the Q2 2026 zip is about
2 GB), so start with two to four and use `--sample-frac`.

## Quick start

```bash
pip install -e '.[xgb,dev]'
drive-failure build data/raw --sample-frac 0.25     # labelled snapshots -> data/snapshots.parquet
drive-failure train                                   # model + REPORT.md
uvicorn drive_failure.api:app --port 8000             # serve
```

No data yet? `make demo` runs everything on a built-in synthetic generator. Its numbers are
meaningless (the generator sets the signal strength) and the report says so.

## Method

- **Snapshots**: one row per drive every 5 days (`--step`, must divide 30). Features use data up
  to the snapshot date only: current SMART raw values for 5, 9, 187, 188, 194, 197, 198, 30-day
  changes of the error counters, capacity and manufacturer.
- **Label**: the drive fails in the next 30 days. Snapshots too close to the end of the data are
  dropped because their label is unknown. Drives that leave the fleet without failing count as
  healthy.
- **Split**: chronological train/validation/test, with a 30-day gap between parts so no training
  label is decided by an event inside the validation or test period. A test showing this is in
  `tests/test_features.py`.
- **Models**: (1) rule: count of SMART 5/187/197/198 above zero; (2) logistic regression;
  (3) XGBoost (falls back to scikit-learn gradient boosting if XGBoost is not installed), with
  early stopping on the validation period.
- **Calibration**: isotonic regression fitted on the validation period, evaluated with Brier score
  and a reliability table on test.
- **Metrics**: PR-AUC, ROC-AUC, recall and precision at a 1% alert budget (ties split
  proportionally), 95% bootstrap intervals over drives.
- **API**: `POST /predict` returns a calibrated probability and an alert flag (probability above
  the validation-period top-1% cut-off).

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d \
 '{"drives":[{"serial_number":"X1","model":"ST4000DM000","capacity_bytes":4e12,
   "readings":{"smart_5_raw":120,"smart_197_raw":40,"d30_smart_5_raw":90}}]}'
```

## Results

Not yet run on real data. After training on real data, paste the test table from `REPORT.md`
here with the quarters used, sample fraction and seed. Report the number of failures in the test
period; with fewer than about 30 the intervals are too wide to rank models.

## Limitations

- The same drives can appear in train and test on different dates. The gap prevents label
  leakage but not drive-level similarity. A drive-disjoint evaluation is a worthwhile extension.
- Many real failures give no SMART warning, so recall is capped well below 100%.
- Failure rates and SMART reporting differ by model and change over time, so the model will drift;
  retrain regularly.
- The Backblaze schema changes between quarters; the loader checks required columns and fills
  missing SMART columns with NaN.
- The real-data loader and the training run on real data were not run in the development
  environment (no network access to Backblaze). Tests use hand-built tables and synthetic data.

## Development

```bash
make install && make test && make lint
```

Layout: `data.py` (loader), `features.py` (snapshots, labels, split), `models.py`, `evaluate.py`,
`pipeline.py` (train, calibrate, report), `api.py`, `cli.py`, `synth.py`.

## License

MIT. Backblaze data is subject to Backblaze's terms (cite as source).
