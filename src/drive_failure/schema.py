"""Column names and feature definitions shared across the package."""

# SMART attributes used. 5/187/197/198 are the ones Backblaze has highlighted as failure signals;
# 9 is power-on hours, 188 command timeout, 194 temperature.
SMART_IDS = (5, 9, 187, 188, 194, 197, 198)
WARNING_IDS = (5, 187, 197, 198)
RAW_COLS = [f"smart_{i}_raw" for i in SMART_IDS]
DELTA_COLS = [f"d30_smart_{i}_raw" for i in SMART_IDS if i not in (9, 194)]
MFRS = ("seagate", "hgst", "wd", "toshiba", "other")
MFR_COLS = [f"mfr_{m}" for m in MFRS]
FEATURES = RAW_COLS + DELTA_COLS + ["capacity_tb"] + MFR_COLS

REQUIRED_DAILY = ("date", "serial_number", "model", "capacity_bytes", "failure")
LAG_DAYS = 30  # window for the delta features; the loader grid step must divide this
