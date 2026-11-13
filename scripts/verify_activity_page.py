"""Render the activity-area page against a temporary valid forecast."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIVE = ROOT / "data/live"
FORECAST = LIVE / "forecast_24h.json"
STATUS = LIVE / "prediction_status.json"
original_forecast = FORECAST.read_bytes() if FORECAST.exists() else None
original_status = STATUS.read_bytes() if STATUS.exists() else None

origin = pd.Timestamp("2026-11-13 17:00", tz="Asia/Hong_Kong")
times = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h")
stations = ["Causeway Bay", "Central", "Central/Western", "Eastern", "Kwai Chung", "Kwun Tong", "Mong Kok", "North", "Sha Tin", "Sham Shui Po", "Southern", "Tai Po", "Tseung Kwan O", "Tsuen Wan", "Tuen Mun", "Tung Chung", "Yuen Long"]
records = [{"forecast_origin": origin.isoformat(), "timestamp": timestamp.isoformat(), "station_id": station, "tsmixer": 2.5 + hour / 24, "patchtst_style": 2.7 + hour / 24, "model_gap": 0.2} for hour, timestamp in enumerate(times) for station in stations]
summary = {"prediction_ready": True, "station_count": 17, "forecast_hours": 24, "forecast_rows": 408, "forecast_origin": origin.isoformat(), "forecast_end": times[-1].isoformat(), "generated_at": origin.isoformat()}

try:
    FORECAST.write_text(json.dumps({"summary": summary, "records": records}), encoding="utf-8")
    STATUS.write_text(json.dumps({"prediction_ready": True}), encoding="utf-8")
    app = AppTest.from_file(ROOT / "app.py").run(timeout=60)
    app.radio[0].set_value("Station forecast").run(timeout=60)
    assert len(app.exception) == 0, app.exception
    labels = [item.label for item in app.selectbox]
    assert "Activity area / monitoring station" in labels, labels
    assert any("Lower-exposure 3h window" == item.label for item in app.metric)
    print("ACTIVITY_PAGE_OK")
finally:
    if original_forecast is None:
        FORECAST.unlink(missing_ok=True)
    else:
        FORECAST.write_bytes(original_forecast)
    if original_status is None:
        STATUS.unlink(missing_ok=True)
    else:
        STATUS.write_bytes(original_status)
