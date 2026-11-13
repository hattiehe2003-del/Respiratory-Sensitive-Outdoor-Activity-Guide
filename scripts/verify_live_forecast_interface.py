"""Focused checks for the live forecast file interface."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live_forecast import load_live_forecast


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


with TemporaryDirectory() as directory:
    root = Path(directory)
    status_path = root / "data/live/prediction_status.json"
    forecast_path = root / "data/live/forecast_24h.json"
    write_json(status_path, {"prediction_ready": False, "available_complete_hours": 66, "required_complete_hours": 72})
    collecting = load_live_forecast(root)
    assert collecting.state == "collecting" and not collecting.usable

    origin = pd.Timestamp("2026-11-13 17:00", tz="Asia/Hong_Kong")
    times = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h")
    records = []
    for timestamp in times:
        for index in range(17):
            records.append({"forecast_origin": origin.isoformat(), "timestamp": timestamp.isoformat(), "station_id": f"Station {index + 1}", "tsmixer": 3.0, "patchtst_style": 3.2})
    summary = {"prediction_ready": True, "station_count": 17, "forecast_hours": 24, "forecast_rows": 408, "forecast_end": times[-1].isoformat()}
    write_json(forecast_path, {"summary": summary, "records": records})
    write_json(status_path, {"prediction_ready": True})
    live = load_live_forecast(root, now=origin)
    assert live.state == "live" and live.usable and len(live.records) == 408

    write_json(status_path, {"prediction_ready": False, "available_complete_hours": 66, "required_complete_hours": 72})
    fallback = load_live_forecast(root, now=origin)
    assert fallback.state == "last_valid" and fallback.usable

print("LIVE_FORECAST_INTERFACE_OK")
