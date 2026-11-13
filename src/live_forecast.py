"""Load and validate website-ready live AQHI forecasts."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"forecast_origin", "timestamp", "station_id", "tsmixer", "patchtst_style"}
EXPECTED_STATIONS = 17
EXPECTED_HOURS = 24
EXPECTED_ROWS = EXPECTED_STATIONS * EXPECTED_HOURS
HK_TZ = "Asia/Hong_Kong"


@dataclass(frozen=True)
class LiveForecastSnapshot:
    records: pd.DataFrame | None
    summary: dict | None
    status: dict | None
    state: str
    message: str
    forecast_expired: bool = False

    @property
    def usable(self) -> bool:
        return self.records is not None and self.summary is not None


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _validate_forecast(payload: dict) -> tuple[pd.DataFrame, dict]:
    summary = payload.get("summary")
    records = payload.get("records")
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError("Forecast JSON must contain summary and records")
    if not bool(summary.get("prediction_ready")):
        raise ValueError("Forecast summary is not marked prediction_ready")
    frame = pd.DataFrame(records)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Forecast records are missing columns: {missing}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Forecast must contain {EXPECTED_ROWS} rows, found {len(frame)}")
    if frame.duplicated(["timestamp", "station_id"]).any():
        raise ValueError("Forecast contains duplicate station-hour rows")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise", utc=True).dt.tz_convert(HK_TZ)
    frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"], errors="raise", utc=True).dt.tz_convert(HK_TZ)
    for column in ("tsmixer", "patchtst_style"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Forecast column {column} contains non-finite values")
    station_counts = frame.groupby("station_id")["timestamp"].nunique()
    if len(station_counts) != EXPECTED_STATIONS or not station_counts.eq(EXPECTED_HOURS).all():
        raise ValueError("Forecast must contain 24 distinct hours for each of 17 stations")
    timestamps = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    if len(timestamps) != EXPECTED_HOURS:
        raise ValueError("Forecast stations do not share one 24-hour time axis")
    if not timestamps.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Forecast timestamps are not consecutive hourly values")
    if frame["forecast_origin"].nunique() != 1:
        raise ValueError("Forecast records contain multiple forecast origins")
    for key, expected in {"station_count": 17, "forecast_hours": 24, "forecast_rows": 408}.items():
        if int(summary.get(key, -1)) != expected:
            raise ValueError(f"Forecast summary {key} does not equal {expected}")
    return frame.sort_values(["timestamp", "station_id"]).reset_index(drop=True), summary


def load_live_forecast(root: Path, now: pd.Timestamp | None = None) -> LiveForecastSnapshot:
    live_dir = Path(root) / "data" / "live"
    status_path = live_dir / "prediction_status.json"
    forecast_path = live_dir / "forecast_24h.json"
    try:
        status = _read_json(status_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return LiveForecastSnapshot(None, None, None, "invalid", f"Invalid prediction status: {error}")
    if not forecast_path.exists():
        available = int((status or {}).get("available_complete_hours", 0))
        required = int((status or {}).get("required_complete_hours", 72))
        state = "collecting" if status is not None else "unavailable"
        return LiveForecastSnapshot(None, None, status, state, f"Waiting for a complete input window: {available}/{required} hours.")
    try:
        payload = _read_json(forecast_path)
        if payload is None:
            raise ValueError("Forecast file is empty")
        records, summary = _validate_forecast(payload)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        return LiveForecastSnapshot(None, None, status, "invalid", f"Invalid forecast output: {error}")
    forecast_end = pd.to_datetime(summary.get("forecast_end"), errors="coerce", utc=True)
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    expired = bool(not pd.isna(forecast_end) and current > forecast_end)
    state = "live" if bool((status or {}).get("prediction_ready")) else "last_valid"
    message = "Latest live forecast is ready." if state == "live" else "Serving the last valid forecast while the newest input window is unavailable."
    if expired:
        state = "expired"
        message = "The last valid forecast has expired and must be labelled historical until refreshed."
    return LiveForecastSnapshot(records, summary, status, state, message, expired)
