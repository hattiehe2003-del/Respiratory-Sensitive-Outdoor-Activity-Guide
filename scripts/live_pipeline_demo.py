"""Proof-of-concept for the rolling live AQHI data and inference pipeline.

This script deliberately writes only to ``data/live_demo``.  It does not modify
the audited historical data or the Streamlit application.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.aqhi_data import STATIONS, parse_aqhi, station_key
from src.data_loader import load_replay, replay_window, valid_origins
from src.models import ForecastBundle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "live_demo"
SOURCE_URL = "https://www.aqhi.gov.hk/epd/ddata/html/out/24aqhi_Eng.xml"
ROLLING_PATH = OUTPUT_DIR / "aqhi_rolling.csv"
STATUS_PATH = OUTPUT_DIR / "status.json"
RAW_PATH = OUTPUT_DIR / "latest_24aqhi_Eng.xml"
REPORT_PATH = OUTPUT_DIR / "demo_report.json"
SIMULATION_INPUT_PATH = OUTPUT_DIR / "simulated_72h_input.csv"
SIMULATION_FORECAST_PATH = OUTPUT_DIR / "simulated_24h_forecasts.csv"
HK_TZ = "Asia/Hong_Kong"


def read_model_config() -> dict:
    path = ROOT / "data/processed/tsmixer/20261111_010719_491327/model_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def download_xml(attempts: int = 3) -> bytes:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "MATH7005-AQHI-pipeline-demo/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"AQHI endpoint returned HTTP {response.status}")
                payload = response.read()
                if not payload:
                    raise RuntimeError("AQHI endpoint returned an empty response")
                return payload
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(attempt * 2)
    raise RuntimeError(f"AQHI download failed after {attempts} attempts") from last_error


def parse_live_xml(payload: bytes) -> tuple[pd.DataFrame, pd.Timestamp]:
    root = ET.fromstring(payload)
    if root.tag != "AQHI24HrReport":
        raise ValueError(f"Unexpected XML root element: {root.tag!r}")
    build_text = root.findtext("lastBuildDate")
    if not build_text:
        raise ValueError("The XML response has no lastBuildDate")
    build_time = pd.Timestamp(parsedate_to_datetime(build_text)).tz_convert(HK_TZ)
    rows: list[dict] = []
    for item in root.findall("item"):
        source_name = (item.findtext("StationName") or "").strip()
        datetime_text = (item.findtext("DateTime") or "").strip()
        aqhi_text = (item.findtext("aqhi") or "").strip()
        station_type = (item.findtext("type") or "").strip()
        if not source_name or not datetime_text:
            raise ValueError("An AQHI item is missing StationName or DateTime")
        parsed = parse_aqhi(aqhi_text)
        capped = 10.0 if parsed["is_10_plus"] else parsed["aqhi_value"]
        rows.append(
            {
                "timestamp": pd.Timestamp(parsedate_to_datetime(datetime_text)).tz_convert(HK_TZ),
                "station_id": station_key(source_name),
                "station_source": source_name,
                "station_type": station_type,
                "aqhi_raw": parsed["aqhi_raw"],
                "capped_aqhi": capped,
                "is_substituted": parsed["is_substituted"],
                "is_10_plus": parsed["is_10_plus"],
                "is_missing": parsed["is_missing"],
                "is_unavailable": parsed["is_unavailable"],
                "is_invalid": parsed["is_invalid"],
            }
        )
    if not rows:
        raise ValueError("The AQHI XML response contains no item records")
    frame = pd.DataFrame(rows).sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    return frame, build_time


def validate_snapshot(frame: pd.DataFrame, build_time: pd.Timestamp) -> dict:
    expected_stations = {station[2] for station in STATIONS}
    observed_stations = set(frame["station_id"])
    timestamps = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    station_counts = frame.groupby("station_id")["timestamp"].nunique()
    duplicate_rows = int(frame.duplicated(["timestamp", "station_id"]).sum())
    expected_grid_rows = len(observed_stations) * len(timestamps)
    missing_grid_rows = expected_grid_rows - len(frame.drop_duplicates(["timestamp", "station_id"]))
    hourly_steps = timestamps.to_series().diff().dropna()
    all_steps_hourly = bool(hourly_steps.eq(pd.Timedelta(hours=1)).all())
    unknown = sorted(observed_stations - expected_stations)
    absent = sorted(expected_stations - observed_stations)
    report = {
        "source_build_time": str(build_time),
        "rows": int(len(frame)),
        "stations": int(frame["station_id"].nunique()),
        "timestamps": int(len(timestamps)),
        "first_timestamp": str(timestamps.min()),
        "latest_timestamp": str(timestamps.max()),
        "duplicate_station_hours": duplicate_rows,
        "missing_grid_rows": int(missing_grid_rows),
        "all_timestamp_steps_are_hourly": all_steps_hourly,
        "minimum_hours_per_station": int(station_counts.min()),
        "maximum_hours_per_station": int(station_counts.max()),
        "unknown_stations": unknown,
        "absent_expected_stations": absent,
        "invalid_values": int(frame["is_invalid"].sum()),
        "missing_values": int(frame["is_missing"].sum()),
        "unavailable_values": int(frame["is_unavailable"].sum()),
        "substituted_values": int(frame["is_substituted"].sum()),
        "ten_plus_values": int(frame["is_10_plus"].sum()),
    }
    report["snapshot_passed"] = bool(
        report["stations"] == 18
        and report["timestamps"] == 24
        and report["rows"] == 18 * 24
        and duplicate_rows == 0
        and missing_grid_rows == 0
        and all_steps_hourly
        and not unknown
        and not absent
        and report["invalid_values"] == 0
    )
    return report


def read_existing_rolling() -> pd.DataFrame:
    if not ROLLING_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(ROLLING_PATH, keep_default_na=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(HK_TZ)
    frame["capped_aqhi"] = pd.to_numeric(frame["capped_aqhi"], errors="coerce")
    for column in [
        "is_substituted",
        "is_10_plus",
        "is_missing",
        "is_unavailable",
        "is_invalid",
    ]:
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    return frame


def consecutive_complete_hours(frame: pd.DataFrame, model_stations: list[str]) -> int:
    selected = frame.loc[frame["station_id"].isin(model_stations)].copy()
    if selected.empty:
        return 0
    panel = selected.pivot(index="timestamp", columns="station_id", values="capped_aqhi")
    panel = panel.reindex(columns=model_stations).sort_index()
    complete = panel.notna().all(axis=1)
    latest = panel.index.max()
    count = 0
    current = latest
    while current in complete.index and bool(complete.loc[current]):
        count += 1
        current -= pd.Timedelta(hours=1)
    return count


def merge_rolling(snapshot: pd.DataFrame, model_stations: list[str]) -> dict:
    existing = read_existing_rolling()
    existing_keys = (
        set(zip(existing["timestamp"], existing["station_id"])) if not existing.empty else set()
    )
    snapshot_keys = set(zip(snapshot["timestamp"], snapshot["station_id"]))
    new_key_rows = len(snapshot_keys - existing_keys)
    combined = pd.concat([existing, snapshot], ignore_index=True)
    combined = combined.sort_values(["timestamp", "station_id"])
    combined = combined.drop_duplicates(["timestamp", "station_id"], keep="last")
    latest = combined["timestamp"].max()
    cutoff = latest - pd.Timedelta(days=7) + pd.Timedelta(hours=1)
    combined = combined.loc[combined["timestamp"].ge(cutoff)].copy()
    combined.to_csv(ROLLING_PATH, index=False, encoding="utf-8")
    continuous = consecutive_complete_hours(combined, model_stations)
    status = {
        "source_url": SOURCE_URL,
        "latest_timestamp": str(latest),
        "rolling_rows": int(len(combined)),
        "rolling_stations": int(combined["station_id"].nunique()),
        "new_station_hour_rows": int(new_key_rows),
        "continuous_complete_model_hours": int(continuous),
        "required_model_hours": 72,
        "model_station_count": len(model_stations),
        "model_ready": bool(continuous >= 72),
        "policy": "No imputation; forecast only when all 17 stations have 72 consecutive hours.",
    }
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def run_live_collection(model_stations: list[str]) -> tuple[dict, dict]:
    payload = download_xml()
    RAW_PATH.write_bytes(payload)
    snapshot, build_time = parse_live_xml(payload)
    snapshot_report = validate_snapshot(snapshot, build_time)
    if not snapshot_report["snapshot_passed"]:
        raise RuntimeError(f"Live AQHI snapshot validation failed: {snapshot_report}")
    rolling_report = merge_rolling(snapshot, model_stations)
    return snapshot_report, rolling_report


def run_historical_72h_simulation() -> dict:
    _, panel, stations = load_replay(ROOT)
    bundle = ForecastBundle(ROOT)
    if stations != bundle.stations:
        raise RuntimeError("Replay station order differs from the frozen model order")
    origins = valid_origins(panel, bundle.lookback, bundle.horizon)
    if len(origins) == 0:
        raise RuntimeError("No complete historical 72/24 window is available")
    origin = origins[-1]
    past, actual = replay_window(panel, origin, bundle.lookback, bundle.horizon)
    predictions = bundle.predict(past.to_numpy(dtype="float32"))
    past.to_csv(SIMULATION_INPUT_PATH, encoding="utf-8")
    forecast_rows: list[dict] = []
    metrics: dict[str, dict] = {}
    actual_values = actual.to_numpy(dtype="float32")
    for model_name, values in predictions.items():
        error = values - actual_values
        metrics[model_name] = {
            "mae_on_demonstration_window": float(np.mean(np.abs(error))),
            "rmse_on_demonstration_window": float(np.sqrt(np.mean(error**2))),
            "minimum_prediction": float(values.min()),
            "maximum_prediction": float(values.max()),
        }
        for time_index, timestamp in enumerate(actual.index):
            for station_index, station in enumerate(stations):
                forecast_rows.append(
                    {
                        "forecast_origin": origin,
                        "forecast_timestamp": timestamp,
                        "station_id": station,
                        "model": model_name,
                        "prediction": float(values[time_index, station_index]),
                        "historical_actual": float(actual_values[time_index, station_index]),
                    }
                )
    pd.DataFrame(forecast_rows).to_csv(
        SIMULATION_FORECAST_PATH, index=False, encoding="utf-8"
    )
    return {
        "simulation_origin": str(origin),
        "input_shape": list(past.shape),
        "target_shape": list(actual.shape),
        "forecast_models": list(predictions),
        "forecast_shape_per_model": [bundle.horizon, len(bundle.stations)],
        "all_predictions_finite": bool(
            all(np.isfinite(values).all() for values in predictions.values())
        ),
        "metrics": metrics,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = read_model_config()
    model_stations = list(config["stations"])
    if len(model_stations) != 17 or int(config["lookback"]) != 72 or int(config["horizon"]) != 24:
        raise RuntimeError("The frozen model does not use the expected 17 x 72 -> 17 x 24 design")

    first_snapshot, first_rolling = run_live_collection(model_stations)
    second_snapshot, second_rolling = run_live_collection(model_stations)
    simulation = run_historical_72h_simulation()
    report = {
        "demo_passed": bool(
            first_snapshot["snapshot_passed"]
            and second_snapshot["snapshot_passed"]
            and second_rolling["new_station_hour_rows"] == 0
            and simulation["input_shape"] == [72, 17]
            and simulation["forecast_shape_per_model"] == [24, 17]
            and simulation["all_predictions_finite"]
        ),
        "live_first_run": {
            "snapshot": first_snapshot,
            "rolling": first_rolling,
        },
        "live_immediate_second_run": {
            "snapshot": second_snapshot,
            "rolling": second_rolling,
            "expected_result": "0 new station-hour rows because the same 24-hour snapshot is deduplicated",
        },
        "historical_72h_model_simulation": simulation,
        "files": {
            "rolling_live_data": str(ROLLING_PATH),
            "live_status": str(STATUS_PATH),
            "latest_raw_xml": str(RAW_PATH),
            "simulated_72h_input": str(SIMULATION_INPUT_PATH),
            "simulated_24h_forecasts": str(SIMULATION_FORECAST_PATH),
        },
        "boundary": "The historical simulation proves the 72h-to-24h inference path; the live feed needs about 48 more hours before a genuine live forecast is allowed.",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["demo_passed"]:
        raise SystemExit("PIPELINE_DEMO_FAILED")
    print("PIPELINE_DEMO_OK")


if __name__ == "__main__":
    main()
