"""Collect one validated rolling snapshot from the official Hong Kong AQHI feed.

The collector is intentionally inference-free: one invocation downloads once,
validates the complete 18-station x 24-hour source grid, merges by
``(timestamp, station_id)``, keeps seven days, and writes a deterministic status
file.  Missing AQHI values are never imputed.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URL = "https://www.aqhi.gov.hk/epd/ddata/html/out/24aqhi_Eng.xml"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "live"
DEFAULT_MODEL_CONFIG = (
    ROOT / "data/processed/tsmixer/20261111_010719_491327/model_config.json"
)
HK_TZ = "Asia/Hong_Kong"
EXPECTED_STATIONS = [
    "Central/Western",
    "Southern",
    "Eastern",
    "Kwun Tong",
    "Sham Shui Po",
    "Kwai Chung",
    "Tsuen Wan",
    "Tseung Kwan O",
    "Yuen Long",
    "Tuen Mun",
    "Tung Chung",
    "Tai Po",
    "Sha Tin",
    "North",
    "Tap Mun",
    "Causeway Bay",
    "Central",
    "Mong Kok",
]
BOOLEAN_COLUMNS = [
    "is_substituted",
    "is_10_plus",
    "is_missing",
    "is_unavailable",
    "is_invalid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def fetch_once(source_url: str, retries: int) -> bytes:
    if retries < 1:
        raise ValueError("retries must be at least 1")
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "MATH7005-AQHI-collector/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = getattr(response, "status", None)
                if status is not None and status != 200:
                    raise RuntimeError(f"AQHI endpoint returned HTTP {status}")
                payload = response.read()
                if not payload:
                    raise RuntimeError("AQHI endpoint returned an empty response")
                return payload
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(attempt * 2)
    raise RuntimeError(f"AQHI download failed after {retries} attempt(s)") from last_error


def parse_aqhi(raw_value: str) -> dict:
    raw = str(raw_value).strip()
    substituted = "*" in raw
    token = raw.replace("*", "").strip()
    missing = token == ""
    unavailable = token == "-"
    capped = token == "10+"
    numeric = token.isdigit() and 1 <= int(token) <= 10
    invalid = not (missing or unavailable or capped or numeric)
    capped_aqhi = 10.0 if capped else float(token) if numeric else None
    return {
        "aqhi_raw": raw,
        "capped_aqhi": capped_aqhi,
        "is_substituted": substituted,
        "is_10_plus": capped,
        "is_missing": missing,
        "is_unavailable": unavailable,
        "is_invalid": invalid,
    }


def parse_snapshot(payload: bytes) -> tuple[pd.DataFrame, pd.Timestamp]:
    root = ET.fromstring(payload)
    if root.tag != "AQHI24HrReport":
        raise ValueError(f"Unexpected XML root: {root.tag!r}")
    build_text = root.findtext("lastBuildDate")
    if not build_text:
        raise ValueError("AQHI XML has no lastBuildDate")
    build_time = pd.Timestamp(parsedate_to_datetime(build_text)).tz_convert(HK_TZ)
    rows: list[dict] = []
    for item in root.findall("item"):
        station = (item.findtext("StationName") or "").strip()
        datetime_text = (item.findtext("DateTime") or "").strip()
        station_type = (item.findtext("type") or "").strip()
        if not station or not datetime_text:
            raise ValueError("AQHI item is missing StationName or DateTime")
        rows.append(
            {
                "timestamp": pd.Timestamp(parsedate_to_datetime(datetime_text)).tz_convert(HK_TZ),
                "station_id": station,
                "station_type": station_type,
                **parse_aqhi(item.findtext("aqhi") or ""),
            }
        )
    if not rows:
        raise ValueError("AQHI XML contains no item records")
    frame = pd.DataFrame(rows).sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    return frame, build_time


def validate_snapshot(frame: pd.DataFrame) -> dict:
    observed = set(frame["station_id"])
    expected = set(EXPECTED_STATIONS)
    timestamps = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    duplicate_rows = int(frame.duplicated(["timestamp", "station_id"]).sum())
    station_counts = frame.groupby("station_id")["timestamp"].nunique()
    expected_grid_rows = len(expected) * len(timestamps)
    missing_grid_rows = expected_grid_rows - len(
        frame.drop_duplicates(["timestamp", "station_id"])
    )
    steps = timestamps.to_series().diff().dropna()
    report = {
        "rows": int(len(frame)),
        "stations": int(frame["station_id"].nunique()),
        "timestamps": int(len(timestamps)),
        "first_timestamp": str(timestamps.min()),
        "latest_timestamp": str(timestamps.max()),
        "duplicate_station_hours": duplicate_rows,
        "missing_grid_rows": int(missing_grid_rows),
        "all_timestamp_steps_are_hourly": bool(
            steps.eq(pd.Timedelta(hours=1)).all()
        ),
        "minimum_hours_per_station": int(station_counts.min()),
        "maximum_hours_per_station": int(station_counts.max()),
        "unknown_stations": sorted(observed - expected),
        "absent_expected_stations": sorted(expected - observed),
        "invalid_values": int(frame["is_invalid"].sum()),
        "missing_values": int(frame["is_missing"].sum()),
        "unavailable_values": int(frame["is_unavailable"].sum()),
        "substituted_values": int(frame["is_substituted"].sum()),
        "ten_plus_values": int(frame["is_10_plus"].sum()),
    }
    report["snapshot_passed"] = bool(
        report["rows"] == 18 * 24
        and report["stations"] == 18
        and report["timestamps"] == 24
        and duplicate_rows == 0
        and missing_grid_rows == 0
        and report["all_timestamp_steps_are_hourly"]
        and report["minimum_hours_per_station"] == 24
        and report["maximum_hours_per_station"] == 24
        and not report["unknown_stations"]
        and not report["absent_expected_stations"]
        and report["invalid_values"] == 0
    )
    return report


def read_model_stations(model_config: Path) -> list[str]:
    config = json.loads(model_config.read_text(encoding="utf-8"))
    stations = list(config["stations"])
    if len(stations) != 17 or int(config["lookback"]) != 72:
        raise ValueError("Frozen model config is not the expected 17-station, 72-hour design")
    unknown = sorted(set(stations) - set(EXPECTED_STATIONS))
    if unknown:
        raise ValueError(f"Model config contains unknown stations: {unknown}")
    return stations


def read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, keep_default_na=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(HK_TZ)
    frame["capped_aqhi"] = pd.to_numeric(frame["capped_aqhi"], errors="coerce")
    for column in BOOLEAN_COLUMNS:
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    return frame


def continuous_complete_hours(frame: pd.DataFrame, model_stations: list[str]) -> int:
    model_rows = frame.loc[frame["station_id"].isin(model_stations)].copy()
    if model_rows.empty:
        return 0
    panel = (
        model_rows.pivot(index="timestamp", columns="station_id", values="capped_aqhi")
        .reindex(columns=model_stations)
        .sort_index()
    )
    complete = panel.notna().all(axis=1)
    current = panel.index.max()
    count = 0
    while current in complete.index and bool(complete.loc[current]):
        count += 1
        current -= pd.Timedelta(hours=1)
    return count


def merge_rolling(
    existing: pd.DataFrame,
    snapshot: pd.DataFrame,
    model_stations: list[str],
) -> tuple[pd.DataFrame, dict]:
    existing_keys = (
        set(zip(existing["timestamp"], existing["station_id"])) if not existing.empty else set()
    )
    snapshot_keys = set(zip(snapshot["timestamp"], snapshot["station_id"]))
    new_rows = len(snapshot_keys - existing_keys)
    revised_rows = 0
    if not existing.empty:
        compare_columns = ["aqhi_raw", "capped_aqhi", *BOOLEAN_COLUMNS]
        old = existing.set_index(["timestamp", "station_id"])[compare_columns]
        new = snapshot.set_index(["timestamp", "station_id"])[compare_columns]
        overlap = old.index.intersection(new.index)
        for key in overlap:
            old_values = old.loc[key].fillna("<NA>").astype(str)
            new_values = new.loc[key].fillna("<NA>").astype(str)
            revised_rows += int(not old_values.equals(new_values))

    combined = pd.concat([existing, snapshot], ignore_index=True)
    combined = combined.sort_values(["timestamp", "station_id"])
    combined = combined.drop_duplicates(["timestamp", "station_id"], keep="last")
    latest = combined["timestamp"].max()
    cutoff = latest - pd.Timedelta(days=7) + pd.Timedelta(hours=1)
    combined = combined.loc[combined["timestamp"].ge(cutoff)].copy()
    continuous = continuous_complete_hours(combined, model_stations)
    return combined, {
        "new_station_hour_rows": int(new_rows),
        "revised_station_hour_rows": int(revised_rows),
        "rolling_rows": int(len(combined)),
        "rolling_stations": int(combined["station_id"].nunique()),
        "rolling_first_timestamp": str(combined["timestamp"].min()),
        "rolling_latest_timestamp": str(latest),
        "continuous_complete_model_hours": int(continuous),
        "required_model_hours": 72,
        "model_station_count": len(model_stations),
        "model_ready": bool(continuous >= 72),
    }


def write_if_changed(path: Path, content: bytes) -> bool:
    if path.exists() and path.read_bytes() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return True


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    rolling_path = output_dir / "aqhi_rolling.csv"
    status_path = output_dir / "status.json"
    raw_path = output_dir / "latest_24aqhi_Eng.xml"
    model_stations = read_model_stations(args.model_config.resolve())

    # No output is changed unless download, XML parsing and source validation all pass.
    payload = fetch_once(args.source_url, args.retries)
    snapshot, build_time = parse_snapshot(payload)
    snapshot_report = validate_snapshot(snapshot)
    if not snapshot_report["snapshot_passed"]:
        raise RuntimeError(f"Official source snapshot failed validation: {snapshot_report}")

    existing = read_existing(rolling_path)
    combined, rolling_report = merge_rolling(existing, snapshot, model_stations)
    run_report = {
        "new_station_hour_rows": rolling_report.pop("new_station_hour_rows"),
        "revised_station_hour_rows": rolling_report.pop("revised_station_hour_rows"),
    }
    status = {
        "source_url": args.source_url,
        "source_build_time": str(build_time),
        "source_snapshot": snapshot_report,
        "rolling": rolling_report,
        "policy": "No imputation; predict only with 72 consecutive complete hours for all 17 model stations.",
    }
    csv_bytes = combined.to_csv(index=False, lineterminator="\n").encode("utf-8")
    status_bytes = (
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    changed = {
        "rolling_csv": write_if_changed(rolling_path, csv_bytes),
        "status_json": write_if_changed(status_path, status_bytes),
        "raw_xml": write_if_changed(raw_path, payload),
    }
    result = {
        "collector_ok": True,
        "changed_files": changed,
        "run": run_report,
        **status,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("LIVE_AQHI_COLLECTOR_OK")


if __name__ == "__main__":
    main()
