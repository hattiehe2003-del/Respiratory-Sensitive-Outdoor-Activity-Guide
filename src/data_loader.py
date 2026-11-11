"""Build and load the compact, traceable historical-replay dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


APP_DATA_RELATIVE = Path("data/processed/app/replay_history_2025_2026.csv.gz")
APP_MANIFEST_RELATIVE = Path("data/processed/app/replay_manifest.json")
TEST_START = pd.Timestamp("2025-01-01 00:00", tz="Asia/Hong_Kong")
TEST_END = pd.Timestamp("2026-07-01 00:00", tz="Asia/Hong_Kong")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    converted = series.astype(str).str.strip().str.lower()
    if not converted.isin(["true", "false"]).all():
        raise ValueError(f"Unexpected Boolean values: {sorted(converted.unique())}")
    return converted.eq("true")


def model_config(root: Path) -> dict:
    path = Path(root) / "data/processed/tsmixer/20261111_010719_491327/model_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_app_data(root: Path, overwrite: bool = False) -> Path:
    """Create an inference-ready subset without changing the audited source."""
    root = Path(root)
    destination = root / APP_DATA_RELATIVE
    if destination.exists() and not overwrite:
        return destination
    cfg = model_config(root)
    stations = list(cfg["stations"])
    source = Path(cfg["source"]) / "history_all_audited.csv.gz"
    start_date = (TEST_START - pd.Timedelta(hours=int(cfg["lookback"]))).date().isoformat()
    columns = [
        "source_date",
        "timestamp",
        "station_id",
        "aqhi_raw",
        "aqhi_value",
        "is_substituted",
        "is_10_plus",
        "source_file",
    ]
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source,
        usecols=columns,
        keep_default_na=False,
        dtype={"source_date": str},
        chunksize=250_000,
    ):
        mask = chunk["source_date"].ge(start_date) & chunk["station_id"].isin(stations)
        if mask.any():
            selected.append(chunk.loc[mask].copy())
    if not selected:
        raise RuntimeError("No replay rows were selected from the audited history")
    replay = pd.concat(selected, ignore_index=True)
    replay["timestamp"] = pd.to_datetime(replay["timestamp"], utc=True).dt.tz_convert(
        "Asia/Hong_Kong"
    )
    replay = replay.loc[replay["timestamp"].le(TEST_END)].copy()
    replay["is_10_plus"] = _as_bool(replay["is_10_plus"])
    replay["is_substituted"] = _as_bool(replay["is_substituted"])
    replay["capped_aqhi"] = pd.to_numeric(replay["aqhi_value"], errors="coerce")
    replay.loc[replay["is_10_plus"], "capped_aqhi"] = 10.0
    if replay.duplicated(["timestamp", "station_id"]).any():
        raise ValueError("Duplicate station-hours detected in the replay subset")
    replay = replay.sort_values(["timestamp", "station_id"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    replay.to_csv(destination, index=False, encoding="utf-8-sig", compression="gzip")
    manifest = {
        "purpose": "Historical replay for the Streamlit demonstration; not a live feed",
        "source": str(source),
        "source_is_modified": False,
        "output": str(destination),
        "stations": stations,
        "rows": int(len(replay)),
        "start": str(replay["timestamp"].min()),
        "end": str(replay["timestamp"].max()),
        "target": "capped AQHI; original 10+ encoded as 10 with marker retained",
        "created_by": "src/data_loader.py::build_app_data",
    }
    (root / APP_MANIFEST_RELATIVE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def load_replay(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    root = Path(root)
    cfg = model_config(root)
    stations = list(cfg["stations"])
    path = root / APP_DATA_RELATIVE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.build_app_data` before starting the app."
        )
    history = pd.read_csv(path, keep_default_na=False)
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True).dt.tz_convert(
        "Asia/Hong_Kong"
    )
    history["capped_aqhi"] = pd.to_numeric(history["capped_aqhi"], errors="coerce")
    for column in ["is_10_plus", "is_substituted"]:
        history[column] = _as_bool(history[column])
    index = pd.date_range(history["timestamp"].min(), history["timestamp"].max(), freq="h")
    panel = (
        history.pivot(index="timestamp", columns="station_id", values="capped_aqhi")
        .reindex(index=index, columns=stations)
        .rename_axis(index="timestamp")
    )
    return history, panel, stations


def valid_origins(panel: pd.DataFrame, lookback: int = 72, horizon: int = 24) -> pd.DatetimeIndex:
    present = pd.Series(np.isfinite(panel.to_numpy()).all(axis=1), index=panel.index)
    input_ok = present.astype(int).rolling(lookback).sum().shift(1).eq(lookback)
    target_ok = present.astype(int).rolling(horizon).sum().shift(-(horizon - 1)).eq(horizon)
    end = panel.index + pd.Timedelta(hours=horizon - 1)
    mask = input_ok.to_numpy() & target_ok.to_numpy()
    mask &= panel.index > TEST_START
    mask &= end <= TEST_END
    return panel.index[mask]


def replay_window(
    panel: pd.DataFrame,
    origin: pd.Timestamp,
    lookback: int = 72,
    horizon: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    origin = pd.Timestamp(origin)
    location = panel.index.get_indexer([origin])[0]
    if location < lookback or location < 0:
        raise ValueError("Forecast origin does not have 72 preceding hours")
    past = panel.iloc[location - lookback : location]
    future = panel.iloc[location : location + horizon]
    if len(past) != lookback or len(future) != horizon:
        raise ValueError("Forecast origin does not have a complete 72/24 replay window")
    if not np.isfinite(past.to_numpy()).all() or not np.isfinite(future.to_numpy()).all():
        raise ValueError("Replay window contains missing observations")
    return past, future


def load_stations(root: Path, model_stations: list[str]) -> pd.DataFrame:
    stations = pd.read_csv(Path(root) / "data/processed/stations.csv")
    stations = stations.loc[stations["station_id"].isin(model_stations)].copy()
    stations["station_order"] = stations["station_id"].map(
        {name: i for i, name in enumerate(model_stations)}
    )
    return stations.sort_values("station_order").drop(columns="station_order")

