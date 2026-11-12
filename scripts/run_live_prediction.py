"""Create a website-ready live 24-hour forecast when the rolling input is ready.

A run exits successfully when fewer than 72 complete joint hours are available,
but writes an explicit status file. No missing values are imputed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE_DIR = ROOT / "data" / "live"
ROLLING_PATH = LIVE_DIR / "aqhi_rolling.csv"
STATUS_PATH = LIVE_DIR / "prediction_status.json"
CSV_PATH = LIVE_DIR / "forecast_24h.csv"
JSON_PATH = LIVE_DIR / "forecast_24h.json"
MODEL_CONFIG = ROOT / "data/processed/tsmixer/20261111_010719_491327/model_config.json"
HK_TZ = "Asia/Hong_Kong"
LOOKBACK = 72
HORIZON = 24


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_status(payload: dict) -> None:
    write_text_atomic(
        STATUS_PATH,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def load_complete_panel() -> tuple[pd.DataFrame, list[str]]:
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    stations = list(config["stations"])
    if len(stations) != 17 or int(config["lookback"]) != LOOKBACK:
        raise ValueError("Frozen model config is not the expected 17-station, 72-hour design")
    frame = pd.read_csv(ROLLING_PATH)
    required = {"timestamp", "station_id", "capped_aqhi"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Rolling input is missing columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(HK_TZ)
    frame["capped_aqhi"] = pd.to_numeric(frame["capped_aqhi"], errors="coerce")
    if frame.duplicated(["timestamp", "station_id"]).any():
        raise ValueError("Rolling input contains duplicate station-hour rows")
    panel = (
        frame.loc[frame["station_id"].isin(stations)]
        .pivot(index="timestamp", columns="station_id", values="capped_aqhi")
        .reindex(columns=stations)
        .sort_index()
    )
    return panel, stations


def trailing_complete_hours(panel: pd.DataFrame) -> int:
    if panel.empty:
        return 0
    complete = panel.notna().all(axis=1)
    current = panel.index.max()
    count = 0
    while current in complete.index and bool(complete.loc[current]):
        count += 1
        current -= pd.Timedelta(hours=1)
    return count


def main() -> None:
    if not ROLLING_PATH.exists():
        status = {
            "prediction_ready": False,
            "reason": "Rolling AQHI input does not exist yet.",
            "required_complete_hours": LOOKBACK,
            "available_complete_hours": 0,
        }
        write_status(status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        print("LIVE_PREDICTION_SKIPPED")
        return

    panel, stations = load_complete_panel()
    available = trailing_complete_hours(panel)
    latest = panel.index.max() if not panel.empty else None
    if available < LOOKBACK:
        status = {
            "prediction_ready": False,
            "reason": "Waiting for 72 consecutive complete hours for all 17 model stations.",
            "required_complete_hours": LOOKBACK,
            "available_complete_hours": int(available),
            "latest_input_timestamp": latest.isoformat() if latest is not None else None,
            "model_station_count": len(stations),
            "policy": "No imputation; an incomplete input window is never predicted.",
        }
        write_status(status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        print("LIVE_PREDICTION_SKIPPED")
        return

    expected_index = pd.date_range(end=latest, periods=LOOKBACK, freq="h")
    past = panel.reindex(expected_index).reindex(columns=stations)
    values = past.to_numpy(dtype="float32")
    if past.shape != (LOOKBACK, len(stations)) or not np.isfinite(values).all():
        raise RuntimeError("The selected live input window is not a complete finite 72 x 17 matrix")

    from src.models import ForecastBundle

    bundle = ForecastBundle(ROOT)
    if stations != bundle.stations or bundle.lookback != LOOKBACK or bundle.horizon != HORIZON:
        raise ValueError("Frozen models do not match the validated live input design")
    predictions = bundle.predict(values)
    tsmixer = predictions["TSMixer"]
    patchtst = predictions["PatchTST-style"]
    if tsmixer.shape != (HORIZON, len(stations)) or patchtst.shape != tsmixer.shape:
        raise RuntimeError("Frozen models returned an unexpected forecast shape")
    if not np.isfinite(tsmixer).all() or not np.isfinite(patchtst).all():
        raise RuntimeError("Frozen models returned non-finite forecasts")

    forecast_index = pd.date_range(
        start=latest + pd.Timedelta(hours=1), periods=HORIZON, freq="h"
    )
    frame = pd.DataFrame(
        {
            "forecast_origin": [latest.isoformat()] * (HORIZON * len(stations)),
            "timestamp": forecast_index.repeat(len(stations)).astype(str),
            "station_id": stations * HORIZON,
            "tsmixer": tsmixer.reshape(-1),
            "patchtst_style": patchtst.reshape(-1),
        }
    )
    frame["model_gap"] = (frame["tsmixer"] - frame["patchtst_style"]).abs()
    summary = {
        "mode": "live_forecast",
        "prediction_ready": True,
        "forecast_origin": latest.isoformat(),
        "input_start": past.index[0].isoformat(),
        "input_end": past.index[-1].isoformat(),
        "forecast_start": forecast_index[0].isoformat(),
        "forecast_end": forecast_index[-1].isoformat(),
        "generated_at": pd.Timestamp.now(tz=HK_TZ).isoformat(),
        "input_hours": LOOKBACK,
        "forecast_hours": HORIZON,
        "station_count": len(stations),
        "forecast_rows": len(frame),
        "models": ["TSMixer", "PatchTST-style"],
        "policy": "No imputation; forecast generated only from a complete 72 x 17 input window.",
    }

    csv_tmp = CSV_PATH.with_name(CSV_PATH.name + ".tmp")
    frame.to_csv(csv_tmp, index=False, encoding="utf-8-sig")
    csv_tmp.replace(CSV_PATH)
    write_text_atomic(
        JSON_PATH,
        json.dumps(
            {"summary": summary, "records": frame.round(4).to_dict(orient="records")},
            ensure_ascii=False,
            indent=2,
        ),
    )
    write_status(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("LIVE_PREDICTION_OK")


if __name__ == "__main__":
    main()
