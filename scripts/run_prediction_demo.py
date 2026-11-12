"""Run one auditable 72-hour to 24-hour forecasting demo.

The script uses archived observations so that the 24-hour forecast can be
compared with known outcomes. It writes website-ready CSV/JSON files and runs
twice with adjacent origins to verify that a newer forecast replaces the old.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_replay, replay_window, valid_origins
from src.models import ForecastBundle


OUTPUT_DIR = ROOT / "data" / "processed" / "demo"
CSV_PATH = OUTPUT_DIR / "forecast_24h.csv"
JSON_PATH = OUTPUT_DIR / "forecast_24h.json"
SUMMARY_PATH = OUTPUT_DIR / "demo_summary.json"


def error_metrics(prediction: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    error = prediction - observed
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def build_forecast(
    origin: pd.Timestamp,
    panel: pd.DataFrame,
    stations: list[str],
    bundle: ForecastBundle,
) -> tuple[pd.DataFrame, dict]:
    past, actual = replay_window(panel, origin, bundle.lookback, bundle.horizon)
    predictions = bundle.predict(past.to_numpy(dtype="float32"))
    observed = actual.to_numpy(dtype="float32")
    tsmixer = predictions["TSMixer"]
    patchtst = predictions["PatchTST-style"]

    assert past.shape == (72, 17)
    assert actual.shape == (24, 17)
    assert tsmixer.shape == observed.shape == patchtst.shape
    assert np.isfinite(tsmixer).all() and np.isfinite(patchtst).all()

    frame = pd.DataFrame(
        {
            "forecast_origin": [origin.isoformat()] * (24 * len(stations)),
            "timestamp": actual.index.repeat(len(stations)).astype(str),
            "station_id": stations * len(actual),
            "tsmixer": tsmixer.reshape(-1),
            "patchtst_style": patchtst.reshape(-1),
            "observed": observed.reshape(-1),
        }
    )
    frame["model_gap"] = (frame["tsmixer"] - frame["patchtst_style"]).abs()
    frame["tsmixer_abs_error"] = (frame["tsmixer"] - frame["observed"]).abs()
    frame["patchtst_abs_error"] = (frame["patchtst_style"] - frame["observed"]).abs()

    summary = {
        "mode": "historical_replay_demo",
        "forecast_origin": origin.isoformat(),
        "input_start": past.index[0].isoformat(),
        "input_end": past.index[-1].isoformat(),
        "forecast_start": actual.index[0].isoformat(),
        "forecast_end": actual.index[-1].isoformat(),
        "input_hours": int(len(past)),
        "forecast_hours": int(len(actual)),
        "station_count": int(len(stations)),
        "forecast_rows": int(len(frame)),
        "metrics": {
            "TSMixer": error_metrics(tsmixer, observed),
            "PatchTST-style": error_metrics(patchtst, observed),
        },
        "note": "Archived outcome is included for demo evaluation only; a live forecast will not have observed values yet.",
    }
    return frame, summary


def write_outputs(frame: pd.DataFrame, summary: dict) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    records = frame.round(4).to_dict(orient="records")
    JSON_PATH.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    digest = hashlib.sha256()
    digest.update(CSV_PATH.read_bytes())
    digest.update(JSON_PATH.read_bytes())
    return digest.hexdigest()


def main() -> None:
    _, panel, stations = load_replay(ROOT)
    bundle = ForecastBundle(ROOT)
    assert stations == bundle.stations
    origins = valid_origins(panel, bundle.lookback, bundle.horizon)
    if len(origins) < 2:
        raise RuntimeError("At least two valid historical origins are required")

    first_origin = origins[-2]
    first_frame, first_summary = build_forecast(first_origin, panel, stations, bundle)
    first_hash = write_outputs(first_frame, first_summary)

    latest_origin = origins[-1]
    latest_frame, latest_summary = build_forecast(latest_origin, panel, stations, bundle)
    latest_hash = write_outputs(latest_frame, latest_summary)

    if first_hash == latest_hash:
        raise AssertionError("The simulated update did not replace the previous output")
    saved = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if saved["summary"]["forecast_origin"] != latest_origin.isoformat():
        raise AssertionError("The final JSON does not contain the latest demo origin")

    result = {
        "demo_ok": True,
        "first_origin": first_origin.isoformat(),
        "latest_origin": latest_origin.isoformat(),
        "update_overwrote_previous": True,
        "final_metrics": latest_summary["metrics"],
        "outputs": [str(CSV_PATH), str(JSON_PATH), str(SUMMARY_PATH)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("PREDICTION_DEMO_OK")


if __name__ == "__main__":
    main()
