"""Fast end-to-end verification for the historical-replay inference path."""
from pathlib import Path

import numpy as np

from src.data_loader import build_app_data, load_replay, replay_window, valid_origins
from src.models import ForecastBundle


ROOT = Path(__file__).resolve().parent
build_app_data(ROOT)
_, panel, stations = load_replay(ROOT)
bundle = ForecastBundle(ROOT)
assert stations == bundle.stations
origins = valid_origins(panel, bundle.lookback, bundle.horizon)
assert len(origins) > 0
origin = origins[-1]
past, actual = replay_window(panel, origin, bundle.lookback, bundle.horizon)
forecasts = bundle.predict(past.to_numpy(dtype="float32"))
assert actual.shape == (24, 17)
for name, values in forecasts.items():
    assert values.shape == (24, 17)
    assert np.isfinite(values).all()
    print(f"{name}: shape={values.shape}, range=({values.min():.3f}, {values.max():.3f})")
print(f"verified_origin={origin}")
print(f"valid_origins={len(origins)}")
print("APP_PIPELINE_OK")

