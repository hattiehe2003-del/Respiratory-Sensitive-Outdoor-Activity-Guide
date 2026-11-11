"""Inference-only model definitions used by the Streamlit application.

The architectures and preprocessing reproduce notebooks 06 and 08.  The
application loads the frozen weights; it never trains or modifies them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Avoid the Anaconda MKL/OpenMP conflict before importing torch or NumPy.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch
from torch import nn


class MixerBlock(nn.Module):
    def __init__(self, length: int, features: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.time_norm = nn.LayerNorm((length, features))
        self.time_linear = nn.Linear(length, length)
        self.dropout = nn.Dropout(dropout)
        self.feature_norm = nn.LayerNorm((length, features))
        self.feature_mlp = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.time_norm(x).transpose(1, 2)
        x = x + self.dropout(torch.relu(self.time_linear(temporal)).transpose(1, 2))
        return x + self.feature_mlp(self.feature_norm(x))


class TSMixer(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        features: int,
        blocks: int = 2,
        hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[MixerBlock(lookback, features, hidden, dropout) for _ in range(blocks)]
        )
        self.projection = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.blocks(x).transpose(1, 2)).transpose(1, 2)


class PatchTSTSmall(nn.Module):
    """The fixed simplified PatchTST-style comparison model from notebook 08."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        patch_length: int = 12,
        stride: int = 12,
        d_model: int = 16,
        heads: int = 4,
        feedforward: int = 32,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_length
        self.stride = stride
        self.patches = (lookback - patch_length) // stride + 1
        self.embed = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.empty(1, self.patches, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(layers)
            ]
        )
        self.head = nn.Linear(self.patches * d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        mean = x.mean(dim=1, keepdim=True).detach()
        scale = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        z = ((x - mean) / scale).transpose(1, 2).unfold(
            -1, self.patch_len, self.stride
        )
        z = z.reshape(batch * channels, self.patches, self.patch_len)
        z = self.dropout(self.embed(z) + self.position)
        for layer in self.layers:
            z = layer(z)
        out = self.head(z.reshape(batch * channels, -1))
        out = out.reshape(batch, channels, -1).transpose(1, 2)
        return out * scale + mean


@dataclass(frozen=True)
class LoadedModel:
    name: str
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray


class ForecastBundle:
    """Frozen two-model inference bundle with shared station ordering."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.tsmixer_dir = (
            self.root / "data/processed/tsmixer/20261111_010719_491327"
        )
        self.patch_dir = (
            self.root / "data/processed/patchtst/20261111_102259_281485"
        )
        self.tsmixer_config = self._read_json(self.tsmixer_dir / "model_config.json")
        self.patch_config = self._read_json(self.patch_dir / "model_config.json")
        self.stations = list(self.tsmixer_config["stations"])
        self.lookback = int(self.tsmixer_config["lookback"])
        self.horizon = int(self.tsmixer_config["horizon"])
        if self.patch_config["stations"] != self.stations:
            raise ValueError("The frozen models use different station ordering")
        if (self.lookback, self.horizon) != (
            int(self.patch_config["lookback"]),
            int(self.patch_config["horizon"]),
        ):
            raise ValueError("The frozen models use incompatible time windows")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.tsmixer = self._load_tsmixer()
        self.patch = self._load_patch()

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_state(path: Path) -> dict[str, torch.Tensor]:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions before weights_only was added
            return torch.load(path, map_location="cpu")

    def _scaler(self, directory: Path) -> tuple[np.ndarray, np.ndarray]:
        scaler = np.load(directory / "scaler.npz")
        if scaler["stations"].tolist() != self.stations:
            raise ValueError(f"Station ordering mismatch in {directory / 'scaler.npz'}")
        return scaler["mean"].astype("float32"), scaler["std"].astype("float32")

    def _load_tsmixer(self) -> LoadedModel:
        cfg = self.tsmixer_config
        model = TSMixer(
            self.lookback,
            self.horizon,
            len(self.stations),
            blocks=int(cfg["blocks"]),
            hidden=int(cfg["hidden"]),
            dropout=float(cfg["dropout"]),
        )
        model.load_state_dict(self._load_state(self.tsmixer_dir / "best_model_state.pt"))
        model.eval()
        mean, std = self._scaler(self.tsmixer_dir)
        return LoadedModel("TSMixer", model, mean, std)

    def _load_patch(self) -> LoadedModel:
        cfg = self.patch_config
        model = PatchTSTSmall(
            self.lookback,
            self.horizon,
            patch_length=int(cfg["patch_length"]),
            stride=int(cfg["stride"]),
            d_model=int(cfg["d_model"]),
            heads=int(cfg["heads"]),
            feedforward=int(cfg["feedforward"]),
            layers=int(cfg["layers"]),
            dropout=float(cfg["dropout"]),
        )
        model.load_state_dict(self._load_state(self.patch_dir / "best_model_state.pt"))
        model.eval()
        mean, std = self._scaler(self.patch_dir)
        return LoadedModel("PatchTST-style", model, mean, std)

    def _predict_one(self, loaded: LoadedModel, window: np.ndarray) -> np.ndarray:
        scaled = (window.astype("float32") - loaded.mean) / loaded.std
        x = torch.from_numpy(scaled[None, ...])
        with torch.inference_mode():
            prediction = loaded.model(x).cpu().numpy()[0]
        return prediction * loaded.std + loaded.mean

    def predict(self, window: np.ndarray) -> dict[str, np.ndarray]:
        expected = (self.lookback, len(self.stations))
        if window.shape != expected:
            raise ValueError(f"Expected input shape {expected}; received {window.shape}")
        if not np.isfinite(window).all():
            raise ValueError("The 72-hour joint input window contains missing values")
        forecasts = {
            "TSMixer": self._predict_one(self.tsmixer, window),
            "PatchTST-style": self._predict_one(self.patch, window),
        }
        for name, values in forecasts.items():
            if values.shape != (self.horizon, len(self.stations)):
                raise RuntimeError(f"Unexpected {name} output shape: {values.shape}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"{name} produced a non-finite forecast")
        return forecasts

