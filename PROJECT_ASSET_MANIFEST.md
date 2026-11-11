# MATH7005 website asset manifest

Checked on 2026-11-11. This file records the frozen inputs that the website is
allowed to use. It does not replace the numbered notebooks, raw data or evaluation
records.

## 1. Audited source data

- Canonical audited history:
  `data/processed/full_history/20261110_170756_651230/history_all_audited.csv.gz`
- Audit summary:
  `data/processed/full_history/20261110_170756_651230/summary.json`
- Station metadata used by the application: `data/processed/stations.csv`
- Coverage: 151 monthly files, 2013-12 to 2026-06, 18 stations.
- Full-history timestamps: 2013-12-30 01:00 to 2026-07-01 00:00 (Asia/Hong_Kong).
- Missing values are not imputed. Windows containing missing station-hours are
  rejected before inference.

## 2. Website replay data

- Compact replay file: `data/processed/app/replay_history_2025_2026.csv.gz`
- Replay manifest: `data/processed/app/replay_manifest.json`
- Purpose: historical replay only; it is not a live EPD forecast.
- Coverage: 223,992 rows from 2024-12-29 01:00 to 2026-07-01 00:00.
- The replay contains the 17 modelled stations. Tap Mun remains available in the
  full history and station metadata, but is not an input/output of the frozen
  forecasting models.

## 3. Primary model: TSMixer

- Frozen directory: `data/processed/tsmixer/20261111_010719_491327`
- Weights: `best_model_state.pt`
- Scaler: `scaler.npz`
- Configuration: `model_config.json`
- Input/output: previous 72 hours across 17 stations -> next 24 hours across the
  same 17 stations.
- Target: capped AQHI; original `10+` observations are encoded as 10 while their
  source marker remains in the audited data.
- Weight SHA-256:
  `9b529817cc8ddde61c28cf5b4d60c16a2b778d3e7e7d88d241771024c8b4bc34`
- Scaler SHA-256:
  `3ebdf7e44d0c41a45dc35bb5ac5d36d37da8146c1c36a035002d5915b1790810`
- The hashes match
  `data/processed/final_test/20261111_011923_374913/evaluation_lock.json`.
- Locked test performance: MAE 0.538345; RMSE 0.750868.

## 4. Auxiliary comparison model: PatchTST-style

- Frozen directory: `data/processed/patchtst/20261111_102259_281485`
- Weights: `best_model_state.pt`
- Scaler: `scaler.npz`
- Configuration: `model_config.json` and `precommitted_config.json`
- Post-hoc test outputs: `posthoc_test/`
- Test performance: MAE 0.551400; RMSE 0.782810.
- This is a simplified PatchTST-style comparator added after the main locked test.
  It must not be described as an independently selected primary model or as an
  uncertainty/confidence estimator.

## 5. Evaluation records

- Baselines: `data/processed/baselines/20261110_181540_048535/`
- Locked TSMixer test: `data/processed/final_test/20261111_011923_374913/`
- PatchTST-style post-hoc test:
  `data/processed/patchtst/20261111_102259_281485/posthoc_test/`
- Test MAE: persistence 0.622452; previous day 0.621884; TSMixer 0.538345;
  PatchTST-style 0.551400.

## 6. Reproducibility record

- `01_data_check.ipynb` through `08_patchtst_comparison.ipynb` are present.
- Every non-empty code cell has an execution count and no saved error output was
  found during this audit.
- Website inference modules load the frozen assets and do not retrain them.

## 7. Application-facing code already present

- Entry point: `app.py`
- Replay preparation/loading: `src/data_loader.py`, `src/build_app_data.py`
- Frozen model definitions and inference: `src/models.py`
- AQHI advice mapping: `src/health_advice.py`
- Pipeline smoke test: `verify_app_pipeline.py`
- Current smoke-test result: `APP_PIPELINE_OK`; both models returned a 24 x 17
  forecast for a valid historical origin.

## 8. Do not use as production inputs

- Earlier timestamped runs under `data/processed/full_history/`,
  `data/processed/tsmixer/` and `data/processed/patchtst/` are retained as an audit
  trail but are not application inputs.
- Files directly under `data/` such as the manually downloaded August 2023 copies
  are not the canonical modelling input.
- The RSS snapshot is not sufficient to create a complete 72-hour live input
  window.
- Do not retrain, rescale, clip outputs or change station order inside the website.

## 9. Readiness decision

The data/model asset layer is ready for website development. Before calling the
website complete, the existing Streamlit prototype still requires functional and
responsive UI verification against the agreed requirements.
