# MATH7005 Outdoor Air Quality Guide

This repository contains the complete audited AQHI workflow, model comparison and
Streamlit decision-support prototype. The website uses a **historical replay** so
that forecasts can be compared with archived outcomes; it is not a live or official
Hong Kong EPD forecast.

## Run the website

From Anaconda Prompt:

```text
cd /d "C:\Users\hxq08259\Desktop\sem1\Math7005"
python -m pip install -r requirements.txt
python -m src.build_app_data
python verify_app_pipeline.py
streamlit run app.py
```

The website provides a network overview, an interactive 72-hour-to-24-hour station
forecast replay, TSMixer/PatchTST-style comparison, activity guidance, model
performance and methodological limitations.

## Reproducible workflow

The eight numbered notebooks remain the analysis record: data audit, full-history
checks, exploratory analysis, prediction design, baselines, TSMixer training, locked
test evaluation and a post-hoc PatchTST-style comparison. Website modules under
`src/` load the frozen weights and never retrain them.

## Boundaries

- Forecast target is capped AQHI; original `10+` is encoded as 10 with its marker retained.
- Missing values are not imputed and incomplete 72/24 windows are rejected.
- TSMixer is the primary model. PatchTST-style is a simplified, post-hoc comparator.
- High-risk observations are rare and are underpredicted; this is not an emergency-warning system.
- No weather, pollutant-concentration or traffic variables are used.
- Current network metadata does not establish historical station locations.

## AI assistance

Codex helped draft ingestion, audit, modelling and application code. The student is
responsible for reviewing, understanding and accurately disclosing this assistance.
