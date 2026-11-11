"""MATH7005 historical-replay AQHI forecasting dashboard."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_replay, load_stations, replay_window, valid_origins
from src.health_advice import EPD_ADVICE_URL, EPD_AQHI_URL, activity_advice, risk_band, risk_colour
from src.models import ForecastBundle


ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="HK Air Quality Outlook",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#17223b; --muted:#61708a; --sky:#eaf6f6; --teal:#0f8b8d; --coral:#ee6c4d; }
    .stApp { background: linear-gradient(145deg, #f8fbfd 0%, #eef7f5 58%, #f9f5ef 100%); color: var(--ink); }
    [data-testid="stSidebar"] { background: #10243e; }
    [data-testid="stSidebar"] * { color: #f6fbff; }
    .hero { padding: 1.35rem 1.55rem; border-radius: 22px; background: linear-gradient(120deg,#10243e,#176b77); color:white; box-shadow:0 18px 50px rgba(16,36,62,.16); margin-bottom:1rem; }
    .hero h1 { font-size:2.25rem; margin:.15rem 0 .4rem; letter-spacing:-.04em; }
    .hero p { color:#d7edf0; margin:0; max-width:850px; }
    .eyebrow { font-size:.75rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; color:#8bd6d2; }
    .mode-chip { display:inline-block; margin-top:.8rem; padding:.32rem .7rem; border-radius:999px; background:#f4b942; color:#17223b; font-size:.78rem; font-weight:800; }
    [data-testid="stMetric"] { background:rgba(255,255,255,.76); border:1px solid rgba(15,139,141,.12); padding:1rem; border-radius:16px; box-shadow:0 8px 24px rgba(16,36,62,.05); }
    .advice { background:rgba(255,255,255,.82); border-left:5px solid #0f8b8d; padding:1rem 1.15rem; border-radius:8px 16px 16px 8px; }
    .fineprint { color:#61708a; font-size:.82rem; }
    .risk-pill { display:inline-block; padding:.32rem .7rem; border-radius:999px; color:white; font-weight:800; }
    div[data-testid="stPlotlyChart"] { background:rgba(255,255,255,.72); border-radius:18px; padding:.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the audited replay data…")
def cached_data():
    history, panel, stations = load_replay(ROOT)
    metadata = load_stations(ROOT, stations)
    return history, panel, stations, metadata


@st.cache_resource(show_spinner="Loading frozen neural-network weights…")
def cached_models():
    return ForecastBundle(ROOT)


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">MATH7005 · Decision-support prototype</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <span class="mode-chip">Historical replay · Not a live forecast</span>
        </section>
        """,
        unsafe_allow_html=True,
    )


def model_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Previous-day baseline", 0.6219, 0.9702, "Locked test"],
            ["TSMixer", 0.5383, 0.7509, "Primary · locked test"],
            ["PatchTST-style", 0.5514, 0.7828, "Post-hoc comparison"],
        ],
        columns=["Model", "MAE", "RMSE", "Evaluation status"],
    )


history, panel, station_order, metadata = cached_data()
bundle = cached_models()
origins = valid_origins(panel, bundle.lookback, bundle.horizon)
if origins.empty:
    st.error("No complete historical replay window is available.")
    st.stop()

st.sidebar.markdown("## HK Air Quality Outlook")
st.sidebar.caption("A transparent 24-hour AQHI forecasting demonstration")
page = st.sidebar.radio("Navigate", ["Network overview", "Station forecast", "Model & data"])
st.sidebar.markdown("---")
st.sidebar.caption("Scope: 17 monitoring stations · AQHI history only · Hong Kong time")

if page == "Network overview":
    hero(
        "Air quality, with tomorrow in view.",
        "Explore a reproducible historical replay of Hong Kong AQHI observations and the frozen TSMixer forecasting system.",
    )
    latest_time = panel.dropna(how="all").index[-1]
    latest = panel.loc[latest_time].rename("AQHI").reset_index()
    snapshot = metadata.merge(latest, on="station_id", how="left")
    cols = st.columns(4)
    cols[0].metric("Modelled stations", len(station_order))
    cols[1].metric("Historical coverage", "2020–2026")
    cols[2].metric("Forecast horizon", "24 hours")
    cols[3].metric("Primary test MAE", "0.538 AQHI")

    st.subheader("Latest recorded snapshot in the replay dataset")
    st.caption(f"Recorded at {latest_time.strftime('%d %b %Y, %H:%M HKT')} · this is archived data, not a live reading.")
    fig = px.scatter(
        snapshot,
        x="longitude",
        y="latitude",
        color="AQHI",
        size=np.repeat(14, len(snapshot)),
        hover_name="station_id",
        hover_data={"station_type": True, "longitude": ":.4f", "latitude": ":.4f", "AQHI": ":.1f"},
        color_continuous_scale=["#2eb67d", "#f3b61f", "#f27d3d", "#df3f5b"],
        range_color=[1, 10],
        labels={"longitude": "Longitude", "latitude": "Latitude"},
        height=470,
    )
    fig.update_traces(marker={"line": {"color": "white", "width": 1.5}})
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("What the system does")
        st.markdown(
            """
            1. Takes a complete **72-hour, 17-station AQHI window**.
            2. Produces the next **24 hourly AQHI estimates** with TSMixer.
            3. Runs a fixed PatchTST-style model as a comparison signal.
            4. Reports model disagreement and risk-band activity guidance.
            """
        )
    with right:
        st.subheader("Evidence at a glance")
        st.dataframe(model_metrics(), hide_index=True, use_container_width=True)

elif page == "Station forecast":
    hero(
        "Replay a 24-hour station forecast.",
        "Select a station and a valid forecast origin. The chart shows information that would have been available at that time, followed by the archived outcome.",
    )
    options = origins[::6].union(pd.DatetimeIndex([origins[-1]])).sort_values(ascending=False)
    controls = st.columns([1, 1.35])
    station = controls[0].selectbox("Monitoring station", station_order, index=0)
    origin = controls[1].selectbox(
        "Forecast origin",
        options,
        format_func=lambda value: value.strftime("%d %b %Y, %H:%M HKT"),
    )
    past, actual = replay_window(panel, origin, bundle.lookback, bundle.horizon)
    predictions = bundle.predict(past.to_numpy(dtype="float32"))
    station_idx = station_order.index(station)
    mixer = predictions["TSMixer"][:, station_idx]
    patch = predictions["PatchTST-style"][:, station_idx]
    observed = actual[station].to_numpy()
    disagreement = np.abs(mixer - patch)
    peak = float(mixer.max())
    peak_band = risk_band(peak)
    agreement_label = "Close" if disagreement.mean() <= 0.5 else "Mixed" if disagreement.mean() <= 1 else "Wide"

    metrics = st.columns(4)
    metrics[0].metric("Latest observed", f"{past[station].iloc[-1]:.1f}")
    metrics[1].metric("Next-hour TSMixer", f"{mixer[0]:.1f}")
    metrics[2].metric("Predicted 24h peak", f"{peak:.1f}", peak_band)
    metrics[3].metric("Mean model gap", f"{disagreement.mean():.2f}", agreement_label)

    fig = go.Figure()
    fig.add_hrect(y0=1, y1=3, fillcolor="#2eb67d", opacity=0.08, line_width=0)
    fig.add_hrect(y0=3, y1=6, fillcolor="#f3b61f", opacity=0.08, line_width=0)
    fig.add_hrect(y0=6, y1=7, fillcolor="#f27d3d", opacity=0.08, line_width=0)
    fig.add_hrect(y0=7, y1=10.5, fillcolor="#df3f5b", opacity=0.07, line_width=0)
    fig.add_trace(go.Scatter(x=past.index, y=past[station], name="Observed input", line=dict(color="#17223b", width=2)))
    fig.add_trace(go.Scatter(x=actual.index, y=observed, name="Archived outcome", line=dict(color="#8091a7", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=actual.index, y=mixer, name="TSMixer forecast", line=dict(color="#0f8b8d", width=3)))
    fig.add_trace(go.Scatter(x=actual.index, y=patch, name="PatchTST-style", line=dict(color="#ee6c4d", width=2)))
    fig.add_vline(x=origin.timestamp() * 1000, line_dash="dash", line_color="#61708a")
    fig.update_layout(
        height=510,
        margin=dict(l=10, r=10, t=35, b=10),
        title=f"{station}: 72 hours observed → 24 hours forecast",
        yaxis_title="Capped AQHI",
        xaxis_title="Hong Kong time",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.5)",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    advice = activity_advice(peak_band)
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"""
            <div class="advice">
              <b>Activity guide · predicted peak</b><br>
              <span class="risk-pill" style="background:{risk_colour(peak_band)}">{peak_band}</span>
              <p><b>General public:</b> {advice['general']}</p>
              <p><b>Sensitive people:</b> {advice['sensitive']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Adapted from Hong Kong EPD precautionary actions. Seek medical advice if in doubt or unwell.")
    with right:
        st.subheader("Hourly forecast table")
        forecast_table = pd.DataFrame(
            {
                "Time (HKT)": actual.index.strftime("%d %b %H:%M"),
                "TSMixer": mixer.round(2),
                "PatchTST-style": patch.round(2),
                "Model gap": disagreement.round(2),
                "Archived AQHI": observed.round(1),
            }
        )
        st.dataframe(forecast_table, hide_index=True, use_container_width=True, height=285)

    st.info(
        "Model agreement is a diagnostic comparison, not a calibrated confidence interval. "
        "The archived outcome is shown only because this page is a historical replay."
    )

else:
    hero(
        "Transparent by design.",
        "The prototype keeps data provenance, model evaluation status and deployment limitations visible rather than presenting a black-box forecast.",
    )
    st.subheader("Data-to-decision pipeline")
    st.markdown(
        "**Official monthly AQHI records** → audited station-hour panel → complete 72/24 windows → frozen neural-network inference → risk-band activity guide"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Forecast design")
        st.markdown(
            """
            - **Target:** capped AQHI; original `10+` is encoded as 10 while its source marker is retained.
            - **Input:** previous 72 hours from 17 stations.
            - **Output:** next 24 hours for all 17 stations.
            - **Missingness:** no imputation; incomplete joint windows are excluded.
            - **Known gap:** 25 Aug–1 Sep 2023 is retained and excluded from training windows.
            - **External variables:** deliberately excluded from this project scope.
            """
        )
    with col2:
        st.markdown("#### Model roles")
        st.markdown(
            """
            - **TSMixer:** primary model, selected before locked test evaluation.
            - **PatchTST-style:** simplified, post-hoc comparison; not a full official PatchTST reproduction.
            - **Previous day:** transparent baseline.
            - The two neural forecasts are **not averaged**, because an ensemble improvement has not been validated.
            """
        )

    st.subheader("Held-out performance")
    metrics = model_metrics()
    fig = px.bar(
        metrics,
        x="Model",
        y=["MAE", "RMSE"],
        barmode="group",
        color_discrete_sequence=["#0f8b8d", "#ee6c4d"],
        height=390,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(metrics, hide_index=True, use_container_width=True)

    st.warning(
        "High-risk events are rare and both neural models underpredict extreme observations. "
        "The system cannot distinguish a future official 10+ category from a numeric 10, so it must not be used as an emergency-warning service."
    )
    st.markdown(
        f"Official references: [What is AQHI?]({EPD_AQHI_URL}) · "
        f"[Recommended precautionary actions]({EPD_ADVICE_URL}) · "
        "[TSMixer paper](https://arxiv.org/abs/2303.06053) · "
        "[PatchTST paper](https://arxiv.org/abs/2211.14730)"
    )
    st.markdown(
        "<p class='fineprint'>Educational decision-support prototype for MATH7005. "
        "It is not an official EPD forecast, a medical device or a substitute for current public-health advice.</p>",
        unsafe_allow_html=True,
    )

