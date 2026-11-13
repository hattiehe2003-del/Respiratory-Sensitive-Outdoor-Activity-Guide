"""MATH7005 historical-replay AQHI forecasting dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_replay, load_stations, replay_window, valid_origins
from src.health_advice import EPD_ADVICE_URL, EPD_AQHI_URL, activity_advice, risk_band, risk_colour
from src.live_forecast import load_live_forecast
from src.models import ForecastBundle


ROOT = Path(__file__).resolve().parent

STATION_LABELS = {
    "Causeway Bay": "Causeway Bay 铜锣湾",
    "Central": "Central 中环",
    "Central/Western": "Central/Western 中西区",
    "Eastern": "Eastern 东区",
    "Kwai Chung": "Kwai Chung 葵涌",
    "Kwun Tong": "Kwun Tong 观塘",
    "Mong Kok": "Mong Kok 旺角",
    "North": "North 北区",
    "Sha Tin": "Sha Tin 沙田",
    "Sham Shui Po": "Sham Shui Po 深水埗",
    "Southern": "Southern 南区",
    "Tai Po": "Tai Po 大埔",
    "Tseung Kwan O": "Tseung Kwan O 将军澳",
    "Tsuen Wan": "Tsuen Wan 荃湾",
    "Tuen Mun": "Tuen Mun 屯门",
    "Tung Chung": "Tung Chung 东涌",
    "Yuen Long": "Yuen Long 元朗",
}

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
    [data-testid="stMetricLabel"] { color:#52627a !important; }
    [data-testid="stMetricValue"] { color:#17223b !important; }
    [data-testid="stMetricDelta"] { color:#35566f !important; }
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] label p { color:#17223b !important; }
    div[data-testid="stPlotlyChart"] { background:rgba(255,255,255,.72); border-radius:18px; padding:.35rem; }
    div[data-testid="stPlotlyChart"] svg text { fill:#17223b !important; }
    div[data-testid="stPlotlyChart"] .modebar { background:rgba(255,255,255,.94) !important; }
    div[data-testid="stPlotlyChart"] .modebar-btn path { fill:#17223b !important; }
    @media (max-width: 768px) {
      .hero { padding:1rem; border-radius:16px; }
      .hero h1 { font-size:1.7rem; }
      .hero p { font-size:.92rem; }
      [data-testid="stMetric"] { padding:.75rem; }
    }
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


@st.cache_data(show_spinner=False)
def cached_demo_output():
    path = ROOT / "data" / "processed" / "demo" / "forecast_24h.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["records"]), payload["summary"]


def cached_live_output():
    snapshot = load_live_forecast(ROOT)
    return (
        snapshot.records,
        snapshot.summary,
        snapshot.status,
        snapshot.state,
        snapshot.message,
    )

def hero(title: str, subtitle: str, live: bool = False):
    mode = "Live 24-hour forecast" if live else "Historical replay · Not a live forecast"
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">MATH7005 · Decision-support prototype</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <span class="mode-chip">{mode}</span>
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


def station_display(station: str) -> str:
    return STATION_LABELS.get(station, station)


def best_activity_window(times: pd.DatetimeIndex, values: np.ndarray, hours: int = 3):
    """Return the lowest predicted consecutive window without implying certainty."""
    numeric = np.asarray(values, dtype=float)
    if len(numeric) < hours or not np.isfinite(numeric).all():
        raise ValueError("A complete forecast is required to rank activity windows")
    means = np.convolve(numeric, np.ones(hours) / hours, mode="valid")
    start_index = int(np.argmin(means))
    start = times[start_index]
    end = times[start_index + hours - 1] + pd.Timedelta(hours=1)
    window = numeric[start_index : start_index + hours]
    return start, end, float(window.mean()), risk_band(float(window.max()))


def apply_readable_plot_theme(fig, plot_bgcolor: str = "rgba(255,255,255,.86)"):
    """Keep Plotly labels readable when the browser prefers a dark theme."""
    ink = "#17223b"
    grid = "rgba(97,112,138,.20)"
    fig.update_layout(
        template="plotly_white",
        font=dict(color=ink, size=14),
        title_font=dict(color=ink, size=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bgcolor,
        legend_font_color=ink,
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#61708a",
            font_color=ink,
        ),
    )
    fig.update_xaxes(
        color=ink,
        gridcolor=grid,
        linecolor="#61708a",
        zerolinecolor=grid,
    )
    fig.update_yaxes(
        color=ink,
        gridcolor=grid,
        linecolor="#61708a",
        zerolinecolor=grid,
    )
    return fig


history, panel, station_order, metadata = cached_data()
bundle = cached_models()
demo_forecast, demo_summary = cached_demo_output()
live_forecast, live_summary, live_status, live_state, live_message = cached_live_output()
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
        "View the latest official-data forecast when ready, with the audited historical replay retained for evidence.",
        live=live_summary is not None and live_state in {"live", "last_valid"},
    )
    latest_time = panel.dropna(how="all").index[-1]
    latest = panel.loc[latest_time].rename("AQHI").reset_index()
    snapshot = metadata.merge(latest, on="station_id", how="left")
    cols = st.columns(4)
    cols[0].metric("Modelled stations", len(station_order))
    cols[1].metric("Historical coverage", "2020–2026")
    cols[2].metric("Forecast horizon", "24 hours")
    cols[3].metric("Primary test MAE", "0.538 AQHI")

    if live_summary is not None:
        live_notice = (
            st.success if live_state == "live" else
            st.warning if live_state == "last_valid" else st.error
        )
        live_notice(
            "Latest live 24-hour forecast loaded: "
            f"{live_summary['forecast_rows']} rows, "
            f"{live_summary['station_count']} stations, "
            f"origin {live_summary['forecast_origin']}."
        )
        with st.expander("Preview latest live 24-hour forecast"):
            st.dataframe(live_forecast.head(24), hide_index=True, use_container_width=True)
    elif live_status is not None:
        available = int(live_status.get("available_complete_hours", 0))
        required = int(live_status.get("required_complete_hours", 72))
        st.info(
            f"Live pipeline is collecting a complete input window: {available}/{required} hours ready. "
            "It will forecast automatically once all 17 stations have 72 consecutive hours."
        )

    if demo_summary is not None:
        st.success(
            "Latest generated demo output loaded: "
            f"{demo_summary['forecast_rows']} rows, "
            f"{demo_summary['station_count']} stations, "
            f"origin {demo_summary['forecast_origin']}."
        )
        with st.expander("Preview historical 24-hour forecast demo"):
            st.dataframe(demo_forecast.head(24), hide_index=True, use_container_width=True)

    st.subheader("Latest recorded snapshot in the replay dataset")
    st.caption(f"Recorded at {latest_time.strftime('%d %b %Y, %H:%M HKT')} · this is archived data, not a live reading.")
    fig = px.scatter(
        snapshot,
        x="longitude",
        y="latitude",
        color="AQHI",
        hover_name="station_id",
        hover_data={"station_type": True, "longitude": ":.4f", "latitude": ":.4f", "AQHI": ":.1f"},
        color_continuous_scale=["#2eb67d", "#f3b61f", "#f27d3d", "#df3f5b"],
        range_color=[1, 10],
        labels={"longitude": "Longitude", "latitude": "Latitude"},
        height=470,
    )
    fig.update_traces(marker={"size": 18, "line": {"color": "white", "width": 1.5}})
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    apply_readable_plot_theme(fig)
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color="#17223b"),
            title_font=dict(color="#17223b"),
            bgcolor="rgba(255,255,255,.88)",
        )
    )
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

    live_columns = {"timestamp", "station_id", "tsmixer", "patchtst_style"}
    live_station_mode = (
        live_forecast is not None
        and not live_forecast.empty
        and live_summary is not None
        and bool(live_summary.get("prediction_ready"))
        and live_columns.issubset(live_forecast.columns)
    )
    if live_station_mode:
        hero(
            "Latest 24-hour station forecast.",
            "Choose a monitoring station to see the newest model output generated from the rolling government AQHI feed.",
            live=True,
        )
        if live_state == "live":
            st.success("Forecast service is current and ready for activity planning.")
        elif live_state == "last_valid":
            st.warning(live_message)
        elif live_state == "expired":
            st.error(live_message)

        st.subheader("Plan an outdoor activity")
        st.caption(
            "Choose the monitoring area nearest to your planned activity. "
            "The recommendation is based on predicted AQHI, not your exact street location."
        )
        available_stations = [
            name for name in station_order
            if name in set(live_forecast["station_id"].astype(str))
        ]
        if not available_stations:
            st.error("The newest output does not contain a recognised monitoring station.")
            st.stop()

        station = st.selectbox(
            "Activity area / monitoring station",
            available_stations,
            index=0,
            format_func=station_display,
        )
        station_live = live_forecast.loc[
            live_forecast["station_id"].astype(str).eq(station)
        ].copy()
        station_live["timestamp"] = pd.to_datetime(
            station_live["timestamp"], errors="coerce", utc=True
        ).dt.tz_convert("Asia/Hong_Kong")
        station_live = station_live.dropna(subset=["timestamp"]).sort_values("timestamp")
        if len(station_live) != 24:
            st.error(
                f"The newest output contains {len(station_live)} hourly rows for {station}; "
                "24 are required. The page will update after the next valid model run."
            )
            st.stop()

        forecast_time = pd.DatetimeIndex(station_live["timestamp"])
        mixer = pd.to_numeric(station_live["tsmixer"], errors="coerce").to_numpy()
        patch = pd.to_numeric(station_live["patchtst_style"], errors="coerce").to_numpy()
        disagreement = np.abs(mixer - patch)
        peak = float(np.nanmax(mixer))
        peak_band = risk_band(peak)
        mean_gap = float(np.nanmean(disagreement))
        agreement_label = "Close" if mean_gap <= 0.5 else "Mixed" if mean_gap <= 1 else "Wide"
        hourly_risk = [risk_band(float(value)) for value in mixer]
        best_start, best_end, best_mean, best_band = best_activity_window(
            forecast_time, mixer
        )
        best_label = f"{best_start:%H:%M}–{best_end:%H:%M}"
        origin = pd.to_datetime(
            live_summary.get("forecast_origin"), errors="coerce", utc=True
        )
        origin_label = (
            origin.tz_convert("Asia/Hong_Kong").strftime("%d %b %H:%M")
            if not pd.isna(origin) else "Latest run"
        )

        st.caption(f"Forecast origin: {origin_label} HKT · next 24 hours")
        metrics = st.columns(4)
        metrics[0].metric("Next-hour AQHI", f"{mixer[0]:.1f}", hourly_risk[0])
        metrics[1].metric("Lower-exposure 3h window", best_label, best_band)
        metrics[2].metric("Predicted 24h peak", f"{peak:.1f}", peak_band)
        metrics[3].metric("Mean model gap", f"{mean_gap:.2f}", agreement_label)

        fig = go.Figure()
        for low, high, colour in [
            (1, 3, "#2eb67d"), (3, 6, "#f3b61f"),
            (6, 7, "#f27d3d"), (7, 10.5, "#df3f5b")
        ]:
            fig.add_hrect(y0=low, y1=high, fillcolor=colour, opacity=0.08, line_width=0)
        fig.add_vrect(
            x0=best_start, x1=best_end, fillcolor="#0f8b8d", opacity=0.10,
            line_width=0, annotation_text="Lower-exposure window",
            annotation_position="top left",
        )
        fig.add_trace(go.Scatter(
            x=forecast_time, y=mixer, name="TSMixer forecast",
            line=dict(color="#0f8b8d", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_time, y=patch, name="PatchTST-style comparison",
            line=dict(color="#ee6c4d", width=2)
        ))
        fig.update_layout(
            height=510, margin=dict(l=10, r=10, t=35, b=10),
            title=f"{station_display(station)}: newest 24-hour forecast",
            yaxis_title="Capped AQHI", xaxis_title="Hong Kong time",
            hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.5)",
            legend=dict(orientation="h", y=1.08, x=0),
        )
        apply_readable_plot_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        advice = activity_advice(peak_band)
        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                f"""
                <div class="advice">
                  <b>Activity guide · predicted peak</b><br>
                  <span class="risk-pill" style="background:{risk_colour(peak_band)}">{peak_band}</span>
                  <p><b>Lower-exposure window:</b> {best_label} HKT · mean AQHI {best_mean:.1f} ({best_band})</p>
                  <p><b>General public:</b> {advice['general']}</p>
                  <p><b>Sensitive people:</b> {advice['sensitive']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Adapted from Hong Kong EPD precautionary actions. "
                "Seek medical advice if in doubt or unwell."
            )
        with right:
            st.subheader("Hourly forecast table")
            forecast_table = pd.DataFrame({
                "Time (HKT)": forecast_time.strftime("%d %b %H:%M"),
                "TSMixer": np.round(mixer, 2),
                "PatchTST-style": np.round(patch, 2),
                "Model gap": np.round(disagreement, 2),
                "Risk": hourly_risk,
            })
            st.dataframe(
                forecast_table, hide_index=True, use_container_width=True, height=285
            )

        generated = pd.to_datetime(
            live_summary.get("generated_at"), errors="coerce", utc=True
        )
        generated_label = (
            generated.tz_convert("Asia/Hong_Kong").strftime("%d %b %Y, %H:%M HKT")
            if not pd.isna(generated) else "the latest successful workflow run"
        )
        st.info(
            "TSMixer is the primary forecast. PatchTST-style is a diagnostic "
            "comparison only; the two outputs are not averaged."
        )
        st.caption(
            f"Generated {generated_label}. Educational model forecast; "
            "not an official EPD forecast or emergency-warning service."
        )
        st.stop()

    hero(
        "Replay a 24-hour station forecast.",
        "Select a station and a valid forecast origin. The chart shows information that would have been available at that time, followed by the archived outcome.",
    )
    if live_status is not None:
        available_hours = int(live_status.get("available_complete_hours", 0))
        required_hours = int(live_status.get("required_complete_hours", 72))
        progress = min(1.0, available_hours / max(required_hours, 1))
        st.progress(progress)
        st.info(
            f"Live forecast preparation: {available_hours}/{required_hours} consecutive complete hours. "
            "Until the input reaches 72 hours, the historical replay below remains available "
            "without being presented as a current forecast."
        )
    options = origins[::6].union(pd.DatetimeIndex([origins[-1]])).sort_values(ascending=False)
    controls = st.columns([1, 1.35])
    station = controls[0].selectbox(
        "Activity area / monitoring station",
        station_order,
        index=0,
        format_func=station_display,
    )
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
        title=f"{station_display(station)}: 72 hours observed → 24 hours forecast",
        yaxis_title="Capped AQHI",
        xaxis_title="Hong Kong time",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.5)",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    apply_readable_plot_theme(fig)
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
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    apply_readable_plot_theme(fig)
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
