"""
app.py
------
Main Streamlit application for Electricity Consumption Forecasting.

Run:
    streamlit run app.py
"""

# ─── Standard / third-party imports ─────────────────────────────────────────
import os
import io
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib

from weather import get_current_temperature
try:
    from kaggle_loader import load_kaggle_dataset
    _KAGGLE_AVAILABLE = True
except ImportError:
    _KAGGLE_AVAILABLE = False

# ─── Project modules ─────────────────────────────────────────────────────────
from generate_dataset import generate_electricity_dataset
from model_training   import (
    load_and_preprocess,
    train_models,
    predict_consumption,
    invalidate_model_cache,
    load_metrics,
    DATA_PATH,
    MODEL_DIR,
    BEST_PATH,
    METRICS_PATH,
)

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ Electricity Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  – Modern dark-energy aesthetic
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google font ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --border:   #1f2d45;
    --accent:   #f59e0b;   /* amber */
    --accent2:  #3b82f6;   /* blue  */
    --good:     #10b981;
    --danger:   #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --card:     #131c2e;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main content padding ── */
.block-container { padding: 2rem 3rem 4rem 3rem; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f1f3d 0%, #0a0e1a 60%, #1a0f2e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(245,158,11,.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f59e0b, #fbbf24, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 .5rem 0;
    line-height: 1.15;
}
.hero p { color: var(--muted); font-size: 1.05rem; margin: 0; }
.hero .badge {
    display: inline-block;
    background: rgba(245,158,11,.15);
    color: var(--accent);
    border: 1px solid rgba(245,158,11,.3);
    border-radius: 999px;
    padding: .25rem .85rem;
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .05em;
    margin-bottom: .75rem;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    transition: border-color .2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card .label { color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .06em; margin-bottom: .4rem; }
.metric-card .value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: var(--accent); }
.metric-card .delta { font-size: .82rem; color: var(--muted); margin-top: .2rem; }

/* ── Section title ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
    margin-left: .75rem;
}

/* ── Alert boxes ── */
.alert-danger {
    background: rgba(239,68,68,.1);
    border: 1px solid rgba(239,68,68,.4);
    border-left: 4px solid var(--danger);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    color: #fca5a5;
}
.alert-success {
    background: rgba(16,185,129,.08);
    border: 1px solid rgba(16,185,129,.3);
    border-left: 4px solid var(--good);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    color: #6ee7b7;
}
.alert-info {
    background: rgba(59,130,246,.08);
    border: 1px solid rgba(59,130,246,.3);
    border-left: 4px solid var(--accent2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    color: #93c5fd;
}

/* ── Prediction result box ── */
.prediction-box {
    background: linear-gradient(135deg, #1a2744, #0f1f3d);
    border: 2px solid var(--accent);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(245,158,11,.12);
}
.prediction-box .pred-label { color: var(--muted); font-size: .9rem; letter-spacing: .08em; text-transform: uppercase; }
.prediction-box .pred-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.prediction-box .pred-unit { color: var(--muted); font-size: 1rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 10px;
    padding: .25rem;
    gap: .25rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--border) !important;
    color: var(--accent) !important;
}

/* ── Plotly chart bg ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    background: rgba(16,185,129,.12);
    color: #34d399;
    border: 1px solid rgba(16,185,129,.35);
    border-radius: 8px;
    padding: .3rem .9rem;
    font-size: .85rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME DEFAULTS
# ════════════════════════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,.6)",
    font=dict(color="#94a3b8", family="DM Sans"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.1)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)", showgrid=True, zeroline=False),
)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE HELPERS
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_dataframe():
    return load_and_preprocess(DATA_PATH)

def ensure_models_trained():
    """Train models if not already done; cache metrics in session state."""
    if "metrics" not in st.session_state or st.session_state.metrics is None:
        metrics = load_metrics()
        if metrics is None:
            with st.spinner("🔧  First launch: training models on synthetic data …"):
                df = get_dataframe()
                metrics = train_models(df)
        st.session_state.metrics = metrics


# ════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>Electricity Consumption<br>Forecasting</h1>
    <p>Machine-learning powered predictions using Linear Regression &amp; Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# ENSURE MODELS ARE TRAINED
# ════════════════════════════════════════════════════════════════════════════
ensure_models_trained()
metrics = st.session_state.metrics
df      = get_dataframe()

# Best model info
best_name = metrics.get("best_model_name", "Random Forest")
lr_m      = metrics["Linear Regression"]
rf_m      = metrics["Random Forest"]



# ════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════════════════════
tab_pred, tab_compare, tab_viz, tab_forecast, tab_explorer = st.tabs([
    "⚡ Prediction",
    "📊 Model Comparison",
    "📈 Visualizations",
    "📅 Forecasts",
    "🔍 Data Explorer",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 – PREDICTION
# ────────────────────────────────────────────────────────────────────────────
with tab_pred:

    st.markdown('<div class="section-title">⚙️ Prediction Inputs</div>', unsafe_allow_html=True)

    in1, in2, in3 = st.columns(3)
    with in1:
        selected_date = st.date_input("📅 Date", value=datetime.date.today())
        selected_time = st.time_input("🕐 Time", value=datetime.time(12, 0))
    with in2:
        prev_cons = st.slider(
            "🔋 Previous Hour Consumption (kWh)",
            min_value=10.0, max_value=250.0, value=85.0, step=0.5,
        )
        city_input = st.text_input("🌍 City for Live Temperature", value="Chennai", key="pred_city")
        fetch_btn  = st.button("🌡️ Fetch Temperature", key="pred_fetch")
        if fetch_btn:
            try:
                fetched_temp, city_display = get_current_temperature(city_input)
                st.session_state["fetched_temp"] = fetched_temp
                st.session_state["city_display"]  = city_display
            except Exception as e:
                st.warning(f"Could not fetch temperature: {e}. Using manual value.")
        if "fetched_temp" in st.session_state:
            st.success(f"🌡️ {st.session_state['city_display']}: **{st.session_state['fetched_temp']}°C**")
        temperature = st.slider(
            "Temperature (°C) — override if needed",
            min_value=-10.0, max_value=50.0,
            value=float(st.session_state.get("fetched_temp", 22.0)),
            step=0.5,
        )
    with in3:
        threshold = st.number_input(
            "⚠️ Alert Threshold (kWh)",
            min_value=50.0, max_value=500.0, value=150.0, step=5.0,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Predict Consumption")
        retrain_btn = st.button("🔄 Retrain Models")
        if _KAGGLE_AVAILABLE:
            kaggle_btn = st.button("📦 Load Real Kaggle Dataset & Retrain")
        else:
            kaggle_btn = False
            st.caption("⚠️ Install `kaggle` + add kaggle.json to enable real data")
        if retrain_btn:
            st.session_state["show_retrain_confirm"] = True
        if kaggle_btn:
            st.session_state["show_kaggle_confirm"] = True
        if st.session_state.get("show_kaggle_confirm"):
            confirm_k = st.checkbox(
                "⚠️ Confirm: delete existing data & download real Kaggle dataset (requires internet + kaggle.json)",
                key="kaggle_confirm_cb"
            )
            if confirm_k:
                st.session_state["show_kaggle_confirm"] = False
                if os.path.exists(DATA_PATH):
                    os.remove(DATA_PATH)
                for f in [
                    os.path.join(MODEL_DIR, "linear_regression.pkl"),
                    os.path.join(MODEL_DIR, "random_forest.pkl"),
                    os.path.join(MODEL_DIR, "scaler.pkl"),
                    os.path.join(MODEL_DIR, "best_model.pkl"),
                    os.path.join(MODEL_DIR, "metrics.pkl"),
                ]:
                    if os.path.exists(f): os.remove(f)
                with st.spinner("📦 Downloading real dataset from Kaggle … (this may take a minute)"):
                    try:
                        load_kaggle_dataset(DATA_PATH)
                        st.success("✅ Real dataset loaded! Retraining models …")
                    except Exception as e:
                        st.error(f"🚨 Kaggle download failed: {e}")
                        st.stop()
                invalidate_model_cache()
                st.cache_data.clear()
                if "metrics" in st.session_state: del st.session_state["metrics"]
                st.rerun()
        if st.session_state.get("show_retrain_confirm"):
            confirm = st.checkbox("⚠️ Confirm retrain — this will delete all saved models and data",
                                  key="retrain_confirm_cb")
            if confirm:
                st.session_state["show_retrain_confirm"] = False
                if os.path.exists(DATA_PATH):
                    os.remove(DATA_PATH)
                for f in [
                    os.path.join(MODEL_DIR, "linear_regression.pkl"),
                    os.path.join(MODEL_DIR, "random_forest.pkl"),
                    os.path.join(MODEL_DIR, "scaler.pkl"),
                    os.path.join(MODEL_DIR, "best_model.pkl"),
                    os.path.join(MODEL_DIR, "metrics.pkl"),
                ]:
                    if os.path.exists(f): os.remove(f)
                invalidate_model_cache()
                st.cache_data.clear()
                if "metrics" in st.session_state: del st.session_state["metrics"]
                st.rerun()

    st.markdown("---")

    col_info, col_model = st.columns([2, 1])
    with col_info:
        st.markdown('<div class="section-title">📌 Input Summary</div>', unsafe_allow_html=True)
        dt_obj = datetime.datetime.combine(selected_date, selected_time)
        info_df = pd.DataFrame({
            "Field":  ["Date/Time", "Hour", "Day of Week", "Month", "Temperature (°C)", "Prev. Consumption (kWh)"],
            "Value": [
                dt_obj.strftime("%d %b %Y  %H:%M"),
                dt_obj.hour,
                dt_obj.strftime("%A"),
                dt_obj.strftime("%B"),
                f"{temperature:.1f} °C",
                f"{prev_cons:.1f} kWh",
            ]
        })
        st.dataframe(info_df, hide_index=True, use_container_width=True)

    with col_model:
        st.markdown('<div class="section-title">🏆 Active Model</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="model-badge">✅ {best_name}</div>', unsafe_allow_html=True)
        best_m = rf_m if best_name == "Random Forest" else lr_m
        st.markdown(f"""
        <br>
        <div class="metric-card">
            <div class="label">R² Score</div>
            <div class="value">{best_m['R2']}</div>
            <div class="delta">MAE: {best_m['MAE']} kWh</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Run prediction ──────────────────────────────────────────────────────
    if predict_btn or "last_prediction" in st.session_state:

        if predict_btn:
            day_of_year = dt_obj.timetuple().tm_yday
            try:
                prediction = predict_consumption(
                    hour=dt_obj.hour,
                    day_of_week=dt_obj.weekday(),
                    month=dt_obj.month,
                    day_of_year=day_of_year,
                    temperature=temperature,
                    prev_consumption=prev_cons,
                )
            except Exception as e:
                st.error(f"🚨 Prediction failed: {e}. Try retraining the models.")
                st.stop()
            st.session_state.last_prediction = prediction
            st.session_state.last_threshold  = threshold

        prediction = st.session_state.last_prediction
        threshold_ = st.session_state.get("last_threshold", threshold)

        st.markdown("---")
        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

        col_r, col_insight = st.columns([1, 1])

        with col_r:
            st.markdown(f"""
            <div class="prediction-box">
                <div class="pred-label">Predicted Electricity Consumption</div>
                <div class="pred-value">{prediction:.2f}</div>
                <div class="pred-unit">kilowatt-hours (kWh)</div>
            </div>
            """, unsafe_allow_html=True)

        with col_insight:
            avg_cons = df["consumption"].mean()
            pct      = (prediction - avg_cons) / avg_cons * 100

            st.markdown('<div class="section-title">💡 Smart Insights</div>', unsafe_allow_html=True)

            # Alert
            if prediction > threshold_:
                st.markdown(f"""
                <div class="alert-danger">
                    🚨 <strong>High Usage Alert!</strong><br>
                    Predicted {prediction:.1f} kWh exceeds your threshold of {threshold_:.1f} kWh.
                    Consider shifting heavy loads to off-peak hours.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-success">
                    ✅ <strong>Within Normal Range</strong><br>
                    Predicted usage is below your threshold of {threshold_:.1f} kWh. 
                    Energy usage looks good!
                </div>""", unsafe_allow_html=True)

            # vs average
            arrow = "▲" if pct > 0 else "▼"
            color = "#ef4444" if pct > 0 else "#10b981"
            st.markdown(f"""
            <div class="alert-info">
                📊 <strong>vs. Historical Average</strong><br>
                Prediction is <span style="color:{color};font-weight:700">
                {arrow} {abs(pct):.1f}%</span> compared to average ({avg_cons:.1f} kWh).
            </div>""", unsafe_allow_html=True)

            # Time-of-day tip
            h = dt_obj.hour
            if 6 <= h < 9:
                tip = "☀️ Morning peak — appliances like geysers & ACs spike demand."
            elif 17 <= h < 22:
                tip = "🌆 Evening peak — highest grid load period of the day."
            elif 0 <= h < 5:
                tip = "🌙 Night off-peak — optimal time to run heavy appliances."
            else:
                tip = "📊 Mid-day period — moderate consumption expected."
            st.markdown(f'<div class="alert-info">🕐 <strong>Time-of-Day Tip</strong><br>{tip}</div>',
                        unsafe_allow_html=True)

        # ── Download report ──────────────────────────────────────────────────
        st.markdown("---")
        report_rows = {
            "Prediction Date":         dt_obj.strftime("%d %b %Y %H:%M"),
            "Predicted Consumption (kWh)": round(prediction, 2),
            "Threshold (kWh)":         threshold_,
            "Exceeded Threshold":      "Yes" if prediction > threshold_ else "No",
            "Avg Historical (kWh)":    round(avg_cons, 2),
            "Deviation from Avg (%)":  round(pct, 2),
            "Model Used":              best_name,
            "Model R²":                best_m["R2"],
            "Model MAE":               best_m["MAE"],
            "Temperature (°C)":        temperature,
            "Prev. Consumption (kWh)": prev_cons,
        }
        report_df  = pd.DataFrame(report_rows.items(), columns=["Metric", "Value"])
        csv_buffer = io.StringIO()
        report_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="📥 Download Prediction Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"electricity_prediction_{dt_obj.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.markdown("""
        <div class="alert-info">
            ☝️ <strong>Set your inputs above</strong> and click 
            <em>"Predict Consumption"</em> to generate a forecast.
        </div>""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 – MODEL COMPARISON
# ────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="section-title">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────
    cols = st.columns(6)
    kpis = [
        ("Linear Reg — MAE",  lr_m["MAE"],  "kWh"),
        ("Linear Reg — MSE",  lr_m["MSE"],  "kWh²"),
        ("Linear Reg — R²",   lr_m["R2"],   ""),
        ("Random Forest — MAE", rf_m["MAE"],"kWh"),
        ("Random Forest — MSE", rf_m["MSE"],"kWh²"),
        ("Random Forest — R²",  rf_m["R2"], ""),
    ]
    for i, (label, val, unit) in enumerate(kpis):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="font-size:1.5rem">{val}</div>
                <div class="delta">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar chart comparison ─────────────────────────────────────────────────
    col_bar, col_scatter = st.columns(2)

    with col_bar:
        col_mae, col_mse, col_r2 = st.columns(3)
        for col_chart, metric, unit in [
            (col_mae, "MAE", "kWh"),
            (col_mse, "MSE", "kWh²"),
            (col_r2,  "R2",  ""),
        ]:
            with col_chart:
                fig_m = go.Figure(data=[
                    go.Bar(name="Linear Regression", x=["LR"], y=[lr_m[metric]],
                           marker_color="#3b82f6", opacity=.85),
                    go.Bar(name="Random Forest",     x=["RF"], y=[rf_m[metric]],
                           marker_color="#f59e0b", opacity=.85),
                ])
                label = "R²" if metric == "R2" else metric
                fig_m.update_layout(**CHART_LAYOUT, title=f"{label} ({unit})".strip(" ()"),
                                    barmode="group", showlegend=False)
                st.plotly_chart(fig_m, use_container_width=True)

    with col_scatter:
        # Actual vs Predicted scatter for the best model
        if best_name == "Random Forest":
            preds   = rf_m["predictions"]
            actuals = rf_m["actuals"]
        else:
            preds   = lr_m["predictions"]
            actuals = lr_m["actuals"]

        # Sub-sample for performance
        rng = np.random.default_rng(42)
        idx = rng.choice(len(preds), min(500, len(preds)), replace=False)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=actuals[idx], y=preds[idx],
            mode="markers",
            marker=dict(color="#f59e0b", opacity=.6, size=5),
            name="Predicted vs Actual",
        ))
        # Perfect-fit line
        lo, hi = actuals.min(), actuals.max()
        fig_sc.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Perfect Fit",
        ))
        fig_sc.update_layout(**CHART_LAYOUT, title=f"Actual vs Predicted — {best_name}",
                             xaxis_title="Actual (kWh)", yaxis_title="Predicted (kWh)")
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Time-series actual vs predicted ──────────────────────────────────────
    st.markdown('<div class="section-title">📉 Actual vs Predicted Over Time</div>', unsafe_allow_html=True)

    n_show  = min(500, len(preds))
    x_axis  = np.arange(n_show)
    fig_ts  = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=x_axis, y=actuals[:n_show],
        mode="lines", name="Actual",
        line=dict(color="#60a5fa", width=1.5),
    ))
    fig_ts.add_trace(go.Scatter(
        x=x_axis, y=preds[:n_show],
        mode="lines", name="Predicted",
        line=dict(color="#f59e0b", width=1.5),
    ))
    fig_ts.update_layout(**CHART_LAYOUT, title="Test-Set: Actual vs Predicted Consumption",
                         xaxis_title="Hour Index", yaxis_title="Consumption (kWh)")
    st.plotly_chart(fig_ts, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 – VISUALIZATIONS
# ────────────────────────────────────────────────────────────────────────────
with tab_viz:
    st.markdown('<div class="section-title">📈 Data Insights & Trend Analysis</div>', unsafe_allow_html=True)

    # ── Aggregate helpers ────────────────────────────────────────────────────
    df_viz = df.copy()
    df_viz["date"] = pd.to_datetime(df_viz["datetime"]).dt.date
    df_viz["week"] = pd.to_datetime(df_viz["datetime"]).dt.isocalendar().week.astype(int)
    df_viz["year_month"] = pd.to_datetime(df_viz["datetime"]).dt.to_period("M").astype(str)

    daily_df   = df_viz.groupby("date")["consumption"].sum().reset_index()
    monthly_df = df_viz.groupby("year_month")["consumption"].sum().reset_index()
    hourly_df  = df_viz.groupby("hour")["consumption"].mean().reset_index()
    dow_df     = df_viz.groupby("day_of_week")["consumption"].mean().reset_index()
    dow_df["day_name"] = dow_df["day_of_week"].map(
        {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    )

    # ── Row 1: Daily & Monthly ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        fig_daily = px.area(
            daily_df, x="date", y="consumption",
            color_discrete_sequence=["#3b82f6"],
            title="Daily Total Consumption (kWh)",
        )
        fig_daily.update_traces(fillcolor="rgba(59,130,246,.15)", line_color="#3b82f6")
        fig_daily.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_daily, use_container_width=True)

    with col_b:
        fig_monthly = px.bar(
            monthly_df, x="year_month", y="consumption",
            color="consumption",
            color_continuous_scale=["#1d4ed8","#f59e0b","#ef4444"],
            title="Monthly Total Consumption (kWh)",
        )
        fig_monthly.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_monthly, use_container_width=True)

    # ── Row 2: Hourly pattern & Day-of-week ─────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(
            x=hourly_df["hour"], y=hourly_df["consumption"],
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=7, color="#fbbf24"),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,.1)",
            name="Avg Consumption",
        ))
        fig_hourly.update_layout(**CHART_LAYOUT,
            title="Average Hourly Consumption Pattern",
            xaxis_title="Hour of Day", yaxis_title="Avg kWh",
            xaxis_tickmode="linear", xaxis_tick0=0, xaxis_dtick=2,
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with col_d:
        fig_dow = px.bar(
            dow_df, x="day_name", y="consumption",
            color="consumption",
            color_continuous_scale=["#1d4ed8","#f59e0b"],
            title="Average Consumption by Day of Week",
        )
        fig_dow.update_layout(**CHART_LAYOUT, coloraxis_showscale=False,
                              xaxis_title="Day", yaxis_title="Avg kWh")
        st.plotly_chart(fig_dow, use_container_width=True)

    # ── Row 3: Temp vs Consumption scatter ──────────────────────────────────
    st.markdown('<div class="section-title">🌡️ Temperature vs Consumption Correlation</div>',
                unsafe_allow_html=True)

    sample_df = df_viz.sample(min(1500, len(df_viz)), random_state=42)
    fig_temp  = px.scatter(
        sample_df, x="temperature", y="consumption",
        color="month", color_continuous_scale="Turbo",
        opacity=0.5, title="Temperature vs Electricity Consumption",
        labels={"temperature":"Temperature (°C)", "consumption":"Consumption (kWh)"},
        trendline="ols",
    )
    fig_temp.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig_temp, use_container_width=True)

    # ── Row 4: Heatmap (Hour × Month) ───────────────────────────────────────
    st.markdown('<div class="section-title">🗓️ Consumption Heatmap — Hour vs Month</div>',
                unsafe_allow_html=True)

    heat_df  = df_viz.groupby(["month","hour"])["consumption"].mean().reset_index()
    heat_piv = heat_df.pivot(index="month", columns="hour", values="consumption")

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_piv.values,
        x=[f"{h:02d}:00" for h in heat_piv.columns],
        y=[month_names.get(m, m) for m in heat_piv.index],
        colorscale="YlOrRd",
        showscale=True,
    ))
    fig_heat.update_layout(**CHART_LAYOUT, title="Avg Consumption by Hour and Month",
                            xaxis_title="Hour", yaxis_title="Month")
    st.plotly_chart(fig_heat, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 – FORECASTS
# ────────────────────────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown('<div class="section-title">📅 Multi-Horizon Forecasts</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        fc_date = st.date_input("📅 Start Date", value=datetime.date.today(), key="fc_date")
    with fc2:
        fc_city = st.text_input("🌍 City for Live Temperature", value="Chennai", key="fc_city")
        fc_fetch = st.button("🌡️ Fetch Temperature", key="fc_fetch")
        if fc_fetch:
            try:
                ft, fd = get_current_temperature(fc_city)
                st.session_state["fc_fetched_temp"] = ft
                st.session_state["fc_city_display"]  = fd
            except Exception as e:
                st.warning(f"Could not fetch temperature: {e}. Using manual value.")
        if "fc_fetched_temp" in st.session_state:
            st.success(f"🌡️ {st.session_state['fc_city_display']}: **{st.session_state['fc_fetched_temp']}°C**")
        fc_temp = st.slider(
            "Temperature (°C) — override if needed",
            -10.0, 50.0,
            float(st.session_state.get("fc_fetched_temp", 22.0)),
            0.5, key="fc_temp",
        )
    with fc3:
        fc_prev = st.slider("🔋 Starting Consumption (kWh)", 10.0, 250.0, 85.0, 0.5, key="fc_prev")

    run_fc = st.button("📊 Generate Forecasts")

    if run_fc or "fc_results" in st.session_state:

        if run_fc:
            # ── helper: predict a sequence of hours rolling prev_consumption ──
            def forecast_hours(start_dt, n_hours, base_temp, start_prev):
                rows = []
                prev = start_prev
                for i in range(n_hours):
                    dt = start_dt + datetime.timedelta(hours=i)
                    pred = predict_consumption(
                        hour=dt.hour,
                        day_of_week=dt.weekday(),
                        month=dt.month,
                        day_of_year=dt.timetuple().tm_yday,
                        temperature=base_temp,
                        prev_consumption=prev,
                    )
                    rows.append({"datetime": dt, "predicted_kwh": round(pred, 2)})
                    prev = pred
                return pd.DataFrame(rows)

            start_dt = datetime.datetime.combine(fc_date, datetime.time(0, 0))

            with st.spinner("Generating forecasts…"):
                df_24h  = forecast_hours(start_dt, 24,   fc_temp, fc_prev)
                df_week = forecast_hours(start_dt, 7*24, fc_temp, fc_prev)
                df_2mo  = forecast_hours(start_dt, 60*24, fc_temp, fc_prev)

            st.session_state.fc_results = {
                "24h":  df_24h,
                "week": df_week,
                "2mo":  df_2mo,
                "date": fc_date,
            }

        res    = st.session_state.fc_results
        df_24h = res["24h"]
        df_week = res["week"]
        df_2mo  = res["2mo"]

        # ── KPI summary row ──────────────────────────────────────────────────
        kf1, kf2, kf3, kf4 = st.columns(4)
        peak_hour = df_24h.loc[df_24h["predicted_kwh"].idxmax(), "datetime"]
        for col, label, val, unit in [
            (kf1, "24h Total",    df_24h["predicted_kwh"].sum(),  "kWh"),
            (kf2, "Weekly Total", df_week["predicted_kwh"].sum(), "kWh"),
            (kf3, "2-Month Total",df_2mo["predicted_kwh"].sum(),  "kWh"),
            (kf4, "Peak Hour",    peak_hour.strftime("%H:%M"),    df_24h["predicted_kwh"].max()),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value" style="font-size:1.5rem">{val if isinstance(val,str) else f"{val:,.1f}"}</div>
                    <div class="delta">{unit}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 24-Hour Forecast ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">🕐 24-Hour Forecast</div>', unsafe_allow_html=True)
        fig_24 = go.Figure()
        fig_24.add_trace(go.Scatter(
            x=df_24h["datetime"].dt.strftime("%H:%M"),
            y=df_24h["predicted_kwh"],
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=7, color="#fbbf24"),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,.1)",
            name="Predicted kWh",
        ))
        fig_24.update_layout(**CHART_LAYOUT,
            title=f"Hourly Forecast — {fc_date.strftime('%d %b %Y')}",
            xaxis_title="Hour", yaxis_title="Consumption (kWh)",
        )
        st.plotly_chart(fig_24, use_container_width=True)

        # download 24h
        buf = io.StringIO()
        df_24h.to_csv(buf, index=False)
        st.download_button("📥 Download 24h Forecast CSV", buf.getvalue(),
            file_name=f"forecast_24h_{fc_date}.csv", mime="text/csv")

        # ── Weekly Forecast ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">📆 7-Day Forecast</div>', unsafe_allow_html=True)
        df_week["date"] = df_week["datetime"].dt.date
        df_week["day_name"] = df_week["datetime"].dt.strftime("%a %d %b")
        daily_week = df_week.groupby(["date","day_name"])["predicted_kwh"].sum().reset_index()

        fig_week = go.Figure()
        fig_week.add_trace(go.Bar(
            x=daily_week["day_name"],
            y=daily_week["predicted_kwh"],
            marker_color="#3b82f6",
            opacity=0.85,
            name="Daily Total kWh",
        ))
        fig_week.update_layout(**CHART_LAYOUT,
            title="7-Day Daily Total Consumption Forecast",
            xaxis_title="Day", yaxis_title="Total kWh",
        )
        st.plotly_chart(fig_week, use_container_width=True)

        buf2 = io.StringIO()
        daily_week[["date","day_name","predicted_kwh"]].to_csv(buf2, index=False)
        st.download_button("📥 Download 7-Day Forecast CSV", buf2.getvalue(),
            file_name=f"forecast_7day_{fc_date}.csv", mime="text/csv")

        # ── 2-Month Forecast ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">📈 2-Month Forecast</div>', unsafe_allow_html=True)
        df_2mo["date"] = df_2mo["datetime"].dt.date
        daily_2mo = df_2mo.groupby("date")["predicted_kwh"].sum().reset_index()

        fig_2mo = px.area(
            daily_2mo, x="date", y="predicted_kwh",
            color_discrete_sequence=["#10b981"],
            title="60-Day Daily Total Consumption Forecast",
            labels={"predicted_kwh": "Total kWh", "date": "Date"},
        )
        fig_2mo.update_traces(fillcolor="rgba(16,185,129,.1)", line_color="#10b981")
        fig_2mo.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_2mo, use_container_width=True)

        buf3 = io.StringIO()
        daily_2mo.to_csv(buf3, index=False)
        st.download_button("📥 Download 2-Month Forecast CSV", buf3.getvalue(),
            file_name=f"forecast_2month_{fc_date}.csv", mime="text/csv")

    else:
        st.markdown("""
        <div class="alert-info">
            ☝️ <strong>Set your inputs above</strong> and click
            <em>"Generate Forecasts"</em> to see 24-hour, 7-day and 2-month predictions.
        </div>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 – DATA EXPLORER
# ────────────────────────────────────────────────────────────────────────────
with tab_explorer:
    st.markdown('<div class="section-title">🔍 Interactive Data Explorer</div>', unsafe_allow_html=True)

    df_ex = df.copy()
    df_ex["datetime"] = pd.to_datetime(df_ex["datetime"])

    # ── Filters row ──────────────────────────────────────────────────────────
    fx1, fx2, fx3, fx4 = st.columns(4)
    with fx1:
        month_options = list(range(1, 13))
        month_labels  = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
        sel_months = st.multiselect(
            "📅 Month", options=month_options,
            format_func=lambda m: month_labels[m-1],
            default=month_options,
        )
    with fx2:
        hour_range = st.slider("🕐 Hour Range", 0, 23, (0, 23))
    with fx3:
        cons_range = st.slider(
            "⚡ Consumption (kWh)",
            float(df_ex["consumption"].min()),
            float(df_ex["consumption"].max()),
            (float(df_ex["consumption"].min()), float(df_ex["consumption"].max())),
        )
    with fx4:
        temp_range = st.slider(
            "🌡️ Temperature (°C)",
            float(df_ex["temperature"].min()),
            float(df_ex["temperature"].max()),
            (float(df_ex["temperature"].min()), float(df_ex["temperature"].max())),
        )

    # Apply filters
    mask = (
        df_ex["month"].isin(sel_months) &
        df_ex["hour"].between(*hour_range) &
        df_ex["consumption"].between(*cons_range) &
        df_ex["temperature"].between(*temp_range)
    )
    df_filtered = df_ex[mask].reset_index(drop=True)

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, val in [
        (k1, "Rows",        f"{len(df_filtered):,}"),
        (k2, "Avg kWh",     f"{df_filtered['consumption'].mean():.2f}"),
        (k3, "Max kWh",     f"{df_filtered['consumption'].max():.2f}"),
        (k4, "Min kWh",     f"{df_filtered['consumption'].min():.2f}"),
        (k5, "Avg Temp °C", f"{df_filtered['temperature'].mean():.1f}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center">
                <div class="label">{label}</div>
                <div class="value" style="font-size:1.5rem">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distribution chart ───────────────────────────────────────────────────
    ex_c1, ex_c2 = st.columns(2)

    with ex_c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_filtered["consumption"],
            nbinsx=50,
            marker_color="#f59e0b",
            opacity=0.8,
            name="Consumption",
        ))
        fig_hist.update_layout(**CHART_LAYOUT,
            title="Consumption Distribution",
            xaxis_title="kWh", yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with ex_c2:
        fig_box = go.Figure()
        for dow, name in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
            subset = df_filtered[df_filtered["day_of_week"] == dow]["consumption"]
            if len(subset):
                fig_box.add_trace(go.Box(
                    y=subset, name=name,
                    marker_color="#3b82f6" if dow < 5 else "#f59e0b",
                    line_color="#60a5fa",
                ))
        fig_box.update_layout(**CHART_LAYOUT,
            title="Consumption Spread by Day of Week",
            yaxis_title="kWh", showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Raw data table ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Filtered Data Table</div>', unsafe_allow_html=True)

    display_cols = ["datetime","hour","day_of_week","month","temperature",
                    "prev_consumption","consumption"]
    st.dataframe(
        df_filtered[display_cols].head(500),
        use_container_width=True,
        hide_index=True,
    )
    if len(df_filtered) > 500:
        st.caption(f"Showing first 500 of {len(df_filtered):,} rows.")

    # ── Export filtered data ─────────────────────────────────────────────────
    ex_buf = io.StringIO()
    df_filtered[display_cols].to_csv(ex_buf, index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=ex_buf.getvalue(),
        file_name="filtered_electricity_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.markdown("""
<br><br>
<div style="text-align:center;color:#334155;font-size:.82rem;padding:1.5rem 0;
            border-top:1px solid #1e293b">
    ⚡ Electricity Forecasting ML App &nbsp;|&nbsp;
    Built with Python · Scikit-learn · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
