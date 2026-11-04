"""
TurboRUL — Turbofan Engine Remaining Useful Life Predictor
============================================================
Streamlit dashboard for predicting engine Remaining Useful Life (RUL)
from C-MAPSS sensor telemetry using a trained Gradient Boosting model.

Required artifacts (place in the same folder as this file):
    gb_model.pkl          - trained GradientBoostingRegressor (joblib)
    scaler.pkl             - fitted StandardScaler (joblib)
    feature_columns.json   - ordered list of feature names used at train time
    feature_stats.json     - per-feature min/max/mean/std from training data
    metrics.json            - optional, model performance metrics for the About panel

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TurboRUL · Engine Health Predictor",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

MODEL_PATH = MODEL_DIR / "gb_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
STATS_PATH = MODEL_DIR / "feature_stats.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Minimum cycle the app will ever report — guarantees we never tell an
# engineer an engine has "0 cycles left", which is operationally meaningless.
MIN_RUL = 1

# Health banding used across the dashboard (cycles remaining)
CRITICAL_THRESHOLD = 20
WARNING_THRESHOLD = 50

# Full reference glossary for every possible NASA C-MAPSS field: 3 operating
# settings + 21 sensors + the two record fields. Whichever subset RFE/Lasso
# selects during training, the dashboard can always show a real engineering
# name instead of a bare column name like "sensor7".
# Format: column -> (short_code, full_name, unit, category)
SENSOR_INFO = {
    "id":       ("ID",   "Engine Unit Identifier",          "",        "Flight Record"),
    "cycle":    ("Cycle","Operating Cycles Completed",      "cycles",  "Flight Record"),

    "setting1": ("Alt",  "Altitude (operating setting)",    "ft",      "Operating Setting"),
    "setting2": ("Mach", "Mach Number (operating setting)", "Mach",    "Operating Setting"),
    "setting3": ("TRA",  "Throttle Resolver Angle",         "deg",     "Operating Setting"),

    "sensor1":  ("T2",   "Total Temperature at Fan Inlet",        "°R",     "Temperature"),
    "sensor2":  ("T24",  "Total Temperature at LPC Outlet",       "°R",     "Temperature"),
    "sensor3":  ("T30",  "Total Temperature at HPC Outlet",       "°R",     "Temperature"),
    "sensor4":  ("T50",  "Total Temperature at LPT Outlet",       "°R",     "Temperature"),
    "sensor5":  ("P2",   "Pressure at Fan Inlet",                 "psia",   "Pressure"),
    "sensor6":  ("P15",  "Total Pressure in Bypass-Duct",         "psia",   "Pressure"),
    "sensor7":  ("P30",  "Total Pressure at HPC Outlet",          "psia",   "Pressure"),
    "sensor8":  ("Nf",   "Physical Fan Speed",                    "rpm",    "Rotational Speed"),
    "sensor9":  ("Nc",   "Physical Core Speed",                   "rpm",    "Rotational Speed"),
    "sensor10": ("EPR",  "Engine Pressure Ratio (P50 / P2)",      "ratio",  "Pressure"),
    "sensor11": ("Ps30", "Static Pressure at HPC Outlet",         "psia",   "Pressure"),
    "sensor12": ("phi",  "Fuel Flow to Ps30 Ratio",                "pps/psi","Flow"),
    "sensor13": ("NRf",  "Corrected Fan Speed",                   "rpm",    "Rotational Speed"),
    "sensor14": ("NRc",  "Corrected Core Speed",                  "rpm",    "Rotational Speed"),
    "sensor15": ("BPR",  "Bypass Ratio",                          "ratio",  "Flow"),
    "sensor16": ("farB", "Burner Fuel-Air Ratio",                 "ratio",  "Flow"),
    "sensor17": ("htBleed", "Bleed Enthalpy",                     "—",      "Other"),
    "sensor18": ("Nf_dmd",  "Demanded Fan Speed",                 "rpm",    "Rotational Speed"),
    "sensor19": ("PCNfR_dmd", "Demanded Corrected Fan Speed",     "rpm",    "Rotational Speed"),
    "sensor20": ("W31",  "HPT Coolant Bleed",                     "lbm/s",  "Flow"),
    "sensor21": ("W32",  "LPT Coolant Bleed",                     "lbm/s",  "Flow"),
}

# Display order for grouped sidebar sections / glossary table
CATEGORY_ORDER = [
    "Flight Record", "Operating Setting", "Temperature",
    "Pressure", "Rotational Speed", "Flow", "Other",
]


def sensor_meta(col):
    """Return (code, full_name, unit, category) for any feature column,
    falling back gracefully for anything not in the glossary."""
    return SENSOR_INFO.get(col, (col, col.replace("_", " ").title(), "", "Other"))


# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        h1, h2, h3, .hud-title { font-family: 'Sora', sans-serif; }

        .stApp {
            background: radial-gradient(circle at 15% 0%, #111827 0%, #0A0E14 45%, #060809 100%);
            color: #E5E7EB;
        }

        section[data-testid="stSidebar"] {
            background: #0D1320;
            border-right: 1px solid #1F2937;
        }
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stNumberInput label {
            color: #9CA3AF !important;
            font-size: 0.82rem;
            font-weight: 500;
        }

        /* Top banner */
        .hud-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.2rem 1.6rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #0F1A2C 0%, #111827 100%);
            border: 1px solid #1F2A3D;
            margin-bottom: 1.4rem;
        }
        .hud-title { font-size: 1.55rem; font-weight: 700; color: #F1F5F9; margin: 0; letter-spacing: 0.3px;}
        .hud-subtitle { font-size: 0.92rem; color: #7C8AA0; margin-top: 2px; }
        .hud-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #38BDF8;
            background: rgba(56,189,248,0.08);
            border: 1px solid rgba(56,189,248,0.35);
            padding: 5px 12px;
            border-radius: 999px;
            letter-spacing: 0.5px;
        }

        /* KPI cards */
        .kpi-card {
            background: #121826;
            border: 1px solid #1F2937;
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            height: 100%;
        }
        .kpi-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #7C8AA0;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.1rem;
            font-weight: 600;
            line-height: 1.1;
        }
        .kpi-sub { font-size: 0.78rem; color: #6B7686; margin-top: 4px; }

        .status-pill {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.3px;
        }
        .status-healthy  { background: rgba(52,211,153,0.12); color:#34D399; border:1px solid rgba(52,211,153,0.4);}
        .status-warning  { background: rgba(251,191,36,0.12); color:#FBBF24; border:1px solid rgba(251,191,36,0.4);}
        .status-critical { background: rgba(248,113,113,0.12); color:#F87171; border:1px solid rgba(248,113,113,0.4);}

        .panel {
            background: #121826;
            border: 1px solid #1F2937;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
        }
        .panel h4 { margin-top:0; color:#E5E7EB; font-size: 1.0rem; }

        .reco-box {
            border-left: 3px solid #38BDF8;
            background: rgba(56,189,248,0.06);
            padding: 0.8rem 1rem;
            border-radius: 6px;
            font-size: 0.92rem;
            color: #D1D9E6;
        }

        .stButton > button {
            background: linear-gradient(135deg, #38BDF8 0%, #0EA5E9 100%);
            color: #051019;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1rem;
            width: 100%;
            letter-spacing: 0.3px;
        }
        .stButton > button:hover { filter: brightness(1.08); }

        .ghost-btn > button {
            background: transparent !important;
            color: #9CA3AF !important;
            border: 1px solid #2A3650 !important;
            font-weight: 500 !important;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = [p.name for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not p.exists()]
    if missing:
        return None, None, None, None, None, missing

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = json.loads(FEATURES_PATH.read_text())

    stats = json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

    return model, scaler, feature_columns, stats, metrics, []


def safe_range(col_stats, fallback_min=0.0, fallback_max=100.0, fallback_mean=50.0):
    if not col_stats:
        return fallback_min, fallback_max, fallback_mean
    lo, hi, mean = col_stats["min"], col_stats["max"], col_stats["mean"]
    if lo == hi:
        hi = lo + 1.0
    return lo, hi, mean


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_ICON = {
    "Flight Record": "🛫",
    "Operating Setting": "🎛️",
    "Temperature": "🌡️",
    "Pressure": "🧭",
    "Rotational Speed": "⚙️",
    "Flow": "💨",
    "Other": "🔧",
}


def render_sidebar(feature_columns, stats):
    st.sidebar.markdown(
        "<div style='font-family:Sora,sans-serif;font-size:1.05rem;font-weight:700;color:#F1F5F9;'>"
        "✈️ TurboRUL Console</div>"
        "<div style='font-size:0.8rem;color:#7C8AA0;margin-bottom:1rem;'>"
        "Enter current engine telemetry below, then run the prediction. "
        "Hover the <b>?</b> next to any field for what it measures.</div>",
        unsafe_allow_html=True,
    )

    # Initialize session state defaults from training-data means
    for col in feature_columns:
        key = f"input_{col}"
        if key not in st.session_state:
            col_stats = stats.get(col)
            _, _, mean = safe_range(col_stats)
            st.session_state[key] = float(round(mean, 4))

    def load_sample():
        for col in feature_columns:
            col_stats = stats.get(col)
            _, _, mean = safe_range(col_stats)
            st.session_state[f"input_{col}"] = float(round(mean, 4))

    st.sidebar.button("🔄  Load typical engine sample", on_click=load_sample, use_container_width=True)
    st.sidebar.markdown("---")

    inputs = {}

    # Group every input feature by engineering category so the sidebar reads
    # like an organized instrument panel instead of a flat list of raw names.
    grouped = {}
    for col in feature_columns:
        _, _, _, category = sensor_meta(col)
        grouped.setdefault(category, []).append(col)

    ordered_categories = [c for c in CATEGORY_ORDER if c in grouped]

    for category in ordered_categories:
        icon = CATEGORY_ICON.get(category, "📊")
        with st.sidebar.expander(f"{icon}  {category}", expanded=(category in ("Flight Record", "Temperature"))):
            for col in grouped[category]:
                code, full_name, unit, _ = sensor_meta(col)
                unit_suffix = f" ({unit})" if unit and unit != "—" else ""
                help_text = f"{full_name}{' · unit: ' + unit if unit and unit != '—' else ''}"

                if col in ("id", "cycle"):
                    col_stats = stats.get(col)
                    lo, hi, _ = safe_range(col_stats, 1, 400, 100)
                    inputs[col] = st.number_input(
                        f"{full_name}",
                        min_value=float(lo), max_value=float(max(hi, lo + 1)),
                        step=1.0, key=f"input_{col}", help=help_text,
                    )
                else:
                    col_stats = stats.get(col)
                    lo, hi, _ = safe_range(col_stats)
                    span = hi - lo
                    step = span / 200 if span > 0 else 0.01
                    inputs[col] = st.slider(
                        f"{full_name} · {code}{unit_suffix}",
                        min_value=float(lo), max_value=float(hi),
                        step=float(round(step, 6)) if step > 0 else 0.01,
                        key=f"input_{col}", help=help_text,
                    )

    st.sidebar.markdown("---")
    predict_clicked = st.sidebar.button("🔮  Predict Remaining Life", use_container_width=True)

    return inputs, predict_clicked


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANELS
# ─────────────────────────────────────────────────────────────────────────────
def status_for(rul):
    if rul <= CRITICAL_THRESHOLD:
        return "Critical", "status-critical", "#F87171"
    if rul <= WARNING_THRESHOLD:
        return "Monitor", "status-warning", "#FBBF24"
    return "Healthy", "status-healthy", "#34D399"


def recommendation_for(label):
    text = {
        "Critical": "Schedule immediate maintenance inspection. Engine is approaching end of useful life — "
                    "prioritize this unit in the next maintenance window.",
        "Monitor": "Plan a maintenance check in the near term. Trend sensor readings over the next few "
                   "cycles to confirm degradation rate.",
        "Healthy": "No immediate action required. Continue routine monitoring on the standard inspection cycle.",
    }
    return text[label]


def render_gauge(rul, gauge_max):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul,
            number={"suffix": " cycles", "font": {"size": 38, "color": "#E5E7EB", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, gauge_max], "tickcolor": "#4B5563", "tickfont": {"color": "#7C8AA0"}},
                "bar": {"color": "#38BDF8", "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, CRITICAL_THRESHOLD], "color": "rgba(248,113,113,0.25)"},
                    {"range": [CRITICAL_THRESHOLD, WARNING_THRESHOLD], "color": "rgba(251,191,36,0.22)"},
                    {"range": [WARNING_THRESHOLD, gauge_max], "color": "rgba(52,211,153,0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#E5E7EB", "width": 2},
                    "thickness": 0.8,
                    "value": rul,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E5E7EB"},
    )
    return fig


def render_deviation_chart(inputs, stats, feature_columns):
    rows = []
    for col in feature_columns:
        if col in ("id",):
            continue
        col_stats = stats.get(col)
        if not col_stats or col_stats.get("std", 0) == 0:
            continue
        z = (inputs[col] - col_stats["mean"]) / col_stats["std"]
        code, full_name, _, _ = sensor_meta(col)
        rows.append((f"{code} · {full_name}", z))

    if not rows:
        return None

    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = []
    for v in values:
        if abs(v) >= 2:
            colors.append("#F87171")
        elif abs(v) >= 1:
            colors.append("#FBBF24")
        else:
            colors.append("#38BDF8")

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Std. deviations from typical engine (z-score)", gridcolor="#1F2937", zerolinecolor="#374151", color="#9CA3AF"),
        yaxis=dict(color="#E5E7EB"),
        font={"color": "#E5E7EB"},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    model, scaler, feature_columns, stats, metrics, missing = load_artifacts()

    st.markdown(
        """
        <div class="hud-banner">
            <div>
                <p class="hud-title">🛩️ TurboRUL — Engine Health Predictor</p>
                <p class="hud-subtitle">Remaining Useful Life estimation for turbofan engines · NASA C-MAPSS (FD001)</p>
            </div>
            <div class="hud-badge">GRADIENT BOOSTING · LIVE INFERENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if missing:
        st.error(
            "**Model artifacts not found.** This dashboard needs the following files in the same "
            f"folder as `app.py`: {', '.join(missing)}.\n\n"
            "Run the artifact-export cell at the end of your training notebook, then place the "
            "generated files alongside `app.py` before launching Streamlit again."
        )
        st.stop()

    inputs, predict_clicked = render_sidebar(feature_columns, stats)

    if "last_rul" not in st.session_state:
        st.session_state.last_rul = None

    if predict_clicked:
        X_input = pd.DataFrame([[inputs[c] for c in feature_columns]], columns=feature_columns)
        X_scaled = scaler.transform(X_input)
        raw_pred = float(model.predict(X_scaled)[0])
        # Never report a meaningless "0 cycles" — floor every prediction at MIN_RUL.
        rul = max(MIN_RUL, int(round(raw_pred)))
        st.session_state.last_rul = rul

    rul = st.session_state.last_rul

    if rul is None:
        st.markdown(
            "<div class='panel' style='text-align:center; padding:3rem 1rem;'>"
            "<h4>No prediction yet</h4>"
            "<p style='color:#7C8AA0;'>Set the engine telemetry in the sidebar and click "
            "<b>Predict Remaining Life</b> to see results here.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    label, pill_class, color = status_for(rul)
    gauge_max = max(125, int(rul * 1.4))

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">Predicted Remaining Life</div>
                    <div class="kpi-value" style="color:{color}">{rul} cycles</div>
                    <div class="kpi-sub">Estimated operational cycles before maintenance is required</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">Engine Status</div>
                    <div style="margin-top:6px;"><span class="status-pill {pill_class}">{label}</span></div>
                    <div class="kpi-sub">Based on remaining-cycle thresholds</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        r2_text = f"{metrics.get('r2'):.3f}" if metrics.get("r2") is not None else "—"
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">Model Confidence (R²)</div>
                    <div class="kpi-value" style="color:#38BDF8">{r2_text}</div>
                    <div class="kpi-sub">Gradient Boosting · held-out test performance</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2 = st.columns([1, 1.2])
    with g1:
        st.markdown("<div class='panel'><h4>Remaining Life Gauge</h4>", unsafe_allow_html=True)
        st.plotly_chart(render_gauge(rul, gauge_max), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<div class='reco-box'><b>Recommendation:</b> {recommendation_for(label)}</div></div>",
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown("<div class='panel'><h4>Sensor Deviation from Typical Engine</h4>", unsafe_allow_html=True)
        fig = render_deviation_chart(inputs, stats, feature_columns)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Sensor statistics unavailable — export `feature_stats.json` to enable this chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📋  Input Telemetry — what was just submitted", expanded=False):
        rows = []
        for col in feature_columns:
            code, full_name, unit, category = sensor_meta(col)
            rows.append(
                {
                    "Category": category,
                    "Code": code,
                    "Measurement": full_name,
                    "Value": round(inputs[col], 4),
                    "Unit": unit if unit != "—" else "",
                }
            )
        review_df = pd.DataFrame(rows)
        st.dataframe(review_df, use_container_width=True, hide_index=True)
        st.caption(
            "This is exactly what was sent to the model, in plain engineering terms — "
            "use it to double-check a reading before trusting the prediction above."
        )

    with st.expander("ℹ️  About this model & pipeline"):
        st.markdown(
            f"""
            **Pipeline:** raw C-MAPSS telemetry → cleaning & outlier removal (Isolation Forest) →
            feature selection (RFE + Lasso, **{len(feature_columns)} features** retained) →
            standardization (`StandardScaler`) → **Gradient Boosting Regressor**.
            """
        )
        if metrics:
            mcols = st.columns(4)
            mcols[0].metric("MAE", f"{metrics.get('mae', 0):.4f}")
            mcols[1].metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            mcols[2].metric("R²", f"{metrics.get('r2', 0):.4f}")
            mcols[3].metric("Training samples", f"{metrics.get('training_samples', '—')}")
        st.caption(
            "Dataset: NASA C-MAPSS Jet Engine Simulated Data (FD001), NASA Open Data Portal. "
            "Predictions are floored at "
            f"{MIN_RUL} cycle to avoid reporting an uninformative zero."
        )


if __name__ == "__main__":
    main()
