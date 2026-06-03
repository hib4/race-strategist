from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
DATA_PATH = ROOT / "data" / "f1_strategy_dataset_v4_validated.csv"
SAMPLE_INPUT_PATH = ROOT / "data" / "sample_prediction_input.json"
MODEL_CANDIDATES = [
    ROOT / "models" / "best_model_candidate.joblib",
    ROOT / "models" / "random_forest.joblib",
]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "WillPitNextLap"
MODEL_INPUT_COLUMNS = [
    "Driver",
    "LapNumber",
    "Compound",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "Race",
    "Year",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Normalized_TyreLife",
    "Position_Change",
]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"]

RED = "#da291c"
RED_DARK = "#9d2211"
CANVAS = "#181818"
SURFACE = "#303030"
TEXT = "#ffffff"
BODY = "#b8b8b8"
MUTED = "#8f8f8f"
HAIRLINE = "#303030"


st.set_page_config(
    page_title="Race Strategist",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --rs-red: {RED};
            --rs-red-dark: {RED_DARK};
            --rs-canvas: {CANVAS};
            --rs-surface: {SURFACE};
            --rs-text: {TEXT};
            --rs-body: {BODY};
            --rs-muted: {MUTED};
            --rs-line: {HAIRLINE};
            --rs-space-1: 0.4rem;
            --rs-space-2: 0.75rem;
            --rs-space-3: 1rem;
            --rs-space-4: 1.5rem;
            --rs-space-5: 2rem;
        }}

        html, body, [class*="stApp"] {{
            background: var(--rs-canvas);
            color: var(--rs-text);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp > header {{
            background: transparent;
        }}

        div[data-testid="stAppToolbar"] {{
            height: 2.75rem;
        }}

        .block-container {{
            padding-top: 4rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }}

        section[data-testid="stSidebar"] {{
            background: #111111;
            border-right: 1px solid var(--rs-line);
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--rs-text);
        }}

        h1, h2, h3 {{
            color: var(--rs-text);
            letter-spacing: 0;
        }}

        h1 {{
            font-size: clamp(2.25rem, 4.5vw, 4.6rem);
            line-height: 1.02;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }}

        h2 {{
            font-size: 1.65rem;
            font-weight: 750;
            margin-top: 0;
        }}

        h3 {{
            font-size: 1.05rem;
            font-weight: 700;
        }}

        p, li, label, div[data-testid="stMarkdownContainer"] {{
            color: var(--rs-body);
        }}

        a {{
            color: var(--rs-text);
        }}

        .rs-kicker {{
            color: var(--rs-red);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12rem;
            text-transform: uppercase;
            margin-bottom: var(--rs-space-1);
        }}

        .rs-subtitle {{
            color: var(--rs-text);
            font-size: clamp(1.05rem, 1.8vw, 1.45rem);
            font-weight: 500;
            max-width: 780px;
            margin-bottom: var(--rs-space-3);
        }}

        .rs-copy {{
            color: var(--rs-body);
            font-size: 0.98rem;
            line-height: 1.65;
            max-width: 860px;
        }}

        .rs-section-head {{
            margin: 0 0 var(--rs-space-4);
        }}

        .rs-section-title {{
            color: var(--rs-text);
            font-size: 1.65rem;
            line-height: 1.15;
            font-weight: 750;
            margin: 0;
        }}

        .rs-section-copy {{
            color: var(--rs-body);
            font-size: 0.98rem;
            line-height: 1.65;
            max-width: 860px;
            margin-top: var(--rs-space-2);
        }}

        .rs-subsection-title {{
            color: var(--rs-text);
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0 0 var(--rs-space-3);
        }}

        .rs-hero {{
            position: relative;
            overflow: hidden;
            min-height: 360px;
            padding: 56px 44px;
            border: 1px solid var(--rs-line);
            margin-bottom: var(--rs-space-4);
            background:
                linear-gradient(115deg, rgba(24,24,24,0.96) 0%, rgba(24,24,24,0.82) 48%, rgba(24,24,24,0.45) 100%),
                repeating-linear-gradient(135deg, rgba(218,41,28,0.18) 0 2px, transparent 2px 16px),
                radial-gradient(circle at 82% 20%, rgba(218,41,28,0.22), transparent 32%),
                #111111;
        }}

        .rs-hero::after {{
            content: "";
            position: absolute;
            right: -10%;
            bottom: 16%;
            width: 58%;
            height: 3px;
            background: var(--rs-red);
            transform: skewX(-22deg);
            box-shadow: -70px 44px 0 rgba(218,41,28,0.38), -150px 88px 0 rgba(218,41,28,0.22);
        }}

        .rs-hero > * {{
            position: relative;
            z-index: 1;
        }}

        .rs-panel {{
            background: var(--rs-surface);
            border: 1px solid #3a3a3a;
            padding: 24px;
            margin: 0;
        }}

        .rs-card {{
            background: #242424;
            border: 1px solid #383838;
            padding: 20px;
            min-height: 145px;
            margin: 0;
        }}

        .rs-card-title {{
            color: var(--rs-text);
            font-size: 0.94rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }}

        .rs-card-body {{
            color: var(--rs-body);
            font-size: 0.9rem;
            line-height: 1.55;
        }}

        .rs-metric {{
            background: #242424;
            border-left: 4px solid var(--rs-red);
            padding: 18px 20px;
            min-height: 105px;
            margin: 0;
        }}

        .rs-metric-label {{
            color: var(--rs-muted);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.1rem;
            text-transform: uppercase;
        }}

        .rs-metric-value {{
            color: var(--rs-text);
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.15;
            margin-top: 0.35rem;
        }}

        .rs-metric-note {{
            color: var(--rs-body);
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }}

        .rs-result {{
            background: #111111;
            border: 1px solid #3a3a3a;
            border-top: 5px solid var(--rs-red);
            padding: 28px;
            margin: var(--rs-space-3) 0 var(--rs-space-3);
        }}

        .rs-result-class {{
            color: var(--rs-text);
            font-size: clamp(2rem, 4vw, 3.8rem);
            line-height: 1;
            font-weight: 850;
            margin: 0.35rem 0 0.6rem;
        }}

        .rs-tag {{
            display: inline-block;
            background: var(--rs-red);
            color: white;
            padding: 7px 10px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }}

        .rs-divider {{
            height: 1px;
            background: #3a3a3a;
            margin: var(--rs-space-4) 0;
        }}

        .rs-caption {{
            color: var(--rs-muted);
            font-size: 0.82rem;
            line-height: 1.5;
            margin-top: -0.6rem;
            margin-bottom: 0.45rem;
        }}

        div.stButton > button {{
            width: 100%;
            height: 48px;
            border-radius: 0;
            border: 1px solid var(--rs-red);
            background: var(--rs-red);
            color: white;
            font-weight: 800;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }}

        div.stButton > button:hover {{
            background: var(--rs-red-dark);
            border-color: var(--rs-red-dark);
            color: white;
        }}

        div[data-testid="stMetric"] {{
            background: #242424;
            border-left: 4px solid var(--rs-red);
            padding: 16px 18px;
        }}

        div[data-testid="stMetric"] * {{
            color: var(--rs-text);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid #3a3a3a;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 0;
            color: var(--rs-body);
            background: #242424;
            padding: 10px 18px;
        }}

        .stTabs [aria-selected="true"] {{
            background: var(--rs-red);
            color: white;
        }}

        .stTabs {{
            margin-top: var(--rs-space-2);
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea {{
            border-radius: 4px !important;
            background: #111111 !important;
            border-color: #454545 !important;
        }}

        div[data-testid="stElementContainer"] {{
            margin-bottom: 0;
        }}

        div[data-testid="stElementContainer"]:has(.rs-hero),
        div[data-testid="stElementContainer"]:has(.rs-section-head),
        div[data-testid="stElementContainer"]:has(.rs-subsection-title),
        div[data-testid="stElementContainer"]:has(.rs-card),
        div[data-testid="stElementContainer"]:has(.rs-metric),
        div[data-testid="stElementContainer"]:has(.rs-result),
        div[data-testid="stElementContainer"]:has(.rs-divider),
        div[data-testid="stElementContainer"]:has(.rs-small),
        div[data-testid="stElementContainer"]:has(.rs-caption) {{
            margin-bottom: 0 !important;
        }}

        div[data-testid="stHorizontalBlock"] {{
            gap: 0.9rem;
            margin-bottom: var(--rs-space-3);
        }}

        div[data-testid="stAlert"] {{
            margin: 0.2rem 0 0.45rem;
        }}

        div[data-testid="stAlert"] > div {{
            padding-top: 0.65rem;
            padding-bottom: 0.65rem;
        }}

        div[data-testid="stCaptionContainer"] {{
            margin: -0.35rem 0 0.9rem;
        }}

        div[data-testid="stPlotlyChart"] {{
            margin-bottom: 0.1rem;
        }}

        div[data-testid="stDataFrame"] {{
            margin-bottom: var(--rs-space-3);
        }}

        div[data-testid="stExpander"] {{
            margin-top: var(--rs-space-2);
        }}

        div[data-testid="stImage"] {{
            margin-bottom: var(--rs-space-4);
        }}

        .stSlider [data-baseweb="slider"] div {{
            color: var(--rs-red);
        }}

        .rs-small {{
            color: var(--rs-muted);
            font-size: 0.82rem;
            line-height: 1.5;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def html_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="rs-card">
            <div class="rs-card-title">{title}</div>
            <div class="rs-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="rs-metric">
            <div class="rs-metric-label">{label}</div>
            <div class="rs-metric-value">{value}</div>
            <div class="rs-metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, copy: str | None = None) -> None:
    copy_html = f'<div class="rs-section-copy">{copy}</div>' if copy else ""
    st.markdown(
        f"""
        <div class="rs-section-head">
            <div class="rs-kicker">{kicker}</div>
            <div class="rs-section-title">{title}</div>
            {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_break() -> None:
    st.markdown('<div class="rs-divider"></div>', unsafe_allow_html=True)


def section_title(title: str) -> None:
    st.markdown(f'<div class="rs-subsection-title">{title}</div>', unsafe_allow_html=True)


def caption_text(text: str) -> None:
    st.markdown(f'<div class="rs-caption">{text}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].fillna("UNKNOWN")
    return df


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model_payload() -> tuple[dict[str, Any] | None, str | None, str | None]:
    for path in MODEL_CANDIDATES:
        if not path.exists():
            continue
        try:
            artifact = joblib.load(path)
            if isinstance(artifact, dict) and "pipeline" in artifact:
                return artifact, path.name, None
            if hasattr(artifact, "predict"):
                return {"pipeline": artifact, "threshold": 0.5}, path.name, None
            return None, path.name, "Model artifact was found but has an unsupported format."
        except Exception as exc:
            return None, path.name, f"Could not load model artifact: {exc}"
    return None, None, "No trained model artifact found in models/."


def model_feature_columns(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return MODEL_INPUT_COLUMNS
    pipe = payload.get("pipeline")
    if pipe is not None and hasattr(pipe, "feature_names_in_"):
        return list(pipe.feature_names_in_)
    return MODEL_INPUT_COLUMNS


def dataset_defaults(df: pd.DataFrame | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "Driver": "NOR",
        "LapNumber": 30,
        "Compound": "HARD",
        "Stint": 2,
        "TyreLife": 13.0,
        "Position": 10,
        "LapTime (s)": 91.167,
        "Race": "Azerbaijan Grand Prix",
        "Year": 2024,
        "LapTime_Delta": 0.0,
        "Cumulative_Degradation": -21.678,
        "RaceProgress": 0.42,
        "Normalized_TyreLife": 0.33,
        "Position_Change": 0.0,
    }

    if df is None or df.empty:
        return defaults

    for col in MODEL_INPUT_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median())
            if col in {"LapNumber", "Stint", "Position", "Year"}:
                defaults[col] = int(round(defaults[col]))
        else:
            defaults[col] = str(series.mode().iloc[0])

    return defaults


def load_sample_or_defaults(df: pd.DataFrame | None) -> dict[str, Any]:
    defaults = dataset_defaults(df)
    sample = load_json(str(SAMPLE_INPUT_PATH))
    if isinstance(sample, dict):
        defaults.update({k: v for k, v in sample.items() if k in MODEL_INPUT_COLUMNS})
    return defaults


def numeric_bounds(df: pd.DataFrame | None, col: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if df is None or col not in df.columns:
        return fallback
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return fallback
    lo = float(vals.quantile(0.01))
    hi = float(vals.quantile(0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return fallback
    return lo, hi


def degradation_levels(df: pd.DataFrame | None) -> dict[str, float]:
    fallback = {"Low": -51.0, "Medium": -21.7, "High": -3.7}
    if df is None or "Cumulative_Degradation" not in df.columns:
        return fallback
    vals = pd.to_numeric(df["Cumulative_Degradation"], errors="coerce").dropna()
    if vals.empty:
        return fallback
    return {
        "Low": float(vals.quantile(0.25)),
        "Medium": float(vals.quantile(0.50)),
        "High": float(vals.quantile(0.75)),
    }


def options_from_dataset(df: pd.DataFrame | None, col: str, fallback: list[str]) -> list[str]:
    if df is None or col not in df.columns:
        return fallback
    values = sorted(str(v) for v in df[col].dropna().unique())
    return values or fallback


def coerce_input_row(values: dict[str, Any], columns: list[str], defaults: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for col in columns:
        value = values.get(col, defaults.get(col))
        if col in {"Driver", "Compound", "Race"}:
            row[col] = "UNKNOWN" if value is None or pd.isna(value) else str(value)
        elif col in {"LapNumber", "Stint", "Position", "Year"}:
            row[col] = int(round(float(value)))
        else:
            row[col] = float(value)
    return pd.DataFrame([row], columns=columns)


def predict(payload: dict[str, Any], row: pd.DataFrame) -> dict[str, Any]:
    pipe = payload["pipeline"]
    threshold = float(payload.get("threshold", 0.5))
    if hasattr(pipe, "predict_proba"):
        probability = float(pipe.predict_proba(row)[:, 1][0])
    else:
        probability = float(pipe.predict(row)[0])
    prediction = int(probability >= threshold)
    return {
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "label": "Pit Next Lap" if prediction else "Stay Out",
    }


def confidence_label(probability: float, threshold: float) -> str:
    distance = abs(probability - threshold)
    if distance >= 0.25:
        return "High"
    if distance >= 0.12:
        return "Medium"
    return "Low"


def strategy_explanation(probability: float, prediction: int) -> str:
    if prediction and probability >= 0.75:
        return "The current tire age, degradation, and race progress suggest that a pit stop is strategically likely soon."
    if prediction:
        return "The model sees enough pit-stop signal in the current lap context to recommend watching for a stop next lap."
    if probability <= 0.25:
        return "The model suggests the driver is likely to continue the current stint."
    return "The model leans toward staying out, but the probability is close enough to monitor tire life and lap-time loss."


def plot_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=CANVAS,
        plot_bgcolor=CANVAS,
        font={"color": TEXT, "family": "Inter, sans-serif"},
        margin={"l": 18, "r": 18, "t": 42, "b": 36},
        legend={"font": {"color": BODY}},
    )
    fig.update_xaxes(gridcolor="#343434", zerolinecolor="#343434", linecolor="#555555")
    fig.update_yaxes(gridcolor="#343434", zerolinecolor="#343434", linecolor="#555555")
    return fig


def page_home(df: pd.DataFrame | None, payload: dict[str, Any] | None, model_name: str | None) -> None:
    st.markdown(
        """
        <div class="rs-hero">
            <div class="rs-kicker">Formula 1 Strategy Dashboard</div>
            <h1>Race Strategist</h1>
            <div class="rs-subtitle">F1 Pit Stop Prediction Prototype Using Classical Machine Learning</div>
            <div class="rs-copy">
                Given the current race situation, this prototype estimates how likely a driver is to pit on the next lap.
                It is designed for lecturers, demo audiences, and non-technical users who want to understand how
                lap-level race data can support strategy decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        html_card(
            "What the model predicts",
            "A binary outcome: <strong>Pit Next Lap</strong> or <strong>Stay Out</strong>, using current lap features such as tire life, compound, race progress, position, and lap-time delta.",
        )
    with c2:
        html_card(
            "Why it matters",
            "Pit timing can decide track position, undercut opportunities, tire performance, and race outcome. Even a simple prediction can make strategy tradeoffs easier to discuss.",
        )
    with c3:
        html_card(
            "How it works",
            "The app sends raw lap conditions into a saved classical ML pipeline, then applies the model's tuned decision threshold to produce a strategy signal.",
        )

    section_break()
    c1, c2, c3, c4 = st.columns(4)
    rows = "Unavailable" if df is None else f"{len(df):,}"
    races = "Unavailable" if df is None or "Race" not in df.columns else f"{df['Race'].nunique():,}"
    drivers = "Unavailable" if df is None or "Driver" not in df.columns else f"{df['Driver'].nunique():,}"
    model = model_name or "Unavailable"
    with c1:
        metric_card("Dataset Rows", rows)
    with c2:
        metric_card("Races", races)
    with c3:
        metric_card("Drivers", drivers)
    with c4:
        metric_card("Loaded Model", model.replace(".joblib", ""))

    section_break()
    section_header(
        "How to Use",
        "Demo flow",
        "Start with the prediction page for a simple race scenario, then use Dataset Insights to explain the data and Model Performance to discuss the strengths and limitations of the trained model.",
    )

    c1, c2 = st.columns(2)
    with c1:
        html_card(
            "1. Try Quick Scenario",
            "Choose tire compound, tire age, race progress, position, lap-time delta, degradation level, and stint. This mode is best for a live project demo.",
        )
    with c2:
        html_card(
            "2. Inspect Advanced Input",
            "Use all model features when you want to show the exact schema behind the saved pipeline and discuss how lap-level features influence prediction.",
        )

    if payload is None:
        st.warning("Prediction is currently unavailable because a trained model artifact could not be loaded.")


def render_prediction_result(result: dict[str, Any]) -> None:
    probability = result["probability"]
    threshold = result["threshold"]
    prediction = result["prediction"]
    confidence = confidence_label(probability, threshold)
    explanation = strategy_explanation(probability, prediction)

    st.markdown(
        f"""
        <div class="rs-result">
            <div class="rs-tag">Prediction</div>
            <div class="rs-result-class">{result["label"]}</div>
            <div class="rs-copy">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Pit Probability", f"{probability:.1%}", "Model probability for next-lap pit")
    with c2:
        metric_card("Decision Threshold", f"{threshold:.3f}", "Saved tuned threshold")
    with c3:
        metric_card("Confidence", confidence, "Distance from model threshold")

    st.info(
        "This is a prototype prediction based on historical lap-level data. "
        "It should be interpreted as a decision-support tool, not a perfect race strategy engine."
    )


def page_predict(df: pd.DataFrame | None, payload: dict[str, Any] | None, model_name: str | None, model_error: str | None) -> None:
    section_header(
        "Prediction",
        "Will the driver pit on the next lap?",
        "Choose a simple scenario for a fast demo, or use advanced input to match the saved model schema feature by feature.",
    )

    defaults = load_sample_or_defaults(df)
    features = model_feature_columns(payload)

    if payload is None:
        st.error(model_error or "Prediction is unavailable because no model artifact could be loaded.")
    else:
        st.caption(f"Loaded model artifact: `{model_name}`")

    if df is None:
        st.warning("Dataset was not found. The app is using safe fallback defaults for input values and degradation levels.")

    tab_quick, tab_advanced = st.tabs(["Quick Scenario", "Advanced Input"])

    with tab_quick:
        levels = degradation_levels(df)
        compounds = [c for c in COMPOUNDS if c in options_from_dataset(df, "Compound", COMPOUNDS)] or COMPOUNDS
        c1, c2, c3 = st.columns(3)
        with c1:
            compound = st.selectbox("Tire compound", compounds, index=compounds.index(defaults.get("Compound", "HARD")) if defaults.get("Compound", "HARD") in compounds else 0)
            tyre_life = st.slider("Tire life", 0, 60, int(min(max(float(defaults.get("TyreLife", 13)), 0), 60)))
            stint = st.number_input("Stint", min_value=1, max_value=6, value=int(min(max(defaults.get("Stint", 2), 1), 6)), step=1)
        with c2:
            race_progress_pct = st.slider("Race progress", 0, 100, int(round(float(defaults.get("RaceProgress", 0.42)) * 100)))
            position = st.slider("Current position", 1, 20, int(min(max(defaults.get("Position", 10), 1), 20)))
            degradation_label = st.selectbox("Degradation level", ["Low", "Medium", "High"], index=1)
        with c3:
            lap_delta = st.slider("Lap time delta", -5.0, 10.0, float(np.clip(defaults.get("LapTime_Delta", 0.0), -5.0, 10.0)), step=0.1)
            race_options = options_from_dataset(df, "Race", [defaults.get("Race", "Azerbaijan Grand Prix")])
            driver_options = options_from_dataset(df, "Driver", [defaults.get("Driver", "NOR")])
            race = st.selectbox("Race context", race_options, index=race_options.index(defaults["Race"]) if defaults["Race"] in race_options else 0)
            driver = st.selectbox("Driver context", driver_options, index=driver_options.index(defaults["Driver"]) if defaults["Driver"] in driver_options else 0)

        race_progress = race_progress_pct / 100
        max_lap = int(df["LapNumber"].max()) if df is not None and "LapNumber" in df.columns else 78
        max_tyre_life = float(df["TyreLife"].max()) if df is not None and "TyreLife" in df.columns else 78.0
        quick_values = dict(defaults)
        quick_values.update(
            {
                "Driver": driver,
                "Race": race,
                "Compound": compound,
                "TyreLife": float(tyre_life),
                "RaceProgress": race_progress,
                "Position": int(position),
                "LapTime_Delta": float(lap_delta),
                "Cumulative_Degradation": levels[degradation_label],
                "Stint": int(stint),
                "LapNumber": max(1, int(round(max_lap * max(race_progress, 0.01)))),
                "Normalized_TyreLife": float(np.clip(tyre_life / max(max_tyre_life, 1.0), 0, 1)),
            }
        )

        if st.button("Predict Quick Scenario", key="quick_predict", disabled=payload is None):
            try:
                row = coerce_input_row(quick_values, features, defaults)
                result = predict(payload, row)  # type: ignore[arg-type]
                render_prediction_result(result)
                with st.expander("Model input row"):
                    st.dataframe(row, use_container_width=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with tab_advanced:
        advanced_values: dict[str, Any] = {}
        race_options = options_from_dataset(df, "Race", [defaults.get("Race", "Azerbaijan Grand Prix")])
        driver_options = options_from_dataset(df, "Driver", [defaults.get("Driver", "NOR")])
        compound_options = COMPOUNDS

        cols = st.columns(3)
        for i, feature in enumerate(features):
            with cols[i % 3]:
                default = defaults.get(feature)
                if feature == "Driver":
                    advanced_values[feature] = st.selectbox(feature, driver_options, index=driver_options.index(default) if default in driver_options else 0)
                elif feature == "Race":
                    advanced_values[feature] = st.selectbox(feature, race_options, index=race_options.index(default) if default in race_options else 0)
                elif feature == "Compound":
                    advanced_values[feature] = st.selectbox(feature, compound_options, index=compound_options.index(default) if default in compound_options else 2)
                elif feature in {"LapNumber", "Stint", "Position", "Year"}:
                    fallback_min, fallback_max = {
                        "LapNumber": (1, 80),
                        "Stint": (1, 8),
                        "Position": (1, 20),
                        "Year": (2022, 2026),
                    }[feature]
                    lo, hi = numeric_bounds(df, feature, (fallback_min, fallback_max))
                    advanced_values[feature] = st.number_input(
                        feature,
                        min_value=int(np.floor(lo)),
                        max_value=int(np.ceil(hi)),
                        value=int(np.clip(int(round(float(default))), int(np.floor(lo)), int(np.ceil(hi)))),
                        step=1,
                    )
                else:
                    fallback = {
                        "TyreLife": (0.0, 80.0),
                        "LapTime (s)": (60.0, 140.0),
                        "LapTime_Delta": (-30.0, 30.0),
                        "Cumulative_Degradation": (-150.0, 80.0),
                        "RaceProgress": (0.0, 1.0),
                        "Normalized_TyreLife": (0.0, 1.0),
                        "Position_Change": (-20.0, 20.0),
                    }.get(feature, (-100.0, 100.0))
                    lo, hi = numeric_bounds(df, feature, fallback)
                    if feature in {"RaceProgress", "Normalized_TyreLife"}:
                        lo, hi = 0.0, 1.0
                    advanced_values[feature] = st.number_input(
                        feature,
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(np.clip(float(default), lo, hi)),
                        step=0.01,
                        format="%.4f" if feature in {"RaceProgress", "Normalized_TyreLife"} else "%.3f",
                    )

        if st.button("Predict Advanced Input", key="advanced_predict", disabled=payload is None):
            try:
                row = coerce_input_row(advanced_values, features, defaults)
                result = predict(payload, row)  # type: ignore[arg-type]
                render_prediction_result(result)
                with st.expander("Model input row"):
                    st.dataframe(row, use_container_width=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


def page_dataset(df: pd.DataFrame | None) -> None:
    section_header(
        "Dataset",
        "Lap-level race data",
        "This page summarizes the historical race data used to train and evaluate the pit-stop prediction prototype.",
    )

    if df is None:
        st.error(f"Dataset not found at `{DATA_PATH}`.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows", f"{len(df):,}")
    with c2:
        metric_card("Features", f"{max(df.shape[1] - 1, 0):,}", "excluding target")
    with c3:
        races = df["Race"].nunique() if "Race" in df.columns else 0
        metric_card("Races", f"{races:,}")
    with c4:
        drivers = df["Driver"].nunique() if "Driver" in df.columns else 0
        metric_card("Drivers", f"{drivers:,}")

    if TARGET in df.columns:
        target_counts = df[TARGET].map({0: "Stay Out", 1: "Pit Next Lap"}).value_counts().reset_index()
        target_counts.columns = ["Outcome", "Rows"]
        fig = px.bar(target_counts, x="Outcome", y="Rows", color="Outcome", color_discrete_map={"Stay Out": "#666666", "Pit Next Lap": RED}, title="Target Distribution")
        st.plotly_chart(plot_layout(fig), use_container_width=True)
        caption_text("Pit stops are rare compared with normal stay-out laps, which makes this an imbalanced classification problem.")

    c1, c2 = st.columns(2)
    with c1:
        if "Compound" in df.columns:
            compound_counts = df["Compound"].fillna("UNKNOWN").value_counts().reset_index()
            compound_counts.columns = ["Compound", "Rows"]
            fig = px.bar(compound_counts, x="Compound", y="Rows", color="Compound", color_discrete_sequence=[RED, "#ffffff", "#8f8f8f", "#4c98b9", "#f6e500", "#666666"], title="Compound Distribution")
            st.plotly_chart(plot_layout(fig), use_container_width=True)
            caption_text("This shows which tire compounds appear most often in the lap-level dataset.")
    with c2:
        if "TyreLife" in df.columns:
            fig = px.histogram(df, x="TyreLife", nbins=36, color_discrete_sequence=[RED], title="Tire Life Distribution")
            st.plotly_chart(plot_layout(fig), use_container_width=True)
            caption_text("Tire life gives a simple view of how long drivers usually stay on a stint before changing tires.")

    c1, c2 = st.columns(2)
    with c1:
        if "RaceProgress" in df.columns:
            fig = px.histogram(df, x="RaceProgress", nbins=32, color_discrete_sequence=["#ffffff"], title="Race Progress Distribution")
            fig.update_traces(marker_line_color=CANVAS, marker_line_width=1)
            st.plotly_chart(plot_layout(fig), use_container_width=True)
            caption_text("Race progress normalizes lap number so early, middle, and late race phases can be compared.")
    with c2:
        if {"Compound", "Cumulative_Degradation"}.issubset(df.columns):
            deg = df.groupby("Compound", dropna=False)["Cumulative_Degradation"].mean().reset_index()
            fig = px.bar(deg, x="Compound", y="Cumulative_Degradation", color="Compound", color_discrete_sequence=[RED, "#ffffff", "#8f8f8f", "#4c98b9", "#f6e500", "#666666"], title="Average Degradation by Compound")
            st.plotly_chart(plot_layout(fig), use_container_width=True)
            caption_text("Average degradation by compound helps explain why tire type can influence pit timing.")

    if {"TyreLife", TARGET}.issubset(df.columns):
        tmp = df.copy()
        tmp["Tire Life Bucket"] = pd.cut(
            tmp["TyreLife"],
            bins=[0, 5, 10, 15, 20, 30, 45, np.inf],
            labels=["1-5", "6-10", "11-15", "16-20", "21-30", "31-45", "46+"],
            include_lowest=True,
        )
        pit_rate = tmp.groupby("Tire Life Bucket", observed=False)[TARGET].mean().reset_index()
        pit_rate["Pit Rate"] = pit_rate[TARGET] * 100
        fig = px.line(pit_rate, x="Tire Life Bucket", y="Pit Rate", markers=True, color_discrete_sequence=[RED], title="Pit Rate by Tire Life Bucket")
        st.plotly_chart(plot_layout(fig), use_container_width=True)
        caption_text("This chart shows how the historical chance of a next-lap pit changes as tires get older.")


def page_performance() -> None:
    section_header(
        "Model",
        "Performance on held-out races",
        "Metrics are loaded from saved evaluation reports. Missing artifacts are reported as missing rather than replaced with invented values.",
    )

    summary = load_json(str(REPORTS_DIR / "training_summary.json"))
    comparison = load_csv(str(REPORTS_DIR / "model_comparison.csv"))

    if summary:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Best Model", str(summary.get("best_model", "Unknown")).replace("_", " ").title())
        with c2:
            val = summary.get("best_pr_auc")
            metric_card("PR-AUC", f"{val:.3f}" if isinstance(val, (int, float)) else "n/a")
        with c3:
            val = summary.get("best_f1")
            metric_card("F1 Score", f"{val:.3f}" if isinstance(val, (int, float)) else "n/a")
        with c4:
            val = summary.get("best_threshold")
            metric_card("Threshold", f"{val:.3f}" if isinstance(val, (int, float)) else "n/a")
    else:
        st.warning("Saved training summary was not found.")

    if comparison is not None and not comparison.empty:
        display_cols = [c for c in ["model", "pr_auc", "roc_auc", "f1", "precision", "recall", "brier", "threshold"] if c in comparison.columns]
        section_break()
        section_title("Classical ML Model Comparison")
        st.dataframe(comparison[display_cols], use_container_width=True, hide_index=True)

        metric_cols = [c for c in ["pr_auc", "roc_auc", "f1", "precision", "recall"] if c in comparison.columns]
        if {"model", "pr_auc"}.issubset(comparison.columns):
            melted = comparison[["model"] + metric_cols].melt(id_vars="model", var_name="Metric", value_name="Score")
            fig = px.bar(melted, x="model", y="Score", color="Metric", barmode="group", title="Model Scores", color_discrete_sequence=[RED, "#ffffff", "#8f8f8f", "#4c98b9", "#f6e500"])
            st.plotly_chart(plot_layout(fig), use_container_width=True)
    else:
        st.warning("Model comparison table was not found.")

    section_break()
    section_title("Confusion Matrix Guide")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        html_card("True Negative", "Correctly predicted <strong>Stay Out</strong>.")
    with c2:
        html_card("False Positive", "Predicted <strong>Pit</strong>, but the driver stayed out.")
    with c3:
        html_card("False Negative", "Predicted <strong>Stay Out</strong>, but the driver pitted.")
    with c4:
        html_card("True Positive", "Correctly predicted <strong>Pit Next Lap</strong>.")

    figure_paths = [
        ("Confusion Matrices", FIGURES_DIR / "confusion_matrices.png"),
        ("Precision-Recall Curves", FIGURES_DIR / "pr_curves.png"),
        ("ROC Curves", FIGURES_DIR / "roc_curves.png"),
        ("Random Forest Feature Importance", FIGURES_DIR / "feature_importance_rf.png"),
    ]

    for title, path in figure_paths:
        if path.exists():
            section_title(title)
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"`{path.relative_to(ROOT)}` was not found.")

    if summary and summary.get("per_race_recall_best"):
        section_title("Per-Race Recall")
        st.dataframe(pd.DataFrame(summary["per_race_recall_best"]), use_container_width=True, hide_index=True)


def main() -> None:
    apply_theme()
    df = load_dataset()
    payload, model_name, model_error = load_model_payload()

    with st.sidebar:
        st.markdown('<div class="rs-kicker">Race Strategist</div>', unsafe_allow_html=True)
        section_title("Pit Stop Prediction")
        page = st.radio(
            "Navigation",
            ["Home", "Predict Pit Stop", "Dataset Insights", "Model Performance"],
            label_visibility="collapsed",
        )
        section_break()
        if df is not None:
            st.caption(f"Dataset: {len(df):,} laps")
        else:
            st.caption("Dataset: unavailable")
        st.caption(f"Model: {model_name or 'unavailable'}")

    if page == "Home":
        page_home(df, payload, model_name)
    elif page == "Predict Pit Stop":
        page_predict(df, payload, model_name, model_error)
    elif page == "Dataset Insights":
        page_dataset(df)
    else:
        page_performance()


if __name__ == "__main__":
    main()
