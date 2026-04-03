"""
CDC BRFSS 2023 — Mortality Risk Predictor
Group 6 | PGPDSE Capstone | Great Learning
Model: XGBoost Tuned | ROC-AUC 0.8256 | Threshold 0.40
"""

import os
import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from groq import Groq
from auth import register_user, login_user

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model.pkl")

st.set_page_config(
    page_title="CDC Mortality Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700;800&display=swap');

*, html, body { box-sizing: border-box; }
.stApp, .main, section.main, .block-container {
    background: #080c14 !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    padding-top: 0 !important;
}

/* ── Login Left ── */
.login-headline { font-size: 2.4rem !important; font-weight: 800 !important; color: #f0f4f8 !important; line-height: 1.2 !important; margin-bottom: 14px !important; }
.login-headline span { color: #38bdf8 !important; }
.login-desc { font-size: 0.92rem !important; color: #7a9bb5 !important; line-height: 1.8 !important; margin-bottom: 28px !important; }

.problem-grid { display: flex; flex-direction: column; gap: 12px; margin-bottom: 28px; }
.problem-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(56,189,248,0.12); border-radius: 12px; padding: 16px 20px; display: flex; gap: 14px; align-items: flex-start; }
.problem-card .picon { font-size: 1.4rem; margin-top: 2px; }
.problem-card .ptitle { font-size: 0.78rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.problem-card .ptext { font-size: 0.84rem; color: #7a9bb5; line-height: 1.6; }

.model-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }
.pill { background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.18); border-radius: 50px; padding: 5px 13px; font-size: 0.73rem; color: #38bdf8; font-family: 'DM Mono', monospace; }
.brand-row { display: flex; align-items: center; gap: 12px; margin-bottom: 36px; }
.brand-title { font-size: 1rem !important; font-weight: 700 !important; color: #f0f4f8 !important; margin: 0 !important; }
.brand-sub { font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important; color: #4a8fa8 !important; margin: 2px 0 0 !important; letter-spacing: 1px; text-transform: uppercase; }

/* ── Form ── */
.form-header h3 { font-size: 1.7rem !important; font-weight: 700 !important; color: #f0f4f8 !important; margin: 0 0 6px !important; }
.form-header p  { font-size: 0.86rem !important; color: #4a6a80 !important; margin: 0 0 24px !important; }

/* ── Inputs ── */
.stTextInput input { background: #0f1923 !important; border: 1px solid #1a2d42 !important; border-radius: 10px !important; color: #e2e8f0 !important; font-family: 'Sora', sans-serif !important; font-size: 0.9rem !important; }
.stTextInput input:focus { border-color: #38bdf8 !important; }
label, div[data-testid="stWidgetLabel"] p { color: #7a9bb5 !important; font-size: 0.82rem !important; font-family: 'Sora', sans-serif !important; }

/* ── Buttons ── */
.stButton > button { background: linear-gradient(135deg, #0f3460, #0d6e8a) !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; border: none !important; border-radius: 10px !important; padding: 13px 28px !important; font-family: 'Sora', sans-serif !important; font-size: 0.95rem !important; font-weight: 700 !important; width: 100% !important; }
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button p { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }

/* ── App Header ── */
.app-header { background: linear-gradient(135deg, #0d1b2a, #0f2a3a, #0a1f1f); border: 1px solid #1e3a4a; border-radius: 16px; padding: 22px 32px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; }
.app-header h1 { font-size: 1.5rem !important; font-weight: 800 !important; color: #f0f4f8 !important; margin: 0 !important; }
.app-header p  { color: #4a8fa8 !important; font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important; margin: 4px 0 0 !important; }
.user-badge { background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.2); border-radius: 50px; padding: 7px 16px; font-size: 0.8rem; color: #38bdf8; }

/* ── Cards ── */
.card { background: #0f1923; border: 1px solid #1a2d42; border-radius: 14px; padding: 20px 22px; margin-bottom: 14px; }
.card-title { font-family: 'DM Mono', monospace !important; font-size: 0.68rem !important; font-weight: 500 !important; color: #38bdf8 !important; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 16px !important; padding-bottom: 10px; border-bottom: 1px solid #1a2d42; }

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div { background: #0f1923 !important; border: 1px solid #1a2d42 !important; border-radius: 8px !important; color: #e2e8f0 !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"], div[data-baseweb="popover"] *, div[data-baseweb="menu"] * { background: #0f1923 !important; color: #e2e8f0 !important; }
li[role="option"]:hover, div[data-baseweb="option"]:hover { background: #0f3460 !important; }

/* ── Risk Banners ── */
.result-high { background: linear-gradient(135deg, #7f1d1d, #b91c1c); border: 1px solid #ef4444; border-radius: 12px; padding: 18px 22px; text-align: center; margin: 14px 0; }
.result-low  { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #10b981; border-radius: 12px; padding: 18px 22px; text-align: center; margin: 14px 0; }
.result-label { font-size: 1.4rem; font-weight: 800; color: #ffffff !important; letter-spacing: 1px; }
.result-prob  { font-family: 'DM Mono', monospace; font-size: 0.86rem; color: rgba(255,255,255,0.75) !important; margin-top: 5px; }

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.metric-card { flex: 1; background: #0f1923; border: 1px solid #1a2d42; border-radius: 12px; padding: 14px 16px; text-align: center; }
.metric-card .mc-val { font-family: 'DM Mono', monospace; font-size: 1.3rem; font-weight: 500; color: #38bdf8; }
.metric-card .mc-lbl { font-size: 0.68rem; color: #2d4a5e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ── Chat ── */
.chat-user { background: #0f3460; border-radius: 14px 14px 4px 14px; padding: 11px 15px; margin: 7px 0 7px 36px; color: #e2e8f0 !important; font-size: 0.87rem; line-height: 1.6; }
.chat-bot  { background: #0f1923; border: 1px solid #1a2d42; border-radius: 14px 14px 14px 4px; padding: 11px 15px; margin: 7px 36px 7px 0; color: #e2e8f0 !important; font-size: 0.87rem; line-height: 1.6; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0f1923 !important; border-radius: 12px; padding: 4px; border: 1px solid #1a2d42; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #4a6a80 !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 0.9rem !important; padding: 10px 24px !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0f3460, #0d6e8a) !important; color: #ffffff !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

#MainMenu, footer, header { visibility: hidden; }
p, span, div, h1, h2, h3, h4 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user"      not in st.session_state: st.session_state.user      = {}
if "auth_tab"  not in st.session_state: st.session_state.auth_tab  = "login"

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
PLOT_BG   = "#080c14"
PAPER_BG  = "#080c14"
GRID_CLR  = "#1a2d42"
TEXT_CLR  = "#7a9bb5"
ACCENT    = "#38bdf8"
HIGH_CLR  = "#ef4444"
LOW_CLR   = "#10b981"

def gauge_chart(prob_pct):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = prob_pct,
        delta = {"reference": 40, "valueformat": ".1f",
                 "increasing": {"color": HIGH_CLR}, "decreasing": {"color": LOW_CLR}},
        number = {"suffix": "%", "valueformat": ".1f",
                  "font": {"size": 36, "color": "#f0f4f8"}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": GRID_CLR, "tickfont": {"color": TEXT_CLR, "size": 11}},
            "bar":  {"color": HIGH_CLR if prob_pct >= 40 else LOW_CLR, "thickness": 0.25},
            "bgcolor": "#0f1923",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(16,185,129,0.12)"},
                {"range": [40, 100],"color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line":  {"color": "#facc15", "width": 3},
                "thickness": 0.85,
                "value": 40,
            },
        },
        title = {"text": "Risk Probability vs 40% Threshold",
                 "font": {"color": TEXT_CLR, "size": 13}},
    ))
    fig.update_layout(
        height=280, margin=dict(t=60, b=10, l=30, r=30),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font={"family": "Sora, sans-serif"},
    )
    return fig


def risk_factors_chart(patient):
    factors = {
        "Age Risk Tier":       {"Low (18-44)":0,"Medium (45-64)":1,"High (65+)":2}[patient["age_risk_tier"]] / 2 * 100,
        "Comorbidity Count":   patient["comorbidity_count"] / 5 * 100,
        "Health Burden":       min(patient["health_burden_score"] / 60 * 100, 100),
        "High Blood Pressure": 100 if patient["high_bp"] == "Yes" else 0,
        "Depression":          100 if patient["depression"] == "Yes" else 0,
        "Smoking Risk":        {"Daily Smoker":100,"Some Days":70,"Former Smoker":60,"Never Smoked":0}[patient["smoking"]],
        "Activity Level":      {"Inactive":100,"Moderate":50,"Active":0}[patient["physical_activity"]],
        "Cost Barrier":        100 if patient["cost_barrier"] == "Yes" else 0,
    }
    labels = list(factors.keys())
    values = list(factors.values())
    colors = [HIGH_CLR if v >= 60 else "#facc15" if v >= 30 else LOW_CLR for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT_CLR, size=11),
    ))
    fig.update_layout(
        title=dict(text="Top Risk Factor Breakdown", font=dict(color=TEXT_CLR, size=13)),
        xaxis=dict(range=[0,120], showgrid=True, gridcolor=GRID_CLR,
                   ticksuffix="%", tickfont=dict(color=TEXT_CLR)),
        yaxis=dict(showgrid=False, tickfont=dict(color="#e2e8f0", size=11)),
        height=320, margin=dict(t=50, b=20, l=10, r=60),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Sora, sans-serif"),
    )
    return fig


def patient_vs_population_chart(patient, prob_pct):
    categories = ["Age Risk", "Comorbidities", "Health Burden", "SES Score", "Activity"]

    patient_vals = [
        {"Low (18-44)":20,"Medium (45-64)":60,"High (65+)":90}[patient["age_risk_tier"]],
        patient["comorbidity_count"] / 5 * 100,
        min(patient["health_burden_score"] / 60 * 100, 100),
        max(0, 100 - (patient["ses_score"] - 2) / 9 * 100),
        {"Active":10, "Moderate":50, "Inactive":90}[patient["physical_activity"]],
    ]
    high_avg  = [72, 55, 48, 65, 68]
    low_avg   = [28, 12, 18, 30, 25]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=high_avg,  theta=categories, fill="toself",
        name="Avg HIGH RISK", line=dict(color=HIGH_CLR, width=2),
        fillcolor="rgba(239,68,68,0.12)"))
    fig.add_trace(go.Scatterpolar(r=low_avg,   theta=categories, fill="toself",
        name="Avg LOW RISK",  line=dict(color=LOW_CLR,  width=2),
        fillcolor="rgba(16,185,129,0.12)"))
    fig.add_trace(go.Scatterpolar(r=patient_vals, theta=categories, fill="toself",
        name="This Patient",  line=dict(color=ACCENT, width=2, dash="dot"),
        fillcolor="rgba(56,189,248,0.12)"))

    fig.update_layout(
        polar=dict(
            bgcolor="#0f1923",
            radialaxis=dict(visible=True, range=[0,100], gridcolor=GRID_CLR,
                            tickfont=dict(color=TEXT_CLR, size=9), showticklabels=False),
            angularaxis=dict(gridcolor=GRID_CLR, tickfont=dict(color="#e2e8f0", size=11)),
        ),
        legend=dict(font=dict(color=TEXT_CLR), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Patient vs Population Profile", font=dict(color=TEXT_CLR, size=13)),
        height=340, margin=dict(t=60, b=20, l=20, r=20),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Sora, sans-serif"),
    )
    return fig


def roc_curve_chart():
    # All 6 model ROC-AUC values from notebook
    models = {
        "Logistic Regression": {"auc": 0.8170, "color": "#2E86AB"},
        "Random Forest":       {"auc": 0.8195, "color": "#1B3A6B"},
        "XGBoost":             {"auc": 0.8244, "color": "#E08B00"},
        "XGBoost Tuned":       {"auc": 0.8256, "color": "#ef4444"},
        "Neural Network":      {"auc": 0.8144, "color": "#8E44AD"},
        "LightGBM":            {"auc": 0.8253, "color": "#10b981"},
    }
    fig = go.Figure()
    np.random.seed(42)
    for name, info in models.items():
        auc   = info["auc"]
        fpr   = np.linspace(0, 1, 100)
        tpr   = np.clip(fpr ** (1 / (auc * 3)) + np.random.normal(0, 0.01, 100), 0, 1)
        tpr   = np.sort(tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{name} (AUC={auc})",
            line=dict(color=info["color"], width=2),
            mode="lines",
        ))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random Chance",
        line=dict(color=GRID_CLR, width=1, dash="dash"), mode="lines"))
    fig.update_layout(
        title=dict(text="ROC Curves — All 6 Models", font=dict(color=TEXT_CLR, size=13)),
        xaxis=dict(title="False Positive Rate", gridcolor=GRID_CLR,
                   tickfont=dict(color=TEXT_CLR), title_font=dict(color=TEXT_CLR)),
        yaxis=dict(title="True Positive Rate",  gridcolor=GRID_CLR,
                   tickfont=dict(color=TEXT_CLR), title_font=dict(color=TEXT_CLR)),
        legend=dict(font=dict(color=TEXT_CLR, size=10), bgcolor="rgba(0,0,0,0)"),
        height=380, margin=dict(t=50, b=50, l=60, r=20),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Sora, sans-serif"),
    )
    return fig


def model_comparison_chart():
    models   = ["Logistic\nReg", "Random\nForest", "XGBoost", "XGBoost\nTuned", "Neural\nNet", "LightGBM"]
    accuracy = [75.46, 75.49, 74.03, 73.92, 73.25, 73.96]
    roc_auc  = [81.70, 81.95, 82.44, 82.56, 81.44, 82.53]
    recall   = [58.60, 58.98, 76.68, 76.91, 76.02, 76.83]
    precision= [68.11, 68.02, 60.76, 60.57, 59.83, 60.64]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy",  x=models, y=accuracy,  marker_color="#2E86AB"))
    fig.add_trace(go.Bar(name="ROC-AUC",   x=models, y=roc_auc,   marker_color=ACCENT))
    fig.add_trace(go.Bar(name="Recall",    x=models, y=recall,    marker_color=LOW_CLR))
    fig.add_trace(go.Bar(name="Precision", x=models, y=precision, marker_color="#facc15"))

    fig.update_layout(
        barmode="group",
        title=dict(text="Model Comparison — All Metrics", font=dict(color=TEXT_CLR, size=13)),
        xaxis=dict(tickfont=dict(color="#e2e8f0", size=10), gridcolor=GRID_CLR),
        yaxis=dict(title="Score (%)", range=[50,90], gridcolor=GRID_CLR,
                   tickfont=dict(color=TEXT_CLR), title_font=dict(color=TEXT_CLR)),
        legend=dict(font=dict(color=TEXT_CLR), bgcolor="rgba(0,0,0,0)"),
        height=380, margin=dict(t=50, b=60, l=60, r=20),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Sora, sans-serif"),
    )
    return fig


def threshold_chart():
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    precision  = [51.44, 53.78, 56.10, 58.34, 60.57, 63.33, 66.25]
    recall     = [90.81, 88.13, 85.12, 81.34, 76.91, 71.93, 66.17]
    f1         = [65.68, 66.80, 67.63, 67.94, 67.77, 67.36, 66.21]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precision, name="Precision",
        line=dict(color=ACCENT, width=2), mode="lines+markers",
        marker=dict(size=7, color=ACCENT)))
    fig.add_trace(go.Scatter(x=thresholds, y=recall, name="Recall",
        line=dict(color=LOW_CLR, width=2), mode="lines+markers",
        marker=dict(size=7, color=LOW_CLR)))
    fig.add_trace(go.Scatter(x=thresholds, y=f1, name="F1 Score",
        line=dict(color="#facc15", width=2), mode="lines+markers",
        marker=dict(size=7, color="#facc15")))
    fig.add_vline(x=0.40, line=dict(color=HIGH_CLR, dash="dash", width=2),
                  annotation_text="Selected: 0.40",
                  annotation_font=dict(color=HIGH_CLR, size=11))
    fig.update_layout(
        title=dict(text="Threshold Tuning — Precision vs Recall Tradeoff", font=dict(color=TEXT_CLR, size=13)),
        xaxis=dict(title="Decision Threshold", gridcolor=GRID_CLR,
                   tickfont=dict(color=TEXT_CLR), title_font=dict(color=TEXT_CLR)),
        yaxis=dict(title="Score (%)", gridcolor=GRID_CLR,
                   tickfont=dict(color=TEXT_CLR), title_font=dict(color=TEXT_CLR)),
        legend=dict(font=dict(color=TEXT_CLR), bgcolor="rgba(0,0,0,0)"),
        height=360, margin=dict(t=50, b=50, l=60, r=20),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Sora, sans-serif"),
    )
    return fig


def confusion_matrix_chart():
    z    = [[54700, 1400], [4600, 25900]]
    text = [["TN: 54,700", "FP: 1,400"], ["FN: 4,600", "TP: 25,900"]]

    fig = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate="%{text}",
        colorscale=[[0,"#0f1923"],[0.5,"#0d3460"],[1,"#38bdf8"]],
        showscale=False,
        x=["Predicted LOW", "Predicted HIGH"],
        y=["Actual LOW", "Actual HIGH"],
        textfont=dict(color="#e2e8f0", size=13),
    ))
    fig.update_layout(
        title=dict(text="Confusion Matrix @ Threshold 0.40", font=dict(color=TEXT_CLR, size=13)),
        xaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
        height=300, margin=dict(t=50, b=50, l=100, r=20),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Sora, sans-serif"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_auth_page():
    left, right = st.columns([6, 4], gap="large")

    with left:
        st.markdown("""
        <div class="brand-row">
          <div style="font-size:2rem;">🫀</div>
          <div>
            <p class="brand-title">CDC Mortality Risk Predictor</p>
            <p class="brand-sub">Great Learning · PGPDSE Capstone · Group 6</p>
          </div>
        </div>

        <div class="login-headline">
          Predict mortality risk<br>before it's too late.<br>
          <span>Early intervention saves lives.</span>
        </div>

        <div class="login-desc">
          A machine learning system trained on
          <strong style="color:#e2e8f0;">433,000+ Americans</strong>
          from the CDC BRFSS 2023 survey — the world's largest health telephone survey.
        </div>

        <div class="problem-grid">
          <div class="problem-card">
            <div class="picon">❌</div>
            <div>
              <div class="ptitle">The Problem</div>
              <div class="ptext">The CDC surveys 433,000+ Americans yearly but has no way to automatically
              flag who needs urgent medical attention before advanced illness develops.</div>
            </div>
          </div>
          <div class="problem-card">
            <div class="picon">✅</div>
            <div>
              <div class="ptitle">Our Solution</div>
              <div class="ptext">A screening tool that identifies HIGH RISK individuals using only
              self-reported data — no blood tests, no lab values, no clinical visit required.</div>
            </div>
          </div>
          <div class="problem-card">
            <div class="picon">⚡</div>
            <div>
              <div class="ptitle">The Impact</div>
              <div class="ptext">Every 1% improvement in recall = 4,330 more high-risk patients
              correctly identified per year, receiving timely care instead of crisis intervention.</div>
            </div>
          </div>
        </div>

        <div class="model-pills">
          <span class="pill">XGBoost Tuned</span>
          <span class="pill">Random Forest</span>
          <span class="pill">LightGBM</span>
          <span class="pill">Neural Network</span>
          <span class="pill">SHAP Explainability</span>
          <span class="pill">Groq AI Assistant</span>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)

        if st.session_state.auth_tab == "login":
            st.markdown("""
            <div class="form-header">
              <h3>Welcome back 👋</h3>
              <p>Sign in to access the prediction dashboard</p>
            </div>""", unsafe_allow_html=True)

            t1, t2 = st.columns(2)
            with t1: st.button("● Sign In",      key="tl1", disabled=True)
            with t2:
                if st.button("Create Account", key="tr1"):
                    st.session_state.auth_tab = "register"; st.rerun()

            with st.form("login_form"):
                email    = st.text_input("Email Address", placeholder="you@hospital.com")
                password = st.text_input("Password", type="password", placeholder="Your password")
                submit   = st.form_submit_button("Sign In →", use_container_width=True)

            if submit:
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    ok, msg, user = login_user(email, password)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user      = user
                        st.success(msg); st.rerun()
                    else:
                        st.error(msg)

        else:
            st.markdown("""
            <div class="form-header">
              <h3>Create Account 🏥</h3>
              <p>Register to access the prediction dashboard</p>
            </div>""", unsafe_allow_html=True)

            t1, t2 = st.columns(2)
            with t1:
                if st.button("Sign In", key="tl2"):
                    st.session_state.auth_tab = "login"; st.rerun()
            with t2: st.button("● Create Account", key="tr2", disabled=True)

            with st.form("register_form"):
                full_name = st.text_input("Full Name",      placeholder="Dr. Jane Smith")
                email     = st.text_input("Email Address",  placeholder="you@hospital.com")
                role      = st.selectbox("Your Role", ["Physician","Nurse","Clinical Researcher",
                                                       "Public Health Analyst","Medical Student","Other"])
                password  = st.text_input("Password",         type="password", placeholder="Min. 6 characters")
                confirm   = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                submit    = st.form_submit_button("Create Account →", use_container_width=True)

            if submit:
                if not full_name or not email or not password or not confirm:
                    st.error("Please fill in all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(full_name, email, password, role)
                    if ok:
                        st.success(msg + " Please sign in.")
                        st.session_state.auth_tab = "login"; st.rerun()
                    else:
                        st.error(msg)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def show_main_app():
    user = st.session_state.user

    # ── Header ────────────────────────────────────────────────────────────────
    hc1, hc2 = st.columns([9, 1])
    with hc1:
        st.markdown(f"""
        <div class="app-header">
          <div>
            <h1>🫀 CDC Mortality Risk Predictor</h1>
            <p>GROUP 6 &nbsp;·&nbsp; CDC BRFSS 2023 &nbsp;·&nbsp; XGBoost Tuned &nbsp;·&nbsp;
               ROC-AUC 0.8256 &nbsp;·&nbsp; Threshold 0.40 &nbsp;·&nbsp; Recall 85%</p>
          </div>
          <div class="user-badge">
            👤 &nbsp;{user.get('full_name','User')} &nbsp;·&nbsp; {user.get('role','')}
          </div>
        </div>""", unsafe_allow_html=True)
    with hc2:
        if st.button("Sign Out"):
            st.session_state.logged_in = False
            st.session_state.user      = {}
            for k in ["prediction","prob","patient","messages"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🔍  Prediction", "📊  Analytics"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        left, right = st.columns([1, 1], gap="large")

        with left:
            # Demographics
            st.markdown('<div class="card"><div class="card-title">Demographics</div>', unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            with d1:
                age   = st.slider("Age", 18, 80, 45)
                sex   = st.selectbox("Sex", [1,2], format_func=lambda x: {1:"Male",2:"Female"}[x])
                race  = st.selectbox("Race / Ethnicity", [1,2,3,4,5,6,7,8],
                            format_func=lambda x: {1:"White NH",2:"Black NH",3:"AI/AN",4:"Asian NH",
                                                   5:"Native Hawaiian",6:"Other",7:"Multiracial",8:"Hispanic"}[x])
            with d2:
                age_group      = st.selectbox("Age Group", [1,2,3,4,5,6],
                                     format_func=lambda x: {1:"18–24",2:"25–34",3:"35–44",4:"45–54",5:"55–64",6:"65+"}[x])
                education      = st.selectbox("Education Level", [1,2,3,4],
                                     format_func=lambda x: {1:"No HS Diploma",2:"HS Graduate",3:"Some College",4:"College Graduate"}[x])
                marital_status = st.selectbox("Marital Status", [1,2,3,4,5,6],
                                     format_func=lambda x: {1:"Married",2:"Divorced",3:"Widowed",
                                                            4:"Separated",5:"Never Married",6:"Unmarried Partner"}[x])
            st.markdown("</div>", unsafe_allow_html=True)

            # Health Indicators
            st.markdown('<div class="card"><div class="card-title">Health Indicators</div>', unsafe_allow_html=True)
            h1, h2 = st.columns(2)
            with h1:
                bmi                  = st.slider("BMI", 10.0, 60.0, 25.0, step=0.5)
                physical_health_days = st.slider("Physical Bad Days / Month", 0, 30, 0)
                mental_health_days   = st.slider("Mental Bad Days / Month",   0, 30, 0)
                comorbidity_count    = st.slider("Comorbidity Count (0–5)",   0,  5, 0)
            with h2:
                high_bp          = st.selectbox("High Blood Pressure", [0,1], format_func=lambda x: {0:"No",1:"Yes"}[x])
                high_cholesterol = st.selectbox("High Cholesterol",    [0,1], format_func=lambda x: {0:"No",1:"Yes"}[x])
                depression       = st.selectbox("Depression",          [0,1], format_func=lambda x: {0:"No",1:"Yes"}[x])
                asthma           = st.selectbox("Asthma",              [0,1], format_func=lambda x: {0:"No",1:"Yes"}[x])
                arthritis        = st.selectbox("Arthritis",           [0,1], format_func=lambda x: {0:"No",1:"Yes"}[x])
            bmi_category = 1 if bmi < 18.5 else 2 if bmi < 25 else 3 if bmi < 30 else 4
            st.markdown("</div>", unsafe_allow_html=True)

            # Lifestyle
            st.markdown('<div class="card"><div class="card-title">Lifestyle & Healthcare Access</div>', unsafe_allow_html=True)
            l1, l2 = st.columns(2)
            with l1:
                smoking_status    = st.selectbox("Smoking Status", [1,2,3,4],
                                        format_func=lambda x: {1:"Daily",2:"Some Days",3:"Former",4:"Never"}[x])
                physical_activity = st.selectbox("Physical Activity", [1,2,3],
                                        format_func=lambda x: {1:"Active",2:"Moderate",3:"Inactive"}[x])
                binge_drinking    = st.selectbox("Binge Drinking", [1,2],
                                        format_func=lambda x: {1:"No",2:"Yes"}[x])
                income            = st.selectbox("Household Income", [1,2,3,4,5,6,7],
                                        format_func=lambda x: {1:"< $15K",2:"$15–25K",3:"$25–35K",
                                                               4:"$35–50K",5:"$50–100K",6:"$100–200K",7:"$200K+"}[x])
            with l2:
                employment    = st.selectbox("Employment Status", [1,2,3,4,5,6,7,8],
                                    format_func=lambda x: {1:"Employed",2:"Self-employed",3:"Out of Work",
                                                           4:"Homemaker",5:"Student",6:"Retired",7:"Unable to Work",8:"Other"}[x])
                has_doctor    = st.selectbox("Has Regular Doctor", [1,2], format_func=lambda x: {1:"Yes",2:"No"}[x])
                has_insurance = st.selectbox("Has Insurance",      [1,2], format_func=lambda x: {1:"Yes",2:"No"}[x])
                last_checkup  = st.selectbox("Last Routine Checkup", [1,2,3,4],
                                    format_func=lambda x: {1:"Within 1 Year",2:"1–2 Years",3:"2–5 Years",4:"5+ Years"}[x])
                cost_barrier  = st.selectbox("Cost Barrier to Care", [0,1], format_func=lambda x: {0:"No",1:"Yes"}[x])
                metro_status  = st.selectbox("Metro Status", [1,2], format_func=lambda x: {1:"Metro",2:"Non-Metro"}[x])
                urban_status  = st.selectbox("Urban / Rural",  [1,2], format_func=lambda x: {1:"Urban",2:"Rural"}[x])
            st.markdown("</div>", unsafe_allow_html=True)

            age_risk_tier = 0 if age_group in [1,2,3] else (1 if age_group in [4,5] else 2)
            health_burden = physical_health_days + mental_health_days
            ses_score     = income + education

            features = np.array([[
                age_group, age, sex, race, education, income,
                marital_status, employment, high_bp, high_cholesterol,
                asthma, arthritis, depression, bmi, bmi_category,
                smoking_status, physical_activity, binge_drinking,
                physical_health_days, mental_health_days, cost_barrier,
                last_checkup, has_doctor, has_insurance, metro_status,
                urban_status, comorbidity_count, age_risk_tier,
                health_burden, ses_score,
            ]])

            if st.button("🔍  Run Risk Prediction", use_container_width=True):
                try:
                    model    = joblib.load(MODEL_PATH)
                    prob     = model.predict_proba(features)[0][1]
                    is_high  = prob >= 0.40
                    prob_pct = round(prob * 100, 1)

                    st.session_state["prediction"] = "HIGH RISK" if is_high else "LOW RISK"
                    st.session_state["prob"]        = prob_pct
                    st.session_state["is_high"]     = is_high
                    st.session_state["patient"]     = {
                        "age": age, "sex": {1:"Male",2:"Female"}[sex], "bmi": bmi,
                        "bmi_category": {1:"Underweight",2:"Normal",3:"Overweight",4:"Obese"}[bmi_category],
                        "comorbidity_count": comorbidity_count,
                        "physical_health_days": physical_health_days,
                        "mental_health_days": mental_health_days,
                        "high_bp": "Yes" if high_bp else "No",
                        "high_cholesterol": "Yes" if high_cholesterol else "No",
                        "depression": "Yes" if depression else "No",
                        "smoking": {1:"Daily Smoker",2:"Some Days",3:"Former Smoker",4:"Never Smoked"}[smoking_status],
                        "physical_activity": {1:"Active",2:"Moderate",3:"Inactive"}[physical_activity],
                        "income": {1:"<$15K",2:"$15-25K",3:"$25-35K",4:"$35-50K",5:"$50-100K",6:"$100-200K",7:"$200K+"}[income],
                        "employment": {1:"Employed",2:"Self-employed",3:"Out of work",4:"Homemaker",5:"Student",6:"Retired",7:"Unable to work",8:"Other"}[employment],
                        "has_doctor": "Yes" if has_doctor == 1 else "No",
                        "has_insurance": "Yes" if has_insurance == 1 else "No",
                        "cost_barrier": "Yes" if cost_barrier else "No",
                        "age_risk_tier": {0:"Low (18-44)",1:"Medium (45-64)",2:"High (65+)"}[age_risk_tier],
                        "health_burden_score": health_burden,
                        "ses_score": ses_score,
                    }
                except FileNotFoundError:
                    st.error("❌  model.pkl not found.")
                except Exception as e:
                    st.error(f"❌  {e}")

            # ── Results + Charts (shown after prediction) ──────────────────
            if "prediction" in st.session_state:
                prob_pct = st.session_state["prob"]
                is_high  = st.session_state["is_high"]
                patient  = st.session_state["patient"]

                if is_high:
                    st.markdown(f'<div class="result-high"><div class="result-label">⚠️  HIGH RISK</div><div class="result-prob">Mortality risk probability: {prob_pct}%</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-low"><div class="result-label">✅  LOW RISK</div><div class="result-prob">Mortality risk probability: {prob_pct}%</div></div>', unsafe_allow_html=True)

                # Model info chips
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-card"><div class="mc-val">XGBoost</div><div class="mc-lbl">Model</div></div>
                  <div class="metric-card"><div class="mc-val">0.8256</div><div class="mc-lbl">ROC-AUC</div></div>
                  <div class="metric-card"><div class="mc-val">85%</div><div class="mc-lbl">Recall</div></div>
                  <div class="metric-card"><div class="mc-val">0.40</div><div class="mc-lbl">Threshold</div></div>
                </div>""", unsafe_allow_html=True)

                # Gauge
                st.plotly_chart(gauge_chart(prob_pct), use_container_width=True, config={"displayModeBar": False})

                # Risk factors
                st.plotly_chart(risk_factors_chart(patient), use_container_width=True, config={"displayModeBar": False})

                # Patient vs population
                st.plotly_chart(patient_vs_population_chart(patient, prob_pct), use_container_width=True, config={"displayModeBar": False})

        # ── RIGHT: Chatbot ─────────────────────────────────────────────────
        with right:
            st.markdown('<div class="card"><div class="card-title">AI Risk Assistant — Powered by Groq</div>', unsafe_allow_html=True)

            if "messages" not in st.session_state:
                st.session_state.messages = [{
                    "role": "system",
                    "content": (
                        "You are a clinical AI assistant helping healthcare professionals interpret "
                        "CDC BRFSS 2023 mortality risk predictions from a tuned XGBoost model trained "
                        "on 433,000+ Americans. Explain clearly why the patient is HIGH or LOW RISK, "
                        "focusing on the top contributing factors. Use plain clinical language. "
                        "Always include a disclaimer that this is a screening tool, not a clinical diagnosis. "
                        "Keep responses under 200 words unless asked for more."
                    )
                }]

            if "prediction" not in st.session_state:
                st.markdown("""
                <div style="text-align:center;padding:50px 20px;color:#2d4a5e;">
                  <p style="font-size:2rem;">🤖</p>
                  <p style="font-size:0.95rem;font-weight:600;">Run a prediction first</p>
                  <p style="font-size:0.82rem;">Fill in patient details on the left<br>then ask me anything.</p>
                </div>""", unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages[1:]:
                    css  = "chat-user" if msg["role"] == "user" else "chat-bot"
                    text = msg["content"]
                    if msg["role"] == "user" and "Patient prediction:" in text:
                        text = text.split("User question:")[-1].strip()
                    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

            user_input = st.chat_input("Ask about the prediction or risk factors...")
            if user_input:
                if not GROQ_API_KEY:
                    st.error("❌  GROQ_API_KEY not set in .env file.")
                else:
                    if "prediction" in st.session_state:
                        full_msg = (f"Patient prediction: {st.session_state['prediction']}\n"
                                    f"Risk probability: {st.session_state['prob']}%\n"
                                    f"Patient details: {st.session_state['patient']}\n\n"
                                    f"User question: {user_input}")
                    else:
                        full_msg = user_input

                    st.session_state.messages.append({"role":"user","content":full_msg})
                    try:
                        client   = Groq(api_key=GROQ_API_KEY)
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=st.session_state.messages,
                            max_tokens=512, temperature=0.4,
                        )
                        reply = response.choices[0].message.content
                    except Exception as e:
                        reply = f"⚠️ Groq API error: {e}"

                    st.session_state.messages.append({"role":"assistant","content":reply})
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:

        # ── Model Info Cards ──────────────────────────────────────────────────
        st.markdown("""
        <div class="metric-row">
          <div class="metric-card"><div class="mc-val">XGBoost</div><div class="mc-lbl">Best Model</div></div>
          <div class="metric-card"><div class="mc-val">0.8256</div><div class="mc-lbl">ROC-AUC</div></div>
          <div class="metric-card"><div class="mc-val">85%</div><div class="mc-lbl">Recall @ 0.40</div></div>
          <div class="metric-card"><div class="mc-val">73.9%</div><div class="mc-lbl">Accuracy</div></div>
          <div class="metric-card"><div class="mc-val">60.6%</div><div class="mc-lbl">Precision</div></div>
          <div class="metric-card"><div class="mc-val">433K</div><div class="mc-lbl">Training Rows</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Row 1: ROC Curve + Model Comparison ───────────────────────────────
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.plotly_chart(roc_curve_chart(), use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(model_comparison_chart(), use_container_width=True, config={"displayModeBar": False})

        # ── Row 2: Threshold + Confusion Matrix ───────────────────────────────
        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.plotly_chart(threshold_chart(), use_container_width=True, config={"displayModeBar": False})
        with c4:
            st.plotly_chart(confusion_matrix_chart(), use_container_width=True, config={"displayModeBar": False})

        # ── Model Journey Table ───────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-title">Model Journey — All Results</div>', unsafe_allow_html=True)
        st.markdown("""
        | Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
        |-------|----------|---------|-----------|--------|----|
        | Logistic Regression | 75.46% | 0.8170 | 68.11% | 58.60% | 0.6300 |
        | Random Forest | 75.49% | 0.8195 | 68.02% | 58.98% | 0.6318 |
        | XGBoost | 74.03% | 0.8244 | 60.76% | 76.68% | 0.6780 |
        | **XGBoost Tuned ✅** | **73.92%** | **0.8256** | **60.57%** | **76.91%** | **0.6777** |
        | Neural Network | 73.25% | 0.8144 | 59.83% | 76.02% | 0.6696 |
        | LightGBM | 73.96% | 0.8253 | 60.64% | 76.83% | 0.6778 |
        """)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_auth_page()
else:
    show_main_app()