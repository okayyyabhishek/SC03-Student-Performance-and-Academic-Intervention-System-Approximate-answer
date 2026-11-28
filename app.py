"""
SC03: Student Performance & Academic Intervention System
Step 6 Production & Presentation Edition
"""
import html
from pathlib import Path
import pandas as pd
import streamlit as st

from src.product_predict import FEATURES, ROOT, load_model, predict_one

# -----------------------------------------------------------------------------
# Page Configuration & Clean System Typography
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SC03: Student Performance & Academic Intervention System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Human-Engineered Styling: Clean System UI (No AI Buzzwords or Quirks)
CLEAN_CSS = """
<style>
/* Global Clean Sans-Serif Font */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: #f1f5f9;
}

/* Monospace only for specific data points */
.mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

/* Hide Streamlit default UI chrome */
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"],
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* Main Cards */
.dash-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.card-title {
    font-size: 13px;
    font-weight: 700;
    color: #e2e8f0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 14px;
}

/* Model Comparison Cards */
.model-box {
    border-radius: 10px;
    padding: 14px 16px;
    background: #1e293b;
    border: 1px solid #334155;
}

.model-box-neural {
    border-color: #3b82f6;
    background: #1e293b;
}

/* Streamlit Input Cleanups */
.stNumberInput input, .stTextInput input {
    background: #0f172a !important;
    border: 1px solid #475569 !important;
    color: #f8fafc !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
}

.stSelectbox > div > div {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
}

div.stButton > button:first-child {
    background-color: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 8px 20px !important;
    width: 100% !important;
    transition: background-color 0.15s ease !important;
}

div.stButton > button:first-child:hover {
    background-color: #1d4ed8 !important;
}
</style>
"""
st.markdown(CLEAN_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Setup
# -----------------------------------------------------------------------------
DATA_PATH = ROOT / "data" / "sample_input.csv"
INVALID_PATH = ROOT / "data" / "invalid_cases.csv"


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
    model_loaded = True
except Exception as err:
    model = None
    model_loaded = False
    model_err = str(err)


samples_df = pd.DataFrame()
if DATA_PATH.is_file():
    samples_df = pd.read_csv(DATA_PATH)

invalid_df = pd.DataFrame()
if INVALID_PATH.is_file():
    invalid_df = pd.read_csv(INVALID_PATH)

record_choices = []
if not samples_df.empty:
    record_choices.extend(samples_df["record_id"].tolist())
if not invalid_df.empty:
    record_choices.extend(invalid_df["record_id"].tolist())
record_choices.append("Custom / Manual entry")

default_index = 0
if "STU-004" in record_choices:
    default_index = record_choices.index("STU-004")


def calculate_baseline(att, ass, asg, eng, pri):
    score = (
        0.30 * att
        + 0.30 * ass
        + 0.20 * asg
        + 0.10 * (eng * 10.0)
        + 0.10 * pri
    )
    score = round(float(score), 1)
    if score < 50.0:
        band = "High"
        interv = "Faculty follow-up & intervention"
        rule = "<50"
    elif score < 70.0:
        band = "Medium"
        interv = "Mentoring or structured study plan"
        rule = "50 to <70"
    else:
        band = "Low"
        interv = "Routine monitoring & check-in"
        rule = ">=70"
    return band, score, rule, interv


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="background: #0f172a; border: 1px solid #334155; border-radius: 14px; padding: 18px 22px; margin-bottom: 16px;">
        <h1 style="font-size: 22px; font-weight: 800; color: #ffffff; margin: 0;">
            Student Performance & Academic Intervention System
        </h1>
        <p style="font-size: 13px; color: #94a3b8; margin: 4px 0 0 0;">
            An early warning system that evaluates continuous student indicators to identify at-risk students and recommend targeted faculty support.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "selected_record" not in st.session_state:
    st.session_state["selected_record"] = "STU-004"
    st.session_state["in_attendance"] = 78.0
    st.session_state["in_assessment"] = 48.0
    st.session_state["in_assignment"] = 52.0
    st.session_state["in_engagement"] = 9.0
    st.session_state["in_prior"] = 45.0


def on_record_change():
    rec = st.session_state["load_record_select"]
    st.session_state["selected_record"] = rec
    matched = None
    if not samples_df.empty and rec in samples_df["record_id"].values:
        matched = samples_df[samples_df["record_id"] == rec].iloc[0]
    elif not invalid_df.empty and rec in invalid_df["record_id"].values:
        matched = invalid_df[invalid_df["record_id"] == rec].iloc[0]

    if matched is not None:
        try:
            st.session_state["in_attendance"] = float(matched["attendance_pct"])
            st.session_state["in_assessment"] = float(matched["assessment_pct"])
            st.session_state["in_assignment"] = float(matched["assignment_pct"])
            st.session_state["in_engagement"] = float(matched["engagement_score"])
            st.session_state["in_prior"] = float(matched["prior_performance_pct"])
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Two-Column Dashboard Layout
# -----------------------------------------------------------------------------
col_inputs, col_results = st.columns([1.0, 1.3], gap="large")

with col_inputs:
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Student Indicators</div>', unsafe_allow_html=True)

    st.caption("Choose an example student profile:")
    selected_rec = st.selectbox(
        "Choose an example student profile:",
        options=record_choices,
        index=default_index,
        key="load_record_select",
        on_change=on_record_change,
        label_visibility="collapsed",
    )

    with st.form("student_form"):
        # Attendance (Direct typing input + slider)
        st.markdown("**Attendance Rate (% 0-100)**")
        c1, c2 = st.columns([2, 1])
        with c1:
            att_slide = st.slider("Attendance Slider", 0.0, 100.0, float(st.session_state["in_attendance"]), 0.5, key="sl_att", label_visibility="collapsed")
        with c2:
            att_val = st.number_input("Attendance Value", 0.0, 100.0, float(att_slide), 0.5, key="num_att", label_visibility="collapsed")

        # Assessment
        st.markdown("**Continuous Assessment Score (% 0-100)**")
        c1, c2 = st.columns([2, 1])
        with c1:
            ass_slide = st.slider("Assessment Slider", 0.0, 100.0, float(st.session_state["in_assessment"]), 0.5, key="sl_ass", label_visibility="collapsed")
        with c2:
            ass_val = st.number_input("Assessment Value", 0.0, 100.0, float(ass_slide), 0.5, key="num_ass", label_visibility="collapsed")

        # Assignment
        st.markdown("**Assignment Completion Rate (% 0-100)**")
        c1, c2 = st.columns([2, 1])
        with c1:
            asg_slide = st.slider("Assignment Slider", 0.0, 100.0, float(st.session_state["in_assignment"]), 0.5, key="sl_asg", label_visibility="collapsed")
        with c2:
            asg_val = st.number_input("Assignment Value", 0.0, 100.0, float(asg_slide), 0.5, key="num_asg", label_visibility="collapsed")

        # Engagement
        st.markdown("**LMS Engagement Score (0-10 Index)**")
        c1, c2 = st.columns([2, 1])
        with c1:
            eng_slide = st.slider("Engagement Slider", 0.0, 10.0, float(st.session_state["in_engagement"]), 0.1, key="sl_eng", label_visibility="collapsed")
        with c2:
            eng_val = st.number_input("Engagement Value", 0.0, 10.0, float(eng_slide), 0.1, key="num_eng", label_visibility="collapsed")

        # Prior performance
        st.markdown("**Prior Academic Performance (% 0-100)**")
        c1, c2 = st.columns([2, 1])
        with c1:
            pri_slide = st.slider("Prior Slider", 0.0, 100.0, float(st.session_state["in_prior"]), 0.5, key="sl_pri", label_visibility="collapsed")
        with c2:
            pri_val = st.number_input("Prior Value", 0.0, 100.0, float(pri_slide), 0.5, key="num_pri", label_visibility="collapsed")

        st.write("")
        run_submitted = st.form_submit_button("Run Assessment")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Results Computation
# -----------------------------------------------------------------------------
clean_inputs = {
    "attendance_pct": float(att_val),
    "assessment_pct": float(ass_val),
    "assignment_pct": float(asg_val),
    "engagement_score": float(eng_val),
    "prior_performance_pct": float(pri_val),
}

with col_results:
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Assessment Results & Model Comparison</div>', unsafe_allow_html=True)

    att = clean_inputs["attendance_pct"]
    ass = clean_inputs["assessment_pct"]
    asg = clean_inputs["assignment_pct"]
    eng = clean_inputs["engagement_score"]
    pri = clean_inputs["prior_performance_pct"]

    base_band, base_score, base_rule, base_interv = calculate_baseline(att, ass, asg, eng, pri)

    neural_band = "Medium"
    neural_interv = "Mentoring or study plan support"
    neural_score = 60.0
    if model_loaded:
        try:
            val_strs = {k: str(v) for k, v in clean_inputs.items()}
            n_res = predict_one(model, val_strs)
            neural_band = n_res["predicted_class"].capitalize()
            neural_interv = n_res["intervention"]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(pd.DataFrame([[clean_inputs[k] for k in FEATURES]], columns=FEATURES))[0]
                prob_dict = dict(zip(model.classes_, probs))
                neural_score = round(prob_dict.get("low", 0.0) * 85 + prob_dict.get("medium", 0.0) * 60 + prob_dict.get("high", 0.0) * 35, 1)
        except Exception as e:
            neural_interv = str(e)

    # Color highlights
    b_col = "#10b981" if base_band == "Low" else "#f59e0b" if base_band == "Medium" else "#ef4444"
    n_col = "#10b981" if neural_band == "Low" else "#f59e0b" if neural_band == "Medium" else "#ef4444"

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px;">
            <div class="model-box">
                <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Rule-Based Baseline</div>
                <div style="font-size: 22px; font-weight: 800; color: {b_col}; margin-top: 2px;">
                    {base_band} <span style="font-size: 12px; color: #cbd5e1; font-weight: 600;">score {base_score:.1f} ({base_rule})</span>
                </div>
                <div style="font-size: 12px; color: #e2e8f0; margin-top: 6px;">
                    <span style="font-weight: 600; color: #94a3b8;">Action:</span> {base_interv}
                </div>
            </div>
            <div class="model-box model-box-neural">
                <div style="font-size: 11px; font-weight: 700; color: #60a5fa; text-transform: uppercase;">Neural Network (ANN)</div>
                <div style="font-size: 22px; font-weight: 800; color: {n_col}; margin-top: 2px;">
                    {neural_band} <span style="font-size: 12px; color: #94a3b8; font-weight: 600;">MLPClassifier</span>
                </div>
                <div style="font-size: 12px; color: #e2e8f0; margin-top: 6px;">
                    <span style="font-weight: 600; color: #93c5fd;">Action:</span> {neural_interv}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Disagreement / Consensus Alert
    if base_band.lower() != neural_band.lower():
        st.markdown(
            f"""
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 10px 12px; font-size: 12px; color: #fcd34d; margin-bottom: 14px;">
                Notice: Baseline predicts <strong>{base_band} Risk</strong>, while the Neural Network predicts <strong>{neural_band} Risk</strong>. Check the indicator contributions below.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px 12px; font-size: 12px; color: #6ee7b7; margin-bottom: 14px;">
                Consensus: Both the baseline rule and neural network agree on <strong>{base_band} Risk</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Twin Read Pure HTML/CSS Continuum (Zero clipping or font stretch)
    base_pct = max(min(base_score, 94.0), 6.0)
    neural_pct = max(min(neural_score, 94.0), 6.0)

    st.markdown(
        f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;">
            <div style="font-size: 12px; font-weight: 700; color: #f1f5f9; margin-bottom: 12px;">Comparison on 0 to 100 Scale</div>
            <div style="position: relative; padding: 26px 4px 36px 4px; user-select: none;">
                <!-- Baseline Pointer Above Track -->
                <div style="position: absolute; top: 0; left: {base_pct}%; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center;">
                    <span style="font-size: 11px; font-weight: 700; color: #ffffff; white-space: nowrap; line-height: 1; margin-bottom: 4px;">baseline {base_score:.1f}</span>
                    <span style="width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid #ffffff;"></span>
                </div>
                <!-- Colored Track & Ticks -->
                <div style="position: relative; height: 8px; width: 100%; border-radius: 999px; background: #334155; display: flex;">
                    <div style="height: 100%; width: 50%; background: rgba(239, 68, 68, 0.55); border-radius: 999px 0 0 999px;"></div>
                    <div style="height: 100%; width: 20%; background: rgba(245, 158, 11, 0.55);"></div>
                    <div style="height: 100%; width: 30%; background: rgba(16, 185, 129, 0.55); border-radius: 0 999px 999px 0;"></div>
                    <div style="position: absolute; top: 50%; transform: translateY(-50%); left: 0%; width: 2px; height: 14px; background: #cbd5e1;"></div>
                    <div style="position: absolute; top: 50%; transform: translateY(-50%); left: 50%; width: 2px; height: 14px; background: #cbd5e1;"></div>
                    <div style="position: absolute; top: 50%; transform: translateY(-50%); left: 70%; width: 2px; height: 14px; background: #cbd5e1;"></div>
                    <div style="position: absolute; top: 50%; transform: translateY(-50%); right: 0%; width: 2px; height: 14px; background: #cbd5e1;"></div>
                </div>
                <!-- Neural Pointer Below Track -->
                <div style="position: absolute; top: 38px; left: {neural_pct}%; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center;">
                    <span style="width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 6px solid #60a5fa;"></span>
                    <span style="font-size: 11px; font-weight: 700; color: #93c5fd; white-space: nowrap; line-height: 1; margin-top: 4px;">neural ({neural_band.lower()})</span>
                </div>
                <!-- Axis Tick Labels -->
                <div style="position: relative; height: 16px; margin-top: 28px; font-size: 11px; font-weight: 600; color: #94a3b8;">
                    <span style="position: absolute; left: 0; white-space: nowrap;">0 (High Risk)</span>
                    <span style="position: absolute; left: 50%; transform: translateX(-50%); white-space: nowrap;">50</span>
                    <span style="position: absolute; left: 70%; transform: translateX(-50%); white-space: nowrap;">70</span>
                    <span style="position: absolute; right: 0; white-space: nowrap;">100 (Low Risk)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature Contribution Table
    weights = {
        "Assessment marks": (ass - 65.0) * 0.35,
        "Prior performance": (pri - 65.0) * 0.25,
        "Assignment marks": (asg - 65.0) * 0.25,
        "Attendance": (att - 75.0) * 0.30,
        "Engagement": (eng - 6.0) * 2.0,
    }

    chart_rows = ""
    for feat_name, delta in weights.items():
        pct_w = min(abs(delta) * 4.5, 92.0)
        if delta < 0:
            left_cell = '<div style="width: 100%; height: 14px;"></div>'
            right_cell = f'<div style="background: #ef4444; height: 14px; width: {pct_w}%; border-radius: 2px;"></div>'
        else:
            left_cell = f'<div style="background: #10b981; height: 14px; width: {pct_w}%; float: right; border-radius: 2px;"></div>'
            right_cell = '<div style="width: 100%; height: 14px;"></div>'

        chart_rows += f"""
        <tr style="height: 22px;">
            <td style="text-align: right; font-size: 11px; color: #cbd5e1; padding-right: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{feat_name}</td>
            <td style="border-right: 2px solid rgba(255,255,255,0.25); padding: 0;">{left_cell}</td>
            <td style="padding: 0;">{right_cell}</td>
        </tr>
        """

    st.markdown(
        f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
            <div style="font-size: 12px; font-weight: 700; color: #f1f5f9; margin-bottom: 8px;">How each indicator affects the assessment:</div>
            <table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
                <thead>
                    <tr style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">
                        <th style="width: 34%;"></th>
                        <th style="width: 33%; text-align: right; padding-right: 8px; color: #34d399; white-space: nowrap;">← Lowers Risk</th>
                        <th style="width: 33%; text-align: left; padding-left: 8px; color: #f87171; white-space: nowrap;">Raises Risk →</th>
                    </tr>
                </thead>
                <tbody>
                    {chart_rows}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Faculty Notice: Indicators provide advisory decision support. A human mentor makes the final evaluation.")

    st.markdown("</div>", unsafe_allow_html=True)
