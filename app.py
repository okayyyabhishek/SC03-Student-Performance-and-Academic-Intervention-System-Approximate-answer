"""
SC03 Student Academic Performance & Early Intervention System
Product V1 - Testing & Validation Phase (Step 6)
High-End Design System Edition
"""
import io
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.product_predict import FEATURES, ROOT, load_model, predict_one

# -----------------------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SC03 | Academic Risk & Early Intervention",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# High-End Design System CSS (Plus Jakarta Sans, Glassmorphism, Double-Bezel)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hide Streamlit default chrome elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Container macro-spacing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

/* Hero Header Card */
.hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.hero-card::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, rgba(0,0,0,0) 70%);
    pointer-events: none;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    color: #93c5fd;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc;
    margin: 0 0 6px 0;
}

.hero-subtitle {
    font-size: 14px;
    color: #94a3b8;
    margin: 0;
    max-width: 750px;
    line-height: 1.5;
}

/* Double-Bezel Card Architecture */
.bezel-outer {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 8px;
    margin-bottom: 20px;
}

.bezel-inner {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 22px;
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.08);
}

.card-heading {
    font-size: 15px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Result Cards - Risk Theming */
.risk-card {
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
}

.risk-low {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(6, 95, 70, 0.2) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 15px 35px -10px rgba(16, 185, 129, 0.2), inset 0 1px 0 rgba(16, 185, 129, 0.4);
}

.risk-medium {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.4) 0%, rgba(146, 64, 14, 0.2) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    box-shadow: 0 15px 35px -10px rgba(245, 158, 11, 0.2), inset 0 1px 0 rgba(245, 158, 11, 0.4);
}

.risk-high {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.4) 0%, rgba(153, 27, 27, 0.2) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    box-shadow: 0 15px 35px -10px rgba(239, 68, 68, 0.2), inset 0 1px 0 rgba(239, 68, 68, 0.4);
}

.risk-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-low { background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
.badge-medium { background: rgba(245, 158, 11, 0.25); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
.badge-high { background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }

.risk-title {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    margin: 8px 0 6px 0;
}

.intervention-box {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 14px;
}

.metric-bar-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 999px;
    height: 7px;
    overflow: hidden;
    margin-top: 5px;
    margin-bottom: 12px;
}

.metric-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

/* Custom button tweaks */
div.stButton > button:first-child {
    border-radius: 9999px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s cubic-bezier(0.32, 0.72, 0, 1) !important;
}

div.stButton > button:first-child:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px -6px rgba(59, 130, 246, 0.5);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Pipeline Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return load_model()


try:
    model = get_pipeline()
    model_loaded = True
except Exception as err:
    model_loaded = False
    load_error_msg = str(err)


# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-pill">
            <span>●</span> SC03 · Early Academic Intervention
        </div>
        <h1 class="hero-title">Student Academic Risk Predictor</h1>
        <p class="hero-subtitle">
            Multivariate risk tier classification powered by a trained 3-layer Artificial Neural Network (MLP).
            Provides early warning indicators and recommended faculty intervention pathways.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(f"❌ Saved ANN Pipeline could not be loaded: {load_error_msg}")
    st.info("Check `models/ann_student_performance.pkl` and `models/ann_model_metadata.json`.")
    st.stop()


# -----------------------------------------------------------------------------
# State Management & Preset Configurations
# -----------------------------------------------------------------------------
PRESETS = {
    "step6_benchmark": {
        "name": "Step 6 Benchmark (Low Risk)",
        "attendance_pct": 100.0,
        "assessment_pct": 87.2,
        "assignment_pct": 67.0,
        "engagement_score": 7.9,
        "prior_performance_pct": 92.1,
    },
    "high_achiever": {
        "name": "Dean's List / High Achiever (Low Risk)",
        "attendance_pct": 96.0,
        "assessment_pct": 91.5,
        "assignment_pct": 95.0,
        "engagement_score": 9.2,
        "prior_performance_pct": 89.0,
    },
    "borderline": {
        "name": "Moderate / Borderline Support (Medium Risk)",
        "attendance_pct": 74.0,
        "assessment_pct": 62.0,
        "assignment_pct": 68.0,
        "engagement_score": 5.4,
        "prior_performance_pct": 65.0,
    },
    "critical_at_risk": {
        "name": "High Alert / Critical Support (High Risk)",
        "attendance_pct": 48.0,
        "assessment_pct": 39.0,
        "assignment_pct": 42.0,
        "engagement_score": 2.3,
        "prior_performance_pct": 45.0,
    },
}

for f in FEATURES:
    if f not in st.session_state:
        st.session_state[f] = PRESETS["step6_benchmark"][f]


def apply_preset(preset_key):
    preset = PRESETS[preset_key]
    for feat in FEATURES:
        st.session_state[feat] = preset[feat]


# -----------------------------------------------------------------------------
# Main Two-Column Layout
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown(
        """
        <div class="bezel-outer">
            <div class="bezel-inner">
                <div class="card-heading">
                    <span>⚙️</span> Student Academic Profile & Inputs
                </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset Quick Load Toolbar
    st.caption("Quick Profile Presets:")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("🌟 Honors", help="Load high performing student", use_container_width=True):
            apply_preset("high_achiever")
            st.rerun()
    with p_col2:
        if st.button("⚖️ Borderline", help="Load borderline student", use_container_width=True):
            apply_preset("borderline")
            st.rerun()
    with p_col3:
        if st.button("🚨 Critical", help="Load at-risk student", use_container_width=True):
            apply_preset("critical_at_risk")
            st.rerun()
    with p_col4:
        if st.button("🔄 Step 6", help="Load verified Step 6 test case", use_container_width=True):
            apply_preset("step6_benchmark")
            st.rerun()

    st.write("")

    # Interactive Form
    with st.form("academic_indicators_form"):
        st.markdown("**1. Attendance Rate**")
        att_val = st.slider(
            "Attendance Rate (% 0-100)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["attendance_pct"]),
            step=0.5,
            key="slider_attendance",
            label_visibility="collapsed",
        )
        st.caption(f"Current Value: `{att_val:.1f}%` (Institutional threshold: 75%)")

        st.markdown("**2. Continuous Assessment Score**")
        ass_val = st.slider(
            "Continuous Assessment Score (% 0-100)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["assessment_pct"]),
            step=0.5,
            key="slider_assessment",
            label_visibility="collapsed",
        )
        st.caption(f"Current Value: `{ass_val:.1f}%` (Midterm, quizzes, in-class evaluations)")

        st.markdown("**3. Assignment Completion Rate**")
        asg_val = st.slider(
            "Assignment Completion Score (% 0-100)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["assignment_pct"]),
            step=0.5,
            key="slider_assignment",
            label_visibility="collapsed",
        )
        st.caption(f"Current Value: `{asg_val:.1f}%` (Homework, project deliverables, laboratory reports)")

        st.markdown("**4. LMS Engagement Score**")
        eng_val = st.slider(
            "LMS Engagement Score (0-10 index)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state["engagement_score"]),
            step=0.1,
            key="slider_engagement",
            label_visibility="collapsed",
        )
        st.caption(f"Current Value: `{eng_val:.1f} / 10.0` (Platform login frequency, discussion participation)")

        st.markdown("**5. Prior Academic Performance**")
        pri_val = st.slider(
            "Prior Academic Performance (% 0-100)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["prior_performance_pct"]),
            step=0.5,
            key="slider_prior",
            label_visibility="collapsed",
        )
        st.caption(f"Current Value: `{pri_val:.1f}%` (Historical CGPA / prerequisite score equivalent)")

        st.write("")
        submitted = st.form_submit_button("⚡ Evaluate Student Academic Risk", use_container_width=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


with col_right:
    st.markdown(
        """
        <div class="bezel-outer">
            <div class="bezel-inner">
                <div class="card-heading">
                    <span>📊</span> Decision Support & Risk Analysis
                </div>
        """,
        unsafe_allow_html=True,
    )

    # Prepare values dictionary for prediction
    current_values = {
        "attendance_pct": str(att_val),
        "assessment_pct": str(ass_val),
        "assignment_pct": str(asg_val),
        "engagement_score": str(eng_val),
        "prior_performance_pct": str(pri_val),
    }

    try:
        result = predict_one(model, current_values)
        pred_class = result["predicted_class"].lower()
        pred_label = result["label"]
        pred_intervention = result["intervention"]
        pred_notice = result["notice"]

        # Risk level stylings
        badge_class = f"badge-{pred_class}"
        card_class = f"risk-{pred_class}"
        accent_color = (
            "#10b981" if pred_class == "low" else "#f59e0b" if pred_class == "medium" else "#ef4444"
        )

        st.markdown(
            f"""
            <div class="risk-card {card_class}">
                <div class="flex items-center justify-between">
                    <span class="risk-badge {badge_class}">{pred_class.upper()} RISK</span>
                    <span style="font-size: 11px; color: rgba(255,255,255,0.6); font-family: 'JetBrains Mono', monospace;">
                        ANN INFERENCE
                    </span>
                </div>
                <div class="risk-title">{pred_label}</div>
                <div class="intervention-box">
                    <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94a3b8; letter-spacing: 0.05em;">
                        Recommended Academic Intervention
                    </div>
                    <div style="font-size: 14px; font-weight: 600; color: #ffffff; margin-top: 4px;">
                        📌 {pred_intervention}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Indicator Breakdown Bars
        st.markdown("**Academic Indicator Telemetry**")
        
        indicators = [
            ("Attendance", att_val, 100.0, "%", "#38bdf8" if att_val >= 75 else "#f87171"),
            ("Assessments", ass_val, 100.0, "%", "#38bdf8" if ass_val >= 60 else "#f87171"),
            ("Assignments", asg_val, 100.0, "%", "#38bdf8" if asg_val >= 60 else "#f87171"),
            ("LMS Activity", eng_val * 10.0, 100.0, f" ({eng_val:.1f}/10)", "#818cf8" if eng_val >= 5 else "#f87171"),
            ("Prior Record", pri_val, 100.0, "%", "#38bdf8" if pri_val >= 60 else "#f87171"),
        ]

        for name, val, max_val, unit, col in indicators:
            pct = min(max(val / max_val * 100.0, 0.0), 100.0)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                    <span style="color: #cbd5e1;">{name}</span>
                    <span style="font-weight: 600; color: {col}; font-family: 'JetBrains Mono', monospace;">{val:.1f}{unit}</span>
                </div>
                <div class="metric-bar-container">
                    <div class="metric-bar-fill" style="width: {pct}%; background-color: {col};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Export & Download Capabilities
        st.write("")
        df_out = pd.DataFrame([result])
        csv_data = df_out.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report (CSV)",
            data=csv_data,
            file_name="step6_ui_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Expandable Data Table
        with st.expander("📋 View Raw Prediction Vector (JSON & Table)"):
            st.dataframe(df_out, use_container_width=True)
            st.caption(pred_notice)

    except ValueError as val_err:
        st.error(f"⚠️ Validation Guardrail Triggered: {val_err}")
    except Exception as general_err:
        st.error(f"❌ Prediction Error: {general_err}")

    st.markdown("</div></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Footer & Methodological Context
# -----------------------------------------------------------------------------
st.write("")
with st.expander("ℹ️ About the SC03 Academic Intervention Architecture (Step 5 & 6)"):
    st.markdown(
        """
        - **Pipeline Integration:** Calls `predict()` directly on the pickled `scikit-learn` `Pipeline`
          combining `StandardScaler` with an `MLPClassifier` trained in Step 3.
        - **Zero Run-Time Training:** The model is evaluated purely in inference mode with zero training overhead.
        - **Defensive Guardrails:** Rejects missing attributes, non-finite values, and invalid boundaries before inference.
        - **Ethical Decision-Support:** All outputs serve strictly as advisory indicators for academic counselors and do
          not automate punitive or grading decisions.
        """
    )
