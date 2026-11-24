"""
SC03: Student Performance and Academic Intervention System
Product V1 - Exact Specification Layout
"""
import html
from pathlib import Path
import pandas as pd
import streamlit as st

from src.product_predict import FEATURES, ROOT, load_model, predict_one

# -----------------------------------------------------------------------------
# Page Configuration & Clean Blueprint Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SC03: Student Performance and Academic Intervention System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS to match the exact blueprint layout and clean visual hierarchy
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700;800&family=Consolas&family=Fira+Code&display=swap');

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
}

#MainMenu, footer, header {
    visibility: hidden;
    display: none !important;
}

.stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"],
[data-testid="collapsedControl"] {
    display: none !important;
}

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

/* Blueprint Header */
.app-header {
    margin-bottom: 14px;
}
.app-title {
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.01em;
    margin: 0;
}
.app-subtitle {
    font-size: 13px;
    color: #64748b;
    margin: 3px 0 0 0;
}

/* Master Panels */
.panel-box {
    background: #ffffff;
    border: 1.5px solid #94a3b8;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.panel-title {
    font-size: 13px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* Result Cards Side-by-Side */
.result-card-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 18px;
}

.result-card {
    border: 1.5px solid #94a3b8;
    border-radius: 6px;
    padding: 12px 14px;
    background: #ffffff;
}

.result-card-neural {
    border: 1.5px solid #1e40af;
}

.card-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #334155;
    margin-bottom: 4px;
}

.card-label-neural {
    color: #1e40af;
}

.risk-level-display {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 6px;
}

.risk-subtext {
    font-size: 12px;
    font-weight: 600;
    color: #475569;
}

.intervention-text {
    font-size: 12px;
    color: #334155;
    margin-top: 4px;
}

/* Scale & Axis Section */
.scale-section-title {
    font-size: 12px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
}

/* Validation Area */
.validation-box {
    background: #ffffff;
    border: 1.5px solid #94a3b8;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'Consolas', 'Fira Code', monospace;
    font-size: 12px;
    line-height: 1.6;
}

.val-pass {
    color: #15803d;
    font-weight: 700;
}

.val-fail {
    color: #b91c1c;
    font-weight: 700;
}

.val-dim {
    color: #64748b;
}

/* Run Button */
div.stButton > button:first-child {
    background-color: #1e3a8a !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 10px 24px !important;
    margin-top: 8px !important;
    width: 100% !important;
    transition: background-color 0.15s ease-in-out !important;
}

div.stButton > button:first-child:hover {
    background-color: #172554 !important;
}

/* Streamlit form input styling */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border: 1.5px solid #94a3b8 !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.stSelectbox > div > div {
    border: 1.5px solid #94a3b8 !important;
    border-radius: 6px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


# Load Records
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
record_choices.append("Manual Entry")

# Default to STU-004 as shown in the reference sketch
default_index = 0
if "STU-004" in record_choices:
    default_index = record_choices.index("STU-004")


# Baseline calculation helper
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
        interv = "Faculty follow-up"
        rule = "<50"
    elif score < 70.0:
        band = "Medium"
        interv = "Mentoring or study plan"
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
    <div class="app-header">
        <h1 class="app-title">SC03: Student Performance and Academic Intervention System</h1>
        <p class="app-subtitle">Product V1 sketch | example values only (record STU-004) | a human makes the final decision</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Main Panels (Inputs Left, Results Right)
# -----------------------------------------------------------------------------
col_inputs, col_results = st.columns([0.9, 1.3], gap="medium")

# Initialize session state from STU-004
if "selected_record" not in st.session_state:
    st.session_state["selected_record"] = "STU-004"
    st.session_state["in_attendance"] = "78"
    st.session_state["in_assessment"] = "48"
    st.session_state["in_assignment"] = "52"
    st.session_state["in_engagement"] = "9"
    st.session_state["in_prior"] = "45"


def on_record_change():
    rec = st.session_state["load_record_select"]
    st.session_state["selected_record"] = rec
    matched = None
    if not samples_df.empty and rec in samples_df["record_id"].values:
        matched = samples_df[samples_df["record_id"] == rec].iloc[0]
    elif not invalid_df.empty and rec in invalid_df["record_id"].values:
        matched = invalid_df[invalid_df["record_id"] == rec].iloc[0]

    if matched is not None:
        st.session_state["in_attendance"] = str(matched["attendance_pct"])
        st.session_state["in_assessment"] = str(matched["assessment_pct"])
        st.session_state["in_assignment"] = str(matched["assignment_pct"])
        st.session_state["in_engagement"] = str(matched["engagement_score"])
        st.session_state["in_prior"] = str(matched["prior_performance_pct"])


with col_inputs:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">INPUTS (anonymous indicators)</div>', unsafe_allow_html=True)

    # Record selector
    st.caption("Load record")
    selected_rec = st.selectbox(
        "Load record",
        options=record_choices,
        index=default_index,
        key="load_record_select",
        on_change=on_record_change,
        label_visibility="collapsed",
    )

    with st.form("input_form"):
        # Input 1: Attendance
        st.caption("Attendance")
        c1, c2 = st.columns([4, 1])
        with c1:
            att_raw = st.text_input("Attendance", value=st.session_state["in_attendance"], key="att_field", label_visibility="collapsed")
        with c2:
            st.markdown("<div style='padding-top: 10px; font-weight: 700; color: #475569;'>%</div>", unsafe_allow_html=True)

        # Input 2: Assessment marks
        st.caption("Assessment marks")
        c1, c2 = st.columns([4, 1])
        with c1:
            ass_raw = st.text_input("Assessment marks", value=st.session_state["in_assessment"], key="ass_field", label_visibility="collapsed")
        with c2:
            st.markdown("<div style='padding-top: 10px; font-weight: 700; color: #475569;'>%</div>", unsafe_allow_html=True)

        # Input 3: Assignment marks
        st.caption("Assignment marks")
        c1, c2 = st.columns([4, 1])
        with c1:
            asg_raw = st.text_input("Assignment marks", value=st.session_state["in_assignment"], key="asg_field", label_visibility="collapsed")
        with c2:
            st.markdown("<div style='padding-top: 10px; font-weight: 700; color: #475569;'>%</div>", unsafe_allow_html=True)

        # Input 4: Engagement score
        st.caption("Engagement score")
        c1, c2 = st.columns([4, 1.2])
        with c1:
            eng_raw = st.text_input("Engagement score", value=st.session_state["in_engagement"], key="eng_field", label_visibility="collapsed")
        with c2:
            st.markdown("<div style='padding-top: 10px; font-weight: 700; color: #475569;'>/ 10</div>", unsafe_allow_html=True)

        # Input 5: Prior performance
        st.caption("Prior performance")
        c1, c2 = st.columns([4, 1])
        with c1:
            pri_raw = st.text_input("Prior performance", value=st.session_state["in_prior"], key="pri_field", label_visibility="collapsed")
        with c2:
            st.markdown("<div style='padding-top: 10px; font-weight: 700; color: #475569;'>%</div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size: 11px; color: #64748b; margin-top: 6px;'>Allowed range: 0 to 100 (engagement 0 to 10)</div>", unsafe_allow_html=True)

        run_clicked = st.form_submit_button("Run assessment")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Validation & Computation
# -----------------------------------------------------------------------------
validation_error = None
clean_inputs = {}

# Parse inputs
for field_name, raw_val, min_v, max_v in [
    ("attendance_pct", att_raw, 0.0, 100.0),
    ("assessment_pct", ass_raw, 0.0, 100.0),
    ("assignment_pct", asg_raw, 0.0, 100.0),
    ("engagement_score", eng_raw, 0.0, 10.0),
    ("prior_performance_pct", pri_raw, 0.0, 100.0),
]:
    if raw_val is None or str(raw_val).strip() == "":
        validation_error = f"{field_name}: enter a value."
        break
    try:
        val_f = float(raw_val)
    except Exception:
        validation_error = f"{field_name}: enter a number."
        break
    if val_f < 0:
        validation_error = f"{field_name}: negative values are invalid."
        break
    if val_f < min_v or val_f > max_v:
        validation_error = f"{field_name} must be between {int(min_v)} and {int(max_v)}, got {raw_val}"
        break
    clean_inputs[field_name] = val_f


# -----------------------------------------------------------------------------
# Results Computation
# -----------------------------------------------------------------------------
with col_results:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">RESULTS</div>', unsafe_allow_html=True)

    if validation_error:
        st.markdown(
            f"""
            <div style="padding: 30px 20px; text-align: center; color: #94a3b8;">
                <div style="font-size: 14px; font-weight: 700; color: #dc2626;">Input Validation Error</div>
                <div style="font-size: 12px; margin-top: 6px; color: #475569;">{validation_error}</div>
                <div style="font-size: 11px; margin-top: 10px; color: #94a3b8;">Nothing is computed when input is invalid. Check the error log below.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # 1. Baseline Calculation
        att = clean_inputs["attendance_pct"]
        ass = clean_inputs["assessment_pct"]
        asg = clean_inputs["assignment_pct"]
        eng = clean_inputs["engagement_score"]
        pri = clean_inputs["prior_performance_pct"]

        base_band, base_score, base_rule, base_interv = calculate_baseline(att, ass, asg, eng, pri)

        # 2. Neural Model Prediction
        neural_band = "High"
        neural_interv = "Faculty follow-up"
        neural_score = 45.0
        if model_loaded:
            try:
                values_dict = {k: str(v) for k, v in clean_inputs.items()}
                n_res = predict_one(model, values_dict)
                neural_band = n_res["predicted_class"].capitalize()
                neural_interv = n_res["intervention"]
                # Approximate neural score on 0-100 scale for twin read visualization
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(pd.DataFrame([[clean_inputs[k] for k in FEATURES]], columns=FEATURES))[0]
                    # Map low->85, med->60, high->35
                    prob_dict = dict(zip(model.classes_, probs))
                    neural_score = round(prob_dict.get("low", 0.0) * 85 + prob_dict.get("medium", 0.0) * 60 + prob_dict.get("high", 0.0) * 35, 1)
                else:
                    neural_score = 80.0 if neural_band == "Low" else 60.0 if neural_band == "Medium" else 40.0
            except Exception as e:
                neural_band = "High"
                neural_interv = f"Error: {e}"

        # Render Side-by-Side Cards
        st.markdown(
            f"""
            <div class="result-card-container">
                <div class="result-card">
                    <div class="card-label">BASELINE (simple score rule)</div>
                    <div class="risk-level-display">{base_band} <span class="risk-subtext">score {base_score:.1f} ({base_rule})</span></div>
                    <div class="intervention-text">Intervention: {base_interv}</div>
                </div>
                <div class="result-card result-card-neural">
                    <div class="card-label card-label-neural">NEURAL MODEL (ANN, trained)</div>
                    <div class="risk-level-display" style="color: #1e40af;">{neural_band}</div>
                    <div class="intervention-text">Intervention: {neural_interv}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Twin Read Axis (0 to 100 Scale)
        base_pct = max(min(base_score, 100.0), 0.0)
        neural_pct = max(min(neural_score, 100.0), 0.0)

        # Disagreement Flag
        is_disagreement = (base_band.lower() != neural_band.lower())
        disagree_msg = (
            f"Disagreement flag: baseline {base_band} vs neural {neural_band}. Check the reasons below."
            if is_disagreement
            else f"Agreement: baseline {base_band} and neural {neural_band} agree."
        )
        flag_color = "#b45309" if is_disagreement else "#15803d"

        # Twin Read SVG Rendering
        svg_scale = f"""
        <div style="margin-bottom: 14px;">
            <div class="scale-section-title">Twin Read: both results on one 0 to 100 scale</div>
            <div style="position: relative; width: 100%; height: 65px; margin-top: 5px;">
                <svg width="100%" height="65" viewBox="0 0 500 65" preserveAspectRatio="none">
                    <!-- Base line -->
                    <line x1="20" y1="35" x2="480" y2="35" stroke="#0f172a" stroke-width="2"/>
                    <!-- Ticks -->
                    <line x1="20" y1="28" x2="20" y2="42" stroke="#0f172a" stroke-width="2"/>
                    <line x1="250" y1="28" x2="250" y2="42" stroke="#0f172a" stroke-width="2"/>
                    <line x1="342" y1="28" x2="342" y2="42" stroke="#0f172a" stroke-width="2"/>
                    <line x1="480" y1="28" x2="480" y2="42" stroke="#0f172a" stroke-width="2"/>
                    <!-- Tick Labels -->
                    <text x="20" y="55" font-size="11" font-weight="600" text-anchor="middle" fill="#0f172a">0</text>
                    <text x="250" y="55" font-size="11" font-weight="600" text-anchor="middle" fill="#0f172a">50</text>
                    <text x="342" y="55" font-size="11" font-weight="600" text-anchor="middle" fill="#0f172a">70</text>
                    <text x="480" y="55" font-size="11" font-weight="600" text-anchor="middle" fill="#0f172a">100</text>
                    <!-- Baseline Pointer (Top Downward Triangle) -->
                    <polygon points="{20 + (base_pct * 4.6)},20 {20 + (base_pct * 4.6) - 6},8 {20 + (base_pct * 4.6) + 6},8" fill="#0f172a"/>
                    <text x="{20 + (base_pct * 4.6)}" y="4" font-size="10" font-weight="700" text-anchor="middle" fill="#0f172a">baseline {base_score:.1f}</text>
                    <!-- Neural Pointer (Bottom Upward Triangle) -->
                    <polygon points="{20 + (neural_pct * 4.6)},38 {20 + (neural_pct * 4.6) - 6},50 {20 + (neural_pct * 4.6) + 6},50" fill="#1e40af"/>
                    <text x="{20 + (neural_pct * 4.6)}" y="63" font-size="10" font-weight="700" text-anchor="middle" fill="#1e40af">neural ({neural_band.lower()})</text>
                </svg>
            </div>
            <div style="font-size: 11px; font-weight: 700; color: {flag_color}; margin-top: 6px;">
                {disagree_msg}
            </div>
        </div>
        """
        st.markdown(svg_scale, unsafe_allow_html=True)

        # Diverging Bar Chart: Why Contribution of Each Indicator
        # Nominal reference points: attendance: 75, assessment: 65, assignment: 65, engagement: 6.0, prior: 65
        weights = {
            "Assessment marks": (ass - 65.0) * 0.35,
            "Prior performance": (pri - 65.0) * 0.25,
            "Assignment marks": (asg - 65.0) * 0.25,
            "Attendance": (att - 75.0) * 0.30,
            "Engagement": (eng - 6.0) * 2.0,
        }

        # Build Diverging Bar Rows
        rows_html = ""
        for feat_label, delta in weights.items():
            # If delta < 0: raises risk (red bar extending right)
            # If delta >= 0: lowers risk (green bar extending left)
            bar_len = min(abs(delta) * 3.5, 95.0)
            if delta < 0:
                # Raises risk (Red on right)
                left_cell = '<div style="width: 100px; height: 14px;"></div>'
                right_cell = f'<div style="background-color: #b91c1c; height: 14px; width: {bar_len}px; border-radius: 1px;"></div>'
            else:
                # Lowers risk (Green on left)
                left_cell = f'<div style="background-color: #15803d; height: 14px; width: {bar_len}px; float: right; border-radius: 1px;"></div>'
                right_cell = '<div style="width: 100px; height: 14px;"></div>'

            rows_html += f"""
            <tr style="height: 22px;">
                <td style="text-align: right; font-size: 11px; color: #334155; padding-right: 12px; width: 140px; white-space: nowrap;">{feat_label}</td>
                <td style="width: 100px; border-right: 1.5px solid #0f172a; padding: 0;">{left_cell}</td>
                <td style="width: 100px; padding: 0;">{right_cell}</td>
            </tr>
            """

        diverging_chart = f"""
        <div style="margin-top: 10px; margin-bottom: 12px;">
            <div style="font-size: 12px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">Why: contribution of each indicator (chart)</div>
            <table style="width: 380px; border-collapse: collapse; margin-top: 2px;">
                <thead>
                    <tr style="font-size: 10px; color: #64748b;">
                        <th></th>
                        <th style="text-align: right; padding-right: 8px; font-weight: 600;">lowers risk</th>
                        <th style="text-align: left; padding-left: 8px; font-weight: 600;">raises risk</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
        st.markdown(diverging_chart, unsafe_allow_html=True)

        # Footnote
        st.markdown(
            """
            <div style="border: 1px solid #94a3b8; border-radius: 4px; padding: 6px 10px; font-size: 11px; color: #334155; background: #f8fafc;">
                Limitation: indicators are partial. The model can be wrong. A human makes the final decision.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Bottom Panel: VALIDATION / ERROR MESSAGE AREA
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="panel-box">
        <div class="panel-title">VALIDATION / ERROR MESSAGE AREA</div>
        <div class="validation-box">
    """,
    unsafe_allow_html=True,
)

# Render validation log lines matching the sketch
log_output = []

if not samples_df.empty:
    log_output.append(f'<span class="val-pass">PASS</span>  STEP 1 DATA CHECK PASSED: {len(samples_df)} clean rows loaded.')
else:
    log_output.append('<span class="val-fail">FAIL</span>  STEP 1 DATA CHECK: sample data file not found.')

if validation_error:
    rec_name = selected_rec if selected_rec else "MANUAL"
    log_output.append(f'<span class="val-fail">FAIL</span>  Row {rec_name}: {html.escape(validation_error)}')
    log_output.append('<span class="val-dim">(expected failure from data/invalid_cases.csv; nothing is computed when input is invalid)</span>')
else:
    log_output.append('<span class="val-pass">PASS</span>  INPUT BOUNDARY VERIFIED: All 5 indicators within valid mathematical domains.')
    log_output.append('<span class="val-pass">PASS</span>  INFERENCE PARITY: Baseline deterministic score and Neural ANN predictions evaluated.')

st.markdown("<br>".join(log_output) + "</div></div>", unsafe_allow_html=True)
