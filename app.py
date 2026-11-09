"""SC03 Step 5: run with python -m streamlit run app.py."""
import pandas as pd
import streamlit as st
from src.product_predict import FEATURES, ROOT, load_model, predict_one

st.set_page_config(page_title="SC03 Academic Intervention - Product V1", layout="centered")
st.title("Student Academic Risk Prediction: Product V1")
st.write("Enter the five academic indicators used by our saved Step 3 ANN.")
st.warning(
    "Teaching and decision-support example only. A model prediction does not replace "
    "faculty mentoring or certify academic disciplinary actions."
)

# Field labels with units
LABELS = {
    "attendance_pct": "Attendance Rate (% 0-100)",
    "assessment_pct": "Continuous Assessment Score (% 0-100)",
    "assignment_pct": "Assignment Completion Score (% 0-100)",
    "engagement_score": "LMS Engagement Score (0-10 index)",
    "prior_performance_pct": "Prior Academic Performance (% 0-100)",
}


@st.cache_resource
def saved_model():
    return load_model()


try:
    model = saved_model()
except Exception as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.info("Use the Step 3 Python environment and check the model and metadata paths.")
    st.stop()

if st.button("Load one existing example"):
    try:
        example = pd.read_csv(ROOT / "data" / "processed" / "validation.csv", nrows=1)
        for name in FEATURES:
            st.session_state[name] = str(example.iloc[0][name])
    except Exception as error:
        st.error(f"Example could not be loaded: {error}")

with st.form("academic_indicators"):
    values = {
        name: st.text_input(LABELS[name], value=st.session_state.get(name, ""), key=name)
        for name in FEATURES
    }
    submitted = st.form_submit_button("Predict using saved ANN")

if submitted:
    try:
        result = predict_one(model, values)
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
    else:
        st.subheader(result["label"])
        st.write(f"**Suggested Intervention:** {result['intervention']}")
        st.write(f"Model class: `{result['predicted_class']}`")
        st.caption(result["notice"])
        output = pd.DataFrame([result])
        st.dataframe(output)
        st.download_button(
            "Download this result as CSV",
            output.to_csv(index=False).encode("utf-8"),
            file_name="step5_prediction.csv",
            mime="text/csv",
        )

with st.expander("How this demonstration works"):
    st.write("The interface validates inputs and calls predict() on the saved pipeline.")
    st.write("The pipeline already includes its fitted scaler. No training runs here.")
    st.write(
        "SC03 is a reference project for Student Academic Performance and Intervention. "
        "Faculty mentors retain discretion over actual student support plans."
    )
