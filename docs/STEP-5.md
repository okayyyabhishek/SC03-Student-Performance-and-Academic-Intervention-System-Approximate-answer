# Step 5 - Product V1

## Team and project
- **Team name:** SC03 Capstone Project Team
- **Project code and title:** SC03 — Student Performance and Academic Intervention System
- **Branch and commit:** `step-5-product-v1` (`76a01ee`)

---

## User task
This interactive web application is designed for **academic faculty mentors, advisors, and course coordinators**. 

The mentor enters five documented continuous academic indicators for an anonymous student:
1. `attendance_pct` — Lecture and lab session attendance rate (0.0% to 100.0%)
2. `assessment_pct` — Continuous internal assessment and quiz score (0.0% to 100.0%)
3. `assignment_pct` — Homework and lab assignment completion score (0.0% to 100.0%)
4. `engagement_score` — Standardized LMS portal activity telemetry index (0.0 to 10.0)
5. `prior_performance_pct` — Cumulative prerequisite coursework grade percentage (0.0% to 100.0%)

The system runs inference using our saved Step 3 Artificial Neural Network (ANN) pipeline to estimate the student's academic risk band (`low`, `medium`, or `high`) and provides an immediate, tailored supportive intervention recommendation (e.g., routine progress checks, peer study mentoring, or urgent 1-on-1 advisor consultations).

---

## Interface and method
- **Interface type:** Streamlit Interactive Web Application
- **Entry file:** `app.py` (repository root)
- **Inference module:** `src/product_predict.py`
- **Saved model/pipeline:** `models/ann_student_performance.pkl` (Scikit-learn `Pipeline` pairing `StandardScaler` with a 3-layer `MLPClassifier` trained in Step 3)
- **Model metadata:** `models/ann_model_metadata.json` (features: `attendance_pct`, `assessment_pct`, `assignment_pct`, `engagement_score`, `prior_performance_pct`; classes: `['high', 'low', 'medium']`)
- **Input schema and units:**
  - `attendance_pct`: Float, unit: `%` (range 0.0 to 100.0)
  - `assessment_pct`: Float, unit: `%` (range 0.0 to 100.0)
  - `assignment_pct`: Float, unit: `%` (range 0.0 to 100.0)
  - `engagement_score`: Float, unit: `index` (range 0.0 to 10.0)
  - `prior_performance_pct`: Float, unit: `%` (range 0.0 to 100.0)
- **Output labels, units and meaning:**
  - `predicted_class`: Categorical class (`low`, `medium`, or `high`)
  - `label`: Plain-language risk tier (`Predicted low risk class`, `Predicted medium risk class`, `Predicted high risk class`)
  - `intervention`: Suggested pedagogical supportive action:
    - *Low risk:* Routine monitoring & check-in (Progress is healthy; periodic check)
    - *Medium risk:* Mentoring or study plan support (Targeted peer tutoring, study skills coaching)
    - *High risk:* Immediate faculty consultation & follow-up (Urgent 1-on-1 diagnostic counseling)
  - `notice`: Mandatory ethical governance disclaimer ("Educational prediction only. This decision-support output does not replace human mentor evaluation or constitute official disciplinary action.")
- **Data flow:** Input form / Load sample button $\rightarrow$ Strict type and bound validation $\rightarrow$ Saved model pipeline inference (`StandardScaler` + `MLPClassifier.predict()`) $\rightarrow$ Actionable explained triage output and CSV export.

---

## Run instructions
- **Tested Python version:** Python 3.11.9
- **Dependency installation command:**
  ```powershell
  python -m pip install -r requirements-product-v1.txt
  ```
- **Working directory:** Repository root (`Capstone Project`)
- **Interface start command:**
  ```powershell
  python -m streamlit run app.py
  ```
- **Expected local address:** `http://localhost:8501`
- **Automated test suite command:**
  ```powershell
  python -m unittest test_product_predict -v
  ```
- **Consistency verification command:**
  ```powershell
  python src/check_product_prediction.py
  ```
- **Demo status:** Local demonstration verified and reproducible.

---

## Actual test evidence

| Case | Actual input or evidence path | Expected behaviour | Observed behaviour | Pass/Fail |
|---|---|---|---|:---:|
| **Normal** | `attendance_pct=100.0`, `assessment_pct=87.2`, `assignment_pct=67.0`, `engagement_score=7.9`, `prior_performance_pct=92.1` (Row 0 of `data/processed/validation.csv`, `STU-07286`) | Clean validation pass; predicts `low` risk; recommends routine monitoring | Successfully displayed `Predicted low risk class`, suggested `Routine monitoring & check-in`, displayed governance notice, rendered data table and CSV download button | **PASS** |
| **Boundary** | Min: All features 0.0<br>Max: Percentages 100.0, Engagement 10.0 | Min and max bounds accepted without error; Min yields `high` risk, Max yields `low` risk | Clean execution without exception; Min produced `high` risk (`Immediate faculty consultation & follow-up`); Max produced `low` risk (`Routine monitoring & check-in`) | **PASS** |
| **Invalid or missing input** | Blank `attendance_pct=""`, Negative `assessment_pct="-5"`, Out-of-bounds `engagement_score="12"`, String `prior_performance_pct="abc"`, Missing `assignment_pct` | Raises specific `ValueError` detailing offending feature; prevents model inference | Form caught all validation errors, displayed field-specific error warnings, prevented model execution, no unhandled traceback | **PASS** |
| **Missing model/service** | Model artifact path points to non-existent path or corrupted file | Raises descriptive `FileNotFoundError` or displays graceful warning in UI | UI caught missing file, halted cleanly, displayed clear instructions to restore Step 3 model artifacts | **PASS** |
| **Same input through UI and inference code** | Input: `STU-07286` evaluated via `src/check_product_prediction.py` and `app.py` | Identical predictions, labels, and output features between script and web application | `data/reports/step5_ui_reference_prediction.csv` matched UI output exactly (Class: `low`, Label: `Predicted low risk class`, Intervention: `Routine monitoring & check-in`) | **PASS** |

---

## Earlier feedback addressed
- **Step 1 Feedback (Data privacy & scoping):** Ensured all test scenarios and demo inputs are strictly anonymous without student PII (using synthetic anonymous identifiers such as `STU-07286`).
- **Step 2 Feedback (Split integrity):** Built input validation module using verified feature bounds established in Step 2 data profiling. Demo sample loading directly references sealed Step 2 validation split without touching training data.
- **Step 3 Feedback (Model artifact serialization):** Embedded `StandardScaler` directly inside the Scikit-learn `Pipeline` artifact (`ann_student_performance.pkl`), eliminating manual rescaling errors at inference time.
- **Step 4 Feedback (Baseline comparison & separation):** Maintained strict independence between the baseline evaluation scripts (`src/baseline.py`, `src/compare_baseline.py`) and the interactive product application (`app.py`, `src/product_predict.py`). No baseline artifacts or earlier step evidence files were overwritten.

---

## Limitations and responsible use
1. **Decision-Support Only:** This application is strictly an early-alert academic advisory tool. Model outputs are suggestions and must **never** be used to automate punitive grading, financial aid revocation, or disciplinary action.
2. **Mandatory Human Faculty Review:** An academic mentor must review student context before initiating any intervention plan.
3. **Cohort Generalization Boundaries:** The neural model was trained on historical institutional patterns. Significant shifts in course curriculum or grading schemes require recalibration and drift validation.

---

## Team contributions
- **Abhishek (`@okayyyabhishek`) — Repository & Backend Lead:** Implemented `src/product_predict.py` inference pipeline, model loading logic, and environment configuration (`requirements-product-v1.txt`).
- **Saumya Rawat (`@saumyaarawatt`) — Data & ML Lead:** Formulated input validation rules, checked feature ranges against Step 2 data distributions, and verified model class consistency.
- **Aman Sharma (`@amansharma-link`) — Baseline & ML Support:** Built `src/check_product_prediction.py` consistency verification and validated feature-by-feature parity between UI and script outputs.
- **Riya Gupta (`@riyargupta09-star`) — Testing & UI Lead:** Developed Streamlit interactive frontend (`app.py`), authored unit test suite (`test_product_predict.py`), executed boundary test cases, and compiled `docs/STEP-5.md`.
