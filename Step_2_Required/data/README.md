# Data Documentation & Data Dictionary — SC03

**Product:** Student Performance and Academic Intervention System  
**Stage:** Step 1 Data Specification & 10,000+ Record Plan  

---

## 1. Data Dictionary

Each field in the SC03 academic schema has been rigorously defined according to Guide Step 1C:

### 1.1 `record_id`
- **One-sentence meaning:** Unique pseudonymous identifier assigned to an individual student record.
- **Classification:** Text identifier (Metadata).
- **Unit:** Alphanumeric code (Format: `^STU-\d{3,}$`).
- **Allowed values for starter sample:** `STU-001` to `STU-020`.
- **Missing value policy:** **Reject record.** No assessment can be associated with an unidentifiable row.
- **Source / Justification:** Follows FERPA/GDPR pseudonymous key guidelines to decouple identity from modeling data.

### 1.2 `attendance_pct`
- **One-sentence meaning:** Percentage of scheduled classroom lectures and laboratory sessions attended by the student.
- **Classification:** Continuous numeric input.
- **Unit:** Percentage (`%`).
- **Allowed values for starter sample:** `0.0` to `100.0` inclusive (e.g., `92.0`).
- **Missing value policy:** **Reject record.** Missing attendance cannot be assumed to be 0 or 100 in live inference.
- **Source / Justification:** Standard university biometric or manual attendance registry tracking minimum mandatory attendance thresholds (e.g., 75% rule).

### 1.3 `assessment_pct`
- **One-sentence meaning:** Percentage scored on continuous internal assessments, in-class quizzes, and mid-term tests.
- **Classification:** Continuous numeric input.
- **Unit:** Percentage (`%`).
- **Allowed values for starter sample:** `0.0` to `100.0` inclusive (e.g., `84.0`).
- **Missing value policy:** **Reject record.** Unrecorded test scores require manual faculty verification rather than automated imputation.
- **Source / Justification:** University continuous internal assessment (CIA) grading logs.

### 1.4 `assignment_pct`
- **One-sentence meaning:** Percentage scored across homework problem sets, practical lab assignments, and term projects.
- **Classification:** Continuous numeric input.
- **Unit:** Percentage (`%`).
- **Allowed values for starter sample:** `0.0` to `100.0` inclusive (e.g., `88.0`).
- **Missing value policy:** **Reject record.**
- **Source / Justification:** Learning Management System (LMS) assignment gradebook.

### 1.5 `engagement_score`
- **One-sentence meaning:** Standardized indicator measuring a student's active participation, LMS portal activity, and discussion forum engagement.
- **Classification:** Discrete / continuous numeric input.
- **Unit:** Score on a normalized `0.0` to `10.0` scale.
- **Allowed values for starter sample:** `0.0` to `10.0` inclusive (e.g., `8.0`).
- **Missing value policy:** **Reject record.**
- **Source / Justification:** Standard LMS telemetry (e.g., Moodle/Canvas activity indices), mapped to a 0–10 scale. Multiplied by 10 in baseline calculations for percentage parity.

### 1.6 `prior_performance_pct`
- **One-sentence meaning:** Cumulative academic performance percentage from prerequisite coursework or previous academic semesters.
- **Classification:** Continuous numeric input.
- **Unit:** Percentage (`%`).
- **Allowed values for starter sample:** `0.0` to `100.0` inclusive (e.g., `81.0`).
- **Missing value policy:** **Reject record.**
- **Source / Justification:** Historical transcript records / cumulative grade point average (CGPA) scaled to 100%.

### 1.7 `risk_label` (Target / Validation Ground Truth)
- **One-sentence meaning:** Categorical label indicating the estimated level of academic risk requiring intervention.
- **Classification:** Categorical target.
- **Unit:** String token (`low`, `medium`, `high`).
- **Allowed values for starter sample:** Exactly one of: `low`, `medium`, `high` (lowercase).
- **Missing value policy:** **Reject record.**
- **Source / Justification:** Derived from independent academic outcomes (e.g., course pass/fail/distinction status).

---

## 2. System Outputs Specification

Outputs are strictly decoupled from model internals when presented to different roles:

| Output Field | Meaning | Values / Format | Role Visibility |
|---|---|---|---|
| **Risk Band** | Estimated level of student risk | `low`, `medium`, `high` | Administrator, Faculty Mentor |
| **Supportive Intervention** | Actionable institutional follow-up | `Monitor`, `Mentoring or study plan`, `Faculty follow-up` | Administrator, Faculty Mentor |
| **Student-Facing Voice** | Non-stigmatizing empathetic status | `On track`, `Some support could help`, `Let's plan support together` | Student only |
| **Baseline Score** | Linear weighted reference score | Float `0.0` to `100.0` | Administrator, Faculty Mentor |
| **Model Output** | Continuous scores & predicted band | Per-class linear outputs (ADALINE) / Softmax probabilities (MLP) | Administrator, Faculty Mentor |
| **Explanation** | Per-indicator contribution breakdown | Direction (`raises_risk` / `lowers_risk`) + bar weight | Administrator, Faculty Mentor |
| **Limitation Banner** | Mandatory ethical disclaimer | *"Indicators are partial. The model can be wrong. A human makes the final decision."* | All Roles |

---

## 3. Starter Sample Composition (`data/sample_input.csv`)

The 20 starter rows represent four distinct real-world student categories:

1. **Mandatory Golden Cases (Rows 1–5):**
   - `STU-001`: Consistently strong indicators across all areas $\implies$ `low` risk.
   - `STU-002`: Strong marks but low attendance ($42\%$) $\implies$ `medium` risk.
   - `STU-003`: Severe deficits across multiple indicators $\implies$ `high` risk.
   - `STU-004`: High engagement ($9/10$) despite weak exam marks ($48\%$) $\implies$ `medium` risk.
   - `STU-005`: Boundary case near exact thresholds ($60\%$) $\implies$ `medium` risk.
2. **Ordinary Valid Rows (Rows 6–10):** Typical distributions of low, medium, and high performers.
3. **Boundary Rows (Rows 11–15):** Records hovering within $\pm 2\%$ of the 50% and 70% threshold lines.
4. **Difficult / Mixed Condition Rows (Rows 16–20):** Conflicting signals (e.g., excellent prior performance but zero recent LMS engagement).

---

## 4. Benchmark Dataset & Processed Corpus (Step 2)

Step 2 scaled the project to an active **12,000-record** benchmark dataset produced by `src/pipeline/build_dataset.py` with seed `42`, incorporating OULAD proxy architecture and non-linear ground truth.

### 4.1 Processed Directory Layout (`data/processed/`)
- `dataset.csv`: 12,000 rows (full combined corpus).
- `train.csv`: 8,400 rows (70% stratified training split).
- `val.csv`: 1,800 rows (15% stratified validation split).
- `test.csv`: 1,800 rows (15% sealed test split).
- `protected_attributes.csv`: 12,000 rows (demographic attributes held out of model features).
- `manifest.json`: Full cryptographic manifest with SHA-256 hashes, class counts, and leakage check.

### 4.2 Documented Feature & Proxy Mapping (TR-015)
- **`attendance_pct` [PROXY]:** Classroom attendance percentage proxy (analogue of OULAD active interaction days relative to course duration).
- **`assessment_pct`:** Continuous internal assessment percentage (analogue of OULAD TMA tutor-marked weighted score).
- **`assignment_pct`:** Homework and practical assignment percentage (analogue of OULAD CMA computer-marked weighted score).
- **`engagement_score` [PROXY]:** Standardized LMS telemetry engagement indicator scaled to 0.0–10.0 (analogue of log-scaled VLE portal interaction clicks).
- **`prior_performance_pct`:** Prerequisite coursework and foundation cumulative percentage.
- **`risk_label` [INDEPENDENT GROUND TRUTH]:** Non-linear latent risk target with interaction penalties and external noise ($44.5\%$ low, $26.1\%$ medium, $29.4\%$ high).

### 4.3 Baseline Leakage Check (TR-012)
- **Test Set Accuracy:** **78.28%**
- **Test Set Macro F1:** **0.7704**
- **Status:** **PASS (Non-Circular).** Ground truth is not derived from the baseline formula.

### 4.4 Cryptographic Checksums (TR-011)
- `dataset.csv`: `424f7eaa2efe4640ae004669e36680ab49233df869b06a9ff56c5b8cbdbda86e`
- `train.csv`: `7de86756ffbcc8abe1a4d20f8071a9852ae8a1145ca9457fc411eeddd0741b2e`
- `val.csv`: `a2fc12efd09e793a4f26135d2bdc193a44d20b4520ac3477e0e5ee46ba631885`
- `test.csv`: `292f333ed4fc5fedbbf06e643d783703f327d22b1c5a22543bc4321116b55dfc`
- `protected_attributes.csv`: `9db174ae5b9138c2acf4bb7e1a276ac8927a71478eeec24ff606c59f8be2170a`

---

## 5. Privacy & Ethical Non-Negotiables
- **No PII:** Names, emails, phone numbers, or institutional roll numbers must never enter this directory.
- **Protected Attributes (TR-014):** Demographic attributes (`gender`, `age_band`, `disability`, `socio_economic_decile`) are stored strictly in `data/processed/protected_attributes.csv` and **never** merged into training features. They are reserved exclusively for the Step 7 fairness audit.

