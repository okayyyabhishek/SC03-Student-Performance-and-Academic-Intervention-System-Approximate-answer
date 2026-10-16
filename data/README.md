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

## 4. Dataset Provenance, Synthetic Generation & Benchmark Corpus (Step 2)

### 4.1 Synthetic Dataset Nature & Provenance Declaration
- **Dataset Name:** SC03 Student Performance Academic Benchmark
- **Plain Statement on Synthetic Nature:** **All 12,000 academic records and ground-truth `risk_label` classifications in this repository are 100% synthetic**, generated algorithmically using a calibrated parametric simulation pipeline (`src/pipeline/build_dataset.py`, seeded with `numpy.random.default_rng(seed=42)`).
- **Not Real Student Observations:** These records are **not empirical observations of real students** and do not represent actual student records from the Open University or any other educational institution. No real student data, human subject experiments, or institutional grading registries were accessed or downloaded.
- **Design Inspiration (OULAD Attribution):** The Open University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta, & Zdrahal, 2017) served strictly as **high-level conceptual and architectural design inspiration** for schema indicator definitions and telemetry names (e.g., analogizing LMS portal engagement to VLE interaction logs, and continuous assessment marks to TMA/CMA coursework evaluations). **None of the records, rows, features, or labels in this repository were downloaded from OULAD, nor is this dataset an extract, slice, or subset of OULAD.**
  - *Academic Reference for Design Inspiration:* Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171
- **Licensing & Project Terms:** **Dedicated Synthetic Capstone Benchmark (MIT License / Capstone Academic Use)**. Any previous or inherited Creative Commons Attribution 4.0 (CC BY 4.0) attribution that implied records were downloaded from OULAD is hereby replaced.

### 4.2 Raw Test Fixture Explanation (`data/raw/student_performance_raw.csv`)
- **Nature of the Raw File:** The 12,005-row file located at `data/raw/student_performance_raw.csv` is a **synthetically generated raw test fixture** created specifically to test, benchmark, and validate the automated data ingestion, validation, and cleaning pipeline (`src/prepare_data.py`). It is **not an original source record file** from an external institutional database.
- **Deliberately Introduced Test Conditions:**
  - **Five Duplicate Rows:** Exactly 5 duplicate student records (1 high-risk, 4 medium-risk duplicates) were **artificially introduced / injected into the raw test fixture** to verify that the deduplication stage (`clean_data() -> drop_duplicates()`) reliably detects and purges duplicate telemetry without human intervention.
  - **Sixty Missing Feature Cells:** Exactly 60 null values (12 missing cells in each of the 5 numerical feature columns, spanning 12 distinct rows) were **intentionally injected into the raw test fixture** to verify that the median-imputation logic in `src/prepare_data.py` correctly fills missing values prior to schema validation.
- **Deduplication and Imputation Outcome:** After running `src/prepare_data.py`, all 5 duplicate rows are dropped and all 60 missing values are imputed, yielding the pristine 12,000-record benchmark dataset (`data/processed/dataset.csv`).

### 4.3 Mandatory Volume Threshold Verification (>= 10,000 Records)
- **Mandatory Course Guide Threshold:** $\ge 10,000$ valid student records.
- **Raw Test Fixture Records:** **12,005**
- **Cleaned Benchmark Records:** **12,000**
- **Compliance Status:** **VERIFIED & EXCEEDED (+20.0% above mandatory 10,000 threshold).**

### 4.4 Exact Generator Equations & Ground-Truth Label Rules
The synthetic benchmark was constructed offline via `src/pipeline/build_dataset.py` with fixed random seed `42` to guarantee exact mathematical reproducibility.

#### 1. Latent Student Traits
Each student record is driven by two unobserved, standard-normal latent factors:
- $A \sim \mathcal{N}(0, 1)$: Academic Aptitude & Prerequisite Domain Competence
- $W \sim \mathcal{N}(0, 1)$: Academic Work Ethic, Diligence, & Routine Study Habits

#### 2. Observable Indicator Generation with Measurement Noise
The 5 observable indicators are generated by linear combinations of latent traits, corrupted by calibrated Gaussian measurement noise $\epsilon_i$, and bounded to valid domain intervals:
- **`attendance_pct`**: $\text{clip}(72.0 + 15.0 \cdot W + 5.0 \cdot A + \epsilon_1, 0.0, 100.0), \quad \epsilon_1 \sim \mathcal{N}(0.0, 12.0)$
- **`assessment_pct`**: $\text{clip}(64.0 + 18.0 \cdot A + 8.0 \cdot W + \epsilon_2, 0.0, 100.0), \quad \epsilon_2 \sim \mathcal{N}(0.0, 14.0)$
- **`assignment_pct`**: $\text{clip}(66.0 + 12.0 \cdot A + 15.0 \cdot W + \epsilon_3, 0.0, 100.0), \quad \epsilon_3 \sim \mathcal{N}(0.0, 13.0)$
- **`engagement_score`**: $\text{clip}(6.2 + 1.8 \cdot W + 0.8 \cdot A + \epsilon_4, 0.0, 10.0), \quad \epsilon_4 \sim \mathcal{N}(0.0, 1.5)$
- **`prior_performance_pct`**: $\text{clip}(65.0 + 16.0 \cdot A + 5.0 \cdot W + \epsilon_5, 0.0, 100.0), \quad \epsilon_5 \sim \mathcal{N}(0.0, 13.0)$
*(All indicator values are rounded to 1 decimal place).*

#### 3. Composite Base Score
A baseline linear composite performance score is computed as:
$$\text{Composite} = 0.35 \cdot \text{assessment\_pct} + 0.25 \cdot \text{assignment\_pct} + 0.20 \cdot \text{attendance\_pct} + 0.12 \cdot (\text{engagement\_score} \times 10) + 0.08 \cdot \text{prior\_performance\_pct}$$

#### 4. Four Non-Linear Domain Interaction Penalties
Real educational struggle is asymmetric and non-linear. Four domain penalties modify the composite score:
1. **Compensatory Illusion (Unproductive Portal Cramming):** Students who exhibit high LMS portal clicks ($\text{engagement\_score} \ge 7.5$) but fail in-class assessments ($\text{assessment\_pct} < 45.0$).  
   $\implies \text{Penalty} += 14.0$
2. **Attendance Disengagement Spiral:** Attendance drops below $50.0\%$, creating compounding gaps and missed continuous assessments.  
   $\implies \text{Penalty} += 16.0$
3. **Prior Strength Breakdown:** High prior performance ($\ge 80.0\%$) fails to protect a student when attendance collapses below $55.0\%$.  
   $\implies \text{Penalty} += 12.0$
4. **Dual Coursework Collapse:** Severe concurrent failure in both continuous assessment and homework assignments ($< 48.0\%$).  
   $\implies \text{Penalty} += 15.0$

#### 5. Unobserved External Life Hardship Noise
Unpredictable external life events (health illness, family distress, financial shock) are simulated via zero-mean Gaussian noise:
$$\eta \sim \mathcal{N}(0.0, 6.0)$$

#### 6. Latent Effective Outcome & Ground-Truth Risk Labeling
The final latent outcome score is:
$$\text{Effective Outcome} = \text{Composite} - \sum \text{Penalties} + \eta$$

The ground-truth `risk_label` is assigned via strict decision thresholds:
- **$\text{Effective Outcome} < 52.0$** $\implies$ `risk_label` = **`high`** (Severe struggle / Intensive faculty intervention needed)
- **$52.0 \le \text{Effective Outcome} < 68.0$** $\implies$ `risk_label` = **`medium`** (Emerging difficulty / Mentoring or study plan needed)
- **$\text{Effective Outcome} \ge 68.0$** $\implies$ `risk_label` = **`low`** (On track / Monitor routine progress)

### 4.5 Exact Record Counts & Split Methodology
Splits were constructed using stratified random sampling partitioned strictly across the target `risk_label` classes to prevent class imbalance drift (`random_state=42`):

| Dataset Partition | File Path | Exact Row Count | Percentage | Class Distribution (`low` / `medium` / `high`) | Purpose |
|---|---|---|---|---|---|
| **Raw Test Fixture** | `data/raw/student_performance_raw.csv` | 12,005 | — | 5,339 (44.47%) / 3,139 (26.15%) / 3,527 (29.38%) | Synthetic uncleaned test corpus with 60 missing cells & 5 duplicates |
| **Cleaned Benchmark** | `data/processed/dataset.csv` | 12,000 | 100.0% | 5,339 (44.49%) / 3,135 (26.13%) / 3,526 (29.38%) | Deduplicated, median-imputed, schema-validated benchmark |
| **Training Split** | `data/processed/train.csv` | 8,400 | 70.0% | 3,737 (44.49%) / 2,195 (26.13%) / 2,468 (29.38%) | Soft computing model training & parameter optimization |
| **Validation Split** | `data/processed/val.csv` / `validation.csv` | 1,800 | 15.0% | 801 (44.50%) / 470 (26.11%) / 529 (29.39%) | Hyperparameter tuning & early-stopping checkpoints |
| **Test Split** | `data/processed/test.csv` | 1,800 | 15.0% | 801 (44.50%) / 470 (26.11%) / 529 (29.39%) | Sealed test set for unseen model evaluation |
| **Testing Scenarios** | `data/generated/generated_student_performance_test_scenarios.csv` | 1,200 | 10.0% ratio | 400 (33.33%) / 400 (33.33%) / 400 (33.33%) | Stress testing & boundary checks (never mixed into splits) |

- **Partition Disjointness:** Verified. $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$ (zero record ID overlap across splits).
- **Stratification Quality:** Maximum deviation between split class proportions and full corpus proportions is $< 0.02\%$.

### 4.6 Feature & Telemetry Nomenclature (Design Inspiration from OULAD)
- **`attendance_pct`:** Classroom lecture and lab attendance percentage (conceptual analogue of OULAD active interaction days relative to course duration).
- **`assessment_pct`:** Continuous internal assessment percentage (conceptual analogue of OULAD TMA tutor-marked weighted score).
- **`assignment_pct`:** Homework and practical assignment percentage (conceptual analogue of OULAD CMA computer-marked weighted score).
- **`engagement_score`:** Standardized LMS telemetry engagement indicator scaled to 0.0–10.0 (conceptual analogue of log-scaled VLE portal interaction clicks).
- **`prior_performance_pct`:** Prerequisite coursework and foundation cumulative percentage.
- **`risk_label` [INDEPENDENT GROUND TRUTH]:** Non-linear latent risk target with interaction penalties and external life noise ($44.5\%$ low, $26.1\%$ medium, $29.4\%$ high).

### 4.7 Baseline Leakage Check (TR-012)
- **Test Set Accuracy:** **78.28%**
- **Test Set Macro F1:** **0.7704**
- **Status:** **PASS (Non-Circular).** Ground truth is not derived from the baseline formula.

### 4.8 Methodological Scope & Real-World Evaluation Disclaimer
> [!WARNING]
> **CRITICAL METHODOLOGICAL DISCLAIMER ON SYNTHETIC EVALUATION:**  
> 1. **Pipeline & Mechanics Validation Only:** Evaluating baseline algorithms, ADALINE, Backpropagation MLPs, or any other machine learning models on ground-truth labels generated from the same simulated feature rules demonstrates algorithmic convergence, pipeline mechanics, and the capacity of non-linear models to separate complex interaction surfaces that linear heuristics fail on.
> 2. **Does NOT Establish Real-World Intervention Performance:** Performance metrics (accuracy, F1-score, recall) achieved on these synthetic labels **do not establish or validate real-world intervention efficacy, student outcome improvement, or operational reliability in a live institutional setting.**
> 3. **Real-World Complexities:** Real students exhibit psychological, socioeconomic, pedagogic, and institutional behaviors that cannot be fully captured by synthetic mathematical generators. Actual deployment requires prospective institutional trials, ethical review board oversight, mentor validation, and longitudinal observational follow-ups.

### 4.9 Cryptographic Checksums (TR-011)
- `dataset.csv`: `424f7eaa2efe4640ae004669e36680ab49233df869b06a9ff56c5b8cbdbda86e`
- `train.csv`: `7de86756ffbcc8abe1a4d20f8071a9852ae8a1145ca9457fc411eeddd0741b2e`
- `val.csv`: `a2fc12efd09e793a4f26135d2bdc193a44d20b4520ac3477e0e5ee46ba631885`
- `test.csv`: `292f333ed4fc5fedbbf06e643d783703f327d22b1c5a22543bc4321116b55dfc`
- `protected_attributes.csv`: `9db174ae5b9138c2acf4bb7e1a276ac8927a71478eeec24ff606c59f8be2170a`

### 4.10 Step 2 Evidence & Verification Reports
- **Data Quality Report:** [`data/reports/data_quality_report.md`](reports/data_quality_report.md)
- **Executed Jupyter Notebook:** [`notebooks/01_data_inspection.ipynb`](../notebooks/01_data_inspection.ipynb)
- **Scenario Generation Report:** [`data/reports/generated_scenarios_report.md`](reports/generated_scenarios_report.md)
- **Dataset Manifest:** [`data/processed/manifest.json`](processed/manifest.json)

---

## 5. Privacy & Ethical Non-Negotiables
- **No PII:** Names, emails, phone numbers, or institutional roll numbers must never enter this directory.
- **Protected Attributes (TR-014):** Demographic attributes (`gender`, `age_band`, `disability`, `socio_economic_decile`) are stored strictly in `data/processed/protected_attributes.csv` and **never** merged into training features. They are reserved exclusively for the Step 7 fairness audit.
- **Decision Support Only:** Model outputs are advisory recommendations for faculty mentors; automated punitive or administrative actions are strictly prohibited.

