# Student Performance and Academic Intervention System (SC03)

> A role-based academic decision-support web application that consumes anonymous academic indicators, estimates an academic risk band, suggests supportive interventions, and contrasts a simple score baseline with transparent neural classifiers.

**One-Sentence Problem Statement:**  
*Given anonymous academic indicators, estimate a risk band and suggest a supportive intervention.*

---

## 1. Branch Navigation & Milestone Roadmap

The repository follows a branch-per-milestone structure. This branch (`main`) serves as the official **Step 1 Foundation & Contract Landing Page**:

| Branch | Milestone | Key Deliverables & Contents | Status |
| :--- | :--- | :--- | :--- |
| [`main`](https://github.com/okayyyabhishek/SC03-Student-Performance-and-Academic-Intervention-System-Approximate-answer/tree/main) | **Step 1** | Project Contract, Data Dictionary, 20-row simulated starter dataset, baseline rules & pseudocode, validator CLI, PASS/FAIL evidence, Product V1 sketch | **COMPLETE (All Evidence Verified)** |
| [`step-2-data-pipeline`](https://github.com/okayyyabhishek/SC03-Student-Performance-and-Academic-Intervention-System-Approximate-answer/tree/step-2-data-pipeline) | **Step 2** | 12,000-record benchmark dataset, non-circular ground-truth generator, stratified 70/15/15 split, demographic isolation, SHA-256 manifest | **COMPLETE** |
| [`step-3-ann-model`](https://github.com/okayyyabhishek/SC03-Student-Performance-and-Academic-Intervention-System-Approximate-answer/tree/step-3-ann-model) | **Step 3** | Executed ANN Jupyter notebook, standalone training script, model pipeline artifact, loss curves, confusion matrices | **COMPLETE** |

---

## 2. Team Members and Responsibilities

The team consists of four members with rotating leadership across project milestones. Step 1 initial responsibilities and milestone specializations are documented below:

| Member | Real Name | GitHub Username | First Responsibility | Exact Step 1 Job | Milestone Focus (M1 / M2) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Member 1** | **Abhishek** | [`@okayyyabhishek`](https://github.com/okayyyabhishek) | Repository & Backend Lead | Create folder structure, configure environment, verify Git workflow, test clone and execution steps | FastAPI backend, RBAC middleware, Docker containerization, deployment |
| **Member 2** | **Saumya Rawat** | [`@saumyaarawatt`](https://github.com/saumyaarawatt) | Data & ML Lead | Define inputs/outputs, units, ranges, compile 20-row simulated starter sample, compile Data Dictionary | 10k+ dataset pipeline, NumPy neural training (ADALINE & Backprop MLP), metrics |
| **Member 3** | **Aman Sharma** | [`@amansharma-link`](https://github.com/amansharma-link) | Baseline & ML Support | Formulate weighted baseline rules, compute expected scores for five mandatory cases, write pseudocode | Baseline engine, model comparisons, noise robustness, demographic fairness audit |
| **Member 4** | **Riya Gupta** | [`@riyargupta09-star`](https://github.com/riyargupta09-star) | Testing & UI Lead | Write standalone validation script, design Product V1 screen sketch, record PASS/FAIL terminal evidence | React frontend, Design System token adherence, Twin Read visualization, WCAG a11y |

*Team Discipline:* Responsibilities rotate across project steps to prevent knowledge silos and ensure all members can defend every component in viva examinations.

---

## 3. Data Dictionary

The SC03 schema processes five academic indicators into a risk level and suggested intervention:

| Field | Meaning in one sentence | Data Type | Physical Unit | Allowed Range / Format | Missing Value Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `record_id` | Unique pseudonymous identifier assigned to a student record | Text (Metadata) | Alphanumeric token | `^STU-\d{3,}$` (e.g., `STU-001`, unique) | **Reject record** |
| `attendance_pct` | Percentage of scheduled lectures and laboratory sessions attended | Float / Integer | Percentage (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `assessment_pct` | Continuous internal assessment marks and quiz score percentage | Float / Integer | Percentage (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `assignment_pct` | Graded practical lab assignments and homework problem sets percentage | Float / Integer | Percentage (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `engagement_score` | Standardized LMS portal interaction telemetry & participation index | Float / Integer | Score index | `0.0` to `10.0` inclusive | **Reject record** |
| `prior_performance_pct` | Cumulative prerequisite coursework or previous semester percentage | Float / Integer | Percentage (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `risk_label` | Expected academic risk category (inference target) | Category | Categorical token | Exactly one of: `low`, `medium`, `high` | **Reject record** |

Full specifications and justification are in [`data/README.md`](data/README.md).

---

## 4. Distinction: Simulated Starter Cases vs. Real Student Records

> [!IMPORTANT]
> **Strict Non-Conflation Policy:**
> - **Starter Cases are Simulated:** The 20 starter rows in [`data/sample_input.csv`](data/sample_input.csv) are **synthetic educational test scenarios** authored by the team to verify schema validation rules and baseline arithmetic. They do **not** represent real human students and must never be interpreted as real student performance.
> - **Zero PII Mandate:** Real student names, email addresses, phone numbers, enrollment numbers, or roll numbers are strictly forbidden and never collected, stored, or processed.
> - **Candidate Benchmark Data (Step 2):** When scaling to 10,000+ records in Step 2, records will originate from vetted, publicly licensed de-identified educational corpora (e.g., Open University Learning Analytics Dataset - OULAD, CC BY 4.0) or reproducible mathematical scenario generators with documented non-circular ground truth.

---

## 5. Outputs and Supportive Interventions

The system pairs every estimated risk category with an actionable, non-punitive supportive intervention:

| Risk Category | Primary Intervention | Student-Facing Interpretation | Intended Follow-up |
| :--- | :--- | :--- | :--- |
| `low` | **Monitor** | *On track* | Routine academic engagement; no mentor intervention needed. |
| `medium` | **Mentoring or study plan** | *Some support could help* | Faculty mentor assigns peer tutoring, study skills coaching, or revision modules. |
| `high` | **Faculty follow-up** | *Let's plan support together* | Direct, confidential mentor consultation to diagnose underlying academic roadblocks. |

> [!NOTE]
> **Human Decision-Support Only:** The system suggests interventions; a qualified faculty mentor always confirms or modifies the action based on individual context. The system **never** automates punishment, disciplinary flags, or public student ranking.

---

## 6. Deterministic Baseline Method

The comparison baseline is a transparent, deterministic linear weighted scoring formula. It is **not** fuzzy logic, an artificial neural network, or a genetic algorithm.

All weights and thresholds below are **TEMPORARY** classroom values subject to empirical tuning in Step 3:
1. **Convert engagement score:** Convert `engagement_score` (0–10 scale) to a percentage:
   $$\text{engagement\_pct} = \text{engagement\_score} \times 10$$
2. **Compute visible score:**
   $$\text{score} = 0.30 \times \text{attendance\_pct} + 0.30 \times \text{assessment\_pct} + 0.20 \times \text{assignment\_pct} + 0.10 \times \text{engagement\_pct} + 0.10 \times \text{prior\_performance\_pct}$$
3. **Map to temporary risk bands:**
   - $\text{score} < 50.0 \implies \mathbf{high}$ risk $\implies$ **Faculty follow-up**
   - $50.0 \le \text{score} < 70.0 \implies \mathbf{medium}$ risk $\implies$ **Mentoring or study plan**
   - $\text{score} \ge 70.0 \implies \mathbf{low}$ risk $\implies$ **Monitor**
4. **Mandatory Disclaimer:** *"A human makes the final decision."*

Formal baseline pseudocode and worked calculations are documented in [`docs/baseline-pseudocode.md`](docs/baseline-pseudocode.md).

---

## 7. Five Mandatory Test Cases (Golden Tests)

| Case ID | Input Indicators (Att, Assess, Assign, Eng, Prior) | Expected Label | Test Objective / Characteristic | Baseline Score | Baseline Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **STU-001** | `92, 84, 88, 8, 81` | `low` | Consistently strong record across all metrics | **86.5** | `low`, Monitor |
| **STU-002** | `42, 75, 78, 7, 74` | `medium` | Low attendance only; strong marks elsewhere | **65.1** | `medium`, Mentoring or study plan |
| **STU-003** | `55, 35, 30, 3, 40` | `high` | Multiple failing indicators | **40.0** | `high`, Faculty follow-up |
| **STU-004** | `78, 48, 52, 9, 45` | `medium` | High engagement despite weak exam marks | **61.7** | `medium`, Mentoring or study plan |
| **STU-005** | `70, 60, 60, 5, 60` | `medium` | Exact boundary threshold case | **62.0** | `medium`, Mentoring or study plan |
| **STU-BAD** | `112, 84, 88, -1, 81, critical` | *Rejected* | Impossible values (Att $>100$, Eng $<0$, bad label) | *Not computed* | **VALIDATION FAILED** (see [`data/invalid_cases.csv`](data/invalid_cases.csv)) |

---

## 8. Documented Repository Map & Evidence Paths

All Step 1 evidence files are present at their exact documented paths on this branch:

```text
SC03-Student-Performance-and-Academic-Intervention-System/
├── README.md                           # Project overview, Data Dictionary, team roles, and run steps
├── STEP-1.md                           # Step 1 Project Contract (root copy)
├── requirements.txt                    # Pinned Python dependencies
├── .gitignore                          # Standard git ignore rules
├── data/
│   ├── README.md                       # Comprehensive Data Dictionary, units, sources & limits
│   ├── sample_input.csv                # 20 clean simulated starter rows
│   └── invalid_cases.csv               # Deliberately invalid test fixture (7 errors caught)
├── docs/
│   ├── STEP-1.md                       # Step 1 Project Contract
│   ├── baseline-pseudocode.md          # Formal pseudocode for deterministic weighted baseline
│   └── product-v1-sketch.png           # Product V1 UI screen sketch
├── src/
│   └── validate_data.py                # Step 1 standalone CLI data validator
└── results/
    └── step1/
        ├── validation_pass.txt         # Terminal capture showing PASS on data/sample_input.csv
        ├── validation_fail.txt         # Terminal capture showing FAIL on data/invalid_cases.csv
        ├── validation_pass.png         # Terminal screenshot of successful validation
        ├── validation_fail.png         # Terminal screenshot of rejected invalid cases
        ├── checklist.md                # 10-point Step 1 Definition of Done approval checklist
        └── faculty_demo_script.md      # Scripted 10-step faculty demonstration walkthrough
```

### Direct Evidence Links
- **Step 1 Project Contract:** [`docs/STEP-1.md`](docs/STEP-1.md) (and [`STEP-1.md`](STEP-1.md))
- **Data Dictionary & Synthetic Data Policy:** [`data/README.md`](data/README.md)
- **20 Valid Simulated Starter Rows:** [`data/sample_input.csv`](data/sample_input.csv)
- **Deliberately Invalid Fixture:** [`data/invalid_cases.csv`](data/invalid_cases.csv)
- **Baseline Pseudocode & Rules:** [`docs/baseline-pseudocode.md`](docs/baseline-pseudocode.md)
- **Product V1 Screen Sketch:** [`docs/product-v1-sketch.png`](docs/product-v1-sketch.png)
- **Data Validator Program:** [`src/validate_data.py`](src/validate_data.py)
- **PASS Validation Terminal Evidence:** [`results/step1/validation_pass.txt`](results/step1/validation_pass.txt)
- **Expected FAIL Validation Terminal Evidence:** [`results/step1/validation_fail.txt`](results/step1/validation_fail.txt)
- **Step 1 DoD Checklist:** [`results/step1/checklist.md`](results/step1/checklist.md)
- **Faculty Demo Walkthrough:** [`results/step1/faculty_demo_script.md`](results/step1/faculty_demo_script.md)

---

## 9. Quickstart & Verification (Step 1)

### 1. Set Up Python Environment
```powershell
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Step 1 Data Validator
Verify that the 20-row valid sample passes and the invalid test fixture is rejected with 7 clear errors:

```powershell
# 1. Valid starter sample (prints PASS):
python src/validate_data.py data/sample_input.csv

# 2. Invalid fixture (prints clear validation errors):
python src/validate_data.py data/invalid_cases.csv
```

**Expected PASS Terminal Output:**
```text
file: data/sample_input.csv
shape: (20, 7)
columns: ['record_id', 'attendance_pct', 'assessment_pct', 'assignment_pct', 'engagement_score', 'prior_performance_pct', 'risk_label']
missing values: {'record_id': 0, 'attendance_pct': 0, 'assessment_pct': 0, 'assignment_pct': 0, 'engagement_score': 0, 'prior_performance_pct': 0, 'risk_label': 0}
STEP 1 DATA CHECK PASSED
```

**Expected FAIL Terminal Output:**
```text
file: data/invalid_cases.csv
shape: (4, 7)
columns: ['record_id', 'attendance_pct', 'assessment_pct', 'assignment_pct', 'engagement_score', 'prior_performance_pct', 'risk_label']
missing values: {'record_id': 0, 'attendance_pct': 0, 'assessment_pct': 1, 'assignment_pct': 0, 'engagement_score': 0, 'prior_performance_pct': 0, 'risk_label': 0}
VALIDATION FAILED: 7 problem(s) found
  1. Row STU-021: assessment_pct is missing
  2. Row STU-022: assignment_pct must be a number, got "abc"
  3. Row STU-BAD: attendance_pct must be between 0 and 100, got 112
  4. Row STU-BAD: engagement_score must be between 0 and 10, got -1
  5. Row STU-BAD: risk_label must be low, medium or high, got critical
  6. Row STU-BAD: record_id must look like STU-001, got STU-BAD
  7. Duplicate record_id: STU-022
```

---

## 10. Ethical Notice & Limitations

- **Decision-Support Tool:** SC03 is developed solely as an educational academic intervention decision-support system. Its outputs must never replace human faculty mentoring or formal committee reviews.
- **Supportive Purpose:** The application is explicitly designed to offer supportive, constructive intervention recommendations rather than automated disciplinary flags or ranking.
- **Student Privacy:** All processed academic indicators remain de-identified. No personal data is stored in the modeling system.
