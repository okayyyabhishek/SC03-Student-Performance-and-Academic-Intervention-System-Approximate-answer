# Student Performance and Academic Intervention System (SC03)

> A role-based academic decision-support web application that consumes anonymous academic indicators, estimates an academic risk band, suggests supportive interventions, and contrasts a simple score baseline with transparent neural classifiers.

---

## 1. Problem Statement

**One-sentence problem (from the course guide):**  
*Given anonymous academic indicators, estimate a risk band and suggest a supportive intervention.*

Mentors often notice struggling students late in a semester, after marks have already dropped. Indicators like attendance, assessment marks, assignments, participation, and earlier grades exist, but are typically checked in isolation. This project provides a single, combined, explainable signal focused on early support, not punitive action.

---

## 2. Core Principles & Governance

1. **Human Decision-Support Only:** The system suggests interventions; a human faculty mentor confirms or overrides with a documented reason. It **never** automates punishment, disciplinary flags, or public ranking.
2. **Anonymous by Design:** The dataset contains no student names, email addresses, phone numbers, or real identifiers. Only anonymous IDs (e.g., `STU-001`) are used.
3. **No Circular Evaluation:** Ground-truth labels are derived independently from the baseline formula, ensuring genuine model comparison.
4. **From-Scratch Soft Computing:** Perceptron, ADALINE, and Backpropagation models are implemented in pure NumPy with visible weights and reproducible gradient descent updates.
5. **Least-Privilege RBAC:** Three strictly decreasing role tiers:
   - **Administrator:** System governance, model registry, fairness audits, dataset metadata.
   - **Faculty Mentor:** Assigned cohort only, pseudonymous records, runs assessments, manages interventions.
   - **Student:** Own record only, supportive wording, no peer comparison or model internals.
6. **Design Discipline:** Strict typography (Literata serif headings, Atkinson Hyperlegible body, IBM Plex Mono code) and dual light/dark themes shipping together.
7. **Simulated Starter Cases vs. Real Records:** The 20 starter rows in `data/sample_input.csv` are strictly simulated synthetic test scenarios constructed to test validator logic and baseline math. They are **not real student records** and must never be conflated with real student data. Real student identifiers (PII) are strictly prohibited.

---

## 3. Team Responsibilities (Step 1)

Roles rotate across project steps to prevent knowledge silos and ensure all members can defend every component in viva examinations:

| Member | Real Name | GitHub Username | Assigned Role | Step 1 Focus | Milestone Focus (M1 / M2) |
|---|---|---|---|---|---|
| **Member 1** | **Abhishek** | [`@okayyyabhishek`](https://github.com/okayyyabhishek) | Repository & Backend Lead | Repository architecture, environment setup, Git workflow, verification | FastAPI backend, RBAC middleware, Docker, deployment |
| **Member 2** | **Saumya Rawat** | [`@saumyaarawatt`](https://github.com/saumyaarawatt) | Data & ML Lead | Data dictionary, 20-row simulated sample, source citations, bounds | 10k+ pipeline, NumPy ADALINE & Backprop training, metrics |
| **Member 3** | **Aman Sharma** | [`@amansharma-link`](https://github.com/amansharma-link) | Baseline & ML Support | Weighted baseline rules, golden test checks, pseudocode documentation | Baseline engine, model comparison, noise & fairness audit |
| **Member 4** | **Riya Gupta** | [`@riyargupta09-star`](https://github.com/riyargupta09-star) | Testing & UI Lead | Data validation scripts, UI wireframe sketch, PASS/FAIL test evidence | React frontend, Design System token adherence, WCAG a11y |

---

## 4. Repository Structure

```text
SC03-Student-Performance-and-Academic-Intervention-System/
├── README.md                      # Project overview, setup, and team roles
├── AGENT.md                       # AI coding agent operating rules
├── requirements.txt               # Pinned Python dependencies
├── .gitignore                     # Git ignore rules
├── docs/                          # Specifications and contracts
│   ├── SC03-Beginner-Step-1-Guide.pdf # Faculty course guide
│   ├── STEP-1.md                  # Step 1 Project Contract
│   ├── PRD.md                     # Product Requirements Document
│   ├── TRD.md                     # Technical Requirements Document
│   ├── ARCHITECTURE.md            # Three-tier role architecture & security
│   ├── DESIGN-SYSTEM.md           # Tokens, typography, and UI rules
│   ├── APP-FLOW.md                # Route journeys and wireframes
│   ├── IMPLEMENTATION-PLAN.md     # 8-step roadmap & milestone gates
│   ├── BASELINE_PSEUDOCODE.md     # Pseudocode for weighted baseline
│   └── DECISIONS.md               # Architectural decision log (ADR)
├── data/                          # Data store (strictly anonymous)
│   ├── README.md                  # Data dictionary, units, sources & limits
│   ├── sample_input.csv           # 20 clean starter rows (Step 1)
│   ├── invalid_cases.csv          # Deliberately invalid test cases
│   └── invalid_input.csv          # Legacy test fixture
├── src/                           # Backend & ML source code
│   ├── validate_data.py           # Step 1 standalone CLI data validator
│   ├── baseline/                  # Step 3 score baseline module
│   ├── ml/                        # Step 4 & 6 NumPy neural models (ADALINE, etc.)
│   ├── pipeline/                  # Step 2 dataset ingestion & manifest builder
│   ├── api/                       # Step 5 FastAPI application & role routers
│   └── db/                        # Database models, schemas & seed scripts
├── notebooks/                     # Exploratory analysis (never primary logic)
├── app/                           # Step 5 React + Vite + TypeScript web application
├── results/                       # Verification evidence per step
│   ├── step1/                     # PASS and FAIL validation terminal captures
│   └── ...                        # step2 through step8 evidence folders
├── report/                        # Interim (M1) and final (M2) report sources
└── tests/                         # Automated pytest test suites
```

---

## 5. Quickstart & Verification

### Prerequisites
- Python 3.11+
- Node.js 18+ (for Step 5 onwards)

### 1. Set Up Python Virtual Environment
```powershell
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Step 1 Data Validator
Verify that the 20-row valid sample passes and the invalid test fixture is rejected with clear errors:
```powershell
# Valid starter sample (prints PASS):
python src/validate_data.py data/sample_input.csv

# Invalid fixture (prints clear validation errors):
python src/validate_data.py data/invalid_cases.csv
```

### 3. Run Step 2 Data Pipeline & Quality Profiling
Generate clean stratified 70/15/15 splits, data quality report, and verification manifest:
```powershell
python src/pipeline/build_dataset.py
python src/pipeline/profile_dataset.py
```
- Documentation: [`docs/STEP-2.md`](docs/STEP-2.md)
- Data Quality Report: [`data/reports/data_quality_report.md`](data/reports/data_quality_report.md)

### 4. Run Step 3 Artificial Neural Network (ANN)
Train the multi-layer feed-forward neural network with backpropagation and evaluate on test data:
```powershell
python src/train_ann.py
```
- Documentation: [`docs/STEP-3.md`](docs/STEP-3.md)
- Executed Notebook: [`notebooks/02_ann_model.ipynb`](notebooks/02_ann_model.ipynb)
- Model Artifact: [`models/ann_student_performance.pkl`](models/ann_student_performance.pkl)

### 5. Run Step 4 Simple Baseline & Empirical Model Comparison
Evaluate the deterministic weighted baseline on the sealed test set ($N=1,800$) and generate comparative metrics:
```powershell
# Full test set evaluation and comparison plot generation:
python src/baseline.py --evaluate
# Or via standard guide entry point:
python src/compare_baseline.py

# Single-student CLI prediction:
python src/baseline.py --attendance 45.0 --assessment 38.0 --assignment 50.0 --engagement 3.5 --prior 52.0

# Automated Step 4 unit tests:
python -m pytest tests/test_baseline.py -v
```
- Documentation: [`docs/STEP-4.md`](docs/STEP-4.md)
- Executed Notebook: [`notebooks/03_baseline.ipynb`](notebooks/03_baseline.ipynb)
- Comparison Metrics: [`data/reports/step4_comparison_metrics.csv`](data/reports/step4_comparison_metrics.csv)
- Full Predictions Table: [`data/reports/step4_test_comparison_full.csv`](data/reports/step4_test_comparison_full.csv)
- Visual Artifacts:
  - Confusion Matrix Comparison: [`results/step4_baseline_confusion_matrix.png`](results/step4_baseline_confusion_matrix.png)
  - Model Performance Bar Chart: [`results/step4_model_comparison_bar.png`](results/step4_model_comparison_bar.png)
  - Twin Read Cases: [`results/step4_twin_read_cases.png`](results/step4_twin_read_cases.png)

---

## 6. Milestone Plan

- **M1 (Mid-Semester):** Deliver working Product V1: anonymous input → validation → baseline score & ADALINE/ANN neural classifier → Twin Read visual comparison → supportive intervention recommendation.
- **M2 (End-Semester):** Implement and compare Perceptron, ADALINE, and Backpropagation; study error patterns, class imbalance, noise robustness, and fairness; deploy Product V2 publicly with full viva evidence package.
