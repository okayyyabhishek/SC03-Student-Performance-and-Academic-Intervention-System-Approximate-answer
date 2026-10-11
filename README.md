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

---

## 3. Team Responsibilities (Step 1)

Roles rotate across project steps to prevent knowledge silos and ensure all members can defend every component in viva examinations:

| Member | Assigned Role | Step 1 Focus | Milestone Focus (M1 / M2) |
|---|---|---|---|
| **Member A** | Repository & Backend Lead | Repository architecture, environment setup, Git workflow | FastAPI backend, RBAC middleware, Docker, deployment |
| **Member B** | Data & ML Lead | Data dictionary, 20-row sample data, source citations | 10k+ pipeline, NumPy ADALINE & Backprop training, metrics |
| **Member C** | Baseline & ML Support | Weighted baseline rules, golden test checks, pseudocode | Baseline engine, model comparison, noise & fairness audit |
| **Member D** | Testing & UI Lead | Data validation scripts, UI wireframe sketch, test evidence | React frontend, Design System token adherence, WCAG a11y |

*(Note: For 3-member teams, Repository Lead and Testing/UI Lead are merged).*

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

---

## 6. Milestone Plan

- **M1 (Mid-Semester):** Deliver working Product V1: anonymous input → validation → baseline score & ADALINE neural classifier → Twin Read visual comparison → supportive intervention recommendation.
- **M2 (End-Semester):** Implement and compare Perceptron, ADALINE, and Backpropagation; study error patterns, class imbalance, noise robustness, and fairness; deploy Product V2 publicly with full viva evidence package.
