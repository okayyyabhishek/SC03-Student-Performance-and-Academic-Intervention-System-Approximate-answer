# Step 1 Project Contract — SC03

**Product:** Student Performance and Academic Intervention System  
**Stage:** Step 1 Foundation, Contract, Data Specification & Evidence  
**Status:** COMPLETE (All Step 1 evidence verified & documented)  

---

## 1. Team and Responsibilities

The team consists of four members with dedicated first responsibilities for Step 1 and aligned milestone leadership:

| Member | Real Name | GitHub Username | First Responsibility | Exact Step 1 Job | Milestone Focus (M1 / M2) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Member 1** | **Abhishek** | [`@okayyyabhishek`](https://github.com/okayyyabhishek) | Repository & Backend Lead | Create folder structure, configure environment, verify Git workflow, test clone and execution steps | FastAPI backend, RBAC middleware, Docker containerization, deployment |
| **Member 2** | **Saumya Rawat** | [`@saumyaarawatt`](https://github.com/saumyaarawatt) | Data & ML Lead | Define inputs/outputs, units, ranges, compile 20-row simulated starter sample, compile Data Dictionary | 10k+ dataset pipeline, NumPy neural training (ADALINE & Backprop MLP), metrics |
| **Member 3** | **Aman Sharma** | [`@amansharma-link`](https://github.com/amansharma-link) | Baseline & ML Support | Formulate weighted baseline rules, compute expected scores for five mandatory cases, write pseudocode | Baseline engine, model comparisons, noise robustness, demographic fairness audit |
| **Member 4** | **Riya Gupta** | [`@riyargupta09-star`](https://github.com/riyargupta09-star) | Testing & UI Lead | Write standalone validation script, design Product V1 screen sketch, record PASS/FAIL terminal evidence | React frontend, Design System token adherence, Twin Read visualization, WCAG a11y |

*Note on Rotation:* These are primary initial responsibilities to prevent knowledge silos, not permanent barriers. Every team member understands and can defend every component in viva examinations.

---

## 2. One-Sentence Problem Statement

> **Given anonymous academic indicators, estimate a risk band and suggest a supportive intervention.**

Mentors frequently identify struggling students too late in a semester, after major examination failures have already taken place. Key indicators—attendance, assessment marks, homework assignments, LMS engagement, and prior grades—exist across disparate systems but are analyzed in isolation. SC03 combines these indicators into an early, transparent, and supportive decision-support signal.

---

## 3. Users of the Product

- **Primary User — Faculty Mentor:** Reviews assigned student cohorts, evaluates risk estimates, reads model explanations, and confirms or customizes supportive intervention actions.
- **Secondary User — Administrator:** Oversees system configuration, dataset governance, model registry, and demographic fairness audits.
- **Secondary User — Student:** Views their own academic indicators and constructive, empathetic feedback without peer rankings or exposure to model internals.

---

## 4. Inputs and Units

All input features are strictly numerical or categorical with well-defined operational units and zero missing-value tolerance:

| Field | Meaning | Type / Unit | Allowed Range / Format | Missing Value Rule |
| :--- | :--- | :--- | :--- | :--- |
| `record_id` | Anonymous row identifier | Text | `^STU-\d{3,}$` (e.g., `STU-001`, unique) | **Reject record** |
| `attendance_pct` | Classroom lecture and lab attendance | Percent (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `assessment_pct` | Continuous internal assessment / test marks | Percent (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `assignment_pct` | Graded homework and lab assignment marks | Percent (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `engagement_score` | LMS portal participation index | Score | `0.0` to `10.0` inclusive | **Reject record** |
| `prior_performance_pct` | Cumulative prerequisite / prior grade | Percent (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `risk_label` | Expected validation risk category (target) | Category | Exactly one of: `low`, `medium`, `high` | **Reject record** |

*Missing Value Policy:* Zero tolerance. Any row containing missing, null, or out-of-range values is rejected immediately by the validator (`src/validate_data.py`). See [data/README.md](data/README.md) for full data dictionary notes.

---

## 5. Outputs and Units

The system produces a transparent risk assessment paired with a non-punitive intervention recommendation:

- **Risk Category:** `low`, `medium`, or `high`.
- **Supportive Intervention:**
  - `low` $\implies$ **Monitor** (routine check-ins, student is on track).
  - `medium` $\implies$ **Mentoring or study plan** (peer tutoring, targeted study skills coaching, revision modules).
  - `high` $\implies$ **Faculty follow-up** (immediate, supportive one-on-one consultation with faculty mentor).

> [!IMPORTANT]
> **Human Decision-Support Only:** The system suggests interventions; a qualified faculty mentor always confirms or modifies the action based on individual context. The system **never** automates punishment, disciplinary flags, or public student ranking.

---

## 6. Baseline Method

The comparison baseline is a transparent, deterministic linear weighted scoring formula. It is **not** fuzzy logic, an artificial neural network, or a genetic algorithm.

All weights and thresholds below are **TEMPORARY** classroom values subject to empirical tuning in Step 3:
1. **Convert engagement score:** Convert `engagement_score` (0–10 scale) to a percentage:
   $$\text{engagement\_pct} = \text{engagement\_score} \times 10$$
2. **Compute visible score:**
   $$\text{score} = 0.30 \times \text{attendance\_pct} + 0.30 \times \text{assessment\_pct} + 0.20 \times \text{assignment\_pct} + 0.10 \times \text{engagement\_pct} + 0.10 \times \text{prior\_performance\_pct}$$
3. **Map to temporary risk bands:**
   - $\text{score} < 50.0 \implies \mathbf{high}$ risk.
   - $50.0 \le \text{score} < 70.0 \implies \mathbf{medium}$ risk.
   - $\text{score} \ge 70.0 \implies \mathbf{low}$ risk.
4. **Map risk band to supportive intervention:**
   - $\mathbf{low} \implies \text{Monitor}$
   - $\mathbf{medium} \implies \text{Mentoring or study plan}$
   - $\mathbf{high} \implies \text{Faculty follow-up}$
5. **Human Authority Disclaimer:** *"A human makes the final decision."*

Formal baseline pseudocode and worked calculations are documented in [docs/baseline-pseudocode.md](docs/baseline-pseudocode.md).

---

## 7. Soft Computing Method for M1 & Advanced Method for M2

- **Milestone 1 (M1):** Train a transparent neural classifier (pure NumPy ADALINE with Widrow-Hoff Delta rule gradient descent or MLP) and contrast its predictions side-by-side with the score baseline in an interactive Product V1 UI using the **Twin Read** visual scale.
- **Milestone 2 (M2):** Compare three from-scratch neural architectures (Perceptron, ADALINE, Backpropagation MLP); evaluate error patterns, class imbalance mitigation, noise robustness ($\sigma = 2\%, 5\%, 10\%$), and demographic fairness audits across protected cohorts.

---

## 8. Dataset and Scenario Sources: Simulated vs. Real Distinction

> [!NOTE]
> **Honesty & Synthetic Data Policy:**
> - **Step 1 Sample (`data/sample_input.csv`):** A small **simulated starter dataset** (20 rows) written by the team. These rows are **synthetic test scenarios** created to verify validator logic and baseline math. They are **not real student records** and must never be represented as real data.
> - **Privacy Guarantee:** The schema strictly forbids personal identifiers. No real student names, email addresses, phone numbers, or institutional roll numbers are collected, stored, or processed.
> - **Step 2 Benchmark Plan:** In Step 2, the team will scale to a 10,000+ record dataset using a cited, publicly licensed de-identified educational dataset (Open University Learning Analytics Dataset - OULAD, CC BY 4.0) or a documented non-circular scenario generator. Ground-truth labels will be derived independently of the baseline formula to prevent circular evaluation.

---

## 9. Five Mandatory Test Cases (Golden Tests)

| Case ID | Input Indicators (Att, Assess, Assign, Eng, Prior) | Expected Label | Test Objective / Characteristic | Baseline Score | Baseline Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **STU-001** | `92, 84, 88, 8, 81` | `low` | Consistently strong record across all metrics | **86.5** | `low`, Monitor |
| **STU-002** | `42, 75, 78, 7, 74` | `medium` | Low attendance only; strong marks elsewhere | **65.1** | `medium`, Mentoring or study plan |
| **STU-003** | `55, 35, 30, 3, 40` | `high` | Multiple failing indicators | **40.0** | `high`, Faculty follow-up |
| **STU-004** | `78, 48, 52, 9, 45` | `medium` | High engagement despite weak exam marks | **61.7** | `medium`, Mentoring or study plan |
| **STU-005** | `70, 60, 60, 5, 60` | `medium` | Exact boundary threshold case | **62.0** | `medium`, Mentoring or study plan |
| **STU-BAD** | `112, 84, 88, -1, 81, critical` | *Rejected* | Impossible values (Att $>100$, Eng $<0$, bad label) | *Not computed* | **VALIDATION FAILED** (see `data/invalid_cases.csv`) |

---

## 10. Product V1 Screen Sketch

![Product V1 Screen Sketch](docs/product-v1-sketch.png)

The Product V1 screen layout incorporates:
1. **Left Input Panel:** Five indicator fields with visible units (`%`, `/10`), valid range hints, and a primary **Run Assessment** action button.
2. **Right Results Panel:** The **Twin Read** visualization displaying the Baseline marker above the scale and the Neural Model marker below the scale, accompanied by color-coded status chips and per-feature explanation bars.
3. **Footer Status Area:** Clear validation message display and the mandatory ethics limitation banner: *"Indicators are partial. The model can be wrong. A human makes the final decision."*

The sketch image is preserved in both [docs/product-v1-sketch.png](docs/product-v1-sketch.png) and root [product-v1-sketch.png](product-v1-sketch.png).

---

## 11. Risks and Assumptions

1. **Simulated Starter Data:** Results on the 20 starter rows test code logic only; they do not generalize to real students.
2. **Temporary Thresholds:** The 50.0 and 70.0 score boundaries are tagged `TEMPORARY = True` and will be tuned empirically against benchmark distributions.
3. **Intentional Baseline Disagreements:** Rows `STU-016`, `STU-017`, and `STU-019` in `data/sample_input.csv` deliberately disagree with the linear baseline to illustrate where soft computing models are required.
4. **Circularity Prevention:** The 10,000+ benchmark dataset ground truth must not use the baseline formula.
5. **Proxy Transparency:** Any proxy indicator used in Step 2 (e.g. LMS click counts as attendance proxy) will be explicitly disclosed in [data/README.md](data/README.md).

---

## 12. Step 1 Completion Evidence

Every promised Step 1 evidence artifact is verified, accessible, and committed at the documented repository paths:

| Evidence Artifact | Documented Path | Secondary / Root Path | Status |
| :--- | :--- | :--- | :--- |
| **Collaborators & Team** | GitHub Settings $\to$ Collaborators | [README.md](README.md#3-team-members-and-responsibilities) | **Verified (4 Members)** |
| **Project Contract** | [`docs/STEP-1.md`](docs/STEP-1.md) | [`STEP-1.md`](STEP-1.md) | **Complete & Verified** |
| **Data Dictionary** | [`data/README.md`](data/README.md) | [`README.md`](README.md#2-data-dictionary) | **Complete & Verified** |
| **20 Valid Simulated Starter Rows** | [`data/sample_input.csv`](data/sample_input.csv) | [`sample_input.csv`](sample_input.csv) | **Complete (20 Valid Rows)** |
| **Deliberately Invalid Cases** | [`data/invalid_cases.csv`](data/invalid_cases.csv) | — | **Complete (7 Errors Caught)** |
| **Baseline Rules & Pseudocode** | [`docs/baseline-pseudocode.md`](docs/baseline-pseudocode.md) | — | **Complete & Verified** |
| **Standalone Data Validator** | [`src/validate_data.py`](src/validate_data.py) | [`validate_data.py`](validate_data.py) | **Complete & Verified** |
| **PASS Execution Evidence** | [`results/step1/validation_pass.txt`](results/step1/validation_pass.txt) | [`results/step1/validation_pass.png`](results/step1/validation_pass.png) | **Terminal Capture Verified** |
| **Expected FAIL Execution Evidence** | [`results/step1/validation_fail.txt`](results/step1/validation_fail.txt) | [`results/step1/validation_fail.png`](results/step1/validation_fail.png) | **Terminal Capture Verified** |
| **Product V1 Screen Sketch** | [`docs/product-v1-sketch.png`](docs/product-v1-sketch.png) | [`product-v1-sketch.png`](product-v1-sketch.png) | **Exported & Verified** |
| **DoD Checklist** | [`results/step1/checklist.md`](results/step1/checklist.md) | — | **All 10 Items PASS** |
| **Faculty Demo Script** | [`results/step1/faculty_demo_script.md`](results/step1/faculty_demo_script.md) | — | **Rehearsed & Verified** |
| **Commits from All Members** | Git Commit History (`answer/main`) | Commits tab | **All 4 Members Active** |
