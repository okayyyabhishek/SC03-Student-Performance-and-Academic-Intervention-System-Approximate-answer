# Data Dictionary & Dataset Documentation (Step 1) — SC03

**Product:** Student Performance and Academic Intervention System  
**Stage:** Step 1 Project Preparation & Data Contract  
**Repository Path:** `data/README.md`  

---

## 1. Data Dictionary

The SC03 schema uses five academic indicators to predict an intervention risk category. All fields, units, allowed ranges, and handling rules are documented below:

| Field | Meaning in one sentence | Data Type | Physical Unit | Allowed Range / Format | Missing Value Policy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `record_id` | Unique pseudonymous identifier assigned to a student record | Text (Metadata) | Alphanumeric token | `^STU-\d{3,}$` (e.g., `STU-001`, unique) | **Reject record** |
| `attendance_pct` | Percentage of scheduled lectures and laboratory sessions attended | Float / Integer | Percentage (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `assessment_pct` | Continuous internal assessment marks and quiz score percentage | Float / Integer | Percentage (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `assignment_pct` | Graded practical lab assignments and homework problem sets percentage | Float / Integer | Percentage (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `engagement_score` | Standardized LMS portal interaction telemetry & participation index | Float / Integer | Score index | `0.0` to `10.0` inclusive | **Reject record** |
| `prior_performance_pct` | Cumulative prerequisite coursework or previous semester percentage | Float / Integer | Percentage (`%`) | `0.0` to `100.0` inclusive | **Reject record** |
| `risk_label` | Expected academic risk category (inference target) | Category | Categorical token | Exactly one of: `low`, `medium`, `high` | **Reject record** |

### Roles of Columns
- **Identifier:** `record_id` (pseudonymous row key)
- **Model Inputs:** `attendance_pct`, `assessment_pct`, `assignment_pct`, `engagement_score`, `prior_performance_pct`
- **Inference Target / Label:** `risk_label`

---

## 2. Distinction: Simulated Starter Cases vs. Real Student Records

> [!IMPORTANT]
> **Strict Non-Conflation Policy:**
> `data/sample_input.csv` contains **simulated synthetic starter test cases** manually authored by the team to verify validator logic and baseline arithmetic.
> - **They are NOT real student records:** The figures and labels represent educational scenarios designed to test edge cases, not empirical measurements of actual human students.
> - **Zero PII Mandate:** Real student names, email addresses, phone numbers, enrollment numbers, or roll numbers are strictly forbidden and never collected or stored.
> - **Candidate Benchmark Data (Step 2):** When scaling to 10,000+ records in Step 2, records will originate from vetted, publicly licensed de-identified educational corpora (e.g., Open University Learning Analytics Dataset - OULAD, CC BY 4.0) or reproducible mathematical scenario generators with documented non-circular ground truth.

---

## 3. Starter Sample Composition (`data/sample_input.csv`)

The 20 simulated starter rows are structured into four intentional testing cohorts:

| Group | Record IDs | Purpose & Characteristics |
| :--- | :--- | :--- |
| **Mandatory Golden Cases** | `STU-001` to `STU-005` | Prescribed by the course guide to test clean baseline behavior: consistently strong (`STU-001`), single weak indicator (`STU-002`), multiple weak indicators (`STU-003`), compensatory interaction (`STU-004`), exact threshold boundary (`STU-005`). |
| **Ordinary Valid Rows** | `STU-006` to `STU-010` | Standard low, medium, and high student distributions across typical operational ranges. |
| **Boundary Rows** | `STU-011` to `STU-015` | Minimum extremes (`0%`), maximum extremes (`100%`), and boundary values right at the `50.0` and `70.0` risk decision thresholds. |
| **Difficult / Conflicting Rows** | `STU-016` to `STU-020` | Realistic conflicting educational signals (e.g., `STU-016` has 95% attendance but 30% assessment; `STU-019` has 0 engagement but strong marks) where linear baseline formulas struggle and neural networks are required. |

---

## 4. Deliberately Invalid Test Fixture (`data/invalid_cases.csv`)

To prove that `src/validate_data.py` rejects invalid data before any model executes, `data/invalid_cases.csv` includes four test cases covering 7 distinct validation violations:

1. **Missing values:** Row `STU-021` has an empty `assessment_pct` cell $\implies$ caught & rejected.
2. **Invalid types:** Row `STU-022` provides string `"abc"` for numeric `assignment_pct` $\implies$ caught & rejected.
3. **Out-of-bounds percentage:** Row `STU-BAD` has `attendance_pct = 112` ($> 100$) $\implies$ caught & rejected.
4. **Out-of-bounds engagement:** Row `STU-BAD` has `engagement_score = -1` ($< 0$) $\implies$ caught & rejected.
5. **Invalid categorical label:** Row `STU-BAD` uses `risk_label = critical` (must be `low`, `medium`, or `high`) $\implies$ caught & rejected.
6. **Malformed identifier:** Row `STU-BAD` violates the regex format `^STU-\d{3,}$` $\implies$ caught & rejected.
7. **Duplicate identifiers:** Two rows share `STU-022` $\implies$ duplicate uniqueness violation caught & rejected.

---

## 5. System Outputs & Supportive Interventions

The system maps indicators to a risk band and suggests a supportive intervention:

| Risk Category | Primary Intervention | Student-Facing Interpretation | Intended Follow-up |
| :--- | :--- | :--- | :--- |
| `low` | **Monitor** | *On track* | Routine academic engagement; no mentor intervention needed. |
| `medium` | **Mentoring or study plan** | *Some support could help* | Faculty mentor assigns peer tutoring, study skills coaching, or revision modules. |
| `high` | **Faculty follow-up** | *Let's plan support together* | Direct, confidential mentor consultation to diagnose underlying academic roadblocks. |

> [!NOTE]
> **Human Authority:** The system provides decision-support only. A faculty mentor always reviews context and makes the final decision. Automated punitive actions, public rankings, and disciplinary flags are strictly prohibited.

---

## 6. Step 2 Plan (10,000+ Record Dataset)

- **Source Corpus:** Open University Learning Analytics Dataset (OULAD) or seeded synthetic generator.
  - **Citation:** Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171
  - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Non-Circularity:** Ground truth target labels will be derived independently of the baseline formula to guarantee scientifically valid model comparison.
- **Proxy Disclosures:** Any proxy mapping (e.g. active interaction days as a proxy for classroom attendance) will be explicitly documented and flagged.
