# SC03 Student Performance and Academic Intervention System

A decision-support app that estimates an academic **risk band** (low, medium, high) from anonymous indicators and suggests a **supportive intervention**. A faculty mentor reads the result; a human always makes the final decision. Nothing happens to a student automatically.

**One-sentence problem:** Given anonymous academic indicators, estimate a risk band and suggest a supportive intervention.

**Status:** Step 1 (project preparation). No neural network or app yet. See `docs/STEP-1.md`.

## Data Dictionary

| Field | Simple meaning | Type/unit | Starter rule |
| :--- | :--- | :--- | :--- |
| `record_id` | Anonymous row name | text | `STU-001`, `STU-002`, ... (unique) |
| `attendance_pct` | Classes attended | percent | 0 to 100 |
| `assessment_pct` | Assessment marks | percent | 0 to 100 |
| `assignment_pct` | Assignment marks | percent | 0 to 100 |
| `engagement_score` | Participation indicator | score | 0 to 10 |
| `prior_performance_pct` | Earlier performance | percent | 0 to 100 |
| `risk_label` | Expected label for testing | category | `low`, `medium`, `high` |

Missing values are not allowed in Step 1: a row with any missing value is rejected.

## Outputs

- Risk category: low, medium or high
- One supportive intervention: monitor, mentoring or study plan, or faculty follow-up

## Repository map

```
README.md                     this file (project overview + Data Dictionary)
requirements.txt              Python packages
data/README.md                field meanings, units, sources, assumptions
data/sample_input.csv         20 valid starter rows
data/invalid_cases.csv        deliberately wrong rows (must FAIL validation)
docs/STEP-1.md                project contract
docs/baseline-pseudocode.md   simple baseline rules
docs/product-v1-sketch.png    Product V1 screen sketch
src/validate_data.py          data validator
results/step1/                PASS and FAIL evidence
```

## How to run (Step 1)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
python src/validate_data.py                          # expect: STEP 1 DATA CHECK PASSED
python src/validate_data.py data/invalid_cases.csv   # expect: VALIDATION FAILED with 7 problems
```

## Team

See `docs/STEP-1.md` for members and responsibilities.
