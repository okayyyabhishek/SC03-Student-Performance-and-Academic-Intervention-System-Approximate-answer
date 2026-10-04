# SC03 Data Quality Report

## Purpose
This report records how the Student Performance and Academic Intervention System (SC03) dataset was inspected, cleaned, validated, and split for model development and evaluation.

## Dataset Source
- **Dataset:** SC03 Student Performance Academic Benchmark
- **Source:** Open University Learning Analytics Dataset (OULAD) proxy architecture & simulated benchmark (Kuzilek et al., 2017 analogue)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original file:** `data/raw/student_performance_raw.csv`
- **Original records:** 12,005
- **Target column:** `risk_label`
  - `low` = On track / Low academic risk
  - `medium` = Emerging difficulty / Medium academic risk
  - `high` = Severe struggle / High academic risk

## Raw-Data Inspection

| Check | Result |
|---|---|
| Rows | 12,005 |
| Columns | 7 |
| Missing values in `attendance_pct` | 12 |
| Missing values in `assessment_pct` | 12 |
| Missing values in `assignment_pct` | 12 |
| Missing values in `engagement_score` | 12 |
| Missing values in `prior_performance_pct` | 12 |
| Total missing values | 60 |
| Duplicate rows | 5 |
| Low risk records | 5,340 |
| Medium risk records | 3,132 |
| High risk records | 3,528 |

## Cleaning Method
- The original raw CSV was preserved unchanged in `data/raw/`.
- Cleaning operated on a dedicated in-memory copy (`clean_df = df.copy()`).
- Missing continuous indicators were imputed using the median of each respective column.
- Exact duplicate rows were removed using `drop_duplicates()` and index was reset.
- The cleaned dataset was strictly checked for missing values, range bounds, target categories, and duplicate prevention.

## Validation Result
All validation checks passed after cleaning:
- No required columns were missing (`record_id`, `attendance_pct`, `assessment_pct`, `assignment_pct`, `engagement_score`, `prior_performance_pct`, `risk_label`).
- No missing values remained (`clean_df.isna().sum().sum() == 0`).
- All indicator percentages were strictly between `0.0` and `100.0`.
- All LMS engagement scores were strictly between `0.0` and `10.0`.
- `risk_label` contained only valid tokens: `low`, `medium`, and `high`.
- No duplicate rows remained (`0 duplicates`).

## Reproducible Split
A fixed random seed of `42` was used. The data was split separately for each risk class to preserve identical class distributions across splits:

| Dataset | Rows | Percentage | Purpose |
|---|---|---|---|
| `train.csv` | 8,400 | 70.0% | Soft computing model training & weight optimization |
| `validation.csv` / `val.csv` | 1,800 | 15.0% | Hyperparameter tuning & early stopping checkpointing |
| `test.csv` | 1,800 | 15.0% | Final unseen evaluation & baseline comparison |
| **Total Cleaned** | **12,000** | **100.0%** | **Complete cleaned benchmark dataset** |

## Output Files
The pipeline created the following artifacts in `data/processed/`:
- `student_performance_clean.csv` (12,000 rows)
- `dataset.csv` (12,000 rows)
- `train.csv` (8,400 rows)
- `validation.csv` (1,800 rows)
- `val.csv` (1,800 rows)
- `test.csv` (1,800 rows)
- `manifest.json` (Cryptographic verification and metadata)
- `protected_attributes.csv` (Isolated demographic audit records)

## Limitation and Next Action
The benchmark dataset contains 12,000 valid records. It must not be duplicated and described as new real observations.
The next step is generating and documenting 1,200 separate stress-testing scenarios. These scenarios will be clearly labelled as generated test scenarios, not real observations.

## Safety Note
This is an educational decision-support project. Predictions suggest supportive interventions and do not replace human mentor evaluation or official academic decisions.
