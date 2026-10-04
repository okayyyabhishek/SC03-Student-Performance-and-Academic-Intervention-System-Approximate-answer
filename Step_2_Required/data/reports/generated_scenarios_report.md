# SC03 Generated Testing Scenarios Report

## Purpose
These records are generated testing scenarios for the SC03 Student Performance and Academic Intervention System. They are not new real student records and are not presented as real observations.

## Source and Size
- **Cleaned cited benchmark records:** 12,000
- **Generated scenario ratio:** 10%
- **Generated scenarios:** 1,200
- **Output file:** `data/generated/generated_student_performance_test_scenarios.csv`

## Reproducible Method
- **Fixed random seed:** 42
- A balanced set was sampled from the three real-data classes (`low`, `medium`, `high`).
- Continuous indicators (`attendance_pct`, `assessment_pct`, `assignment_pct`, `engagement_score`, `prior_performance_pct`) were adjusted with Gaussian jitter scaled to the feature interquartile range (IQR).
- Values were clipped to observed valid domains: percentages between `0.0` and `100.0`, and engagement score between `0.0` and `10.0`.
- Expected risk label is inherited from the sampled source class for testing expectations only; it is not a fresh mentor-assigned label.

## Validation Performed
- Exactly 1,200 scenario IDs were created (`GEN-STU-0001` to `GEN-STU-1200`).
- Zero missing values, zero duplicate scenario IDs, zero duplicate generated feature rows, and zero exact copies of real feature rows.
- All non-negative bounds and percentage checks passed.
- Expected labels: `low`: 400, `medium`: 400, `high`: 400.

## Strict Usage Rule
Use these scenarios ONLY for product testing, UI input validation, and robustness demonstration. Do not mix them into `train.csv`, `val.csv`, `validation.csv`, or `test.csv`, and do not use them to claim model learning accuracy.
