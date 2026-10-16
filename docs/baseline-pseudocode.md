# Baseline pseudocode

The baseline is a deliberately simple comparison method. It is **not** fuzzy logic, an ANN or a GA. All weights and thresholds are **TEMPORARY** classroom values, not official standards.

```
READ one valid input row
IF the row is invalid:
    SHOW a clear error (field, value, rule)
ELSE:
    engagement_pct = engagement_score * 10
    score = 0.30 * attendance_pct
          + 0.30 * assessment_pct
          + 0.20 * assignment_pct
          + 0.10 * engagement_pct
          + 0.10 * prior_performance_pct

    IF score < 50:
        risk = high      intervention = faculty follow-up
    ELSE IF score < 70:
        risk = medium    intervention = mentoring or study plan
    ELSE:
        risk = low       intervention = monitor

    PRINT risk, score, intervention, and the rule that was used
    PRINT "A human makes the final decision."
SAVE the result for later comparison
```

## Expected baseline behaviour for the five mandatory cases

| Case | Input (att, assess, assign, eng, prior) | Score | Baseline band | Expected label | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| STU-001 | 92, 84, 88, 8, 81 | 86.5 | low | low | Consistently strong record |
| STU-002 | 42, 75, 78, 7, 74 | 65.1 | medium | medium | Low attendance only |
| STU-003 | 55, 35, 30, 3, 40 | 40.0 | high | high | Several weak indicators |
| STU-004 | 78, 48, 52, 9, 45 | 61.7 | medium | medium | High engagement but weak marks |
| STU-005 | 70, 60, 60, 5, 60 | 62.0 | medium | medium | Boundary record |
| Invalid | STU-BAD, 112, 84, 88, -1, 81 | not computed | rejected | n/a | Validator error |

Worked example, STU-004: 0.30x78 = 23.4; 0.30x48 = 14.4; 0.20x52 = 10.4; 0.10x90 = 9.0; 0.10x45 = 4.5. Total 61.7, which is between 50 and 70, so medium risk and a mentoring or study plan.

## Known limits (why we need Soft Computing later)

STU-016 (high attendance, very weak marks) scores 54.5 so the baseline says medium while the team label is high. STU-017 (low attendance, strong marks) scores 73.9 so the baseline says low while the team label is medium. STU-019 (zero engagement) scores 75.3 so the baseline says low while the team label is medium. A fixed weighted sum cannot capture these interactions.
