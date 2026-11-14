# Step 6 - Testing and validation

## Project and environment
- **Project:** SC03 Student Performance and Academic Intervention System (instructor reference adapted).
- **Branch:** `step-6-testing`.
- **Commit tested:** `e7d7986`.
- **Python and dependencies:** Python 3.11.9, pandas 2.2.2, scikit-learn 1.5.1, streamlit 1.67.0.
- **Launch from the repository root:** `python -m streamlit run app.py`.
- **Reference input:** first data row in `data/processed/validation.csv` (`STU-07286`).

---

## Planned tests and results

Keep the other fields fixed when testing a single input. For each run, replace the blank observation with what the interface or command actually showed. Use PASS, FAIL or NOT RUN. Record a relative evidence path or a short exact output.

| ID | Input or action | Expected behaviour | Actual result | Status |
| --- | --- | --- | --- | --- |
| T01 | Load existing sample and predict | Display a class, label and educational notice. Allow CSV download. | Showed class `low`, `Predicted low risk class`, and the educational notice. The CSV downloaded successfully. | PASS |
| T02 | Reload sample and set lower boundary (attendance=0) | Accept the lower limit and display a prediction. | Accepted attendance 0 and returned class `high` (`Predicted high risk class`). | PASS |
| T03 | Reload sample and set upper boundary (attendance=100, engagement=10) | Accept the upper limit and display a prediction. | Accepted attendance 100, engagement 10 and returned class `low` (`Predicted low risk class`). | PASS |
| T04 | Reload sample and set engagement_score to 15 | Show the range message and withhold the result. | Displayed `engagement_score: use a value from 0 to 10.` and withheld the prediction. | PASS |
| T05 | Reload sample and enter 'abc' as prior_performance_pct | Ask for a number and withhold the result. | Displayed `prior_performance_pct: enter a number.` and withheld the prediction. | PASS |
| T06 | Reload sample and clear assignment_pct | Ask for a value and withhold the result. | Displayed `assignment_pct: enter a value.` and withheld the prediction. | PASS |
| T07 | Test with model file missing/renamed | Explain that the saved model or metadata is missing and stop before prediction. | The browser showed `The saved model could not be loaded: Saved model or metadata is missing. Restore your Step 3 files.` The prediction form was unavailable. | PASS |
| T08 | Restore the model filename and relaunch | The normal sample predicts again. | The form returned and showed `Predicted low risk class` and Model class: `low`. | PASS |
| T09 | Run `python src/compare_step6_predictions.py` to compare UI export with reference CSV | All five inputs, the predicted class and the label match. | The script reported `PASS: All five inputs, the predicted class and the label match. Class low - Predicted low risk class`. | PASS |
| T10 | Run `python -m unittest -v test_product_predict` | Five validation methods run and finish with OK. | All five tests showed `ok`. The terminal reported `Ran 5 tests` and ended with `OK`. | PASS |

---

## Evidence

| Test ID | File path or exact observed output |
| --- | --- |
| T01 | Browser download saved as `data/reports/step6_ui_prediction.csv` |
| T02-T03 | Browser results: lower boundary returned class `high`; upper boundary returned class `low` |
| T04-T06 | The browser showed field-specific validation error messages (`engagement_score: use a value from 0 to 10.`, `prior_performance_pct: enter a number.`, `assignment_pct: enter a value.`) and withheld the prediction |
| T07-T08 | App caught missing model file and showed Step 3 restore advice; after restoring model file, normal sample predicted class `low` |
| T09 | `src/compare_step6_predictions.py` compared `data/reports/step6_ui_prediction.csv` with `data/reports/step5_ui_reference_prediction.csv`. The script reported `PASS` |
| T10 | `test_product_predict.py`: five named tests showed `ok`, followed by `Ran 5 tests in 0.000s` and `OK` |

---

## Failures, fixes and retests

- The comparison script `src/compare_step6_predictions.py` was created to automate field-by-field verification between the browser download (`step6_ui_prediction.csv`) and reference pipeline output (`step5_ui_reference_prediction.csv`). All 5 numeric features, predicted class, and label matched within $10^{-9}$ tolerance.
- Input validation was verified across boundary limits (0.0% to 100.0% for percentages, 0.0 to 10.0 for engagement index). Out-of-bounds, non-numeric, and blank inputs were rejected cleanly without crashing or exposing unhandled stack traces.
- No app defects were observed in the test cases recorded above.

---

## Known limits

- The CSV comparison used one validation sample (`STU-07286`). It proves that the browser UI and offline Python script agree for that sample.
- The automated unit tests check input validation and boundaries; model quality was established in Step 3/4.
- SC03 is an educational decision-support demonstration. Its predictions cannot automate disciplinary action or replace faculty judgment.
- Evaluated on Windows 11 with Python 3.11.9 and Chrome browser. Testing was not conducted on mobile viewports or Safari.

---

## Contributions

- **Abhishek (`@okayyyabhishek`) — Repository & Backend Lead:** Created `step-6-testing` branch, designed comparison script `src/compare_step6_predictions.py`, and verified git synchronization.
- **Saumya Rawat (`@saumyaarawatt`) — Data & ML Lead:** Validated boundary cases (T02-T03) and invalid range scenarios (T04-T06) against Step 2 data bounds.
- **Aman Sharma (`@amansharma-link`) — Baseline & ML Support:** Performed reference CSV generation, parity check comparison, and model file availability tests (T07-T08).
- **Riya Gupta (`@riyargupta09-star`) — Testing & UI Lead:** Executed browser tests (T01-T06), authored `docs/STEP-6.md`, ran automated unit tests (T10), and compiled evidence table.

