# SC03 Step 4 — Simple Baseline & Empirical Model Comparison

## 1. Comparison Question
> **"Does the Step 3 Artificial Neural Network (ANN) classify multi-tier student academic risk bands (`low`, `medium`, `high`) and suggest appropriate supportive interventions more accurately and reliably than a deterministic weighted-sum threshold baseline on the identical 1,800-record sealed synthetic test split?"**

- **Target Output:** `risk_label` (`low`, `medium`, `high`) and mapped supportive intervention (`Monitor`, `Mentoring or study plan`, `Faculty follow-up`).
- **Primary Comparison Measures:**
  1. **Overall Accuracy** — Global multi-tier correctness across all $N=1,800$ held-out synthetic test records.
  2. **Macro F1 Score** — Unweighted harmonic mean balancing precision and recall across all three classes without majority-class bias.
  3. **High-Risk Recall (Critical Educational Triage Measure)** — Sensitivity in detecting synthetic student records modeled under acute academic jeopardy requiring human advisor review.
  4. **Per-Tier Confusion Matrices** — Full error attribution showing misclassification patterns between adjacent bands.

---

## 2. Baseline Specification
Before executing empirical comparisons, our team specified a transparent, reproducible, domain-grounded baseline matching institutional advising practices as documented in [`baseline-pseudocode.md`](baseline-pseudocode.md).

### A. Mathematical Formulation
$$\text{Score} = 0.30 \cdot \text{attendance\_pct} + 0.30 \cdot \text{assessment\_pct} + 0.20 \cdot \text{assignment\_pct} + 0.10 \cdot (\text{engagement\_score} \times 10) + 0.10 \cdot \text{prior\_performance\_pct}$$

### B. Threshold Decision Rules & Interventions
- **High Risk ($\text{Score} < 50.0$):** Acute academic struggle.  
  $\rightarrow$ *Intervention:* **Faculty follow-up** (immediate 1-on-1 academic consultation).
- **Medium Risk ($50.0 \le \text{Score} < 70.0$):** Emerging deficit or inconsistent engagement.  
  $\rightarrow$ *Intervention:* **Mentoring or study plan** (peer tutoring, TA study sessions).
- **Low Risk ($\text{Score} \ge 70.0$):** On track with healthy milestone progress.  
  $\rightarrow$ *Intervention:* **Monitor** (routine automated weekly progress check).

### C. Why this is a Fair Starting Point (Not a Final Product)
1. **Pedagogical Alignment:** Mirrors standard university weighted syllabi and learning management system (LMS) composite scorecards.
2. **Reproducibility & Independence:** Requires zero machine learning libraries, no GPU compute, and no training data. An academic adviser can calculate this on paper with zero black-box obscurity.
3. **Strict Separation:** It does not secretly borrow ANN hidden weights or tuned thresholds.
4. **Pedagogical Flaw (Why it is not a final product):** The linear additive formula assumes perfect **substitutability** between features. In education, a student who earns 9/10 LMS engagement clicks but receives 10% on exams cannot have their conceptual deficit "repaired" by clicks. The baseline is mathematically blind to feature interactions.

---

## 3. Experimental Setup & Identical Test Partitions
To guarantee an honest benchmark:
- **Identical Sealed Partition:** Both the Simple Baseline and the Step 3 ANN were evaluated on the exact held-out test split created in Step 2: [`../data/processed/test.csv`](../data/processed/test.csv) ($N=1,800$, 15% stratified split of 12,000 synthetic records).
- **Identical Features:** Both methods consumed the exact 5 academic features (`attendance_pct`, `assessment_pct`, `assignment_pct`, `engagement_score`, `prior_performance_pct`).
- **Explicit Alignment Validation:** Rather than assuming raw row ordering, `src/baseline.py` executes an explicit programmatic alignment check verifying 1-to-1 match on `record_id`, feature vectors, and ground-truth `risk_label` before computing metrics.
- **No Test Leakage:** Neither method was tuned, trained, or calibrated on the held-out test set.
- **Reference Standard:** Evaluated alongside a naive **Majority-Class Baseline** (always predicting `low`, which represents 44.50% of the population).

---

## 4. Quantitative Results & Comparison Table

The side-by-side evaluation across all 1,800 sealed synthetic test records yielded the following empirical metrics (persisted in [`../data/reports/step4_comparison_metrics.csv`](../data/reports/step4_comparison_metrics.csv)):

| Metric | Simple Baseline | Majority-Class | Step 3 ANN | Difference ($\Delta$) | Pct Change | Simulated Triage & Decision Impact |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Overall Accuracy** | **78.28%** | 44.50% | **84.11%** | **+5.83%** | +5.83 pp | **+105 net synthetic records** correctly categorized |
| **Macro F1 Score** | **0.7704** | 0.2053 | **0.8290** | **+0.0587** | +5.87 pp | Balanced quality across all three tiers |
| **Macro Precision** | **0.8100** | 0.1483 | **0.8287** | **+0.0188** | +1.88 pp | Reduced false alarms across tiers |
| **Macro Recall** | **0.7704** | 0.3333 | **0.8294** | **+0.0590** | +5.90 pp | Broader coverage of at-risk simulated profiles |
| **High-Risk Recall** | **63.71%** | 0.00% | **89.41%** | **+25.71%** | **+25.71 pp** | **Primary Gain:** ANN detects **136 additional modeled high-risk records** |
| **Medium-Risk Recall** | 81.28% | 0.00% | 71.28% | -10.00% | -10.00 pp | Trade-off: Baseline over-predicted `medium` (dumping ground) |
| **Low-Risk Recall** | 86.14% | 100.00% | 88.14% | +2.00% | +2.00 pp | Consistent, stable identification of on-track records |
| **High-Risk F1** | 0.7712 | 0.0000 | 0.8984 | +0.1272 | +12.72 pp | Superior balance of high-risk detection |
| **Medium-Risk F1** | 0.6626 | 0.0000 | 0.7023 | +0.0397 | +3.97 pp | Higher net F1 despite lower raw recall |
| **Low-Risk F1** | 0.8773 | 0.6159 | 0.8864 | +0.0091 | +0.91 pp | High reliability on safe records |

### Detailed Confusion Matrix Comparison ($N=1,800$)
Rebuilt directly from saved test comparison data ([`../data/reports/step4_test_comparison_full.csv`](../data/reports/step4_test_comparison_full.csv)):

```
SIMPLE BASELINE CONFUSION MATRIX:
               Predicted High   Predicted Low   Predicted Medium   Total
Actual High         337 (63.7%)         2 (0.4%)        190 (35.9%)     529
Actual Low            0 (0.0%)        690 (86.1%)       111 (13.9%)     801
Actual Medium         8 (1.7%)         80 (17.0%)       382 (81.3%)     470
Total               345               772               683           1,800

STEP 3 ANN CONFUSION MATRIX:
               Predicted High   Predicted Low   Predicted Medium   Total
Actual High         473 (89.4%)         2 (0.4%)         54 (10.2%)     529
Actual Low            0 (0.0%)        706 (88.1%)        95 (11.9%)     801
Actual Medium        51 (10.9%)        84 (17.9%)       335 (71.3%)     470
Total               524               792               484           1,800
```

---

## 5. Evidence Artifacts
All generated evaluation artifacts are checked into the repository:
1. **Interactive Notebook:** [`../notebooks/03_baseline.ipynb`](../notebooks/03_baseline.ipynb) — Executed notebook with complete outputs and side-by-side visualizations.
2. **Implementation Module:** [`../src/baseline.py`](../src/baseline.py) — Standalone Python script supporting programmatic alignment checks, full test evaluation, and single-record CLI inference.
3. **Automated Unit Tests:** [`../tests/test_baseline.py`](../tests/test_baseline.py) — 7 unit tests covering scoring formulas, cutoffs, 5 golden cases, test split metrics, and explicit dataset alignment.
4. **Comparative Metrics Table:** [`../data/reports/step4_comparison_metrics.csv`](../data/reports/step4_comparison_metrics.csv) — Formal metric comparison table with definitions and percentage changes.
5. **Full Prediction Records:** [`../data/reports/step4_test_comparison_full.csv`](../data/reports/step4_test_comparison_full.csv) — 1,800 rows containing ground truth, baseline scores, baseline predictions, and ANN predictions.
6. **Comparative Visualizations:**
   - [`../results/step4_model_comparison_bar.png`](../results/step4_model_comparison_bar.png) — Grouped bar chart illustrating Accuracy, Macro F1, and High-Risk Recall across Majority-Class, Baseline, and ANN.
   - [`../results/step4_baseline_confusion_matrix.png`](../results/step4_baseline_confusion_matrix.png) — Side-by-side heatmap comparison of Baseline vs. Step 3 ANN error matrices.
   - [`../results/step4_twin_read_cases.png`](../results/step4_twin_read_cases.png) — Twin Read comparison graphic showing representative verified test records where the baseline fails.

---

## 6. Analysis: Where the Advanced Method Helps and Fails

### A. Where the Advanced Method Helps (The $+25.71\%$ High-Risk Recall Surge)
In educational decision-support modeling, missing a modeled high-risk record (False Negative) is the primary triage concern. A simulated record mislabeled as "medium" or "low" fails to trigger urgent human faculty consultation in the simulated advising pipeline.
- The Simple Baseline correctly identifies only 337 out of 529 high-risk synthetic records (**63.71% recall**), leaving **192 high-risk records unflagged for urgent review** (190 misclassified as medium, 2 misclassified as low).
- The Step 3 ANN correctly detects 473 out of 529 high-risk synthetic records (**89.41% recall**), missing only 56 records.
- **The ANN detects 136 additional modeled high-risk records**, delivering a substantial $+25.71$ percentage point recall upgrade for academic triage.

#### Mathematical Explanation: Overcoming the "Compensatory Illusion"
Why does the baseline miss 190 high-risk synthetic records? The linear weighted sum allows strong values in one indicator to mask critical failure in another:
$$\text{Score} = 0.30 \times \text{att} + 0.30 \times \text{assess} + 0.20 \times \text{assign} + 0.10 \times (\text{eng} \times 10) + 0.10 \times \text{prior}$$

Consider verified test record `STU-01217` from [`../data/reports/step4_test_comparison_full.csv`](../data/reports/step4_test_comparison_full.csv):
- $\text{attendance} = 93.3\%$, $\text{assessment} = 10.1\%$, $\text{assignment} = 72.8\%$, $\text{engagement} = 7.2$, $\text{prior} = 30.4\%$.
- Baseline Score: $0.30(93.3) + 0.30(10.1) + 0.20(72.8) + 0.10(72.0) + 0.10(30.4) = 27.99 + 3.03 + 14.56 + 7.20 + 3.04 = \mathbf{55.82}$.
- Because $55.82 \ge 50.0$, the baseline assigns `medium` risk, recommending only informal study tips.
- Yet this record exhibits a catastrophic **10.1% in internal assessments**, representing severe exam failure!
- **ANN Non-Linear Rectification:** The ANN's multi-layer perceptron (with ReLU activations) learns non-linear bottleneck interactions: if assessment falls below 40%, high attendance and homework cannot compensate. The ANN predicts `high` risk with $94.21\%$ model probability ($P(\text{high}) = 0.9421$), correctly triggering **Faculty follow-up**.

### B. Where the Advanced Method Fails or Struggles
1. **Medium-Risk Ambiguity:** The ANN's recall on `medium` risk is $71.28\%$, compared to the baseline's $81.28\%$.  
   - However, the baseline's apparent advantage is an artifact of over-prediction: the baseline assigned 683 records into `medium` (achieving only $55.93\%$ precision; 382/683).
   - The ANN predicts `medium` more selectively (484 records), achieving higher precision ($69.21\%$; 335/484) and higher F1 ($0.7023$ vs. $0.6626$).
2. **False Alarms in Borderline Cases:** The ANN misclassified 51 medium-risk synthetic records as high risk (compared to 8 for the baseline). This represents an increased review volume for academic advisors (reviewing a student profile that could have been handled through peer mentoring). In decision-support systems, this trade-off is often acceptable to avoid unflagged academic collapse, provided advisors retain final decision discretion.

### C. Twin Read Case Studies (Verified Test Records)
Below are representative cases from [`../data/reports/step4_test_comparison_full.csv`](../data/reports/step4_test_comparison_full.csv) cross-referenced with exact model softmax probabilities from [`../data/reports/step3_test_predictions.csv`](../data/reports/step3_test_predictions.csv):

| Record ID | Att% | Assess% | Assign% | Engage | Prior% | Ground Truth | Baseline Pred (Score) | ANN Pred (Probability) | Pedagogical & Decision Rationale |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **STU-01217** | 93.3 | **10.1** | 72.8 | 7.2 | 30.4 | **high** | `medium` (55.82) | **high** ($P=0.942$) | **Compensatory Illusion:** Baseline inflated by attendance and homework; ANN detects critical assessment failure (10.1%). |
| **STU-09402** | **27.5** | 59.7 | 60.4 | 8.4 | 63.9 | **high** | `medium` (53.03) | **high** ($P=0.993$) | **Attendance Collapse:** Severe absenteeism (<30%) triggers de-registration risk; linear baseline averages away risk with LMS clicks. |
| **STU-07276** | 50.7 | **43.1** | 57.3 | 5.4 | 54.9 | **high** | `medium` (50.49) | **high** ($P=0.794$) | **Multiple Deficits:** Borderline attendance and failing exams detected by ANN hidden-layer interaction; baseline barely clears 50.0 cutoff. |
| **STU-05643** | 51.9 | 69.2 | **42.2** | 5.1 | 59.4 | **medium** | **medium** (55.81) | `high` ($P=0.555$) | **ANN False Alarm:** Sub-50% assignment score caused ANN to over-predict high; baseline linear weighting correctly retained medium. |
| **STU-11733** | 83.9 | **58.2** | 63.3 | 5.9 | 76.9 | **low** | `medium` (68.88) | `medium` ($P=0.625$) | **Borderline Caution:** Sub-60% assessment marks led both models to recommend peer study plan rather than passive monitoring. |

---

## 7. Limitations & Operational Trade-Offs

1. **Synthetic Data Nature & Absence of Real-World Clinical Efficacy Claims:**
   In accordance with Step 2 documentation, this dataset comprises 12,000 synthetic records generated from calibrated behavioral rules inspired by OULAD. While effective for isolating algorithm mechanics, mathematical properties, and feature interactions, **these empirical metrics do not establish proven real-world educational intervention benefit or human student retention improvement**. True intervention efficacy requires prospective, human-supervised observational trials with real student cohorts, institutional ethics approval, and advisor feedback.
2. **Computational & Deployment Overhead:**
   - *Baseline:* Evaluates in $<1$ microsecond, requires 0 dependencies, and can run in SQL queries or Excel.
   - *Step 3 ANN:* Requires scikit-learn, pipeline serialization (`models/ann_student_performance.pkl`), and feature scaling.
3. **Interpretability Trade-off:** The baseline's decision can be explained to an advisee in 10 seconds. The ANN requires post-hoc explanation techniques (e.g. SHAP values, feature importance) to explain why a specific interaction triggered an alert.
4. **Mandatory Human-in-the-Loop:** Neither the baseline nor the ANN makes final automated academic decisions. All classifications are decision-support indicators requiring mentor review before intervention dispatch.

---

## 8. Reproducibility & Run Instructions

### A. Run Full Baseline Evaluation & Generate Artifacts
```powershell
python src/baseline.py --evaluate
```

### B. Run Single-Record Interactive Prediction
```powershell
python src/baseline.py --attendance 45.0 --assessment 38.0 --assignment 50.0 --engagement 3.5 --prior 52.0
```

### C. Run Automated Test Suite
```powershell
python -m pytest tests/test_baseline.py -v
```

### D. Execute Step 4 Notebook
Open and run all cells in [`../notebooks/03_baseline.ipynb`](../notebooks/03_baseline.ipynb).

---

## 9. Team Member Responsibilities

| Team Member | Official Role | Milestone 1 · Step 4 Specific Contributions |
|---|---|---|
| **Abhishek (Lead)** | System Architect & ML Lead | Formulated comparison research question; designed deterministic weighting equation and cutoffs; directed empirical benchmark evaluation; authored side-by-side comparison CSV pipeline; reviewed git synchronization. |
| **Saumya Rawat** | Data & Feature Engineer | Implemented `src/baseline.py` inference and CLI functions; verified test set schema and programmatic alignment checks; performed confusion matrix error calculations; investigated medium-risk transition dynamics. |
| **Aman Sharma** | QA & Evaluation Engineer | Built `tests/test_baseline.py` test suite covering formulas, cutoffs, golden cases, and alignment validation; developed matplotlib comparative plotting routines; generated `results/step4_*.png` visual artifacts. |
| **Riya Gupta** | Documentation & Ethics Lead | Authored `docs/STEP-4.md`; conducted Twin Read case study error analysis; documented compensatory illusion failure modes; articulated human-in-the-loop ethical guidelines and operational limitations on synthetic data. |

