# SC03 Step 3 — Artificial Neural Network (ANN) Model

## 1. Goal
Train a feed-forward Artificial Neural Network (ANN) using backpropagation to learn the relationship between five continuous student academic indicators and the academic intervention risk label (`low`, `medium`, `high`).

## 2. Inputs and Output
### Inputs (5 indicators)
1. `attendance_pct` — Lecture and lab session attendance percentage (0.0 to 100.0%)
2. `assessment_pct` — Continuous internal assessment and quiz score percentage (0.0 to 100.0%)
3. `assignment_pct` — Homework problem sets and lab assignment percentage (0.0 to 100.0%)
4. `engagement_score` — Standardized LMS portal activity and telemetry index (0.0 to 10.0)
5. `prior_performance_pct` — Cumulative prerequisite coursework percentage (0.0 to 100.0%)

### Output
- `risk_label` — Academic intervention category:
  - `low` (On track / Routine monitoring)
  - `medium` (Emerging deficit / Mentoring and study plan support)
  - `high` (Severe struggle / Urgent faculty follow-up)

### Architecture
- **Structure:** `5 inputs -> 16 hidden neurons -> 8 hidden neurons -> 3 output neurons (Softmax)`
- **Preprocessing:** `StandardScaler` to prevent larger-magnitude features from dominating gradient updates.
- **Activation:** Rectified Linear Unit (`relu`) for hidden representations.
- **Optimizer:** `adam` with initial learning rate $\eta = 0.001$.
- **Regularization:** Early stopping enabled with 15% validation fraction, $n=25$ iterations patience, and fixed random seed 42.

## 3. Data Strategy and Performance Results
### Data Splits
Model development strictly adheres to the clean Step 2 splits:
- `data/processed/train.csv` (8,400 rows, 70%)
- `data/processed/validation.csv` (1,800 rows, 15%)
- `data/processed/test.csv` (1,800 rows, 15%)

Testing scenarios (`data/generated/`) are strictly isolated for synthetic stress testing and are never used in training or evaluation splits.

### Training Progress
- **Iterations completed:** 79 iterations
- **Final training cross-entropy loss:** 0.36699
- Loss curve demonstrates smooth, monotonically decreasing error convergence without divergent oscillation.

### Quantitative Metrics

| Split | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| **Validation** | **84.94%** | **0.8351** | **0.8354** | **0.8352** |
| **Test (Sealed)** | **84.11%** | **0.8287** | **0.8294** | **0.8290** |

Detailed class reports confirm balanced detection across all three risk categories:
- High risk: Precision = 0.90, Recall = 0.89, F1 = 0.90 (Test split)
- Low risk: Precision = 0.89, Recall = 0.88, F1 = 0.89 (Test split)
- Medium risk: Precision = 0.69, Recall = 0.71, F1 = 0.70 (Test split)

## 4. Evidence Artifacts
All required Step 3 evidence files are saved and committed:
1. `notebooks/02_ann_model.ipynb` — Executed interactive notebook with visible outputs, metric tables, and loss curve.
2. `src/train_ann.py` — Repeatable standalone Python script for non-notebook execution.
3. `results/step3_ann_loss_curve.png` — Training loss curve figure.
4. `results/step3_validation_confusion_matrix.png` — Labelled confusion matrix for validation split.
5. `results/step3_test_confusion_matrix.png` — Labelled confusion matrix for unseen test split.
6. `data/reports/step3_ann_metrics.csv` — Tabular metrics recorded across splits.
7. `data/reports/step3_validation_predictions.csv` — Validation predictions with class probabilities.
8. `data/reports/step3_test_predictions.csv` — Test predictions with class probabilities.
9. `models/ann_student_performance.pkl` — Serialized model pipeline artifact.
10. `models/ann_model_metadata.json` — Comprehensive model architecture and audit metadata.

## 5. Limitations & Ethical Notice
- **Decision-Support Only:** This educational machine learning model is intended as an early-warning guidance tool to help mentors identify students who may need additional academic support.
- **Human Oversight:** Predictions do **not** replace mentor evaluations, faculty counseling, or official academic committee decisions. A human mentor must always review context before initiating intervention actions.
