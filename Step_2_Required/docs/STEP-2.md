# Step 2: 10,000+ Record Dataset Pipeline & Manifest — SC03

**Product:** Student Performance and Academic Intervention System  
**Stage:** Step 2 Dataset Ingestion, Non-Circular Ground Truth, & Manifest  
**Status:** COMPLETE (All gates passed, 18 automated tests passing)  

---

## 1. Overview & Objectives

In accordance with Guide Step 2, [docs/IMPLEMENTATION_PLAN.md](file:///c:/Users/Abhishek/Capstone%20Project/docs/IMPLEMENTATION-PLAN.md), and technical requirements TR-010 through TR-015:
- We scaled from the 20-row starter sample to an official **12,000-record** benchmark dataset.
- We constructed an offline, fully reproducible generation pipeline (`src/pipeline/build_dataset.py`) seeded at `42`.
- We enforced an **independent non-circular ground-truth target** via latent aptitude modeling, non-linear interaction penalties, and external life noise (TR-012).
- We partitioned the data into an exact **stratified 70 / 15 / 15 split** (`train.csv`, `val.csv`, `test.csv`) (TR-013).
- We decoupled demographic attributes into an isolated file (`data/processed/protected_attributes.csv`) strictly excluded from training features (TR-014).
- We produced a cryptographic manifest (`data/processed/manifest.json`) recording parameters, proxy flags, class counts, and SHA-256 checksums (TR-011).

---

## 2. Mathematical Generation Model & Non-Circularity (TR-012)

### 2.1 The Non-Circularity Mandate (Non-Negotiable Rule 4)
If the ground truth `risk_label` were derived from the linear baseline formula ($30\%$ attendance $+ 30\%$ assessment $+ 20\%$ assignment $+ 10\%$ engagement $+ 10\%$ prior performance), the baseline classifier would achieve $100\%$ accuracy by definition. This would render any Soft Computing / neural network comparison scientifically invalid.

### 2.2 Latent Factor Structure
Each student record is driven by two unobserved latent traits:
- $A \sim \mathcal{N}(0, 1)$: Academic Aptitude & Prerequisite Knowledge
- $W \sim \mathcal{N}(0, 1)$: Academic Work Ethic, Diligence, & Routine Habit

### 2.3 Observable Indicator Equations
Observable indicators are generated with calibrated measurement noise:
- **`attendance_pct`**: $\text{clip}(72.0 + 15.0 \cdot W + 5.0 \cdot A + \epsilon_1, 0, 100)$, $\epsilon_1 \sim \mathcal{N}(0, 12)$
- **`assessment_pct`**: $\text{clip}(64.0 + 18.0 \cdot A + 8.0 \cdot W + \epsilon_2, 0, 100)$, $\epsilon_2 \sim \mathcal{N}(0, 14)$
- **`assignment_pct`**: $\text{clip}(66.0 + 12.0 \cdot A + 15.0 \cdot W + \epsilon_3, 0, 100)$, $\epsilon_3 \sim \mathcal{N}(0, 13)$
- **`engagement_score`**: $\text{clip}(6.2 + 1.8 \cdot W + 0.8 \cdot A + \epsilon_4, 0, 10)$, $\epsilon_4 \sim \mathcal{N}(0, 1.5)$
- **`prior_performance_pct`**: $\text{clip}(65.0 + 16.0 \cdot A + 5.0 \cdot W + \epsilon_5, 0, 100)$, $\epsilon_5 \sim \mathcal{N}(0, 13)$

### 2.4 Non-Linear Interaction Penalties
Real educational failure is asymmetric and non-linear. Four realistic domain interaction penalties modify the latent risk:
1. **Compensatory Illusion (Cramming):** Students with high LMS engagement ($\ge 7.5$) who fail exam fundamentals ($\text{assessment} < 45\%$) exhibit unproductive struggle. Penalty: $-14.0$ points to effective outcome.
2. **Attendance Disengagement Spiral:** Attendance drops below $50\%$ compound unrecorded quiz failures and missed learning scaffolding. Penalty: $-16.0$ points to effective outcome.
3. **Prior Strength Breakdown:** High prior performance ($\ge 80\%$) fails to protect a student whose attendance collapses below $55\%$. Penalty: $-12.0$ points to effective outcome.
4. **Dual Coursework Collapse:** Severe deficits in both assessment and assignments ($< 48\%$) represent active course failure regardless of attendance. Penalty: $-15.0$ points to effective outcome.
5. **External Life Disruptions:** Gaussian noise $\eta \sim \mathcal{N}(0, 6.0)$ simulates unpredictable personal, health, and economic hardships.

### 2.5 Ground Truth Thresholding
$$\text{Effective Outcome} = (0.35 \cdot \text{Assess} + 0.25 \cdot \text{Assign} + 0.20 \cdot \text{Att} + 0.12 \cdot \text{Eng}_{\%} + 0.08 \cdot \text{Prior}) - \text{Penalties} + \eta$$

- **Effective Outcome $< 52.0$** $\implies$ `high` risk
- **$52.0 \le$ Effective Outcome $< 68.0$** $\implies$ `medium` risk
- **Effective Outcome $\ge 68.0$** $\implies$ `low` risk

---

## 3. Stratified Partitioning & Row Counts (TR-013)

The dataset was partitioned using `sklearn.model_selection.train_test_split` with `seed=42`, stratified by `risk_label`:

| Partition | Purpose | Rows | Proportion | Class Proportions (Low / Medium / High) |
|---|---|---|---|---|
| `dataset.csv` | Full benchmark corpus | 12,000 | 100.0% | 44.5% / 26.1% / 29.4% |
| `train.csv` | Model fitting & gradient updates | 8,400 | 70.0% | 44.5% / 26.1% / 29.4% |
| `val.csv` | Hyperparameter & early stopping | 1,800 | 15.0% | 44.5% / 26.1% / 29.4% |
| `test.csv` | Sealed evaluation (touched once) | 1,800 | 15.0% | 44.5% / 26.1% / 29.4% |

*Rule:* The `test.csv` set remains completely unseen during training and validation tuning.

---

## 4. Protected Attributes Isolation Policy (TR-014, NFR-001)

- Demographic attributes are stored strictly in `data/processed/protected_attributes.csv`.
- Columns: `record_id`, `gender`, `age_band`, `disability`, `socio_economic_decile`.
- **Architectural Isolation:** These fields are **never** present in `train.csv`, `val.csv`, or `test.csv`, and are never input to the neural classifiers.
- **Fairness Audit:** In Step 7, model predictions will be linked via `record_id` in an isolated admin audit script to calculate demographic parity and equal opportunity ratios without compromising student privacy.

---

## 5. Baseline Leakage & Non-Circularity Proof

The deterministic linear baseline was evaluated against the unseen 1,800-row `test.csv` split:
- **Baseline Accuracy:** **78.28%**
- **Baseline Macro F1:** **0.7704**
- **Circular Evaluation Check:** **PASS** (strictly between 65% and 85%)

### Confusion Matrix on Test Split ($N = 1,800$)

| Actual \ Predicted | Predicted Low | Predicted Medium | Predicted High | Total Actual | Class Recall |
|---|---|---|---|---|---|
| **Actual Low** | 690 | 111 | 0 | 801 | **86.1%** |
| **Actual Medium** | 80 | 382 | 8 | 470 | **81.3%** |
| **Actual High** | 2 | 190 | 337 | 529 | **63.7%** |

### Critical Viva Takeaway
> **Viva Question:** *"Why does the baseline achieve only 63.7% recall on high-risk students?"*  
> **Explanation:** The baseline incorrectly classifies 190 truly high-risk students as `medium` risk. These are students with good attendance or high prior performance who experienced severe coursework collapse or unproductive portal cramming. Because the baseline uses static linear weights, it averages these numbers away. In Step 4, our ADALINE neural network will learn decision boundaries that capture these non-linear patterns.

---

## 6. Cryptographic Manifest Checksums (TR-011)

All files in `data/processed/` are cryptographically verified:
- `dataset.csv`: `424f7eaa2efe4640ae004669e36680ab49233df869b06a9ff56c5b8cbdbda86e`
- `train.csv`: `7de86756ffbcc8abe1a4d20f8071a9852ae8a1145ca9457fc411eeddd0741b2e`
- `val.csv`: `a2fc12efd09e793a4f26135d2bdc193a44d20b4520ac3477e0e5ee46ba631885`
- `test.csv`: `292f333ed4fc5fedbbf06e643d783703f327d22b1c5a22543bc4321116b55dfc`
- `protected_attributes.csv`: `9db174ae5b9138c2acf4bb7e1a276ac8927a71478eeec24ff606c59f8be2170a`

---

## 7. Verification Commands

To reproduce and verify Step 2 from a clean checkout:
```powershell
# 1. Regenerate dataset and manifest
python src/pipeline/build_dataset.py --seed 42 --num-records 12000

# 2. Run data validation on all partitions
python src/validate_data.py data/processed/train.csv
python src/validate_data.py data/processed/val.csv
python src/validate_data.py data/processed/test.csv

# 3. Generate profiling report
python src/pipeline/profile_dataset.py

# 4. Run automated test suite
python -m pytest -v tests/test_dataset.py
```
