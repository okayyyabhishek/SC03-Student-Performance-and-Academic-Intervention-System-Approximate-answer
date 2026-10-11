# SC03 — Student Performance & Academic Intervention System

<p align="center">
  <strong>Explainable academic-risk decision support for earlier, more supportive student interventions.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Machine Learning" src="https://img.shields.io/badge/ML-ANN%20%7C%20Backpropagation-8A2BE2">
  <img alt="Data" src="https://img.shields.io/badge/Dataset-12%2C000%20records-16803C">
  <img alt="Status" src="https://img.shields.io/badge/Status-Step%203%20ANN%20available-blue">
</p>

## Overview

The **Student Performance and Academic Intervention System (SC03)** is a soft-computing capstone project designed to estimate an academic risk category from anonymous academic indicators and suggest a supportive next step. It aims to help faculty mentors identify potential learning difficulties earlier—not to rank, punish, or automatically make decisions about students.

**Problem statement:** Given anonymous academic indicators, estimate a risk band and suggest a supportive intervention.

### Risk categories and intended support

| Risk category | Intended interpretation | Possible support |
|---|---|---|
| `low` | Student appears on track | Routine monitoring |
| `medium` | Potential emerging academic difficulty | Mentoring or a study plan |
| `high` | Signals of significant difficulty | Faculty follow-up |

These categories are educational decision-support outputs, not official academic determinations. A faculty member must review the situation and make the final decision.

## Current project scope

The repository tracks work by stages using Git branches:

- **Step 1 — Project preparation:** problem definition, data contract, starter sample, baseline rules, and data validation.
- **Step 2 — Data pipeline:** processed dataset, train/validation/test partitions, data-quality evidence, and manifest.
- **Step 3 — ANN model:** feed-forward neural classifier, recorded evaluation metrics, prediction files, model artifacts, loss curve, and confusion matrices.

The repository currently contains the Step 3 work on the `step-3-ann-model` branch. The default branch, `main`, is used as the repository landing page; select the appropriate branch to inspect its specific milestone.

## Branch overview

| Branch | Purpose | Key contents |
|---|---|---|
| [`main`](https://github.com/okayyyabhishek/SC03-Student-Performance-and-Academic-Intervention-System-Approximate-answer/tree/main) | Repository landing page and project overview | README, initial project contract, starter sample and validator |
| [`step-2-data-pipeline`](https://github.com/okayyyabhishek/SC03-Student-Performance-and-Academic-Intervention-System-Approximate-answer/tree/step-2-data-pipeline) | Step 2 milestone | Dataset preparation, processed splits, manifest, data-quality reports, and Step 2 evidence |
| [`step-3-ann-model`](https://github.com/okayyyabhishek/SC03-Student-Performance-and-Academic-Intervention-System-Approximate-answer/tree/step-3-ann-model) | Step 3 ANN milestone | Executed notebook, ANN training script, model files, evaluation CSVs, plots, metadata, and `docs/STEP-3.md` |

### Key files on Step 3

- [Step 2 documentation](docs/STEP-2.md) — dataset pipeline and split strategy.
- [Step 3 documentation](docs/STEP-3.md) — ANN design, recorded results, evidence artifacts, and limitations.
- [ANN notebook](notebooks/02_ann_model.ipynb) — training and evaluation workflow with saved outputs.
- [Reusable training script](src/train_ann.py) — standalone training and artifact generation.
- [Requirements](requirements.txt) — Python dependency ranges.

## Model inputs

The ANN uses five academic indicators:

| Feature | Meaning | Allowed scale described by the project |
|---|---|---|
| `attendance_pct` | Attendance percentage | 0–100 |
| `assessment_pct` | Assessment and quiz performance | 0–100 |
| `assignment_pct` | Assignment performance | 0–100 |
| `engagement_score` | Learning-platform engagement indicator | 0–10 |
| `prior_performance_pct` | Prior academic performance | 0–100 |

**Prediction target:** `risk_label`, with the classes `low`, `medium`, and `high`.

The dataset documentation describes a 12,000-record benchmark, split into 8,400 training rows, 1,800 validation rows, and 1,800 test rows. The generated testing scenarios are kept separate from these model-development splits. Review the Step 2 data documentation for the target-generation assumptions and synthetic/proxy limitations.

## ANN approach

The Step 3 model is a scikit-learn `MLPClassifier` wrapped in a pipeline:

- **Architecture:** 5 input features → 16 hidden neurons → 8 hidden neurons → 3 output classes.
- **Preprocessing:** `StandardScaler`.
- **Hidden-layer activation:** ReLU.
- **Optimizer:** Adam.
- **Early stopping:** enabled.
- **Reproducibility:** random seed 42.

### Recorded evaluation results

The figures below are the values recorded in `data/reports/step3_ann_metrics.csv` on the Step 3 branch; they are not a guarantee of performance on real student data.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 84.94% | 84.11% |
| Macro precision | 83.51% | 82.87% |
| Macro recall | 83.54% | 82.94% |
| Macro F1-score | 83.52% | 82.90% |

Additional evidence includes the training loss curve and separately labelled validation and test confusion matrices.

## Repository structure

The following is a guide to the main project paths. Some milestone-specific files are available only on the corresponding branch.

```text
.
├── README.md
├── STEP-1.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   ├── test.csv
│   │   └── manifest.json
│   ├── generated/
│   └── reports/
├── docs/
│   ├── STEP-2.md
│   └── STEP-3.md
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   └── 02_ann_model.ipynb
├── src/
│   ├── validate_data.py
│   ├── prepare_data.py
│   ├── generate_test_scenarios.py
│   ├── pipeline/
│   └── train_ann.py
├── models/
│   ├── ann_student_performance.pkl
│   ├── ann_model.pkl
│   └── ann_model_metadata.json
└── results/
    ├── step2/
    ├── step3_ann_loss_curve.png
    ├── step3_validation_confusion_matrix.png
    └── step3_test_confusion_matrix.png
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/okayyyabhishek/SC03-Student-Performance-and-Academic-Intervention-System-Approximate-answer.git
cd SC03-Student-Performance-and-Academic-Intervention-System-Approximate-answer
```

### 2. Select the milestone branch

For the most complete published modeling workflow:

```bash
git checkout step-3-ann-model
```

### 3. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Inspect the model notebook

Open `notebooks/02_ann_model.ipynb` in Jupyter or VS Code and select the project virtual environment as the kernel.

### 6. Run the training script

Run from the repository root on the Step 3 branch:

```bash
python src/train_ann.py
```

The script expects these files to exist first:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

It writes evaluation metrics and predictions to `data/reports/`, saved model artifacts to `models/`, and plot images to `results/`. Running it may overwrite these generated artifacts with new outputs.

If the processed splits are missing, review [Step 2 documentation](docs/STEP-2.md) and complete the data pipeline before training.

## Data, privacy, and responsible use

- Use anonymous or appropriately de-identified records; never commit real student names, email addresses, phone numbers, or other direct identifiers.
- Treat starter and generated scenarios according to their documented status. Synthetic/generated examples are not real labelled student observations.
- Keep training, validation, and test data roles separate. Do not repeatedly tune a model against the final test split.
- Inspect class-wise metrics and confusion matrices rather than relying on accuracy alone.
- Review the dataset source, license, assumptions, target-generation method, and limitations in `data/README.md` and `docs/STEP-2.md`.
- A prediction is not a diagnosis or an official academic decision. A qualified human must consider context before any intervention.
- Do not use the system to automate punishment, disciplinary decisions, or public student rankings.

## Roadmaps

The repository's project contract describes later work such as baseline comparison, additional neural methods (including Perceptron and ADALINE), error analysis, noise robustness, fairness evaluation, and application integation. These should be treated as roadmap items unless the corresponding implementation and evidence are present on a branch.

## Contributing and milestone workflow

1. Check out the branch for the milestone you are working on.
2. Read the relevant `docs/STEP-*.md` contract before changing the pipeline or model.
3. Keep training logic reproducible and record meaningful changes in Git.
4. Save metrics, plots, metadata, and documentation with the code that produces them.
5. Review staged files before committing; do not commit credentials, personal records, local environment folders, or unrelated generated files.

## License and dataset terms

No project license or third-party dataset license should be assumed from this README alone. Check the repository and dataset source for applicable license terms before reuse or redistribution.

---

**Educational use only:** SC03 is a student project for academic support research and demonstration. Its outputs must not replace faculty judgment or official academic processes.
