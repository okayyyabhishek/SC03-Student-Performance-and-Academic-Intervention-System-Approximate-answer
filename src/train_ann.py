"""SC03 Step 3 - train an ANN student-performance risk classifier.

This script trains a feed-forward Artificial Neural Network (MLPClassifier)
with feature standardization to predict academic risk levels (low, medium, high).
Generates reproducible model artifacts, loss curves, confusion matrices, and metrics.
"""

from pathlib import Path
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for folder in [REPORT_DIR, MODEL_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "engagement_score",
    "prior_performance_pct",
]
TARGET = "risk_label"
RANDOM_SEED = 42


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Complete Step 2 first.")
    frame = pd.read_csv(path)
    required = FEATURES + [TARGET]
    if frame[required].isna().sum().sum() != 0:
        raise ValueError(f"{name}.csv still has missing values.")
    return frame


def metric_row(y_true, y_pred, split_name: str) -> dict:
    return {
        "split": split_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
    }


def show_confusion_matrix(y_true, y_pred, split_name: str) -> None:
    labels = ["low", "medium", "high"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=f"ANN Confusion Matrix - {split_name.title()} Split",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=[0, 1, 2],
        yticks=[0, 1, 2],
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
    )
    for row in range(3):
        for column in range(3):
            val = matrix[row, column]
            color = "white" if val > matrix.max() / 2 else "black"
            axis.text(column, row, str(val), ha="center", va="center", color=color, fontweight="bold")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / f"step3_{split_name}_confusion_matrix.png", dpi=180)
    plt.close(figure)


def main() -> None:
    print(f"Project folder: {PROJECT_ROOT}")
    print("SC03 STEP 3: ANN MODEL TRAINING AND EVALUATION")
    print("-" * 50)

    train_data = load_split("train")
    validation_data = load_split("validation")
    test_data = load_split("test")

    print(f"Train: {len(train_data)} rows, {len(train_data.columns)} columns")
    print(train_data[TARGET].value_counts(normalize=True).round(3))
    print(f"\nValidation: {len(validation_data)} rows, {len(validation_data.columns)} columns")
    print(validation_data[TARGET].value_counts(normalize=True).round(3))
    print(f"\nTest: {len(test_data)} rows, {len(test_data.columns)} columns")
    print(test_data[TARGET].value_counts(normalize=True).round(3))

    X_train, y_train = train_data[FEATURES], train_data[TARGET]
    X_validation, y_validation = validation_data[FEATURES], validation_data[TARGET]
    X_test, y_test = test_data[FEATURES], test_data[TARGET]

    print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Build ANN pipeline (StandardScaler + MLPClassifier)
    ann_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "ann",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    print("ANN architecture: 5 inputs -> 16 -> 8 -> 3 output classes")
    print("Training ANN...")
    ann_model.fit(X_train, y_train)

    trained_ann = ann_model.named_steps["ann"]
    print(f"Training iterations completed: {trained_ann.n_iter_}")
    print(f"Final training loss: {round(trained_ann.loss_, 5)}")

    # Validation metrics
    validation_predictions = ann_model.predict(X_validation)
    validation_metrics = metric_row(y_validation, validation_predictions, "validation")

    # Plot loss curve
    losses = trained_ann.loss_curve_
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", markersize=2, color="#1f77b4")
    plt.title("ANN Training Loss Curve")
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "step3_ann_loss_curve.png", dpi=180)
    plt.close()

    # Test evaluation
    test_predictions = ann_model.predict(X_test)
    test_metrics = metric_row(y_test, test_predictions, "test")

    metrics_df = pd.DataFrame([validation_metrics, test_metrics])
    print("\nMetrics summary:")
    print(metrics_df.to_string(index=False))

    print("\nValidation classification report:")
    print(classification_report(y_validation, validation_predictions, zero_division=0))

    print("Test classification report:")
    print(classification_report(y_test, test_predictions, zero_division=0))

    # Confusion matrices
    show_confusion_matrix(y_validation, validation_predictions, "validation")
    show_confusion_matrix(y_test, test_predictions, "test")

    # Save evidence files
    metrics_df.to_csv(REPORT_DIR / "step3_ann_metrics.csv", index=False)

    for split_name, frame, predictions in [
        ("validation", validation_data, validation_predictions),
        ("test", test_data, test_predictions),
    ]:
        output = frame.copy()
        output["ann_prediction"] = predictions
        probs = ann_model.predict_proba(frame[FEATURES])
        for idx, cls_name in enumerate(ann_model.classes_):
            output[f"prob_{cls_name}"] = probs[:, idx].round(4)
        output.to_csv(REPORT_DIR / f"step3_{split_name}_predictions.csv", index=False)

    with (MODEL_DIR / "ann_student_performance.pkl").open("wb") as model_file:
        pickle.dump(ann_model, model_file)

    with (MODEL_DIR / "ann_model.pkl").open("wb") as model_file:
        pickle.dump(ann_model, model_file)

    metadata = {
        "technique": "feed-forward ANN / backpropagation (scikit-learn MLPClassifier)",
        "architecture": "5 inputs -> 16 hidden neurons -> 8 hidden neurons -> 3 output neurons (Softmax)",
        "features": FEATURES,
        "target": TARGET,
        "classes": list(ann_model.classes_),
        "random_seed": RANDOM_SEED,
        "training_iterations": int(trained_ann.n_iter_),
        "final_loss": round(float(trained_ann.loss_), 5),
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_macro_f1": float(validation_metrics["f1_score"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["f1_score"]),
    }

    (MODEL_DIR / "ann_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nSaved model and Step 3 evidence files.")
    print("Files created:")
    print(f"- {RESULTS_DIR / 'step3_ann_loss_curve.png'}")
    print(f"- {RESULTS_DIR / 'step3_validation_confusion_matrix.png'}")
    print(f"- {RESULTS_DIR / 'step3_test_confusion_matrix.png'}")
    print(f"- {REPORT_DIR / 'step3_ann_metrics.csv'}")
    print(f"- {REPORT_DIR / 'step3_validation_predictions.csv'}")
    print(f"- {REPORT_DIR / 'step3_test_predictions.csv'}")
    print(f"- {MODEL_DIR / 'ann_student_performance.pkl'}")
    print(f"- {MODEL_DIR / 'ann_model_metadata.json'}")


if __name__ == "__main__":
    main()
