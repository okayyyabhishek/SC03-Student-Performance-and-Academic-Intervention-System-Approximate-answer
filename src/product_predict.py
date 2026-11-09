"""SC03 Product V1 inference. Adapted for Student Performance and Academic Intervention System."""
from pathlib import Path
import json
import math
import pickle
import pandas as pd

FEATURES = [
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "engagement_score",
    "prior_performance_pct",
]

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ann_student_performance.pkl"
if not MODEL_PATH.is_file() and (ROOT / "models" / "ann_model.pkl").is_file():
    MODEL_PATH = ROOT / "models" / "ann_model.pkl"
META_PATH = ROOT / "models" / "ann_model_metadata.json"

LABELS_INTERVENTIONS = {
    "low": {
        "label": "Predicted low risk class",
        "intervention": "Routine monitoring & check-in",
    },
    "medium": {
        "label": "Predicted medium risk class",
        "intervention": "Mentoring or study plan support",
    },
    "high": {
        "label": "Predicted high risk class",
        "intervention": "Immediate faculty consultation & follow-up",
    },
}

NOTICE_TEXT = (
    "Educational prediction only. This decision-support output does not replace "
    "human mentor evaluation or constitute official disciplinary action."
)


def validate_inputs(values):
    """Reject blank, nonnumeric, nonfinite and out-of-range academic indicator values."""
    clean = {}
    for name in FEATURES:
        raw = values.get(name)
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"{name}: enter a value.")
        try:
            number = float(raw)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name}: enter a number.") from error
        if not math.isfinite(number):
            raise ValueError(f"{name}: use a finite number.")
        if number < 0:
            raise ValueError(f"{name}: negative values are invalid.")
        if name == "engagement_score" and not 0.0 <= number <= 10.0:
            raise ValueError("engagement_score: use a value from 0 to 10.")
        if name != "engagement_score" and not 0.0 <= number <= 100.0:
            raise ValueError(f"{name}: use a value from 0 to 100.")
        clean[name] = number
    return clean


def load_model():
    """Load only the model you created and trust in this repository."""
    if not MODEL_PATH.is_file() or not META_PATH.is_file():
        raise FileNotFoundError(
            "Saved model or metadata is missing. Restore your Step 3 files."
        )
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    if metadata.get("features") != FEATURES:
        raise ValueError("Model metadata feature order does not match the app.")
    with MODEL_PATH.open("rb") as source:
        model = pickle.load(source)
    if list(getattr(model, "feature_names_in_", [])) != FEATURES:
        raise ValueError("Saved model input columns do not match the app.")
    return model


def predict_one(model, values):
    clean = validate_inputs(values)
    frame = pd.DataFrame([[clean[name] for name in FEATURES]], columns=FEATURES)
    raw_pred = model.predict(frame)[0]
    prediction = str(raw_pred).lower()
    if prediction not in LABELS_INTERVENTIONS:
        raise ValueError("Unexpected model class. Check the saved model.")

    details = LABELS_INTERVENTIONS[prediction]
    return {
        "predicted_class": prediction,
        "label": details["label"],
        "intervention": details["intervention"],
        "notice": NOTICE_TEXT,
        **clean,
    }
