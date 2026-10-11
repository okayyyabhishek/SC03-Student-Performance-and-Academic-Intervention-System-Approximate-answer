"""
SC03 Student Performance and Academic Intervention System
Step 2 Reusable Data Preparation Pipeline
Meets Student Master Guide Step 2 Requirements (Actions 23, 43-47)
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "student_performance_raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "engagement_score",
    "prior_performance_pct",
]

REQUIRED_COLUMNS = [
    "record_id",
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "engagement_score",
    "prior_performance_pct",
    "risk_label",
]

NUMERIC_COLUMNS = [
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "engagement_score",
    "prior_performance_pct",
]

SEED = 42
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing feature values with column median and remove duplicate rows."""
    cleaned = dataframe.copy()
    cleaned[FEATURE_COLUMNS] = cleaned[FEATURE_COLUMNS].fillna(
        cleaned[FEATURE_COLUMNS].median()
    )
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def validate_data(dataframe: pd.DataFrame) -> None:
    """Validate cleaned academic data against the contract before model splitting."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_count = int(dataframe[REQUIRED_COLUMNS].isna().sum().sum())
    if missing_count > 0:
        raise ValueError(f"Missing values still present after cleaning: {missing_count}")

    if not dataframe["attendance_pct"].between(0.0, 100.0).all():
        raise ValueError("attendance_pct values outside valid range 0 to 100.")
    if not dataframe["assessment_pct"].between(0.0, 100.0).all():
        raise ValueError("assessment_pct values outside valid range 0 to 100.")
    if not dataframe["assignment_pct"].between(0.0, 100.0).all():
        raise ValueError("assignment_pct values outside valid range 0 to 100.")
    if not dataframe["engagement_score"].between(0.0, 10.0).all():
        raise ValueError("engagement_score values outside valid range 0 to 10.")
    if not dataframe["prior_performance_pct"].between(0.0, 100.0).all():
        raise ValueError("prior_performance_pct values outside valid range 0 to 100.")

    valid_labels = {"low", "medium", "high"}
    if not dataframe["risk_label"].isin(valid_labels).all():
        raise ValueError("risk_label must contain only 'low', 'medium', or 'high'.")

    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"Duplicate rows remain after cleaning: {duplicate_count}")


from sklearn.model_selection import train_test_split


def create_stratified_splits(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible balanced train, validation, and test datasets."""
    val_test_ratio = VALIDATION_RATIO + TEST_RATIO  # 0.30
    train_data, val_test_data = train_test_split(
        dataframe,
        test_size=val_test_ratio,
        random_state=SEED,
        stratify=dataframe["risk_label"],
    )

    validation_data, test_data = train_test_split(
        val_test_data,
        test_size=0.50,
        random_state=SEED,
        stratify=val_test_data["risk_label"],
    )

    return train_data, validation_data, test_data


def run_pipeline() -> None:
    """Run the complete Step 2 SC03 data pipeline."""
    raw_data = pd.read_csv(RAW_FILE)

    duplicate_rows_before = int(raw_data.duplicated().sum())
    missing_values_before = int(raw_data[FEATURE_COLUMNS].isna().sum().sum())

    cleaned_data = clean_data(raw_data)
    validate_data(cleaned_data)

    train_data, validation_data, test_data = create_stratified_splits(cleaned_data)

    if len(train_data) + len(validation_data) + len(test_data) != len(cleaned_data):
        raise ValueError("Split sizes do not match the cleaned dataset size.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_data.to_csv(
        PROCESSED_DIR / "student_performance_clean.csv", index=False
    )
    train_data.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation_data.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    validation_data.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_data.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("SC03 STEP 2 DATA PIPELINE COMPLETED")
    print("-" * 45)
    print(f"Raw rows: {len(raw_data)}")
    print(f"Missing values before cleaning: {missing_values_before}")
    print(f"Duplicate rows before cleaning: {duplicate_rows_before}")
    print(f"Cleaned rows: {len(cleaned_data)}")
    print(f"Training rows: {len(train_data)}")
    print(f"Validation rows: {len(validation_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Random seed: {SEED}")
    print("Output location:")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    run_pipeline()
