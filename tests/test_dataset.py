"""
SC03 Student Performance and Academic Intervention System
Step 2 Automated Test Suite (tests/test_dataset.py)

Verifies:
- File presence and schema integrity (TR-010, TR-011)
- Validation passes on all processed splits
- Exact 70/15/15 stratified splits (TR-013)
- Strict isolation of protected attributes (TR-014)
- Manifest cryptographic SHA-256 verification (TR-011)
- Honest, non-circular baseline accuracy between 65% and 85% (TR-012)
"""

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from src.validate_data import validate

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MANIFEST_FILE = DATA_DIR / "manifest.json"
DATASET_FILE = DATA_DIR / "dataset.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
TEST_FILE = DATA_DIR / "test.csv"
PROTECTED_FILE = DATA_DIR / "protected_attributes.csv"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_processed_files_exist():
    """Verify all 5 processed CSVs and manifest exist."""
    assert MANIFEST_FILE.is_file(), "manifest.json missing"
    assert DATASET_FILE.is_file(), "dataset.csv missing"
    assert TRAIN_FILE.is_file(), "train.csv missing"
    assert VAL_FILE.is_file(), "val.csv missing"
    assert TEST_FILE.is_file(), "test.csv missing"
    assert PROTECTED_FILE.is_file(), "protected_attributes.csv missing"


@pytest.mark.parametrize("file_path", [DATASET_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE])
def test_all_splits_pass_schema_validation(file_path: Path):
    """Verify every split passes TR-001..TR-009 validation rules with 0 errors."""
    errors = validate(file_path)
    assert errors == [], f"Validation failed for {file_path.name}: {errors}"


def test_record_counts_and_split_ratios():
    """Verify exact 70/15/15 row counts and disjoint partitions."""
    df_all = pd.read_csv(DATASET_FILE)
    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)
    df_test = pd.read_csv(TEST_FILE)

    assert len(df_all) == 12000, f"Expected 12000 rows, got {len(df_all)}"
    assert len(df_train) == 8400, f"Expected 8400 train rows, got {len(df_train)}"
    assert len(df_val) == 1800, f"Expected 1800 val rows, got {len(df_val)}"
    assert len(df_test) == 1800, f"Expected 1800 test rows, got {len(df_test)}"

    # Disjointness check
    ids_train = set(df_train["record_id"])
    ids_val = set(df_val["record_id"])
    ids_test = set(df_test["record_id"])

    assert ids_train.isdisjoint(ids_val), "Train and Val contain overlapping record IDs"
    assert ids_train.isdisjoint(ids_test), "Train and Test contain overlapping record IDs"
    assert ids_val.isdisjoint(ids_test), "Val and Test contain overlapping record IDs"
    assert len(ids_train | ids_val | ids_test) == 12000


def test_stratification_consistency():
    """Verify class proportions are strictly stratified across train, val, test."""
    df_all = pd.read_csv(DATASET_FILE)
    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)
    df_test = pd.read_csv(TEST_FILE)

    for c in ["low", "medium", "high"]:
        all_pct = (df_all["risk_label"] == c).mean()
        train_pct = (df_train["risk_label"] == c).mean()
        val_pct = (df_val["risk_label"] == c).mean()
        test_pct = (df_test["risk_label"] == c).mean()

        assert abs(train_pct - all_pct) < 0.005, f"Train imbalance for class {c}"
        assert abs(val_pct - all_pct) < 0.005, f"Val imbalance for class {c}"
        assert abs(test_pct - all_pct) < 0.005, f"Test imbalance for class {c}"


def test_protected_attributes_isolation_and_privacy():
    """Verify TR-014: Protected attributes are completely decoupled from model features."""
    df_prot = pd.read_csv(PROTECTED_FILE)
    df_train = pd.read_csv(TRAIN_FILE)

    assert len(df_prot) == 12000
    assert "record_id" in df_prot.columns
    assert set(df_prot.columns) == {
        "record_id",
        "gender",
        "age_band",
        "disability",
        "socio_economic_decile",
    }

    # Model features must NOT have any demographic column
    protected_cols = {"gender", "age_band", "disability", "socio_economic_decile"}
    for col in protected_cols:
        assert col not in df_train.columns, f"Leaked protected column '{col}' into train.csv"

    # Protected attributes must NOT have academic indicator columns
    academic_cols = {
        "attendance_pct",
        "assessment_pct",
        "assignment_pct",
        "engagement_score",
        "prior_performance_pct",
        "risk_label",
    }
    for col in academic_cols:
        assert col not in df_prot.columns, f"Leaked academic column '{col}' into protected_attributes.csv"


def test_sha256_manifest_integrity():
    """Verify TR-011: Cryptographic hashes in manifest match actual files."""
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_hashes = manifest["sha256"]
    for filename, expected_hash in expected_hashes.items():
        filepath = DATA_DIR / filename
        assert filepath.is_file(), f"File {filename} in manifest does not exist"
        actual_hash = compute_sha256(filepath)
        assert actual_hash == expected_hash, (
            f"SHA-256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
        )


def test_non_circular_baseline_performance():
    """Verify TR-012: Baseline accuracy on test set is between 65% and 85% (not 100%)."""
    df_test = pd.read_csv(TEST_FILE)

    scores = (
        0.30 * df_test["attendance_pct"]
        + 0.30 * df_test["assessment_pct"]
        + 0.20 * df_test["assignment_pct"]
        + 0.10 * (df_test["engagement_score"] * 10.0)
        + 0.10 * df_test["prior_performance_pct"]
    )

    preds = pd.Series("medium", index=df_test.index)
    preds[scores < 50.0] = "high"
    preds[scores >= 70.0] = "low"

    targets = df_test["risk_label"]
    accuracy = (preds == targets).mean()

    # Must be realistic non-circular performance
    assert 0.65 <= accuracy <= 0.85, (
        f"Baseline accuracy ({accuracy:.4f}) is out of expected range [0.65, 0.85]. "
        "Accuracy >= 0.98 indicates circular label leakage."
    )
