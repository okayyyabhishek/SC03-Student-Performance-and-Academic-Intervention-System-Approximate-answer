"""
SC03 Student Performance and Academic Intervention System
Step 1 Data Validation Script (TR-001 through TR-009)

Verifies file shape, column headers, data types, value boundaries,
uniqueness, missing values, and ensures strict privacy (no PII columns).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
import pandas as pd

# Canonical definitions per data dictionary
REQUIRED_COLUMNS = [
    "record_id",
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "engagement_score",
    "prior_performance_pct",
    "risk_label",
]

PERCENTAGE_COLUMNS = [
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "prior_performance_pct",
]

FORBIDDEN_PII_PATTERNS = ["name", "email", "phone", "roll", "address", "dob", "ssn"]
ALLOWED_RISK_LABELS = {"low", "medium", "high"}
RECORD_ID_REGEX = re.compile(r"^STU-\d{3,}$")


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Validate DataFrame against TR-001 through TR-009 rules.

    Returns a list of error strings. Empty list indicates full PASS.
    """
    errors: list[str] = []

    # TR-008: Check for forbidden personal data columns
    for col in df.columns:
        col_lower = str(col).strip().lower()
        for forbidden in FORBIDDEN_PII_PATTERNS:
            if forbidden in col_lower:
                errors.append(f"TR-008 Forbidden personal-data column found: '{col}'")

    # TR-001: Check for exact column presence and spelling
    for req in REQUIRED_COLUMNS:
        if req not in df.columns:
            # Check if accidental whitespace exists
            stripped_matches = [c for c in df.columns if str(c).strip() == req]
            if stripped_matches:
                errors.append(
                    f"TR-001 Column '{req}' has leading/trailing whitespace in header: '{stripped_matches[0]}'"
                )
            else:
                errors.append(f"TR-001 Missing required column: '{req}'")

    if errors:
        # If columns are fundamentally broken, return immediately
        return errors

    # TR-007: Uniqueness and format of record_id
    seen_ids: set[str] = set()
    for idx, raw_id in enumerate(df["record_id"]):
        row_num = idx + 1
        str_id = str(raw_id).strip()
        if pd.isna(raw_id) or not str_id:
            errors.append(f"TR-002 Row {row_num}: record_id is missing or empty")
            continue
        if not RECORD_ID_REGEX.match(str_id):
            errors.append(
                f"TR-007 Row {row_num} ({str_id}): record_id must match format ^STU-\\d{{3,}}$, got '{str_id}'"
            )
        if str_id in seen_ids:
            errors.append(f"TR-007 Duplicate record_id found: '{str_id}' at row {row_num}")
        seen_ids.add(str_id)

    # Validate each row's fields
    for idx, row in df.iterrows():
        rec_id = str(row.get("record_id", f"Row-{idx + 1}"))

        # TR-002: Check for missing values in row
        for col in REQUIRED_COLUMNS:
            val = row[col]
            if pd.isna(val) or (isinstance(val, str) and not val.strip()):
                errors.append(f"TR-002 Row {rec_id}: '{col}' is missing")

        # TR-004: Validate Percentage columns
        for col in PERCENTAGE_COLUMNS:
            val = row[col]
            if pd.notna(val):
                try:
                    num_val = float(val)
                    if num_val < 0.0 or num_val > 100.0:
                        errors.append(
                            f"TR-004 Row {rec_id}: {col} must be between 0 and 100, got {num_val}"
                        )
                except (ValueError, TypeError):
                    errors.append(
                        f"TR-003 Row {rec_id}: {col} must be numeric, got '{val}'"
                    )

        # TR-005: Validate engagement_score (0 to 10)
        eng_val = row["engagement_score"]
        if pd.notna(eng_val):
            try:
                num_eng = float(eng_val)
                if num_eng < 0.0 or num_eng > 10.0:
                    errors.append(
                        f"TR-005 Row {rec_id}: engagement_score must be between 0 and 10, got {num_eng}"
                    )
            except (ValueError, TypeError):
                errors.append(
                    f"TR-003 Row {rec_id}: engagement_score must be numeric, got '{eng_val}'"
                )

        # TR-006: Validate risk_label
        label = row["risk_label"]
        if pd.notna(label):
            str_label = str(label).strip()
            if str_label not in ALLOWED_RISK_LABELS:
                errors.append(
                    f"TR-006 Row {rec_id}: risk_label must be 'low', 'medium', or 'high', got '{str_label}'"
                )

    return errors


def validate(path: Path | str) -> list[str]:
    """Load a CSV file and run comprehensive validation."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_csv(p)
    return validate_dataframe(df)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_file = root / "data" / "sample_input.csv"
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_file

    print(f"Validating dataset: {target_path}")
    if not target_path.exists():
        print(f"Error: File does not exist at {target_path}")
        sys.exit(1)

    df = pd.read_csv(target_path)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isna().sum().to_dict()}")
    print("-" * 60)

    errors = validate_dataframe(df)

    if not errors:
        print("STEP 1 DATA CHECK PASSED")
        sys.exit(0)
    else:
        print(f"FAILED with {len(errors)} error(s):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
