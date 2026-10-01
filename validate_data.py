"""Step 1 data validator for the SC03 Student Performance and Academic
Intervention System.

Why this exists: bad data must be caught before any baseline or neural model
sees it. The validator reports EVERY problem, not only the first one.

Usage (run from the repository root):
    python src/validate_data.py                      # checks data/sample_input.csv
    python src/validate_data.py data/invalid_cases.csv   # must FAIL with clear messages
"""
import re
import sys

import pandas as pd

DEFAULT_FILE = "data/sample_input.csv"

REQUIRED = [
    "record_id",
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "engagement_score",
    "prior_performance_pct",
    "risk_label",
]
PERCENT_COLUMNS = [
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "prior_performance_pct",
]
NUMERIC = PERCENT_COLUMNS + ["engagement_score"]
ALLOWED_LABELS = {"low", "medium", "high"}
ID_PATTERN = re.compile(r"^STU-\d{3,}$")
# Headers that would indicate personal data. The project must stay anonymous.
FORBIDDEN_WORDS = ["name", "email", "phone", "roll", "address", "dob"]


def validate(df: pd.DataFrame) -> list:
    """Return a list of human-readable problems. Empty list means valid."""
    problems = []

    # Rule: no personal-data columns.
    for column in df.columns:
        if any(word in str(column).lower() for word in FORBIDDEN_WORDS):
            problems.append(f"Forbidden column: {column} (data must be anonymous)")

    # Rule: required columns present, exact spelling.
    missing_columns = [c for c in REQUIRED if c not in df.columns]
    for column in missing_columns:
        problems.append(f"Missing column: {column}")
    if missing_columns:
        return problems  # row checks need the columns to exist

    # Rule: no missing values (report row by row).
    for index, row in df.iterrows():
        for column in REQUIRED:
            if pd.isna(row[column]):
                problems.append(f"Row {_name(row, index)}: {column} is missing")

    # Rule: numeric columns must be numbers.
    numeric_values = {}
    for column in NUMERIC:
        converted = pd.to_numeric(df[column], errors="coerce")
        numeric_values[column] = converted
        for index, row in df.iterrows():
            if pd.notna(row[column]) and pd.isna(converted[index]):
                problems.append(
                    f'Row {_name(row, index)}: {column} must be a number, got "{row[column]}"'
                )

    # Rule: percentages 0 to 100, engagement 0 to 10.
    for column in PERCENT_COLUMNS:
        for index, value in numeric_values[column].items():
            if pd.notna(value) and not 0 <= value <= 100:
                problems.append(
                    f"Row {_name(df.loc[index], index)}: {column} must be between 0 and 100, got {_fmt(value)}"
                )
    for index, value in numeric_values["engagement_score"].items():
        if pd.notna(value) and not 0 <= value <= 10:
            problems.append(
                f"Row {_name(df.loc[index], index)}: engagement_score must be between 0 and 10, got {_fmt(value)}"
            )

    # Rule: risk_label must be low, medium or high (lower case).
    for index, row in df.iterrows():
        label = row["risk_label"]
        if pd.notna(label) and label not in ALLOWED_LABELS:
            problems.append(
                f"Row {_name(row, index)}: risk_label must be low, medium or high, got {label}"
            )

    # Rule: identifiers unique and well formed.
    for index, row in df.iterrows():
        record_id = row["record_id"]
        if pd.notna(record_id) and not ID_PATTERN.match(str(record_id)):
            problems.append(
                f"Row {_name(row, index)}: record_id must look like STU-001, got {record_id}"
            )
    for record_id in df["record_id"][df["record_id"].duplicated()].unique():
        problems.append(f"Duplicate record_id: {record_id}")

    return problems


def _name(row, index) -> str:
    """Use the record id when known, else the CSV line number."""
    record_id = row.get("record_id")
    return str(record_id) if pd.notna(record_id) else f"on line {index + 2}"


def _fmt(value) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"VALIDATION FAILED: file not found: {path}")
        return 1

    print("file:", path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("missing values:", df.isna().sum().to_dict())

    problems = validate(df)
    if problems:
        print(f"VALIDATION FAILED: {len(problems)} problem(s) found")
        for number, problem in enumerate(problems, start=1):
            print(f"  {number}. {problem}")
        return 1

    print("STEP 1 DATA CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
