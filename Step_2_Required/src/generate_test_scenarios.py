"""
SC03 Student Performance and Academic Intervention System
Step 2 Testing Scenario Generator (Actions 64, 67)

Generates 1,200 testing-only scenarios (10% ratio of the 12,000 benchmark dataset).
These scenarios are STRICTLY reserved for stress testing, boundary analysis,
and product verification. They are NEVER mixed into train.csv, validation.csv, or test.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_ROOT / "data" / "processed" / "student_performance_clean.csv"
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_FILE = GENERATED_DIR / "generated_student_performance_test_scenarios.csv"
REPORT_FILE = PROJECT_ROOT / "data" / "reports" / "generated_scenarios_report.md"

FEATURE_COLUMNS = [
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "engagement_score",
    "prior_performance_pct",
]

PERCENTAGE_COLUMNS = [
    "attendance_pct",
    "assessment_pct",
    "assignment_pct",
    "prior_performance_pct",
]

SEED = 42
SCENARIO_RATIO = 0.10  # 10% of real dataset


def validate_scenarios(
    generated: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    expected_count: int,
) -> None:
    """Prove that scenarios are complete, valid, and not copied real records."""
    if len(generated) != expected_count:
        raise ValueError(
            f"Expected {expected_count} scenarios but found {len(generated)}."
        )

    if generated.isna().sum().sum() != 0:
        raise ValueError("Generated scenarios contain missing values.")

    if generated["scenario_id"].duplicated().any():
        raise ValueError("Generated scenario IDs must be unique.")

    if generated[FEATURE_COLUMNS].duplicated().any():
        raise ValueError("Generated scenarios must not duplicate one another.")

    for col in PERCENTAGE_COLUMNS:
        if not generated[col].between(0.0, 100.0).all():
            raise ValueError(f"Generated {col} values must remain between 0 and 100.")

    if not generated["engagement_score"].between(0.0, 10.0).all():
        raise ValueError("Generated engagement_score values must remain between 0 and 10.")

    valid_labels = {"low", "medium", "high"}
    if not generated["expected_risk_label"].isin(valid_labels).all():
        raise ValueError("expected_risk_label must contain only 'low', 'medium', or 'high'.")

    # Anti-leakage safeguard: ensure no generated scenario is an exact duplicate of a real record
    real_feature_rows = set(
        tuple(round(float(x), 1) for x in row)
        for row in cleaned_data[FEATURE_COLUMNS].itertuples(index=False, name=None)
    )
    generated_feature_rows = set(
        tuple(round(float(x), 1) for x in row)
        for row in generated[FEATURE_COLUMNS].itertuples(index=False, name=None)
    )
    copied_rows = real_feature_rows.intersection(generated_feature_rows)
    if copied_rows:
        raise ValueError(
            f"Generated scenarios must not be exact copies of real records. Found: {len(copied_rows)}"
        )


def generate_scenarios(
    cleaned_data: pd.DataFrame,
    count: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate perturbed edge cases and testing scenarios from the cleaned dataset."""
    rng = np.random.default_rng(seed)

    real_feature_rows = set(
        tuple(round(float(x), 1) for x in row)
        for row in cleaned_data[FEATURE_COLUMNS].itertuples(index=False, name=None)
    )

    # Sample representative rows across classes
    sampled_indices = []
    classes = cleaned_data["risk_label"].unique()
    per_class = count // len(classes)

    for cls in classes:
        cls_idx = cleaned_data[cleaned_data["risk_label"] == cls].index
        sampled = rng.choice(cls_idx, size=per_class, replace=True)
        sampled_indices.extend(sampled)

    remainder = count - len(sampled_indices)
    if remainder > 0:
        extra = rng.choice(cleaned_data.index, size=remainder, replace=True)
        sampled_indices.extend(extra)

    base_sample = cleaned_data.loc[sampled_indices].copy().reset_index(drop=True)

    jittered = base_sample.copy()
    for col in FEATURE_COLUMNS:
        iqr = float(cleaned_data[col].quantile(0.75) - cleaned_data[col].quantile(0.25))
        jitter_std = max(iqr * 0.05, 0.5)
        noise = rng.normal(0.0, jitter_std, size=count)
        # Nudge zero noise
        noise[np.abs(noise) < 0.2] = 0.6
        jittered[col] = (jittered[col] + noise).round(1)

    # Clip to legal physical bounds
    for col in PERCENTAGE_COLUMNS:
        jittered[col] = jittered[col].clip(0.0, 100.0)
    jittered["engagement_score"] = jittered["engagement_score"].clip(0.0, 10.0)

    # Ensure no exact copies of real records and no internal duplicate scenarios
    seen_tuples = set()
    for i in range(count):
        tup = tuple(round(float(jittered.loc[i, c]), 1) for c in FEATURE_COLUMNS)
        while tup in real_feature_rows or tup in seen_tuples:
            jittered.loc[i, "attendance_pct"] = round(float((jittered.loc[i, "attendance_pct"] + 0.7) % 100.0), 1)
            jittered.loc[i, "assessment_pct"] = round(float((jittered.loc[i, "assessment_pct"] + 0.3) % 100.0), 1)
            tup = tuple(round(float(jittered.loc[i, c]), 1) for c in FEATURE_COLUMNS)
        seen_tuples.add(tup)


    # Assign clean scenario metadata
    jittered["scenario_id"] = [f"GEN-STU-{i+1:04d}" for i in range(count)]
    jittered["expected_risk_label"] = base_sample["risk_label"]

    columns_order = [
        "scenario_id",
        "attendance_pct",
        "assessment_pct",
        "assignment_pct",
        "engagement_score",
        "prior_performance_pct",
        "expected_risk_label",
    ]
    return jittered[columns_order]


def run_generator() -> pd.DataFrame:
    """Generate, validate, and save the testing scenario suite."""
    cleaned_df = pd.read_csv(CLEAN_FILE)
    expected_count = int(round(len(cleaned_df) * SCENARIO_RATIO))

    scenarios = generate_scenarios(cleaned_df, count=expected_count, seed=SEED)
    validate_scenarios(scenarios, cleaned_df, expected_count=expected_count)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(OUTPUT_FILE, index=False)

    print("SC03 GENERATED TESTING SCENARIOS COMPLETED")
    print("-" * 45)
    print(f"Real cleaned records: {len(cleaned_df):,}")
    print(f"Generated testing scenarios: {len(scenarios):,}")
    print(f"Generation seed: {SEED}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Report file: {REPORT_FILE}")
    print(f"Scenario table shape: {scenarios.shape}")
    print("\nExpected class balance:")
    print(scenarios["expected_risk_label"].value_counts().to_string())
    print("\nFirst five generated testing scenarios:")
    print(scenarios.head().to_string(index=False))

    return scenarios


if __name__ == "__main__":
    run_generator()
