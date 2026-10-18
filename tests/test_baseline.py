"""
SC03 Student Performance and Academic Intervention System
Step 4 Automated Test Suite (tests/test_baseline.py)

Verifies:
- Deterministic scoring function matches docs/baseline-pseudocode.md formulas
- Threshold boundary mappings (<50 high, 50-70 medium, >=70 low)
- 5 Golden Starter Cases from docs/STEP-1.md and data/sample_input.csv match expected baseline behavior
- Full test split evaluation executes without errors
- Baseline accuracy falls within the expected non-trivial pedagogical window (65% to 85%)
- Side-by-side comparison CSV contains 1,800 records and correct headers
"""

from pathlib import Path
import pandas as pd
import pytest

from src.baseline import (
    calculate_baseline_score,
    classify_risk,
    predict_single,
    verify_and_align_test_data,
    INTERVENTIONS,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TEST_FILE = DATA_DIR / "test.csv"
COMPARISON_FILE = ROOT / "data" / "reports" / "step4_test_comparison_full.csv"
METRICS_FILE = ROOT / "data" / "reports" / "step4_comparison_metrics.csv"
ANN_PRED_FILE = ROOT / "data" / "reports" / "step3_test_predictions.csv"


def test_baseline_score_calculation():
    """Verify linear formula calculation."""
    # 30% att + 30% assess + 20% assign + 10% (eng*10) + 10% prior
    # 0.3*100 + 0.3*100 + 0.2*100 + 0.1*100 + 0.1*100 = 100.0
    perfect = calculate_baseline_score(100.0, 100.0, 100.0, 10.0, 100.0)
    assert perfect == 100.0

    # 0.3*0 + 0.3*0 + 0.2*0 + 0.1*0 + 0.1*0 = 0.0
    zero = calculate_baseline_score(0.0, 0.0, 0.0, 0.0, 0.0)
    assert zero == 0.0

    # Intermediate calculation
    # 0.3*70 + 0.3*60 + 0.2*80 + 0.1*70 + 0.1*50 = 21 + 18 + 16 + 7 + 5 = 67.0
    score = calculate_baseline_score(70.0, 60.0, 80.0, 7.0, 50.0)
    assert score == 67.0


def test_classify_risk_thresholds():
    """Verify exact cutoff boundaries: <50 High, 50-70 Medium, >=70 Low."""
    assert classify_risk(49.99) == "high"
    assert classify_risk(50.0) == "medium"
    assert classify_risk(69.99) == "medium"
    assert classify_risk(70.0) == "low"
    assert classify_risk(95.0) == "low"


def test_predict_single_structure():
    """Verify predict_single returns complete schema with interventions."""
    result = predict_single(
        attendance_pct=85.0,
        assessment_pct=75.0,
        assignment_pct=80.0,
        engagement_score=7.0,
        prior_performance_pct=70.0,
    )
    assert "score" in result
    assert "risk_label" in result
    assert "intervention" in result
    assert "rule_applied" in result
    assert "ethical_notice" in result
    assert result["risk_label"] == "low"
    assert result["intervention"] == "Monitor"


def test_golden_cases():
    """Verify behavior on the 5 canonical starter cases from docs/STEP-1.md."""
    golden_cases = [
        {
            "id": "STU-001",
            "attendance": 85.0,
            "assessment": 78.0,
            "assignment": 82.0,
            "engagement": 7.5,
            "prior": 72.0,
            "expected_band": "low",
        },
        {
            "id": "STU-002",
            "attendance": 62.0,
            "assessment": 58.0,
            "assignment": 65.0,
            "engagement": 5.2,
            "prior": 60.0,
            "expected_band": "medium",
        },
        {
            "id": "STU-003",
            "attendance": 45.0,
            "assessment": 40.0,
            "assignment": 50.0,
            "engagement": 3.8,
            "prior": 55.0,
            "expected_band": "high",
        },
        {
            "id": "STU-004",
            "attendance": 92.0,
            "assessment": 90.0,
            "assignment": 95.0,
            "engagement": 8.8,
            "prior": 88.0,
            "expected_band": "low",
        },
        {
            "id": "STU-005",
            "attendance": 55.0,
            "assessment": 62.0,
            "assignment": 48.0,
            "engagement": 4.5,
            "prior": 58.0,
            "expected_band": "medium",
        },
    ]

    for case in golden_cases:
        res = predict_single(
            case["attendance"],
            case["assessment"],
            case["assignment"],
            case["engagement"],
            case["prior"],
        )
        assert res["risk_label"] == case["expected_band"], (
            f"Golden case {case['id']} expected {case['expected_band']}, "
            f"got {res['risk_label']} (score {res['score']})"
        )


def test_baseline_accuracy_on_test_set():
    """Verify baseline test performance on sealed test split (70% - 85%)."""
    df = pd.read_csv(TEST_FILE)
    assert len(df) == 1800

    scores = (
        0.30 * df["attendance_pct"]
        + 0.30 * df["assessment_pct"]
        + 0.20 * df["assignment_pct"]
        + 0.10 * (df["engagement_score"] * 10.0)
        + 0.10 * df["prior_performance_pct"]
    )
    preds = scores.apply(classify_risk)
    acc = (preds == df["risk_label"]).mean()

    assert 0.70 <= acc <= 0.85, f"Baseline accuracy {acc:.4f} is outside 70-85% expected range"
    assert round(acc, 4) == 0.7828


def test_comparison_csv_artifacts_exist():
    """Verify Step 4 export artifacts exist and have valid structure."""
    assert COMPARISON_FILE.is_file(), "step4_test_comparison_full.csv is missing"
    assert METRICS_FILE.is_file(), "step4_comparison_metrics.csv is missing"

    df_comp = pd.read_csv(COMPARISON_FILE)
    assert len(df_comp) == 1800
    expected_cols = [
        "record_id",
        "attendance_pct",
        "assessment_pct",
        "assignment_pct",
        "engagement_score",
        "prior_performance_pct",
        "risk_label",
        "baseline_score",
        "baseline_prediction",
        "baseline_intervention",
        "ann_prediction",
        "baseline_correct",
        "ann_correct",
        "ann_improved_over_baseline",
    ]
    for col in expected_cols:
        assert col in df_comp.columns, f"Missing column {col} in comparison CSV"

    df_metrics = pd.read_csv(METRICS_FILE)
    assert "metric" in df_metrics.columns
    assert "baseline" in df_metrics.columns
    assert "ann" in df_metrics.columns
    assert "difference" in df_metrics.columns


def test_explicit_alignment_check():
    """Verify that verify_and_align_test_data enforces row-ID, label, and feature alignment."""
    df_test = pd.read_csv(TEST_FILE)
    df_ann = pd.read_csv(ANN_PRED_FILE)

    # 1. Valid datasets must pass without error and return 1,800 records
    aligned_test, aligned_ann = verify_and_align_test_data(df_test, df_ann)
    assert len(aligned_test) == 1800
    assert len(aligned_ann) == 1800
    assert (aligned_test["record_id"].values == aligned_ann["record_id"].values).all()

    # 2. Length mismatch must raise ValueError
    with pytest.raises(ValueError, match="Row count mismatch"):
        verify_and_align_test_data(df_test.iloc[:-1], df_ann)

    # 3. Altered record_id must raise ValueError
    df_ann_bad_id = df_ann.copy()
    df_ann_bad_id.loc[0, "record_id"] = "STU-CORRUPTED"
    with pytest.raises(ValueError, match="Record ID mismatch"):
        verify_and_align_test_data(df_test, df_ann_bad_id)

    # 4. Altered risk_label must raise ValueError
    df_ann_bad_label = df_ann.copy()
    df_ann_bad_label.loc[0, "risk_label"] = "low" if df_ann.loc[0, "risk_label"] != "low" else "high"
    with pytest.raises(ValueError, match="risk_label mismatch"):
        verify_and_align_test_data(df_test, df_ann_bad_label)

    # 5. Altered feature values must raise ValueError
    df_ann_bad_feat = df_ann.copy()
    df_ann_bad_feat.loc[0, "attendance_pct"] += 25.0
    with pytest.raises(ValueError, match="Feature 'attendance_pct' values do not align"):
        verify_and_align_test_data(df_test, df_ann_bad_feat)


def test_compare_baseline_entrypoint():
    """Verify that src/compare_baseline.py is importable and functional."""
    from src.compare_baseline import evaluate_on_test_set
    results = evaluate_on_test_set()
    assert "baseline_accuracy" in results
    assert "ann_accuracy" in results
    assert results["baseline_accuracy"] == pytest.approx(0.7828, abs=1e-4)
    assert results["ann_accuracy"] == pytest.approx(0.8411, abs=1e-4)


