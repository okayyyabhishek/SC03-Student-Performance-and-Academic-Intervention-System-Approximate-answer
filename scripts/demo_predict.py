"""Demonstration runner for SC03 Product V1 Student Academic Risk Prediction."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.product_predict import load_model, predict_one

def main():
    model = load_model()
    scenarios = [
        {
            "title": "Case 1: Excelling Student (Low Risk)",
            "inputs": {
                "attendance_pct": "95.0",
                "assessment_pct": "88.0",
                "assignment_pct": "92.0",
                "engagement_score": "8.5",
                "prior_performance_pct": "85.0",
            },
        },
        {
            "title": "Case 2: Borderline / Struggling Student (Medium Risk)",
            "inputs": {
                "attendance_pct": "72.0",
                "assessment_pct": "60.0",
                "assignment_pct": "65.0",
                "engagement_score": "5.2",
                "prior_performance_pct": "62.0",
            },
        },
        {
            "title": "Case 3: Critically At-Risk Student (High Risk)",
            "inputs": {
                "attendance_pct": "45.0",
                "assessment_pct": "38.0",
                "assignment_pct": "40.0",
                "engagement_score": "2.1",
                "prior_performance_pct": "42.0",
            },
        },
        {
            "title": "Case 4: Input Validation Guardrail (Negative Attendance)",
            "inputs": {
                "attendance_pct": "-10.0",
                "assessment_pct": "50.0",
                "assignment_pct": "50.0",
                "engagement_score": "5.0",
                "prior_performance_pct": "50.0",
            },
        },
        {
            "title": "Case 5: Input Validation Guardrail (Out-of-range LMS Score)",
            "inputs": {
                "attendance_pct": "80.0",
                "assessment_pct": "75.0",
                "assignment_pct": "70.0",
                "engagement_score": "15.0",  # Max is 10.0
                "prior_performance_pct": "72.0",
            },
        },
    ]

    print("=" * 85)
    print("      SC03 STUDENT PERFORMANCE & ACADEMIC INTERVENTION PREDICTOR - DEMONSTRATION")
    print("=" * 85)

    results = []
    for sc in scenarios:
        print(f"\n>>> {sc['title']}")
        print(f"    Inputs: {sc['inputs']}")
        try:
            res = predict_one(model, sc["inputs"])
            print(f"    -> Status: SUCCESS")
            print(f"    -> Predicted Class: {res['predicted_class'].upper()}")
            print(f"    -> Display Label:   {res['label']}")
            print(f"    -> Recommended Action: {res['intervention']}")
            results.append({
                "Scenario": sc["title"],
                "Prediction": res["predicted_class"],
                "Intervention": res["intervention"],
                "Status": "Valid",
            })
        except ValueError as err:
            print(f"    -> Status: REJECTED (Safety Validation)")
            print(f"    -> Error Caught: {err}")
            results.append({
                "Scenario": sc["title"],
                "Prediction": "N/A",
                "Intervention": f"Rejected: {err}",
                "Status": "Blocked",
            })

    print("\n" + "=" * 85)
    print("SUMMARY TABLE:")
    print("=" * 85)
    df_summary = pd.DataFrame(results)
    print(df_summary.to_string(index=False))
    print("=" * 85)

if __name__ == "__main__":
    main()
