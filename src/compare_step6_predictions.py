"""Compare the SC03 reference CSV with the normal-case app download."""
import argparse
import csv
from math import isclose
from pathlib import Path
import sys

# Support running both from project root and from src/
try:
    from src.product_predict import FEATURES, ROOT
except ImportError:
    from product_predict import FEATURES, ROOT

REPORT_DIR = ROOT / "data" / "reports"


def read_prediction(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise ValueError(f"{path.name}: expected exactly one prediction row.")
    row = rows[0]
    for field in [*FEATURES, "predicted_class", "label"]:
        if not row.get(field):
            raise ValueError(f"{path.name}: missing value for {field}.")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference",
        type=Path,
        nargs="?",
        default=REPORT_DIR / "step5_ui_reference_prediction.csv",
    )
    parser.add_argument(
        "app_download",
        type=Path,
        nargs="?",
        default=REPORT_DIR / "step6_ui_prediction.csv",
    )
    args = parser.parse_args()
    try:
        reference = read_prediction(args.reference)
        app = read_prediction(args.app_download)
        differences = [
            name
            for name in FEATURES
            if not isclose(
                float(reference[name]),
                float(app[name]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ]
        differences.extend(
            name
            for name in ("predicted_class", "label")
            if reference[name].strip() != app[name].strip()
        )
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"FAIL: {error}\n")
    if differences:
        parser.exit(1, f"FAIL: Different values in {', '.join(differences)}.\n")
    print("PASS: All five inputs, the predicted class and the label match.")
    print(f"Class {app['predicted_class']} - {app['label']}")


if __name__ == "__main__":
    main()

