"""SC03 Product V1 validation and unit tests."""
import unittest
from src.product_predict import FEATURES, validate_inputs


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.values = {name: "75.0" for name in FEATURES}
        self.values["engagement_score"] = "7.5"

    def test_normal(self):
        clean = validate_inputs(self.values)
        self.assertEqual(clean["attendance_pct"], 75.0)
        self.assertEqual(clean["engagement_score"], 7.5)

    def test_boundaries(self):
        for value in ("0", "100"):
            self.values["attendance_pct"] = value
            self.assertEqual(validate_inputs(self.values)["attendance_pct"], float(value))
        for value in ("0", "10"):
            self.values["engagement_score"] = value
            self.assertEqual(validate_inputs(self.values)["engagement_score"], float(value))

    def test_invalid_inputs(self):
        for value in ("-1", "105", "", "abc", "nan", "inf"):
            self.values["attendance_pct"] = value
            with self.assertRaises(ValueError):
                validate_inputs(self.values)
        for value in ("-1", "12", "invalid"):
            self.values["engagement_score"] = value
            with self.assertRaises(ValueError):
                validate_inputs(self.values)

    def test_negative_measurement(self):
        self.values["assessment_pct"] = "-5"
        with self.assertRaises(ValueError):
            validate_inputs(self.values)

    def test_missing_feature(self):
        self.values.pop("attendance_pct")
        with self.assertRaises(ValueError):
            validate_inputs(self.values)


if __name__ == "__main__":
    unittest.main()
