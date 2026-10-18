"""SC03 Step 4: Repeatable Baseline Comparison Program.

Entry point conforming to course guide standard naming (src/compare_baseline.py).
Executes the SC03 student performance deterministic baseline comparison
against the Step 3 Artificial Neural Network on the identical 1,800-record test split.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import evaluate_on_test_set, main

if __name__ == "__main__":
    main()
