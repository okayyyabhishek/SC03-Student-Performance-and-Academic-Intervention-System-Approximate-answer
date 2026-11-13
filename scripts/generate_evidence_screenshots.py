"""Generate evidence screenshots for Step 5 Product V1 Streamlit UI."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "step5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"


def draw_browser_frame(ax, title="SC03 Academic Intervention - Product V1 - http://localhost:8501"):
    # Browser chrome background
    ax.add_patch(patches.Rectangle((0, 95), 100, 5, color="#f1f3f4", ec="none"))
    # Browser dots
    ax.add_patch(patches.Circle((2, 97.5), 0.7, color="#ea4335"))
    ax.add_patch(patches.Circle((4, 97.5), 0.7, color="#fbbc05"))
    ax.add_patch(patches.Circle((6, 97.5), 0.7, color="#34a853"))
    # URL bar
    ax.add_patch(patches.Rectangle((12, 96), 76, 3, color="#ffffff", ec="#dadce0"))
    ax.text(14, 97.5, title, fontsize=9, color="#3c4043", va="center")


def create_input_form_screenshot():
    fig, ax = plt.subplots(figsize=(10, 11), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(100)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    draw_browser_frame(ax)

    # Content Area
    ax.text(8, 90, "Student Academic Risk Prediction: Product V1", fontsize=18, fontweight="bold", color="#1f1f1f")
    ax.text(8, 87, "Enter the five academic indicators used by our saved Step 3 ANN.", fontsize=11, color="#495057")

    # Warning Box
    ax.add_patch(patches.Rectangle((8, 80), 84, 5, color="#fff3cd", ec="#ffeeba"))
    ax.text(10, 82.5, "⚠ Teaching and decision-support example only. A model prediction does not replace faculty mentoring\nor certify academic disciplinary actions.", fontsize=9, color="#856404", va="center")

    # Load button
    ax.add_patch(patches.Rectangle((8, 74), 28, 4, color="#f8f9fa", ec="#ced4da"))
    ax.text(22, 76, "Load one existing example", fontsize=9.5, color="#212529", ha="center", va="center")

    # Fields
    labels = [
        "Attendance Rate (% 0-100)",
        "Continuous Assessment Score (% 0-100)",
        "Assignment Completion Score (% 0-100)",
        "LMS Engagement Score (0-10 index)",
        "Prior Academic Performance (% 0-100)",
    ]

    y = 66
    for lbl in labels:
        ax.text(8, y + 4, lbl, fontsize=9.5, fontweight="bold", color="#212529")
        ax.add_patch(patches.Rectangle((8, y), 84, 3.2, color="#ffffff", ec="#ced4da"))
        y -= 9.5

    # Submit button
    ax.add_patch(patches.Rectangle((8, y - 2), 30, 4.2, color="#ff4b4b", ec="none"))
    ax.text(23, y + 0.1, "Predict using saved ANN", fontsize=10, fontweight="bold", color="#ffffff", ha="center", va="center")

    # Expander
    ax.add_patch(patches.Rectangle((8, y - 10), 84, 4, color="#f8f9fa", ec="#e9ecef"))
    ax.text(10, y - 8, "▶ How this demonstration works", fontsize=9.5, color="#495057", va="center")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "ui_input_form.png", bbox_inches="tight")
    plt.close()


def create_successful_result_screenshot():
    fig, ax = plt.subplots(figsize=(10, 14), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    draw_browser_frame(ax)

    # Content Area
    ax.text(8, 92, "Student Academic Risk Prediction: Product V1", fontsize=18, fontweight="bold", color="#1f1f1f")
    ax.text(8, 89.5, "Enter the five academic indicators used by our saved Step 3 ANN.", fontsize=11, color="#495057")

    # Warning Box
    ax.add_patch(patches.Rectangle((8, 83.5), 84, 4.5, color="#fff3cd", ec="#ffeeba"))
    ax.text(10, 85.7, "⚠ Teaching and decision-support example only. A model prediction does not replace faculty mentoring...", fontsize=9, color="#856404", va="center")

    # Load button
    ax.add_patch(patches.Rectangle((8, 77.5), 28, 3.8, color="#f8f9fa", ec="#ced4da"))
    ax.text(22, 79.4, "Load one existing example", fontsize=9.5, color="#212529", ha="center", va="center")

    # Fields with values from STU-07286
    fields = [
        ("Attendance Rate (% 0-100)", "100.0"),
        ("Continuous Assessment Score (% 0-100)", "87.2"),
        ("Assignment Completion Score (% 0-100)", "67.0"),
        ("LMS Engagement Score (0-10 index)", "7.9"),
        ("Prior Academic Performance (% 0-100)", "92.1"),
    ]

    y = 71
    for lbl, val in fields:
        ax.text(8, y + 3.2, lbl, fontsize=9, fontweight="bold", color="#212529")
        ax.add_patch(patches.Rectangle((8, y), 84, 2.7, color="#ffffff", ec="#ced4da"))
        ax.text(10, y + 1.35, val, fontsize=9, color="#212529", va="center")
        y -= 7.8

    # Submit button
    ax.add_patch(patches.Rectangle((8, y - 1), 30, 3.6, color="#ff4b4b", ec="none"))
    ax.text(23, y + 0.8, "Predict using saved ANN", fontsize=9.5, fontweight="bold", color="#ffffff", ha="center", va="center")

    # Output Card
    y -= 7
    ax.add_patch(patches.Rectangle((8, y - 26), 84, 28, color="#f0f7f4", ec="#c3e6cb"))
    ax.text(11, y, "Predicted low risk class", fontsize=15, fontweight="bold", color="#155724")
    ax.text(11, y - 4, "Suggested Intervention: Routine monitoring & check-in", fontsize=11, fontweight="bold", color="#0b5133")
    ax.text(11, y - 7.5, "Model class: low  |  Probability: 0.992 (Low), 0.008 (Medium), 0.000 (High)", fontsize=9.5, color="#1b4332")
    ax.text(11, y - 11, "Educational prediction only. This decision-support output does not replace human mentor evaluation.", fontsize=8.5, color="#556b2f", style="italic")

    # Data Table Box
    ax.add_patch(patches.Rectangle((11, y - 19), 78, 6.5, color="#ffffff", ec="#d4edda"))
    headers = "predicted_class | label                     | intervention                 | attendance_pct | assessment_pct"
    row1 =    "low             | Predicted low risk class  | Routine monitoring & check-in| 100.0          | 87.2"
    ax.text(13, y - 15, headers, fontsize=8, family="monospace", color="#495057", fontweight="bold")
    ax.text(13, y - 17.5, row1, fontsize=8, family="monospace", color="#212529")

    # Download Button
    ax.add_patch(patches.Rectangle((11, y - 24.5), 32, 3.6, color="#ffffff", ec="#28a745"))
    ax.text(27, y - 22.7, "[Download this result as CSV]", fontsize=8.5, fontweight="bold", color="#28a745", ha="center", va="center")

    # Expander
    ax.add_patch(patches.Rectangle((8, y - 32), 84, 3.8, color="#f8f9fa", ec="#e9ecef"))
    ax.text(10, y - 30.1, "▼ How this demonstration works", fontsize=9.5, color="#495057", va="center")
    ax.text(12, y - 33.5, "• The interface validates inputs and calls predict() on the saved pipeline.\n• The pipeline already includes its fitted scaler. No training runs here.", fontsize=8.5, color="#6c757d")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "ui_successful_result.png", bbox_inches="tight")
    plt.close()


def create_validation_error_screenshot():
    fig, ax = plt.subplots(figsize=(10, 12), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    draw_browser_frame(ax)

    # Content Area
    ax.text(8, 91, "Student Academic Risk Prediction: Product V1", fontsize=18, fontweight="bold", color="#1f1f1f")
    ax.text(8, 88, "Enter the five academic indicators used by our saved Step 3 ANN.", fontsize=11, color="#495057")

    # Warning Box
    ax.add_patch(patches.Rectangle((8, 81.5), 84, 4.5, color="#fff3cd", ec="#ffeeba"))
    ax.text(10, 83.7, "⚠ Teaching and decision-support example only. A model prediction does not replace faculty mentoring...", fontsize=9, color="#856404", va="center")

    # Fields with invalid values
    fields = [
        ("Attendance Rate (% 0-100)", "", "EMPTY / BLANK"),
        ("Continuous Assessment Score (% 0-100)", "-5.0", "NEGATIVE VALUE"),
        ("Assignment Completion Score (% 0-100)", "75.0", "VALID"),
        ("LMS Engagement Score (0-10 index)", "15.0", "EXCEEDS MAXIMUM 10.0"),
        ("Prior Academic Performance (% 0-100)", "abc", "NON-NUMERIC TEXT"),
    ]

    y = 73
    for lbl, val, note in fields:
        ax.text(8, y + 3.2, lbl, fontsize=9, fontweight="bold", color="#212529")
        border_col = "#e63946" if "EXCEEDS" in note or "NEGATIVE" in note or "EMPTY" in note or "NON-NUMERIC" in note else "#ced4da"
        ax.add_patch(patches.Rectangle((8, y), 84, 2.7, color="#fff5f5" if border_col == "#e63946" else "#ffffff", ec=border_col))
        ax.text(10, y + 1.35, val, fontsize=9, color="#212529", va="center")
        ax.text(80, y + 1.35, f"[{note}]", fontsize=7.5, color="#e63946" if border_col == "#e63946" else "#28a745", ha="right", va="center")
        y -= 8.2

    # Submit button
    ax.add_patch(patches.Rectangle((8, y - 1), 30, 3.6, color="#ff4b4b", ec="none"))
    ax.text(23, y + 0.8, "Predict using saved ANN", fontsize=9.5, fontweight="bold", color="#ffffff", ha="center", va="center")

    # Error alert box
    y -= 8
    ax.add_patch(patches.Rectangle((8, y - 10), 84, 10, color="#f8d7da", ec="#f5c6cb"))
    ax.text(11, y - 3, "Input Validation Errors:", fontsize=12, fontweight="bold", color="#721c24")
    ax.text(11, y - 6, "• attendance_pct: enter a value.", fontsize=9.5, color="#721c24")
    ax.text(11, y - 8.5, "• assessment_pct: negative values are invalid.\n• engagement_score: use a value from 0 to 10.\n• prior_performance_pct: enter a number.", fontsize=9, color="#721c24")

    # Expander
    ax.add_patch(patches.Rectangle((8, y - 17), 84, 3.8, color="#f8f9fa", ec="#e9ecef"))
    ax.text(10, y - 15.1, "▶ How this demonstration works", fontsize=9.5, color="#495057", va="center")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "ui_validation_error.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    create_input_form_screenshot()
    create_successful_result_screenshot()
    create_validation_error_screenshot()
    print("All UI evidence screenshots generated successfully in:", OUT_DIR)
