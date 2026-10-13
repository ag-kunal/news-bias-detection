#!/usr/bin/env python3
"""Generate a publication-ready Executive Results Dashboard for Readability Experiments."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "readability" / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "readability" / "outputs" / "plots"

# Set stylish typography and aesthetics
sns.set_theme(style="white", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def create_results_dashboard() -> None:
    # 1. Load Data
    results_path = RESULTS_DIR / "readability_results.csv"
    report_path = RESULTS_DIR / "readability_classification_report.csv"
    cm_path = RESULTS_DIR / "confusion_matrices" / "Combined_Dataset_confusion_matrix.csv"

    if not (results_path.exists() and report_path.exists() and cm_path.exists()):
        raise FileNotFoundError("Missing baseline result CSVs in readability/outputs/results/")

    results_df = pd.read_csv(results_path)
    report_df = pd.read_csv(report_path)
    cm_df = pd.read_csv(cm_path, index_col=0)

    # Figure Canvas: 2x2 Grid with sleek dimensions
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.patch.set_facecolor("#fafbfc")

    # -------------------------------------------------------------
    # PANEL 1: Overall Performance Metrics
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#ffffff")
    metrics = ["Accuracy", "Balanced Acc.", "Macro-F1", "Weighted-F1"]
    values = [
        results_df["accuracy"].iloc[0] * 100,
        results_df["balanced_accuracy"].iloc[0] * 100,
        results_df["macro_f1"].iloc[0] * 100,
        results_df["weighted_f1"].iloc[0] * 100,
    ]
    colors = ["#2b5c8f", "#3c8dbc", "#41b6c4", "#7fcdbb"]

    bars = ax1.bar(metrics, values, color=colors, width=0.55, edgecolor="#1f3b5c", linewidth=1.2, zorder=3)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax1.set_ylim(0, 60)
    ax1.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax1.set_title("A. Overall Baseline Classification Performance\n(Logistic Regression, N=6,900 Test Samples)", 
                  fontsize=13, fontweight="bold", pad=12, loc="left", color="#1a252f")

    # Add data labels
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1a252f")

    # Add benchmark baseline reference (chance level for 3 balanced classes = 33.3%)
    ax1.axhline(33.33, color="#e74c3c", linestyle=":", linewidth=1.8, label="Random Chance Baseline (33.3%)", zorder=4)
    ax1.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=10)

    # -------------------------------------------------------------
    # PANEL 2: Per-Class Precision, Recall, and F1 Breakdown
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#ffffff")
    classes = ["Left", "Center", "Right"]
    x = np.arange(len(classes))
    bar_width = 0.25

    # Match class orders
    report_dict = report_df.set_index("class").to_dict(orient="index")
    precision = [report_dict[c.lower()]["precision"] * 100 for c in classes]
    recall = [report_dict[c.lower()]["recall"] * 100 for c in classes]
    f1 = [report_dict[c.lower()]["f1"] * 100 for c in classes]

    b1 = ax2.bar(x - bar_width, precision, width=bar_width, label="Precision", color="#4575b4", edgecolor="#224269", zorder=3)
    b2 = ax2.bar(x, recall, width=bar_width, label="Recall", color="#74add1", edgecolor="#396a8a", zorder=3)
    b3 = ax2.bar(x + bar_width, f1, width=bar_width, label="F1-Score", color="#fdae61", edgecolor="#a85d13", zorder=3)

    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 75)
    ax2.set_ylabel("Metric Score (%)", fontsize=12, fontweight="bold")
    ax2.set_title("B. Per-Class Granular Performance Breakdown\n(Precision vs. Recall vs. F1-Score)", 
                  fontsize=13, fontweight="bold", pad=12, loc="left", color="#1a252f")
    ax2.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=10)

    for b_group in [b1, b2, b3]:
        for bar in b_group:
            h = bar.get_height()
            ax2.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # -------------------------------------------------------------
    # PANEL 3: Normalized Confusion Matrix Heatmap
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#ffffff")
    cm_matrix = cm_df.values
    cm_row_sums = cm_matrix.sum(axis=1, keepdims=True)
    cm_norm = (cm_matrix / cm_row_sums) * 100.0

    class_labels = ["Left", "Center", "Right"]
    annot = []
    for r in range(3):
        row_annot = []
        for c in range(3):
            count = cm_matrix[r, c]
            pct = cm_norm[r, c]
            row_annot.append(f"{count:,}\n({pct:.1f}%)")
        annot.append(row_annot)

    sns.heatmap(
        cm_norm,
        annot=np.array(annot),
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax3,
        linewidths=1.5,
        linecolor="#ffffff",
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    ax3.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=8)
    ax3.set_ylabel("Actual Label", fontsize=12, fontweight="bold", labelpad=8)
    ax3.set_title("C. Confusion Matrix (Counts & Class Recall %)\n(Evaluation on Stratified Holdout Test Set)", 
                  fontsize=13, fontweight="bold", pad=12, loc="left", color="#1a252f")

    # -------------------------------------------------------------
    # PANEL 4: Distinctive Stylistic Features Across Leanings
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#ffffff")

    # Showcase key stylistic differentiators found in analysis
    features_show = [
        "Headline Char Length",
        "Average Sentence Length (words)",
        "Complex Words Rate (% × 2)",
        "Flesch Reading Ease (/2)",
    ]
    center_vals = [60.34, 24.05, 17.58 * 2, 41.86 / 2]
    left_vals = [61.71, 24.73, 17.25 * 2, 42.29 / 2]
    right_vals = [66.87, 24.17, 17.30 * 2, 42.72 / 2]

    y_pos = np.arange(len(features_show))
    h_width = 0.25

    ax4.barh(y_pos - h_width, left_vals, height=h_width, label="Left", color="#2b83ba", edgecolor="#1a557a", zorder=3)
    ax4.barh(y_pos, center_vals, height=h_width, label="Center", color="#78c679", edgecolor="#418042", zorder=3)
    ax4.barh(y_pos + h_width, right_vals, height=h_width, label="Right", color="#d7191c", edgecolor="#871012", zorder=3)

    ax4.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(features_show, fontsize=11, fontweight="bold")
    ax4.set_xlabel("Normalized Metric Value", fontsize=12, fontweight="bold")
    ax4.set_xlim(0, 80)
    ax4.set_title("D. Stylistic Readability Comparison Across Bias\n(Key Linguistic Markers & Headline Differences)", 
                  fontsize=13, fontweight="bold", pad=12, loc="left", color="#1a252f")
    ax4.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=10)

    # Key takeaway text box placed cleanly in upper-left without overlapping bars
    takeaway_text = (
        "Key Research Insights:\n"
        "• Body prose is consistent (~42 FRE, college reading level across all bias classes).\n"
        "• Right articles feature +10.8% longer headlines (66.9 vs 60.3 characters).\n"
        "• Readability provides an orthogonal baseline (40.3% Macro-F1 > 33.3% chance)."
    )
    ax4.text(
        0.02, 0.28, takeaway_text,
        transform=ax4.transAxes,
        fontsize=9.5,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9db", edgecolor="#e6c229", alpha=0.95),
    )

    # Overall Header & Footer
    fig.suptitle(
        "NEWS BIAS DETECTION — READABILITY FEATURE EXPERIMENT RESULTS",
        fontsize=18,
        fontweight="bold",
        color="#0f2027",
        y=0.985,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    # Save to both results and plots directories
    out1 = RESULTS_DIR / "readability_results_dashboard.png"
    out2 = PLOTS_DIR / "00_results_executive_dashboard.png"

    plt.savefig(out1, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(out2, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Executive Results Dashboard generated successfully:")
    print(f" - {out1}")
    print(f" - {out2}")


if __name__ == "__main__":
    create_results_dashboard()
