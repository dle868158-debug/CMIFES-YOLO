"""
Generate Figure 7 (training curves) for the paper from Ultralytics `results.csv`.

Input:
  - paper_v7/figures/data/cmife_yolo_v7_results.csv

Outputs:
  - paper_v7/figures/fig_training_curves.pdf
  - paper_v7/figures/fig_training_curves_preview.png
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _style_axes(ax, xmin: int, xmax: int) -> None:
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([1, 50, 100, 150, 200, 250, 300])


def main() -> None:
    root = Path(__file__).resolve().parent
    csv_path = root / "data" / "cmife_yolo_v7_results.csv"
    out_pdf = root / "fig_training_curves.pdf"
    out_png = root / "fig_training_curves_preview.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    epochs = df["epoch"].astype(int)
    xmin, xmax = int(epochs.min()), int(epochs.max())

    # Keep the look paper-friendly (serif, subtle grid, vector-friendly PDF text).
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    blue = "#2563EB"
    orange = "#DD6B20"
    green = "#2F855A"
    teal = "#0F766E"
    purple = "#7C3AED"
    magenta = "#C026D3"

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.0), constrained_layout=True)

    # (a) Box loss
    ax = axes[0, 0]
    ax.plot(epochs, df["train/box_loss"], color=blue, linewidth=1.4, label="Train")
    ax.plot(
        epochs, df["val/box_loss"], color=orange, linewidth=1.2, linestyle="--", label="Val"
    )
    ax.set_title("(a) Box Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    _style_axes(ax, xmin, xmax)
    ax.legend(frameon=False, loc="upper right")

    # (b) Cls loss
    ax = axes[0, 1]
    ax.plot(epochs, df["train/cls_loss"], color=blue, linewidth=1.4, label="Train")
    ax.plot(
        epochs, df["val/cls_loss"], color=orange, linewidth=1.2, linestyle="--", label="Val"
    )
    ax.set_title("(b) Classification Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    _style_axes(ax, xmin, xmax)
    ax.legend(frameon=False, loc="upper right")

    # (c) DFL loss
    ax = axes[0, 2]
    ax.plot(epochs, df["train/dfl_loss"], color=blue, linewidth=1.4, label="Train")
    ax.plot(
        epochs, df["val/dfl_loss"], color=orange, linewidth=1.2, linestyle="--", label="Val"
    )
    ax.set_title("(c) DFL Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    _style_axes(ax, xmin, xmax)
    ax.legend(frameon=False, loc="upper right")

    # (d) Precision & Recall
    ax = axes[1, 0]
    precision = df["metrics/precision(B)"] * 100.0
    recall = df["metrics/recall(B)"] * 100.0
    ax.plot(epochs, precision, color=green, linewidth=1.4, label="Precision")
    ax.plot(epochs, recall, color=teal, linewidth=1.4, label="Recall")
    ax.set_title("(d) Precision and Recall")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Epoch")
    _style_axes(ax, xmin, xmax)
    ax.set_ylim(0, max(60, float(precision.max()) + 5))
    ax.legend(frameon=False, loc="lower right")

    # (e) mAP@0.5
    ax = axes[1, 1]
    map50 = df["metrics/mAP50(B)"] * 100.0
    ax.plot(epochs, map50, color=purple, linewidth=1.6, label="mAP@0.5")
    ax.set_title("(e) mAP@0.5")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Epoch")
    _style_axes(ax, xmin, xmax)
    ax.set_ylim(0, max(50, float(map50.max()) + 5))
    ax.legend(frameon=False, loc="lower right")
    ax.annotate(
        f"{map50.iloc[-1]:.2f}%",
        xy=(epochs.iloc[-1], map50.iloc[-1]),
        xytext=(-32, -4),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8,
        color=purple,
    )

    # (f) mAP@0.5:0.95
    ax = axes[1, 2]
    map5095 = df["metrics/mAP50-95(B)"] * 100.0
    ax.plot(epochs, map5095, color=magenta, linewidth=1.6, label="mAP@0.5:0.95")
    ax.set_title("(f) mAP@0.5:0.95")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Epoch")
    _style_axes(ax, xmin, xmax)
    ax.set_ylim(0, max(35, float(map5095.max()) + 5))
    ax.legend(frameon=False, loc="lower right")
    ax.annotate(
        f"{map5095.iloc[-1]:.2f}%",
        xy=(epochs.iloc[-1], map5095.iloc[-1]),
        xytext=(-32, -4),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8,
        color=magenta,
    )

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()

