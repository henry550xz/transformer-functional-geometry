#!/usr/bin/env python3
"""Regenerate all README figures from the sanitized public results snapshot."""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def overview() -> None:
    stages = [
        ("Predictive KV", "killed", "#9a9a9a"),
        ("Latent block dynamics", "killed", "#9a9a9a"),
        ("Global functional space", "killed", "#9a9a9a"),
        ("Local functional geometry", "supported", "#2878b5"),
        ("Reusable local regimes", "killed", "#9a9a9a"),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    for index, (name, outcome, color) in enumerate(stages):
        x = 0.08 + index * 0.21
        ax.text(x, 0.63, name, ha="center", va="center", fontsize=10.5, color="#20252b",
                bbox=dict(boxstyle="round,pad=0.55", facecolor="#f7f8fa", edgecolor=color, linewidth=1.8))
        ax.text(x, 0.39, outcome.upper(), ha="center", color=color, fontsize=9, weight="bold")
        if index < len(stages) - 1:
            ax.annotate("", xy=(x + 0.15, 0.63), xytext=(x + 0.07, 0.63), arrowprops=dict(arrowstyle="->", color="#66717d"))
    ax.text(0.5, 0.10, "Surviving observation: functional geometry is locally concentrated, but context-dependent.", ha="center", fontsize=12, weight="bold", color="#174b73")
    fig.tight_layout()
    fig.savefig(ASSETS / "overview.png", dpi=220, transparent=False)
    fig.savefig(ASSETS / "overview.pdf", bbox_inches="tight")
    fig.savefig(ASSETS / "overview.svg", bbox_inches="tight")
    plt.close(fig)


def geometry_and_dictionary() -> None:
    data = json.loads((ROOT / "results/local_spectrum_summary.json").read_text())
    x = np.array(data["dimensions"])
    median = np.array(data["phase1_local_median_cumulative"])
    low = np.array(data["phase1_local_q25_cumulative"])
    high = np.array(data["phase1_local_q75_cumulative"])
    aggregate = np.array(data["matched_aggregate_cumulative"])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.fill_between(x, low, high, color="#69a9d0", alpha=0.23, label="Local IQR (12 calibration tokens)")
    ax.plot(x, median, color="#176ca4", linewidth=2.2, label="Median pointwise Fisher")
    ax.plot(x, aggregate, color="#d17a22", linewidth=2.2, label="Matched H=32 aggregate")
    ax.axhline(0.95, color="0.35", linestyle="--", linewidth=1)
    ax.annotate("local median r95≈20.5", xy=(21, .95), xytext=(52, .77), arrowprops=dict(arrowstyle="->", color="#176ca4"), color="#176ca4")
    ax.annotate("aggregate r95≈107", xy=(107, .95), xytext=(135, .86), arrowprops=dict(arrowstyle="->", color="#d17a22"), color="#a85f18")
    ax.set(xlim=(1, 256), ylim=(0, 1.02), xlabel="Retained hidden-state directions", ylabel="Cumulative estimated functional energy", title="Pointwise geometry is more concentrated than matched aggregate geometry")
    ax.grid(alpha=.2); ax.legend(loc="lower right", fontsize=9); fig.tight_layout()
    fig.savefig(ASSETS / "local_vs_global_geometry.png", dpi=220)
    fig.savefig(ASSETS / "local_vs_global_geometry.pdf")
    plt.close(fig)

    grid = [row for row in data["grid_validation"] if row["q"] == 128]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot([r["K"] for r in grid], [r["dictionary_median"] for r in grid], marker="o", linewidth=2, color="#176ca4", label="Prototype dictionary")
    ax.plot([r["K"] for r in grid], [r["global_median"] for r in grid], marker="s", linewidth=2, color="#d17a22", label="One global q-space")
    ax.axhline(grid[0]["random_expected"], color="0.5", linestyle=":", label="Random q-space expectation")
    ax.set(xlabel="Number of prototypes K", ylabel="Validation median functional-energy capture", title="More prototypes barely improve held-out capture", xticks=[1, 2, 4, 8, 16], ylim=(0.12, .30))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(ASSETS / "dictionary_recurrence.png", dpi=220)
    fig.savefig(ASSETS / "dictionary_recurrence.pdf")
    plt.close(fig)


def timeline() -> None:
    labels = ["Temporal KV AR", "2D token×depth", "Predictive transform", "Latent linear", "Global functional space", "Prehabilitation", "Local low rank", "Reusable regimes"]
    supported = [False, False, False, False, False, False, True, False]
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    y = np.arange(len(labels))[::-1]
    colors = ["#2778b5" if value else "#a1a6aa" for value in supported]
    ax.scatter(np.zeros(len(labels)), y, s=120, c=colors, zorder=3)
    ax.plot(np.zeros(len(labels)), y, color="#c5c9cc", zorder=1)
    for yi, label, value in zip(y, labels, supported):
        ax.text(.05, yi, label, va="center", fontsize=10)
        ax.text(-.05, yi, "SUPPORTED" if value else "KILLED", va="center", ha="right", color="#2778b5" if value else "#70767b", fontsize=9, weight="bold")
    ax.set(xlim=(-.55, .75), ylim=(-.7, len(labels)-.3), title="A sequence of predeclared falsification gates")
    ax.axis("off"); fig.tight_layout(); fig.savefig(ASSETS / "hypothesis_timeline.png", dpi=220); fig.savefig(ASSETS / "hypothesis_timeline.pdf"); plt.close(fig)


if __name__ == "__main__":
    overview(); geometry_and_dictionary(); timeline(); print(f"Wrote figures to {ASSETS}")

