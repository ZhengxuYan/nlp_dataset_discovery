#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


OUT = Path("publish/acl-style-files-master/figures")

COL = {
    "blue": "#4C78A8",
    "blue_light": "#B9CAD8",
    "orange": "#F58518",
    "green": "#54A24B",
    "green_dark": "#3F7F45",
    "gray": "#777777",
    "lightgray": "#E6E6E6",
    "text": "#222222",
}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.55,
        "xtick.color": COL["text"],
        "ytick.color": COL["text"],
        "figure.dpi": 200,
        "savefig.dpi": 300,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"{name}{suffix}", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def pct(value: float, digits: int = 0) -> str:
    return f"{100 * value:.{digits}f}%"


def contribution_direction_figure() -> None:
    labels = [
        "Scale/coverage",
        "Annotation protocol",
        "Data/source",
        "Task/domain",
        "Evaluation use",
        "Availability/quality",
        "Governance/ethics",
    ]
    n_dcus = [9615, 8469, 7174, 6105, 3683, 2442, 202]
    total = sum(n_dcus)
    dcu_share = [n / total for n in n_dcus]
    no_prior = [0.396, 0.056, 0.105, 0.131, 0.128, 0.197, 0.183]

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(6.9, 2.55),
        gridspec_kw={"width_ratios": [1.03, 1.0], "wspace": 0.54},
    )
    y = list(range(len(labels)))
    colors = [COL["orange"] if lab == "Scale/coverage" else COL["blue_light"] for lab in labels]

    ax0.barh(y, dcu_share, color=colors, height=0.58, edgecolor="none")
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.invert_yaxis()
    ax0.set_xlim(0, 0.30)
    ax0.set_xticks([0, 0.10, 0.20, 0.30])
    ax0.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax0.grid(axis="x", color=COL["lightgray"], lw=0.6)
    ax0.set_axisbelow(True)
    ax0.set_xlabel("Share of scored DCUs")
    ax0.tick_params(axis="y", length=0)
    for i, val in enumerate(dcu_share):
        ax0.text(val + 0.006, i, pct(val), va="center", fontsize=6.4, color=COL["text"])

    ax1.barh(y, no_prior, color=colors, height=0.58, edgecolor="none")
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.45)
    ax1.set_xticks([0, 0.15, 0.30, 0.45])
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.grid(axis="x", color=COL["lightgray"], lw=0.6)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("No-prior-match rate")
    ax1.tick_params(axis="y", length=0)
    for i, val in enumerate(no_prior):
        ax1.text(val + 0.008, i, pct(val), va="center", fontsize=6.4, color=COL["text"])

    save(fig, "figure3_contribution_directions")


def llm_contribution_figure() -> None:
    years = [2023, 2024, 2025]
    llm_share = [0.360, 0.494, 0.621]
    methods = [
        "Human annotated",
        "Translation / multilingual",
        "Benchmark aggregation",
        "LLM / synthetic",
    ]
    no_prior = [0.222, 0.200, 0.199, 0.168]
    ns = [1549, 613, 1678, 7062]

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.55),
        gridspec_kw={"width_ratios": [0.82, 1.18], "wspace": 0.58},
    )

    x = list(range(len(years)))
    ax0.bar(x, llm_share, color=COL["green_dark"], width=0.56, edgecolor="none", alpha=0.86)
    ax0.set_xticks(x)
    ax0.set_xticklabels([str(y) for y in years])
    ax0.set_xlim(-0.28, 2.10)
    ax0.set_ylim(0, 0.72)
    ax0.set_yticks([0, 0.20, 0.40, 0.60])
    ax0.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax0.grid(axis="y", color=COL["lightgray"], lw=0.6)
    ax0.set_axisbelow(True)
    ax0.set_ylabel("Share of dataset records")
    for xi, val in zip(x, llm_share):
        ax0.text(xi, val + 0.025, pct(val, 1), ha="center", va="bottom", fontsize=6.5, color=COL["text"])

    y = list(range(len(methods)))
    colors = [COL["blue_light"], COL["blue_light"], COL["blue_light"], COL["green_dark"]]
    ax1.barh(y, no_prior, color=colors, height=0.58, edgecolor="none")
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.25)
    ax1.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.grid(axis="x", color=COL["lightgray"], lw=0.6)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("No-prior-match rate")
    ax1.tick_params(axis="y", length=0)
    for i, (val, n) in enumerate(zip(no_prior, ns)):
        ax1.text(val + 0.005, i - 0.10, pct(val, 1), va="center", fontsize=6.4, color=COL["text"])
        ax1.text(val + 0.005, i + 0.13, f"N={n:,}", va="center", fontsize=5.8, color=COL["gray"])

    save(fig, "figure4_llm_contribution")


def main() -> None:
    setup()
    contribution_direction_figure()
    llm_contribution_figure()


if __name__ == "__main__":
    main()
