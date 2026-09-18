#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / ".matplotlib-cache")))
os.environ.setdefault("XDG_CACHE_HOME", str((Path(__file__).resolve().parents[1] / ".cache")))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from build_census_dimension_summary import bank_llm_assisted  # noqa: E402


COL = {
    "blue": "#4C78A8",
    "blue_dark": "#3F6F9F",
    "blue_light": "#B9CAD8",
    "blue_gray": "#7D91A3",
    "orange": "#F58518",
    "green": "#54A24B",
    "green_dark": "#5A9367",
    "green_light": "#C9DCCB",
    "purple": "#B279A2",
    "teal": "#72B7B2",
    "gray": "#777777",
    "axis": "#4A4A4A",
    "lightgray": "#E5E5E5",
    "text": "#222222",
}

TYPE_LABELS = {
    "task/domain": "Task/domain",
    "data/source": "Data/source",
    "annotation/protocol": "Annotation protocol",
    "scale/coverage": "Scale/coverage",
    "evaluation/use": "Evaluation use",
    "availability/quality": "Availability/quality",
    "governance/ethics": "Governance/ethics",
    "other": "Other",
}

METHOD_LABELS = {
    "LLM/synthetic": "LLM / synthetic",
    "human annotated": "Human annotated",
    "translation/multilingual": "Translation / multilingual",
    "benchmark aggregation": "Benchmark aggregation",
    "derived/filtered": "Derived / filtered",
    "newly collected": "Newly collected",
    "unknown/not reported": "Unknown / not reported",
}

IMPORTANCE_WEIGHTS = {"low": 0.5, "medium": 1.0, "high": 1.5}
LABEL_VALUES = {"covered": 0.0, "partially_covered": 0.5, "not_covered": 1.0}


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.4,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COL["axis"],
        "axes.linewidth": 0.55,
        "xtick.color": COL["text"],
        "ytick.color": COL["text"],
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.dpi": 200,
        "savefig.dpi": 300,
    })


def pct(value: float, digits: int = 0) -> str:
    return f"{100 * float(value):.{digits}f}%"


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in {path.suffix, ".png", ".pdf"}:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 1000, seed: int = 13) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def included_records(record_level: pd.DataFrame) -> pd.DataFrame:
    records = record_level[record_level["included"].astype(bool)].copy()
    records = records[records["added_information_score"].notna()].copy()
    return records


def included_scored_dcus(dcu_level: pd.DataFrame) -> pd.DataFrame:
    dcus = dcu_level[dcu_level["included"].astype(bool)].copy()
    dcus["claim_score"] = dcus["label"].map(LABEL_VALUES)
    dcus["importance_weight"] = dcus["importance"].map(IMPORTANCE_WEIGHTS).fillna(1.0)
    dcus = dcus[dcus["claim_score"].notna()].copy()
    dcus["score_mass"] = dcus["claim_score"] * dcus["importance_weight"]
    return dcus


def build_score_landscape(records: pd.DataFrame, out: Path) -> pd.DataFrame:
    scores = records["added_information_score"].astype(float)
    by_year = (
        records.groupby("year")["added_information_score"]
        .agg(records="count", mean_score="mean", median_score="median")
        .reset_index()
        .sort_values("year")
    )
    by_year[["ci_low", "ci_high"]] = by_year["year"].apply(
        lambda y: pd.Series(bootstrap_ci(records.loc[records["year"].eq(y), "added_information_score"].to_numpy()))
    )

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.75),
        gridspec_kw={"width_ratios": [1.2, 0.9], "wspace": 0.42},
    )

    bins = np.linspace(0, 1, 21)
    ax0.hist(scores, bins=bins, color=COL["blue"], alpha=0.88, edgecolor="white", linewidth=0.5)
    median = float(scores.median())
    mean = float(scores.mean())
    ax0.axvline(median, color=COL["orange"], lw=1.5, label=f"Median {median:.2f}")
    ax0.axvline(mean, color=COL["text"], lw=1.1, ls="--", label=f"Mean {mean:.2f}")
    ax0.set_title("A. Score distribution", loc="left", fontweight="bold")
    ax0.set_xlabel("Added-information score")
    ax0.set_ylabel("Dataset records")
    ax0.set_xlim(0, 1)
    ax0.grid(axis="y", color=COL["lightgray"], lw=0.7)
    ax0.legend(frameon=False, loc="upper left")

    x = np.arange(len(by_year))
    yerr = np.vstack([
        by_year["mean_score"].to_numpy() - by_year["ci_low"].to_numpy(),
        by_year["ci_high"].to_numpy() - by_year["mean_score"].to_numpy(),
    ])
    ax1.bar(x, by_year["mean_score"], yerr=yerr, color=COL["green"], alpha=0.88, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(year)) for year in by_year["year"]])
    ax1.set_ylim(0, 0.75)
    ax1.set_title("B. Mean score by year", loc="left", fontweight="bold")
    ax1.set_xlabel("Query year")
    ax1.set_ylabel("Mean added-information score")
    ax1.grid(axis="y", color=COL["lightgray"], lw=0.7)
    for i, row in by_year.iterrows():
        ax1.text(x[i], row["mean_score"] + 0.035, f"{row['mean_score']:.2f}", ha="center", fontsize=7)

    fig.suptitle("Dataset-level added-information scores", y=1.03, fontsize=10.0, fontweight="bold")
    save_figure(fig, out / "figure3_score_landscape.png")

    landscape = records[["bank_id", "paper_id", "dataset_name", "year", "source_corpus", "added_information_score", "n_query_dcus"]].copy()
    landscape.to_csv(out / "score_landscape.csv", index=False)
    by_year.to_csv(out / "score_by_year.csv", index=False)
    return by_year


def build_score_decomposition(dcus: pd.DataFrame, out: Path, *, min_n: int = 100) -> pd.DataFrame:
    agg = (
        dcus.groupby("query_dcu_type")
        .agg(
            n_dcus=("label", "size"),
            score_mass=("score_mass", "sum"),
            weight_mass=("importance_weight", "sum"),
            mean_claim_score=("claim_score", "mean"),
        )
        .reset_index()
    )
    agg = agg[(agg["query_dcu_type"] != "other") & (agg["n_dcus"] >= min_n)].copy()
    agg["score_mass_share"] = agg["score_mass"] / agg["score_mass"].sum()
    agg["weighted_mean_claim_score"] = agg["score_mass"] / agg["weight_mass"]
    agg["type_label"] = agg["query_dcu_type"].map(TYPE_LABELS).fillna(agg["query_dcu_type"])

    plot_df = agg.sort_values("score_mass_share", ascending=False).reset_index(drop=True)

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(6.6, 2.45),
        gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.52},
    )
    y = np.arange(len(plot_df))
    bar_colors = [COL["blue_dark"] if t == "scale/coverage" else COL["blue_light"] for t in plot_df["query_dcu_type"]]
    ax0.barh(y, plot_df["score_mass_share"], color=bar_colors, height=0.55, edgecolor="none")
    ax0.set_yticks(y)
    ax0.set_yticklabels(plot_df["type_label"])
    ax0.invert_yaxis()
    ax0.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax0.set_xlim(0, 0.35)
    ax0.set_xticks([0, 0.10, 0.20, 0.30])
    ax0.grid(axis="x", color="#EDEDED", lw=0.55)
    ax0.set_axisbelow(True)
    ax0.set_title("A. Share of total score mass", loc="left", fontweight="bold")
    ax0.set_xlabel("Share of total added-information score")
    ax0.tick_params(axis="y", length=0)
    for i, row in plot_df.iterrows():
        ax0.text(row["score_mass_share"] + 0.006, i, pct(row["score_mass_share"]), va="center", fontsize=6.6, color=COL["text"])

    ax1.barh(y, plot_df["weighted_mean_claim_score"], color=bar_colors, height=0.55, edgecolor="none")
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.75)
    ax1.set_xticks([0, 0.25, 0.50, 0.75])
    ax1.grid(axis="x", color="#EDEDED", lw=0.55)
    ax1.set_axisbelow(True)
    ax1.set_title("B. Mean claim score", loc="left", fontweight="bold")
    ax1.set_xlabel("Weighted mean claim score")
    ax1.tick_params(axis="y", length=0)
    for i, row in plot_df.iterrows():
        ax1.text(row["weighted_mean_claim_score"] + 0.012, i, f"{row['weighted_mean_claim_score']:.2f}", va="center", fontsize=6.5, color=COL["text"])

    fig.subplots_adjust(top=0.90, bottom=0.20, left=0.20, right=0.985)
    save_figure(fig, out / "figure4_score_decomposition.png")
    out_df = agg[[
        "query_dcu_type",
        "type_label",
        "n_dcus",
        "score_mass",
        "score_mass_share",
        "weighted_mean_claim_score",
        "mean_claim_score",
    ]].sort_values("score_mass_share", ascending=False)
    out_df.to_csv(out / "score_decomposition_by_type.csv", index=False)
    return out_df


def build_llm_by_year(bank_jsonl: Path) -> pd.DataFrame:
    rows = []
    for row in read_jsonl(bank_jsonl):
        year = row.get("year")
        if year not in {2023, 2024, 2025}:
            continue
        rows.append({
            "bank_year": int(year),
            "llm_assisted": bank_llm_assisted(row),
        })
    data = pd.DataFrame(rows)
    if data.empty:
        return pd.DataFrame(columns=["bank_year", "records", "llm_records", "llm_share"])
    return (
        data.groupby("bank_year")
        .agg(records=("llm_assisted", "size"), llm_records=("llm_assisted", "sum"), llm_share=("llm_assisted", "mean"))
        .reset_index()
        .sort_values("bank_year")
    )


def build_llm_score(records: pd.DataFrame, bank_llm_by_year: pd.DataFrame, out: Path) -> pd.DataFrame:
    keep = ["human annotated", "translation/multilingual", "benchmark aggregation", "LLM/synthetic"]
    methods = records[records["construction_method"].isin(keep)].copy()
    rows = []
    for method, group in methods.groupby("construction_method"):
        values = group["added_information_score"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(values)
        rows.append({
            "construction_method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "n_records": int(len(group)),
            "mean_score": float(np.mean(values)),
            "median_score": float(np.median(values)),
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    score_by_method = pd.DataFrame(rows)
    method_order = ["LLM/synthetic", "human annotated", "benchmark aggregation", "translation/multilingual"]
    score_by_method["method_order"] = score_by_method["construction_method"].map({m: i for i, m in enumerate(method_order)})
    score_by_method = score_by_method.sort_values("method_order").drop(columns=["method_order"]).reset_index(drop=True)

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(6.7, 2.45),
        gridspec_kw={"width_ratios": [0.72, 1.28], "wspace": 0.72},
    )

    years = bank_llm_by_year["bank_year"].to_numpy(dtype=int)
    vals = bank_llm_by_year["llm_share"].to_numpy(dtype=float)
    xpos = np.arange(len(years))
    ax0.bar(xpos, vals, color=COL["green_dark"], width=0.56, edgecolor="none")
    for x, val in zip(xpos, vals):
        ax0.text(x, val + 0.026, pct(val, 0), ha="center", va="bottom", fontsize=6.7, color=COL["text"])
    ax0.set_xticks(xpos)
    ax0.set_xticklabels([str(y) for y in years])
    ax0.set_ylim(0, 0.78)
    ax0.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax0.set_ylabel("Share of dataset records")
    ax0.grid(axis="y", color="#EDEDED", lw=0.55)
    ax0.set_axisbelow(True)

    y = np.arange(len(score_by_method))
    xerr = np.vstack([
        score_by_method["mean_score"].to_numpy() - score_by_method["ci_low"].to_numpy(),
        score_by_method["ci_high"].to_numpy() - score_by_method["mean_score"].to_numpy(),
    ])
    llm_mask = score_by_method["construction_method"].eq("LLM/synthetic")
    colors = [COL["green_dark"] if is_llm else COL["blue_light"] for is_llm in llm_mask]
    ax1.barh(y, score_by_method["mean_score"], color=colors, height=0.56, edgecolor="none", zorder=2)
    ax1.errorbar(
        score_by_method["mean_score"],
        y,
        xerr=xerr,
        fmt="none",
        ecolor=COL["axis"],
        elinewidth=0.75,
        capsize=2.0,
        capthick=0.75,
        zorder=3,
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    for i, row in score_by_method.iterrows():
        ax1.text(
            0.4975,
            i - 0.12,
            row["method_label"],
            ha="right",
            va="center",
            fontsize=6.7,
            color=COL["text"],
            clip_on=False,
        )
        ax1.text(
            0.4975,
            i + 0.13,
            f"N={int(row['n_records']):,}",
            ha="right",
            va="center",
            fontsize=5.8,
            color=COL["gray"],
            clip_on=False,
        )
    ax1.invert_yaxis()
    ax1.set_xlim(0.50, 0.62)
    ax1.set_xticks([0.50, 0.54, 0.58, 0.62])
    ax1.axvline(records["added_information_score"].mean(), color="#9A9A9A", lw=0.65, ls=(0, (3, 3)), zorder=1)
    ax1.grid(axis="x", color="#EDEDED", lw=0.55)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("Mean evidence-conditioned added-information score")
    ax1.tick_params(axis="y", length=0)
    for i, row in score_by_method.iterrows():
        ax1.text(
            row["mean_score"] + 0.004,
            i,
            f"{row['mean_score']:.3f}",
            va="center",
            fontsize=6.4,
            color=COL["text"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.92},
        )

    fig.subplots_adjust(top=0.90, bottom=0.20, left=0.085, right=0.985)
    save_figure(fig, out / "figure5_llm_score.png")
    score_by_method.to_csv(out / "score_by_construction.csv", index=False)
    bank_llm_by_year.to_csv(out / "llm_by_year_for_score_figure.csv", index=False)
    return score_by_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="data/census/census_scale_analysis")
    parser.add_argument("--dimension-dir", default="data/census/census_scale_analysis/dimension_summaries")
    parser.add_argument("--dataset-bank-jsonl", default="data/census/integrated_fulltext_dataset_bank_2023_2025.jsonl")
    parser.add_argument(
        "--extraction-jsonl",
        action="append",
        default=[
            "data/census/fulltext_dataset_extractions_pdf_all.jsonl",
            "data/census/arxiv_fulltext_dataset_extractions_2023_2025.jsonl",
        ],
    )
    parser.add_argument("--out-dir", default="data/census/census_scale_analysis/score_artifacts")
    args = parser.parse_args()

    setup_matplotlib()
    analysis_dir = Path(args.analysis_dir)
    dimension_dir = Path(args.dimension_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = included_records(pd.read_csv(analysis_dir / "record_level.csv"))
    dcus = included_scored_dcus(pd.read_csv(analysis_dir / "dcu_level.csv"))
    bank_llm_by_year = build_llm_by_year(Path(args.dataset_bank_jsonl))

    by_year = build_score_landscape(records, out)
    decomp = build_score_decomposition(dcus, out)
    construction = build_llm_score(records, bank_llm_by_year, out)

    summary = [
        "# Census Score Artifact Summary",
        "",
        f"- Adequacy-passing records: {len(records):,}",
        f"- Adequacy-passing scored DCUs: {len(dcus):,}",
        f"- Mean score: {records['added_information_score'].mean():.3f}",
        f"- Median score: {records['added_information_score'].median():.3f}",
        "",
        "## Score by year",
        by_year.to_markdown(index=False),
        "",
        "## Score decomposition by type",
        decomp.to_markdown(index=False),
        "",
        "## Score by construction method",
        construction.to_markdown(index=False),
        "",
        "## LLM-assisted share by year",
        bank_llm_by_year.to_markdown(index=False),
        "",
    ]
    (out / "score_artifact_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Wrote census score artifacts to {out}")


if __name__ == "__main__":
    main()
