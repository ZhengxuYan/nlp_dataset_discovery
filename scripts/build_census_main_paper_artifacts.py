#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / ".matplotlib-cache")))
os.environ.setdefault("XDG_CACHE_HOME", str((Path(__file__).resolve().parents[1] / ".cache")))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


COL = {
    "blue": "#4C78A8",
    "teal": "#72B7B2",
    "orange": "#F58518",
    "green": "#54A24B",
    "gray": "#777777",
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

REL_LABELS = {
    "source/reuse": "Source / reuse",
    "comparison/evaluation prior": "Comparison / evaluation prior",
    "no prior mention": "No prior mention",
    "inspired/shared task": "Inspired / shared task",
    "knowledge/lexical resource": "Knowledge / lexical resource",
    "other": "Other",
}


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
    })


def read_csv(csv_dir: Path, name: str) -> pd.DataFrame:
    path = csv_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in {path.suffix, ".png", ".pdf"}:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def pct(value: float, digits: int = 0) -> str:
    return f"{100 * float(value):.{digits}f}%"


def latex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, tgt in replacements.items():
        value = value.replace(src, tgt)
    return value


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    *,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
    column_spec: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.1f}" if "pp" in col else f"{val:.1f}")
            else:
                vals.append(latex_escape(val))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.1f}" if col.endswith("_pct") or col.endswith("_pp") else f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_figure3(dcu_type: pd.DataFrame, out: Path, *, min_n: int = 100) -> pd.DataFrame:
    df = dcu_type.copy()
    df = df[(df["query_dcu_type"] != "other") & (df["n_dcus"] >= min_n)].copy()
    df["type_label"] = df["query_dcu_type"].map(TYPE_LABELS).fillna(df["query_dcu_type"])
    df["not_covered_share"] = df["not_covered"] / df["not_covered"].sum()

    rate_df = df.sort_values("not_covered_rate", ascending=True).reset_index(drop=True)
    share_df = df.sort_values("not_covered_share", ascending=True).reset_index(drop=True)

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.95),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.62},
    )

    y = np.arange(len(rate_df))
    ax0.barh(y, rate_df["not_covered_rate"], color=COL["orange"], height=0.62)
    ax0.set_yticks(y)
    ax0.set_yticklabels(rate_df["type_label"])
    ax0.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax0.set_xlim(0, max(0.45, float(rate_df["not_covered_rate"].max()) + 0.06))
    ax0.grid(axis="x", color=COL["lightgray"], lw=0.7)
    ax0.set_title("A. No-prior-match rate", loc="left", fontweight="bold")
    ax0.set_xlabel("Share of query DCUs with no prior-evidence match")
    for i, row in rate_df.iterrows():
        ax0.text(row["not_covered_rate"] + 0.012, i, pct(row["not_covered_rate"]), va="center", fontsize=6.8)

    y = np.arange(len(share_df))
    ax1.barh(y, share_df["not_covered_share"], color=COL["blue"], height=0.62)
    ax1.set_yticks(y)
    ax1.set_yticklabels(share_df["type_label"])
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.set_xlim(0, max(0.60, float(share_df["not_covered_share"].max()) + 0.08))
    ax1.grid(axis="x", color=COL["lightgray"], lw=0.7)
    ax1.set_title("B. Share of all no-prior-match claims", loc="left", fontweight="bold")
    ax1.set_xlabel("Among no-prior-match DCUs")
    for i, row in share_df.iterrows():
        ax1.text(row["not_covered_share"] + 0.01, i, pct(row["not_covered_share"]), va="center", fontsize=6.8)

    fig.suptitle("Prior-evidence matches by contribution type", y=1.03, fontsize=10.2, fontweight="bold")
    save_figure(fig, out / "figure3_dimension_evidence_gap.png")
    summary = df[["query_dcu_type", "type_label", "n_dcus", "not_covered", "not_covered_rate", "not_covered_share"]].copy()
    return summary.sort_values("not_covered_share", ascending=False)


def build_figure4(bank_llm: pd.DataFrame, construction_methods: pd.DataFrame, out: Path) -> pd.DataFrame:
    by = bank_llm.sort_values("bank_year").copy()
    methods = construction_methods.copy()
    keep = ["human annotated", "translation/multilingual", "benchmark aggregation", "LLM/synthetic"]
    methods = methods[methods["construction_method"].isin(keep)].copy()
    methods["method_label"] = methods["construction_method"].map(METHOD_LABELS).fillna(methods["construction_method"])
    order = {m: i for i, m in enumerate(["LLM/synthetic", "benchmark aggregation", "translation/multilingual", "human annotated"])}
    methods["order"] = methods["construction_method"].map(order)
    methods = methods.sort_values("not_covered_rate", ascending=True).reset_index(drop=True)

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.85),
        gridspec_kw={"width_ratios": [0.9, 1.15], "wspace": 0.55},
    )

    x = np.arange(len(by))
    ax0.plot(x, by["llm_share"], marker="o", color=COL["green"], lw=2)
    ax0.fill_between(x, 0, by["llm_share"], color=COL["green"], alpha=0.12)
    ax0.set_xticks(x)
    ax0.set_xticklabels([str(int(year)) for year in by["bank_year"]])
    ax0.set_ylim(0, max(0.72, float(by["llm_share"].max()) + 0.08))
    ax0.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax0.set_title("A. LLM-assisted construction by year", loc="left", fontweight="bold")
    ax0.set_ylabel("Share of dataset records")
    ax0.grid(axis="y", color=COL["lightgray"], lw=0.7)
    for i, val in enumerate(by["llm_share"]):
        ax0.text(i, val + 0.02, pct(val, 1), ha="center", fontsize=7)

    y = np.arange(len(methods))
    colors = [COL["green"] if m == "LLM/synthetic" else COL["orange"] for m in methods["construction_method"]]
    ax1.barh(y, methods["not_covered_rate"], color=colors, height=0.62)
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods["method_label"])
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.set_xlim(0, max(0.25, float(methods["not_covered_rate"].max()) + 0.04))
    ax1.grid(axis="x", color=COL["lightgray"], lw=0.7)
    ax1.set_title("B. No-prior-match rate by construction method", loc="left", fontweight="bold")
    ax1.set_xlabel("Share of attributed DCUs with no prior-evidence match")
    for i, row in methods.iterrows():
        ax1.text(row["not_covered_rate"] + 0.006, i, f"{pct(row['not_covered_rate'], 1)}  N={int(row['n_records']):,}", va="center", fontsize=6.8)

    fig.suptitle("LLM-assisted construction and prior-evidence matches", y=1.03, fontsize=10.0, fontweight="bold")
    save_figure(fig, out / "figure4_llm_construction_gap.png")
    return methods[["construction_method", "method_label", "n_records", "n_dcus", "not_covered_rate"]].copy()


def build_relationship_table(relationship: pd.DataFrame, out: Path) -> pd.DataFrame:
    keep = ["source/reuse", "comparison/evaluation prior", "no prior mention", "inspired/shared task"]
    table = relationship[relationship["relationship_group"].isin(keep)].copy()
    order = {name: i for i, name in enumerate(keep)}
    table["order"] = table["relationship_group"].map(order)
    table = table.sort_values("order")
    table["relationship"] = table["relationship_group"].map(REL_LABELS).fillna(table["relationship_group"])
    table["records"] = table["n_records"].astype(int)
    table["mentions"] = table["n_mentions"].astype(int)
    table["llm_share_pct"] = 100 * table["llm_share"]
    table["not_covered_pct"] = 100 * table["not_covered_rate"]
    table["dominant_gap"] = table["dominant_not_covered_type"].map(TYPE_LABELS).fillna(table["dominant_not_covered_type"])
    out_table = table[["relationship", "records", "mentions", "llm_share_pct", "not_covered_pct", "dominant_gap"]].copy()
    out_table.to_csv(out / "main_table_prior_relationship.csv", index=False)
    latex_table = out_table.copy()
    latex_table["llm_share_pct"] = latex_table["llm_share_pct"].map(lambda v: f"{v:.1f}%")
    latex_table["not_covered_pct"] = latex_table["not_covered_pct"].map(lambda v: f"{v:.1f}%")
    write_latex_table(
        latex_table,
        out / "main_table_prior_relationship.tex",
        columns=["relationship", "records", "mentions", "llm_share_pct", "not_covered_pct", "dominant_gap"],
        headers=["Prior relationship", "Records", "Mentions", "LLM share", "No-prior-match rate", "Dominant gap"],
        caption=(
            "Prior-dataset relationship diagnostics. Records may have multiple relationship types. "
            "No-prior-match rates are computed over adequacy-passing query DCUs associated with each relationship group."
        ),
        label="tab:prior_relationships",
        column_spec="lrrrrl",
    )
    return out_table


def compact_appendix_tables(csv_dir: Path, out: Path) -> dict[str, pd.DataFrame]:
    appendix_dir = out / "appendix_tables"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}

    table_specs = {
        "appendix_primary_use_resource": (
            ["primary_use_group_profile.csv", "resource_group_profile.csv"],
            "Primary-use and resource-type diagnostics.",
        ),
        "appendix_language_scale": (
            ["language_group_profile.csv", "language_group_scale_dcus.csv", "size_bin_scale_dcus.csv"],
            "Language and scale diagnostics.",
        ),
        "appendix_task_domain": (
            ["top_task_profile.csv", "top_domain_profile.csv"],
            "Top task and domain diagnostics.",
        ),
        "appendix_documentation": (
            ["doc_feature_pass_delta.csv", "doc_score_profile.csv"],
            "Documentation and governance diagnostics.",
        ),
        "appendix_year_source": (
            ["year_source_profile.csv"],
            "Year and corpus-source diagnostics.",
        ),
        "appendix_relationship_full": (
            ["relationship_type_profile.csv", "relationship_group_x_type.csv"],
            "Full prior-relationship diagnostics.",
        ),
    }

    md_sections = ["# Appendix Census Diagnostic Tables", ""]
    for section, (files, title) in table_specs.items():
        md_sections.extend([f"## {title}", ""])
        for filename in files:
            df = read_csv(csv_dir, filename)
            path = appendix_dir / filename
            df.to_csv(path, index=False)
            outputs[filename] = df
            show = df.head(30)
            md_sections.extend([f"### `{filename}`", "", markdown_table(show, list(show.columns)), ""])
    (out / "appendix_census_tables.md").write_text("\n".join(md_sections), encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="data/census/census_scale_analysis/dimension_summaries")
    parser.add_argument("--out-dir", default="data/census/census_scale_analysis/main_paper_artifacts")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    fig3 = build_figure3(read_csv(csv_dir, "dcu_type_profile.csv"), out)
    fig4 = build_figure4(
        read_csv(csv_dir, "bank_llm_by_year_2023_2025.csv"),
        read_csv(csv_dir, "construction_method_profile.csv"),
        out,
    )
    rel_table = build_relationship_table(read_csv(csv_dir, "relationship_group_profile.csv"), out)
    compact_appendix_tables(csv_dir, out)

    fig3.to_csv(out / "figure3_dimension_evidence_gap_data.csv", index=False)
    fig4.to_csv(out / "figure4_llm_construction_gap_data.csv", index=False)

    summary = [
        "# Census Main-Paper Artifact Summary",
        "",
        "## Main Figure 3",
        "- File: `figure3_dimension_evidence_gap.png` / `.pdf`",
        "- Claim: Scale and coverage claims dominate the evidence gap.",
        markdown_table(fig3, ["query_dcu_type", "n_dcus", "not_covered", "not_covered_rate", "not_covered_share"]),
        "",
        "## Main Figure 4",
        "- File: `figure4_llm_construction_gap.png` / `.pdf`",
        "- Claim: LLM-assisted construction is growing, but is not the construction mode with the largest evidence gap.",
        markdown_table(fig4, ["construction_method", "n_records", "n_dcus", "not_covered_rate"]),
        "",
        "## Main Table Candidate",
        "- File: `main_table_prior_relationship.tex` / `.csv`",
        "- Claim: Most records explicitly reuse or compare against prior datasets; across relationship groups, the dominant evidence gap remains scale/coverage.",
        markdown_table(rel_table, ["relationship", "records", "mentions", "llm_share_pct", "not_covered_pct", "dominant_gap"]),
        "",
        "## Appendix",
        "- File: `appendix_census_tables.md`",
        "- Directory: `appendix_tables/`",
    ]
    (out / "artifact_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote census main-paper artifacts to {out}")


if __name__ == "__main__":
    main()
