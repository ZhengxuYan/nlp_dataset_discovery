#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

Path("artifacts/.mplconfig").mkdir(parents=True, exist_ok=True)
Path("artifacts/.cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("artifacts/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


YEAR_KEYS = ["2023", "2024", "2025"]
METHOD_ORDER = ["lexical", "dense", "fusion", "gpt_5_4_listwise_rerank"]
METRIC_ORDER = ["mrr", "recall@1", "recall@3", "recall@5", "recall@10"]
SUPPORT_ORDER = ["unsupported", "partially_supported", "supported", "not_comparable", "contradicted"]
ADEQUACY_ORDER = ["high", "medium", "low"]
RISK_ORDER = ["low", "medium", "high"]
DELTA_ORDER = [
    "scale/coverage",
    "annotation/protocol",
    "task/domain",
    "data/source",
    "evaluation/use",
    "availability/quality",
    "governance/ethics",
    "other",
]

METHOD_LABELS = {
    "lexical": "Lexical",
    "dense": "Dense",
    "fusion": "Fusion",
    "gpt_5_4_listwise_rerank": "GPT-5.4 rerank",
}
GOVERNANCE_LABELS = {
    "ethics_discussed": "Ethics",
    "pii_discussed": "PII",
    "consent_discussed": "Consent",
    "copyright_discussed": "Copyright",
    "bias_or_fairness_discussed": "Bias/fairness",
}
ROLE_LABELS = {
    "introduced_dataset": "Introduced",
    "training_data": "Training",
    "benchmark": "Benchmark",
    "evaluation_set": "Evaluation",
    "shared_task_dataset": "Shared task",
    "knowledge_resource": "Knowledge",
    "lexical_resource": "Lexical",
    "pretraining_corpus": "Pretraining",
    "instruction_tuning_data": "Instruction",
    "other": "Other",
}
PRIMARY_USE_LABELS = {
    "benchmarking": "Benchmarking",
    "evaluation": "Evaluation",
    "training": "Training",
    "fine_tuning": "Fine-tuning",
    "instruction_tuning": "Instruction tuning",
    "analysis": "Analysis",
    "pretraining": "Pretraining",
}
SECTOR_LABELS = {
    "academic_only": "Academic",
    "academic_industry_collab": "Acad.+industry",
    "industry_only": "Industry",
}
COLOR = {
    "blue": "#3B6EA8",
    "teal": "#2A9D8F",
    "amber": "#E9A23B",
    "orange": "#D95F02",
    "red": "#B23A48",
    "green": "#4C956C",
    "gray": "#6C757D",
    "light_gray": "#E9ECEF",
    "ink": "#1F2933",
}
SUPPORT_COLORS = {
    "unsupported": COLOR["orange"],
    "partially_supported": COLOR["amber"],
    "supported": COLOR["green"],
    "not_comparable": COLOR["gray"],
    "contradicted": COLOR["red"],
}
ADEQUACY_COLORS = {
    "high": COLOR["blue"],
    "medium": COLOR["teal"],
    "low": COLOR["amber"],
}
RISK_COLORS = {
    "low": COLOR["blue"],
    "medium": COLOR["teal"],
    "high": COLOR["amber"],
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = str(value)
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
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_table(headers: list[str], rows: list[list[Any]], alignment: str) -> str:
    lines = [
        "\\begin{tabular}{" + alignment + "}",
        "\\toprule",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(x) for x in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_tex(path: Path, headers: list[str], rows: list[list[Any]], alignment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex_table(headers, rows, alignment), encoding="utf-8")


def fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def fmt_float(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, name: str, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"type": "figure", "name": name, "png": str(png), "pdf": str(pdf)})


def load_added_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def read_dataset_bank(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def read_paper_sector_map(path: str | Path) -> dict[str, str]:
    sectors: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return sectors
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            profile = row.get("institution_profile") or {}
            sectors[row.get("paper_id") or ""] = profile.get("paper_sector") or "unknown"
    return sectors


def value_counts(rows: list[dict[str, Any]], key: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                if item not in (None, ""):
                    counter[str(item)] += 1
        elif value not in (None, ""):
            counter[str(value)] += 1
    return counter


def year_rows(rows: list[dict[str, Any]], year: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("year")) == year]


def year_share(rows: list[dict[str, Any]], year: str, predicate) -> float:
    subset = year_rows(rows, year)
    return 100.0 * sum(1 for row in subset if predicate(row)) / max(len(subset), 1)


def pct_by_year_counter(rows: list[dict[str, Any]], year: str, key: str, values: set[str]) -> float:
    subset = year_rows(rows, year)
    return 100.0 * sum(1 for row in subset if str(row.get(key)) in values) / max(len(subset), 1)


def has_explicit_license(row: dict[str, Any]) -> bool:
    return str(row.get("license") or "").strip().lower() not in {"", "unclear", "unknown", "none", "n/a"}


def has_external_docs(row: dict[str, Any]) -> bool:
    return row.get("documentation_type") in {"website", "datasheet", "data_statement", "data_card", "dataset_card"}


def has_clear_quality_control(row: dict[str, Any]) -> bool:
    value = str(row.get("quality_control") or "").strip().lower()
    if value in {"", "unclear", "unknown", "none", "n/a", "not specified"}:
        return False
    if any(marker in value for marker in ["none mentioned", "not explicitly", "not specified", "no quality control", "not described"]):
        return False
    return True


def artifact_present(row: dict[str, Any]) -> bool:
    artifacts = row.get("artifacts") or {}
    return any(bool(artifacts.get(key)) for key in ["dataset_urls", "github_repos", "huggingface_ids", "code_urls", "project_page_urls", "zenodo_urls", "osf_urls", "kaggle_urls"])


def normalize_model_name(name: str) -> str:
    text = name.strip()
    lower = text.lower()
    if "gpt-4o-mini" in lower or "gpt-4o mini" in lower:
        return "GPT-4o-mini"
    if "gpt-4o" in lower:
        return "GPT-4o"
    if "gpt-4v" in lower:
        return "GPT-4V"
    if "gpt-4" in lower:
        return "GPT-4"
    if "chatgpt" in lower:
        return "ChatGPT"
    if "gpt-3.5" in lower or "gpt3.5" in lower:
        return "GPT-3.5"
    if "claude" in lower:
        return "Claude"
    if "llama" in lower:
        return "Llama"
    if "gemini" in lower:
        return "Gemini"
    if "deepseek" in lower:
        return "DeepSeek"
    if "qwen" in lower:
        return "Qwen"
    if "mistral" in lower:
        return "Mistral"
    return text


def plot_pipeline(fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    stages = [
        ("ACL full-text\npapers", "10,079 papers\n2023-2025"),
        ("Dataset/resource\ncensus", "10,422 resources\n36,726 ACUs"),
        ("ACU bank", "Atomic claims over\ncoverage, source,\nannotation, use"),
        ("Prior candidate\nretrieval", "Explicit matches +\nTF-IDF candidate pool"),
        ("LLM ACU\nattribution", "Support labels and\nimportance weights"),
        ("Added-information\nprofile", "Score + prior-set\nadequacy/risk"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 2.35))
    ax.set_axis_off()
    x_positions = np.linspace(0.04, 0.86, len(stages))
    width = 0.13
    height = 0.56
    for idx, ((title, subtitle), x) in enumerate(zip(stages, x_positions)):
        box = FancyBboxPatch(
            (x, 0.24),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=COLOR["blue"] if idx in {0, len(stages) - 1} else COLOR["gray"],
            facecolor="#F8FAFC",
        )
        ax.add_patch(box)
        ax.text(x + width / 2, 0.63, title, ha="center", va="center", fontsize=8.8, weight="bold", color=COLOR["ink"])
        ax.text(x + width / 2, 0.39, subtitle, ha="center", va="center", fontsize=7.5, color="#334155")
        if idx < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.008, 0.52),
                (x_positions[idx + 1] - 0.008, 0.52),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color=COLOR["gray"],
            )
            ax.add_patch(arrow)
    ax.text(0.5, 0.08, "Output is conditional on the observed prior-support set; adequacy is reported separately.", ha="center", fontsize=8, color=COLOR["gray"])
    save_figure(fig, "figure1_pipeline_overview", fig_dir, manifest)


def count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 5:
        return "2-5"
    return ">5"


def acu_type_counts(rows: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        for acu in row.get("acus") or []:
            counts[acu.get("type") or "other"] += 1
    return counts


def plot_growth_composition(census: dict[str, Any], bank_rows: list[dict[str, Any]], fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.55, 5.75))
    x = np.arange(len(YEAR_KEYS))

    ax = axes[0, 0]
    lineage_specs = [
        ("Claimed new", lambda row: row.get("is_new_dataset") is True, COLOR["gray"]),
        ("Prior mentions", lambda row: bool(row.get("prior_dataset_mentions")), COLOR["teal"]),
        ("Named source datasets", lambda row: bool(row.get("source_datasets")), COLOR["blue"]),
    ]
    for label, predicate, color in lineage_specs:
        values = [year_share(bank_rows, year, predicate) for year in YEAR_KEYS]
        ax.plot(x, values, marker="o", linewidth=1.7, color=color, label=label)
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_ylim(60, 102)
    ax.set_title("Novelty claims and lineage")
    ax.set_ylabel("% of datasets")
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    ax = axes[0, 1]
    mention_buckets = ["0", "1", "2-5", ">5"]
    bucket_colors = ["#E5E7EB", COLOR["blue"], COLOR["teal"], COLOR["orange"]]
    bottom = np.zeros(len(YEAR_KEYS))
    for bucket, color in zip(mention_buckets, bucket_colors):
        values = []
        for year in YEAR_KEYS:
            subset = year_rows(bank_rows, year)
            values.append(100 * sum(count_bucket(len(row.get("prior_dataset_mentions") or [])) == bucket for row in subset) / max(len(subset), 1))
        values = np.array(values)
        ax.bar(x, values, bottom=bottom, color=color, label=bucket)
        bottom += values
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_ylim(0, 100)
    ax.set_title("Prior dataset mentions per resource")
    ax.set_ylabel("% of datasets")
    ax.legend(title="mentions", frameon=False, fontsize=6.8, title_fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    ax = axes[1, 0]
    source_buckets = ["0", "1", "2-5", ">5"]
    bottom = np.zeros(len(YEAR_KEYS))
    for bucket, color in zip(source_buckets, bucket_colors):
        values = []
        for year in YEAR_KEYS:
            subset = year_rows(bank_rows, year)
            values.append(100 * sum(count_bucket(len(row.get("source_datasets") or [])) == bucket for row in subset) / max(len(subset), 1))
        values = np.array(values)
        ax.bar(x, values, bottom=bottom, color=color, label=bucket)
        bottom += values
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_ylim(0, 100)
    ax.set_title("Named source datasets per resource")
    ax.set_ylabel("% of datasets")
    ax.legend(title="sources", frameon=False, fontsize=6.8, title_fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    ax = axes[1, 1]
    acu_types = [
        ("scale/coverage", "Scale/coverage", COLOR["orange"]),
        ("annotation/protocol", "Annotation", COLOR["blue"]),
        ("data/source", "Data/source", COLOR["teal"]),
        ("task/domain", "Task/domain", COLOR["amber"]),
        ("evaluation/use", "Evaluation", COLOR["gray"]),
    ]
    bottom = np.zeros(len(YEAR_KEYS))
    yearly_counts = {year: acu_type_counts(year_rows(bank_rows, year)) for year in YEAR_KEYS}
    yearly_totals = np.array([sum(yearly_counts[year].values()) or 1 for year in YEAR_KEYS], dtype=float)
    for key, label, color in acu_types:
        values = np.array([yearly_counts[year].get(key, 0) for year in YEAR_KEYS], dtype=float) / yearly_totals * 100
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_ylim(0, 100)
    ax.set_title("Claim type composition")
    ax.set_ylabel("% of ACUs")
    ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.subplots_adjust(hspace=0.5, wspace=0.28)
    save_figure(fig, "figure2_growth_composition", fig_dir, manifest)


def plot_primary_use_shifts(census: dict[str, Any], bank_rows: list[dict[str, Any]], fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    x = np.arange(len(YEAR_KEYS))
    primary_keys = [
        ("benchmarking", "Benchmarking", COLOR["blue"]),
        ("evaluation", "Evaluation", COLOR["teal"]),
        ("training", "Training", COLOR["amber"]),
        ("fine_tuning", "Fine-tuning", COLOR["orange"]),
    ]
    for key, label, color in primary_keys:
        values = [year_share(bank_rows, y, lambda row, key=key: row.get("primary_use") == key) for y in YEAR_KEYS]
        ax.plot(x, values, marker="o", linewidth=1.6, color=color, label=label)
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_title("Primary use shifts")
    ax.set_ylabel("% of datasets")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, "appendix_primary_use_shifts", fig_dir, manifest)


def plot_llm_construction(census: dict[str, Any], bank_rows: list[dict[str, Any]], fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.65, 5.85))
    x = np.arange(len(YEAR_KEYS))

    ax = axes[0, 0]
    synth = census.get("synthetic_generation_by_year") or {}
    synth_true = np.array([(synth.get(year) or {}).get("True", 0) for year in YEAR_KEYS])
    synth_false = np.array([(synth.get(year) or {}).get("False", 0) for year in YEAR_KEYS])
    totals = synth_true + synth_false
    ax.bar(x, synth_false, color="#CBD5E1", label="Not LLM-generated")
    ax.bar(x, synth_true, bottom=synth_false, color=COLOR["orange"], label="LLM-generated")
    for i, value in enumerate(synth_true):
        pct_value = 100 * value / max(totals[i], 1)
        ax.text(i, synth_false[i] + value / 2, f"{fmt_int(value)}\n({pct_value:.0f}%)", ha="center", va="center", fontsize=7.2, color="white", weight="bold")
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_title("Rise of LLM-generated datasets")
    ax.set_ylabel("Datasets")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[0, 1]
    status_order = [("yes", "Human verified", COLOR["green"]), ("partial", "Partially verified", COLOR["teal"]), ("no", "No verification", COLOR["orange"]), ("unclear", "Unclear", COLOR["gray"])]
    bottom = np.zeros(len(YEAR_KEYS))
    for key, label, color in status_order:
        values = []
        for year in YEAR_KEYS:
            subset = [row for row in year_rows(bank_rows, year) if row.get("uses_llm_synthetic_generation") is True]
            values.append(100 * sum(row.get("synthetic_human_verification") == key for row in subset) / max(len(subset), 1))
        values = np.array(values)
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_ylim(0, 100)
    ax.set_title("Verification of LLM-generated data")
    ax.set_ylabel("% of LLM-generated datasets")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))

    ax = axes[1, 0]
    model_counts: Counter = Counter()
    for row in bank_rows:
        for name in row.get("synthetic_model_names") or []:
            cleaned = normalize_model_name(str(name))
            if cleaned and cleaned.lower() not in {"unclear", "unknown", "llm"}:
                model_counts[cleaned] += 1
    top_models = model_counts.most_common(10)
    labels = [name for name, _ in top_models][::-1]
    values = [count for _, count in top_models][::-1]
    ax.barh(labels, values, color=COLOR["blue"])
    ax.set_title("Most reported generator models")
    ax.set_xlabel("Mentions")
    ax.tick_params(axis="y", labelsize=6.6)

    ax = axes[1, 1]
    groups = [
        ("LLM-generated", [row for row in bank_rows if row.get("uses_llm_synthetic_generation") is True], COLOR["orange"]),
        ("Not LLM-generated", [row for row in bank_rows if row.get("uses_llm_synthetic_generation") is not True], COLOR["gray"]),
    ]
    metrics = [
        ("QC described", has_clear_quality_control),
        ("PII discussed", lambda row: (row.get("governance") or {}).get("pii_discussed") == "yes"),
        ("Bias discussed", lambda row: (row.get("governance") or {}).get("bias_or_fairness_discussed") == "yes"),
        ("License explicit", has_explicit_license),
    ]
    y = np.arange(len(metrics))
    height = 0.34
    for offset, (group_label, subset, color) in zip([-height / 2, height / 2], groups):
        values = [100 * sum(predicate(row) for row in subset) / max(len(subset), 1) for _, predicate in metrics]
        ax.barh(y + offset, values, height=height, color=color, label=group_label)
    ax.set_yticks(y, [label for label, _ in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_title("Disclosure and quality-control signals")
    ax.set_xlabel("% of datasets")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.subplots_adjust(hspace=0.58, wspace=0.34)
    save_figure(fig, "figure3_release_governance", fig_dir, manifest)


def sector_order_present(bank_rows: list[dict[str, Any]], paper_sector: dict[str, str]) -> list[str]:
    order = ["academic_only", "academic_industry_collab", "industry_only"]
    present = {paper_sector.get(row.get("paper_id") or "") for row in bank_rows}
    return [sector for sector in order if sector in present]


def score_rows_with_bank(rows: list[dict[str, Any]], bank_by_id: dict[str, dict[str, Any]], paper_sector: dict[str, str]) -> pd.DataFrame:
    data = []
    for row in rows:
        if not has_adequate_prior_set(row):
            continue
        bank_row = bank_by_id.get(row.get("query_bank_id") or "")
        if not bank_row:
            continue
        score = (row.get("profile") or {}).get("added_information_score")
        if score is None:
            continue
        sector = paper_sector.get(bank_row.get("paper_id") or "", "unknown")
        if sector not in SECTOR_LABELS:
            continue
        data.append({"sector": SECTOR_LABELS[sector], "score": float(score)})
    return pd.DataFrame(data)


def plot_release_governance(
    census: dict[str, Any],
    bank_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
    bank_by_id: dict[str, dict[str, Any]],
    paper_sector: dict[str, str],
    fig_dir: Path,
    manifest: list[dict[str, str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.7))
    sectors = sector_order_present(bank_rows, paper_sector)
    sector_labels = [SECTOR_LABELS[sector] for sector in sectors]
    sector_rows = {
        sector: [row for row in bank_rows if paper_sector.get(row.get("paper_id") or "") == sector]
        for sector in sectors
    }
    palette = [COLOR["blue"], COLOR["teal"], COLOR["orange"]]

    ax = axes[0, 0]
    size_data = []
    for sector in sectors:
        for row in sector_rows[sector]:
            value = (row.get("scale") or {}).get("num_instances")
            if isinstance(value, (int, float)) and value > 0:
                size_data.append({"sector": SECTOR_LABELS[sector], "log_instances": np.log10(float(value)), "instances": float(value)})
    size_df = pd.DataFrame(size_data)
    sns.boxplot(data=size_df, x="sector", y="log_instances", order=sector_labels, ax=ax, palette=dict(zip(sector_labels, palette)), showfliers=False, width=0.58)
    ax.set_title("Dataset size by sector")
    ax.set_xlabel("")
    ax.set_ylabel("log10 instances")
    for i, sector in enumerate(sector_labels):
        vals = size_df.loc[size_df["sector"] == sector, "instances"].tolist()
        if vals:
            ax.text(i, ax.get_ylim()[0] + 0.05, f"med={fmt_int(np.median(vals))}", ha="center", va="bottom", fontsize=6.5)

    ax = axes[0, 1]
    width = 0.28
    x = np.arange(len(sectors))
    lang_values = []
    domain_values = []
    for sector in sectors:
        rows_for_sector = sector_rows[sector]
        lang_values.append(100 * sum(((row.get("scale") or {}).get("num_languages") or 0) > 1 for row in rows_for_sector) / max(len(rows_for_sector), 1))
        domain_values.append(100 * sum(((row.get("scale") or {}).get("num_domains") or 0) > 1 for row in rows_for_sector) / max(len(rows_for_sector), 1))
    ax.bar(x - width / 2, lang_values, width=width, color=COLOR["blue"], label=">1 language")
    ax.bar(x + width / 2, domain_values, width=width, color=COLOR["teal"], label=">1 domain")
    ax.set_xticks(x, sector_labels)
    ax.set_ylim(0, max(lang_values + domain_values) * 1.25)
    ax.set_title("Coverage breadth by sector")
    ax.set_ylabel("% of datasets")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    metric_specs = [
        ("LLM synthetic", lambda row: row.get("uses_llm_synthetic_generation") is True, COLOR["orange"]),
        ("Benchmarking", lambda row: row.get("primary_use") == "benchmarking", COLOR["blue"]),
        ("Open release", lambda row: row.get("access_restrictions") == "open", COLOR["green"]),
    ]
    offsets = np.linspace(-0.24, 0.24, len(metric_specs))
    for offset, (label, predicate, color) in zip(offsets, metric_specs):
        values = [100 * sum(predicate(row) for row in sector_rows[sector]) / max(len(sector_rows[sector]), 1) for sector in sectors]
        ax.bar(x + offset, values, width=0.22, color=color, label=label)
    ax.set_xticks(x, sector_labels)
    ax.set_ylim(0, 100)
    ax.set_title("Construction/use/release by sector")
    ax.set_ylabel("% of datasets")
    ax.legend(frameon=False, fontsize=6.8, ncol=1, loc="upper right")

    ax = axes[1, 1]
    score_df = score_rows_with_bank(added_rows, bank_by_id, paper_sector)
    sns.boxplot(data=score_df, x="sector", y="score", order=sector_labels, ax=ax, palette=dict(zip(sector_labels, palette)), showfliers=False, width=0.58)
    sns.stripplot(data=score_df.sample(min(len(score_df), 900), random_state=13), x="sector", y="score", order=sector_labels, ax=ax, color=COLOR["gray"], alpha=0.16, size=1.5)
    ax.set_ylim(-0.02, 1.04)
    ax.set_title("Added-information by sector")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    for i, sector in enumerate(sector_labels):
        vals = score_df.loc[score_df["sector"] == sector, "score"].tolist()
        if vals:
            ax.text(i, 0.02, f"n={len(vals):,}", ha="center", va="bottom", fontsize=6.8)
    fig.tight_layout()
    save_figure(fig, "figure6_sector_scale_score", fig_dir, manifest)


def reusability_category(row: dict[str, Any]) -> str:
    released = row.get("release_status") == "released"
    artifact = artifact_present(row)
    licensed = has_explicit_license(row)
    documented = has_external_docs(row)
    maintained = row.get("maintenance_status") == "maintained"
    if released and artifact and licensed and (documented or maintained):
        return "Strong reusable"
    if released and artifact and not licensed:
        return "Released, no license"
    if released and not artifact:
        return "Released, no URL"
    if row.get("release_status") in {"unclear", "promised", "available_on_request", "partially_released", "not_released"}:
        return "No clear release"
    return "Other"


def plot_reusability_gap(
    bank_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
    bank_by_id: dict[str, dict[str, Any]],
    fig_dir: Path,
    manifest: list[dict[str, str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.65, 5.85))
    x = np.arange(len(YEAR_KEYS))

    ax = axes[0, 0]
    signals = [
        ("Released", lambda row: row.get("release_status") == "released", COLOR["green"]),
        ("Artifact URL", artifact_present, COLOR["blue"]),
        ("License", has_explicit_license, COLOR["orange"]),
        ("External docs", has_external_docs, COLOR["teal"]),
        ("Maintained", lambda row: row.get("maintenance_status") == "maintained", COLOR["gray"]),
    ]
    for label, predicate, color in signals:
        values = [year_share(bank_rows, year, predicate) for year in YEAR_KEYS]
        ax.plot(x, values, marker="o", linewidth=1.7, color=color, label=label)
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_ylim(0, 100)
    ax.set_title("Reusability signals over time")
    ax.set_ylabel("% of datasets")
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))

    ax = axes[0, 1]
    category_order = ["Strong reusable", "Released, no license", "Released, no URL", "No clear release"]
    category_colors = [COLOR["green"], COLOR["orange"], COLOR["blue"], COLOR["gray"]]
    bottom = np.zeros(len(YEAR_KEYS))
    for category, color in zip(category_order, category_colors):
        values = []
        for year in YEAR_KEYS:
            subset = year_rows(bank_rows, year)
            values.append(100 * sum(reusability_category(row) == category for row in subset) / max(len(subset), 1))
        values = np.array(values)
        ax.bar(x, values, bottom=bottom, color=color, label=category)
        bottom += values
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_ylim(0, 100)
    ax.set_title("Release status is not enough")
    ax.set_ylabel("% of datasets")
    ax.legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    ax = axes[1, 0]
    artifact_metrics = [
        ("Dataset URL", lambda row: bool((row.get("artifacts") or {}).get("dataset_urls"))),
        ("GitHub/code", lambda row: bool((row.get("artifacts") or {}).get("github_repos") or (row.get("artifacts") or {}).get("code_urls"))),
        ("HuggingFace", lambda row: bool((row.get("artifacts") or {}).get("huggingface_ids"))),
        ("Project page", lambda row: bool((row.get("artifacts") or {}).get("project_page_urls"))),
    ]
    for label, predicate in artifact_metrics:
        values = [year_share(bank_rows, year, predicate) for year in YEAR_KEYS]
        ax.plot(x, values, marker="o", linewidth=1.6, label=label)
    ax.set_xticks(x, YEAR_KEYS)
    ax.set_ylim(0, max([year_share(bank_rows, year, predicate) for _, predicate in artifact_metrics for year in YEAR_KEYS]) * 1.25)
    ax.set_title("Artifact channels")
    ax.set_ylabel("% of datasets")
    ax.legend(frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    ax = axes[1, 1]
    score_data = []
    for row in added_rows:
        if not has_adequate_prior_set(row):
            continue
        bank_row = bank_by_id.get(row.get("query_bank_id") or "")
        score = (row.get("profile") or {}).get("added_information_score")
        if not bank_row or score is None:
            continue
        category = reusability_category(bank_row)
        if category in category_order:
            score_data.append({"category": category, "score": float(score)})
    score_df = pd.DataFrame(score_data)
    present_order = [category for category in category_order if category in set(score_df["category"])]
    sns.boxplot(data=score_df, y="category", x="score", order=present_order, ax=ax, color="#E9F3F1", showfliers=False, width=0.55)
    ax.set_xlim(-0.02, 1.04)
    ax.set_title("Added-information by reusability")
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    for i, category in enumerate(present_order):
        vals = score_df.loc[score_df["category"] == category, "score"].tolist()
        ax.text(1.02, i, f"n={len(vals):,}", va="center", fontsize=6.5)
    fig.subplots_adjust(hspace=0.58, wspace=0.34)
    save_figure(fig, "figure6_reusability_gap", fig_dir, manifest)


def plot_retrieval_validation(report: dict[str, Any], fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    retrieval = report.get("retrieval") or {}
    data = []
    for method in METHOD_ORDER:
        if method not in retrieval:
            continue
        for metric in METRIC_ORDER:
            data.append({"method": METHOD_LABELS.get(method, method), "metric": metric.upper().replace("RECALL@", "R@"), "value": retrieval[method].get(metric, 0.0)})
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    sns.barplot(data=df, x="metric", y="value", hue="method", ax=ax, palette=[COLOR["gray"], COLOR["blue"], COLOR["teal"], COLOR["orange"]])
    ax.set_ylim(0, 1.0)
    ax.set_title("Prior-support retrieval benchmark")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "figure4_retrieval_validation", fig_dir, manifest)


def added_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float((row.get("profile") or {}).get("added_information_score")) for row in rows if (row.get("profile") or {}).get("added_information_score") is not None]
    support_values: dict[str, list[float]] = defaultdict(list)
    unsupported_by_delta: Counter = Counter()
    adequacy_counts: Counter = Counter()
    risk_counts: Counter = Counter()
    for row in rows:
        profile = row.get("profile") or {}
        assessment = row.get("prior_set_assessment") or {}
        adequacy_counts[assessment.get("prior_set_adequacy") or "unknown"] += 1
        risk_counts[assessment.get("missing_prior_risk") or "unknown"] += 1
        unsupported_by_delta.update(profile.get("unsupported_by_delta_type") or {})
        for status in SUPPORT_ORDER:
            support_values[status].append(float((profile.get("support_percentages") or {}).get(status, 0.0)))
    return {
        "rows": len(rows),
        "mean": safe_mean(scores),
        "median": float(np.median(scores)) if scores else 0.0,
        "full_score_rows": sum(score == 1.0 for score in scores),
        "support": {status: safe_mean(values) for status, values in support_values.items()},
        "unsupported_by_delta_type": dict(unsupported_by_delta),
        "adequacy_counts": dict(adequacy_counts),
        "risk_counts": dict(risk_counts),
    }


def has_adequate_prior_set(row: dict[str, Any]) -> bool:
    return ((row.get("prior_set_assessment") or {}).get("prior_set_adequacy") in {"high", "medium"})


def plot_score_distribution(rows: list[dict[str, Any]], fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    scores = [float((row.get("profile") or {}).get("added_information_score")) for row in rows if (row.get("profile") or {}).get("added_information_score") is not None]
    mean_value = safe_mean(scores)
    median_value = float(np.median(scores))
    full_count = sum(score == 1.0 for score in scores)
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    bins = np.linspace(0, 1, 21)
    ax.hist(scores, bins=bins, color=COLOR["orange"], edgecolor="white")
    ax.axvline(mean_value, color=COLOR["blue"], linestyle="--", linewidth=1.5, label=f"Mean {mean_value:.3f}")
    ax.axvline(median_value, color=COLOR["teal"], linestyle=":", linewidth=1.8, label=f"Median {median_value:.3f}")
    ax.set_title(f"2025 added-information score distribution (n={len(scores):,})")
    ax.set_xlabel("Added-information score")
    ax.set_ylabel("Datasets")
    ax.text(0.98, 0.92, f"Score = 1.0: {full_count:,}", ha="right", va="top", transform=ax.transAxes, fontsize=8, color=COLOR["ink"])
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "figure5_added_information_distribution_2025", fig_dir, manifest)


def plot_filtered_score_distribution(rows: list[dict[str, Any]], bank_by_id: dict[str, dict[str, Any]], fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    filtered_rows = [row for row in rows if has_adequate_prior_set(row)]
    scores = [
        float((row.get("profile") or {}).get("added_information_score"))
        for row in filtered_rows
        if (row.get("profile") or {}).get("added_information_score") is not None
    ]
    mean_value = safe_mean(scores)
    median_value = float(np.median(scores)) if scores else 0.0
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(scores, bins=bins, color=COLOR["teal"], edgecolor="white")
    ax.axvline(mean_value, color=COLOR["blue"], linestyle="--", linewidth=1.4, label=f"Mean {mean_value:.3f}")
    ax.axvline(median_value, color=COLOR["orange"], linestyle=":", linewidth=1.8, label=f"Median {median_value:.3f}")
    ax.set_title(f"Score distribution (n={len(scores):,})")
    ax.set_xlabel("Added-information score")
    ax.set_ylabel("Datasets")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    primary_order = ["benchmarking", "evaluation", "training", "fine_tuning"]
    primary_rows = []
    for row in filtered_rows:
        bank_row = bank_by_id.get(row.get("query_bank_id") or "")
        primary_use = (bank_row or {}).get("primary_use")
        score = (row.get("profile") or {}).get("added_information_score")
        if primary_use in primary_order and score is not None:
            primary_rows.append({"primary_use": PRIMARY_USE_LABELS.get(primary_use, primary_use), "score": float(score)})
    primary_df = pd.DataFrame(primary_rows)
    ax = axes[1]
    label_order = [PRIMARY_USE_LABELS.get(key, key) for key in primary_order if PRIMARY_USE_LABELS.get(key, key) in set(primary_df["primary_use"])]
    sns.boxplot(data=primary_df, y="primary_use", x="score", order=label_order, ax=ax, color="#DDEFEA", showfliers=False, width=0.55)
    sns.stripplot(data=primary_df.sample(min(len(primary_df), 900), random_state=11), y="primary_use", x="score", order=label_order, ax=ax, color=COLOR["teal"], alpha=0.18, size=1.6)
    ax.set_xlim(-0.02, 1.12)
    ax.set_title("By primary use")
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    for i, label in enumerate(label_order):
        vals = primary_df.loc[primary_df["primary_use"] == label, "score"].tolist()
        ax.text(1.02, i, f"n={len(vals):,}", va="center", fontsize=7)
    fig.suptitle("Added-information after prior-coverage filter", y=1.03, fontsize=12)
    fig.tight_layout()
    save_figure(fig, "figure4_added_information_distribution_filtered", fig_dir, manifest)


def plot_score_by_group(rows: list[dict[str, Any]], group_key: str, order: list[str], colors: dict[str, str], title: str, name: str, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    data = []
    for row in rows:
        assessment = row.get("prior_set_assessment") or {}
        profile = row.get("profile") or {}
        score = profile.get("added_information_score")
        if score is None:
            continue
        data.append({"group": assessment.get(group_key) or "unknown", "score": float(score)})
    df = pd.DataFrame(data)
    order = [g for g in order if g in set(df["group"])]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    sns.violinplot(data=df, x="group", y="score", hue="group", order=order, hue_order=order, ax=ax, palette={g: colors[g] for g in order}, inner=None, cut=0, linewidth=0.8, legend=False)
    sns.boxplot(data=df, x="group", y="score", order=order, ax=ax, width=0.20, showcaps=True, boxprops={"facecolor": "white", "alpha": 0.85}, showfliers=False, whiskerprops={"linewidth": 1})
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Added-information score")
    tick_labels = []
    for i, group in enumerate(order):
        values = df.loc[df["group"] == group, "score"].tolist()
        tick_labels.append(f"{group}\nn={len(values):,}\nmean={safe_mean(values):.3f}")
    ax.set_xticks(range(len(order)), tick_labels)
    fig.tight_layout()
    save_figure(fig, name, fig_dir, manifest)


def plot_delta_types(rows: list[dict[str, Any]], bank_by_id: dict[str, dict[str, Any]], fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    filtered_rows = [row for row in rows if has_adequate_prior_set(row)]
    counts: Counter = Counter()
    primary_delta: dict[str, Counter] = defaultdict(Counter)
    primary_order = ["benchmarking", "evaluation", "training", "fine_tuning"]
    for row in filtered_rows:
        bank_row = bank_by_id.get(row.get("query_bank_id") or "")
        primary_use = (bank_row or {}).get("primary_use")
        for item in row.get("attributions") or []:
            if item.get("support_status") != "unsupported":
                continue
            delta = item.get("delta_type") or "other"
            counts[delta] += 1
            if primary_use in primary_order:
                primary_delta[primary_use][delta] += 1
    rows = [(label, counts.get(label, 0)) for label in DELTA_ORDER if counts.get(label, 0)]
    labels, values = zip(*rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 4.0), gridspec_kw={"width_ratios": [1.0, 1.2]})
    ax = axes[0]
    y = np.arange(len(labels))
    ax.barh(y, values, color=COLOR["orange"])
    ax.set_yticks(y, [label.replace("/", " / ") for label in labels])
    ax.invert_yaxis()
    ax.set_title("Total unsupported ACUs")
    ax.set_xlabel("Unsupported ACUs")
    for i, value in enumerate(values):
        ax.text(value + max(values) * 0.01, i, fmt_int(value), va="center", fontsize=7)

    heat_labels = [label for label in DELTA_ORDER if counts.get(label, 0)]
    heat_primary = [primary_use for primary_use in primary_order if primary_delta.get(primary_use)]
    matrix = []
    for delta in heat_labels:
        row = []
        for primary_use in heat_primary:
            total = sum(primary_delta[primary_use].values()) or 1
            row.append(100 * primary_delta[primary_use].get(delta, 0) / total)
        matrix.append(row)
    ax = axes[1]
    sns.heatmap(
        pd.DataFrame(matrix, index=[label.replace("/", " / ") for label in heat_labels], columns=[PRIMARY_USE_LABELS.get(primary_use, primary_use) for primary_use in heat_primary]),
        ax=ax,
        cmap=sns.light_palette(COLOR["orange"], as_cmap=True),
        cbar_kws={"label": "% within primary use"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title("Composition by primary use")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save_figure(fig, "figure7_delta_types", fig_dir, manifest)


def plot_method_schema(fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_axis_off()
    sections = [
        ("ACU support labels", "supported = 0\npartially_supported = 0.5\nunsupported = 1\ncontradicted / not_comparable = excluded"),
        ("Importance weights", "low = 0.5\nmedium = 1.0\nhigh = 1.5"),
        ("Added-information score", "weighted mean of support deltas\nover scored query ACUs"),
        ("Prior-set assessment", "adequacy: high / medium / low\nmissing-prior risk: low / medium / high"),
    ]
    positions = [(0.06, 0.55), (0.55, 0.55), (0.06, 0.13), (0.55, 0.13)]
    for (title, body), (x, y) in zip(sections, positions):
        box = FancyBboxPatch((x, y), 0.38, 0.30, boxstyle="round,pad=0.018,rounding_size=0.015", linewidth=1.0, edgecolor=COLOR["gray"], facecolor="#F8FAFC")
        ax.add_patch(box)
        ax.text(x + 0.02, y + 0.24, title, ha="left", va="center", fontsize=10, weight="bold", color=COLOR["ink"])
        ax.text(x + 0.02, y + 0.12, body, ha="left", va="center", fontsize=8, color="#334155")
    ax.text(0.5, 0.92, "Attribution schema and scoring rule", ha="center", fontsize=12, weight="bold", color=COLOR["ink"])
    ax.text(0.5, 0.04, "Scores are conditional on observed prior-support ACUs; low adequacy is interpreted as high missing-prior risk.", ha="center", fontsize=8, color=COLOR["gray"])
    save_figure(fig, "appendix_a1_method_schema", fig_dir, manifest)


def plot_appendix_census_decomposition(census: dict[str, Any], fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.7))
    x = np.arange(len(YEAR_KEYS))
    specs = [
        ("roles_by_year", "Roles by year", ["introduced_dataset", "training_data", "benchmark", "evaluation_set"]),
        ("paper_sector_by_year", "Paper sector by year", ["academic_only", "academic_industry_collab", "industry_only", "government"]),
        ("release_status_by_year", "Release status by year", ["released", "unclear", "promised", "not_released"]),
        ("artifact_by_year", "Artifact links by year", ["github_or_code", "dataset_url", "huggingface", "project_page"]),
    ]
    palette = [COLOR["blue"], COLOR["teal"], COLOR["amber"], COLOR["orange"], COLOR["gray"]]
    for ax, (key, title, subkeys) in zip(axes.flat, specs):
        data = census.get(key) or {}
        bottom = np.zeros(len(YEAR_KEYS))
        for idx, subkey in enumerate(subkeys):
            values = [(data.get(y) or {}).get(subkey, 0) for y in YEAR_KEYS]
            ax.bar(x, values, bottom=bottom, color=palette[idx], label=subkey.replace("_", " "))
            bottom += np.array(values)
        ax.set_title(title)
        ax.set_xticks(x, YEAR_KEYS)
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save_figure(fig, "appendix_a2_census_decompositions", fig_dir, manifest)


def plot_sensitivity(summary_paths: list[Path], fig_dir: Path, manifest: list[dict[str, str]]) -> list[dict[str, Any]]:
    labels = ["top10 / 40 ACUs", "top20 / 80 ACUs", "top20 / 120 ACUs"]
    rows = []
    for label, path in zip(labels, summary_paths):
        if not path.exists():
            continue
        payload = read_json(path)
        rows.append(
            {
                "setting": label,
                "mean": payload.get("mean_added_information_score", 0.0),
                "median": payload.get("median_added_information_score", 0.0),
                "low_adequacy": (payload.get("prior_set_adequacy_counts") or {}).get("low", 0),
                "medium_adequacy": (payload.get("prior_set_adequacy_counts") or {}).get("medium", 0),
                "high_adequacy": (payload.get("prior_set_adequacy_counts") or {}).get("high", 0),
            }
        )
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(df))
    ax1.plot(x, df["mean"], marker="o", color=COLOR["blue"], label="Mean score")
    ax1.plot(x, df["median"], marker="s", color=COLOR["teal"], label="Median score")
    ax1.set_ylim(0.75, 1.0)
    ax1.set_xticks(x, df["setting"], rotation=12, ha="right")
    ax1.set_ylabel("Score")
    ax1.set_title("Added-information sensitivity to prior budget")
    ax2 = ax1.twinx()
    ax2.bar(x, df["low_adequacy"], width=0.25, color=COLOR["amber"], alpha=0.45, label="Low adequacy")
    ax2.set_ylabel("Low-adequacy rows")
    lines, labels1 = ax1.get_legend_handles_labels()
    bars, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels1 + labels2, frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "appendix_a3_sensitivity", fig_dir, manifest)
    return rows


def plot_2024_secondary(path: str | Path | None, table_dir: Path, fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    if not path or not Path(path).exists():
        return
    summary = read_json(path)
    rows = [
        ["Rows processed", fmt_int(summary.get("rows", 0))],
        ["Errors", fmt_int(summary.get("errors", 0))],
        ["Mean score", fmt_float(summary.get("mean_added_information_score", 0.0))],
        ["Median score", fmt_float(summary.get("median_added_information_score", 0.0))],
        ["Caveat", "Partial/left-censored unless full 2024 has completed"],
    ]
    write_tex(table_dir / "appendix_a5_2024_secondary_summary.tex", ["Quantity", "Value"], rows, "ll")
    counts = summary.get("prior_set_adequacy_counts") or {}
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    order = [k for k in ADEQUACY_ORDER if k in counts]
    ax.bar(order, [counts[k] for k in order], color=[ADEQUACY_COLORS[k] for k in order])
    ax.set_title("2024 secondary analysis coverage")
    ax.set_xlabel("Prior-set adequacy")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    save_figure(fig, "appendix_a5_2024_secondary", fig_dir, manifest)


def make_corpus_table(census: dict[str, Any], table_dir: Path) -> None:
    rows = [
        ["Papers", fmt_int(census.get("unique_papers", census.get("input_rows", 0)))],
        ["Datasets/resources", fmt_int(census.get("datasets", 0))],
        ["ACUs", fmt_int(census.get("total_acus", 0))],
        ["Prior dataset mentions", fmt_int(census.get("total_prior_dataset_mentions", 0))],
        ["Mean datasets per paper", fmt_float(census.get("mean_datasets_per_paper", 0.0))],
        ["Mean ACUs per dataset", fmt_float(census.get("mean_acus_per_dataset", 0.0))],
    ]
    for y in YEAR_KEYS:
        rows.append([f"{y} papers / datasets", f"{fmt_int(census.get('papers_by_year', {}).get(y, 0))} / {fmt_int(census.get('datasets_by_year', {}).get(y, 0))}"])
    write_tex(table_dir / "table1_corpus_census_summary.tex", ["Quantity", "Value"], rows, "lr")
    write_csv(table_dir / "table1_corpus_census_summary.csv", [{"quantity": row[0], "value": row[1]} for row in rows], ["quantity", "value"])


def make_added_info_filter_table(rows: list[dict[str, Any]], table_dir: Path) -> None:
    specs = [
        ("All 2025 rows", rows),
        ("Included by prior-coverage filter", [row for row in rows if has_adequate_prior_set(row)]),
        ("Excluded by prior-coverage filter", [row for row in rows if not has_adequate_prior_set(row)]),
    ]
    out_rows = []
    csv_rows = []
    for label, group_rows in specs:
        scores = [
            float((row.get("profile") or {}).get("added_information_score"))
            for row in group_rows
            if (row.get("profile") or {}).get("added_information_score") is not None
        ]
        full = sum(score == 1.0 for score in scores)
        support = defaultdict(list)
        for row in group_rows:
            for status, value in ((row.get("profile") or {}).get("support_percentages") or {}).items():
                support[status].append(float(value))
        row = [
            label,
            fmt_int(len(scores)),
            fmt_float(safe_mean(scores)),
            fmt_float(float(np.median(scores)) if scores else 0.0),
            f"{full / len(scores):.1%}" if scores else "0.0%",
            fmt_float(safe_mean(support.get("unsupported", []))),
            fmt_float(safe_mean(support.get("partially_supported", []))),
            fmt_float(safe_mean(support.get("supported", []))),
        ]
        out_rows.append(row)
        csv_rows.append(
            {
                "subset": row[0],
                "n": row[1],
                "mean": row[2],
                "median": row[3],
                "score_1_pct": row[4],
                "unsupported": row[5],
                "partial": row[6],
                "supported": row[7],
            }
        )
    headers = ["Subset", "N", "Mean", "Median", "Score=1", "Unsup.", "Partial", "Supp."]
    write_tex(table_dir / "table_added_information_filter_summary.tex", headers, out_rows, "lccccccc")
    write_csv(table_dir / "table_added_information_filter_summary.csv", csv_rows, ["subset", "n", "mean", "median", "score_1_pct", "unsupported", "partial", "supported"])


def make_retrieval_table(report: dict[str, Any], table_dir: Path) -> None:
    retrieval = report.get("retrieval") or {}
    rows = []
    csv_rows = []
    for method in METHOD_ORDER:
        if method not in retrieval:
            continue
        values = retrieval[method]
        row = [
            METHOD_LABELS.get(method, method),
            fmt_float(values.get("mrr", 0.0)),
            fmt_float(values.get("recall@1", 0.0)),
            fmt_float(values.get("recall@3", 0.0)),
            fmt_float(values.get("recall@5", 0.0)),
            fmt_float(values.get("recall@10", 0.0)),
        ]
        rows.append(row)
        csv_rows.append({"method": row[0], "mrr": row[1], "r1": row[2], "r3": row[3], "r5": row[4], "r10": row[5]})
    headers = ["Method", "MRR", "R@1", "R@3", "R@5", "R@10"]
    write_tex(table_dir / "figure4_retrieval_validation_table.tex", headers, rows, "lccccc")
    write_csv(table_dir / "figure4_retrieval_validation_table.csv", csv_rows, ["method", "mrr", "r1", "r3", "r5", "r10"])


def make_human_validation_table(paths: list[Path], table_dir: Path) -> None:
    rows = []
    csv_rows = []
    for path in paths:
        if not path.exists():
            continue
        payload = read_json(path)
        quality = ((payload.get("added_information_attribution") or {}).get("evidence_quality") or {})
        annotator = "Jason" if "jason" in path.name else "Jiaxin" if "jiaxin" in path.name else path.stem
        row = [
            annotator,
            fmt_int(quality.get("n", 0)),
            fmt_float(quality.get("evidence_label_accuracy", 0.0)),
            fmt_float(quality.get("macro_f1", 0.0)),
            fmt_float(quality.get("evidence_precision", 0.0)),
            fmt_float(quality.get("rationale_groundedness_rate", 0.0)),
        ]
        rows.append(row)
        csv_rows.append({"annotator": row[0], "n": row[1], "accuracy": row[2], "macro_f1": row[3], "precision": row[4], "groundedness": row[5]})
    if rows:
        headers = ["Annotator", "N", "Acc.", "Macro F1", "Precision", "Grounded"]
        write_tex(table_dir / "table2_human_llm_validation.tex", headers, rows, "lccccc")
        write_csv(table_dir / "table2_human_llm_validation.csv", csv_rows, ["annotator", "n", "accuracy", "macro_f1", "precision", "groundedness"])


def short_text(text: str, width: int = 90) -> str:
    return textwrap.shorten(" ".join(str(text).split()), width=width, placeholder="...")


def make_case_table(rows: list[dict[str, Any]], table_dir: Path) -> None:
    selected = []
    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        level = (row.get("prior_set_assessment") or {}).get("prior_set_adequacy") or "unknown"
        by_level[level].append(row)
    targets = [("high", 2), ("medium", 2), ("low", 2)]
    for level, n in targets:
        candidates = sorted(
            by_level.get(level, []),
            key=lambda r: abs(float((r.get("profile") or {}).get("added_information_score") or 0.0) - {"high": 0.55, "medium": 0.80, "low": 1.0}[level]),
        )
        selected.extend(candidates[:n])
    out_rows = []
    csv_rows = []
    for row in selected[:6]:
        prior_names = []
        seen = set()
        for acu in row.get("prior_acus") or []:
            name = acu.get("prior_dataset_name")
            if name and name not in seen:
                seen.add(name)
                prior_names.append(name)
            if len(prior_names) >= 3:
                break
        unsupported = [
            item.get("query_acu") or item.get("query_acu_text") or ""
            for item in row.get("attributions") or []
            if item.get("support_status") == "unsupported"
        ][:2]
        score = (row.get("profile") or {}).get("added_information_score")
        adequacy = (row.get("prior_set_assessment") or {}).get("prior_set_adequacy") or "unknown"
        interpretation = (row.get("prior_set_assessment") or {}).get("prior_set_rationale") or ""
        out_row = [
            short_text(row.get("query_dataset_name") or "", 42),
            short_text("; ".join(prior_names), 55),
            fmt_float(score or 0.0),
            adequacy,
            short_text(" | ".join(unsupported), 105),
            short_text(interpretation, 105),
        ]
        out_rows.append(out_row)
        csv_rows.append(
            {
                "query_dataset": out_row[0],
                "closest_priors": out_row[1],
                "score": out_row[2],
                "adequacy": out_row[3],
                "unsupported_acus": out_row[4],
                "interpretation": out_row[5],
            }
        )
    headers = ["Dataset", "Closest priors", "Score", "Adequacy", "Unsupported ACUs", "Interpretation"]
    write_tex(table_dir / "table3_case_studies.tex", headers, out_rows, "p{0.13\\linewidth}p{0.16\\linewidth}cp{0.09\\linewidth}p{0.27\\linewidth}p{0.25\\linewidth}")
    write_csv(table_dir / "table3_case_studies.csv", csv_rows, ["query_dataset", "closest_priors", "score", "adequacy", "unsupported_acus", "interpretation"])


def make_appendix_census_tables(census_table_dir: Path, table_dir: Path, manifest: list[dict[str, str]]) -> None:
    source_names = [
        "top_tasks",
        "top_domains",
        "top_languages",
        "top_modalities",
        "top_prior_dataset_names",
        "top_source_dataset_names",
        "resource_types",
        "documentation_type",
        "maintenance_status",
    ]
    combined_rows = []
    for name in source_names:
        path = census_table_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).head(15)
        df.to_csv(table_dir / f"appendix_a1_{name}.csv", index=False)
        rows = df.astype(str).values.tolist()
        headers = list(df.columns)
        write_tex(table_dir / f"appendix_a1_{name}.tex", headers, rows, "lr" if len(headers) == 2 else "l" * len(headers))
        combined_rows.append({"table": name, "csv": str(table_dir / f"appendix_a1_{name}.csv"), "tex": str(table_dir / f"appendix_a1_{name}.tex")})
    manifest.extend({"type": "appendix_table", **row} for row in combined_rows)


def make_retrieval_miss_table(report: dict[str, Any], table_dir: Path) -> None:
    misses = ((report.get("miss_examples") or {}).get("gpt_5_4_listwise_rerank") or (report.get("miss_examples") or {}).get("fusion") or [])[:8]
    rows = []
    csv_rows = []
    for miss in misses:
        gold = "; ".join(miss.get("gold_prior_paper_ids") or [])
        top = "; ".join((miss.get("top_retrieved") or [])[:3])
        row = [short_text(miss.get("query_dataset_name") or "", 45), short_text(gold, 70), short_text(top, 90)]
        rows.append(row)
        csv_rows.append({"query_dataset": row[0], "gold_prior_ids": row[1], "top_retrieved": row[2]})
    if rows:
        write_tex(table_dir / "appendix_a2_retrieval_miss_examples.tex", ["Query dataset", "Gold prior", "Top retrieved"], rows, "p{0.22\\linewidth}p{0.34\\linewidth}p{0.38\\linewidth}")
        write_csv(table_dir / "appendix_a2_retrieval_miss_examples.csv", csv_rows, ["query_dataset", "gold_prior_ids", "top_retrieved"])


def make_failure_accounting_table(summary_2025: dict[str, Any], summary_2024_path: str | Path | None, table_dir: Path) -> None:
    rows = [["2025 main", fmt_int(summary_2025.get("rows", 0)), fmt_int(summary_2025.get("errors", 0)), "Complete after retry"]]
    if summary_2024_path and Path(summary_2024_path).exists():
        summary_2024 = read_json(summary_2024_path)
        note = "Partial/secondary unless full 2024 has completed"
        rows.append(["2024 secondary", fmt_int(summary_2024.get("rows", 0)), fmt_int(summary_2024.get("errors", 0)), note])
    write_tex(table_dir / "appendix_a3_failure_accounting.tex", ["Run", "Rows", "Errors", "Note"], rows, "lrrp{0.42\\linewidth}")
    write_csv(table_dir / "appendix_a3_failure_accounting.csv", [{"run": r[0], "rows": r[1], "errors": r[2], "note": r[3]} for r in rows], ["run", "rows", "errors", "note"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-summary", default="artifacts/paper_results/fulltext_census_summary.json")
    parser.add_argument("--census-table-dir", default="artifacts/paper_results/fulltext_census_tables")
    parser.add_argument("--fulltext-extractions-jsonl", default="data/census/fulltext_dataset_extractions_pdf_all.jsonl")
    parser.add_argument("--dataset-bank-jsonl", default="data/census/fulltext_dataset_bank.jsonl")
    parser.add_argument("--added-info-2025", default="data/census/fulltext_added_information_attribution_2025_all_adequacy_v3_top20prior80acu.jsonl")
    parser.add_argument("--added-info-2025-summary", default="data/census/fulltext_added_information_attribution_2025_all_adequacy_v3_top20prior80acu_summary.json")
    parser.add_argument("--retrieval-report", default="data/benchmark/retrieval_cache/acl_pdf130_retrieval_with_gpt54mini_rerank_report.json")
    parser.add_argument("--human-eval-reports", nargs="*", default=[
        "data/benchmark/retrieval_cache/acl_pdf130_gpt54mini_human_eval_jason_report.json",
        "data/benchmark/retrieval_cache/acl_pdf130_gpt54mini_human_eval_jiaxin_report.json",
    ])
    parser.add_argument("--sensitivity-summaries", nargs="*", default=[
        "data/census/fulltext_added_information_attribution_2025_pilot100_adequacy_v3_summary.json",
        "data/census/fulltext_added_information_attribution_2025_pilot100_adequacy_v3_top20prior80acu_summary.json",
        "data/census/fulltext_added_information_attribution_2025_pilot100_adequacy_v3_top20prior120acu_summary.json",
    ])
    parser.add_argument("--added-info-2024-summary", default="data/census/fulltext_added_information_attribution_2024_all_adequacy_v3_top20prior80acu_summary.json")
    parser.add_argument("--output-dir", default="artifacts/paper_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    census = read_json(args.census_summary)
    bank_rows = read_dataset_bank(args.dataset_bank_jsonl)
    bank_by_id = {row.get("bank_id"): row for row in bank_rows if row.get("bank_id")}
    paper_sector = read_paper_sector_map(args.fulltext_extractions_jsonl)
    retrieval_report = read_json(args.retrieval_report)
    added_rows_2025 = load_added_rows(args.added_info_2025)
    filtered_added_rows_2025 = [row for row in added_rows_2025 if has_adequate_prior_set(row)]
    added_summary_2025 = read_json(args.added_info_2025_summary) if Path(args.added_info_2025_summary).exists() else added_summary(added_rows_2025)
    filtered_added_summary_2025 = added_summary(filtered_added_rows_2025)

    plot_pipeline(fig_dir, manifest)
    plot_growth_composition(census, bank_rows, fig_dir, manifest)
    plot_llm_construction(census, bank_rows, fig_dir, manifest)
    plot_release_governance(census, bank_rows, added_rows_2025, bank_by_id, paper_sector, fig_dir, manifest)
    plot_reusability_gap(bank_rows, added_rows_2025, bank_by_id, fig_dir, manifest)
    plot_retrieval_validation(retrieval_report, fig_dir, manifest)
    plot_filtered_score_distribution(added_rows_2025, bank_by_id, fig_dir, manifest)
    plot_delta_types(added_rows_2025, bank_by_id, fig_dir, manifest)
    plot_method_schema(fig_dir, manifest)
    plot_appendix_census_decomposition(census, fig_dir, manifest)
    sensitivity_rows = plot_sensitivity([Path(p) for p in args.sensitivity_summaries], fig_dir, manifest)

    make_corpus_table(census, table_dir)
    make_added_info_filter_table(added_rows_2025, table_dir)
    make_retrieval_table(retrieval_report, table_dir)
    make_human_validation_table([Path(p) for p in args.human_eval_reports], table_dir)
    make_appendix_census_tables(Path(args.census_table_dir), table_dir, manifest)
    make_retrieval_miss_table(retrieval_report, table_dir)
    make_failure_accounting_table(added_summary_2025, args.added_info_2024_summary, table_dir)
    if sensitivity_rows:
        write_csv(table_dir / "appendix_a3_sensitivity.csv", sensitivity_rows, ["setting", "mean", "median", "low_adequacy", "medium_adequacy", "high_adequacy"])

    summary = {
        "figures": sum(1 for item in manifest if item.get("type") == "figure"),
        "appendix_tables": sum(1 for item in manifest if item.get("type") == "appendix_table"),
        "output_dir": str(output_dir),
        "figure_dir": str(fig_dir),
        "table_dir": str(table_dir),
        "main_2025_rows": len(added_rows_2025),
        "main_2025_mean_score": added_summary_2025.get("mean_added_information_score"),
        "filtered_2025_rows": len(filtered_added_rows_2025),
        "filtered_2025_mean_score": filtered_added_summary_2025.get("mean"),
        "main_2025_errors": added_summary_2025.get("errors", 0),
        "manifest": manifest,
    }
    write_json(output_dir / "paper_figure_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
