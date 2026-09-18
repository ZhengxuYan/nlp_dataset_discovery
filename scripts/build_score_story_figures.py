#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


ACU_TYPES = [
    "task/domain",
    "data/source",
    "annotation/protocol",
    "scale/coverage",
    "evaluation/use",
    "availability/quality",
    "governance/ethics",
    "other",
]

FEATURE_LABELS = {
    "is_new_dataset": "New dataset",
    "size_reported": "Dataset size reported",
    "num_languages": "Languages",
    "num_domains": "Domains",
    "uses_llm": "LLM-generated",
    "human_verification_clear": "Human verification",
    "quality_control_clear": "Quality control",
    "released": "Released",
    "license_present": "License present",
    "documentation_present": "Documentation",
    "maintenance_stated": "Maintenance stated",
    "ethics_discussed": "Ethics discussed",
    "pii_discussed": "PII discussed",
    "copyright_discussed": "Copyright discussed",
    "prior_mentions": "Prior mentions",
    "used_for_evaluation": "Evaluation use",
    "used_for_training": "Training use",
    "human_evaluation_present": "Human evaluation",
    "data_study_present": "Ablation/data study",
}

ARCHETYPE_LABELS = [
    "Reusable benchmarks",
    "LLM-generated datasets",
    "Domain expert resources",
    "Scale-driven corpora",
    "Dataset extensions",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_bank_id(value: Any) -> str:
    text = str(value or "")
    return text[4:] if text.startswith("ACL:") else text


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(safe_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k} {safe_text(v)}" for k, v in value.items())
    return str(value)


def is_clear(value: Any) -> bool:
    text = safe_text(value).strip().lower()
    if not text or text in {"none", "no", "n/a", "na", "unknown", "unclear", "not specified", "not reported"}:
        return False
    return True


def has_yes(value: Any) -> bool:
    return str(value or "").strip().lower() == "yes"


def maybe_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def artifact_any(artifacts: dict[str, Any], *keys: str) -> bool:
    return any(bool(artifacts.get(key)) for key in keys)


def venue_group(row: pd.Series) -> str:
    text = " ".join([str(row.get("paper_id", "")), str(row.get("event", "")), str(row.get("venue_prefix", ""))]).lower()
    if "findings" in text:
        return "Findings"
    if "emnlp" in text:
        return "EMNLP"
    if "naacl" in text:
        return "NAACL"
    if "eacl" in text:
        return "EACL"
    if re.search(r"(^|[.\-_])acl([.\-_]|$)", text) or "acl-" in text:
        return "ACL"
    return "Workshops/other"


def source_bucket(row: pd.Series) -> str:
    text = " ".join([
        safe_text(row.get("source_data_origin")),
        safe_text(row.get("source_datasets")),
        safe_text(row.get("domains")),
        safe_text(row.get("tasks")),
        safe_text(row.get("transformation_types")),
    ]).lower()
    checks = [
        ("clinical/medical", ["clinical", "medical", "biomedical", "health", "patient", "mimic", "pubmed"]),
        ("legal", ["legal", "law", "court", "contract"]),
        ("education", ["education", "student", "exam", "tutor", "school"]),
        ("conversation/chat", ["conversation", "dialog", "dialogue", "chat", "social web"]),
        ("social media", ["twitter", "reddit", "weibo", "social media", "forum"]),
        ("web crawl", ["web crawl", "common crawl", "website", "web page", "internet"]),
        ("crowdsourced", ["crowd", "mturk", "prolific", "annotator"]),
        ("LLM-generated", ["llm", "gpt", "chatgpt", "claude", "llama", "synthetic_generation"]),
        ("existing dataset transformation", ["derived", "source_dataset", "translated", "translation", "filtered", "transformation"]),
    ]
    for label, needles in checks:
        if any(needle in text for needle in needles):
            return label
    return "other/unclear"


def model_family(names: list[Any]) -> str:
    text = " ".join(safe_text(name) for name in names).lower()
    if not text:
        return "unclear"
    if "gpt-4" in text or "gpt4" in text:
        return "GPT-4"
    if "chatgpt" in text:
        return "ChatGPT"
    if "gpt-3.5" in text or "gpt3.5" in text:
        return "GPT-3.5"
    if "claude" in text:
        return "Claude"
    if "llama" in text:
        return "Llama"
    if "gemini" in text or "palm" in text:
        return "PaLM/Gemini"
    return "other"


def load_dataset_frame(root: Path) -> pd.DataFrame:
    bank_rows = read_jsonl(root / "data/census/fulltext_dataset_bank.jsonl")
    by_bank_id = {norm_bank_id(row.get("bank_id")): row for row in bank_rows}

    score_paths = [
        root / "data/census/fulltext_added_information_attribution_2024_all_adequacy_v3_top20prior80acu.jsonl",
        root / "data/census/fulltext_added_information_attribution_2025_all_adequacy_v3_top20prior80acu.jsonl",
    ]
    score_rows: list[dict[str, Any]] = []
    for path in score_paths:
        score_rows.extend(read_jsonl(path))

    records = []
    for score in score_rows:
        prior_set_assessment = score.get("prior_set_assessment") or {}
        prior_set_adequacy = str(
            prior_set_assessment.get("prior_set_adequacy")
            or prior_set_assessment.get("adequacy")
            or ""
        ).strip().lower()
        if prior_set_adequacy == "low":
            continue
        bank_id = norm_bank_id(score.get("query_bank_id"))
        bank = by_bank_id.get(bank_id, {})
        profile = score.get("profile") or {}
        support = profile.get("support_percentages") or {}
        artifacts = bank.get("artifacts") or {}
        scale = bank.get("scale") or {}
        governance = bank.get("governance") or {}
        priors = bank.get("prior_dataset_mentions") or []
        acus = bank.get("acus") or []
        acu_counts = Counter((acu.get("type") or "other") for acu in acus if isinstance(acu, dict))
        n_acus = max(len(acus), 1)

        release_status = str(bank.get("release_status") or "").lower()
        documentation_type = str(bank.get("documentation_type") or "").lower()
        maintenance_status = str(bank.get("maintenance_status") or "").lower()
        primary_use = str(bank.get("primary_use") or "").lower()
        role = str(bank.get("role") or "").lower()
        usage = " ".join([primary_use, role, safe_text(bank.get("usage_description"))]).lower()
        construction_text = " ".join([
            safe_text(bank.get("collection_method")),
            safe_text(bank.get("annotation_protocol")),
            safe_text(bank.get("quality_control")),
            safe_text(bank.get("added_information_summary")),
            safe_text(acus),
        ]).lower()

        record = {
            "bank_id": bank_id,
            "paper_id": score.get("query_paper_id") or bank.get("paper_id"),
            "dataset_name": score.get("query_dataset_name") or bank.get("dataset_name"),
            "title": score.get("query_title") or bank.get("title"),
            "year": int(score.get("query_year") or bank.get("year")),
            "event": bank.get("event"),
            "venue_prefix": bank.get("venue_prefix"),
            "score": maybe_float(profile.get("added_information_score")) * 100.0,
            "prior_set_adequacy": prior_set_adequacy,
            "missing_prior_risk": prior_set_assessment.get("missing_prior_risk"),
            "supported": maybe_float(support.get("supported"), 0.0),
            "partially_supported": maybe_float(support.get("partially_supported"), 0.0),
            "unsupported": maybe_float(support.get("unsupported"), 0.0),
            "n_query_acus": int(profile.get("n_query_acus") or len(score.get("query_acus") or [])),
            "is_new_dataset": bool(bank.get("is_new_dataset")),
            "role": bank.get("role"),
            "resource_type": bank.get("resource_type"),
            "primary_use": bank.get("primary_use"),
            "tasks": bank.get("tasks") or [],
            "domains": bank.get("domains") or [],
            "languages": bank.get("languages") or [],
            "modalities": bank.get("modalities") or [],
            "source_data_origin": bank.get("source_data_origin"),
            "source_datasets": bank.get("source_datasets") or [],
            "transformation_types": bank.get("transformation_types") or [],
            "uses_llm": bool(bank.get("uses_llm_synthetic_generation")),
            "synthetic_model_names": bank.get("synthetic_model_names") or [],
            "human_verification_clear": is_clear(bank.get("synthetic_human_verification")),
            "quality_control_clear": is_clear(bank.get("quality_control")),
            "release_status": bank.get("release_status"),
            "released": release_status == "released",
            "license_present": is_clear(bank.get("license")),
            "access_open": str(bank.get("access_restrictions") or "").lower() == "open",
            "documentation_present": documentation_type not in {"", "paper_only", "none", "unclear", "unknown"},
            "maintenance_stated": maintenance_status in {"maintained", "static"},
            "dataset_url": artifact_any(artifacts, "dataset_urls"),
            "github_or_code": artifact_any(artifacts, "code_urls", "github_repos"),
            "huggingface": artifact_any(artifacts, "huggingface_ids"),
            "project_page": artifact_any(artifacts, "project_page_urls"),
            "ethics_discussed": has_yes(governance.get("ethics_discussed")),
            "pii_discussed": has_yes(governance.get("pii_discussed")),
            "consent_discussed": has_yes(governance.get("consent_discussed")),
            "copyright_discussed": has_yes(governance.get("copyright_discussed")),
            "bias_or_fairness_discussed": has_yes(governance.get("bias_or_fairness_discussed")),
            "prior_mentions": len(priors),
            "prior_relationships": [p.get("relationship_type") for p in priors if isinstance(p, dict)],
            "num_instances": scale.get("num_instances"),
            "size_reported": pd.notna(scale.get("num_instances")),
            "num_languages": scale.get("num_languages") or len(bank.get("languages") or []),
            "num_domains": scale.get("num_domains") or len(bank.get("domains") or []),
            "used_for_evaluation": "evaluation" in usage or "benchmark" in usage or "test" in usage,
            "used_for_training": "training" in usage or "train" in usage or "pretraining" in usage or "fine-tuning" in usage,
            "human_evaluation_present": "human evaluation" in construction_text or "human eval" in construction_text,
            "data_study_present": "ablation" in construction_text or "data study" in construction_text or "analysis" in construction_text,
            "confidence": bank.get("confidence"),
        }
        for acu_type in ACU_TYPES:
            record[f"acu_{acu_type}"] = acu_counts.get(acu_type, 0) / n_acus
            record[f"acu_count_{acu_type}"] = acu_counts.get(acu_type, 0)
        records.append(record)

    df = pd.DataFrame(records)
    return finalize_feature_frame(df, scored=True)


def bank_record(bank: dict[str, Any]) -> dict[str, Any]:
    artifacts = bank.get("artifacts") or {}
    scale = bank.get("scale") or {}
    governance = bank.get("governance") or {}
    priors = bank.get("prior_dataset_mentions") or []
    acus = bank.get("acus") or []
    acu_counts = Counter((acu.get("type") or "other") for acu in acus if isinstance(acu, dict))
    n_acus = max(len(acus), 1)

    release_status = str(bank.get("release_status") or "").lower()
    documentation_type = str(bank.get("documentation_type") or "").lower()
    maintenance_status = str(bank.get("maintenance_status") or "").lower()
    primary_use = str(bank.get("primary_use") or "").lower()
    role = str(bank.get("role") or "").lower()
    usage = " ".join([primary_use, role, safe_text(bank.get("usage_description"))]).lower()
    construction_text = " ".join([
        safe_text(bank.get("collection_method")),
        safe_text(bank.get("annotation_protocol")),
        safe_text(bank.get("quality_control")),
        safe_text(bank.get("added_information_summary")),
        safe_text(acus),
    ]).lower()

    record = {
        "bank_id": norm_bank_id(bank.get("bank_id")),
        "paper_id": bank.get("paper_id"),
        "dataset_name": bank.get("dataset_name"),
        "title": bank.get("title"),
        "year": int(bank.get("year")),
        "event": bank.get("event"),
        "venue_prefix": bank.get("venue_prefix"),
        "n_query_acus": len(acus),
        "is_new_dataset": bool(bank.get("is_new_dataset")),
        "role": bank.get("role"),
        "resource_type": bank.get("resource_type"),
        "primary_use": bank.get("primary_use"),
        "tasks": bank.get("tasks") or [],
        "domains": bank.get("domains") or [],
        "languages": bank.get("languages") or [],
        "modalities": bank.get("modalities") or [],
        "source_data_origin": bank.get("source_data_origin"),
        "source_datasets": bank.get("source_datasets") or [],
        "transformation_types": bank.get("transformation_types") or [],
        "uses_llm": bool(bank.get("uses_llm_synthetic_generation")),
        "synthetic_model_names": bank.get("synthetic_model_names") or [],
        "human_verification_clear": is_clear(bank.get("synthetic_human_verification")),
        "quality_control_clear": is_clear(bank.get("quality_control")),
        "release_status": bank.get("release_status"),
        "released": release_status == "released",
        "license_present": is_clear(bank.get("license")),
        "access_open": str(bank.get("access_restrictions") or "").lower() == "open",
        "documentation_present": documentation_type not in {"", "paper_only", "none", "unclear", "unknown"},
        "maintenance_stated": maintenance_status in {"maintained", "static"},
        "dataset_url": artifact_any(artifacts, "dataset_urls"),
        "github_or_code": artifact_any(artifacts, "code_urls", "github_repos"),
        "huggingface": artifact_any(artifacts, "huggingface_ids"),
        "project_page": artifact_any(artifacts, "project_page_urls"),
        "ethics_discussed": has_yes(governance.get("ethics_discussed")),
        "pii_discussed": has_yes(governance.get("pii_discussed")),
        "consent_discussed": has_yes(governance.get("consent_discussed")),
        "copyright_discussed": has_yes(governance.get("copyright_discussed")),
        "bias_or_fairness_discussed": has_yes(governance.get("bias_or_fairness_discussed")),
        "prior_mentions": len(priors),
        "prior_relationships": [p.get("relationship_type") for p in priors if isinstance(p, dict)],
        "num_instances": scale.get("num_instances"),
        "size_reported": pd.notna(scale.get("num_instances")),
        "num_languages": scale.get("num_languages") or len(bank.get("languages") or []),
        "num_domains": scale.get("num_domains") or len(bank.get("domains") or []),
        "used_for_evaluation": "evaluation" in usage or "benchmark" in usage or "test" in usage,
        "used_for_training": "training" in usage or "train" in usage or "pretraining" in usage or "fine-tuning" in usage,
        "human_evaluation_present": "human evaluation" in construction_text or "human eval" in construction_text,
        "data_study_present": "ablation" in construction_text or "data study" in construction_text or "analysis" in construction_text,
        "confidence": bank.get("confidence"),
    }
    for acu_type in ACU_TYPES:
        record[f"acu_{acu_type}"] = acu_counts.get(acu_type, 0) / n_acus
        record[f"acu_count_{acu_type}"] = acu_counts.get(acu_type, 0)
    return record


def finalize_feature_frame(df: pd.DataFrame, scored: bool) -> pd.DataFrame:
    df = df.copy()
    df["venue_group"] = df.apply(venue_group, axis=1)
    if scored:
        df = df.dropna(subset=["score"])
        df["score_quartile"] = pd.qcut(
            df["score"].rank(method="first"),
            4,
            labels=["Bottom 25%", "25-50%", "50-75%", "Top 25%"],
        )
    df["source_bucket"] = df.apply(source_bucket, axis=1)
    df["model_family"] = df["synthetic_model_names"].apply(model_family)
    df["log_num_instances"] = df["num_instances"].apply(lambda x: math.log10(float(x) + 1.0) if pd.notna(x) else 0.0)
    df["prior_lineage_bucket"] = df.apply(lineage_bucket, axis=1)
    df["reusability_level"] = df.apply(reusability_level, axis=1)
    return df


def load_bank_frame(root: Path) -> pd.DataFrame:
    bank_rows = read_jsonl(root / "data/census/fulltext_dataset_bank.jsonl")
    return finalize_feature_frame(pd.DataFrame([bank_record(row) for row in bank_rows]), scored=False)


def lineage_bucket(row: pd.Series) -> str:
    relationships = set(str(x or "").lower() for x in row.get("prior_relationships", []))
    if row.get("is_new_dataset") and not relationships:
        return "new/no explicit prior"
    if {"source_dataset", "extended_from", "combined_with"} & relationships:
        return "dataset family extension"
    if relationships:
        return "related prior positioned"
    return "unclear prior relation"


def reusability_level(row: pd.Series) -> int:
    if not row.get("released"):
        return 0
    if not (row.get("dataset_url") or row.get("github_or_code") or row.get("huggingface") or row.get("project_page")):
        return 1
    if not row.get("license_present"):
        return 2
    if not row.get("documentation_present"):
        return 3
    if not row.get("maintenance_stated"):
        return 4
    return 5


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tex_table(headers: list[str], rows: list[list[str]], align: str) -> str:
    def esc(value: Any) -> str:
        text = str(value)
        replacements = {
            "\\": "\\textbackslash{}",
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}",
        }
        return "".join(replacements.get(char, char) for char in text)

    lines = ["\\begin{tabular}{" + align + "}", "\\toprule"]
    lines.append(" & ".join(esc(header) for header in headers) + " \\\\")
    lines.append("\\midrule")
    lines.extend(" & ".join(esc(cell) for cell in row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def mean_pct(series: pd.Series) -> str:
    return pct(float(series.mean())) if len(series) else "0.0"


def fmt(value: float) -> str:
    return f"{value:.1f}"


def build_fig1(df: pd.DataFrame, outdir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axes[0]
    sns.histplot(df["score"], bins=30, kde=True, color="#3972a8", ax=ax)
    mean = df["score"].mean()
    median = df["score"].median()
    q1, q3 = df["score"].quantile([0.25, 0.75])
    for value, color, label in [
        (mean, "#d95f02", "mean"),
        (median, "#1b9e77", "median"),
        (q1, "#7570b3", "Q1/Q3"),
        (q3, "#7570b3", None),
    ]:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.3, label=label)
    ax.set_title("a. Score distribution")
    ax.set_xlabel("Dataset contribution score")
    ax.set_ylabel("Datasets")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    sns.violinplot(data=df, x="year", y="score", color="#b9d4ea", inner=None, cut=0, ax=ax)
    sns.boxplot(data=df, x="year", y="score", width=0.25, color="white", fliersize=0, ax=ax)
    ax.set_title("b. Score by year")
    ax.set_xlabel("")
    ax.set_ylabel("Score")

    ax = axes[2]
    order = ["ACL", "EMNLP", "NAACL", "EACL", "Findings", "Workshops/other"]
    present = [x for x in order if x in set(df["venue_group"])]
    sns.boxplot(data=df, x="venue_group", y="score", order=present, color="#d8e6c3", fliersize=1.5, ax=ax)
    ax.set_title("c. Score by venue/event type")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35)
    savefig(fig, outdir, "figure1_score_landscape")


def build_fig2(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.4))

    support = (
        df.groupby("score_quartile", observed=False)[["supported", "partially_supported", "unsupported"]]
        .mean()
        .rename(columns={
            "supported": "Supported by priors",
            "partially_supported": "Partially supported",
            "unsupported": "Unsupported / added",
        })
    )
    support.plot(kind="bar", stacked=True, ax=axes[0], width=0.8, color=["#7f8c8d", "#e6b85c", "#4c78a8"])
    axes[0].set_title("a. Prior-support profile by score quartile")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean share of query ACUs")
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
    axes[0].tick_params(axis="x", rotation=25)

    binary_features = [
        ("Evaluation use", "used_for_evaluation"),
        ("Training use", "used_for_training"),
        ("Released", "released"),
        ("License present", "license_present"),
        ("Documentation", "documentation_present"),
        ("Human verification", "human_verification_clear"),
        ("Quality control", "quality_control_clear"),
        ("LLM-generated", "uses_llm"),
        ("Size reported", "size_reported"),
        ("Ethics discussed", "ethics_discussed"),
        ("PII discussed", "pii_discussed"),
        ("Copyright discussed", "copyright_discussed"),
    ]
    bottom = df[df["score_quartile"].astype(str) == "Bottom 25%"]
    top = df[df["score_quartile"].astype(str) == "Top 25%"]
    deltas = []
    for label, col in binary_features:
        deltas.append({
            "label": label,
            "delta_pp": 100.0 * (top[col].mean() - bottom[col].mean()),
            "top": 100.0 * top[col].mean(),
            "bottom": 100.0 * bottom[col].mean(),
        })
    delta_df = pd.DataFrame(deltas).sort_values("delta_pp")
    colors = np.where(delta_df["delta_pp"] >= 0, "#4c78a8", "#c44e52")
    axes[1].barh(delta_df["label"], delta_df["delta_pp"], color=colors)
    axes[1].axvline(0, color="#555555", linewidth=0.9)
    axes[1].set_title("b. Top minus bottom quartile metadata gaps")
    axes[1].set_xlabel("Difference in prevalence (percentage points)")
    axes[1].set_ylabel("")

    signals = [
        ("Evaluation use", "used_for_evaluation"),
        ("Released + license", "released_and_licensed"),
        ("LLM + verification", "llm_and_verification"),
        ("Quality control", "quality_control_clear"),
        ("Prior mentioned", "has_prior_mentions"),
        ("Size reported", "size_reported"),
    ]
    signal_df = df.assign(
        released_and_licensed=df["released"] & df["license_present"],
        llm_and_verification=df["uses_llm"] & df["human_verification_clear"],
        has_prior_mentions=df["prior_mentions"] > 0,
    )
    rows = []
    for label, col in signals:
        rows.append({"Signal": label, "Status": "Absent", "Mean score": signal_df.loc[~signal_df[col], "score"].mean()})
        rows.append({"Signal": label, "Status": "Present", "Mean score": signal_df.loc[signal_df[col], "score"].mean()})
    signal_plot = pd.DataFrame(rows)
    sns.pointplot(
        data=signal_plot,
        y="Signal",
        x="Mean score",
        hue="Status",
        dodge=0.35,
        linestyle="none",
        palette={"Absent": "#a6a6a6", "Present": "#4c78a8"},
        ax=axes[2],
    )
    axes[2].set_title("c. Mean score with concrete reporting signals")
    axes[2].set_xlabel("Mean score")
    axes[2].set_ylabel("")
    axes[2].legend(frameon=False, title="", loc="lower right")
    savefig(fig, outdir, "figure2_score_drivers")
    return delta_df.sort_values("delta_pp", ascending=False)


def build_fig3(df: pd.DataFrame, scored_df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    yearly = df.groupby("year")["uses_llm"].mean().reset_index()
    sns.barplot(data=yearly, x="year", y="uses_llm", color="#6ca6a6", ax=axes[0, 0])
    axes[0, 0].set_title("a. LLM-assisted construction by year")
    axes[0, 0].set_ylabel("Fraction of datasets")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylim(0, max(0.65, yearly["uses_llm"].max() + 0.08))

    pivot = scored_df.pivot_table(
        index="uses_llm",
        columns="human_verification_clear",
        values="score",
        aggfunc=["count", "mean"],
        fill_value=0,
    )
    matrix = np.zeros((2, 2))
    labels = np.empty((2, 2), dtype=object)
    for i, llm in enumerate([False, True]):
        for j, hv in enumerate([False, True]):
            count = int(pivot.loc[llm, ("count", hv)]) if (llm in pivot.index and ("count", hv) in pivot.columns) else 0
            mean_score = float(pivot.loc[llm, ("mean", hv)]) if (llm in pivot.index and ("mean", hv) in pivot.columns) else 0.0
            matrix[i, j] = mean_score
            labels[i, j] = f"n={count}\n{mean_score:.1f}"
    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="YlGnBu",
        xticklabels=["No/unclear verification", "Clear verification"],
        yticklabels=["No LLM", "LLM used"],
        ax=axes[0, 1],
        cbar_kws={"label": "Mean score"},
    )
    axes[0, 1].set_title("b. LLM use and human verification")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")

    qc = (
        df.assign(llm_group=np.where(df["uses_llm"], "LLM-generated", "Non-LLM"))
        .groupby("llm_group")["quality_control_clear"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
    )
    qc["quality_control_clear"] = qc["quality_control_clear"].map({True: "clear", False: "unclear/none"})
    sns.barplot(data=qc, x="llm_group", y="share", hue="quality_control_clear", ax=axes[1, 0])
    axes[1, 0].set_title("c. Quality-control disclosure")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Share")
    axes[1, 0].legend(frameon=False, title="")

    model_counts = df[df["uses_llm"]]["model_family"].value_counts().reindex(
        ["GPT-4", "ChatGPT", "GPT-3.5", "Claude", "Llama", "PaLM/Gemini", "other", "unclear"],
        fill_value=0,
    )
    sns.barplot(x=model_counts.values, y=model_counts.index, color="#b8875e", ax=axes[1, 1])
    axes[1, 1].set_title("d. Model families named in construction")
    axes[1, 1].set_xlabel("Datasets")
    axes[1, 1].set_ylabel("")
    savefig(fig, outdir, "figure3_llm_construction_shift")


def build_fig4(df: pd.DataFrame, scored_df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0, 1.4]})
    ladder = df.groupby(["year", "reusability_level"]).size().reset_index(name="count")
    ladder["share"] = ladder["count"] / ladder.groupby("year")["count"].transform("sum")
    ladder["reusability_level"] = ladder["reusability_level"].astype(int)
    ladder_pivot = ladder.pivot(index="year", columns="reusability_level", values="share").fillna(0)
    ladder_pivot.plot(kind="bar", stacked=True, ax=axes[0], colormap="viridis")
    axes[0].set_title("a. Reusability ladder by year")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Share of datasets")
    axes[0].legend(title="Level", fontsize=7, frameon=False)
    axes[0].tick_params(axis="x", rotation=0)

    sns.boxplot(data=scored_df, x="reusability_level", y="score", color="#c9d7a6", fliersize=1.5, ax=axes[1])
    axes[1].set_title("b. Score by reusability level")
    axes[1].set_xlabel("Reusability level")
    axes[1].set_ylabel("Score")

    gov_cols = ["ethics_discussed", "pii_discussed", "consent_discussed", "copyright_discussed", "bias_or_fairness_discussed"]
    gov = df.groupby("source_bucket")[gov_cols].mean()
    source_order = ["web crawl", "social media", "conversation/chat", "clinical/medical", "legal", "education", "crowdsourced", "LLM-generated", "existing dataset transformation", "other/unclear"]
    gov = gov.reindex([x for x in source_order if x in gov.index])
    short_cols = {
        "web crawl": "web",
        "social media": "social",
        "conversation/chat": "chat",
        "clinical/medical": "clinical",
        "legal": "legal",
        "education": "edu",
        "crowdsourced": "crowd",
        "LLM-generated": "LLM",
        "existing dataset transformation": "derived",
        "other/unclear": "other",
    }
    gov = gov.rename(index=short_cols)
    sns.heatmap(gov.T, cmap="YlOrRd", vmin=0, vmax=max(0.55, gov.max().max()), ax=axes[2], cbar_kws={"label": "Disclosure rate"})
    axes[2].set_title("c. Governance disclosure by source risk")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    axes[2].set_yticklabels(["ethics", "PII", "consent", "copyright", "bias/fairness"], rotation=0)
    axes[2].tick_params(axis="x", rotation=35, labelsize=8)
    savefig(fig, outdir, "figure4_reusability_governance")


def assign_archetypes(df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score_by_bank_id = scored_df.drop_duplicates("bank_id").set_index("bank_id")["score"]
    adequacy_by_bank_id = scored_df.drop_duplicates("bank_id").set_index("bank_id")["prior_set_adequacy"]
    out["score"] = out["bank_id"].map(score_by_bank_id)
    out["prior_set_adequacy"] = out["bank_id"].map(adequacy_by_bank_id)
    out["has_prior_mentions"] = out["prior_mentions"] > 0

    source = out["source_bucket"].astype(str)
    large_or_training = (
        out["used_for_training"]
        | (out["num_instances"].fillna(0).astype(float) >= 100000)
        | (out["acu_scale/coverage"].fillna(0).astype(float) >= 0.45)
    )
    expert_domain = (
        source.isin(["clinical/medical", "legal", "education"])
        | (out["quality_control_clear"] & (out["human_verification_clear"] | (out["num_domains"].fillna(0).astype(float) >= 2)))
    )
    extension = (out["prior_mentions"] > 0) & (~out["is_new_dataset"])
    reusable = out["used_for_evaluation"] & (out["reusability_level"] >= 3)

    out["archetype"] = "Dataset extensions"
    out.loc[large_or_training, "archetype"] = "Scale-driven corpora"
    out.loc[expert_domain, "archetype"] = "Domain expert resources"
    out.loc[out["uses_llm"], "archetype"] = "LLM-generated datasets"
    out.loc[reusable, "archetype"] = "Reusable benchmarks"
    out.loc[extension & ~reusable & ~out["uses_llm"] & ~expert_domain & ~large_or_training, "archetype"] = "Dataset extensions"
    return out


def build_fig5(df: pd.DataFrame, scored_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    clustered = assign_archetypes(df, scored_df)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})

    order = [x for x in ARCHETYPE_LABELS if x in set(clustered["archetype"])]
    count_score = (
        clustered.groupby("archetype")
        .agg(datasets=("bank_id", "count"), mean_score=("score", "mean"))
        .reindex(order)
        .reset_index()
    )
    ax_count = axes[0]
    sns.barplot(data=count_score, y="archetype", x="datasets", color="#8ab6d6", ax=ax_count)
    ax_count.set_title("a. Archetype size and mean score")
    ax_count.set_xlabel("Datasets")
    ax_count.set_ylabel("")
    ax_score = ax_count.twiny()
    ax_score.plot(count_score["mean_score"], np.arange(len(count_score)), "o", color="#c44e52", markersize=5)
    ax_score.set_xlabel("Mean score", color="#c44e52")
    ax_score.tick_params(axis="x", colors="#c44e52")
    for i, value in enumerate(count_score["datasets"]):
        ax_count.text(value, i, f" {int(value)}", va="center", fontsize=7)

    profile_cols = {
        "Mean score": "score",
        "Task/domain ACU": "acu_task/domain",
        "Scale ACU": "acu_scale/coverage",
        "Evaluation ACU": "acu_evaluation/use",
        "Availability ACU": "acu_availability/quality",
        "Governance ACU": "acu_governance/ethics",
        "LLM use": "uses_llm",
        "Human verification": "human_verification_clear",
        "Release": "released",
        "License": "license_present",
        "Documentation": "documentation_present",
        "Has prior mentions": "has_prior_mentions",
    }
    profile = clustered.groupby("archetype")[[*profile_cols.values()]].mean().rename(columns={v: k for k, v in profile_cols.items()})
    profile = profile.reindex(order)
    display_profile = profile.drop(columns=["Mean score"]).T * 100.0
    sns.heatmap(display_profile, cmap="Blues", vmin=0, vmax=100, annot=True, fmt=".0f", ax=axes[1], cbar_kws={"label": "% / mean share"})
    axes[1].set_title("b. Archetype profile")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].tick_params(axis="y", labelsize=8)

    trend = clustered.groupby(["year", "archetype"]).size().reset_index(name="count")
    trend["share"] = trend["count"] / trend.groupby("year")["count"].transform("sum")
    pivot = trend.pivot(index="year", columns="archetype", values="share").fillna(0)
    pivot = pivot[[x for x in ARCHETYPE_LABELS if x in pivot.columns]]
    pivot.plot(kind="bar", stacked=True, ax=axes[2], colormap="tab20")
    axes[2].set_title("c. Archetype trend over time")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Share")
    axes[2].tick_params(axis="x", rotation=0)
    axes[2].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    savefig(fig, outdir, "figure5_contribution_archetypes")
    return clustered


def build_tables(df: pd.DataFrame, bank_df: pd.DataFrame, clustered: pd.DataFrame, outdir: Path) -> None:
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for year, part in bank_df.groupby("year"):
        rows.append([
            str(year),
            str(part["paper_id"].nunique()),
            str(len(part)),
            str(int(part["is_new_dataset"].sum())),
            str(int((part["prior_mentions"] > 0).sum())),
            str(int((part["n_query_acus"] > 0).sum())),
            fmt(part["n_query_acus"].mean()),
            str(int((part["prior_mentions"] > 0).sum())),
            f"{int((part['confidence'] == 'high').sum())}/{int((part['confidence'] == 'medium').sum())}/{int((part['confidence'] == 'low').sum())}",
        ])
    write_text(
        table_dir / "table1_corpus_extraction_summary.tex",
        tex_table(
            ["Year", "Papers", "Datasets", "New", "Extensions/prior", "With ACUs", "Avg ACUs", "Prior mentions", "Conf. H/M/L"],
            rows,
            "lrrrrrrrr",
        ),
    )

    features = [
        ("Mean score", lambda p: fmt(p["score"].mean()), lambda a, b: fmt(a["score"].mean() - b["score"].mean())),
        ("% new dataset", lambda p: mean_pct(p["is_new_dataset"]), lambda a, b: pct(a["is_new_dataset"].mean() - b["is_new_dataset"].mean())),
        ("Avg # ACUs", lambda p: fmt(p["n_query_acus"].mean()), lambda a, b: fmt(a["n_query_acus"].mean() - b["n_query_acus"].mean())),
        ("Avg # prior mentions", lambda p: fmt(p["prior_mentions"].mean()), lambda a, b: fmt(a["prior_mentions"].mean() - b["prior_mentions"].mean())),
        ("% LLM-generated", lambda p: mean_pct(p["uses_llm"]), lambda a, b: pct(a["uses_llm"].mean() - b["uses_llm"].mean())),
        ("% human verification", lambda p: mean_pct(p["human_verification_clear"]), lambda a, b: pct(a["human_verification_clear"].mean() - b["human_verification_clear"].mean())),
        ("% quality control", lambda p: mean_pct(p["quality_control_clear"]), lambda a, b: pct(a["quality_control_clear"].mean() - b["quality_control_clear"].mean())),
        ("% released", lambda p: mean_pct(p["released"]), lambda a, b: pct(a["released"].mean() - b["released"].mean())),
        ("% license present", lambda p: mean_pct(p["license_present"]), lambda a, b: pct(a["license_present"].mean() - b["license_present"].mean())),
        ("% documentation", lambda p: mean_pct(p["documentation_present"]), lambda a, b: pct(a["documentation_present"].mean() - b["documentation_present"].mean())),
        ("% maintenance stated", lambda p: mean_pct(p["maintenance_stated"]), lambda a, b: pct(a["maintenance_stated"].mean() - b["maintenance_stated"].mean())),
        ("% used for evaluation", lambda p: mean_pct(p["used_for_evaluation"]), lambda a, b: pct(a["used_for_evaluation"].mean() - b["used_for_evaluation"].mean())),
        ("% used for training", lambda p: mean_pct(p["used_for_training"]), lambda a, b: pct(a["used_for_training"].mean() - b["used_for_training"].mean())),
        ("Avg # languages", lambda p: fmt(p["num_languages"].mean()), lambda a, b: fmt(a["num_languages"].mean() - b["num_languages"].mean())),
        ("Median instances", lambda p: fmt(p.loc[p["num_instances"].notna(), "num_instances"].median() if p["num_instances"].notna().any() else 0), lambda a, b: fmt((a.loc[a["num_instances"].notna(), "num_instances"].median() if a["num_instances"].notna().any() else 0) - (b.loc[b["num_instances"].notna(), "num_instances"].median() if b["num_instances"].notna().any() else 0))),
        ("% ethics discussed", lambda p: mean_pct(p["ethics_discussed"]), lambda a, b: pct(a["ethics_discussed"].mean() - b["ethics_discussed"].mean())),
        ("% PII discussed", lambda p: mean_pct(p["pii_discussed"]), lambda a, b: pct(a["pii_discussed"].mean() - b["pii_discussed"].mean())),
        ("% copyright discussed", lambda p: mean_pct(p["copyright_discussed"]), lambda a, b: pct(a["copyright_discussed"].mean() - b["copyright_discussed"].mean())),
    ]
    quartiles = ["Bottom 25%", "25-50%", "50-75%", "Top 25%"]
    parts = {q: df[df["score_quartile"].astype(str) == q] for q in quartiles}
    qrows = []
    for label, formatter, diff_formatter in features:
        qrows.append([label] + [formatter(parts[q]) for q in quartiles] + [diff_formatter(parts["Top 25%"], parts["Bottom 25%"])])
    write_text(
        table_dir / "table2_score_quartile_comparison.tex",
        tex_table(["Feature", "Bottom 25%", "25-50%", "50-75%", "Top 25%", "Top-Bottom"], qrows, "lrrrrr"),
    )

    archetype_rows = []
    for archetype, part in clustered.groupby("archetype"):
        top = part.sort_values("score", ascending=False).head(3)
        common_acus = (
            part[[f"acu_{t}" for t in ACU_TYPES]]
            .mean()
            .sort_values(ascending=False)
            .head(2)
            .index.str.replace("acu_", "", regex=False)
            .tolist()
        )
        signals = []
        if part["uses_llm"].mean() > 0.5:
            signals.append("LLM use")
        if part["released"].mean() > 0.8:
            signals.append("released")
        if part["license_present"].mean() > 0.5:
            signals.append("licensed")
        if part["prior_mentions"].mean() > df["prior_mentions"].mean():
            signals.append("clear prior lineage")
        if part["score"].mean() > df["score"].mean():
            signals.append("high score")
        archetype_rows.append([
            archetype,
            str(len(part)),
            fmt(part["score"].mean()),
            ", ".join(signals) or "mixed signals",
            ", ".join(common_acus),
            "LLM-assisted" if part["uses_llm"].mean() > 0.5 else ("expert/curated" if part["quality_control_clear"].mean() > 0.4 else "mixed"),
            f"release {pct(part['released'].mean())}%, license {pct(part['license_present'].mean())}%",
            "; ".join(top["dataset_name"].fillna(top["title"]).astype(str).head(3).tolist()),
        ])
    archetype_rows = sorted(archetype_rows, key=lambda r: ARCHETYPE_LABELS.index(r[0]) if r[0] in ARCHETYPE_LABELS else 999)
    write_text(
        table_dir / "table3_archetype_summary.tex",
        tex_table(
            ["Archetype", "Datasets", "Mean score", "Defining signals", "Common ACUs", "Typical construction", "Reusability", "Examples"],
            archetype_rows,
            "lrrlllll",
        ),
    )

    df.to_csv(table_dir / "score_story_dataset_level.csv", index=False)
    clustered.to_csv(table_dir / "score_story_dataset_level_with_archetypes.csv", index=False)


def build_summary(df: pd.DataFrame, importance: pd.DataFrame, outdir: Path) -> None:
    lines = [
        "# Score Story Figure/Table Summary",
        "",
        f"- Scored dataset rows: {len(df):,}",
        f"- Years: {', '.join(str(y) for y in sorted(df['year'].unique()))}",
        f"- Mean score: {df['score'].mean():.1f}",
        f"- Median score: {df['score'].median():.1f}",
        f"- 25th/75th percentiles: {df['score'].quantile(0.25):.1f} / {df['score'].quantile(0.75):.1f}",
        f"- LLM-generated fraction: {100 * df['uses_llm'].mean():.1f}%",
        f"- Released fraction: {100 * df['released'].mean():.1f}%",
        f"- License-present fraction: {100 * df['license_present'].mean():.1f}%",
        "",
        "## Largest top-vs-bottom quartile metadata gaps",
        "",
    ]
    for _, row in importance.head(10).iterrows():
        lines.append(f"- {row['label']}: {row['delta_pp']:.1f} percentage points")
    write_text(outdir / "score_story_summary.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_results/score_story"))
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    figures_dir = output_dir / "figures"

    df = load_dataset_frame(root)
    bank_df = load_bank_frame(root)
    build_fig1(df, figures_dir)
    importance = build_fig2(df, figures_dir)
    build_fig3(bank_df, df, figures_dir)
    build_fig4(bank_df, df, figures_dir)
    clustered = build_fig5(bank_df, df, figures_dir)
    build_tables(df, bank_df, clustered, output_dir)
    build_summary(df, importance, output_dir)


if __name__ == "__main__":
    main()
