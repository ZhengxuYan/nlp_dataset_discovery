#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / ".matplotlib-cache")))
os.environ.setdefault("XDG_CACHE_HOME", str((Path(__file__).resolve().parents[1] / ".cache")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


LABEL_ORDER = [
    "covered",
    "partially_covered",
    "not_covered",
    "contradicted",
    "not_comparable",
]

LABEL_DISPLAY = {
    "covered": "Covered",
    "partially_covered": "Partially covered",
    "not_covered": "Not covered",
    "contradicted": "Contradicted",
    "not_comparable": "Not comparable",
}

TYPE_ORDER = [
    "task/domain",
    "data/source",
    "annotation/protocol",
    "scale/coverage",
    "evaluation/use",
    "availability/quality",
    "governance/ethics",
    "other",
]

GOVERNANCE_KEYS = [
    "ethics_discussed",
    "pii_discussed",
    "consent_discussed",
    "copyright_discussed",
    "bias_or_fairness_discussed",
]

REPORTING_FEATURE_DISPLAY = {
    "model_family_named": "Model family named",
    "released": "Released",
    "open_access": "Open access",
    "license_specified": "License specified",
    "source_or_origin_disclosed": "Source/origin disclosed",
    "quality_control_reported": "Quality control",
    "human_verification_reported": "Human verification",
    "documentation_beyond_paper": "Documentation beyond paper",
    "ethics_discussed": "Ethics",
    "pii_discussed": "PII",
    "consent_discussed": "Consent",
    "copyright_discussed": "Copyright",
    "bias_or_fairness_discussed": "Bias/fairness",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return {
        "supported": "covered",
        "partially_supported": "partially_covered",
        "unsupported": "not_covered",
        "covered": "covered",
        "partially_covered": "partially_covered",
        "not_covered": "not_covered",
        "contradicted": "contradicted",
        "not_comparable": "not_comparable",
    }.get(raw, raw or "unknown")


def norm_text(value: Any) -> str:
    return str(value or "").strip()


def has_known(value: Any) -> bool:
    text = norm_text(value).lower()
    return bool(text) and text not in {"unknown", "unclear", "none", "n/a", "not reported", "not_applicable"}


def list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def any_artifact(bank: dict[str, Any], *names: str) -> bool:
    artifacts = bank.get("artifacts")
    if isinstance(artifacts, list):
        # Integrated bank flattens artifacts imperfectly in some historical runs.
        joined = " ".join(str(x) for x in artifacts).lower()
        return any(name.replace("_", "").lower() in joined for name in names)
    if isinstance(artifacts, dict):
        return any(bool(artifacts.get(name)) for name in names)
    return False


def contains_any(values: Iterable[Any], needles: Iterable[str]) -> bool:
    text = " ".join(str(v).lower() for v in values if v is not None)
    return any(needle in text for needle in needles)


def normalize_model_family(name: Any) -> str:
    text = norm_text(name).lower()
    if not text or text in {"unknown", "unclear", "none", "n/a", "unspecified", "llm", "large language model"}:
        return ""
    if "gpt" in text or "chatgpt" in text or "openai" in text:
        return "OpenAI GPT"
    if "claude" in text:
        return "Claude"
    if "gemini" in text or "palm" in text or "bard" in text:
        return "Gemini/PaLM"
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    if "deepseek" in text:
        return "DeepSeek"
    if "mistral" in text or "mixtral" in text:
        return "Mistral"
    return text.split()[0][:32]


def classify_construction(row: dict[str, Any]) -> str:
    transformations = [str(x).lower() for x in row.get("transformation_types") or []]
    collection = norm_text(row.get("collection_method")).lower()
    origin = norm_text(row.get("source_data_origin")).lower()
    annotator = norm_text(row.get("annotator_type")).lower()
    protocol = norm_text(row.get("annotation_protocol")).lower()
    primary_use = norm_text(row.get("primary_use")).lower()
    resource_type = norm_text(row.get("resource_type")).lower()
    source_count = list_len(row.get("source_datasets"))
    languages = row.get("languages") or []
    uses_llm = bool(row.get("uses_llm_synthetic_generation"))
    synthetic_models = row.get("synthetic_model_names") or []

    text_blob = " ".join([*transformations, collection, origin, annotator, protocol, primary_use, resource_type])
    if uses_llm or synthetic_models or contains_any([text_blob], ["llm", "synthetic", "generated by", "chatgpt", "gpt-"]):
        return "LLM/synthetic"
    if contains_any([text_blob], ["translation", "translated", "multilingual", "cross-lingual"]) or len(languages) >= 3:
        return "translation/multilingual"
    if contains_any([text_blob], ["aggregate", "aggregation", "benchmark suite", "benchmark"]) and (
        "benchmark" in resource_type or "benchmark" in primary_use
    ):
        return "benchmark aggregation"
    if "annotation" in transformations or has_known(protocol) or annotator in {
        "crowd",
        "expert",
        "human",
        "author",
        "student",
        "professional",
    }:
        return "human annotated"
    if source_count or contains_any([text_blob], ["filter", "filtered", "derived", "converted", "extracted", "cleaned", "extension"]):
        return "derived/filtered"
    if contains_any([text_blob], ["collection", "collected", "crawl", "scrape", "survey"]):
        return "newly collected"
    return "unknown/not reported"


def has_llm_synthetic_generation(row: dict[str, Any], construction_method: str | None = None) -> bool:
    synthetic_models = row.get("synthetic_model_names") or []
    if bool(row.get("uses_llm_synthetic_generation")) or synthetic_models:
        return True
    if construction_method == "LLM/synthetic":
        return True
    fields = [
        row.get("collection_method"),
        row.get("source_data_origin"),
        row.get("annotator_type"),
        row.get("annotation_protocol"),
        row.get("primary_use"),
        row.get("resource_type"),
        " ".join(str(x) for x in row.get("transformation_types") or []),
    ]
    return contains_any(fields, ["llm", "synthetic", "generated by", "chatgpt", "gpt-"])


def named_model_families(row: dict[str, Any]) -> list[str]:
    families = []
    for name in row.get("synthetic_model_names") or []:
        family = normalize_model_family(name)
        if family:
            families.append(family)
    return sorted(set(families))


def build_doc_features(row: dict[str, Any]) -> dict[str, bool]:
    release_status = norm_text(row.get("release_status")).lower()
    license_text = norm_text(row.get("license")).lower()
    access = norm_text(row.get("access_restrictions")).lower()
    docs = norm_text(row.get("documentation_type")).lower()
    source_datasets = row.get("source_datasets") or []
    quality = norm_text(row.get("quality_control")).lower()
    human_verification = norm_text(row.get("synthetic_human_verification")).lower()
    governance = row.get("governance") or {}
    artifacts_flat = " ".join(str(x).lower() for x in row.get("artifacts") or [])

    def artifact_name(name: str) -> bool:
        return name in artifacts_flat or any_artifact(row, name)

    return {
        "released": release_status == "released",
        "open_access": access == "open",
        "license_specified": has_known(license_text),
        "source_or_origin_disclosed": bool(source_datasets) or has_known(row.get("source_data_origin")),
        "quality_control_reported": has_known(quality),
        "human_verification_reported": human_verification == "yes",
        "documentation_beyond_paper": docs not in {"", "unknown", "unclear", "paper_only"},
        "ethics_discussed": str(governance.get("ethics_discussed") or "").lower() == "yes",
        "pii_discussed": str(governance.get("pii_discussed") or "").lower() == "yes",
        "consent_discussed": str(governance.get("consent_discussed") or "").lower() == "yes",
        "copyright_discussed": str(governance.get("copyright_discussed") or "").lower() == "yes",
        "bias_or_fairness_discussed": str(governance.get("bias_or_fairness_discussed") or "").lower() == "yes",
    }


def inclusion_under(row: dict[str, Any], *, max_low: float, max_risk: float, min_interp: float) -> bool:
    if "profile" not in row:
        return (
            float(row.get("low_adequacy_rate") or 0.0) <= max_low
            and float(row.get("high_missing_prior_risk_rate") or 0.0) <= max_risk
            and float(row.get("adequate_query_dcu_rate") or 0.0) >= min_interp
        )
    profile = row.get("profile") or {}
    return (
        float(profile.get("low_adequacy_rate") or 0.0) <= max_low
        and float(profile.get("high_missing_prior_risk_rate") or 0.0) <= max_risk
        and float(profile.get("adequate_query_acu_rate") or 0.0) >= min_interp
    )


def flatten(attribution_rows: list[dict[str, Any]], bank_by_id: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_rows: list[dict[str, Any]] = []
    dcu_rows: list[dict[str, Any]] = []
    for row in attribution_rows:
        profile = row.get("profile") or {}
        bank = bank_by_id.get(row.get("query_bank_id"), {})
        query_year = int(row.get("query_year") or bank.get("year") or 0)
        source = row.get("source_corpus") or bank.get("source_corpus") or "unknown"
        included = bool(profile.get("analysis_inclusion"))
        construction = classify_construction(bank)
        uses_llm = has_llm_synthetic_generation(bank, construction)
        model_families = named_model_families(bank)
        doc_features = build_doc_features(bank)
        rec = {
            "bank_id": row.get("query_bank_id"),
            "paper_id": row.get("query_paper_id"),
            "dataset_name": row.get("query_dataset_name"),
            "year": query_year,
            "source_corpus": source,
            "included": included,
            "n_query_dcus": int(profile.get("n_query_acus") or len(row.get("query_acus") or [])),
            "pass_dcus": 0,
            "added_information_score": profile.get("added_information_score"),
            "low_adequacy_rate": profile.get("low_adequacy_rate"),
            "high_missing_prior_risk_rate": profile.get("high_missing_prior_risk_rate"),
            "adequate_query_dcu_rate": profile.get("adequate_query_acu_rate"),
            "construction_method": construction,
            "release_status": bank.get("release_status") or "unknown",
            "access_restrictions": bank.get("access_restrictions") or "unknown",
            "license": bank.get("license") or "unknown",
            "documentation_type": bank.get("documentation_type") or "unknown",
            "paper_sector": "unknown",
            "uses_llm_synthetic_generation": uses_llm,
            "construction_group": "LLM-assisted" if uses_llm else "Non-LLM/other",
            "model_family_named": bool(model_families),
            "model_families": "; ".join(model_families),
            **doc_features,
        }
        record_rows.append(rec)
        for attr in row.get("attributions") or []:
            label = normalize_label(attr.get("support_status"))
            dcu_type = attr.get("query_acu_type") or attr.get("delta_type") or "other"
            dcu = {
                "bank_id": row.get("query_bank_id"),
                "paper_id": row.get("query_paper_id"),
                "dataset_name": row.get("query_dataset_name"),
                "year": query_year,
                "source_corpus": source,
                "included": included,
                "construction_method": construction,
                "construction_group": "LLM-assisted" if uses_llm else "Non-LLM/other",
                "query_dcu_id": attr.get("query_acu_id"),
                "query_dcu_type": dcu_type if dcu_type in TYPE_ORDER else "other",
                "label": label,
                "evidence_adequacy": attr.get("evidence_adequacy") or "unknown",
                "missing_prior_risk": attr.get("missing_prior_risk") or "unknown",
                "importance": attr.get("importance") or "unknown",
                "delta_type": attr.get("delta_type") or "other",
            }
            dcu_rows.append(dcu)
            if included:
                rec["pass_dcus"] += 1
    return pd.DataFrame(record_rows), pd.DataFrame(dcu_rows)


def proportion_ci_by_record(
    df: pd.DataFrame,
    *,
    group_col: str,
    condition_col: str,
    condition_value: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group, g in df.groupby(group_col):
        # Bootstrap over dataset records. Pre-aggregate numerator/denominator by
        # record so each bootstrap replicate only resamples small numeric arrays.
        per_record = (
            g.assign(_hit=(g[condition_col] == condition_value).astype(int))
            .groupby("bank_id")["_hit"]
            .agg(["sum", "count"])
            .reset_index()
        )
        if per_record.empty:
            continue
        sums = per_record["sum"].to_numpy(dtype=float)
        counts = per_record["count"].to_numpy(dtype=float)
        n_records = len(per_record)
        point = float(sums.sum() / counts.sum()) if counts.sum() else 0.0
        idx = rng.integers(0, n_records, size=(n_boot, n_records))
        boot_den = counts[idx].sum(axis=1)
        boot_num = sums[idx].sum(axis=1)
        boot = np.divide(boot_num, boot_den, out=np.zeros_like(boot_num, dtype=float), where=boot_den != 0)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            group_col: group,
            "n_records": n_records,
            "n_dcus": len(g),
            f"{condition_value}_rate": point,
            "ci_low": float(lo),
            "ci_high": float(hi),
        })
    return pd.DataFrame(rows)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def save_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def esc(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(
    x: float,
    y: float,
    text: Any,
    *,
    size: int = 12,
    weight: str = "normal",
    anchor: str = "start",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" font-weight="{weight}" '
        f'text-anchor="{anchor}"{transform}>{esc(text)}</text>'
    )


def svg_rect(x: float, y: float, w: float, h: float, fill: str, *, stroke: str = "none", opacity: float = 1.0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0, w):.1f}" height="{max(0, h):.1f}" fill="{fill}" stroke="{stroke}" opacity="{opacity:.2f}"/>'


def svg_line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = "#333", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}"/>'


def write_svg(path: Path, width: int, height: int, elements: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *elements,
        "</svg>",
    ]
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")


def summarize_overview(records: pd.DataFrame, dcus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    splits: list[tuple[str, pd.Series]] = []
    for year in sorted(records["year"].dropna().unique()):
        splits.append((str(year), records["year"] == year))
    for source in ["acl", "arxiv"]:
        splits.append((source.upper(), records["source_corpus"] == source))
    splits.append(("Overall", pd.Series([True] * len(records), index=records.index)))
    for name, mask in splits:
        rec = records[mask]
        dcu = dcus[dcus["bank_id"].isin(set(rec["bank_id"]))]
        pass_rec = rec[rec["included"]]
        pass_dcu = dcu[dcu["included"]]
        label_counts = pass_dcu["label"].value_counts().to_dict()
        rows.append({
            "split": name,
            "records": len(rec),
            "query_dcus": len(dcu),
            "adequacy_pass_records": len(pass_rec),
            "adequacy_pass_dcus": len(pass_dcu),
            "pass_rate": len(pass_rec) / len(rec) if len(rec) else 0,
            "covered_rate": label_counts.get("covered", 0) / len(pass_dcu) if len(pass_dcu) else 0,
            "partial_rate": label_counts.get("partially_covered", 0) / len(pass_dcu) if len(pass_dcu) else 0,
            "not_covered_rate": label_counts.get("not_covered", 0) / len(pass_dcu) if len(pass_dcu) else 0,
        })
    return pd.DataFrame(rows)


def label_profile(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, g in df.groupby(group_col):
        counts = g["label"].value_counts().to_dict()
        total = len(g)
        row = {group_col: group, "n_dcus": total, "n_records": g["bank_id"].nunique()}
        for label in LABEL_ORDER:
            row[label] = counts.get(label, 0)
            row[f"{label}_rate"] = counts.get(label, 0) / total if total else 0
        rows.append(row)
    return pd.DataFrame(rows)



def plot_type_profile(type_profile: pd.DataFrame, ci: pd.DataFrame, out: Path) -> None:
    plot_df = type_profile.set_index("query_dcu_type").reindex([t for t in TYPE_ORDER if t in set(type_profile["query_dcu_type"])])
    labels = ["covered", "partially_covered", "not_covered"]
    colors = {"covered": "#4C78A8", "partially_covered": "#72B7B2", "not_covered": "#F58518"}
    ci_df = ci.set_index("query_dcu_type").reindex(plot_df.index)
    elements: list[str] = []
    width, height = 1100, 430
    elements.append(svg_text(40, 30, "Evidence-coverage profile by DCU type", size=16, weight="bold"))
    elements.append(svg_text(660, 30, "Not-covered rate with bootstrap 95% CI", size=16, weight="bold"))
    chart_x, chart_y, chart_w, chart_h = 60, 70, 520, 240
    bar_w = chart_w / max(1, len(plot_df)) * 0.72
    gap = chart_w / max(1, len(plot_df))
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        x = chart_x + i * gap + (gap - bar_w) / 2
        y0 = chart_y + chart_h
        for label in labels:
            h = chart_h * float(row.get(f"{label}_rate", 0))
            y0 -= h
            elements.append(svg_rect(x, y0, bar_w, h, colors[label]))
        elements.append(svg_text(x + bar_w / 2, chart_y + chart_h + 18, idx, size=10, anchor="end", rotate=-35))
        elements.append(svg_text(x + bar_w / 2, chart_y + chart_h + 45, f"N={int(row['n_dcus'])}", size=9, anchor="middle"))
    elements.append(svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h))
    for j, label in enumerate(labels):
        elements.append(svg_rect(60 + j * 150, 350, 14, 14, colors[label]))
        elements.append(svg_text(80 + j * 150, 362, LABEL_DISPLAY[label], size=11))

    bx, by, bw, row_h = 720, 70, 300, 28
    max_hi = max(0.75, float(ci_df["ci_high"].max()) + 0.05)
    for i, (idx, row) in enumerate(ci_df.iterrows()):
        y = by + i * row_h
        rate = float(row["not_covered_rate"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        elements.append(svg_text(640, y + 16, idx, size=10, anchor="end"))
        elements.append(svg_rect(bx, y + 4, bw * rate / max_hi, 14, "#F58518", opacity=0.9))
        elements.append(svg_line(bx + bw * lo / max_hi, y + 11, bx + bw * hi / max_hi, y + 11, stroke="#111"))
        elements.append(svg_line(bx + bw * lo / max_hi, y + 6, bx + bw * lo / max_hi, y + 16, stroke="#111"))
        elements.append(svg_line(bx + bw * hi / max_hi, y + 6, bx + bw * hi / max_hi, y + 16, stroke="#111"))
        elements.append(svg_text(bx + bw + 8, y + 16, f"{100*rate:.1f}%", size=10))
    elements.append(svg_line(bx, by + len(ci_df) * row_h + 4, bx + bw, by + len(ci_df) * row_h + 4))
    write_svg(out, width, height, elements)


def plot_construction(records: pd.DataFrame, dcus: pd.DataFrame, out: Path) -> None:
    years = sorted(records["year"].dropna().unique())
    llm_share = records.groupby("year")["uses_llm_synthetic_generation"].mean().reindex(years)
    llm_reporting_features = [
        "model_family_named",
        "human_verification_reported",
        "quality_control_reported",
        "documentation_beyond_paper",
    ]
    llm_records = records[records["uses_llm_synthetic_generation"]]
    reporting_rates = {
        feature: float(llm_records[feature].mean()) if len(llm_records) else 0.0
        for feature in llm_reporting_features
    }
    inc = dcus[dcus["included"]]
    profile = label_profile(inc, "construction_group").set_index("construction_group").reindex(["LLM-assisted", "Non-LLM/other"]).dropna(how="all")
    label_colors = {"covered": "#4C78A8", "partially_covered": "#72B7B2", "not_covered": "#F58518"}
    elements: list[str] = []
    width, height = 1160, 430
    elements.append(svg_text(45, 30, "LLM-assisted construction by year", size=16, weight="bold"))
    elements.append(svg_text(415, 30, "Reporting among LLM-assisted records", size=16, weight="bold"))
    elements.append(svg_text(800, 30, "Evidence profile by construction group", size=16, weight="bold"))
    chart_x, chart_y, chart_w, chart_h = 70, 70, 250, 230
    bar_w = 58
    for i, year in enumerate(years):
        x = chart_x + i * 105
        val = float(llm_share.loc[year])
        h = chart_h * val
        elements.append(svg_rect(x, chart_y + chart_h - h, bar_w, h, "#54A24B"))
        elements.append(svg_text(x + bar_w / 2, chart_y + chart_h - h - 8, f"{100*val:.1f}%", size=10, anchor="middle"))
        elements.append(svg_text(x + bar_w / 2, chart_y + chart_h + 22, year, size=12, anchor="middle"))
    elements.append(svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h))

    rx, ry, rw, row_h = 590, 76, 190, 42
    display = {
        "model_family_named": "Model family named",
        "human_verification_reported": "Human verification",
        "quality_control_reported": "Quality control",
        "documentation_beyond_paper": "Docs beyond paper",
    }
    for i, feature in enumerate(llm_reporting_features):
        y = ry + i * row_h
        rate = reporting_rates[feature]
        elements.append(svg_text(rx - 8, y + 17, display[feature], size=10, anchor="end"))
        elements.append(svg_rect(rx, y + 4, rw * rate, 17, "#4C78A8"))
        elements.append(svg_text(rx + rw + 8, y + 18, f"{100*rate:.1f}%", size=10))

    bx, by, bw, row_h2 = 870, 82, 210, 46
    for i, group in enumerate(profile.index):
        y = by + i * row_h2
        elements.append(svg_text(bx - 10, y + 17, group, size=10, anchor="end"))
        x0 = bx
        for label in ["covered", "partially_covered", "not_covered"]:
            val = float(profile.loc[group, f"{label}_rate"]) if group in profile.index else 0
            w = bw * val
            elements.append(svg_rect(x0, y + 4, w, 16, label_colors[label]))
            x0 += w
        elements.append(svg_text(bx + bw + 8, y + 16, f"N={int(profile.loc[group, 'n_dcus'])}", size=9))
    for j, label in enumerate(["covered", "partially_covered", "not_covered"]):
        elements.append(svg_rect(790 + j * 118, 300, 12, 12, label_colors[label]))
        elements.append(svg_text(808 + j * 118, 311, LABEL_DISPLAY[label], size=10))
    write_svg(out, width, height, elements)


def plot_release_governance(records: pd.DataFrame, out: Path) -> None:
    features = [
        "released", "open_access", "license_specified", "source_or_origin_disclosed", "quality_control_reported",
        "human_verification_reported", "documentation_beyond_paper", "ethics_discussed", "pii_discussed",
        "consent_discussed", "copyright_discussed", "bias_or_fairness_discussed",
    ]
    rows = []
    for feature in features:
        for year, g in records.groupby("year"):
            rows.append({"feature": feature, "year": year, "rate": float(g[feature].mean())})
    heat = pd.DataFrame(rows).pivot(index="feature", columns="year", values="rate").reindex(features)
    pass_rows = []
    for feature in features:
        yes = records[records[feature]]
        no = records[~records[feature]]
        pass_rows.append({
            "feature": feature,
            "with_feature_pass_rate": float(yes["included"].mean()) if len(yes) else math.nan,
            "without_feature_pass_rate": float(no["included"].mean()) if len(no) else math.nan,
            "n_with": len(yes),
            "n_without": len(no),
        })
    plot = pd.DataFrame(pass_rows).sort_values("with_feature_pass_rate", ascending=True)
    elements: list[str] = []
    width, height = 1220, 650
    elements.append(svg_text(40, 30, "Reporting/documentation rates by year", size=16, weight="bold"))
    elements.append(svg_text(670, 30, "Adequacy pass rate by documentation feature", size=16, weight="bold"))
    hx, hy, cell_w, cell_h = 260, 60, 58, 26
    for j, year in enumerate(heat.columns):
        elements.append(svg_text(hx + j * cell_w + cell_w / 2, hy - 10, year, size=11, anchor="middle"))
    for i, feature in enumerate(features):
        y = hy + i * cell_h
        elements.append(svg_text(hx - 8, y + 17, feature.replace("_", " "), size=9, anchor="end"))
        for j, year in enumerate(heat.columns):
            rate = float(heat.loc[feature, year])
            blue = int(245 - 140 * rate)
            fill = f"rgb({blue},{min(255, blue+5)},255)"
            elements.append(svg_rect(hx + j * cell_w, y, cell_w - 2, cell_h - 2, fill, stroke="#fff"))
            elements.append(svg_text(hx + j * cell_w + cell_w / 2, y + 17, f"{100*rate:.0f}", size=8, anchor="middle"))

    bx, by, bw, row_h = 820, 60, 240, 28
    for i, row in enumerate(plot.itertuples(index=False)):
        y = by + i * row_h
        feature = str(row.feature)
        with_rate = float(row.with_feature_pass_rate) if not pd.isna(row.with_feature_pass_rate) else 0
        without_rate = float(row.without_feature_pass_rate) if not pd.isna(row.without_feature_pass_rate) else 0
        elements.append(svg_text(bx - 8, y + 18, feature.replace("_", " "), size=9, anchor="end"))
        elements.append(svg_rect(bx, y + 3, bw * without_rate, 9, "#BAB0AC"))
        elements.append(svg_rect(bx, y + 15, bw * with_rate, 9, "#4C78A8"))
        elements.append(svg_text(bx + bw + 8, y + 18, f"{100*with_rate:.0f}/{100*without_rate:.0f}", size=8))
    elements.append(svg_rect(840, 580, 12, 12, "#4C78A8"))
    elements.append(svg_text(858, 591, "with feature", size=10))
    elements.append(svg_rect(950, 580, 12, 12, "#BAB0AC"))
    elements.append(svg_text(968, 591, "without feature", size=10))
    write_svg(out, width, height, elements)


# Matplotlib replacements for paper-ready PNG/PDF figures.  These definitions
# intentionally override the compact SVG debug plots above.
COVERAGE_COLORS = {
    "covered": "#4C78A8",
    "partially_covered": "#72B7B2",
    "not_covered": "#F58518",
}
TEXT = "#222222"
GRID = "#E6E6E6"

TYPE_LABELS = {
    "task/domain": "Task/domain",
    "data/source": "Data/source",
    "annotation/protocol": "Annotation\nprotocol",
    "scale/coverage": "Scale/coverage",
    "evaluation/use": "Evaluation/use",
    "availability/quality": "Availability\nquality",
    "governance/ethics": "Governance\nethics",
    "other": "Other",
}

DOC_FEATURES = [
    ("released", "Released"),
    ("open_access", "Open access"),
    ("license_specified", "License specified"),
    ("source_or_origin_disclosed", "Source/origin disclosed"),
    ("quality_control_reported", "Quality control reported"),
    ("human_verification_reported", "Human verification reported"),
    ("documentation_beyond_paper", "Docs beyond paper"),
    ("ethics_discussed", "Ethics discussed"),
    ("pii_discussed", "PII discussed"),
    ("consent_discussed", "Consent discussed"),
    ("copyright_discussed", "Copyright discussed"),
    ("bias_or_fairness_discussed", "Bias/fairness discussed"),
]


def _setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_figure(fig: plt.Figure, out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    if out_path.suffix.lower() != ".png":
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _pct_label(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.0f}%"


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    vals = series.astype(str).str.lower().str.strip()
    return vals.isin({"true", "yes", "y", "1", "released", "open", "clear", "present", "specified"})


def _included_mask(df: pd.DataFrame) -> pd.Series:
    col = _first_existing(df, ["included", "adequacy_pass", "passes_adequacy", "is_included"])
    if col is None:
        return pd.Series(True, index=df.index)
    return _truthy_series(df[col])


def _coverage_profile_for_plot(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns or "label" not in df.columns:
        return pd.DataFrame(columns=[group_col, "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"])
    tmp = df.copy()
    tmp["_label"] = tmp["label"].astype(str).str.lower().str.replace("-", "_")
    rows = []
    for group, sub in tmp.groupby(group_col, dropna=False):
        n = len(sub)
        rows.append({
            group_col: group,
            "n_dcus": n,
            "covered_rate": float((sub["_label"] == "covered").mean()),
            "partially_covered_rate": float(sub["_label"].isin(["partially_covered", "partial", "partially covered"]).mean()),
            "not_covered_rate": float(sub["_label"].isin(["not_covered", "unsupported", "not covered"]).mean()),
        })
    return pd.DataFrame(rows)


def _plot_stacked_barh(ax: Any, df: pd.DataFrame, y_col: str, title: str, *, n_col: str = "n_dcus") -> None:
    labels = df[y_col].tolist()
    y = np.arange(len(labels))
    left = np.zeros(len(df))
    for key, label in [
        ("covered_rate", "Covered"),
        ("partially_covered_rate", "Partially covered"),
        ("not_covered_rate", "Not covered"),
    ]:
        vals = df[key].fillna(0).to_numpy()
        ax.barh(y, vals, left=left, height=0.64, color=COVERAGE_COLORS[key.replace("_rate", "")], label=label)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_title(title, loc="left", fontweight="bold")
    for i, row in df.reset_index(drop=True).iterrows():
        n_txt = f"N={int(row[n_col]):,}" if n_col in row and pd.notna(row[n_col]) else ""
        ax.text(1.02, i, f"{_pct_label(row.get('not_covered_rate', np.nan))}  {n_txt}", va="center", ha="left", fontsize=7.5, color=TEXT)
    ax.set_xlabel("Share of query DCUs")


def plot_type_profile(type_prof: pd.DataFrame, type_ci: pd.DataFrame, out_path: Path | str, *, min_n: int = 100) -> None:
    _setup_matplotlib()
    df = type_prof.copy()
    df = df[df["query_dcu_type"].isin(TYPE_ORDER)].copy()
    df["order"] = df["query_dcu_type"].map({t: i for i, t in enumerate(TYPE_ORDER)})
    df = df[(df["query_dcu_type"] != "other") & (df["n_dcus"] >= min_n)].sort_values("order")
    df["display"] = df["query_dcu_type"].map(TYPE_LABELS).fillna(df["query_dcu_type"])
    ci = type_ci.copy() if type_ci is not None else pd.DataFrame()
    if not ci.empty:
        ci = ci[[c for c in ["query_dcu_type", "ci_low", "ci_high", "not_covered_rate"] if c in ci.columns]]
        df = df.merge(ci, on="query_dcu_type", how="left", suffixes=("", "_ci"))
    for col in ["ci_low", "ci_high"]:
        if col not in df.columns:
            df[col] = df["not_covered_rate"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.34}
    )
    _plot_stacked_barh(ax1, df, "display", "A. Evidence profile by contribution type")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3, frameon=False)

    y = np.arange(len(df))
    x = df["not_covered_rate"].to_numpy()
    lo = np.maximum(0, x - df["ci_low"].to_numpy())
    hi = np.maximum(0, df["ci_high"].to_numpy() - x)
    ax2.errorbar(
        x,
        y,
        xerr=[lo, hi],
        fmt="o",
        color=COVERAGE_COLORS["not_covered"],
        ecolor=TEXT,
        elinewidth=1.0,
        capsize=2.5,
        markersize=4.2,
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(0.55, float(np.nanmax(df["ci_high"].fillna(0.0))) + 0.05))
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.grid(axis="x", color=GRID, linewidth=0.7)
    ax2.set_xlabel("Not-covered rate")
    ax2.set_title("B. Not-covered rate\n(bootstrap 95% CI)", loc="left", fontweight="bold")
    for i, val in enumerate(x):
        ax2.text(ax2.get_xlim()[1] + 0.01, i, _pct_label(val), va="center", ha="left", fontsize=7.5, color=TEXT)
    fig.suptitle(
        "Added information is concentrated in specific contribution dimensions",
        x=0.01,
        y=1.04,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    _save_figure(fig, out_path)


def _infer_llm_assisted(records: pd.DataFrame) -> pd.Series:
    col = _first_existing(records, ["llm_assisted", "uses_llm_synthetic_generation", "uses_llm", "synthetic_uses_llm", "construction_uses_llm"])
    if col is not None:
        return _truthy_series(records[col])
    candidates = [c for c in records.columns if "model" in c.lower() or "construction" in c.lower() or "synthetic" in c.lower()]
    if not candidates:
        return pd.Series(False, index=records.index)
    text = records[candidates].astype(str).agg(" ".join, axis=1).str.lower()
    return text.str.contains(r"\b(gpt|chatgpt|claude|llama|gemini|palm|mistral|qwen|llm|large language model)\b", regex=True)


def _get_year_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing(df, ["year", "query_year", "paper_year"])


def plot_construction(records: pd.DataFrame, dcus: pd.DataFrame, out_path: Path | str) -> None:
    _setup_matplotlib()
    rec = records.copy()
    rec = rec[_included_mask(rec)].copy()
    if rec.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No adequacy-passing records", ha="center", va="center")
        ax.axis("off")
        _save_figure(fig, out_path)
        return

    year_col = _get_year_col(rec)
    rec["_llm_assisted"] = _infer_llm_assisted(rec)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [0.95, 1.45], "wspace": 0.48})

    if year_col:
        by_year = rec.groupby(year_col)["_llm_assisted"].agg(records="size", llm_share="mean").reset_index().sort_values(year_col)
        ax1.plot(by_year[year_col], by_year["llm_share"], marker="o", color="#59A14F", linewidth=2.0)
        ax1.fill_between(by_year[year_col], by_year["llm_share"], color="#59A14F", alpha=0.12)
        for _, row in by_year.iterrows():
            ax1.text(row[year_col], row["llm_share"] + 0.035, _pct_label(row["llm_share"]), ha="center", fontsize=8)
        ax1.set_xticks(by_year[year_col].tolist())
        ax1.set_ylim(0, min(1.0, max(0.75, float(by_year["llm_share"].max()) + 0.15)))
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax1.grid(axis="y", color=GRID, linewidth=0.7)
        ax1.set_ylabel("Share of dataset records")
        ax1.set_title("A. LLM-assisted share", loc="left", fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "Year column not found", ha="center", va="center")
        ax1.axis("off")

    d = dcus.copy()
    if not d.empty:
        if "included" in d.columns:
            d = d[_included_mask(d)].copy()
        group_col = None
        if "construction_group" in d.columns:
            group_col = "construction_group"
            d["construction_group"] = d["construction_group"].replace({"Non-LLM/other": "Other", "non-llm/other": "Other"})
        elif "llm_assisted" in d.columns:
            d["construction_group"] = np.where(_truthy_series(d["llm_assisted"]), "LLM-assisted", "Other")
            group_col = "construction_group"
        if group_col:
            prof = _coverage_profile_for_plot(d, group_col)
            prof[group_col] = prof[group_col].replace({True: "LLM-assisted", False: "Other", "Non-LLM/other": "Other"})
            prof = prof[prof[group_col].isin(["LLM-assisted", "Other"])].copy()
            prof["order"] = prof[group_col].map({"LLM-assisted": 0, "Other": 1}).fillna(99)
            prof = prof.sort_values("order").rename(columns={group_col: "display"})
            _plot_stacked_barh(ax2, prof, "display", "B. Evidence profile by group")
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=3, frameon=False)
        else:
            ax2.text(0.5, 0.5, "No LLM group available for DCUs", ha="center", va="center")
            ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "No DCU records", ha="center", va="center")
        ax2.axis("off")

    fig.suptitle(
        "LLM-mediated construction is growing, but contribution profiles remain claim-specific",
        x=0.01,
        y=1.04,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    _save_figure(fig, out_path)


def _feature_series(records: pd.DataFrame, key: str) -> pd.Series:
    if key in records.columns:
        return _truthy_series(records[key])
    if key == "released":
        col = _first_existing(records, ["release_status", "availability.release_status"])
        return records[col].astype(str).str.lower().str.contains("released|available|public") if col else pd.Series(False, index=records.index)
    if key == "open_access":
        col = _first_existing(records, ["access_restrictions", "availability.access_restrictions"])
        return ~records[col].astype(str).str.lower().str.contains("gated|restricted|request|unclear|unknown") if col else pd.Series(False, index=records.index)
    if key == "license_specified":
        col = _first_existing(records, ["license", "availability.license"])
        return ~records[col].astype(str).str.lower().isin(["", "unclear", "unknown", "none", "nan"]) if col else pd.Series(False, index=records.index)
    if key == "source_or_origin_disclosed":
        col = _first_existing(records, ["source_data_origin", "construction.source_data_origin"])
        return ~records[col].astype(str).str.lower().isin(["", "unclear", "unknown", "none", "nan"]) if col else pd.Series(False, index=records.index)
    if key == "quality_control_reported":
        col = _first_existing(records, ["quality_control", "construction.quality_control"])
        return ~records[col].astype(str).str.lower().isin(["", "unclear", "unknown", "none", "nan"]) if col else pd.Series(False, index=records.index)
    if key == "human_verification_reported":
        col = _first_existing(records, ["human_verification", "synthetic_human_verification", "construction.synthetic_generation.human_verification"])
        return records[col].astype(str).str.lower().str.contains("human|manual|expert|verified|yes|clear") if col else pd.Series(False, index=records.index)
    if key == "documentation_beyond_paper":
        col = _first_existing(records, ["documentation_type", "availability.documentation_type"])
        return ~records[col].astype(str).str.lower().isin(["", "unclear", "unknown", "none", "paper", "paper_only", "nan"]) if col else pd.Series(False, index=records.index)
    gov_map = {
        "ethics_discussed": ["ethics_discussed", "governance.ethics_discussed"],
        "pii_discussed": ["pii_discussed", "governance.pii_discussed"],
        "consent_discussed": ["consent_discussed", "governance.consent_discussed"],
        "copyright_discussed": ["copyright_discussed", "governance.copyright_discussed"],
        "bias_or_fairness_discussed": ["bias_or_fairness_discussed", "governance.bias_or_fairness_discussed"],
    }
    if key in gov_map:
        col = _first_existing(records, gov_map[key])
        return records[col].astype(str).str.lower().str.contains("yes|discussed|clear|present|true") if col else pd.Series(False, index=records.index)
    return pd.Series(False, index=records.index)


def plot_release_governance(records: pd.DataFrame, out_path: Path | str) -> None:
    _setup_matplotlib()
    rec = records.copy()
    if rec.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No records", ha="center", va="center")
        ax.axis("off")
        _save_figure(fig, out_path)
        return

    year_col = _get_year_col(rec)
    features = [(key, label) for key, label in DOC_FEATURES]
    for key, _ in features:
        rec[f"_feat_{key}"] = _feature_series(rec, key)
    rec["_doc_score"] = rec[[f"_feat_{key}" for key, _ in features]].sum(axis=1)
    rec["_included"] = _included_mask(rec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 4.2), gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.68})

    if year_col:
        years = sorted([year for year in rec[year_col].dropna().unique()])
    else:
        years = ["All"]
        rec["_year_all"] = "All"
        year_col = "_year_all"

    heat = []
    ylabels = []
    for key, label in features:
        row = []
        for year in years:
            sub = rec[rec[year_col] == year]
            row.append(float(sub[f"_feat_{key}"].mean()) if len(sub) else np.nan)
        heat.append(row)
        ylabels.append(label)
    heat_arr = np.array(heat)
    im = ax1.imshow(heat_arr, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax1.set_xticks(np.arange(len(years)))
    ax1.set_xticklabels([str(int(year)) if isinstance(year, (int, float, np.integer)) else str(year) for year in years])
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_yticklabels(ylabels)
    ax1.set_title("A. Reporting/documentation rates", loc="left", fontweight="bold")
    for i in range(len(ylabels)):
        for j in range(len(years)):
            val = heat_arr[i, j]
            txt = "--" if np.isnan(val) else f"{100 * val:.0f}"
            color = "white" if not np.isnan(val) and val > 0.55 else TEXT
            ax1.text(j, i, txt, ha="center", va="center", fontsize=7.3, color=color)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    bins = [-0.1, 2, 4, 6, 20]
    labels = ["0--2", "3--4", "5--6", "7+"]
    rec["_doc_bin"] = pd.cut(rec["_doc_score"], bins=bins, labels=labels)
    grouped = rec.groupby("_doc_bin", observed=False).agg(records=("_included", "size"), pass_rate=("_included", "mean")).reset_index()
    grouped = grouped[grouped["records"] > 0]
    x = np.arange(len(grouped))
    ax2.bar(x, grouped["pass_rate"], color="#4C78A8", width=0.62)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in grouped["_doc_bin"]])
    ax2.set_ylim(0, min(1, max(0.75, float(grouped["pass_rate"].max()) + 0.10)))
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.grid(axis="y", color=GRID, linewidth=0.7)
    ax2.set_xlabel("Documentation/reporting features present")
    ax2.set_ylabel("Adequacy pass rate", labelpad=10)
    ax2.set_title("B. Documentation and auditability", loc="left", fontweight="bold")
    for i, row in grouped.iterrows():
        ax2.text(i, row["pass_rate"] + 0.025, f"{100 * row['pass_rate']:.0f}%\nN={int(row['records']):,}", ha="center", va="bottom", fontsize=7.2)

    fig.suptitle(
        "Release and governance reporting shape whether dataset contributions are auditable",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    _save_figure(fig, out_path)


def markdown_table(df: pd.DataFrame, cols: list[str], *, max_rows: int | None = None) -> str:
    show = df[cols].copy()
    if max_rows:
        show = show.head(max_rows)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if col in {"year", "records", "llm_records", "n_records", "n_dcus", "n_with", "n_without"} and not pd.isna(val):
                vals.append(str(int(val)))
            elif isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-bank-jsonl", default="data/census/integrated_fulltext_dataset_bank_2023_2025.jsonl")
    parser.add_argument("--attribution-jsonl", action="append", default=[
        "data/census/integrated_dcu_native_attribution_2024_gemini31_flashlite_top50_datasetcompact_v5.jsonl",
        "data/census/integrated_dcu_native_attribution_2025_gemini31_flashlite_top50_datasetcompact_v5.jsonl",
    ])
    parser.add_argument("--output-dir", default="data/census/census_scale_analysis")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reuse-flattened", action="store_true", help="Reuse record_level.csv and dcu_level.csv from output dir.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    record_csv = out_dir / "record_level.csv"
    dcu_csv = out_dir / "dcu_level.csv"
    if args.reuse_flattened and record_csv.exists() and dcu_csv.exists():
        print("Reusing flattened CSVs...", flush=True)
        records = pd.read_csv(record_csv)
        dcus = pd.read_csv(dcu_csv)
    else:
        print("Loading dataset bank...", flush=True)
        bank_rows = read_jsonl(args.dataset_bank_jsonl)
        bank_by_id = {row.get("bank_id"): row for row in bank_rows}
        attribution_rows: list[dict[str, Any]] = []
        for path in args.attribution_jsonl:
            print(f"Loading attribution rows: {path}", flush=True)
            attribution_rows.extend(read_jsonl(path))

        print("Flattening records/DCUs...", flush=True)
        records, dcus = flatten(attribution_rows, bank_by_id)

        print("Writing flattened CSVs...", flush=True)
        records.to_csv(record_csv, index=False)
        dcus.to_csv(dcu_csv, index=False)

    for col in ["included"]:
        if col in records.columns:
            records[col] = records[col].astype(str).str.lower().isin({"true", "1", "yes"})
        if col in dcus.columns:
            dcus[col] = dcus[col].astype(str).str.lower().isin({"true", "1", "yes"})
    if "source_or_origin_disclosed" not in records.columns and "source_dataset_disclosed" in records.columns:
        records["source_or_origin_disclosed"] = records["source_dataset_disclosed"]
    if "uses_llm_synthetic_generation" not in records.columns:
        records["uses_llm_synthetic_generation"] = records["construction_method"].astype(str).eq("LLM/synthetic")
    if "construction_group" not in records.columns:
        records["construction_group"] = np.where(records["uses_llm_synthetic_generation"], "LLM-assisted", "Non-LLM/other")
    if "construction_group" not in dcus.columns:
        dcus["construction_group"] = np.where(dcus["construction_method"].astype(str).eq("LLM/synthetic"), "LLM-assisted", "Non-LLM/other")
    if "model_family_named" not in records.columns:
        records["model_family_named"] = False
    if "model_families" not in records.columns:
        records["model_families"] = ""
    for col in [
        "uses_llm_synthetic_generation",
        "model_family_named",
        "released",
        "open_access",
        "license_specified",
        "source_or_origin_disclosed",
        "quality_control_reported",
        "human_verification_reported",
        "documentation_beyond_paper",
        *GOVERNANCE_KEYS,
    ]:
        if col in records.columns:
            records[col] = records[col].astype(str).str.lower().isin({"true", "1", "yes"})
    pass_dcus = dcus[dcus["included"]].copy()

    print("Summarizing overview...", flush=True)
    overview = summarize_overview(records, dcus)
    save_csv(out_dir / "census_overview.csv", overview)

    print("Finding 1 summaries and bootstrap...", flush=True)
    type_prof = label_profile(pass_dcus, "query_dcu_type")
    type_prof["sort"] = type_prof["query_dcu_type"].map({v: i for i, v in enumerate(TYPE_ORDER)}).fillna(99)
    type_prof = type_prof.sort_values("sort").drop(columns=["sort"])
    type_ci = proportion_ci_by_record(pass_dcus, group_col="query_dcu_type", condition_col="label", condition_value="not_covered", n_boot=args.bootstrap, seed=args.seed)
    type_ci["sort"] = type_ci["query_dcu_type"].map({v: i for i, v in enumerate(TYPE_ORDER)}).fillna(99)
    type_ci = type_ci.sort_values("sort").drop(columns=["sort"])
    save_csv(out_dir / "finding1_dcu_type_profile.csv", type_prof)
    save_csv(out_dir / "finding1_not_covered_bootstrap_ci.csv", type_ci)

    print("Finding 2 construction summaries...", flush=True)
    construction_counts = records.groupby(["year", "construction_method"]).size().reset_index(name="records")
    construction_counts["year_total"] = construction_counts.groupby("year")["records"].transform("sum")
    construction_counts["share"] = construction_counts["records"] / construction_counts["year_total"]
    save_csv(out_dir / "finding2_construction_by_year.csv", construction_counts)
    construction_profile = label_profile(pass_dcus, "construction_method").sort_values("n_dcus", ascending=False)
    save_csv(out_dir / "finding2_construction_label_profile.csv", construction_profile)
    llm_by_year = (
        records.groupby("year")
        .agg(
            records=("bank_id", "count"),
            llm_records=("uses_llm_synthetic_generation", "sum"),
            llm_share=("uses_llm_synthetic_generation", "mean"),
        )
        .reset_index()
    )
    save_csv(out_dir / "finding2_llm_by_year.csv", llm_by_year)
    llm_reporting_features = [
        "model_family_named",
        "human_verification_reported",
        "quality_control_reported",
        "documentation_beyond_paper",
    ]
    llm_records = records[records["uses_llm_synthetic_generation"]]
    llm_reporting = pd.DataFrame([
        {
            "feature": feature,
            "display": REPORTING_FEATURE_DISPLAY.get(feature, feature.replace("_", " ")),
            "records": len(llm_records),
            "rate": float(llm_records[feature].mean()) if len(llm_records) else math.nan,
        }
        for feature in llm_reporting_features
    ])
    save_csv(out_dir / "finding2_llm_reporting.csv", llm_reporting)
    construction_group_profile = label_profile(pass_dcus, "construction_group").sort_values("n_dcus", ascending=False)
    save_csv(out_dir / "finding2_llm_group_label_profile.csv", construction_group_profile)
    family_counts: Counter[str] = Counter()
    for families in records.loc[records["uses_llm_synthetic_generation"], "model_families"].fillna(""):
        for family in str(families).split(";"):
            family = family.strip()
            if family:
                family_counts[family] += 1
    model_families = pd.DataFrame([
        {"model_family": family, "records": count}
        for family, count in family_counts.most_common()
    ])
    save_csv(out_dir / "finding2_model_families.csv", model_families)

    print("Finding 3 documentation/governance summaries...", flush=True)
    doc_features = [
        "released", "open_access", "license_specified", "source_or_origin_disclosed", "quality_control_reported",
        "human_verification_reported", "documentation_beyond_paper", *GOVERNANCE_KEYS,
    ]
    doc_rows = []
    for feature in doc_features:
        for year, g in records.groupby("year"):
            doc_rows.append({
                "feature": feature,
                "year": year,
                "records": len(g),
                "rate": float(g[feature].mean()),
            })
    doc_by_year = pd.DataFrame(doc_rows)
    save_csv(out_dir / "finding3_release_governance_by_year.csv", doc_by_year)
    doc_pass_rows = []
    for feature in doc_features:
        yes = records[records[feature]]
        no = records[~records[feature]]
        doc_pass_rows.append({
            "feature": feature,
            "n_with": len(yes),
            "n_without": len(no),
            "with_feature_pass_rate": float(yes["included"].mean()) if len(yes) else math.nan,
            "without_feature_pass_rate": float(no["included"].mean()) if len(no) else math.nan,
            "pass_rate_delta": (
                float(yes["included"].mean()) - float(no["included"].mean())
                if len(yes) and len(no) else math.nan
            ),
        })
    doc_pass = pd.DataFrame(doc_pass_rows).sort_values("pass_rate_delta", ascending=False)
    save_csv(out_dir / "finding3_documentation_vs_adequacy.csv", doc_pass)

    print("Sensitivity summaries...", flush=True)
    sensitivity_rows = []
    settings = {
        "main": (0.50, 0.34, 0.50),
        "strict": (0.25, 0.20, 0.75),
        "loose": (0.75, 0.50, 0.50),
    }
    for name, (max_low, max_risk, min_interp) in settings.items():
        mask = records.apply(lambda r: inclusion_under(r.to_dict(), max_low=max_low, max_risk=max_risk, min_interp=min_interp), axis=1)
        ids = set(records.loc[mask, "bank_id"])
        sd = dcus[dcus["bank_id"].isin(ids)]
        counts = sd["label"].value_counts().to_dict()
        sensitivity_rows.append({
            "setting": name,
            "max_low_adequacy_rate": max_low,
            "max_high_risk_rate": max_risk,
            "min_interpretable_rate": min_interp,
            "records": int(mask.sum()),
            "record_pass_rate": float(mask.mean()),
            "dcus": len(sd),
            "covered_rate": counts.get("covered", 0) / len(sd) if len(sd) else 0,
            "partial_rate": counts.get("partially_covered", 0) / len(sd) if len(sd) else 0,
            "not_covered_rate": counts.get("not_covered", 0) / len(sd) if len(sd) else 0,
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    save_csv(out_dir / "sensitivity_adequacy_thresholds.csv", sensitivity)

    print("Rendering figures...", flush=True)
    plot_type_profile(type_prof, type_ci, fig_dir / "figure3_dcu_type_profile.png")
    plot_construction(records, dcus, fig_dir / "figure4_llm_construction.png")
    plot_release_governance(records, fig_dir / "figure5_release_governance.png")

    print("Writing reports...", flush=True)
    summary = {
        "inputs": {
            "dataset_bank_jsonl": args.dataset_bank_jsonl,
            "attribution_jsonl": args.attribution_jsonl,
        },
        "n_records": int(len(records)),
        "n_dcus": int(len(dcus)),
        "adequacy_pass_records": int(records["included"].sum()),
        "adequacy_pass_dcus": int(pass_dcus.shape[0]),
        "adequacy_pass_rate": float(records["included"].mean()),
        "overall_pass_label_rates": {
            label: float((pass_dcus["label"] == label).mean()) for label in LABEL_ORDER
        },
        "top_not_covered_types": type_prof.sort_values("not_covered_rate", ascending=False)[
            ["query_dcu_type", "n_dcus", "not_covered_rate"]
        ].to_dict(orient="records"),
        "construction_methods": construction_counts.to_dict(orient="records"),
        "llm_by_year": llm_by_year.to_dict(orient="records"),
        "llm_reporting": llm_reporting.to_dict(orient="records"),
        "llm_group_label_profile": construction_group_profile.to_dict(orient="records"),
        "model_families": model_families.head(12).to_dict(orient="records"),
        "documentation_vs_adequacy_top_deltas": doc_pass.head(8).to_dict(orient="records"),
        "sensitivity": sensitivity.to_dict(orient="records"),
        "outputs": {
            "overview_csv": str(out_dir / "census_overview.csv"),
            "finding1_type_profile_csv": str(out_dir / "finding1_dcu_type_profile.csv"),
            "finding1_ci_csv": str(out_dir / "finding1_not_covered_bootstrap_ci.csv"),
            "finding2_construction_by_year_csv": str(out_dir / "finding2_construction_by_year.csv"),
            "finding2_construction_profile_csv": str(out_dir / "finding2_construction_label_profile.csv"),
            "finding2_llm_by_year_csv": str(out_dir / "finding2_llm_by_year.csv"),
            "finding2_llm_reporting_csv": str(out_dir / "finding2_llm_reporting.csv"),
            "finding2_llm_group_label_profile_csv": str(out_dir / "finding2_llm_group_label_profile.csv"),
            "finding2_model_families_csv": str(out_dir / "finding2_model_families.csv"),
            "finding3_release_governance_by_year_csv": str(out_dir / "finding3_release_governance_by_year.csv"),
            "finding3_documentation_vs_adequacy_csv": str(out_dir / "finding3_documentation_vs_adequacy.csv"),
            "sensitivity_csv": str(out_dir / "sensitivity_adequacy_thresholds.csv"),
            "figures": [
                str(fig_dir / "figure3_dcu_type_profile.png"),
                str(fig_dir / "figure4_llm_construction.png"),
                str(fig_dir / "figure5_release_governance.png"),
            ],
        },
    }
    write_json(out_dir / "census_scale_findings_summary.json", summary)

    report = [
        "# Census-Scale Analysis Report",
        "",
        "## Scope",
        f"- Records processed: {len(records):,}",
        f"- Query DCUs: {len(dcus):,}",
        f"- Adequacy-passing records: {int(records['included'].sum()):,} ({pct(records['included'].mean())})",
        f"- Adequacy-passing DCUs: {len(pass_dcus):,}",
        "",
        "## Overview",
        markdown_table(overview, ["split", "records", "query_dcus", "adequacy_pass_records", "adequacy_pass_dcus", "pass_rate", "covered_rate", "partial_rate", "not_covered_rate"]),
        "",
        "## Finding 1: Added information is dimension-specific",
        "Evidence-coverage profiles vary strongly by DCU type among adequacy-passing records.",
        markdown_table(type_prof, ["query_dcu_type", "n_records", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"]),
        "",
        "Bootstrap CIs for not-covered rates:",
        markdown_table(type_ci, ["query_dcu_type", "n_records", "n_dcus", "not_covered_rate", "ci_low", "ci_high"]),
        "",
        "## Finding 2: Construction practices shift, but contribution deltas are type-specific",
        "LLM-mediated construction is increasingly common, but LLM use alone does not determine the evidence profile.",
        markdown_table(llm_by_year, ["year", "records", "llm_records", "llm_share"]),
        "",
        "Reporting among LLM-assisted records:",
        markdown_table(llm_reporting, ["display", "records", "rate"]),
        "",
        "Evidence profile by LLM construction group:",
        markdown_table(construction_group_profile, ["construction_group", "n_records", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"]),
        "",
        "Most frequent named model families:",
        markdown_table(model_families, ["model_family", "records"], max_rows=10) if not model_families.empty else "_No named model families._",
        "",
        "Construction-method distribution:",
        markdown_table(construction_counts.sort_values(["year", "share"], ascending=[True, False]), ["year", "construction_method", "records", "share"]),
        "",
        "Evidence profile by construction method:",
        markdown_table(construction_profile, ["construction_method", "n_records", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"]),
        "",
        "## Finding 3: Documentation and release practices shape auditability",
        markdown_table(doc_pass, ["feature", "n_with", "n_without", "with_feature_pass_rate", "without_feature_pass_rate", "pass_rate_delta"]),
        "",
        "## Sensitivity",
        markdown_table(sensitivity, ["setting", "records", "record_pass_rate", "dcus", "covered_rate", "partial_rate", "not_covered_rate"]),
        "",
        "## Figure catalog",
        "- `figures/figure3_dcu_type_profile.png`: horizontal evidence-coverage profile by DCU type plus bootstrap CI for not-covered rates.",
        "- `figures/figure4_llm_construction.png`: LLM-assisted construction trend and evidence profile by construction group.",
        "- `figures/figure5_release_governance.png`: reporting heatmap and documentation-score association with adequacy pass rate.",
    ]
    (out_dir / "analysis-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    fig_catalog = [
        "# Figure Catalog",
        "",
        "## Figure 3: DCU type profile",
        "- Purpose: show that evidence coverage is dimension-specific rather than a single dataset-level novelty property.",
        "- Reader should notice: not-covered evidence concentrates unevenly by contribution type.",
        "- Caveat: labels are evidence-indexed and restricted to adequacy-passing records.",
        "",
        "## Figure 4: LLM-mediated construction",
        "- Purpose: connect the rise of LLM-assisted dataset construction to reporting practices and evidence-coverage profiles.",
        "- Reader should notice: LLM-assisted construction becomes common, but the evidence profile remains mostly partially covered rather than globally new.",
        "- Caveat: construction method is an extracted primary category, not a human-validated taxonomy.",
        "",
        "## Figure 5: Release and governance",
        "- Purpose: show reporting/documentation rates and how documentation features relate to interpretable attribution.",
        "- Reader should notice: some documentation features correlate with higher adequacy pass rates.",
        "- Caveat: these are descriptive associations, not causal effects.",
    ]
    (out_dir / "figure-catalog.md").write_text("\n".join(fig_catalog) + "\n", encoding="utf-8")

    stats = [
        "# Stats Appendix",
        "",
        "All confidence intervals are nonparametric bootstrap intervals over dataset records.",
        f"Bootstrap replicates: {args.bootstrap}; seed: {args.seed}.",
        "",
        "No hypothesis tests are reported because these are descriptive census analyses over a constructed paper bank rather than randomized experimental comparisons.",
        "",
        "## Main sensitivity check",
        markdown_table(sensitivity, ["setting", "max_low_adequacy_rate", "max_high_risk_rate", "min_interpretable_rate", "records", "record_pass_rate", "not_covered_rate"]),
    ]
    (out_dir / "stats-appendix.md").write_text("\n".join(stats) + "\n", encoding="utf-8")

    figure_decisions = [
        "# Main-vs-Appendix Figure Decisions",
        "",
        "| Analysis | Placement | Rationale |",
        "| --- | --- | --- |",
        "| Evidence-coverage profile by DCU type | Main | Directly supports the paper thesis that dataset contribution is claim-level and multidimensional. |",
        "| LLM-mediated construction trend + reporting + contribution profile | Main | Timely ecosystem finding tied to construction practice and evidence attribution. |",
        "| Release/governance reporting and adequacy association | Main | Actionable auditability finding; descriptive association only. |",
        "| Scalar score distribution/year/venue landscape | Appendix | Useful descriptive context, but risks pulling the story back to a scalar novelty score. |",
        "| Score drivers or quartile metadata gaps | Appendix | Descriptive and potentially confounded; should not be framed causally. |",
        "| Contribution archetypes | Appendix or short synthesis | Potentially useful, but requires additional explanation of rule/clustering construction. |",
    ]
    (out_dir / "main_appendix_figure_decisions.md").write_text("\n".join(figure_decisions) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
