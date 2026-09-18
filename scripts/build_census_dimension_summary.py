#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABELS = ["covered", "partially_covered", "not_covered"]
DOC_FEATURES = [
    "released",
    "open_access",
    "license_specified",
    "source_or_origin_disclosed",
    "quality_control_reported",
    "human_verification_reported",
    "documentation_beyond_paper",
    "ethics_discussed",
    "pii_discussed",
    "consent_discussed",
    "copyright_discussed",
    "bias_or_fairness_discussed",
]
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


def boolify(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().str.strip().isin({"true", "1", "yes", "y"})


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{100 * x:.1f}%"


def fmt_num(x: Any) -> str:
    if isinstance(x, (float, np.floating)):
        return f"{x:.3f}"
    return str(x)


def md_table(df: pd.DataFrame, cols: list[str], *, max_rows: int | None = None) -> str:
    show = df[cols].copy()
    if max_rows:
        show = show.head(max_rows)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        vals = [fmt_num(row[col]) for col in cols]
        vals = [
            str(int(row[col])) if col in {"year", "records", "llm_records", "n_records", "n_dcus", "n_with", "n_without", "adequacy_pass_records"} and pd.notna(row[col])
            else fmt_num(row[col])
            for col in cols
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def label_profile(dcus: pd.DataFrame, group_cols: str | list[str], *, min_dcus: int = 0) -> pd.DataFrame:
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    rows = []
    for group, g in dcus.groupby(group_cols, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        total = len(g)
        if total < min_dcus:
            continue
        counts = g["label"].value_counts()
        row = {col: val for col, val in zip(group_cols, group)}
        row["n_records"] = g["bank_id"].nunique()
        row["n_dcus"] = total
        for label in LABELS:
            row[f"{label}_rate"] = float(counts.get(label, 0) / total) if total else 0.0
            row[label] = int(counts.get(label, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def dcu_type_spread(type_profile: pd.DataFrame) -> pd.DataFrame:
    major = type_profile[type_profile["query_dcu_type"].ne("other")].copy()
    return major.sort_values("not_covered_rate", ascending=False)


def doc_score(records: pd.DataFrame) -> pd.Series:
    available = [col for col in DOC_FEATURES if col in records.columns]
    if not available:
        return pd.Series(0, index=records.index)
    bools = records[available].apply(boolify)
    return bools.sum(axis=1)


def norm(value: Any) -> str:
    return str(value or "").strip()


def known(value: Any) -> bool:
    text = norm(value).lower()
    return bool(text) and text not in {"unknown", "unclear", "none", "n/a", "nan", "null"}


def coarse_relationship(value: Any) -> str:
    raw = norm(value).lower()
    if raw in {"source_dataset", "training_data", "extended_from", "combined_with"}:
        return "source/reuse"
    if raw in {"comparison_dataset", "comparison_benchmark", "baseline_benchmark", "closest_prior_dataset"}:
        return "comparison/evaluation prior"
    if raw in {"inspired_by", "shared_task"}:
        return "inspired/shared task"
    if raw in {"knowledge_resource", "lexical_resource"}:
        return "knowledge/lexical resource"
    if not raw or raw in {"missing", "unknown", "unclear", "none"}:
        return "unknown"
    return "other"


def coarse_primary_use(value: Any) -> str:
    raw = norm(value).lower()
    if raw in {"benchmarking", "evaluation"}:
        return "evaluation/benchmarking"
    if raw in {"training", "fine_tuning", "pretraining", "instruction_tuning", "instruction_tuning_data", "rlhf_preference", "safety_fine_tuning", "mid_training"}:
        return "training/fine-tuning"
    if raw in {"analysis", "research", "simulation"}:
        return "analysis/research"
    if raw in {"retrieval", "retrieval_augmented_generation", "inference_augmentation"}:
        return "retrieval/RAG"
    if not raw or raw in {"missing", "unknown", "unclear", "none"}:
        return "unknown"
    return "other"


def coarse_resource_type(value: Any) -> str:
    raw = norm(value).lower()
    if raw in {"dataset", "corpus", "annotation_set", "treebank"}:
        return "dataset/corpus"
    if raw == "benchmark":
        return "benchmark"
    if raw in {"knowledge_base", "knowledge_resource", "lexicon", "lexical_resource"}:
        return "knowledge/lexical"
    if raw in {"multimodal_resource"}:
        return "multimodal"
    if raw in {"tool_output", "model_output"}:
        return "model/tool output"
    if not raw or raw in {"missing", "unknown", "unclear", "none"}:
        return "unknown"
    return "other"


def language_group(languages: Any, scale: dict[str, Any] | None = None) -> tuple[int, str]:
    langs = languages if isinstance(languages, list) else []
    clean = [norm(x) for x in langs if known(x)]
    count = len(set(clean))
    if count == 0 and isinstance(scale, dict) and scale.get("num_languages"):
        try:
            count = int(scale.get("num_languages"))
        except Exception:
            count = 0
    lower = {x.lower() for x in clean}
    if count == 0:
        return 0, "unknown"
    if count == 1 and ("english" in lower or "en" in lower):
        return count, "English-only"
    if count == 1:
        return count, "non-English monolingual"
    if count <= 5:
        return count, "multilingual 2-5"
    if count <= 20:
        return count, "multilingual 6-20"
    return count, "multilingual 20+"


def size_value_and_bin(scale: dict[str, Any] | None) -> tuple[float | None, str]:
    if not isinstance(scale, dict):
        return None, "unknown"
    fields = ["num_instances", "num_documents", "num_dialogues", "num_images", "num_audio_hours"]
    vals = []
    for field in fields:
        value = scale.get(field)
        if value is None:
            continue
        try:
            vals.append(float(value))
        except Exception:
            continue
    if not vals:
        return None, "unknown"
    value = max(vals)
    if value < 1_000:
        return value, "<1k"
    if value < 10_000:
        return value, "1k-10k"
    if value < 100_000:
        return value, "10k-100k"
    if value < 1_000_000:
        return value, "100k-1M"
    return value, "1M+"


def bank_llm_assisted(row: dict[str, Any]) -> bool:
    if bool(row.get("uses_llm_synthetic_generation")) or bool(row.get("synthetic_model_names")):
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
    text = " ".join(norm(x).lower() for x in fields)
    return any(
        needle in text
        for needle in ["llm", "synthetic", "generated by", "chatgpt", "gpt-", "claude", "llama", "gemini"]
    )


def flatten_bank_metadata(bank_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record_rows = []
    relationship_rows = []
    task_rows = []
    domain_rows = []
    for row in bank_rows:
        bank_id = row.get("bank_id")
        year = row.get("year")
        scale = row.get("scale") if isinstance(row.get("scale"), dict) else {}
        lang_count, lang_group = language_group(row.get("languages"), scale)
        size_value, size_bin = size_value_and_bin(scale)
        institution = row.get("institution_profile") if isinstance(row.get("institution_profile"), dict) else {}
        paper_sector = institution.get("paper_sector") or row.get("paper_sector") or row.get("sector") or "unknown"
        uses_llm = bank_llm_assisted(row)
        record_rows.append({
            "bank_id": bank_id,
            "bank_year": year,
            "bank_source_corpus": row.get("source_corpus") or "unknown",
            "role": row.get("role") or "unknown",
            "resource_type": row.get("resource_type") or "unknown",
            "resource_group": coarse_resource_type(row.get("resource_type")),
            "primary_use": row.get("primary_use") or "unknown",
            "primary_use_group": coarse_primary_use(row.get("primary_use")),
            "language_count": lang_count,
            "language_group": lang_group,
            "size_value": size_value,
            "size_bin": size_bin,
            "num_domains": len(row.get("domains") or []) or scale.get("num_domains") or 0,
            "num_tasks": len(row.get("tasks") or []),
            "paper_sector_from_bank": paper_sector,
            "bank_uses_llm_synthetic_generation": uses_llm,
        })
        rels = row.get("prior_dataset_mentions") or []
        seen_rel = set()
        for mention in rels:
            if not isinstance(mention, dict):
                continue
            rel = mention.get("relationship_type") or mention.get("relationship") or mention.get("type") or "unknown"
            rel_key = norm(rel).lower() or "unknown"
            relationship_rows.append({
                "bank_id": bank_id,
                "relationship_type": rel_key,
                "relationship_group": coarse_relationship(rel_key),
                "prior_name": mention.get("name") or "",
            })
            seen_rel.add(rel_key)
        if not rels:
            relationship_rows.append({
                "bank_id": bank_id,
                "relationship_type": "no_prior_mention",
                "relationship_group": "no prior mention",
                "prior_name": "",
            })
        for task in row.get("tasks") or []:
            if known(task):
                task_rows.append({"bank_id": bank_id, "task": norm(task).lower()})
        for domain in row.get("domains") or []:
            if known(domain):
                domain_rows.append({"bank_id": bank_id, "domain": norm(domain).lower()})
    return (
        pd.DataFrame(record_rows),
        pd.DataFrame(relationship_rows),
        pd.DataFrame(task_rows),
        pd.DataFrame(domain_rows),
    )


def record_pass_profile(records: pd.DataFrame, group_col: str, *, min_records: int = 0) -> pd.DataFrame:
    rows = []
    for group, g in records.groupby(group_col, dropna=False):
        if len(g) < min_records:
            continue
        rows.append({
            group_col: group,
            "records": len(g),
            "query_dcus": int(g["n_query_dcus"].sum()) if "n_query_dcus" in g else np.nan,
            "adequacy_pass_records": int(g["included"].sum()),
            "pass_rate": float(g["included"].mean()) if len(g) else 0.0,
            "mean_dcus_per_record": float(g["n_query_dcus"].mean()) if "n_query_dcus" in g else np.nan,
        })
    return pd.DataFrame(rows)


def feature_pass_deltas(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in DOC_FEATURES:
        if feature not in records.columns:
            continue
        mask = boolify(records[feature])
        yes = records[mask]
        no = records[~mask]
        if len(yes) == 0 or len(no) == 0:
            continue
        rows.append({
            "feature": feature,
            "n_with": len(yes),
            "n_without": len(no),
            "with_pass_rate": float(yes["included"].mean()),
            "without_pass_rate": float(no["included"].mean()),
            "delta_pp": float(100 * (yes["included"].mean() - no["included"].mean())),
        })
    return pd.DataFrame(rows).sort_values("delta_pp", ascending=False)


def relationship_tables(
    records: pd.DataFrame,
    pass_dcus: pd.DataFrame,
    relationships: pd.DataFrame,
    *,
    group_col: str,
    min_records: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel = relationships.drop_duplicates(["bank_id", group_col]).copy()
    rel_records = rel.merge(records[["bank_id", "included", "uses_llm_synthetic_generation"]], on="bank_id", how="left")
    mention_counts = relationships.groupby(group_col).size().rename("n_mentions").reset_index()
    record_counts = (
        rel_records.groupby(group_col)
        .agg(
            n_records=("bank_id", "nunique"),
            adequacy_pass_records=("included", "sum"),
            record_pass_rate=("included", "mean"),
            llm_share=("uses_llm_synthetic_generation", "mean"),
        )
        .reset_index()
    )
    dcu_rel = pass_dcus.merge(rel[["bank_id", group_col]], on="bank_id", how="inner")
    prof = label_profile(dcu_rel, group_col)
    if prof.empty:
        summary = record_counts.merge(mention_counts, on=group_col, how="left")
        return summary, pd.DataFrame()
    prof = prof.rename(columns={"n_records": "n_attributed_records"})
    dominant = (
        dcu_rel[dcu_rel["label"].eq("not_covered")]
        .groupby([group_col, "query_dcu_type"])
        .size()
        .reset_index(name="not_covered_dcus")
        .sort_values([group_col, "not_covered_dcus"], ascending=[True, False])
        .drop_duplicates(group_col)
        .rename(columns={"query_dcu_type": "dominant_not_covered_type"})
    )
    summary = (
        record_counts.merge(mention_counts, on=group_col, how="left")
        .merge(prof, on=group_col, how="left")
        .merge(dominant[[group_col, "dominant_not_covered_type", "not_covered_dcus"]], on=group_col, how="left")
        .sort_values(["n_records", "n_mentions"], ascending=False)
    )
    summary = summary[summary["n_records"].fillna(0) >= min_records].copy()
    matrix = label_profile(dcu_rel, [group_col, "query_dcu_type"], min_dcus=100)
    return summary, matrix


def top_value_profile(
    records: pd.DataFrame,
    pass_dcus: pd.DataFrame,
    values: pd.DataFrame,
    value_col: str,
    *,
    top_n: int = 15,
    min_records: int = 50,
) -> pd.DataFrame:
    if values.empty:
        return pd.DataFrame()
    top = values.groupby(value_col)["bank_id"].nunique().sort_values(ascending=False).head(top_n).index
    vals = values[values[value_col].isin(top)].drop_duplicates(["bank_id", value_col])
    rec_counts = vals.merge(records[["bank_id", "included", "uses_llm_synthetic_generation"]], on="bank_id", how="left")
    rec_summary = (
        rec_counts.groupby(value_col)
        .agg(
            n_records=("bank_id", "nunique"),
            adequacy_pass_records=("included", "sum"),
            record_pass_rate=("included", "mean"),
            llm_share=("uses_llm_synthetic_generation", "mean"),
        )
        .reset_index()
    )
    d = pass_dcus.merge(vals[["bank_id", value_col]], on="bank_id", how="inner")
    prof = label_profile(d, value_col).rename(columns={"n_records": "n_attributed_records"})
    out = rec_summary.merge(prof, on=value_col, how="left").sort_values("n_records", ascending=False)
    return out[out["n_records"] >= min_records].copy()


def add_doc_bins(records: pd.DataFrame) -> pd.DataFrame:
    out = records.copy()
    out["doc_score"] = doc_score(out)
    out["doc_score_bin"] = pd.cut(
        out["doc_score"],
        bins=[-0.1, 2, 4, 6, 20],
        labels=["0-2", "3-4", "5-6", "7+"],
    )
    return out


def write_summary(
    out_dir: Path,
    *,
    records: pd.DataFrame,
    dcus: pd.DataFrame,
    pass_dcus: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    type_prof = tables["dcu_type_profile"]
    type_x_llm = tables["dcu_type_x_llm"]
    type_x_year = tables["dcu_type_x_year"]
    construction = tables["construction_profile"]
    feature_delta = tables["doc_feature_pass_delta"]
    doc_bin = tables["doc_score_profile"]
    llm_reporting = tables["llm_reporting"]
    bank_llm_by_year = tables["bank_llm_by_year_2023_2025"]
    source_year = tables["year_source_profile"]
    rel_group = tables["relationship_group_profile"]
    rel_matrix = tables["relationship_group_x_type"]
    use_profile = tables["primary_use_group_profile"]
    resource_profile = tables["resource_group_profile"]
    language_profile = tables["language_group_profile"]
    language_scale = tables["language_group_scale_dcus"]
    size_scale = tables["size_bin_scale_dcus"]
    sector_profile = tables["sector_profile"]
    task_profile = tables["top_task_profile"]
    domain_profile = tables["top_domain_profile"]

    major_type = dcu_type_spread(type_prof)
    strong_type_gap = (
        float(major_type["not_covered_rate"].max() - major_type["not_covered_rate"].min())
        if len(major_type) else np.nan
    )
    llm = construction[construction["construction_group"].eq("LLM-assisted")]
    other = construction[construction["construction_group"].eq("Non-LLM/other")]
    llm_delta = np.nan
    if len(llm) and len(other):
        llm_delta = float(llm.iloc[0]["not_covered_rate"] - other.iloc[0]["not_covered_rate"])

    report = [
        "# Census Dimension Summary",
        "",
        "This file summarizes dimensions before deciding which figure combinations belong in the paper.",
        "",
        "## Scope",
        f"- Records: {len(records):,}",
        f"- Query DCUs: {len(dcus):,}",
        f"- Adequacy-passing records: {int(records['included'].sum()):,} ({pct(records['included'].mean())})",
        f"- Adequacy-passing DCUs: {len(pass_dcus):,}",
        "",
        "## Strongest Dimension-Level Signals",
        f"- DCU type has a large not-covered spread across major types: {100 * strong_type_gap:.1f} percentage points.",
        f"- LLM-assisted construction is not a higher-not-covered group in this run: LLM minus non-LLM not-covered delta is {100 * llm_delta:.1f} percentage points.",
        "- Documentation/auditability features show mixed associations; source/origin disclosure and quality-control reporting are the cleanest positive signals.",
        "- Sensitivity matters: not-covered rates depend strongly on adequacy filtering, so paper wording should stay evidence-conditioned.",
        "",
        "## 1. DCU Type Profile",
        "This is the cleanest thesis-aligned dimension: it directly shows that contribution evidence coverage differs by claim type.",
        md_table(
            type_prof,
            ["query_dcu_type", "n_records", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"],
        ),
        "",
        "Recommended use: main paper. Keep `other` out of the main figure; include it only in appendix/table.",
        "",
        "## 2. Year and Source",
        "This is useful as scope/context, but weak as a standalone finding.",
        md_table(
            source_year,
            ["year", "source_corpus", "records", "adequacy_pass_records", "pass_rate", "n_dcus", "not_covered_rate"],
        ),
        "",
        "Recommended use: compact overview table, not a main figure unless paired with another dimension.",
        "",
        "## 3. DCU Type x Year",
        "Use this to check whether Finding 1 is stable across 2024 and 2025.",
        md_table(
            type_x_year.sort_values(["query_dcu_type", "year"]),
            ["query_dcu_type", "year", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"],
        ),
        "",
        "Recommended use: appendix robustness. It is diagnostic, not the cleanest main narrative.",
        "",
        "## 4. Construction Method",
        "Construction categories are useful, but should not be interpreted as novelty categories.",
        md_table(
            tables["construction_method_profile"],
            ["construction_method", "n_records", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"],
        ),
        "",
        "Recommended use: main only if framed as creation-practice shift plus evidence profile, not as a ranking.",
        "",
        "## 5. LLM-Assisted Construction",
        "This is timely and clear: LLM use is rising, but its evidence profile remains mostly partially covered.",
        "Bank-level construction metadata trend, including 2023:",
        md_table(
            bank_llm_by_year,
            ["bank_year", "records", "llm_records", "llm_share"],
        ),
        "",
        "Attribution-run records only:",
        md_table(
            tables["llm_by_year"],
            ["year", "records", "llm_records", "llm_share"],
        ),
        "",
        md_table(
            construction,
            ["construction_group", "n_records", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"],
        ),
        "",
        "Reporting among LLM-assisted records:",
        md_table(llm_reporting, ["feature", "records", "rate"]),
        "",
        "Recommended use: main paper as a two-panel finding. Do not claim LLM use implies more added information.",
        "",
        "## 6. DCU Type x LLM Group",
        "This is the most useful cross-check for the LLM finding: it asks whether LLM-assisted construction changes where not-covered evidence appears.",
        md_table(
            type_x_llm.sort_values(["query_dcu_type", "construction_group"]),
            ["query_dcu_type", "construction_group", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"],
        ),
        "",
        "Recommended use: consider as appendix or as a small companion table if the LLM finding needs more substance.",
        "",
        "## 7. Release/Governance/Documentation Features",
        "Feature-level rates are descriptive. The strongest paper-safe version is auditability, not causality.",
        md_table(feature_delta, ["feature", "n_with", "n_without", "with_pass_rate", "without_pass_rate", "delta_pp"]),
        "",
        "Documentation score bins:",
        md_table(doc_bin, ["doc_score_bin", "records", "adequacy_pass_records", "pass_rate", "mean_dcus_per_record"]),
        "",
        "Recommended use: main only if written as descriptive association with auditability. Avoid causal language.",
        "",
        "## 8. Documentation Score x DCU Type",
        "This checks whether better reporting changes the evidence profile by contribution dimension.",
        md_table(
            tables["doc_bin_x_type"],
            ["doc_score_bin", "query_dcu_type", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"],
            max_rows=80,
        ),
        "",
        "Recommended use: appendix unless a very clean monotonic pattern is visible.",
        "",
        "## 9. Importance",
        "This is useful for sanity-checking whether high-importance claims behave differently.",
        md_table(
            tables["importance_profile"],
            ["importance", "n_records", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"],
        ),
        "",
        "Recommended use: appendix/error analysis.",
        "",
        "## Candidate Main-Paper Combinations",
        "",
        "| Candidate | Evidence strength | Why it works | Risk | Placement |",
        "| --- | --- | --- | --- | --- |",
        "| DCU type profile | Strong | Directly supports claim-level added-information thesis | Must keep evidence-conditioned wording | Main |",
        "| LLM share + LLM evidence profile | Medium-strong | Timely and tied to construction shift | LLM category is extracted, not manually validated | Main or short main |",
        "| Release/governance reporting + adequacy pass | Medium | Actionable auditability angle | Associations are mixed and non-causal | Main if space; otherwise appendix |",
        "| DCU type x LLM group | Medium | Adds substance to LLM finding | More complex table/figure | Appendix or small companion |",
        "| Year/source breakdown | Low as finding | Good scope/context | Dashboard-like if overemphasized | Overview table |",
        "| Scalar score landscape | Low for current story | Exploratory only | Pulls story back to scalar novelty | Appendix only |",
    ]
    extra = [
        "",
        "## 10. Prior Dataset Relationship",
        "This is the most thesis-aligned candidate third finding if relationship labels are clean enough.",
        md_table(
            rel_group,
            [
                "relationship_group",
                "n_mentions",
                "n_records",
                "llm_share",
                "n_dcus",
                "not_covered_rate",
                "dominant_not_covered_type",
            ],
        ),
        "",
        "Prior relationship x DCU type:",
        md_table(
            rel_matrix.sort_values(["relationship_group", "query_dcu_type"]),
            ["relationship_group", "query_dcu_type", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"],
            max_rows=80,
        ),
        "",
        "Recommended use: strong candidate for a third main finding if categories are acceptable after manual spot-checking.",
        "",
        "## 11. Primary Use and Resource Type",
        "These dimensions explain why datasets are created, and are easier to interpret than fine-grained task/domain labels.",
        "Primary use:",
        md_table(use_profile, ["primary_use_group", "n_records", "n_dcus", "llm_share", "covered_rate", "partially_covered_rate", "not_covered_rate"]),
        "",
        "Resource type:",
        md_table(resource_profile, ["resource_group", "n_records", "n_dcus", "llm_share", "covered_rate", "partially_covered_rate", "not_covered_rate"]),
        "",
        "Recommended use: fallback third finding if prior relationship labels are too noisy.",
        "",
        "## 12. Language and Scale",
        "These dimensions should be used to deepen Finding 1, especially scale/coverage claims.",
        "Language group profile:",
        md_table(language_profile, ["language_group", "records", "pass_rate", "n_dcus", "not_covered_rate"]),
        "",
        "Language group restricted to scale/coverage DCUs:",
        md_table(language_scale, ["language_group", "n_records", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"]),
        "",
        "Size bin restricted to scale/coverage DCUs:",
        md_table(size_scale, ["size_bin", "n_records", "n_dcus", "covered_rate", "partially_covered_rate", "not_covered_rate"]),
        "",
        "Recommended use: use as Finding 1 supporting analysis, not as a standalone dashboard.",
        "",
        "## 13. Sector",
        "Sector is currently weak because the integrated bank does not preserve reliable paper-sector labels.",
        md_table(sector_profile, ["paper_sector_from_bank", "records", "pass_rate", "n_dcus", "not_covered_rate"], max_rows=20),
        "",
        "Recommended use: do not use unless sector extraction is rebuilt or recovered from full extraction records.",
        "",
        "## 14. Task and Domain",
        "These are high-cardinality and best treated as appendix scope unless a clear cluster-level finding emerges.",
        "Top tasks:",
        md_table(task_profile, ["task", "n_records", "n_dcus", "llm_share", "not_covered_rate"], max_rows=20),
        "",
        "Top domains:",
        md_table(domain_profile, ["domain", "n_records", "n_dcus", "llm_share", "not_covered_rate"], max_rows=20),
        "",
        "Recommended use: appendix scope table. Avoid a large main-figure dashboard.",
    ]
    report.extend(extra)
    (out_dir / "dimension_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="data/census/census_scale_analysis")
    parser.add_argument("--dataset-bank-jsonl", default="data/census/integrated_fulltext_dataset_bank_2023_2025.jsonl")
    parser.add_argument("--output-dir", default="data/census/census_scale_analysis/dimension_summaries")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(analysis_dir / "record_level.csv")
    dcus = pd.read_csv(analysis_dir / "dcu_level.csv")
    bank_rows = read_jsonl(Path(args.dataset_bank_jsonl))
    bank_meta, relationships, task_values, domain_values = flatten_bank_metadata(bank_rows)
    records["included"] = boolify(records["included"])
    dcus["included"] = boolify(dcus["included"])
    for col in DOC_FEATURES + ["uses_llm_synthetic_generation", "model_family_named"]:
        if col in records.columns:
            records[col] = boolify(records[col])
    if "construction_group" not in records.columns:
        records["construction_group"] = np.where(records.get("uses_llm_synthetic_generation", False), "LLM-assisted", "Non-LLM/other")
    if "construction_group" not in dcus.columns:
        dcus["construction_group"] = np.where(dcus["construction_method"].astype(str).eq("LLM/synthetic"), "LLM-assisted", "Non-LLM/other")
    # Add schema dimensions that are not preserved in the flattened attribution files.
    records = records.merge(bank_meta, on="bank_id", how="left")
    records["primary_use_group"] = records["primary_use_group"].fillna("unknown")
    records["resource_group"] = records["resource_group"].fillna("unknown")
    records["language_group"] = records["language_group"].fillna("unknown")
    records["size_bin"] = records["size_bin"].fillna("unknown")
    records["paper_sector_from_bank"] = records["paper_sector_from_bank"].fillna("unknown")
    dcus = dcus.merge(
        records[[
            "bank_id",
            "primary_use_group",
            "resource_group",
            "language_group",
            "size_bin",
            "paper_sector_from_bank",
        ]],
        on="bank_id",
        how="left",
    )

    pass_dcus = dcus[dcus["included"]].copy()
    pass_records = records[records["included"]].copy()
    records = add_doc_bins(records)
    pass_records = add_doc_bins(pass_records)
    dcus = dcus.merge(records[["bank_id", "doc_score_bin"]], on="bank_id", how="left")
    pass_dcus = pass_dcus.merge(records[["bank_id", "doc_score_bin"]], on="bank_id", how="left")

    tables: dict[str, pd.DataFrame] = {}
    tables["dcu_type_profile"] = label_profile(pass_dcus, "query_dcu_type").assign(
        sort=lambda df: df["query_dcu_type"].map({v: i for i, v in enumerate(TYPE_ORDER)}).fillna(99)
    ).sort_values("sort").drop(columns=["sort"])
    tables["year_source_profile"] = label_profile(pass_dcus, ["year", "source_corpus"]).merge(
        record_pass_profile(records, "year").rename(columns={"records": "year_records"}),
        on="year",
        how="left",
    )
    # Replace with a clearer per year-source record profile.
    rec_ys = (
        records.groupby(["year", "source_corpus"])
        .agg(records=("bank_id", "count"), adequacy_pass_records=("included", "sum"), pass_rate=("included", "mean"))
        .reset_index()
    )
    dcu_ys = label_profile(pass_dcus, ["year", "source_corpus"])
    tables["year_source_profile"] = rec_ys.merge(dcu_ys, on=["year", "source_corpus"], how="left")
    tables["dcu_type_x_year"] = label_profile(pass_dcus, ["query_dcu_type", "year"], min_dcus=100)
    tables["construction_method_profile"] = label_profile(pass_dcus, "construction_method").sort_values("n_dcus", ascending=False)
    tables["construction_profile"] = label_profile(pass_dcus, "construction_group").sort_values("n_dcus", ascending=False)
    llm_by_year = (
        records.groupby("year")
        .agg(records=("bank_id", "count"), llm_records=("uses_llm_synthetic_generation", "sum"), llm_share=("uses_llm_synthetic_generation", "mean"))
        .reset_index()
    )
    tables["llm_by_year"] = llm_by_year
    bank_llm_by_year = (
        bank_meta[bank_meta["bank_year"].isin([2023, 2024, 2025])]
        .groupby("bank_year")
        .agg(
            records=("bank_id", "count"),
            llm_records=("bank_uses_llm_synthetic_generation", "sum"),
            llm_share=("bank_uses_llm_synthetic_generation", "mean"),
        )
        .reset_index()
    )
    tables["bank_llm_by_year_2023_2025"] = bank_llm_by_year
    llm_records = records[records["uses_llm_synthetic_generation"]]
    tables["llm_reporting"] = pd.DataFrame([
        {"feature": feature, "records": len(llm_records), "rate": float(llm_records[feature].mean())}
        for feature in ["model_family_named", "human_verification_reported", "quality_control_reported", "documentation_beyond_paper"]
        if feature in records.columns
    ])
    tables["dcu_type_x_llm"] = label_profile(pass_dcus, ["query_dcu_type", "construction_group"], min_dcus=100)
    tables["doc_feature_pass_delta"] = feature_pass_deltas(records)
    tables["doc_score_profile"] = record_pass_profile(records, "doc_score_bin")
    tables["doc_bin_x_type"] = label_profile(pass_dcus, ["doc_score_bin", "query_dcu_type"], min_dcus=100)
    tables["importance_profile"] = label_profile(pass_dcus, "importance").sort_values("n_dcus", ascending=False)

    # Schema-dimension diagnostics.
    rel_group, rel_group_matrix = relationship_tables(records, pass_dcus, relationships, group_col="relationship_group", min_records=50)
    rel_raw, rel_raw_matrix = relationship_tables(records, pass_dcus, relationships, group_col="relationship_type", min_records=50)
    tables["relationship_group_profile"] = rel_group
    tables["relationship_group_x_type"] = rel_group_matrix
    tables["relationship_type_profile"] = rel_raw
    tables["relationship_type_x_type"] = rel_raw_matrix

    use_prof = label_profile(pass_dcus, "primary_use_group").merge(
        records.groupby("primary_use_group").agg(n_input_records=("bank_id", "count"), llm_share=("uses_llm_synthetic_generation", "mean")).reset_index(),
        on="primary_use_group",
        how="left",
    ).sort_values("n_dcus", ascending=False)
    resource_prof = label_profile(pass_dcus, "resource_group").merge(
        records.groupby("resource_group").agg(n_input_records=("bank_id", "count"), llm_share=("uses_llm_synthetic_generation", "mean")).reset_index(),
        on="resource_group",
        how="left",
    ).sort_values("n_dcus", ascending=False)
    tables["primary_use_group_profile"] = use_prof
    tables["resource_group_profile"] = resource_prof

    lang_rec = record_pass_profile(records, "language_group")
    lang_dcu = label_profile(pass_dcus, "language_group")
    tables["language_group_profile"] = lang_rec.merge(lang_dcu, on="language_group", how="left")
    scale_dcus = pass_dcus[pass_dcus["query_dcu_type"].eq("scale/coverage")]
    tables["language_group_scale_dcus"] = label_profile(scale_dcus, "language_group", min_dcus=50)
    tables["size_bin_scale_dcus"] = label_profile(scale_dcus, "size_bin", min_dcus=50)

    sector_rec = record_pass_profile(records, "paper_sector_from_bank")
    sector_dcu = label_profile(pass_dcus, "paper_sector_from_bank")
    tables["sector_profile"] = sector_rec.merge(sector_dcu, on="paper_sector_from_bank", how="left")

    task_profile = top_value_profile(records, pass_dcus, task_values, "task", top_n=30, min_records=50)
    domain_profile = top_value_profile(records, pass_dcus, domain_values, "domain", top_n=30, min_records=50)
    tables["top_task_profile"] = task_profile
    tables["top_domain_profile"] = domain_profile

    for name, df in tables.items():
        save_csv(df, out_dir / f"{name}.csv")

    write_summary(out_dir, records=records, dcus=dcus, pass_dcus=pass_dcus, tables=tables)
    summary = {
        "analysis_dir": str(analysis_dir),
        "dataset_bank_jsonl": args.dataset_bank_jsonl,
        "output_dir": str(out_dir),
        "records": int(len(records)),
        "dcus": int(len(dcus)),
        "adequacy_pass_records": int(records["included"].sum()),
        "adequacy_pass_dcus": int(pass_dcus.shape[0]),
        "outputs": {name: str(out_dir / f"{name}.csv") for name in tables},
        "dimension_summary_md": str(out_dir / "dimension_summary.md"),
    }
    (out_dir / "dimension_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
