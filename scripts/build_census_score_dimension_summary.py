#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from build_census_dimension_summary import (  # noqa: E402
    DOC_FEATURES,
    TYPE_ORDER,
    add_doc_bins,
    boolify,
    flatten_bank_metadata,
    read_jsonl,
)


IMPORTANCE_WEIGHTS = {"low": 0.5, "medium": 1.0, "high": 1.5}
LABEL_VALUES = {"covered": 0.0, "partially_covered": 0.5, "not_covered": 1.0}


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def md_table(df: pd.DataFrame, cols: list[str], *, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    show = df[cols].copy()
    if max_rows:
        show = show.head(max_rows)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def bootstrap_ci(values: pd.Series, *, seed: int = 17, n_boot: int = 1000) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def record_score_profile(records: pd.DataFrame, group_cols: str | list[str], *, min_records: int = 0) -> pd.DataFrame:
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    rows = []
    for group, g in records.groupby(group_cols, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        if len(g) < min_records:
            continue
        scores = g["added_information_score"].astype(float)
        row = {col: val for col, val in zip(group_cols, group)}
        row.update({
            "n_records": int(len(g)),
            "n_dcus": int(g["n_query_dcus"].sum()) if "n_query_dcus" in g else int(g["bank_id"].nunique()),
            "mean_score": float(scores.mean()),
            "median_score": float(scores.median()),
        })
        row["ci_low"], row["ci_high"] = bootstrap_ci(scores)
        rows.append(row)
    return pd.DataFrame(rows)


def dcu_score_profile(dcus: pd.DataFrame, group_cols: str | list[str], *, min_dcus: int = 0) -> pd.DataFrame:
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    rows = []
    total_mass = float(dcus["score_mass"].sum())
    for group, g in dcus.groupby(group_cols, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        if len(g) < min_dcus:
            continue
        row = {col: val for col, val in zip(group_cols, group)}
        mass = float(g["score_mass"].sum())
        denom = float(g["importance_weight"].sum())
        row.update({
            "n_records": int(g["bank_id"].nunique()),
            "n_dcus": int(len(g)),
            "score_mass": mass,
            "score_mass_share": mass / total_mass if total_mass else np.nan,
            "weighted_mean_claim_score": mass / denom if denom else np.nan,
            "mean_claim_score": float(g["claim_score"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_inputs(analysis_dir: Path, bank_jsonl: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(analysis_dir / "record_level.csv")
    dcus = pd.read_csv(analysis_dir / "dcu_level.csv")
    records["included"] = boolify(records["included"])
    dcus["included"] = boolify(dcus["included"])
    for col in DOC_FEATURES + ["uses_llm_synthetic_generation", "model_family_named"]:
        if col in records.columns:
            records[col] = boolify(records[col])

    bank_rows = read_jsonl(bank_jsonl)
    bank_meta, relationships, task_values, domain_values = flatten_bank_metadata(bank_rows)
    records = records.merge(bank_meta, on="bank_id", how="left")
    records["primary_use_group"] = records["primary_use_group"].fillna("unknown")
    records["resource_group"] = records["resource_group"].fillna("unknown")
    records["language_group"] = records["language_group"].fillna("unknown")
    records["size_bin"] = records["size_bin"].fillna("unknown")
    records["paper_sector_from_bank"] = records["paper_sector_from_bank"].fillna("unknown")
    records = add_doc_bins(records)

    dcus = dcus.merge(
        records[[
            "bank_id",
            "primary_use_group",
            "resource_group",
            "language_group",
            "size_bin",
            "paper_sector_from_bank",
            "doc_score_bin",
        ]],
        on="bank_id",
        how="left",
    )
    pass_records = records[records["included"] & records["added_information_score"].notna()].copy()
    pass_dcus = dcus[dcus["included"]].copy()
    pass_dcus["claim_score"] = pass_dcus["label"].map(LABEL_VALUES)
    pass_dcus["importance_weight"] = pass_dcus["importance"].map(IMPORTANCE_WEIGHTS).fillna(1.0)
    pass_dcus = pass_dcus[pass_dcus["claim_score"].notna()].copy()
    pass_dcus["score_mass"] = pass_dcus["claim_score"] * pass_dcus["importance_weight"]
    return records, pass_records, pass_dcus, relationships, task_values, domain_values


def top_value_score_profile(
    records: pd.DataFrame,
    dcus: pd.DataFrame,
    values: pd.DataFrame,
    value_col: str,
    *,
    top_n: int = 20,
    min_records: int = 50,
) -> pd.DataFrame:
    if values.empty:
        return pd.DataFrame()
    top = values.groupby(value_col)["bank_id"].nunique().sort_values(ascending=False).head(top_n).index
    vals = values[values[value_col].isin(top)].drop_duplicates(["bank_id", value_col])
    rec_join = records.merge(vals[["bank_id", value_col]], on="bank_id", how="inner")
    rec_prof = record_score_profile(rec_join, value_col, min_records=min_records)
    dcu_join = dcus.merge(vals[["bank_id", value_col]], on="bank_id", how="inner")
    dcu_prof = dcu_score_profile(dcu_join, value_col)
    out = rec_prof.merge(
        dcu_prof[[value_col, "score_mass_share", "weighted_mean_claim_score"]],
        on=value_col,
        how="left",
        suffixes=("", "_claim"),
    )
    return out.sort_values("n_records", ascending=False)


def build_tables(records: pd.DataFrame, dcus: pd.DataFrame, relationships: pd.DataFrame, task_values: pd.DataFrame, domain_values: pd.DataFrame, bank_meta_source: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    type_profile = dcu_score_profile(dcus, "query_dcu_type")
    type_profile["sort"] = type_profile["query_dcu_type"].map({v: i for i, v in enumerate(TYPE_ORDER)}).fillna(99)
    tables["score_dcu_type_profile"] = type_profile.sort_values("sort").drop(columns=["sort"])
    tables["score_year_source_profile"] = record_score_profile(records, ["year", "source_corpus"]).sort_values(["year", "source_corpus"])
    tables["score_dcu_type_x_year"] = dcu_score_profile(dcus, ["query_dcu_type", "year"], min_dcus=100).sort_values(["query_dcu_type", "year"])
    tables["score_construction_method_profile"] = record_score_profile(records, "construction_method").sort_values("n_records", ascending=False)
    tables["score_construction_group_profile"] = record_score_profile(records, "construction_group").sort_values("n_records", ascending=False)
    bank_llm = (
        bank_meta_source[bank_meta_source["bank_year"].isin([2023, 2024, 2025])]
        .groupby("bank_year")
        .agg(records=("bank_id", "count"), llm_records=("bank_uses_llm_synthetic_generation", "sum"), llm_share=("bank_uses_llm_synthetic_generation", "mean"))
        .reset_index()
    )
    tables["bank_llm_by_year_2023_2025"] = bank_llm
    tables["score_dcu_type_x_llm"] = dcu_score_profile(dcus, ["query_dcu_type", "construction_group"], min_dcus=100).sort_values(["query_dcu_type", "construction_group"])

    feature_rows = []
    all_records = records.copy()
    for feature in DOC_FEATURES:
        if feature not in all_records.columns:
            continue
        mask = boolify(all_records[feature])
        yes = all_records[mask & all_records["included"] & all_records["added_information_score"].notna()]
        no = all_records[(~mask) & all_records["included"] & all_records["added_information_score"].notna()]
        if len(yes) == 0 or len(no) == 0:
            continue
        feature_rows.append({
            "feature": feature,
            "n_with": int(mask.sum()),
            "n_without": int((~mask).sum()),
            "with_mean_score": float(yes["added_information_score"].mean()),
            "without_mean_score": float(no["added_information_score"].mean()),
            "delta_score": float(yes["added_information_score"].mean() - no["added_information_score"].mean()),
        })
    tables["score_doc_feature_delta"] = pd.DataFrame(feature_rows).sort_values("delta_score", ascending=False)
    tables["score_doc_score_profile"] = record_score_profile(records, "doc_score_bin").sort_values("doc_score_bin")
    tables["score_doc_bin_x_type"] = dcu_score_profile(dcus, ["doc_score_bin", "query_dcu_type"], min_dcus=100).sort_values(["doc_score_bin", "query_dcu_type"])
    tables["score_importance_profile"] = dcu_score_profile(dcus, "importance").sort_values("n_dcus", ascending=False)

    rel = relationships.drop_duplicates(["bank_id", "relationship_group"]).copy()
    rel_rec = records.merge(rel[["bank_id", "relationship_group"]], on="bank_id", how="inner")
    rel_dcu = dcus.merge(rel[["bank_id", "relationship_group"]], on="bank_id", how="inner")
    tables["score_relationship_group_profile"] = record_score_profile(rel_rec, "relationship_group", min_records=50).sort_values("n_records", ascending=False)
    rel_type_dcu = dcus.merge(rel[["bank_id", "relationship_group"]], on="bank_id", how="inner")
    tables["score_relationship_group_x_type"] = dcu_score_profile(rel_type_dcu, ["relationship_group", "query_dcu_type"], min_dcus=100).sort_values(["relationship_group", "query_dcu_type"])

    tables["score_primary_use_group_profile"] = record_score_profile(records, "primary_use_group").sort_values("n_records", ascending=False)
    tables["score_resource_group_profile"] = record_score_profile(records, "resource_group").sort_values("n_records", ascending=False)
    tables["score_language_group_profile"] = record_score_profile(records, "language_group").sort_values("n_records", ascending=False)
    scale_dcus = dcus[dcus["query_dcu_type"].eq("scale/coverage")]
    tables["score_language_group_scale_dcus"] = dcu_score_profile(scale_dcus, "language_group", min_dcus=50).sort_values("n_dcus", ascending=False)
    tables["score_size_bin_scale_dcus"] = dcu_score_profile(scale_dcus, "size_bin", min_dcus=50).sort_values("n_dcus", ascending=False)
    tables["score_sector_profile"] = record_score_profile(records, "paper_sector_from_bank").sort_values("n_records", ascending=False)
    tables["score_top_task_profile"] = top_value_score_profile(records, dcus, task_values, "task", top_n=30, min_records=50)
    tables["score_top_domain_profile"] = top_value_score_profile(records, dcus, domain_values, "domain", top_n=30, min_records=50)
    return tables


def write_summary(out_dir: Path, records: pd.DataFrame, dcus: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    type_prof = tables["score_dcu_type_profile"]
    llm = tables["score_construction_group_profile"]
    llm_assisted = llm[llm["construction_group"].eq("LLM-assisted")]
    other = llm[llm["construction_group"].eq("Non-LLM/other")]
    llm_delta = np.nan
    if len(llm_assisted) and len(other):
        llm_delta = float(llm_assisted.iloc[0]["mean_score"] - other.iloc[0]["mean_score"])

    lines = [
        "# Census Score Dimension Summary",
        "",
        "This file preserves the full dimension sweep, but all main diagnostics are expressed with the evidence-conditioned added-information score.",
        "",
        "## Scope",
        f"- Adequacy-passing records: {len(records):,}",
        f"- Adequacy-passing scored DCUs: {len(dcus):,}",
        f"- Mean score: {records['added_information_score'].mean():.3f}",
        f"- Median score: {records['added_information_score'].median():.3f}",
        "",
        "## Strongest Score-Level Signals",
        f"- Scale/coverage contributes {100 * float(type_prof.loc[type_prof['query_dcu_type'].eq('scale/coverage'), 'score_mass_share'].iloc[0]):.1f}% of total score mass.",
        f"- Scale/coverage has weighted mean claim score {float(type_prof.loc[type_prof['query_dcu_type'].eq('scale/coverage'), 'weighted_mean_claim_score'].iloc[0]):.3f}.",
        f"- LLM-assisted minus Non-LLM/other mean score delta is {llm_delta:.3f}; LLM use is not a higher-score proxy.",
        "- Documentation/governance score associations remain mixed and should stay appendix/diagnostic.",
        "",
        "## 1. DCU Type Profile",
        md_table(type_prof, ["query_dcu_type", "n_records", "n_dcus", "score_mass_share", "weighted_mean_claim_score", "mean_claim_score"]),
        "",
        "Recommended use: main paper as score decomposition. Keep `other` out of the main figure.",
        "",
        "## 2. Year and Source",
        md_table(tables["score_year_source_profile"], ["year", "source_corpus", "n_records", "mean_score", "median_score", "ci_low", "ci_high"]),
        "",
        "Recommended use: scope/context or appendix robustness.",
        "",
        "## 3. DCU Type x Year",
        md_table(tables["score_dcu_type_x_year"], ["query_dcu_type", "year", "n_dcus", "score_mass_share", "weighted_mean_claim_score"], max_rows=80),
        "",
        "Recommended use: appendix robustness for Finding 2.",
        "",
        "## 4. Construction Method",
        md_table(tables["score_construction_method_profile"], ["construction_method", "n_records", "mean_score", "median_score", "ci_low", "ci_high"]),
        "",
        "Recommended use: main only paired with LLM trend; do not rank construction methods as novelty categories.",
        "",
        "## 5. LLM-Assisted Construction",
        "Bank-level construction metadata trend, including 2023:",
        md_table(tables["bank_llm_by_year_2023_2025"], ["bank_year", "records", "llm_records", "llm_share"]),
        "",
        "Score by LLM group:",
        md_table(tables["score_construction_group_profile"], ["construction_group", "n_records", "mean_score", "median_score", "ci_low", "ci_high"]),
        "",
        "Recommended use: main paper. Claim only that LLM-assisted construction rises but is not a higher-score proxy.",
        "",
        "## 6. DCU Type x LLM Group",
        md_table(tables["score_dcu_type_x_llm"], ["query_dcu_type", "construction_group", "n_dcus", "score_mass_share", "weighted_mean_claim_score"], max_rows=80),
        "",
        "Recommended use: appendix or companion diagnostic.",
        "",
        "## 7. Release/Governance/Documentation Features",
        md_table(tables["score_doc_feature_delta"], ["feature", "n_with", "n_without", "with_mean_score", "without_mean_score", "delta_score"]),
        "",
        "Documentation score bins:",
        md_table(tables["score_doc_score_profile"], ["doc_score_bin", "n_records", "mean_score", "median_score", "ci_low", "ci_high"]),
        "",
        "Recommended use: appendix only unless reframed as mixed descriptive association.",
        "",
        "## 8. Documentation Score x DCU Type",
        md_table(tables["score_doc_bin_x_type"], ["doc_score_bin", "query_dcu_type", "n_dcus", "score_mass_share", "weighted_mean_claim_score"], max_rows=80),
        "",
        "Recommended use: appendix.",
        "",
        "## 9. Importance",
        md_table(tables["score_importance_profile"], ["importance", "n_records", "n_dcus", "score_mass_share", "weighted_mean_claim_score"]),
        "",
        "Recommended use: appendix/error analysis.",
        "",
        "## 10. Prior Dataset Relationship",
        md_table(tables["score_relationship_group_profile"], ["relationship_group", "n_records", "mean_score", "median_score", "ci_low", "ci_high"]),
        "",
        "Prior relationship x DCU type:",
        md_table(tables["score_relationship_group_x_type"], ["relationship_group", "query_dcu_type", "n_dcus", "score_mass_share", "weighted_mean_claim_score"], max_rows=80),
        "",
        "Recommended use: appendix support; relationship groups do not currently produce a sharp main finding.",
        "",
        "## 11. Primary Use and Resource Type",
        "Primary use:",
        md_table(tables["score_primary_use_group_profile"], ["primary_use_group", "n_records", "mean_score", "median_score", "ci_low", "ci_high"]),
        "",
        "Resource type:",
        md_table(tables["score_resource_group_profile"], ["resource_group", "n_records", "mean_score", "median_score", "ci_low", "ci_high"]),
        "",
        "Recommended use: appendix or fallback table.",
        "",
        "## 12. Language and Scale",
        "Language group profile:",
        md_table(tables["score_language_group_profile"], ["language_group", "n_records", "mean_score", "median_score", "ci_low", "ci_high"]),
        "",
        "Language group restricted to scale/coverage DCUs:",
        md_table(tables["score_language_group_scale_dcus"], ["language_group", "n_records", "n_dcus", "score_mass_share", "weighted_mean_claim_score"]),
        "",
        "Size bin restricted to scale/coverage DCUs:",
        md_table(tables["score_size_bin_scale_dcus"], ["size_bin", "n_records", "n_dcus", "score_mass_share", "weighted_mean_claim_score"]),
        "",
        "Recommended use: deepen scale/coverage finding; complete tables in appendix.",
        "",
        "## 13. Sector",
        md_table(tables["score_sector_profile"], ["paper_sector_from_bank", "n_records", "mean_score", "median_score"], max_rows=20),
        "",
        "Recommended use: do not use until sector labels are recovered.",
        "",
        "## 14. Task and Domain",
        "Top tasks:",
        md_table(tables["score_top_task_profile"], ["task", "n_records", "mean_score", "median_score", "score_mass_share", "weighted_mean_claim_score"], max_rows=20),
        "",
        "Top domains:",
        md_table(tables["score_top_domain_profile"], ["domain", "n_records", "mean_score", "median_score", "score_mass_share", "weighted_mean_claim_score"], max_rows=20),
        "",
        "Recommended use: appendix scope table; avoid a large main-figure dashboard.",
    ]
    (out_dir / "score_dimension_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="data/census/census_scale_analysis")
    parser.add_argument("--dataset-bank-jsonl", default="data/census/integrated_fulltext_dataset_bank_2023_2025.jsonl")
    parser.add_argument("--output-dir", default="data/census/census_scale_analysis/score_dimension_summaries")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_all, records, dcus, relationships, task_values, domain_values = prepare_inputs(Path(args.analysis_dir), Path(args.dataset_bank_jsonl))
    bank_meta, _, _, _ = flatten_bank_metadata(read_jsonl(Path(args.dataset_bank_jsonl)))
    tables = build_tables(records, dcus, relationships, task_values, domain_values, bank_meta)
    for name, df in tables.items():
        save_csv(df, out_dir / f"{name}.csv")
    write_summary(out_dir, records, dcus, tables)
    summary = {
        "analysis_dir": args.analysis_dir,
        "dataset_bank_jsonl": args.dataset_bank_jsonl,
        "output_dir": str(out_dir),
        "records": int(len(records)),
        "dcus": int(len(dcus)),
        "outputs": {name: str(out_dir / f"{name}.csv") for name in tables},
        "score_dimension_summary_md": str(out_dir / "score_dimension_summary.md"),
    }
    (out_dir / "score_dimension_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
