#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from build_census_dimension_summary import (  # noqa: E402
    DOC_FEATURES,
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
    if max_rows is not None:
        show = show.head(max_rows)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def bootstrap_ci(values: pd.Series, *, seed: int = 19, n_boot: int = 1000) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def canonical_sector(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "academic_only":
        return "academic_only"
    if raw in {"industry_only", "industry", "industry_orgs"}:
        return "industry_only"
    if raw in {"academic_industry_collab", "mixed"}:
        return "academic_industry_collab"
    if raw in {"government", "nonprofit"}:
        return raw
    return "other/unknown"


def load_sector_map(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                paper_id = row.get("paper_id")
                if not paper_id or paper_id in seen:
                    continue
                inst = row.get("institution_profile") if isinstance(row.get("institution_profile"), dict) else {}
                sector = inst.get("paper_sector") or row.get("paper_sector") or "unknown"
                rows.append({
                    "paper_id": paper_id,
                    "recovered_paper_sector": sector,
                    "sector_group": canonical_sector(sector),
                    "lead_author_sector": inst.get("lead_author_sector") or "unknown",
                    "industry_orgs": "; ".join(inst.get("industry_orgs") or []),
                    "academic_orgs": "; ".join(inst.get("academic_orgs") or []),
                })
                seen.add(paper_id)
    return pd.DataFrame(rows)


def prepare_inputs(analysis_dir: Path, bank_jsonl: Path, extraction_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(analysis_dir / "record_level.csv")
    dcus = pd.read_csv(analysis_dir / "dcu_level.csv")
    records["included"] = boolify(records["included"])
    dcus["included"] = boolify(dcus["included"])
    for col in DOC_FEATURES + ["uses_llm_synthetic_generation", "model_family_named"]:
        if col in records.columns:
            records[col] = boolify(records[col])

    sector_map = load_sector_map(extraction_paths)
    records = records.merge(sector_map, on="paper_id", how="left")
    records["recovered_paper_sector"] = records["recovered_paper_sector"].fillna("unknown")
    records["sector_group"] = records["sector_group"].fillna("other/unknown")

    bank_rows = read_jsonl(bank_jsonl)
    bank_meta, _, _, _ = flatten_bank_metadata(bank_rows)
    records = records.merge(
        bank_meta[[
            "bank_id",
            "primary_use_group",
            "resource_group",
            "language_group",
            "size_bin",
            "language_count",
            "size_value",
            "num_domains",
            "num_tasks",
        ]],
        on="bank_id",
        how="left",
    )
    for col in ["primary_use_group", "resource_group", "language_group", "size_bin"]:
        records[col] = records[col].fillna("unknown")

    dcus = dcus.merge(
        records[[
            "bank_id",
            "sector_group",
            "recovered_paper_sector",
            "primary_use_group",
            "resource_group",
            "language_group",
            "size_bin",
        ]],
        on="bank_id",
        how="left",
    )
    dcus["claim_score"] = dcus["label"].map(LABEL_VALUES)
    dcus["importance_weight"] = dcus["importance"].map(IMPORTANCE_WEIGHTS).fillna(1.0)
    dcus["score_mass"] = dcus["claim_score"] * dcus["importance_weight"]
    return records, dcus, sector_map


def record_profile(records: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, g_all in records.groupby(group_col, dropna=False):
        g = g_all[g_all["included"] & g_all["added_information_score"].notna()].copy()
        scores = g["added_information_score"].astype(float)
        row = {
            group_col: group,
            "records_total": int(len(g_all)),
            "adequacy_pass_records": int(len(g)),
            "pass_rate": float(len(g) / len(g_all)) if len(g_all) else np.nan,
            "mean_score": float(scores.mean()) if len(g) else np.nan,
            "median_score": float(scores.median()) if len(g) else np.nan,
            "mean_dcus_per_pass_record": float(g["n_query_dcus"].mean()) if len(g) else np.nan,
            "llm_share": float(g["uses_llm_synthetic_generation"].mean()) if len(g) and "uses_llm_synthetic_generation" in g else np.nan,
        }
        row["score_ci_low"], row["score_ci_high"] = bootstrap_ci(scores)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("adequacy_pass_records", ascending=False)


def scale_profile(dcus: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    pass_dcus = dcus[dcus["included"] & dcus["claim_score"].notna()].copy()
    for group, g in pass_dcus.groupby(group_col, dropna=False):
        total_mass = float(g["score_mass"].sum())
        scale = g[g["query_dcu_type"].eq("scale/coverage")]
        scale_mass = float(scale["score_mass"].sum())
        denom = float(scale["importance_weight"].sum())
        rows.append({
            group_col: group,
            "n_dcus": int(len(g)),
            "scale_dcus": int(len(scale)),
            "scale_dcu_share": float(len(scale) / len(g)) if len(g) else np.nan,
            "scale_score_mass_share": float(scale_mass / total_mass) if total_mass else np.nan,
            "scale_weighted_mean_claim_score": float(scale_mass / denom) if denom else np.nan,
            "overall_weighted_mean_claim_score": float(total_mass / g["importance_weight"].sum()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_dcus", ascending=False)


def feature_profile(records: pd.DataFrame, group_col: str) -> pd.DataFrame:
    pass_records = records[records["included"] & records["added_information_score"].notna()].copy()
    rows = []
    for group, g in pass_records.groupby(group_col, dropna=False):
        row: dict[str, Any] = {group_col: group, "n_records": int(len(g))}
        for feature in [
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
        ]:
            if feature in g.columns:
                row[f"{feature}_rate"] = float(g[feature].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_records", ascending=False)


def key_metrics_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sectors = ["academic_only", "academic_industry_collab", "industry_only"]
    records = tables["sector_record_profile"].set_index("sector_group")
    scale = tables["sector_scale_profile"].set_index("sector_group")
    features = tables["sector_feature_profile"].set_index("sector_group")
    size = tables["sector_size_profile"]
    rows = []
    for sector in sectors:
        if sector not in records.index:
            continue
        size_g = size[size["sector_group"].eq(sector)]
        size_total = float(size_g["n_records"].sum())
        row = {
            "sector_group": sector,
            "pass_records": int(records.loc[sector, "adequacy_pass_records"]),
            "mean_score": float(records.loc[sector, "mean_score"]),
            "score_ci_low": float(records.loc[sector, "score_ci_low"]),
            "score_ci_high": float(records.loc[sector, "score_ci_high"]),
            "llm_share": float(records.loc[sector, "llm_share"]),
            "pass_rate": float(records.loc[sector, "pass_rate"]),
            "scale_dcu_share": float(scale.loc[sector, "scale_dcu_share"]),
            "scale_score_mass_share": float(scale.loc[sector, "scale_score_mass_share"]),
            "scale_weighted_mean_claim_score": float(scale.loc[sector, "scale_weighted_mean_claim_score"]),
            "released_rate": float(features.loc[sector, "released_rate"]),
            "open_access_rate": float(features.loc[sector, "open_access_rate"]),
            "human_verification_reported_rate": float(features.loc[sector, "human_verification_reported_rate"]),
            "ethics_discussed_rate": float(features.loc[sector, "ethics_discussed_rate"]),
            "pii_discussed_rate": float(features.loc[sector, "pii_discussed_rate"]),
            "bias_or_fairness_discussed_rate": float(features.loc[sector, "bias_or_fairness_discussed_rate"]),
        }
        for _, size_row in size_g.iterrows():
            row[f"size_{size_row['size_bin']}_share"] = float(size_row["n_records"] / size_total) if size_total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def category_score(records: pd.DataFrame, group_cols: list[str], *, min_records: int = 25) -> pd.DataFrame:
    pass_records = records[records["included"] & records["added_information_score"].notna()].copy()
    rows = []
    for group, g in pass_records.groupby(group_cols, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        if len(g) < min_records:
            continue
        row = {col: value for col, value in zip(group_cols, group)}
        row.update({
            "n_records": int(len(g)),
            "mean_score": float(g["added_information_score"].mean()),
            "llm_share": float(g["uses_llm_synthetic_generation"].mean()) if "uses_llm_synthetic_generation" in g else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["n_records"])


def build_summary(tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    sector = tables["sector_record_profile"]
    exact = tables["exact_sector_profile"]
    scale = tables["sector_scale_profile"]
    features = tables["sector_feature_profile"]
    language = tables["sector_language_profile"]
    size = tables["sector_size_profile"]
    primary = tables["sector_primary_use_profile"]
    resource = tables["sector_resource_profile"]
    key = tables["sector_key_metrics"]

    lines = [
        "# Academia / Industry Sector Comparison",
        "",
        "Sectors are recovered from `institution_profile.paper_sector` in full-text extraction files and joined back to census records by `paper_id`.",
        "Use this as a diagnostic rather than a validated sociological claim.",
        "",
        "## Sector Score Profile",
        md_table(sector, [
            "sector_group",
            "records_total",
            "adequacy_pass_records",
            "pass_rate",
            "mean_score",
            "median_score",
            "score_ci_low",
            "score_ci_high",
            "llm_share",
            "mean_dcus_per_pass_record",
        ]),
        "",
        "## Key Academic vs Industry Metrics",
        md_table(key, [
            "sector_group",
            "pass_records",
            "mean_score",
            "score_ci_low",
            "score_ci_high",
            "llm_share",
            "scale_dcu_share",
            "scale_score_mass_share",
            "released_rate",
            "open_access_rate",
            "human_verification_reported_rate",
            "ethics_discussed_rate",
            "pii_discussed_rate",
        ]),
        "",
        "## Exact Sector Labels",
        md_table(exact, [
            "recovered_paper_sector",
            "records_total",
            "adequacy_pass_records",
            "pass_rate",
            "mean_score",
            "median_score",
            "llm_share",
        ]),
        "",
        "## Scale / Coverage Profile",
        md_table(scale, [
            "sector_group",
            "n_dcus",
            "scale_dcus",
            "scale_dcu_share",
            "scale_score_mass_share",
            "scale_weighted_mean_claim_score",
            "overall_weighted_mean_claim_score",
        ]),
        "",
        "## Release / Governance / Documentation Rates",
        md_table(features, [
            "sector_group",
            "n_records",
            "released_rate",
            "open_access_rate",
            "license_specified_rate",
            "source_or_origin_disclosed_rate",
            "quality_control_reported_rate",
            "human_verification_reported_rate",
            "ethics_discussed_rate",
            "pii_discussed_rate",
            "bias_or_fairness_discussed_rate",
        ]),
        "",
        "## Language Groups",
        md_table(language, ["sector_group", "language_group", "n_records", "mean_score", "llm_share"], max_rows=40),
        "",
        "## Size Bins",
        md_table(size, ["sector_group", "size_bin", "n_records", "mean_score", "llm_share"], max_rows=40),
        "",
        "## Primary Use",
        md_table(primary, ["sector_group", "primary_use_group", "n_records", "mean_score", "llm_share"], max_rows=40),
        "",
        "## Resource Type",
        md_table(resource, ["sector_group", "resource_group", "n_records", "mean_score", "llm_share"], max_rows=40),
        "",
        "## Quick Interpretation",
        "- The most defensible comparison is `academic_only` vs `industry_only` vs `academic_industry_collab`; other categories are small.",
        "- Score differences should be read as evidence-conditioned added-information differences, not absolute novelty.",
        "- Scale should be read in two ways: share of DCUs that are scale/coverage, and how much of the score mass comes from scale/coverage DCUs.",
    ]
    out_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/census/census_scale_analysis"))
    parser.add_argument("--dataset-bank-jsonl", type=Path, default=Path("data/census/integrated_fulltext_dataset_bank_2023_2025.jsonl"))
    parser.add_argument("--extraction-jsonl", type=Path, action="append", default=[
        Path("data/census/fulltext_dataset_extractions_pdf_all.jsonl"),
        Path("data/census/arxiv_fulltext_dataset_extractions_2023_2025.jsonl"),
    ])
    parser.add_argument("--output-dir", type=Path, default=Path("data/census/census_scale_analysis/sector_comparison"))
    args = parser.parse_args()

    records, dcus, sector_map = prepare_inputs(args.analysis_dir, args.dataset_bank_jsonl, args.extraction_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "sector_map": sector_map,
        "sector_record_profile": record_profile(records, "sector_group"),
        "exact_sector_profile": record_profile(records, "recovered_paper_sector"),
        "sector_scale_profile": scale_profile(dcus, "sector_group"),
        "sector_feature_profile": feature_profile(records, "sector_group"),
        "sector_language_profile": category_score(records, ["sector_group", "language_group"]),
        "sector_size_profile": category_score(records, ["sector_group", "size_bin"]),
        "sector_primary_use_profile": category_score(records, ["sector_group", "primary_use_group"]),
        "sector_resource_profile": category_score(records, ["sector_group", "resource_group"]),
    }
    tables["sector_key_metrics"] = key_metrics_table(tables)

    for name, table in tables.items():
        save_csv(table, args.output_dir / f"{name}.csv")
    build_summary(tables, args.output_dir / "sector_comparison_summary.md")
    print(f"Wrote sector comparison diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
