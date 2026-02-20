import argparse
import json
import os
from typing import List, Optional, Tuple

DEFAULT_OUTPUT_DIR = "data/visualizations"
DEFAULT_SCV_FILE = "data/processed/final_scv_200.jsonl"
DEFAULT_CATALOG_FILE = "data/processed/arxiv_nlp_conf_papers_2023_2025.csv"
DEFAULT_CACHE_ROOT = os.path.join(DEFAULT_OUTPUT_DIR, ".cache")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(DEFAULT_OUTPUT_DIR, ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", DEFAULT_CACHE_ROOT)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

AFFILIATION_COLORS = {
    "Industry": "#1f77b4",
    "Academia": "#ff7f0e",
    "Mixed": "#2ca02c",
    "Unknown": "#7f7f7f",
}

sns.set_theme(style="whitegrid", context="talk")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate demonstration-ready visualizations for Scientific Contribution Vector (SCV) outputs."
    )
    parser.add_argument(
        "--scv-file",
        default=DEFAULT_SCV_FILE,
        help="Path to the final SCV JSONL file.",
    )
    parser.add_argument(
        "--catalog-file",
        default=DEFAULT_CATALOG_FILE,
        help="CSV file that lists all scraped papers for computing baseline counts.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=3,
        help="Window (in years) for temporal trend plots.",
    )
    return parser.parse_args()


def load_jsonl(filepath: str) -> List[dict]:
    records = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_catalog(filepath: Optional[str]) -> Optional[pd.DataFrame]:
    if not filepath or not os.path.exists(filepath):
        print(f"[visualize] Catalog file '{filepath}' not found. Skipping baseline counts.")
        return None

    df = pd.read_csv(filepath)
    date_columns = ["Publication Date", "published_date", "date"]
    id_columns = ["arXiv ID", "arxiv_id", "id"]

    date_col = next((col for col in date_columns if col in df.columns), None)
    id_col = next((col for col in id_columns if col in df.columns), None)
    if not date_col or not id_col:
        print("[visualize] Catalog is missing required columns. Skipping baseline counts.")
        return None

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["paper_id"] = df[id_col].astype(str)
    df = df.dropna(subset=["date"])
    return df[["paper_id", "date"]]


def classify_affiliation(authors: List[dict]) -> str:
    industry = [
        "Google",
        "DeepMind",
        "Meta",
        "Facebook",
        "Microsoft",
        "Amazon",
        "Apple",
        "Anthropic",
        "Cohere",
        "OpenAI",
        "IBM",
        "Nvidia",
        "Salesforce",
        "Adobe",
        "Tencent",
        "Alibaba",
        "Baidu",
        "Huawei",
        "ByteDance",
        "HuggingFace",
        "AI2",
        "Allen Institute",
    ]
    academia = [
        "University",
        "College",
        "Institute",
        "School",
        "Academy",
        "Polytechnic",
        "MIT",
        "Caltech",
        "Stanford",
        "Harvard",
        "CMU",
        "ETH",
        "Inria",
        "MPI",
        "CNRS",
        "Riken",
        "KAIST",
        "UC ",
        "Georgia Tech",
    ]

    text = " ".join(a.get("affiliation", "") or "" for a in authors).lower()
    is_industry = any(name.lower() in text for name in industry)
    is_academia = any(name.lower() in text for name in academia)

    if is_industry and is_academia:
        return "Mixed"
    if is_industry:
        return "Industry"
    if is_academia:
        return "Academia"
    return "Unknown"


def simplify_domain(domain: str) -> str:
    if not domain:
        return "General/Other"
    d = domain.lower()
    if any(key in d for key in ["bio", "med", "clinic"]):
        return "Biomedical"
    if any(key in d for key in ["social", "twitter", "reddit"]):
        return "Social Media"
    if any(key in d for key in ["finance", "econ"]):
        return "Finance"
    if any(key in d for key in ["legal", "law"]):
        return "Legal"
    if any(key in d for key in ["dialog", "conversa", "chatbot"]):
        return "Dialogue"
    if any(key in d for key in ["vision", "image", "video", "multi"]):
        return "Multimodal"
    if any(key in d for key in ["sci", "scholar"]):
        return "Scientific"
    return "General/Other"


def to_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def extract_num_samples(size_info) -> Optional[float]:
    if not isinstance(size_info, dict):
        return None
    raw = size_info.get("num_samples")
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    if isinstance(raw, str):
        cleaned = "".join(ch for ch in raw if ch.isdigit())
        if cleaned:
            return float(cleaned)
    return None


def bucket_languages(count: int) -> str:
    if count is None or count <= 0:
        return "0 languages"
    if count == 1:
        return "1 language"
    if count <= 3:
        return "2-3 languages"
    if count <= 10:
        return "4-10 languages"
    return "10+ languages"


def bucket_samples(num_samples: Optional[float]) -> str:
    if num_samples is None or num_samples <= 0:
        return "Unknown size"
    if num_samples < 1_000:
        return "<1K samples"
    if num_samples < 10_000:
        return "1K-10K"
    if num_samples < 100_000:
        return "10K-100K"
    if num_samples < 1_000_000:
        return "100K-1M"
    return "1M+"


def build_frames(records: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paper_rows = []
    dataset_rows = []

    for record in records:
        metadata = record.get("metadata", {})
        date_str = metadata.get("date")
        if not date_str:
            continue
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            continue

        authors = metadata.get("authors", [])
        affiliation = classify_affiliation(authors)
        introduced = [
            ds for ds in record.get("datasets", []) if ds.get("info", {}).get("is_introduced")
        ]
        # Filter out datasets with novelty score of 1.0
        introduced = [
            ds for ds in introduced 
            if ds.get("scv", {}).get("novelty") != 1.0
        ]

        paper_rows.append(
            {
                "paper_id": record.get("arxiv_id"),
                "date": dt,
                "quarter": dt.to_period("Q"),
                "affiliation": affiliation,
                "has_dataset": bool(introduced),
            }
        )

        for ds in introduced:
            info = ds.get("info", {})
            scv = ds.get("scv", {})
            languages = to_list(info.get("languages"))
            tasks = to_list(info.get("main_tasks"))
            num_samples = extract_num_samples(info.get("size"))

            dataset_rows.append(
                {
                    "paper_id": record.get("arxiv_id"),
                    "dataset_name": info.get("name", "Unknown Dataset"),
                    "date": dt,
                    "quarter": dt.to_period("Q"),
                    "affiliation": affiliation,
                    "domain": simplify_domain(info.get("domain", "")),
                    "raw_domain": info.get("domain", "Unknown"),
                    "primary_task": tasks[0] if tasks else "Other / Unknown",
                    "languages_count": len(languages),
                    "languages_bucket": bucket_languages(len(languages)),
                    "num_samples": num_samples,
                    "size_bucket": bucket_samples(num_samples),
                    "novelty": scv.get("novelty", np.nan),
                    "diversity": scv.get("diversity", np.nan),
                    "quality": scv.get("quality", np.nan),
                    "embedding": record.get("paper_embedding", []),
                }
            )

    df_papers = pd.DataFrame(paper_rows)
    df_datasets = pd.DataFrame(dataset_rows)
    return df_papers, df_datasets


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def temporal_trend_plot(
    df_papers: pd.DataFrame,
    catalog: Optional[pd.DataFrame],
    output_dir: str,
    start_date: pd.Timestamp,
):
    if df_papers.empty:
        return

    subset = df_papers[df_papers["date"] >= start_date]
    if subset.empty:
        return

    dataset_counts = subset.groupby("quarter")["paper_id"].nunique()
    summary = pd.DataFrame({"dataset_papers": dataset_counts})

    if catalog is not None:
        catalog_subset = catalog[catalog["date"] >= start_date]
        catalog_subset["quarter"] = catalog_subset["date"].dt.to_period("Q")
        totals = catalog_subset.groupby("quarter")["paper_id"].nunique()
        summary = summary.join(totals.rename("all_papers"), how="outer")
    else:
        summary["all_papers"] = np.nan

    summary = summary.fillna(0).sort_index()
    summary["fraction"] = np.where(
        summary["all_papers"] > 0,
        summary["dataset_papers"] / summary["all_papers"],
        np.nan,
    )
    quarters = summary.index.to_timestamp()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(
        quarters,
        summary["dataset_papers"],
        width=70,
        color="#92b4f4",
        label="Dataset papers",
    )
    ax1.set_ylabel("Dataset papers per quarter")
    ax1.set_title("Dataset introductions vs. total NLP papers")
    ax1.set_xlabel("Quarter")

    if summary["fraction"].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(
            quarters,
            summary["fraction"],
            color="#d62728",
            marker="o",
            linewidth=2,
            label="Fraction introducing datasets",
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Share of all papers")
        lines, labels = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + l2, labels + lab2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "temporal_fraction.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def affiliation_stack_plot(df_datasets: pd.DataFrame, output_dir: str, start_date: pd.Timestamp):
    if df_datasets.empty:
        return
    subset = df_datasets[df_datasets["date"] >= start_date]
    if subset.empty:
        return

    pivot = (
        subset.groupby(["quarter", "affiliation"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[c for c in AFFILIATION_COLORS if c in subset["affiliation"].unique()])
    )
    quarters = pivot.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 6))
    stacks = []
    for aff in pivot.columns:
        stacks.append(pivot[aff])
    colors = [AFFILIATION_COLORS.get(aff, "#999999") for aff in pivot.columns]
    ax.stackplot(quarters, stacks, labels=pivot.columns, colors=colors, alpha=0.85)
    ax.set_title("Who releases new datasets?")
    ax.set_ylabel("Datasets per quarter")
    ax.set_xlabel("Quarter")
    ax.legend(loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "affiliation_area.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def domain_novelty_plot(df_datasets: pd.DataFrame, output_dir: str):
    if df_datasets.empty:
        return
    df = df_datasets.copy()
    top_domains = df["domain"].value_counts().head(6).index
    df["domain_plot"] = np.where(df["domain"].isin(top_domains), df["domain"], "Other")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=df,
        x="domain_plot",
        y="novelty",
        inner="quartile",
        cut=0,
        ax=ax,
    )
    ax.set_title("Novelty distribution by domain")
    ax.set_xlabel("Domain (top 6 + other)")
    ax.set_ylabel("Novelty score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "novelty_by_domain.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def language_diversity_scatter(df_datasets: pd.DataFrame, output_dir: str):
    if df_datasets.empty:
        return
    df = df_datasets.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="languages_count",
        y="diversity",
        hue="domain",
        size="quality",
        sizes=(40, 250),
        palette="tab10",
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Diversity vs. language coverage")
    ax.set_xlabel("Languages referenced in paper")
    ax.set_ylabel("Diversity score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(output_dir, "language_diversity_scatter.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def size_vs_novelty(df_datasets: pd.DataFrame, output_dir: str):
    df = df_datasets.dropna(subset=["num_samples", "novelty"]).copy()
    if df.empty:
        return
    df["log_samples"] = np.log10(df["num_samples"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        data=df,
        x="log_samples",
        y="novelty",
        scatter_kws={"alpha": 0.6, "s": 80},
        line_kws={"color": "black", "lw": 2},
        ax=ax,
    )
    ax.set_title("Does scale correlate with novelty?")
    ax.set_xlabel("log10(# samples)")
    ax.set_ylabel("Novelty score")
    plt.tight_layout()
    path = os.path.join(output_dir, "size_vs_novelty.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def scv_hexbin(df_datasets: pd.DataFrame, output_dir: str):
    if df_datasets.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(
        df_datasets["novelty"],
        df_datasets["quality"],
        gridsize=20,
        cmap="crest",
        mincnt=1,
    )
    ax.set_xlabel("Novelty")
    ax.set_ylabel("Quality / transparency")
    ax.set_title("Density of Scientific Contribution Vector (SCV) scores")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Datasets")
    plt.tight_layout()
    path = os.path.join(output_dir, "scv_density.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def embedding_pca(df_datasets: pd.DataFrame, output_dir: str):
    if df_datasets.empty:
        return
    mask = df_datasets["embedding"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    if mask.sum() < 5:
        return

    emb = np.array(df_datasets.loc[mask, "embedding"].tolist())
    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(emb)
    df_plot = df_datasets.loc[mask].copy()
    df_plot["pc1"] = coords[:, 0]
    df_plot["pc2"] = coords[:, 1]
    df_plot["year"] = df_plot["date"].dt.year

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df_plot["pc1"],
        df_plot["pc2"],
        c=df_plot["year"],
        cmap="viridis",
        s=100,
        alpha=0.8,
    )
    ax.set_title("Landscape of dataset papers (SPECTER2 PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Publication year")
    plt.tight_layout()
    path = os.path.join(output_dir, "embedding_pca.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def affiliation_radar(df_datasets: pd.DataFrame, output_dir: str):
    if df_datasets.empty:
        return
    metrics = ["novelty", "diversity", "quality"]
    summary = df_datasets.groupby("affiliation")[metrics].mean().dropna()
    order = [aff for aff in AFFILIATION_COLORS if aff in summary.index]
    if not order:
        return

    labels = metrics
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for aff in order:
        values = summary.loc[aff].tolist()
        values += values[:1]
        color = AFFILIATION_COLORS.get(aff, "#999999")
        ax.plot(angles, values, label=aff, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label.title() for label in labels])
    ax.set_ylim(0, 1)
    ax.set_title("Average Scientific Contribution Vector (SCV) profile by affiliation")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    path = os.path.join(output_dir, "affiliation_radar.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def visualize():
    args = parse_args()
    ensure_output_dir(args.output_dir)
    cache_dir = os.environ.get("MPLCONFIGDIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        os.makedirs(cache_root, exist_ok=True)
        os.makedirs(os.path.join(cache_root, "fontconfig"), exist_ok=True)

    print(f"[visualize] Loading SCV records from {args.scv_file}")
    records = load_jsonl(args.scv_file)
    df_papers, df_datasets = build_frames(records)
    catalog = load_catalog(args.catalog_file)

    if df_papers.empty or df_datasets.empty:
        print("[visualize] No data available after parsing. Exiting.")
        return

    latest_date = df_papers["date"].max()
    start_date = latest_date - pd.DateOffset(years=args.lookback_years)
    print(f"[visualize] Using lookback window starting {start_date.date()}")

    temporal_trend_plot(df_papers, catalog, args.output_dir, start_date)
    affiliation_stack_plot(df_datasets, args.output_dir, start_date)
    domain_novelty_plot(df_datasets, args.output_dir)
    language_diversity_scatter(df_datasets, args.output_dir)
    size_vs_novelty(df_datasets, args.output_dir)
    scv_hexbin(df_datasets, args.output_dir)
    embedding_pca(df_datasets, args.output_dir)
    affiliation_radar(df_datasets, args.output_dir)


if __name__ == "__main__":
    visualize()
