"""
Analyze extracted dataset metadata to generate insights and visualizations.

Usage:
    python analyze_metadata.py results.csv
    python analyze_metadata.py results.csv --report output_report.txt
    python analyze_metadata.py results.csv --graphs output_dir/
"""

import pandas as pd
import argparse
from pathlib import Path
from collections import Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns


def load_metadata(csv_file: str) -> pd.DataFrame:
    """Load metadata CSV."""
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} papers from {csv_file}")
    return df


def analyze_dataset_roles(df: pd.DataFrame) -> dict:
    """Analyze dataset roles and contributions."""

    print("\n" + "=" * 80)
    print("DATASET ROLES AND CONTRIBUTIONS")
    print("=" * 80)

    results = {}

    # Role distribution
    if "role_in_paper" in df.columns:
        role_counts = df["role_in_paper"].value_counts()
        print("\nRole in Paper:")
        for role, count in role_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {role:40s}: {count:5d} ({pct:5.1f}%)")
        results["role_distribution"] = role_counts.to_dict()

    # Intended use
    if "intended_primary_use" in df.columns:
        uses = []
        for val in df["intended_primary_use"].dropna():
            uses.extend([u.strip() for u in str(val).split(",")])
        use_counts = Counter(uses)
        print("\nIntended Primary Use:")
        for use, count in use_counts.most_common():
            print(f"  {use:40s}: {count:5d}")
        results["use_distribution"] = dict(use_counts)

    # Scope
    if "scope_of_use" in df.columns:
        scope_counts = df["scope_of_use"].value_counts()
        print("\nScope of Use:")
        for scope, count in scope_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {scope:40s}: {count:5d} ({pct:5.1f}%)")
        results["scope_distribution"] = scope_counts.to_dict()

    return results


def analyze_tasks_and_capabilities(df: pd.DataFrame) -> dict:
    """Analyze task families and targeted capabilities."""

    print("\n" + "=" * 80)
    print("TASKS AND CAPABILITIES")
    print("=" * 80)

    results = {}

    # Task families
    if "task_family" in df.columns:
        task_counts = df["task_family"].value_counts()
        print("\nTask Family:")
        for task, count in task_counts.head(15).items():
            pct = (count / len(df)) * 100
            print(f"  {task:40s}: {count:5d} ({pct:5.1f}%)")
        results["task_distribution"] = task_counts.to_dict()

    # Targeted capabilities
    if "targeted_capability" in df.columns:
        capabilities = []
        for val in df["targeted_capability"].dropna():
            capabilities.extend([c.strip() for c in str(val).split(",")])
        cap_counts = Counter(capabilities)
        print("\nTargeted Capabilities:")
        for cap, count in cap_counts.most_common(10):
            print(f"  {cap:40s}: {count:5d}")
        results["capability_distribution"] = dict(cap_counts)

    # Output types
    if "output_type" in df.columns:
        output_counts = df["output_type"].value_counts()
        print("\nOutput Type:")
        for output, count in output_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {output:40s}: {count:5d} ({pct:5.1f}%)")
        results["output_distribution"] = output_counts.to_dict()

    return results


def analyze_data_content(df: pd.DataFrame) -> dict:
    """Analyze data content and sources."""

    print("\n" + "=" * 80)
    print("DATA CONTENT AND SOURCES")
    print("=" * 80)

    results = {}

    # Language coverage
    if "language_coverage" in df.columns:
        lang_counts = df["language_coverage"].value_counts()
        print("\nLanguage Coverage:")
        for lang, count in lang_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {lang:40s}: {count:5d} ({pct:5.1f}%)")
        results["language_distribution"] = lang_counts.to_dict()

    # Domains
    if "domain" in df.columns:
        domain_counts = df["domain"].value_counts()
        print("\nDomain:")
        for domain, count in domain_counts.head(15).items():
            pct = (count / len(df)) * 100
            print(f"  {domain:40s}: {count:5d} ({pct:5.1f}%)")
        results["domain_distribution"] = domain_counts.to_dict()

    # Data sources
    if "source_of_text" in df.columns:
        sources = []
        for val in df["source_of_text"].dropna():
            sources.extend([s.strip() for s in str(val).split(",")])
        source_counts = Counter(sources)
        print("\nSource of Text:")
        for source, count in source_counts.most_common(10):
            print(f"  {source:40s}: {count:5d}")
        results["source_distribution"] = dict(source_counts)

    return results


def analyze_construction(df: pd.DataFrame) -> dict:
    """Analyze construction methods and scale."""

    print("\n" + "=" * 80)
    print("CONSTRUCTION AND SCALE")
    print("=" * 80)

    results = {}

    # Collection method
    if "collection_method" in df.columns:
        method_counts = df["collection_method"].value_counts()
        print("\nCollection Method:")
        for method, count in method_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {method:40s}: {count:5d} ({pct:5.1f}%)")
        results["method_distribution"] = method_counts.to_dict()

    # LLM involvement
    if "llm_involvement" in df.columns:
        llm_counts = df["llm_involvement"].value_counts()
        print("\nLLM Involvement:")
        for llm, count in llm_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {llm:40s}: {count:5d} ({pct:5.1f}%)")
        results["llm_distribution"] = llm_counts.to_dict()

        # Calculate percentage with LLM involvement
        llm_involved = df[
            df["llm_involvement"].str.contains("LLM", case=False, na=False)
        ]
        llm_pct = (len(llm_involved) / len(df)) * 100
        print(f"\n  → {len(llm_involved)} datasets ({llm_pct:.1f}%) involve LLMs")

    # Scale
    if "approximate_scale" in df.columns:
        scale_counts = df["approximate_scale"].value_counts()
        print("\nApproximate Scale:")
        # Order by size
        scale_order = ["<10K", "10K-100K", "100K-1M", ">1M"]
        for scale in scale_order:
            if scale in scale_counts:
                count = scale_counts[scale]
                pct = (count / len(df)) * 100
                print(f"  {scale:40s}: {count:5d} ({pct:5.1f}%)")
        results["scale_distribution"] = scale_counts.to_dict()

    return results


def analyze_novelty(df: pd.DataFrame) -> dict:
    """Analyze novelty and relationships to prior work."""

    print("\n" + "=" * 80)
    print("NOVELTY AND RELATIONSHIPS")
    print("=" * 80)

    results = {}

    # Relationship to prior datasets
    if "relationship_to_prior" in df.columns:
        relationships = []
        for val in df["relationship_to_prior"].dropna():
            relationships.extend([r.strip() for r in str(val).split(",")])
        rel_counts = Counter(relationships)
        print("\nRelationship to Prior Datasets:")
        for rel, count in rel_counts.most_common():
            print(f"  {rel:40s}: {count:5d}")
        results["relationship_distribution"] = dict(rel_counts)

    # Claimed contribution
    if "claimed_contribution" in df.columns:
        contrib_counts = df["claimed_contribution"].value_counts()
        print("\nClaimed Contribution:")
        for contrib, count in contrib_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {contrib:40s}: {count:5d} ({pct:5.1f}%)")
        results["contribution_distribution"] = contrib_counts.to_dict()

    # Expected adoption scope
    if "expected_adoption_scope" in df.columns:
        adoption_counts = df["expected_adoption_scope"].value_counts()
        print("\nExpected Adoption Scope:")
        for adoption, count in adoption_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {adoption:40s}: {count:5d} ({pct:5.1f}%)")
        results["adoption_distribution"] = adoption_counts.to_dict()

    return results


def analyze_temporal_trends(df: pd.DataFrame) -> dict:
    """Analyze temporal trends if publication dates are available."""

    if "publication_date" not in df.columns:
        return {}

    print("\n" + "=" * 80)
    print("TEMPORAL TRENDS")
    print("=" * 80)

    results = {}

    # Convert to datetime
    df["pub_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["pub_year"] = df["pub_date"].dt.year

    # Papers per year
    year_counts = df["pub_year"].value_counts().sort_index()
    print("\nPapers per Year:")
    for year, count in year_counts.items():
        if pd.notna(year):
            print(f"  {int(year):4d}: {count:5d} papers")
    results["yearly_distribution"] = year_counts.to_dict()

    # LLM involvement over time
    if "llm_involvement" in df.columns:
        llm_by_year = (
            df[df["llm_involvement"].str.contains("LLM", case=False, na=False)]
            .groupby("pub_year")
            .size()
        )
        total_by_year = df.groupby("pub_year").size()
        llm_pct_by_year = (llm_by_year / total_by_year * 100).round(1)

        print("\nLLM Involvement by Year:")
        for year, pct in llm_pct_by_year.items():
            if pd.notna(year):
                print(f"  {int(year):4d}: {pct:5.1f}% of papers")
        results["llm_temporal_trend"] = llm_pct_by_year.to_dict()

    return results


def find_interesting_subsets(df: pd.DataFrame):
    """Identify interesting subsets of datasets."""

    print("\n" + "=" * 80)
    print("INTERESTING SUBSETS")
    print("=" * 80)

    # New large-scale datasets
    if "role_in_paper" in df.columns and "approximate_scale" in df.columns:
        new_large = df[
            (df["role_in_paper"].str.contains("New", case=False, na=False))
            & (df["approximate_scale"] == ">1M")
        ]
        print(f"\nNew Large-Scale Datasets (>1M examples): {len(new_large)}")
        if len(new_large) > 0 and len(new_large) <= 10:
            for _, row in new_large.iterrows():
                print(f"  - {row['title']}")

    # LLM-generated datasets
    if "llm_involvement" in df.columns:
        llm_generated = df[
            df["llm_involvement"].str.contains("generated", case=False, na=False)
        ]
        print(f"\nLLM-Generated Datasets: {len(llm_generated)}")
        if len(llm_generated) > 0 and len(llm_generated) <= 10:
            for _, row in llm_generated.iterrows():
                print(f"  - {row['title']}")

    # Multilingual datasets
    if "language_coverage" in df.columns:
        multilingual = df[
            df["language_coverage"].str.contains("Multilingual", case=False, na=False)
        ]
        print(f"\nMultilingual Datasets: {len(multilingual)}")

    # Safety/toxicity datasets
    if "targeted_capability" in df.columns:
        safety = df[
            df["targeted_capability"].str.contains(
                "safety|toxicity", case=False, na=False
            )
        ]
        print(f"\nSafety/Toxicity Datasets: {len(safety)}")
        if len(safety) > 0 and len(safety) <= 10:
            for _, row in safety.iterrows():
                print(f"  - {row['title']}")

    # Benchmark suites
    if "scope_of_use" in df.columns:
        benchmarks = df[
            df["scope_of_use"].str.contains("Multi-task", case=False, na=False)
        ]
        print(f"\nMulti-Task Benchmark Suites: {len(benchmarks)}")
        if len(benchmarks) > 0 and len(benchmarks) <= 10:
            for _, row in benchmarks.iterrows():
                print(f"  - {row['title']}")


def generate_summary_stats(df: pd.DataFrame) -> dict:
    """Generate overall summary statistics."""

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    stats = {
        "total_papers": len(df),
        "date_range": (
            str(df["publication_date"].min())
            if "publication_date" in df.columns
            else "N/A",
            str(df["publication_date"].max())
            if "publication_date" in df.columns
            else "N/A",
        ),
    }

    print(f"\nTotal Papers: {stats['total_papers']}")
    print(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")

    # Count missing values
    print("\nData Completeness:")
    for col in df.columns:
        if col not in ["arxiv_id", "title", "authors", "publication_date", "abstract"]:
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            if missing > 0:
                print(
                    f"  {col:30s}: {len(df) - missing:5d} / {len(df):5d} ({100 - missing_pct:5.1f}% complete)"
                )

    return stats


def analyze_dataset_classification(df: pd.DataFrame) -> dict:
    """Analyze dataset vs non-dataset classification."""

    print("\n" + "=" * 80)
    print("DATASET CLASSIFICATION")
    print("=" * 80)

    results = {}

    if "is_dataset_paper" in df.columns:
        dataset_counts = df["is_dataset_paper"].value_counts()
        print("\nDataset Paper Classification:")
        for classification, count in dataset_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {classification:20s}: {count:5d} ({pct:5.1f}%)")
        results["dataset_classification"] = dataset_counts.to_dict()

        # Get dataset papers only
        dataset_papers = df[df["is_dataset_paper"].str.lower().isin(["yes", "y"])]
        print(f"\nTotal Dataset Papers: {len(dataset_papers)}")
        print(f"Total Non-Dataset Papers: {len(df) - len(dataset_papers)}")

    return results


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualizations for the metadata analysis."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # Filter to dataset papers only
    dataset_df = (
        df[df["is_dataset_paper"].str.lower().isin(["yes", "y"])]
        if "is_dataset_paper" in df.columns
        else df
    )

    print(f"\nCreating visualizations in {output_dir}...")

    # 1. Dataset vs Non-Dataset Papers
    if "is_dataset_paper" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        dataset_counts = df["is_dataset_paper"].value_counts()
        colors = ["#2ecc71", "#e74c3c"]
        wedges, texts, autotexts = ax.pie(
            dataset_counts.values,
            labels=dataset_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
        )
        ax.set_title("Dataset vs Non-Dataset Papers", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            output_path / "01_dataset_classification.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Created: 01_dataset_classification.png")

    # 2. Role in Paper (for dataset papers only)
    if "role_in_paper" in dataset_df.columns and len(dataset_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        role_counts = dataset_df["role_in_paper"].value_counts().head(10)
        role_counts.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Count", fontsize=12)
        ax.set_title("Top 10 Dataset Roles", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path / "02_role_in_paper.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ Created: 02_role_in_paper.png")

    # 3. Intended Primary Use
    if "intended_primary_use" in dataset_df.columns and len(dataset_df) > 0:
        uses = []
        for val in dataset_df["intended_primary_use"].dropna():
            uses.extend([u.strip() for u in str(val).split(",")])
        use_counts = Counter(uses).most_common(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        labels, values = zip(*use_counts)
        ax.barh(range(len(labels)), values, color="coral")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Count", fontsize=12)
        ax.set_title("Top 10 Intended Primary Uses", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path / "03_intended_use.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ Created: 03_intended_use.png")

    # 4. Language Coverage
    if "language_coverage" in dataset_df.columns and len(dataset_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        lang_counts = dataset_df["language_coverage"].value_counts().head(10)
        lang_counts.plot(kind="bar", ax=ax, color="mediumseagreen")
        ax.set_xlabel("Language", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Top 10 Language Coverage", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            output_path / "04_language_coverage.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Created: 04_language_coverage.png")

    # 5. LLM Involvement
    if "llm_involvement" in dataset_df.columns and len(dataset_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        llm_counts = dataset_df["llm_involvement"].value_counts()
        colors_llm = sns.color_palette("Set2", len(llm_counts))
        llm_counts.plot(kind="bar", ax=ax, color=colors_llm)
        ax.set_xlabel("LLM Involvement", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            "LLM Involvement in Dataset Creation", fontsize=14, fontweight="bold"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            output_path / "05_llm_involvement.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Created: 05_llm_involvement.png")

    # 6. Approximate Scale
    if "approximate_scale" in dataset_df.columns and len(dataset_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        scale_order = ["<10K", "10K-100K", "100K-1M", ">1M"]
        scale_counts = dataset_df["approximate_scale"].value_counts()
        # Reorder according to scale_order
        ordered_counts = [scale_counts.get(scale, 0) for scale in scale_order]

        ax.bar(scale_order, ordered_counts, color="skyblue", edgecolor="navy")
        ax.set_xlabel("Dataset Scale", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Dataset Scale Distribution", fontsize=14, fontweight="bold")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path / "06_dataset_scale.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ Created: 06_dataset_scale.png")

    # 7. Task Family (top 15)
    if "task_family" in dataset_df.columns and len(dataset_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        task_counts = dataset_df["task_family"].value_counts().head(15)
        task_counts.plot(kind="barh", ax=ax, color="mediumpurple")
        ax.set_xlabel("Count", fontsize=12)
        ax.set_title("Top 15 Task Families", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path / "07_task_families.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ Created: 07_task_families.png")

    # 8. Domain (top 15)
    if "domain" in dataset_df.columns and len(dataset_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        domain_counts = dataset_df["domain"].value_counts().head(15)
        domain_counts.plot(kind="barh", ax=ax, color="lightcoral")
        ax.set_xlabel("Count", fontsize=12)
        ax.set_title("Top 15 Domains", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path / "08_domains.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ Created: 08_domains.png")

    # 9. Claimed Contribution
    if "claimed_contribution" in dataset_df.columns and len(dataset_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        contrib_counts = dataset_df["claimed_contribution"].value_counts()
        contrib_counts.plot(kind="bar", ax=ax, color="gold", edgecolor="darkorange")
        ax.set_xlabel("Contribution Type", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Claimed Contributions", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            output_path / "09_claimed_contributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ Created: 09_claimed_contributions.png")

    # 10. Scope of Use
    if "scope_of_use" in dataset_df.columns and len(dataset_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        scope_counts = dataset_df["scope_of_use"].value_counts()
        colors_scope = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
        wedges, texts, autotexts = ax.pie(
            scope_counts.values,
            labels=scope_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors_scope[: len(scope_counts)],
        )
        ax.set_title("Scope of Use Distribution", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path / "10_scope_of_use.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ Created: 10_scope_of_use.png")

    print(f"\n✓ All visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset metadata extraction results"
    )
    parser.add_argument("metadata_csv", help="Path to metadata CSV file")
    parser.add_argument("--report", "-r", help="Save analysis report to file")
    parser.add_argument("--json", "-j", help="Save analysis results as JSON")
    parser.add_argument("--graphs", "-g", help="Directory to save visualization graphs")

    args = parser.parse_args()

    # Load data
    df = load_metadata(args.metadata_csv)

    # Run analyses
    all_results = {}

    all_results["dataset_classification"] = analyze_dataset_classification(df)
    all_results["summary"] = generate_summary_stats(df)
    all_results["roles"] = analyze_dataset_roles(df)
    all_results["tasks"] = analyze_tasks_and_capabilities(df)
    all_results["content"] = analyze_data_content(df)
    all_results["construction"] = analyze_construction(df)
    all_results["novelty"] = analyze_novelty(df)
    all_results["temporal"] = analyze_temporal_trends(df)

    find_interesting_subsets(df)

    # Create visualizations if requested
    if args.graphs:
        create_visualizations(df, args.graphs)

    # Save report if requested
    if args.report:
        # TODO: Capture stdout to file
        print(f"\nReport would be saved to: {args.report}")

    # Save JSON if requested
    if args.json:
        # Convert to JSON-serializable format
        json_results = {}
        for key, value in all_results.items():
            if isinstance(value, dict):
                json_results[key] = {
                    str(k): (int(v) if isinstance(v, (int, float)) else str(v))
                    for k, v in value.items()
                }

        with open(args.json, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nJSON results saved to: {args.json}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
