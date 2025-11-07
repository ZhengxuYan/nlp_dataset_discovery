"""
Filter arXiv papers by publication date

This script removes papers published after a specified date.
"""

import pandas as pd
from datetime import datetime
import argparse


def filter_papers_by_date(
    input_csv, output_csv, cutoff_date, date_column="Publication Date"
):
    """
    Filter papers to only include those published before or on the cutoff date

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        cutoff_date: Cutoff date (YYYY-MM-DD)
        date_column: Name of the date column
    """
    print(f"Loading papers from {input_csv}...")
    df = pd.read_csv(input_csv)

    original_count = len(df)
    print(f"Total papers: {original_count}")

    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Parse cutoff date
    cutoff = pd.to_datetime(cutoff_date)

    print(f"\nFiltering papers published on or before {cutoff_date}...")

    # Filter papers
    df_filtered = df[df[date_column] <= cutoff]

    removed_count = original_count - len(df_filtered)

    print(f"\nResults:")
    print(f"  Papers kept: {len(df_filtered)}")
    print(f"  Papers removed: {removed_count}")
    print(
        f"  Date range: {df_filtered[date_column].min()} to {df_filtered[date_column].max()}"
    )

    # Show year distribution
    year_dist = df_filtered[date_column].dt.year.value_counts().sort_index()
    print(f"\nPapers by year:")
    for year, count in year_dist.items():
        print(f"  {int(year)}: {count}")

    # Save filtered data
    print(f"\nSaving to {output_csv}...")
    df_filtered.to_csv(output_csv, index=False)

    print(f"✓ Done! Filtered dataset saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter arXiv papers by publication date"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/arxiv_results.csv",
        help="Input CSV file (default: results/arxiv_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/arxiv_results_filtered.csv",
        help="Output CSV file (default: results/arxiv_results_filtered.csv)",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2025-03-15",
        help="Cutoff date in YYYY-MM-DD format (default: 2025-03-15)",
    )
    parser.add_argument(
        "--date-column",
        type=str,
        default="Publication Date",
        help="Name of the date column (default: 'Publication Date')",
    )

    args = parser.parse_args()

    filter_papers_by_date(
        input_csv=args.input,
        output_csv=args.output,
        cutoff_date=args.cutoff_date,
        date_column=args.date_column,
    )


if __name__ == "__main__":
    main()

