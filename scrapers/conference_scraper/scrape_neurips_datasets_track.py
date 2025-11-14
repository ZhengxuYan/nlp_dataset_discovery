#!/usr/bin/env python3
"""
Scrape NeurIPS Datasets and Benchmarks Track

Scrapes papers from the official NeurIPS Datasets and Benchmarks Track.
URL: https://openreview.net/group?id=NeurIPS.cc/2024/Datasets_and_Benchmarks_Track
"""

from scrape_openreview import OpenReviewScraper
import os


def main():
    print("=" * 80)
    print("NeurIPS Datasets and Benchmarks Track Scraper")
    print("=" * 80)

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Initialize scraper for NeurIPS Datasets and Benchmarks Track
    # You can change these years to scrape different years (2024 and 2025 are available)
    start_year = 2024
    end_year = 2025  # Change to 2024 if you only want 2024

    print(
        f"\nInitializing scraper for NeurIPS {start_year}-{end_year} Datasets and Benchmarks Track..."
    )
    print(
        "Source: https://openreview.net/group?id=NeurIPS.cc/2024/Datasets_and_Benchmarks_Track"
    )
    print(
        "        https://openreview.net/group?id=NeurIPS.cc/2025/Datasets_and_Benchmarks_Track"
    )

    scraper = OpenReviewScraper(
        conference="NeurIPS",
        start_year=start_year,
        end_year=end_year,
        fetch_decisions=False,  # Set to True if you want accept/reject info (SLOW!)
        fetch_scores=False,  # Set to True if you want review scores (SLOW!)
        track="Datasets_and_Benchmarks_Track",  # Specify the track
    )

    # Scrape all papers from the track
    print("\nScraping Datasets and Benchmarks Track papers...")
    print("This should be quick - only scraping one specific track...")
    scraper.scrape_all()

    # Filter by abstract length
    print("\nFiltering out papers with short abstracts...")
    scraper.filter_papers(min_abstract_length=100)

    # Save results
    year_suffix = (
        f"{start_year}_{end_year}" if start_year != end_year else str(start_year)
    )
    output_file = f"results/neurips_{year_suffix}_datasets_benchmarks_track.csv"
    print(f"\nSaving results to {output_file}...")
    df = scraper.save_to_csv(output_file)

    # Show summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if df is not None and len(df) > 0:
        print(f"✓ Total papers in Datasets and Benchmarks Track: {len(df)}")

        # Show venue distribution
        if "venue" in df.columns:
            print("\nVenue distribution:")
            print(df["venue"].value_counts())

        # Show primary areas
        if "primary_area" in df.columns and df["primary_area"].notna().any():
            print("\nPrimary areas distribution:")
            print(df["primary_area"].value_counts())

        # Show keyword analysis
        if "keywords" in df.columns:
            print("\nTop keywords:")
            all_keywords = []
            for kws in df["keywords"]:
                if isinstance(kws, str) and kws:
                    all_keywords.extend([k.strip() for k in kws.split(",")])

            if all_keywords:
                from collections import Counter

                keyword_counts = Counter(all_keywords)
                for keyword, count in keyword_counts.most_common(15):
                    if keyword:  # Skip empty keywords
                        print(f"  {keyword}: {count}")

        # Show sample papers
        print("\n" + "-" * 80)
        print("Sample Dataset and Benchmark Papers:")
        print("-" * 80)
        for idx in range(min(10, len(df))):
            row = df.iloc[idx]
            print(f"\n{idx + 1}. {row['title']}")
            if "venue" in row and row["venue"]:
                print(f"   Venue: {row['venue']}")
            if "keywords" in row and row["keywords"]:
                keywords_str = str(row["keywords"])[:100]
                print(f"   Keywords: {keywords_str}...")
            print(f"   URL: {row['openreview_url']}")

        if len(df) > 10:
            print(f"\n... and {len(df) - 10} more papers")

        print(f"\n✓ Full results saved to: {output_file}")
        print(f"\n{'='*80}")
        print("You can now analyze these dataset papers for NLP-related content!")
        print(f"{'='*80}")
    else:
        print("⚠ No papers found in Datasets and Benchmarks track")

    print("\nDone!")


if __name__ == "__main__":
    main()
