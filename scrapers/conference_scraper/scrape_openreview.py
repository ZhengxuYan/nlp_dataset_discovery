"""
OpenReview Conference Paper Scraper

A general scraper for conferences hosted on OpenReview (ICLR, NeurIPS, etc.)
Supports both API v1 (2017-2023) and v2 (2024+).
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime


class OpenReviewScraper:
    """General scraper for OpenReview conference papers"""

    def __init__(
        self,
        conference="NeurIPS",
        start_year=2025,
        end_year=2025,
        fetch_decisions=False,
        fetch_scores=False,
        track=None,
    ):
        """
        Initialize the OpenReview scraper

        Args:
            conference (str): Conference name (e.g., "ICLR", "NeurIPS")
            start_year (int): First year to scrape
            end_year (int): Last year to scrape
            fetch_decisions (bool): Whether to fetch decisions (SLOW - takes hours!)
            fetch_scores (bool): Whether to fetch review scores (SLOW - takes hours!)
            track (str): Optional track name (e.g., "Datasets_and_Benchmarks_Track")
        """
        self.conference = conference
        self.start_year = start_year
        self.end_year = end_year
        self.fetch_decisions = fetch_decisions
        self.fetch_scores = fetch_scores
        self.track = track
        self.papers = []

        # OpenReview API endpoints
        self.api_v1 = "https://api.openreview.net/notes"
        self.api_v2 = "https://api2.openreview.net/notes"

        # Query types for different submission statuses
        self.query_types = [
            "submission",
            "Submission",
            "Blind_Submission",
            "Withdrawn_Submission",
            "Rejected_Submission",
            "Desk_Rejected_Submission",
            "",  # For 2024+ to get all papers
        ]

        # Rate limiting
        self.request_delay = 1  # seconds between requests
        self.rate_limit_delay = 30  # seconds to wait when rate limited

    def make_request(self, url):
        """Make API request with rate limiting handling"""
        try:
            response = requests.get(url)
            data = response.json()

            # Check for rate limiting
            if "name" in data and data["name"] == "RateLimitError":
                print(f"\nRate limited! Waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
                response = requests.get(url)
                data = response.json()

            time.sleep(self.request_delay)
            return data

        except Exception as e:
            print(f"\nError making request to {url}: {e}")
            return None

    def build_url(self, year, query_type, offset=0, track=None):
        """Build API URL based on year and query type

        Args:
            year: Conference year
            query_type: Submission type (Submission, Withdrawn_Submission, etc.)
            offset: Pagination offset
            track: Optional track name (e.g., "Datasets_and_Benchmarks_Track")
        """
        conf = self.conference

        if year <= 2017:
            if query_type == "":
                return None
            return f"{self.api_v1}?invitation={conf}.cc%2F{year}%2Fconference%2F-%2F{query_type}&offset={offset}"

        elif year <= 2023:
            if query_type == "":
                return None
            return f"{self.api_v1}?invitation={conf}.cc%2F{year}%2FConference%2F-%2F{query_type}&offset={offset}"

        else:  # 2024+
            # Handle track-specific queries
            if track:
                return f"{self.api_v2}?content.venueid={conf}.cc/{year}/{track}&offset={offset}"

            query_suffix = f"/{query_type}" if query_type != "" else ""
            return f"{self.api_v2}?content.venueid={conf}.cc/{year}/Conference{query_suffix}&offset={offset}"

    def extract_paper_data(self, note, year, query_type):
        """Extract relevant data from a note (paper) object"""
        try:
            content = note.get("content", {})

            # Handle different API versions (v1 vs v2)
            if year < 2024:
                title = content.get("title", "").strip()
                abstract = content.get("abstract", "").strip()
                keywords = content.get("keywords", [])
                authors_list = content.get("authors", [])

                # Handle author names
                if isinstance(authors_list, list):
                    authors = ", ".join(authors_list)
                else:
                    authors = authors_list if authors_list else ""

                # Handle author IDs (email-based for older years)
                if year == 2017:
                    # 2017 uses different field names
                    if "authorids" in content:
                        author_ids = ", ".join(content["authorids"])
                    elif "author_emails" in content:
                        author_ids = content["author_emails"]
                    else:
                        author_ids = ""
                else:
                    author_ids = ", ".join(content.get("authorids", []))

            else:  # 2024+
                title = content.get("title", {}).get("value", "").strip()
                abstract = content.get("abstract", {}).get("value", "").strip()
                keywords = content.get("keywords", {}).get("value", [])

                if "authors" in content:
                    authors = ", ".join(content.get("authors", {}).get("value", []))
                    author_ids = ", ".join(
                        content.get("authorids", {}).get("value", [])
                    )
                else:
                    authors = ""
                    author_ids = ""

            # Extract venue/track information (available in 2024+)
            if year >= 2024:
                venue = content.get("venue", {}).get("value", "")
                primary_area = content.get("primary_area", {}).get("value", "")
            else:
                venue = ""
                primary_area = ""

            # Remove author IDs for years <= 2020 (email-based IDs are not useful)
            if year <= 2020:
                author_ids = ""

            # Determine initial decision status based on query type
            if "Withdrawn_Submission" in query_type:
                decision = "Withdrawn"
            elif "Desk_Rejected_Submission" in query_type:
                decision = "Desk rejected"
            elif "Rejected_Submission" in query_type:
                decision = "Reject"
            else:
                decision = ""

            paper = {
                "year": year,
                "id": note.get("forum", note.get("id", "")),
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "author_ids": author_ids,
                "keywords": [k.lower() for k in keywords]
                if isinstance(keywords, list)
                else keywords,
                "venue": venue,
                "primary_area": primary_area,
                "decision": decision,
                "scores": [],
                "openreview_url": f"https://openreview.net/forum?id={note.get('forum', note.get('id', ''))}",
            }

            return paper

        except Exception as e:
            print(f"\nError extracting paper data: {e}")
            return None

    def fetch_decisions_and_scores_for_papers(self):
        """
        Fetch decisions and scores by querying each paper's forum individually.
        WARNING: This is VERY SLOW - takes several hours for thousands of papers!
        Based on the notebook approach in cell 6.
        """
        print("\n" + "=" * 80)
        print(f"FETCHING DECISIONS AND SCORES (This will take several hours!)")
        print(f"Total papers to process: {len(self.papers)}")
        print("=" * 80)

        for num, paper in enumerate(self.papers):
            # Progress indicator
            if (num + 1) % 1000 == 0:
                print("*", end="", flush=True)
            elif (num + 1) % 100 == 0:
                print(".", end="", flush=True)

            year = paper["year"]
            forum_id = paper["id"]

            # Skip if decision already set (from query type)
            if not self.fetch_decisions and paper["decision"] != "":
                continue

            # Build forum URL based on API version
            if year < 2024:
                forum_url = f"{self.api_v1}?forum={forum_id}"
            else:
                forum_url = f"{self.api_v2}?forum={forum_id}"

            # Make request
            data = self.make_request(forum_url)
            if data is None:
                continue

            notes = data.get("notes", [])

            # Fetch decision if requested
            if self.fetch_decisions and paper["decision"] == "":
                found_decision = False
                for note in notes:
                    content = note.get("content", {})

                    # Check for decision field
                    if "decision" in content:
                        decision = content["decision"]
                        if year >= 2024 and isinstance(decision, dict):
                            decision = decision.get("value", "")
                        paper["decision"] = decision
                        found_decision = True
                        break

                    # Check for withdrawal confirmation (2024+)
                    if "withdrawal_confirmation" in content:
                        paper["decision"] = "Withdrawn"
                        found_decision = True
                        break

                    # Check for desk rejection
                    if "desk_reject_comments" in content:
                        paper["decision"] = "Desk rejected"
                        found_decision = True
                        break

                    # Check for recommendation field (used in some years)
                    if "recommendation" in content:
                        decision = content["recommendation"]
                        if isinstance(decision, dict):
                            decision = decision.get("value", "")
                        paper["decision"] = decision
                        found_decision = True
                        break

                    # Check for withdrawal field (ICLR 2017)
                    if "withdrawal" in content:
                        if content["withdrawal"] == "Confirmed":
                            paper["decision"] = "Withdrawn"
                            found_decision = True
                            break

            # Fetch scores if requested
            if self.fetch_scores:
                scores = []
                for note in notes:
                    content = note.get("content", {})

                    # Check for rating field
                    if "rating" in content:
                        rating = content["rating"]
                        if year >= 2024 and isinstance(rating, dict):
                            rating = rating.get("value", "")

                        # Extract numeric score
                        if isinstance(rating, str):
                            try:
                                score = int(rating.split(":")[0])
                                scores.append(score)
                            except:
                                pass
                        elif isinstance(rating, (int, float)):
                            scores.append(int(rating))

                paper["scores"] = scores

        print("\n")
        return self.papers

    def scrape_year(self, year):
        """Scrape all papers for a given year"""
        track_info = f" ({self.track})" if self.track else ""
        print(f"\nScraping {self.conference} {year}{track_info}:")
        year_papers = []

        # If scraping a specific track, don't use query types
        if self.track:
            offset = 0
            while True:
                url = self.build_url(year, "", offset, track=self.track)

                if url is None:
                    break

                data = self.make_request(url)

                if data is None:
                    break

                notes = data.get("notes", [])

                if len(notes) == 0:
                    break

                print(f"{len(notes)} ", end="", flush=True)

                for note in notes:
                    paper = self.extract_paper_data(note, year, "")
                    if paper:
                        year_papers.append(paper)

                # Check if we need to continue pagination
                if len(notes) < 1000:
                    break

                offset += 1000
        else:
            # Original logic for scraping all papers
            for query_type in self.query_types:
                offset = 0

                while True:
                    url = self.build_url(year, query_type, offset)

                    if url is None:
                        break

                    data = self.make_request(url)

                    if data is None:
                        break

                    notes = data.get("notes", [])

                    if len(notes) == 0:
                        break

                    print(f"{len(notes)} ", end="", flush=True)

                    for note in notes:
                        paper = self.extract_paper_data(note, year, query_type)
                        if paper:
                            year_papers.append(paper)

                    # Check if we need to continue pagination
                    if len(notes) < 1000:
                        break

                    offset += 1000

        print(f"\nTotal papers for {year}: {len(year_papers)}")
        return year_papers

    def scrape_all(self):
        """Scrape all papers from start_year to end_year"""
        print(
            f"Scraping {self.conference} papers from {self.start_year} to {self.end_year}"
        )
        print("=" * 60)

        all_papers = []

        for year in range(self.start_year, self.end_year + 1):
            year_papers = self.scrape_year(year)
            all_papers.extend(year_papers)

        self.papers = all_papers
        print("\n" + "=" * 60)
        print(f"Total papers scraped: {len(all_papers)}")

        # Fetch decisions and scores if requested
        if self.fetch_decisions or self.fetch_scores:
            self.fetch_decisions_and_scores_for_papers()

        return all_papers

    def filter_papers(self, min_abstract_length=100):
        """Filter out papers with very short abstracts (likely placeholders)"""
        print(
            f"\nFiltering papers with abstract length < {min_abstract_length} characters..."
        )

        filtered_papers = []
        removed_count = 0

        for paper in self.papers:
            if len(paper["abstract"]) >= min_abstract_length:
                filtered_papers.append(paper)
            else:
                removed_count += 1

        print(f"Removed {removed_count} papers with short abstracts")
        self.papers = filtered_papers

        return filtered_papers

    def filter_by_track(self, track_keyword):
        """
        Filter papers by track/venue keyword (case-insensitive)

        Useful examples:
            - "Datasets and Benchmarks" for NeurIPS Datasets and Benchmarks track
            - "poster" for poster presentations
            - "oral" for oral presentations
        """
        print(f"\nFiltering papers with '{track_keyword}' in venue...")

        filtered_papers = []
        for paper in self.papers:
            venue = paper.get("venue", "").lower()
            if track_keyword.lower() in venue:
                filtered_papers.append(paper)

        print(f"Found {len(filtered_papers)} papers matching '{track_keyword}'")
        self.papers = filtered_papers

        return filtered_papers

    def to_dataframe(self):
        """Convert papers to pandas DataFrame"""
        if not self.papers:
            print("No papers to convert!")
            return None

        df = pd.DataFrame(self.papers)

        # Sort by year and id
        df = df.sort_values(by=["year", "id"]).reset_index(drop=True)

        # Reorder columns
        columns = [
            "year",
            "id",
            "title",
            "abstract",
            "authors",
            "author_ids",
            "venue",
            "primary_area",
            "decision",
            "scores",
            "keywords",
            "openreview_url",
        ]
        df = df[columns]

        return df

    def save_to_csv(self, output_file="papers.csv"):
        """Save scraped papers to CSV file"""
        df = self.to_dataframe()
        if df is None:
            return None

        # Convert lists to strings for CSV
        df_csv = df.copy()
        df_csv["scores"] = df_csv["scores"].apply(
            lambda x: ", ".join(map(str, x))
            if isinstance(x, list) and len(x) > 0
            else ""
        )
        df_csv["keywords"] = df_csv["keywords"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

        # Clean text fields: replace newlines and multiple spaces
        text_fields = [
            "title",
            "abstract",
            "authors",
            "author_ids",
            "venue",
            "primary_area",
        ]
        for field in text_fields:
            if field in df_csv.columns:
                df_csv[field] = df_csv[field].apply(
                    lambda x: " ".join(str(x).split()) if pd.notna(x) else ""
                )

        # Save with proper escaping and quoting
        df_csv.to_csv(
            output_file, index=False, encoding="utf-8", escapechar="\\", quoting=1
        )
        print(f"\nSaved {len(df_csv)} papers to {output_file}")

        return df

    def save_to_parquet(self, output_file="papers.parquet"):
        """Save scraped papers to Parquet file"""
        df = self.to_dataframe()
        if df is None:
            return None

        # Save to parquet (preserves list types)
        df.to_parquet(output_file)
        print(f"\nSaved {len(df)} papers to {output_file}")

        return df


def main():
    """
    Main function to run the scraper

    Example usage:
        # Quick scrape without decisions (fast - 10 minutes)
        python scrape_iclr.py --conference ICLR --start-year 2024 --end-year 2024

        # Full scrape with decisions and scores (slow - several hours!)
        python scrape_iclr.py --conference ICLR --start-year 2024 --end-year 2025 --fetch-decisions --fetch-scores
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape papers from OpenReview conferences"
    )
    parser.add_argument(
        "--conference",
        type=str,
        default="ICLR",
        help="Conference name (e.g., ICLR, NeurIPS)",
    )
    parser.add_argument("--start-year", type=int, default=2024, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument(
        "--fetch-decisions",
        action="store_true",
        help="Fetch accept/reject decisions (SLOW - takes hours!)",
    )
    parser.add_argument(
        "--fetch-scores",
        action="store_true",
        help="Fetch review scores (SLOW - takes hours!)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "parquet"],
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (auto-generated if not provided)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Initialize scraper
    print(
        f"Initializing scraper for {args.conference} ({args.start_year}-{args.end_year})"
    )
    if args.fetch_decisions or args.fetch_scores:
        print("WARNING: Fetching decisions/scores will take several hours!")

    scraper = OpenReviewScraper(
        conference=args.conference,
        start_year=args.start_year,
        end_year=args.end_year,
        fetch_decisions=args.fetch_decisions,
        fetch_scores=args.fetch_scores,
    )

    # Scrape all papers
    scraper.scrape_all()

    # Filter out placeholder abstracts
    scraper.filter_papers(min_abstract_length=100)

    # Generate output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conf_lower = args.conference.lower()
        ext = "csv" if args.format == "csv" else "parquet"
        output_file = f"results/{conf_lower}_papers_{args.start_year}_{args.end_year}_{timestamp}.{ext}"
    else:
        output_file = args.output

    # Save to file
    if args.format == "csv":
        df = scraper.save_to_csv(output_file)
    else:
        df = scraper.save_to_parquet(output_file)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(df.groupby("year")["id"].count().rename("papers_per_year"))

    if args.fetch_decisions:
        print("\nDecision breakdown:")
        print(df["decision"].value_counts())

    if args.fetch_scores:
        # Calculate average scores
        scores_df = df[df["scores"].apply(lambda x: len(x) > 0)]
        if len(scores_df) > 0:
            avg_scores = scores_df["scores"].apply(lambda x: sum(x) / len(x))
            print(f"\nAverage score: {avg_scores.mean():.2f}")

    print("\n")


if __name__ == "__main__":
    main()
