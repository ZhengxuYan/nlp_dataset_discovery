import os
import time
from datetime import datetime
import scrapy
from scrapy import Request
import urllib.parse
import csv
import xml.etree.ElementTree as ET


class ArxivSpider(scrapy.Spider):
    name = "arxiv"

    sleep_time = 3  # arXiv API recommends 3 seconds between requests
    articles_fetched = 0  # Counter for articles written to CSV
    articles_processed = 0  # Counter for total articles processed (including skipped)
    start_index = 0  # For pagination
    current_category_index = 0  # Track which CS subcategory we're querying

    # Max results per API request - arXiv API allows up to 2000 per request
    max_results_per_request = 2000

    # Default date range for fetching papers (can be overridden with -a start_date=... -a end_date=...)
    start_date = "2025-10-01"
    end_date = "2025-11-01"

    # Computer Science subcategories relevant for NLP datasets
    cs_categories = [
        "cs.CL",  # Primary home for NLP datasets - Computation and Language
        "cs.LG",  # Some dataset papers frame dataset as broader ML contribution
        "cs.IR",  # Search, retrieval, ranking, web data, QA corpora
        "cs.HC",  # Human annotation studies, dialog, user behavior corpora
        "cs.SI",  # Social networks, discourse, online communities, misinformation
        "cs.CY",  # Societal impact, bias, toxicity, fairness datasets
        "cs.MA",  # Multi-agent interactions in language
        "cs.MM",  # Multimodal datasets (text + images/video/audio)
        "cs.DL",  # Digital library, corpus curation, document processing
    ]

    def __init__(self, *args, **kwargs):
        super(ArxivSpider, self).__init__(*args, **kwargs)

        # Create results directory if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Open file using relative path
        results_file = os.path.join(results_dir, "arxiv_results.csv")

        # Check if file exists and has content to determine if we need header
        file_exists = os.path.exists(results_file) and os.path.getsize(results_file) > 0

        # Open file in append mode
        self.file = open(results_file, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.max_articles = 200000  # Maximum number of articles you want to fetch

        # Write the header row only if file is new/empty
        if not file_exists:
            self.writer.writerow(
                [
                    "Link/DOI",
                    "Publication Date",
                    "Title",
                    "Authors",
                    "Abstract",
                    "Categories",
                ]
            )

    def close_spider(self, spider):
        self.file.close()

    def build_api_url(self, start_index=0, category_index=None):
        """Build arXiv API query URL for specific CS subcategory in date range"""
        if category_index is None:
            category_index = self.current_category_index

        # Get current category
        if category_index >= len(self.cs_categories):
            return None  # No more categories to query

        category = self.cs_categories[category_index]

        # Format dates for arXiv API (YYYYMMDD)
        start_date_formatted = self.start_date.replace("-", "")
        end_date_formatted = self.end_date.replace("-", "")

        # Search specific CS subcategory with date range
        search_query = f"cat:{category} AND submittedDate:[{start_date_formatted} TO {end_date_formatted}]"

        query_params = {
            "search_query": search_query,
            "start": start_index,
            "max_results": self.max_results_per_request,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        api_url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            query_params
        )
        return api_url

    def start_requests(self):
        self.logger.info(f"Fetching papers from {self.start_date} to {self.end_date}")
        self.logger.info("Using arXiv API (official method, no CAPTCHA restrictions)")
        self.logger.info(
            f"Searching {len(self.cs_categories)} Computer Science subcategories"
        )
        self.logger.info(f"Starting with category: {self.cs_categories[0]}")
        yield Request(
            url=self.build_api_url(start_index=0),
            callback=self.parse_api_response,
            meta={"dont_obey_robotstxt": True},
        )

    def parse_api_response(self, response):
        """Parse XML response from arXiv API"""
        self.logger.info(f"Response status: {response.status}")

        # Parse XML response
        try:
            root = ET.fromstring(response.body)
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse XML: {e}")
            return

        # Define namespaces
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        # Get total results
        total_results_elem = root.find("atom:totalResults", namespaces)
        total_results = (
            int(total_results_elem.text) if total_results_elem is not None else 0
        )
        self.logger.info(f"Total results available: {total_results}")

        # Get all entry elements (papers)
        entries = root.findall("atom:entry", namespaces)
        self.logger.info(
            f"Found {len(entries)} papers in this batch (start_index: {self.start_index})"
        )

        # Process each paper (if any)
        for entry in entries:
            self.articles_processed += 1

            try:
                # Extract paper details from XML
                title_elem = entry.find("atom:title", namespaces)
                title = (
                    title_elem.text.strip().replace("\n", " ")
                    if title_elem is not None
                    else ""
                )

                # Get arXiv URL from the id element
                id_elem = entry.find("atom:id", namespaces)
                arxiv_url = id_elem.text if id_elem is not None else ""

                # Get published date
                published_elem = entry.find("atom:published", namespaces)
                if published_elem is not None:
                    published_date = datetime.fromisoformat(
                        published_elem.text.replace("Z", "+00:00")
                    )
                    publication_date = published_date.date().isoformat()
                else:
                    publication_date = ""

                # Filter by date range - skip papers outside our range
                if publication_date:
                    if publication_date < self.start_date:
                        # We've gone past our date range (papers are sorted descending)
                        # Stop processing further
                        self.logger.info(
                            f"Reached papers before start_date ({self.start_date}). Stopping."
                        )
                        self.logger.info(
                            f"Processed {self.articles_processed} papers total, "
                            f"fetched {self.articles_fetched} within date range"
                        )
                        return
                    elif publication_date > self.end_date:
                        # Skip papers that are too recent
                        self.logger.debug(
                            f"Skipping paper from {publication_date} (after end_date)"
                        )
                        continue

                # Get authors
                author_elems = entry.findall("atom:author", namespaces)
                authors = []
                for author in author_elems:
                    name_elem = author.find("atom:name", namespaces)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                authors_str = "; ".join(authors)

                # Get abstract
                summary_elem = entry.find("atom:summary", namespaces)
                abstract = (
                    summary_elem.text.strip().replace("\n", " ")
                    if summary_elem is not None
                    else ""
                )

                # Get categories
                category_elems = entry.findall("atom:category", namespaces)
                categories = []
                for category in category_elems:
                    term = category.get("term")
                    if term:
                        categories.append(term)
                categories_str = "; ".join(categories)

                # Write to CSV
                self.writer.writerow(
                    [
                        arxiv_url,
                        publication_date,
                        title,
                        authors_str,
                        abstract,
                        categories_str,
                    ]
                )

                self.articles_fetched += 1

                # Check if we've reached the limit
                if self.articles_fetched >= self.max_articles:
                    self.logger.info(
                        f"Reached the limit of {self.max_articles} articles."
                    )
                    return

            except Exception as e:
                self.logger.error(f"Error processing entry: {e}")
                continue

        # Update start index for next batch
        self.start_index += len(entries)

        self.logger.info(f"Total articles fetched so far: {self.articles_fetched}")

        # Check if there are more results to fetch
        if self.articles_fetched >= self.max_articles:
            # Already reached our limit
            self.logger.info(
                f"Reached the limit of {self.max_articles} articles. Stopping."
            )
            return

        if len(entries) > 0:
            # Got entries, keep going with same category
            self.logger.info(
                f"Fetching next batch from {self.cs_categories[self.current_category_index]} (start_index: {self.start_index})"
            )
            self.logger.info(
                f"Sleeping for {self.sleep_time} seconds (arXiv API rate limiting)"
            )
            time.sleep(self.sleep_time)

            yield Request(
                url=self.build_api_url(start_index=self.start_index),
                callback=self.parse_api_response,
                meta=response.meta,
            )
        else:
            # No entries - move to next category
            self.current_category_index += 1
            self.start_index = 0  # Reset start index for new category

            if self.current_category_index >= len(self.cs_categories):
                # All categories exhausted
                self.logger.info(
                    f"Finished all {len(self.cs_categories)} categories! Total articles fetched: {self.articles_fetched}"
                )
                return

            # Move to next category
            next_category = self.cs_categories[self.current_category_index]
            self.logger.info(f"Finished category, moving to next: {next_category}")
            self.logger.info(
                f"Sleeping for {self.sleep_time} seconds (arXiv API rate limiting)"
            )
            time.sleep(self.sleep_time)

            yield Request(
                url=self.build_api_url(start_index=0),
                callback=self.parse_api_response,
                meta=response.meta,
            )
