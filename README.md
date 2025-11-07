# NLP Dataset Discovery

A comprehensive toolkit for discovering and tracking new NLP dataset papers from arXiv and top-tier ML conferences.

## Overview

This project provides automated scrapers to help researchers discover newly published NLP dataset papers from two key sources:

1.**arXiv**: Scrapes papers from CS categories most relevant to NLP datasets

2.**Conference Papers**: Scrapes papers from OpenReview conferences, with special focus on the NeurIPS Datasets and Benchmarks Track

## Project Structure

```

nlp_dataset_discovery/

├── arxiv_scraper/           # Scrapy-based arXiv scraper

│   ├── arxiv_scraper/

│   │   ├── spiders/

│   │   │   └── arxiv.py     # Main arXiv spider

│   │   ├── pipelines.py

│   │   └── settings.py

│   ├── results/

│   │   └── arxiv_results.csv

│   └── filter_by_date.py

│

├── conference_scraper/      # OpenReview conference scraper

│   ├── scrape_openreview.py          # Main scraper class

│   ├── scrape_neurips_datasets_track.py  # NeurIPS D&B track scraper

│   └── results/

│       └── neurips_2024_2025_datasets_benchmarks_track.csv

│

└── README.md

```

## Features

### arXiv Scraper

- Scrapes papers from 9 CS categories most relevant to NLP datasets
- Filters by date range
- Handles arXiv API rate limiting automatically
- Outputs comprehensive CSV with titles, abstracts, authors, categories, and links

**Targeted Categories:**

-`cs.CL` - Computation and Language (primary home for NLP datasets)

-`cs.LG` - Machine Learning (dataset papers as ML contributions)

-`cs.IR` - Information Retrieval (search, ranking, QA corpora)

-`cs.HC` - Human-Computer Interaction (annotation studies, dialog)

-`cs.SI` - Social Networks (discourse, social language)

-`cs.CY` - Computers and Society (bias, fairness, toxicity datasets)

-`cs.MA` - Multiagent Systems (multi-agent language interactions)

-`cs.MM` - Multimedia (multimodal datasets: text + images/video/audio)

-`cs.DL` - Digital Libraries (corpus curation, document processing)

### Conference Scraper

- Scrapes papers from OpenReview conferences (ICLR, NeurIPS, etc.)
- Special support for NeurIPS Datasets and Benchmarks Track
- Extracts venue, primary area, keywords, decisions, and review scores
- Handles both API v1 (2017-2023) and v2 (2024+)
- Proper CSV formatting (no multi-row papers)

## Installation

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Setup

1.**Clone the repository:**

```bash

cd"nlp_dataset_discovery"

```

2.**Set up the Scrapy environment (for arXiv scraper):**

```bash

# Activate the existing virtual environment

sourcescrapyenv/bin/activate


# Or create a new one

python3-mvenvscrapyenv

sourcescrapyenv/bin/activate

pipinstallscrapy

```

3.**Install dependencies for conference scraper:**

```bash

pipinstallrequestspandaspyarrow

```

## Usage

### 1. Scraping arXiv Papers

**Basic usage:**

```bash

cdarxiv_scraper

scrapycrawlarxiv

```

**With custom date range:**

```bash

scrapycrawlarxiv-astart_date=2025-10-01-aend_date=2025-11-01

```

**Configuration:**

- Edit `arxiv_scraper/spiders/arxiv.py` to modify:

-`start_date` / `end_date` (default: "2025-10-01" to "2025-11-01")

-`cs_categories` (list of arXiv categories to scrape)

-`max_articles` (limit on number of papers)

**Output:**

- CSV file: `arxiv_scraper/results/arxiv_results.csv`
- Columns: Link/DOI, Publication Date, Title, Authors, Abstract, Categories

### 2. Scraping NeurIPS Datasets and Benchmarks Track

**Quick start:**

```bash

cdconference_scraper

python3scrape_neurips_datasets_track.py

```

This will scrape both NeurIPS 2024 and 2025 Datasets and Benchmarks Track papers.

**Configuration:**

Edit `scrape_neurips_datasets_track.py` to modify:

```python

start_year =2024# Change to desired start year

end_year =2025# Change to desired end year

```

**Output:**

- CSV file: `conference_scraper/results/neurips_2024_2025_datasets_benchmarks_track.csv`
- Columns: year, id, title, abstract, authors, author_ids, venue, primary_area, decision, scores, keywords, openreview_url

### 3. General Conference Scraping (ICLR, NeurIPS, etc.)

**Command line:**

```bash

cdconference_scraper


# Scrape ICLR 2024

python3scrape_openreview.py--conferenceICLR--start-year2024--end-year2024


# Scrape NeurIPS 2024 (all papers)

python3scrape_openreview.py--conferenceNeurIPS--start-year2024--end-year2024


# With decisions and scores (SLOW - takes hours!)

python3scrape_openreview.py--conferenceICLR--start-year2024--end-year2024--fetch-decisions--fetch-scores

```

**Python API:**

```python

from scrape_openreview import OpenReviewScraper


# Scrape a specific track

scraper =OpenReviewScraper(

conference="NeurIPS",

start_year=2024,

end_year=2024,

track="Datasets_and_Benchmarks_Track"# Optional: specify track

)


scraper.scrape_all()

scraper.filter_papers(min_abstract_length=100)


# Save to CSV

scraper.save_to_csv("output.csv")


# Or save to Parquet (preserves list types)

scraper.save_to_parquet("output.parquet")

```

## Data Files

### Current Dataset Collections

1.**arXiv Papers** (`arxiv_scraper/results/arxiv_results.csv`)

- ~5,868 papers from Oct-Nov 2025
- 9 NLP-relevant CS categories

  2.**NeurIPS Datasets and Benchmarks Track** (`conference_scraper/results/neurips_2024_2025_datasets_benchmarks_track.csv`)

- 956 papers (459 from 2024, 497 from 2025)
- Includes posters, spotlights, and oral presentations
- Official track: [NeurIPS 2024 D&amp;B](https://openreview.net/group?id=NeurIPS.cc/2024/Datasets_and_Benchmarks_Track) | [NeurIPS 2025 D&amp;B](https://openreview.net/group?id=NeurIPS.cc/2025/Datasets_and_Benchmarks_Track)

## Key Features

### arXiv Scraper Features

- ✅ Respects arXiv API rate limits (3 seconds between requests)
- ✅ Pagination support (handles 2000+ papers per category)
- ✅ Date-based filtering
- ✅ Automatic deduplication
- ✅ Resume support (appends to existing CSV)

### Conference Scraper Features

- ✅ Track-specific scraping (e.g., Datasets and Benchmarks)
- ✅ Multi-year support (2017-2025+)
- ✅ Venue and primary area extraction
- ✅ Keyword analysis
- ✅ Optional decision and review score fetching
- ✅ Proper CSV formatting (no newlines breaking rows)
- ✅ Both CSV and Parquet export

## Examples

### Filter arXiv Results by Keyword

```python

import pandas as pd


df = pd.read_csv("arxiv_scraper/results/arxiv_results.csv")


# Find papers with "dataset" in title or abstract

dataset_papers = df[

    df['Title'].str.contains('dataset', case=False, na=False) |

    df['Abstract'].str.contains('dataset', case=False, na=False)

]


print(f"Found {len(dataset_papers)} dataset papers")

```

### Analyze NeurIPS Dataset Papers by Keywords

```python

import pandas as pd

from collections import Counter


df = pd.read_csv("conference_scraper/results/neurips_2024_2025_datasets_benchmarks_track.csv")


# Extract all keywords

all_keywords = []

for kws in df['keywords']:

ifisinstance(kws, str) and kws:

        all_keywords.extend([k.strip() for k in kws.split(',')])


# Top 10 keywords

keyword_counts =Counter(all_keywords)

print(keyword_counts.most_common(10))


# Filter for NLP-related papers

nlp_keywords = ['nlp', 'language', 'text', 'linguistic', 'dialogue', 'translation']

nlp_papers = df[df['keywords'].str.contains('|'.join(nlp_keywords), case=False, na=False)]

print(f"\nFound {len(nlp_papers)} NLP-related dataset papers")

```

## Tips and Best Practices

### arXiv Scraper

- Use date ranges to avoid overwhelming the API
- The scraper sleeps 3 seconds between requests (arXiv requirement)
- Expected runtime: ~10-20 minutes for a month of papers across all categories
- Results append to CSV, so you can run incrementally

### Conference Scraper

- Scraping without `--fetch-decisions` and `--fetch-scores` is fast (~10 minutes)
- Adding decisions/scores requires individual API calls per paper (several hours!)
- Use `track` parameter to scrape specific tracks only
- CSV files have cleaned text (no newlines)

## Contributing

Feel free to extend the scrapers to support:

- Additional arXiv categories
- Other conferences (ACL, EMNLP, ICML, etc.)
- More sophisticated filtering (ML-based relevance detection)
- Automatic paper classification (dataset type, domain, etc.)

## Resources

- [arXiv API Documentation](https://arxiv.org/help/api/)
- [OpenReview API](https://docs.openreview.net/reference/api-v2/entities/note)
- [NeurIPS 2024 Datasets and Benchmarks Track](https://openreview.net/group?id=NeurIPS.cc/2024/Datasets_and_Benchmarks_Track)
- [NeurIPS 2025 Datasets and Benchmarks Track](https://openreview.net/group?id=NeurIPS.cc/2025/Datasets_and_Benchmarks_Track)

## License

This project is for research and educational purposes.

## Notes

- arXiv scraper uses Scrapy framework
- Conference scraper uses requests + pandas
- All text fields are cleaned (newlines removed) in CSV outputs
- Rate limiting is automatically handled for both scrapers
