# NLP Dataset Discovery

A comprehensive toolkit for discovering and tracking new NLP dataset papers from arXiv and top-tier ML conferences.

## Overview

This project provides automated scrapers to help researchers discover newly published NLP dataset papers from two key sources:

1.**arXiv**: Scrapes papers from CS categories most relevant to NLP datasets

2.**Conference Papers**: Scrapes papers from OpenReview conferences, with special focus on the NeurIPS Datasets and Benchmarks Track

## Project Structure

```
nlp_dataset_discovery/
├── data/                    # All data files (raw and processed)
│   ├── raw/                 # Raw scraped data
│   │   └── arxiv_results.csv
│   ├── processed/           # LLM-extracted metadata
│   │   └── arxiv_results_metadata_curator_*.csv
│   ├── visualizations/      # Generated charts and graphs
│   └── reports/             # JSON analysis reports
│
├── scrapers/                # Data collection scripts
│   ├── arxiv_scraper/       # Scrapy-based arXiv scraper
│   │   └── arxiv_scraper/
│   │       └── spiders/
│   │           └── arxiv.py
│   └── conference_scraper/  # OpenReview scraper
│       ├── scrape_openreview.py
│       └── scrape_neurips_datasets_track.py
│
├── scripts/                 # Metadata extraction scripts
│   └── dataset_metadata_extractor_curator.py  # LLM-based extraction
│
├── utils/                   # Analysis and utility scripts
│   ├── analyze_metadata.py           # Metadata analysis & visualization
│   └── merge_metadata_batches.py     # Batch merging utility
│
└── scrapyenv/              # Virtual environment
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

3.**Install dependencies:**

```bash
# For conference scrapers
pip install requests pandas pyarrow

# For metadata extraction (LLM-based)
pip install bespokelabs-curator python-dotenv datasets

# For analysis and visualization
pip install matplotlib seaborn
```

4.**Set up environment variables:**

Create a `.env` file in the project root:

```bash
# OpenAI API key (required for metadata extraction)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Enable Curator Viewer for live processing visualization
CURATOR_VIEWER=1
```

## Usage

### 1. Scraping arXiv Papers

**Basic usage:**

```bash
cd scrapers/arxiv_scraper
scrapy crawl arxiv
```

Results are automatically saved to `data/raw/arxiv_results.csv`

**With custom date range:**

```bash
scrapy crawl arxiv -a start_date=2025-10-01 -a end_date=2025-11-01
```

**Configuration:**

- Edit `arxiv_scraper/spiders/arxiv.py` to modify:

-`start_date` / `end_date` (default: "2025-10-01" to "2025-11-01")

-`cs_categories` (list of arXiv categories to scrape)

-`max_articles` (limit on number of papers)

**Output:**

- CSV file: `data/raw/arxiv_results.csv`
- Columns: arXiv ID, Title, Authors, Abstract, Categories, Publication Date, URLs

### 2. Extracting Dataset Metadata with LLMs

**Extract metadata from papers using fast batched processing:**

```bash
# Process all papers in the default input file
python scripts/dataset_metadata_extractor_curator.py

# Process specific number of papers
python scripts/dataset_metadata_extractor_curator.py --max-papers 100

# Use custom input file
python scripts/dataset_metadata_extractor_curator.py data/raw/arxiv_results.csv

# Use different model
python scripts/dataset_metadata_extractor_curator.py --model gpt-4o
```

The extraction uses a two-stage process: first filtering for dataset papers, then extracting 14 metadata dimensions including role, use, scope, task, capabilities, and scale. Results are saved to `data/processed/arxiv_results_metadata_curator_TIMESTAMP.csv`

### 3. Analyzing Extracted Metadata

Run analysis and generate visualizations:

```bash
# Basic analysis (console output only)
python utils/analyze_metadata.py data/processed/arxiv_results_metadata_curator_TIMESTAMP.csv

# Generate visualizations
python utils/analyze_metadata.py data/processed/arxiv_results_metadata_curator_TIMESTAMP.csv --graphs data/visualizations/

# Save JSON analysis
python utils/analyze_metadata.py data/processed/arxiv_results_metadata_curator_TIMESTAMP.csv --json results.json
```

The analysis provides statistics on dataset classification, roles, intended uses, language coverage, LLM involvement, scale distribution, task families, domains, and contributions. When run with `--graphs`, it generates 10 visualization charts in PNG format.

### 4. Scraping NeurIPS Datasets and Benchmarks Track

```bash
cd scrapers/conference_scraper
python3 scrape_neurips_datasets_track.py
```

This scrapes both NeurIPS 2024 and 2025 Datasets and Benchmarks Track papers.

Configuration:

Edit `scrape_neurips_datasets_track.py` to modify:

```python

start_year =2024# Change to desired start year

end_year =2025# Change to desired end year

```

Output:

- CSV file: `conference_scraper/results/neurips_2024_2025_datasets_benchmarks_track.csv`
- Columns: year, id, title, abstract, authors, author_ids, venue, primary_area, decision, scores, keywords, openreview_url

### 5. General Conference Scraping (ICLR, NeurIPS, etc.)

Command line usage:

```bash

cdconference_scraper


# Scrape ICLR 2024

python3scrape_openreview.py--conferenceICLR--start-year2024--end-year2024


# Scrape NeurIPS 2024 (all papers)

python3scrape_openreview.py--conferenceNeurIPS--start-year2024--end-year2024


# With decisions and scores (SLOW - takes hours!)

python3scrape_openreview.py--conferenceICLR--start-year2024--end-year2024--fetch-decisions--fetch-scores

```

Python API:

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

All data is organized in the `data/` directory:

### `data/raw/` - Raw Scraped Data

1. arXiv Papers (`arxiv_results.csv`)

   - Papers from specified date range
   - 9 NLP-relevant CS categories
   - Updated via `scrapy crawl arxiv`

2. Conference Papers (various files)
   - NeurIPS, ICLR, etc. papers
   - Updated via conference scrapers

### `data/processed/` - Extracted Metadata

- Dataset Metadata (`arxiv_results_metadata_curator_*.csv`)
  - LLM-extracted structured metadata
  - 14 dimensions: role, use, scope, task, capabilities, scale, etc.
  - Generated via `dataset_metadata_extractor_curator.py`
  - Includes both dataset and non-dataset papers with classification

### `data/visualizations/` - Analysis Charts

- 10 visualization charts (PNG format)
  - Dataset classification, roles, uses, languages
  - LLM involvement, scale distribution
  - Task families, domains, contributions
  - Generated via `analyze_metadata.py --graphs`

### `data/reports/` - JSON Reports

- Analysis results in JSON format
  - Structured data for programmatic access
  - Generated via `analyze_metadata.py --json`

## Key Features

### arXiv Scraper Features

- Respects arXiv API rate limits (3 seconds between requests)
- Pagination support (handles 2000+ papers per category)
- Date-based filtering
- Automatic deduplication
- Resume support (appends to existing CSV)

### Conference Scraper Features

- Track-specific scraping (e.g., Datasets and Benchmarks)
- Multi-year support (2017-2025+)
- Venue and primary area extraction
- Keyword analysis
- Optional decision and review score fetching
- Proper CSV formatting (no newlines breaking rows)
- Both CSV and Parquet export

## Examples

### Analyze Dataset Metadata Statistics

```python
import pandas as pd

# Load extracted metadata
df = pd.read_csv("data/processed/arxiv_results_metadata_curator_20251114_014109.csv")

# Get dataset papers only
dataset_papers = df[df['is_dataset_paper'] == 'Yes']
print(f"Found {len(dataset_papers)} dataset papers out of {len(df)} total")

# Analyze by scale
scale_dist = dataset_papers['approximate_scale'].value_counts()
print("\nDataset Scale Distribution:")
print(scale_dist)

# Find multilingual datasets
multilingual = dataset_papers[
    dataset_papers['language_coverage'].str.contains('Multilingual', case=False, na=False)
]
print(f"\nMultilingual datasets: {len(multilingual)}")

# Analyze LLM involvement
llm_stats = dataset_papers['llm_involvement'].value_counts()
print("\nLLM Involvement:")
print(llm_stats)
```

### Filter arXiv Results by Keyword

```python
import pandas as pd

df = pd.read_csv("data/raw/arxiv_results.csv")

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

## Complete Workflow Example

Here's a typical end-to-end workflow:

```bash
# Step 1: Scrape papers from arXiv
cd scrapers/arxiv_scraper
scrapy crawl arxiv -a start_date=2025-10-01 -a end_date=2025-11-01
cd ../..

# Step 2: Extract metadata from papers (processes all papers)
python scripts/dataset_metadata_extractor_curator.py

# Step 3: Analyze and visualize the results
python utils/analyze_metadata.py \
    data/processed/arxiv_results_metadata_curator_*.csv \
    --graphs data/visualizations/ \
    --json data/reports/analysis.json
```

Expected timeline:

- Step 1 (Scraping): ~10-20 minutes for one month of papers
- Step 2 (Metadata extraction): ~20-30 minutes for 7000 papers
- Step 3 (Analysis): <1 minute

## Tips and Best Practices

### arXiv Scraper

- Use date ranges to avoid overwhelming the API
- The scraper sleeps 3 seconds between requests (arXiv requirement)
- Expected runtime: ~10-20 minutes for a month of papers across all categories
- Results append to CSV, so you can run incrementally

### Metadata Extraction

- Set `CURATOR_VIEWER=1` in `.env` to watch live progress
- Use `--max-papers` for testing before processing full dataset
- Adjust `--batch-size` based on your API rate limits
- Default model `gpt-4o-mini` is fast and cost-effective

### Analysis

- Generate visualizations with `--graphs` for quick insights
- Export to JSON with `--json` for programmatic access
- Combine with pandas for custom analysis
- All visualizations are high-resolution (300 DPI) PNG files

### Conference Scraper

- Scraping without `--fetch-decisions` and `--fetch-scores` is fast (~10 minutes)
- Adding decisions/scores requires individual API calls per paper (several hours!)
- Use `track` parameter to scrape specific tracks only
- CSV files have cleaned text (no newlines)

## Contributing

Potential extensions:

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
