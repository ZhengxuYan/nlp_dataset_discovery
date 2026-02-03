# Scientific Contribution Vector (SCV) Pipeline

This pipeline processes research papers to extract dataset information, analyze their novelty, and compute a "Scientific Contribution Vector" (SCV) consisting of novelty, diversity, and quality scores.

## Overview

The pipeline runs in two stages:
1.  **Extraction Stage**: Downloads papers, extracts structured data (using LLMs), and computes embeddings.
2.  **Analysis Stage**: Sorts papers by date, compares "Novelty" against prior work (history), and computes final scores.

## Requirements

- Python 3.9+
- `bespokelabs` (Curator)
- `sentence-transformers`
- `pandas`, `pypdf`, `requests`
- `pydantic`

## Usage

### 1. Extraction Stage

Run this command to process papers from a CSV file. It filters for papers that *introduce* a new dataset.

```bash
python scripts/run_scv.py --stage extract --limit 200 --output data/processed/scv_intermediate.jsonl
```

**What it does:**
- Reads input CSV (default: `data/processed/arxiv_nlp_conf_papers_2023_2025.csv`).
- Downloads PDF/Source for each paper.
- Uses `gpt-5-mini` to extract:
    - Dataset metadata (Name, Usage, Tasks, Languages, Size, etc.).
    - **Novelty Claims**: "Atomic Content Units" (ACUs) - short claims about what is new.
- Computes `SPECTER2` embeddings for the paper abstract.
- **Filters**: Only saves papers that introduce at least one new dataset.

**Output:** `data/processed/scv_intermediate.jsonl`

### 2. Analysis Stage

Run this command to analyze the extracted data.

```bash
python scripts/run_scv.py --stage analyze --input data/processed/scv_intermediate.jsonl --output data/processed/scv_final_results.jsonl
```

**What it does:**
- Loads the intermediate JSONL file.
- **Sorts by Publication Date** to simulate a historical timeline.
- Iterates through papers:
    - **Novelty Score**: Compares the paper's ACUs against the ACUs of *all previous* papers in the timeline using NLI (Natural Language Inference).
        - High Entailment by history = Low Novelty.
        - Low Entailment = High Novelty.
    - **Diversity Score**: Heuristic based on languages, domain, and size.
    - **Quality Score**: Heuristic based on transparency (license, links) and availability.
- Updates the "History" with the current paper's ACUs (so it becomes prior work for future papers).

**Output:** `data/processed/scv_final_results.jsonl`

## Configuration

- **Models**:
    - Extraction: `gpt-5-mini` (via Curator).
    - Embedding: `allenai/specter2_base`.
    - NLI: `cross-encoder/nli-deberta-v3-small`.
- **Paths**: Defined in `scripts/scv_pipeline.py` (modifiable via arguments or code constants).

## Output Format

The final JSONL contains records like:

```json
{
  "arxiv_id": "2303.18121",
  "is_nlp": true,
  "datasets": [
    {
      "info": {
        "name": "New Dataset X",
        "role": "Main Contribution",
        "is_introduced": true,
        "novelty_summary": "First dataset for...",
        "acus": ["We introduce dataset X.", "It covers 50 languages."],
        ...
      },
      "scv": {
        "novelty": 0.85,    # 0.0 - 1.0 (Higher is more novel)
        "diversity": 0.6,   # 0.0 - 1.0
        "quality": 0.9      # 0.0 - 1.0
      }
    }
  ],
  "paper_embedding": [...]
}
```
