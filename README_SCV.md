# Scientific Contribution Vector (SCV) Pipeline

This pipeline processes research papers to extract dataset information, analyze their added information relative to prior datasets, and compute a "Scientific Contribution Vector" (SCV) consisting of added-information, diversity, and quality scores.

## Overview

The pipeline runs in two stages:
1.  **Extraction Stage**: Downloads papers, extracts structured data (using LLMs), and computes embeddings.
2.  **Analysis Stage**: Sorts papers by date, compares each introduced dataset's ACUs against prior support, and computes final scores.

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
    - **Added-Information Claims**: "Atomic Content Units" (ACUs) - short claims about what the dataset adds.
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
    - **Added-Information Score**: Compares the paper's ACUs against the ACUs of *all previous* papers in the timeline using NLI or an LLM judge.
        - High support by history = Low added information.
        - Low support by history = High added information.
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
        "added_information_summary": "First dataset for...",
        "acus": ["We introduce dataset X.", "It covers 50 languages."],
        ...
      },
      "scv": {
        "added_information": 0.85,    # 0.0 - 1.0 (Higher means more unsupported new information)
        "novelty": 0.85,              # Legacy alias
        "diversity": 0.6,   # 0.0 - 1.0
        "quality": 0.9      # 0.0 - 1.0
      }
    }
  ],
  "paper_embedding": [...]
}
```

## Benchmark Workflow

Use the scaffold script to bootstrap a real manually reviewed benchmark:

```bash
python scripts/bootstrap_real_benchmark.py \
  --input data/processed/final_scv_200.jsonl \
  --output data/benchmark/real_added_information_benchmark_template.jsonl \
  --limit 25
```

Each row contains:

- a real query dataset and its ACUs
- a candidate prior-support pool
- empty gold fields for manual annotation:
  - `gold_prior_support_ids`
  - `gold_prior_support_acus`
  - `gold_added_information_label`
  - `gold_added_information_rationale`

Evaluate retrieval and added-information estimation with:

```bash
python run_benchmark.py --input data/benchmark/real_added_information_benchmark_template.jsonl
```

Implemented retrieval ablations:

- `dense`
- `lexical`
- `splade`
- `colbert`
- `rank_fusion`
- `fusion`
- `hybrid_rerank`

Implemented evaluation splits:

- retrieval quality
- oracle prior support
- end-to-end prior support + added-information estimation
