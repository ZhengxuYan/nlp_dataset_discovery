"""
Dataset metadata extraction using Curator (faster batched processing).
"""

import pandas as pd
from typing import List, Dict
import argparse
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from bespokelabs import curator
from dotenv import load_dotenv
from datasets import Dataset

load_dotenv()


# Response schemas
class RelevanceResponse(BaseModel):
    is_dataset_paper: str = Field(
        description="Is this paper primarily about creating, introducing, or analyzing an NLP/text dataset? Answer: 'Yes' or 'No'"
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) for the decision"
    )


class MinimalMetadataResponse(BaseModel):
    role_in_paper: str = Field(
        description="'New dataset', 'New version', 'Reannotation', 'Composite', or 'Only existing'"
    )
    intended_primary_use: str = Field(
        description="'Pretraining', 'Fine-tuning', 'Evaluation', 'RLHF', 'Probing', or 'Synthetic' (comma-sep)"
    )
    scope_of_use: str = Field(
        description="'General purpose', 'Multi-task', 'Narrow capability', or 'Domain-specific'"
    )
    task_family: str = Field(description="Main task type")
    targeted_capability: str = Field(description="High-level capabilities (comma-sep)")
    output_type: str = Field(
        description="'Categorical', 'Multiple labels', 'Span', 'Generation', or 'Structured'"
    )
    language_coverage: str = Field(description="Language scope")
    domain: str = Field(description="Primary domain/application area")
    source_of_text: str = Field(description="Where text comes from (comma-sep)")
    collection_method: str = Field(description="How dataset was collected")
    llm_involvement: str = Field(
        description="'No LLM', 'Generated', 'Assisted', 'Filtering', or 'Unclear'"
    )
    approximate_scale: str = Field(
        description="'<10K', '10K-100K', '100K-1M', or '>1M'"
    )
    relationship_to_prior: str = Field(
        description="'From scratch', 'Filtered', 'Re-annotation', 'Augmentation', 'Mixture' (comma-sep)"
    )
    claimed_contribution: str = Field(
        description="'New task', 'New domain', 'New annotation', 'New scale', 'New protocol', or 'Re-packaging'"
    )


# Prompts
RELEVANCE_PROMPT = """Determine if a paper is about an NLP dataset.

Title: {title}

Abstract: {abstract}

Is this paper primarily about creating, introducing, or analyzing an NLP/text dataset? 
Answer 'Yes' or 'No' and provide a brief explanation (1-2 sentences).
"""

MINIMAL_METADATA_PROMPT = """Extract metadata from a dataset paper.

Title: {title}

Abstract: {abstract}

Extract the following 14 metadata dimensions from this NLP dataset paper:

1. Role in paper: Is this a 'New dataset', 'New version', 'Reannotation', 'Composite', or 'Only existing'?

2. Intended primary use: 'Pretraining', 'Fine-tuning', 'Evaluation', 'RLHF', 'Probing', or 'Synthetic' (can be comma-separated)

3. Scope of use: 'General purpose', 'Multi-task', 'Narrow capability', or 'Domain-specific'

4. Task family: What is the main task type? (e.g., 'Text classification', 'Question answering', 'Translation', 'Summarization', etc.)

5. Targeted capability: What high-level capabilities does it target? (comma-separated if multiple)

6. Output type: 'Categorical', 'Multiple labels', 'Span', 'Generation', or 'Structured'

7. Language coverage: What languages are covered? (e.g., 'English only', 'Multilingual', etc.)

8. Domain: What is the primary domain or application area?

9. Source of text: Where does the text come from? (comma-separated if multiple)

10. Collection method: How was the dataset collected?

11. LLM involvement: 'No LLM', 'Generated', 'Assisted', 'Filtering', or 'Unclear'

12. Approximate scale: '<10K', '10K-100K', '100K-1M', or '>1M' examples

13. Relationship to prior datasets: 'From scratch', 'Filtered', 'Re-annotation', 'Augmentation', 'Mixture' (comma-separated)

14. Claimed contribution: 'New task', 'New domain', 'New annotation', 'New scale', 'New protocol', or 'Re-packaging'

Provide concise, accurate answers for each dimension based on the title and abstract.
"""


# Curator LLMs
class RelevanceChecker(curator.LLM):
    response_format = RelevanceResponse

    def prompt(self, row: dict) -> str:
        return RELEVANCE_PROMPT.format(title=row["Title"], abstract=row["Abstract"])

    def parse(self, input: dict, response: RelevanceResponse) -> List[Dict]:
        return [
            {
                **input,
                "is_dataset_paper": response.is_dataset_paper,
                "skip_reasoning": response.reasoning,
            }
        ]


class MinimalMetadataExtractor(curator.LLM):
    response_format = MinimalMetadataResponse

    def prompt(self, row: dict) -> str:
        return MINIMAL_METADATA_PROMPT.format(
            title=row["Title"], abstract=row["Abstract"]
        )

    def parse(self, input: dict, response: MinimalMetadataResponse) -> List[Dict]:
        return [
            {
                **input,
                "is_dataset_paper": "Yes",
                "role_in_paper": response.role_in_paper,
                "intended_primary_use": response.intended_primary_use,
                "scope_of_use": response.scope_of_use,
                "task_family": response.task_family,
                "targeted_capability": response.targeted_capability,
                "output_type": response.output_type,
                "language_coverage": response.language_coverage,
                "domain": response.domain,
                "source_of_text": response.source_of_text,
                "collection_method": response.collection_method,
                "llm_involvement": response.llm_involvement,
                "approximate_scale": response.approximate_scale,
                "relationship_to_prior": response.relationship_to_prior,
                "claimed_contribution": response.claimed_contribution,
            }
        ]


def clean_field(value):
    """Clean a field value for CSV output."""
    if value is None:
        return ""
    value = str(value).replace("\n", " ").replace("\r", " ")
    value = " ".join(value.split())
    return value.strip()


def process_with_curator(
    input_csv: str,
    output_csv: str,
    model: str = "gpt-4o-mini",
    max_papers: int = None,
    start_index: int = 0,
    skip_relevance: bool = False,
    batch_size: int = 10,
):
    """
    Process papers using Curator for fast batched inference.

    Args:
        batch_size: Number of papers to process concurrently (default: 10)
    """
    print(f"Loading papers from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Handle slicing
    if max_papers:
        df = df.iloc[start_index : start_index + max_papers]
    else:
        df = df.iloc[start_index:]

    print(f"Processing {len(df)} papers (starting from index {start_index})...")
    print(f"Batch size: {batch_size} concurrent requests")
    print(f"Model: {model}")

    # Convert DataFrame to HuggingFace Dataset
    input_dataset = Dataset.from_pandas(df, preserve_index=False)

    if skip_relevance:
        print("WARNING: Skipping relevance check - processing all papers")
        # Process all papers directly with metadata extraction
        extractor = MinimalMetadataExtractor(model_name=model)
        result_dataset = extractor(input_dataset)
        results = result_dataset.dataset.to_pandas().to_dict("records")

    else:
        print("Step 1/2: Checking relevance...")
        # First filter by relevance
        checker = RelevanceChecker(model_name=model)
        relevance_result = checker(input_dataset)
        relevance_df = relevance_result.dataset.to_pandas()

        # Separate dataset and non-dataset papers
        dataset_papers = relevance_df[
            relevance_df["is_dataset_paper"].str.lower().isin(["yes", "y"])
        ]
        non_dataset_papers = relevance_df[
            ~relevance_df["is_dataset_paper"].str.lower().isin(["yes", "y"])
        ]

        print(f"Found {len(dataset_papers)} dataset papers")
        print(f"Skipped {len(non_dataset_papers)} non-dataset papers")

        if len(dataset_papers) > 0:
            print("\nStep 2/2: Extracting metadata from dataset papers...")
            # Extract metadata only for dataset papers
            # Convert back to Dataset
            dataset_papers_ds = Dataset.from_pandas(
                dataset_papers, preserve_index=False
            )
            extractor = MinimalMetadataExtractor(model_name=model)
            metadata_result = extractor(dataset_papers_ds)
            dataset_results = metadata_result.dataset.to_pandas().to_dict("records")

            # Combine with non-dataset papers
            non_dataset_results = non_dataset_papers.to_dict("records")
            results = dataset_results + non_dataset_results
        else:
            results = non_dataset_papers.to_dict("records")

    # Clean all fields
    cleaned_results = []
    for result in results:
        cleaned_result = {}
        # Map column names to match expected output
        column_mapping = {
            "arXiv ID": "arxiv_id",
            "Title": "title",
            "Authors": "authors",
            "Publication Date": "publication_date",
            "Abstract": "abstract",
        }
        for key, value in result.items():
            # Map original column names
            if key in column_mapping:
                cleaned_result[column_mapping[key]] = clean_field(value)
            else:
                cleaned_result[key] = clean_field(value)
        cleaned_results.append(cleaned_result)

    # Save results
    results_df = pd.DataFrame(cleaned_results)
    results_df.to_csv(output_csv, index=False)

    print(f"\nCompleted! Results saved to {output_csv}")
    print(f"Total papers processed: {len(results_df)}")
    if not skip_relevance:
        dataset_count = len(results_df[results_df["is_dataset_paper"] == "Yes"])
        print(f"Dataset papers: {dataset_count}")
        print(f"Skipped (not dataset papers): {len(results_df) - dataset_count}")
        if len(results_df) > 0:
            print(f"Success rate: {(dataset_count / len(results_df) * 100):.1f}%")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract dataset metadata using Curator (fast batched processing)"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help="Path to input CSV file with papers (default: data/raw/arxiv_results.csv)",
    )
    parser.add_argument("--output", "-o", help="Path to output CSV file")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument(
        "--max-papers", "-n", type=int, help="Maximum number of papers to process"
    )
    parser.add_argument(
        "--start-index", "-s", type=int, default=0, help="Starting index in CSV"
    )
    parser.add_argument(
        "--skip-relevance",
        action="store_true",
        help="Skip the dataset relevance check and process all papers",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )

    args = parser.parse_args()

    # Set default input path if not provided
    if args.input_csv is None:
        project_root = Path(__file__).parent.parent
        args.input_csv = str(project_root / "data" / "raw" / "arxiv_results.csv")
        print(f"Using default input: {args.input_csv}")

    # Determine output path
    if args.output:
        output_csv = args.output
    else:
        input_path = Path(args.input_csv)
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = str(
            output_dir / f"{input_path.stem}_metadata_{timestamp}.csv"
        )

    # Process papers
    process_with_curator(
        input_csv=args.input_csv,
        output_csv=output_csv,
        model=args.model,
        max_papers=args.max_papers,
        start_index=args.start_index,
        skip_relevance=args.skip_relevance,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
