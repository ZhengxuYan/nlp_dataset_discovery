"""
Dataset metadata extraction using DSPy.
"""

import dspy
import pandas as pd
from typing import Optional
import argparse
from pathlib import Path
from datetime import datetime


class DatasetRelevanceSignature(dspy.Signature):
    """Determine if a paper is about an NLP dataset."""

    title = dspy.InputField(desc="Paper title")
    abstract = dspy.InputField(desc="Paper abstract")

    is_dataset_paper = dspy.OutputField(
        desc="Is this paper primarily about creating, introducing, or analyzing an NLP/text dataset? Answer: 'Yes' or 'No'"
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation (1-2 sentences) for the decision"
    )


class DatasetRoleSignature(dspy.Signature):
    """Extract dataset role and intended use information from a paper."""

    title = dspy.InputField(desc="Paper title")
    abstract = dspy.InputField(desc="Paper abstract")

    role_in_paper = dspy.OutputField(
        desc="Role: 'New dataset', 'New version or extension', 'Reannotation', 'Composite/mixture', or 'Only uses existing'"
    )
    intended_primary_use = dspy.OutputField(
        desc="Primary use (comma-separated if multiple): 'Pretraining', 'Fine-tuning', 'Evaluation', 'RLHF/feedback', 'Probing/analysis', or 'Synthetic-data generation'"
    )
    scope_of_use = dspy.OutputField(
        desc="Scope: 'General purpose NLP', 'Broad multi-task evaluation', 'Narrow capability-specific', or 'Domain-specific'"
    )


class TaskPhenomenonSignature(dspy.Signature):
    """Extract task, phenomenon, and supervision information."""

    title = dspy.InputField(desc="Paper title")
    abstract = dspy.InputField(desc="Paper abstract")

    task_family = dspy.OutputField(
        desc="Primary task: 'Text classification', 'Sequence labeling', 'Question answering', 'Summarization', 'Translation', 'Dialogue', 'Retrieval', 'NLI', 'Code-related', 'Structured prediction', 'Multi-task benchmark', or 'Other'"
    )
    targeted_capability = dspy.OutputField(
        desc="Capabilities (comma-separated): 'General understanding', 'Commonsense', 'Mathematical reasoning', 'Social intelligence', 'Safety/toxicity', 'Bias/fairness', 'Multilingual', 'Information extraction', or 'Other'"
    )
    output_type = dspy.OutputField(
        desc="Output type: 'Single categorical', 'Multiple labels', 'Extractive span', 'Free-form generation', or 'Structured output'"
    )
    supervision_level = dspy.OutputField(
        desc="Supervision: 'Fully supervised', 'Weak supervision', 'Unlabeled', 'Human preference', or 'Mixed'"
    )


class DataContentSignature(dspy.Signature):
    """Extract data content and population information."""

    title = dspy.InputField(desc="Paper title")
    abstract = dspy.InputField(desc="Paper abstract")

    language_coverage = dspy.OutputField(
        desc="Languages: 'English only', 'Bilingual', 'Multilingual high-resource', 'Multilingual with low-resource', or 'Code-mixed'"
    )
    domain = dspy.OutputField(
        desc="Domain: 'General web/news', 'Social media', 'Conversational', 'Scientific', 'Legal', 'Educational', 'Healthcare', 'Code/software', or 'Other specialized'"
    )
    source_of_text = dspy.OutputField(
        desc="Source (comma-separated): 'Web crawl', 'Platform logs', 'Crowdsourcing', 'Expert-authored', 'Existing corpora', 'Synthetic/LLM-generated', or 'Mixed'"
    )
    unit_of_analysis = dspy.OutputField(
        desc="Unit: 'Single sentence', 'Short paragraph', 'Document', 'Conversation turn', 'Multi-turn conversation', or 'Multi-document'"
    )
    human_population = dspy.OutputField(
        desc="Population: 'General', 'Children/students', 'Professionals', 'Geographic/cultural community', 'Marginalized/vulnerable groups', or 'Not specified'"
    )


class ConstructionQualitySignature(dspy.Signature):
    """Extract construction process, quality, and governance information."""

    title = dspy.InputField(desc="Paper title")
    abstract = dspy.InputField(desc="Paper abstract")

    collection_method = dspy.OutputField(
        desc="Method: 'Fully scraped', 'Human-authored', 'Crowd-annotated', 'Expert-annotated', 'System logs', or 'Automatically constructed'"
    )
    llm_involvement = dspy.OutputField(
        desc="LLM use: 'No LLM', 'LLM-generated text', 'LLM-assisted annotation', 'LLM filtering', or 'Unclear'"
    )
    approximate_scale = dspy.OutputField(
        desc="Scale: '<10K', '10K-100K', '100K-1M', or '>1M' examples"
    )
    label_granularity = dspy.OutputField(
        desc="Labels: 'Coarse discrete', 'Fine-grained', 'Continuous scores', or 'Mixed'"
    )
    difficulty_level = dspy.OutputField(
        desc="Difficulty: 'Trivial/near-solved', 'Moderate', or 'Hard/frontier'"
    )
    ethical_risk = dspy.OutputField(
        desc="Risk: 'Low risk', 'Contains PII', 'Contains hate speech/toxicity', 'Sensitive medical/content', or 'Not specified'"
    )
    license_accessibility = dspy.OutputField(
        desc="License: 'Open access', 'Restricted access', 'Not released', or 'License unclear'"
    )


class NoveltyRelationshipSignature(dspy.Signature):
    """Extract relationship to existing datasets and novelty claims."""

    title = dspy.InputField(desc="Paper title")
    abstract = dspy.InputField(desc="Paper abstract")

    relationship_to_prior = dspy.OutputField(
        desc="Relationship (comma-separated): 'Built from scratch', 'Filtered subset', 'Re-annotation', 'Synthetic augmentation', or 'Union/mixture'"
    )
    claimed_contribution = dspy.OutputField(
        desc="Contribution: 'New task', 'New domain', 'New annotation scheme', 'New scale', 'New evaluation protocol', or 'Re-packaging'"
    )
    expected_adoption_scope = dspy.OutputField(
        desc="Adoption: 'General benchmark', 'Subarea-specific', 'Very narrow', or 'Authors own project'"
    )
    overlap_with_existing = dspy.OutputField(
        desc="Overlap: 'Mostly novel', 'Strongly derived', 'Mixed', or 'Unclear'"
    )
    base_datasets = dspy.OutputField(
        desc="List any specific existing dataset names mentioned (comma-separated, or 'None' if not applicable)"
    )


class DatasetMetadataExtractor(dspy.Module):
    """Orchestrates metadata extraction across all dimensions."""

    def __init__(self):
        super().__init__()

        # Initialize relevance checker
        self.check_relevance = dspy.ChainOfThought(DatasetRelevanceSignature)

        # Initialize all extractors
        self.extract_role = dspy.ChainOfThought(DatasetRoleSignature)
        self.extract_task = dspy.ChainOfThought(TaskPhenomenonSignature)
        self.extract_content = dspy.ChainOfThought(DataContentSignature)
        self.extract_construction = dspy.ChainOfThought(ConstructionQualitySignature)
        self.extract_novelty = dspy.ChainOfThought(NoveltyRelationshipSignature)

    def forward(self, title: str, abstract: str, skip_relevance_check: bool = False):
        """Extract all metadata dimensions from a paper."""

        # First check if this is actually a dataset paper
        if not skip_relevance_check:
            relevance = self.check_relevance(title=title, abstract=abstract)
            if relevance.is_dataset_paper.strip().lower() not in ["yes", "y"]:
                return dspy.Prediction(
                    is_relevant=False, reasoning=relevance.reasoning, metadata=None
                )

        # Extract each category
        role_output = self.extract_role(title=title, abstract=abstract)
        task_output = self.extract_task(title=title, abstract=abstract)
        content_output = self.extract_content(title=title, abstract=abstract)
        construction_output = self.extract_construction(title=title, abstract=abstract)
        novelty_output = self.extract_novelty(title=title, abstract=abstract)

        # Combine all outputs into structured metadata
        metadata = {
            # 1. Dataset role and intended use
            "role_in_paper": role_output.role_in_paper,
            "intended_primary_use": role_output.intended_primary_use,
            "scope_of_use": role_output.scope_of_use,
            # 2. Task, phenomenon, and supervision
            "task_family": task_output.task_family,
            "targeted_capability": task_output.targeted_capability,
            "output_type": task_output.output_type,
            "supervision_level": task_output.supervision_level,
            # 3. Data content and population
            "language_coverage": content_output.language_coverage,
            "domain": content_output.domain,
            "source_of_text": content_output.source_of_text,
            "unit_of_analysis": content_output.unit_of_analysis,
            "human_population": content_output.human_population,
            # 4. Construction process, quality, and governance
            "collection_method": construction_output.collection_method,
            "llm_involvement": construction_output.llm_involvement,
            "approximate_scale": construction_output.approximate_scale,
            "label_granularity": construction_output.label_granularity,
            "difficulty_level": construction_output.difficulty_level,
            "ethical_risk": construction_output.ethical_risk,
            "license_accessibility": construction_output.license_accessibility,
            # 5. Relationship to existing datasets and novelty
            "relationship_to_prior": novelty_output.relationship_to_prior,
            "claimed_contribution": novelty_output.claimed_contribution,
            "expected_adoption_scope": novelty_output.expected_adoption_scope,
            "overlap_with_existing": novelty_output.overlap_with_existing,
            "base_datasets": novelty_output.base_datasets,
        }

        return dspy.Prediction(is_relevant=True, metadata=metadata, reasoning=None)


def clean_field(value):
    """Clean a field value for CSV output by removing problematic characters."""
    if value is None:
        return ""
    # Convert to string and replace newlines with spaces
    value = str(value).replace("\n", " ").replace("\r", " ")
    # Replace multiple spaces with single space
    value = " ".join(value.split())
    return value.strip()


def process_papers(
    input_csv: str,
    output_csv: str,
    model: str = "gpt-4o-mini",
    max_papers: Optional[int] = None,
    start_index: int = 0,
    skip_relevance: bool = False,
):
    # Configure DSPy with OpenAI model
    lm = dspy.OpenAI(model=model)
    dspy.configure(lm=lm)

    # Initialize extractor
    extractor = DatasetMetadataExtractor()

    # Load input CSV
    print(f"Loading papers from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Handle slicing
    if max_papers:
        df = df.iloc[start_index : start_index + max_papers]
    else:
        df = df.iloc[start_index:]

    print(f"Processing {len(df)} papers (starting from index {start_index})...")

    # Process each paper
    results = []
    skipped_count = 0
    for idx, row in df.iterrows():
        try:
            print(f"\nProcessing paper {idx + 1}/{len(df)}: {row['Title'][:60]}...")

            # Extract metadata (includes relevance check unless disabled)
            prediction = extractor(
                title=str(row["Title"]),
                abstract=str(row["Abstract"]),
                skip_relevance_check=skip_relevance,
            )

            # Check if paper is relevant (only if relevance check was done)
            if not skip_relevance and not prediction.is_relevant:
                skipped_count += 1
                print(f"  SKIPPED (not a dataset paper): {prediction.reasoning}")
                results.append(
                    {
                        "arxiv_id": clean_field(row["arXiv ID"]),
                        "title": clean_field(row["Title"]),
                        "authors": clean_field(row["Authors"]),
                        "publication_date": clean_field(row["Publication Date"]),
                        "abstract": clean_field(row["Abstract"]),
                        "is_dataset_paper": "No",
                        "skip_reasoning": clean_field(prediction.reasoning),
                    }
                )
                continue

            # Combine with original data and clean all fields
            result = {
                "arxiv_id": clean_field(row["arXiv ID"]),
                "title": clean_field(row["Title"]),
                "authors": clean_field(row["Authors"]),
                "publication_date": clean_field(row["Publication Date"]),
                "abstract": clean_field(row["Abstract"]),
                "is_dataset_paper": "Yes",
            }

            # Clean metadata fields
            for key, value in prediction.metadata.items():
                result[key] = clean_field(value)

            results.append(result)

            # Save incrementally every 10 papers
            if (idx + 1) % 10 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(output_csv, index=False)
                print(f"Saved checkpoint at {idx + 1} papers")

        except Exception as e:
            print(f"Error processing paper {idx}: {e}")
            # Add a row with error
            results.append(
                {
                    "arxiv_id": clean_field(row["arXiv ID"]),
                    "title": clean_field(row["Title"]),
                    "authors": clean_field(row["Authors"]),
                    "publication_date": clean_field(row["Publication Date"]),
                    "abstract": clean_field(row["Abstract"]),
                    "error": clean_field(str(e)),
                }
            )

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nCompleted! Results saved to {output_csv}")
    print(f"Total papers processed: {len(results)}")
    print(f"Dataset papers: {len(results) - skipped_count}")
    print(f"Skipped (not dataset papers): {skipped_count}")
    if len(results) > 0:
        print(
            f"Success rate: {((len(results) - skipped_count) / len(results) * 100):.1f}%"
        )

    return results_df


def process_minimal_schema(
    input_csv: str,
    output_csv: str,
    model: str = "gpt-4o-mini",
    max_papers: Optional[int] = None,
    start_index: int = 0,
    skip_relevance: bool = False,
):
    # Configure DSPy with OpenAI model
    lm = dspy.OpenAI(model=model)
    dspy.configure(lm=lm)

    # Initialize relevance checker
    relevance_checker = dspy.ChainOfThought(DatasetRelevanceSignature)

    # Create a combined signature for minimal schema
    class MinimalMetadataSignature(dspy.Signature):
        """Extract prioritized metadata dimensions from a dataset paper."""

        title = dspy.InputField(desc="Paper title")
        abstract = dspy.InputField(desc="Paper abstract")

        # Role and use (3)
        role_in_paper = dspy.OutputField(
            desc="'New dataset', 'New version', 'Reannotation', 'Composite', or 'Only existing'"
        )
        intended_primary_use = dspy.OutputField(
            desc="'Pretraining', 'Fine-tuning', 'Evaluation', 'RLHF', 'Probing', or 'Synthetic' (comma-sep)"
        )
        scope_of_use = dspy.OutputField(
            desc="'General purpose', 'Multi-task', 'Narrow capability', or 'Domain-specific'"
        )

        # Task (3)
        task_family = dspy.OutputField(desc="Main task type")
        targeted_capability = dspy.OutputField(
            desc="High-level capabilities (comma-sep)"
        )
        output_type = dspy.OutputField(
            desc="'Categorical', 'Multiple labels', 'Span', 'Generation', or 'Structured'"
        )

        # Content (3)
        language_coverage = dspy.OutputField(desc="Language scope")
        domain = dspy.OutputField(desc="Primary domain/application area")
        source_of_text = dspy.OutputField(desc="Where text comes from (comma-sep)")

        # Construction (3)
        collection_method = dspy.OutputField(desc="How dataset was collected")
        llm_involvement = dspy.OutputField(
            desc="'No LLM', 'Generated', 'Assisted', 'Filtering', or 'Unclear'"
        )
        approximate_scale = dspy.OutputField(
            desc="'<10K', '10K-100K', '100K-1M', or '>1M'"
        )

        # Novelty (2)
        relationship_to_prior = dspy.OutputField(
            desc="'From scratch', 'Filtered', 'Re-annotation', 'Augmentation', 'Mixture' (comma-sep)"
        )
        claimed_contribution = dspy.OutputField(
            desc="'New task', 'New domain', 'New annotation', 'New scale', 'New protocol', or 'Re-packaging'"
        )

    # Initialize extractor
    extractor = dspy.ChainOfThought(MinimalMetadataSignature)

    # Load input CSV
    print(f"Loading papers from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Handle slicing
    if max_papers:
        df = df.iloc[start_index : start_index + max_papers]
    else:
        df = df.iloc[start_index:]

    print(
        f"Processing {len(df)} papers with minimal schema (starting from index {start_index})..."
    )

    # Process each paper
    results = []
    skipped_count = 0
    for idx, row in df.iterrows():
        try:
            print(f"\nProcessing paper {idx + 1}/{len(df)}: {row['Title'][:60]}...")

            # First check relevance (unless skipping)
            if not skip_relevance:
                relevance = relevance_checker(
                    title=str(row["Title"]), abstract=str(row["Abstract"])
                )

                if relevance.is_dataset_paper.strip().lower() not in ["yes", "y"]:
                    skipped_count += 1
                    print(f"  SKIPPED (not a dataset paper): {relevance.reasoning}")
                    results.append(
                        {
                            "arxiv_id": clean_field(row["arXiv ID"]),
                            "title": clean_field(row["Title"]),
                            "authors": clean_field(row["Authors"]),
                            "publication_date": clean_field(row["Publication Date"]),
                            "abstract": clean_field(row["Abstract"]),
                            "is_dataset_paper": "No",
                            "skip_reasoning": clean_field(relevance.reasoning),
                        }
                    )
                    continue

            # Extract metadata
            output = extractor(title=str(row["Title"]), abstract=str(row["Abstract"]))

            # Combine with original data and clean all fields
            result = {
                "arxiv_id": clean_field(row["arXiv ID"]),
                "title": clean_field(row["Title"]),
                "authors": clean_field(row["Authors"]),
                "publication_date": clean_field(row["Publication Date"]),
                "abstract": clean_field(row["Abstract"]),
                "is_dataset_paper": "Yes",
                "role_in_paper": clean_field(output.role_in_paper),
                "intended_primary_use": clean_field(output.intended_primary_use),
                "scope_of_use": clean_field(output.scope_of_use),
                "task_family": clean_field(output.task_family),
                "targeted_capability": clean_field(output.targeted_capability),
                "output_type": clean_field(output.output_type),
                "language_coverage": clean_field(output.language_coverage),
                "domain": clean_field(output.domain),
                "source_of_text": clean_field(output.source_of_text),
                "collection_method": clean_field(output.collection_method),
                "llm_involvement": clean_field(output.llm_involvement),
                "approximate_scale": clean_field(output.approximate_scale),
                "relationship_to_prior": clean_field(output.relationship_to_prior),
                "claimed_contribution": clean_field(output.claimed_contribution),
            }

            results.append(result)

            # Save incrementally every 10 papers
            if (idx + 1) % 10 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(output_csv, index=False)
                print(f"Saved checkpoint at {idx + 1} papers")

        except Exception as e:
            print(f"Error processing paper {idx}: {e}")
            results.append(
                {
                    "arxiv_id": clean_field(row["arXiv ID"]),
                    "title": clean_field(row["Title"]),
                    "error": clean_field(str(e)),
                }
            )

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nCompleted! Results saved to {output_csv}")
    print(f"Total papers processed: {len(results)}")
    print(f"Dataset papers: {len(results) - skipped_count}")
    print(f"Skipped (not dataset papers): {skipped_count}")
    if len(results) > 0:
        print(
            f"Success rate: {((len(results) - skipped_count) / len(results) * 100):.1f}%"
        )

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract dataset metadata from research papers using DSPy"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help="Path to input CSV file with papers (default: data/raw/arxiv_results.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output CSV file (default: input_name + _metadata.csv)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-papers",
        "-n",
        type=int,
        help="Maximum number of papers to process (default: all)",
    )
    parser.add_argument(
        "--start-index",
        "-s",
        type=int,
        default=0,
        help="Starting index in CSV (for resuming, default: 0)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use minimal 14-dimension schema for faster initial pass",
    )
    parser.add_argument(
        "--skip-relevance",
        action="store_true",
        help="Skip the dataset relevance check and process all papers",
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
        # Default to saving in data/processed/ directory
        input_path = Path(args.input_csv)
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        schema_type = "minimal" if args.minimal else "full"
        output_csv = str(
            output_dir / f"{input_path.stem}_metadata_{schema_type}_{timestamp}.csv"
        )

    # Process papers
    if args.minimal:
        print("Using MINIMAL schema (14 dimensions)")
        if args.skip_relevance:
            print("WARNING: Relevance check disabled - processing all papers")
        process_minimal_schema(
            input_csv=args.input_csv,
            output_csv=output_csv,
            model=args.model,
            max_papers=args.max_papers,
            start_index=args.start_index,
            skip_relevance=args.skip_relevance,
        )
    else:
        print("Using FULL schema (all dimensions)")
        if args.skip_relevance:
            print("WARNING: Relevance check disabled - processing all papers")
        process_papers(
            input_csv=args.input_csv,
            output_csv=output_csv,
            model=args.model,
            max_papers=args.max_papers,
            start_index=args.start_index,
            skip_relevance=args.skip_relevance,
        )


if __name__ == "__main__":
    main()
