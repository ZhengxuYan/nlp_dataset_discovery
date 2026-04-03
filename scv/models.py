from typing import List, Optional
from pydantic import BaseModel, Field

class DatasetSize(BaseModel):
    size_str: str = Field(description="Original text describing size (e.g. '10k examples').", default="Unknown")
    num_samples: Optional[int] = Field(description="Estimated number of samples as an integer. -1 if unknown.", default=None)
    num_classes: Optional[int] = Field(description="Number of classes/labels as an integer. -1 if unknown or not applicable.", default=None)
    storage_size: str = Field(description="Storage size (e.g. '500MB') if mentioned.", default="Unknown")

class DatasetScvInfo(BaseModel):
    name: str = Field(description="Exact name of the dataset as mentioned in the paper.")
    usage_description: str = Field(description="Detailed explanation of how and why the dataset is used.")
    role: str = Field(description="Role: 'Main Contribution', 'Training Data', 'Evaluation Benchmark', 'Fine-tuning', 'Reference/Prior Work'.")
    is_introduced: bool = Field(description="True if this paper introduces this dataset.")
    creators: str = Field(description="Who created this dataset? (Authors vs External).")
    
    # Transformation details
    transformation_type: str = Field(description="Type of transformation (e.g. 'Subset', 'Preprocessing', 'None').", default="None")
    source_dataset: str = Field(description="Name of source dataset if transformed. 'None' otherwise.", default="None")
    
    # Extended Metadata
    main_tasks: List[str] = Field(description="Main tasks/applications (e.g., Classification, QA, Summarization).", default=[])
    languages: List[str] = Field(description="Languages covered (e.g., English, Chinese, Multilingual).", default=[])
    
    # Structure/Size
    size: DatasetSize = Field(description="Structured size information.")
    
    domain: str = Field(description="Domain/Subject matter (e.g., Biomedical, Social Media, General).", default="General")
    
    # Quality & Availability
    license: str = Field(description="License type or access info (e.g., MIT, CC-BY, Restricted, Unknown).", default="Unknown")
    availability_url: str = Field(description="URL to code/data repository if mentioned.", default="None")
    documentation_type: str = Field(description="Type of documentation mentioned (e.g. 'Datasheet', 'Readme', 'Appendix', 'None').", default="None")
    maintenance_status: str = Field(description="Any mention of maintenance plan? ('Yes', 'No', 'Unknown').", default="Unknown")
    
    # Added Information & Contribution (Only if introduced)
    added_information_summary: str = Field(
        description="Short statement summarizing what new information this dataset adds relative to prior work. 'None' if not introduced.",
        default="None",
    )
    novelty_summary: str = Field(
        description="Legacy alias for added_information_summary. 'None' if not introduced.",
        default="None",
    )
    acus: List[str] = Field(description="Atomic Content Units: Decomposed short claims about the dataset's contribution. Empty if not introduced.", default=[])
    previous_work_acus: List[str] = Field(description="ACUs of previous work mentioned in the paper. Extract 3-5 key claims/facts about prior datasets/methods if mentioned.", default=[])

    # Quality Signals (Issues)
    transparency_issues: List[str] = Field(description="List of potential issues: 'Missing License', 'No Link', 'Unclear Provenance', or 'None'.", default=[])
    
    confidence: str = Field(description="Extraction confidence: High, Medium, Low.")

class AuthorInfo(BaseModel):
    name: str = Field(description="Full name of the author.")
    affiliation: str = Field(description="Affiliation of the author (University, Company, etc.). If not explicitly stated, infer from context or strictly state 'Unknown'.")

class ScvPaperAnalysis(BaseModel):
    is_nlp_paper: bool = Field(description="Is this an NLP-related paper?")
    nlp_relevance_explanation: str = Field(description="Brief explanation of why this paper is considered NLP-related or not.")
    paper_contribution_summary: str = Field(description="Brief summary of the paper's main scientific contribution.")
    
    # We rely on CSV for Title, Date, Venue per user request.
    # We ONLY extract Authors to get affiliations.
    authors: List[AuthorInfo] = Field(description="List of authors and their affiliations identified in the paper.", default=[])
    
    datasets: List[DatasetScvInfo] = Field(description="List of datasets explicitly named and used/introduced.", default=[])

class AddedInformationScoreDimension(BaseModel):
    dimension_name: str = Field(description="Name of the dimension (e.g., Task/Domain Delta, Methodology Delta, Coverage Delta).")
    explanation: str = Field(description="Explanation of the score comparing the new dataset to prior support.")
    score: float = Field(description="Score between 0.0 (fully supported by prior work) and 1.0 (substantial added information).")

class AddedInformationScoringResult(BaseModel):
    dimensions: List[AddedInformationScoreDimension] = Field(
        description="Scoring along different dimensions (Task/Domain Delta, Methodology Delta, Coverage Delta)."
    )
    average_added_information_score: float = Field(
        description="The average of the scores across all dimensions, representing the final added-information score."
    )
    average_novelty_score: Optional[float] = Field(
        description="Legacy alias for average_added_information_score.",
        default=None,
    )
