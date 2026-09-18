from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


ACU_TYPES = {
    "task/domain",
    "data/source",
    "annotation/protocol",
    "scale/coverage",
    "evaluation/use",
    "availability/quality",
    "governance/ethics",
    "other",
}

IMPORTANCE_LEVELS = {"low", "medium", "high"}


class SourceTextInfo(BaseModel):
    type: str = Field(default="unknown")
    path: str | None = Field(default=None)
    char_count: int = Field(default=0)
    truncated: bool = Field(default=False)


class AuthorSectorInfo(BaseModel):
    name: str = ""
    affiliations: list[str] = Field(default_factory=list)
    sector: str = "unknown"


class InstitutionProfile(BaseModel):
    authors: list[AuthorSectorInfo] = Field(default_factory=list)
    lead_author_sector: str = "unknown"
    paper_sector: str = "unknown"
    industry_orgs: list[str] = Field(default_factory=list)
    academic_orgs: list[str] = Field(default_factory=list)
    countries_or_regions: list[str] = Field(default_factory=list)


class DatasetIdentity(BaseModel):
    canonical_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    acronym: str | None = None
    version: str | None = None
    is_new_dataset: bool | None = None
    is_dataset_family: bool = False
    parent_dataset_family: str | None = None


class SourceDatasetReference(BaseModel):
    name: str = ""
    relationship: str = "unclear"
    evidence: str = ""


class SyntheticGenerationInfo(BaseModel):
    uses_llm: bool | None = None
    model_names: list[str] = Field(default_factory=list)
    human_verification: str = "unclear"


class DatasetConstruction(BaseModel):
    source_data_origin: str = "unclear"
    source_datasets: list[SourceDatasetReference] = Field(default_factory=list)
    collection_method: str = "unclear"
    transformation_types: list[str] = Field(default_factory=list)
    annotation_protocol: str = "unclear"
    annotator_type: str = "unknown"
    num_annotators: int | None = None
    quality_control: str = "unclear"
    synthetic_generation: SyntheticGenerationInfo = Field(default_factory=SyntheticGenerationInfo)


class DatasetScale(BaseModel):
    size_text: str = "unknown"
    num_instances: int | None = None
    num_tokens: int | None = None
    num_documents: int | None = None
    num_dialogues: int | None = None
    num_images: int | None = None
    num_audio_hours: float | None = None
    num_languages: int | None = None
    num_domains: int | None = None
    splits: dict[str, Any] = Field(default_factory=dict)


class ArtifactLinks(BaseModel):
    dataset_urls: list[str] = Field(default_factory=list)
    project_page_urls: list[str] = Field(default_factory=list)
    code_urls: list[str] = Field(default_factory=list)
    huggingface_ids: list[str] = Field(default_factory=list)
    github_repos: list[str] = Field(default_factory=list)
    zenodo_urls: list[str] = Field(default_factory=list)
    osf_urls: list[str] = Field(default_factory=list)
    kaggle_urls: list[str] = Field(default_factory=list)
    paperswithcode_urls: list[str] = Field(default_factory=list)
    other_urls: list[str] = Field(default_factory=list)


class DatasetAvailability(BaseModel):
    release_status: str = "unclear"
    artifacts: ArtifactLinks = Field(default_factory=ArtifactLinks)
    license: str = "unclear"
    access_restrictions: str = "unclear"
    documentation_type: str = "unclear"
    maintenance_status: str = "unclear"


class DatasetGovernance(BaseModel):
    ethics_discussed: str = "unclear"
    pii_discussed: str = "unclear"
    consent_discussed: str = "unclear"
    copyright_discussed: str = "unclear"
    bias_or_fairness_discussed: str = "unclear"
    known_limitations: list[str] = Field(default_factory=list)


class DatasetEvaluation(BaseModel):
    used_for_training: bool | None = None
    used_for_evaluation: bool | None = None
    benchmark_metrics: list[str] = Field(default_factory=list)
    baseline_models: list[str] = Field(default_factory=list)
    compared_datasets: list[str] = Field(default_factory=list)
    reported_improvement: str = "unclear"
    human_evaluation: str = "unclear"
    ablation_or_data_study: str = "unclear"


class DatasetCoverage(BaseModel):
    tasks: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    language_family_or_region: list[str] = Field(default_factory=list)
    modality: list[str] = Field(default_factory=list)
    genre: list[str] = Field(default_factory=list)
    unit_of_analysis: str = "unclear"
    input_output_format: str = "unclear"
    label_space: str = "unclear"


class DatasetAcu(BaseModel):
    id: str = ""
    text: str = ""
    type: str = "other"
    importance: str = "medium"
    evidence: str = ""
    section: str = "unknown"


class PriorDatasetMention(BaseModel):
    name: str = ""
    relationship_type: str = "loosely_related"
    cited_paper_title: str | None = None
    cited_paper_id: str | None = None
    evidence: str = ""
    prior_support_acus: list[str] = Field(default_factory=list)


class ExtractedDatasetRecord(BaseModel):
    dataset_id: str = ""
    dataset_identity: DatasetIdentity = Field(default_factory=DatasetIdentity)
    role: str = "other"
    resource_type: str = "other"
    primary_use: str = "other"
    is_reusable_resource: bool | None = None
    usage_description: str = ""
    coverage: DatasetCoverage = Field(default_factory=DatasetCoverage)
    construction: DatasetConstruction = Field(default_factory=DatasetConstruction)
    scale: DatasetScale = Field(default_factory=DatasetScale)
    availability: DatasetAvailability = Field(default_factory=DatasetAvailability)
    governance: DatasetGovernance = Field(default_factory=DatasetGovernance)
    evaluation: DatasetEvaluation = Field(default_factory=DatasetEvaluation)
    added_information_summary: str = ""
    acus: list[DatasetAcu] = Field(default_factory=list)
    prior_dataset_mentions: list[PriorDatasetMention] = Field(default_factory=list)
    confidence: str = "medium"
    ambiguities: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class ExtractionQuality(BaseModel):
    confidence: str = "medium"
    text_source: str = "unknown"
    missing_sections: list[str] = Field(default_factory=list)
    ambiguous_dataset_identity: bool = False
    possible_false_positive_dataset_paper: bool = False
    needs_human_review: bool = False
    notes: list[str] = Field(default_factory=list)


class FullTextDatasetExtraction(BaseModel):
    paper_id: str = ""
    acl_id: str = ""
    title: str = ""
    year: int | None = None
    venue_prefix: str = ""
    event: str = ""
    booktitle: str = ""
    abstract: str = ""
    anthology_url: str = ""
    pdf_url: str = ""
    source_text: SourceTextInfo = Field(default_factory=SourceTextInfo)
    paper_contribution_summary: str = ""
    institution_profile: InstitutionProfile = Field(default_factory=InstitutionProfile)
    datasets: list[ExtractedDatasetRecord] = Field(default_factory=list)
    extraction_quality: ExtractionQuality = Field(default_factory=ExtractionQuality)
    extracted_at: str = ""
    model: str = ""
    prompt_version: str = ""


STRINGISH_KEYS = {
    "type",
    "path",
    "name",
    "sector",
    "lead_author_sector",
    "paper_sector",
    "canonical_name",
    "acronym",
    "version",
    "parent_dataset_family",
    "relationship",
    "evidence",
    "source_data_origin",
    "collection_method",
    "annotation_protocol",
    "annotator_type",
    "quality_control",
    "human_verification",
    "size_text",
    "release_status",
    "license",
    "access_restrictions",
    "documentation_type",
    "maintenance_status",
    "ethics_discussed",
    "pii_discussed",
    "consent_discussed",
    "copyright_discussed",
    "bias_or_fairness_discussed",
    "reported_improvement",
    "human_evaluation",
    "ablation_or_data_study",
    "unit_of_analysis",
    "input_output_format",
    "label_space",
    "id",
    "text",
    "importance",
    "section",
    "relationship_type",
    "cited_paper_title",
    "cited_paper_id",
    "dataset_id",
    "role",
    "resource_type",
    "primary_use",
    "usage_description",
    "added_information_summary",
    "confidence",
    "paper_id",
    "acl_id",
    "title",
    "venue_prefix",
    "event",
    "booktitle",
    "abstract",
    "anthology_url",
    "pdf_url",
    "paper_contribution_summary",
    "text_source",
    "extracted_at",
    "model",
    "prompt_version",
}


def _normalize_null_text_fields(value: Any, key: str | None = None) -> Any:
    if value is None and key in STRINGISH_KEYS:
        return "unclear"
    if isinstance(value, dict):
        return {
            child_key: _normalize_null_text_fields(child_value, child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_normalize_null_text_fields(item, key) for item in value]
    return value


def parse_model_payload(payload: dict[str, Any]) -> FullTextDatasetExtraction:
    """Validate a model payload while keeping the schema permissive for pilot runs."""
    payload = _normalize_null_text_fields(payload)
    return FullTextDatasetExtraction(**payload)


def validate_extraction_for_bank(record: FullTextDatasetExtraction) -> list[str]:
    errors: list[str] = []
    if not record.paper_id:
        errors.append("missing paper_id")
    for dataset_index, dataset in enumerate(record.datasets):
        name = dataset.dataset_identity.canonical_name.strip()
        if not name:
            errors.append(f"datasets[{dataset_index}] missing canonical_name")
        for acu_index, acu in enumerate(dataset.acus):
            if not acu.text.strip():
                errors.append(f"datasets[{dataset_index}].acus[{acu_index}] missing text")
            if acu.type not in ACU_TYPES:
                errors.append(f"datasets[{dataset_index}].acus[{acu_index}] invalid type: {acu.type}")
            if acu.importance not in IMPORTANCE_LEVELS:
                errors.append(f"datasets[{dataset_index}].acus[{acu_index}] invalid importance: {acu.importance}")
            if not acu.evidence.strip():
                errors.append(f"datasets[{dataset_index}].acus[{acu_index}] missing evidence")
    return errors


def quality_warnings_for_bank(record: FullTextDatasetExtraction) -> list[str]:
    warnings: list[str] = []
    for dataset_index, dataset in enumerate(record.datasets):
        if not dataset.acus:
            warnings.append(f"datasets[{dataset_index}] has no ACUs")
        elif len(dataset.acus) < 4:
            warnings.append(f"datasets[{dataset_index}] has fewer than 4 ACUs")
        if not dataset.prior_dataset_mentions:
            warnings.append(f"datasets[{dataset_index}] has no prior dataset mentions")
    return warnings
