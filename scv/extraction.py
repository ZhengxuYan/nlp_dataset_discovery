from bespokelabs import curator
from .models import ScvPaperAnalysis


def condense_text(text: str, max_chars: int) -> str:
    """Keep prompt size bounded while preserving start and end context."""
    if not text or len(text) <= max_chars:
        return text

    head_chars = int(max_chars * 0.7)
    tail_chars = max_chars - head_chars
    return f"{text[:head_chars]}\n\n[... truncated ...]\n\n{text[-tail_chars:]}"

ANALYSIS_PROMPT = """Analyze the research paper text to extract structured dataset information for a Scientific Contribution Vector (SCV).

Metadata from Paper Header (if available):
Title: {title}
Abstract: {abstract}

Paper Text (Excerpt):
{text}

Instructions:
1. **Paper Analysis**:
    - **Authors**: Extract full names AND affiliations.
    - **Relevance**: Is this an NLP paper? + Explanation.
    - **Contribution**: Brief summary of the main contribution.
    
    (Note: Title and Date are already known, do not extract them).

2. **Datasets**: Identify all **explicitly named** datasets.
3. For each dataset, extract:
    - Basic info: Name, Role, Introduction status, Creators.
    - Transformation: If it's a version of another dataset.
    - **Extended Metadata**: 
        - Tasks, Languages, Domain.
        - **Size (Structured)**: 
            - `size_str` (original text e.g. "10k sentences").
            - `num_samples` (ESTIMATED INTEGER e.g. 10000. Use -1 if unknown).
            - `num_classes` (INTEGER e.g. 2, 10, 1000. Use -1 if unknown).
            - `storage_size` (e.g. "5GB").
    - **Quality & Availability**: 
        - License (e.g. MIT, CC-BY).
        - **Availability URL**: Look for GitHub, HuggingFace, or project links.
        - **Documentation**: Datasheets, Readmes mentioned?
        - Maintenance: Any maintenance plan mentioned?
    - **Added Information (Crucial)**: If the dataset is NEW (introduced here):
        - `added_information_summary`: A concise statement of what new information this dataset adds relative to prior work.
        - `novelty_summary`: Mirror of `added_information_summary` for backward compatibility.
        - `acus`: List of Atomic Content Units. Break information into extremely granular, short, independent sentences. Each ACU must contain only ONE single, simple fact. Avoid compound sentences and conjunctions (e.g., break "Data and code are released on GitHub" into "Data is released on GitHub." and "Code is released on GitHub.").
        - `previous_work_acus`: Extract 3-5 ACUs of relevant PREVIOUS WORK mentioned in the text (e.g. "Previous Dataset Y only covered English.", "Dataset Z used heuristic filtering."). This helps in comparing the new contribution.
    - **Transparency Issues**: Note if links/license/provenance are missing.

Constraints:
- Be grounded in the text.
- If a field is not found, use "Unknown" or "None".
- Do NOT hallucinate dataset names.

Output JSON matching the schema.
IMPORTANT: 
- You MUST populate `acus` for any introduced dataset.
- ACUs should be extremely short, concise, and contain only ONE factual claim per sentence. Break down any complex or compound sentences into multiple simpler ACUs.
- Populate `previous_work_acus` if the paper discusses prior art constraints or features.
- Populate the `size` object fields carefully.
"""

class ScvExtractor(curator.LLM):
    response_format = ScvPaperAnalysis

    def __init__(self, *args, text_char_limit: int = 16000, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_char_limit = text_char_limit

    def prompt(self, input: dict) -> str:
        # Using title/abstract + first chunk of text
        return ANALYSIS_PROMPT.format(
            title=input.get("title", ""),
            abstract=input.get("abstract", ""),
            text=condense_text(input.get("text", ""), self.text_char_limit)
        )
