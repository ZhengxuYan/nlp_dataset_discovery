import pandas as pd
import requests
import os
import time
import tarfile
import shutil
import json
import glob
from tqdm import tqdm
from pypdf import PdfReader
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from bespokelabs import curator
from dotenv import load_dotenv

load_dotenv()

# Configuration
CSV_PATH = 'data/processed/arxiv_nlp_conf_papers_2023_2025.csv'
OUTPUT_FILE = 'data/processed/arxiv_nlp_conf_papers_2023_2025_dataset_analysis.jsonl'
TEMP_DIR = 'data/temp_papers'
ARXIV_SRC_URL = 'https://arxiv.org/src/{}'
ARXIV_PDF_URL = 'https://arxiv.org/pdf/{}.pdf'
DELAY_SECONDS = 3

# --- Pydantic Models for LLM ---

class AuthorInfo(BaseModel):
    name: str = Field(description="Full name of the author.")
    affiliation: str = Field(description="Affiliation of the author (University, Company, etc.). If not explicitly stated, infer from context or strictly state 'Unknown'.")

class DatasetUsage(BaseModel):
    name: str = Field(description="Exact name of the dataset as mentioned in the paper.")
    usage_description: str = Field(description="Detailed explanation of how and why the dataset is used in this paper. Must be grounded in the text. Mention if it is used for training, evaluation, or as a benchmark.")
    role: str = Field(description="Specific role: 'Main Contribution' (if introduced here), 'Training Data', 'Evaluation Benchmark', 'Fine-tuning', 'Reference/Prior Work'.")
    is_introduced: bool = Field(description="True if this paper introduces or creates this dataset; False otherwise.")
    confidence: str = Field(description="Confidence level about the extracted information: 'High', 'Medium', or 'Low'.")
    transformation_type: Optional[str] = Field(description="If this dataset is a transformation of another (e.g., 'Subsampling', 'Preprocessing', 'Upstream'), specify the type. 'None' if original.", default="None")
    transformation_type: Optional[str] = Field(description="If this dataset is a transformation of another (e.g., 'Subsampling', 'Preprocessing', 'Upstream'), specify the type. 'None' if original.", default="None")
    source_dataset: Optional[str] = Field(description="If this is a transformed dataset, the name of the source dataset. 'None' if not applicable.", default="None")
    creators: str = Field(description="Who created this dataset? E.g. 'Original Authors', 'Google', 'Stanford University', etc.")

class PaperAnalysisResponse(BaseModel):
    is_nlp_paper: bool = Field(description="True if the paper describes research related to Natural Language Processing (NLP), Computational Linguistics, or uses text data for ML tasks.", default=False)
    nlp_relevance_explanation: str = Field(description="Brief explanation of why this paper is considered NLP-related or not.", default="None")
    publication_venue: str = Field(description="The derived publication venue (e.g. 'ACL 2023', 'ArXiv', 'NeurIPS 2024') based on the provided metadata and text.", default="Unknown")
    authors: List[AuthorInfo] = Field(description="List of authors and their affiliations identified in the paper.", default=[])
    is_dataset_mentioned: bool = Field(description="True ONLY if specific named datasets are explicitly mentioned and used.", default=False)
    datasets: List[DatasetUsage] = Field(description="List of datasets found. Include ONLY datasets explicitly named in the text.", default=[])

ANALYSIS_PROMPT = """Analyze the provided text from a research paper to identify dataset usage, author information, and publication details.

Metadata:
Comment: {comment}
Journal Reference: {journal_ref}

Paper Text:
{text}

Instructions:
1. **Determine if this is an NLP paper**: Does it deal with language modeling, text processing, speech, machine translation, or other NLP topics?
2. **Extract Publication Venue**: Use the provided metadata (Comment/Journal Reference) and the text to determine the publication venue (e.g., "ACL 2023", "EMNLP 2022 Findings").
3. **Extract Authors**: Identify extracted author names and their affiliations from the text.
4. Identify all **explicitly named** datasets mentioned in the text. Do NOT infer or hallucinate dataset names.
5. For each dataset, determine:
   - **Role**: Is it the main contribution of this paper? Is it used for training a model? Is it used as a benchmark for evaluation?
   - **Usage**: Why is it used? How is it processed or utilized?
   - **Introduction**: Does this paper introduce this dataset for the first time?
   - **Creators**: Who created this dataset? (e.g. "Authors of this paper" or name of org/institution).
   - **Transformation**: Is this dataset a modified version of another? (e.g., a subset, a preprocessed version). If so, identify the **Source Dataset**.
   - **Confidence**: How confident are you in this extraction based on the text? (High/Medium/Low)

Constraints:
- Be **grounded**: Only report information explicitly stated in the text.
- If a dataset is mentioned only in passing (e.g., "unlike previous datasets..."), mark its role as 'Reference/Prior Work' or exclude it if it's not substantive.
- Do NOT invent usage details. If the text doesn't say how it's used, state "Unspecified".
- **CRITICAL**: Do NOT output null values for datasets. If a dataset is not found, do not include it in the list.

Output a JSON object matching the schema below:
{schema}
"""

# --- curator LLM ---

class PaperAnalyzer(curator.LLM):
    # response_format = PaperAnalysisResponse  # Disabled to avoid strict structured output not supported error

    def prompt(self, input: dict) -> str:
        # Truncate text to fit context window if necessary (rough estimate)
        text = input["text"][:100000] 
        return ANALYSIS_PROMPT.format(
            text=text, 
            comment=input.get("comment", ""), 
            journal_ref=input.get("journal_ref", ""),
            schema=json.dumps(PaperAnalysisResponse.model_json_schema(), indent=2)
        )
    
    def parse(self, input: dict, response: str) -> dict:
        try:
            # Basic cleanup if model wraps json in markdown
            response_text = response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Helper to clean common JSON syntax errors from LLMs
            def clean_json_string(text):
                """
                Robustly clean a JSON string by handling invalid escape sequences.
                Iterates through the string and preserves only valid JSON escapes:
                \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX.
                All other backslashes are escaped to \\\\.
                """
                out = []
                i = 0
                n = len(text)
                while i < n:
                    char = text[i]
                    if char == '\\':
                        # Check next character
                        if i + 1 < n:
                            next_char = text[i+1]
                            if next_char in '"\\/bfnrt':
                                out.append('\\' + next_char)
                                i += 2
                            elif next_char == 'u':
                                # Check if next 4 chars are hex digits
                                if i + 5 < n:
                                    hex_part = text[i+2:i+6]
                                    import string
                                    if all(c in string.hexdigits for c in hex_part):
                                        out.append('\\u' + hex_part)
                                        i += 6
                                    else:
                                        # Invalid unicode escape, escape the backslash
                                        out.append('\\\\')
                                        i += 1
                                else:
                                    out.append('\\\\')
                                    i += 1
                            else:
                                # Invalid escape char (e.g. \e, \s), escape the backslash
                                out.append('\\\\')
                                i += 1
                        else:
                            # Backslash at end of string
                            out.append('\\\\')
                            i += 1
                    elif 0 <= ord(char) < 32 and char not in '\t\n\r':
                         # Skip non-printable control chars
                         i += 1
                    else:
                        out.append(char)
                        i += 1
                return "".join(out)

            response_text = clean_json_string(response_text)

            data = json.loads(response_text)
            
            # Validate with Pydantic - allow partial filling if model misses fields? 
            # Strict mode might be too harsh for a weak instruction follower.
            # Let's try to fill valid fields and default others if validation fails on the first pass?
            try:
                validated = PaperAnalysisResponse(**data)
                return validated.model_dump()
            except Exception as validation_error:
                # If it's just missing fields, we might be able to salvage partial data
                # For now, let's just log it as a failure but maybe try to constructing it manually
                # But Pydantic error is safer to catch bad structure.
                 raise validation_error

        except Exception as e:
            # print(f"Error parsing response: {e}")
            # print(f"Response was: {response}")
            return {
                "is_nlp_paper": False,
                "nlp_relevance_explanation": f"Failed to parse model response: {e}",
                "publication_venue": "Unknown",
                "authors": [],
                "is_dataset_mentioned": False,
                "datasets": []
            }



# --- Helper Functions ---

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_tar_gz(file_path, extract_path):
    try:
        if tarfile.is_tarfile(file_path):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
            return True
        return False
    except Exception:
        return False

def get_text_from_latex(directory):
    text_content = []
    # Find all .tex files recursively
    tex_files = glob.glob(os.path.join(directory, '**/*.tex'), recursive=True)
    
    # Heuristic: Try to find main.tex or similar first, or just read all
    # Reading all is safer to catch everything
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                text_content.append(f"--- File: {os.path.basename(tex_file)} ---\n")
                text_content.append(f.read())
                text_content.append("\n")
        except Exception as e:
            print(f"Error reading {tex_file}: {e}")
            
    return "\n".join(text_content)

def get_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def process_papers(limit=None, model_name="gpt-5.4", backend=None, backend_params=None, overwrite=False):
    ensure_dir(TEMP_DIR)
    
    if overwrite and os.path.exists(OUTPUT_FILE):
        print(f"Overwriting {OUTPUT_FILE}...")
        os.remove(OUTPUT_FILE)

    # Load existing progress
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['arxiv_id'])
                except:
                    pass
    
    df = pd.read_csv(CSV_PATH)
    # Rename columns to match expected names
    df.rename(columns={'arXiv ID': 'arxiv_id'}, inplace=True)
    
    # Filter out already processed
    dataset_papers = df[~df['arxiv_id'].astype(str).isin(processed_ids)]
    
    print(f"Found {len(dataset_papers)} papers to process.")
    
    if limit:
        dataset_papers = dataset_papers.head(limit)
        print(f"Processing next {limit} papers...")
    
    analyzer = PaperAnalyzer(
        model_name=model_name,
        backend=backend,
        backend_params=backend_params,
        generation_params={"max_tokens": 16384}
    )
    
    BATCH_SIZE = 20
    
    # Process in batches
    total_papers = len(dataset_papers)
    for i in range(0, total_papers, BATCH_SIZE):
        batch_df = dataset_papers.iloc[i : i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1} / {(total_papers + BATCH_SIZE - 1) // BATCH_SIZE} (Papers {i} to {min(i + BATCH_SIZE, total_papers)})")
        
        batch_inputs = []
        batch_ids = []
        
        # 1. Download Phase
        print("  Downloading papers...")
        for index, row in tqdm(batch_df.iterrows(), total=len(batch_df), leave=False):
            arxiv_id = str(row['arxiv_id'])
            paper_dir = os.path.join(TEMP_DIR, arxiv_id)
            ensure_dir(paper_dir)
            
            text = ""
            source_type = ""
            download_success = False
            
            # Try Source
            src_url = ARXIV_SRC_URL.format(arxiv_id)
            src_save_path = os.path.join(paper_dir, 'source.tar.gz')
            
            if download_file(src_url, src_save_path):
                if extract_tar_gz(src_save_path, paper_dir):
                    extracted_text = get_text_from_latex(paper_dir)
                    if len(extracted_text) > 100:
                        text = extracted_text
                        source_type = "latex"
                        download_success = True
            
            # Fallback to PDF
            if not download_success:
                pdf_url = ARXIV_PDF_URL.format(arxiv_id)
                pdf_save_path = os.path.join(paper_dir, 'paper.pdf')
                if download_file(pdf_url, pdf_save_path):
                    extracted_text = get_text_from_pdf(pdf_save_path)
                    if len(extracted_text) > 100:
                        text = extracted_text
                        source_type = "pdf"
                        download_success = True
            
            if download_success and text:
                input_data = {
                    "arxiv_id": arxiv_id, 
                    "text": text, 
                    "source_type": source_type,
                    "comment": str(row.get('Comment', '')),
                    "journal_ref": str(row.get('Journal Reference', '')),
                    "title": str(row.get('Title', '')),
                    "published_date": str(row.get('Publication Date', ''))
                }
                batch_inputs.append(input_data)
                batch_ids.append(arxiv_id)
            else:
                print(f"  Failed to get content for {arxiv_id}")
                # Cleanup failed immediately
                try:
                    shutil.rmtree(paper_dir)
                except:
                    pass

        # 2. Analysis Phase
        if batch_inputs:
            print(f"  Analyzing {len(batch_inputs)} papers...")
            try:
                results = analyzer(batch_inputs)
                
                # 3. Save Phase
                
                # Iterate over results.dataset and inputs together
                with open(OUTPUT_FILE, 'a') as f:
                    for input_data, response in zip(batch_inputs, results.dataset):
                        # response is a Dict (from Curator/Arrow dataset) or the result of parse
                        
                        # Handle authors list (list of dicts)
                        authors_list = response.get("authors", [])
                        
                        # Handle datasets list (list of dicts)
                        datasets_list = response.get("datasets", [])
                        datasets_dict = {}
                        if datasets_list:
                            for d in datasets_list:
                                if isinstance(d, dict) and "name" in d:
                                    datasets_dict[d["name"]] = d
                        
                        record = {
                            "arxiv_id": input_data["arxiv_id"],
                            "title": input_data.get("title"),
                            "published_date": input_data.get("published_date"),
                            "source_type": input_data["source_type"],
                            "is_nlp_paper": response.get("is_nlp_paper"),
                            "nlp_relevance_explanation": response.get("nlp_relevance_explanation"),
                            "publication_venue": response.get("publication_venue"),
                            "authors": authors_list,
                            "is_dataset_mentioned": response.get("is_dataset_mentioned"),
                            "datasets": datasets_dict
                        }
                        f.write(json.dumps(record) + "\n")

                             
                print(f"  Saved results for batch.")
                
            except Exception as e:
                print(f"  Error in batch analysis: {e}")
        
        # 4. Cleanup Phase
        print("  Cleaning up batch...")
        for arxiv_id in batch_ids:
            paper_dir = os.path.join(TEMP_DIR, arxiv_id)
            try:
                shutil.rmtree(paper_dir)
            except Exception as e:
                print(f"  Error cleaning up {paper_dir}: {e}")
        
        # Delay between batches
        time.sleep(DELAY_SECONDS)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and analyze dataset papers.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of papers to process.")
    parser.add_argument("--model", type=str, default="gpt-5.4", help="Model name to use for analysis (default: gpt-5.4).")
    parser.add_argument("--backend", type=str, default=None, help="Backend to use (e.g., litellm, openai). Defaults to None (or env BACKEND).")
    parser.add_argument("--backend-params", type=str, default=None, help="JSON string for backend parameters.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file and start from scratch (process the first N papers again).")
    args = parser.parse_args()

    # Handle backend configuration
    backend = args.backend if args.backend else os.environ.get("BACKEND")
    
    backend_params = None
    if args.backend_params:
        try:
            backend_params = json.loads(args.backend_params)
        except json.JSONDecodeError:
            print("Error parsing --backend-params. Must be valid JSON string.")
            exit(1)
    elif os.environ.get("BACKEND_PARAMS"):
         try:
            backend_params = json.loads(os.environ.get("BACKEND_PARAMS"))
         except json.JSONDecodeError:
             print("Error parsing BACKEND_PARAMS env var. Must be valid JSON string.")
    
    # Fallback default params if not set
    if backend_params is None and os.environ.get("BACKEND_PARAMS") is None:
         # Default params similar to example if not specified
         backend_params = {
            "max_requests_per_minute": 2_000,
            "max_tokens_per_minute": 4_000_000,
         }

    import litellm
    # Register custom model to avoid "model not mapped" error
    # Pricing: Input: $0.200/1M, Output: $0.600/1M (from user logs, note 0.600 not 6.000 as previously tried)
    # Register both original and lowercase versions as LiteLLM might normalize keys
    model_key = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
    cost_dict = {
        "input_cost_per_token": 0.200 / 1_000_000,
        "output_cost_per_token": 0.600 / 1_000_000,
        "max_tokens": 32768, 
        "litellm_provider": "together_ai",
        "mode": "chat"
    }
    litellm.model_cost[model_key] = cost_dict
    litellm.model_cost[model_key.lower()] = cost_dict
    litellm.model_cost["qwen/qwen3-235b-a22b-instruct-2507-tput"] = cost_dict # Case sometimes stripped of provider prefix in internal checks?

    process_papers(limit=args.limit, model_name=args.model, backend=backend, backend_params=backend_params, overwrite=args.overwrite)
