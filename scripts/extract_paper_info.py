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
CSV_PATH = 'data/processed/arxiv_results_metadata_curator_20251114_014109.csv'
OUTPUT_FILE = 'data/processed/paper_dataset_analysis.jsonl'
TEMP_DIR = 'data/temp_papers'
ARXIV_SRC_URL = 'https://arxiv.org/src/{}'
ARXIV_PDF_URL = 'https://arxiv.org/pdf/{}.pdf'
DELAY_SECONDS = 3

# --- Pydantic Models for LLM ---

class DatasetUsage(BaseModel):
    name: str = Field(description="Exact name of the dataset as mentioned in the paper.")
    usage_description: str = Field(description="Detailed explanation of how and why the dataset is used in this paper. Must be grounded in the text. Mention if it is used for training, evaluation, or as a benchmark.")
    role: str = Field(description="Specific role: 'Main Contribution' (if introduced here), 'Training Data', 'Evaluation Benchmark', 'Fine-tuning', 'Reference/Prior Work'.")
    is_introduced: bool = Field(description="True if this paper introduces or creates this dataset; False otherwise.")

class PaperAnalysisResponse(BaseModel):
    is_dataset_mentioned: bool = Field(description="True ONLY if specific named datasets are explicitly mentioned and used.")
    datasets: Dict[str, DatasetUsage] = Field(description="Dictionary of datasets found. Key is the canonical name. Include ONLY datasets explicitly named in the text.", default={})

# --- Prompts ---

ANALYSIS_PROMPT = """Analyze the provided text from a research paper to identify and characterize the datasets used.

Paper Text (truncated):
{text}

Instructions:
1. Identify all **explicitly named** datasets mentioned in the text. Do NOT infer or hallucinate dataset names.
2. For each dataset, determine:
   - **Role**: Is it the main contribution of this paper? Is it used for training a model? Is it used as a benchmark for evaluation?
   - **Usage**: Why is it used? How is it processed or utilized?
   - **Introduction**: Does this paper introduce this dataset for the first time?

Constraints:
- Be **grounded**: Only report information explicitly stated in the text.
- If a dataset is mentioned only in passing (e.g., "unlike previous datasets..."), mark its role as 'Reference/Prior Work' or exclude it if it's not substantive.
- Do NOT invent usage details. If the text doesn't say how it's used, state "Unspecified".

Output a JSON object matching the schema.
"""

# --- Curator LLM ---

class PaperAnalyzer(curator.LLM):
    response_format = PaperAnalysisResponse

    def prompt(self, input: dict) -> str:
        # Truncate text to fit context window if necessary (rough estimate)
        text = input["text"][:100000] 
        return ANALYSIS_PROMPT.format(text=text)

    def parse(self, input: dict, response: PaperAnalysisResponse) -> List[Dict]:
        return [
            {
                "arxiv_id": input["arxiv_id"],
                "source_type": input["source_type"],
                "is_dataset_mentioned": response.is_dataset_mentioned,
                "datasets": {name: data.dict() for name, data in response.datasets.items()}
            }
        ]

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

def process_papers(limit=None):
    ensure_dir(TEMP_DIR)
    
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
    dataset_papers = df[df['is_dataset_paper'] == 'Yes']
    
    # Filter out already processed
    dataset_papers = dataset_papers[~dataset_papers['arxiv_id'].astype(str).isin(processed_ids)]
    
    print(f"Found {len(dataset_papers)} papers to process.")
    
    if limit:
        dataset_papers = dataset_papers.head(limit)
        print(f"Processing next {limit} papers...")
    
    analyzer = PaperAnalyzer(model_name="gpt-4o-mini")
    
    for index, row in tqdm(dataset_papers.iterrows(), total=len(dataset_papers)):
        arxiv_id = str(row['arxiv_id'])
        paper_dir = os.path.join(TEMP_DIR, arxiv_id)
        ensure_dir(paper_dir)
        
        text = ""
        source_type = ""
        
        # 1. Try Source
        src_url = ARXIV_SRC_URL.format(arxiv_id)
        src_save_path = os.path.join(paper_dir, 'source.tar.gz')
        
        download_success = False
        if download_file(src_url, src_save_path):
            if extract_tar_gz(src_save_path, paper_dir):
                extracted_text = get_text_from_latex(paper_dir)
                if len(extracted_text) > 100: # Arbitrary threshold
                    text = extracted_text
                    source_type = "latex"
                    download_success = True
        
        # 2. Fallback to PDF
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
            # Run Analysis
            try:
                result = analyzer([{"arxiv_id": arxiv_id, "text": text, "source_type": source_type}])
                
                # Save result
                with open(OUTPUT_FILE, 'a') as f:
                    f.write(json.dumps(result.dataset.to_pandas().to_dict('records')[0]) + "\n")
                    
            except Exception as e:
                print(f"Error analyzing {arxiv_id}: {e}")
        else:
            print(f"Failed to get content for {arxiv_id}")
            # Log failure?
        
        # Cleanup
        try:
            shutil.rmtree(paper_dir)
        except Exception as e:
            print(f"Error cleaning up {paper_dir}: {e}")
            
        time.sleep(DELAY_SECONDS)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and analyze dataset papers.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of papers to process.")
    args = parser.parse_args()

    process_papers(limit=args.limit)
