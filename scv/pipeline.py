import os
import json
import pandas as pd
from tqdm import tqdm
from .models import ScvPaperAnalysis, DatasetScvInfo
from .extraction import ScvExtractor
from .utils import ensure_dir, download_file, get_text_from_pdf
from .analysis import (
    compute_embeddings, 
    calculate_diversity_score, 
    calculate_quality_score, 
    analyze_novelty_and_get_score, 
    construct_scv, 
    NoveltyAnalyzer
)

TEMP_DIR = 'data/temp_scv'
ARXIV_PDF_URL = 'https://arxiv.org/pdf/{}.pdf'

# --- Helper for Sampling ---
def sample_papers_uniformly(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Sample papers uniformly across months from 2023-2025."""
    try:
        if 'Publication Date' not in df.columns:
            return df.head(limit)
            
        # Parse dates
        df['dt'] = pd.to_datetime(df['Publication Date'], errors='coerce')
        df = df.dropna(subset=['dt'])
        df['month_year'] = df['dt'].dt.to_period('M')
        
        # Group by month and sample
        groups = [g for _, g in df.groupby('month_year')]
        if not groups:
            return df.head(limit)
            
        # Distribute limit among groups
        per_group = max(1, limit // len(groups))
        
        sampled_frames = []
        for g in groups:
            # If group smaller than per_group, take all. Else sample.
            n = min(len(g), per_group)
            sampled_frames.append(g.sample(n=n, random_state=42))
            
        sampled = pd.concat(sampled_frames)
        
        # If we need more to fill limit, take random from remaining
        if len(sampled) < limit:
            remaining = df.drop(sampled.index)
            n_needed = limit - len(sampled)
            if n_needed > 0 and not remaining.empty:
                extra = remaining.sample(n=min(len(remaining), n_needed), random_state=42)
                sampled = pd.concat([sampled, extra])
        
        return sampled.head(limit)
    except Exception as e:
        print(f"Sampling failed: {e}. Falling back to head().")
        return df.head(limit)

# --- Main Pipeline Stages ---

def run_extraction_stage(limit: int = 200, output_file: str = 'data/processed/scv_intermediate.jsonl'):
    """Stage 1: Extract info and embeddings."""
    ensure_dir(TEMP_DIR)
    
    # Load input data
    CSV_PATH = 'data/processed/arxiv_nlp_conf_papers_2023_2025.csv'
    if not os.path.exists(CSV_PATH):
        print(f"Data file {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # SAMPLING: "sample papers from different months across 2023 to 2025"
    papers_to_process = sample_papers_uniformly(df, limit)
    print(f"[Extraction] Processing {len(papers_to_process)} papers (sampled)...")
    
    extractor = ScvExtractor(model_name="gpt-5-mini") 
    
    batch_size = 5
    processed_count = 0
    
    for i in range(0, len(papers_to_process), batch_size):
        batch = papers_to_process.iloc[i:i+batch_size]
        batch_inputs = []
        
        # 1. Download & Prepare Text
        for _, row in batch.iterrows():
            arxiv_id = str(row['arXiv ID']) 
            paper_dir = os.path.join(TEMP_DIR, arxiv_id)
            ensure_dir(paper_dir)
            
            text = ""
            pdf_url = ARXIV_PDF_URL.format(arxiv_id)
            pdf_path = os.path.join(paper_dir, 'paper.pdf')
            
            if not os.path.exists(pdf_path):
                 download_file(pdf_url, pdf_path)
            
            if os.path.exists(pdf_path):
                text = get_text_from_pdf(pdf_path)
            
            if text:
                # Extract extra CSV metadata to pass through
                # 'Journal Reference' or 'Comment' often has venue info.
                venue = str(row.get('Journal Reference', ''))
                if not venue or venue == 'nan':
                     venue = str(row.get('Comment', 'ArXiv'))
                
                batch_inputs.append({
                    "arxiv_id": arxiv_id,
                    "title": str(row.get('Title', '')),
                    "abstract": str(row.get('Abstract', '')),
                    "text": text,
                    "date": str(row.get('Publication Date', '')),
                    "venue": venue,
                    "source_type": "ArXiv", # Default for this dataset
                    # Extra Metadata per user request
                    "categories": str(row.get('Categories', '')),
                    "primary_category": str(row.get('Primary Category', '')),
                    "doi": str(row.get('DOI', ''))
                })
        
        if not batch_inputs:
            continue
            
        # 2. Extract Info
        try:
            llm_outputs = extractor(batch_inputs)
            
            # 3. Process & Save Intermediate
            for inp, analysis in zip(batch_inputs, llm_outputs.dataset):
                
                # Compute Paper Embedding
                paper_embedding = compute_embeddings(inp['abstract'])
                
                # Check formatting/conversion if needed
                # Hydrate Pydantic model if analysis is a dict
                if isinstance(analysis, dict):
                    try:
                        analysis_obj = ScvPaperAnalysis(**analysis)
                    except Exception as e:
                        print(f"Error parsing analysis for {inp['arxiv_id']}: {e}")
                        continue
                else:
                    analysis_obj = analysis

                api_datasets = [d.dict() for d in analysis_obj.datasets]
                
                # FILTER Logic Update: 
                # User request: "if a paper that doesn't introduce a dataset. we should still save the same extraction result."
                # So we SAVE ALL, but mark them.
                
                # Check if ANY dataset is introduced
                has_introduced = any(d.get('is_introduced', False) for d in api_datasets)
                
                record = {
                    "arxiv_id": inp['arxiv_id'],
                    # Merge CSV metadata (inp) with LLM extracted data (analysis_obj)
                    "title": inp['title'], # From CSV
                    "abstract": inp['abstract'],
                    "published_date": inp['date'], # From CSV
                    "source_type": inp.get('source_type', 'Unknown'), # From CSV 
                    "publication_venue": inp.get('venue', 'Unknown'), # From CSV
                    
                    # Extra Metadata
                    "categories": inp.get('categories', ''),
                    "primary_category": inp.get('primary_category', ''),
                    "doi": inp.get('doi', ''),
                    
                    # LLM Fields
                    "is_nlp_paper": analysis_obj.is_nlp_paper,
                    "nlp_relevance_explanation": analysis_obj.nlp_relevance_explanation,
                    "contribution_summary": analysis_obj.paper_contribution_summary,
                    "authors": [a.dict() for a in analysis_obj.authors], # Extracted Authors
                    
                    "datasets": api_datasets, # Save ALL datasets extracted
                    "paper_embedding": paper_embedding
                }
                
                with open(output_file, 'a') as f:
                    f.write(json.dumps(record) + "\n")
                
                processed_count += 1
                    
        except Exception as e:
            print(f"Batch failed: {e}")
            
    print(f"[Extraction] Done. Saved {processed_count} records to {output_file}.")


def run_analysis_stage(input_file: str = 'data/processed/scv_intermediate.jsonl', output_file: str = 'data/processed/scv_final_results.jsonl'):
    """Stage 2: Sort, Analyze Novelty against history, Compute SCV."""
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return
        
    print(f"[Analysis] Loading extracted data from {input_file}...")
    records = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                 records.append(json.loads(line))
            except:
                pass
                
    # Sort by date
    def parse_date(d_str):
        try:
            return pd.to_datetime(d_str)
        except:
            return pd.to_datetime('2023-01-01') # Default fallback
            
    records.sort(key=lambda x: parse_date(x.get('published_date', '')))
    print(f"[Analysis] Sorted {len(records)} records by date.")
    
    # Init Analyzer
    analyzer = NoveltyAnalyzer()
    
    results = []
    
    for rec in tqdm(records, desc="Analyzing Novelty"):
        datasets = rec.get('datasets', [])
        
        # Hydrate
        dataset_objs = []
        has_intro = False
        for d in datasets:
            try:
                ds = DatasetScvInfo(**d)
                dataset_objs.append(ds)
                if ds.is_introduced:
                    has_intro = True
            except:
                pass
        
        # FILTER: "don't need to run analysis for those [non-introduced]"
        if not has_intro:
             # Skip analysis for this paper, do not add to final record?
             # Or add as empty?
             # User implies "don't run", so we probably shouldn't include it in final SCV results 
             # OR we include it with null SCV. 
             # Let's skip it to keep the final file clean with only "contributions".
             continue
        
        processed_datasets = []
        for ds in dataset_objs:
            # Only analyze introduced datasets for SCV
            if not ds.is_introduced:
                continue
            
            # Metrics
            # Quality needs to use the NEW robust fields
            qual_score = calculate_quality_score(ds)
            div_score = calculate_diversity_score(ds)
            nov_score = analyze_novelty_and_get_score(ds, analyzer)
            
            scv = construct_scv(nov_score, div_score, qual_score)
            
            # Update History
            if ds.acus:
                analyzer.add_acus(ds.acus)
            
            processed_datasets.append({
                "info": ds.dict(),
                "scv": scv
            })
            
        final_record = {
            "arxiv_id": rec['arxiv_id'],
            "is_nlp": rec.get('is_nlp_paper', True), 
            "paper_embedding": rec.get('paper_embedding', []),
            "datasets": processed_datasets,
            "metadata": {
                "title": rec.get('title'),
                "date": rec.get('published_date'),
                 "source": rec.get('source_type'),
                 "venue": rec.get('publication_venue'),
                 "authors": rec.get('authors')
            }
        }
        
        results.append(final_record)
        
    # Save Final
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"[Analysis] Done. Saved {len(results)} analyzed records to {output_file}.")
