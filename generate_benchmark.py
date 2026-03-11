import os
import json
import argparse
from typing import List, Dict
from pydantic import BaseModel, Field
from bespokelabs import curator
from dotenv import load_dotenv

class Dataset(BaseModel):
    name: str = Field(description="Name of the dataset")
    acus: List[str] = Field(description="List of 3-5 short Atomic Content Units describing the dataset")

class ScoredDataset(Dataset):
    ground_truth_novelty: float = Field(description="A ground truth novelty score from 0.0 to 1.0 (relative to the true_ancestor).")

class LineageCluster(BaseModel):
    domain: str = Field(description="The NLP domain or task for this cluster, e.g., 'Biomedical Named Entity Recognition'")
    true_ancestor: Dataset = Field(description="The foundational semantic ancestor dataset.")
    
    # Phase 1: Retrieval Eval 
    query_dataset: Dataset = Field(description="The descendant dataset we will use as the retrieval query.")
    hard_negative: Dataset = Field(description="A distractor dataset that has high lexical/word overlap with the query but is semantically unrelated (different domain/task).")
    soft_negative: Dataset = Field(description="A dataset in a similar domain but fundamentally a different lineage or task.")
    
    # Phase 2: Novelty Scoring Eval (Relative to true_ancestor)
    incremental_descendant: ScoredDataset = Field(description="Scenario A: An incremental descendant holding a minor new contribution. Expected novelty: 0.3 - 0.5")
    breakthrough_dataset: ScoredDataset = Field(description="Scenario B: A completely new paradigm or massively scaled dataset in the same domain. Expected novelty: 0.8 - 1.0")
    reproduction_dataset: ScoredDataset = Field(description="Scenario C: A pure reproduction/translation that offers zero new methods/tasks. Expected novelty: 0.0 - 0.1")


class BenchmarkGeneration(BaseModel):
    clusters: List[LineageCluster] = Field(description="A list of generated dataset lineage clusters")

class BenchmarkGenerator(curator.LLM):
    response_format = BenchmarkGeneration
    
    def prompt(self, input: dict) -> str:
        domain = input.get("domain", "General NLP")
        title = input.get("title", "")
        abstract = input.get("abstract", "")
        introduced_dataset = input.get("introduced_dataset", "")
        
        return f"""You are an expert NLP researcher. Your task is to generate 1 highly realistic synthetic benchmark 'Dataset Lineage Cluster'.

You are grounding this cluster around a REAL peer-reviewed paper that introduced a dataset:
--- 
NLP Domain/Topic: {domain}
Paper Title: {title}
Paper Abstract: {abstract}
Introduced Dataset: {introduced_dataset}
---

For this cluster, invent highly realistic datasets with these constraints (use 3-5 ACUs per dataset):
1. `true_ancestor`: Must represent the actual '{introduced_dataset}' from the paper above. Use the provided abstract to form its ACUs accurately.

--- RETRIEVAL EVALUATION COMPONENT ---
2. `query_dataset`: A new descendant dataset (builds logically on the ancestor).
3. `hard_negative`: COMPLETELY DIFFERENT TASK/DOMAIN but engineered to share EXACT KEYWORDS with the `query_dataset`'s ACUs (trap for BM25).
4. `soft_negative`: Exact same domain/task as query, but distinctly separate effort (not an ancestor).

--- NOVELTY SCORING EVALUATION COMPONENT ---
You must generate three relative datasets compared to the `true_ancestor` and assign them a `ground_truth_novelty` score (0.0=Identical, 1.0=Groundbreaking):
5. `incremental_descendant`: Scenario A. A minor iteration (e.g., +10% more data or adding one language). Ground Truth: ~0.3 - 0.5
6. `breakthrough_dataset`: Scenario B. A massive leap in scale or an entirely novel task paradigm. Ground Truth: ~0.8 - 1.0
7. `reproduction_dataset`: Scenario C. An exact re-annotation or direct translation without new methods. Ground Truth: ~0.0 - 0.1

ACU formatting: Each ACU must be a single, short, fully independent sentence making ONE claim. Do not use real datasets (invent names like 'ClinicalNER-2024').
"""

def generate_benchmark(output_file: str, input_csv: str, input_jsonl: str, num_clusters: int):
    import pandas as pd
    import random
    
    print("Loading data for grounding...")
    df_csv = pd.read_csv(input_csv)
    # create mapping from arxiv id to abstract and title
    id_to_abstract = dict(zip(df_csv['arXiv ID'].astype(str), df_csv['Abstract']))
    id_to_title = dict(zip(df_csv['arXiv ID'].astype(str), df_csv['Title']))
    
    introduced_datasets = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line)
            arxiv_id = str(item.get("arxiv_id", ""))
            domain = item.get("nlp_domain", "General NLP")
            if "datasets" in item:
                for ds_name, ds_info in item["datasets"].items():
                    if ds_info.get("is_introduced", False):
                        # Construct context
                        paper_title = item.get("title", id_to_title.get(arxiv_id, "Unknown Title"))
                        paper_abstract = item.get("abstract", id_to_abstract.get(arxiv_id, ""))
                        if not paper_abstract:
                            print(f"Warning: Abstract missing for {arxiv_id}, it may impact generation.")
                            
                        introduced_datasets.append({
                            "domain": domain,
                            "title": paper_title,
                            "abstract": paper_abstract,
                            "introduced_dataset": ds_name
                        })
    
    print(f"Found {len(introduced_datasets)} unique introduced dataset anchors.")
    
    generator = BenchmarkGenerator(model_name="gpt-5-mini")
    
    # Sample the required number of clusters
    if num_clusters > len(introduced_datasets):
        print(f"Warning: Requested {num_clusters} clusters but only found {len(introduced_datasets)} anchors. Sampling with replacement.")
        sampled_anchors = random.choices(introduced_datasets, k=num_clusters)
    else:
        sampled_anchors = random.sample(introduced_datasets, num_clusters)
        
    print(f"Generating benchmark with {num_clusters} clusters...")
    
    requests = sampled_anchors
    
    batch_size = 20
    print(f"Dividing {len(requests)} requests into batches of {batch_size} to prevent 503 Upstream API timeouts...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Clear output file first
    with open(output_file, 'w') as f:
        pass
        
    total_generated = 0
    
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(requests) + batch_size - 1)//batch_size}...")
        results = generator(batch)
        
        batch_clusters = []
        dataset_results = getattr(results, "dataset", results)
        for res in dataset_results:
            # Check if the result uses parsed_response_message or directly maps
            if hasattr(res, "clusters"):
                batch_clusters.extend(res.clusters)
            elif isinstance(res, dict) and "clusters" in res:
                batch_clusters.extend(res["clusters"])
            elif isinstance(res, dict) and "parsed_response_message" in res:
                msg = res["parsed_response_message"]
                if hasattr(msg, "clusters"):
                    batch_clusters.extend(msg.clusters)
                elif isinstance(msg, dict) and "clusters" in msg:
                     batch_clusters.extend(msg["clusters"])
                     
        # Flatten to our expected JSONL format
        flat_data = []
        for c in batch_clusters:
            # Convert Pydantic object to dict
            if hasattr(c, "model_dump"):
                flat_data.append(c.model_dump())
            elif hasattr(c, "dict"):
                flat_data.append(c.dict())
            else:
                 flat_data.append(c) # if already dict
                 
        with open(output_file, 'a') as f:
            for item in flat_data:
                f.write(json.dumps(item) + "\n")
                
        total_generated += len(flat_data)
        print(f"  -> Batch complete. Wrote {len(flat_data)} clusters. Total so far: {total_generated}")
            
    print(f"Successfully generated {total_generated} clusters and saved to {output_file}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/benchmark/retrieval_benchmark.jsonl")
    parser.add_argument("--input_csv", type=str, default="data/processed/arxiv_nlp_conf_papers_2023_2025.csv")
    parser.add_argument("--input_jsonl", type=str, default="data/processed/arxiv_nlp_conf_papers_2023_2025_dataset_analysis(gpt-5-mini).jsonl")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters to generate")
    args = parser.parse_args()
    
    generate_benchmark(args.output, args.input_csv, args.input_jsonl, args.num_clusters)
