import sys
import os
import json
import numpy as np
from typing import List, Dict


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scv.analysis import NoveltyAnalyzer, SENTENCE_MODEL, load_models

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

def calculate_metrics(ranks: List[int]) -> Dict[str, float]:
    """Calculate standard IR metrics given a list of ranks of the True Positive (1-indexed)."""
    if not ranks:
        return {"MRR": 0.0, "Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0}
        
    mrr = np.mean([1.0 / r for r in ranks])
    r1 = np.mean([1 if r <= 1 else 0 for r in ranks])
    r3 = np.mean([1 if r <= 3 else 0 for r in ranks])
    r5 = np.mean([1 if r <= 5 else 0 for r in ranks])
    
    return {
        "MRR": float(mrr),
        "Recall@1": float(r1),
        "Recall@3": float(r3),
        "Recall@5": float(r5)
    }

def run_benchmark(input_file: str):
    if not os.path.exists(input_file):
        print(f"Error: Benchmark file {input_file} not found. Run generate_benchmark.py first.")
        return
        
    print(f"Loading benchmark data from {input_file}...")
    clusters = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))
                
    print(f"Loaded {len(clusters)} lineage clusters.")
    
    load_models()
    analyzer = NoveltyAnalyzer()
    
    # 1. Build the global history corpus
    history_acus = []
    history_sources = []
    
    for i, cluster in enumerate(clusters):
        cid = f"C{i}"
        
        # Ancestor
        if 'true_ancestor' in cluster and 'acus' in cluster['true_ancestor']:
            for acu in cluster['true_ancestor']['acus']:
                history_acus.append(acu)
                history_sources.append(f"{cid}_ANCESTOR_{cluster['true_ancestor']['name']}")
            
        # Hard Negative
        if 'hard_negative' in cluster and 'acus' in cluster['hard_negative']:
            for acu in cluster['hard_negative']['acus']:
                history_acus.append(acu)
                history_sources.append(f"{cid}_HARDNEG_{cluster['hard_negative']['name']}")
            
        # Soft Negative
        if 'soft_negative' in cluster and 'acus' in cluster['soft_negative']:
            for acu in cluster['soft_negative']['acus']:
                history_acus.append(acu)
                history_sources.append(f"{cid}_SOFTNEG_{cluster['soft_negative']['name']}")

    print(f"Global History Corpus Size: {len(history_acus)} ACUs")
    analyzer.add_acus(history_acus)
    
    # Initialize BM25 Model on the exact same corpus
    bm25 = None
    if HAS_BM25:
        tokenized_corpus = [doc.lower().split(" ") for doc in history_acus]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        print("Warning: rank_bm25 not installed. BM25 baseline will be skipped.")
        
    # 2. Evaluate Queries
    from scv.analysis import SENTENCE_MODEL
    from sentence_transformers import util
    
    scv_ranks = []
    bm25_ranks = []
    
    # Track how often the model fell for the Hard Negative trap
    scv_hn_trap_count = 0 
    bm25_hn_trap_count = 0
    total_queries = len(clusters)
    
    print("\nEvaluating Queries...")
    
    for i, cluster in enumerate(clusters):
        cid = f"C{i}"
        target_ancestor_id = f"{cid}_ANCESTOR_{cluster['true_ancestor']['name']}"
        target_hardneg_id = f"{cid}_HARDNEG_{cluster['hard_negative']['name']}"
        
        query_acus = cluster['query_dataset']['acus']
        
        # ---------- SCV Pipeline Evaluation ----------
        desc_embs = SENTENCE_MODEL.encode(query_acus, convert_to_numpy=True)
        # We retrieve top-K per ACU and aggregate
        hits = util.semantic_search(desc_embs, analyzer.history_embeddings, top_k=20)
        
        scv_scores = {}
        for query_hits in hits:
            for hit in query_hits:
                source = history_sources[hit['corpus_id']]
                if source not in scv_scores:
                    scv_scores[source] = 0.0
                scv_scores[source] += hit['score']
                
        sorted_scv = sorted(scv_scores.items(), key=lambda x: x[1], reverse=True)
        
        scv_tp_rank = float('inf')
        scv_hn_rank = float('inf')
        for rank, (source, score) in enumerate(sorted_scv):
            if source == target_ancestor_id and scv_tp_rank == float('inf'):
                scv_tp_rank = rank + 1
            if source == target_hardneg_id and scv_hn_rank == float('inf'):
                scv_hn_rank = rank + 1
                
        scv_ranks.append(scv_tp_rank if scv_tp_rank != float('inf') else total_queries * 3) # Penalty rank
        if scv_hn_rank < scv_tp_rank:
            scv_hn_trap_count += 1
            
        # ---------- BM25 Pipeline Evaluation ----------
        if bm25:
            bm25_scores = {}
            for query_acu in query_acus:
                tokenized_query = query_acu.lower().split(" ")
                doc_scores = bm25.get_scores(tokenized_query)
                for idx, score in enumerate(doc_scores):
                    if score > 0: # Only care about non-zero hits
                        source = history_sources[idx]
                        if source not in bm25_scores:
                            bm25_scores[source] = 0.0
                        bm25_scores[source] += score
                        
            sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            
            bm25_tp_rank = float('inf')
            bm25_hn_rank = float('inf')
            for rank, (source, score) in enumerate(sorted_bm25):
                if source == target_ancestor_id and bm25_tp_rank == float('inf'):
                    bm25_tp_rank = rank + 1
                if source == target_hardneg_id and bm25_hn_rank == float('inf'):
                    bm25_hn_rank = rank + 1
                    
            bm25_ranks.append(bm25_tp_rank if bm25_tp_rank != float('inf') else total_queries * 3)
            if bm25_hn_rank < bm25_tp_rank:
                bm25_hn_trap_count += 1

    # 3. Print Results
    scv_metrics = calculate_metrics(scv_ranks)
    print("\n==================================")
    print("        SCV PIPELINE METRICS       ")
    print("==================================")
    print(f"MRR:       {scv_metrics['MRR']:.4f}")
    print(f"Recall@1:  {scv_metrics['Recall@1']:.4f}")
    print(f"Recall@3:  {scv_metrics['Recall@3']:.4f}")
    print(f"Recall@5:  {scv_metrics['Recall@5']:.4f}")
    print(f"Hard Negative Error Rate: {scv_hn_trap_count/total_queries:.4f} ({scv_hn_trap_count}/{total_queries})")
    
    if bm25:
        bm25_metrics = calculate_metrics(bm25_ranks)
        print("\n==================================")
        print("         BM25 PIPELINE METRICS     ")
        print("==================================")
        print(f"MRR:       {bm25_metrics['MRR']:.4f}")
        print(f"Recall@1:  {bm25_metrics['Recall@1']:.4f}")
        print(f"Recall@3:  {bm25_metrics['Recall@3']:.4f}")
        print(f"Recall@5:  {bm25_metrics['Recall@5']:.4f}")
        print(f"Hard Negative Error Rate: {bm25_hn_trap_count/total_queries:.4f} ({bm25_hn_trap_count}/{total_queries})")

def evaluate_novelty_scoring(input_file: str):
    print(f"\n==================================")
    print("      PHASE 2: NOVELTY SCORING      ")
    print("==================================")
    
    if not os.path.exists(input_file):
        print(f"Error: Benchmark file {input_file} not found.")
        return
        
    clusters = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))
                
    load_models()
    
    analyzer_nli = NoveltyAnalyzer()
    analyzer_llm = NoveltyAnalyzer()
    
    # We must patch NoveltyAnalyzer to NOT skip LLM for this specific test
    # (Since we mocked it out earlier in analysis.py for Exp A)
    from scv.analysis import LlmNoveltyEvaluator
    if "OPENAI_API_KEY" in os.environ:
        analyzer_llm.llm_evaluator = LlmNoveltyEvaluator(model_name="gpt-5-mini")
    else:
        print("Warning: OPENAI_API_KEY not found. LLM Novelty Scoring will be skipped.")
    
    nli_errors = []
    llm_errors = []
    
    for i, cluster in enumerate(clusters):
        # The history for novelty scoring is just the True Ancestor
        if 'true_ancestor' not in cluster:
            continue
            
        ancestor_acus = cluster['true_ancestor']['acus']
        
        # Test 3 Scenarios per cluster
        scenarios = []
        if 'incremental_descendant' in cluster:
            scenarios.append(("Incremental", cluster['incremental_descendant']))
        if 'breakthrough_dataset' in cluster:
            scenarios.append(("Breakthrough", cluster['breakthrough_dataset']))
        if 'reproduction_dataset' in cluster:
            scenarios.append(("Reproduction", cluster['reproduction_dataset']))
            
        for scenario_name, dataset in scenarios:
            new_acus = dataset['acus']
            ground_truth = dataset.get('ground_truth_novelty', 0.5)
            
            # Evaluate NLI
            from scv.analysis import SENTENCE_MODEL
            analyzer_nli.history_acus = ancestor_acus
            analyzer_nli.history_embeddings = SENTENCE_MODEL.encode(ancestor_acus, convert_to_numpy=True)
            nli_score = analyzer_nli.calculate_novelty_score_nli(new_acus)
            nli_errors.append(abs(nli_score - ground_truth))
            
            # Evaluate LLM
            if analyzer_llm.llm_evaluator:
                analyzer_llm.history_acus = ancestor_acus
                analyzer_llm.history_embeddings = analyzer_nli.history_embeddings 
                llm_score = analyzer_llm.calculate_novelty_score_llm(new_acus)
                llm_errors.append(abs(llm_score - ground_truth))
                
    print(f"Evaluated {len(nli_errors)} novelty scenario pairs.")
    print(f"NLI Cross-Encoder MAE (Mean Absolute Error): {np.mean(nli_errors):.4f}")
    
    if llm_errors:
        print(f"LLM Judge MAE (Mean Absolute Error):         {np.mean(llm_errors):.4f}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/benchmark/retrieval_benchmark.jsonl")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip Phase 1 Retrieval Eval")
    parser.add_argument("--skip_novelty", action="store_true", help="Skip Phase 2 Novelty Eval")
    args = parser.parse_args()
    
    if not args.skip_retrieval:
        run_benchmark(args.input)
        
    if not args.skip_novelty:
        evaluate_novelty_scoring(args.input)
