import sys
import os
import json
import numpy as np

# Adjust path to import scv package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scv.analysis import NoveltyAnalyzer, load_models

# Ensure models are loaded
load_models()

# Basic BM25 setup (if you want to implement the baseline)
# You might need: pip install rank_bm25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    print("Warning: rank_bm25 package not found. BM25 baseline will be skipped.")
    print("Run: pip install rank_bm25")
    HAS_BM25 = False

# ==========================================
# EXPERIMENT A: Known Lineage Data
# Format: List of Families. 
# Each family is [Ancestor, Descendant 1, Descendant 2...]
# ==========================================
DATASET_LINEAGES = [
    {
        "family": "SQuAD",
        "ancestor_name": "SQuAD 1.0",
        "ancestor_acus": [
            "SQuAD 1.0 contains 100,000+ question-answer pairs on 536 Wikipedia articles.",
            "Answers in SQuAD 1.0 are text segments resulting from crowd-workers.",
            "SQuAD 1.0 focuses on reading comprehension."
        ],
        "descendant_name": "SQuAD 2.0",
        "descendant_acus": [
            "SQuAD 2.0 combines existing reading comprehension questions with 50,000 unanswerable questions.",
            "The unanswerable questions are written adversarially by crowdworkers.",
            "Systems must not only answer reading comprehension questions but also know when no answer exists."
        ]
    },
    {
        "family": "NLI",
        "ancestor_name": "SNLI",
        "ancestor_acus": [
            "SNLI is a collection of 570k human-written English sentence pairs manually labeled for balanced classification.",
            "SNLI provides labels for entailment, contradiction, and neutral semantic relationships.",
            "The premise data is drawn from image captions."
        ],
        "descendant_name": "MultiNLI",
        "descendant_acus": [
            "MultiNLI contains 433k sentence pairs for textual entailment.",
            "The dataset covers 10 distinct genres of spoken and written English.",
            "It is designed as a drop-in replacement for the SNLI dataset."
        ]
    },
    {
        "family": "GLUE",
        "ancestor_name": "GLUE",
        "ancestor_acus": [
            "GLUE is a benchmark of 9 diverse English sentence understanding tasks.",
            "It includes question answering, sentiment analysis, and textual entailment tasks.",
            "GLUE provides an online platform for evaluating and comparing models."
        ],
        "descendant_name": "SuperGLUE",
        "descendant_acus": [
            "SuperGLUE is a new benchmark styled after GLUE with a new set of more difficult language understanding tasks.",
            "Tasks include coreference resolution and complex question answering.",
            "It features comprehensive human baselines and public leaderboards."
        ]
    }
]

def run_experiment_a():
    print("Running Experiment A: Known Lineage Retrieval")
    
    analyzer = NoveltyAnalyzer()
    
    # 1. Build History (Add Ancestors)
    history_acus = []
    history_sources = [] # Track which dataset each ACU came from
    
    # Add our known ancestors
    for lineage in DATASET_LINEAGES:
        for acu in lineage["ancestor_acus"]:
            history_acus.append(acu)
            history_sources.append(lineage["ancestor_name"])
            
    # Optionally: Add some noise (random ACUs) to the history to make it realistic
    noise_acus = [
        "The Penn Treebank provides parsed text corpus for English.",
        "WMT14 contains parallel corpora for English-German translation.",
        "The BookCorpus consists of unpublished novels from independent authors.",
        "Yelp Reviews dataset is used for sentiment classification."
    ]
    for acu in noise_acus:
        history_acus.append(acu)
        history_sources.append("Noise")
        
    # Populate the analyzer
    analyzer.add_acus(history_acus)
    print(f"History built with {len(history_acus)} total ACUs.")
    
    # Setup BM25
    bm25 = None
    if HAS_BM25:
        tokenized_corpus = [doc.lower().split(" ") for doc in history_acus]
        bm25 = BM25Okapi(tokenized_corpus)
    
    # 2. Evaluate Descendant Retrieval
    for lineage in DATASET_LINEAGES:
        print(f"\n--- Testing Family: {lineage['family']} ---")
        print(f"Querying Descendant: {lineage['descendant_name']}")
        
        target_ancestor = lineage['ancestor_name']
        
        # SCV Method (Semantic)
        from sentence_transformers import util
        from scv.analysis import SENTENCE_MODEL
        desc_embs = SENTENCE_MODEL.encode(lineage["descendant_acus"], convert_to_numpy=True)
        hits = util.semantic_search(desc_embs, analyzer.history_embeddings, top_k=5)
        
        # We aggregate hits across the descendant's ACUs. 
        # For simplicity in this eval, let's see if ANY top-5 retrieved ACU points to the ancestor.
        scv_found_ancestor = False
        print("  SCV Pipeline Top Hits:")
        for query_idx, query_hits in enumerate(hits):
            for hit in query_hits:
                corpus_id = hit['corpus_id']
                retrieved_dataset = history_sources[corpus_id]
                retrieved_acu = analyzer.history_acus[corpus_id]
                score = hit['score']
                # print(f"    Q[{query_idx}] -> {retrieved_dataset} (Score: {score:.3f}): {retrieved_acu}")
                
                if retrieved_dataset == target_ancestor:
                    scv_found_ancestor = True
                    break
        
        if scv_found_ancestor:
            print("  [✓] SCV Method successfully retrieved the Ancestor.")
        else:
            print("  [✗] SCV Method missed the Ancestor.")

        # BM25 Method (Lexical)
        if bm25:
            bm25_found_ancestor = False
            for query_acu in lineage["descendant_acus"]:
                tokenized_query = query_acu.lower().split(" ")
                doc_scores = bm25.get_scores(tokenized_query)
                # Get top 5 indices
                top_k_indices = np.argsort(doc_scores)[::-1][:5]
                
                for idx in top_k_indices:
                    if history_sources[idx] == target_ancestor:
                        bm25_found_ancestor = True
                        break
                        
            if bm25_found_ancestor:
                print("  [✓] BM25 successfully retrieved the Ancestor.")
            else:
                print("  [✗] BM25 missed the Ancestor.")


# ==========================================
# EXPERIMENT C: Hard Negatives
# ==========================================
HARD_NEGATIVES_DATA = [
    {
        "query_name": "Medical Translation Dataset",
        "query_acus": [
            "The dataset contains 10,000 parallel sentences for English-Spanish medical translation.",
            "It is designed to evaluate clinical terminology translation models."
        ],
        "true_positive": {
            "name": "BioMT English-Spanish",
            "acus": [
                "BioMT provides a parallel corpus for biomedical machine translation between English and Spanish.",
                "The data is sourced from biological literature and clinical notes."
            ]
        },
        "hard_negatives": [
            {
                "name": "Legal Translation Dataset (Domain Mismatch)",
                "acus": [
                    "The dataset contains 10,000 parallel sentences for English-Spanish legal translation.",
                    "It is designed to evaluate legal contract translation models."
                ]
            },
            {
                "name": "Medical NER Dataset (Task Mismatch)",
                "acus": [
                    "The dataset contains 10,000 English and Spanish clinical notes for Named Entity Recognition.",
                    "It is designed to evaluate clinical terminology extraction models."
                ]
            }
        ]
    }
]

def run_experiment_c():
    print("\n\nRunning Experiment C: Contrastive Hard Negatives")
    
    analyzer = NoveltyAnalyzer()
    
    for case in HARD_NEGATIVES_DATA:
        print(f"\n--- Testing Query: {case['query_name']} ---")
        
        # Build history with TP and Hard Negatives
        history_acus = []
        history_sources = []
        
        # Add True Positive
        for acu in case['true_positive']['acus']:
            history_acus.append(acu)
            history_sources.append(case['true_positive']['name'])
            
        # Add Hard Negatives
        for hn in case['hard_negatives']:
            for acu in hn['acus']:
                history_acus.append(acu)
                history_sources.append(hn['name'])
                
        analyzer.add_acus(history_acus)
        
        # Setup BM25
        bm25 = None
        if HAS_BM25:
            tokenized_corpus = [doc.lower().split(" ") for doc in history_acus]
            bm25 = BM25Okapi(tokenized_corpus)
            
        # Find Nearest Neighbors for Query
        from scv.analysis import SENTENCE_MODEL
        from sentence_transformers import util
        
        desc_embs = SENTENCE_MODEL.encode(case["query_acus"], convert_to_numpy=True)
        hits = util.semantic_search(desc_embs, analyzer.history_embeddings, top_k=3)
        
        print("\n  SCV Pipeline Ranking:")
        scv_tp_rank = -1
        # To keep it simple, we'll just look at the average rank/score across query ACUs for each source
        scv_scores = {}
        for query_idx, query_hits in enumerate(hits):
            for rank, hit in enumerate(query_hits):
                corpus_id = hit['corpus_id']
                retrieved_source = history_sources[corpus_id]
                score = hit['score']
                
                if retrieved_source not in scv_scores:
                    scv_scores[retrieved_source] = 0.0
                scv_scores[retrieved_source] += score
                
        # Sort by total score
        sorted_scv = sorted(scv_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (source, score) in enumerate(sorted_scv):
            print(f"    {rank+1}. {source} (Aggr Score: {score:.3f})")
            if source == case['true_positive']['name'] and scv_tp_rank == -1:
                scv_tp_rank = rank + 1
                
        if scv_tp_rank == 1:
            print("  [✓] SCV Method correctly ranked the True Positive #1.")
        else:
            print("  [✗] SCV Method failed to rank the True Positive #1.")

        # BM25 Ranking
        if bm25:
            print("\n  BM25 Pipeline Ranking:")
            bm25_scores = {}
            for query_acu in case["query_acus"]:
                tokenized_query = query_acu.lower().split(" ")
                doc_scores = bm25.get_scores(tokenized_query)
                for idx, score in enumerate(doc_scores):
                    source = history_sources[idx]
                    if source not in bm25_scores:
                        bm25_scores[source] = 0.0
                    bm25_scores[source] += score
                    
            sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            bm25_tp_rank = -1
            for rank, (source, score) in enumerate(sorted_bm25):
                print(f"    {rank+1}. {source} (Aggr Score: {score:.3f})")
                if source == case['true_positive']['name'] and bm25_tp_rank == -1:
                    bm25_tp_rank = rank + 1
                    
            if bm25_tp_rank == 1:
                print("  [✓] BM25 correctly ranked the True Positive #1.")
            else:
                print("  [✗] BM25 failed to rank the True Positive #1.")



# ==========================================
# EXPERIMENT B: LLM As A Judge Scaffold
# ==========================================
def run_experiment_b():
    print("\n\nRunning Experiment B: LLM-as-a-Judge Scaffold")
    print("  Note: Requires OPENAI_API_KEY to be set in the environment.")
    
    # Check if OPENAI_API_KEY is available
    import os
    if "OPENAI_API_KEY" not in os.environ:
        print("  [Skipped] OPENAI_API_KEY not found. This experiment requires an LLM to judge relevance.")
        return

    # To avoid modifying the core system's LLMNoveltyEvaluator which we mocked out in analysis.py,
    # we'll build a standalone evaluator using curator.
    try:
        from bespokelabs import curator
        from pydantic import BaseModel, Field
        
        class RelevanceJudgment(BaseModel):
            relevance_score: int = Field(description="Score from 0-3 (0=Unrelated, 1=Vaguely related domain, 2=Highly related task/method, 3=Direct antecedent/variant).")
            explanation: str = Field(description="Brief explanation of the score.")

        class RelevanceEvaluator(curator.LLM):
            response_format = RelevanceJudgment
            
            def prompt(self, input: dict) -> str:
                return f"""Given Query ACU: "{input['query_acu']}"
And Retrieved ACU: "{input['retrieved_acu']}"

Rate their relationship on a scale of 0-3 (0=Unrelated, 1=Vaguely related domain, 2=Highly related task/method, 3=Direct antecedent/variant).
"""

        evaluator = RelevanceEvaluator(model_name="gpt-4o")
        
        # Test case
        query_acu = "The dataset contains 10,000 parallel sentences for English-Spanish medical translation."
        retrieved_acus = [
            "BioMT provides a parallel corpus for biomedical machine translation between English and Spanish.",
            "The dataset contains 10,000 parallel sentences for English-Spanish legal translation.",
        ]
        
        for retrieved in retrieved_acus:
            print(f"\n  Query: {query_acu}")
            print(f"  Retrieved: {retrieved}")
            
            # This would run the API call
            eval_res = evaluator([{"query_acu": query_acu, "retrieved_acu": retrieved}])
            if eval_res and eval_res.dataset:
                result = eval_res.dataset[0]
                # depending on curator version: result might be a dict or a pydantic model
                if hasattr(result, "relevance_score"):
                    score = result.relevance_score
                    exp = result.explanation
                elif isinstance(result, dict) and "relevance_score" in result:
                    score = result["relevance_score"]
                    exp = result["explanation"]
                elif isinstance(result, dict) and "parsed_response_message" in result:
                    msg = result["parsed_response_message"]
                    score = msg.get("relevance_score")
                    exp = msg.get("explanation")
                else:
                    score = "Unknown"
                    exp = "Failed to parse."
                    
                print(f"    --> Score: {score} | Reason: {exp}")
            else:
                 print("    --> Failed to get evaluation result.")

    except Exception as e:
        print(f"  Error setting up/running LLM Evaluator: {e}")


# ==========================================
# EXPERIMENT D: Human Eval Scaffold
# ==========================================
def run_experiment_d():
    print("\n\nRunning Experiment D: Blind Human Evaluation Scaffold")
    print("  This generates a blind A/B test for a human annotator to grade relevance.")
    
    # We will reuse the query and history from Experiment C to show the setup
    case = HARD_NEGATIVES_DATA[0]
    
    # Simulate generating the blind output
    import random
    
    method_1_name = "Method Alpha"
    method_2_name = "Method Beta"
    
    # Suppose SCV outputted BioMT and Legal, and BM25 outputted Legal and Medical NER
    scv_outputs = [
        "BioMT provides a parallel corpus for biomedical machine translation between English and Spanish.",
        "The dataset contains 10,000 parallel sentences for English-Spanish legal translation."
    ]
    bm25_outputs = [
        "The dataset contains 10,000 parallel sentences for English-Spanish legal translation.",
        "The dataset contains 10,000 English and Spanish clinical notes for Named Entity Recognition."
    ]
    
    # Randomize for blind test
    if random.choice([True, False]):
        alpha_outputs = scv_outputs
        beta_outputs = bm25_outputs
        key = {"Method Alpha": "SCV", "Method Beta": "BM25"}
    else:
        alpha_outputs = bm25_outputs
        beta_outputs = scv_outputs
        key = {"Method Alpha": "BM25", "Method Beta": "SCV"}
        
    print("\n================= ANNOTATION TASK =================")
    print("QUERY ACUs:")
    for acu in case['query_acus']:
        print(f" - {acu}")
        
    print(f"\n[{method_1_name}] Retrieved Context:")
    for i, out in enumerate(alpha_outputs):
        print(f" {i+1}. {out}")
        
    print(f"\n[{method_2_name}] Retrieved Context:")
    for i, out in enumerate(beta_outputs):
        print(f" {i+1}. {out}")
        
    print("\nQ: Which method returns more RELEVANT context for evaluating the novelty of the query?")
    print("  [ ] Method Alpha")
    print("  [ ] Method Beta")
    print("  [ ] Tie / Hard to tell")
    print("==================================================")
    
    # Do not print the key usually, but for debug purposes here:
    print(f"  [Debug Key]: {key}")

if __name__ == "__main__":
    run_experiment_a()
    run_experiment_c()
    run_experiment_b()
    run_experiment_d()
