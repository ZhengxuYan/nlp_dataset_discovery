from typing import List, Dict
import numpy as np
from .models import DatasetScvInfo, NoveltyScoringResult
from bespokelabs import curator

# Model Global State (Lazy Load)
SPECTER_MODEL = None
SENTENCE_MODEL = None
NLI_MODEL = None

def load_models():
    global SPECTER_MODEL, SENTENCE_MODEL, NLI_MODEL
    if SPECTER_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            print("Loading models...")
            SPECTER_MODEL = SentenceTransformer('allenai/specter2_base')
            SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            # NLI model might take more memory, load carefully
            NLI_MODEL = CrossEncoder('cross-encoder/nli-deberta-v3-small')
        except Exception as e:
            print(f"Error loading models: {e}")

def compute_embeddings(text: str) -> List[float]:
    """Step 2: Compute embedding using SPECTER2."""
    load_models()
    if SPECTER_MODEL and text:
        return SPECTER_MODEL.encode(text, convert_to_numpy=True).tolist()
    return []

def calculate_diversity_score(dataset: DatasetScvInfo) -> float:
    """Step 4: Heuristic diversity score."""
    score = 0.0
    # Language bonus
    if len(dataset.languages) > 1:
        score += 0.3
    elif dataset.languages and dataset.languages[0].lower() != 'english':
        score += 0.2
    
    # Domain bonus
    if dataset.domain.lower() not in ['general', 'unknown']:
        score += 0.2
        
    # Size bonus (Check num_samples and storage)
    if dataset.size:
        # Check num_samples (Integer check)
        if dataset.size.num_samples and dataset.size.num_samples > 0:
            ns = dataset.size.num_samples
            if ns >= 10_000_000: # Massive (Top ~1-2%)
                score += 0.3
            elif ns >= 1_000_000: # Large (Top ~5-10%)
                score += 0.2
            elif ns >= 100_000: # Medium (Top ~15%)
                score += 0.1
                
        # Fallback/Additional check on storage strings if samples not captured
        if not dataset.size.num_samples or dataset.size.num_samples <= 0:
             size_signals = [dataset.size.storage_size, dataset.size.size_str]
             size_text = " ".join([s.lower() for s in size_signals if s])
             if any(x in size_text for x in ['billion', 'trillion', 'tb', 'petabyte']):
                 score += 0.3
             elif any(x in size_text for x in ['million', 'gb', '100k', '500k']):
                 score += 0.2
        
        # Check num_classes (Complexity bonus)
        if dataset.size.num_classes and dataset.size.num_classes > 10:
            score += 0.1

    return min(1.0, score)

def calculate_quality_score(dataset: DatasetScvInfo) -> float:
    """Step 5: Quality/Transparency score (Robust & Additive)."""
    score = 0.0 # Base score (additive model)
    
    # 1. Availability / Access (Max +0.4)
    # Check both the specific URL field and generic usage description
    has_url = (dataset.availability_url and dataset.availability_url.lower() not in ['none', 'unknown', 'nan', ''])
    if not has_url:
        if dataset.usage_description:
            desc_lower = dataset.usage_description.lower()
            if 'url' in desc_lower or 'link' in desc_lower or 'http' in desc_lower:
                has_url = True
    
    if has_url:
        score += 0.3
        # Bonus for known high-quality hosts
        url_lower = (dataset.availability_url or "").lower()
        if any(host in url_lower for host in ['github.com', 'huggingface.co', 'paperswithcode.com', 'kaggle.com', 'zenodo.org']):
            score += 0.1

    # 2. License / Usability (Max +0.2)
    has_license = (dataset.license and dataset.license.lower() not in ['unknown', 'none', 'nan', ''])
    if has_license:
        score += 0.1
        # Bonus for open licenses
        lic_lower = dataset.license.lower()
        if any(open_lic in lic_lower for open_lic in ['apache', 'mit', 'creative commons', 'bsd', 'cc-by', 'public domain']):
            score += 0.1

    # 3. Documentation (Max +0.2)
    has_docs = (dataset.documentation_type and dataset.documentation_type.lower() not in ['none', 'unknown', 'nan', ''])
    if has_docs:
        score += 0.1
        # Bonus for structured/comprehensive docs
        doc_lower = dataset.documentation_type.lower()
        if any(type_name in doc_lower for type_name in ['datasheet', 'card', 'readme', 'appendix']):
            score += 0.1

    # 4. Metadata Completeness (Max +0.1)
    # Language known
    if dataset.languages and dataset.languages[0].lower() not in ['unknown', 'none']:
        score += 0.05
    # Domain known
    if dataset.domain and dataset.domain.lower() not in ['unknown', 'general', 'none']:
        score += 0.05
        
    # 5. Maintenance (Max +0.05)
    if dataset.maintenance_status and dataset.maintenance_status.lower() == 'yes':
        score += 0.05

    # 6. Transparency Issues (Penalty)
    # Filter out "Missing License" as it's common (handled by lack of bonus above)
    if dataset.transparency_issues:
        relevant_issues = [i for i in dataset.transparency_issues if 'license' not in i.lower()]
        score -= 0.05 * len(relevant_issues)

    return max(0.0, min(1.0, score))

class LlmNoveltyEvaluator(curator.LLM):
    response_format = NoveltyScoringResult

    def prompt(self, input: dict) -> str:
        new_acus_text = "\n".join(f"- {a}" for a in input.get("new_acus", []))
        history_acus_text = "\n".join(f"- {a}" for a in input.get("history_acus", []))
        if not history_acus_text.strip():
            history_acus_text = "None. Assume high baseline novelty as no direct history is provided."
            
        return f"""You are an expert NLP researcher evaluating the novelty of a newly introduced dataset. 
You are given a list of Atomic Content Units (ACUs) representing the claims made about the NEW dataset, and a list of ACUs representing previously existing datasets (History).

Compare the NEW dataset's ACUs against the History ACUs. Evaluate the novelty of the new dataset across the following 3 dimensions:
1. **Task/Domain Novelty**: Does the new dataset cover tasks, subjects, or domains that are non-existent or rare in the history?
2. **Methodological Novelty**: Does the new dataset introduce a novel collection, annotation, or generation technique compared to history?
3. **Data Scale/Coverage**: Is the dataset significantly larger, more diverse, or spanning more languages compared to what has been done before?

For each dimension, provide:
- A brief explanation comparing the new ACUs to the most relevant history ACUs.
- A score from 0.0 (identical/no novelty) to 1.0 (highly novel/groundbreaking).

Finally, calculate the `average_novelty_score` as the mean of the three dimension scores.

NEW DATASET ACUS:
{new_acus_text}

HISTORY ACUS (Prior Work Context):
{history_acus_text}
"""

class NoveltyAnalyzer:
    def __init__(self):
        self.history_acus = [] # List of text
        self.history_embeddings = None # Matrix
        load_models()
        # We only need embeddings for this experiment, so we can ignore the LLM
        self.llm_evaluator = None
        
    def add_acus(self, acus: List[str]):
        if not acus or not SENTENCE_MODEL:
            return
        
        new_embeddings = SENTENCE_MODEL.encode(acus, convert_to_numpy=True)
        self.history_acus.extend(acus)
        
        if self.history_embeddings is None:
            self.history_embeddings = new_embeddings
        else:
            self.history_embeddings = np.vstack([self.history_embeddings, new_embeddings])
            
    def calculate_novelty_score_nli(self, new_acus: List[str], forced_context: List[str] = None) -> float:
        """
        Check if new_acus are entailed by history OR forced_context using NLI cross-encoder.
        Returns a score 0.0 (Not Novel) to 1.0 (Highly Novel).
        """
        if not new_acus:
            return 0.0
            
        # 1. Embed new ACUs
        new_embs = SENTENCE_MODEL.encode(new_acus, convert_to_numpy=True)
        
        # Prepare retrieval candidates
        hits = []
        if self.history_embeddings is not None and len(self.history_embeddings) > 0:
            from sentence_transformers import util
            # Retrieve Top-K candidates for each ACU from history
            hits = util.semantic_search(new_embs, self.history_embeddings, top_k=5)
        else:
            # No history, fill with empty lists to match indexing
            hits = [[] for _ in range(len(new_acus))]
        
        entailment_scores = []
        
        for i, hit_list in enumerate(hits):
            acu_text = new_acus[i]
            
            # Gather all candidate premises for this ACU
            candidate_premises = []
            
            # A. Forced Context
            if forced_context:
                candidate_premises.extend(forced_context)
                
            # B. Retrieved History
            for hit in hit_list:
                corpus_id = hit['corpus_id']
                candidate_premises.append(self.history_acus[corpus_id])
            
            if not candidate_premises:
                # No history and no forced context -> Novel
                entailment_scores.append(1.0)
                continue
                
            # Construct pairs: (premise, hypothesis)
            pairs = [(premise, acu_text) for premise in candidate_premises]
            
            # Batch predict
            scores = NLI_MODEL.predict(pairs)
            
            # Convert logits to probabilities (Softmax)
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
            entail_probs = probs[:, 1]
            
            # Get max entailment probability across all candidates
            max_entailment_prob = float(np.max(entail_probs))
            
            entailment_scores.append(1.0 - max_entailment_prob)
            
        return float(np.mean(entailment_scores))

    def calculate_novelty_score_llm(self, new_acus: List[str], forced_context: List[str] = None) -> float:
        """
        Check if new_acus are novel compared to history OR forced_context using an LLM.
        Returns a score 0.0 (Not Novel) to 1.0 (Highly Novel).
        """
        if not new_acus:
            return 0.0
            
        # 1. Embed new ACUs
        new_embs = SENTENCE_MODEL.encode(new_acus, convert_to_numpy=True)
        
        # Prepare retrieval candidates
        candidate_premises = set()
        
        # A. Forced Context
        if forced_context:
            for c in forced_context:
                candidate_premises.add(c)
                
        # B. Retrieved History
        if self.history_embeddings is not None and len(self.history_embeddings) > 0:
            from sentence_transformers import util
            hits = util.semantic_search(new_embs, self.history_embeddings, top_k=5)
            for hit_list in hits:
                for hit in hit_list:
                    corpus_id = hit['corpus_id']
                    candidate_premises.add(self.history_acus[corpus_id])
        
        history_acus_list = list(candidate_premises)
        
        try:
            # Optionally log what's being passed
            print(f"Calling LLM for Novelty Scoring with {len(new_acus)} new ACUs and {len(history_acus_list)} history ACUs")
            eval_res = self.llm_evaluator([
                {"new_acus": new_acus, "history_acus": history_acus_list}
            ])
            
            if not eval_res or not eval_res.dataset:
                print("LLM evaluation returned empty dataset.")
                return 0.0
                
            # CuratorResponse.dataset is a list of results
            row_result = eval_res.dataset[0]
            
            # Since evaluator(..., response_format=...) is used, the returned item might be
            # the Pydantic object itself, OR a dict containing parsed_response_message.
            # Let's handle both dynamically:
            if hasattr(row_result, "average_novelty_score"):
                 result = row_result
            elif isinstance(row_result, dict) and "average_novelty_score" in row_result:
                 result = NoveltyScoringResult(**row_result)
            elif isinstance(row_result, dict) and "parsed_response_message" in row_result:
                 result = row_result["parsed_response_message"]
                 if isinstance(result, dict):
                     result = NoveltyScoringResult(**result)
            else:
                 print(f"Unrecognized response format: {type(row_result)}")
                 return 0.0
            
            # (Optional) Log the detailed dimensions
            for dim in result.dimensions:
                print(f" - [{dim.dimension_name} ({dim.score:.2f})]: {dim.explanation}")
            print(f" - Average Score: {result.average_novelty_score:.2f}")
            
            return result.average_novelty_score
        except Exception as e:
            print(f"Error evaluating novelty with LLM: {e}")
            return 0.0

def analyze_novelty_and_get_score(dataset: DatasetScvInfo, analyzer: NoveltyAnalyzer, forced_context: List[str] = None, method: str = 'llm') -> float:
    if not dataset.is_introduced:
        return 0.0
    
    if method == 'nli':
        return analyzer.calculate_novelty_score_nli(dataset.acus, forced_context=forced_context)
    else:
        return analyzer.calculate_novelty_score_llm(dataset.acus, forced_context=forced_context)

def construct_scv(novelty: float, diversity: float, quality: float) -> Dict[str, float]:
    """Step 6: Construct SCV."""
    return {
        "novelty": novelty,
        "diversity": diversity,
        "quality": quality,
    }
