from typing import List, Dict
import numpy as np
from .models import DatasetScvInfo

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
            # NLI model might take more memory, load carefuly
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

class NoveltyAnalyzer:
    def __init__(self):
        self.history_acus = [] # List of text
        self.history_embeddings = None # Matrix
        load_models()
        
    def add_acus(self, acus: List[str]):
        if not acus or not SENTENCE_MODEL:
            return
        
        new_embeddings = SENTENCE_MODEL.encode(acus, convert_to_numpy=True)
        self.history_acus.extend(acus)
        
        if self.history_embeddings is None:
            self.history_embeddings = new_embeddings
        else:
            self.history_embeddings = np.vstack([self.history_embeddings, new_embeddings])
            
    def calculate_novelty_score(self, new_acus: List[str]) -> float:
        """
        Check if new_acus are entailed by history.
        Returns a score 0.0 (Not Novel) to 1.0 (Highly Novel).
        """
        if not new_acus:
            return 0
            
        # 1. Embed new ACUs
        new_embs = SENTENCE_MODEL.encode(new_acus, convert_to_numpy=True)
        
        if self.history_embeddings is None or len(self.history_embeddings) == 0:
            return 1.0 # First entry is novel by definition (relative to empty history)
            
        from sentence_transformers import util
        # 2. Retrieve Top-K candidates for each ACU
        hits = util.semantic_search(new_embs, self.history_embeddings, top_k=5)
        
        entailment_scores = []
        
        for i, hit_list in enumerate(hits):
            acu_text = new_acus[i]
            max_entailment_prob = 0.0
            
            for hit in hit_list:
                corpus_id = hit['corpus_id']
                premise = self.history_acus[corpus_id]
                
                scores = NLI_MODEL.predict([(premise, acu_text)])[0] 
                probs = np.exp(scores) / np.sum(np.exp(scores))
                
                # Index 1 is entailment for nli-deberta-v3-small (usually)
                entail_prob = probs[1] 
                if entail_prob > 0.5: 
                    max_entailment_prob = max(max_entailment_prob, entail_prob)
            
            entailment_scores.append(1.0 - max_entailment_prob)
            
        return float(np.mean(entailment_scores))

def analyze_novelty_and_get_score(dataset: DatasetScvInfo, analyzer: NoveltyAnalyzer) -> float:
    if not dataset.is_introduced:
        return 0.0
    return analyzer.calculate_novelty_score(dataset.acus)

def construct_scv(novelty: float, diversity: float, quality: float) -> Dict[str, float]:
    """Step 6: Construct SCV."""
    return {
        "novelty": novelty,
        "diversity": diversity,
        "quality": quality,
    }
