import os
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)

# Use Gemini 1.5 Pro
model = genai.GenerativeModel(
    'gemini-3.1-flash-lite',
    generation_config={"response_mime_type": "application/json"}
)

PROMPT_TEMPLATE = """You are an expert human annotator evaluating an LLM's attribution predictions.
Your goal is to evaluate the provided prediction and fill in three missing fields.

You are given:
- Query ACU: The claim being evaluated.
- Predicted Support Status: The LLM's prediction.
- Predicted Best Prior ACUs: The evidence the LLM selected.
- Predicted Rationale: The LLM's reasoning.

You must output a JSON object with EXACTLY these three keys:
1. "gold_support_status": Your human judgment of the correct support label. Must be one of: ["supported", "partially_supported", "unsupported", "contradicted", "not_comparable"].
   - "supported": prior ACU supports the core claim.
   - "partially_supported": prior ACU supports part, but query ACU has important new details (e.g., numbers, sizes).
   - "unsupported": prior ACU does not support the claim, or is only tangentially related.
   - "contradicted": prior ACU explicitly conflicts.
   - "not_comparable": query ACU and prior ACU are not comparable.
2. "selected_evidence_relevant": boolean (true/false).
   - true if the LLM selected relevant prior ACUs. 
   - true if pred_best_prior_acus is empty [] and you believe it correctly found no evidence.
   - false if selected evidence is wrong/drifted, or if it is empty but you strongly suspect evidence exists.
3. "rationale_grounded": boolean (true/false).
   - true if rationale doesn't hallucinate and accurately reflects query/evidence.
   - false if rationale hallucinates, calls unsupported things supported, or cites fake facts.

Input Data:
Query ACU: {query_acu}
Predicted Support Status: {pred_support_status}
Predicted Best Prior ACUs: {pred_best_prior_acus}
Predicted Rationale: {pred_rationale}

Output ONLY a JSON object:
{{
    "gold_support_status": "string",
    "selected_evidence_relevant": true,
    "rationale_grounded": true
}}
"""

def main():
    input_file = "artifacts/paper_results/human_evidence_sample.jsonl"
    output_file = "data/benchmark/retrieval_cache/gpt54mini_attribution_human_sample_labeled.jsonl"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    labeled_lines = []
    
    print(f"Processing {len(lines)} rows...")
    
    for i, line in enumerate(lines):
        data = json.loads(line.strip())
        
        prompt = PROMPT_TEMPLATE.format(
            query_acu=data.get('query_acu', ''),
            pred_support_status=data.get('pred_support_status', ''),
            pred_best_prior_acus=json.dumps(data.get('pred_best_prior_acus', [])),
            pred_rationale=data.get('pred_rationale', '')
        )
        
        # Retry loop for API calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                result = json.loads(response.text)
                
                data['gold_support_status'] = result.get('gold_support_status')
                data['selected_evidence_relevant'] = result.get('selected_evidence_relevant')
                data['rationale_grounded'] = result.get('rationale_grounded')
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error on line {i}: {e}")
                time.sleep(2)
                
        labeled_lines.append(data)
        print(f"Processed {i + 1}/{len(lines)}")
        
        # Avoid rate limits
        time.sleep(1)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in labeled_lines:
            f.write(json.dumps(item) + '\n')
            
    print(f"Finished! Output saved to {output_file}")

if __name__ == "__main__":
    main()
