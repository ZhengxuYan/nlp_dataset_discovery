import os
import json
import time
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROMPT_TEMPLATE = """You are an expert human annotator evaluating an LLM's attribution predictions.
Your goal is to evaluate the provided prediction and fill in three missing fields.

You are given:
- Query ACU: The claim being evaluated.
- Predicted Support Status: The LLM's prediction.
- Predicted Best Prior ACUs: The evidence the LLM selected.
- Predicted Rationale: The LLM's reasoning.

You must output a JSON object with EXACTLY these three keys:
1. "gold_support_status": Your human judgment of the correct support label. Must be one of: ["supported", "partially_supported", "unsupported"].
   - "supported": prior ACU supports the core claim.
   - "partially_supported": prior ACU supports part, but query ACU has important new details (e.g., numbers, sizes).
   - "unsupported": prior ACU does not support the claim, or is only tangentially related.
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

def process_with_openai(prompt, model_name):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return response.choices[0].message.content

def process_with_gemini(prompt, model_name):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name,
        generation_config={"response_mime_type": "application/json", "temperature": 0.0}
    )
    response = model.generate_content(prompt)
    return response.text

def main():
    parser = argparse.ArgumentParser(description="Auto-annotate human evaluation sample.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", 
                        help="Model to use (e.g., gpt-4o-mini, gemini-3.1-flash-lite)")
    args = parser.parse_args()

    if args.model.startswith("gpt"):
        process_func = process_with_openai
    elif args.model.startswith("gemini"):
        process_func = process_with_gemini
    else:
        print(f"Unknown model prefix for '{args.model}'. Defaulting to OpenAI API.")
        process_func = process_with_openai

    input_file = "data/benchmark/retrieval_cache/acl_pdf130_gpt54mini_oracle_human_eval_sample100.jsonl"
    
    # Ensure correct extension depending on your preference, sticking to .jsonl as the format is line-by-line JSON
    model_slug = args.model.replace(".", "").replace("-", "")
    output_file = f"data/benchmark/retrieval_cache/acl_pdf130_gpt54mini_oracle_human_eval_sample100_labeled_{model_slug}.jsonl"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
        
    labeled_lines = []
    print(f"Processing {len(lines)} rows with {args.model}...")
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        data = json.loads(line.strip())
        
        prompt = PROMPT_TEMPLATE.format(
            query_acu=data.get('query_acu', ''),
            pred_support_status=data.get('pred_support_status', ''),
            pred_best_prior_acus=json.dumps(data.get('pred_best_prior_acus', [])),
            pred_rationale=data.get('pred_rationale', '')
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_text = process_func(prompt, args.model)
                result = json.loads(result_text)
                
                data['gold_support_status'] = result.get('gold_support_status')
                data['selected_evidence_relevant'] = result.get('selected_evidence_relevant')
                data['rationale_grounded'] = result.get('rationale_grounded')
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error on line {i}: {e}")
                    data['gold_support_status'] = None
                    data['selected_evidence_relevant'] = None
                    data['rationale_grounded'] = None
                time.sleep(2)
                
        labeled_lines.append(data)
        
        if (i + 1) % 10 == 0 or i == len(lines) - 1:
            print(f"Processed {i + 1}/{len(lines)}")
            
        # Write incrementally
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in labeled_lines:
                f.write(json.dumps(item) + '\n')
        
        # Avoid rate limits
        time.sleep(0.5 if args.model.startswith("gpt") else 1.0)
            
    print(f"Finished! Output fully saved to {output_file}")

if __name__ == "__main__":
    main()
