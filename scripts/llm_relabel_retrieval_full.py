import os
import csv
import json
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

PROMPT_GOLD_PRIOR = """You are an expert NLP annotator reviewing a claim-level prior evidence retrieval benchmark.
Your task is to relabel the gold prior DCU set to ensure high precision.

Input is a query DCU and a candidate prior DCU. 
Task: Determine if the prior DCU should be the gold prior evidence for the query DCU.

Rules for 'human_gold_prior_decision':
1. 'covered': The prior DCU expresses basically the same dataset contribution (task, domain, language, modality, data source, annotation protocol, scale, release status, or evaluation use).
2. 'partially_covered': The prior DCU has real overlap but differs in at least one important dimension (task, domain, language, modality, source, etc).
3. 'not_evidence': The candidate prior DCU should NOT be a gold prior. E.g., it only has broad topical similarity, it's an unrelated claim in the same paper, it's a generic dataset statement, different contribution dimension, or only model/method/experiment related. If unsure, default to 'not_evidence' (we need high precision).

Quick Decision Test: "If a retrieval system returns this prior DCU for the query DCU, should we reward it?"
- If yes -> covered / partially_covered.
- If no -> not_evidence.

Input Row Data:
{row_data}

Output JSON with EXACTLY these keys:
{{
  "human_gold_prior_decision": "covered" | "partially_covered" | "not_evidence",
  "human_gold_prior_note": "A short reasoning explaining your choice."
}}
"""

PROMPT_HARD_NEGATIVE = """You are an expert NLP annotator reviewing a claim-level prior evidence retrieval benchmark.
Your task is to validate proposed hard negative DCUs to ensure they are high-quality distractors.

Input is a query DCU, a gold prior DCU set, and a proposed hard negative DCU.
Task: Determine if the hard negative is valid.

Rules for 'human_hard_negative_decision':
1. 'valid_hard_negative': The prior DCU seems relevant and easily confused with the query, but does NOT cover or partially cover the query DCU. It is a reasonable difficult distractor.
2. 'actually_evidence': The supposed negative is actually valid prior evidence. If it should be marked as covered/partially_covered for the query, select this.
3. 'irrelevant_not_plausible': The negative is too unrelated to be a true hard negative (e.g., random unrelated claim, completely different domain/task/contribution type, no real confusion potential).

Quick Decision Test: "Would this be a plausible wrong retrieval result that a system should rank below the gold prior?"
- If yes -> valid_hard_negative.
- If it should be rewarded -> actually_evidence.
- If it's too unrelated -> irrelevant_not_plausible.

Input Row Data:
{row_data}

Output JSON with EXACTLY these keys:
{{
  "human_hard_negative_decision": "valid_hard_negative" | "actually_evidence" | "irrelevant_not_plausible",
  "human_hard_negative_note": "A short reasoning explaining your choice."
}}
"""

def process_with_openai(prompt, model_name):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Enforcing seed and temp 0.0 for maximum consistency across duplicate runs
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        seed=42
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
    parser = argparse.ArgumentParser(description="Auto-relabel full retrieval benchmark.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    args = parser.parse_args()

    if args.model.startswith("gpt"):
        process_func = process_with_openai
    elif args.model.startswith("gemini"):
        process_func = process_with_gemini
    else:
        print(f"Unknown model prefix for '{args.model}'. Defaulting to OpenAI API.")
        process_func = process_with_openai

    file_basename = os.path.basename(args.input_file)
    
    if "gold_prior" in file_basename:
        prompt_template = PROMPT_GOLD_PRIOR
        human_keys = ["human_gold_prior_decision", "human_gold_prior_note"]
    elif "hard_negative" in file_basename:
        prompt_template = PROMPT_HARD_NEGATIVE
        human_keys = ["human_hard_negative_decision", "human_hard_negative_note"]
    else:
        raise ValueError(f"Could not determine task type from filename: {file_basename}. Filename must contain 'gold_prior' or 'hard_negative'.")

    model_slug = args.model.replace(".", "").replace("-", "")
    
    base, ext = os.path.splitext(args.input_file)
    output_file = f"{base}_labeled_{model_slug}.csv"
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    print(f"Loaded {len(rows)} rows from {args.input_file}")
    
    for k in human_keys:
        if k not in fieldnames:
            fieldnames.append(k)
    if 'annotator_id' not in fieldnames:
        fieldnames.append('annotator_id')

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"Processing with {args.model} (Temp 0.0, Seed 42 for consistency)...")
    
    for i, row in enumerate(rows):
        # Exclude existing human fields from input context
        input_data = {k: v for k, v in row.items() if k and not k.startswith("human_") and k != "annotator_id"}
        
        prompt = prompt_template.format(row_data=json.dumps(input_data, indent=2, ensure_ascii=False))
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_text = process_func(prompt, args.model)
                result = json.loads(result_text)
                
                for k in human_keys:
                    row[k] = result.get(k, "")
                row['annotator_id'] = args.model
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error on row {i}: {e}")
                    row['annotator_id'] = f"{args.model}_error"
                time.sleep(2)
                
        with open(output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            
        if (i + 1) % 50 == 0 or i == len(rows) - 1:
            print(f"Processed {i + 1}/{len(rows)}")
            
        time.sleep(0.5 if args.model.startswith("gpt") else 1.0)
            
    print(f"Finished! Output fully saved to {output_file}")

if __name__ == "__main__":
    main()
