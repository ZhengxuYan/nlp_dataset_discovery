import os
import csv
import json
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

PROMPT_ADJUDICATE_GOLD = """You are an expert NLP Senior Adjudicator. 
Your task is to resolve a disagreement between two annotators regarding a claim-level prior evidence retrieval benchmark.

Task: Determine if the candidate prior DCU should be the gold prior evidence for the query DCU.

Rules for 'adjudicated_gold_prior_decision':
1. 'covered': The prior DCU expresses basically the same dataset contribution (task, domain, language, modality, data source, annotation protocol, scale, release status, or evaluation use).
2. 'partially_covered': The prior DCU has real overlap but differs in at least one important dimension (task, domain, language, modality, source, etc).
3. 'not_evidence': The candidate prior DCU should NOT be a gold prior. E.g., it only has broad topical similarity, it's an unrelated claim in the same paper, it's a generic dataset statement, different contribution dimension, or only model/method/experiment related. If unsure, default to 'not_evidence' (we need high precision).

Input Disagreement Row (contains Jason's and Jiaxin's decisions/notes):
{row_data}

Output JSON with EXACTLY these keys:
{{
  "adjudicated_gold_prior_decision": "covered" | "partially_covered" | "not_evidence",
  "adjudicated_gold_prior_note": "A short explanation of why you chose this final adjudication."
}}
"""

PROMPT_ADJUDICATE_HARD_NEGATIVE = """You are an expert NLP Senior Adjudicator. 
Your task is to resolve a disagreement between two annotators regarding proposed hard negative DCUs for a retrieval benchmark.

Task: Determine if the hard negative is valid.

Rules for 'adjudicated_hard_negative_decision':
1. 'valid_hard_negative': The prior DCU seems relevant and easily confused with the query, but does NOT cover or partially cover the query DCU. It is a reasonable difficult distractor.
2. 'actually_evidence': The supposed negative is actually valid prior evidence. If it should be marked as covered/partially_covered for the query, select this.
3. 'irrelevant_not_plausible': The negative is too unrelated to be a true hard negative (e.g., random unrelated claim, completely different domain/task/contribution type, no real confusion potential).

Input Disagreement Row (contains Jason's and Jiaxin's decisions/notes):
{row_data}

Output JSON with EXACTLY these keys:
{{
  "adjudicated_hard_negative_decision": "valid_hard_negative" | "actually_evidence" | "irrelevant_not_plausible",
  "adjudicated_hard_negative_note": "A short explanation of why you chose this final adjudication."
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

def main():
    parser = argparse.ArgumentParser(description="Auto-adjudicate full retrieval benchmark relabeling.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    args = parser.parse_args()

    file_basename = os.path.basename(args.input_file)
    
    if "gold_prior" in file_basename:
        prompt_template = PROMPT_ADJUDICATE_GOLD
        human_keys = ["adjudicated_gold_prior_decision", "adjudicated_gold_prior_note"]
    elif "hard_negative" in file_basename:
        prompt_template = PROMPT_ADJUDICATE_HARD_NEGATIVE
        human_keys = ["adjudicated_hard_negative_decision", "adjudicated_hard_negative_note"]
    else:
        raise ValueError(f"Could not determine task type from filename: {file_basename}. Must contain 'gold_prior' or 'hard_negative'.")

    model_slug = args.model.replace(".", "").replace("-", "")
    
    base, ext = os.path.splitext(args.input_file)
    output_file = f"{base}_resolved_{model_slug}.csv"
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    print(f"Loaded {len(rows)} adjudication rows from {args.input_file}")
    
    for k in human_keys:
        if k not in fieldnames:
            fieldnames.append(k)
    if 'adjudicator_id' not in fieldnames:
        fieldnames.append('adjudicator_id')

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"Processing with {args.model}...")
    
    for i, row in enumerate(rows):
        # We pass the entire row to the LLM so it can read the notes and decisions from Jason/Jiaxin
        # Exclude previously added adjudicated_ fields to avoid confusing the context
        input_data = {k: v for k, v in row.items() if k and not k.startswith("adjudicated_")}
        
        prompt = prompt_template.format(row_data=json.dumps(input_data, indent=2, ensure_ascii=False))
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_text = process_with_openai(prompt, args.model)
                result = json.loads(result_text)
                
                for k in human_keys:
                    row[k] = result.get(k, "")
                row['adjudicator_id'] = f"auto_{args.model}"
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error on row {i}: {e}")
                    row['adjudicator_id'] = f"auto_{args.model}_error"
                time.sleep(2)
                
        with open(output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            
        if (i + 1) % 20 == 0 or i == len(rows) - 1:
            print(f"Processed {i + 1}/{len(rows)}")
            
        time.sleep(0.5)
            
    print(f"Finished! Output fully saved to {output_file}")

if __name__ == "__main__":
    main()
