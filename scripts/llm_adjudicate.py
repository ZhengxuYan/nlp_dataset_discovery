import os
import csv
import json
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

PROMPT_ADJUDICATE = """You are an expert NLP Senior Adjudicator. 
Your task is to resolve a disagreement between two human annotators (or an annotator and a model) regarding a dataset-contribution validation task.

You are given a CSV row representing the disagreement. It includes the audit type, query details, the original model label, the specific fields they disagreed on, and notes from annotators 'Jason' and 'Jiaxin'.

Rules for Adjudication (Based on our Codebook):
1. For Retrieval Benchmark: 
   - A 'gold prior' is 'covered' or 'partially_covered' ONLY if it provides evidence for the SAME contribution dimension. If it's just topically similar, it is 'not_evidence'.
   - A 'hard negative' is 'valid_hard_negative' if plausible but does NOT cover the query. If it actually covers the claim, it's 'actually_evidence'. If completely irrelevant, 'irrelevant_not_plausible'.
2. For Attribution Coverage:
   - If the prior evidence is a useful comparator for the SAME contribution dimension, lean towards 'partially_covered'. Pure topical similarity is 'not_covered'.
3. For Evidence Adequacy / Risk:
   - 'adequacy_correct=yes' means retrieved evidence is sufficient to explain the current model label.
   - 'missing_prior_risk_correct=yes' means the model's internal risk estimate was reasonable.
   - 'false_not_covered=yes' ONLY if found prior actually flips 'not_covered' to 'covered'/'partially_covered'.
4. For Metadata Extraction:
   - Focus on standardizing metadata errors into a semicolon-separated list of exact field names.

Your job:
Adjudicate the row. Output the final correct label/validity based on the strict rules above.

Input Disagreement Row:
{row_data}

Output JSON with these exact keys:
{{
  "adjudicated_final_label": "The final coverage label (covered/partially_covered/not_covered/contradicted/not_comparable) OR blank if not an attribution row",
  "adjudicated_validity": "The final validity status for retrieval rows (covered/not_evidence/valid_hard_negative/actually_evidence etc.) OR blank if not retrieval",
  "adjudicated_notes": "A brief explanation of why you chose this final adjudication, referencing the annotators' notes if applicable"
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
    parser = argparse.ArgumentParser(description="Auto-adjudicate disagreements using LLMs.")
    parser.add_argument("--input_file", type=str, default="data/human_validation/human_validation_adjudication_queue.csv", help="Path to the adjudication CSV file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use (e.g., gpt-4o-mini, gemini-3.1-flash-lite)")
    args = parser.parse_args()

    if args.model.startswith("gpt"):
        process_func = process_with_openai
    elif args.model.startswith("gemini"):
        process_func = process_with_gemini
    else:
        print(f"Unknown model prefix for '{args.model}'. Defaulting to OpenAI API.")
        process_func = process_with_openai

    model_slug = args.model.replace(".", "").replace("-", "")
    
    base, ext = os.path.splitext(args.input_file)
    output_file = f"{base}_resolved_{model_slug}.csv"
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    print(f"Loaded {len(rows)} adjudication rows from {args.input_file}")
    
    human_keys = ["adjudicated_final_label", "adjudicated_validity", "adjudicated_notes", "adjudicator_id"]
    for k in human_keys:
        if k not in fieldnames:
            fieldnames.append(k)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"Processing with {args.model}...")
    
    for i, row in enumerate(rows):
        # We pass the entire row to the LLM so it can read the notes from Jason/Jiaxin
        input_data = {k: v for k, v in row.items() if k and not k.startswith("adjudicated_")}
        
        prompt = PROMPT_ADJUDICATE.format(row_data=json.dumps(input_data, indent=2, ensure_ascii=False))
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_text = process_func(prompt, args.model)
                result = json.loads(result_text)
                
                row["adjudicated_final_label"] = result.get("adjudicated_final_label", "")
                row["adjudicated_validity"] = result.get("adjudicated_validity", "")
                row["adjudicated_notes"] = result.get("adjudicated_notes", "")
                row['adjudicator_id'] = f"auto_{args.model}"
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error on row {i}: {e}")
                    row['adjudicator_id'] = f"auto_{args.model}_error"
                time.sleep(2)
                
        # Append progressively
        with open(output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"Processed {i + 1}/{len(rows)}")
            
        time.sleep(0.5 if args.model.startswith("gpt") else 1.0)
            
    print(f"Finished! Output fully saved to {output_file}")

if __name__ == "__main__":
    main()
