import os
import csv
import json
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

PROMPT_RESOLVE_CONFLICT = """You are an expert NLP Senior Adjudicator. 
Your task is to resolve a critical conflict in a dataset-contribution retrieval benchmark. 

The conflict: A specific prior DCU was judged as 'actually_evidence' during the hard-negative review, but was simultaneously judged as 'not_evidence' during the gold-prior review. These contradict each other.

Task: Carefully read the query DCU, the prior DCU, and the previous notes/judgments from both sides. Determine the absolute final truth: Is this prior DCU actually valid evidence for the query DCU?

Rules for 'resolution_decision':
1. 'covered': The prior DCU expresses basically the same dataset contribution.
2. 'partially_covered': The prior DCU has real overlap but differs in at least one important dimension (task, domain, language, modality, source, etc).
3. 'not_evidence': The prior DCU does NOT cover the query DCU. It might just be topically similar, a generic statement, or unrelated. (If you choose this, it means the gold review was right, and the hard-negative review was wrong to call it 'actually_evidence').

Input Conflict Row Data:
{row_data}

Output JSON with EXACTLY these keys:
{{
  "resolution_decision": "covered" | "partially_covered" | "not_evidence",
  "resolution_note": "A short, definitive explanation of why you chose this final resolution, addressing why one side of the conflict was wrong."
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
    parser = argparse.ArgumentParser(description="Auto-resolve gold vs hard-negative conflicts.")
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

    model_slug = args.model.replace(".", "").replace("-", "")
    
    base, ext = os.path.splitext(args.input_file)
    output_file = f"{base}_resolved_{model_slug}.csv"
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    print(f"Loaded {len(rows)} conflict rows from {args.input_file}")
    
    human_keys = ["resolution_decision", "resolution_note", "resolver_id"]
    for k in human_keys:
        if k not in fieldnames:
            fieldnames.append(k)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"Processing with {args.model}...")
    
    for i, row in enumerate(rows):
        # Exclude previously added resolution_ fields to avoid confusing the context on reruns
        input_data = {k: v for k, v in row.items() if k and not k.startswith("resolution_") and k != "resolver_id"}
        
        prompt = PROMPT_RESOLVE_CONFLICT.format(row_data=json.dumps(input_data, indent=2, ensure_ascii=False))
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_text = process_func(prompt, args.model)
                result = json.loads(result_text)
                
                row["resolution_decision"] = result.get("resolution_decision", "")
                row["resolution_note"] = result.get("resolution_note", "")
                row['resolver_id'] = f"auto_{args.model}"
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error on row {i}: {e}")
                    row['resolver_id'] = f"auto_{args.model}_error"
                time.sleep(2)
                
        with open(output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"Processed {i + 1}/{len(rows)}")
            
        time.sleep(0.5 if args.model.startswith("gpt") else 1.0)
            
    print(f"Finished! Output fully saved to {output_file}")

if __name__ == "__main__":
    main()
