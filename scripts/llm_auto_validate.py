import os
import csv
import json
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

PROMPT_ATTRIBUTION = """You are an expert human annotator reviewing dataset-contribution attribution predictions for NLP papers.
Your goal is to validate if the model correctly judged whether a 'prior evidence' DCU (Dataset Contribution Unit) covers the 'query claim'.

Rules:
- Coverage Labels: 'covered', 'partially_covered', 'not_covered', 'contradicted', 'not_comparable'.
- Do not require the prior DCU to mention the query dataset name exactly.
- Topical similarity alone is not enough; must be useful for the same dataset-contribution dimension.
- If the prior evidence is a useful comparator for the SAME contribution dimension, lean towards 'partially_covered'. Pure topical similarity is 'not_covered'.

Task: Evaluate the model's judgment using ONLY the shown evidence.

Input Row Data:
{row_data}

Output JSON with these exact keys:
{{
  "human_coverage_label_correct": "yes" | "no" | "uncertain",
  "human_corrected_coverage_label": "covered" | "partially_covered" | "not_covered" | "contradicted" | "not_comparable" | "" (blank if model is correct),
  "human_selected_evidence_relevance": "direct" | "partial" | "none" | "na" (Use 'na' ONLY if no selected prior evidence is present. Use 'none' if present but irrelevant. Use 'partial' if same dimension but incomplete. Use 'direct' if directly covers the same claim),
  "human_rationale_groundedness": "yes" | "partial" | "no",
  "human_evidence_sufficient": "yes" | "no" | "uncertain",
  "human_missing_prior_risk": "high" | "low" | "uncertain",
  "human_notes": "Your reasoning or blank"
}}
"""

PROMPT_EXTRACTION = """You are an expert human annotator validating NLP dataset/benchmark extraction records.
Your goal is to check whether the dataset claims, role, type, and metadata are correct and grounded in the evidence span.

Task: Evaluate the extraction record using ONLY the shown evidence.

Input Row Data:
{row_data}

Output JSON with these exact keys:
{{
  "human_record_valid": "yes" | "no" | "uncertain",
  "human_role_correct": "yes" | "no" | "uncertain",
  "human_dcu_grounded": "yes" | "no" | "uncertain",
  "human_dcu_type_correct": "yes" | "no" | "uncertain",
  "human_metadata_errors": "Format as a semicolon-separated list of exact field names (e.g., 'license; scale; source_data_origin') or blank if no errors. Do not use natural language.",
  "human_notes": "Your reasoning or blank"
}}
"""

PROMPT_RETRIEVAL = """You are an expert human annotator reviewing retrieval benchmark label validity.
Your goal is to evaluate if the labeled 'gold prior', 'hard negative', or 'empty gold' cases are genuinely correct for evaluating retrieval systems.

Rules:
- A 'gold prior' must be marked 'covered' or 'partially_covered' ONLY if it provides evidence for the SAME contribution dimension. If it's just topically similar, it is 'not_evidence'.
- A 'hard negative' is ONLY 'valid_hard_negative' if it is plausible/topical BUT does NOT cover the query DCU. If completely irrelevant, it's 'irrelevant_not_plausible'. If it actually covers the claim, it's 'actually_evidence'.

Task: Evaluate the validity using ONLY the shown evidence.

Input Row Data:
{row_data}

Output JSON with these exact keys (leave irrelevant fields blank based on the case type):
{{
  "human_gold_prior_validity": "covered" | "partially_covered" | "not_evidence" | "uncertain" | "",
  "human_hard_negative_validity": "valid_hard_negative" | "actually_evidence" | "irrelevant_not_plausible" | "uncertain" | "",
  "human_empty_gold_validity": "valid_empty" | "missed_cited_prior" | "uncertain" | "",
  "human_corrected_label": "Corrected label if needed or blank",
  "human_notes": "Your reasoning or blank"
}}
"""

PROMPT_EXTERNAL = """You are an expert human annotator performing a missing-prior external audit for dataset contributions.
Your goal is to determine if there are significant prior datasets/claims that should have been covered, but the model missed.

Rules:
- 'human_adequacy_correct'=yes means the retrieved evidence is sufficient to explain the current model label.
- 'human_missing_prior_risk_correct'=yes means the model's internal estimate of risk was reasonable given the context.
- 'human_final_false_not_covered'=yes ONLY if a found prior actually flips the status from 'not_covered' to 'covered' or 'partially_covered'.

Task: Use your parametric memory and broad knowledge of NLP datasets (simulating a lightweight external search) to check if the query dataset claim is actually covered by prior work not shown in the immediate evidence. 

Input Row Data:
{row_data}

Output JSON with these exact keys:
{{
  "human_external_search_done": "yes",
  "human_found_missed_prior": "yes" | "no",
  "human_missed_prior_title_or_url": "Name/URL of missed prior or blank",
  "human_missed_prior_evidence": "Short description of the prior evidence or blank",
  "human_final_false_not_covered": "yes" | "no",
  "human_adequacy_correct": "yes" | "no" | "uncertain",
  "human_missing_prior_risk_correct": "yes" | "no" | "uncertain",
  "human_notes": "Your reasoning or blank"
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

def get_prompt_and_keys(audit_type):
    if "extraction" in audit_type or "dcu" in audit_type:
        return PROMPT_EXTRACTION, ["human_record_valid", "human_role_correct", "human_dcu_grounded", "human_dcu_type_correct", "human_metadata_errors", "human_notes"]
    elif "retrieval" in audit_type or "benchmark" in audit_type:
        return PROMPT_RETRIEVAL, ["human_gold_prior_validity", "human_hard_negative_validity", "human_empty_gold_validity", "human_corrected_label", "human_notes"]
    elif "external" in audit_type or "missing_prior_external" in audit_type:
        return PROMPT_EXTERNAL, ["human_external_search_done", "human_found_missed_prior", "human_missed_prior_title_or_url", "human_missed_prior_evidence", "human_final_false_not_covered", "human_adequacy_correct", "human_missing_prior_risk_correct", "human_notes"]
    else:
        # Default to attribution audit
        return PROMPT_ATTRIBUTION, ["human_coverage_label_correct", "human_corrected_coverage_label", "human_selected_evidence_relevance", "human_rationale_groundedness", "human_evidence_sufficient", "human_missing_prior_risk", "human_notes"]

def main():
    parser = argparse.ArgumentParser(description="Auto-annotate CSV files for human validation using LLMs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
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
    
    file_basename = os.path.basename(args.input_file)
    if "missing_prior_external" in file_basename or "missing_prior" in file_basename:
        forced_audit_type = "external"
    elif "retrieval" in file_basename or "benchmark" in file_basename:
        forced_audit_type = "retrieval"
    elif "extraction" in file_basename or "dcu" in file_basename:
        forced_audit_type = "extraction"
    else:
        forced_audit_type = "attribution"
        
    prompt_template, human_keys = get_prompt_and_keys(forced_audit_type)
    
    base, ext = os.path.splitext(args.input_file)
    output_file = f"{base}_labeled_{model_slug}.csv"
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    print(f"Loaded {len(rows)} rows from {args.input_file}")
    
    # Ensure all required output keys are in fieldnames to avoid DictWriter ValueError
    for k in human_keys:
        if k not in fieldnames:
            fieldnames.append(k)
    if 'annotator_id' not in fieldnames:
        fieldnames.append('annotator_id')

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"Processing with {args.model}...")
    
    for i, row in enumerate(rows):
        
        # Prepare context by filtering out existing human_ columns and annotator_id to save context
        input_data = {k: v for k, v in row.items() if k and not k.startswith("human_") and k != "annotator_id"}
        
        prompt = prompt_template.format(row_data=json.dumps(input_data, indent=2, ensure_ascii=False))
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_text = process_func(prompt, args.model)
                result = json.loads(result_text)
                
                # Apply updates
                for k in human_keys:
                    row[k] = result.get(k, "")
                row['annotator_id'] = args.model
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error on row {i}: {e}")
                    row['annotator_id'] = f"{args.model}_error"
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
