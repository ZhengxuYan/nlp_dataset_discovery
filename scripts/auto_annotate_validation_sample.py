import os
import json
import time
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROMPT_TEMPLATE = """You are an expert human annotator evaluating whether a natural language processing paper introduces a new dataset.
Your goal is to evaluate the provided paper title, abstract, and previous predictions, and determine if it is a dataset-introducing paper.

You are given:
- Title: The paper's title.
- Abstract: The paper's abstract (if available).
- Predicted Is Dataset Introducing: A previous model's prediction.
- Predicted Datasets: A previous model's extracted dataset names.
- Predicted Exclusion Reason: A previous model's reason for exclusion.

You must output a JSON object with EXACTLY these three keys:
1. "gold_is_dataset_introducing": boolean (true/false).
   - true if the paper introduces a new dataset.
   - false if the paper does NOT introduce a new dataset.
2. "gold_dataset_names": A list of strings containing the names of the datasets introduced (or an empty list [] if none or not introduced).
3. "notes": A string explaining your reasoning based on the criteria (or null if reasoning is trivial).

**Judging Criteria:**
Count as dataset-introducing (true) if the abstract contains signals like:
- "we introduce a benchmark/dataset/corpus"
- "we release a dataset"
- "we construct/curate/annotate/translate/generate a dataset"
- "new test set / evaluation set / training set"

Do NOT count as dataset-introducing (false) if:
- "we evaluate on existing datasets"
- "we use dataset X"
- "we present a model/tool/system"
- "we release code/prompts only"
- "survey / analysis only"

**Edge Cases:**
- New annotation over existing dataset: Count as true.
- Synthetic training data: If contributed as a resource/training set, count as true.
- Benchmark/test set: Count as true.
- Shared-task system paper: Usually false.
- Shared-task overview paper with dataset/test set: Count as true.
- Toolkit + existing datasets: Count as false.
- Toolkit + newly released annotated data: Count as true.

Input Data:
Title: {title}
Abstract: {abstract}
Predicted Is Dataset Introducing: {pred_is_dataset}
Predicted Datasets: {pred_datasets}
Predicted Exclusion Reason: {pred_exclusion}

Output ONLY a valid JSON object:
{{
    "gold_is_dataset_introducing": true,
    "gold_dataset_names": ["DatasetName1"],
    "notes": "Clear statement of introducing a new benchmark."
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
    parser = argparse.ArgumentParser(description="Auto-annotate dataset validation sample.")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", 
                        help="Model to use (e.g., gpt-4o-mini, gemini-3.1-flash-lite)")
    args = parser.parse_args()

    # Determine which API to use based on model name prefix
    if args.model.startswith("gpt"):
        process_func = process_with_openai
    elif args.model.startswith("gemini"):
        process_func = process_with_gemini
    else:
        print(f"Unknown model prefix for '{args.model}'. Defaulting to OpenAI API.")
        process_func = process_with_openai

    input_file = "data/census/acl_gemini_flashlite_all_clean_validation_sample_with_abstracts.jsonl"
    
    # Generate output file name based on model
    model_slug = args.model.replace(".", "").replace("-", "")
    output_file = f"data/census/acl_validation_sample_annotated_{model_slug}.jsonl"
    
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
        data = json.loads(line.strip())
        
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        
        prompt = PROMPT_TEMPLATE.format(
            title=title,
            abstract=abstract if abstract else "N/A",
            pred_is_dataset=data.get('pred_is_dataset_introducing', ''),
            pred_datasets=json.dumps(data.get('pred_datasets', [])),
            pred_exclusion=data.get('pred_exclusion_reason', '')
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_text = process_func(prompt, args.model)
                result = json.loads(result_text)
                
                # Fill in the gold fields
                data['gold_is_dataset_introducing'] = result.get('gold_is_dataset_introducing')
                data['gold_dataset_names'] = result.get('gold_dataset_names', [])
                data['notes'] = result.get('notes')
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error on line {i}: {e}")
                    # Keep original data structure but indicate failure
                    data['gold_is_dataset_introducing'] = None
                    data['gold_dataset_names'] = None
                    data['notes'] = f"Error processing: {e}"
                time.sleep(2)
                
        labeled_lines.append(data)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == len(lines) - 1:
            print(f"Processed {i + 1}/{len(lines)}")
            
        # Write intermediate results to avoid losing data
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in labeled_lines:
                f.write(json.dumps(item) + '\n')
        
        # Rate limit pause
        time.sleep(0.5 if args.model.startswith("gpt") else 1.0)
            
    print(f"Finished! Output fully saved to {output_file}")

if __name__ == "__main__":
    main()
