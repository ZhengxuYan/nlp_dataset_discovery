import json
from collections import Counter
from pathlib import Path

def analyze_metadata(file_path):
    print(f"Analyzing metadata distribution in {file_path}...\n")
    
    languages = Counter()
    domains = Counter()
    licenses = Counter()
    doc_types = Counter()
    has_url_count = 0
    total_datasets = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                datasets = data.get('datasets', [])
                for ds in datasets:
                    total_datasets += 1
                    
                    # Languages
                    langs = ds.get('languages', [])
                    if langs:
                        for l in langs:
                            languages[l.strip()] += 1
                    else:
                        languages['Unknown'] += 1
                        
                    # Domain
                    dom = ds.get('domain', 'Unknown')
                    domains[dom] += 1
                    
                    # Quality Signals
                    lic = ds.get('license', 'Unknown')
                    licenses[lic] += 1
                    
                    url = ds.get('availability_url', 'None')
                    if url and url.lower() not in ['none', 'unknown', 'nan']:
                        has_url_count += 1
                        
                    doc = ds.get('documentation_type', 'None')
                    doc_types[doc] += 1
                    
            except Exception as e:
                continue
                
    print(f"Total Datasets Analyzed: {total_datasets}\n")
    
    print("--- Top 20 Languages ---")
    for l, c in languages.most_common(20):
        print(f"{l}: {c} ({c/total_datasets*100:.1f}%)")
        
    print("\n--- Top 20 Domains ---")
    for d, c in domains.most_common(20):
        print(f"{d}: {c} ({c/total_datasets*100:.1f}%)")
        
    print("\n--- Top Licenses ---")
    for l, c in licenses.most_common(15):
        print(f"{l}: {c} ({c/total_datasets*100:.1f}%)")
        
    print("\n--- Availability ---")
    print(f"Has URL: {has_url_count} ({has_url_count/total_datasets*100:.1f}%)")
    
    print("\n--- Documentation Types ---")
    for d, c in doc_types.most_common(10):
        print(f"{d}: {c} ({c/total_datasets*100:.1f}%)")

if __name__ == "__main__":
    analyze_metadata("data/processed/scv_extraction_200_sampled_v3.jsonl")
