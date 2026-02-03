import json
import sys
from typing import Dict, Any, Counter

def load_jsonl(filepath: str) -> Dict[str, Any]:
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                data[item['arxiv_id']] = item
            except json.JSONDecodeError:
                pass
    return data

from difflib import SequenceMatcher

def is_similar(name1: str, name2: str) -> bool:
    n1 = name1.lower()
    n2 = name2.lower()
    
    # Exact match after lowercasing
    if n1 == n2:
        return True
    
    # Substring match (if one is a significant part of the other)
    if len(n1) > 4 and n1 in n2:
        return True
    if len(n2) > 4 and n2 in n1:
        return True
        
    # Similarity ratio
    ratio = SequenceMatcher(None, n1, n2).ratio()
    if ratio > 0.8:
        return True
        
    return False

def compare_datasets(id: str, d1: Dict[str, Any], d2: Dict[str, Any], name1: str, name2: str, stats: Counter, global_stats: Dict[str, int]):
    # print(f"--- Paper: {id} ---") # Reduce verbosity for stats view
    datasets1 = d1.get('datasets', {})
    datasets2 = d2.get('datasets', {})
    
    keys1 = set(datasets1.keys())
    keys2 = set(datasets2.keys())
    
    common = keys1 & keys2
    only_in_1 = list(keys1 - keys2)
    only_in_2 = list(keys2 - keys1)
    
    # Fuzzy matching to find naming differences
    fuzzy_matches = []
    
    # Iterate through a copy to allow modification or just track indices
    # We'll try to match each item in only_in_1 to something in only_in_2
    
    matched_in_2 = set()
    matched_in_1 = set()
    
    for k1 in only_in_1:
        best_match = None
        best_score = 0
        
        for k2 in only_in_2:
            if k2 in matched_in_2:
                continue
            
            if is_similar(k1, k2):
                # If multiple matches, take the first one (simplification)
                matched_in_1.add(k1)
                matched_in_2.add(k2)
                fuzzy_matches.append((k1, k2))
                break
    
    # Calculate True Misses
    true_unique_1 = [k for k in only_in_1 if k not in matched_in_1]
    true_unique_2 = [k for k in only_in_2 if k not in matched_in_2]
    
    # Update global stats
    global_stats['total_datasets_1'] += len(keys1)
    global_stats['total_datasets_2'] += len(keys2)
    
    global_stats['unique_to_1'] += len(true_unique_1)
    global_stats['unique_to_2'] += len(true_unique_2)
    global_stats['naming_differences'] += len(fuzzy_matches)
    
    has_difference = False
    
    if true_unique_1 or true_unique_2 or fuzzy_matches:
        has_difference = True
        # Uncomment print statements for unique datasets to inspect them
        if true_unique_1:
            print(f"Only in {name1}: {', '.join(true_unique_1)}")
        if true_unique_2:
            print(f"Only in {name2}: {', '.join(true_unique_2)}")
        if fuzzy_matches:
            print(f"Naming differences: {', '.join([f'{m1} vs {m2}' for m1, m2 in fuzzy_matches])}")
        
    for ds_name in common:
        ds1 = datasets1[ds_name]
        ds2 = datasets2[ds_name]
        
        # Compare specific fields
        fields = ['role', 'confidence', 'transformation_type', 'source_dataset', 'is_introduced']
        for field in fields:
            val1 = ds1.get(field)
            val2 = ds2.get(field)
            # Normalize None/Strings for comparison
            if str(val1).lower() != str(val2).lower():
                stats[field] += 1
                has_difference = True

    if has_difference:
        global_stats['papers_with_differences'] += 1

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare dataset extraction results between two models.")
    parser.add_argument("--file1", default="data/processed/arxiv_nlp_conf_papers_2023_2025_dataset_analysis(qwen-235b).jsonl", help="Path to first JSONL file")
    parser.add_argument("--file2", default="data/processed/arxiv_nlp_conf_papers_2023_2025_dataset_analysis(gpt-5-mini).jsonl", help="Path to second JSONL file")
    parser.add_argument("--name1", default="Qwen-235B", help="Display name for first model")
    parser.add_argument("--name2", default="GPT-5-mini", help="Display name for second model")
    
    args = parser.parse_args()

    print(f"Loading {args.name1} from {args.file1}...")
    data1 = load_jsonl(args.file1)
    print(f"Loading {args.name2} from {args.file2}...")
    data2 = load_jsonl(args.file2)
    
    common_ids = set(data1.keys()) & set(data2.keys())
    print(f"Found {len(common_ids)} common papers.")
    
    stats = Counter()
    global_stats = {
        'total_datasets_1': 0,
        'total_datasets_2': 0,
        'unique_to_1': 0, # Missed by 2
        'unique_to_2': 0, # Missed by 1
        'naming_differences': 0,
        'papers_with_differences': 0
    }
    
    for arxiv_id in common_ids:
        compare_datasets(arxiv_id, data1[arxiv_id], data2[arxiv_id], args.name1, args.name2, stats, global_stats)

    print("\n--- Missed Datasets & Row Differences ---")
    print(f"Total Papers Compared: {len(common_ids)}")
    print(f"Papers with Differences: {global_stats['papers_with_differences']} ({(global_stats['papers_with_differences']/len(common_ids))*100:.1f}%)")
    print("-" * 40)
    print(f"Total Datasets Found by {args.name1}: {global_stats['total_datasets_1']}")
    print(f"Total Datasets Found by {args.name2}:  {global_stats['total_datasets_2']}")
    print("-" * 40)
    print(f"Datasets Matched (Exact Name): {global_stats['total_datasets_1'] - global_stats['unique_to_1'] - global_stats['naming_differences']}")
    print(f"Datasets Matched (Naming Difference): {global_stats['naming_differences']}")
    print("-" * 40)
    print(f"True Misses by {args.name1} (Unique to {args.name2}): {global_stats['unique_to_2']}")
    print(f"True Misses by {args.name2} (Unique to {args.name1}): {global_stats['unique_to_1']}")
    
    print("\n--- Field Differences (for common datasets) ---")
    for field, count in stats.most_common():
        print(f"{field:<25} | {count:<10}")

if __name__ == "__main__":
    main()
