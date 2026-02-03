import json
import numpy as np
from pathlib import Path

def analyze_distributions(file_path):
    print(f"Analyzing {file_path}...")
    
    samples = []
    classes = []
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                datasets = data.get('datasets', [])
                for ds in datasets:
                    size = ds.get('size', {})
                    
                    # Num Samples
                    ns = size.get('num_samples')
                    if ns is not None and isinstance(ns, (int, float)) and ns > 0:
                        samples.append(ns)
                        
                    # Num Classes
                    nc = size.get('num_classes')
                    if nc is not None and isinstance(nc, (int, float)) and nc > 0:
                        classes.append(nc)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
                
    print(f"\nFound {len(samples)} valid num_samples entries.")
    print(f"Found {len(classes)} valid num_classes entries.")
    
    def print_stats(name, data):
        if not data:
            print(f"\nNo valid data for {name}.")
            return
            
        data = np.array(data)
        print(f"\n--- {name} Distribution ---")
        print(f"Min: {np.min(data)}")
        print(f"Max: {np.max(data)}")
        print(f"Mean: {np.mean(data):.2f}")
        print(f"Median (50%): {np.median(data)}")
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(data, percentiles)
        
        for p, v in zip(percentiles, pct_values):
            print(f"{p}th percentile: {v:.0f}")

    print_stats("Num Samples", samples)
    print_stats("Num Classes", classes)

if __name__ == "__main__":
    analyze_distributions("data/processed/scv_extraction_200_sampled_v3.jsonl")
