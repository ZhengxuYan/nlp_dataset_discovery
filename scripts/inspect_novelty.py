import json

def inspect_novelty():
    low_novelty_count = 0
    total_count = 0
    
    print("--- Datasets with Novelty < 1.0 ---")
    with open('data/processed/final_scv_200.jsonl', 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)
                datasets = rec.get('datasets', [])
                for d in datasets:
                    total_count += 1
                    scv = d.get('scv', {})
                    nov = scv.get('novelty', 1.0)
                    
                    if nov < 0.99: # Allow for float point weirdness, check strictly non-1
                        low_novelty_count += 1
                        info = d.get('info', {})
                        print(f"Paper: {rec.get('arxiv_id')}")
                        print(f"Dataset: {info.get('name')}")
                        print(f"Use: {info.get('usage_description')[:100]}...")
                        print(f"Role: {info.get('role')}")
                        print(f"Score: {nov:.3f}")
                        print("-" * 30)
            except:
                pass
                
    print(f"\nTotal Datasets: {total_count}")
    print(f"Low Novelty Count: {low_novelty_count}")
    print(f"High Novelty (1.0) Count: {total_count - low_novelty_count}")

if __name__ == "__main__":
    inspect_novelty()
