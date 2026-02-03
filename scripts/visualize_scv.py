import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Ensure output directory
OUTPUT_DIR = 'data/visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

def classify_affiliation(authors):
    # Simple heuristic
    industry_keywords = [
        "Google", "Facebook", "Meta", "Microsoft", "Amazon", "Apple", "IBM", 
        "OpenAI", "DeepMind", "Twitter", "Salesforce", "Adobe", "Intel", 
        "Nvidia", "Baidu", "Tencent", "Alibaba", "ByteDance", "Huawei", 
        "Samsung", "Naver", "Kakao", "LG", "Sony", "Uber", "Lyft", "Snap", 
        "Pinterest", "LinkedIn", "Netflix", "Spotify", "Oracle", "SAP", "Cisco",
        "Anthropic", "Cohere", "HuggingFace", "Allen Institute", "AI2" # AI2 is non-profit but often behaves like industry research labs in unmatched resources
    ]
    academic_keywords = [
        "University", "College", "Institute", "School", "Academy", "Polytechnic", 
        "MIT", "Caltech", "Stanford", "Harvard", "CMU", "ETH", "Inria", "MPI", 
        "CNRS", "Riken", "Kaist", "UC ", "Georgia Tech"
    ]
    
    affiliations = [a.get('affiliation', '') for a in authors]
    aff_text = " ".join(affiliations)
    
    is_industry = any(k.lower() in aff_text.lower() for k in industry_keywords)
    is_academic = any(k.lower() in aff_text.lower() for k in academic_keywords)
    
    if is_industry and is_academic:
        return "Mixed"
    elif is_industry:
        return "Industry"
    elif is_academic:
        return "Academia"
    else:
        return "Unknown"

def simplify_domain(domain):
    d = domain.lower()
    if 'bio' in d or 'med' in d or 'clinic' in d: return 'Biomedical'
    if 'social' in d or 'twitter' in d or 'reddit' in d: return 'Social Media'
    if 'finance' in d or 'econ' in d: return 'Finance'
    if 'legal' in d or 'law' in d: return 'Legal'
    if 'dialogue' in d or 'conversa' in d: return 'Dialogue'
    if 'vision' in d or 'image' in d or 'video' in d or 'multi' in d: return 'Multimodal'
    if 'sci' in d or 'scholarly' in d: return 'Scientific'
    return 'General/Other'

def visualize(input_file):
    print(f"Loading data from {input_file}...")
    records = load_data(input_file)
    print(f"Loaded {len(records)} records.")
    
    dataset_papers = []
    all_papers = []
    
    scv_points = []
    
    for r in records:
        # Basic Info
        date_str = r['metadata'].get('date')
        if not date_str: continue
        
        try:
            dt = pd.to_datetime(date_str)
        except:
            continue
            
        is_dataset = False
        introduced_datasets = []
        
        # Check if introduced datasets exist
        if 'datasets' in r:
            for d in r['datasets']:
                # The schema in final_scv is list of objects with 'info' and 'scv'
                info = d.get('info', {})
                if info.get('is_introduced'):
                    is_dataset = True
                    introduced_datasets.append(d)
        
        paper_info = {
            "id": r['arxiv_id'],
            "date": dt,
            "q": f"{dt.year}-Q{dt.quarter}",
            "affiliation": classify_affiliation(r['metadata'].get('authors', [])),
            "is_dataset": is_dataset
        }
        all_papers.append(paper_info)
        
        if is_dataset:
            dataset_papers.append(paper_info)
            # Collect SCV points
            # One paper might introduce multiple, let's take the best or all? All.
            for d in introduced_datasets:
                scv = d.get('scv', {})
                info = d.get('info', {})
                scv_points.append({
                    "id": r['arxiv_id'],
                    "novelty": scv.get('novelty', 0),
                    "diversity": scv.get('diversity', 0),
                    "quality": scv.get('quality', 0),
                    "embedding": r.get('paper_embedding', []),
                    "domain": simplify_domain(info.get('domain', 'Unknown')),
                    "affiliation": paper_info['affiliation']
                })

    df_papers = pd.DataFrame(all_papers)
    df_scv = pd.DataFrame(scv_points)
    
    # 1. Temporal Trend
    print("Generating Temporal Trend Plot...")
    if not df_papers.empty:
        trend = df_papers.groupby('q').agg(
            total=('id', 'count'),
            dataset_papers=('is_dataset', 'sum')
        ).sort_index()
        trend['fraction'] = trend['dataset_papers'] / trend['total']
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Bar for Total
        # sns.barplot(data=trend.reset_index(), x='q', y='total', ax=ax1, color='lightgray', alpha=0.5, label='Total Papers')
        # Actually easier to just plot Line for fraction as requested
        
        sns.lineplot(data=trend.reset_index(), x='q', y='fraction', ax=ax1, marker='o', linewidth=2, color='blue')
        ax1.set_ylabel('Fraction of Dataset Papers')
        ax1.set_xlabel('Quarter')
        ax1.set_title('Fraction of Papers Introducing New Datasets over Time')
        ax1.grid(True)
        
        plt.savefig(f"{OUTPUT_DIR}/temporal_trend.png")
        plt.close()
        
    # 2. Industry vs Academia
    print("Generating Affiliation Plot...")
    if not df_scv.empty:
        # Count unique papers or datasets? Let's count datasets (rows in df_scv)
        counts = df_scv['affiliation'].value_counts()
        
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        plt.title('Origin of New Datasets (Industry vs Academia)')
        plt.savefig(f"{OUTPUT_DIR}/affiliation_split.png")
        plt.close()
        
    # 3. SCV Distributions
    print("Generating SCV Distributions...")
    if not df_scv.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(df_scv['novelty'], bins=10, ax=axes[0], kde=True, color='purple')
        axes[0].set_title('Novelty Score Distribution')
        
        sns.histplot(df_scv['diversity'], bins=10, ax=axes[1], kde=True, color='orange')
        axes[1].set_title('Diversity Score Distribution')
        
        sns.histplot(df_scv['quality'], bins=10, ax=axes[2], kde=True, color='green')
        axes[2].set_title('Quality Score Distribution')
        
        plt.savefig(f"{OUTPUT_DIR}/scv_distributions.png")
        plt.close()
        
    # 4. Novelty vs Quality Scatter
    print("Generating Novelty vs Quality Scatter...")
    if not df_scv.empty:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df_scv, 
            x='novelty', 
            y='quality', 
            hue='affiliation', 
            size='diversity', 
            sizes=(20, 200),
            alpha=0.7,
            palette='deep'
        )
        plt.title('Dataset SCV: Novelty vs Quality')
        plt.axhline(0.5, linestyle='--', color='gray', alpha=0.5)
        plt.axvline(0.5, linestyle='--', color='gray', alpha=0.5)
        plt.savefig(f"{OUTPUT_DIR}/novelty_quality_scatter.png")
        plt.close()
        
    # 5. Embedding PCA
    print("Generating Embedding Space/PCA...")
    if not df_scv.empty:
        # Filter rows with valid embeddings
        valid_embs = [e for e in df_scv['embedding'] if isinstance(e, list) and len(e) > 0]
        if len(valid_embs) > 5:
            X = np.array(valid_embs)
            # Match filtered df
            mask = df_scv['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)
            df_pca = df_scv[mask].copy()
            
            pca = PCA(n_components=2)
            X_r = pca.fit_transform(X)
            
            df_pca['pc1'] = X_r[:, 0]
            df_pca['pc2'] = X_r[:, 1]
            
            plt.figure(figsize=(12, 10))
            sns.scatterplot(
                data=df_pca,
                x='pc1',
                y='pc2',
                hue='domain',
                style='affiliation',
                s=100,
                alpha=0.8
            )
            plt.title('Dataset Landscape (PCA of SPECTER Embeddings)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/landscape_pca.png")
            plt.close()

if __name__ == "__main__":
    visualize("data/processed/final_scv_200.jsonl")
