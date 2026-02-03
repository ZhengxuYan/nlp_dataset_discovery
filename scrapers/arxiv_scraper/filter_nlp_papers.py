import csv
import os
import re

def filter_nlp_papers():
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    input_file = os.path.join(project_root, "data", "raw", "arxiv_results_2023_2025.csv")
    output_file = os.path.join(project_root, "data", "processed", "arxiv_nlp_conf_papers_2023_2025.csv")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # List of NLP venues/keywords based on ACL Anthology
    # Using regex compatible strings or just checking substring
    keywords = [
        "AACL", "ACL", "ANLP", "ArabicNLP", 
        "Computational Linguistics", "CoNLL", "EACL", "EMNLP", 
        "Findings", "IWSLT", "NAACL", "SemEval", r"\*SEM", "StarSEM", "TACL", "WMT", 
        "AIME-Con", "ALTA", "AMTA", "CCL", "CLiC-it", "COLING", "EAMT", 
        "HLT", "IJCLCLP", "IJCNLP", "IWSDS", "JEP", "TALN", "RECITAL", 
        "JLCL", "KONVENS", "LILT", "LREC", "MTSummit", "MUC", "NEJLT", "NoDaLiDa", 
        "PACLIC", "RANLP", "ROCLING", "TINLAP", "TIPSTER"
    ]
    
    # Short acronyms like "CL", "WS", "TAL" can be noisy if unchecked. 
    # We'll be slightly more strict with them or rely on the user's list implies they want them.
    # But "CL" appears in categories "cs.CL". We are searching Comment and Journal Ref.
    # "CL" as a journal is usually "Computational Linguistics" or "J. Comput. Lang."
    # We will add them but maybe require word boundaries for short ones if using regex.
    
    print(f"Reading from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    # Compile regex for better matching (word boundaries for short acronyms)
    # We transform keywords into a regex pattern
    # Escape special chars like * in *SEM
    
    # Logic: If any keyword is found in Comment or Journal Reference
    
    count_processed = 0
    count_kept = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f_in, \
         open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            print("Empty input file.")
            return
            
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            count_processed += 1
            
            comment = row.get("Comment", "").lower()
            journal_ref = row.get("Journal Reference", "").lower()
            
            # Combine text to search
            text_to_check = f"{comment} {journal_ref}"
            
            found = False
            for kw in keywords:
                # Simple substring check (case-insensitive done by lower())
                # Handle regex-like kw for *SEM if needed, but simple replacing * might be easier
                # For this list, we'll just check substring presence for most
                
                kw_clean = kw.lower().replace(r"\*", "*")
                
                # For very short keywords, this might be noisy (e.g. "ws" inside "news")
                # But "WS" wasn't explicitly in my safe list above, I used "Workshop".
                # "ACL" is safe enough. "CL" is risky ("clean").
                # User asked for "CL", "WS", "TAL". 
                # Let's use regex word boundary for short ones (len < 4)
                
                if len(kw_clean) < 4:
                    if re.search(r'\b' + re.escape(kw_clean) + r'\b', text_to_check):
                        found = True
                        break
                else:
                    if kw_clean in text_to_check:
                        found = True
                        break
            
            if found:
                writer.writerow(row)
                count_kept += 1
                
    print(f"Finished.")
    print(f"Processed {count_processed} papers.")
    print(f"Kept {count_kept} papers matching NLP venues.")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    filter_nlp_papers()
