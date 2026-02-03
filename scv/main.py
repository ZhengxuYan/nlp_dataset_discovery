import argparse
from .pipeline import run_extraction_stage, run_analysis_stage
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="Max papers to process in extraction.")
    parser.add_argument("--stage", type=str, choices=['extract', 'analyze'], required=True, help="Pipeline stage to run.")
    parser.add_argument("--input", type=str, default=None, help="Input file path.")
    parser.add_argument("--output", type=str, default=None, help="Output file path.")
    args = parser.parse_args()
    
    if args.stage == 'extract':
        out_file = args.output if args.output else 'data/processed/scv_intermediate.jsonl'
        run_extraction_stage(limit=args.limit, output_file=out_file)
    elif args.stage == 'analyze':
        in_file = args.input if args.input else 'data/processed/scv_intermediate.jsonl'
        out_file = args.output if args.output else 'data/processed/scv_final_results.jsonl'
        run_analysis_stage(input_file=in_file, output_file=out_file)

if __name__ == "__main__":
    main()
