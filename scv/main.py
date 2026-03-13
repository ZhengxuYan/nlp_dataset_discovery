import argparse
import json
import os
from .pipeline import run_extraction_stage, run_analysis_stage
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Max papers to process in extraction.")
    parser.add_argument("--stage", type=str, choices=['extract', 'analyze'], required=True, help="Pipeline stage to run.")
    parser.add_argument("--input", type=str, default=None, help="Input file path.")
    parser.add_argument("--output", type=str, default=None, help="Output file path.")
    parser.add_argument("--batch-size", type=int, default=10, help="Concurrent LLM requests per extraction batch.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for each failed extraction batch.")
    parser.add_argument("--pdf-workers", type=int, default=4, help="Parallel workers for PDF download/text prep.")
    parser.add_argument("--text-char-limit", type=int, default=16000, help="Maximum paper text characters sent to the extractor.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="LLM model name for extraction.")
    parser.add_argument("--backend", type=str, default=None, help="Curator backend, for example `litellm`.")
    parser.add_argument("--backend-params", type=str, default=None, help="JSON string for backend parameters.")
    args = parser.parse_args()

    backend = args.backend if args.backend else os.environ.get("BACKEND")
    backend_params = None
    raw_backend_params = args.backend_params if args.backend_params else os.environ.get("BACKEND_PARAMS")
    if raw_backend_params:
        try:
            backend_params = json.loads(raw_backend_params)
        except json.JSONDecodeError:
            raise SystemExit("Error parsing backend params JSON.")
    
    if args.stage == 'extract':
        out_file = args.output if args.output else 'data/processed/scv_intermediate.jsonl'
        run_extraction_stage(
            limit=args.limit,
            output_file=out_file,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            pdf_workers=args.pdf_workers,
            text_char_limit=args.text_char_limit,
            model_name=args.model,
            backend=backend,
            backend_params=backend_params
        )
    elif args.stage == 'analyze':
        in_file = args.input if args.input else 'data/processed/scv_intermediate.jsonl'
        out_file = args.output if args.output else 'data/processed/scv_final_results.jsonl'
        run_analysis_stage(
            input_file=in_file,
            output_file=out_file,
            novelty_model_name=args.model,
            novelty_backend=backend,
            novelty_backend_params=backend_params
        )

if __name__ == "__main__":
    main()
