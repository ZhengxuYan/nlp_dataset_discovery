import subprocess
import time
import argparse
from datetime import datetime, timedelta
import calendar
from pathlib import Path
from typing import Sequence

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day).date()

def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Run the arXiv scraper over date intervals.")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--interval-months", type=int, default=3)
    parser.add_argument("--interval-days", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-file")
    parser.add_argument("--categories", help="Comma-separated arXiv categories, for example cs.CL,cs.LG.")
    parser.add_argument("--stop-on-error", action="store_true", default=True)
    parser.add_argument("--keep-going", action="store_false", dest="stop_on_error")
    args = parser.parse_args(argv)
    project_dir = Path(__file__).resolve().parent

    overall_start = parse_date(args.start_date)
    overall_end = parse_date(args.end_date)
    if overall_end < overall_start:
        raise SystemExit("--end-date must be on or after --start-date")
    if args.interval_days is not None and args.interval_days <= 0:
        raise SystemExit("--interval-days must be positive")
    if args.interval_months <= 0:
        raise SystemExit("--interval-months must be positive")
    
    current_start = overall_start
    
    while current_start <= overall_end:
        if args.interval_days is not None:
            next_start = current_start + timedelta(days=args.interval_days)
        else:
            # Calculate end date for this interval (3 months later minus 1 day)
            # e.g., Jan 1 to Mar 31
            next_start = add_months(current_start, args.interval_months)
        current_end = next_start - timedelta(days=1)
        
        # Cap at overall end date
        if current_end > overall_end:
            current_end = overall_end
        
        print(f"Running scraper for interval: {current_start} to {current_end}")
        
        cmd = [
            "scrapy", "crawl", "arxiv",
            "-a", f"start_date={current_start}",
            "-a", f"end_date={current_end}",
        ]
        if args.output_file:
            cmd.extend(["-a", f"output_file={args.output_file}"])
        if args.categories:
            cmd.extend(["-a", f"categories={args.categories}"])
        
        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
        else:
            try:
                subprocess.run(cmd, cwd=project_dir, check=True)
                print(f"Completed interval: {current_start} to {current_end}")
            except subprocess.CalledProcessError as e:
                print(f"Error running for interval {current_start} to {current_end}: {e}")
                if args.stop_on_error:
                    raise SystemExit(e.returncode)
        
        # Determine strict next start (it is just next_start)
        current_start = next_start
        
        if current_start > overall_end:
            break
            
        if not args.dry_run and args.sleep_seconds > 0:
            print(f"Waiting {args.sleep_seconds:g} seconds before next run...")
            time.sleep(args.sleep_seconds)

if __name__ == "__main__":
    main()
