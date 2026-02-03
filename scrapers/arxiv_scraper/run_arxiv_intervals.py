import subprocess
import time
from datetime import datetime, timedelta
import calendar

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day).date()

def main():
    # Define start and end of the entire period
    overall_start = datetime(2023, 1, 1).date()
    overall_end = datetime(2025, 12, 31).date()
    
    current_start = overall_start
    
    while current_start <= overall_end:
        # Calculate end date for this interval (3 months later minus 1 day)
        # e.g., Jan 1 to Mar 31
        next_start = add_months(current_start, 3)
        current_end = next_start - timedelta(days=1)
        
        # Cap at overall end date
        if current_end > overall_end:
            current_end = overall_end
        
        print(f"Running scraper for interval: {current_start} to {current_end}")
        
        cmd = [
            "scrapy", "crawl", "arxiv",
            "-a", f"start_date={current_start}",
            "-a", f"end_date={current_end}"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed interval: {current_start} to {current_end}")
        except subprocess.CalledProcessError as e:
            print(f"Error running for interval {current_start} to {current_end}: {e}")
            # Optionally break or continue based on preference. 
            # We'll continue to try the next interval.
        
        # Determine strict next start (it is just next_start)
        current_start = next_start
        
        if current_start > overall_end:
            break
            
        print("Waiting 10 seconds before next run...")
        time.sleep(10)

if __name__ == "__main__":
    main()
