#!/usr/bin/env python3
"""
Parallel UTMB Score Fetcher

Runs multiple step 2 processes in parallel to speed up UTMB score fetching.
Each worker handles one race file, with live progress bars for each.

USAGE:
------
# Process all files in parallel (4 parallel workers by default)
python3 parallel_step2.py --cookie "YOUR_COOKIE"

# Process specific tenant/year with 6 parallel workers
python3 parallel_step2.py --cookie "YOUR_COOKIE" --tenant kullamannen --year 2025 --workers 6

# Skip files that already have scores
python3 parallel_step2.py --cookie "YOUR_COOKIE" --skip-existing

# Dry run - show what would be processed
python3 parallel_step2.py --cookie "YOUR_COOKIE" --dry-run

TIPS:
-----
- Use 4-8 workers for best performance (more may hit rate limits)
- Cookie expires after ~30 min, so start with fewer files first
- Use --skip-existing to resume interrupted runs
"""

import argparse
import sys
import time
import threading
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install 'tqdm' for progress bars: pip install tqdm")

# Import from utmb_scraper
from utmb_scraper import (
    get_data_dir, 
    read_csv_with_metadata,
    write_csv_with_metadata,
    csv_has_utmb_scores,
    extract_access_token,
    fetch_runner_race_history,
    get_cached_race_metadata,
    UTMB_API_HEADERS,
)


class RaceProcessor:
    """Process a single race file to fetch UTMB scores."""
    
    def __init__(self, csv_file, tenant, year, race_id, access_token, position=0):
        self.csv_file = csv_file
        self.tenant = tenant
        self.year = year
        self.race_id = race_id
        self.access_token = access_token
        self.position = position
        self.scores_found = 0
        self.total_runners = 0
        self.status = "pending"
        self.error = None
        
    def process(self):
        """Process the race and return (success, message)."""
        try:
            # Load CSV
            df, metadata = read_csv_with_metadata(self.csv_file)
            self.total_runners = len(df)
            
            # Get runners with profiles
            runners_with_profiles = df[df['utmb_profile_url'].notna()]
            
            if len(runners_with_profiles) == 0:
                self.status = "no_profiles"
                return (True, "No runners with UTMB profiles")
            
            # Get race name pattern for matching (case-insensitive)
            cached = get_cached_race_metadata(self.tenant, self.year, self.race_id)
            if cached:
                race_name_pattern = cached.get('race_name', '').split(' - ')[-1] if cached.get('race_name') else self.race_id
            else:
                race_name_pattern = self.race_id
            race_name_pattern_lower = race_name_pattern.lower()
            
            # Create thread-local session
            session = requests.Session()
            session.headers.update(UTMB_API_HEADERS)
            
            # Fetch scores
            scores = {}
            runners_to_fetch = [
                (idx, row['utmb_profile_url'])
                for idx, row in runners_with_profiles.iterrows()
            ]
            
            self.status = "processing"
            desc = f"{self.tenant}/{self.year}/{self.race_id}"
            
            if TQDM_AVAILABLE:
                pbar = tqdm(
                    runners_to_fetch,
                    desc=desc[:30].ljust(30),
                    unit="runner",
                    position=self.position,
                    leave=False,
                    ncols=100,
                    bar_format='{desc} {bar}| {n_fmt}/{total_fmt} [{rate_fmt}]'
                )
                
                for idx, url in pbar:
                    history = fetch_runner_race_history(url, self.access_token, session=session)
                    
                    if history:
                        for race in history.get('results', []):
                            race_name = race.get('race', '')
                            race_name_lower = race_name.lower()
                            # Case-insensitive matching
                            if race_name_pattern_lower in race_name_lower and str(self.year) in race_name:
                                score = race.get('index')
                                if score:
                                    scores[idx] = score
                                    self.scores_found += 1
                                break
                    
                    pbar.set_postfix({'scores': self.scores_found}, refresh=False)
                
                pbar.close()
            else:
                for i, (idx, url) in enumerate(runners_to_fetch):
                    history = fetch_runner_race_history(url, self.access_token, session=session)
                    
                    if history:
                        for race in history.get('results', []):
                            race_name = race.get('race', '')
                            race_name_lower = race_name.lower()
                            # Case-insensitive matching
                            if race_name_pattern_lower in race_name_lower and str(self.year) in race_name:
                                score = race.get('index')
                                if score:
                                    scores[idx] = score
                                    self.scores_found += 1
                                break
            
            # Update DataFrame
            for idx, score in scores.items():
                df.loc[idx, 'utmb_score'] = score
            
            # Save
            write_csv_with_metadata(df, self.csv_file, metadata)
            
            self.status = "completed"
            return (True, f"{self.scores_found}/{len(runners_to_fetch)} scores")
            
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            return (False, str(e))


def get_files_to_process(tenant_filter=None, year_filter=None, skip_existing=False):
    """Get list of CSV files to process."""
    data_dir = get_data_dir()
    if not data_dir.exists():
        return []
    
    csv_files = list(data_dir.glob("*.csv"))
    files_to_process = []
    
    for csv_file in csv_files:
        parts = csv_file.stem.split('_')
        if len(parts) >= 3:
            tenant = parts[0]
            try:
                year = int(parts[1])
            except ValueError:
                continue
            race_id = '_'.join(parts[2:])
            
            if tenant_filter and tenant != tenant_filter:
                continue
            if year_filter and year != year_filter:
                continue
            
            if skip_existing and csv_has_utmb_scores(csv_file):
                continue
            
            # Get runner count
            try:
                df, _ = read_csv_with_metadata(csv_file)
                runner_count = len(df)
            except Exception:
                runner_count = 0
            
            files_to_process.append((csv_file, tenant, year, race_id, runner_count))
    
    # Sort by runner count (smallest first)
    files_to_process.sort(key=lambda x: x[4])
    
    return files_to_process


def run_parallel_step2(cookie, tenant_filter=None, year_filter=None, 
                       skip_existing=False, num_workers=4, dry_run=False):
    """Run step 2 in parallel for multiple races."""
    
    files = get_files_to_process(tenant_filter, year_filter, skip_existing)
    
    if not files:
        print("\nNo files to process!")
        if skip_existing:
            print("All files already have UTMB scores.")
            print("Remove --skip-existing to re-fetch scores.")
        return
    
    # Extract access token
    access_token = extract_access_token(cookie)
    if not access_token:
        print("ERROR: Could not extract access_token from cookie!")
        return
    
    print(f"Access token extracted (length: {len(access_token)})")
    
    # Quick auth test
    print("Testing authentication...")
    from utmb_scraper import fetch_runner_race_history
    test_result = fetch_runner_race_history("758585.alexandre.boucheix", access_token)
    if test_result:
        if test_result.get('races_with_scores', 0) > 0:
            print(f"  ✓ Authentication working! Found {test_result['races_with_scores']} scores in test runner's history")
        else:
            print(f"  ⚠ WARNING: API responded but returned 0 scores.")
            print(f"    This usually means the cookie has EXPIRED.")
            print(f"    Please get a fresh cookie from your browser and try again.")
            return
    else:
        print("  ✗ ERROR: API request failed!")
        return
    
    # Calculate totals
    total_runners = sum(f[4] for f in files)
    
    print("\n" + "=" * 70)
    print("PARALLEL UTMB SCORE FETCHER")
    print("=" * 70)
    print(f"\nFiles to process: {len(files)}")
    print(f"Total runners: {total_runners:,}")
    print(f"Parallel workers: {num_workers}")
    
    if len(files) <= 20:
        print("\nFiles (sorted by runner count):")
        for csv_file, tenant, year, race_id, count in files:
            print(f"  {tenant}/{year}/{race_id}: {count:,} runners")
    else:
        print(f"\nFirst 10 files:")
        for csv_file, tenant, year, race_id, count in files[:10]:
            print(f"  {tenant}/{year}/{race_id}: {count:,} runners")
        print(f"  ... and {len(files) - 10} more")
    
    if dry_run:
        print("\n" + "-" * 70)
        print("DRY RUN: No actual processing performed.")
        return
    
    print("\n" + "-" * 70)
    print(f"Starting {num_workers} parallel workers...")
    print("Each race shows its own progress bar.\n")
    
    results = {'success': [], 'failed': []}
    start_time = time.time()
    completed_count = 0
    
    # Process in batches to manage progress bar positions
    def process_race(args):
        csv_file, tenant, year, race_id, _, position = args
        processor = RaceProcessor(csv_file, tenant, year, race_id, access_token, position)
        success, message = processor.process()
        return (tenant, year, race_id, success, message)
    
    try:
        # Process files with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Add position for progress bar
            work_items = [
                (f[0], f[1], f[2], f[3], f[4], i % num_workers)
                for i, f in enumerate(files)
            ]
            
            futures = {executor.submit(process_race, item): item for item in work_items}
            
            # Overall progress bar at bottom
            if TQDM_AVAILABLE:
                overall_pbar = tqdm(
                    total=len(files),
                    desc="Overall progress".ljust(30),
                    position=num_workers,
                    unit="race",
                    ncols=100,
                    bar_format='{desc} {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )
            
            for future in as_completed(futures):
                tenant, year, race_id, success, message = future.result()
                completed_count += 1
                
                if success:
                    results['success'].append((tenant, year, race_id, message))
                else:
                    results['failed'].append((tenant, year, race_id, message))
                
                if TQDM_AVAILABLE:
                    overall_pbar.update(1)
                    overall_pbar.set_postfix({
                        'ok': len(results['success']), 
                        'fail': len(results['failed'])
                    })
            
            if TQDM_AVAILABLE:
                overall_pbar.close()
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    
    elapsed = time.time() - start_time
    
    # Clear progress bar area
    if TQDM_AVAILABLE:
        print("\n" * (num_workers + 1))
    
    # Print summary
    print("=" * 70)
    print("PARALLEL PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nElapsed time: {elapsed/60:.1f} minutes")
    print(f"Processing rate: {completed_count / (elapsed/60):.1f} races/minute")
    
    print(f"\n✓ Successful: {len(results['success'])} races")
    if len(results['success']) <= 20:
        for tenant, year, race_id, message in results['success']:
            print(f"    {tenant}/{year}/{race_id}: {message}")
    else:
        # Show summary by score count
        with_scores = sum(1 for _, _, _, msg in results['success'] if msg and '/' in msg and not msg.startswith('0/'))
        print(f"    {with_scores} races with scores found")
        print(f"    {len(results['success']) - with_scores} races with 0 scores (expected for short/future races)")
    
    if results['failed']:
        print(f"\n✗ Failed: {len(results['failed'])} races")
        for tenant, year, race_id, message in results['failed'][:10]:
            print(f"    {tenant}/{year}/{race_id}: {message}")
        if len(results['failed']) > 10:
            print(f"    ... and {len(results['failed']) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel UTMB Score Fetcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--cookie', type=str, required=True,
                        help='Browser cookie for authentication')
    parser.add_argument('--tenant', type=str, default=None,
                        help='Only process this tenant')
    parser.add_argument('--year', type=int, default=None,
                        help='Only process this year')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files that already have UTMB scores')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without actually processing')
    
    args = parser.parse_args()
    
    run_parallel_step2(
        cookie=args.cookie,
        tenant_filter=args.tenant,
        year_filter=args.year,
        skip_existing=args.skip_existing,
        num_workers=args.workers,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
