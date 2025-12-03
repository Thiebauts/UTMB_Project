#!/usr/bin/env python3
"""
=============================================================================
UTMB Race Data Scraper - Clean Pipeline for Analysis
=============================================================================

Scrapes race results from UTMB World Series events and outputs clean CSV files
with race metadata headers and final results (no checkpoint times).

USAGE:
------
Step 1: Scrape race results (no authentication needed)
    python3 utmb_scraper.py --step 1 --tenant kullamannen --year 2025 --race 100M

Step 2: Add UTMB scores (authentication required)
    python3 utmb_scraper.py --step 2 --tenant kullamannen --year 2025 --race 100M --cookie "YOUR_COOKIE"

BATCH MODE: Scrape all races from events.json
    python3 utmb_scraper.py --step 1 --all                    # All events, all years, all races
    python3 utmb_scraper.py --step 1 --all --tenant kullamannen   # All years/races for one event
    python3 utmb_scraper.py --step 1 --all --tenant kullamannen --year 2025  # All races for one event/year
    python3 utmb_scraper.py --step 1 --all --skip-existing    # Skip races already scraped

HOW TO GET YOUR COOKIE:
-----------------------
1. Log in to utmb.world in your browser
2. Open Developer Tools (F12)
3. Go to Network tab
4. Refresh the page
5. Click on any request to utmb.world
6. Find "Cookie:" in Request Headers
7. Copy the entire cookie value

OUTPUT:
-------
CSV files in data/ folder with format:
  # Race: Kullamannen by UTMB 2025 - Ultra 100 Miles
  # Distance: 173 km
  # D+: 2300 m
  # D-: 2300 m
  # Date: 2025-10-31
  # URL: https://live.utmb.world/kullamannen/2025
  bib,name,country,age,sex,category,club,...

=============================================================================
"""

import requests
import re
import json
import pandas as pd
import time
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Import race data functions from update_race_data
from update_race_data import get_race_metadata as get_cached_race_metadata, load_events_data

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install 'tqdm' for better progress bars: pip install tqdm")


# =============================================================================
# CONFIGURATION
# =============================================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Accept': 'text/html,application/json',
}

UTMB_RUNNER_HISTORY_API = 'https://api.utmb.world/runners/{runner_uri}/results?limit=500&lang=en'

UTMB_API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Origin': 'https://utmb.world',
    'Referer': 'https://utmb.world/',
    'X-Tenant-Id': 'worldseries',
}

# Columns to keep in final output (no checkpoint data)
RESULT_COLUMNS = [
    'bib',
    'name',
    'country',
    'age',
    'sex',
    'category',
    'club',
    'utmb_profile_url',
    'utmb_index',
    'rank_scratch',
    'rank_sex',
    'rank_category',
    'race_time',
    'race_time_seconds',
    'is_finisher',
    'status',
    'utmb_score',
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_data_dir():
    """Get the data directory path."""
    return Path(__file__).parent / 'data'


def get_output_filename(tenant: str, year: int, race_id: str) -> Path:
    """Generate output file path."""
    return get_data_dir() / f"{tenant}_{year}_{race_id}.csv"


def fetch_event_metadata(tenant: str, year: int) -> dict:
    """
    Fetch event and race metadata from live.utmb.world.
    
    This extracts the actual distance, elevation gain, and dates from the
    live results page, ensuring accurate data for each specific year.
    
    Args:
        tenant: Event tenant name (e.g., 'kullamannen')
        year: Event year
    
    Returns:
        dict with event info and races metadata, or None on failure
    """
    url = f"https://live.utmb.world/{tenant}/{year}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            print(f"  Warning: Could not fetch event page (status {response.status_code})")
            return None
        
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            response.text,
            re.DOTALL
        )
        if not match:
            print("  Warning: Could not find event data on page")
            return None
        
        data = json.loads(match.group(1))
        page_props = data.get('props', {}).get('pageProps', {})
        
        current_event = page_props.get('currentEvent', {})
        homepage = page_props.get('homepage', {})
        
        # Build races dict indexed by raceId
        races = {}
        for race in homepage.get('races', []):
            race_id = race.get('raceId')
            if race_id:
                # Parse start date
                start_date = race.get('startDate', '')
                try:
                    date_str = start_date.split('T')[0] if start_date else None
                except:
                    date_str = None
                
                races[race_id] = {
                    'name': race.get('name'),
                    'distance_km': race.get('distance'),
                    'elevation_gain': race.get('elevationGain'),
                    # D- is not available from API, default to D+ for loop courses
                    'elevation_loss': race.get('elevationLoss') or race.get('elevationGain'),
                    'date': date_str,
                    'category': race.get('category'),
                    'stones': race.get('stones'),
                }
        
        return {
            'event_name': homepage.get('eventName') or current_event.get('title'),
            'tenant': tenant,
            'year': year,
            'timezone': homepage.get('eventTimezone'),
            'country': current_event.get('country'),
            'country_code': current_event.get('countryCode'),
            'url': f"https://live.utmb.world/{tenant}/{year}",
            'races': races,
        }
        
    except Exception as e:
        print(f"  Warning: Error fetching event metadata: {e}")
        return None


def get_race_metadata(tenant: str, year: int, race_id: str) -> dict:
    """
    Get race metadata, trying live data first, then falling back to events.json cache.
    
    Args:
        tenant: Event tenant name
        year: Event year
        race_id: Race ID
    
    Returns:
        dict with race metadata
    """
    # Try to fetch live data first
    print(f"\nFetching race metadata from live.utmb.world...")
    event_data = fetch_event_metadata(tenant, year)
    
    if event_data and race_id in event_data.get('races', {}):
        race = event_data['races'][race_id]
        print(f"  ✓ Found live data: {race.get('name')} - {race.get('distance_km')}km, D+{race.get('elevation_gain')}m")
        
        return {
            'race_name': f"{event_data['event_name']} {year} - {race['name']}",
            'distance_km': race.get('distance_km'),
            'elevation_gain': race.get('elevation_gain'),
            'elevation_loss': race.get('elevation_loss'),
            'date': race.get('date'),
            'url': event_data['url'],
            'source': 'live',
        }
    
    # Fall back to events.json cache
    print(f"  ⚠ Live data not available, checking events.json cache...")
    cached = get_cached_race_metadata(tenant, year, race_id)
    
    if cached:
        print(f"  ✓ Found cached data: {cached.get('race_name')}")
        return {
            'race_name': cached.get('race_name'),
            'distance_km': cached.get('distance_km'),
            'elevation_gain': cached.get('elevation_gain'),
            'elevation_loss': cached.get('elevation_loss'),
            'date': cached.get('date'),
            'url': cached.get('url'),
            'source': 'cache',
        }
    
    # No data available
    print(f"  ⚠ No data found for {tenant}/{year}/{race_id}")
    print(f"  Tip: Run 'python3 update_race_data.py --event {tenant} --year {year}' to fetch race data")
    return {
        'race_name': f"{tenant} {year} - {race_id}",
        'distance_km': None,
        'elevation_gain': None,
        'elevation_loss': None,
        'date': None,
        'url': f"https://live.utmb.world/{tenant}/{year}",
        'source': 'none',
    }


def get_next_data(url, headers=None, max_retries=3):
    """
    Extract __NEXT_DATA__ JSON from a page with retry logic.
    
    Handles transient network errors (DNS resolution, connection timeouts)
    with exponential backoff.
    """
    if headers is None:
        headers = HEADERS
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                return None
            match = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                response.text,
                re.DOTALL
            )
            if match:
                return json.loads(match.group(1))
            return None
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            # Transient network errors - retry with backoff
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                time.sleep(wait_time)
                continue
            # Final attempt failed - log but don't spam
            return None
        except Exception as e:
            # Other errors - log and fail
            print(f"Error fetching {url}: {e}")
            return None
    return None


def extract_access_token(cookie: str) -> str:
    """Extract access_token from a cookie string."""
    if not cookie:
        return None
    match = re.search(r'access_token=([^;]+)', cookie)
    return match.group(1) if match else None


def fetch_runner_race_history(runner_uri: str, access_token: str = None, max_retries: int = 3) -> dict:
    """
    Fetch complete race history for a runner using the UTMB API.
    
    Args:
        runner_uri: Runner URI (e.g., "758585.alexandre.boucheix")
        access_token: Optional access token for authenticated requests
        max_retries: Number of retries for transient network errors
    
    Returns:
        dict with race history data, or None on failure
    """
    if not runner_uri or pd.isna(runner_uri):
        return None
    
    # Extract runner URI from full URL if needed
    if 'utmb.world/runner/' in str(runner_uri):
        runner_uri = runner_uri.split('/runner/')[-1]
    
    url = UTMB_RUNNER_HISTORY_API.format(runner_uri=runner_uri)
    
    headers = UTMB_API_HEADERS.copy()
    if access_token:
        headers['Authorization'] = f'Bearer {access_token}'
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                scores_count = sum(1 for r in results if r.get('index') is not None)
                
                return {
                    'runner_uri': runner_uri,
                    'results': results,
                    'total_races': len(results),
                    'races_with_scores': scores_count,
                }
            return None
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError):
            # Transient network errors - retry with backoff
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue
            return None
        except Exception:
            return None
    
    return None


# =============================================================================
# CSV I/O WITH METADATA HEADERS
# =============================================================================

def write_csv_with_metadata(df: pd.DataFrame, filepath: Path, metadata: dict):
    """
    Write DataFrame to CSV with metadata header comments.
    
    Args:
        df: DataFrame to write
        filepath: Output file path
        metadata: dict with race metadata (race_name, distance_km, etc.)
    """
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Build header comments
    header_lines = [
        f"# Race: {metadata.get('race_name', 'Unknown')}",
        f"# Distance: {metadata.get('distance_km', 'N/A')} km",
        f"# D+: {metadata.get('elevation_gain', 'N/A')} m",
        f"# D-: {metadata.get('elevation_loss', 'N/A')} m",
        f"# Date: {metadata.get('date', 'N/A')}",
        f"# URL: {metadata.get('url', 'N/A')}",
    ]
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in header_lines:
            f.write(line + '\n')
        df.to_csv(f, index=False)
    
    print(f"  Saved to {filepath}")


def read_csv_with_metadata(filepath: Path) -> tuple:
    """
    Read CSV file with metadata header comments.
    
    Args:
        filepath: Input file path
    
    Returns:
        tuple (DataFrame, metadata_dict)
    """
    metadata = {}
    header_lines = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                header_lines += 1
                # Parse metadata
                if ':' in line:
                    key_part = line[2:].split(':', 1)
                    if len(key_part) == 2:
                        key = key_part[0].strip().lower().replace(' ', '_')
                        value = key_part[1].strip()
                        # Clean up key names
                        if key == 'd+':
                            key = 'elevation_gain'
                            value = value.replace(' m', '')
                        elif key == 'd-':
                            key = 'elevation_loss'
                            value = value.replace(' m', '')
                        elif key == 'distance':
                            key = 'distance_km'
                            value = value.replace(' km', '')
                        metadata[key] = value
            else:
                break
    
    # Read DataFrame, skipping header comments
    df = pd.read_csv(filepath, skiprows=header_lines)
    
    return df, metadata


# =============================================================================
# STEP 1: SCRAPE RACE RESULTS (NO AUTH REQUIRED)
# =============================================================================

def get_runner_data(base_url: str, tenant: str, year: int, bib: int) -> dict:
    """Fetch detailed data for a single runner."""
    url = f"{base_url}/{tenant}/{year}/runners/{bib}"
    data = get_next_data(url)
    
    if not data:
        return None
    
    runner = data['props']['pageProps'].get('runner', {})
    if not runner:
        return None
    
    resume = runner.get('resume', {})
    detail = runner.get('detail', {})
    
    return {
        'bib': bib,
        'race_id': resume.get('raceId'),
        'resume': resume,
        'detail': detail,
    }


def discover_bib_range(base_url: str, tenant: str, year: int, race_id: str, 
                       num_workers: int = 20, max_search_bib: int = 10000) -> tuple:
    """
    Discover the actual bib range for a specific race by sampling.
    
    Returns (min_bib, max_bib) for the race, or None if not found.
    """
    print(f"\nDiscovering bib range for {race_id}...")
    
    def get_race_id_for_bib(bib):
        url = f"{base_url}/{tenant}/{year}/runners/{bib}"
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                return None
            match = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                response.text,
                re.DOTALL
            )
            if match:
                data = json.loads(match.group(1))
                runner = data['props']['pageProps'].get('runner', {})
                if runner:
                    return runner.get('resume', {}).get('raceId')
        except:
            pass
        return None
    
    # Phase 1: Coarse sampling
    sample_step = 50
    sample_bibs = list(range(1, max_search_bib + 1, sample_step))
    found_bibs = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(get_race_id_for_bib, bib): bib for bib in sample_bibs}
        
        if TQDM_AVAILABLE:
            pbar = tqdm(as_completed(futures), total=len(futures), 
                       desc="Coarse scan", unit="bib", ncols=80,
                       miniters=10, mininterval=0.5)
            for future in pbar:
                bib = futures[future]
                rid = future.result()
                if rid == race_id:
                    found_bibs.append(bib)
            pbar.close()
        else:
            for future in as_completed(futures):
                bib = futures[future]
                rid = future.result()
                if rid == race_id:
                    found_bibs.append(bib)
    
    if not found_bibs:
        print(f"  No bibs found for {race_id} in coarse scan")
        return None
    
    # Estimate range
    min_sample = min(found_bibs)
    max_sample = max(found_bibs)
    estimated_min = max(1, min_sample - sample_step)
    estimated_max = max_sample + sample_step
    
    print(f"  Coarse scan found bibs in range ~{min_sample}-{max_sample}")
    
    # Phase 2: Fine scan at boundaries
    boundary_bibs = set()
    for b in range(estimated_min, min(estimated_min + 100, estimated_max + 1)):
        boundary_bibs.add(b)
    for b in range(max(estimated_max - 100, estimated_min), estimated_max + 1):
        boundary_bibs.add(b)
    
    boundary_bibs = [b for b in boundary_bibs if b not in found_bibs]
    
    if boundary_bibs:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(get_race_id_for_bib, bib): bib for bib in boundary_bibs}
            
            if TQDM_AVAILABLE:
                pbar = tqdm(as_completed(futures), total=len(futures), 
                           desc="Fine scan", unit="bib", ncols=80,
                           miniters=10, mininterval=0.5)
                for future in pbar:
                    bib = futures[future]
                    rid = future.result()
                    if rid == race_id:
                        found_bibs.append(bib)
                pbar.close()
            else:
                for future in as_completed(futures):
                    bib = futures[future]
                    rid = future.result()
                    if rid == race_id:
                        found_bibs.append(bib)
    
    actual_min = min(found_bibs)
    actual_max = max(found_bibs)
    final_min = max(1, actual_min - 5)
    final_max = actual_max + 5
    
    print(f"  Estimated bib range: {final_min} - {final_max} ({final_max - final_min + 1} bibs)")
    
    return (final_min, final_max)


def scan_for_runners(base_url: str, tenant: str, year: int, race_id: str,
                     bib_range: tuple = None, num_workers: int = 20, 
                     max_search_bib: int = 10000) -> list:
    """Scan for all runners in a race using parallel requests."""
    
    if bib_range is None:
        bib_range = discover_bib_range(base_url, tenant, year, race_id, 
                                       num_workers=20, max_search_bib=max_search_bib)
        if bib_range is None:
            print(f"Could not discover bib range. Falling back to 1-{max_search_bib}.")
            bib_range = (1, max_search_bib)
    
    min_bib, max_bib = bib_range
    print(f"\nScanning for runners (bibs {min_bib} to {max_bib})...")
    
    def fetch_single_bib(bib):
        try:
            runner_data = get_runner_data(base_url, tenant, year, bib)
            if runner_data and runner_data.get('race_id') == race_id:
                return runner_data
        except Exception:
            pass
        return None
    
    all_runners = []
    bibs = list(range(min_bib, max_bib + 1))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_single_bib, bib): bib for bib in bibs}
        
        if TQDM_AVAILABLE:
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching runners",
                unit="bib",
                ncols=100,
                miniters=10,  # Update at most every 10 iterations
                mininterval=0.5,  # Update at most every 0.5 seconds
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            for future in pbar:
                result = future.result()
                if result:
                    all_runners.append(result)
                # Only update postfix occasionally to avoid excessive refreshes
                if len(all_runners) % 50 == 0 or len(all_runners) == 1:
                    pbar.set_postfix({'found': len(all_runners)}, refresh=False)
            pbar.close()
        else:
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result:
                    all_runners.append(result)
                if completed % 200 == 0:
                    print(f"  Progress: {completed}/{len(bibs)} bibs, found {len(all_runners)} runners...")
    
    return all_runners


def create_results_dataframe(runners: list) -> pd.DataFrame:
    """Convert runner data to a clean DataFrame (no checkpoint columns)."""
    
    rows = []
    for runner in runners:
        resume = runner.get('resume', {})
        detail = runner.get('detail', {})
        info = resume.get('info', {})
        ranking = resume.get('ranking', {})
        
        row = {
            'bib': runner.get('bib'),
            'name': info.get('fullname'),
            'country': info.get('countryCode'),
            'age': info.get('age'),
            'sex': info.get('sex'),
            'category': info.get('category'),
            'club': info.get('club'),
            'utmb_profile_url': info.get('url'),
            'utmb_index': info.get('index'),
            'rank_scratch': ranking.get('scratch'),
            'rank_sex': ranking.get('sex'),
            'rank_category': ranking.get('category'),
            'race_time': resume.get('raceTime'),
            'race_time_seconds': detail.get('raceTimeInSeconds'),
            'is_finisher': resume.get('isFinisher'),
            'status': resume.get('status'),
            'utmb_score': None,  # Will be filled in Step 2
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('rank_scratch', na_position='last')
    
    # Ensure columns are in the right order
    cols = [c for c in RESULT_COLUMNS if c in df.columns]
    df = df[cols]
    
    return df


def extract_race_date(runners: list) -> str:
    """Extract race date from runner data."""
    for runner in runners:
        start_time = runner.get('resume', {}).get('start')
        if start_time:
            try:
                # Parse ISO format datetime
                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d')
            except:
                pass
    return 'N/A'


def run_step1(tenant: str, year: int, race_id: str, 
              bib_range: tuple = None, max_search_bib: int = 10000):
    """
    STEP 1: Scrape race results from live.utmb.world (no authentication needed)
    
    Race metadata (distance, elevation) is automatically fetched from the
    live results page, ensuring accurate data for each specific year.
    """
    print("=" * 70)
    print(f"STEP 1: Scraping Race Results")
    print(f"Event: {tenant} | Year: {year} | Race: {race_id}")
    print("=" * 70)
    
    # Get race metadata (automatically fetched from live page)
    metadata = get_race_metadata(tenant, year, race_id)
    
    # Base URL is always the same
    base_url = 'https://live.utmb.world'
    
    # Scan for runners
    scan_start = time.time()
    all_runners = scan_for_runners(base_url, tenant, year, race_id, 
                                   bib_range, max_search_bib=max_search_bib)
    elapsed = time.time() - scan_start
    
    print(f"\nScan complete in {elapsed:.1f} seconds")
    print(f"Found {len(all_runners)} runners")
    
    if not all_runners:
        print("No runners found!")
        return None
    
    # Create DataFrame
    df = create_results_dataframe(all_runners)
    
    # Summary
    finishers = df['is_finisher'].sum()
    print(f"\nSummary:")
    print(f"  Total runners: {len(df)}")
    print(f"  Finishers: {finishers}")
    print(f"  DNF: {len(df) - finishers}")
    
    # If date wasn't in metadata, extract from runner data
    if not metadata.get('date'):
        metadata['date'] = extract_race_date(all_runners)
    
    # Save CSV with metadata
    output_file = get_output_filename(tenant, year, race_id)
    write_csv_with_metadata(df, output_file, metadata)
    
    print(f"\n{'=' * 70}")
    print("STEP 1 COMPLETE")
    print("=" * 70)
    print(f"\nNext: Run Step 2 with your browser cookie to get UTMB scores:")
    print(f"  python3 utmb_scraper.py --step 2 --tenant {tenant} --year {year} --race {race_id} --cookie 'YOUR_COOKIE'")
    
    return df


# =============================================================================
# STEP 2: FETCH UTMB SCORES (AUTH REQUIRED)
# =============================================================================

def run_step2(tenant: str, year: int, race_id: str, cookie: str):
    """
    STEP 2: Fetch UTMB scores using authenticated API
    """
    print("=" * 70)
    print(f"STEP 2: Fetching UTMB Scores")
    print(f"Event: {tenant} | Year: {year} | Race: {race_id}")
    print("=" * 70)
    
    # Check if CSV exists
    csv_file = get_output_filename(tenant, year, race_id)
    if not csv_file.exists():
        print(f"ERROR: {csv_file} not found. Run Step 1 first!")
        return
    
    # Load CSV with metadata
    df, metadata = read_csv_with_metadata(csv_file)
    print(f"Loaded {len(df)} runners from {csv_file}")
    
    # Extract access token
    access_token = extract_access_token(cookie)
    if not access_token:
        print("ERROR: Could not extract access_token from cookie!")
        print("Make sure your cookie contains 'access_token=...'")
        return
    
    print("  Access token extracted successfully")
    
    # Test authentication
    print("\nTesting authentication...")
    test_url = df[df['utmb_profile_url'].notna()]['utmb_profile_url'].iloc[0]
    test_result = fetch_runner_race_history(test_url, access_token)
    
    if not test_result:
        print("ERROR: API request failed! Cookie may be expired.")
        return
    
    if test_result.get('races_with_scores', 0) == 0 and test_result.get('total_races', 0) > 0:
        print("ERROR: Authentication failed! No scores returned.")
        print("Please get a fresh cookie from your browser.")
        return
    
    print(f"  Authentication successful!")
    
    # Get race name pattern for matching scores
    cached = get_cached_race_metadata(tenant, year, race_id)
    if cached:
        # Extract just the race name (not full "Event Year - Race" format)
        race_name_pattern = cached.get('race_name', '').split(' - ')[-1] if cached.get('race_name') else race_id
    else:
        race_name_pattern = race_id
    
    # Fetch scores for all runners
    runners_with_profiles = df[df['utmb_profile_url'].notna()]
    print(f"\nFetching UTMB scores for {len(runners_with_profiles)} runners...")
    
    scores = {}
    
    runners_to_fetch = [
        (idx, row['utmb_profile_url'])
        for idx, row in runners_with_profiles.iterrows()
    ]
    
    def fetch_score_for_runner(runner_info):
        idx, url = runner_info
        history = fetch_runner_race_history(url, access_token)
        
        if history:
            # Find the score for this specific race
            for race in history.get('results', []):
                race_name = race.get('race', '')
                if race_name_pattern in race_name and str(year) in race_name:
                    return (idx, race.get('index'))
        return None
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(fetch_score_for_runner, r): r for r in runners_to_fetch}
        
        if TQDM_AVAILABLE:
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching scores",
                unit="runner",
                ncols=100,
                miniters=10,
                mininterval=0.5,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            for future in pbar:
                result = future.result()
                if result:
                    idx, score = result
                    scores[idx] = score
                # Only update postfix occasionally to avoid excessive refreshes
                if len(scores) % 50 == 0 or len(scores) == 1:
                    pbar.set_postfix({'scores': len(scores)}, refresh=False)
            pbar.close()
        else:
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result:
                    idx, score = result
                    scores[idx] = score
                if completed % 50 == 0:
                    print(f"  Progress: {completed}/{len(futures)}, scores: {len(scores)}...")
    
    # Update DataFrame with scores
    for idx, score in scores.items():
        df.loc[idx, 'utmb_score'] = score
    
    # Save updated CSV
    write_csv_with_metadata(df, csv_file, metadata)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("STEP 2 COMPLETE")
    print("=" * 70)
    
    finishers = df[df['is_finisher'] == True]
    finishers_with_score = finishers[finishers['utmb_score'].notna()]
    
    print(f"\nFinishers with UTMB score: {len(finishers_with_score)}/{len(finishers)}")
    
    if len(finishers_with_score) > 0:
        scores_series = df['utmb_score'].dropna()
        print(f"Score range: {scores_series.min():.0f} - {scores_series.max():.0f}")
        print(f"Mean score: {scores_series.mean():.0f}")
    
    print("\nTop 10 finishers:")
    cols = ['rank_scratch', 'name', 'race_time', 'utmb_index', 'utmb_score']
    print(finishers[cols].head(10).to_string(index=False))


# =============================================================================
# BATCH MODE: SCRAPE ALL RACES FROM EVENTS.JSON
# =============================================================================

def get_races_to_scrape(tenant_filter: str = None, year_filter: int = None, 
                        skip_existing: bool = False) -> list:
    """
    Get list of races to scrape from events.json.
    
    Args:
        tenant_filter: Optional - only include this tenant
        year_filter: Optional - only include this year
        skip_existing: If True, skip races that already have CSV files
    
    Returns:
        List of (tenant, year, race_id, race_name) tuples
    """
    data = load_events_data()
    events = data.get("events", {})
    
    races_to_scrape = []
    
    for tenant, event_data in events.items():
        # Filter by tenant if specified
        if tenant_filter and tenant != tenant_filter:
            continue
        
        years = event_data.get("years", {})
        
        for year_str, year_data in years.items():
            year = int(year_str)
            
            # Filter by year if specified
            if year_filter and year != year_filter:
                continue
            
            races = year_data.get("races", {})
            
            for race_id, race_info in races.items():
                # Check if should skip existing
                if skip_existing:
                    output_file = get_output_filename(tenant, year, race_id)
                    if output_file.exists():
                        continue
                
                race_name = race_info.get("name", race_id)
                races_to_scrape.append((tenant, year, race_id, race_name))
    
    return races_to_scrape


def run_batch_step1(tenant_filter: str = None, year_filter: int = None,
                    skip_existing: bool = False, max_search_bib: int = 10000,
                    dry_run: bool = False):
    """
    BATCH MODE: Scrape all races from events.json
    
    Args:
        tenant_filter: Optional - only scrape this tenant
        year_filter: Optional - only scrape this year
        skip_existing: If True, skip races that already have CSV files
        max_search_bib: Maximum bib number to search during auto-discovery
        dry_run: If True, only show what would be scraped without actually scraping
    """
    races = get_races_to_scrape(tenant_filter, year_filter, skip_existing)
    
    if not races:
        print("\nNo races to scrape!")
        if skip_existing:
            print("All races already have CSV files. Remove --skip-existing to re-scrape.")
        else:
            print("Make sure events.json has data. Run: python3 update_race_data.py --list")
        return
    
    # Group by tenant for display
    by_tenant = {}
    for tenant, year, race_id, race_name in races:
        if tenant not in by_tenant:
            by_tenant[tenant] = []
        by_tenant[tenant].append((year, race_id, race_name))
    
    print("\n" + "=" * 70)
    print("BATCH MODE: Scraping All Races" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 70)
    print(f"\nRaces to scrape: {len(races)}")
    print(f"Events: {len(by_tenant)}")
    
    for tenant, race_list in sorted(by_tenant.items()):
        print(f"\n  {tenant}:")
        for year, race_id, race_name in sorted(race_list):
            print(f"    - {year}/{race_id}: {race_name}")
    
    if dry_run:
        print("\n" + "-" * 70)
        print("DRY RUN: No actual scraping performed.")
        print("Remove --dry-run to start scraping.")
        return
    
    print("\n" + "-" * 70)
    input("Press Enter to start scraping (Ctrl+C to cancel)...")
    print()
    
    # Track results
    results = {
        'success': [],
        'failed': [],
        'skipped': [],
    }
    
    for i, (tenant, year, race_id, race_name) in enumerate(races, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(races)}] {tenant} {year} - {race_name} ({race_id})")
        print("=" * 70)
        
        try:
            df = run_step1(tenant, year, race_id, bib_range=None, max_search_bib=max_search_bib)
            
            if df is not None and len(df) > 0:
                results['success'].append((tenant, year, race_id, len(df)))
            else:
                results['failed'].append((tenant, year, race_id, "No runners found"))
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            results['failed'].append((tenant, year, race_id, str(e)))
    
    # Print summary
    print("\n" + "=" * 70)
    print("BATCH SCRAPING COMPLETE")
    print("=" * 70)
    
    print(f"\n✓ Successfully scraped: {len(results['success'])} races")
    for tenant, year, race_id, count in results['success']:
        print(f"    {tenant}/{year}/{race_id}: {count} runners")
    
    if results['failed']:
        print(f"\n✗ Failed: {len(results['failed'])} races")
        for tenant, year, race_id, error in results['failed']:
            print(f"    {tenant}/{year}/{race_id}: {error}")
    
    print(f"\nCSV files saved to: {get_data_dir()}/")


def run_batch_step2(tenant_filter: str = None, year_filter: int = None,
                    cookie: str = None):
    """
    BATCH MODE: Add UTMB scores to all existing CSV files
    
    Args:
        tenant_filter: Optional - only process this tenant
        year_filter: Optional - only process this year
        cookie: Browser cookie for authentication
    """
    if not cookie:
        print("ERROR: --cookie is required for step 2")
        return
    
    # Find all CSV files in data directory
    data_dir = get_data_dir()
    if not data_dir.exists():
        print(f"No data directory found: {data_dir}")
        return
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found. Run Step 1 first.")
        return
    
    # Filter files based on tenant/year
    files_to_process = []
    for csv_file in csv_files:
        # Parse filename: tenant_year_race.csv
        parts = csv_file.stem.split('_')
        if len(parts) >= 3:
            tenant = parts[0]
            try:
                year = int(parts[1])
            except ValueError:
                continue
            race_id = '_'.join(parts[2:])
            
            # Apply filters
            if tenant_filter and tenant != tenant_filter:
                continue
            if year_filter and year != year_filter:
                continue
            
            files_to_process.append((csv_file, tenant, year, race_id))
    
    if not files_to_process:
        print("No matching CSV files found.")
        return
    
    print("\n" + "=" * 70)
    print("BATCH MODE: Adding UTMB Scores")
    print("=" * 70)
    print(f"\nFiles to process: {len(files_to_process)}")
    
    for csv_file, tenant, year, race_id in files_to_process:
        print(f"  - {csv_file.name}")
    
    print("\n" + "-" * 70)
    input("Press Enter to start (Ctrl+C to cancel)...")
    print()
    
    for i, (csv_file, tenant, year, race_id) in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] Processing {csv_file.name}...")
        
        try:
            run_step2(tenant, year, race_id, cookie)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            break
        except Exception as e:
            print(f"ERROR: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UTMB Race Data Scraper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--step', type=int, choices=[1, 2], required=True,
                        help='Step to run: 1=scrape results, 2=fetch UTMB scores')
    parser.add_argument('--tenant', type=str, default=None,
                        help='Event tenant name (default: kullamannen for single race mode)')
    parser.add_argument('--year', type=int, default=None,
                        help='Event year (default: 2025 for single race mode)')
    parser.add_argument('--race', type=str, default=None,
                        help='Race ID (default: 100M for single race mode)')
    parser.add_argument('--all', action='store_true',
                        help='Batch mode: scrape all races from events.json')
    parser.add_argument('--skip-existing', action='store_true',
                        help='With --all: skip races that already have CSV files')
    parser.add_argument('--dry-run', action='store_true',
                        help='With --all: show what would be scraped without actually scraping')
    parser.add_argument('--bib-min', type=int, default=None,
                        help='Minimum bib number (auto-discovered if not set)')
    parser.add_argument('--bib-max', type=int, default=None,
                        help='Maximum bib number (auto-discovered if not set)')
    parser.add_argument('--max-search', type=int, default=10000,
                        help='Maximum bib to search during auto-discovery (default: 10000)')
    parser.add_argument('--cookie', type=str,
                        help='Browser cookie for authentication (required for step 2)')
    
    args = parser.parse_args()
    
    # BATCH MODE
    if args.all:
        if args.step == 1:
            run_batch_step1(
                tenant_filter=args.tenant,
                year_filter=args.year,
                skip_existing=args.skip_existing,
                max_search_bib=args.max_search,
                dry_run=args.dry_run
            )
        elif args.step == 2:
            if not args.cookie:
                print("ERROR: --cookie is required for step 2")
                print("\nHow to get your cookie:")
                print("1. Log in to utmb.world in your browser")
                print("2. Open Developer Tools (F12)")
                print("3. Go to Network tab")
                print("4. Click on any request to utmb.world")
                print("5. Find 'Cookie:' in Request Headers")
                print("6. Copy the entire cookie value")
                return
            run_batch_step2(
                tenant_filter=args.tenant,
                year_filter=args.year,
                cookie=args.cookie
            )
        return
    
    # SINGLE RACE MODE
    # Apply defaults for single race mode
    tenant = args.tenant or 'kullamannen'
    year = args.year or 2025
    race = args.race or '100M'
    
    # Determine bib range
    bib_range = None
    if args.bib_min is not None and args.bib_max is not None:
        bib_range = (args.bib_min, args.bib_max)
    elif args.bib_min is not None or args.bib_max is not None:
        print("WARNING: Both --bib-min and --bib-max must be set together. Using auto-discovery.")
    
    if args.step == 1:
        run_step1(tenant, year, race, bib_range, args.max_search)
    elif args.step == 2:
        if not args.cookie:
            print("ERROR: --cookie is required for step 2")
            print("\nHow to get your cookie:")
            print("1. Log in to utmb.world in your browser")
            print("2. Open Developer Tools (F12)")
            print("3. Go to Network tab")
            print("4. Refresh the page")
            print("5. Click on any request to utmb.world")
            print("6. Find 'Cookie:' in Request Headers")
            print("7. Copy the entire cookie value")
            return
        run_step2(tenant, year, race, args.cookie)


if __name__ == "__main__":
    main()

