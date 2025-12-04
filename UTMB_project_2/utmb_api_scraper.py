#!/usr/bin/env python3
"""
=============================================================================
UTMB Race Data Scraper v2 - Fast API-Based Pipeline
=============================================================================

Scrapes race results from UTMB World Series events using the direct API endpoint.
This is MUCH faster than scraping individual runner pages (no checkpoint data).

USAGE:
------
Step 1: List available races for an event:
    python3 utmb_api_scraper.py --step 1 --tenant kullamannen --list

Step 1: Scrape a specific race (no auth needed):
    python3 utmb_api_scraper.py --step 1 --tenant kullamannen --year 2025 --race "Ultra 100 Miles"

Step 1: Scrape all races for an event/year:
    python3 utmb_api_scraper.py --step 1 --tenant kullamannen --year 2025 --all

Step 2: Add UTMB scores to existing CSV files (auth required):
    python3 utmb_api_scraper.py --step 2 --cookie "YOUR_COOKIE"
    python3 utmb_api_scraper.py --step 2 --cookie "YOUR_COOKIE" --tenant kullamannen --year 2025

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
  # Tenant: kullamannen
  # Year: 2025
  # Race URI: 10675.kullamannenbyutmbultra100miles.2025
  # Total Finishers: 650
  rank,name,time,country,country_code,age_group,gender,utmb_index,utmb_score,runner_uri
  1,Christian MALMSTROM,16:18:22,Sweden,SE,45-49,H,,850,1243141.christian.malmstrom
  ...

=============================================================================
"""

import requests
import re
import json
import pandas as pd
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

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
    'Accept': 'application/json',
}

# API endpoint for race results
RESULTS_API = 'https://api.utmb.world/races/{race_uri}/results'

# Results page URL to get available races
RESULTS_PAGE_URL = 'https://{tenant}.utmb.world/runners/results'

# API endpoint for runner race history (to get UTMB scores)
UTMB_RUNNER_HISTORY_API = 'https://api.utmb.world/runners/{runner_uri}/results?limit=500&lang=en'

# Headers for authenticated UTMB API requests
UTMB_API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Origin': 'https://utmb.world',
    'Referer': 'https://utmb.world/',
    'X-Tenant-Id': 'worldseries',
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_data_dir():
    """Get the data directory path."""
    return Path(__file__).parent / 'data'


def get_output_filename(tenant: str, year: int, race_name: str) -> Path:
    """Generate output file path."""
    # Sanitize race name for filename
    safe_name = re.sub(r'[^\w\s-]', '', race_name).strip().replace(' ', '_')
    return get_data_dir() / f"{tenant}_{year}_{safe_name}.csv"


def time_to_seconds(time_str: str) -> Optional[int]:
    """Convert HH:MM:SS time string to seconds."""
    if not time_str:
        return None
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
    except (ValueError, AttributeError):
        return None
    return None


# =============================================================================
# API FUNCTIONS
# =============================================================================

def fetch_available_races(tenant: str) -> List[Dict]:
    """
    Fetch list of available races and years from the results page.
    
    Args:
        tenant: Event tenant name (e.g., 'kullamannen')
    
    Returns:
        List of dicts with year, race info
    """
    url = RESULTS_PAGE_URL.format(tenant=tenant)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            print(f"Error: Could not fetch results page (status {response.status_code})")
            return []
        
        # Extract __NEXT_DATA__ JSON
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            response.text,
            re.DOTALL
        )
        if not match:
            print("Error: Could not find page data")
            return []
        
        data = json.loads(match.group(1))
        page_props = data.get('props', {}).get('pageProps', {})
        slices = page_props.get('slices', [])
        
        # Find the results slice
        for slice_data in slices:
            if slice_data.get('type') == 'results':
                return slice_data.get('data', {}).get('results', [])
        
        print("Warning: No results slice found on page")
        return []
        
    except Exception as e:
        print(f"Error fetching available races: {e}")
        return []


def fetch_race_results(race_uri: str, limit: int = 1000) -> Tuple[List[Dict], int]:
    """
    Fetch race results from the API.
    
    Args:
        race_uri: Race URI (e.g., '10675.kullamannenbyutmbultra100miles.2025')
        limit: Maximum number of results per request
    
    Returns:
        Tuple of (list of result dicts, total count)
    """
    url = RESULTS_API.format(race_uri=race_uri)
    all_results = []
    offset = 0
    total = None
    
    while True:
        try:
            params = {'limit': limit, 'offset': offset}
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"Error: API returned status {response.status_code}")
                break
            
            data = response.json()
            results = data.get('results', [])
            
            if total is None:
                total = data.get('nbHits', 0)
            
            if not results:
                break
            
            all_results.extend(results)
            offset += len(results)
            
            # Check if we've fetched all results
            if offset >= total:
                break
                
        except Exception as e:
            print(f"Error fetching results: {e}")
            break
    
    return all_results, total or len(all_results)


# =============================================================================
# UTMB SCORE FETCHING (REQUIRES AUTHENTICATION)
# =============================================================================

def extract_access_token(cookie: str) -> Optional[str]:
    """Extract access_token from a cookie string."""
    if not cookie:
        return None
    match = re.search(r'access_token=([^;]+)', cookie)
    return match.group(1) if match else None


def fetch_runner_race_history(runner_uri: str, access_token: str = None, 
                               max_retries: int = 3, session=None) -> Optional[Dict]:
    """
    Fetch complete race history for a runner using the UTMB API.
    
    Args:
        runner_uri: Runner URI (e.g., "758585.alexandre.boucheix")
        access_token: Optional access token for authenticated requests
        max_retries: Number of retries for transient network errors
        session: Optional requests.Session for connection reuse
    
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
    
    # Use session if provided for connection reuse
    requester = session if session else requests
    
    for attempt in range(max_retries):
        try:
            response = requester.get(url, headers=headers, timeout=15)
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


def add_utmb_scores_to_csv(csv_file: Path, cookie: str) -> bool:
    """
    Add UTMB scores to an existing CSV file.
    
    Args:
        csv_file: Path to CSV file
        cookie: Browser cookie for authentication
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Adding UTMB Scores to: {csv_file.name}")
    print(f"{'='*60}")
    
    # Load CSV with metadata
    df, metadata = read_csv_with_metadata(csv_file)
    print(f"  Loaded {len(df)} runners")
    
    # Extract access token
    access_token = extract_access_token(cookie)
    if not access_token:
        print("  ✗ Error: Could not extract access_token from cookie!")
        print("    Make sure your cookie contains 'access_token=...'")
        return False
    
    print("  ✓ Access token extracted")
    
    # Check if runner_uri column exists
    if 'runner_uri' not in df.columns:
        print("  ✗ Error: CSV file missing 'runner_uri' column")
        return False
    
    # Test authentication with first runner
    print("  Testing authentication...")
    test_uri = df[df['runner_uri'].notna()]['runner_uri'].iloc[0]
    test_result = fetch_runner_race_history(test_uri, access_token)
    
    if not test_result:
        print("  ✗ Error: API request failed! Cookie may be expired.")
        return False
    
    if test_result.get('races_with_scores', 0) == 0 and test_result.get('total_races', 0) > 0:
        print("  ✗ Error: Authentication failed! No scores returned.")
        print("    Please get a fresh cookie from your browser.")
        return False
    
    print("  ✓ Authentication successful!")
    
    # Get race info for matching scores
    race_uri = metadata.get('race_uri', '')
    year = metadata.get('year', '')
    
    # Extract race name pattern from URI for matching
    # e.g., "10675.kullamannenbyutmbultra100miles.2025" -> look for "ultra" and year
    race_name_parts = race_uri.split('.')
    race_pattern = race_name_parts[1] if len(race_name_parts) > 1 else ''
    
    # Add utmb_score column if not exists
    if 'utmb_score' not in df.columns:
        df['utmb_score'] = None
    
    # Fetch scores for all runners
    runners_with_uris = df[df['runner_uri'].notna()]
    print(f"\n  Fetching UTMB scores for {len(runners_with_uris)} runners...")
    
    scores = {}
    
    runners_to_fetch = [
        (idx, row['runner_uri'])
        for idx, row in runners_with_uris.iterrows()
    ]
    
    # Use thread-local sessions for connection reuse
    thread_local = threading.local()
    
    def get_session():
        if not hasattr(thread_local, 'session'):
            thread_local.session = requests.Session()
            thread_local.session.headers.update(UTMB_API_HEADERS)
        return thread_local.session
    
    def fetch_score_for_runner(runner_info):
        idx, uri = runner_info
        session = get_session()
        history = fetch_runner_race_history(uri, access_token, session=session)
        
        if history:
            # Find the score for this specific race
            for race in history.get('results', []):
                race_name = race.get('race', '').lower()
                # Match by race pattern and year
                if race_pattern.lower() in race_name.replace(' ', '').replace('-', '') and str(year) in race_name:
                    score = race.get('index')
                    if score is not None:
                        return (idx, score)
        return None
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(fetch_score_for_runner, r): r for r in runners_to_fetch}
        
        if TQDM_AVAILABLE:
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="  Fetching scores",
                unit="runner",
                ncols=80,
                miniters=10,
                mininterval=0.5,
            )
            for future in pbar:
                result = future.result()
                if result:
                    idx, score = result
                    scores[idx] = score
                if len(scores) % 50 == 0 or len(scores) == 1:
                    pbar.set_postfix({'found': len(scores)}, refresh=False)
            pbar.close()
        else:
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result:
                    idx, score = result
                    scores[idx] = score
                if completed % 100 == 0:
                    print(f"    Progress: {completed}/{len(futures)}, scores found: {len(scores)}")
    
    # Update DataFrame with scores
    for idx, score in scores.items():
        df.loc[idx, 'utmb_score'] = score
    
    # Reorder columns to put utmb_score after utmb_index
    cols = list(df.columns)
    if 'utmb_score' in cols and 'utmb_index' in cols:
        cols.remove('utmb_score')
        idx = cols.index('utmb_index') + 1
        cols.insert(idx, 'utmb_score')
        df = df[cols]
    
    # Save updated CSV
    write_csv_with_metadata(df, csv_file, metadata)
    
    # Summary
    print(f"\n  Summary:")
    scores_count = df['utmb_score'].notna().sum()
    print(f"    Runners with UTMB score: {scores_count}/{len(df)}")
    
    if scores_count > 0:
        scores_series = df['utmb_score'].dropna()
        print(f"    Score range: {scores_series.min():.0f} - {scores_series.max():.0f}")
        print(f"    Mean score: {scores_series.mean():.0f}")
    
    print(f"\n  Top 10 finishers:")
    cols_to_show = ['rank_scratch', 'name', 'race_time', 'utmb_score']
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    print(df[cols_to_show].head(10).to_string(index=False))
    
    return True


def run_batch_step2(tenant_filter: str = None, year_filter: int = None,
                    cookie: str = None, skip_existing: bool = False):
    """
    BATCH MODE: Add UTMB scores to CSV files (optionally filtered by tenant/year).
    
    Args:
        tenant_filter: Optional - only process files for this tenant
        year_filter: Optional - only process files for this year
        cookie: Browser cookie for authentication
        skip_existing: If True, skip files that already have scores
    """
    data_dir = get_data_dir()
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found. Run step 1 first.")
        return
    
    # Filter files by tenant/year and skip_existing
    files_to_process = []
    files_skipped = []
    
    for csv_file in csv_files:
        # Parse filename: tenant_year_race.csv
        parts = csv_file.stem.split('_')
        if len(parts) >= 3:
            tenant = parts[0]
            try:
                year = int(parts[1])
            except ValueError:
                continue
            
            # Apply filters
            if tenant_filter and tenant != tenant_filter:
                continue
            if year_filter and year != year_filter:
                continue
        
        # Check if should skip files with existing scores
        if skip_existing:
            df, _ = read_csv_with_metadata(csv_file)
            if 'utmb_score' in df.columns and df['utmb_score'].notna().sum() > len(df) * 0.1:
                files_skipped.append(csv_file)
                continue
        
        files_to_process.append(csv_file)
    
    print(f"\n{'='*70}")
    print("STEP 2: Adding UTMB Scores to CSV Files")
    print(f"{'='*70}")
    
    if tenant_filter or year_filter:
        filters = []
        if tenant_filter:
            filters.append(f"tenant={tenant_filter}")
        if year_filter:
            filters.append(f"year={year_filter}")
        print(f"  Filters: {', '.join(filters)}")
    
    print(f"\n  Files to process: {len(files_to_process)}")
    if files_skipped:
        print(f"  Files skipped (already have scores): {len(files_skipped)}")
    
    for csv_file in files_to_process:
        print(f"    - {csv_file.name}")
    
    if not files_to_process:
        print("\n  No files to process.")
        return
    
    # Process each file
    success = 0
    failed = 0
    
    for csv_file in files_to_process:
        try:
            if add_utmb_scores_to_csv(csv_file, cookie):
                success += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            break
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print("STEP 2 COMPLETE")
    print(f"{'='*70}")
    print(f"  ✓ Successful: {success}")
    print(f"  ✗ Failed: {failed}")


# =============================================================================
# CSV I/O WITH METADATA HEADERS
# =============================================================================

def write_csv_with_metadata(df: pd.DataFrame, filepath: Path, metadata: dict):
    """
    Write DataFrame to CSV with metadata header comments.
    
    Args:
        df: DataFrame to write
        filepath: Output file path
        metadata: dict with race metadata
    """
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Build header comments
    header_lines = [
        f"# Race: {metadata.get('race_name', 'Unknown')}",
        f"# Tenant: {metadata.get('tenant', 'N/A')}",
        f"# Year: {metadata.get('year', 'N/A')}",
        f"# Race URI: {metadata.get('race_uri', 'N/A')}",
        f"# Total Finishers: {metadata.get('total_finishers', 'N/A')}",
        f"# Scraped: {datetime.now().isoformat()}",
    ]
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in header_lines:
            f.write(line + '\n')
        df.to_csv(f, index=False)
    
    print(f"  ✓ Saved to {filepath}")


def read_csv_with_metadata(filepath: Path) -> Tuple[pd.DataFrame, dict]:
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
                        metadata[key] = value
            else:
                break
    
    # Read DataFrame, skipping header comments
    df = pd.read_csv(filepath, skiprows=header_lines)
    
    return df, metadata


# =============================================================================
# MAIN SCRAPING FUNCTIONS
# =============================================================================

def scrape_race(tenant: str, year: int, race_info: dict) -> Optional[pd.DataFrame]:
    """
    Scrape a single race and save to CSV.
    
    Args:
        tenant: Event tenant name
        year: Race year
        race_info: Dict with raceName, raceUri, hasResults
    
    Returns:
        DataFrame with results, or None on failure
    """
    race_name = race_info.get('raceName', 'Unknown')
    race_uri = race_info.get('raceUri', '')
    has_results = race_info.get('hasResults', False)
    
    print(f"\n{'='*60}")
    print(f"Scraping: {race_name} ({year})")
    print(f"{'='*60}")
    
    if not has_results:
        print("  ⚠ No results available for this race")
        return None
    
    if not race_uri:
        print("  ⚠ No race URI found")
        return None
    
    print(f"  Race URI: {race_uri}")
    print(f"  Fetching results...")
    
    # Fetch results from API
    results, total = fetch_race_results(race_uri)
    
    if not results:
        print("  ⚠ No results returned from API")
        return None
    
    print(f"  ✓ Fetched {len(results)} of {total} results")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Rename columns for clarity
    column_mapping = {
        'runnerUri': 'runner_uri',
        'fullname': 'name',
        'time': 'race_time',
        'rank': 'rank_scratch',
        'nationality': 'country',
        'nationalityCode': 'country_code',
        'index': 'utmb_index',
        'ageGroup': 'age_group',
        'gender': 'sex',
    }
    df = df.rename(columns=column_mapping)
    
    # Add time in seconds
    df['race_time_seconds'] = df['race_time'].apply(time_to_seconds)
    
    # Add UTMB profile URL
    df['utmb_profile_url'] = df['runner_uri'].apply(
        lambda x: f"https://utmb.world/runner/{x}" if pd.notna(x) else None
    )
    
    # Reorder columns
    columns = [
        'rank_scratch', 'name', 'race_time', 'race_time_seconds',
        'country', 'country_code', 'age_group', 'sex',
        'utmb_index', 'runner_uri', 'utmb_profile_url'
    ]
    df = df[[c for c in columns if c in df.columns]]
    
    # Prepare metadata
    metadata = {
        'race_name': f"{tenant.title()} {year} - {race_name}",
        'tenant': tenant,
        'year': year,
        'race_uri': race_uri,
        'total_finishers': len(df),
    }
    
    # Save to CSV
    output_file = get_output_filename(tenant, year, race_name)
    write_csv_with_metadata(df, output_file, metadata)
    
    # Print summary
    print(f"\n  Summary:")
    print(f"    Total finishers: {len(df)}")
    if len(df) > 0:
        print(f"    Winner: {df.iloc[0]['name']} ({df.iloc[0]['race_time']})")
        
        # Gender breakdown
        if 'sex' in df.columns:
            gender_counts = df['sex'].value_counts()
            print(f"    Men: {gender_counts.get('H', 0)}")
            print(f"    Women: {gender_counts.get('F', 0)}")
    
    return df


def list_available_races(tenant: str):
    """List all available races for an event."""
    print(f"\n{'='*60}")
    print(f"Available Races for: {tenant}")
    print(f"{'='*60}")
    
    races_data = fetch_available_races(tenant)
    
    if not races_data:
        print("  No races found")
        return
    
    for year_data in races_data:
        year = year_data.get('year')
        races = year_data.get('races', [])
        
        print(f"\n  {year}:")
        for race in races:
            name = race.get('raceName', 'Unknown')
            has_results = race.get('hasResults', False)
            status = "✓" if has_results else "✗"
            print(f"    {status} {name}")


def scrape_all_races(tenant: str, year: Optional[int] = None, skip_existing: bool = False):
    """
    Scrape all races for an event (optionally filtered by year).
    
    Args:
        tenant: Event tenant name
        year: Optional year filter
        skip_existing: If True, skip races that already have CSV files
    """
    print(f"\n{'='*60}")
    print(f"Scraping All Races for: {tenant}" + (f" ({year})" if year else ""))
    print(f"{'='*60}")
    
    races_data = fetch_available_races(tenant)
    
    if not races_data:
        print("  No races found")
        return
    
    # Filter by year if specified
    if year:
        races_data = [y for y in races_data if y.get('year') == year]
    
    # Count total races with results
    total_races = sum(
        len([r for r in y.get('races', []) if r.get('hasResults')])
        for y in races_data
    )
    
    print(f"\n  Found {total_races} races with results")
    
    # Track results
    success = 0
    skipped = 0
    failed = 0
    
    for year_data in races_data:
        race_year = year_data.get('year')
        races = year_data.get('races', [])
        
        for race in races:
            if not race.get('hasResults'):
                continue
            
            race_name = race.get('raceName', 'Unknown')
            
            # Check if should skip existing
            if skip_existing:
                output_file = get_output_filename(tenant, race_year, race_name)
                if output_file.exists():
                    print(f"\n  Skipping {race_name} ({race_year}) - already exists")
                    skipped += 1
                    continue
            
            try:
                df = scrape_race(tenant, race_year, race)
                if df is not None and len(df) > 0:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ Error scraping {race_name}: {e}")
                failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"  ✓ Successful: {success}")
    print(f"  ⊘ Skipped: {skipped}")
    print(f"  ✗ Failed: {failed}")
    print(f"\n  CSV files saved to: {get_data_dir()}/")


def find_race_by_name(races_data: List[Dict], year: int, race_name: str) -> Optional[Dict]:
    """Find a race by year and name (case-insensitive partial match)."""
    for year_data in races_data:
        if year_data.get('year') != year:
            continue
        
        for race in year_data.get('races', []):
            if race_name.lower() in race.get('raceName', '').lower():
                return race
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UTMB Race Data Scraper v2 - Fast API-Based',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Step selection (same as UTMB_project)
    parser.add_argument('--step', type=int, choices=[1, 2], required=True,
                        help='Step to run: 1=scrape results, 2=fetch UTMB scores')
    
    # Common arguments (same as UTMB_project)
    parser.add_argument('--tenant', type=str, default=None,
                        help='Event tenant name (e.g., kullamannen)')
    parser.add_argument('--year', type=int, default=None,
                        help='Race year (required for single race, optional for --all)')
    parser.add_argument('--race', type=str, default=None,
                        help='Race name (partial match, e.g., "Ultra 100")')
    parser.add_argument('--all', action='store_true',
                        help='Batch mode: Step 1 scrapes all races, Step 2 adds scores to all CSVs')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip races/files that already exist or have scores')
    parser.add_argument('--cookie', type=str, default=None,
                        help='Browser cookie for authentication (required for step 2)')
    
    # Step 1 specific options
    parser.add_argument('--list', action='store_true',
                        help='Step 1: List available races')
    
    args = parser.parse_args()
    
    # STEP 2: Add UTMB scores
    if args.step == 2:
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
        
        # Filter by tenant/year if specified
        run_batch_step2(
            tenant_filter=args.tenant,
            year_filter=args.year,
            cookie=args.cookie,
            skip_existing=args.skip_existing
        )
        return
    
    # STEP 1: Scrape race results
    # For step 1, tenant is required
    if not args.tenant:
        print("ERROR: --tenant is required for step 1")
        print("Example: --tenant kullamannen")
        return
    
    # List mode
    if args.list:
        list_available_races(args.tenant)
        return
    
    # Scrape all mode
    if args.all:
        scrape_all_races(args.tenant, args.year, args.skip_existing)
        return
    
    # Single race mode
    if not args.year:
        print("ERROR: --year is required for single race scraping")
        print("Use --list to see available races, or --all to scrape all")
        return
    
    if not args.race:
        print("ERROR: --race is required for single race scraping")
        print("Use --list to see available races, or --all to scrape all")
        return
    
    # Find the race
    races_data = fetch_available_races(args.tenant)
    race_info = find_race_by_name(races_data, args.year, args.race)
    
    if not race_info:
        print(f"ERROR: Race '{args.race}' not found for year {args.year}")
        print("Use --list to see available races")
        return
    
    # Scrape the race
    scrape_race(args.tenant, args.year, race_info)


if __name__ == "__main__":
    main()

