#!/usr/bin/env python3
"""
Convert existing race data from the old format to the new UTMB_project format.
This script reads the existing CSV and creates a new clean CSV with metadata headers.

Race metadata is automatically fetched from live.utmb.world to ensure accurate
distance/elevation data for each specific year.
"""

import pandas as pd
from pathlib import Path
import sys

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent))

from race_configs import get_race_config, get_full_race_name, get_race_url
from utmb_scraper import get_race_metadata

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


def write_csv_with_metadata(df: pd.DataFrame, filepath: Path, metadata: dict):
    """Write DataFrame to CSV with metadata header comments."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    header_lines = [
        f"# Race: {metadata.get('race_name', 'Unknown')}",
        f"# Distance: {metadata.get('distance_km', 'N/A')} km",
        f"# D+: {metadata.get('elevation_gain', 'N/A')} m",
        f"# D-: {metadata.get('elevation_loss', 'N/A')} m",
        f"# Date: {metadata.get('date', 'N/A')}",
        f"# URL: {metadata.get('url', 'N/A')}",
    ]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in header_lines:
            f.write(line + '\n')
        df.to_csv(f, index=False)


def convert_existing_csv(source_file: Path, tenant: str, year: int, race_id: str):
    """Convert an existing CSV file to the new format."""
    
    print(f"\nConverting {source_file.name}...")
    
    # Read existing data
    df = pd.read_csv(source_file)
    print(f"  Loaded {len(df)} runners")
    
    # Rename score column if it exists
    old_score_col = f'utmb_score_{tenant}_{year}'
    if old_score_col in df.columns:
        df['utmb_score'] = df[old_score_col]
        print(f"  Found UTMB scores in column: {old_score_col}")
    else:
        df['utmb_score'] = None
        print(f"  No UTMB scores found (column {old_score_col} not present)")
    
    # Select only the columns we want
    available_cols = [c for c in RESULT_COLUMNS if c in df.columns]
    df_clean = df[available_cols].copy()
    
    # Sort by rank
    df_clean = df_clean.sort_values('rank_scratch', na_position='last')
    
    # Get race metadata (automatically fetched from live page)
    metadata = get_race_metadata(tenant, year, race_id)
    
    # If date wasn't in metadata, extract from start_time column
    if not metadata.get('date') and 'start_time' in df.columns:
        start_time = df['start_time'].dropna().iloc[0] if not df['start_time'].dropna().empty else None
        if start_time:
            try:
                metadata['date'] = start_time.split('T')[0]
            except:
                pass
    
    # Write to new location
    output_dir = Path(__file__).parent / 'data'
    output_file = output_dir / f"{tenant}_{year}_{race_id}.csv"
    
    write_csv_with_metadata(df_clean, output_file, metadata)
    
    # Summary
    finishers = df_clean['is_finisher'].sum()
    scores = df_clean['utmb_score'].notna().sum()
    
    print(f"  Output: {output_file}")
    print(f"  Total: {len(df_clean)} runners, {finishers} finishers, {scores} with UTMB score")
    
    return df_clean


def discover_existing_files(search_dir: Path) -> list:
    """
    Discover existing CSV files with the pattern {tenant}_{race}_{year}_results.csv
    
    Returns list of tuples: (filepath, tenant, year, race_id)
    """
    import re
    
    files = []
    pattern = re.compile(r'^([a-z]+)_([A-Za-z0-9]+)_(\d{4})_results\.csv$')
    
    for csv_file in search_dir.glob('*_results.csv'):
        match = pattern.match(csv_file.name)
        if match:
            tenant = match.group(1)
            race_id = match.group(2)
            year = int(match.group(3))
            files.append((csv_file, tenant, year, race_id))
    
    return sorted(files, key=lambda x: (x[1], x[2], x[3]))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert existing race data to UTMB_project format'
    )
    parser.add_argument('--file', type=str, nargs='+',
                        help='Specific file(s) to convert (format: tenant_race_year_results.csv)')
    parser.add_argument('--search-dir', type=str, default=None,
                        help='Directory to search for files (default: parent directory)')
    parser.add_argument('--discover', action='store_true',
                        help='Auto-discover all matching files in search directory')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    search_dir = Path(args.search_dir) if args.search_dir else project_root
    
    print("=" * 60)
    print("Converting existing data to UTMB_project format")
    print("=" * 60)
    
    conversions = []
    
    if args.file:
        # Convert specific files
        import re
        pattern = re.compile(r'^([a-z]+)_([A-Za-z0-9]+)_(\d{4})_results\.csv$')
        for f in args.file:
            filepath = Path(f) if Path(f).is_absolute() else search_dir / f
            match = pattern.match(filepath.name)
            if match:
                tenant = match.group(1)
                race_id = match.group(2)
                year = int(match.group(3))
                conversions.append((filepath, tenant, year, race_id))
            else:
                print(f"  Warning: {f} doesn't match expected pattern")
    elif args.discover:
        # Auto-discover files
        conversions = discover_existing_files(search_dir)
        print(f"Discovered {len(conversions)} files to convert")
    else:
        # Default: known files
        default_files = [
            (project_root / 'kullamannen_100M_2025_results.csv', 'kullamannen', 2025, '100M'),
        ]
        conversions = [(f, t, y, r) for f, t, y, r in default_files if f.exists()]
    
    if not conversions:
        print("No files to convert!")
        print(f"\nUsage:")
        print(f"  python3 convert_existing_data.py --discover")
        print(f"  python3 convert_existing_data.py --file kullamannen_100M_2025_results.csv")
        return
    
    for source_file, tenant, year, race_id in conversions:
        if source_file.exists():
            convert_existing_csv(source_file, tenant, year, race_id)
        else:
            print(f"  Skipping {source_file.name} (file not found)")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

