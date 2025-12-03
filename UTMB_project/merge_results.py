#!/usr/bin/env python3
"""
=============================================================================
UTMB Results Merger - Combine All Race CSVs into Master File
=============================================================================

This script reads all race CSV files from the data/ folder and combines them
into a single master_results.csv file for analysis.

USAGE:
------
    python3 merge_results.py

The script will:
1. Scan data/ folder for all CSV files
2. Parse metadata headers from each file
3. Add race metadata as columns to each row
4. Combine all data into master_results.csv

OUTPUT:
-------
master_results.csv with columns:
- Race metadata: race_name, distance_km, elevation_gain, elevation_loss, race_date, race_url
- Runner data: bib, name, country, age, sex, category, club, etc.
- Results: rank_scratch, race_time, utmb_score, etc.

=============================================================================
"""

import pandas as pd
from pathlib import Path
import argparse


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent / 'data'


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
                        # Clean up key names and values
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


def merge_all_results(data_dir: Path = None, output_file: Path = None, 
                      finishers_only: bool = False) -> pd.DataFrame:
    """
    Merge all race CSV files into a single DataFrame.
    
    Args:
        data_dir: Directory containing race CSV files (default: data/)
        output_file: Output file path (default: master_results.csv)
        finishers_only: If True, only include finishers in output
    
    Returns:
        Combined DataFrame
    """
    if data_dir is None:
        data_dir = get_data_dir()
    
    if output_file is None:
        output_file = Path(__file__).parent / 'master_results.csv'
    
    # Find all CSV files in data directory
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None
    
    print(f"Found {len(csv_files)} race files to merge:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Process each file
    all_dfs = []
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        
        try:
            df, metadata = read_csv_with_metadata(csv_file)
        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")
            continue
        
        # Add metadata columns
        df['race_name'] = metadata.get('race', 'Unknown')
        df['distance_km'] = pd.to_numeric(metadata.get('distance_km', None), errors='coerce')
        df['elevation_gain'] = pd.to_numeric(metadata.get('elevation_gain', None), errors='coerce')
        df['elevation_loss'] = pd.to_numeric(metadata.get('elevation_loss', None), errors='coerce')
        df['race_date'] = metadata.get('date', None)
        df['race_url'] = metadata.get('url', None)
        
        # Add source file for reference
        df['source_file'] = csv_file.name
        
        # Filter to finishers only if requested
        if finishers_only and 'is_finisher' in df.columns:
            original_count = len(df)
            df = df[df['is_finisher'] == True]
            print(f"  Loaded {len(df)} finishers (of {original_count} total)")
        else:
            print(f"  Loaded {len(df)} runners")
        
        all_dfs.append(df)
    
    if not all_dfs:
        print("\nNo data to merge!")
        return None
    
    # Combine all DataFrames
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Reorder columns: metadata first, then runner data
    metadata_cols = ['race_name', 'distance_km', 'elevation_gain', 'elevation_loss', 
                     'race_date', 'race_url', 'source_file']
    other_cols = [c for c in combined.columns if c not in metadata_cols]
    combined = combined[metadata_cols + other_cols]
    
    # Sort by race_date and rank
    if 'race_date' in combined.columns and 'rank_scratch' in combined.columns:
        combined = combined.sort_values(['race_date', 'rank_scratch'], 
                                        ascending=[False, True], 
                                        na_position='last')
    
    # Save to file
    combined.to_csv(output_file, index=False)
    
    print(f"\n{'=' * 60}")
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nTotal runners: {len(combined)}")
    print(f"Races included: {combined['race_name'].nunique()}")
    print(f"Output file: {output_file}")
    
    # Summary statistics
    if 'utmb_score' in combined.columns:
        scores = combined['utmb_score'].dropna()
        if len(scores) > 0:
            print(f"\nUTMB Score statistics:")
            print(f"  Count: {len(scores)}")
            print(f"  Mean: {scores.mean():.1f}")
            print(f"  Std: {scores.std():.1f}")
            print(f"  Min: {scores.min():.0f}")
            print(f"  Max: {scores.max():.0f}")
    
    # Race summary
    print("\nRaces in master file:")
    for race_name in combined['race_name'].unique():
        race_df = combined[combined['race_name'] == race_name]
        finishers = race_df['is_finisher'].sum() if 'is_finisher' in race_df.columns else len(race_df)
        scores = race_df['utmb_score'].notna().sum() if 'utmb_score' in race_df.columns else 0
        print(f"  {race_name}: {len(race_df)} runners, {finishers} finishers, {scores} with UTMB score")
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Merge all UTMB race results into a master file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing race CSV files (default: data/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: master_results.csv)')
    parser.add_argument('--finishers-only', action='store_true',
                        help='Only include finishers in output')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else None
    output_file = Path(args.output) if args.output else None
    
    merge_all_results(data_dir, output_file, args.finishers_only)


if __name__ == "__main__":
    main()

