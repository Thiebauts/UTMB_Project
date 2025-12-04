#!/usr/bin/env python3
"""
Data preparation module for UTMB Score prediction.

Loads race data from master_results.csv, filters for records with valid UTMB scores,
cleans the data, and provides train/test splits.
"""

from __future__ import annotations
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# Define paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MASTER_CSV = PROJECT_ROOT / "UTMB_project" / "master_results.csv"

# Feature columns and target
FEATURE_COLS = ["distance_km", "elevation_gain", "race_time_seconds"]
TARGET_COL = "utmb_score"


def load_raw_data(csv_path: Path = MASTER_CSV) -> pd.DataFrame:
    """Load the master results CSV file."""
    # Skip comment lines that start with #
    df = pd.read_csv(csv_path, comment="#")
    print(f"Loaded {len(df):,} total records from {csv_path.name}")
    return df


def filter_valid_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to records that have valid UTMB scores and are usable for training.
    
    Criteria:
    - utmb_score is not null/empty
    - is_finisher is True
    - race_time_seconds > 0
    - distance_km > 0
    - elevation_gain >= 0
    """
    initial_count = len(df)
    
    # Convert utmb_score to numeric, coercing errors to NaN
    df["utmb_score"] = pd.to_numeric(df["utmb_score"], errors="coerce")
    
    # Filter for valid records
    mask = (
        df["utmb_score"].notna() &
        (df["utmb_score"] > 0) &
        (df["is_finisher"] == True) &
        (df["race_time_seconds"] > 0) &
        (df["distance_km"] > 0) &
        (df["elevation_gain"] >= 0)
    )
    
    df_filtered = df[mask].copy()
    
    print(f"Filtered to {len(df_filtered):,} valid records ({len(df_filtered)/initial_count*100:.1f}%)")
    print(f"  - Removed {initial_count - len(df_filtered):,} records")
    
    return df_filtered


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target from the dataframe.
    
    Returns:
        X: DataFrame with feature columns
        y: Series with target values
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    
    # Ensure numeric types
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    
    # Drop any rows with NaN in features
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Prepared {len(X):,} samples with {len(FEATURE_COLS)} features")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Target: {TARGET_COL}")
    
    return X, y


def get_train_test_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train/Test split: {len(X_train):,} train, {len(X_test):,} test ({test_size*100:.0f}% test)")
    
    return X_train, X_test, y_train, y_test


def load_and_prepare_data(
    csv_path: Path = MASTER_CSV,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Main function to load, filter, and prepare all data.
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test, and metadata
    """
    print("=" * 60)
    print("UTMB Score Prediction - Data Preparation")
    print("=" * 60)
    
    # Load and process
    df = load_raw_data(csv_path)
    df_valid = filter_valid_records(df)
    X, y = prepare_features(df_valid)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size, random_state)
    
    # Print summary statistics
    print("\n" + "-" * 40)
    print("Data Summary:")
    print("-" * 40)
    print(f"Target (utmb_score) statistics:")
    print(f"  Min:  {y.min():.1f}")
    print(f"  Max:  {y.max():.1f}")
    print(f"  Mean: {y.mean():.1f}")
    print(f"  Std:  {y.std():.1f}")
    
    print(f"\nFeature statistics:")
    for col in FEATURE_COLS:
        print(f"  {col}:")
        print(f"    Range: [{X[col].min():.1f}, {X[col].max():.1f}]")
        print(f"    Mean:  {X[col].mean():.1f}")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": FEATURE_COLS,
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


if __name__ == "__main__":
    # Run data preparation and print summary
    data = load_and_prepare_data()
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"Ready for training with {data['n_train']:,} samples")

