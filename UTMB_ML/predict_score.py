#!/usr/bin/env python3
"""
UTMB Score prediction CLI tool.

Predict the UTMB score for a given race performance based on:
- Distance (km)
- Elevation gain (m)
- Finish time (HH:MM:SS or seconds)

Usage:
    python3 predict_score.py --distance 100 --elevation 4000 --time "10:30:00"
    python3 predict_score.py -d 173 -e 10000 -t "24:15:30"
    python3 predict_score.py --distance 50 --elevation 2500 --time-seconds 18000
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Check for xgboost (required for loading the model)
try:
    import xgboost
except ImportError:
    print("❌ Error: xgboost is not installed.")
    print("\nPlease install it using:")
    print("  pip install xgboost")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# Paths
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_MODEL = "xgboost_utmb_score"


def parse_time_string(time_str: str) -> float:
    """
    Parse time string in various formats to seconds.
    
    Supported formats:
    - "HH:MM:SS" (e.g., "10:30:45")
    - "H:MM:SS" (e.g., "9:30:45")
    - "MM:SS" (e.g., "45:30" -> 45 min 30 sec)
    - Decimal hours (e.g., "10.5" -> 10 hours 30 min)
    
    Returns time in seconds.
    """
    time_str = time_str.strip()
    
    # Check for HH:MM:SS or H:MM:SS format
    match = re.match(r"^(\d{1,3}):(\d{2}):(\d{2})$", time_str)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    
    # Check for MM:SS format
    match = re.match(r"^(\d{1,3}):(\d{2})$", time_str)
    if match:
        minutes, seconds = map(int, match.groups())
        return minutes * 60 + seconds
    
    # Check for decimal hours (e.g., "10.5")
    try:
        hours = float(time_str)
        return hours * 3600
    except ValueError:
        pass
    
    raise ValueError(
        f"Invalid time format: '{time_str}'. "
        "Use 'HH:MM:SS' (e.g., '10:30:00') or decimal hours (e.g., '10.5')"
    )


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_model(model_name: str = DEFAULT_MODEL):
    """Load the trained model."""
    model_path = MODELS_DIR / f"{model_name}.joblib"
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("\nPlease train the model first by running:")
        print("  python3 train_model.py")
        sys.exit(1)
    
    return joblib.load(model_path)


def predict_score(
    model,
    distance_km: float,
    elevation_gain: float,
    time_seconds: float
) -> float:
    """
    Predict UTMB score for given race parameters.
    
    Args:
        model: Trained XGBoost model
        distance_km: Race distance in kilometers
        elevation_gain: Elevation gain in meters
        time_seconds: Finish time in seconds
    
    Returns:
        Predicted UTMB score
    """
    # Create feature array in the same order as training
    features = pd.DataFrame({
        "distance_km": [distance_km],
        "elevation_gain": [elevation_gain],
        "race_time_seconds": [time_seconds]
    })
    
    prediction = model.predict(features)[0]
    
    return prediction


def print_prediction(
    distance_km: float,
    elevation_gain: float,
    time_seconds: float,
    predicted_score: float
):
    """Print formatted prediction result."""
    print("\n" + "=" * 50)
    print("🏃 UTMB Score Prediction")
    print("=" * 50)
    print(f"\n📊 Race Parameters:")
    print(f"   Distance:       {distance_km:.1f} km")
    print(f"   Elevation Gain: {elevation_gain:.0f} m")
    print(f"   Finish Time:    {format_time(time_seconds)} ({time_seconds:.0f}s)")
    print(f"\n🎯 Predicted UTMB Score: {predicted_score:.0f}")
    print("=" * 50 + "\n")


def interactive_mode(model):
    """Run in interactive mode, prompting for input."""
    print("\n" + "=" * 50)
    print("🏃 UTMB Score Predictor - Interactive Mode")
    print("=" * 50)
    print("Enter race parameters (or 'q' to quit)\n")
    
    while True:
        try:
            # Get distance
            distance_input = input("Distance (km): ").strip()
            if distance_input.lower() == 'q':
                break
            distance_km = float(distance_input)
            
            # Get elevation
            elevation_input = input("Elevation gain (m): ").strip()
            if elevation_input.lower() == 'q':
                break
            elevation_gain = float(elevation_input)
            
            # Get time
            time_input = input("Finish time (HH:MM:SS): ").strip()
            if time_input.lower() == 'q':
                break
            time_seconds = parse_time_string(time_input)
            
            # Predict
            score = predict_score(model, distance_km, elevation_gain, time_seconds)
            
            print(f"\n🎯 Predicted UTMB Score: {score:.0f}\n")
            print("-" * 30 + "\n")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}\n")
        except KeyboardInterrupt:
            print("\n")
            break
    
    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Predict UTMB score for a given race performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 predict_score.py --distance 100 --elevation 4000 --time "10:30:00"
  python3 predict_score.py -d 173 -e 10000 -t "24:15:30"
  python3 predict_score.py --distance 50 --elevation 2500 --time-seconds 18000
  python3 predict_score.py --interactive
        """
    )
    
    parser.add_argument(
        "-d", "--distance", type=float,
        help="Race distance in kilometers"
    )
    parser.add_argument(
        "-e", "--elevation", type=float,
        help="Elevation gain in meters"
    )
    parser.add_argument(
        "-t", "--time", type=str,
        help="Finish time in HH:MM:SS format (e.g., '10:30:00')"
    )
    parser.add_argument(
        "--time-seconds", type=float,
        help="Finish time in seconds (alternative to --time)"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model)
        return
    
    # Check required arguments for single prediction
    if args.distance is None or args.elevation is None:
        if args.distance is None and args.elevation is None and args.time is None:
            # No arguments provided, show help
            parser.print_help()
            print("\n💡 Tip: Use --interactive for interactive mode")
            return
        else:
            parser.error("--distance and --elevation are required for prediction")
    
    if args.time is None and args.time_seconds is None:
        parser.error("Either --time or --time-seconds is required")
    
    # Parse time
    if args.time_seconds is not None:
        time_seconds = args.time_seconds
    else:
        try:
            time_seconds = parse_time_string(args.time)
        except ValueError as e:
            parser.error(str(e))
    
    # Validate inputs
    if args.distance <= 0:
        parser.error("Distance must be positive")
    if args.elevation < 0:
        parser.error("Elevation gain cannot be negative")
    if time_seconds <= 0:
        parser.error("Time must be positive")
    
    # Make prediction
    score = predict_score(model, args.distance, args.elevation, time_seconds)
    
    # Print result
    print_prediction(args.distance, args.elevation, time_seconds, score)


if __name__ == "__main__":
    main()

