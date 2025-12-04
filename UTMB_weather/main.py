"""
Main pipeline for UTMB Weather Analysis.

This script orchestrates the full weather analysis workflow:
1. Parse events.json to extract race information
2. Geocode start locations to coordinates
3. Fetch historical weather data from Open-Meteo
4. Compute weather quality scores
5. Export results to CSV
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from geocoder import LocationGeocoder
from weather_fetcher import WeatherFetcher, WeatherData
from weather_scorer import WeatherScorer


def load_events(events_path: str) -> dict:
    """
    Load events data from JSON file.
    
    Args:
        events_path: Path to events.json
        
    Returns:
        Dictionary with events data
    """
    with open(events_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_race_datetime(date_str: str) -> Optional[datetime]:
    """
    Parse race start datetime from ISO format string.
    
    Args:
        date_str: ISO format datetime string (e.g., "2023-05-21T05:49:59.000Z")
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    try:
        # Handle various ISO formats
        date_str = date_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(date_str)
        # Convert to naive UTC for consistency
        return dt.replace(tzinfo=None)
    except ValueError:
        print(f"Warning: Could not parse date '{date_str}'")
        return None


def extract_races(events_data: dict) -> list[dict]:
    """
    Extract all races from events data into a flat list.
    
    Args:
        events_data: Full events.json data
        
    Returns:
        List of race dictionaries with relevant fields
    """
    races = []
    
    events = events_data.get('events', {})
    
    for event_key, event_data in events.items():
        event_name = event_data.get('event_name', event_key)
        
        years = event_data.get('years', {})
        for year, year_data in years.items():
            event_info = year_data.get('event_info', {})
            country = event_info.get('country', '')
            country_code = event_info.get('country_code', '')
            
            race_list = year_data.get('races', {})
            for race_code, race_data in race_list.items():
                start_date_str = race_data.get('start_date', '')
                start_datetime = parse_race_datetime(start_date_str)
                
                # Skip races without valid date or from the future
                if not start_datetime:
                    continue
                    
                # Skip future races (historical data won't be available)
                if start_datetime > datetime.now():
                    continue
                
                races.append({
                    'event_key': event_key,
                    'event_name': event_name,
                    'race_code': race_code,
                    'race_name': race_data.get('name', race_code),
                    'year': int(year),
                    'start_datetime': start_datetime,
                    'start_place': race_data.get('start_place', ''),
                    'country': country,
                    'country_code': country_code,
                    'distance_km': race_data.get('distance_km', 0),
                    'elevation_gain': race_data.get('elevation_gain', 0),
                    'utmb_category': race_data.get('utmb_category', '')
                })
    
    return races


def process_races(
    races: list[dict],
    geocoder: LocationGeocoder,
    fetcher: WeatherFetcher,
    scorer: WeatherScorer,
    hours_window: int = 24
) -> list[dict]:
    """
    Process all races: geocode, fetch weather, compute scores.
    
    Args:
        races: List of race dictionaries
        geocoder: LocationGeocoder instance
        fetcher: WeatherFetcher instance  
        scorer: WeatherScorer instance
        hours_window: Hours of weather data to analyze
        
    Returns:
        List of result dictionaries
    """
    results = []
    total = len(races)
    
    print(f"\nProcessing {total} races...")
    
    for i, race in enumerate(races, 1):
        city = race['start_place']
        country = race['country']
        
        print(f"[{i}/{total}] {race['event_name']} - {race['race_name']} ({race['year']})", end="")
        
        # Skip if no location
        if not city:
            print(" - Skipped (no location)")
            continue
        
        # Geocode location
        coords = geocoder.geocode(city, country)
        if not coords:
            print(f" - Skipped (geocoding failed for {city})")
            continue
        
        lat, lon = coords
        
        # Fetch weather
        weather = fetcher.fetch_weather(
            lat, lon, 
            race['start_datetime'],
            hours=hours_window
        )
        
        if not weather:
            print(" - Skipped (weather fetch failed)")
            continue
        
        # Compute score
        score = scorer.compute_score(weather)
        
        # Build result record
        result = {
            'event_name': race['event_name'],
            'race_name': race['race_name'],
            'race_code': race['race_code'],
            'year': race['year'],
            'start_date': race['start_datetime'].strftime('%Y-%m-%d'),
            'start_time': race['start_datetime'].strftime('%H:%M'),
            'location': city,
            'country': country,
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'distance_km': race['distance_km'],
            'elevation_gain': race['elevation_gain'],
            'utmb_category': race['utmb_category'],
            **weather.to_dict(),
            **score.to_dict()
        }
        
        results.append(result)
        print(f" - Score: {score.overall_score:.1f} ({score.category})")
    
    return results


def save_results(results: list[dict], output_path: str) -> None:
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path for output CSV
    """
    if not results:
        print("No results to save.")
        return
    
    df = pd.DataFrame(results)
    
    # Sort by event, year, race
    df = df.sort_values(['event_name', 'year', 'race_name'])
    
    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total races analyzed: {len(df)}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Weather Summary by Category:")
    print(df['weather_category'].value_counts().to_string())
    
    print("\nAverage Weather Score by Event:")
    event_scores = df.groupby('event_name')['weather_score'].mean().sort_values(ascending=False)
    for event, score in event_scores.head(10).items():
        print(f"  {event}: {score:.1f}")


def main():
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent
    events_path = base_dir.parent / "UTMB_project" / "events.json"
    output_path = base_dir / "data" / "weather_results.csv"
    
    # Configuration
    hours_window = 24  # 24-hour weather window
    
    print("="*60)
    print("UTMB Weather Analysis Pipeline")
    print("="*60)
    
    # Check events file exists
    if not events_path.exists():
        print(f"Error: Events file not found at {events_path}")
        sys.exit(1)
    
    print(f"\nLoading events from: {events_path}")
    events_data = load_events(events_path)
    
    # Extract races
    races = extract_races(events_data)
    print(f"Found {len(races)} historical races to analyze")
    
    if not races:
        print("No races found to analyze.")
        sys.exit(0)
    
    # Initialize components
    # Note: verify_ssl=False is used as a workaround for some environments with SSL issues
    geocoder = LocationGeocoder()
    fetcher = WeatherFetcher(verify_ssl=False)
    scorer = WeatherScorer()
    
    # Process races
    results = process_races(
        races, 
        geocoder, 
        fetcher, 
        scorer,
        hours_window=hours_window
    )
    
    # Save results
    save_results(results, output_path)
    
    # Print geocoder cache stats
    print(f"\nGeocoder cache: {geocoder.get_cache_stats()}")


if __name__ == "__main__":
    main()

