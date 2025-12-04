# UTMB Weather Analysis

Analyze historical weather conditions during UTMB World Series races and quantify how "good" or "bad" the weather was using a comprehensive scoring system.

## Overview

This tool:
1. Extracts race start times and locations from `events.json`
2. Geocodes start locations to coordinates using Nominatim/OpenStreetMap
3. Fetches historical weather data from Open-Meteo (free, no API key required)
4. Computes a weather quality score (0-100) for each race
5. Exports results to CSV

## Installation

```bash
cd UTMB_weather
pip install -r requirements.txt
```

## Usage

Run the main pipeline:

```bash
python3 main.py
```

This will:
- Read race data from `../UTMB_project/events.json`
- Geocode all unique start locations (cached in `data/locations_cache.json`)
- Fetch weather for each historical race (24-hour window from start)
- Output results to `data/weather_results.csv`

## Weather Scoring System

### Score Scale
- **0-100** where 100 = perfect conditions, 0 = extreme conditions

### Factors and Weights

| Factor | Weight | Optimal Range | Notes |
|--------|--------|---------------|-------|
| Temperature | 30% | 8-15°C | Penalty for too hot or too cold |
| Precipitation | 35% | 0 mm | Linear penalty per mm of rain/snow |
| Wind | 20% | < 15 km/h | Uses max wind speed (gusts) |
| Humidity | 15% | 40-70% | Extremes cause dehydration or heat stress |

### Weather Categories

| Score Range | Category | Description |
|-------------|----------|-------------|
| 80-100 | Excellent | Ideal running conditions |
| 60-79 | Good | Minor weather challenges |
| 40-59 | Moderate | Noticeable impact on performance |
| 20-39 | Challenging | Difficult conditions affecting safety |
| 0-19 | Extreme | Dangerous, races may be altered |

## Output Format

The CSV output includes:

| Column | Description |
|--------|-------------|
| event_name | UTMB event name |
| race_name | Specific race name |
| race_code | Race identifier code |
| year | Year of the race |
| start_date | Start date (YYYY-MM-DD) |
| start_time | Start time (HH:MM) |
| location | Start city/place |
| country | Country name |
| latitude | Start location latitude |
| longitude | Start location longitude |
| distance_km | Race distance |
| elevation_gain | Total elevation gain (m) |
| avg_temp_c | Average temperature during race window |
| min_temp_c | Minimum temperature |
| max_temp_c | Maximum temperature |
| total_precip_mm | Total precipitation (24h) |
| max_wind_kmh | Maximum wind speed |
| avg_wind_kmh | Average wind speed |
| avg_humidity_pct | Average relative humidity |
| temp_score | Temperature sub-score (0-100) |
| precip_score | Precipitation sub-score (0-100) |
| wind_score | Wind sub-score (0-100) |
| humidity_score | Humidity sub-score (0-100) |
| weather_score | Overall composite score (0-100) |
| weather_category | Category label |

## Data Sources

- **Weather Data**: [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api)
  - Free, no API key required
  - Historical data available from 1940 to ~5 days ago
  - Hourly resolution
  
- **Geocoding**: [Nominatim/OpenStreetMap](https://nominatim.org/)
  - Free, no API key required
  - Results are cached locally to minimize API calls

## Files

```
UTMB_weather/
├── main.py              # Main pipeline orchestrator
├── geocoder.py          # City name to coordinates conversion
├── weather_fetcher.py   # Open-Meteo API client
├── weather_scorer.py    # Weather quality scoring logic
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── data/
    ├── locations_cache.json  # Cached geocoding results
    └── weather_results.csv   # Output with weather scores
```

## Notes

- **Rate Limiting**: Nominatim requires 1 request/second. The geocoder automatically handles this.
- **Historical Data Only**: Races with future dates are skipped (no forecast data).
- **Caching**: Geocoding results are cached to speed up subsequent runs.
- **24-Hour Window**: Weather is analyzed for 24 hours starting from race start time.

