"""
Weather fetcher module for retrieving historical weather data from Open-Meteo API.
Fetches hourly weather data and aggregates over a 24-hour race window.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import urllib3

# Suppress SSL warnings in problematic environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class WeatherData:
    """Container for aggregated weather data over a race period."""
    avg_temp_c: float
    min_temp_c: float
    max_temp_c: float
    total_precip_mm: float
    max_wind_kmh: float
    avg_wind_kmh: float
    avg_humidity_pct: float
    hours_analyzed: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for easy DataFrame creation."""
        return {
            'avg_temp_c': round(self.avg_temp_c, 1),
            'min_temp_c': round(self.min_temp_c, 1),
            'max_temp_c': round(self.max_temp_c, 1),
            'total_precip_mm': round(self.total_precip_mm, 1),
            'max_wind_kmh': round(self.max_wind_kmh, 1),
            'avg_wind_kmh': round(self.avg_wind_kmh, 1),
            'avg_humidity_pct': round(self.avg_humidity_pct, 1),
            'hours_analyzed': self.hours_analyzed
        }


class WeatherFetcher:
    """
    Fetches historical weather data from Open-Meteo API.
    
    Open-Meteo is a free, open-source weather API that provides historical
    weather data without requiring an API key.
    """
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # Weather variables to fetch
    HOURLY_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m", 
        "precipitation",
        "wind_speed_10m"
    ]
    
    def __init__(self, verify_ssl: bool = True):
        """
        Initialize the weather fetcher.
        
        Args:
            verify_ssl: Whether to verify SSL certificates (set False for testing)
        """
        self.session = requests.Session()
        self.verify_ssl = verify_ssl
        
        # Configure retries
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        start_datetime: datetime,
        hours: int = 24
    ) -> Optional[WeatherData]:
        """
        Fetch weather data for a location and time period.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            start_datetime: Race start datetime (timezone-aware or naive UTC)
            hours: Number of hours to analyze (default 24)
            
        Returns:
            WeatherData object with aggregated metrics, or None on error
        """
        # Calculate date range
        end_datetime = start_datetime + timedelta(hours=hours)
        
        # Open-Meteo uses date strings (not datetime)
        start_date = start_datetime.strftime("%Y-%m-%d")
        end_date = end_datetime.strftime("%Y-%m-%d")
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.HOURLY_VARIABLES),
            "timezone": "UTC"
        }
        
        try:
            response = self.session.get(
                self.BASE_URL, 
                params=params, 
                timeout=30,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                print(f"Warning: No hourly data in response for {latitude}, {longitude}")
                return None
            
            return self._aggregate_hourly_data(
                data["hourly"], 
                start_datetime, 
                hours
            )
            
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Error parsing weather response: {e}")
            return None
    
    def _aggregate_hourly_data(
        self, 
        hourly: dict, 
        start_datetime: datetime,
        hours: int
    ) -> Optional[WeatherData]:
        """
        Aggregate hourly data into race-period statistics.
        
        Args:
            hourly: Dictionary with hourly arrays from API
            start_datetime: Race start time
            hours: Number of hours to include
            
        Returns:
            WeatherData with aggregated metrics
        """
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        precip = hourly.get("precipitation", [])
        wind = hourly.get("wind_speed_10m", [])
        
        if not times:
            return None
        
        # Find indices for our time window
        # Times are in format "2023-05-21T05:00"
        start_hour = start_datetime.replace(minute=0, second=0, microsecond=0)
        
        selected_temps = []
        selected_humidity = []
        selected_precip = []
        selected_wind = []
        
        for i, time_str in enumerate(times):
            try:
                time_dt = datetime.fromisoformat(time_str)
                
                # Check if this hour is within our window
                if start_hour <= time_dt < start_hour + timedelta(hours=hours):
                    if i < len(temps) and temps[i] is not None:
                        selected_temps.append(temps[i])
                    if i < len(humidity) and humidity[i] is not None:
                        selected_humidity.append(humidity[i])
                    if i < len(precip) and precip[i] is not None:
                        selected_precip.append(precip[i])
                    if i < len(wind) and wind[i] is not None:
                        selected_wind.append(wind[i])
                        
            except ValueError:
                continue
        
        # Need at least some data
        if not selected_temps:
            print("Warning: No temperature data found in time window")
            return None
        
        return WeatherData(
            avg_temp_c=sum(selected_temps) / len(selected_temps),
            min_temp_c=min(selected_temps),
            max_temp_c=max(selected_temps),
            total_precip_mm=sum(selected_precip) if selected_precip else 0,
            max_wind_kmh=max(selected_wind) if selected_wind else 0,
            avg_wind_kmh=sum(selected_wind) / len(selected_wind) if selected_wind else 0,
            avg_humidity_pct=sum(selected_humidity) / len(selected_humidity) if selected_humidity else 50,
            hours_analyzed=len(selected_temps)
        )
    
    def check_data_availability(self, date: datetime) -> bool:
        """
        Check if historical data is available for a given date.
        
        Open-Meteo historical data has a delay of about 5 days from current date.
        
        Args:
            date: Date to check
            
        Returns:
            True if data should be available
        """
        # Historical data typically available up to 5 days ago
        cutoff = datetime.now() - timedelta(days=5)
        return date < cutoff


if __name__ == "__main__":
    # Test the weather fetcher
    fetcher = WeatherFetcher()
    
    # Test with Chamonix coordinates and a past UTMB date
    lat, lon = 45.9237, 6.8694  # Chamonix
    test_date = datetime(2023, 8, 25, 18, 0)  # UTMB 2023 start
    
    print(f"Fetching weather for Chamonix on {test_date}...")
    weather = fetcher.fetch_weather(lat, lon, test_date)
    
    if weather:
        print(f"Weather data retrieved:")
        for key, value in weather.to_dict().items():
            print(f"  {key}: {value}")
    else:
        print("Failed to fetch weather data")

