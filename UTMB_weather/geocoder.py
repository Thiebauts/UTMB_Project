"""
Geocoder module for converting city names to latitude/longitude coordinates.
Uses Nominatim (OpenStreetMap) via geopy with caching to avoid repeated API calls.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
import ssl
import urllib3

# Suppress SSL warnings in problematic environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LocationGeocoder:
    """Geocoder with persistent caching for race start locations."""
    
    def __init__(self, cache_file: str = "data/locations_cache.json"):
        """
        Initialize the geocoder.
        
        Args:
            cache_file: Path to the JSON cache file for storing geocoded results
        """
        self.cache_file = Path(__file__).parent / cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        self.geolocator = Nominatim(user_agent="utmb_weather_analysis")
        
    def _load_cache(self) -> dict:
        """Load the geocoding cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self) -> None:
        """Save the geocoding cache to disk."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def _make_cache_key(self, city: str, country: str) -> str:
        """Create a unique cache key for a location."""
        city = city or ""
        country = country or ""
        return f"{city.lower().strip()}|{country.lower().strip()}"
    
    def geocode(
        self, 
        city: str, 
        country: str, 
        max_retries: int = 3
    ) -> Optional[Tuple[float, float]]:
        """
        Get latitude and longitude for a city.
        
        Args:
            city: City/place name (e.g., "Chamonix")
            country: Country name or code (e.g., "France" or "FR")
            max_retries: Number of retries on timeout
            
        Returns:
            Tuple of (latitude, longitude) or None if geocoding fails
        """
        # Handle None/empty values
        if not city:
            return None
        
        cache_key = self._make_cache_key(city, country)
        
        # Check cache first
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if cached is not None:
                return (cached['lat'], cached['lon'])
            return None
        
        # Build query string
        if country:
            query = f"{city}, {country}"
        else:
            query = city
        
        for attempt in range(max_retries):
            try:
                # Rate limiting - Nominatim requires 1 request per second
                time.sleep(1.1)
                
                location = self.geolocator.geocode(query, timeout=10)
                
                if location:
                    result = {'lat': location.latitude, 'lon': location.longitude}
                    self.cache[cache_key] = result
                    self._save_cache()
                    return (location.latitude, location.longitude)
                else:
                    # Location not found, cache as None
                    self.cache[cache_key] = None
                    self._save_cache()
                    print(f"Warning: Could not geocode '{query}'")
                    return None
                    
            except GeocoderTimedOut:
                if attempt < max_retries - 1:
                    print(f"Timeout geocoding '{query}', retrying...")
                    time.sleep(2)
                    continue
                print(f"Error: Geocoding timeout for '{query}' after {max_retries} attempts")
                return None
                
            except (GeocoderServiceError, GeocoderUnavailable, ssl.SSLError) as e:
                print(f"Error: Geocoding service error for '{query}': {e}")
                # Cache as None to avoid repeated failures
                self.cache[cache_key] = None
                self._save_cache()
                return None
            except Exception as e:
                # Catch any other SSL-related errors
                if 'SSL' in str(e) or 'ssl' in str(e) or 'certificate' in str(e).lower():
                    print(f"Error: SSL error for '{query}': {e}")
                    return None
                raise
        
        return None
    
    def geocode_batch(
        self, 
        locations: list[dict]
    ) -> dict[str, Optional[Tuple[float, float]]]:
        """
        Geocode multiple locations.
        
        Args:
            locations: List of dicts with 'city' and 'country' keys
            
        Returns:
            Dictionary mapping cache keys to coordinates
        """
        results = {}
        
        for loc in locations:
            city = loc.get('city', '')
            country = loc.get('country', '')
            
            if not city or not country:
                continue
                
            cache_key = self._make_cache_key(city, country)
            coords = self.geocode(city, country)
            results[cache_key] = coords
            
        return results
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the cache."""
        total = len(self.cache)
        successful = sum(1 for v in self.cache.values() if v is not None)
        failed = total - successful
        
        return {
            'total_entries': total,
            'successful': successful,
            'failed': failed
        }


if __name__ == "__main__":
    # Test the geocoder
    geocoder = LocationGeocoder()
    
    test_locations = [
        ("Chamonix", "France"),
        ("Barr", "France"),
        ("Båstad", "Sweden"),
        ("Queenstown", "New Zealand"),
    ]
    
    print("Testing geocoder...")
    for city, country in test_locations:
        coords = geocoder.geocode(city, country)
        if coords:
            print(f"  {city}, {country}: {coords[0]:.4f}, {coords[1]:.4f}")
        else:
            print(f"  {city}, {country}: Not found")
    
    print(f"\nCache stats: {geocoder.get_cache_stats()}")

