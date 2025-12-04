"""
Weather scoring module for quantifying race weather conditions.

Scoring System (0-100 scale, 100 = perfect conditions):
- Temperature: 30% weight - Optimal at 8-15°C
- Precipitation: 35% weight - 0mm = 100, heavy rain = low score  
- Wind: 20% weight - Calm = 100, gale = low score
- Humidity: 15% weight - 40-70% = optimal
"""

from dataclasses import dataclass
from typing import Optional
from weather_fetcher import WeatherData


@dataclass
class WeatherScore:
    """Container for weather scores."""
    temp_score: float
    precip_score: float
    wind_score: float
    humidity_score: float
    overall_score: float
    category: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'temp_score': round(self.temp_score, 1),
            'precip_score': round(self.precip_score, 1),
            'wind_score': round(self.wind_score, 1),
            'humidity_score': round(self.humidity_score, 1),
            'weather_score': round(self.overall_score, 1),
            'weather_category': self.category
        }


class WeatherScorer:
    """
    Calculates weather quality scores for trail running conditions.
    
    All individual scores are on a 0-100 scale where:
    - 100 = Perfect/optimal conditions
    - 0 = Extreme/dangerous conditions
    """
    
    # Scoring weights
    WEIGHT_TEMP = 0.30
    WEIGHT_PRECIP = 0.35
    WEIGHT_WIND = 0.20
    WEIGHT_HUMIDITY = 0.15
    
    # Temperature parameters (Celsius)
    TEMP_OPTIMAL_LOW = 8.0
    TEMP_OPTIMAL_HIGH = 15.0
    TEMP_EXTREME_COLD = -10.0
    TEMP_EXTREME_HOT = 35.0
    
    # Precipitation parameters (mm over 24h)
    PRECIP_LIGHT = 5.0      # Up to 5mm is light rain
    PRECIP_MODERATE = 20.0  # Up to 20mm is moderate
    PRECIP_HEAVY = 50.0     # 50mm+ is heavy
    
    # Wind parameters (km/h)
    WIND_CALM = 15.0        # Up to 15 km/h is calm
    WIND_MODERATE = 30.0    # Up to 30 km/h is moderate
    WIND_STRONG = 50.0      # Up to 50 km/h is strong
    WIND_EXTREME = 80.0     # 80+ km/h is dangerous
    
    # Humidity parameters (%)
    HUMIDITY_OPTIMAL_LOW = 40.0
    HUMIDITY_OPTIMAL_HIGH = 70.0
    
    # Category thresholds
    CATEGORIES = [
        (80, "Excellent"),
        (60, "Good"),
        (40, "Moderate"),
        (20, "Challenging"),
        (0, "Extreme")
    ]
    
    def score_temperature(self, avg_temp: float) -> float:
        """
        Score temperature conditions.
        
        Optimal range: 8-15°C (returns 100)
        Score decreases as temperature deviates from optimal range.
        
        Args:
            avg_temp: Average temperature in Celsius
            
        Returns:
            Score from 0-100
        """
        if self.TEMP_OPTIMAL_LOW <= avg_temp <= self.TEMP_OPTIMAL_HIGH:
            return 100.0
        
        if avg_temp < self.TEMP_OPTIMAL_LOW:
            # Cold penalty
            deviation = self.TEMP_OPTIMAL_LOW - avg_temp
            range_size = self.TEMP_OPTIMAL_LOW - self.TEMP_EXTREME_COLD
            penalty = (deviation / range_size) * 100
            return max(0.0, 100.0 - penalty)
        else:
            # Heat penalty
            deviation = avg_temp - self.TEMP_OPTIMAL_HIGH
            range_size = self.TEMP_EXTREME_HOT - self.TEMP_OPTIMAL_HIGH
            penalty = (deviation / range_size) * 100
            return max(0.0, 100.0 - penalty)
    
    def score_precipitation(self, total_precip_mm: float) -> float:
        """
        Score precipitation conditions.
        
        0mm = 100 (perfect)
        Score decreases with increasing precipitation.
        
        Args:
            total_precip_mm: Total precipitation in mm over 24h
            
        Returns:
            Score from 0-100
        """
        if total_precip_mm <= 0:
            return 100.0
        
        if total_precip_mm <= self.PRECIP_LIGHT:
            # Light rain: 100 -> 80
            return 100.0 - (total_precip_mm / self.PRECIP_LIGHT) * 20
        
        if total_precip_mm <= self.PRECIP_MODERATE:
            # Moderate rain: 80 -> 50
            excess = total_precip_mm - self.PRECIP_LIGHT
            range_size = self.PRECIP_MODERATE - self.PRECIP_LIGHT
            return 80.0 - (excess / range_size) * 30
        
        if total_precip_mm <= self.PRECIP_HEAVY:
            # Heavy rain: 50 -> 20
            excess = total_precip_mm - self.PRECIP_MODERATE
            range_size = self.PRECIP_HEAVY - self.PRECIP_MODERATE
            return 50.0 - (excess / range_size) * 30
        
        # Extreme rain: 20 -> 0
        excess = total_precip_mm - self.PRECIP_HEAVY
        return max(0.0, 20.0 - excess * 0.4)
    
    def score_wind(self, max_wind_kmh: float) -> float:
        """
        Score wind conditions.
        
        Uses max wind speed as gusts are most impactful for runners.
        
        Args:
            max_wind_kmh: Maximum wind speed in km/h
            
        Returns:
            Score from 0-100
        """
        if max_wind_kmh <= self.WIND_CALM:
            return 100.0
        
        if max_wind_kmh <= self.WIND_MODERATE:
            # Moderate wind: 100 -> 70
            excess = max_wind_kmh - self.WIND_CALM
            range_size = self.WIND_MODERATE - self.WIND_CALM
            return 100.0 - (excess / range_size) * 30
        
        if max_wind_kmh <= self.WIND_STRONG:
            # Strong wind: 70 -> 40
            excess = max_wind_kmh - self.WIND_MODERATE
            range_size = self.WIND_STRONG - self.WIND_MODERATE
            return 70.0 - (excess / range_size) * 30
        
        if max_wind_kmh <= self.WIND_EXTREME:
            # Very strong wind: 40 -> 10
            excess = max_wind_kmh - self.WIND_STRONG
            range_size = self.WIND_EXTREME - self.WIND_STRONG
            return 40.0 - (excess / range_size) * 30
        
        # Extreme/dangerous wind
        return max(0.0, 10.0 - (max_wind_kmh - self.WIND_EXTREME) * 0.2)
    
    def score_humidity(self, avg_humidity_pct: float) -> float:
        """
        Score humidity conditions.
        
        Optimal range: 40-70%
        Low humidity: dehydration risk
        High humidity: heat stress risk
        
        Args:
            avg_humidity_pct: Average relative humidity percentage
            
        Returns:
            Score from 0-100
        """
        if self.HUMIDITY_OPTIMAL_LOW <= avg_humidity_pct <= self.HUMIDITY_OPTIMAL_HIGH:
            return 100.0
        
        if avg_humidity_pct < self.HUMIDITY_OPTIMAL_LOW:
            # Low humidity penalty (dehydration risk)
            deviation = self.HUMIDITY_OPTIMAL_LOW - avg_humidity_pct
            # 0% humidity = score of 50
            return max(50.0, 100.0 - deviation * 1.25)
        else:
            # High humidity penalty (heat stress)
            deviation = avg_humidity_pct - self.HUMIDITY_OPTIMAL_HIGH
            # 100% humidity = score of 40
            return max(40.0, 100.0 - deviation * 2.0)
    
    def get_category(self, score: float) -> str:
        """
        Get weather category based on overall score.
        
        Args:
            score: Overall weather score (0-100)
            
        Returns:
            Category string
        """
        for threshold, category in self.CATEGORIES:
            if score >= threshold:
                return category
        return "Extreme"
    
    def compute_score(self, weather: WeatherData) -> WeatherScore:
        """
        Compute overall weather score from weather data.
        
        Args:
            weather: WeatherData object with metrics
            
        Returns:
            WeatherScore object with individual and overall scores
        """
        temp_score = self.score_temperature(weather.avg_temp_c)
        precip_score = self.score_precipitation(weather.total_precip_mm)
        wind_score = self.score_wind(weather.max_wind_kmh)
        humidity_score = self.score_humidity(weather.avg_humidity_pct)
        
        overall = (
            temp_score * self.WEIGHT_TEMP +
            precip_score * self.WEIGHT_PRECIP +
            wind_score * self.WEIGHT_WIND +
            humidity_score * self.WEIGHT_HUMIDITY
        )
        
        category = self.get_category(overall)
        
        return WeatherScore(
            temp_score=temp_score,
            precip_score=precip_score,
            wind_score=wind_score,
            humidity_score=humidity_score,
            overall_score=overall,
            category=category
        )


if __name__ == "__main__":
    # Test the scorer with various conditions
    scorer = WeatherScorer()
    
    # Test individual scoring functions
    print("Temperature scoring:")
    for temp in [-5, 0, 8, 12, 15, 20, 25, 30, 35]:
        score = scorer.score_temperature(temp)
        print(f"  {temp}°C: {score:.1f}")
    
    print("\nPrecipitation scoring:")
    for precip in [0, 2, 5, 10, 20, 30, 50, 75]:
        score = scorer.score_precipitation(precip)
        print(f"  {precip}mm: {score:.1f}")
    
    print("\nWind scoring:")
    for wind in [5, 15, 25, 35, 50, 65, 80]:
        score = scorer.score_wind(wind)
        print(f"  {wind}km/h: {score:.1f}")
    
    print("\nHumidity scoring:")
    for humidity in [20, 40, 55, 70, 85, 95]:
        score = scorer.score_humidity(humidity)
        print(f"  {humidity}%: {score:.1f}")
    
    # Test full scoring
    print("\n" + "="*50)
    print("Full weather scoring examples:")
    
    # Perfect conditions
    perfect = WeatherData(
        avg_temp_c=12.0, min_temp_c=10.0, max_temp_c=14.0,
        total_precip_mm=0.0, max_wind_kmh=10.0, avg_wind_kmh=5.0,
        avg_humidity_pct=55.0, hours_analyzed=24
    )
    result = scorer.compute_score(perfect)
    print(f"\nPerfect conditions: {result.overall_score:.1f} ({result.category})")
    
    # Challenging conditions
    challenging = WeatherData(
        avg_temp_c=5.0, min_temp_c=2.0, max_temp_c=8.0,
        total_precip_mm=25.0, max_wind_kmh=45.0, avg_wind_kmh=30.0,
        avg_humidity_pct=90.0, hours_analyzed=24
    )
    result = scorer.compute_score(challenging)
    print(f"Challenging conditions: {result.overall_score:.1f} ({result.category})")

