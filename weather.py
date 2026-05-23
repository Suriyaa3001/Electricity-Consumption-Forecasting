"""
weather.py
----------
Fetches real-time temperature using Open-Meteo (free, no API key required).
Uses the Open-Meteo geocoding API to resolve city names to coordinates.
"""

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def get_coordinates(city: str) -> tuple[float, float, str]:
    """
    Resolve a city name to (latitude, longitude, display_name).
    Raises ValueError if city not found.
    """
    resp = requests.get(GEOCODE_URL, params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=5)
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"City '{city}' not found.")
    r = results[0]
    return r["latitude"], r["longitude"], f"{r['name']}, {r.get('country', '')}"


def get_current_temperature(city: str) -> tuple[float, str]:
    """
    Returns (temperature_celsius, display_name) for the given city.
    Uses Open-Meteo current weather endpoint.
    """
    lat, lon, display = get_coordinates(city)
    resp = requests.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m",
            "timezone": "auto",
        },
        timeout=5,
    )
    resp.raise_for_status()
    temp = resp.json()["current"]["temperature_2m"]
    temp = float(temp)
    if temp != temp:  # NaN check
        raise ValueError("Weather API returned an invalid temperature (NaN).")
    return temp, display
