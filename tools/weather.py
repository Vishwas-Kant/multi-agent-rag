import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv
from utils.cache import cached

load_dotenv()

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=5,
    pool_maxsize=10,
    max_retries=requests.adapters.Retry(total=2, backoff_factor=0.3),
)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


@cached(ttl=300)
def _fetch_weather_cached(city: str) -> str:
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "Error: OpenWeatherMap API key not found in environment variables."

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = _session.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"].get("feels_like", temp)
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        visibility = data.get("visibility", "N/A")

        return (
            f"Weather in {city}: {weather_desc}\n"
            f"🌡️ Temperature: {temp}°C (feels like {feels_like}°C)\n"
            f"💧 Humidity: {humidity}%\n"
            f"💨 Wind Speed: {wind_speed} m/s\n"
            f"👁️ Visibility: {visibility}m"
        )

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except KeyError:
        return f"Error parsing weather data. Please check if the city '{city}' exists."


@tool
def fetch_weather(city: str) -> str:
    return _fetch_weather_cached(city)
