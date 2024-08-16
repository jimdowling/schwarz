import os
import datetime
import time
import requests
import pandas as pd
import json
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pathlib import Path

def get_hourly_weather_forecast(latitude=49.1427, longitude=9.2109):

    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "precipitation"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    
    hourly_data = {"ds": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s"),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["outdoortemperature"] = hourly_temperature_2m
#     hourly_data["precipitation_sum"] = hourly_precipitation

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    hourly_dataframe = hourly_dataframe.dropna()
    return hourly_dataframe


def get_historical_weather(start_date,  end_date, latitude=49.1427, longitude=9.2109):
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "precipitation_sum"]
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()

    daily_data = {"ds": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s"),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["outdoortemperature"] = daily_temperature_2m_mean
#     daily_data["precipitation_sum"] = daily_precipitation_sum

    daily_dataframe = pd.DataFrame(data = daily_data)
    daily_dataframe = daily_dataframe.dropna()
    return daily_dataframe