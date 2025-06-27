import os
import requests

def plan_route(input_data):
    origin = input_data["origin"]  # (lat, lon)
    destination = input_data["destination"]  # (lat, lon)

    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {
        "Authorization": os.getenv("ORS_API_KEY"),
        "Content-Type": "application/json"
    }

    payload = {
        "coordinates": [[origin[1], origin[0]], [destination[1], destination[0]]]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        route = data["features"][0]
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve route from ORS: {e}")

    coords = route["geometry"]["coordinates"]
    segment = route["properties"]["segments"][0]

    input_data["route_info"] = {
        "coordinates": coords,
        "distance_km": segment["distance"] / 1000,
        "base_route_time_mins": segment["duration"] / 60,
        "elevation_gain": 200  # Simulated or hardcoded
    }

    return input_data
