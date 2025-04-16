import os
import requests
import json
import numpy as np
import polyline
from math import radians, sin, cos, sqrt, atan2

def geocode_location(city_name):
    """
    Geocode a city name to get its coordinates using OpenRouteService API.
    
    Args:
        city_name: Name of the city to geocode
        
    Returns:
        Tuple of (latitude, longitude) if successful, None otherwise
    """
    api_key = os.getenv('ORS_API_KEY')
    if not api_key:
        raise ValueError("ORS_API_KEY not found in environment variables")
    
    base_url = "https://api.openrouteservice.org/geocode/search"
    
    params = {
        'api_key': api_key,
        'text': city_name,
        'size': 1,
        'layers': 'locality'
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if 'features' in data and len(data['features']) > 0:
            # Get the coordinates in the format [longitude, latitude]
            coordinates = data['features'][0]['geometry']['coordinates']
            # Return as (latitude, longitude) for consistency with other functions
            return (coordinates[1], coordinates[0])
        else:
            return None
    except Exception as e:
        print(f"Error geocoding location: {e}")
        return None

def get_route_between_locations(start_lat, start_lng, end_lat, end_lng):
    """
    Get a route between two locations using OpenRouteService API.
    
    Args:
        start_lat: Latitude of the starting location
        start_lng: Longitude of the starting location
        end_lat: Latitude of the ending location
        end_lng: Longitude of the ending location
        
    Returns:
        Dictionary with route information including distance, duration, and coordinates
    """
    api_key = os.getenv('ORS_API_KEY')
    if not api_key:
        raise ValueError("ORS_API_KEY not found in environment variables")
    
    base_url = "https://api.openrouteservice.org/v2/directions/driving-car"
    
    headers = {
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'coordinates': [
            [start_lng, start_lat],
            [end_lng, end_lat]
        ]
    }
    
    try:
        response = requests.post(base_url, headers=headers, json=data)
        result = response.json()
        
        if 'routes' in result and len(result['routes']) > 0:
            route = result['routes'][0]
            
            # Extract route information
            distance_km = route['summary']['distance'] / 1000  # Convert meters to kilometers
            duration_sec = route['summary']['duration']  # Duration in seconds
            
            # Decode the polyline to get coordinates
            encoded_polyline = route['geometry']
            coordinates = polyline.decode(encoded_polyline)
            
            return {
                'distance': distance_km,
                'duration': duration_sec,
                'coordinates': coordinates
            }
        else:
            return None
    except Exception as e:
        print(f"Error getting route: {e}")
        return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the Earth.
    
    Args:
        lat1, lon1: Coordinates of the first point (in degrees)
        lat2, lon2: Coordinates of the second point (in degrees)
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  # Earth radius in kilometers
    
    return distance

def format_duration(seconds):
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m"
    else:
        return f"{int(minutes)}m {int(seconds)}s"

def calculate_shortest_paths_dijkstra(locations):
    """
    Calculate shortest paths between all locations using Dijkstra's algorithm,
    implemented by calling the OpenRouteService API for real-world routing.
    Also handles ordering of visits using a greedy nearest neighbor approach.
    
    Args:
        locations: List of location dictionaries, starting with the origin
        
    Returns:
        Tuple of (shortest_paths, total_distance, ordered_visits, route_coordinates, total_duration)
    """
    n = len(locations)
    
    # Create a matrix to store distances and durations between all locations
    shortest_paths = [[None for _ in range(n)] for _ in range(n)]
    
    # Calculate distances and durations between all pairs of locations
    for i in range(n):
        for j in range(n):
            if i != j:
                route = get_route_between_locations(
                    locations[i]['lat'], locations[i]['lng'],
                    locations[j]['lat'], locations[j]['lng']
                )
                
                if route:
                    shortest_paths[i][j] = {
                        'distance': route['distance'],
                        'duration': route['duration'],
                        'coordinates': route['coordinates']
                    }
                else:
                    # If route couldn't be fetched, use haversine distance as fallback
                    dist = haversine_distance(
                        locations[i]['lat'], locations[i]['lng'],
                        locations[j]['lat'], locations[j]['lng']
                    )
                    shortest_paths[i][j] = {
                        'distance': dist,
                        'duration': dist * 60,  # Rough estimate: 1 km takes 60 seconds
                        'coordinates': []
                    }
    
    # Use greedy algorithm (nearest neighbor) to determine visit order
    ordered_indices = [0]  # Start with the origin
    unvisited = list(range(1, n))
    
    while unvisited:
        current = ordered_indices[-1]
        next_idx = min(unvisited, key=lambda i: shortest_paths[current][i]['distance'])
        ordered_indices.append(next_idx)
        unvisited.remove(next_idx)
    
    # Calculate total distance and collect coordinates for the entire route
    total_distance = 0
    total_duration = 0
    route_coordinates = []
    ordered_visits = [locations[i] for i in ordered_indices]
    
    for i in range(len(ordered_indices) - 1):
        from_idx = ordered_indices[i]
        to_idx = ordered_indices[i + 1]
        
        total_distance += shortest_paths[from_idx][to_idx]['distance']
        total_duration += shortest_paths[from_idx][to_idx]['duration']
        
        coords = shortest_paths[from_idx][to_idx]['coordinates']
        if coords:
            route_coordinates.extend(coords)
    
    return shortest_paths, total_distance, ordered_visits, route_coordinates, total_duration