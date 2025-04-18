import os
import requests
import json
import numpy as np
import polyline
from math import radians, sin, cos, sqrt, atan2, inf

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

def branch_and_bound_tsp(distance_matrix):
    """
    Branch and Bound algorithm for the Traveling Salesperson Problem.
    Finds an optimal route visiting all locations starting and ending at the depot.
    
    Args:
        distance_matrix: Matrix of distances between all pairs of locations
        
    Returns:
        Tuple of (optimal_path, optimal_cost)
    """
    n = len(distance_matrix)  # Number of cities
    
    # Initialize global variables
    optimal_path = []
    optimal_cost = float('inf')
    
    # Helper function to calculate the lower bound for a partial path
    def calculate_lower_bound(path, visited):
        current = path[-1]
        lb = sum(distance_matrix[i][j] for i, j in zip(path[:-1], path[1:]))  # Cost of current path
        
        # Add minimum outgoing edge for each unvisited node
        for i in range(n):
            if not visited[i]:
                min_edge = float('inf')
                for j in range(n):
                    if j != i and distance_matrix[i][j] < min_edge:
                        min_edge = distance_matrix[i][j]
                lb += min_edge
                
        # Add minimum edge from current to an unvisited node
        if n - len(path) > 0:  # If there are unvisited nodes
            min_edge = float('inf')
            for i in range(n):
                if not visited[i] and distance_matrix[current][i] < min_edge:
                    min_edge = distance_matrix[current][i]
            lb += min_edge
        
        # Add minimum edge from an unvisited node to depot
        if n - len(path) > 0:  # If there are unvisited nodes
            min_edge = float('inf')
            for i in range(n):
                if not visited[i] and distance_matrix[i][0] < min_edge:
                    min_edge = distance_matrix[i][0]
            lb += min_edge
        elif len(path) == n:  # All nodes visited, add cost to return to depot
            lb += distance_matrix[current][0]
            
        return lb
    
    # Recursive branch and bound function
    def branch_and_bound(path, cost, visited):
        nonlocal optimal_path, optimal_cost
        
        # If all nodes have been visited
        if len(path) == n:
            # Add cost to return to depot (node 0)
            total_cost = cost + distance_matrix[path[-1]][0]
            if total_cost < optimal_cost:
                optimal_cost = total_cost
                optimal_path = path.copy()  # Make a copy of the path
            return
        
        # Calculate lower bound for current partial path
        lower_bound = calculate_lower_bound(path, visited)
        
        # Prune if lower bound exceeds the current optimal cost
        if lower_bound >= optimal_cost:
            return
        
        # Try adding each unvisited node to the path
        current = path[-1]
        for next_node in range(n):  # Consider all nodes (not just starting from 1)
            if not visited[next_node]:
                visited[next_node] = True
                new_cost = cost + distance_matrix[current][next_node]
                
                # Recurse only if the new cost doesn't exceed optimal_cost
                if new_cost < optimal_cost:
                    branch_and_bound(path + [next_node], new_cost, visited)
                
                visited[next_node] = False
    
    # Start from the depot (node 0)
    visited = [False] * n
    visited[0] = True
    branch_and_bound([0], 0, visited)
    
    # If we found a solution, add the return to depot
    if optimal_path:
        return optimal_path, optimal_cost
    else:
        # Fallback to simple ordering if no solution found
        return list(range(n)), float('inf')

def calculate_shortest_paths_dijkstra(locations):
    """
    Calculate shortest paths between all locations using real-world routing,
    then optimize the route using Branch and Bound for TSP.
    
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
    
    # Create distance matrix for Branch and Bound TSP
    distance_matrix = [[shortest_paths[i][j]['distance'] if i != j else 0 for j in range(n)] for i in range(n)]
    
    # Solve TSP using Branch and Bound
    optimal_path, _ = branch_and_bound_tsp(distance_matrix)
    
    # Check if optimal path was found (should always be the case now with the fallback)
    ordered_indices = optimal_path
    
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