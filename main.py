import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import os
from dotenv import load_dotenv
import json
from utils.routing import (
    geocode_location, 
    get_route_between_locations, 
    calculate_shortest_paths_dijkstra,
    format_duration
)

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Intelligent Parcel Delivery System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTextInput, .stSelectbox, .stNumberInput, .stButton {
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
    }
    .stats-box {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #004D40;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #E65100;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #01579B;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def knapsack_algorithm(parcels_df, max_weight):
    """
    Implementation of 0/1 Knapsack algorithm to select the most valuable parcels 
    without exceeding the maximum weight limit.
    
    Args:
        parcels_df: DataFrame containing parcel data including weight and value
        max_weight: Maximum total weight that can be carried
        
    Returns:
        List of selected parcel indices
    """
    n = len(parcels_df)
    weights = parcels_df['weight'].tolist()
    values = parcels_df['value'].tolist()
    
    # Convert max_weight to an integer by scaling
    scale_factor = 1000
    weights_scaled = [int(w * scale_factor) for w in weights]
    max_weight_scaled = int(max_weight * scale_factor)
    
    # Initialize DP table
    dp = [[0 for _ in range(max_weight_scaled + 1)] for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(max_weight_scaled + 1):
            if weights_scaled[i-1] <= w:
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w-weights_scaled[i-1]], 
                    dp[i-1][w]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # Backtrack to find selected items
    selected_parcels = []
    w = max_weight_scaled
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_parcels.append(i-1)
            w -= weights_scaled[i-1]
    
    return selected_parcels[::-1]  # Reverse to maintain original order

def display_route_map(locations, route_coordinates):
    """
    Display a map with the optimized route using Folium.
    
    Args:
        locations: List of location data including coordinates
        route_coordinates: List of coordinates for the route
    """
    # Create a folium map centered at the first location
    m = folium.Map(location=[locations[0]['lat'], locations[0]['lng']], 
                   zoom_start=10, 
                   tiles="CartoDB dark_matter")
    
    # Add markers for each location
    for i, loc in enumerate(locations):
        popup_text = f"Location {i+1}: {loc['city']}"
        icon_color = 'red' if i == 0 else ('green' if i == len(locations)-1 else 'blue')
        
        folium.Marker(
            [loc['lat'], loc['lng']],
            popup=popup_text,
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(m)
    
    # Add the route as a polyline
    if route_coordinates and len(route_coordinates) > 1:
        folium.PolyLine(
            route_coordinates,
            weight=5,
            color='blue',
            opacity=0.8
        ).add_to(m)
    
    # Display the map
    folium_static(m)

def main():
    st.title("📦 Intelligent Parcel Delivery System")
    
    st.markdown("""
    <div class="info-box">
    Optimize your delivery operations using advanced algorithms:
    <ul>
        <li>0/1 Knapsack algorithm for optimal parcel selection</li>
        <li>Nearest Neighbor algorithm for route optimization</li>
        <li>Dijkstra algorithm for shortest paths between selected cities</li>
        <li>Real-world routing using OpenRouteService API</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Configuration")
        
        # Input method selection
        input_method = st.radio("Input Method", ["Manual Entry", "Upload CSV"])
        
        if input_method == "Manual Entry":
            # Get number of parcels to input
            num_parcels = st.number_input("Number of Parcels", min_value=1, max_value=20, value=3)
            
            # Create empty dataframe
            parcel_data = []
            
            st.subheader("Enter Parcel Details")
            
            # Input for each parcel
            for i in range(num_parcels):
                st.markdown(f"#### Parcel {i+1}")
                
                city = st.text_input(f"City {i+1}", key=f"city_{i}")
                weight = st.number_input(f"Weight (kg) {i+1}", 
                                         min_value=0.1, 
                                         max_value=1000.0, 
                                         value=10.0, 
                                         step=0.1,
                                         key=f"weight_{i}")
                value = st.number_input(f"Value ($) {i+1}", 
                                        min_value=1, 
                                        max_value=10000, 
                                        value=100, 
                                        step=10,
                                        key=f"value_{i}")
                
                parcel_data.append({
                    "id": i+1,
                    "city": city,
                    "weight": weight,
                    "value": value
                })
            
            parcels_df = pd.DataFrame(parcel_data)
        
        else:  # Upload CSV
            st.subheader("Upload Parcel Data")
            st.markdown("CSV should have columns: id, city, weight, value")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    parcels_df = pd.read_csv(uploaded_file)
                    if not all(col in parcels_df.columns for col in ["id", "city", "weight", "value"]):
                        st.error("CSV must contain columns: id, city, weight, value")
                        return
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    return
            else:
                # Use sample data if no file uploaded
                st.info("No file uploaded. Using sample data.")
                sample_data = [
                    {"id": 1, "city": "Berlin", "weight": 10.0, "value": 200},
                    {"id": 2, "city": "Hamburg", "weight": 5.0, "value": 100},
                    {"id": 3, "city": "Munich", "weight": 15.0, "value": 300},
                    {"id": 4, "city": "Frankfurt", "weight": 8.0, "value": 150},
                    {"id": 5, "city": "Cologne", "weight": 12.0, "value": 250}
                ]
                parcels_df = pd.DataFrame(sample_data)
        
        # Delivery constraints
        st.subheader("Delivery Constraints")
        max_weight = st.number_input("Maximum Weight Capacity (kg)", 
                                     min_value=1.0, 
                                     max_value=2000.0, 
                                     value=30.0, 
                                     step=1.0)
        
        # Starting city
        st.subheader("Starting Location")
        start_city = st.text_input("Starting City", "Berlin")
        
        # Optimization button
        optimize_button = st.button("Optimize Delivery", type="primary")
    
    # Main panel
    if 'parcels_df' in locals() and len(parcels_df) > 0:
        st.subheader("Parcel Data")
        st.dataframe(parcels_df)
        
        if optimize_button:
            with st.spinner("Optimizing delivery route..."):
                try:
                    # Step 1: Geocode all locations including the starting point
                    locations = []
                    
                    # Add starting location first
                    start_location = geocode_location(start_city)
                    if start_location:
                        locations.append({
                            'id': 0,
                            'city': start_city,
                            'lat': start_location[0],
                            'lng': start_location[1]
                        })
                    else:
                        st.error(f"Could not geocode starting city: {start_city}")
                        return
                    
                    # Geocode all parcel locations
                    for _, row in parcels_df.iterrows():
                        location = geocode_location(row['city'])
                        if location:
                            locations.append({
                                'id': row['id'],
                                'city': row['city'],
                                'lat': location[0],
                                'lng': location[1],
                                'weight': row['weight'],
                                'value': row['value']
                            })
                        else:
                            st.warning(f"Could not geocode city: {row['city']}")
                    
                    # Step 2: Apply Knapsack algorithm to select parcels
                    locations_df = pd.DataFrame(locations[1:])  # Exclude starting point
                    selected_indices = knapsack_algorithm(locations_df, max_weight)
                    
                    if not selected_indices:
                        st.error("No parcels could be selected within the weight constraint.")
                        return
                    
                    # Create a list of selected locations (including start point)
                    selected_locations = [locations[0]]  # Starting point
                    for idx in selected_indices:
                        selected_locations.append(locations[idx + 1])  # +1 because indices are for locations_df
                    
                    # Display selected parcels
                    selected_parcels = locations_df.iloc[selected_indices].copy()
                    
                    st.subheader("Selected Parcels")
                    st.markdown(f"""
                    <div class="success-box">
                        <p>Optimally selected {len(selected_parcels)} parcels out of {len(parcels_df)} available.</p>
                        <p>Total Weight: {selected_parcels['weight'].sum():.2f} kg out of {max_weight:.2f} kg maximum.</p>
                        <p>Total Value: ${selected_parcels['value'].sum()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(selected_parcels)
                    
                    # Step 3: Apply Dijkstra algorithm for shortest paths between selected cities
                    st.subheader("Shortest Paths Between Selected Cities")
                    
                    # Create graph from selected locations
                    with st.spinner("Calculating shortest paths..."):
                        # Include only the starting location and selected parcels
                        shortest_paths, total_distance, ordered_visits, route_coordinates, total_duration = calculate_shortest_paths_dijkstra(selected_locations)
                        
                        # Display results
                        if shortest_paths:
                            st.markdown(f"""
                            <div class="stats-box">
                                <p>Total Distance: {total_distance:.2f} km</p>
                                <p>Estimated Total Duration: {format_duration(total_duration)}</p>
                                <p>Delivery Sequence: {' → '.join([loc['city'] for loc in ordered_visits])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display the route details
                            st.subheader("Delivery Route Details")
                            route_details = []
                            
                            for i in range(len(ordered_visits) - 1):
                                from_city = ordered_visits[i]['city']
                                to_city = ordered_visits[i + 1]['city']
                                segment_distance = shortest_paths[i][i+1]['distance']
                                segment_duration = shortest_paths[i][i+1]['duration']
                                
                                route_details.append({
                                    "From": from_city,
                                    "To": to_city,
                                    "Distance (km)": f"{segment_distance:.2f}",
                                    "Duration": format_duration(segment_duration)
                                })
                            
                            st.table(pd.DataFrame(route_details))
                            
                            # Display route on map
                            st.subheader("Delivery Route Map")
                            display_route_map(ordered_visits, route_coordinates)
                        else:
                            st.error("Could not calculate routes between locations. Please check your locations and try again.")
                
                except Exception as e:
                    st.error(f"An error occurred during optimization: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Intelligent Parcel Delivery System | Made with ❤️ using Python & Streamlit")

if __name__ == "__main__":
    main()