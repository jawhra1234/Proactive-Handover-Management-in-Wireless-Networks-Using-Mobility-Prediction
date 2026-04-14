"""
Base Station definitions and management for wireless network simulation.
Handles base station placement and user assignment based on distance.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean


class BaseStation:
    """Represents a single base station with fixed coordinates."""
    
    def __init__(self, bs_id, x, y):
        """
        Initialize a base station.
        
        Args:
            bs_id: Unique identifier for the base station
            x: X-coordinate
            y: Y-coordinate
        """
        self.bs_id = bs_id
        self.x = x
        self.y = y
        self.position = np.array([x, y])
    
    def distance_to_user(self, user_x, user_y):
        """Calculate Euclidean distance to a user position."""
        user_pos = np.array([user_x, user_y])
        return euclidean(self.position, user_pos)
    
    def __repr__(self):
        return f"BS{self.bs_id}({self.x:.1f}, {self.y:.1f})"


def create_base_stations(num_bs=5, grid_size=100.0, seed=42):
    """
    Create base stations randomly distributed in the grid.
    
    Args:
        num_bs: Number of base stations to create
        grid_size: Size of simulation grid
        seed: Random seed
    
    Returns:
        List of BaseStation objects
    """
    np.random.seed(seed)
    base_stations = []
    
    for bs_id in range(num_bs):
        x = np.random.uniform(0, grid_size)
        y = np.random.uniform(0, grid_size)
        base_stations.append(BaseStation(bs_id, x, y))
    
    return base_stations


def get_nearest_base_station(base_stations, user_x, user_y):
    """
    Find the nearest base station to a user's position.
    
    Args:
        base_stations: List of BaseStation objects
        user_x: User X-coordinate
        user_y: User Y-coordinate
    
    Returns:
        Tuple: (nearest_base_station, distance)
    """
    min_distance = float('inf')
    nearest_bs = None
    
    for bs in base_stations:
        dist = bs.distance_to_user(user_x, user_y)
        if dist < min_distance:
            min_distance = dist
            nearest_bs = bs
    
    return nearest_bs, min_distance


def get_k_nearest_base_stations(base_stations, user_x, user_y, k=3):
    """
    Find the k-nearest base stations to a user's position (for handover candidates).
    
    Args:
        base_stations: List of BaseStation objects
        user_x: User X-coordinate
        user_y: User Y-coordinate
        k: Number of nearest stations to return
    
    Returns:
        List of tuples: [(BaseStation, distance), ...]
    """
    distances = [
        (bs, bs.distance_to_user(user_x, user_y))
        for bs in base_stations
    ]
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def get_base_station_positions(base_stations):
    """
    Get all base station positions as a DataFrame for visualization.
    
    Args:
        base_stations: List of BaseStation objects
    
    Returns:
        DataFrame with columns: bs_id, x, y
    """
    data = [{
        'bs_id': bs.bs_id,
        'x': bs.x,
        'y': bs.y
    } for bs in base_stations]
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    base_stations = create_base_stations(num_bs=5, grid_size=100.0)
    print(f"Created {len(base_stations)} base stations:")
    for bs in base_stations:
        print(f"  {bs}")
    
    # Test nearest BS assignment
    user_x, user_y = 50.0, 50.0
    nearest, dist = get_nearest_base_station(base_stations, user_x, user_y)
    print(f"\nNearest BS to user ({user_x}, {user_y}): {nearest} at distance {dist:.2f}")
