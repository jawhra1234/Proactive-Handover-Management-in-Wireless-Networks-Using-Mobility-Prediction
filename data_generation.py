"""
Generate synthetic user mobility data for wireless network simulation.
Simulates realistic movement patterns using linear trajectories with noise.
"""

import numpy as np
import pandas as pd


def generate_user_mobility(
    num_users=3,
    num_timesteps=500,
    speed=2.0,
    noise_scale=0.5,
    grid_size=100.0,
    seed=42
):
    """
    Generate synthetic mobility data for users in a wireless network.
    
    Args:
        num_users: Number of users to simulate
        num_timesteps: Number of time steps to simulate
        speed: Average movement speed per timestep
        noise_scale: Standard deviation of random noise
        grid_size: Size of the simulation grid (0 to grid_size)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: user_id, time, x, y
    """
    np.random.seed(seed)
    
    data = []
    
    for user_id in range(num_users):
        # Initialize random starting position
        x_start = np.random.uniform(0, grid_size)
        y_start = np.random.uniform(0, grid_size)
        
        # Random direction (angle in radians)
        angle = np.random.uniform(0, 2 * np.pi)
        direction_x = np.cos(angle)
        direction_y = np.sin(angle)
        
        x, y = x_start, y_start
        
        for timestep in range(num_timesteps):
            # Add noise and occasional direction changes
            if timestep % 50 == 0 and timestep > 0:  # Change direction every 50 steps
                angle = np.random.uniform(0, 2 * np.pi)
                direction_x = np.cos(angle)
                direction_y = np.sin(angle)
            
            # Movement: linear trajectory + random noise
            noise_x = np.random.normal(0, noise_scale)
            noise_y = np.random.normal(0, noise_scale)
            
            x += speed * direction_x + noise_x
            y += speed * direction_y + noise_y
            
            # Bounce off boundaries (reflective boundary conditions)
            if x < 0 or x > grid_size:
                direction_x = -direction_x
                x = np.clip(x, 0, grid_size)
            if y < 0 or y > grid_size:
                direction_y = -direction_y
                y = np.clip(y, 0, grid_size)
            
            data.append({
                'user_id': user_id,
                'time': timestep,
                'x': x,
                'y': y
            })
    
    df = pd.DataFrame(data)
    return df


def get_user_trajectory(df, user_id):
    """
    Extract trajectory for a specific user.
    
    Args:
        df: Mobility DataFrame
        user_id: User ID to extract
    
    Returns:
        DataFrame with trajectory for the user
    """
    return df[df['user_id'] == user_id].reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    mobility_data = generate_user_mobility(num_users=3, num_timesteps=100)
    print("Generated mobility data:")
    print(mobility_data.head(10))
    print(f"\nShape: {mobility_data.shape}")
