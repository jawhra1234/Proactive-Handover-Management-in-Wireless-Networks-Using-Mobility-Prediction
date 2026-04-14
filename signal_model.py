"""
Signal Strength Model for wireless network simulation.
Simulates Received Signal Strength (RSS) based on distance and path loss.
"""

import numpy as np


class SignalStrengthModel:
    """
    Models Received Signal Strength (RSS) using path loss model.
    RSS = 1 / (distance + epsilon) + noise
    """
    
    def __init__(
        self,
        max_power=1.0,
        epsilon=0.1,
        noise_scale=0.05,
        seed=42
    ):
        """
        Initialize signal strength model.
        
        Args:
            max_power: Maximum signal strength (power at distance 0)
            epsilon: Small constant to avoid division by zero
            noise_scale: Standard deviation of Gaussian noise
            seed: Random seed
        """
        self.max_power = max_power
        self.epsilon = epsilon
        self.noise_scale = noise_scale
        np.random.seed(seed)
    
    def compute_rss(self, distance, add_noise=True):
        """
        Compute Received Signal Strength based on distance.
        
        Path loss model: RSS = max_power / (distance + epsilon)
        
        Args:
            distance: Distance from base station to user
            add_noise: Whether to add Gaussian noise
        
        Returns:
            Received Signal Strength value
        """
        # Basic path loss model: strength decreases with distance
        if distance < 0:
            distance = 0
        
        rss = self.max_power / (distance + self.epsilon)
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale)
            rss = np.clip(rss + noise, 0, self.max_power)
        else:
            rss = np.clip(rss, 0, self.max_power)
        
        return rss
    
    def compute_rss_dbm(self, distance, add_noise=True):
        """
        Compute RSS in dBm (decibel-milliwatts) - more realistic units.
        
        Args:
            distance: Distance from base station to user
            add_noise: Whether to add Gaussian noise
        
        Returns:
            RSS in dBm
        """
        rss_linear = self.compute_rss(distance, add_noise)
        # Convert to dBm (0.001W reference)
        rss_dbm = 10 * np.log10(rss_linear / 0.001)
        return rss_dbm
    
    def get_signal_quality(self, rss, threshold_low=0.3, threshold_high=0.7):
        """
        Classify signal quality based on RSS.
        
        Args:
            rss: Received Signal Strength value
            threshold_low: Lower threshold for poor signal
            threshold_high: Upper threshold for good signal
        
        Returns:
            String: 'good', 'fair', or 'poor'
        """
        if rss >= threshold_high:
            return 'good'
        elif rss >= threshold_low:
            return 'fair'
        else:
            return 'poor'


def compute_all_rss_values(base_stations, user_x, user_y, signal_model):
    """
    Compute RSS from all base stations to a user position.
    
    Args:
        base_stations: List of BaseStation objects
        user_x: User X-coordinate
        user_y: User Y-coordinate
        signal_model: SignalStrengthModel instance
    
    Returns:
        Dict: {bs_id: rss_value}
    """
    rss_values = {}
    
    for bs in base_stations:
        distance = bs.distance_to_user(user_x, user_y)
        rss = signal_model.compute_rss(distance, add_noise=True)
        rss_values[bs.bs_id] = rss
    
    return rss_values


def packet_loss_rate(rss, threshold=0.3):
    """
    Simulate packet loss based on signal strength.
    Poor signal -> higher packet loss.
    
    Args:
        rss: Received Signal Strength
        threshold: Threshold below which packet loss increases significantly
    
    Returns:
        Packet loss rate (0.0 to 1.0)
    """
    if rss < threshold:
        # Exponential increase in packet loss below threshold
        return 1.0 - (rss / threshold) ** 2
    else:
        # Low packet loss above threshold
        return 0.01


def simulate_throughput(rss, max_throughput=100.0):
    """
    Simulate throughput inversely proportional to distance (via RSS).
    
    Args:
        rss: Received Signal Strength
        max_throughput: Maximum throughput in Mbps
    
    Returns:
        Estimated throughput in Mbps
    """
    # Throughput decreases as RSS decreases
    throughput = max_throughput * (rss ** 1.5)
    return max(0.1, throughput)  # Minimum throughput of 0.1 Mbps


if __name__ == "__main__":
    # Example usage
    signal_model = SignalStrengthModel(max_power=1.0, epsilon=0.1)
    
    distances = [5, 10, 20, 50, 100]
    print("Distance (m) | RSS (W) | RSS (dBm) | Quality")
    print("-" * 50)
    for dist in distances:
        rss = signal_model.compute_rss(dist, add_noise=False)
        rss_dbm = signal_model.compute_rss_dbm(dist, add_noise=False)
        quality = signal_model.get_signal_quality(rss)
        print(f"{dist:12} | {rss:7.3f} | {rss_dbm:9.2f} | {quality}")
