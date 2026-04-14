#!/usr/bin/env python3
"""
Quick Start Guide - Proactive Handover Management Simulation

This script demonstrates how to use individual modules from the project.
"""

# Example 1: Generate Mobility Data
print("="*70)
print("Example 1: Generating User Mobility Data")
print("="*70)

from data_generation import generate_user_mobility, get_user_trajectory
import pandas as pd

mobility_data = generate_user_mobility(num_users=2, num_timesteps=100)
print(f"\nGenerated trajectory for {mobility_data['user_id'].nunique()} users")
print(f"Total data points: {len(mobility_data)}")
print("\nSample data (first 5 rows):")
print(mobility_data.head())

print(f"\nSample data (last 5 rows):")
print(mobility_data.tail())

# Example 2: Base Station Management
print("\n" + "="*70)
print("Example 2: Base Station Setup and Assignment")
print("="*70)

from base_station import create_base_stations, get_nearest_base_station, get_k_nearest_base_stations

base_stations = create_base_stations(num_bs=5)
print(f"\nCreated {len(base_stations)} base stations:")
for bs in base_stations:
    print(f"  {bs}")

# Find nearest BS for a user position
user_x, user_y = 50.0, 50.0
nearest_bs, distance = get_nearest_base_station(base_stations, user_x, user_y)
print(f"\nNearest BS to user at ({user_x}, {user_y}): {nearest_bs} at {distance:.2f}m")

# Find k-nearest BS
k_nearest = get_k_nearest_base_stations(base_stations, user_x, user_y, k=3)
print(f"\n3-Nearest base stations:")
for bs, dist in k_nearest:
    print(f"  {bs}: {dist:.2f}m")

# Example 3: Signal Strength Modeling
print("\n" + "="*70)
print("Example 3: Signal Strength (RSS) Model")
print("="*70)

from signal_model import SignalStrengthModel

signal_model = SignalStrengthModel(max_power=1.0, epsilon=0.1)
print("\nSignal Strength vs Distance:")
print("Distance (m) | RSS (W)   | Quality")
print("-"*40)

distances = [5, 10, 20, 50, 100]
for dist in distances:
    rss = signal_model.compute_rss(dist, add_noise=False)
    quality = signal_model.get_signal_quality(rss)
    print(f"{dist:12} | {rss:9.4f} | {quality}")

# Example 4: ML Mobility Prediction
print("\n" + "="*70)
print("Example 4: ML Mobility Prediction")
print("="*70)

from lstm_model import MobilityPredictor

predictor = MobilityPredictor(sequence_length=20)

# Prepare training data
trajectories = []
for user_id in range(2):
    traj = get_user_trajectory(mobility_data, user_id)
    trajectories.append(traj)

print(f"\nTraining ML model on {len(trajectories)} trajectories...")
predictor.train(trajectories, epochs=10, verbose=0)
print("Training completed!")

# Make a prediction
user_traj = trajectories[0]
recent_positions = user_traj[['x', 'y']].values[-20:]
pred_x, pred_y = predictor.predict_next_position(recent_positions)
print(f"\nLast known position: ({user_traj['x'].iloc[-1]:.2f}, {user_traj['y'].iloc[-1]:.2f})")
print(f"Predicted next position: ({pred_x:.2f}, {pred_y:.2f})")

# Example 5: Handover Decision Making
print("\n" + "="*70)
print("Example 5: Reactive vs Proactive Handover")
print("="*70)

from handover import ReactiveHandover, ProactiveHandover

reactive = ReactiveHandover(rss_threshold=0.3)
proactive = ProactiveHandover(predictor, rss_threshold=0.25)

print("\nReactive Handover:")
print(f"  - Threshold: {reactive.rss_threshold}")
print(f"  - Hysteresis: {reactive.hysteresis}")
print(f"  - Current handovers: {reactive.handover_count}")

print("\nProactive Handover:")
print(f"  - Threshold: {proactive.rss_threshold}")
print(f"  - Predictor: {predictor.__class__.__name__}")
print(f"  - Current handovers: {proactive.handover_count}")

# Example 6: Metrics Calculation
print("\n" + "="*70)
print("Example 6: Performance Metrics")
print("="*70)

from metrics import MetricsCalculator

calculator = MetricsCalculator()

# Sample metrics
reactive_stats = {
    'handover_count': 40,
    'unnecessary_handovers': 10,
    'latency': 800,
    'total_timesteps': 200
}

reactive_signal = {
    'avg_rss': 0.50,
    'min_rss': 0.20,
    'max_rss': 0.95,
    'time_below_threshold': 80,
    'avg_packet_loss': 0.10,
    'avg_throughput': 60.0
}

metrics = calculator.calculate_metrics(reactive_stats, reactive_signal)
print("\nCalculated Metrics:")
for key, value in sorted(metrics.items()):
    if isinstance(value, float):
        print(f"  {key:.<35} {value:>10.4f}")
    else:
        print(f"  {key:.<35} {value:>10}")

# Example 7: Visualization
print("\n" + "="*70)
print("Example 7: Visualization Functions Available")
print("="*70)

from visualization import (
    plot_user_trajectory_and_base_stations,
    plot_signal_strength_over_time,
    plot_metrics_comparison
)

print("\nAvailable visualization functions:")
print("  1. plot_user_trajectory_and_base_stations() - Trajectory with BS locations")
print("  2. plot_signal_strength_over_time() - RSS over time comparison")
print("  3. plot_metrics_comparison() - Bar charts for key metrics")
print("  4. plot_handover_timeline() - Handover events timeline")
print("  5. plot_base_station_assignment() - BS assignment changes over time")
print("  6. create_summary_figure() - Comprehensive summary with all metrics")

print("\nAll visualization functions save to 'results/' directory")

print("\n" + "="*70)
print("Quick Start Guide Complete!")
print("="*70)
print("\nNext Steps:")
print("  1. Run: python main.py")
print("  2. Check the 'results/' directory for generated plots")
print("  3. Explore individual modules in the code")
print("  4. Customize parameters for your experiments")
print("\n")
