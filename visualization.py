"""
Visualization module for wireless network simulation results.
Generates plots and charts for analysis and comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from base_station import get_base_station_positions, get_nearest_base_station
from data_generation import get_user_trajectory


def set_plot_style():
    """Set consistent plot styling."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def plot_user_trajectory_and_base_stations(
    mobility_data,
    base_stations,
    user_id=0,
    title="User Trajectory and Base Stations"
):
    """
    Plot user trajectory with base station locations.
    
    Args:
        mobility_data: DataFrame with mobility data
        base_stations: List of BaseStation objects
        user_id: User ID to plot
        title: Plot title
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get user trajectory
    user_traj = get_user_trajectory(mobility_data, user_id)
    
    # Plot trajectory
    ax.plot(user_traj['x'], user_traj['y'], 'b-', alpha=0.5, label='User Trajectory', linewidth=1)
    ax.plot(user_traj.iloc[0]['x'], user_traj.iloc[0]['y'], 'go', markersize=10, label='Start')
    ax.plot(user_traj.iloc[-1]['x'], user_traj.iloc[-1]['y'], 'r*', markersize=15, label='End')
    
    # Plot base stations
    bs_positions = get_base_station_positions(base_stations)
    ax.scatter(bs_positions['x'], bs_positions['y'], s=300, c='red', 
               marker='s', edgecolors='darkred', linewidth=2, label='Base Stations', zorder=5)
    
    # Add BS labels
    for idx, row in bs_positions.iterrows():
        ax.annotate(f"BS{int(row['bs_id'])}", 
                   (row['x'], row['y']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Add coverage circles (visualization only)
    for bs in base_stations:
        circle = patches.Circle((bs.x, bs.y), 30, fill=False, 
                              edgecolor='gray', linestyle='--', alpha=0.3)
        ax.add_patch(circle)
    
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig, ax


def plot_handover_points(
    mobility_data,
    base_stations,
    handover_history,
    user_id=0,
    title="Handover Points - Reactive vs Proactive"
):
    """
    Plot handover points on trajectory.
    
    Args:
        mobility_data: DataFrame with mobility data
        base_stations: List of BaseStation objects
        handover_history: List of handover events with timestamps
        user_id: User ID to plot
        title: Plot title
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get user trajectory
    user_traj = get_user_trajectory(mobility_data, user_id)
    
    # Plot trajectory
    ax.plot(user_traj['x'], user_traj['y'], 'b-', alpha=0.3, linewidth=1, label='Trajectory')
    
    # Plot base stations
    bs_positions = get_base_station_positions(base_stations)
    ax.scatter(bs_positions['x'], bs_positions['y'], s=300, c='red',
               marker='s', edgecolors='darkred', linewidth=2, label='Base Stations', zorder=5)
    
    # Plot handover points
    if handover_history:
        handover_times = [ho['time'] for ho in handover_history]
        handover_positions = user_traj.iloc[handover_times][['x', 'y']]
        ax.scatter(handover_positions['x'], handover_positions['y'], s=100, c='orange',
                  marker='o', edgecolors='darkorange', linewidth=2, label='Handover Points', zorder=4)
    
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig, ax


def plot_signal_strength_over_time(
    reactive_rss,
    proactive_rss,
    threshold=0.3,
    title="Signal Strength Over Time"
):
    """
    Plot RSS over time for both approaches.
    
    Args:
        reactive_rss: List of RSS values for reactive approach
        proactive_rss: List of RSS values for proactive approach
        threshold: RSS threshold for handover
        title: Plot title
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    time_steps = np.arange(len(reactive_rss))
    
    ax.plot(time_steps, reactive_rss, label='Reactive Handover', linewidth=1.5, alpha=0.8)
    ax.plot(time_steps, proactive_rss, label='Proactive Handover', linewidth=1.5, alpha=0.8)
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Handover Threshold ({threshold})')
    
    ax.fill_between(time_steps, 0, threshold, alpha=0.1, color='red', label='Poor Signal Zone')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Signal Strength (RSS)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_metrics_comparison(reactive_metrics, proactive_metrics):
    """
    Create bar charts comparing key metrics.
    
    Args:
        reactive_metrics: Dict with reactive handover metrics
        proactive_metrics: Dict with proactive handover metrics
    """
    set_plot_style()
    
    # Select key metrics to compare
    key_metrics = [
        'total_handovers',
        'unnecessary_handovers',
        'accumulated_latency',
        'time_below_threshold',
        'avg_packet_loss',
        'avg_throughput'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        
        reactive_val = reactive_metrics.get(metric, 0)
        proactive_val = proactive_metrics.get(metric, 0)
        
        approaches = ['Reactive', 'Proactive']
        values = [reactive_val, proactive_val]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(approaches, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        metric_label = metric.replace('_', ' ').title()
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} Comparison")
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_handover_timeline(reactive_handovers, proactive_handovers):
    """
    Plot handover events on timeline.
    
    Args:
        reactive_handovers: List of reactive handover events
        proactive_handovers: List of proactive handover events
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot reactive handovers
    if reactive_handovers:
        reactive_times = [ho['time'] for ho in reactive_handovers]
        ax.scatter(reactive_times, [1]*len(reactive_times), s=100, c='red',
                  marker='v', label='Reactive Handover', alpha=0.7, zorder=3)
    
    # Plot proactive handovers
    if proactive_handovers:
        proactive_times = [ho['time'] for ho in proactive_handovers]
        ax.scatter(proactive_times, [2]*len(proactive_times), s=100, c='green',
                  marker='^', label='Proactive Handover', alpha=0.7, zorder=3)
    
    ax.set_ylim(0.5, 2.5)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Reactive', 'Proactive'])
    ax.set_xlabel('Time Step')
    ax.set_title('Handover Timeline - Reactive vs Proactive')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig, ax


def plot_base_station_assignment(mobility_data, reactive_bs, proactive_bs, user_id=0):
    """
    Plot base station assignments over time.
    
    Args:
        mobility_data: DataFrame with mobility data
        reactive_bs: List of BS assignments (reactive)
        proactive_bs: List of BS assignments (proactive)
        user_id: User ID
    """
    set_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    time_steps = np.arange(len(reactive_bs))
    
    # Reactive
    ax1.fill_between(time_steps, np.array(reactive_bs) - 0.5, 
                     np.array(reactive_bs) + 0.5, alpha=0.3, step='mid')
    ax1.plot(time_steps, reactive_bs, 'o-', markersize=2, label='Base Station Assignment')
    ax1.set_ylabel('Base Station ID')
    ax1.set_title(f'Reactive Handover - Base Station Assignment (User {user_id})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Proactive
    ax2.fill_between(time_steps, np.array(proactive_bs) - 0.5,
                     np.array(proactive_bs) + 0.5, alpha=0.3, step='mid', color='green')
    ax2.plot(time_steps, proactive_bs, 'o-', markersize=2, color='green',
            label='Base Station Assignment')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Base Station ID')
    ax2.set_title(f'Proactive Handover - Base Station Assignment (User {user_id})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    return fig


def create_summary_figure(
    mobility_data,
    base_stations,
    reactive_results,
    proactive_results,
    reactive_metrics,
    proactive_metrics
):
    """
    Create a comprehensive summary figure with multiple subplots.
    
    Args:
        mobility_data: DataFrame with mobility data
        base_stations: List of BaseStation objects
        reactive_results: Reactive handover results dict
        proactive_results: Proactive handover results dict
        reactive_metrics: Reactive metrics dict
        proactive_metrics: Proactive metrics dict
    """
    set_plot_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Trajectory
    ax1 = fig.add_subplot(gs[0, :2])
    user_traj = get_user_trajectory(mobility_data, 0)
    bs_positions = get_base_station_positions(base_stations)
    ax1.plot(user_traj['x'], user_traj['y'], 'b-', alpha=0.5, label='Trajectory')
    ax1.scatter(bs_positions['x'], bs_positions['y'], s=300, c='red', marker='s',
               edgecolors='darkred', linewidth=2, label='Base Stations', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('User Trajectory and Base Stations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Metrics comparison
    ax2 = fig.add_subplot(gs[0, 2])
    metrics_labels = ['Handovers', 'Latency', 'Packet Loss']
    reactive_vals = [
        reactive_metrics.get('total_handovers', 0),
        reactive_metrics.get('accumulated_latency', 0) / 100,
        reactive_metrics.get('avg_packet_loss', 0) * 100
    ]
    proactive_vals = [
        proactive_metrics.get('total_handovers', 0),
        proactive_metrics.get('accumulated_latency', 0) / 100,
        proactive_metrics.get('avg_packet_loss', 0) * 100
    ]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    ax2.bar(x - width/2, reactive_vals, width, label='Reactive', alpha=0.7)
    ax2.bar(x + width/2, proactive_vals, width, label='Proactive', alpha=0.7)
    ax2.set_ylabel('Value')
    ax2.set_title('Key Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Signal Strength
    ax3 = fig.add_subplot(gs[1, :])
    time_steps = np.arange(min(len(reactive_results['history']['rss']),
                              len(proactive_results['history']['rss'])))
    ax3.plot(time_steps, reactive_results['history']['rss'][:len(time_steps)],
            label='Reactive', alpha=0.7, linewidth=1)
    ax3.plot(time_steps, proactive_results['history']['rss'][:len(time_steps)],
            label='Proactive', alpha=0.7, linewidth=1)
    ax3.axhline(y=0.3, color='red', linestyle='--', label='Threshold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Signal Strength (RSS)')
    ax3.set_title('Signal Strength Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Throughput
    ax4 = fig.add_subplot(gs[2, 0])
    time_steps = np.arange(min(len(reactive_results['history']['throughput']),
                              len(proactive_results['history']['throughput'])))
    ax4.plot(time_steps, reactive_results['history']['throughput'][:len(time_steps)],
            label='Reactive', alpha=0.7, linewidth=1)
    ax4.plot(time_steps, proactive_results['history']['throughput'][:len(time_steps)],
            label='Proactive', alpha=0.7, linewidth=1)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Throughput (Mbps)')
    ax4.set_title('Throughput Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Packet Loss
    ax5 = fig.add_subplot(gs[2, 1])
    time_steps = np.arange(min(len(reactive_results['history']['packet_loss']),
                              len(proactive_results['history']['packet_loss'])))
    ax5.plot(time_steps, reactive_results['history']['packet_loss'][:len(time_steps)],
            label='Reactive', alpha=0.7, linewidth=1)
    ax5.plot(time_steps, proactive_results['history']['packet_loss'][:len(time_steps)],
            label='Proactive', alpha=0.7, linewidth=1)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Packet Loss Rate')
    ax5.set_title('Packet Loss Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Handover Count
    ax6 = fig.add_subplot(gs[2, 2])
    reactive_count = reactive_metrics.get('total_handovers', 0)
    proactive_count = proactive_metrics.get('total_handovers', 0)
    approaches = ['Reactive', 'Proactive']
    counts = [reactive_count, proactive_count]
    bars = ax6.bar(approaches, counts, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    ax6.set_ylabel('Number of Handovers')
    ax6.set_title('Total Handovers Comparison')
    ax6.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Wireless Network Handover Simulation - Summary', 
                fontsize=14, fontweight='bold', y=0.995)
    
    return fig


if __name__ == "__main__":
    print("Visualization module loaded. Import and use functions as needed.")
