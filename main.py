"""
Main entry point for Proactive Handover Management in Wireless Networks using Mobility Prediction.
Run with: python main.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from simulation import WirelessNetworkSimulation
from visualization import (
    plot_user_trajectory_and_base_stations,
    plot_signal_strength_over_time,
    plot_metrics_comparison,
    plot_handover_timeline,
    plot_base_station_assignment,
    create_summary_figure,
    set_plot_style
)
from metrics import MetricsCalculator


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_section(text):
    """Print a formatted section."""
    print(f"\n{'─'*80}")
    print(f"  {text}")
    print(f"{'─'*80}")


def main():
    """Main execution function."""
    
    print_header("PROACTIVE HANDOVER MANAGEMENT IN WIRELESS NETWORKS")
    print("Using Mobility Prediction with Machine Learning\n")
    
    # Configuration
    print("[Config] Simulation Parameters:")
    num_users = 3
    num_base_stations = 5
    simulation_time = 500
    sequence_length = 20
    
    print(f"  - Number of users: {num_users}")
    print(f"  - Number of base stations: {num_base_stations}")
    print(f"  - Simulation duration: {simulation_time} time steps")
    print(f"  - ML sequence length: {sequence_length}")
    print(f"  - Random seed: 42 (for reproducibility)")
    
    # Initialize and setup simulation
    print_section("PHASE 1: INITIALIZATION")
    print("[1/4] Creating wireless network simulation...")
    
    simulation = WirelessNetworkSimulation(
        num_users=num_users,
        num_base_stations=num_base_stations,
        simulation_time=simulation_time,
        sequence_length=sequence_length,
        seed=42
    )
    
    print("[2/4] Setting up simulation components...")
    simulation.setup()
    
    # Run simulation
    print_section("PHASE 2: SIMULATION")
    print("[1/1] Running comparative handover simulation...")
    simulation.run_simulation()
    
    # Calculate metrics
    print_section("PHASE 3: ANALYSIS")
    print("[1/2] Calculating performance metrics...")
    reactive_metrics, proactive_metrics = simulation.calculate_metrics()
    
    # Display comparison table
    print("[2/2] Generating comparison results...")
    print_section("PERFORMANCE METRICS COMPARISON")
    comparison_table = simulation.get_comparison_table()
    print("\n" + comparison_table.to_string(index=False))
    
    # Print detailed statistics
    print_section("DETAILED STATISTICS")
    
    print("\nREACTIVE HANDOVER:")
    print(f"  Total Handovers: {reactive_metrics['total_handovers']}")
    print(f"  Unnecessary Handovers: {reactive_metrics['unnecessary_handovers']}")
    print(f"  Total Latency: {reactive_metrics['accumulated_latency']:.2f} ms")
    print(f"  Average RSS: {reactive_metrics['avg_rss']:.4f}")
    print(f"  Poor Signal Time: {reactive_metrics['time_below_threshold']} steps")
    print(f"  Average Packet Loss: {reactive_metrics['avg_packet_loss']:.4f}")
    print(f"  Average Throughput: {reactive_metrics['avg_throughput']:.2f} Mbps")
    
    print("\nPROACTIVE HANDOVER:")
    print(f"  Total Handovers: {proactive_metrics['total_handovers']}")
    print(f"  Unnecessary Handovers: {proactive_metrics['unnecessary_handovers']}")
    print(f"  Total Latency: {proactive_metrics['accumulated_latency']:.2f} ms")
    print(f"  Average RSS: {proactive_metrics['avg_rss']:.4f}")
    print(f"  Poor Signal Time: {proactive_metrics['time_below_threshold']} steps")
    print(f"  Average Packet Loss: {proactive_metrics['avg_packet_loss']:.4f}")
    print(f"  Average Throughput: {proactive_metrics['avg_throughput']:.2f} Mbps")
    
    # Calculate improvements
    print_section("PROACTIVE HANDOVER IMPROVEMENTS")
    
    handover_reduction = (
        (1 - proactive_metrics['total_handovers'] / 
         max(1, reactive_metrics['total_handovers'])) * 100
    )
    latency_reduction = (
        (1 - proactive_metrics['accumulated_latency'] / 
         max(1, reactive_metrics['accumulated_latency'])) * 100
    )
    packet_loss_reduction = (
        (1 - proactive_metrics['avg_packet_loss'] / 
         max(0.001, reactive_metrics['avg_packet_loss'])) * 100
    )
    throughput_improvement = (
        ((proactive_metrics['avg_throughput'] - reactive_metrics['avg_throughput']) /
         reactive_metrics['avg_throughput']) * 100
    )
    poor_signal_reduction = (
        (1 - proactive_metrics['time_below_threshold'] / 
         max(1, reactive_metrics['time_below_threshold'])) * 100
    )
    
    print(f"\n✓ Handover Reduction: {handover_reduction:.2f}%")
    print(f"  ({reactive_metrics['total_handovers']} → {proactive_metrics['total_handovers']})")
    
    print(f"\n✓ Latency Reduction: {latency_reduction:.2f}%")
    print(f"  ({reactive_metrics['accumulated_latency']:.2f}ms → {proactive_metrics['accumulated_latency']:.2f}ms)")
    
    print(f"\n✓ Packet Loss Reduction: {packet_loss_reduction:.2f}%")
    print(f"  ({reactive_metrics['avg_packet_loss']:.4f} → {proactive_metrics['avg_packet_loss']:.4f})")
    
    print(f"\n✓ Throughput Improvement: {throughput_improvement:.2f}%")
    print(f"  ({reactive_metrics['avg_throughput']:.2f} → {proactive_metrics['avg_throughput']:.2f} Mbps)")
    
    print(f"\n✓ Poor Signal Reduction: {poor_signal_reduction:.2f}%")
    print(f"  ({reactive_metrics['time_below_threshold']} → {proactive_metrics['time_below_threshold']} steps)")
    
    # Generate visualizations
    print_section("PHASE 4: VISUALIZATION")
    print("[1/5] Generating trajectory plot...")
    
    results = simulation.get_results_summary()
    
    # Create output directory for plots
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    set_plot_style()
    
    # Plot 1: Trajectory
    fig1, _ = plot_user_trajectory_and_base_stations(
        results['mobility_data'],
        results['base_stations'],
        user_id=0,
        title="User Trajectory with Base Stations (User 0)"
    )
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "01_trajectory.png"), dpi=150, bbox_inches='tight')
    print(f"      Saved: {output_dir}/01_trajectory.png")
    
    # Plot 2: Signal Strength
    print("[2/5] Generating signal strength plot...")
    fig2, _ = plot_signal_strength_over_time(
        simulation.reactive_results['history']['rss'],
        simulation.proactive_results['history']['rss'],
        threshold=0.3,
        title="Signal Strength Comparison: Reactive vs Proactive"
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "02_signal_strength.png"), dpi=150, bbox_inches='tight')
    print(f"      Saved: {output_dir}/02_signal_strength.png")
    
    # Plot 3: Metrics Comparison
    print("[3/5] Generating metrics comparison plot...")
    fig3 = plot_metrics_comparison(reactive_metrics, proactive_metrics)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "03_metrics_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"      Saved: {output_dir}/03_metrics_comparison.png")
    
    # Plot 4: Handover Timeline
    print("[4/5] Generating handover timeline plot...")
    fig4, _ = plot_handover_timeline(
        results['reactive_handovers'],
        results['proactive_handovers']
    )
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "04_handover_timeline.png"), dpi=150, bbox_inches='tight')
    print(f"      Saved: {output_dir}/04_handover_timeline.png")
    
    # Plot 5: Summary Figure
    print("[5/5] Generating comprehensive summary figure...")
    fig5 = create_summary_figure(
        results['mobility_data'],
        results['base_stations'],
        simulation.reactive_results,
        simulation.proactive_results,
        reactive_metrics,
        proactive_metrics
    )
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_dir, "05_summary_figure.png"), dpi=150, bbox_inches='tight')
    print(f"      Saved: {output_dir}/05_summary_figure.png")
    
    # Display plots
    print("\nDisplaying plots...")
    plt.show()
    
    # Final summary
    print_header("SIMULATION COMPLETE")
    print("\n✓ Simulation successfully completed!")
    print(f"✓ All plots saved to '{output_dir}/' directory")
    print(f"\nKey Finding:")
    print(f"  Proactive handover using mobility prediction achieved:")
    print(f"  - {handover_reduction:.1f}% reduction in handovers")
    print(f"  - {latency_reduction:.1f}% reduction in latency")
    print(f"  - {packet_loss_reduction:.1f}% reduction in packet loss")
    print(f"  - {throughput_improvement:.1f}% improvement in throughput")
    print(f"\nThis demonstrates the significant benefits of using mobility")
    print(f"prediction for proactive handover management in wireless networks.")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
