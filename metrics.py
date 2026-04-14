"""
Performance Metrics Calculation for handover comparison.
Computes various metrics to evaluate handover effectiveness.
"""

import numpy as np
import pandas as pd


class MetricsCalculator:
    """Calculate and track handover performance metrics."""
    
    @staticmethod
    def calculate_metrics(handover_stats, signal_stats):
        """
        Calculate comprehensive metrics for handover performance.
        
        Args:
            handover_stats: Dict with handover information
            signal_stats: Dict with signal quality information
        
        Returns:
            Dict with calculated metrics
        """
        metrics = {}
        
        # Handover metrics
        metrics['total_handovers'] = handover_stats.get('handover_count', 0)
        metrics['unnecessary_handovers'] = handover_stats.get('unnecessary_handovers', 0)
        metrics['handover_rate'] = (
            metrics['total_handovers'] / max(1, handover_stats.get('total_timesteps', 1))
        )
        
        # Latency metrics
        metrics['accumulated_latency'] = handover_stats.get('latency', 0)
        metrics['avg_latency_per_handover'] = (
            metrics['accumulated_latency'] / max(1, metrics['total_handovers'])
        )
        
        # Signal quality metrics
        metrics['avg_rss'] = signal_stats.get('avg_rss', 0)
        metrics['min_rss'] = signal_stats.get('min_rss', 0)
        metrics['max_rss'] = signal_stats.get('max_rss', 0)
        metrics['time_below_threshold'] = signal_stats.get('time_below_threshold', 0)
        metrics['poor_signal_percentage'] = (
            (metrics['time_below_threshold'] / 
             max(1, handover_stats.get('total_timesteps', 1))) * 100
        )
        
        # Packet loss metrics
        metrics['avg_packet_loss'] = signal_stats.get('avg_packet_loss', 0)
        metrics['total_packets_lost'] = signal_stats.get('total_packets_lost', 0)
        
        # Throughput metrics
        metrics['avg_throughput'] = signal_stats.get('avg_throughput', 0)
        
        return metrics
    
    @staticmethod
    def calculate_prediction_accuracy(actual_positions, predicted_positions):
        """
        Calculate prediction accuracy metrics.
        
        Args:
            actual_positions: Array of actual positions (n, 2)
            predicted_positions: Array of predicted positions (n, 2)
        
        Returns:
            Dict with accuracy metrics
        """
        if len(actual_positions) == 0:
            return {}
        
        # Calculate Euclidean distances
        distances = np.sqrt(
            np.sum((actual_positions - predicted_positions) ** 2, axis=1)
        )
        
        metrics = {
            'mean_prediction_error': np.mean(distances),
            'median_prediction_error': np.median(distances),
            'std_prediction_error': np.std(distances),
            'max_prediction_error': np.max(distances),
            'min_prediction_error': np.min(distances),
            'rmse': np.sqrt(np.mean(distances ** 2))
        }
        
        return metrics
    
    @staticmethod
    def create_comparison_table(reactive_metrics, proactive_metrics):
        """
        Create a comparison table of metrics.
        
        Args:
            reactive_metrics: Dict with reactive handover metrics
            proactive_metrics: Dict with proactive handover metrics
        
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        all_keys = set(reactive_metrics.keys()) | set(proactive_metrics.keys())
        
        for key in sorted(all_keys):
            reactive_val = reactive_metrics.get(key, 0)
            proactive_val = proactive_metrics.get(key, 0)
            
            # Calculate improvement
            if reactive_val != 0 and proactive_val != 0:
                # For metrics where lower is better (latency, packet loss, handovers)
                if key in ['total_handovers', 'unnecessary_handovers', 'accumulated_latency',
                          'avg_latency_per_handover', 'avg_packet_loss', 'time_below_threshold',
                          'poor_signal_percentage', 'total_packets_lost']:
                    improvement = ((reactive_val - proactive_val) / abs(reactive_val)) * 100
                # For metrics where higher is better (RSS, throughput)
                else:
                    improvement = ((proactive_val - reactive_val) / abs(reactive_val)) * 100
            else:
                improvement = 0
            
            comparison_data.append({
                'Metric': key.replace('_', ' ').title(),
                'Reactive': f"{reactive_val:.4f}" if isinstance(reactive_val, float) else reactive_val,
                'Proactive': f"{proactive_val:.4f}" if isinstance(proactive_val, float) else proactive_val,
                'Improvement (%)': f"{improvement:.2f}"
            })
        
        return pd.DataFrame(comparison_data)


def collect_signal_statistics(rss_history, threshold=0.3):
    """
    Collect signal quality statistics from RSS history.
    
    Args:
        rss_history: List of RSS values over time
        threshold: Signal strength threshold
    
    Returns:
        Dict with signal statistics
    """
    if not rss_history:
        return {}
    
    rss_array = np.array(rss_history)
    
    return {
        'avg_rss': np.mean(rss_array),
        'min_rss': np.min(rss_array),
        'max_rss': np.max(rss_array),
        'time_below_threshold': np.sum(rss_array < threshold),
        'avg_packet_loss': np.mean([
            max(0, 1.0 - (rss / threshold) ** 2) if rss < threshold else 0.01
            for rss in rss_array
        ]),
        'avg_throughput': np.mean([
            100.0 * (rss ** 1.5) for rss in rss_array
        ])
    }


def collect_throughput_statistics(throughput_history):
    """
    Collect throughput statistics.
    
    Args:
        throughput_history: List of throughput values over time
    
    Returns:
        Dict with throughput statistics
    """
    if not throughput_history:
        return {'avg_throughput': 0, 'total_packets_lost': 0}
    
    throughput_array = np.array(throughput_history)
    
    return {
        'avg_throughput': np.mean(throughput_array),
        'min_throughput': np.min(throughput_array),
        'max_throughput': np.max(throughput_array)
    }


def format_metrics_for_display(metrics):
    """
    Format metrics dictionary for pretty printing.
    
    Args:
        metrics: Dict with metrics
    
    Returns:
        Formatted string
    """
    output = []
    output.append("=" * 60)
    output.append("PERFORMANCE METRICS")
    output.append("=" * 60)
    
    for key, value in sorted(metrics.items()):
        formatted_key = key.replace('_', ' ').title()
        if isinstance(value, float):
            output.append(f"{formatted_key:.<40} {value:>15.4f}")
        else:
            output.append(f"{formatted_key:.<40} {value:>15}")
    
    output.append("=" * 60)
    return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    calculator = MetricsCalculator()
    
    # Sample data
    reactive_stats = {
        'handover_count': 45,
        'unnecessary_handovers': 12,
        'latency': 900,  # 900ms total
        'total_timesteps': 500
    }
    
    reactive_signal = {
        'avg_rss': 0.45,
        'min_rss': 0.15,
        'max_rss': 0.95,
        'time_below_threshold': 85,
        'avg_packet_loss': 0.08,
        'avg_throughput': 65.5
    }
    
    proactive_stats = {
        'handover_count': 32,
        'unnecessary_handovers': 5,
        'latency': 480,  # 480ms total
        'total_timesteps': 500
    }
    
    proactive_signal = {
        'avg_rss': 0.58,
        'min_rss': 0.25,
        'max_rss': 0.96,
        'time_below_threshold': 45,
        'avg_packet_loss': 0.03,
        'avg_throughput': 78.2
    }
    
    reactive_metrics = calculator.calculate_metrics(reactive_stats, reactive_signal)
    proactive_metrics = calculator.calculate_metrics(proactive_stats, proactive_signal)
    
    print(calculator.create_comparison_table(reactive_metrics, proactive_metrics))
