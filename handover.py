"""
Handover Management - Reactive and Proactive approaches.
Handles handover decisions and tracking for wireless network simulation.
"""

import numpy as np


class ReactiveHandover:
    """
    Traditional reactive handover approach.
    Switches to a new base station when signal drops below threshold.
    """
    
    def __init__(self, rss_threshold=0.3, hysteresis=0.05):
        """
        Initialize reactive handover.
        
        Args:
            rss_threshold: Signal strength threshold for handover
            hysteresis: Hysteresis margin to prevent ping-pong effect
        """
        self.rss_threshold = rss_threshold
        self.hysteresis = hysteresis
        self.current_bs = None
        self.handover_count = 0
        self.unnecessary_handovers = 0
        self.previous_bs = None
        self.last_handover_time = -1
        self.latency_accumulation = 0
        self.handover_history = []
    
    def decide_handover(self, rss_values, current_bs, time_step, min_handover_interval=1):
        """
        Decide whether to perform handover based on signal strength.
        
        Args:
            rss_values: Dict {bs_id: rss_value}
            current_bs: Current base station object
            time_step: Current simulation time step
            min_handover_interval: Minimum steps between handovers
        
        Returns:
            Tuple: (new_bs, handover_occurred)
        """
        current_rss = rss_values.get(current_bs.bs_id, 0)
        
        # Check if handover needed
        if current_rss < self.rss_threshold:
            # Find best alternative BS
            best_bs = None
            best_rss = current_rss
            
            for bs_id, rss in rss_values.items():
                if bs_id != current_bs.bs_id and rss > best_rss + self.hysteresis:
                    if rss > best_rss:
                        best_bs_rss = rss
                        best_bs = None
                        for bs_id_temp, rss_temp in rss_values.items():
                            if rss_temp > best_bs_rss:
                                best_bs_rss = rss_temp
                                best_bs = bs_id_temp
            
            # Simpler approach: just find the best BS
            best_bs = max(rss_values.keys(), key=lambda x: rss_values[x])
            best_rss = rss_values[best_bs]
            
            # Only perform handover if there's a significant improvement
            if best_bs != current_bs.bs_id and best_rss > current_rss + self.hysteresis:
                if time_step - self.last_handover_time >= min_handover_interval:
                    self.handover_count += 1
                    self.previous_bs = current_bs
                    
                    # Simulate latency during handover
                    self.latency_accumulation += 20  # 20ms latency per handover
                    
                    # Track if this is an unnecessary handover (ping-pong)
                    if self.previous_bs and best_bs == self.previous_bs.bs_id:
                        self.unnecessary_handovers += 1
                    
                    self.last_handover_time = time_step
                    self.handover_history.append({
                        'time': time_step,
                        'from_bs': current_bs.bs_id,
                        'to_bs': best_bs,
                        'rss_before': current_rss,
                        'rss_after': best_rss,
                        'type': 'reactive'
                    })
                    return best_bs, True
        
        return current_bs.bs_id, False
    
    def reset(self):
        """Reset handover statistics."""
        self.handover_count = 0
        self.unnecessary_handovers = 0
        self.previous_bs = None
        self.last_handover_time = -1
        self.latency_accumulation = 0
        self.handover_history = []


class ProactiveHandover:
    """
    Proactive handover using mobility prediction.
    Performs handover before signal drops using predicted next position.
    """
    
    def __init__(self, predictor, rss_threshold=0.25, lead_time=0):
        """
        Initialize proactive handover.
        
        Args:
            predictor: MobilityPredictor instance
            rss_threshold: Signal strength threshold for proactive trigger
            lead_time: Time steps ahead to use for prediction (0 = next step)
        """
        self.predictor = predictor
        self.rss_threshold = rss_threshold
        self.lead_time = lead_time
        self.current_bs = None
        self.handover_count = 0
        self.unnecessary_handovers = 0
        self.previous_bs = None
        self.last_handover_time = -1
        self.latency_accumulation = 0
        self.handover_history = []
        self.prediction_errors = []
    
    def decide_handover(
        self,
        rss_values,
        current_bs,
        recent_positions,
        base_stations,
        time_step,
        signal_model,
        min_handover_interval=1
    ):
        """
        Decide handover based on predicted next position.
        
        Args:
            rss_values: Dict {bs_id: rss_value} current RSS
            current_bs: Current base station object
            recent_positions: Array of recent (x, y) coordinates
            base_stations: List of all base station objects
            time_step: Current simulation time step
            signal_model: SignalStrengthModel instance
            min_handover_interval: Minimum steps between handovers
        
        Returns:
            Tuple: (new_bs, handover_occurred)
        """
        current_rss = rss_values.get(current_bs.bs_id, 0)
        
        # Predict next position
        pred_x, pred_y = self.predictor.predict_next_position(recent_positions)
        
        # Calculate RSS at predicted position for all BSs
        pred_rss_values = {}
        for bs in base_stations:
            distance = bs.distance_to_user(pred_x, pred_y)
            pred_rss = signal_model.compute_rss(distance, add_noise=False)
            pred_rss_values[bs.bs_id] = pred_rss
        
        # Get predicted best BS
        pred_best_bs_id = max(pred_rss_values.keys(), key=lambda x: pred_rss_values[x])
        pred_best_rss = pred_rss_values[pred_best_bs_id]
        
        # Perform proactive handover if:
        # 1. Predicted BS differs from current BS
        # 2. OR predicted signal will be weak
        if pred_best_bs_id != current_bs.bs_id or pred_best_rss < self.rss_threshold:
            if time_step - self.last_handover_time >= min_handover_interval:
                # Find which BS to switch to
                if pred_best_bs_id != current_bs.bs_id:
                    new_bs_id = pred_best_bs_id
                else:
                    # Find alternative to avoid poor signal
                    new_bs_id = max(
                        pred_rss_values.keys(),
                        key=lambda x: pred_rss_values[x] if x != current_bs.bs_id else -1
                    )
                
                self.handover_count += 1
                self.previous_bs = current_bs
                
                # Reduced latency for proactive handover (better preparation)
                self.latency_accumulation += 10  # 10ms latency (reduced)
                
                self.last_handover_time = time_step
                self.handover_history.append({
                    'time': time_step,
                    'from_bs': current_bs.bs_id,
                    'to_bs': new_bs_id,
                    'rss_before': current_rss,
                    'rss_after': rss_values.get(new_bs_id, 0),
                    'predicted_pos': (pred_x, pred_y),
                    'type': 'proactive'
                })
                
                return new_bs_id, True
        
        return current_bs.bs_id, False
    
    def reset(self):
        """Reset handover statistics."""
        self.handover_count = 0
        self.unnecessary_handovers = 0
        self.previous_bs = None
        self.last_handover_time = -1
        self.latency_accumulation = 0
        self.handover_history = []
        self.prediction_errors = []


def compare_handover_strategies(reactive_stats, proactive_stats):
    """
    Compare reactive and proactive handover performance.
    
    Args:
        reactive_stats: Dict with reactive handover metrics
        proactive_stats: Dict with proactive handover metrics
    
    Returns:
        Dict with comparison results
    """
    comparison = {
        'handover_count_diff': (
            proactive_stats['handover_count'] - reactive_stats['handover_count']
        ),
        'handover_count_reduction_percent': (
            (1 - proactive_stats['handover_count'] / (reactive_stats['handover_count'] + 1))
            * 100
        ),
        'latency_diff': (
            proactive_stats['latency'] - reactive_stats['latency']
        ),
        'latency_reduction_percent': (
            (1 - proactive_stats['latency'] / (reactive_stats['latency'] + 1))
            * 100
        ),
        'packet_loss_improvement': (
            reactive_stats['packet_loss'] - proactive_stats['packet_loss']
        )
    }
    return comparison


if __name__ == "__main__":
    # Example usage
    reactive = ReactiveHandover(rss_threshold=0.3)
    print(f"Created reactive handover controller")
    print(f"  Threshold: {reactive.rss_threshold}")
    print(f"  Hysteresis: {reactive.hysteresis}")
