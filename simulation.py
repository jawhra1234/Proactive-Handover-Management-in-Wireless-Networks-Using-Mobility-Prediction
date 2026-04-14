"""
Main Simulation Loop for Wireless Network Handover Comparison.
Orchestrates the comparison between reactive and proactive handover.
"""

import numpy as np
import pandas as pd
from data_generation import generate_user_mobility, get_user_trajectory
from base_station import create_base_stations, get_nearest_base_station
from signal_model import SignalStrengthModel, compute_all_rss_values, packet_loss_rate, simulate_throughput
from lstm_model import MobilityPredictor
from handover import ReactiveHandover, ProactiveHandover
from metrics import MetricsCalculator, collect_signal_statistics, collect_throughput_statistics


class WirelessNetworkSimulation:
    """
    Comprehensive wireless network simulation with reactive and proactive handover.
    """
    
    def __init__(
        self,
        num_users=3,
        num_base_stations=5,
        simulation_time=500,
        sequence_length=20,
        seed=42
    ):
        """
        Initialize simulation parameters.
        
        Args:
            num_users: Number of mobile users
            num_base_stations: Number of base stations
            simulation_time: Duration of simulation in time steps
            sequence_length: ML sequence length
            seed: Random seed
        """
        self.num_users = num_users
        self.num_base_stations = num_base_stations
        self.simulation_time = simulation_time
        self.sequence_length = sequence_length
        self.seed = seed
        
        # Data structures
        self.mobility_data = None
        self.base_stations = None
        self.signal_model = None
        self.predictor = None
        self.reactive_handover = None
        self.proactive_handover = None
        
        # Results
        self.reactive_results = {}
        self.proactive_results = {}
        self.comparison_results = {}
    
    def setup(self):
        """Initialize all simulation components."""
        print("[Setup] Generating mobility data...")
        self.mobility_data = generate_user_mobility(
            num_users=self.num_users,
            num_timesteps=self.simulation_time,
            speed=2.0,
            noise_scale=0.5,
            grid_size=100.0,
            seed=self.seed
        )
        
        print("[Setup] Creating base stations...")
        self.base_stations = create_base_stations(
            num_bs=self.num_base_stations,
            grid_size=100.0,
            seed=self.seed
        )
        print(f"         {len(self.base_stations)} base stations created")
        
        print("[Setup] Initializing signal model...")
        self.signal_model = SignalStrengthModel(
            max_power=1.0,
            epsilon=0.1,
            noise_scale=0.05,
            seed=self.seed
        )
        
        print("[Setup] Building and training ML model...")
        self.predictor = MobilityPredictor(sequence_length=self.sequence_length, seed=self.seed)
        
        # Prepare training data
        trajectories = [
            get_user_trajectory(self.mobility_data, user_id)
            for user_id in range(self.num_users)
        ]
        
        # Train on first 80% of data
        train_trajectories = []
        for traj in trajectories:
            train_size = int(0.8 * len(traj))
            train_trajectories.append(traj.iloc[:train_size])
        
        self.predictor.train(train_trajectories, epochs=20, verbose=0)
        print("         ML model trained successfully")
        
        print("[Setup] Initializing handover controllers...")
        self.reactive_handover = ReactiveHandover(rss_threshold=0.3)
        self.proactive_handover = ProactiveHandover(
            predictor=self.predictor,
            rss_threshold=0.25
        )
    
    def run_simulation(self):
        """Run the main simulation loop."""
        print("\n[Simulation] Starting comparison simulation...")
        
        # Initialize data collectors
        reactive_history = {
            'rss': [],
            'bs_assignment': [],
            'handovers': [],
            'throughput': [],
            'packet_loss': []
        }
        
        proactive_history = {
            'rss': [],
            'bs_assignment': [],
            'handovers': [],
            'throughput': [],
            'packet_loss': []
        }
        
        # Simulate for each user
        for user_id in range(self.num_users):
            print(f"\n  [User {user_id}] Simulating trajectory...")
            
            user_trajectory = get_user_trajectory(self.mobility_data, user_id)
            
            # Initialize current BS for both approaches
            x0, y0 = user_trajectory.iloc[0][['x', 'y']]
            starting_bs, _ = get_nearest_base_station(self.base_stations, x0, y0)
            
            # Reset handover controllers
            self.reactive_handover.current_bs = starting_bs
            self.proactive_handover.current_bs = starting_bs
            
            # Maintain recent position buffer for ML prediction
            recent_positions = []
            
            # Simulate each timestep
            for t in range(1, len(user_trajectory)):
                x_t = user_trajectory.iloc[t]['x']
                y_t = user_trajectory.iloc[t]['y']
                
                # Update recent positions buffer
                recent_positions.append([x_t, y_t])
                if len(recent_positions) > self.sequence_length:
                    recent_positions.pop(0)
                
                # ===== REACTIVE HANDOVER =====
                rss_reactive = compute_all_rss_values(
                    self.base_stations, x_t, y_t, self.signal_model
                )
                current_rss_reactive = rss_reactive[self.reactive_handover.current_bs.bs_id]
                
                # Decide on handover
                new_bs_id_reactive, ho_occurred = self.reactive_handover.decide_handover(
                    rss_reactive, self.reactive_handover.current_bs, t
                )
                
                # Update current BS if handover occurred
                if ho_occurred:
                    for bs in self.base_stations:
                        if bs.bs_id == new_bs_id_reactive:
                            self.reactive_handover.current_bs = bs
                            break
                
                # Record metrics
                reactive_history['rss'].append(current_rss_reactive)
                reactive_history['bs_assignment'].append(self.reactive_handover.current_bs.bs_id)
                reactive_history['handovers'].append(len(self.reactive_handover.handover_history))
                throughput_r = simulate_throughput(current_rss_reactive)
                reactive_history['throughput'].append(throughput_r)
                ploss_r = packet_loss_rate(current_rss_reactive)
                reactive_history['packet_loss'].append(ploss_r)
                
                # ===== PROACTIVE HANDOVER =====
                rss_proactive = compute_all_rss_values(
                    self.base_stations, x_t, y_t, self.signal_model
                )
                current_rss_proactive = rss_proactive[self.proactive_handover.current_bs.bs_id]
                
                if len(recent_positions) >= self.sequence_length:
                    # Decide on handover
                    new_bs_id_proactive, ho_occurred = self.proactive_handover.decide_handover(
                        rss_proactive,
                        self.proactive_handover.current_bs,
                        np.array(recent_positions),
                        self.base_stations,
                        t,
                        self.signal_model
                    )
                    
                    # Update current BS if handover occurred
                    if ho_occurred:
                        for bs in self.base_stations:
                            if bs.bs_id == new_bs_id_proactive:
                                self.proactive_handover.current_bs = bs
                                break
                
                # Record metrics
                proactive_history['rss'].append(current_rss_proactive)
                proactive_history['bs_assignment'].append(self.proactive_handover.current_bs.bs_id)
                proactive_history['handovers'].append(len(self.proactive_handover.handover_history))
                throughput_p = simulate_throughput(current_rss_proactive)
                proactive_history['throughput'].append(throughput_p)
                ploss_p = packet_loss_rate(current_rss_proactive)
                proactive_history['packet_loss'].append(ploss_p)
        
        # Store results
        self.reactive_results = {
            'history': reactive_history,
            'handover_obj': self.reactive_handover
        }
        
        self.proactive_results = {
            'history': proactive_history,
            'handover_obj': self.proactive_handover
        }
        
        print("\n[Simulation] Simulation completed!")
    
    def calculate_metrics(self):
        """Calculate performance metrics for both approaches."""
        print("\n[Metrics] Calculating performance metrics...")
        
        calculator = MetricsCalculator()
        
        # Reactive metrics
        reactive_handover_stats = {
            'handover_count': self.reactive_handover.handover_count,
            'unnecessary_handovers': self.reactive_handover.unnecessary_handovers,
            'latency': self.reactive_handover.latency_accumulation,
            'total_timesteps': self.simulation_time
        }
        reactive_signal_stats = collect_signal_statistics(
            self.reactive_results['history']['rss'],
            threshold=0.3
        )
        reactive_signal_stats['total_packets_lost'] = np.sum(
            self.reactive_results['history']['packet_loss']
        )
        
        self.reactive_metrics = calculator.calculate_metrics(
            reactive_handover_stats, reactive_signal_stats
        )
        
        # Proactive metrics
        proactive_handover_stats = {
            'handover_count': self.proactive_handover.handover_count,
            'unnecessary_handovers': self.proactive_handover.unnecessary_handovers,
            'latency': self.proactive_handover.latency_accumulation,
            'total_timesteps': self.simulation_time
        }
        proactive_signal_stats = collect_signal_statistics(
            self.proactive_results['history']['rss'],
            threshold=0.3
        )
        proactive_signal_stats['total_packets_lost'] = np.sum(
            self.proactive_results['history']['packet_loss']
        )
        
        self.proactive_metrics = calculator.calculate_metrics(
            proactive_handover_stats, proactive_signal_stats
        )
        
        return self.reactive_metrics, self.proactive_metrics
    
    def get_comparison_table(self):
        """Generate comparison table."""
        calculator = MetricsCalculator()
        return calculator.create_comparison_table(
            self.reactive_metrics, self.proactive_metrics
        )
    
    def get_reactive_handovers(self):
        """Get reactive handover history."""
        return self.reactive_handover.handover_history
    
    def get_proactive_handovers(self):
        """Get proactive handover history."""
        return self.proactive_handover.handover_history
    
    def get_results_summary(self):
        """Get summary of results."""
        return {
            'reactive_metrics': self.reactive_metrics,
            'proactive_metrics': self.proactive_metrics,
            'reactive_handovers': self.get_reactive_handovers(),
            'proactive_handovers': self.get_proactive_handovers(),
            'base_stations': self.base_stations,
            'mobility_data': self.mobility_data
        }


if __name__ == "__main__":
    # Example usage
    sim = WirelessNetworkSimulation(
        num_users=3,
        num_base_stations=5,
        simulation_time=300,
        sequence_length=20
    )
    
    sim.setup()
    sim.run_simulation()
    reactive_m, proactive_m = sim.calculate_metrics()
    
    print("\n" + "="*80)
    print("SIMULATION RESULTS")
    print("="*80)
    print(sim.get_comparison_table())
