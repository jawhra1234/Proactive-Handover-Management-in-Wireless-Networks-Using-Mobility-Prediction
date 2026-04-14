"""
Machine Learning-based Mobility Prediction Model.
Predicts future user position based on historical trajectory using Linear Regression.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MobilityPredictor:
    """ML model for predicting next user position using Linear Regression."""

    def __init__(self, sequence_length=20, prediction_steps=5, seed=42):
        """
        Initialize mobility predictor.

        Args:
            sequence_length: Number of past steps to use for prediction (k)
            prediction_steps: Number of steps ahead to predict
            seed: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.scaler = StandardScaler()
        self.model_x = LinearRegression()  # Model for predicting X coordinate
        self.model_y = LinearRegression()  # Model for predicting Y coordinate
        self.is_trained = False

        # Set seeds for reproducibility
        np.random.seed(seed)

    def create_feature_vectors(self, data, sequence_length):
        """
        Create feature vectors for ML training.

        Args:
            data: Input data array of shape (n_samples, 2) with [x, y] coordinates
            sequence_length: Length of each sequence

        Returns:
            Tuple: (X, y_x, y_y) where X has shape (n_sequences, sequence_length*2)
                   and y_x, y_y are target coordinates
        """
        X, y_x, y_y = [], [], []

        for i in range(len(data) - sequence_length):
            # Create feature vector: flatten last k positions [x1,y1,x2,y2,...,xk,yk]
            sequence = data[i:i + sequence_length]
            features = sequence.flatten()  # Shape: (sequence_length * 2,)
            X.append(features)

            # Target: next position after the sequence
            target_x, target_y = data[i + sequence_length]
            y_x.append(target_x)
            y_y.append(target_y)

        return np.array(X), np.array(y_x), np.array(y_y)

    def prepare_data(self, user_trajectory):
        """
        Prepare trajectory data for model training.

        Args:
            user_trajectory: DataFrame with columns 'x', 'y'

        Returns:
            Trajectory data as numpy array
        """
        data = user_trajectory[['x', 'y']].values
        return data

    def train(self, user_trajectories, epochs=None, batch_size=None, verbose=0):
        """
        Train the ML model on user mobility data.

        Args:
            user_trajectories: List of DataFrames, each with columns 'x', 'y'
            epochs: Ignored (for compatibility with LSTM interface)
            batch_size: Ignored (for compatibility with LSTM interface)
            verbose: Verbosity level for training

        Returns:
            Dict with training information
        """
        if verbose > 0:
            print("Training ML mobility predictor...")

        # Combine all trajectories
        all_features = []
        all_targets_x = []
        all_targets_y = []

        for traj in user_trajectories:
            data = self.prepare_data(traj)
            X, y_x, y_y = self.create_feature_vectors(data, self.sequence_length)

            all_features.append(X)
            all_targets_x.append(y_x)
            all_targets_y.append(y_y)

        # Combine all data
        X_combined = np.vstack(all_features)
        y_x_combined = np.concatenate(all_targets_x)
        y_y_combined = np.concatenate(all_targets_y)

        if verbose > 0:
            print(f"Training data shape: {X_combined.shape}")
            print(f"Target X shape: {y_x_combined.shape}")
            print(f"Target Y shape: {y_y_combined.shape}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_combined)

        # Train separate models for X and Y coordinates
        self.model_x.fit(X_scaled, y_x_combined)
        self.model_y.fit(X_scaled, y_y_combined)

        self.is_trained = True

        if verbose > 0:
            print("ML model trained successfully!")

        # Return dummy history for compatibility
        return {
            'model_trained': True,
            'training_samples': len(X_combined),
            'features_per_sample': X_combined.shape[1]
        }

    def predict_next_position(self, recent_positions):
        """
        Predict next position given recent trajectory.

        Args:
            recent_positions: Array of shape (k, 2) with recent [x, y] coordinates

        Returns:
            Tuple: (predicted_x, predicted_y) - predicted coordinates
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")

        # Ensure we have the right number of points
        if len(recent_positions) < self.sequence_length:
            # Pad with first position if needed
            padding = self.sequence_length - len(recent_positions)
            padding_data = np.tile(recent_positions[0:1], (padding, 1))
            recent_positions = np.vstack([padding_data, recent_positions])
        else:
            recent_positions = recent_positions[-self.sequence_length:]

        # Create feature vector: flatten the sequence
        features = recent_positions.flatten().reshape(1, -1)  # Shape: (1, sequence_length*2)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Make predictions
        pred_x = self.model_x.predict(features_scaled)[0]
        pred_y = self.model_y.predict(features_scaled)[0]

        return float(pred_x), float(pred_y)

    def predict_multiple_steps(self, recent_positions, num_steps=None):
        """
        Predict multiple future positions iteratively.

        Args:
            recent_positions: Array of shape (k, 2)
            num_steps: Number of steps to predict (default: prediction_steps)

        Returns:
            Array of shape (num_steps, 2) with predicted positions
        """
        if num_steps is None:
            num_steps = self.prediction_steps

        predictions = []
        current_sequence = recent_positions[-self.sequence_length:]

        for _ in range(num_steps):
            next_pos = self.predict_next_position(current_sequence)
            predictions.append([next_pos[0], next_pos[1]])
            # Update sequence with prediction
            current_sequence = np.vstack([current_sequence[1:], [next_pos]])

        return np.array(predictions)

    def evaluate(self, test_trajectories):
        """
        Evaluate model on test trajectories.

        Args:
            test_trajectories: List of test trajectory DataFrames

        Returns:
            Dict with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")

        all_actual_x, all_actual_y = [], []
        all_pred_x, all_pred_y = [], []

        for traj in test_trajectories:
            data = self.prepare_data(traj)
            X, y_x, y_y = self.create_feature_vectors(data, self.sequence_length)

            if len(X) == 0:
                continue

            X_scaled = self.scaler.transform(X)

            pred_x = self.model_x.predict(X_scaled)
            pred_y = self.model_y.predict(X_scaled)

            all_actual_x.extend(y_x)
            all_actual_y.extend(y_y)
            all_pred_x.extend(pred_x)
            all_pred_y.extend(pred_y)

        if not all_actual_x:
            return {'mse': 0, 'mae': 0, 'rmse': 0}

        # Calculate metrics for X and Y separately, then average
        mse_x = mean_squared_error(all_actual_x, all_pred_x)
        mse_y = mean_squared_error(all_actual_y, all_pred_y)
        mse = (mse_x + mse_y) / 2

        mae_x = mean_absolute_error(all_actual_x, all_pred_x)
        mae_y = mean_absolute_error(all_actual_y, all_pred_y)
        mae = (mae_x + mae_y) / 2

        rmse = np.sqrt(mse)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }


if __name__ == "__main__":
    # Example usage
    from data_generation import generate_user_mobility, get_user_trajectory

    # Generate sample data
    print("Generating mobility data...")
    mobility_data = generate_user_mobility(num_users=3, num_timesteps=500, seed=42)

    # Create predictor
    predictor = MobilityPredictor(sequence_length=20)

    # Prepare training data
    trajectories = []
    for user_id in range(3):
        traj = get_user_trajectory(mobility_data, user_id)
        trajectories.append(traj)

    # Train model
    print("Training ML model...")
    predictor.train(trajectories, verbose=1)

    # Test prediction
    print("\nTesting prediction...")
    user_0_traj = trajectories[0]
    recent_pos = user_0_traj[['x', 'y']].values[-20:]
    next_pos = predictor.predict_next_position(recent_pos)
    print(f"Predicted next position: ({next_pos[0]:.2f}, {next_pos[1]:.2f})")

    # Evaluate model
    print("\nEvaluating model...")
    metrics = predictor.evaluate(trajectories)
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"Root Mean Square Error: {metrics['rmse']:.4f}")
