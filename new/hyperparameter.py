# hyperparameter_tuning.py
import optuna
import pandas as pd
import numpy as np
import os
import time
import pickle
import logging
from datetime import datetime

from relative_gnn import (
    read_data_from_pickle,
    create_causal_spatiotemporal_graph,
    RelativeGNNAftershockPredictor
)

def setup_logging():
    """Set up logging to both console and file"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set up timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/hyperparameter_tuning_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logging.info(f"Logging to {log_file}")
    return log_file

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Graph construction hyperparameters
    time_window = trial.suggest_categorical("time_window", [72, 120, 168, 240, 336])
    spatial_threshold = trial.suggest_categorical("spatial_threshold", [50, 75, 100, 150, 200])
    min_connections = trial.suggest_categorical("min_connections", [1, 2, 3, 5, 7])
    
    # Model hyperparameters
    gnn_type = trial.suggest_categorical("gnn_type", ["gat", "sage"])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    
    # Training hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Log current trial parameters
    logging.info(f"\nTrial {trial.number}:")
    logging.info(f"Graph params: time_window={time_window}, spatial_threshold={spatial_threshold}, min_connections={min_connections}")
    logging.info(f"Model params: gnn_type={gnn_type}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    logging.info(f"Training params: lr={learning_rate}, batch_size={batch_size}, weight_decay={weight_decay}")
    
    # Load and prepare data
    df = read_data_from_pickle("aftershock_data.pkl")
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp").drop("timestamp", axis=1)
    df_sorted = df_sorted

    df_sorted = df_sorted[2:].reset_index(drop=True)
    
    try:
        # Create graphs with current hyperparameters
        logging.info(f"Creating graphs with time_window={time_window}h, spatial_threshold={spatial_threshold}km, min_connections={min_connections}...")
        graph_data_list, reference_coords = create_causal_spatiotemporal_graph(
            df_sorted,
            time_window=time_window,
            spatial_threshold=spatial_threshold,
            min_connections=min_connections
        )
        
        # Skip if not enough graphs were created
        if len(graph_data_list) < 50:
            logging.info(f"Not enough graphs created: {len(graph_data_list)} < 50, skipping trial")
            return float('inf')
            
        logging.info(f"Created {len(graph_data_list)} graphs, proceeding with model training")
        
        # Initialize and train model
        predictor = RelativeGNNAftershockPredictor(
            graph_data_list=graph_data_list,
            reference_coords=reference_coords,
            gnn_type=gnn_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
        )
        
        # Set up a log capture for training
        class LogCapture:
            def __init__(self):
                self.logs = []
            
            def write(self, message):
                if message.strip():  # Only capture non-empty lines
                    self.logs.append(message.strip())
                    logging.info(message.strip())
            
            def flush(self):
                pass
        
        # Capture print output during training
        import sys
        original_stdout = sys.stdout
        log_capture = LogCapture()
        sys.stdout = log_capture
        
        # Train with fewer epochs and early stopping for efficiency
        logging.info(f"Training {gnn_type.upper()} model...")
        try:
            predictor.train(num_epochs=50, patience=10)
        finally:
            # Restore stdout
            sys.stdout = original_stdout
        
        # Evaluate on validation set
        val_metrics, _, _, _ = predictor.validate_with_metrics()
        
        # Get horizontal and 3D errors
        h_error = val_metrics["mean_horizontal_error"]
        d_error = val_metrics["mean_depth_error"]
        error_3d = val_metrics["mean_3d_error"]
        
        logging.info(f"Validation metrics - Horizontal: {h_error:.2f}km, Depth: {d_error:.2f}km, 3D: {error_3d:.2f}km")
        
        # Save trial results
        trial_results = {
            "params": trial.params,
            "metrics": val_metrics,
            "num_graphs": len(graph_data_list),
            "training_logs": log_capture.logs
        }
        
        # Create trials directory if it doesn't exist
        os.makedirs("hp_trials", exist_ok=True)
        
        # Save trial results
        with open(f"hp_trials/trial_{trial.number}.pkl", "wb") as f:
            pickle.dump(trial_results, f)
        
        # Return horizontal error (lower is better)
        return h_error
        
    except Exception as e:
        logging.error(f"Error in trial {trial.number}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return float('inf')

def main():
    """Run hyperparameter optimization"""
    
    # Set up logging
    log_file = setup_logging()
    
    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("hp_trials", exist_ok=True)
    
    logging.info("Starting hyperparameter optimization...")
    start_time = time.time()
    
    # Create and run study
    study = optuna.create_study(direction="minimize", 
                               study_name="earthquake_gnn_tuning",
                               storage="sqlite:///hp_optimization.db",
                               load_if_exists=True)
    
    try:
        study.optimize(objective, n_trials=30, timeout=86400)  # 24 hour timeout
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user.")
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nOptimization completed in {elapsed_time/3600:.2f} hours")
    
    # Print best parameters and metrics
    logging.info("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        logging.info(f"  {key}: {value}")
    
    logging.info(f"\nBest validation error: {study.best_value:.2f} km")
    
    # Save results
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials": study.trials,
        "datetime": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "log_file": log_file
    }
    
    with open("hp_optimization_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Create visualization
    try:
        logging.info("Creating visualizations...")
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("results/param_importance.png")
        
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("results/optimization_history.png")
        logging.info("Visualizations saved.")
    except Exception as e:
        logging.error(f"Error creating visualizations: {str(e)}")
    
    logging.info("Optimization results saved.")
    
    # Train final model with best parameters
    logging.info("\nTraining final model with best parameters...")
    
    # Implement final model training here with best params
    try:
        df = read_data_from_pickle("aftershock_data.pkl")
        df_sorted = df.copy()
        df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
        df_sorted = df_sorted.sort_values("timestamp").drop("timestamp", axis=1)

        df_sorted = df_sorted[2:].reset_index(drop=True)
        
        logging.info("Creating graphs with best parameters...")
        graph_data_list, reference_coords = create_causal_spatiotemporal_graph(
            df_sorted,
            time_window=study.best_params["time_window"],
            spatial_threshold=study.best_params["spatial_threshold"],
            min_connections=study.best_params["min_connections"]
        )
        
        logging.info(f"Created {len(graph_data_list)} graphs for final model")
        
        # Train final model
        final_model = RelativeGNNAftershockPredictor(
            graph_data_list=graph_data_list,
            reference_coords=reference_coords,
            **study.best_params
        )
        
        logging.info("Training final model...")
        final_model.train(num_epochs=100, patience=15)
        
        # Evaluate on test set
        test_metrics, y_true, y_pred, errors = final_model.test()
        
        # Log test metrics
        logging.info("\nFinal model test metrics:")
        for key, value in test_metrics.items():
            logging.info(f"  {key}: {value:.4f}")
        
        # Save best model results
        with open("best_model_results.pkl", "wb") as f:
            pickle.dump({
                "params": study.best_params,
                "test_metrics": test_metrics,
                "y_true": y_true,
                "y_pred": y_pred,
                "errors": errors
            }, f)
        
        logging.info("Final model evaluation complete and results saved.")
        
    except Exception as e:
        logging.error(f"Error training final model: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    
    logging.info("Hyperparameter optimization process complete.")
    
if __name__ == "__main__":
    main()