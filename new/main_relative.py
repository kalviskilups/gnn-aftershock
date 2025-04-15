# main_relative.py Telplaika grafu modelēšana ar relatīvām koordinātēm precīzai pēcgrūdienu lokalizācijas prognozēšanai

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from datetime import datetime

# Import from our relative GNN implementation
from relative_gnn import (
    read_data_from_pickle,
    create_causal_spatiotemporal_graph,  # Using the causal version now
    RelativeGNNAftershockPredictor,
    plot_relative_results
)

def compare_methods(rel_file="rel_results.pkl", abs_file="abs_results.pkl"):
    """Compare the relative and absolute location methods with visualization"""
    
    # Load result files
    try:
        with open(rel_file, "rb") as f:
            rel_results = pickle.load(f)
            
        with open(abs_file, "rb") as f:
            abs_results = pickle.load(f)
    except FileNotFoundError:
        print("Result files not found. Please run both methods first.")
        return
    
    # Extract metrics for comparison
    rel_metrics = {
        model: rel_results[model]["metrics"] 
        for model in rel_results.keys()
    }
    
    abs_metrics = {
        model: abs_results[model]["metrics"] 
        for model in abs_results.keys()
    }
    
    # Compare horizontal errors
    plt.figure(figsize=(12, 8))
    
    models = []
    horiz_errors = []
    depth_errors = []
    method_colors = []
    
    # Get metrics from absolute methods
    for model in abs_metrics:
        models.append(f"Abs-{model.upper()}")
        horiz_errors.append(abs_metrics[model]['mean_horizontal_error'])
        depth_errors.append(abs_metrics[model]['mean_depth_error'])
        method_colors.append('skyblue')
    
    # Get metrics from relative methods
    for model in rel_metrics:
        models.append(f"Rel-{model.upper()}")
        # Use absolute coordinates for fair comparison
        horiz_errors.append(rel_metrics[model]['abs_mean_horizontal_error'])
        depth_errors.append(rel_metrics[model]['abs_mean_depth_error'])
        method_colors.append('salmon')
    
    # Plot horizontal errors
    plt.subplot(1, 2, 1)
    plt.bar(models, horiz_errors, color=method_colors)
    plt.ylabel('Mean Horizontal Error (km)')
    plt.title('Horizontal Error Comparison')
    plt.xticks(rotation=45)
    
    # Plot depth errors
    plt.subplot(1, 2, 2)
    plt.bar(models, depth_errors, color=method_colors)
    plt.ylabel('Mean Depth Error (km)')
    plt.title('Depth Error Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("results/method_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create success rate plots
    thresholds = [5, 10, 15, 20, 50]
    
    plt.figure(figsize=(15, 10))
    
    # Horizontal success rates
    plt.subplot(2, 1, 1)
    for i, model in enumerate(abs_metrics):
        rates = [abs_metrics[model][f'horizontal_{t}km'] for t in thresholds]
        plt.plot(thresholds, rates, 'o-', label=f"Abs-{model.upper()}", color=f"C{i}")
    
    for i, model in enumerate(rel_metrics):
        # For relative models, we need to create equivalent metrics if they don't exist
        if f'horizontal_{thresholds[0]}km' in rel_metrics[model]:
            rates = [rel_metrics[model][f'horizontal_{t}km'] for t in thresholds]
        else:
            # Try to compute from the errors tuple
            abs_err_tuple = rel_results[model]["errors"][1]  # Second tuple has absolute errors
            abs_horiz_errors = abs_err_tuple[0]
            rates = [100 * np.mean(abs_horiz_errors < t) for t in thresholds]
            
        plt.plot(thresholds, rates, 's--', label=f"Rel-{model.upper()}", color=f"C{i+len(abs_metrics)}")
    
    plt.xlabel('Distance Threshold (km)')
    plt.ylabel('Success Rate (%)')
    plt.title('Horizontal Location Success Rates')
    plt.grid(True)
    plt.legend()
    
    # Depth success rates
    plt.subplot(2, 1, 2)
    for i, model in enumerate(abs_metrics):
        rates = [abs_metrics[model][f'depth_{t}km'] for t in thresholds]
        plt.plot(thresholds, rates, 'o-', label=f"Abs-{model.upper()}", color=f"C{i}")
    
    for i, model in enumerate(rel_metrics):
        if f'depth_{thresholds[0]}km' in rel_metrics[model]:
            rates = [rel_metrics[model][f'depth_{t}km'] for t in thresholds]
        else:
            # Try to compute from the errors tuple
            abs_err_tuple = rel_results[model]["errors"][1]  # Second tuple has absolute errors
            abs_depth_errors = abs_err_tuple[1]
            rates = [100 * np.mean(abs_depth_errors < t) for t in thresholds]
            
        plt.plot(thresholds, rates, 's--', label=f"Rel-{model.upper()}", color=f"C{i+len(abs_metrics)}")
    
    plt.xlabel('Distance Threshold (km)')
    plt.ylabel('Success Rate (%)')
    plt.title('Depth Location Success Rates')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/success_rate_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Comparison visualizations saved to results directory")

def main_func():
    """Demo of the causal relative location GNN implementation"""
    
    # Create the output directory
    os.makedirs("results", exist_ok=True)
    
    print("Starting causal relative location GNN demonstration")
    
    # Load the data
    if not os.path.exists("aftershock_data.pkl"):
        print("Error: aftershock_data.pkl not found")
        return
    
    # Load the data
    df = read_data_from_pickle("aftershock_data.pkl")
    
    # Sort data chronologically but DO NOT shuffle
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp").drop("timestamp", axis=1)
    
    # Just use the sorted dataframe as is - no shuffling!
    df = df_sorted[2:].reset_index(drop=True)

    print(df)

    print(f"Loaded data with {len(df)} events")
    
    # Create relative coordinate graphs using the causal approach
    graph_data_list, reference_coords = create_causal_spatiotemporal_graph(
        df,
        time_window=168,  # One week in hours
        spatial_threshold=100,  # 100 km
        min_connections=3
    )
    
    print(f"Created {len(graph_data_list)} causal graphs in relative coordinate system")
    print(f"Reference coordinates: {reference_coords}")
    
    # Train a model with our fixed causal approach
    model_type = "sage"  # Graph Attention Network
    
    predictor = RelativeGNNAftershockPredictor(
        graph_data_list=graph_data_list,
        reference_coords=reference_coords,
        gnn_type=model_type,
        hidden_dim=128,
        num_layers=4,
        learning_rate=0.001,
        batch_size=16
    )
    
    # Debug the data structure to verify no leakage
    predictor.debug_data_structure()
    
    # Train the model
    predictor.train(num_epochs=75, patience=15)
    
    # Test the model
    print(f"Testing causal {model_type.upper()} model...")
    metrics, y_true, y_pred, errors = predictor.test()
    
    # Print metrics
    print(f"\nCausal {model_type.upper()} Relative Prediction Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot results
    plot_relative_results(
        y_true, y_pred, errors,
        reference_coords=reference_coords,
        model_name=f"Causal_{model_type}"
    )
    
    # Save results for potential comparison
    results = {
        model_type: {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred, 
            "errors": errors
        }
    }
    
    with open("rel_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print("Demonstration complete. Results saved to results directory")
    
    # Try to compare with absolute method if results available
    if os.path.exists("abs_results.pkl"):
        print("\nComparing with absolute coordinate results...")
        compare_methods()

if __name__ == "__main__":
    main_func()