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
    plot_relative_results,
    plot_3d_aftershocks
)


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

    print(f"Loaded data with {len(df)} events")

    # Create relative coordinate graphs using the causal approach
    graph_data_list, reference_coords = create_causal_spatiotemporal_graph(
        df,
        time_window=120,  # One week in hours
        spatial_threshold=75,  # 100 km
        min_connections=5,
    )

    print(f"Created {len(graph_data_list)} causal graphs in relative coordinate system")
    print(f"Reference coordinates: {reference_coords}")

    # Train a model with our fixed causal approach
    model_type = "gat"  # Graph Attention Network

    predictor = RelativeGNNAftershockPredictor(
        graph_data_list=graph_data_list,
        reference_coords=reference_coords,
        gnn_type=model_type,
        hidden_dim=128,
        num_layers=3,
        learning_rate=0.0026154033981211277,
        batch_size=8,
        weight_decay=6.4617557814088705e-06,
    )

    # Debug the data structure to verify no leakage
    predictor.debug_data_structure()

    # Train the model
    predictor.train(num_epochs=100, patience=20)

    # Test the model
    print(f"Testing causal {model_type.upper()} model...")
    metrics, y_true, y_pred, errors = predictor.test()

    # Print metrics
    print(f"\nCausal {model_type.upper()} Relative Prediction Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Plot results
    plot_relative_results(
        y_true,
        y_pred,
        errors,
        reference_coords=reference_coords,
        model_name=f"{model_type}",
    )

    plot_3d_aftershocks(
        y_true, y_pred, 
        reference_coords=reference_coords,
        model_name=f"{model_type}"
    )


    # # Save results for potential comparison
    # results = {
    #     model_type: {
    #         "metrics": metrics,
    #         "y_true": y_true,
    #         "y_pred": y_pred,
    #         "errors": errors,
    #     }
    # }

    print("Demonstration complete. Results saved to results directory")


if __name__ == "__main__":
    main_func()
