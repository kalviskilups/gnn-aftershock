"""
Simplified Aftershock Prediction Pipeline with relative position targets
Fixed version to properly handle data matching issues
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


# Parse arguments with reduced default sequence length and added debugging options
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Simplified Aftershock GNN with Relative Position Prediction"
    )

    # Data parameters
    parser.add_argument(
        "--time_window",
        type=int,
        default=48,
        help="Time window for connecting events (hours)",
    )
    parser.add_argument(
        "--distance_threshold",
        type=int,
        default=40,
        help="Distance threshold for connections (km)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=5,  # Reduced default sequence length
        help="Number of events in each sequence",
    )
    parser.add_argument(
        "--max_waveforms",
        type=int,
        default=13400,
        help="Maximum number of waveforms to process",
    )

    # Model parameters
    parser.add_argument(
        "--hidden_channels", type=int, default=64, help="Size of hidden layers"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of GNN layers"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=300, help="Maximum training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--patience", type=int, default=100, help="Early stopping patience"
    )

    # Execution parameters
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip model training (load from file)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/simplified_aftershock_model.pt",
        help="Path to save/load model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with extra print statements",
    )

    return parser.parse_args()


def build_graphs_from_sequences(sequences, distance_threshold_km=25, debug=False):
    """
    Build graph representations from aftershock sequences with relative targets
    """
    from torch_geometric.data import Data
    import torch
    import numpy as np

    graph_dataset = []
    reference_coords = []

    # Extract waveform feature names from the first sequence
    if len(sequences) > 0:
        first_sequence_waveform_features = sequences[0][1]
        feature_names = sorted(list(first_sequence_waveform_features[0].keys()))
        print(f"Using {len(feature_names)} waveform features: {feature_names[:5]}...")
    else:
        feature_names = []
        return graph_dataset, feature_names, reference_coords

    for seq_idx, (
        metadata_features,
        waveform_features_list,
        target,
        reference,
    ) in enumerate(sequences):
        if debug and seq_idx < 3:
            print(f"\nSequence {seq_idx}:")
            print(f"  Metadata shape: {metadata_features.shape}")
            print(f"  Waveform features: {len(waveform_features_list)} events")
            print(f"  Target: {target}")
            print(f"  Reference: {reference}")

        num_nodes = len(metadata_features)

        # Convert metadata features to torch tensors
        metadata_tensor = torch.tensor(metadata_features, dtype=torch.float)

        # Convert waveform features to torch tensors
        waveform_feature_matrix = []
        for waveform_features in waveform_features_list:
            # Extract features in consistent order
            feature_values = [
                waveform_features.get(name, 0.0) for name in feature_names
            ]
            waveform_feature_matrix.append(feature_values)

        waveform_tensor = torch.tensor(waveform_feature_matrix, dtype=torch.float)

        # Convert target to torch tensor
        target_tensor = torch.tensor(target, dtype=torch.float).view(1, 2)

        # Create edges based on spatiotemporal proximity
        edge_list = []

        # Calculate distances between all pairs of events
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue  # Skip self-loops

                # Calculate distance between events
                lat1, lon1 = metadata_features[i][0], metadata_features[i][1]
                lat2, lon2 = metadata_features[j][0], metadata_features[j][1]

                # Approximate distance using Haversine
                R = 6371  # Earth radius in kilometers
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                )
                c = 2 * np.arcsin(np.sqrt(a))
                distance = R * c

                # Calculate time difference
                time1 = metadata_features[i][3]  # hours since mainshock
                time2 = metadata_features[j][3]
                time_diff = abs(time2 - time1)

                # Add edge based on distance and time relationship
                if (
                    distance < distance_threshold_km and time_diff < 24
                ):  # Within distance threshold and 24 hours
                    edge_list.append([i, j])

        # If no edges created, add connections to temporal neighbors
        if len(edge_list) == 0:
            for i in range(num_nodes - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])

        # Convert edge list to torch tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Create PyTorch Geometric Data object
        graph = Data(
            metadata=metadata_tensor,
            waveform=waveform_tensor,
            edge_index=edge_index,
            y=target_tensor,
            num_nodes=metadata_tensor.size(0),
        )

        if debug and seq_idx < 3:
            print(
                f"  Created graph with {graph.num_nodes} nodes and {graph.edge_index.size(1)} edges"
            )

        graph_dataset.append(graph)
        reference_coords.append(reference)

    reference_coords = np.array(reference_coords)
    print(f"Built {len(graph_dataset)} graph representations with relative targets")
    return graph_dataset, feature_names, reference_coords


def main():
    """Main execution function for the simplified aftershock prediction"""
    # Parse arguments
    args = parse_arguments()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Record start time
    start_time = datetime.now()
    print(f"Starting Simplified Aftershock GNN at {start_time}")

    # Step 1: Load and preprocess data with waveforms
    print("\n=== Loading and Preprocessing Data with Waveforms ===")
    # Import these here to avoid circular import issues
    from waveform_gnn import (
        WaveformFeatureExtractor,
        load_aftershock_data_with_waveforms,
        identify_mainshock_and_aftershocks,
        consolidate_station_recordings,
        normalize_waveform_features,
    )

    # Import the simplified model
    from enhanced_model import SimplifiedAfterShockGNN

    # Import the fixed sequence preparation function
    from relative_prediction import prepare_sequences_with_relative_targets

    metadata, iquique, waveform_features_dict = load_aftershock_data_with_waveforms(
        max_waveforms=args.max_waveforms
    )

    # Consolidate recordings from the same event
    print(f"Original dataset: {len(metadata)} recordings")
    metadata, waveform_features_dict = consolidate_station_recordings(
        metadata, waveform_features_dict
    )
    print(f"Consolidated dataset: {len(metadata)} unique events")

    # Identify mainshock and aftershocks
    mainshock, aftershocks = identify_mainshock_and_aftershocks(metadata)

    # Print the first few keys of waveform_features_dict and aftershocks index for debugging
    if args.debug:
        print("\nDebugging information:")
        print(
            f"First few waveform_features_dict keys: {list(waveform_features_dict.keys())[:5]}"
        )
        print(f"First few aftershocks indices: {list(aftershocks.index)[:5]}")
        if "event_id" in aftershocks.columns:
            print(
                f"First few aftershocks event_ids: {aftershocks['event_id'].tolist()[:5]}"
            )

    # Step 2: Create aftershock sequences with waveform features and relative targets
    print("\n=== Creating Aftershock Sequences with Relative Targets ===")
    sequences = prepare_sequences_with_relative_targets(
        aftershocks,
        waveform_features_dict,
        sequence_length=args.sequence_length,
        time_window_hours=args.time_window,
    )

    if not sequences:
        print("No valid sequences created. Exiting.")
        return

    # Step 3: Build graph representations
    print("\n=== Building Graph Representations ===")
    graph_dataset, feature_names, reference_coords = build_graphs_from_sequences(
        sequences, distance_threshold_km=args.distance_threshold, debug=args.debug
    )

    if len(graph_dataset) == 0:
        print("No valid graph representations created. Exiting.")
        return

    # Step 4: Create the simplified GNN model
    print("\n=== Creating Simplified GNN Model ===")
    metadata_channels = 4  # latitude, longitude, depth, hours since mainshock
    waveform_channels = len(feature_names)  # Number of waveform features

    # Create the simplified model
    model = SimplifiedAfterShockGNN(
        metadata_channels=metadata_channels,
        waveform_channels=waveform_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(model)

    # Import the train and evaluate functions
    from train_and_evaluate import train_model, evaluate_model

    # Step 5: Train the model (or load from file)
    if not args.skip_training:
        print("\n=== Training Simplified GNN Model ===")
        model, train_losses, val_losses = train_model(
            graph_dataset,
            model,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
        )
    else:
        print(f"\n=== Loading Model from {args.model_path} ===")
        try:
            model.load_state_dict(torch.load(args.model_path))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    # Step 6: Evaluate the model
    print("\n=== Evaluating Model ===")
    mean_error, median_error, errors_km, coords = evaluate_model(
        model, graph_dataset, mainshock, reference_coords
    )
    pred_lats, pred_lons, actual_lats, actual_lons = coords

    # Record end time and print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n=== Simplified Aftershock GNN Execution Summary ===")
    print(f"Started at: {start_time}")
    print(f"Completed at: {end_time}")
    print(f"Total duration: {duration}")
    print(f"\nDataset: Iquique Earthquake (2014)")
    print(f"Total sequences: {len(graph_dataset)}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Time window: {args.time_window} hours")
    print(f"Distance threshold: {args.distance_threshold} km")

    print(f"\nModel parameters:")
    print(f"  - Hidden channels: {args.hidden_channels}")
    print(f"  - Number of layers: {args.num_layers}")
    print(f"  - Dropout rate: {args.dropout}")

    print(f"\nPerformance:")
    print(f"  - Mean error: {mean_error:.2f} km")
    print(f"  - Median error: {median_error:.2f} km")


if __name__ == "__main__":
    main()
