"""
Modified Aftershock Prediction Pipeline with improved training approach (run_enhanced_aftershock_gnn.py)
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


# Parse arguments with added options for improved training
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Improved Aftershock GNN with Better Training"
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
        default=50,
        help="Distance threshold for connections (km)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=5,
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
        "--hidden_channels", type=int, default=128, help="Size of hidden layers"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gcn",
        choices=["gat", "gcn", "baseline"],
        help="Type of model to use (gat, gcn, baseline)",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--scale_targets",
        action="store_true",
        help="Scale target values (divide by 10)",
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
        default="results/improved_aftershock_model.pt",
        help="Path to save/load model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with extra print statements",
    )

    return parser.parse_args()


def prepare_sequences_with_relative_targets(
    aftershocks,
    waveform_features_dict,
    sequence_length=5,
    time_window_hours=72,
    max_sequences=5000,
    scale_targets=False,
):
    """
    Create aftershock sequences with waveform features using relative targets
    with optional scaling
    """
    import numpy as np
    import pandas as pd

    sequences = []
    total_aftershocks = len(aftershocks)

    # Print keys to debug matching
    print(
        f"First few waveform_features_dict keys: {list(waveform_features_dict.keys())[:5]}"
    )
    print(f"First few aftershocks indices: {list(aftershocks.index)[:5]}")

    # Check if aftershocks have 'event_id' column
    if "event_id" in aftershocks.columns:
        print("Using 'event_id' column to match aftershocks with waveform features")
        # Check which aftershocks have waveform features using event_id
        aftershocks_with_features = aftershocks[
            aftershocks["event_id"].isin(waveform_features_dict.keys())
        ]
    else:
        print("Using index to match aftershocks with waveform features")
        # Check which aftershocks have waveform features using index
        aftershocks_with_features = aftershocks[
            aftershocks.index.isin(waveform_features_dict.keys())
        ]

    print(f"Total aftershocks: {total_aftershocks}")
    print(f"Aftershocks with waveform features: {len(aftershocks_with_features)}")

    # Adjust sequence length if needed
    if len(aftershocks_with_features) < sequence_length + 1:
        original_sequence_length = sequence_length
        sequence_length = min(5, len(aftershocks_with_features) - 1)
        if sequence_length <= 0:
            print(f"Not enough aftershocks with waveform features to create sequences")
            return []

        print(
            f"Adjusting sequence length from {original_sequence_length} to {sequence_length} due to limited data"
        )

    # Sort aftershocks by time
    aftershocks_sorted = aftershocks_with_features.sort_values("hours_since_mainshock")

    # Use a sliding window approach
    step_size = 1

    for i in range(0, len(aftershocks_sorted) - sequence_length, step_size):
        # Get sequence of aftershocks
        current_sequence = aftershocks_sorted.iloc[i : i + sequence_length]
        target_aftershock = aftershocks_sorted.iloc[i + sequence_length]

        # Check if the sequence spans less than the time window
        seq_duration = (
            current_sequence.iloc[-1]["hours_since_mainshock"]
            - current_sequence.iloc[0]["hours_since_mainshock"]
        )
        if seq_duration > time_window_hours:
            continue

        # Extract metadata features for each aftershock in the sequence
        metadata_features = current_sequence[
            [
                "source_latitude_deg",
                "source_longitude_deg",
                "source_depth_km",
                "hours_since_mainshock",
            ]
        ].values

        # Extract waveform features for each aftershock in the sequence
        sequence_waveform_features = []
        valid_sequence = True

        for idx, row in current_sequence.iterrows():
            # Try using event_id if available, otherwise use index
            if "event_id" in current_sequence.columns:
                feature_key = row["event_id"]
            else:
                feature_key = idx

            if feature_key in waveform_features_dict:
                features = waveform_features_dict[feature_key]

                # Check if we have valid features
                if features and len(features) > 0:
                    sequence_waveform_features.append(features)
                else:
                    valid_sequence = False
                    break
            else:
                valid_sequence = False
                break

        # Skip if any event in the sequence doesn't have valid waveform features
        if not valid_sequence:
            continue

        # Use relative target (delta from last event in sequence)
        last_event_lat = current_sequence.iloc[-1]["source_latitude_deg"]
        last_event_lon = current_sequence.iloc[-1]["source_longitude_deg"]

        target_lat = target_aftershock["source_latitude_deg"]
        target_lon = target_aftershock["source_longitude_deg"]

        # Convert to kilometers for more numerically stable targets
        # Approximate conversion at these latitudes
        lat_km_per_degree = 111.0  # Approximate km per degree latitude
        lon_km_per_degree = 111.0 * np.cos(
            np.radians(last_event_lat)
        )  # Varies with latitude

        delta_lat_km = (target_lat - last_event_lat) * lat_km_per_degree
        delta_lon_km = (target_lon - last_event_lon) * lon_km_per_degree

        # Scale targets if requested (helps with training stability)
        if scale_targets:
            delta_lat_km = delta_lat_km / scale_factor  # Scale to tens of km
            delta_lon_km = delta_lon_km / scale_factor

        # Store as relative target
        target = np.array([delta_lat_km, delta_lon_km])

        # Also store reference coordinates for conversion back to absolute
        reference = np.array([last_event_lat, last_event_lon])

        # Store scale factor if we scaled targets
        scale_factor = 100.0 if scale_targets else 1.0

        sequences.append(
            (
                metadata_features,
                sequence_waveform_features,
                target,
                reference,
                scale_factor,
            )
        )

        # Limit the number of sequences
        if len(sequences) >= max_sequences:
            print(f"Reached maximum number of sequences ({max_sequences})")
            break

    print(f"Created {len(sequences)} aftershock sequences with relative targets")
    return sequences


def build_graphs_from_sequences(sequences, distance_threshold_km=25, debug=False):
    """
    Build graph representations from aftershock sequences with absolute targets
    """
    from torch_geometric.data import Data
    import torch
    import numpy as np

    graph_dataset = []

    # Extract waveform feature names from the first sequence
    if len(sequences) > 0:
        first_sequence_waveform_features = sequences[0][1]
        feature_names = sorted(list(first_sequence_waveform_features[0].keys()))
        print(f"Using {len(feature_names)} waveform features: {feature_names[:5]}...")
    else:
        feature_names = []
        return graph_dataset, feature_names

    for seq_idx, (
        metadata_features,
        waveform_features_list,
        target,
    ) in enumerate(sequences):
        if debug and seq_idx < 3:
            print(f"\nSequence {seq_idx}:")
            print(f"  Metadata shape: {metadata_features.shape}")
            print(f"  Waveform features: {len(waveform_features_list)} events")
            print(f"  Target: {target}")

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
                if distance < distance_threshold_km and time_diff < 24:
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

    print(f"Built {len(graph_dataset)} graph representations with absolute targets")
    return graph_dataset, feature_names


def convert_relative_to_absolute_predictions(
    pred_deltas, references, scale_factors=None
):
    """
    Convert relative predictions (in km) back to absolute coordinates (lat/lon)
    with support for scaled predictions
    """
    absolute_coords = np.zeros_like(references)

    for i in range(len(pred_deltas)):
        # Get reference point
        ref_lat = references[i, 0]
        ref_lon = references[i, 1]

        # Get predicted deltas (in km)
        delta_lat_km = pred_deltas[i, 0]
        delta_lon_km = pred_deltas[i, 1]

        # Apply scaling if needed
        if scale_factors is not None:
            delta_lat_km = delta_lat_km * scale_factors[i]
            delta_lon_km = delta_lon_km * scale_factors[i]

        # Convert km back to degrees
        lat_km_per_degree = 111.0
        lon_km_per_degree = 111.0 * np.cos(np.radians(ref_lat))

        delta_lat_deg = delta_lat_km / lat_km_per_degree
        delta_lon_deg = delta_lon_km / lon_km_per_degree

        # Calculate absolute coordinates
        absolute_coords[i, 0] = ref_lat + delta_lat_deg
        absolute_coords[i, 1] = ref_lon + delta_lon_deg

    return absolute_coords


def evaluate_model(model, graph_dataset, mainshock):
    """
    Evaluate the model with absolute position predictions
    """
    # Handle the case where we have very few data points
    test_size = 0.2
    if len(graph_dataset) < 5:
        test_size = 0.4  # Use more for testing if we have limited data

    # Split into training and testing sets
    _, test_data = train_test_split(graph_dataset, test_size=test_size, random_state=42)

    # Create data loader - use batch size of 1 for very small datasets
    batch_size = 32
    if len(test_data) < 10:
        batch_size = 1

    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Model in evaluation mode
    model.eval()

    # Lists to store actual and predicted coordinates
    actual_lats = []
    actual_lons = []
    pred_lats = []
    pred_lons = []

    # Evaluate on test set
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Forward pass
            lat_pred, lon_pred = model(
                batch.metadata, batch.waveform, batch.edge_index, batch.batch
            )

            # Extract predictions and targets
            lat_np = lat_pred.cpu().numpy().flatten()
            lon_np = lon_pred.cpu().numpy().flatten()

            # Get targets
            targets = batch.y.cpu().numpy()

            # Store predictions and targets
            pred_lats.extend(lat_np)
            pred_lons.extend(lon_np)
            actual_lats.extend(targets[:, 0])
            actual_lons.extend(targets[:, 1])

    # Calculate error in kilometers
    errors_km = []
    for i in range(len(actual_lats)):
        # Haversine distance
        R = 6371  # Earth radius in kilometers
        lat1, lon1 = np.radians(actual_lats[i]), np.radians(actual_lons[i])
        lat2, lon2 = np.radians(pred_lats[i]), np.radians(pred_lons[i])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c

        errors_km.append(distance)

    mean_error = np.mean(errors_km)
    median_error = np.median(errors_km)

    print(f"Evaluation Results:")
    print(f"Mean Error: {mean_error:.2f} km")
    print(f"Median Error: {median_error:.2f} km")

    # Plot spatial predictions
    plt.figure(figsize=(12, 10))

    # Plot mainshock
    plt.scatter(
        mainshock["source_longitude_deg"],
        mainshock["source_latitude_deg"],
        s=200,
        c="red",
        marker="*",
        label="Mainshock",
        edgecolor="black",
        zorder=10,
    )

    # Plot actual aftershocks
    plt.scatter(
        actual_lons,
        actual_lats,
        s=50,
        c="blue",
        alpha=0.7,
        label="Actual Aftershocks",
        edgecolor="black",
    )

    # Plot predicted aftershocks
    plt.scatter(
        pred_lons,
        pred_lats,
        s=30,
        c="green",
        alpha=0.7,
        marker="x",
        label="Predicted Aftershocks",
    )

    # Connect actual to predicted with lines
    for i in range(len(actual_lats)):
        plt.plot(
            [actual_lons[i], pred_lons[i]],
            [actual_lats[i], pred_lats[i]],
            "k-",
            alpha=0.2,
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial Distribution of Actual vs Predicted Aftershocks")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simplified_spatial_predictions.png", dpi=300)
    plt.close()

    # Calculate spatial distribution statistics
    if len(pred_lats) >= 2:  # Need at least 2 points for std
        pred_std_lat = np.std(pred_lats)
        pred_std_lon = np.std(pred_lons)
        actual_std_lat = np.std(actual_lats)
        actual_std_lon = np.std(actual_lons)

        print(f"\nSpatial Distribution Analysis:")
        print(f"  - Actual latitude std: {actual_std_lat:.4f}°")
        print(f"  - Predicted latitude std: {pred_std_lat:.4f}°")
        print(f"  - Actual longitude std: {actual_std_lon:.4f}°")
        print(f"  - Predicted longitude std: {pred_std_lon:.4f}°")

        # Calculate coverage ratio (how much of the actual area is covered by predictions)
        actual_lat_range = np.max(actual_lats) - np.min(actual_lats)
        actual_lon_range = np.max(actual_lons) - np.min(actual_lons)
        pred_lat_range = np.max(pred_lats) - np.min(pred_lats)
        pred_lon_range = np.max(pred_lons) - np.min(pred_lons)

        lat_coverage = pred_lat_range / actual_lat_range if actual_lat_range > 0 else 0
        lon_coverage = pred_lon_range / actual_lon_range if actual_lon_range > 0 else 0

        print(f"  - Latitude range coverage: {lat_coverage:.2f}")
        print(f"  - Longitude range coverage: {lon_coverage:.2f}")

    return (
        mean_error,
        median_error,
        errors_km,
        (pred_lats, pred_lons, actual_lats, actual_lons),
    )


def prepare_sequences_with_absolute_targets(
    aftershocks,
    waveform_features_dict,
    sequence_length=5,
    time_window_hours=72,
    max_sequences=5000,
):
    """
    Create aftershock sequences with waveform features using absolute targets
    """
    sequences = []
    total_aftershocks = len(aftershocks)

    # Print keys to debug matching
    print(
        f"First few waveform_features_dict keys: {list(waveform_features_dict.keys())[:5]}"
    )
    print(f"First few aftershocks indices: {list(aftershocks.index)[:5]}")

    # Check if aftershocks have 'event_id' column
    if "event_id" in aftershocks.columns:
        print("Using 'event_id' column to match aftershocks with waveform features")
        # Check which aftershocks have waveform features using event_id
        aftershocks_with_features = aftershocks[
            aftershocks["event_id"].isin(waveform_features_dict.keys())
        ]
    else:
        print("Using index to match aftershocks with waveform features")
        # Check which aftershocks have waveform features using index
        aftershocks_with_features = aftershocks[
            aftershocks.index.isin(waveform_features_dict.keys())
        ]

    print(f"Total aftershocks: {total_aftershocks}")
    print(f"Aftershocks with waveform features: {len(aftershocks_with_features)}")

    # Adjust sequence length if needed
    if len(aftershocks_with_features) < sequence_length + 1:
        original_sequence_length = sequence_length
        sequence_length = min(5, len(aftershocks_with_features) - 1)
        if sequence_length <= 0:
            print(f"Not enough aftershocks with waveform features to create sequences")
            return []

        print(
            f"Adjusting sequence length from {original_sequence_length} to {sequence_length} due to limited data"
        )

    # Sort aftershocks by time
    aftershocks_sorted = aftershocks_with_features.sort_values("hours_since_mainshock")

    # Use a sliding window approach
    step_size = 1

    for i in range(0, len(aftershocks_sorted) - sequence_length, step_size):
        # Get sequence of aftershocks
        current_sequence = aftershocks_sorted.iloc[i : i + sequence_length]
        target_aftershock = aftershocks_sorted.iloc[i + sequence_length]

        # Check if the sequence spans less than the time window
        seq_duration = (
            current_sequence.iloc[-1]["hours_since_mainshock"]
            - current_sequence.iloc[0]["hours_since_mainshock"]
        )
        if seq_duration > time_window_hours:
            continue

        # Extract metadata features for each aftershock in the sequence
        metadata_features = current_sequence[
            [
                "source_latitude_deg",
                "source_longitude_deg",
                "source_depth_km",
                "hours_since_mainshock",
            ]
        ].values

        # Extract waveform features for each aftershock in the sequence
        sequence_waveform_features = []
        valid_sequence = True

        for idx, row in current_sequence.iterrows():
            # Try using event_id if available, otherwise use index
            if "event_id" in current_sequence.columns:
                feature_key = row["event_id"]
            else:
                feature_key = idx

            if feature_key in waveform_features_dict:
                features = waveform_features_dict[feature_key]

                # Check if we have valid features
                if features and len(features) > 0:
                    sequence_waveform_features.append(features)
                else:
                    valid_sequence = False
                    break
            else:
                valid_sequence = False
                break

        # Skip if any event in the sequence doesn't have valid waveform features
        if not valid_sequence:
            continue

        # Use absolute target coordinates directly
        target_lat = target_aftershock["source_latitude_deg"]
        target_lon = target_aftershock["source_longitude_deg"]

        # Store as absolute target
        target = np.array([target_lat, target_lon])

        sequences.append((metadata_features, sequence_waveform_features, target))

        # Limit the number of sequences
        if len(sequences) >= max_sequences:
            print(f"Reached maximum number of sequences ({max_sequences})")
            break

    print(f"Created {len(sequences)} aftershock sequences with absolute targets")
    return sequences


def main():
    """Main execution function for the absolute aftershock prediction"""
    # Parse arguments
    args = parse_arguments()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Record start time
    start_time = datetime.now()
    print(f"Starting Absolute Aftershock GNN at {start_time}")

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
    from train_and_evaluate import (
        train_model_with_diagnostics
    )

    # Import the appropriate model based on user choice
    if args.model_type == "gcn":
        from enhanced_model import SimplerAfterShockGNN as ModelClass

        print("Using GCN-based model architecture")
    elif args.model_type == "baseline":
        from enhanced_model import BaselineRegressor as ModelClass

        print("Using baseline regressor model (non-graph)")
    else:
        from enhanced_model import SimplifiedAfterShockGNN as ModelClass

        print("Using GAT-based model architecture")

    # Load and process data
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

    # Step 2: Create aftershock sequences with waveform features and absolute targets
    print("\n=== Creating Aftershock Sequences with Absolute Targets ===")
    sequences = prepare_sequences_with_absolute_targets(
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
    graph_dataset, feature_names = build_graphs_from_sequences(
        sequences, distance_threshold_km=args.distance_threshold, debug=args.debug
    )

    if len(graph_dataset) == 0:
        print("No valid graph representations created. Exiting.")
        return

    # Step 4: Create the appropriate model
    print(f"\n=== Creating {args.model_type.upper()} Model ===")
    metadata_channels = 4  # latitude, longitude, depth, hours since mainshock
    waveform_channels = len(feature_names)  # Number of waveform features

    # Create the model
    model = ModelClass(
        metadata_channels=metadata_channels,
        waveform_channels=waveform_channels,
        hidden_channels=args.hidden_channels,
        num_layers=3,
        dropout=args.dropout,
    ).to(device)

    print(model)

    # Step 5: Train the model (or load from file)
    if not args.skip_training:
        print("\n=== Training Model with Enhanced Diagnostics ===")
        model, train_losses, val_losses = train_model_with_diagnostics(
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
        model, graph_dataset, mainshock
    )
    pred_lats, pred_lons, actual_lats, actual_lons = coords

    # Record end time and print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n=== Absolute Aftershock GNN Execution Summary ===")
    print(f"Started at: {start_time}")
    print(f"Completed at: {end_time}")
    print(f"Total duration: {duration}")
    print(f"\nDataset: Iquique Earthquake (2014)")
    print(f"Total sequences: {len(graph_dataset)}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Time window: {args.time_window} hours")
    print(f"Distance threshold: {args.distance_threshold} km")

    print(f"\nModel parameters:")
    print(f"  - Model type: {args.model_type}")
    print(f"  - Hidden channels: {args.hidden_channels}")
    print(f"  - Dropout rate: {args.dropout}")

    print(f"\nPerformance:")
    print(f"  - Mean error: {mean_error:.2f} km")
    print(f"  - Median error: {median_error:.2f} km")


if __name__ == "__main__":
    main()
