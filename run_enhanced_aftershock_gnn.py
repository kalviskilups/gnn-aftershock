"""
Modified Aftershock Prediction Pipeline with improved training approach (run_enhanced_aftershock_gnn.py)
"""

import os
import argparse
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import RobustScaler
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from new.debug import *


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
        default=3,
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
    parser.add_argument(
        "--epochs", type=int, default=300, help="Maximum training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
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
    """Build graph representations with improved edge creation"""
    from torch_geometric.data import Data

    graph_dataset = []

    # Extract feature names from first sequence
    if len(sequences) > 0:
        first_sequence_waveform_features = sequences[0][1]
        feature_names = sorted(list(first_sequence_waveform_features[0].keys()))
    else:
        return [], []

    for seq_idx, (metadata_features, waveform_features_list, target) in enumerate(
        sequences
    ):
        num_nodes = len(metadata_features)

        # Convert features to tensors
        metadata_tensor = torch.tensor(metadata_features, dtype=torch.float)

        # Process waveform features
        waveform_feature_matrix = []
        for waveform_features in waveform_features_list:
            feature_values = [
                waveform_features.get(name, 0.0) for name in feature_names
            ]
            waveform_feature_matrix.append(feature_values)
        waveform_tensor = torch.tensor(waveform_feature_matrix, dtype=torch.float)

        # Convert target
        target_tensor = torch.tensor(target, dtype=torch.float).view(1, 2)

        # Create edges based on temporal and spatial proximity
        edge_list = []

        # Always add temporal edges
        for i in range(num_nodes - 1):
            edge_list.append([i, i + 1])  # Forward
            edge_list.append([i + 1, i])  # Backward

        # Add spatial edges if within threshold
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Calculate distance
                lat1, lon1 = metadata_features[i][0], metadata_features[i][1]
                lat2, lon2 = metadata_features[j][0], metadata_features[j][1]

                # Haversine distance
                R = 6371  # Earth radius in km
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                )
                c = 2 * np.arcsin(np.sqrt(a))
                distance = R * c

                # Add bidirectional edges if within threshold
                if distance < distance_threshold_km:
                    edge_list.append([i, j])
                    edge_list.append([j, i])

        # Convert edge list to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Create graph
        graph = Data(
            metadata=metadata_tensor,
            waveform=waveform_tensor,
            edge_index=edge_index,
            y=target_tensor,
            num_nodes=num_nodes,
        )

        graph_dataset.append(graph)

        if debug and seq_idx < 3:
            print(f"\nGraph {seq_idx + 1}:")
            print(f"Nodes: {num_nodes}")
            print(f"Edges: {len(edge_list)}")
            print(f"Metadata shape: {metadata_tensor.shape}")
            print(f"Waveform shape: {waveform_tensor.shape}")
            print(f"Target: {target}")

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
    from train_and_evaluate import train_model_with_diagnostics

    # Import the appropriate model based on user choice
    if args.model_type == "gcn":
        from enhanced_model import BalancedAfterShockGNN as ModelClass

        print("Using GCN-based model architecture")
    elif args.model_type == "baseline":
        from enhanced_model import BaselineRegressor as ModelClass

        print("Using baseline regressor model (non-graph)")
    else:
        from enhanced_model import SimplifiedAfterShockGNN as ModelClass

        print("Using GAT-based model architecture")

    """Main execution function with enhanced debugging"""
    # Parse arguments
    args = parse_arguments()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Record start time
    start_time = datetime.now()
    print(f"Starting Enhanced Aftershock GNN at {start_time}")

    # Step 1: Load and preprocess data with waveforms
    print("\n=== Loading and Preprocessing Data with Waveforms ===")
    metadata, iquique, waveform_features_dict = load_aftershock_data_with_waveforms(
        max_waveforms=args.max_waveforms
    )

    # Debug metadata and waveform features
    debug_waveform_features(waveform_features_dict)

    # Consolidate recordings
    print("\n=== Consolidating Station Recordings ===")
    metadata, waveform_features_dict = consolidate_station_recordings(
        metadata, waveform_features_dict
    )

    # Add normalization here
    print("\n=== Normalizing Waveform Features ===")

    def normalize_features(features_dict):
        """
        Normalize waveform features handling missing features and outliers
        """
        print("Starting feature normalization...")

        # First collect all possible feature names
        all_feature_names = set()
        for features in features_dict.values():
            all_feature_names.update(features.keys())
        all_feature_names = sorted(list(all_feature_names))

        print(f"Found {len(all_feature_names)} unique features")

        # Create a matrix of all features, filling missing values with 0
        feature_matrix = []
        event_ids = []
        for event_id, features in features_dict.items():
            feature_vector = [features.get(fname, 0.0) for fname in all_feature_names]
            feature_matrix.append(feature_vector)
            event_ids.append(event_id)

        feature_matrix = np.array(feature_matrix)

        # Replace infinities with max/min finite values
        feature_matrix[np.isinf(feature_matrix)] = np.nan
        finite_max = np.nanmax(np.abs(feature_matrix[np.isfinite(feature_matrix)]))
        feature_matrix[np.isnan(feature_matrix)] = finite_max

        # Normalize each feature independently
        scaler = (
            RobustScaler()
        )  # Use RobustScaler instead of StandardScaler to handle outliers better
        try:
            normalized = scaler.fit_transform(feature_matrix)

            # Additional outlier handling: clip values to [-5, 5] range
            normalized = np.clip(normalized, -5, 5)

            print("Feature normalization statistics:")
            print(f"Mean range: [{normalized.mean():.4f}, {normalized.std():.4f}]")
            print(f"Value range: [{normalized.min():.4f}, {normalized.max():.4f}]")

            # Convert back to dictionary format
            normalized_dict = {}
            for idx, event_id in enumerate(event_ids):
                normalized_dict[event_id] = {
                    fname: normalized[idx, i]
                    for i, fname in enumerate(all_feature_names)
                }

            return normalized_dict

        except Exception as e:
            print(f"Error during normalization: {e}")
            print("Feature matrix statistics:")
            print(f"Shape: {feature_matrix.shape}")
            print(f"Number of NaNs: {np.isnan(feature_matrix).sum()}")
            print(f"Number of infinities: {np.isinf(feature_matrix).sum()}")
            raise e

    waveform_features_dict = normalize_features(waveform_features_dict)

    # Print some statistics about the normalized features
    sample_event = next(iter(waveform_features_dict.values()))
    print(f"Feature statistics after normalization:")
    print(
        f"Mean range: [{min(sample_event.values()):.4f}, {max(sample_event.values()):.4f}]"
    )

    # Identify mainshock and aftershocks
    mainshock, aftershocks = identify_mainshock_and_aftershocks(metadata)

    # Debug metadata after processing
    debug_metadata(metadata, aftershocks, mainshock)

    # Step 2: Create aftershock sequences
    print("\n=== Creating Aftershock Sequences ===")
    sequences = prepare_sequences_with_absolute_targets(
        aftershocks,
        waveform_features_dict,
        sequence_length=args.sequence_length,
        time_window_hours=args.time_window,
    )

    # Debug sequences
    debug_sequences(sequences)

    if not sequences:
        print("No valid sequences created. Exiting.")
        return

    # Step 3: Build graph representations
    print("\n=== Building Graph Representations ===")
    graph_dataset, feature_names = build_graphs_from_sequences(
        sequences, distance_threshold_km=args.distance_threshold, debug=args.debug
    )

    # Debug graphs
    debug_graphs(graph_dataset, feature_names)

    if len(graph_dataset) == 0:
        print("No valid graph representations created. Exiting.")
        return

    # Visualize sample graphs
    for i in range(min(3, len(graph_dataset))):
        visualize_graph_structure(graph_dataset[i], i)

    # Step 4: Create the appropriate model
    print(f"\n=== Creating {args.model_type.upper()} Model ===")
    metadata_channels = 4  # latitude, longitude, depth, hours since mainshock
    waveform_channels = len(feature_names)  # Number of waveform features

    # Create the model
    model = ModelClass(
        metadata_channels=metadata_channels,
        waveform_channels=waveform_channels,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    ).to(device)

    print(model)

    # Step 5: Train the model (or load from file)
    if not args.skip_training:
        print("\n=== Training Model with Enhanced Diagnostics ===")
        model = train_model_with_diagnostics(
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
