from datetime import datetime
import os
import argparse
from sklearn.preprocessing import RobustScaler
import torch
import numpy as np
from debug import *


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
        default=7,
        help="Number of events in each sequence",
    )
    parser.add_argument(
        "--max_waveforms",
        type=int,
        default=13400,
        help="Maximum number of waveforms to process",
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
        "--debug",
        action="store_true",
        help="Enable debug mode with extra print statements",
    )

    return parser.parse_args()


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


def build_graphs_from_sequences_with_edge_attr(
    sequences, distance_threshold_km=25, debug=False
):
    """Build graph representations with edge attributes for weights"""
    from torch_geometric.data import Data
    import torch
    import numpy as np

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
        edge_attr_list = []  # To store edge attributes

        # Always add temporal edges
        for i in range(num_nodes - 1):
            # Compute temporal distance in hours
            temp_dist = abs(metadata_features[i + 1][3] - metadata_features[i][3])
            # Normalize temporal distance (inverse so closer = higher weight)
            temp_weight = 1.0 / (1.0 + temp_dist)

            # Forward edge
            edge_list.append([i, i + 1])
            edge_attr_list.append(
                [temp_weight, 1.0, 0.0]
            )  # [temporal_weight, is_temporal, is_spatial]

            # Backward edge
            edge_list.append([i + 1, i])
            edge_attr_list.append([temp_weight, 1.0, 0.0])

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

                # Add bidirectional edges with spatial weights if within threshold
                if distance < distance_threshold_km:
                    # Normalize spatial distance (inverse so closer = higher weight)
                    spatial_weight = 1.0 - (distance / distance_threshold_km)

                    # Forward edge
                    edge_list.append([i, j])
                    edge_attr_list.append(
                        [spatial_weight, 0.0, 1.0]
                    )  # [spatial_weight, is_temporal, is_spatial]

                    # Backward edge
                    edge_list.append([j, i])
                    edge_attr_list.append([spatial_weight, 0.0, 1.0])

        # Validation check: Ensure edge indices are within bounds
        for edge in edge_list:
            if edge[0] >= num_nodes or edge[1] >= num_nodes:
                if debug:
                    print(
                        f"Warning: Edge {edge} out of bounds for graph with {num_nodes} nodes!"
                    )

        # Filter out any edges that might be out of bounds
        valid_edges = []
        valid_edge_attrs = []
        for i, edge in enumerate(edge_list):
            if edge[0] < num_nodes and edge[1] < num_nodes:
                valid_edges.append(edge)
                valid_edge_attrs.append(edge_attr_list[i])

        # Convert edge list and attributes to tensors
        edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(valid_edge_attrs, dtype=torch.float)

        # Create graph with edge attributes
        graph = Data(
            metadata=metadata_tensor,
            waveform=waveform_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,  # Add edge attributes
            y=target_tensor,
            num_nodes=num_nodes,
        )

        # Final validation check
        if graph.edge_index.max() >= graph.num_nodes:
            if debug:
                print(
                    f"Error: Graph {seq_idx} still has invalid edges after filtering!"
                )
                print(
                    f"Max edge index: {graph.edge_index.max().item()}, Num nodes: {graph.num_nodes}"
                )
            continue

        graph_dataset.append(graph)

        if debug and seq_idx < 3:
            print(f"\nGraph {seq_idx + 1}:")
            print(f"Nodes: {num_nodes}")
            print(f"Edges: {len(valid_edges)}")
            print(f"Edge attributes shape: {edge_attr.shape}")
            print(f"Metadata shape: {metadata_tensor.shape}")
            print(f"Waveform shape: {waveform_tensor.shape}")
            print(f"Target: {target}")

    print(f"Created {len(graph_dataset)} valid graphs with edge attributes")
    return graph_dataset, feature_names


def main():

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n=== Loading and Preprocessing Data with Waveforms ===")
    # Import these here to avoid circular import issues
    from functions import (
        load_aftershock_data_with_waveforms,
        identify_mainshock_and_aftershocks,
        consolidate_station_recordings,
        create_domain_specific_features,
        build_graphs_with_optimized_features,
        analyze_feature_importance,
        evaluate_with_spatial_binning,
        train_with_spatial_regularization,
        custom_temporal_train_split
    )

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

    # Step 3: Build graph representations with fixed edge attributes
    print("\n=== Building Graph Representations with Edge Attributes ===")
    graph_dataset, feature_names = build_graphs_from_sequences_with_edge_attr(
        sequences, distance_threshold_km=args.distance_threshold, debug=args.debug
    )

    # Debug graphs before using them
    print("\n=== Validating Graphs Before Training ===")
    for i, graph in enumerate(graph_dataset[:5]):  # Check first 5 graphs
        print(f"Graph {i}: Nodes={graph.num_nodes}, Edges={graph.edge_index.size(1)}")
        print(
            f"  Edge index range: {graph.edge_index.min().item()} to {graph.edge_index.max().item()}"
        )
        if graph.edge_index.max() >= graph.num_nodes:
            print(f"  WARNING: Invalid edge indices detected!")

    if len(graph_dataset) == 0:
        print("No valid graph representations created. Exiting.")
        return

    # Step 4: Apply feature optimization
    print("\n=== Optimizing Waveform Features ===")

    # Option 1: Feature selection based on importance
    selected_indices, feature_results = analyze_feature_importance(
        graph_dataset, num_features=20  # Select top 30 features
    )
    optimized_dataset = build_graphs_with_optimized_features(
        graph_dataset, selected_indices
    )

    # Option 2: Apply PCA for dimensionality reduction
    # Uncomment to use PCA instead of feature selection
    # pca_dataset, pca = apply_pca_to_features(graph_dataset, n_components=20)

    # Option 3: Add domain-specific features
    print("\n=== Adding Domain-Specific Features ===")
    enhanced_dataset = create_domain_specific_features(
        optimized_dataset, mainshock  # Use already optimized features
    )

    # Use the enhanced dataset for training
    final_dataset = enhanced_dataset

    # Step 5: Train with improved approach
    print("\n=== Training with Enhanced Approach ===")
    
    # Import the enhanced model
    from models import EnhancedGNN
    
    # Get feature dimensions from the enhanced dataset
    metadata_channels = final_dataset[0].metadata.size(1)
    waveform_channels = final_dataset[0].waveform.size(1)
    
    print(f"Training with {waveform_channels} waveform features")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create improved train/test split
    train_dataset, test_dataset = custom_temporal_train_split(final_dataset)
    
    # Train with spatial regularization
    model, training_metrics = train_with_spatial_regularization(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_class=EnhancedGNN,
        metadata_channels=metadata_channels,
        waveform_channels=waveform_channels,
        hidden_channels=128,
        num_layers=3,
        batch_size=4,
        learning_rate=0.0002,  # Reduced learning rate
        coverage_weight=0.2,   # Weight for spatial coverage regularization
        device=device
    )
    
    # Evaluate with enhanced spatial analysis
    print("\n=== Enhanced Spatial Evaluation ===")
    results = evaluate_with_spatial_binning(
        model=model,
        test_dataset=test_dataset,
        mainshock=mainshock,
        device=device
    )
    
    # Save results
    print("\n=== Saving Results ===")
    try:
        # Save model
        torch.save(model.state_dict(), "results/enhanced_model.pt")
        print("Model saved to results/enhanced_model.pt")
        
        # Save selected features
        if selected_indices is not None:
            np.save("results/selected_features.npy", selected_indices)
            print(f"Selected {len(selected_indices)} features saved to results/selected_features.npy")
        
        # Save error metrics
        error_metrics = {
            "mean_error": results["mean_error"],
            "median_error": results["median_error"],
            "percentile_90": results["percentile_90"]
        }
        
        np.save("results/error_metrics.npy", error_metrics)
        print("Error metrics saved to results/error_metrics.npy")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Print final summary
    print("\n=== Final Results ===")
    print(f"Mean Error: {results['mean_error']:.2f} km")
    print(f"Median Error: {results['median_error']:.2f} km")
    print(f"90th Percentile Error: {results['percentile_90']:.2f} km")
    
    # Record end time
    end_time = datetime.now()
    print(f"Finished at {end_time}")
    print(f"Total runtime: {end_time - start_time}")


if __name__ == "__main__":
    main()
