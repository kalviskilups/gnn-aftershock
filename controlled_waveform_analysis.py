import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import torch
import seaborn as sns
from scipy import stats
import pickle
import inspect
import copy

# Import your existing functions
from relative_gnn import (
    read_data_from_pickle,
    create_causal_spatiotemporal_graph,
    RelativeGNNAftershockPredictor,
    plot_relative_results,
    plot_3d_aftershocks,
    extract_waveform_features,
    CausalGNN,  # Make sure this is exposed for the custom model creation
)


class WaveformPreprocessor:
    """
    Specialized preprocessing for waveform features to improve their contribution
    """

    def __init__(self, log_transform=True, feature_selection=True, top_k_features=10):
        self.log_transform = log_transform
        self.feature_selection = feature_selection
        self.top_k = top_k_features
        self.selected_indices = None
        self.feature_importance = None

    def fit(self, waveform_features):
        """Find most informative waveform features"""
        # Apply log transform if needed
        if self.log_transform:
            # Add small constant to avoid log(0)
            features = self._safe_log_transform(waveform_features)
        else:
            features = waveform_features

        if self.feature_selection:
            # Calculate variance for each feature
            variances = np.var(features, axis=0)

            # Calculate coefficient of variation (normalized variance)
            means = np.mean(np.abs(features), axis=0)
            cv = np.zeros_like(variances)
            nonzero_means = means != 0
            cv[nonzero_means] = variances[nonzero_means] / means[nonzero_means]

            # Select top_k features with highest CV
            self.selected_indices = np.argsort(cv)[-self.top_k :]
            self.feature_importance = cv

            print(f"Selected {len(self.selected_indices)} waveform features")
            print(f"Top feature importance values: {cv[self.selected_indices]}")
        else:
            # Use all features
            self.selected_indices = np.arange(waveform_features.shape[1])

        return self

    def transform(self, waveform_features):
        """Apply preprocessing to waveform features"""
        if self.log_transform:
            features = self._safe_log_transform(waveform_features)
        else:
            features = waveform_features

        if self.selected_indices is not None:
            return features[:, self.selected_indices]
        return features

    def _safe_log_transform(self, data):
        """Apply log transform while handling negative values"""
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            col = data[:, i]
            min_val = np.min(col)

            if min_val < 0:
                # Shift to make all values positive
                shifted = col - min_val + 1e-6
                result[:, i] = np.log1p(shifted)
            else:
                # Add small constant to avoid log(0)
                result[:, i] = np.log1p(col + 1e-6)

        return result


def analyze_feature_importance(waveform_features, preprocessor):
    """Analyze which features were selected and their importance"""

    # Define feature names
    feature_names = []
    components = ["Vertical", "North-South", "East-West"]
    time_features = [
        "Max Amplitude",
        "Mean Amplitude",
        "StdDev",
        "Skewness",
        "Kurtosis",
    ]
    freq_features = [
        "Spectral Centroid",
        "Low Freq Energy",
        "Mid Freq Energy",
        "High Freq Energy",
    ]

    for comp in components:
        for feat in time_features:
            feature_names.append(f"{comp} {feat}")
        for feat in freq_features:
            feature_names.append(f"{comp} {feat}")

    # Get feature importance from preprocessor
    importance = preprocessor.feature_importance
    selected = preprocessor.selected_indices

    # Print mapping of feature names to importance
    print("Selected Features and their Importance:")
    for idx, imp in sorted(
        zip(selected, importance[selected]), key=lambda x: x[1], reverse=True
    ):
        print(f"{feature_names[idx]}: {imp:.6f}")

    return feature_names, importance, selected


def extract_and_preprocess_features(df, top_k=10):
    """
    Extract and preprocess waveform features.
    Returns all necessary information without modifying global state.
    """
    # Extract raw waveform features
    print("Extracting raw waveform features...")
    waveform_features = []
    for waveform in tqdm(df["waveform"]):
        features = extract_waveform_features(waveform)
        waveform_features.append(features)

    waveform_features = np.array(waveform_features)

    # Preprocess features
    print("Preprocessing waveform features...")
    preprocessor = WaveformPreprocessor(
        log_transform=True, feature_selection=True, top_k_features=top_k
    )
    preprocessor.fit(waveform_features)

    # Get processed features
    processed_features = preprocessor.transform(waveform_features)

    # Analyze feature importance
    feature_names, importance, selected = analyze_feature_importance(
        waveform_features, preprocessor
    )

    # Return comprehensive data dictionary
    return {
        "raw_features": waveform_features,
        "processed_features": processed_features,
        "selected_indices": selected,
        "feature_importance": importance,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
    }


def create_graphs_with_processor(
    df, process_waveform_func, time_window, spatial_threshold, min_connections
):
    """
    Create causal spatiotemporal graphs using the provided waveform processing function.
    This is a clean implementation that doesn't modify global state.

    Args:
        df: DataFrame with aftershock data
        process_waveform_func: Function to process waveform features
        time_window, spatial_threshold, min_connections: Graph creation parameters

    Returns:
        List of graph objects, reference coordinates
    """
    print(
        f"Creating causal relative spatiotemporal graphs (time window: {time_window}h, spatial threshold: {spatial_threshold}km)..."
    )

    # Identify mainshock (first event in chronological order)
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp")

    # Get reference coordinates from mainshock
    reference_lat = df_sorted["source_latitude_deg"].iloc[0]
    reference_lon = df_sorted["source_longitude_deg"].iloc[0]
    reference_depth = df_sorted["source_depth_km"].iloc[0]

    print(
        f"Reference event coordinates: Lat={reference_lat:.4f}, Lon={reference_lon:.4f}, Depth={reference_depth:.2f}km"
    )

    # Calculate temporal differences in hours
    df_sorted["time_hours"] = (
        df_sorted["timestamp"] - df_sorted["timestamp"].min()
    ).dt.total_seconds() / 3600

    # Calculate relative coordinates for all events
    print("Computing relative coordinates...")
    relative_coords = []
    for i in range(len(df_sorted)):
        lat = df_sorted["source_latitude_deg"].iloc[i]
        lon = df_sorted["source_longitude_deg"].iloc[i]
        depth = df_sorted["source_depth_km"].iloc[i]

        # Import the function
        from relative_gnn import calculate_relative_coordinates

        # Calculate east-west and north-south distances in km
        ew_km, ns_km = calculate_relative_coordinates(
            lat, lon, reference_lat, reference_lon
        )

        # Calculate relative depth
        depth_rel = depth - reference_depth

        # Store relative coordinates
        relative_coords.append([ew_km, ns_km, depth_rel])

    relative_coords = np.array(relative_coords)

    # Add relative coordinates to dataframe
    df_sorted["ew_rel_km"] = relative_coords[:, 0]
    df_sorted["ns_rel_km"] = relative_coords[:, 1]
    df_sorted["depth_rel_km"] = relative_coords[:, 2]

    # Extract waveform features for each event using the provided function
    print("Extracting waveform features for nodes...")
    waveform_features = []
    for waveform in tqdm(df_sorted["waveform"]):
        features = process_waveform_func(waveform)
        waveform_features.append(features)

    # Create graph data objects
    graph_data_list = []

    # Create a graph for each event (except the first two to ensure we have context)
    for i in tqdm(range(2, len(df_sorted))):
        current_time = df_sorted["time_hours"].iloc[i]

        # Get ONLY past events within the time window
        past_indices = [
            j
            for j in range(i)  # Only consider events before current one
            if current_time - df_sorted["time_hours"].iloc[j] <= time_window
        ]

        if len(past_indices) > 0:
            # Initialize the subgraph structure
            connected_indices = []
            edges = []
            edge_attrs = []

            # Always include the target node (current event) first
            connected_indices.append(i)

            # Find relevant past events to connect
            for j in past_indices:
                past_ew = df_sorted["ew_rel_km"].iloc[j]
                past_ns = df_sorted["ns_rel_km"].iloc[j]
                past_depth = df_sorted["depth_rel_km"].iloc[j]

                # Current event's coordinates (for distance calculation only)
                curr_ew = df_sorted["ew_rel_km"].iloc[i]
                curr_ns = df_sorted["ns_rel_km"].iloc[i]
                curr_depth = df_sorted["depth_rel_km"].iloc[i]

                # Calculate spatial distance in relative coordinates
                horizontal_dist = np.sqrt(
                    (curr_ew - past_ew) ** 2 + (curr_ns - past_ns) ** 2
                )

                # Calculate 3D distance
                depth_diff = abs(curr_depth - past_depth)
                spatial_dist_3d = np.sqrt(horizontal_dist**2 + depth_diff**2)

                # If within spatial threshold, add to the graph
                if spatial_dist_3d <= spatial_threshold:
                    # Add this past event to the graph
                    if j not in connected_indices:
                        connected_indices.append(j)

                    # Get local node indices in this subgraph
                    tgt_idx = 0  # Target is always first in our connected_indices list
                    src_idx = connected_indices.index(j)  # Source (past event)

                    # Create directed edge from past event to current event ONLY
                    edges.append([src_idx, tgt_idx])

                    # Edge attributes
                    temporal_dist = current_time - df_sorted["time_hours"].iloc[j]

                    # Calculate angle from past event to current event
                    ew_diff = curr_ew - past_ew
                    ns_diff = curr_ns - past_ns
                    angle = np.degrees(np.arctan2(ew_diff, ns_diff)) % 360

                    # Relative position edge features
                    spatial_weight = 1.0 / (1.0 + spatial_dist_3d / 10.0)
                    temporal_weight = 1.0 / (1.0 + temporal_dist / 5.0)

                    # Add rupture directivity information
                    depth_similarity = np.exp(-depth_diff / 10.0)

                    # Approximate Coulomb stress change using relative position
                    stress_proxy = (
                        spatial_weight
                        * depth_similarity
                        * np.cos(np.radians(2 * angle))
                    )

                    # Edge attributes for source → target direction
                    edge_attrs.append(
                        [
                            spatial_weight,
                            temporal_weight,
                            stress_proxy,
                            1.0,  # Direction flag (1 = forward)
                            angle / 360.0,  # Normalized angle as feature
                        ]
                    )

            # Only create graph if we have enough connections
            if len(edges) >= min_connections:
                # Create separate feature sets for context nodes vs target node
                node_features = []

                # Target node features WITHOUT its own coordinates (no data leakage)
                wf_features = waveform_features[i]
                if len(wf_features) > 0:  # With waveform features
                    target_features = np.concatenate(
                        [
                            wf_features,
                            [df_sorted["time_hours"].iloc[i]],
                            [
                                0,
                                0,
                                0,
                            ],  # Zero values for coordinate features (no leakage)
                        ]
                    )
                else:  # Without waveform features
                    target_features = np.array(
                        [
                            df_sorted["time_hours"].iloc[i],
                            0,
                            0,
                            0,  # Zero values for coordinate features (no leakage)
                        ]
                    )

                node_features.append(target_features)

                # Add features for context nodes (past events)
                for idx in connected_indices[1:]:  # Skip target node (index 0)
                    wf_features = waveform_features[idx]
                    if len(wf_features) > 0:  # With waveform features
                        past_features = np.concatenate(
                            [
                                wf_features,
                                [df_sorted["time_hours"].iloc[idx]],
                                [df_sorted["ew_rel_km"].iloc[idx]],
                                [df_sorted["ns_rel_km"].iloc[idx]],
                                [df_sorted["depth_rel_km"].iloc[idx]],
                            ]
                        )
                    else:  # Without waveform features
                        past_features = np.array(
                            [
                                df_sorted["time_hours"].iloc[idx],
                                df_sorted["ew_rel_km"].iloc[idx],
                                df_sorted["ns_rel_km"].iloc[idx],
                                df_sorted["depth_rel_km"].iloc[idx],
                            ]
                        )

                    node_features.append(past_features)

                # Convert to numpy arrays
                node_features_array = np.array(node_features, dtype=np.float32)
                edge_attrs_array = np.array(edge_attrs, dtype=np.float32)

                # Convert to tensor format
                x = torch.tensor(node_features_array, dtype=torch.float)
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs_array, dtype=torch.float)

                # Target value (what we're trying to predict)
                target_coords = np.array(
                    [
                        df_sorted["ew_rel_km"].iloc[i],
                        df_sorted["ns_rel_km"].iloc[i],
                        df_sorted["depth_rel_km"].iloc[i],
                    ],
                    dtype=np.float32,
                ).reshape(1, 3)

                y = torch.tensor(target_coords, dtype=torch.float)

                # Create PyTorch Geometric data object
                from torch_geometric.data import Data

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    num_nodes=len(node_features),
                )

                graph_data_list.append(data)

    print(f"Created {len(graph_data_list)} causal graphs")

    # Quick graph structure check
    if len(graph_data_list) > 0:
        print(
            f"First graph has {graph_data_list[0].num_nodes} nodes and {graph_data_list[0].edge_index.shape[1]} edges"
        )

    # Reference coordinates for conversion back to absolute coordinates
    reference_coords = {
        "latitude": reference_lat,
        "longitude": reference_lon,
        "depth": reference_depth,
    }

    return graph_data_list, reference_coords


def create_train_evaluate_with_waveforms(df, waveform_data, params, seed):
    """
    Create, train, and evaluate a model WITH waveform features.
    This is a completely self-contained function with no side effects.
    """
    # Reset RNG state to ensure consistency
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Define a function for extracting and preprocessing features
    preprocessor = waveform_data["preprocessor"]

    def process_waveform(waveform):
        """Local function to process waveform features with the fitted preprocessor"""
        raw_features = extract_waveform_features(waveform)
        return preprocessor.transform(raw_features.reshape(1, -1)).flatten()

    # Create graphs WITH waveform features - custom implementation
    print("Creating graphs WITH waveform features...")
    graphs_with_waveforms, reference_coords = create_graphs_with_processor(
        df,
        process_waveform_func=process_waveform,
        time_window=params["time_window"],
        spatial_threshold=params["spatial_threshold"],
        min_connections=params["min_connections"],
    )

    print(f"Created {len(graphs_with_waveforms)} graphs WITH waveform features")

    # Check feature dimensions
    if len(graphs_with_waveforms) > 0:
        with_features_dim = graphs_with_waveforms[0].x.shape[1]
        print(f"Feature dimensions WITH waveforms: {with_features_dim}")
    else:
        with_features_dim = 0
        print("No graphs created WITH waveforms")

    # Train and evaluate model
    print("\n===== TRAINING AND EVALUATING MODEL: with_waveforms =====")

    metrics_with, y_true_with, y_pred_with, train_losses_with, val_losses_with = (
        train_and_evaluate_model(
            graphs_with_waveforms, "with_waveforms", params, reference_coords, seed
        )
    )

    # Store selected feature info
    selected_indices = waveform_data["selected_indices"]
    feature_importance = waveform_data["feature_importance"]
    feature_names = waveform_data["feature_names"]

    # Return results
    return {
        "metrics": metrics_with,
        "y_true": y_true_with,
        "y_pred": y_pred_with,
        "train_losses": train_losses_with,
        "val_losses": val_losses_with,
        "feature_dim": with_features_dim,
        "reference_coords": reference_coords,
        "selected_indices": selected_indices,
        "feature_importance": feature_importance[selected_indices],
        "feature_names": [feature_names[i] for i in selected_indices],
    }


def create_train_evaluate_without_waveforms(df, params, seed):
    """
    Create, train, and evaluate a model WITHOUT waveform features.
    This is a completely self-contained function with no side effects.
    """
    # Reset RNG state to ensure consistency
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Define a function that returns empty waveform features
    def empty_waveform_features(waveform):
        """Return empty array for waveform features"""
        return np.array([])

    # Create graphs WITHOUT waveform features - custom implementation
    print("Creating graphs WITHOUT waveform features...")
    graphs_without_waveforms, reference_coords = create_graphs_with_processor(
        df,
        process_waveform_func=empty_waveform_features,
        time_window=params["time_window"],
        spatial_threshold=params["spatial_threshold"],
        min_connections=params["min_connections"],
    )

    print(f"Created {len(graphs_without_waveforms)} graphs WITHOUT waveform features")

    # Check feature dimensions
    if len(graphs_without_waveforms) > 0:
        without_features_dim = graphs_without_waveforms[0].x.shape[1]
        print(f"Feature dimensions WITHOUT waveforms: {without_features_dim}")
    else:
        without_features_dim = 0
        print("No graphs created WITHOUT waveforms")

    # Train and evaluate model
    print("\n===== TRAINING AND EVALUATING MODEL: without_waveforms =====")

    (
        metrics_without,
        y_true_without,
        y_pred_without,
        train_losses_without,
        val_losses_without,
    ) = train_and_evaluate_model(
        graphs_without_waveforms, "without_waveforms", params, reference_coords, seed
    )

    # Return results
    return {
        "metrics": metrics_without,
        "y_true": y_true_without,
        "y_pred": y_pred_without,
        "train_losses": train_losses_without,
        "val_losses": val_losses_without,
        "feature_dim": without_features_dim,
        "reference_coords": reference_coords,
    }


def train_and_evaluate_model(graphs, model_name, params, reference_coords, seed):
    """
    Train and evaluate a model with the given graphs.
    Uses the original RelativeGNNAftershockPredictor but with fixed seed.
    """
    # Reset random seed to ensure reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Calculate feature dimension from the graphs
    num_features = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1] if hasattr(graphs[0], "edge_attr") else 0

    print(f"Model input dimension: {num_features}")

    # Create predictor
    predictor = RelativeGNNAftershockPredictor(
        graph_data_list=graphs,
        reference_coords=reference_coords,
        gnn_type=params["model_type"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        weight_decay=params["weight_decay"],
    )

    # Train the model
    train_losses, val_losses = predictor.train(
        num_epochs=params["epochs"], patience=params["patience"]
    )

    # Test the model
    metrics, y_true, y_pred, errors = predictor.test()

    # Print metrics
    print(f"\n{model_name} Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Plot results
    plot_relative_results(
        y_true,
        y_pred,
        errors,
        reference_coords=reference_coords,
        model_name=f"{model_name}_seed{seed}",
    )

    plot_3d_aftershocks(
        y_true,
        y_pred,
        reference_coords=reference_coords,
        model_name=f"{model_name}_seed{seed}",
    )

    # Save metrics and predictions to file
    results = {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    with open(f"results/{model_name}_seed{seed}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    return metrics, y_true, y_pred, train_losses, val_losses


def compare_results(with_results, without_results, seed):
    """
    Compare the results of the two models and create visualizations.
    """
    # Extract metrics
    metrics_with = with_results["metrics"]
    metrics_without = without_results["metrics"]

    # Define key metrics to compare
    key_metrics = [
        "mean_horizontal_error",
        "median_horizontal_error",
        "mean_depth_error",
        "median_depth_error",
        "mean_3d_error",
        "median_3d_error",
        "horizontal_5km",
        "horizontal_10km",
        "horizontal_15km",
        "depth_5km",
        "depth_10km",
        "depth_15km",
        "3d_5km",
        "3d_10km",
        "3d_15km",
    ]

    # Create comparison dataframe
    metrics_df = pd.DataFrame(
        {
            "Metric": key_metrics,
            "With Waveforms": [metrics_with.get(m, np.nan) for m in key_metrics],
            "Without Waveforms": [metrics_without.get(m, np.nan) for m in key_metrics],
        }
    )

    # Calculate improvement percentage
    metrics_df["Improvement"] = np.nan

    for i, metric in enumerate(key_metrics):
        with_val = metrics_with.get(metric, np.nan)
        without_val = metrics_without.get(metric, np.nan)

        if np.isnan(with_val) or np.isnan(without_val) or without_val == 0:
            continue

        if "error" in metric:
            # For error metrics, lower is better
            improvement = ((without_val - with_val) / without_val) * 100
        else:
            # For success rate metrics, higher is better
            improvement = ((with_val - without_val) / without_val) * 100

        metrics_df.loc[i, "Improvement"] = improvement

    # Save comparison to CSV
    metrics_df.to_csv(f"results/metrics_comparison_seed{seed}.csv", index=False)

    # Create comparison bar chart for this seed
    plt.figure(figsize=(15, 10))

    # Error metrics (lower is better)
    error_metrics = [m for m in key_metrics if "error" in m]
    error_df = metrics_df[metrics_df["Metric"].isin(error_metrics)]

    x = np.arange(len(error_df))
    width = 0.35

    plt.subplot(2, 1, 1)
    plt.bar(x - width / 2, error_df["With Waveforms"], width, label="With Waveforms")
    plt.bar(
        x + width / 2, error_df["Without Waveforms"], width, label="Without Waveforms"
    )
    plt.xlabel("Metric")
    plt.ylabel("Error (km)")
    plt.title(f"Error Metrics Comparison (Seed {seed}) - Lower is Better")
    plt.xticks(x, error_df["Metric"], rotation=45)
    plt.legend()

    # Success rate metrics (higher is better)
    success_metrics = [m for m in key_metrics if "km" in m and "error" not in m]
    success_df = metrics_df[metrics_df["Metric"].isin(success_metrics)]

    x = np.arange(len(success_df))

    plt.subplot(2, 1, 2)
    plt.bar(x - width / 2, success_df["With Waveforms"], width, label="With Waveforms")
    plt.bar(
        x + width / 2, success_df["Without Waveforms"], width, label="Without Waveforms"
    )
    plt.xlabel("Metric")
    plt.ylabel("Success Rate (%)")
    plt.title(f"Success Rate Metrics Comparison (Seed {seed}) - Higher is Better")
    plt.xticks(x, success_df["Metric"], rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"results/metrics_comparison_seed{seed}.png", dpi=300, bbox_inches="tight"
    )

    # Create summary report for this seed
    with open(f"results/waveform_comparison_summary_seed{seed}.txt", "w") as f:
        f.write("================================================\n")
        f.write("   WAVEFORM FEATURES CONTRIBUTION EXPERIMENT    \n")
        f.write("================================================\n\n")

        f.write(f"Seed: {seed}\n\n")

        f.write("This experiment compared two models:\n")
        f.write(
            f"1. WITH waveform features (input dim: {with_results['feature_dim']})\n"
        )
        f.write(
            f"2. WITHOUT waveform features (input dim: {without_results['feature_dim']})\n\n"
        )

        f.write("Top waveform features by importance:\n")
        for feat_name, imp in zip(
            with_results["feature_names"], with_results["feature_importance"]
        ):
            f.write(f"- {feat_name}: {imp:.6f}\n")
        f.write("\n")

        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-----------------------\n")
        for i, row in metrics_df.iterrows():
            metric = row["Metric"]
            with_val = row["With Waveforms"]
            without_val = row["Without Waveforms"]
            improvement = row["Improvement"]

            if pd.isna(improvement):
                continue

            better = "better" if improvement > 0 else "worse"
            f.write(f"{metric}:\n")
            f.write(f"  With waveforms:    {with_val:.4f}\n")
            f.write(f"  Without waveforms: {without_val:.4f}\n")
            f.write(
                f"  Difference:        {abs(with_val - without_val):.4f} ({abs(improvement):.2f}% {better})\n\n"
            )


def run_single_seed_experiment(seed, top_k=10):
    """
    Run a single experiment with a specific random seed.
    This function maintains complete separation between with/without waveform experiments.
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load the data
    if not os.path.exists("aftershock_data.pkl"):
        print("Error: aftershock_data.pkl not found")
        return {"with_waveforms": {}, "without_waveforms": {}}

    # Load the data
    df = read_data_from_pickle("aftershock_data_best.pkl")

    # Sort data chronologically but DO NOT shuffle
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df = df_sorted.sort_values("timestamp").drop("timestamp", axis=1).reset_index(drop=True)

    # Just use the sorted dataframe as is - no shuffling!
    print(f"Loaded data with {len(df)} events")

    # Base parameters for both models
    base_params = {
        "time_window": 120,  # hours
        "spatial_threshold": 75,  # km
        "min_connections": 5,
        "model_type": "gat",
        "hidden_dim": 128,
        "num_layers": 3,
        "learning_rate": 0.0025,
        "batch_size": 8,
        "weight_decay": 5e-6,
        "epochs": 50,
        "patience": 10,
    }

    # Extract and preprocess waveform features
    print("\n1. EXTRACTING AND PREPROCESSING WAVEFORM FEATURES")
    waveform_data = extract_and_preprocess_features(df, top_k=top_k)

    # 1. Create and train model WITH waveform features
    print("\n2. CREATING AND TRAINING MODEL WITH WAVEFORM FEATURES")
    with_waveforms_results = create_train_evaluate_with_waveforms(
        df, waveform_data, base_params, seed
    )

    # 2. Create and train model WITHOUT waveform features
    print("\n3. CREATING AND TRAINING MODEL WITHOUT WAVEFORM FEATURES")
    without_waveforms_results = create_train_evaluate_without_waveforms(
        df, base_params, seed
    )

    # 3. Compare results
    print("\n4. COMPARING MODEL PERFORMANCE FOR SEED", seed)
    compare_results(with_waveforms_results, without_waveforms_results, seed)

    return {
        "with_waveforms": with_waveforms_results["metrics"],
        "without_waveforms": without_waveforms_results["metrics"],
    }


def analyze_results_statistically(all_results, seeds):
    """
    Perform statistical analysis on results from multiple seeds.
    """
    print("\n========== AGGREGATING RESULTS ACROSS SEEDS ==========")

    # Define key metrics
    key_metrics = [
        "mean_horizontal_error",
        "median_horizontal_error",
        "mean_depth_error",
        "median_depth_error",
        "mean_3d_error",
        "median_3d_error",
        "horizontal_5km",
        "horizontal_10km",
        "horizontal_15km",
        "depth_5km",
        "depth_10km",
        "depth_15km",
        "3d_5km",
        "3d_10km",
        "3d_15km",
    ]

    # Prepare data structure for aggregation
    aggregate_metrics = {"with_waveforms": {}, "without_waveforms": {}}

    # Collect all metrics across seeds
    for metric in key_metrics:
        # For "with waveforms" model
        values = [m.get(metric, np.nan) for m in all_results["with_waveforms"]]
        values = [v for v in values if not np.isnan(v)]
        if values:
            aggregate_metrics["with_waveforms"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }

        # For "without waveforms" model
        values = [m.get(metric, np.nan) for m in all_results["without_waveforms"]]
        values = [v for v in values if not np.isnan(v)]
        if values:
            aggregate_metrics["without_waveforms"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }

    # Create aggregate comparison dataframe
    agg_df = pd.DataFrame(
        {
            "Metric": key_metrics,
            "With Waveforms (Mean)": [
                aggregate_metrics["with_waveforms"].get(m, {}).get("mean", np.nan)
                for m in key_metrics
            ],
            "With Waveforms (Std)": [
                aggregate_metrics["with_waveforms"].get(m, {}).get("std", np.nan)
                for m in key_metrics
            ],
            "Without Waveforms (Mean)": [
                aggregate_metrics["without_waveforms"].get(m, {}).get("mean", np.nan)
                for m in key_metrics
            ],
            "Without Waveforms (Std)": [
                aggregate_metrics["without_waveforms"].get(m, {}).get("std", np.nan)
                for m in key_metrics
            ],
        }
    )

    # Calculate improvement percentage and statistical significance
    agg_df["Improvement (%)"] = np.nan
    agg_df["p-value"] = np.nan

    for i, metric in enumerate(key_metrics):
        with_vals = (
            aggregate_metrics["with_waveforms"].get(metric, {}).get("values", [])
        )
        without_vals = (
            aggregate_metrics["without_waveforms"].get(metric, {}).get("values", [])
        )

        if not with_vals or not without_vals:
            continue

        with_mean = np.mean(with_vals)
        without_mean = np.mean(without_vals)

        if "error" in metric:
            # For error metrics, lower is better
            improvement = ((without_mean - with_mean) / without_mean) * 100
        else:
            # For success rate metrics, higher is better
            if without_mean == 0:
                improvement = np.inf if with_mean > 0 else 0
            else:
                improvement = ((with_mean - without_mean) / without_mean) * 100

        agg_df.loc[i, "Improvement (%)"] = improvement

        # Perform statistical test if we have multiple seeds
        if len(with_vals) > 1 and len(with_vals) == len(without_vals):
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(with_vals, without_vals)
            agg_df.loc[i, "p-value"] = p_val

    # Save aggregate comparison to CSV
    agg_df.to_csv("results/aggregate_comparison.csv", index=False)

    # Create aggregate comparison visualization
    plt.figure(figsize=(15, 10))

    # Error metrics (lower is better)
    error_metrics = [m for m in key_metrics if "error" in m]
    error_df = agg_df[agg_df["Metric"].isin(error_metrics)]

    x = np.arange(len(error_df))
    width = 0.35

    plt.subplot(2, 1, 1)
    rects1 = plt.bar(
        x - width / 2,
        error_df["With Waveforms (Mean)"],
        width,
        yerr=error_df["With Waveforms (Std)"],
        label="With Waveforms",
        capsize=5,
    )
    rects2 = plt.bar(
        x + width / 2,
        error_df["Without Waveforms (Mean)"],
        width,
        yerr=error_df["Without Waveforms (Std)"],
        label="Without Waveforms",
        capsize=5,
    )
    plt.xlabel("Metric")
    plt.ylabel("Error (km)")
    plt.title(
        f"Error Metrics Comparison (Average of {len(seeds)} Seeds) - Lower is Better"
    )
    plt.xticks(x, error_df["Metric"], rotation=45)
    plt.legend()

    # Add significance markers
    for i, p in enumerate(error_df["p-value"]):
        if not pd.isna(p) and p < 0.05:
            plt.text(
                i,
                max(
                    error_df["With Waveforms (Mean)"][i],
                    error_df["Without Waveforms (Mean)"][i],
                )
                + max(
                    error_df["With Waveforms (Std)"][i],
                    error_df["Without Waveforms (Std)"][i],
                )
                + 0.5,
                "*",
                ha="center",
                fontsize=16,
            )

    # Success rate metrics (higher is better)
    success_metrics = [m for m in key_metrics if "km" in m and "error" not in m]
    success_df = agg_df[agg_df["Metric"].isin(success_metrics)]
    success_df = success_df.reset_index(drop=True)

    x = np.arange(len(success_df))

    plt.subplot(2, 1, 2)
    rects1 = plt.bar(
        x - width / 2,
        success_df["With Waveforms (Mean)"],
        width,
        yerr=success_df["With Waveforms (Std)"],
        label="With Waveforms",
        capsize=5,
    )
    rects2 = plt.bar(
        x + width / 2,
        success_df["Without Waveforms (Mean)"],
        width,
        yerr=success_df["Without Waveforms (Std)"],
        label="Without Waveforms",
        capsize=5,
    )
    plt.xlabel("Metric")
    plt.ylabel("Success Rate (%)")
    plt.title(
        f"Success Rate Metrics Comparison (Average of {len(seeds)} Seeds) - Higher is Better"
    )
    plt.xticks(x, success_df["Metric"], rotation=45)
    plt.legend()

    # Add significance markers
    for i, p in enumerate(success_df['p-value']):
        if not pd.isna(p) and p < 0.05:
            plt.text(i, max(success_df['With Waveforms (Mean)'].iloc[i], success_df['Without Waveforms (Mean)'].iloc[i]) + 
                    max(success_df['With Waveforms (Std)'].iloc[i], success_df['Without Waveforms (Std)'].iloc[i]) + 1, 
                    '*', ha='center', fontsize=16)

    plt.tight_layout()
    plt.savefig("results/aggregate_comparison.png", dpi=300, bbox_inches="tight")

    # Create aggregate summary report
    with open("results/aggregate_waveform_comparison_summary.txt", "w") as f:
        f.write("================================================\n")
        f.write("   WAVEFORM FEATURES CONTRIBUTION EXPERIMENT    \n")
        f.write("================================================\n\n")

        f.write(f"Aggregate results across {len(seeds)} seeds: {seeds}\n\n")

        f.write("This experiment compared two models:\n")
        f.write("1. WITH waveform features (preprocessed)\n")
        f.write("2. WITHOUT waveform features (properly excluded)\n\n")

        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-----------------------\n")
        for i, row in agg_df.iterrows():
            metric = row["Metric"]
            with_mean = row["With Waveforms (Mean)"]
            with_std = row["With Waveforms (Std)"]
            without_mean = row["Without Waveforms (Mean)"]
            without_std = row["Without Waveforms (Std)"]
            improvement = row["Improvement (%)"]
            p_value = row["p-value"]

            if pd.isna(improvement):
                continue

            better = "better" if improvement > 0 else "worse"
            significant = "*" if (not pd.isna(p_value) and p_value < 0.05) else ""

            f.write(f"{metric}:{significant}\n")
            f.write(f"  With waveforms:    {with_mean:.4f} ± {with_std:.4f}\n")
            f.write(f"  Without waveforms: {without_mean:.4f} ± {without_std:.4f}\n")
            f.write(
                f"  Difference:        {abs(with_mean - without_mean):.4f} ({abs(improvement):.2f}% {better})\n"
            )
            if not pd.isna(p_value):
                f.write(f"  p-value:           {p_value:.4f}")
                f.write(
                    " (statistically significant)\n\n" if p_value < 0.05 else "\n\n"
                )
            else:
                f.write("\n")

        f.write("\nCONCLUSION:\n")
        f.write("-----------\n")

        # Calculate overall improvement
        error_metrics_df = agg_df[agg_df["Metric"].str.contains("error")]
        success_metrics_df = agg_df[~agg_df["Metric"].str.contains("error")]

        error_improvements = error_metrics_df["Improvement (%)"].mean()
        success_improvements = success_metrics_df["Improvement (%)"].mean()

        significant_improvements = agg_df[
            (agg_df["p-value"] < 0.05) & (~agg_df["Improvement (%)"].isna())
        ]
        num_significant = len(significant_improvements)

        f.write(f"Based on {len(seeds)} different random seeds:\n\n")

        if error_improvements > 0:
            f.write(
                f"The model WITH waveform features showed improvement in error metrics, "
            )
            f.write(f"with an average error reduction of {error_improvements:.2f}%.\n")
        else:
            f.write(
                f"The model WITHOUT waveform features performed better on error metrics, "
            )
            f.write(
                f"with {-error_improvements:.2f}% lower errors than the model with waveforms.\n"
            )

        if success_improvements > 0:
            f.write(f"The model WITH waveform features showed improved success rates, ")
            f.write(f"with an average improvement of {success_improvements:.2f}%.\n")
        else:
            f.write(f"The model WITHOUT waveform features had better success rates, ")
            f.write(
                f"with {-success_improvements:.2f}% higher success rates than the model with waveforms.\n"
            )

        if num_significant > 0:
            f.write(
                f"\n{num_significant} metrics showed statistically significant differences (p < 0.05):\n"
            )
            for i, row in significant_improvements.iterrows():
                metric = row["Metric"]
                improvement = row["Improvement (%)"]
                better = "better" if improvement > 0 else "worse"
                f.write(
                    f"- {metric}: {abs(improvement):.2f}% {better} with waveforms (p={row['p-value']:.4f})\n"
                )
        else:
            f.write(
                "\nNo metrics showed statistically significant differences (p < 0.05).\n"
            )


def run_experiment_with_seeds(seeds=[42, 123, 456, 789, 1024], top_k=10):
    """
    Run a properly controlled experiment comparing models with and without waveform features
    using multiple random seeds for statistical validity.
    """
    # Create the output directory
    os.makedirs("results", exist_ok=True)

    # Store results for each seed and model type
    all_results = {"with_waveforms": [], "without_waveforms": []}

    # Run experiment for each seed and collect results
    for seed_idx, seed in enumerate(seeds):
        print(
            f"\n========== EXPERIMENT WITH SEED {seed} ({seed_idx+1}/{len(seeds)}) =========="
        )

        # Run experiment for this seed
        results = run_single_seed_experiment(seed, top_k=top_k)

        # Store results
        all_results["with_waveforms"].append(results["with_waveforms"])
        all_results["without_waveforms"].append(results["without_waveforms"])

    # Perform statistical analysis if we have multiple seeds
    if len(seeds) > 1:
        analyze_results_statistically(all_results, seeds)

    print("\nExperiment complete. Results saved to results directory")
    return all_results


def properly_exclude_waveforms_experiment(seeds=[42, 123, 456, 789, 1024], top_k=10):
    """
    Run a controlled experiment comparing models with and without waveform features
    using proper isolation and multiple seeds for statistical validity.
    """
    return run_experiment_with_seeds(seeds, top_k)


if __name__ == "__main__":
    # Run with multiple seeds for statistical validity
    # Adjust seeds and top_k as needed
    properly_exclude_waveforms_experiment(seeds=[42, 123, 456, 789, 1024], top_k=5)
