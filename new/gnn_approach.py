import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sys
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import logging
from main_approach import *


# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(
    log_dir,
    f"aftershock_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)

# Set style for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper")

# Reuse your existing functions for data loading and feature extraction
# We'll focus on the GNN implementation


class AfterShockGNN(nn.Module):
    """
    Graph Neural Network for aftershock location prediction
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=3):
        super(AfterShockGNN, self).__init__()

        self.num_layers = num_layers

        # Input layer
        self.conv_first = GCNConv(input_dim, hidden_dim)

        # Hidden layers
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Final layer for prediction
        self.conv_last = GCNConv(hidden_dim, hidden_dim)

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, batch=None):
        # First layer
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Hidden layers
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # Last layer
        x = self.conv_last(x, edge_index)
        x = F.relu(x)

        # If batch is not None, we're in batch mode
        if batch is not None:
            x = global_mean_pool(x, batch)

        # Final prediction through MLP
        x = self.mlp(x)

        return x


class GATAfterShockModel(nn.Module):
    """
    Graph Attention Network for aftershock location prediction
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=3, heads=4):
        super(GATAfterShockModel, self).__init__()

        self.num_layers = num_layers

        # Input layer with multi-head attention
        self.conv_first = GATv2Conv(input_dim, hidden_dim // heads, heads=heads)

        # Hidden layers
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads))

        # Final GAT layer
        self.conv_last = GATv2Conv(hidden_dim, hidden_dim, heads=1)

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, batch=None):
        # First layer
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Hidden layers
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # Last layer
        x = self.conv_last(x, edge_index)
        x = F.relu(x)

        # If batch is not None, we're in batch mode
        if batch is not None:
            x = global_mean_pool(x, batch)

        # Final prediction through MLP
        x = self.mlp(x)

        return x


def build_aftershock_graph(
    ml_df, selected_features, max_distance=100, max_time_diff=48
):
    """
    Build a graph from aftershock data

    Args:
        ml_df: DataFrame with aftershock data
        selected_features: List of features to use for node attributes
        max_distance: Maximum distance (km) for connecting nodes
        max_time_diff: Maximum time difference (hours) for connecting nodes

    Returns:
        PyTorch Geometric Data object
    """
    # Sort by time
    ml_df = ml_df.sort_values("hours_since_mainshock").reset_index(drop=True)

    # Extract features and targets
    X = ml_df[selected_features].values
    y = ml_df[["latitude", "longitude"]].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to torch tensors
    x = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    # Create edge list
    edge_index = []
    edge_attr = []

    # Iterate through all pairs of aftershocks
    for i in range(len(ml_df)):
        for j in range(i):  # Only connect to previous aftershocks
            # Calculate time difference
            time_diff = (
                ml_df.iloc[i]["hours_since_mainshock"]
                - ml_df.iloc[j]["hours_since_mainshock"]
            )

            # Calculate spatial distance
            distance = haversine_distance(
                ml_df.iloc[i]["latitude"],
                ml_df.iloc[i]["longitude"],
                ml_df.iloc[j]["latitude"],
                ml_df.iloc[j]["longitude"],
            )

            # Add edges if within thresholds
            if time_diff <= max_time_diff and distance <= max_distance:
                edge_index.append([j, i])  # j -> i (past to future)
                edge_attr.append([time_diff, distance])

    # Check if we have any edges
    if not edge_index:
        logging.warning(
            "No edges found in the graph! Consider adjusting max_distance and max_time_diff"
        )
        # Add placeholder edges to avoid errors
        edge_index = [[0, 0]]
        edge_attr = [[0, 0]]

    # Convert edge lists to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Add self-loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(ml_df))

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    logging.info(f"Built graph with {len(ml_df)} nodes and {edge_index.size(1)} edges")

    return data, scaler


def train_gnn_model(graph_data, model, epochs=200, lr=0.001):
    """
    Train a GNN model for aftershock location prediction

    Args:
        graph_data: PyTorch Geometric Data object
        model: The GNN model
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        Trained model and training history
    """
    # Split into train and test sets (time-aware)
    num_nodes = graph_data.x.size(0)
    train_size = int(0.8 * num_nodes)

    # Time-ordered split
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:train_size] = True
    test_mask[train_size:] = True

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    history = {
        "train_loss": [],
        "val_loss": [],
        "mean_error_km": [],
        "median_error_km": [],
    }

    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)

        # Calculate loss
        loss = criterion(out[train_mask], graph_data.y[train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log training loss
        history["train_loss"].append(loss.item())

        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(graph_data.x, graph_data.edge_index)
                val_loss = criterion(val_out[test_mask], graph_data.y[test_mask])
                history["val_loss"].append(val_loss.item())

                # Calculate error in kilometers
                pred_coords = val_out[test_mask].detach().numpy()
                true_coords = graph_data.y[test_mask].detach().numpy()

                errors_km = []
                for i in range(len(pred_coords)):
                    error = haversine_distance(
                        true_coords[i, 0],
                        true_coords[i, 1],
                        pred_coords[i, 0],
                        pred_coords[i, 1],
                    )
                    errors_km.append(error)

                mean_error = np.mean(errors_km)
                median_error = np.median(errors_km)
                history["mean_error_km"].append(mean_error)
                history["median_error_km"].append(median_error)

                logging.info(
                    f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                    f"Mean Error: {mean_error:.2f} km, Median Error: {median_error:.2f} km"
                )

            model.train()

    return model, history


def evaluate_gnn_model(graph_data, model, scaler, test_mask):
    """
    Evaluate GNN model performance with detailed metrics

    Args:
        graph_data: PyTorch Geometric Data object
        model: Trained GNN model
        scaler: Feature scaler
        test_mask: Mask for test nodes

    Returns:
        Dictionary of performance metrics
    """
    model.eval()

    with torch.no_grad():
        # Get predictions
        pred = model(graph_data.x, graph_data.edge_index)
        pred_test = pred[test_mask].detach().numpy()
        true_test = graph_data.y[test_mask].detach().numpy()

        # Calculate errors in kilometers
        errors_km = []
        for i in range(len(pred_test)):
            error = haversine_distance(
                true_test[i, 0], true_test[i, 1], pred_test[i, 0], pred_test[i, 1]
            )
            errors_km.append(error)

        # Calculate metrics
        mean_error = np.mean(errors_km)
        median_error = np.median(errors_km)
        mae = np.mean(np.abs(pred_test - true_test))
        mse = np.mean((pred_test - true_test) ** 2)
        rmse = np.sqrt(mse)

        # Return metrics
        metrics = {
            "mean_error_km": mean_error,
            "median_error_km": median_error,
            "MAE": mae,
            "RMSE": rmse,
            "errors_km": errors_km,
            "pred_coords": pred_test,
            "true_coords": true_test,
        }

    return metrics


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers"""
    R = 6371  # Earth radius in kilometers
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def visualize_gnn_results(metrics, ml_df, name="gnn"):
    """
    Visualize GNN prediction results

    Args:
        metrics: Dictionary of performance metrics
        ml_df: Original DataFrame with aftershock data
        name: Name prefix for saving figures
    """
    # Extract data
    pred_coords = metrics["pred_coords"]
    true_coords = metrics["true_coords"]
    errors_km = metrics["errors_km"]

    # 1. Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors_km, bins=20)
    plt.xlabel("Location Error (km)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Location Prediction Errors (GNN)")
    plt.axvline(
        np.median(errors_km),
        color="r",
        linestyle="--",
        label=f"Median Error: {np.median(errors_km):.2f} km",
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_error_distribution.png")
    plt.close()

    # 2. Actual vs Predicted Map
    plt.figure(figsize=(12, 10))
    plt.scatter(
        true_coords[:, 1], true_coords[:, 0], c="blue", label="Actual", alpha=0.7, s=50
    )
    plt.scatter(
        pred_coords[:, 1],
        pred_coords[:, 0],
        c="red",
        label="Predicted",
        alpha=0.5,
        s=30,
    )

    # Connect points with lines
    for i in range(len(true_coords)):
        plt.plot(
            [true_coords[i, 1], pred_coords[i, 1]],
            [true_coords[i, 0], pred_coords[i, 0]],
            "k-",
            alpha=0.2,
        )

    plt.title(
        f"Actual vs Predicted Locations (GNN)\n"
        f"Median Error: {np.median(errors_km):.2f} km, Mean Error: {np.mean(errors_km):.2f} km"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add stats in a box
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    textstr = "\n".join(
        (
            f"Total points: {len(true_coords)}",
            f"Median error: {np.median(errors_km):.2f} km",
            f"Mean error: {np.mean(errors_km):.2f} km",
            f"Min error: {min(errors_km):.2f} km",
            f"Max error: {max(errors_km):.2f} km",
        )
    )
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(f"{name}_spatial_prediction.png", dpi=300)
    plt.close()

    # 3. Training history
    if "train_loss" in metrics:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(metrics["train_loss"], label="Train Loss")
        plt.plot(
            range(0, len(metrics["train_loss"]), 10),
            metrics["val_loss"],
            label="Val Loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(
            range(0, len(metrics["train_loss"]), 10),
            metrics["mean_error_km"],
            label="Mean Error",
        )
        plt.plot(
            range(0, len(metrics["train_loss"]), 10),
            metrics["median_error_km"],
            label="Median Error",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Error (km)")
        plt.title("Error Metrics During Training")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{name}_training_history.png")
        plt.close()


def visualize_graph_structure(graph_data, ml_df, name="aftershock_graph"):
    """
    Visualize the structure of the aftershock graph

    Args:
        graph_data: PyTorch Geometric Data object
        ml_df: Original DataFrame with aftershock data
        name: Name prefix for saving figures
    """
    try:
        import networkx as nx
        from torch_geometric.utils import to_networkx

        # Convert to networkx graph
        G = to_networkx(graph_data, to_undirected=True)

        # Get node positions from coordinates
        pos = {}
        for i in range(len(ml_df)):
            pos[i] = (ml_df.iloc[i]["longitude"], ml_df.iloc[i]["latitude"])

        # Plot graph with physical coordinates
        plt.figure(figsize=(12, 10))

        # Color nodes by time
        time_values = ml_df["hours_since_mainshock"].values
        node_colors = plt.cm.viridis(
            (time_values - min(time_values))
            / (max(time_values) - min(time_values) + 1e-10)
        )

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.7)

        # Draw edges with alpha proportional to inverse distance
        edges = list(G.edges())
        edge_colors = []

        for u, v in edges:
            if u == v:  # Skip self-loops for clarity
                continue

            # Calculate distance for edge color
            dist = haversine_distance(
                ml_df.iloc[u]["latitude"],
                ml_df.iloc[u]["longitude"],
                ml_df.iloc[v]["latitude"],
                ml_df.iloc[v]["longitude"],
            )

            # Normalize distance for color (closer = darker)
            max_dist = 100  # Maximum expected distance
            norm_dist = max(0, 1 - dist / max_dist)
            edge_colors.append(norm_dist)

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, width=0.5, alpha=0.3, edge_color=edge_colors, edge_cmap=plt.cm.Blues
        )

        plt.title("Aftershock Graph Structure\nNodes colored by time since mainshock")
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Hours since mainshock"
        )

        plt.tight_layout()
        plt.savefig(f"{name}_structure.png", dpi=300)
        plt.close()

        # Also visualize edge distribution
        plt.figure(figsize=(10, 6))
        degrees = [G.degree(n) for n in G.nodes()]
        plt.hist(degrees, bins=20)
        plt.xlabel("Node Degree")
        plt.ylabel("Frequency")
        plt.title("Distribution of Node Degrees in Aftershock Graph")
        plt.grid(True)
        plt.savefig(f"{name}_degree_distribution.png")
        plt.close()

    except ImportError:
        logging.warning(
            "NetworkX not installed, skipping graph structure visualization"
        )


def gnn_feature_importance(graph_data, model, selected_features):
    """
    Estimate feature importance for the GNN model using perturbation

    Args:
        graph_data: PyTorch Geometric Data object
        model: Trained GNN model
        selected_features: List of feature names

    Returns:
        DataFrame with feature importance scores
    """
    model.eval()

    # Get baseline prediction
    with torch.no_grad():
        baseline_pred = model(graph_data.x, graph_data.edge_index)

    # For each feature, perturb and measure impact
    feature_scores = []

    for i in range(graph_data.x.size(1)):
        # Create perturbed feature matrix
        x_perturbed = graph_data.x.clone()
        x_perturbed[:, i] = torch.mean(x_perturbed[:, i])  # Replace with mean

        # Get prediction with perturbed feature
        with torch.no_grad():
            perturbed_pred = model(x_perturbed, graph_data.edge_index)

        # Calculate impact
        impact = torch.mean(torch.abs(baseline_pred - perturbed_pred)).item()
        feature_scores.append(impact)

    # Normalize scores
    total_impact = sum(feature_scores)
    if total_impact > 0:
        feature_importance = [score / total_impact for score in feature_scores]
    else:
        feature_importance = feature_scores

    # Create DataFrame
    importance_df = pd.DataFrame(
        {
            "Feature": selected_features[: len(feature_importance)],
            "Importance": feature_importance,
        }
    ).sort_values("Importance", ascending=False)

    return importance_df


def dynamic_graph_construction(ml_df, selected_features, window_size=50, step=10):
    """
    Create a dynamic graph representation by sliding a window through the aftershock sequence

    Args:
        ml_df: DataFrame with aftershock data
        selected_features: List of features to use for node attributes
        window_size: Number of aftershocks in each window
        step: Step size for sliding window

    Returns:
        List of PyTorch Geometric Data objects
    """
    # Sort by time
    ml_df = ml_df.sort_values("hours_since_mainshock").reset_index(drop=True)

    # Create list of graphs
    graphs = []
    window_starts = range(0, max(0, len(ml_df) - window_size), step)

    for start in window_starts:
        end = min(start + window_size, len(ml_df))
        window_df = ml_df.iloc[start:end]

        # Only build graph if we have enough data
        if len(window_df) >= 5:  # Minimum number of nodes
            graph, _ = build_aftershock_graph(window_df, selected_features)
            graphs.append(graph)

    logging.info(f"Created {len(graphs)} dynamic graphs from sliding window")
    return graphs


def main_gnn_pipeline(ml_df, selected_features):
    """
    Main pipeline for GNN-based aftershock prediction

    Args:
        ml_df: DataFrame with aftershock data
        selected_features: List of features to use

    Returns:
        Trained model and evaluation metrics
    """
    # 1. Build graph from aftershock data
    logging.info("Building aftershock graph...")
    graph_data, scaler = build_aftershock_graph(ml_df, selected_features)

    # 2. Visualize graph structure
    logging.info("Visualizing graph structure...")
    visualize_graph_structure(graph_data, ml_df)

    # 3. Define and train GNN model
    input_dim = len(selected_features)
    logging.info(f"Initializing GNN model with {input_dim} input features...")

    # Try both GCN and GAT models
    gcn_model = AfterShockGNN(input_dim)
    gat_model = GATAfterShockModel(input_dim)

    # Train GCN model
    logging.info("Training GCN model...")
    trained_gcn, gcn_history = train_gnn_model(graph_data, gcn_model)

    # Train GAT model
    logging.info("Training GAT model...")
    trained_gat, gat_history = train_gnn_model(graph_data, gat_model)

    # 4. Evaluate models
    logging.info("Evaluating GNN models...")

    # Create test mask (last 20% of nodes)
    num_nodes = graph_data.x.size(0)
    train_size = int(0.8 * num_nodes)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[train_size:] = True

    # Evaluate GCN
    gcn_metrics = evaluate_gnn_model(graph_data, trained_gcn, scaler, test_mask)
    gcn_metrics.update(gcn_history)  # Add training history

    # Evaluate GAT
    gat_metrics = evaluate_gnn_model(graph_data, trained_gat, scaler, test_mask)
    gat_metrics.update(gat_history)  # Add training history

    # 5. Visualize results
    logging.info("Visualizing results...")
    visualize_gnn_results(gcn_metrics, ml_df, name="gcn")
    visualize_gnn_results(gat_metrics, ml_df, name="gat")

    # 6. Analyze feature importance
    logging.info("Analyzing feature importance...")
    gcn_importance = gnn_feature_importance(graph_data, trained_gcn, selected_features)
    gat_importance = gnn_feature_importance(graph_data, trained_gat, selected_features)

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    gcn_importance.plot.barh()
    plt.title("Feature Importance (GCN)")
    plt.tight_layout()
    plt.savefig("gcn_feature_importance.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    gat_importance.plot.barh()
    plt.title("Feature Importance (GAT)")
    plt.tight_layout()
    plt.savefig("gat_feature_importance.png")
    plt.close()

    # 7. Return results
    results = {
        "gcn_model": trained_gcn,
        "gat_model": trained_gat,
        "gcn_metrics": gcn_metrics,
        "gat_metrics": gat_metrics,
        "gcn_importance": gcn_importance,
        "gat_importance": gat_importance,
        "graph_data": graph_data,
    }

    return results


def modified_main():
    start_time = datetime.datetime.now()

    logging.info("=== Aftershock Location Prediction Model ===")
    logging.info(f"Started at: {start_time}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"NumPy version: {np.__version__}")
    logging.info(f"Pandas version: {pd.__version__}")

    # 1. Load the dataset (limit to 5000 waveforms to keep runtime reasonable)
    logging.info("\nStep 1: Loading and preprocessing data...")
    metadata, iquique, waveform_features_dict = load_aftershock_data_with_waveforms(
        max_waveforms=13400
    )

    # 2. Identify mainshock and aftershocks
    logging.info("\nStep 2: Identifying mainshock and aftershocks...")
    mainshock, aftershocks = identify_mainshock_and_aftershocks(metadata)

    # 3. Consolidate station recordings
    logging.info("\nStep 3: Consolidating station recordings...")
    consolidated_metadata, consolidated_features = consolidate_station_recordings(
        metadata, waveform_features_dict
    )

    # 3.5. Match event_ids to aftershocks
    logging.info("\nStep 3.5: Matching event IDs to aftershocks...")
    # Create key columns in both DataFrames to match events
    aftershocks["key"] = aftershocks.apply(
        lambda row: f"{row['source_origin_time']}_{row['source_latitude_deg']:.4f}_{row['source_longitude_deg']:.4f}_{row['source_depth_km']:.1f}",
        axis=1,
    )
    consolidated_metadata["key"] = consolidated_metadata.apply(
        lambda row: f"{row['source_origin_time']}_{row['lat_rounded']}_{row['lon_rounded']}_{row['depth_rounded']}",
        axis=1,
    )

    # Merge event_id into aftershocks
    aftershocks = aftershocks.merge(
        consolidated_metadata[["key", "event_id"]], on="key", how="left"
    )

    # Check if merge was successful
    missing_ids = aftershocks["event_id"].isna().sum()
    logging.info(
        f"Aftershocks without matched event_id: {missing_ids}/{len(aftershocks)}"
    )

    # Filter to keep only aftershocks with event_id
    aftershocks = aftershocks[~aftershocks["event_id"].isna()]
    logging.info(f"Proceeding with {len(aftershocks)} aftershocks that have event_ids")

    # 4. Prepare ML dataset
    logging.info("\nStep 4: Preparing machine learning dataset...")
    ml_df = prepare_ml_dataset(aftershocks, consolidated_features)
    logging.info(f"Dataset shape: {ml_df.shape}")

    # Log some basic statistics about the dataset
    if len(ml_df) > 0:
        logging.info("\nDataset statistics:")
        for col in ["hours_since_mainshock", "distance_from_mainshock_km", "depth_km"]:
            if col in ml_df.columns:
                logging.info(
                    f"  {col}: min={ml_df[col].min():.2f}, max={ml_df[col].max():.2f}, mean={ml_df[col].mean():.2f}"
                )

        # Count how many events have each feature
        feature_counts = {
            col: ml_df[col].count()
            for col in ml_df.columns
            if col not in ["latitude", "longitude"]
        }
        logging.info("\nFeature availability counts:")
        for feature, count in sorted(
            feature_counts.items(), key=lambda x: x[1], reverse=True
        ):
            if count < len(ml_df):
                logging.info(
                    f"  {feature}: {count}/{len(ml_df)} ({count/len(ml_df)*100:.1f}%)"
                )
    else:
        logging.error("Empty dataset! Cannot proceed with training.")
        return

    # 5. Engineer features
    logging.info("\nStep 5: Engineering features...")
    ml_df = safe_engineer_features(ml_df)
    logging.info(f"Dataset shape after feature engineering: {ml_df.shape}")
    # Your existing code for data loading and preprocessing
    # ...

    # 6. Define and select features
    selected_features = [
        # Your chosen features here
        "N_spec_dom_freq_std",
        "S_Z_LH_ratio",
        "S_E_LH_ratio",
        "log_hours",
        "hours_since_mainshock",
        "N_PS_ratio",
        "Z_energy_ratio",
        "Z_wavelet_band_ratio_0_1",
        "Z_spec_dom_freq_std",
        "Z_low_freq_decay_rate",
        "N_energy_ratio",
        "Z_PS_ratio",
        "P_E_LH_ratio",
        "day_number",
        "P_linearity",
        "depth_km",
        "depth_rolling_avg",
    ]
    # Ensure only available features are used
    selected_features = [feat for feat in selected_features if feat in ml_df.columns]
    logging.info(f"Selected {len(selected_features)} features for prediction.")

    # Build ml_df and selected_features as in your original code
    # ...

    try:
        # Run GNN pipeline
        gnn_results = main_gnn_pipeline(ml_df, selected_features)

        # Log results
        logging.info("\nGNN Model Results:")
        logging.info(
            f"GCN Median Error: {gnn_results['gcn_metrics']['median_error_km']:.2f} km"
        )
        logging.info(
            f"GAT Median Error: {gnn_results['gat_metrics']['median_error_km']:.2f} km"
        )

        # Compare with traditional models
        traditional_results = compare_models(ml_df, selected_features)

        # Create comparison table with both traditional and GNN models
        all_results = {**traditional_results}  # Start with traditional results
        all_results["GCN"] = {
            "median_error_km": gnn_results["gcn_metrics"]["median_error_km"],
            "mean_error_km": gnn_results["gcn_metrics"]["mean_error_km"],
            "MAE": gnn_results["gcn_metrics"]["MAE"],
            "RMSE": gnn_results["gcn_metrics"]["RMSE"],
        }
        all_results["GAT"] = {
            "median_error_km": gnn_results["gat_metrics"]["median_error_km"],
            "mean_error_km": gnn_results["gat_metrics"]["mean_error_km"],
            "MAE": gnn_results["gat_metrics"]["MAE"],
            "RMSE": gnn_results["gat_metrics"]["RMSE"],
        }

        # Visualize comparison
        visualize_model_comparison(all_results)

    except Exception as e:
        logging.error(f"Error in GNN pipeline: {e}")
        import traceback

        logging.error(traceback.format_exc())


if __name__ == "__main__":
    modified_main()
