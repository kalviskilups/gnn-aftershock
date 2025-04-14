import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os
import time
from datetime import datetime
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, GraphConv, GATConv, SAGEConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def read_data_from_pickle(file_path):
    """
    Load the data from pickle file.
    """
    # Load the pickle file that contains the data dictionary
    with open(file_path, "rb") as file:
        data_dict = pickle.load(file)

    # Extract the metadata from each event entry
    data_list = [
        {**entry["metadata"], "waveform": entry["waveform"]}
        for entry in data_dict.values()
    ]
    # Convert the list of metadata dictionaries into a DataFrame
    df = pd.DataFrame(data_list)

    return df


def extract_waveform_features(waveform):
    """
    Extract time and frequency domain features from waveform data.
    """
    features = []
    for component in range(waveform.shape[0]):
        signal = waveform[component]

        # Time domain features
        features.extend(
            [
                np.max(np.abs(signal)),
                np.mean(np.abs(signal)),
                np.std(signal),
                scipy.stats.skew(signal),
                scipy.stats.kurtosis(signal),
            ]
        )

        # Frequency domain features
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal))

        # Spectral centroid
        if np.sum(fft) > 0:
            spectral_centroid = np.sum(freqs * fft) / np.sum(fft)
            features.append(spectral_centroid)
        else:
            features.append(0)

        # Energy in different frequency bands
        if len(freqs) > 3:
            low_idx = int(len(freqs) * 0.33)
            mid_idx = int(len(freqs) * 0.67)

            features.extend(
                [
                    np.sum(fft[:low_idx]),
                    np.sum(fft[low_idx:mid_idx]),
                    np.sum(fft[mid_idx:]),
                ]
            )

    return np.array(features)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points in km.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r


def azimuth_between_points(lat1, lon1, lat2, lon2):
    """
    Calculate the azimuth (bearing) from point 1 to point 2 in degrees.
    Returns an angle between 0 and 360 degrees, where 0 is North.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate azimuth
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    azimuth = np.arctan2(y, x)

    # Convert to degrees and normalize to 0-360
    azimuth_deg = np.degrees(azimuth)
    azimuth_deg = (azimuth_deg + 360) % 360

    return azimuth_deg


def calculate_relative_coordinates(lat1, lon1, reference_lat, reference_lon):
    """
    Calculate relative coordinates in km (east-west, north-south)
    with respect to a reference point.

    Returns: (east_west_km, north_south_km)
    Positive east_west_km means east of reference
    Positive north_south_km means north of reference
    """
    # North-south distance
    ns_distance = haversine_distance(reference_lat, reference_lon, lat1, reference_lon)
    # Add sign (positive if north, negative if south)
    ns_distance = ns_distance if lat1 > reference_lat else -ns_distance

    # East-west distance
    ew_distance = haversine_distance(reference_lat, reference_lon, reference_lat, lon1)
    # Add sign (positive if east, negative if west)
    ew_distance = ew_distance if lon1 > reference_lon else -ew_distance

    return ew_distance, ns_distance


def identify_mainshock(df):
    """
    Identify mainshock in the dataset based on event time and magnitude.
    For Iquique dataset, we'll use the earliest significant event.
    """
    # Sort by time to find the earliest events
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp")

    # For Iquique, we'll assume the mainshock is the first event in the sorted dataframe
    # In a real application, you might want to explicitly select it based on magnitude
    mainshock_idx = df_sorted.index[0]

    return mainshock_idx


def relative_haversine_loss(y_pred, y_true):
    """
    Custom loss function based on relative coordinates.

    Args:
        y_pred: tensor of shape (batch_size, 3) [east_west_km, north_south_km, depth_rel_km]
        y_true: tensor of shape (batch_size, 3) [east_west_km, north_south_km, depth_rel_km]
    """
    # Horizontal distance error (Euclidean in the relative coordinate system)
    h_dist = torch.sqrt(
        (y_pred[:, 0] - y_true[:, 0]) ** 2 + (y_pred[:, 1] - y_true[:, 1]) ** 2
    )

    # Depth error
    d_dist = torch.abs(y_pred[:, 2] - y_true[:, 2])

    # Total 3D distance
    dist_3d = torch.sqrt(h_dist**2 + d_dist**2)

    # Return mean distance
    return torch.mean(dist_3d)


def create_relative_spatiotemporal_graph(
    df, mainshock_idx=None, time_window=24, spatial_threshold=100, min_connections=1
):
    """
    Create spatiotemporal graphs using relative coordinate system.

    Args:
        df: DataFrame with aftershock information
        mainshock_idx: Index of mainshock event (if None, will be determined automatically)
        time_window: Time window in hours to consider events connected
        spatial_threshold: Maximum distance in km between connected events
        min_connections: Minimum number of connections required to create a graph

    Returns:
        List of PyTorch Geometric Data objects, reference coordinates for converting back
    """
    print(
        f"Creating relative spatiotemporal graphs (time window: {time_window}h, spatial threshold: {spatial_threshold}km)..."
    )

    # Identify mainshock if not provided
    if mainshock_idx is None:
        mainshock_idx = identify_mainshock(df)

    # Get reference coordinates from mainshock
    reference_lat = df["source_latitude_deg"].iloc[mainshock_idx]
    reference_lon = df["source_longitude_deg"].iloc[mainshock_idx]
    reference_depth = df["source_depth_km"].iloc[mainshock_idx]

    print(
        f"Reference event coordinates: Lat={reference_lat:.4f}, Lon={reference_lon:.4f}, Depth={reference_depth:.2f}km"
    )

    # Convert timestamps to datetime
    timestamps = pd.to_datetime(df["source_origin_time"])

    # Sort events by time
    df_sorted = df.copy()
    df_sorted["timestamp"] = timestamps
    df_sorted = df_sorted.sort_values("timestamp")

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

    # Extract waveform features
    print("Extracting waveform features for nodes...")
    waveform_features = np.array(
        [extract_waveform_features(w) for w in tqdm(df_sorted["waveform"])]
    )

    # Create a dataframe for node features
    node_df = pd.DataFrame(waveform_features)

    # Add additional node features
    node_df["time_hours"] = df_sorted["time_hours"].values
    node_df["ew_rel_km"] = df_sorted["ew_rel_km"].values
    node_df["ns_rel_km"] = df_sorted["ns_rel_km"].values
    node_df["depth_rel_km"] = df_sorted["depth_rel_km"].values

    # Scale node features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_df.values)

    # Create target values in relative coordinates
    targets = df_sorted[["ew_rel_km", "ns_rel_km", "depth_rel_km"]].values

    # Create graph data objects
    graph_data_list = []

    # We'll create a graph for each event (except the first one)
    for i in tqdm(range(1, len(df_sorted))):
        # Get current event
        current_time = df_sorted["time_hours"].iloc[i]
        curr_ew = df_sorted["ew_rel_km"].iloc[i]
        curr_ns = df_sorted["ns_rel_km"].iloc[i]
        curr_depth = df_sorted["depth_rel_km"].iloc[i]

        # Get past events within the time window
        time_indices = [
            j
            for j in range(i)
            if current_time - df_sorted["time_hours"].iloc[j] <= time_window
        ]

        if len(time_indices) > 0:
            # Initialize the subgraph structure
            connected_indices = []
            edges = []
            edge_attrs = []

            # Always include the current event (target node)
            connected_indices.append(i)

            # Find spatially connected events
            for j in time_indices:
                past_ew = df_sorted["ew_rel_km"].iloc[j]
                past_ns = df_sorted["ns_rel_km"].iloc[j]
                past_depth = df_sorted["depth_rel_km"].iloc[j]

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
                    tgt_idx = connected_indices.index(i)  # Target (current event)
                    src_idx = connected_indices.index(j)  # Source (past event)

                    # Create directed edge from past event to current event
                    edges.append([src_idx, tgt_idx])

                    # Create edge from current event to past event (bidirectional)
                    edges.append([tgt_idx, src_idx])

                    # Edge attributes
                    temporal_dist = current_time - df_sorted["time_hours"].iloc[j]

                    # Calculate angle from past event to current event in relative coordinates
                    ew_diff = curr_ew - past_ew
                    ns_diff = curr_ns - past_ns
                    angle = np.degrees(np.arctan2(ew_diff, ns_diff)) % 360

                    # Relative position edge features (distance and direction information)
                    spatial_weight = 1.0 / (1.0 + spatial_dist_3d / 10.0)
                    temporal_weight = 1.0 / (1.0 + temporal_dist / 5.0)

                    # Add rupture directivity information
                    # Stress concentration tends to be higher at the ends of the rupture
                    depth_similarity = np.exp(-depth_diff / 10.0)

                    # Approximate Coulomb stress change using relative position
                    # This is a simplification - real Coulomb modeling would be more complex
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

                    # Edge attributes for target → source direction
                    back_angle = (angle + 180) % 360
                    edge_attrs.append(
                        [
                            spatial_weight,
                            temporal_weight,
                            stress_proxy,
                            0.0,  # Direction flag (0 = backward)
                            back_angle / 360.0,  # Normalized angle as feature
                        ]
                    )

            # Only create graph if we have enough connections
            num_connections = (
                len(edges) // 2
            )  # Divide by 2 because of bidirectional edges

            if num_connections >= min_connections:
                # Get node features for connected events
                node_indices = sorted(connected_indices)
                sub_features = node_features[node_indices]
                sub_targets = targets[node_indices]

                # Find the index of the target node in the new ordering
                target_idx = node_indices.index(i)

                # Convert to tensors
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
                x = torch.tensor(sub_features, dtype=torch.float)

                # Create mask for target node
                mask = torch.zeros(len(node_indices), dtype=torch.bool)
                mask[target_idx] = True

                # Create PyTorch Geometric data object
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=sub_targets,  # All node coordinates
                    mask=mask,  # Mask identifying the target node
                    target_idx=torch.tensor([target_idx]),  # Index of target node
                )

                data.num_nodes = len(node_indices)
                graph_data_list.append(data)

    print(f"Created {len(graph_data_list)} graphs")

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


def relative_to_absolute_coordinates(ew_km, ns_km, depth_rel, reference_coords):
    """
    Convert relative coordinates back to absolute lat/lon/depth.

    Args:
        ew_km: East-west distance in km (positive = east)
        ns_km: North-south distance in km (positive = north)
        depth_rel: Relative depth (positive = deeper than reference)
        reference_coords: Dictionary with reference lat/lon/depth

    Returns:
        lat, lon, depth (absolute coordinates)
    """
    ref_lat = reference_coords["latitude"]
    ref_lon = reference_coords["longitude"]
    ref_depth = reference_coords["depth"]

    # Earth radius in km
    R = 6371.0

    # Convert distances to angles in radians
    # For small distances, this approximation works well
    lat_offset = ns_km / R * (180.0 / np.pi)
    lon_offset = ew_km / (R * np.cos(np.radians(ref_lat))) * (180.0 / np.pi)

    # Calculate absolute coordinates
    lat = ref_lat + lat_offset
    lon = ref_lon + lon_offset
    depth = ref_depth + depth_rel

    return lat, lon, depth


def calculate_prediction_errors_relative(y_true, y_pred, reference_coords=None):
    """
    Calculate various error metrics for relative location predictions.

    Args:
        y_true: Array of true relative coordinates [ew_km, ns_km, depth_rel]
        y_pred: Array of predicted relative coordinates [ew_km, ns_km, depth_rel]
        reference_coords: If provided, will also compute errors in absolute coordinates

    Returns:
        Metrics dictionary, horizontal errors, depth errors, 3D errors
    """
    # Calculate horizontal errors (Euclidean distance in the EW-NS plane)
    horizontal_errors = np.sqrt(
        (y_true[:, 0] - y_pred[:, 0]) ** 2 + (y_true[:, 1] - y_pred[:, 1]) ** 2
    )

    # Depth errors
    depth_errors = np.abs(y_true[:, 2] - y_pred[:, 2])

    # 3D Euclidean error
    euclidean_3d_errors = np.sqrt(horizontal_errors**2 + depth_errors**2)

    # Calculate absolute coordinate errors if reference is provided
    if reference_coords is not None:
        # Convert predicted and true coordinates to absolute
        abs_true = np.array(
            [
                relative_to_absolute_coordinates(
                    y_true[i, 0], y_true[i, 1], y_true[i, 2], reference_coords
                )
                for i in range(len(y_true))
            ]
        )

        abs_pred = np.array(
            [
                relative_to_absolute_coordinates(
                    y_pred[i, 0], y_pred[i, 1], y_pred[i, 2], reference_coords
                )
                for i in range(len(y_pred))
            ]
        )

        # Calculate haversine distances for absolute coordinates
        abs_horizontal_errors = np.array(
            [
                haversine_distance(
                    abs_true[i, 0], abs_true[i, 1], abs_pred[i, 0], abs_pred[i, 1]
                )
                for i in range(len(abs_true))
            ]
        )

        # Absolute depth errors
        abs_depth_errors = np.abs(abs_true[:, 2] - abs_pred[:, 2])

        # 3D errors using absolute coordinates
        abs_3d_errors = np.sqrt(abs_horizontal_errors**2 + abs_depth_errors**2)

    # Calculate success rates at different thresholds
    thresholds = [5, 10, 15, 20, 50]
    success_rates = {}
    for threshold in thresholds:
        success_rates[f"horizontal_{threshold}km"] = (
            np.mean(horizontal_errors < threshold) * 100
        )
        success_rates[f"depth_{threshold}km"] = np.mean(depth_errors < threshold) * 100
        success_rates[f"3d_{threshold}km"] = (
            np.mean(euclidean_3d_errors < threshold) * 100
        )

    # Combine all metrics
    metrics = {
        "mean_horizontal_error": np.mean(horizontal_errors),
        "median_horizontal_error": np.median(horizontal_errors),
        "mean_depth_error": np.mean(depth_errors),
        "median_depth_error": np.median(depth_errors),
        "mean_3d_error": np.mean(euclidean_3d_errors),
        "median_3d_error": np.median(euclidean_3d_errors),
        **success_rates,
    }

    # Add absolute metrics if available
    if reference_coords is not None:
        abs_metrics = {
            "abs_mean_horizontal_error": np.mean(abs_horizontal_errors),
            "abs_median_horizontal_error": np.median(abs_horizontal_errors),
            "abs_mean_depth_error": np.mean(abs_depth_errors),
            "abs_median_depth_error": np.median(abs_depth_errors),
            "abs_mean_3d_error": np.mean(abs_3d_errors),
            "abs_median_3d_error": np.median(abs_3d_errors),
        }
        metrics.update(abs_metrics)

        # Return both relative and absolute errors
        return (
            metrics,
            (horizontal_errors, depth_errors, euclidean_3d_errors),
            (abs_horizontal_errors, abs_depth_errors, abs_3d_errors),
        )

    # Return only relative errors if no reference provided
    return metrics, horizontal_errors, depth_errors, euclidean_3d_errors


class RelativeGNNModule(torch.nn.Module):
    """
    Enhanced Graph Neural Network for relative aftershock prediction
    """

    def __init__(
        self,
        num_node_features,
        edge_dim=5,  # Increased edge dimension with directional features
        hidden_dim=64,
        output_dim=3,
        num_layers=3,
        gnn_type="gat",
        dropout=0.3,
    ):
        super(RelativeGNNModule, self).__init__()

        # Select GNN layer type
        if gnn_type == "gcn":
            self.conv1 = GCNConv(num_node_features, hidden_dim)
            self.convs = torch.nn.ModuleList(
                [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
            )
        elif gnn_type == "graph":
            self.conv1 = GraphConv(num_node_features, hidden_dim, aggr="mean")
            self.convs = torch.nn.ModuleList(
                [
                    GraphConv(hidden_dim, hidden_dim, aggr="mean")
                    for _ in range(num_layers - 1)
                ]
            )
        elif gnn_type == "gat":
            # For GAT, we'll use the edge attributes in a custom way
            self.conv1 = GATConv(
                num_node_features, hidden_dim, heads=4, concat=False, dropout=dropout
            )
            self.convs = torch.nn.ModuleList(
                [
                    GATConv(
                        hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout
                    )
                    for _ in range(num_layers - 1)
                ]
            )
            # Edge feature processing for GAT
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        elif gnn_type == "sage":
            self.conv1 = SAGEConv(num_node_features, hidden_dim)
            self.convs = torch.nn.ModuleList(
                [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Additional directional convolution layer for more explicit relative positioning
        self.dir_conv = nn.Linear(
            hidden_dim + 3, hidden_dim
        )  # +3 for relative coordinates

        # Prediction layers
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

        # Normalization layers
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Store GNN type
        self.gnn_type = gnn_type

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Process edge attributes if available
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

        # Initial layer with appropriate edge attributes
        if self.gnn_type == "gat" and edge_attr is not None:
            # For GAT, use edge features to modulate attention
            edge_weight = self.edge_encoder(edge_attr).view(-1)
            x = self.conv1(x, edge_index, edge_weight)
        elif self.gnn_type == "graph" and edge_attr is not None:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            # For GCN and SAGE which don't use edge_attr by default
            x = self.conv1(x, edge_index)

        x = F.elu(x)
        x = self.dropout(x)

        # Hidden layers
        for conv in self.convs:
            if self.gnn_type == "gat" and edge_attr is not None:
                # Update edge weights for each layer
                edge_weight = self.edge_encoder(edge_attr).view(-1)
                x_new = conv(x, edge_index, edge_weight)
            elif self.gnn_type == "graph" and edge_attr is not None:
                x_new = conv(x, edge_index, edge_attr)
            else:
                x_new = conv(x, edge_index)

            # Residual connection
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new

            x = F.elu(x)
            x = self.dropout(x)

        # Add relative coordinate information to node embeddings
        # This helps the model learn position-aware representations
        rel_coords = data.x[:, -3:]  # Last 3 features are relative coordinates
        x_with_coords = torch.cat([x, rel_coords], dim=1)
        x = F.elu(self.dir_conv(x_with_coords))

        # Normalization (layer norm works better for graph data)
        x = self.ln(x)

        # Final prediction layers
        x = F.elu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return x


class RelativeGNNAftershockPredictor:
    """
    Wrapper class for training and evaluating the relative coordinate GNN model
    """

    def __init__(
        self,
        graph_data_list,
        reference_coords,
        gnn_type="gat",
        hidden_dim=64,
        num_layers=3,
        learning_rate=0.001,
        batch_size=32,
        weight_decay=1e-5,
    ):
        self.graph_data_list = graph_data_list
        self.reference_coords = reference_coords
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # Split data into train/val/test
        self.train_data, self.val_data, self.test_data = self._train_val_test_split()

        # Setup model
        if len(self.graph_data_list) > 0:
            # Get feature dimensions from the first graph
            num_features = self.graph_data_list[0].x.shape[1]
            edge_dim = (
                self.graph_data_list[0].edge_attr.shape[1]
                if hasattr(self.graph_data_list[0], "edge_attr")
                else 0
            )

            # Initialize model
            self.model = RelativeGNNModule(
                num_node_features=num_features,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                output_dim=3,  # [ew_km, ns_km, depth_rel]
                num_layers=num_layers,
                gnn_type=gnn_type,
            )

            # Loss and optimizer
            self.criterion = relative_haversine_loss
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

            # Learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )

            # Setup data loaders
            self.train_loader = PyGDataLoader(
                self.train_data, batch_size=batch_size, shuffle=True
            )
            self.val_loader = PyGDataLoader(self.val_data, batch_size=batch_size)
            self.test_loader = PyGDataLoader(self.test_data, batch_size=batch_size)

    def _train_val_test_split(self, val_ratio=0.15, test_ratio=0.15):
        """Split graph data into train/val/test sets"""
        n = len(self.graph_data_list)
        indices = list(range(n))

        # Use a deterministic split for reproducibility
        np.random.seed(42)
        np.random.shuffle(indices)

        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)

        test_indices = indices[:test_size]
        val_indices = indices[test_size : test_size + val_size]
        train_indices = indices[test_size + val_size :]

        train_data = [self.graph_data_list[i] for i in train_indices]
        val_data = [self.graph_data_list[i] for i in val_indices]
        test_data = [self.graph_data_list[i] for i in test_indices]

        print(
            f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} graphs"
        )
        return train_data, val_data, test_data

    def train_epoch(self):
        """Train for one epoch with relative coordinates"""
        self.model.train()
        epoch_loss = 0
        num_graphs = 0

        for data in self.train_loader:
            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(data)

            # Get device
            device = out.device

            # Get batch information
            batch_size = 1 if not hasattr(data, "batch") else data.num_graphs
            num_graphs += batch_size

            # Initialize lists for predictions and targets
            predictions = []
            targets = []

            # Process each graph in the batch
            for i in range(batch_size):
                if hasattr(data, "batch"):
                    # Get nodes for this graph
                    graph_mask = data.batch == i
                    graph_indices = torch.where(graph_mask)[0]

                    # Get target node index
                    if hasattr(data, "target_idx"):
                        target_idx_in_graph = data.target_idx[i].item()
                        # Get global index in the batch
                        target_idx = graph_indices[target_idx_in_graph]
                    else:
                        # Default to last node
                        target_idx = graph_indices[-1]

                    # Get prediction for target node
                    pred = out[target_idx]
                    predictions.append(pred)

                    # Get target value for this target node
                    if isinstance(data.y, list):
                        # data.y is a list of numpy arrays, one per graph
                        # Get the array for this graph
                        graph_data = data.y[i]

                        # The target is specifically for the target node (often the last one)
                        # So we need to find that specific node's coordinates
                        target_coords = torch.tensor(
                            graph_data[target_idx_in_graph],
                            dtype=torch.float,
                            device=device,
                        )
                        targets.append(target_coords)
                    else:
                        target_coords = data.y[target_idx]
                        targets.append(target_coords)
                else:
                    # For a single graph (not batched)
                    if hasattr(data, "target_idx"):
                        target_idx = data.target_idx.item()
                    else:
                        target_idx = data.num_nodes - 1

                    pred = out[target_idx]
                    predictions.append(pred)

                    # Get target coordinates
                    if isinstance(data.y, list):
                        target_coords = torch.tensor(
                            data.y[0][target_idx], dtype=torch.float, device=device
                        )
                    else:
                        target_coords = data.y[target_idx]

                    targets.append(target_coords)

            # Stack predictions and targets
            pred_tensor = torch.stack(predictions)
            target_tensor = torch.stack(targets)

            # Calculate loss
            loss = self.criterion(pred_tensor, target_tensor)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * batch_size

        return epoch_loss / num_graphs

    def validate(self):
        """Validate model with relative coordinates"""
        self.model.eval()
        val_loss = 0
        num_graphs = 0

        with torch.no_grad():
            for data in self.val_loader:
                # Forward pass
                out = self.model(data)

                # Get device
                device = out.device

                # Get batch information
                batch_size = 1 if not hasattr(data, "batch") else data.num_graphs
                num_graphs += batch_size

                # Initialize lists for predictions and targets
                predictions = []
                targets = []

                # Process each graph in the batch
                for i in range(batch_size):
                    if hasattr(data, "batch"):
                        # Get nodes for this graph
                        graph_mask = data.batch == i
                        graph_indices = torch.where(graph_mask)[0]

                        # Get target node index
                        if hasattr(data, "target_idx"):
                            target_idx_in_graph = data.target_idx[i].item()
                            # Get global index in the batch
                            target_idx = graph_indices[target_idx_in_graph]
                        else:
                            # Default to last node
                            target_idx = graph_indices[-1]

                        # Get prediction for target node
                        pred = out[target_idx]
                        predictions.append(pred)

                        # Get target value for this target node
                        if isinstance(data.y, list):
                            # Get coordinates for the target node in this graph
                            graph_data = data.y[i]
                            target_coords = torch.tensor(
                                graph_data[target_idx_in_graph],
                                dtype=torch.float,
                                device=device,
                            )
                            targets.append(target_coords)
                        else:
                            target_coords = data.y[target_idx]
                            targets.append(target_coords)
                    else:
                        # For a single graph (not batched)
                        if hasattr(data, "target_idx"):
                            target_idx = data.target_idx.item()
                        else:
                            target_idx = data.num_nodes - 1

                        pred = out[target_idx]
                        predictions.append(pred)

                        # Get target coordinates
                        if isinstance(data.y, list):
                            target_coords = torch.tensor(
                                data.y[0][target_idx], dtype=torch.float, device=device
                            )
                        else:
                            target_coords = data.y[target_idx]

                        targets.append(target_coords)

                # Stack predictions and targets
                pred_tensor = torch.stack(predictions)
                target_tensor = torch.stack(targets)

                # Calculate loss
                loss = self.criterion(pred_tensor, target_tensor)
                val_loss += loss.item() * batch_size

        return val_loss / num_graphs

    def test(self):
        """Test model and return predictions with relative coordinates"""
        self.model.eval()
        all_preds = []
        all_targets = []
        num_graphs = 0

        with torch.no_grad():
            for data in self.test_loader:
                # Forward pass
                out = self.model(data)

                # Get device
                device = out.device

                # Get batch information
                batch_size = 1 if not hasattr(data, "batch") else data.num_graphs
                num_graphs += batch_size

                # Process each graph in the batch
                for i in range(batch_size):
                    if hasattr(data, "batch"):
                        # Get nodes for this graph
                        graph_mask = data.batch == i
                        graph_indices = torch.where(graph_mask)[0]

                        # Get target node index
                        if hasattr(data, "target_idx"):
                            target_idx_in_graph = data.target_idx[i].item()
                            # Get global index in the batch
                            target_idx = graph_indices[target_idx_in_graph]
                        else:
                            # Default to last node
                            target_idx = graph_indices[-1]

                        # Get prediction and target
                        pred = out[target_idx].cpu().numpy()

                        if isinstance(data.y, list):
                            # Get coordinates for the target node
                            target = data.y[i][target_idx_in_graph]
                        else:
                            target = data.y[target_idx].cpu().numpy()

                        all_preds.append(pred)
                        all_targets.append(target)
                    else:
                        # For a single graph (not batched)
                        if hasattr(data, "target_idx"):
                            target_idx = data.target_idx.item()
                        else:
                            target_idx = data.num_nodes - 1

                        pred = out[target_idx].cpu().numpy()

                        if isinstance(data.y, list):
                            target = data.y[0][target_idx]
                        else:
                            target = data.y[target_idx].cpu().numpy()

                        all_preds.append(pred)
                        all_targets.append(target)

        # Concatenate results
        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)

        # Calculate errors - both in relative coordinates and absolute if reference is provided
        if self.reference_coords:
            # Calculate errors in both relative and absolute coordinates
            metrics, rel_errors, abs_errors = calculate_prediction_errors_relative(
                y_true, y_pred, self.reference_coords
            )
            return metrics, y_true, y_pred, (rel_errors, abs_errors)
        else:
            # Only relative coordinate errors
            metrics, horizontal_errors, depth_errors, euclidean_3d_errors = (
                calculate_prediction_errors_relative(y_true, y_pred)
            )
            return (
                metrics,
                y_true,
                y_pred,
                (horizontal_errors, depth_errors, euclidean_3d_errors),
            )

    def train(self, num_epochs=100, patience=10):
        """Train the model with early stopping"""
        print(
            f"Training Relative GNN model ({self.gnn_type}) for {num_epochs} epochs..."
        )

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Initialize tracking variables
        best_val_loss = float("inf")
        best_epoch = 0
        no_improve = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0

                # Save best model
                torch.save(
                    self.model.state_dict(), f"best_rel_gnn_model_{self.gnn_type}.pt"
                )
            else:
                no_improve += 1

            # Update learning rate
            self.scheduler.step(val_loss)

            # Print progress
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(torch.load(f"best_rel_gnn_model_{self.gnn_type}.pt"))

        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.axvline(
            x=best_epoch,
            color="r",
            linestyle="--",
            label=f"Best Epoch ({best_epoch+1})",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss - Relative {self.gnn_type.upper()}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"results/learning_curve_relative_{self.gnn_type}.png",
            dpi=300,
            bbox_inches="tight",
        )

        return train_losses, val_losses

    def debug_data_structure(self):
        """Analyze the first batch to debug data structures"""
        print("\n===== Debugging Relative Coordinate Data Structure =====")

        # Get first batch
        for data in self.train_loader:
            # Basic info
            print(f"Data object type: {type(data)}")
            print(f"Available attributes: {data.keys}")

            # Check node features
            print(f"\nNode features (x):")
            print(f"  Type: {type(data.x)}")
            print(f"  Shape: {data.x.shape}")

            # Display first few rows to check relative coordinates
            print(
                f"  Last 3 columns (should be rel coords): \n{data.x[0, -3:].cpu().numpy()}"
            )

            # Check edge index
            print(f"\nEdge index:")
            print(f"  Type: {type(data.edge_index)}")
            print(f"  Shape: {data.edge_index.shape}")

            # Check edge attributes if available
            if hasattr(data, "edge_attr"):
                print(f"\nEdge attributes:")
                print(f"  Type: {type(data.edge_attr)}")
                print(f"  Shape: {data.edge_attr.shape}")
                print(f"  First row: {data.edge_attr[0].cpu().numpy()}")

            # Check target values
            print(f"\nTarget values (y):")
            print(f"  Type: {type(data.y)}")
            if isinstance(data.y, torch.Tensor):
                print(f"  Shape: {data.y.shape}")
                print(f"  Sample: {data.y[0].cpu().numpy()}")
            elif isinstance(data.y, list):
                print(f"  Length: {len(data.y)}")
                print(f"  First element type: {type(data.y[0])}")
                print(f"  First element: {data.y[0]}")

            # Check batch assignment if batched
            if hasattr(data, "batch"):
                print(f"\nBatch assignment:")
                print(f"  Type: {type(data.batch)}")
                print(f"  Shape: {data.batch.shape}")
                print(f"  Unique values: {torch.unique(data.batch).cpu().numpy()}")
                print(f"  Number of graphs: {data.num_graphs}")

            # Check target indices if available
            if hasattr(data, "target_idx"):
                print(f"\nTarget indices:")
                print(f"  Type: {type(data.target_idx)}")
                print(f"  Shape: {data.target_idx.shape}")
                print(f"  Value: {data.target_idx.cpu().numpy()}")

            # Check mask if available
            if hasattr(data, "mask"):
                print(f"\nMask:")
                print(f"  Type: {type(data.mask)}")
                print(f"  Shape: {data.mask.shape}")
                print(f"  Sum (number of True values): {data.mask.sum().item()}")

            # Only process the first batch
            break

        print("\n=================================")

        # Return to allow chaining
        return self


def plot_relative_results(
    y_true, y_pred, errors, reference_coords=None, model_name="RelativeGNN"
):
    """
    Create visualizations of prediction results for relative coordinates.

    Args:
        y_true: True relative coordinates [ew_km, ns_km, depth_rel]
        y_pred: Predicted relative coordinates [ew_km, ns_km, depth_rel]
        errors: Tuple containing error metrics
        reference_coords: Dictionary with reference coordinates for conversion to absolute
        model_name: Name to use in plot titles and filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Unpack error metrics
    if len(errors) == 2 and reference_coords is not None:
        # We have both relative and absolute errors
        rel_errors, abs_errors = errors
        horizontal_errors, depth_errors, euclidean_3d_errors = rel_errors
        abs_horizontal_errors, abs_depth_errors, abs_3d_errors = abs_errors

        # Convert to absolute coordinates for plotting
        abs_true = np.array(
            [
                relative_to_absolute_coordinates(
                    y_true[i, 0], y_true[i, 1], y_true[i, 2], reference_coords
                )
                for i in range(len(y_true))
            ]
        )

        abs_pred = np.array(
            [
                relative_to_absolute_coordinates(
                    y_pred[i, 0], y_pred[i, 1], y_pred[i, 2], reference_coords
                )
                for i in range(len(y_pred))
            ]
        )

        # Plot in both coordinate systems
        plot_relative = True
        plot_absolute = True
    else:
        # We only have relative errors
        horizontal_errors, depth_errors, euclidean_3d_errors = errors
        plot_relative = True
        plot_absolute = False

    # Plot 1: Map of relative coordinates (East-West vs North-South)
    if plot_relative:
        plt.figure(figsize=(12, 10))

        # Plot true relative positions
        plt.scatter(y_true[:, 0], y_true[:, 1], c="blue", s=30, alpha=0.6, label="True")

        # Plot predicted relative positions
        plt.scatter(
            y_pred[:, 0], y_pred[:, 1], c="red", s=30, alpha=0.6, label="Predicted"
        )

        # Draw lines connecting true and predicted points
        for i in range(len(y_true)):
            plt.plot(
                [y_true[i, 0], y_pred[i, 0]],
                [y_true[i, 1], y_pred[i, 1]],
                "k-",
                alpha=0.15,
            )

        plt.xlabel("East-West Distance (km)")
        plt.ylabel("North-South Distance (km)")
        plt.title(f"True vs Predicted Relative Locations - {model_name}")
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.savefig(
            f"results/relative_location_map_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

    # Plot 2: Map in absolute coordinates (if available)
    if plot_absolute:
        plt.figure(figsize=(12, 10))

        # Plot true absolute locations
        plt.scatter(
            abs_true[:, 1], abs_true[:, 0], c="blue", s=30, alpha=0.6, label="True"
        )

        # Plot predicted absolute locations
        plt.scatter(
            abs_pred[:, 1], abs_pred[:, 0], c="red", s=30, alpha=0.6, label="Predicted"
        )

        # Draw lines connecting true and predicted points
        for i in range(len(abs_true)):
            plt.plot(
                [abs_true[i, 1], abs_pred[i, 1]],
                [abs_true[i, 0], abs_pred[i, 0]],
                "k-",
                alpha=0.15,
            )

        # Plot reference point (mainshock)
        plt.scatter(
            [reference_coords["longitude"]],
            [reference_coords["latitude"]],
            c="green",
            s=100,
            marker="*",
            label="Reference (Mainshock)",
        )

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"True vs Predicted Absolute Locations - {model_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"results/absolute_location_map_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

    # Plot 3: Error distributions
    plt.figure(figsize=(15, 12))

    # Relative coordinate errors
    plt.subplot(3, 2, 1)
    plt.hist(horizontal_errors, bins=30, color="skyblue", edgecolor="black")
    plt.axvline(
        np.mean(horizontal_errors),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(horizontal_errors):.2f} km",
    )
    plt.axvline(
        np.median(horizontal_errors),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(horizontal_errors):.2f} km",
    )
    plt.xlabel("Relative Horizontal Error (km)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.hist(depth_errors, bins=30, color="lightgreen", edgecolor="black")
    plt.axvline(
        np.mean(depth_errors),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(depth_errors):.2f} km",
    )
    plt.axvline(
        np.median(depth_errors),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(depth_errors):.2f} km",
    )
    plt.xlabel("Relative Depth Error (km)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.hist(euclidean_3d_errors, bins=30, color="salmon", edgecolor="black")
    plt.axvline(
        np.mean(euclidean_3d_errors),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(euclidean_3d_errors):.2f} km",
    )
    plt.axvline(
        np.median(euclidean_3d_errors),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(euclidean_3d_errors):.2f} km",
    )
    plt.xlabel("Relative 3D Error (km)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    # Absolute coordinate errors (if available)
    if plot_absolute:
        plt.subplot(3, 2, 4)
        plt.hist(abs_horizontal_errors, bins=30, color="skyblue", edgecolor="black")
        plt.axvline(
            np.mean(abs_horizontal_errors),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(abs_horizontal_errors):.2f} km",
        )
        plt.axvline(
            np.median(abs_horizontal_errors),
            color="green",
            linestyle="--",
            label=f"Median: {np.median(abs_horizontal_errors):.2f} km",
        )
        plt.xlabel("Absolute Horizontal Error (km)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.hist(abs_depth_errors, bins=30, color="lightgreen", edgecolor="black")
        plt.axvline(
            np.mean(abs_depth_errors),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(abs_depth_errors):.2f} km",
        )
        plt.axvline(
            np.median(abs_depth_errors),
            color="green",
            linestyle="--",
            label=f"Median: {np.median(abs_depth_errors):.2f} km",
        )
        plt.xlabel("Absolute Depth Error (km)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 2, 6)
        plt.hist(abs_3d_errors, bins=30, color="salmon", edgecolor="black")
        plt.axvline(
            np.mean(abs_3d_errors),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(abs_3d_errors):.2f} km",
        )
        plt.axvline(
            np.median(abs_3d_errors),
            color="green",
            linestyle="--",
            label=f"Median: {np.median(abs_3d_errors):.2f} km",
        )
        plt.xlabel("Absolute 3D Error (km)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        f"results/error_distribution_{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Plot 4: Depth errors
    plt.figure(figsize=(10, 8))

    if plot_absolute:
        # Plot true vs predicted depths
        plt.scatter(abs_true[:, 2], abs_pred[:, 2], c="blue", alpha=0.6)
        plt.plot(
            [min(abs_true[:, 2]), max(abs_true[:, 2])],
            [min(abs_true[:, 2]), max(abs_true[:, 2])],
            "r--",
        )
        plt.xlabel("True Depth (km)")
        plt.ylabel("Predicted Depth (km)")
    else:
        # Plot relative depths
        plt.scatter(y_true[:, 2], y_pred[:, 2], c="blue", alpha=0.6)
        plt.plot(
            [min(y_true[:, 2]), max(y_true[:, 2])],
            [min(y_true[:, 2]), max(y_true[:, 2])],
            "r--",
        )
        plt.xlabel("True Relative Depth (km)")
        plt.ylabel("Predicted Relative Depth (km)")

    plt.title(f"Depth Prediction Performance - {model_name}")
    plt.grid(True)
    plt.savefig(
        f"results/depth_prediction_{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Plot 5: Position error vs distance from reference (mainshock)
    if plot_absolute:
        # Calculate distance from each event to reference point
        distances_from_ref = np.array(
            [
                haversine_distance(
                    abs_true[i, 0],
                    abs_true[i, 1],
                    reference_coords["latitude"],
                    reference_coords["longitude"],
                )
                for i in range(len(abs_true))
            ]
        )

        plt.figure(figsize=(10, 8))
        plt.scatter(distances_from_ref, abs_horizontal_errors, alpha=0.6)

        # Add trend line
        z = np.polyfit(distances_from_ref, abs_horizontal_errors, 1)
        p = np.poly1d(z)
        plt.plot(
            distances_from_ref,
            p(distances_from_ref),
            "r--",
            label=f"Trend: y={z[0]:.4f}x+{z[1]:.2f}",
        )

        plt.xlabel("Distance from Mainshock (km)")
        plt.ylabel("Horizontal Prediction Error (km)")
        plt.title("Prediction Error vs. Distance from Mainshock")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"results/error_vs_distance_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )


def main():
    """
    Main function to run the relative coordinate GNN aftershock prediction pipeline.
    """
    print("Starting Relative Coordinate GNN Aftershock Prediction...")

    # Check if data exists
    if not os.path.exists("aftershock_data.pkl"):
        print("Error: Data file 'aftershock_data.pkl' not found.")
        return

    # Read data
    print("Reading data from pickle file...")
    df = read_data_from_pickle("aftershock_data.pkl")

    print(df.head())

    # Display data info
    print(f"Dataset loaded. Total events: {len(df)}")

    # Create spatiotemporal graphs with relative coordinates
    graph_data_list, reference_coords = create_relative_spatiotemporal_graph(
        df, time_window=168, spatial_threshold=100, min_connections=1  # 1 week
    )

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train models with different GNN types
    gnn_types = ["gat", "sage"]  # Focus on the best performing architectures
    results = {}

    for gnn_type in gnn_types:
        print(f"\n===== Training Relative {gnn_type.upper()} model =====")

        # Initialize GNN predictor
        predictor = RelativeGNNAftershockPredictor(
            graph_data_list=graph_data_list,
            reference_coords=reference_coords,
            gnn_type=gnn_type,
            hidden_dim=128,
            num_layers=4,
            learning_rate=0.001,
            batch_size=16,
            weight_decay=1e-5,
        )

        # Debug data structure - useful for troubleshooting
        predictor.debug_data_structure()

        # Train model
        predictor.train(num_epochs=75, patience=15)

        # Evaluate on test set
        print(f"Evaluating relative {gnn_type.upper()} model on test set...")
        metrics, y_true, y_pred, errors = predictor.test()

        # Print metrics
        print(f"\n{gnn_type.upper()} Relative Prediction Error Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Plot results
        plot_relative_results(
            y_true,
            y_pred,
            errors,
            reference_coords=reference_coords,
            model_name=f"Relative_{gnn_type.upper()}",
        )

        # Store results
        results[gnn_type] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "errors": errors,
        }

    # Compare different GNN models
    plt.figure(figsize=(12, 8))

    metrics_to_plot = ["mean_horizontal_error", "mean_depth_error", "mean_3d_error"]
    if "abs_mean_horizontal_error" in list(results.values())[0]["metrics"]:
        metrics_to_plot.extend(
            ["abs_mean_horizontal_error", "abs_mean_depth_error", "abs_mean_3d_error"]
        )

    x = np.arange(len(metrics_to_plot))
    width = 0.2

    for i, gnn_type in enumerate(gnn_types):
        values = [results[gnn_type]["metrics"][m] for m in metrics_to_plot]
        plt.bar(x + i * width, values, width, label=gnn_type.upper())

    plt.xlabel("Metric")
    plt.ylabel("Error (km)")
    plt.title("Comparison of Different Relative GNN Models")
    plt.xticks(x + width, metrics_to_plot, rotation=45)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(
        f"results/relative_gnn_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Compare with previous absolute coordinate results if available
    try:
        # Load previous absolute results or compare directly if you have them
        abs_results = pickle.load(open("absolute_results.pkl", "rb"))

        # Compare relative vs absolute approaches
        plt.figure(figsize=(12, 8))

        # Common metrics for both approaches
        common_metrics = ["mean_horizontal_error", "mean_depth_error", "mean_3d_error"]
        x = np.arange(len(common_metrics))
        width = 0.15

        # Plot bars for each model
        for i, abs_model in enumerate(abs_results.keys()):
            values = [abs_results[abs_model]["metrics"][m] for m in common_metrics]
            plt.bar(x + i * width, values, width, label=f"Abs {abs_model.upper()}")

        for i, rel_model in enumerate(results.keys()):
            values = [results[rel_model]["metrics"][f"abs_{m}"] for m in common_metrics]
            plt.bar(
                x + (i + len(abs_results)) * width,
                values,
                width,
                label=f"Rel {rel_model.upper()}",
            )

        plt.xlabel("Metric")
        plt.ylabel("Error (km)")
        plt.title("Absolute vs Relative Coordinate GNN Approaches")
        plt.xticks(
            x + width * (len(abs_results) + len(results) - 1) / 2, common_metrics
        )
        plt.legend()
        plt.grid(True, axis="y")
        plt.savefig(
            f"results/absolute_vs_relative_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
    except Exception as e:
        print(f"Could not compare with absolute results: {e}")

    print("Relative GNN prediction complete. Results saved in the 'results' directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
