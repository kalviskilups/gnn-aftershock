# gnn_approach.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os
import time
from datetime import datetime

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
import scipy


def read_data_from_pickle(file_path):
    """
    Load the data from pickle file.
    """
    # Load the pickle file that contains the data dictionary
    with open(file_path, "rb") as file:
        data_dict = pickle.load(file)

    # Extract the metadata from each event entry.
    data_list = [
        {**entry["metadata"], "waveform": entry["waveform"]}
        for entry in data_dict.values()
    ]
    # Convert the list of metadata dictionaries into a DataFrame
    df = pd.DataFrame(data_list)

    return df


def haversine_loss(y_pred, y_true):
    """
    Custom loss function based on Haversine distance for lat/lon and depth.

    Args:
        y_pred: tensor of shape (batch_size, 3) [lat, lon, depth]
        y_true: tensor of shape (batch_size, 3) [lat, lon, depth]
    """
    # Convert to radians
    lat1, lon1 = torch.deg2rad(y_pred[:, 0]), torch.deg2rad(y_pred[:, 1])
    lat2, lon2 = torch.deg2rad(y_true[:, 0]), torch.deg2rad(y_true[:, 1])

    # Haversine formula for horizontal distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.asin(torch.sqrt(a))
    r = 6371.0  # Radius of Earth in kilometers

    # Horizontal distance
    h_dist = c * r

    # Depth error (in km)
    d_dist = torch.abs(y_pred[:, 2] - y_true[:, 2])

    # Total 3D distance (Euclidean approximation)
    dist_3d = torch.sqrt(h_dist**2 + d_dist**2)

    # Return mean distance
    return torch.mean(dist_3d)


def debug_graph_structure(graph_data_list, num_samples=3):
    """
    Debug and visualize the created graph structure.

    Args:
        graph_data_list: List of PyTorch Geometric Data objects
        num_samples: Number of sample graphs to analyze
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx

    print(f"\n===== Debugging Graph Structure =====")

    # Overall statistics
    num_graphs = len(graph_data_list)
    nodes_per_graph = [g.num_nodes for g in graph_data_list]
    edges_per_graph = [g.edge_index.shape[1] for g in graph_data_list]

    print(f"Total graphs: {num_graphs}")
    print(
        f"Nodes per graph: min={min(nodes_per_graph)}, max={max(nodes_per_graph)}, avg={sum(nodes_per_graph)/num_graphs:.2f}"
    )
    print(
        f"Edges per graph: min={min(edges_per_graph)}, max={max(edges_per_graph)}, avg={sum(edges_per_graph)/num_graphs:.2f}"
    )

    # Checking for isolated nodes
    isolated_nodes = []
    for i, g in enumerate(graph_data_list):
        edge_set = set(g.edge_index[0].tolist() + g.edge_index[1].tolist())
        all_nodes = set(range(g.num_nodes))
        isolated = all_nodes - edge_set
        if isolated:
            isolated_nodes.append((i, isolated))

    print(f"Graphs with isolated nodes: {len(isolated_nodes)}/{num_graphs}")

    # Examining graph connectivity
    for i, g in enumerate(graph_data_list[:num_samples]):
        print(f"\nGraph {i} details:")
        print(f"  Number of nodes: {g.num_nodes}")
        print(f"  Number of edges: {g.edge_index.shape[1]}")
        print(f"  Node feature dimensions: {g.x.shape}")
        print(
            f"  Edge attribute dimensions: {g.edge_attr.shape if hasattr(g, 'edge_attr') else 'None'}"
        )

        # Check target node
        target_idx = g.num_nodes - 1  # Last node is target
        target_node_connections = 0
        for j in range(g.edge_index.shape[1]):
            if (
                g.edge_index[0, j].item() == target_idx
                or g.edge_index[1, j].item() == target_idx
            ):
                target_node_connections += 1

        print(
            f"  Target node (idx {target_idx}) has {target_node_connections} connections"
        )

        # Visualize graph
        plt.figure(figsize=(10, 8))
        G = to_networkx(g, to_undirected=True)
        pos = nx.spring_layout(G, seed=42)  # For consistent layout

        # Get node colors (target node highlighted)
        node_colors = ["red" if n == target_idx else "skyblue" for n in G.nodes()]

        # Draw the graph
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_size=10,
            font_weight="bold",
        )
        plt.title(f"Graph {i} Structure")
        plt.savefig(f"results/graph_structure_{i}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Check for potential issues
    potential_issues = []

    # 1. Check for very small graphs
    small_graphs = sum(1 for n in nodes_per_graph if n < 3)
    if small_graphs > 0:
        potential_issues.append(f"Found {small_graphs} graphs with fewer than 3 nodes")

    # 2. Check for very sparse graphs
    sparse_graphs = sum(
        1
        for i, g in enumerate(graph_data_list)
        if g.edge_index.shape[1] < g.num_nodes * 0.5
    )
    if sparse_graphs > num_graphs * 0.3:  # If more than 30% of graphs are sparse
        potential_issues.append(
            f"Found {sparse_graphs} sparse graphs (low edge density)"
        )

    # 3. Check for disconnected target nodes
    disconnected_targets = 0
    for g in graph_data_list:
        target_idx = g.num_nodes - 1
        connected = False
        for j in range(g.edge_index.shape[1]):
            if (
                g.edge_index[0, j].item() == target_idx
                or g.edge_index[1, j].item() == target_idx
            ):
                connected = True
                break
        if not connected:
            disconnected_targets += 1

    if disconnected_targets > 0:
        potential_issues.append(
            f"Found {disconnected_targets} graphs with disconnected target nodes"
        )

    # Print potential issues
    if potential_issues:
        print("\nPotential issues detected:")
        for issue in potential_issues:
            print(f"- {issue}")
    else:
        print("\nNo obvious structural issues detected in the graphs")


def extract_waveform_features(waveform):
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
        # Low, mid, high frequency energy
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


def calculate_prediction_errors(y_true, y_pred):
    """
    Calculate various error metrics for location predictions.
    """
    # Calculate distance errors in km
    horizontal_errors = np.array(
        [
            haversine_distance(y_true[i, 0], y_true[i, 1], y_pred[i, 0], y_pred[i, 1])
            for i in range(len(y_true))
        ]
    )

    # Depth errors in km
    depth_errors = np.abs(y_true[:, 2] - y_pred[:, 2])

    # 3D Euclidean error (approximation)
    euclidean_3d_errors = np.sqrt(horizontal_errors**2 + depth_errors**2)

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

    return metrics, horizontal_errors, depth_errors, euclidean_3d_errors


def create_improved_graph(df, time_window=24, spatial_threshold=100, min_connections=1):
    """
    Create spatiotemporal graphs for aftershock prediction with physically meaningful connections.

    Args:
        df: DataFrame with aftershock information
        time_window: Time window in hours to consider events connected
        spatial_threshold: Maximum distance in km between connected events
        min_connections: Minimum number of connections required to create a graph

    Returns:
        List of PyTorch Geometric Data objects
    """
    print(
        f"Creating improved spatiotemporal graphs (time window: {time_window}h, spatial threshold: {spatial_threshold}km)..."
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

    # Extract waveform features
    print("Extracting waveform features for nodes...")
    waveform_features = np.array(
        [extract_waveform_features(w) for w in tqdm(df_sorted["waveform"])]
    )

    # Create a dataframe for node features
    node_df = pd.DataFrame(waveform_features)

    # Add some additional node features
    node_df["time_hours"] = df_sorted["time_hours"].values
    node_df["latitude"] = df_sorted["source_latitude_deg"].values
    node_df["longitude"] = df_sorted["source_longitude_deg"].values
    node_df["depth"] = df_sorted["source_depth_km"].values

    # Scale node features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_df.values)

    # Create target values
    targets = df_sorted[
        ["source_latitude_deg", "source_longitude_deg", "source_depth_km"]
    ].values

    # Create improved spatiotemporal graph data objects
    graph_data_list = []

    # We'll create a graph for each event (except the first one)
    for i in tqdm(range(1, len(df_sorted))):
        # Get current event
        current_time = df_sorted["time_hours"].iloc[i]
        curr_lat = df_sorted["source_latitude_deg"].iloc[i]
        curr_lon = df_sorted["source_longitude_deg"].iloc[i]
        curr_depth = df_sorted["source_depth_km"].iloc[i]

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
                past_lat = df_sorted["source_latitude_deg"].iloc[j]
                past_lon = df_sorted["source_longitude_deg"].iloc[j]
                past_depth = df_sorted["source_depth_km"].iloc[j]

                # Calculate spatial distance
                horizontal_dist = haversine_distance(
                    curr_lat, curr_lon, past_lat, past_lon
                )

                # Calculate 3D distance (approximate)
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
                    # This allows information to flow both ways in the graph
                    edges.append([tgt_idx, src_idx])

                    # Edge attributes
                    temporal_dist = current_time - df_sorted["time_hours"].iloc[j]

                    # Spatial weight (closer = stronger influence)
                    # Use inverse distance for weight calculation
                    spatial_weight = 1.0 / (1.0 + spatial_dist_3d / 10.0)

                    # Temporal weight (more recent = stronger influence)
                    temporal_weight = 1.0 / (1.0 + temporal_dist / 5.0)

                    # Try to approximate stress transfer
                    # (inversely proportional to distance, higher at similar depths)
                    depth_similarity = np.exp(-depth_diff / 10.0)
                    stress_proxy = spatial_weight * depth_similarity

                    # Edge attributes for both directions
                    # Forward direction (past → current)
                    edge_attrs.append(
                        [
                            spatial_weight,
                            temporal_weight,
                            stress_proxy,
                            1.0,  # Direction flag (1 = forward)
                        ]
                    )

                    # Backward direction (current → past)
                    edge_attrs.append(
                        [
                            spatial_weight,
                            temporal_weight,
                            stress_proxy,
                            0.0,  # Direction flag (0 = backward)
                        ]
                    )

            # Only create graph if we have enough connections
            num_connections = (
                len(edges) // 2
            )  # Divide by 2 because we create bidirectional edges

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

                # Only the target node (current event) has a label
                # Create a mask for the target node
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

        # Count connections to target node in the first graph
        target_idx = graph_data_list[0].target_idx.item()
        target_connections = sum(
            1
            for j in range(graph_data_list[0].edge_index.shape[1])
            if graph_data_list[0].edge_index[0, j].item() == target_idx
            or graph_data_list[0].edge_index[1, j].item() == target_idx
        )

        print(f"Target node has {target_connections} connections")

        # Additional diagnostic info
        avg_nodes = sum(g.num_nodes for g in graph_data_list) / len(graph_data_list)
        avg_edges = sum(g.edge_index.shape[1] for g in graph_data_list) / len(
            graph_data_list
        )

        print(f"Average nodes per graph: {avg_nodes:.2f}")
        print(f"Average edges per graph: {avg_edges:.2f}")

        # Check for small graphs
        small_graphs = sum(1 for g in graph_data_list if g.num_nodes < 3)
        if small_graphs > 0:
            print(f"Warning: Found {small_graphs} graphs with fewer than 3 nodes")

    return graph_data_list


class GNNModule(torch.nn.Module):
    """
    Graph Neural Network for aftershock prediction
    """

    def __init__(
        self,
        num_node_features,
        hidden_dim=64,
        output_dim=3,
        num_layers=3,
        gnn_type="gat",
    ):
        super(GNNModule, self).__init__()

        # Select GNN layer type
        if gnn_type == "gcn":
            gnn_layer = GCNConv
        elif gnn_type == "graph":
            gnn_layer = GraphConv
        elif gnn_type == "gat":
            gnn_layer = GATConv
        elif gnn_type == "sage":
            gnn_layer = SAGEConv
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Input layer
        self.conv1 = gnn_layer(num_node_features, hidden_dim)

        # Hidden layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(gnn_layer(hidden_dim, hidden_dim))

        # Output layers
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)

        # Batch normalization
        self.bn = torch.nn.BatchNorm1d(hidden_dim)

        # Dropouts
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If edge attributes are available
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Initial layer with appropriate edge attributes
        if isinstance(self.conv1, GATConv):
            # GAT doesn't use edge_attr in the standard implementation
            x = self.conv1(x, edge_index)
        elif isinstance(self.conv1, SAGEConv):
            # SAGEConv doesn't use edge_attr
            x = self.conv1(x, edge_index)
        elif isinstance(self.conv1, GCNConv):
            # GCNConv doesn't use edge_attr
            x = self.conv1(x, edge_index)
        else:
            # GraphConv and other types can use edge_attr
            if edge_attr is not None:
                x = self.conv1(x, edge_index, edge_attr)
            else:
                x = self.conv1(x, edge_index)
        
        x = F.relu(x)
        x = self.dropout(x)

        # Hidden layers
        for conv in self.convs:
            if isinstance(conv, SAGEConv):
                # SAGEConv doesn't use edge_attr
                x = conv(x, edge_index)
            elif isinstance(conv, GATConv) or isinstance(conv, GCNConv):
                x = conv(x, edge_index)
            else:
                if edge_attr is not None:
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)

            x = F.relu(x)
            x = self.dropout(x)

        # Apply prediction to each node
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return x


class GNNAftershockPredictor:
    """
    Wrapper class for training and evaluating the GNN model
    """

    def __init__(
        self,
        graph_data_list,
        gnn_type="gat",
        hidden_dim=64,
        num_layers=3,
        learning_rate=0.001,
        batch_size=32,
    ):
        self.graph_data_list = graph_data_list
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Split data into train/val/test
        self.train_data, self.val_data, self.test_data = self._train_val_test_split()

        # Setup model
        num_features = self.graph_data_list[0].x.shape[1]
        self.model = GNNModule(
            num_features,
            hidden_dim,
            output_dim=3,
            num_layers=num_layers,
            gnn_type=gnn_type,
        )

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
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
        """Train for one epoch with the improved graph structure"""
        self.model.train()
        epoch_loss = 0
        num_graphs = 0
        
        for data in self.train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data)
            
            # Get devices
            device = out.device
            
            # Get batch information
            batch_size = 1 if not hasattr(data, 'batch') else data.num_graphs
            num_graphs += batch_size
            
            # Initialize lists for predictions and targets
            predictions = []
            targets = []
            
            # Process each graph in the batch
            for i in range(batch_size):
                if hasattr(data, 'batch'):
                    # Get nodes for this graph
                    graph_mask = data.batch == i
                    graph_indices = torch.where(graph_mask)[0]
                    
                    # Get target node index
                    if hasattr(data, 'target_idx'):
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
                        target_coords = torch.tensor(graph_data[target_idx_in_graph], 
                                                dtype=torch.float, device=device)
                        targets.append(target_coords)
                else:
                    # For a single graph (not batched)
                    if hasattr(data, 'target_idx'):
                        target_idx = data.target_idx.item()
                    else:
                        target_idx = data.num_nodes - 1
                    
                    pred = out[target_idx]
                    predictions.append(pred)
                    
                    # Get target coordinates
                    if isinstance(data.y, list):
                        target_coords = torch.tensor(data.y[0][target_idx], 
                                                dtype=torch.float, device=device)
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
        """Validate model with the improved graph structure"""
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
                batch_size = 1 if not hasattr(data, 'batch') else data.num_graphs
                num_graphs += batch_size
                
                # Initialize lists for predictions and targets
                predictions = []
                targets = []
                
                # Process each graph in the batch
                for i in range(batch_size):
                    if hasattr(data, 'batch'):
                        # Get nodes for this graph
                        graph_mask = data.batch == i
                        graph_indices = torch.where(graph_mask)[0]
                        
                        # Get target node index
                        if hasattr(data, 'target_idx'):
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
                            target_coords = torch.tensor(graph_data[target_idx_in_graph], 
                                                    dtype=torch.float, device=device)
                            targets.append(target_coords)
                    else:
                        # For a single graph (not batched)
                        if hasattr(data, 'target_idx'):
                            target_idx = data.target_idx.item()
                        else:
                            target_idx = data.num_nodes - 1
                        
                        pred = out[target_idx]
                        predictions.append(pred)
                        
                        # Get target coordinates
                        if isinstance(data.y, list):
                            target_coords = torch.tensor(data.y[0][target_idx], 
                                                    dtype=torch.float, device=device)
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
        """Test model and return predictions with the improved graph structure"""
        self.model.eval()
        test_loss = 0
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
                batch_size = 1 if not hasattr(data, 'batch') else data.num_graphs
                num_graphs += batch_size
                
                # Process each graph in the batch
                for i in range(batch_size):
                    if hasattr(data, 'batch'):
                        # Get nodes for this graph
                        graph_mask = data.batch == i
                        graph_indices = torch.where(graph_mask)[0]
                        
                        # Get target node index
                        if hasattr(data, 'target_idx'):
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
                        if hasattr(data, 'target_idx'):
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
        
        # Calculate errors
        metrics, horizontal_errors, depth_errors, euclidean_3d_errors = (
            calculate_prediction_errors(y_true, y_pred)
        )
        
        return (
            metrics,
            y_true,
            y_pred,
            (horizontal_errors, depth_errors, euclidean_3d_errors),
        )
    
    def debug_data_structure(self):
        """Analyze the first batch to debug data structures"""
        print("\n===== Debugging Data Structure =====")
        
        # Get first batch
        for data in self.train_loader:
            # Basic info
            print(f"Data object type: {type(data)}")
            print(f"Available attributes: {data.keys}")
            
            # Check node features
            print(f"\nNode features (x):")
            print(f"  Type: {type(data.x)}")
            print(f"  Shape: {data.x.shape}")
            
            # Check edge index
            print(f"\nEdge index:")
            print(f"  Type: {type(data.edge_index)}")
            print(f"  Shape: {data.edge_index.shape}")
            
            # Check edge attributes if available
            if hasattr(data, 'edge_attr'):
                print(f"\nEdge attributes:")
                print(f"  Type: {type(data.edge_attr)}")
                print(f"  Shape: {data.edge_attr.shape}")
            
            # Check target values
            print(f"\nTarget values (y):")
            print(f"  Type: {type(data.y)}")
            if isinstance(data.y, torch.Tensor):
                print(f"  Shape: {data.y.shape}")
                print(f"  Sample: {data.y[0]}")
            elif isinstance(data.y, list):
                print(f"  Length: {len(data.y)}")
                print(f"  First element type: {type(data.y[0])}")
                print(f"  First element: {data.y[0]}")
            
            # Check batch assignment if batched
            if hasattr(data, 'batch'):
                print(f"\nBatch assignment:")
                print(f"  Type: {type(data.batch)}")
                print(f"  Shape: {data.batch.shape}")
                print(f"  Unique values: {torch.unique(data.batch)}")
                print(f"  Number of graphs: {data.num_graphs}")
            
            # Check target indices if available
            if hasattr(data, 'target_idx'):
                print(f"\nTarget indices:")
                print(f"  Type: {type(data.target_idx)}")
                print(f"  Shape or len: {data.target_idx.shape if isinstance(data.target_idx, torch.Tensor) else len(data.target_idx)}")
                print(f"  Value: {data.target_idx}")
            
            # Check mask if available
            if hasattr(data, 'mask'):
                print(f"\nMask:")
                print(f"  Type: {type(data.mask)}")
                print(f"  Shape: {data.mask.shape}")
                print(f"  Sum (number of True values): {data.mask.sum()}")
            
            # Only process the first batch
            break
        
        print("\n=================================")
        
        # Return to allow chaining
        return self

    def train(self, num_epochs=100, patience=10):
        """Train the model with early stopping"""
        print(f"Training GNN model ({self.gnn_type}) for {num_epochs} epochs...")

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
                    self.model.state_dict(), f"best_gnn_model_{self.gnn_type}.pt"
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
        self.model.load_state_dict(torch.load(f"best_gnn_model_{self.gnn_type}.pt"))

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
        plt.title(f"Training and Validation Loss ({self.gnn_type.upper()})")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"results/learning_curve_{self.gnn_type}.png",
            dpi=300,
            bbox_inches="tight",
        )

        return train_losses, val_losses


def plot_results(y_true, y_pred, errors, model_name="GNN"):
    """
    Create visualizations of prediction results.
    """
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Map of true vs predicted locations
    plt.figure(figsize=(12, 10))

    # Plot true test events
    plt.scatter(y_true[:, 1], y_true[:, 0], c="blue", s=30, alpha=0.6, label="True")

    # Plot predicted events
    plt.scatter(y_pred[:, 1], y_pred[:, 0], c="red", s=30, alpha=0.6, label="Predicted")

    # Draw lines connecting true and predicted points
    for i in range(len(y_true)):
        plt.plot(
            [y_true[i, 1], y_pred[i, 1]], [y_true[i, 0], y_pred[i, 0]], "k-", alpha=0.15
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"True vs Predicted Aftershock Locations - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"results/location_map_{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Plot 2: Error distribution
    plt.figure(figsize=(15, 10))

    horizontal_errors, depth_errors, euclidean_3d_errors = errors

    plt.subplot(2, 2, 1)
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
    plt.xlabel("Horizontal Error (km)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
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
    plt.xlabel("Depth Error (km)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
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
    plt.xlabel("3D Error (km)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        f"results/error_distribution_{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )


def main():
    """
    Main function to run the GNN aftershock prediction pipeline.
    """
    print("Starting GNN Aftershock Prediction...")

    # Check if data exists
    if not os.path.exists("aftershock_data.pkl"):
        print("Error: Data file 'aftershock_data.pkl' not found.")
        return

    # Read data
    print("Reading data from pickle file...")
    df = read_data_from_pickle("aftershock_data.pkl")

    print(df)

    # Display data info
    print(f"Dataset loaded. Total events: {len(df)}")

    # Create spatiotemporal graphs
    graph_data_list = create_improved_graph(
        df, time_window=168, spatial_threshold=30, min_connections=1
    )

    debug_graph_structure(graph_data_list, num_samples=3)

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train models with different GNN types
    gnn_types = ["gat", "gcn", "sage"]
    results = {}

    for gnn_type in gnn_types:
        print(f"\n===== Training {gnn_type.upper()} model =====")

        # Initialize GNN predictor
        predictor = GNNAftershockPredictor(
            graph_data_list=graph_data_list,
            gnn_type=gnn_type,
            hidden_dim=128,
            num_layers=4,
            learning_rate=0.001,
            batch_size=16,
        )

        predictor.debug_data_structure()

        # Train model
        predictor.train(num_epochs=75, patience=15)

        # Evaluate on test set
        print(f"Evaluating {gnn_type.upper()} model on test set...")
        metrics, y_true, y_pred, errors = predictor.test()

        # Print metrics
        print(f"\n{gnn_type.upper()} Prediction Error Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Plot results
        plot_results(y_true, y_pred, errors, model_name=f"GNN_{gnn_type.upper()}")

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
    x = np.arange(len(metrics_to_plot))
    width = 0.2

    for i, gnn_type in enumerate(gnn_types):
        values = [results[gnn_type]["metrics"][m] for m in metrics_to_plot]
        plt.bar(x + i * width, values, width, label=gnn_type.upper())

    plt.xlabel("Metric")
    plt.ylabel("Error (km)")
    plt.title("Comparison of Different GNN Models")
    plt.xticks(x + width, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig(
        f"results/gnn_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    print("GNN prediction complete. Results saved in the 'results' directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
