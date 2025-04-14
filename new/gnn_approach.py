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
    print(f"Nodes per graph: min={min(nodes_per_graph)}, max={max(nodes_per_graph)}, avg={sum(nodes_per_graph)/num_graphs:.2f}")
    print(f"Edges per graph: min={min(edges_per_graph)}, max={max(edges_per_graph)}, avg={sum(edges_per_graph)/num_graphs:.2f}")
    
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
        print(f"  Edge attribute dimensions: {g.edge_attr.shape if hasattr(g, 'edge_attr') else 'None'}")
        
        # Check target node
        target_idx = g.num_nodes - 1  # Last node is target
        target_node_connections = 0
        for j in range(g.edge_index.shape[1]):
            if g.edge_index[0, j].item() == target_idx or g.edge_index[1, j].item() == target_idx:
                target_node_connections += 1
        
        print(f"  Target node (idx {target_idx}) has {target_node_connections} connections")
        
        # Visualize graph
        plt.figure(figsize=(10, 8))
        G = to_networkx(g, to_undirected=True)
        pos = nx.spring_layout(G, seed=42)  # For consistent layout
        
        # Get node colors (target node highlighted)
        node_colors = ['red' if n == target_idx else 'skyblue' for n in G.nodes()]
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=500, font_size=10, font_weight='bold')
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
    sparse_graphs = sum(1 for i, g in enumerate(graph_data_list) 
                      if g.edge_index.shape[1] < g.num_nodes * 0.5)
    if sparse_graphs > num_graphs * 0.3:  # If more than 30% of graphs are sparse
        potential_issues.append(f"Found {sparse_graphs} sparse graphs (low edge density)")
    
    # 3. Check for disconnected target nodes
    disconnected_targets = 0
    for g in graph_data_list:
        target_idx = g.num_nodes - 1
        connected = False
        for j in range(g.edge_index.shape[1]):
            if g.edge_index[0, j].item() == target_idx or g.edge_index[1, j].item() == target_idx:
                connected = True
                break
        if not connected:
            disconnected_targets += 1
    
    if disconnected_targets > 0:
        potential_issues.append(f"Found {disconnected_targets} graphs with disconnected target nodes")
    
    # Print potential issues
    if potential_issues:
        print("\nPotential issues detected:")
        for issue in potential_issues:
            print(f"- {issue}")
    else:
        print("\nNo obvious structural issues detected in the graphs")


def extract_waveform_features(waveform):
    """
    Extract enhanced features from a seismic waveform.
    """
    features = []
    for component in range(waveform.shape[0]):
        signal = waveform[component]

        # Time domain features
        features.extend(
            [
                np.max(np.abs(signal)),  # Peak amplitude
                np.mean(np.abs(signal)),  # Mean amplitude
                np.std(signal),  # Standard deviation
                np.percentile(signal, 25),  # 25th percentile
                np.percentile(signal, 75),  # 75th percentile
                np.median(signal),  # Median
                np.mean(np.diff(signal) ** 2),  # Mean squared difference
                np.max(signal) - np.min(signal),  # Range
                np.sum(np.abs(signal)),  # Total energy
                np.mean(np.abs(np.diff(signal))),  # Mean absolute change
                # Zero crossing rate
                np.sum(np.abs(np.diff(np.signbit(signal)))) / (2 * len(signal)),
            ]
        )

        # Frequency domain features
        fft = np.abs(np.fft.rfft(signal))

        # Basic spectral features
        features.extend(
            [
                np.argmax(fft),  # Dominant frequency index
                np.max(fft),  # Maximum amplitude
                np.sum(fft),  # Total spectral energy
                np.mean(fft),  # Mean spectral amplitude
                np.std(fft),  # Spectral standard deviation
            ]
        )

        # Energy in frequency bands (customized for seismic data)
        # Adjust frequency bands as needed for your specific data
        low_band = fft[: len(fft) // 4]  # 0-25% of frequencies
        mid_low_band = fft[len(fft) // 4 : len(fft) // 2]  # 25-50%
        mid_high_band = fft[len(fft) // 2 : 3 * len(fft) // 4]  # 50-75%
        high_band = fft[3 * len(fft) // 4 :]  # 75-100%

        low_energy = np.sum(low_band)
        mid_low_energy = np.sum(mid_low_band)
        mid_high_energy = np.sum(mid_high_band)
        high_energy = np.sum(high_band)
        total_energy = np.sum(fft)

        # Absolute energy in bands
        features.extend([low_energy, mid_low_energy, mid_high_energy, high_energy])

        # Energy ratios (if non-zero)
        if total_energy > 0:
            features.extend(
                [
                    low_energy / total_energy,
                    mid_low_energy / total_energy,
                    mid_high_energy / total_energy,
                    high_energy / total_energy,
                ]
            )
        else:
            features.extend([0, 0, 0, 0])

        # Seismic-specific features
        # P-wave and S-wave energy ratio (simplified)
        # Assuming first 1/3 is mostly P-waves, last 2/3 mostly S-waves
        p_wave_section = signal[: len(signal) // 3]
        s_wave_section = signal[len(signal) // 3 :]

        p_energy = np.sum(p_wave_section**2)
        s_energy = np.sum(s_wave_section**2)

        if p_energy > 0:
            s_p_ratio = s_energy / p_energy
        else:
            s_p_ratio = 0

        features.extend([p_energy, s_energy, s_p_ratio])

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


def create_spatiotemporal_graphs(df, time_window=240, max_distance=None):
    """
    Create a list of graph data objects where each node is an aftershock
    and edges fully connect all events within the time window.
    
    Args:
        df: DataFrame with aftershock information
        time_window: Time window in hours to consider events connected
        max_distance: Optional maximum distance for normalization (km)
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    print("Creating fully connected spatiotemporal graphs...")

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

    # Create spatiotemporal graph data objects
    graph_data_list = []

    # We'll create a graph for each event, including its history
    for i in tqdm(range(1, len(df_sorted))):
        # Get current event and all previous events
        current_time = df_sorted["time_hours"].iloc[i]
        
        # Past events (including the current one)
        past_indices = list(range(i + 1))  # Include the current event

        # Filter events within the time window
        time_indices = [
            j
            for j in past_indices
            if current_time - df_sorted["time_hours"].iloc[j] <= time_window
        ]

        if len(time_indices) > 1:  # Need at least 2 events to form a graph
            # Get node features for this subgraph
            sub_features = node_features[time_indices]
            sub_targets = targets[time_indices]

            # Create fully connected edges (every node connects to every other node)
            edges = []
            edge_attrs = []

            # Get all coordinates and times for efficient computation
            lats = df_sorted["source_latitude_deg"].iloc[time_indices].values
            lons = df_sorted["source_longitude_deg"].iloc[time_indices].values
            times = df_sorted["time_hours"].iloc[time_indices].values
            
            # Find maximum distance for normalization if not provided
            if max_distance is None:
                max_dist = 0
                for idx1 in range(len(time_indices)):
                    for idx2 in range(idx1 + 1, len(time_indices)):
                        dist = haversine_distance(lats[idx1], lons[idx1], lats[idx2], lons[idx2])
                        max_dist = max(max_dist, dist)
                # Add 10% to avoid division by zero issues
                max_dist = max_dist * 1.1 if max_dist > 0 else 100
            else:
                max_dist = max_distance
                
            # Connect every node to every other node (fully connected)
            for idx1 in range(len(time_indices)):
                for idx2 in range(len(time_indices)):
                    if idx1 != idx2:  # No self-loops
                        # Calculate spatial distance
                        spatial_dist = haversine_distance(lats[idx1], lons[idx1], lats[idx2], lons[idx2])
                        
                        # Calculate temporal distance
                        temporal_dist = abs(times[idx1] - times[idx2])
                        
                        # Add edge
                        edges.append([idx1, idx2])
                        
                        # Edge attributes: normalized inverse distance (closer = stronger)
                        # Normalize spatial distance to [0,1] range before inversion
                        norm_spatial_dist = min(spatial_dist / max_dist, 1.0)
                        spatial_weight = 1.0 / (1.0 + norm_spatial_dist)
                        
                        # Normalize temporal distance to [0,1] range before inversion
                        norm_temporal_dist = min(temporal_dist / time_window, 1.0)
                        temporal_weight = 1.0 / (1.0 + norm_temporal_dist)
                        
                        # Combined weight reflecting both spatial and temporal proximity
                        edge_attrs.append([spatial_weight, temporal_weight])

            # Convert to tensor
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            x = torch.tensor(sub_features, dtype=torch.float)
            y = torch.tensor(sub_targets, dtype=torch.float)

            # Create PyTorch Geometric data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            # The target node is always the last one (most recent event)
            data.num_nodes = len(time_indices)

            graph_data_list.append(data)

    print(f"Created {len(graph_data_list)} graphs")
    
    # Quick graph structure check
    print(f"First graph has {graph_data_list[0].num_nodes} nodes and {graph_data_list[0].edge_index.shape[1]} edges")
    target_connections = sum(1 for j in range(graph_data_list[0].edge_index.shape[1]) 
                          if graph_data_list[0].edge_index[0, j].item() == graph_data_list[0].num_nodes - 1 
                          or graph_data_list[0].edge_index[1, j].item() == graph_data_list[0].num_nodes - 1)
    print(f"Target node has {target_connections} connections")
    
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
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

        # Initial layer
        if isinstance(self.conv1, SAGEConv):
            # SAGEConv doesn't use edge_attr
            x = self.conv1(x, edge_index)
        elif isinstance(self.conv1, GATConv) or isinstance(self.conv1, GCNConv):
            x = self.conv1(x, edge_index)
        else:
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
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0

        for data in self.train_loader:
            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(data)

            # Get target indices (the most recent event in each graph)
            target_indices = []
            offset = 0

            # In batched mode, PyTorch Geometric combines graphs into a single large graph
            # with batch information telling us which nodes belong to which graph
            if hasattr(data, "batch"):
                # This is a batched graph
                batch_size = data.num_graphs
                ptr = data.ptr.tolist()  # Pointers to where each graph starts

                for i in range(batch_size):
                    # Get the number of nodes in this graph
                    if i < batch_size - 1:
                        graph_size = ptr[i + 1] - ptr[i]
                    else:
                        graph_size = data.x.size(0) - ptr[i]

                    # Get indices of nodes in this graph
                    start_idx = ptr[i]
                    # The target is the last node in each graph (most recent event)
                    target_idx = start_idx + graph_size - 1
                    target_indices.append(target_idx)
            else:
                # Single graph - target is the last node
                target_indices.append(data.num_nodes - 1)

            # Get predictions and targets
            pred = out[target_indices, :]
            target = data.y[target_indices, :]

            # Calculate loss
            loss = self.criterion(pred, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * len(target_indices)

        return epoch_loss / len(self.train_data)

    def validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for data in self.val_loader:
                # Forward pass
                out = self.model(data)

                # Get target indices
                target_indices = []

                # In batched mode, PyTorch Geometric combines graphs into a single large graph
                # with batch information telling us which nodes belong to which graph
                if hasattr(data, "batch"):
                    # This is a batched graph
                    batch_size = data.num_graphs
                    ptr = data.ptr.tolist()  # Pointers to where each graph starts

                    for i in range(batch_size):
                        # Get the number of nodes in this graph
                        if i < batch_size - 1:
                            graph_size = ptr[i + 1] - ptr[i]
                        else:
                            graph_size = data.x.size(0) - ptr[i]

                        # Get indices of nodes in this graph
                        start_idx = ptr[i]
                        # The target is the last node in each graph (most recent event)
                        target_idx = start_idx + graph_size - 1
                        target_indices.append(target_idx)
                else:
                    # Single graph - target is the last node
                    target_indices.append(data.num_nodes - 1)

                # Get predictions and targets
                pred = out[target_indices, :]
                target = data.y[target_indices, :]

                # Calculate loss
                loss = self.criterion(pred, target)
                val_loss += loss.item() * len(target_indices)

        return val_loss / len(self.val_data)

    def test(self):
        """Test model and return predictions"""
        self.model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in self.test_loader:
                # Forward pass
                out = self.model(data)

                # Get target indices
                target_indices = []

                # In batched mode, PyTorch Geometric combines graphs into a single large graph
                # with batch information telling us which nodes belong to which graph
                if hasattr(data, "batch"):
                    # This is a batched graph
                    batch_size = data.num_graphs
                    ptr = data.ptr.tolist()  # Pointers to where each graph starts

                    for i in range(batch_size):
                        # Get the number of nodes in this graph
                        if i < batch_size - 1:
                            graph_size = ptr[i + 1] - ptr[i]
                        else:
                            graph_size = data.x.size(0) - ptr[i]

                        # Get indices of nodes in this graph
                        start_idx = ptr[i]
                        # The target is the last node in each graph (most recent event)
                        target_idx = start_idx + graph_size - 1
                        target_indices.append(target_idx)
                else:
                    # Single graph - target is the last node
                    target_indices.append(data.num_nodes - 1)

                # Get predictions and targets
                pred = out[target_indices, :]
                target = data.y[target_indices, :]

                # Calculate loss
                loss = self.criterion(pred, target)
                test_loss += loss.item() * len(target_indices)

                # Save predictions and targets
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        # Concatenate results
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)

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
    graph_data_list = create_spatiotemporal_graphs(
        df, time_window=168, max_distance=None
    )

    debug_graph_structure(graph_data_list, num_samples=20)

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train models with different GNN types
    gnn_types = ["gcn", "sage"]
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

        # Train model
        predictor.train(num_epochs=150, patience=15)

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
        f'results/gnn_comparison.png',
        dpi=300,
        bbox_inches="tight",
    )

    print("GNN prediction complete. Results saved in the 'results' directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
