#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved aftershock prediction model using GNN with waveform features.
Includes optimizations for better performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats
import pickle
import scipy
from torch_geometric.nn import GENConv  # Improved graph convolution
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool


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


def simplified_extract_waveform_features(waveform):
    """
    Extract a smaller set of essential waveform features to reduce dimensionality.
    
    Args:
        waveform: 3D seismic waveform data with shape [components, samples]
        
    Returns:
        Array of extracted features
    """
    features = []
    
    for component in range(waveform.shape[0]):
        signal = waveform[component]
        
        # 1. Amplitude features (2 per component)
        max_amp = np.max(np.abs(signal))
        mean_amp = np.mean(np.abs(signal))
        features.extend([max_amp, mean_amp])
        
        # 2. Simple shape features (2 per component)
        std_dev = np.std(signal)
        zero_crossings = np.sum(np.diff(np.signbit(signal)))
        features.extend([std_dev, zero_crossings])
        
        # 3. Basic frequency domain feature (1 per component)
        fft = np.abs(np.fft.rfft(signal))
        if np.sum(fft) > 0:
            # Dominant frequency
            freqs = np.fft.rfftfreq(len(signal))
            dominant_freq = freqs[np.argmax(fft)]
            features.append(dominant_freq)
        else:
            features.append(0)
    
    # Total: 5 features per component = 15 features for 3-component data
    return np.array(features)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in km."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r


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
    """
    # Sort by time to find the earliest events
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp")

    # For this dataset, we'll assume the mainshock is the first event
    mainshock_idx = df_sorted.index[0]
    return mainshock_idx


def improved_relative_haversine_loss(y_pred, y_true, horizontal_weight=1.0, depth_weight=0.33):
    """
    Improved loss function with more efficient computation and better weighting.
    
    Args:
        y_pred: tensor of shape (batch_size, 3) [east_west_km, north_south_km, depth_rel_km]
        y_true: tensor of shape (batch_size, 3) [east_west_km, north_south_km, depth_rel_km]
        horizontal_weight: Weight for horizontal error component (default 1.0)
        depth_weight: Weight for depth error component (default 0.33 = ~1/3)
    """
    # Ensure both tensors have a batch dimension
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)

    # More efficient implementation with separate horizontal and depth components
    # Horizontal error (Euclidean distance in the EW-NS plane)
    horizontal_error = torch.sqrt(((y_pred[:, :2] - y_true[:, :2]) ** 2).sum(dim=-1))
    
    # Depth error (absolute difference)
    depth_error = torch.abs(y_pred[:, 2] - y_true[:, 2])
    
    # Weighted sum
    weighted_error = horizontal_weight * horizontal_error + depth_weight * depth_error

    # Return mean error
    return torch.mean(weighted_error)


def relative_to_absolute_coordinates(ew_km, ns_km, depth_rel, reference_coords):
    """
    Convert relative coordinates back to absolute lat/lon/depth.
    """
    ref_lat = reference_coords["latitude"]
    ref_lon = reference_coords["longitude"]
    ref_depth = reference_coords["depth"]

    # Earth radius in km
    R = 6371.0

    # Convert distances to angles in radians
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


class ImprovedAftershockGNN(torch.nn.Module):
    """
    Improved GNN for aftershock prediction with optimized architecture.
    Uses GENConv to better utilize edge features and improved feature handling.
    """
    def __init__(
        self,
        num_features,      # Total number of node features (waveform + time + coords)
        edge_dim=2,        # Dimension of edge features (temporal features)
        hidden_dim=32,     # Hidden dimension
        output_dim=3,      # Output dimension (3 for EW, NS, depth)
        num_layers=2,      # Number of layers
        dropout=0.2,       # Dropout rate
        waveform_feature_dim=15,  # Dimension of waveform features
    ):
        super(ImprovedAftershockGNN, self).__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.waveform_feature_dim = waveform_feature_dim
        
        # Simple encoder for all features
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge encoder - optimize for use with GENConv
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ) if edge_dim > 0 else None
        
        # Use GENConv instead of GraphConv to better utilize edge attributes
        self.conv_layers = nn.ModuleList([
            GENConv(hidden_dim, hidden_dim, aggr='mean', edge_dim=1) 
            for _ in range(num_layers)
        ])
        
        # Direct decoder for coordinate prediction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Simple time encoder
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU()
        )

    def forward(self, past_graph, query_time):
        """Forward pass with improved edge attribute handling"""
        # Process graph features
        x, edge_index = past_graph.x, past_graph.edge_index
        batch = past_graph.batch if hasattr(past_graph, "batch") else None
        edge_attr = past_graph.edge_attr if hasattr(past_graph, "edge_attr") and past_graph.edge_attr is not None else None
        
        # Create single batch index if none exists
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Basic feature encoding
        h = self.encoder(x)
        
        # Process edge attributes for GENConv (requires edge_dim=1)
        edge_weights = None
        if self.edge_encoder is not None and edge_attr is not None:
            edge_weights = self.edge_encoder(edge_attr)  # Shape: [num_edges, 1]
        
        # Apply graph convolutions with improved edge handling
        for conv in self.conv_layers:
            if edge_weights is not None:
                h_new = conv(h, edge_index, edge_weights)
            else:
                h_new = conv(h, edge_index)
            
            # Residual connection
            h = h + h_new
            h = F.relu(h)
        
        # Pool node features to graph level
        pooled = global_mean_pool(h, batch)
        
        # Ensure query_time has the right shape
        if query_time.dim() == 1:
            query_time = query_time.unsqueeze(1)
        
        # Simple concatenation of pooled features with query time embedding
        time_embedding = self.time_encoder(query_time)
        
        # Make dimensions match for addition
        if pooled.size(0) == 1 and time_embedding.size(0) > 1:
            pooled = pooled.repeat(time_embedding.size(0), 1)
        elif time_embedding.size(0) == 1 and pooled.size(0) > 1:
            time_embedding = time_embedding.repeat(pooled.size(0), 1)
        
        # Combine by addition instead of concatenation (simpler)
        combined = pooled + time_embedding
        
        # Direct prediction of coordinates
        pred = self.decoder(combined)
        
        return pred


def create_operational_aftershock_data(
    df, time_window=72, spatial_threshold=60, min_connections=3, max_context=128, verbose=True, use_simplified_features=True
):
    """
    Create operational aftershock data where context graphs contain ONLY past events.
    Each data item is a tuple of (past_graph, query_time, target_coords).
    
    Args:
        df: DataFrame with aftershock data
        time_window: Time window in hours to consider events connected
        spatial_threshold: Maximum distance in km between connected events
        min_connections: Minimum number of connections required to create a graph
        max_context: Maximum number of past events to include
        verbose: Whether to print progress information
        use_simplified_features: Whether to use simplified waveform features
    """
    if verbose:
        print(
            f"Creating operational aftershock data (time window: {time_window}h, spatial threshold: {spatial_threshold}km)..."
        )

    # Identify mainshock (first event in chronological order)
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp").reset_index(drop=True)

    # Get reference coordinates from mainshock
    mainshock_idx = 0  # First event after sorting
    reference_lat = df_sorted["source_latitude_deg"].iloc[mainshock_idx]
    reference_lon = df_sorted["source_longitude_deg"].iloc[mainshock_idx]
    reference_depth = df_sorted["source_depth_km"].iloc[mainshock_idx]

    if verbose:
        print(
            f"Reference event coordinates: Lat={reference_lat:.4f}, Lon={reference_lon:.4f}, Depth={reference_depth:.2f}km"
        )

    # Calculate temporal differences in hours
    df_sorted["time_hours"] = (
        df_sorted["timestamp"] - df_sorted["timestamp"].min()
    ).dt.total_seconds() / 3600

    # Calculate relative coordinates for all events
    if verbose:
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

    # Extract waveform features once for all events
    if verbose:
        print("Extracting waveform features...")
    waveform_features = []
    
    # Choose which feature extraction method to use
    extract_fn = simplified_extract_waveform_features if use_simplified_features else None
    
    for waveform in tqdm(df_sorted["waveform"], disable=not verbose):
        if extract_fn:
            features = extract_fn(waveform)
        else:
            # If simplified not available and no extraction function provided
            # Use the original extract_waveform_features from relative_gnn.py if imported
            try:
                from relative_gnn import extract_waveform_features
                features = extract_waveform_features(waveform)
            except ImportError:
                # Fallback to simplified extraction if relative_gnn not available
                features = simplified_extract_waveform_features(waveform)
        
        waveform_features.append(features)
    
    waveform_features = np.array(waveform_features)
    
    if verbose:
        print(f"Waveform features shape: {waveform_features.shape}")

    # Create data items (past_graph, query_time, target_coords)
    context_target_pairs = []

    # Create causal edges function 
    def create_causal_edges(past_indices, df_sorted, time_window):
        """Create simple causal edges between past events."""
        edges = []
        edge_attrs = []
        
        # Create edges from earlier to later events only
        for idx1, j1 in enumerate(past_indices):
            time1 = df_sorted["time_hours"].iloc[j1]
            
            for idx2, j2 in enumerate(past_indices):
                if j1 != j2:
                    time2 = df_sorted["time_hours"].iloc[j2]
                    
                    # Only create edges from earlier to later events
                    if time1 < time2:
                        # Calculate temporal distance in hours
                        temporal_dist = time2 - time1
                        
                        # Only create edge if close in time
                        if temporal_dist <= time_window:
                            # Add edge (earlier → later)
                            edges.append([idx1, idx2])
                            
                            # Improved edge attributes with more physical meaning
                            # 1. Temporal decay (Omori-like)
                            omori_c = 0.1  # small constant for stability
                            omori_weight = 1.0 / (temporal_dist + omori_c)
                            
                            # Add as edge feature
                            edge_attrs.append([temporal_dist, omori_weight])
        
        return edges, edge_attrs

    # Create an item for each event (except the first few to ensure we have context)
    for i in tqdm(range(min_connections + 2, len(df_sorted)), disable=not verbose):
        current_time = df_sorted["time_hours"].iloc[i]

        # Get ONLY past events within the time window
        past_indices = [
            j
            for j in range(i)  # Only consider events before current one
            if current_time - df_sorted["time_hours"].iloc[j] <= time_window
        ]

        # Limit context size if needed
        if len(past_indices) > max_context:
            # Just take the most recent events
            past_indices = sorted(
                past_indices, 
                key=lambda j: df_sorted["time_hours"].iloc[j], 
                reverse=True
            )[:max_context]

        # Only proceed if we have enough past events for context
        if len(past_indices) >= min_connections:
            # 1. Create node features for past events
            node_features = []
            
            for j in past_indices:
                # Include waveform features, time and known coordinates for past events
                past_features = np.concatenate([
                    waveform_features[j],
                    [df_sorted["time_hours"].iloc[j]],
                    [df_sorted["ew_rel_km"].iloc[j]],
                    [df_sorted["ns_rel_km"].iloc[j]],
                    [df_sorted["depth_rel_km"].iloc[j]],
                ])
                node_features.append(past_features)
            
            # 2. Create edges between past events (causal only)
            edges, edge_attrs = create_causal_edges(past_indices, df_sorted, time_window)
            
            # Convert to numpy arrays
            node_features_array = np.array(node_features, dtype=np.float32)
            
            # 3. Create the past events graph
            x = torch.tensor(node_features_array, dtype=torch.float)
            
            # Only include edges if we have any
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(np.array(edge_attrs, dtype=np.float32), dtype=torch.float)
                
                past_graph = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(node_features),
                )
            else:
                # No edges, just nodes
                past_graph = Data(
                    x=x,
                    num_nodes=len(node_features),
                )
            
            # 4. Create query time tensor
            query_time = torch.tensor([current_time], dtype=torch.float)
            
            # 5. Create target coordinates tensor
            target_coords = torch.tensor(
                [
                    df_sorted["ew_rel_km"].iloc[i],
                    df_sorted["ns_rel_km"].iloc[i],
                    df_sorted["depth_rel_km"].iloc[i],
                ],
                dtype=torch.float
            )
            
            # Add to context-target pairs list
            context_target_pairs.append((past_graph, query_time, target_coords))

    if verbose:
        print(f"Created {len(context_target_pairs)} operational aftershock data items")

        # Quick data structure check
        if len(context_target_pairs) > 0:
            first_past_graph, first_query_time, first_target = context_target_pairs[0]
            print(f"First past graph has {first_past_graph.num_nodes} nodes")
            print(f"Node feature dimension: {first_past_graph.x.shape[1]} (includes waveform features)")
            if hasattr(first_past_graph, "edge_index") and first_past_graph.edge_index is not None:
                print(f"First past graph has {first_past_graph.edge_index.shape[1]} edges")
            print(f"Query time: {first_query_time.item():.2f} hours")
            print(f"Target coordinates: {first_target.numpy()}")

    # Reference coordinates for conversion back to absolute coordinates
    reference_coords = {
        "latitude": reference_lat,
        "longitude": reference_lon,
        "depth": reference_depth,
    }

    return context_target_pairs, reference_coords


class ImprovedAftershockPredictor:
    """
    Improved predictor class for aftershock forecasting with better feature scaling
    and edge attribute handling.
    """

    def __init__(
        self,
        context_target_pairs,
        reference_coords,
        hidden_dim=32,
        num_layers=2,
        learning_rate=0.001,
        batch_size=16,
        weight_decay=1e-4,
        horizontal_weight=1.0,
        depth_weight=0.33,
    ):
        self.context_target_pairs = context_target_pairs
        self.reference_coords = reference_coords
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.horizontal_weight = horizontal_weight
        self.depth_weight = depth_weight

        # Split data into train/val/test
        self.train_data, self.val_data, self.test_data = self._chronological_split()

        # Setup model
        if len(self.context_target_pairs) > 0:
            # Get feature dimensions from the first context graph
            past_graph, _, _ = self.context_target_pairs[0]
            num_features = past_graph.x.shape[1]
            edge_dim = (
                past_graph.edge_attr.shape[1]
                if hasattr(past_graph, "edge_attr") and past_graph.edge_attr is not None
                else 0
            )

            # Calculate waveform feature dimension - default is 15 for simplified features
            waveform_feature_dim = 15  
                
            print(f"Using waveform feature dimension: {waveform_feature_dim}")
            print(f"Total feature dimension: {num_features}")

            # Initialize improved GNN model
            self.model = ImprovedAftershockGNN(
                num_features=num_features,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                output_dim=3,  # [ew_km, ns_km, depth_rel]
                num_layers=num_layers,
                waveform_feature_dim=waveform_feature_dim,
            )

            # Loss and optimizer
            self.criterion = lambda y_pred, y_true: improved_relative_haversine_loss(
                y_pred,
                y_true,
                horizontal_weight=self.horizontal_weight,
                depth_weight=self.depth_weight,
            )

            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

            # Learning rate scheduler with cosine decay
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=50, eta_min=1e-5
            )

            # Feature scaling (fit only on training data)
            self.train_data = self._scale_features(self.train_data, is_train=True)
            self.val_data = self._scale_features(self.val_data, is_train=False)
            self.test_data = self._scale_features(self.test_data, is_train=False)

    def _chronological_split(self, val_ratio=0.2, test_ratio=0.2):
        """Split data chronologically into train/val/test sets"""
        n = len(self.context_target_pairs)

        # Use a chronological split - earlier events for train, later for test
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        # No shuffling - maintain chronological order
        train_data = self.context_target_pairs[:train_end]
        val_data = self.context_target_pairs[train_end:val_end]
        test_data = self.context_target_pairs[val_end:]

        print(
            f"Chronological split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} pairs"
        )
        return train_data, val_data, test_data

    def _scale_features(self, data_pairs, is_train=True):
        """
        Improved feature scaling that handles waveform, time, and coordinate features separately.
        """
        # ------------------------------------------------------------------
        # 1) Fit scalers on the training set
        # ------------------------------------------------------------------
        if is_train:
            # Get first sample to determine feature structure
            past_graph, _, _ = data_pairs[0]
            
            # Assume waveform features are first, followed by time and coordinates
            # Assuming a structure of [waveform_features, time, ew, ns, depth]
            num_features = past_graph.x.shape[1]
            waveform_end = num_features - 4  # Everything before time and coords
            
            # Collect features for fitting scalers
            all_waveform_features = []
            all_time_features = []
            all_coord_features = []
            
            for past_graph, _, _ in data_pairs:
                # Split features into components
                waveform_part = past_graph.x[:, :waveform_end].numpy()
                time_part = past_graph.x[:, waveform_end].numpy().reshape(-1, 1)
                coord_part = past_graph.x[:, waveform_end+1:].numpy()
                
                all_waveform_features.append(waveform_part)
                all_time_features.append(time_part)
                all_coord_features.append(coord_part)
            
            # Stack all samples
            all_waveform_features = np.vstack(all_waveform_features)
            all_time_features = np.vstack(all_time_features)
            all_coord_features = np.vstack(all_coord_features)
            
            # Create and fit separate scalers
            self.waveform_scaler = StandardScaler()
            self.waveform_scaler.fit(all_waveform_features)
            
            # For time features, just store mean and std
            self.time_mean = np.mean(all_time_features)
            self.time_std = np.std(all_time_features)
            
            self.coord_scaler = StandardScaler()
            self.coord_scaler.fit(all_coord_features)
            
            # Remember feature dimensions for applying later
            self.waveform_end = waveform_end
            
        # ------------------------------------------------------------------
        # 2) Apply the scalers with separate normalization for each component
        # ------------------------------------------------------------------
        scaled_pairs = []
        for past_graph, query_time, target_coords in data_pairs:
            # Split features into components
            waveform_end = self.waveform_end
            waveform_part = past_graph.x[:, :waveform_end].numpy()
            time_col = past_graph.x[:, waveform_end].numpy().reshape(-1, 1)
            coord_part = past_graph.x[:, waveform_end+1:].numpy()
            
            # Scale each component separately
            waveform_scaled = self.waveform_scaler.transform(waveform_part)
            time_scaled = (time_col - self.time_mean) / self.time_std  # hours → N(0,1)
            coord_scaled = self.coord_scaler.transform(coord_part)     # EW-NS-depth
            
            # Recombine in original order
            x_scaled = np.hstack([
                waveform_scaled, 
                time_scaled, 
                coord_scaled
            ]).astype(np.float32)

            # Create new Data object with scaled features
            scaled_past_graph = Data(
                x=torch.from_numpy(x_scaled),
                edge_index=past_graph.edge_index,
                edge_attr=past_graph.edge_attr,
                num_nodes=past_graph.num_nodes,
            )

            # Also normalize query time
            query_time_scaled = (query_time - self.time_mean) / self.time_std

            scaled_pairs.append(
                (scaled_past_graph, query_time_scaled, target_coords)
            )

        return scaled_pairs

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_samples = 0

        # Process data in batches
        for i in range(0, len(self.train_data), self.batch_size):
            # Get batch
            batch_data = self.train_data[i:i+self.batch_size]
            
            if not batch_data:
                continue
                
            # Prepare batch data
            past_graphs = [item[0] for item in batch_data]
            query_times = torch.stack([item[1] for item in batch_data])
            targets = torch.stack([item[2] for item in batch_data])
            
            # Create batch from past graphs
            from torch_geometric.data import Batch
            batched_past_graphs = Batch.from_data_list(past_graphs)
            
            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(batched_past_graphs, query_times)

            # Calculate loss
            loss = self.criterion(pred, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * len(batch_data)
            num_samples += len(batch_data)

        return epoch_loss / num_samples if num_samples > 0 else 0

    def validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        num_samples = 0

        with torch.no_grad():
            for i in range(0, len(self.val_data), self.batch_size):
                # Get batch
                batch_data = self.val_data[i:i+self.batch_size]
                
                if not batch_data:
                    continue
                
                # Prepare batch data
                past_graphs = [item[0] for item in batch_data]
                query_times = torch.stack([item[1] for item in batch_data])
                targets = torch.stack([item[2] for item in batch_data])
                
                # Create batch from past graphs
                from torch_geometric.data import Batch
                batched_past_graphs = Batch.from_data_list(past_graphs)
                
                # Forward pass
                pred = self.model(batched_past_graphs, query_times)

                # Calculate loss
                loss = self.criterion(pred, targets)

                val_loss += loss.item() * len(batch_data)
                num_samples += len(batch_data)

        return val_loss / num_samples if num_samples > 0 else 0

    def test(self):
        """Test model and return evaluation metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i in range(0, len(self.test_data), self.batch_size):
                # Get batch
                batch_data = self.test_data[i:i+self.batch_size]
                
                if not batch_data:
                    continue
                
                # Prepare batch data
                past_graphs = [item[0] for item in batch_data]
                query_times = torch.stack([item[1] for item in batch_data])
                targets = torch.stack([item[2] for item in batch_data])
                
                # Create batch from past graphs
                from torch_geometric.data import Batch
                batched_past_graphs = Batch.from_data_list(past_graphs)
                
                # Forward pass
                pred = self.model(batched_past_graphs, query_times)

                # Collect predictions and targets
                all_preds.append(pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Concatenate results
        if all_preds and all_targets:
            y_pred = np.vstack(all_preds)
            y_true = np.vstack(all_targets)
        else:
            print("Warning: No predictions or targets collected")
            return {}, None, None, None

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

    def train(self, num_epochs=50, patience=10):
        """Train the model with early stopping"""
        print(f"Training ImprovedAftershockGNN model for {num_epochs} epochs...")
        print(f"Loss weighting: horizontal={self.horizontal_weight}, depth={self.depth_weight}")

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

            # Update scheduler
            self.scheduler.step()

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
                    self.model.state_dict(),
                    f"best_improved_aftershock_gnn_model.pt",
                )
            else:
                no_improve += 1

            # Print progress
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )

            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(
            torch.load(f"best_improved_aftershock_gnn_model.pt")
        )

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
        plt.title(f"Training and Validation Loss - ImprovedAftershockGNN")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"results/learning_curve_improved_aftershock_gnn.png",
            dpi=300,
            bbox_inches="tight",
        )

        return train_losses, val_losses


def plot_relative_results(
    y_true, y_pred, errors, reference_coords=None, model_name="ImprovedAftershockGNN"
):
    """
    Create visualizations of prediction results for relative coordinates.
    """
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

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


def plot_3d_aftershocks(y_true, y_pred, reference_coords, model_name="ImprovedAftershockGNN"):
    """
    Create 3D visualization of aftershock locations showing true vs predicted positions
    with depth information.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import os
    from matplotlib.lines import Line2D
    
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
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
    
    # Create figure with 3D axes
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot true locations - blue
    ax.scatter(
        abs_true[:, 1],  # longitude
        abs_true[:, 0],  # latitude
        abs_true[:, 2],  # depth
        c='blue', 
        s=30, 
        alpha=0.6,
        label="True Location"
    )
    
    # Plot predicted locations - red
    ax.scatter(
        abs_pred[:, 1],  # longitude
        abs_pred[:, 0],  # latitude
        abs_pred[:, 2],  # depth
        c='red', 
        s=30, 
        alpha=0.6,
        label="Predicted Location"
    )
    
    # Draw lines connecting true and predicted points
    for i in range(len(abs_true)):
        ax.plot(
            [abs_true[i, 1], abs_pred[i, 1]],  # longitude
            [abs_true[i, 0], abs_pred[i, 0]],  # latitude
            [abs_true[i, 2], abs_pred[i, 2]],  # depth
            'k-', 
            alpha=0.15
        )
    
    # Plot reference point (mainshock)
    ax.scatter(
        [reference_coords["longitude"]],
        [reference_coords["latitude"]],
        [reference_coords["depth"]],
        c='green', 
        s=100, 
        marker='*',
        label="Reference (Mainshock)"
    )
    
    # Customize the plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth (km)')
    
    # Invert z-axis so that depth increases downward (as is standard in seismology)
    ax.invert_zaxis()
    
    # Add a title
    ax.set_title(f'3D Aftershock Location Visualization - {model_name}')
    
    # Add grid lines for better depth perception
    ax.grid(True)
    
    # Add legend
    ax.legend()
    
    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Save the figure
    plt.savefig(
        f"results/3d_location_visualization_{model_name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    
    # Create an alternative view (top-down view with color-coded depth)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    # Plot true and predicted locations with depth as color
    sc1 = ax.scatter(
        abs_true[:, 1],  # longitude
        abs_true[:, 0],  # latitude
        c=abs_true[:, 2],  # depth as color
        cmap='Blues',
        s=50,
        alpha=0.7,
        edgecolors='navy',
        label="True"
    )
    
    sc2 = ax.scatter(
        abs_pred[:, 1],  # longitude
        abs_pred[:, 0],  # latitude
        c=abs_pred[:, 2],  # depth as color
        cmap='Reds',
        s=50,
        alpha=0.7,
        edgecolors='darkred',
        label="Predicted"
    )
    
    # Add color bars
    cb1 = plt.colorbar(sc1, ax=ax, pad=0.01)
    cb1.set_label('True Depth (km)')
    
    cb2 = plt.colorbar(sc2, ax=ax, pad=0.06)
    cb2.set_label('Predicted Depth (km)')
    
    # Connect corresponding true and predicted points
    for i in range(len(abs_true)):
        ax.plot(
            [abs_true[i, 1], abs_pred[i, 1]],
            [abs_true[i, 0], abs_pred[i, 0]],
            'k-', 
            alpha=0.15
        )
    
    # Plot reference point (mainshock)
    ax.scatter(
        [reference_coords["longitude"]],
        [reference_coords["latitude"]],
        c='green', 
        s=150, 
        marker='*',
        edgecolors='darkgreen'
    )
    
    # Custom legend (since we're using colormaps for the main scatters)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', markersize=10, label='True Location'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=10, label='Predicted Location'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=15, label='Mainshock')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Map View with Depth-Coded Colors - {model_name}')
    ax.grid(True)
    
    # Save the figure
    plt.savefig(
        f"results/depth_coded_map_{model_name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    
    print(f"Created 3D visualizations for {model_name}")


def run_aftershock_experiment(
    seed,
    horizontal_weight=1.0,
    depth_weight=0.33,
):
    """
    Run an experiment using the improved aftershock forecasting approach.
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Load the data - now preferring the best recording for each event
    data_path = "aftershock_data_best.pkl" if os.path.exists("aftershock_data_best.pkl") else "aftershock_data.pkl"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return {}

    # Load the data
    df = read_data_from_pickle(data_path)
    print(f"Loaded data with {len(df)} events from {data_path}")

    # Improved model parameters
    params = {
        "time_window": 72,
        "spatial_threshold": 60,
        "min_connections": 3,
        "max_context": 32,  # Reduced context size
        "hidden_dim": 48,   # Increased from 32 to 48
        "num_layers": 2,    # Kept at 2 layers
        "learning_rate": 1e-3,
        "batch_size": 16,
        "weight_decay": 1e-4,
        "epochs": 70,       # Increased from 50 to 70
        "patience": 15,     # Increased from 10 to 15
    }

    # Create operational aftershock data (context-target pairs)
    print("\nCreating operational aftershock data with improved features...")
    context_target_pairs, reference_coords = create_operational_aftershock_data(
        df,
        time_window=params["time_window"],
        spatial_threshold=params["spatial_threshold"],
        min_connections=params["min_connections"],
        max_context=params["max_context"],
        use_simplified_features=True,
    )

    if len(context_target_pairs) == 0:
        print("Error: No context-target pairs created")
        return {}

    # Create a custom name based on model configuration
    model_name = f"improved_aftershock_model_seed{seed}"

    print(f"\n===== TRAINING AND EVALUATING IMPROVED MODEL: {model_name} =====")

    # Use the ImprovedAftershockPredictor
    predictor = ImprovedAftershockPredictor(
        context_target_pairs=context_target_pairs,
        reference_coords=reference_coords,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        weight_decay=params["weight_decay"],
        horizontal_weight=horizontal_weight,
        depth_weight=depth_weight,
    )

    # Train the model
    train_losses, val_losses = predictor.train(
        num_epochs=params["epochs"], patience=params["patience"]
    )

    # Test the model
    metrics, y_true, y_pred, errors = predictor.test()

    # Plot results
    plot_relative_results(
        y_true, y_pred, errors, reference_coords=reference_coords, model_name=model_name
    )

    plot_3d_aftershocks(
        y_true, y_pred, reference_coords=reference_coords, model_name=model_name
    )

    # Save results
    results = {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    with open(f"results/{model_name}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Create a summary file with key metrics
    with open(f"results/{model_name}_summary.txt", "w") as f:
        f.write("==================================================\n")
        f.write("   IMPROVED AFTERSHOCK FORECASTING EXPERIMENT   \n")
        f.write("==================================================\n\n")

        f.write(f"Model: ImprovedAftershockGNN\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Hidden Dimension: {params['hidden_dim']}\n")
        f.write(f"Number of Layers: {params['num_layers']}\n")
        f.write(f"Horizontal Weight: {horizontal_weight}\n")
        f.write(f"Depth Weight: {depth_weight}\n\n")

        f.write("MODEL IMPROVEMENTS:\n")
        f.write("------------------\n")
        f.write("- Improved feature scaling with separate normalization for all feature types\n")
        f.write("- Using GENConv instead of GraphConv for better edge attribute utilization\n")
        f.write("- Optimized loss function weighting (3:1 horizontal to depth ratio)\n")
        f.write("- Simplified waveform features with 5 features per component\n")
        f.write("- Better normalization of time features\n")
        f.write("- Enhanced edge attributes with Omori-like decay\n\n")

        f.write("KEY METRICS:\n")
        f.write("-----------\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    # Return results dictionary
    return results


def main():
    """Main function to run experiments with the improved model."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Improved Aftershock Forecasting experiments"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds to use",
    )
    parser.add_argument(
        "--horizontal_weight",
        type=float,
        default=1.0,
        help="Weight for horizontal errors in loss function",
    )
    parser.add_argument(
        "--depth_weight",
        type=float,
        default=0.33,
        help="Weight for depth errors in loss function (default 0.33 = ~1/3 horizontal)",
    )

    args = parser.parse_args()

    # Run experiment with specified seed(s)
    for seed in args.seeds:
        print(f"\n========== RUNNING EXPERIMENT WITH SEED {seed} ==========")
        
        results = run_aftershock_experiment(
            seed,
            horizontal_weight=args.horizontal_weight,
            depth_weight=args.depth_weight,
        )

    print("Done!")


if __name__ == "__main__":
    main()