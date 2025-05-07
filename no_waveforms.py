# no_waveforms.py

"""
SpatiotemporalGNN: A specialized Graph Neural Network for aftershock prediction
that focuses exclusively on spatial and temporal patterns without waveform features.

This architecture is specifically designed for the spatial and temporal characteristics
of earthquake sequences, with emphasis on the physics of stress transfer and
space-time patterns of aftershock sequences.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats
import pickle
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# Import necessary functions from relative_gnn.py
from relative_gnn import (
    read_data_from_pickle,
    calculate_relative_coordinates,
    plot_relative_results,
    plot_3d_aftershocks,
    RelativeGNNAftershockPredictor,
)

# Import the graph structure printing functions
from graph_validation import (
    print_graph_structures,
    print_batch_structure,
    print_spatiotemporal_graph_structures
)


class SpatiotemporalGNN(torch.nn.Module):
    """
    Specialized GNN architecture focused exclusively on spatial and temporal patterns
    for aftershock prediction. Designed from the ground up without consideration for
    waveform features.

    Key innovations:
    1. Spatial-temporal embeddings with physics-informed edge representations
    2. Multi-scale temporal attention
    3. Stress field encoding
    4. Custom message passing based on earthquake physics
    """

    def __init__(
        self,
        num_features,  # Number of node features (time + coords)
        num_node_features=None,
        edge_dim=5,  # Dimension of edge features
        hidden_dim=64,  # Hidden dimension
        output_dim=3,  # Output dimension (3 for EW, NS, depth)
        num_layers=3,  # Number of GNN layers
        temporal_scales=[1, 5, 20],  # Multiple time scales for attention (in hours)
        stress_encoding=True,  # Whether to encode stress patterns
        dropout=0.3,  # Dropout rate
        gnn_type="gat",  # Type of GNN layer (GATConv)
        debug_mode=False,  # Whether to print debug information
    ):
        super(SpatiotemporalGNN, self).__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.stress_encoding = stress_encoding
        self.temporal_scales = temporal_scales
        self.debug_mode = debug_mode

        # 1. Input encoding layers
        self.spatial_encoder = nn.Linear(num_features, hidden_dim)

        # 2. Multi-scale temporal attention
        self.temporal_attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )
                for _ in temporal_scales
            ]
        )

        # 3. Stress field encoding (if enabled)
        if stress_encoding:
            # Encode directional stress patterns
            self.stress_encoder = nn.Sequential(
                nn.Linear(2, hidden_dim // 2),  # Angle and distance
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.Tanh(),  # Produces positive and negative stress values
            )

        # 4. Edge encoder with directional information
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # 5. Multi-level spatial attention GNN layers
        self.conv1 = GATConv(
            hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout
        )

        self.convs = nn.ModuleList(
            [
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
                for _ in range(num_layers - 1)
            ]
        )

        # 6. Coordinate-system-aware output layers
        # Separate prediction paths for horizontal and vertical components
        self.horizontal_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # EW, NS
        )

        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Depth
        )

        # 7. Additional layers
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def debug_input_shapes(self, data):
        """Print input tensor shapes for debugging"""
        if not self.debug_mode:
            return
            
        print("\n=== Debugging SpatiotemporalGNN Input Shapes ===")
        if hasattr(data, 'x'):
            print(f"Node features (x): {data.x.shape}")
            if data.x.shape[0] > 0:
                print(f"First node features: {data.x[0]}")
                if data.x.shape[0] > 1:
                    print(f"Second node features: {data.x[1]}")
        else:
            print("No node features (x) found!")
            
        if hasattr(data, 'edge_index'):
            print(f"Edge index: {data.edge_index.shape}")
            # Count edges to target node (index 0)
            num_to_target = torch.sum(data.edge_index[1] == 0).item()
            print(f"Edges to target node: {num_to_target}")
            
            # Check for causality violations
            num_from_target = torch.sum(data.edge_index[0] == 0).item()
            if num_from_target > 0:
                print(f"WARNING: Target node has {num_from_target} outgoing edges (causality violation)!")
        else:
            print("No edge_index found!")
            
        if hasattr(data, 'edge_attr'):
            print(f"Edge attributes: {data.edge_attr.shape}")
            if data.edge_attr.shape[0] > 0:
                print(f"First edge attributes: {data.edge_attr[0]}")
        else:
            print("No edge_attr found!")
            
        if hasattr(data, 'y'):
            print(f"Target (y): {data.y.shape}")
            print(f"Target values: {data.y}")
        else:
            print("No target (y) found!")
            
        if hasattr(data, 'batch') and data.batch is not None:
            print(f"Batch index: {data.batch.shape}")
            print(f"Number of graphs in batch: {int(data.batch.max()) + 1}")
        else:
            print("No batch index or single graph")
        
        print("=================================================\n")

    def forward(self, data):
        # Debug input shapes if debug mode is enabled
        self.debug_input_shapes(data)
        
        # Debug batch structure in more detail if needed
        if self.debug_mode and hasattr(data, 'batch') and data.batch is not None:
            print_batch_structure(data, batch_idx=0)  # Show details of first graph in batch
        
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else None
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

        # Extract temporal features for multi-scale attention
        # Assuming the time feature is the first column after any waveform features
        # For this architecture, our input is [time, ew_rel_km, ns_rel_km, depth_rel_km]
        time_features = x[:, 0].unsqueeze(-1)  # Extract time feature

        # 1. Encode spatial features
        h = self.spatial_encoder(x)

        # 2. Apply multi-scale temporal attention
        for i, scale in enumerate(self.temporal_scales):
            # Normalize time by scale
            normalized_time = time_features / scale
            # Calculate attention weights
            attn = self.temporal_attention[i](normalized_time)
            # Apply attention
            h = h * attn

        # 3. Apply stress field encoding if enabled
        if self.stress_encoding and edge_attr is not None:
            # Extract directional information from edge attributes
            # Assuming angle is the last feature and spatial_weight is the first
            angle_features = edge_attr[:, -1].unsqueeze(-1)  # Normalized angle
            distance_features = 1.0 - edge_attr[:, 0].unsqueeze(
                -1
            )  # 1 - spatial_weight for distance

            # Combine angle and distance for stress encoding
            stress_features = torch.cat([angle_features, distance_features], dim=1)
            stress_encoding = self.stress_encoder(stress_features)

            # Use stress encoding to modulate message passing
            edge_weights = self.edge_encoder(edge_attr) * stress_encoding
        else:
            # Use simpler edge weighting without stress encoding
            edge_weights = (
                self.edge_encoder(edge_attr) if edge_attr is not None else None
            )

        # 4. Apply GNN layers with edge-aware message passing
        # First GNN layer
        if edge_weights is not None:
            h = self.conv1(h, edge_index, edge_weights.squeeze())
        else:
            h = self.conv1(h, edge_index)

        h = F.elu(h)
        h = self.dropout(h)

        # Remaining GNN layers with residual connections
        for conv in self.convs:
            if edge_weights is not None:
                h_new = conv(h, edge_index, edge_weights.squeeze())
            else:
                h_new = conv(h, edge_index)

            # Apply residual connection if shapes match
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new

            h = F.elu(h)
            h = self.dropout(h)

        # Apply layer normalization
        h = self.ln(h)

        # Handle batched data properly - focusing on target nodes
        if batch is not None:
            # Get indices of target nodes (always first node in each graph)
            target_indices = []
            batch_size = int(batch.max()) + 1

            for i in range(batch_size):
                mask = batch == i
                graph_indices = torch.nonzero(mask).squeeze()

                if graph_indices.dim() > 0:
                    if graph_indices.dim() == 0:  # Only one node
                        target_indices.append(graph_indices.item())
                    else:
                        target_indices.append(graph_indices[0].item())

            targets_tensor = torch.tensor(
                target_indices, dtype=torch.long, device=h.device
            )
            target_h = h[targets_tensor]
        else:
            # Single graph - target is the first node
            target_h = h[0].unsqueeze(0)

        # 5. Predict coordinates using specialized output heads
        horizontal_coords = self.horizontal_predictor(target_h)  # EW, NS
        depth = self.depth_predictor(target_h)  # Depth

        # Combine predictions
        coords_pred = torch.cat([horizontal_coords, depth], dim=1)

        return coords_pred


def create_spatiotemporal_graphs(
    df, time_window=120, spatial_threshold=75, min_connections=5, verbose=True
):
    """
    Create graphs with ONLY spatial and temporal features (no waveform features).
    This function is specifically designed to build graphs that focus on space-time
    patterns without any consideration for waveform data.

    Args:
        df: DataFrame with aftershock data
        time_window: Time window in hours
        spatial_threshold: Maximum distance in km
        min_connections: Minimum number of connections required
        verbose: Whether to print progress messages

    Returns:
        List of graph objects, reference coordinates
    """
    if verbose:
        print(
            f"Creating spatiotemporal graphs (time window: {time_window}h, spatial threshold: {spatial_threshold}km)..."
        )

    # Identify mainshock (first event in chronological order)
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp")

    # Get reference coordinates from mainshock
    reference_lat = df_sorted["source_latitude_deg"].iloc[0]
    reference_lon = df_sorted["source_longitude_deg"].iloc[0]
    reference_depth = df_sorted["source_depth_km"].iloc[0]

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

    # Create graph data objects
    graph_data_list = []

    # Create a graph for each event (except the first two to ensure we have context)
    for i in tqdm(range(2, len(df_sorted)), disable=not verbose):
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

                    # Calculate more sophisticated edge features

                    # 1. Temporal features (with Omori's law decay)
                    temporal_dist = current_time - df_sorted["time_hours"].iloc[j]
                    c = 0.01  # Small constant from Omori's law
                    p = 1.1  # Typical p-value from Omori's law
                    omori_weight = 1.0 / ((temporal_dist + c) ** p)

                    # 2. Spatial weight with more physically realistic scaling
                    # Stress decay ~ 1/r^3 for static stress changes
                    spatial_weight = 1.0 / (1.0 + (spatial_dist_3d / 10.0) ** 3)

                    # 3. Calculate angle from past event to current event
                    ew_diff = curr_ew - past_ew
                    ns_diff = curr_ns - past_ns
                    angle = np.degrees(np.arctan2(ew_diff, ns_diff)) % 360

                    # 4. Depth similarity (important for fault planes)
                    depth_similarity = np.exp(-depth_diff / 15.0)

                    # 5. Better Coulomb stress approximation
                    # Integrate regional stress field orientation (approximated)
                    regional_stress_angle = 30  # Example - would be region-specific
                    stress_alignment = np.cos(
                        np.radians(2 * (angle - regional_stress_angle))
                    )
                    stress_proxy = spatial_weight * depth_similarity * stress_alignment

                    # Edge attributes
                    edge_attrs.append(
                        [
                            spatial_weight,
                            omori_weight,
                            stress_proxy,
                            depth_similarity,
                            angle / 360.0,  # Normalized angle as feature
                        ]
                    )

            # Only create graph if we have enough connections
            if len(edges) >= min_connections:
                # Create node features - ONLY temporal and spatial features
                node_features = []

                # Target node features
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
                    # Only include time and coordinates - NO WAVEFORM FEATURES
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
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    num_nodes=len(node_features),
                )

                graph_data_list.append(data)

    if verbose:
        print(f"Created {len(graph_data_list)} spatiotemporal graphs")

        # Quick graph structure check
        if len(graph_data_list) > 0:
            print(
                f"First graph has {graph_data_list[0].num_nodes} nodes and {graph_data_list[0].edge_index.shape[1]} edges"
            )
            print(f"Node feature dimension: {graph_data_list[0].x.shape[1]}")

    # Reference coordinates for conversion back to absolute coordinates
    reference_coords = {
        "latitude": reference_lat,
        "longitude": reference_lon,
        "depth": reference_depth,
    }

    return graph_data_list, reference_coords


def run_spatiotemporal_experiment(
    seed, temporal_scales=[1, 10, 50], stress_encoding=True
):
    """
    Run an experiment using the SpatiotemporalGNN model that completely
    excludes waveform features from the architecture.

    Args:
        seed: Random seed for reproducibility
        temporal_scales: List of temporal scales for attention (in hours)
        stress_encoding: Whether to use stress field encoding

    Returns:
        Dictionary with experiment results
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create output directory
    os.makedirs("spatiotemporal_results", exist_ok=True)

    # Load the data
    if not os.path.exists("aftershock_data.pkl"):
        print("Error: aftershock_data.pkl not found")
        return {}

    # Load the data
    df = read_data_from_pickle("aftershock_data.pkl")

    # Sort data chronologically
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp").drop("timestamp", axis=1)

    # Use the sorted dataframe
    df = df_sorted[2:].reset_index(drop=True)
    print(f"Loaded data with {len(df)} events")

    # Model parameters
    params = {
        "time_window": 120,
        "spatial_threshold": 75,
        "min_connections": 5,
        "hidden_dim": 128,
        "num_layers": 3,
        "learning_rate": 0.0025,
        "batch_size": 8,
        "weight_decay": 5e-6,
        "epochs": 50,
        "patience": 10,
    }

    # Create spatiotemporal graphs (NO WAVEFORM FEATURES AT ALL)
    print("\nCreating spatiotemporal graphs (no waveform features)...")
    spatiotemporal_graphs, reference_coords = create_spatiotemporal_graphs(
        df,
        time_window=params["time_window"],
        spatial_threshold=params["spatial_threshold"],
        min_connections=params["min_connections"],
    )

    if len(spatiotemporal_graphs) == 0:
        print("Error: No graphs created")
        return {}

    # ========== NEW: Print Graph Structures ==========
    print("\nAnalyzing graph structures...")
    structure_summary = print_spatiotemporal_graph_structures(
        spatiotemporal_graphs,
        reference_coords=reference_coords
    )
    
    # Check for serious issues
    if structure_summary["causality_violations"] > 0:
        print(f"WARNING: Found {structure_summary['causality_violations']} graphs with causality violations!")
        print("This means target nodes have outgoing edges, which breaks causality constraints.")
    
    if structure_summary["isolated_target_nodes"] > len(spatiotemporal_graphs) * 0.1:
        print(f"WARNING: {structure_summary['isolated_target_nodes']} target nodes ({structure_summary['isolated_target_nodes']/len(spatiotemporal_graphs)*100:.1f}%) are isolated!")
        print("More than 10% of graphs have no incoming connections to target nodes.")
    
    print("\nGraph structure analysis complete.")
    # =====================================================

    # Get the input feature dimension
    num_features = spatiotemporal_graphs[0].x.shape[1]
    edge_dim = spatiotemporal_graphs[0].edge_attr.shape[1]

    # Create the SpatiotemporalGNN model
    model = SpatiotemporalGNN(
        num_features=num_features,
        edge_dim=edge_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        temporal_scales=temporal_scales,
        stress_encoding=stress_encoding,
        dropout=0.3,
        debug_mode=False,  # Set to True for detailed debugging output
    )

    # Create a custom name based on model configuration
    model_config = (
        f"scales-{'-'.join(map(str, temporal_scales))}_stress-{stress_encoding}"
    )
    model_name = f"spatiotemporal_{model_config}_seed{seed}"

    print(f"\n===== TRAINING AND EVALUATING MODEL: {model_name} =====")

    # Use the RelativeGNNAftershockPredictor as a wrapper for training and evaluation
    predictor = RelativeGNNAftershockPredictor(
        graph_data_list=spatiotemporal_graphs,
        reference_coords=reference_coords,
        gnn_type="gat", 
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        learning_rate=params['learning_rate'],
        batch_size=params['batch_size'], 
        weight_decay=params['weight_decay'],
        model_class=SpatiotemporalGNN,  # Use the class directly, not instance.__class__
        model_kwargs={
            "num_features": num_features,
            # Remove edge_dim from here - it's already passed by the predictor
            "temporal_scales": temporal_scales,
            "stress_encoding": stress_encoding,
            "debug_mode": False,  # Set to True to see detailed per-batch debugging
        }
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
        "model_config": model_config,
        "structure_summary": structure_summary,  # Include graph structure summary
    }

    with open(f"spatiotemporal_results/{model_name}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning Curves - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"spatiotemporal_results/{model_name}_learning_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Create a summary file with key metrics
    with open(f"spatiotemporal_results/{model_name}_summary.txt", "w") as f:
        f.write("==================================================\n")
        f.write("   SPATIOTEMPORAL GNN ARCHITECTURE EXPERIMENT    \n")
        f.write("==================================================\n\n")

        f.write(f"Model: SpatiotemporalGNN\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Temporal Scales: {temporal_scales}\n")
        f.write(f"Stress Encoding: {stress_encoding}\n\n")

        f.write("MODEL ARCHITECTURE:\n")
        f.write("------------------\n")
        f.write(
            "- Special architecture designed for spatial and temporal features only\n"
        )
        f.write("- Multi-scale temporal attention\n")
        if stress_encoding:
            f.write("- Physics-based stress field encoding\n")
        f.write("- Specialized horizontal and vertical prediction paths\n\n")

        f.write("KEY METRICS:\n")
        f.write("-----------\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
            
        # Add graph structure summary
        f.write("\nGRAPH STRUCTURE SUMMARY:\n")
        f.write("------------------------\n")
        f.write(f"Total graphs: {structure_summary['graph_count']}\n")
        f.write(f"Nodes per graph: min={structure_summary['node_count_stats']['min']}, " +
                f"max={structure_summary['node_count_stats']['max']}, " +
                f"mean={structure_summary['node_count_stats']['mean']:.2f}\n")
        f.write(f"Edges per graph: min={structure_summary['edge_count_stats']['min']}, " +
                f"max={structure_summary['edge_count_stats']['max']}, " +
                f"mean={structure_summary['edge_count_stats']['mean']:.2f}\n")
        f.write(f"Causality violations: {structure_summary['causality_violations']} " +
                f"({structure_summary['causality_violations']/structure_summary['graph_count']*100:.2f}%)\n")
        f.write(f"Isolated target nodes: {structure_summary['isolated_target_nodes']} " +
                f"({structure_summary['isolated_target_nodes']/structure_summary['graph_count']*100:.2f}%)\n")

    return results


def run_multiple_seeds(
    seeds=[42, 123, 456, 789, 1024], temporal_scales=[1, 10, 50], stress_encoding=True
):
    """
    Run experiments with multiple seeds to establish statistical significance.

    Args:
        seeds: List of random seeds
        temporal_scales: List of temporal scales for attention
        stress_encoding: Whether to use stress field encoding

    Returns:
        Dictionary with results for each seed
    """
    # Create output directory
    os.makedirs("spatiotemporal_results", exist_ok=True)

    # Store results for each seed
    all_results = []

    # Run experiment for each seed
    for seed_idx, seed in enumerate(seeds):
        print(
            f"\n========== EXPERIMENT WITH SEED {seed} ({seed_idx+1}/{len(seeds)}) =========="
        )

        # Run experiment for this seed
        results = run_spatiotemporal_experiment(seed, temporal_scales, stress_encoding)

        # Store results
        all_results.append(results)

    # Perform statistical analysis
    analyze_results(all_results, seeds, temporal_scales, stress_encoding)

    return all_results


def analyze_results(all_results, seeds, temporal_scales, stress_encoding):
    """
    Analyze results from multiple seeds to establish statistical significance.

    Args:
        all_results: List of result dictionaries from each seed
        seeds: List of random seeds used
        temporal_scales: Temporal scales used in the model
        stress_encoding: Whether stress encoding was used
    """
    # Create a unique configuration identifier
    config = f"scales-{'-'.join(map(str, temporal_scales))}_stress-{stress_encoding}"

    # Define key metrics to analyze
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

    # Collect metrics from all seeds
    metrics_dict = {metric: [] for metric in key_metrics}
    for result in all_results:
        if not result:
            continue

        metrics = result["metrics"]
        for metric in key_metrics:
            if metric in metrics:
                metrics_dict[metric].append(metrics[metric])

    # Calculate statistics
    stats_dict = {}
    for metric, values in metrics_dict.items():
        if values:
            stats_dict[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

    # Save statistics to CSV
    stats_df = pd.DataFrame(
        {
            "Metric": key_metrics,
            "Mean": [stats_dict.get(m, {}).get("mean", np.nan) for m in key_metrics],
            "Std": [stats_dict.get(m, {}).get("std", np.nan) for m in key_metrics],
            "Min": [stats_dict.get(m, {}).get("min", np.nan) for m in key_metrics],
            "Max": [stats_dict.get(m, {}).get("max", np.nan) for m in key_metrics],
        }
    )

    stats_df.to_csv(f"spatiotemporal_results/aggregate_stats_{config}.csv", index=False)

    # Create visualization of mean error metrics
    error_metrics = [m for m in key_metrics if "error" in m]
    error_df = stats_df[stats_df["Metric"].isin(error_metrics)]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(error_df))

    plt.bar(x, error_df["Mean"], yerr=error_df["Std"], capsize=5)
    plt.xticks(x, error_df["Metric"], rotation=45)
    plt.ylabel("Error (km)")
    plt.title(f"Mean Error Metrics (Average of {len(seeds)} Seeds)")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(
        f"spatiotemporal_results/error_metrics_{config}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Create visualization of success rates
    success_metrics = [m for m in key_metrics if "km" in m and "error" not in m]
    success_df = stats_df[stats_df["Metric"].isin(success_metrics)]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(success_df))

    plt.bar(x, success_df["Mean"], yerr=success_df["Std"], capsize=5)
    plt.xticks(x, success_df["Metric"], rotation=45)
    plt.ylabel("Success Rate (%)")
    plt.title(f"Mean Success Rates (Average of {len(seeds)} Seeds)")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(
        f"spatiotemporal_results/success_rates_{config}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Create comprehensive summary report
    with open(f"spatiotemporal_results/aggregate_summary_{config}.txt", "w") as f:
        f.write("==================================================\n")
        f.write("   SPATIOTEMPORAL GNN ARCHITECTURE EXPERIMENT    \n")
        f.write("==================================================\n\n")

        f.write(f"Configuration: {config}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Temporal Scales: {temporal_scales}\n")
        f.write(f"Stress Encoding: {stress_encoding}\n\n")

        f.write("MODEL ARCHITECTURE:\n")
        f.write("------------------\n")
        f.write(
            "- Special architecture designed for spatial and temporal features only\n"
        )
        f.write("- Multi-scale temporal attention\n")
        if stress_encoding:
            f.write("- Physics-based stress field encoding\n")
        f.write("- Specialized horizontal and vertical prediction paths\n\n")

        f.write("AGGREGATE METRICS:\n")
        f.write("-----------------\n")
        for metric in key_metrics:
            if metric in stats_dict:
                stats = stats_dict[metric]
                f.write(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")


def compare_with_baseline(spatiotemporal_results, baseline_results_path):
    """
    Compare SpatiotemporalGNN results with baseline models from previous experiments.

    Args:
        spatiotemporal_results: Results from SpatiotemporalGNN experiments
        baseline_results_path: Path to baseline results CSV file
    """
    # Check if baseline results file exists
    if not os.path.exists(baseline_results_path):
        print(f"Error: Baseline results file '{baseline_results_path}' not found")
        return

    # Load baseline results
    baseline_df = pd.read_csv(baseline_results_path)

    # Create spatiotemporal results dataframe
    config = spatiotemporal_results[0]["model_config"]

    # Define key metrics
    key_metrics = [
        "mean_horizontal_error",
        "median_horizontal_error",
        "mean_depth_error",
        "median_depth_error",
        "mean_3d_error",
        "median_3d_error",
    ]

    # Calculate statistics for spatiotemporal results
    metrics_values = {metric: [] for metric in key_metrics}
    for result in spatiotemporal_results:
        if not result:
            continue

        for metric in key_metrics:
            if metric in result["metrics"]:
                metrics_values[metric].append(result["metrics"][metric])

    # Create comparison dataframe
    comparison_data = {"Metric": key_metrics}

    # Add spatiotemporal results
    comparison_data["SpatiotemporalGNN (Mean)"] = [
        np.mean(metrics_values[m]) if metrics_values[m] else np.nan for m in key_metrics
    ]
    comparison_data["SpatiotemporalGNN (Std)"] = [
        np.std(metrics_values[m]) if metrics_values[m] else np.nan for m in key_metrics
    ]

    # Add baseline results
    for metric in key_metrics:
        if (
            f"With Waveforms (Mean)" in baseline_df.columns
            and metric in baseline_df["Metric"].values
        ):
            mean_val = baseline_df.loc[
                baseline_df["Metric"] == metric, "With Waveforms (Mean)"
            ].values[0]
            std_val = baseline_df.loc[
                baseline_df["Metric"] == metric, "With Waveforms (Std)"
            ].values[0]
            comparison_data[f"With Waveforms (Mean)"] = comparison_data.get(
                f"With Waveforms (Mean)", []
            ) + [mean_val]
            comparison_data[f"With Waveforms (Std)"] = comparison_data.get(
                f"With Waveforms (Std)", []
            ) + [std_val]

        if (
            f"Without Waveforms (Mean)" in baseline_df.columns
            and metric in baseline_df["Metric"].values
        ):
            mean_val = baseline_df.loc[
                baseline_df["Metric"] == metric, "Without Waveforms (Mean)"
            ].values[0]
            std_val = baseline_df.loc[
                baseline_df["Metric"] == metric, "Without Waveforms (Std)"
            ].values[0]
            comparison_data[f"Without Waveforms (Mean)"] = comparison_data.get(
                f"Without Waveforms (Mean)", []
            ) + [mean_val]
            comparison_data[f"Without Waveforms (Std)"] = comparison_data.get(
                f"Without Waveforms (Std)", []
            ) + [std_val]

    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)

    # Calculate improvement percentages
    if "Without Waveforms (Mean)" in comparison_df.columns:
        comparison_df["vs No-WF Improvement (%)"] = np.nan

        for i, metric in enumerate(key_metrics):
            if metric in comparison_df["Metric"].values:
                idx = comparison_df.index[comparison_df["Metric"] == metric].tolist()[0]
                spatiotemporal_val = comparison_df.loc[idx, "SpatiotemporalGNN (Mean)"]
                nowf_val = comparison_df.loc[idx, "Without Waveforms (Mean)"]

                if (
                    not pd.isna(spatiotemporal_val)
                    and not pd.isna(nowf_val)
                    and nowf_val != 0
                ):
                    if "error" in metric:
                        # For error metrics, lower is better
                        improvement = ((nowf_val - spatiotemporal_val) / nowf_val) * 100
                    else:
                        # For success metrics, higher is better
                        improvement = ((spatiotemporal_val - nowf_val) / nowf_val) * 100

                    comparison_df.loc[idx, "vs No-WF Improvement (%)"] = improvement

    # Save comparison to CSV
    comparison_df.to_csv(
        f"spatiotemporal_results/baseline_comparison_{config}.csv", index=False
    )

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Filter to error metrics only
    error_df = comparison_df[comparison_df["Metric"].str.contains("error")]

    x = np.arange(len(error_df))
    width = 0.25

    # Plot bars for each model
    plt.bar(
        x - width,
        error_df["SpatiotemporalGNN (Mean)"],
        width,
        yerr=error_df["SpatiotemporalGNN (Std)"],
        label="SpatiotemporalGNN",
        capsize=5,
    )

    if "Without Waveforms (Mean)" in error_df.columns:
        plt.bar(
            x,
            error_df["Without Waveforms (Mean)"],
            width,
            yerr=error_df["Without Waveforms (Std)"],
            label="Standard (No Waveforms)",
            capsize=5,
        )

    if "With Waveforms (Mean)" in error_df.columns:
        plt.bar(
            x + width,
            error_df["With Waveforms (Mean)"],
            width,
            yerr=error_df["With Waveforms (Std)"],
            label="Standard (With Waveforms)",
            capsize=5,
        )

    plt.xlabel("Metric")
    plt.ylabel("Error (km)")
    plt.title("Error Metrics Comparison - Lower is Better")
    plt.xticks(x, error_df["Metric"], rotation=45)
    plt.legend()
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(
        f"spatiotemporal_results/baseline_comparison_{config}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Create summary report
    with open(f"spatiotemporal_results/comparison_summary_{config}.txt", "w") as f:
        f.write("==================================================\n")
        f.write("   SPATIOTEMPORAL GNN vs. BASELINE COMPARISON    \n")
        f.write("==================================================\n\n")

        f.write(f"SpatiotemporalGNN Configuration: {config}\n\n")

        f.write("PERFORMANCE COMPARISON:\n")
        f.write("----------------------\n")

        for i, row in error_df.iterrows():
            metric = row["Metric"]
            spatiotemporal_mean = row["SpatiotemporalGNN (Mean)"]
            spatiotemporal_std = row["SpatiotemporalGNN (Std)"]

            f.write(f"{metric}:\n")
            f.write(
                f"  SpatiotemporalGNN:       {spatiotemporal_mean:.4f} ± {spatiotemporal_std:.4f}\n"
            )

            if "Without Waveforms (Mean)" in row:
                nowf_mean = row["Without Waveforms (Mean)"]
                nowf_std = row["Without Waveforms (Std)"]
                f.write(
                    f"  Standard (No Waveforms): {nowf_mean:.4f} ± {nowf_std:.4f}\n"
                )

            if "With Waveforms (Mean)" in row:
                wf_mean = row["With Waveforms (Mean)"]
                wf_std = row["With Waveforms (Std)"]
                f.write(f"  Standard (With Waveforms): {wf_mean:.4f} ± {wf_std:.4f}\n")

            if "vs No-WF Improvement (%)" in row and not pd.isna(
                row["vs No-WF Improvement (%)"]
            ):
                improvement = row["vs No-WF Improvement (%)"]
                better = "better" if improvement > 0 else "worse"
                f.write(
                    f"  Improvement vs. No Waveforms: {abs(improvement):.2f}% {better}\n"
                )

            f.write("\n")

        # Overall conclusion
        if "vs No-WF Improvement (%)" in error_df.columns:
            avg_improvement = error_df["vs No-WF Improvement (%)"].mean()
            if avg_improvement > 0:
                f.write(
                    "\nCONCLUSION: The SpatiotemporalGNN architecture specifically designed without waveform features\n"
                )
                f.write(
                    f"outperforms the standard architecture by an average of {avg_improvement:.2f}% on error metrics.\n"
                )
                f.write(
                    "This suggests that specialized architecture design focusing on spatial-temporal patterns\n"
                )
                f.write(
                    "can be more effective than simply removing waveform features from a general architecture.\n"
                )
            else:
                f.write(
                    "\nCONCLUSION: The standard architecture without waveforms still outperforms the specialized\n"
                )
                f.write(
                    f"SpatiotemporalGNN architecture by {-avg_improvement:.2f}% on average for error metrics.\n"
                )
                f.write(
                    "This suggests that the standard architecture already effectively captures\n"
                )
                f.write(
                    "important spatial-temporal patterns even without specialized temporal attention.\n"
                )


def main():
    """Main function to run experiments."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SpatiotemporalGNN experiments")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1024],
        help="Random seeds to use",
    )
    parser.add_argument(
        "--temporal_scales",
        type=int,
        nargs="+",
        default=[1, 10, 50],
        help="Temporal scales for attention (in hours)",
    )
    parser.add_argument(
        "--stress_encoding",
        type=bool,
        default=True,
        help="Whether to use stress field encoding",
    )
    parser.add_argument(
        "--baseline_results",
        type=str,
        default="",
        help="Path to baseline results CSV file for comparison",
    )
    parser.add_argument(
        "--single_run", action="store_true", help="Run a single experiment with seed 42"
    )
    parser.add_argument(
        "--print_only", action="store_true", help="Only print graph structures without training"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Print detailed graph structure information"
    )

    args = parser.parse_args()
    
    # If print_only is set, just create and print graph structures without training
    if args.print_only:
        print("Running graph structure analysis only (no training)...")
        # Set random seed
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Load the data
        if not os.path.exists("aftershock_data.pkl"):
            print("Error: aftershock_data.pkl not found")
            return
            
        # Load and prepare data
        df = read_data_from_pickle("aftershock_data.pkl")
        df_sorted = df.copy()
        df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
        df_sorted = df_sorted.sort_values("timestamp").drop("timestamp", axis=1)
        df = df_sorted[2:].reset_index(drop=True)
        
        # Create spatiotemporal graphs
        spatiotemporal_graphs, reference_coords = create_spatiotemporal_graphs(
            df, time_window=120, spatial_threshold=75, min_connections=5
        )
        
        # Print graph structures
        print_graph_structures(
            spatiotemporal_graphs,
            num_samples=10 if not args.detailed else 5,  # More samples in basic mode, fewer in detailed
            detailed=args.detailed  # Show detailed info if requested
        )
        
        print("Graph structure analysis complete.")
        return

    if args.single_run:
        # Run a single experiment with seed 42
        results = run_spatiotemporal_experiment(
            42, args.temporal_scales, args.stress_encoding
        )
    else:
        # Run multiple experiments with different seeds
        results = run_multiple_seeds(
            args.seeds, args.temporal_scales, args.stress_encoding
        )

        # Compare with baseline if provided
        if args.baseline_results and os.path.exists(args.baseline_results):
            compare_with_baseline(results, args.baseline_results)

    print("Done!")


if __name__ == "__main__":
    main()