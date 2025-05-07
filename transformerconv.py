# transformerconv.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats
import pickle
from torch_geometric.nn import GATv2Conv, TransformerConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataLoader


# Import necessary functions from relative_gnn.py
from relative_gnn import (
    read_data_from_pickle,
    calculate_relative_coordinates,
    plot_relative_results,
    plot_3d_aftershocks,
    relative_haversine_loss,
    identify_mainshock,
    haversine_distance
)


class ImprovedSpatiotemporalGNN(torch.nn.Module):
    """
    Specialized GNN architecture focused exclusively on spatial and temporal patterns
    for aftershock prediction. Designed from the ground up without consideration for
    waveform features.

    Key innovations:
    1. Spatial-temporal embeddings with physics-informed edge representations
    2. Multi-scale temporal attention
    3. Stress field encoding
    4. Custom message passing based on earthquake physics
    5. Improved edge feature handling with GATv2Conv or TransformerConv
    6. Better angle representation using sin/cos encoding
    7. BatchNorm instead of LayerNorm for small graphs
    """

    def __init__(
        self,
        num_features,  # Number of node features (time + coords)
        edge_dim=6,    # Dimension of edge features (including sin/cos angle)
        hidden_dim=64,  # Hidden dimension
        output_dim=3,  # Output dimension (3 for EW, NS, depth)
        num_layers=3,  # Number of GNN layers
        temporal_scales=[1, 5, 20],  # Multiple time scales for attention (in hours)
        stress_encoding=True,  # Whether to encode stress patterns
        dropout=0.3,  # Dropout rate
        use_transformer=False,  # Whether to use TransformerConv instead of GATv2Conv
        debug_mode=False,  # Whether to print debug information
    ):
        super(ImprovedSpatiotemporalGNN, self).__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.stress_encoding = stress_encoding
        self.temporal_scales = temporal_scales
        self.debug_mode = debug_mode
        self.time_feature_idx = 0  # Explicitly set time feature index for clarity
        self.use_transformer = use_transformer

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
            # Encode directional stress patterns - now with sin/cos for angle
            self.stress_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),  # sin(angle), cos(angle), and distance
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

        # 5. Multi-level spatial attention GNN layers with proper edge_dim
        if use_transformer:
            # TransformerConv handles edge features natively and often converges faster
            self.conv1 = TransformerConv(
                hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout, edge_dim=hidden_dim
            )
            
            self.convs = nn.ModuleList(
                [
                    TransformerConv(
                        hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout, edge_dim=hidden_dim
                    )
                    for _ in range(num_layers - 1)
                ]
            )
        else:
            # GATv2Conv is an improved version of GATConv with better attention mechanisms
            self.conv1 = GATv2Conv(
                hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout, edge_dim=hidden_dim
            )
            
            self.convs = nn.ModuleList(
                [
                    GATv2Conv(
                        hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout, edge_dim=hidden_dim
                    )
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

        # 7. Additional layers - Replace LayerNorm with BatchNorm1d
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def debug_input_shapes(self, data):
        """Print input tensor shapes for debugging"""
        if not self.debug_mode:
            return

        print("\n=== Debugging ImprovedSpatiotemporalGNN Input Shapes ===")
        if hasattr(data, "x"):
            print(f"Node features (x): {data.x.shape}")
            if data.x.shape[0] > 0:
                print(f"First node features: {data.x[0]}")
                if data.x.shape[0] > 1:
                    print(f"Second node features: {data.x[1]}")
        else:
            print("No node features (x) found!")

        if hasattr(data, "edge_index"):
            print(f"Edge index: {data.edge_index.shape}")
            # Count edges to target node (index 0)
            num_to_target = torch.sum(data.edge_index[1] == 0).item()
            print(f"Edges to target node: {num_to_target}")

            # Check for causality violations
            num_from_target = torch.sum(data.edge_index[0] == 0).item()
            if num_from_target > 0:
                print(
                    f"WARNING: Target node has {num_from_target} outgoing edges (causality violation)!"
                )
        else:
            print("No edge_index found!")

        if hasattr(data, "edge_attr"):
            print(f"Edge attributes: {data.edge_attr.shape}")
            if data.edge_attr.shape[0] > 0:
                print(f"First edge attributes: {data.edge_attr[0]}")
        else:
            print("No edge_attr found!")

        if hasattr(data, "y"):
            print(f"Target (y): {data.y.shape}")
            print(f"Target values: {data.y}")
        else:
            print("No target (y) found!")

        if hasattr(data, "batch") and data.batch is not None:
            print(f"Batch index: {data.batch.shape}")
            print(f"Number of graphs in batch: {int(data.batch.max()) + 1}")
        else:
            print("No batch index or single graph")

        print("=================================================\n")

    def forward(self, data):
        # Debug input shapes if debug mode is enabled
        self.debug_input_shapes(data)

        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else None
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

        # Extract temporal features for multi-scale attention
        # Explicitly using time_feature_idx to extract time feature
        time_features = x[:, self.time_feature_idx].unsqueeze(-1)  # Extract time feature

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
            # Assuming the last two features are sin_angle and cos_angle
            sin_angle = edge_attr[:, -2].unsqueeze(-1)  # Sin of angle
            cos_angle = edge_attr[:, -1].unsqueeze(-1)  # Cos of angle
            
            # Distance feature (1 - spatial_weight for distance)
            distance_features = 1.0 - edge_attr[:, 0].unsqueeze(-1)

            # Combine sin(angle), cos(angle), and distance for stress encoding
            stress_features = torch.cat([sin_angle, cos_angle, distance_features], dim=1)
            stress_encoding = self.stress_encoder(stress_features)

            # Use stress encoding to modulate message passing - maintain shape [E, hidden_dim]
            edge_weights = self.edge_encoder(edge_attr) * stress_encoding
        else:
            # Use simpler edge weighting without stress encoding
            edge_weights = self.edge_encoder(edge_attr) if edge_attr is not None else None

        # 4. Apply GNN layers with edge-aware message passing
        # First GNN layer - Important: Do not squeeze edge_weights to maintain [E, hidden_dim] shape
        if edge_weights is not None:
            h = self.conv1(h, edge_index, edge_weights)
        else:
            h = self.conv1(h, edge_index)

        h = F.elu(h)
        h = self.dropout(h)

        # Remaining GNN layers with residual connections
        for conv in self.convs:
            if edge_weights is not None:
                h_new = conv(h, edge_index, edge_weights)
            else:
                h_new = conv(h, edge_index)

            # Apply residual connection if shapes match
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new

            h = F.elu(h)
            h = self.dropout(h)

        # Apply BatchNorm1d instead of LayerNorm
        h = self.bn(h)

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


def improved_relative_haversine_loss(y_pred, y_true, horizontal_weight=1.0, depth_weight=0.3):
    """
    Improved custom loss function based on relative coordinates with the option to weight 
    horizontal errors more heavily than depth errors.

    Args:
        y_pred: tensor of shape (batch_size, 3) or (3) [east_west_km, north_south_km, depth_rel_km]
        y_true: tensor of shape (batch_size, 3) or (3) [east_west_km, north_south_km, depth_rel_km]
        horizontal_weight: Weight for horizontal error component
        depth_weight: Weight for depth error component
    """
    # Ensure both tensors have a batch dimension
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)
    
    # Horizontal distance error (Euclidean in the relative coordinate system)
    h_dist = torch.sqrt(
        (y_pred[:, 0] - y_true[:, 0]) ** 2 + (y_pred[:, 1] - y_true[:, 1]) ** 2
    )

    # Depth error
    d_dist = torch.abs(y_pred[:, 2] - y_true[:, 2])

    # Weighted combination of horizontal and depth errors
    weighted_error = horizontal_weight * h_dist + depth_weight * d_dist
    
    # Return mean distance
    return torch.mean(weighted_error)


class ImprovedSpatiotemporalGNNPredictor:
    """
    Improved predictor class for training and evaluating SpatiotemporalGNN models.
    Incorporates fixes for edge features, better angle representation, and improved loss function.
    """

    def __init__(
        self,
        graph_data_list,
        reference_coords,
        hidden_dim=64,
        num_layers=3,
        temporal_scales=[1, 10, 50],
        stress_encoding=True,
        learning_rate=0.001,
        batch_size=32,
        weight_decay=1e-5,
        use_transformer=False,
        horizontal_weight=1.0,
        depth_weight=0.3,
    ):
        self.graph_data_list = graph_data_list
        self.reference_coords = reference_coords
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temporal_scales = temporal_scales
        self.stress_encoding = stress_encoding
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.use_transformer = use_transformer
        self.horizontal_weight = horizontal_weight
        self.depth_weight = depth_weight

        # Split data into train/val/test
        self.train_data, self.val_data, self.test_data = self._chronological_split()

        # Setup model
        if len(self.graph_data_list) > 0:
            # Get feature dimensions from the first graph
            num_features = self.graph_data_list[0].x.shape[1]
            edge_dim = (
                self.graph_data_list[0].edge_attr.shape[1]
                if hasattr(self.graph_data_list[0], "edge_attr")
                else 0
            )

            # Initialize improved SpatiotemporalGNN model
            self.model = ImprovedSpatiotemporalGNN(
                num_features=num_features,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                output_dim=3,  # [ew_km, ns_km, depth_rel]
                num_layers=num_layers,
                temporal_scales=temporal_scales,
                stress_encoding=stress_encoding,
                dropout=0.3,
                use_transformer=use_transformer,
            )

            # Loss and optimizer
            # Replace MSE with improved haversine loss function
            self.criterion = lambda y_pred, y_true: improved_relative_haversine_loss(
                y_pred, y_true, 
                horizontal_weight=self.horizontal_weight, 
                depth_weight=self.depth_weight
            )
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

            # Learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )

            # Feature scaling (fit only on training data)
            self.train_data, self.val_data, self.test_data = self._scale_features()

            # Setup data loaders (no shuffling to maintain chronological order)
            self.train_loader = PyGDataLoader(
                self.train_data, batch_size=batch_size, shuffle=False
            )
            self.val_loader = PyGDataLoader(self.val_data, batch_size=batch_size)
            self.test_loader = PyGDataLoader(self.test_data, batch_size=batch_size)

    def _chronological_split(self, val_ratio=0.2, test_ratio=0.1):
        """Split graph data chronologically into train/val/test sets"""
        n = len(self.graph_data_list)

        # Use a chronological split - earlier events for train, later for test
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        # No shuffling - maintain chronological order
        train_data = self.graph_data_list[:train_end]
        val_data = self.graph_data_list[train_end:val_end]
        test_data = self.graph_data_list[val_end:]

        print(
            f"Chronological split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} graphs"
        )
        return train_data, val_data, test_data

    def _scale_features(self):
        """Scale features using only training data statistics"""

        # Extract all node features from training data
        all_train_features = torch.cat(
            [data.x for data in self.train_data], dim=0
        ).numpy()

        # Fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(all_train_features)

        # Function to apply scaling to a dataset
        def apply_scaling(dataset):
            scaled_dataset = []
            for data in dataset:
                # Get original features
                x = data.x.numpy()

                # Scale features
                x_scaled = scaler.transform(x)

                # Create new Data object with scaled features
                new_data = Data(
                    x=torch.tensor(x_scaled, dtype=torch.float),
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    y=data.y,
                    num_nodes=data.num_nodes,
                )
                scaled_dataset.append(new_data)
            return scaled_dataset

        # Scale all datasets using training data statistics
        train_data_scaled = apply_scaling(self.train_data)
        val_data_scaled = apply_scaling(self.val_data)
        test_data_scaled = apply_scaling(self.test_data)

        return train_data_scaled, val_data_scaled, test_data_scaled

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_graphs = 0

        for data in self.train_loader:
            self.optimizer.zero_grad()

            # Forward pass - model outputs prediction for target node
            pred = self.model(data)

            # Get target value
            target = data.y

            # Ensure both tensors have compatible shapes
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)  # Add batch dimension if missing
            if target.dim() == 1:
                target = target.unsqueeze(0)

            # For batches, check if shapes match
            if pred.shape[0] != target.shape[0]:
                # This shouldn't happen with our improved model, but just in case
                print(
                    f"Warning: Shape mismatch - pred: {pred.shape}, target: {target.shape}"
                )

                # Make them compatible if possible
                if pred.shape[0] == 1 and target.shape[0] > 1:
                    pred = pred.repeat(target.shape[0], 1)
                elif pred.shape[0] > 1 and target.shape[0] == 1:
                    target = target.repeat(pred.shape[0], 1)

            # Calculate loss - using improved haversine loss
            loss = self.criterion(pred, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * data.num_graphs
            num_graphs += data.num_graphs

        return epoch_loss / num_graphs

    def validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        num_graphs = 0

        with torch.no_grad():
            for data in self.val_loader:
                # Forward pass
                pred = self.model(data)

                # Get target
                target = data.y

                # Ensure both tensors have compatible shapes
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)  # Add batch dimension if missing
                if target.dim() == 1:
                    target = target.unsqueeze(0)

                # For batches, check if shapes match
                if pred.shape[0] != target.shape[0]:
                    # This shouldn't happen with our improved model, but just in case
                    if pred.shape[0] == 1 and target.shape[0] > 1:
                        pred = pred.repeat(target.shape[0], 1)
                    elif pred.shape[0] > 1 and target.shape[0] == 1:
                        target = target.repeat(pred.shape[0], 1)

                # Calculate loss - using improved haversine loss
                loss = self.criterion(pred, target)

                val_loss += loss.item() * data.num_graphs
                num_graphs += data.num_graphs

        return val_loss / num_graphs

    def test(self):
        """Test model and return evaluation metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in self.test_loader:
                # Forward pass
                pred = self.model(data)

                # Ensure pred has correct shape
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)

                # Get target and ensure correct shape
                target = data.y
                if target.dim() == 1:
                    target = target.unsqueeze(0)

                # Collect predictions and targets
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        # Concatenate results safely
        if all_preds and all_targets:
            try:
                y_pred = np.vstack(all_preds)
                y_true = np.vstack(all_targets)
            except ValueError:
                # Handle irregular shapes by ensuring each array has same shape
                print("Warning: Irregular shapes detected in predictions or targets")
                # Convert to consistent shape (may need adjustment)
                fixed_preds = []
                fixed_targets = []
                for p, t in zip(all_preds, all_targets):
                    if p.ndim == 1:
                        p = p.reshape(1, -1)
                    if t.ndim == 1:
                        t = t.reshape(1, -1)
                    fixed_preds.append(p)
                    fixed_targets.append(t)
                y_pred = np.vstack(fixed_preds)
                y_true = np.vstack(fixed_targets)
        else:
            print("Warning: No predictions or targets collected")
            return {}, None, None, None

        # Calculate errors - both in relative coordinates and absolute if reference is provided
        from relative_gnn import calculate_prediction_errors_relative
        
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
        print(f"Training ImprovedSpatiotemporalGNN model for {num_epochs} epochs...")
        print(f"Using {'TransformerConv' if self.use_transformer else 'GATv2Conv'} with edge features")
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

            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0

                # Save best model
                torch.save(self.model.state_dict(), f"best_improved_spatiotemporal_gnn_model.pt")
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
        self.model.load_state_dict(torch.load(f"best_improved_spatiotemporal_gnn_model.pt"))

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
        plt.title(f"Training and Validation Loss - ImprovedSpatiotemporalGNN")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"results/learning_curve_improved_spatiotemporal_gnn.png",
            dpi=300,
            bbox_inches="tight",
        )

        return train_losses, val_losses


def create_improved_spatiotemporal_graphs(
    df, time_window=120, spatial_threshold=75, min_connections=5, verbose=True
):
    """
    Create graphs with ONLY spatial and temporal features (no waveform features).
    This improved version includes proper angle representation using sin/cos.

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
            f"Creating improved spatiotemporal graphs (time window: {time_window}h, spatial threshold: {spatial_threshold}km)..."
        )

    # Identify mainshock (first event in chronological order)
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp")

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
                    # with improved sin/cos representation
                    ew_diff = curr_ew - past_ew
                    ns_diff = curr_ns - past_ns
                    angle = np.degrees(np.arctan2(ew_diff, ns_diff)) % 360
                    sin_angle = np.sin(np.radians(angle))
                    cos_angle = np.cos(np.radians(angle))

                    # 4. Depth similarity (important for fault planes)
                    depth_similarity = np.exp(-depth_diff / 15.0)

                    # 5. Better Coulomb stress approximation
                    # Integrate regional stress field orientation (approximated)
                    regional_stress_angle = 30  # Example - would be region-specific
                    stress_alignment = np.cos(
                        np.radians(2 * (angle - regional_stress_angle))
                    )
                    stress_proxy = spatial_weight * depth_similarity * stress_alignment

                    # Edge attributes with sin/cos angle representation
                    edge_attrs.append(
                        [
                            spatial_weight,
                            omori_weight,
                            stress_proxy,
                            depth_similarity,
                            sin_angle,  # Sin of angle
                            cos_angle,  # Cos of angle
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
        print(f"Created {len(graph_data_list)} improved spatiotemporal graphs")

        # Quick graph structure check
        if len(graph_data_list) > 0:
            print(
                f"First graph has {graph_data_list[0].num_nodes} nodes and {graph_data_list[0].edge_index.shape[1]} edges"
            )
            print(f"Node feature dimension: {graph_data_list[0].x.shape[1]}")
            print(f"Edge feature dimension: {graph_data_list[0].edge_attr.shape[1]}")

    # Reference coordinates for conversion back to absolute coordinates
    reference_coords = {
        "latitude": reference_lat,
        "longitude": reference_lon,
        "depth": reference_depth,
    }

    return graph_data_list, reference_coords


def run_improved_spatiotemporal_experiment(
    seed, temporal_scales=[1, 10, 50], stress_encoding=True,
    use_transformer=True, horizontal_weight=1.0, depth_weight=0.3
):
    """
    Run an experiment using the improved SpatiotemporalGNN model that completely
    excludes waveform features from the architecture.

    Args:
        seed: Random seed for reproducibility
        temporal_scales: List of temporal scales for attention (in hours)
        stress_encoding: Whether to use stress field encoding
        use_transformer: Whether to use TransformerConv instead of GATv2Conv
        horizontal_weight: Weight for horizontal errors in loss function
        depth_weight: Weight for depth errors in loss function

    Returns:
        Dictionary with experiment results
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create output directory
    os.makedirs("improved_spatiotemporal_results", exist_ok=True)

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

    # Create improved spatiotemporal graphs (NO WAVEFORM FEATURES AT ALL)
    print("\nCreating improved spatiotemporal graphs (no waveform features)...")
    spatiotemporal_graphs, reference_coords = create_improved_spatiotemporal_graphs(
        df,
        time_window=params["time_window"],
        spatial_threshold=params["spatial_threshold"],
        min_connections=params["min_connections"],
    )

    if len(spatiotemporal_graphs) == 0:
        print("Error: No graphs created")
        return {}

    # Create a custom name based on model configuration
    model_config = (
        f"scales-{'-'.join(map(str, temporal_scales))}_stress-{stress_encoding}_"
        f"transformer-{use_transformer}_hw-{horizontal_weight}_dw-{depth_weight}"
    )
    model_name = f"improved_spatiotemporal_{model_config}_seed{seed}"

    print(f"\n===== TRAINING AND EVALUATING MODEL: {model_name} =====")

    # Use the ImprovedSpatiotemporalGNNPredictor
    predictor = ImprovedSpatiotemporalGNNPredictor(
        graph_data_list=spatiotemporal_graphs,
        reference_coords=reference_coords,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        temporal_scales=temporal_scales,
        stress_encoding=stress_encoding,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        weight_decay=params["weight_decay"],
        use_transformer=use_transformer,
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
        "model_config": model_config,
    }

    with open(f"improved_spatiotemporal_results/{model_name}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Create a summary file with key metrics
    with open(f"improved_spatiotemporal_results/{model_name}_summary.txt", "w") as f:
        f.write("==================================================\n")
        f.write("   IMPROVED SPATIOTEMPORAL GNN EXPERIMENT        \n")
        f.write("==================================================\n\n")

        f.write(f"Model: ImprovedSpatiotemporalGNN\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Temporal Scales: {temporal_scales}\n")
        f.write(f"Stress Encoding: {stress_encoding}\n")
        f.write(f"Use TransformerConv: {use_transformer}\n")
        f.write(f"Horizontal Weight: {horizontal_weight}\n")
        f.write(f"Depth Weight: {depth_weight}\n\n")

        f.write("MODEL ARCHITECTURE:\n")
        f.write("------------------\n")
        f.write("- Spatial-temporal embeddings with physics-informed edge representations\n")
        f.write("- Multi-scale temporal attention\n")
        if stress_encoding:
            f.write("- Improved stress field encoding\n")
        f.write("- Proper angle representation using sin/cos\n")
        f.write(f"- Using {'TransformerConv' if use_transformer else 'GATv2Conv'} with edge features\n")
        f.write("- BatchNorm instead of LayerNorm for better small graph handling\n")
        f.write("- Weighted loss function prioritizing horizontal accuracy\n\n")

        f.write("KEY METRICS:\n")
        f.write("-----------\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    # Return results dictionary
    return results


def run_multiple_seeds(
    seeds=[42, 123, 456, 789, 1024], 
    temporal_scales=[1, 10, 50], 
    stress_encoding=True,
    use_transformer=True,
    horizontal_weight=1.0, 
    depth_weight=0.3
):
    """
    Run experiments with multiple seeds to establish statistical significance.

    Args:
        seeds: List of random seeds
        temporal_scales: List of temporal scales for attention
        stress_encoding: Whether to use stress field encoding
        use_transformer: Whether to use TransformerConv instead of GATv2Conv
        horizontal_weight: Weight for horizontal errors in loss function
        depth_weight: Weight for depth errors in loss function

    Returns:
        Dictionary with results for each seed
    """
    # Create output directory
    os.makedirs("improved_spatiotemporal_results", exist_ok=True)

    # Store results for each seed
    all_results = []

    # Run experiment for each seed
    for seed_idx, seed in enumerate(seeds):
        print(
            f"\n========== EXPERIMENT WITH SEED {seed} ({seed_idx+1}/{len(seeds)}) =========="
        )

        # Run experiment for this seed
        results = run_improved_spatiotemporal_experiment(
            seed, 
            temporal_scales, 
            stress_encoding,
            use_transformer,
            horizontal_weight,
            depth_weight
        )

        # Store results
        all_results.append(results)

    # Perform statistical analysis
    analyze_improved_results(
        all_results, 
        seeds, 
        temporal_scales, 
        stress_encoding,
        use_transformer,
        horizontal_weight,
        depth_weight
    )

    return all_results


def analyze_improved_results(
    all_results, 
    seeds, 
    temporal_scales, 
    stress_encoding,
    use_transformer,
    horizontal_weight,
    depth_weight
):
    """
    Analyze results from multiple seeds to establish statistical significance.

    Args:
        all_results: List of result dictionaries from each seed
        seeds: List of random seeds used
        temporal_scales: Temporal scales used in the model
        stress_encoding: Whether stress encoding was used
        use_transformer: Whether TransformerConv was used
        horizontal_weight: Weight used for horizontal errors
        depth_weight: Weight used for depth errors
    """
    # Create a unique configuration identifier
    config = (
        f"scales-{'-'.join(map(str, temporal_scales))}_stress-{stress_encoding}_"
        f"transformer-{use_transformer}_hw-{horizontal_weight}_dw-{depth_weight}"
    )

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

    stats_df.to_csv(f"improved_spatiotemporal_results/aggregate_stats_{config}.csv", index=False)

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
        f"improved_spatiotemporal_results/error_metrics_{config}.png",
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
        f"improved_spatiotemporal_results/success_rates_{config}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Create comprehensive summary report
    with open(f"improved_spatiotemporal_results/aggregate_summary_{config}.txt", "w") as f:
        f.write("==================================================\n")
        f.write("   IMPROVED SPATIOTEMPORAL GNN EXPERIMENT        \n")
        f.write("==================================================\n\n")

        f.write(f"Configuration: {config}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Temporal Scales: {temporal_scales}\n")
        f.write(f"Stress Encoding: {stress_encoding}\n")
        f.write(f"Use TransformerConv: {use_transformer}\n")
        f.write(f"Horizontal Weight: {horizontal_weight}\n")
        f.write(f"Depth Weight: {depth_weight}\n\n")

        f.write("MODEL ARCHITECTURE:\n")
        f.write("------------------\n")
        f.write("- Spatial-temporal embeddings with physics-informed edge representations\n")
        f.write("- Multi-scale temporal attention\n")
        if stress_encoding:
            f.write("- Improved stress field encoding\n")
        f.write("- Proper angle representation using sin/cos\n")
        f.write(f"- Using {'TransformerConv' if use_transformer else 'GATv2Conv'} with edge features\n")
        f.write("- BatchNorm instead of LayerNorm for better small graph handling\n")
        f.write("- Weighted loss function prioritizing horizontal accuracy\n\n")

        f.write("AGGREGATE METRICS:\n")
        f.write("-----------------\n")
        for metric in key_metrics:
            if metric in stats_dict:
                stats = stats_dict[metric]
                f.write(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")


def compare_with_original(
    improved_results, 
    original_results_path,
    temporal_scales=[1, 10, 50], 
    stress_encoding=True,
    use_transformer=True,
    horizontal_weight=1.0, 
    depth_weight=0.3
):
    """
    Compare improved SpatiotemporalGNN results with original model results.

    Args:
        improved_results: Results from improved SpatiotemporalGNN experiments
        original_results_path: Path to original results CSV file
        temporal_scales: Temporal scales used in the model
        stress_encoding: Whether stress encoding was used
        use_transformer: Whether TransformerConv was used
        horizontal_weight: Weight used for horizontal errors
        depth_weight: Weight used for depth errors
    """
    # Check if original results file exists
    if not os.path.exists(original_results_path):
        print(f"Error: Original results file '{original_results_path}' not found")
        return

    # Load original results
    original_df = pd.read_csv(original_results_path)

    # Create configuration string
    config = (
        f"scales-{'-'.join(map(str, temporal_scales))}_stress-{stress_encoding}_"
        f"transformer-{use_transformer}_hw-{horizontal_weight}_dw-{depth_weight}"
    )

    # Define key metrics to compare
    key_metrics = [
        "mean_horizontal_error",
        "median_horizontal_error",
        "mean_depth_error",
        "median_depth_error",
        "mean_3d_error",
        "median_3d_error",
        "horizontal_15km",
        "depth_10km",
        "3d_15km",
    ]

    # Calculate statistics for improved results
    metrics_values = {metric: [] for metric in key_metrics}
    for result in improved_results:
        if not result:
            continue

        for metric in key_metrics:
            if metric in result["metrics"]:
                metrics_values[metric].append(result["metrics"][metric])

    # Create comparison dataframe
    comparison_data = {"Metric": key_metrics}

    # Add improved results
    comparison_data["Improved (Mean)"] = [
        np.mean(metrics_values[m]) if metrics_values[m] else np.nan for m in key_metrics
    ]
    comparison_data["Improved (Std)"] = [
        np.std(metrics_values[m]) if metrics_values[m] else np.nan for m in key_metrics
    ]

    # Add original results if they're in the dataframe
    for metric in key_metrics:
        if metric in original_df["Metric"].values:
            mean_val = original_df.loc[
                original_df["Metric"] == metric, "Mean"
            ].values[0]
            std_val = original_df.loc[
                original_df["Metric"] == metric, "Std"
            ].values[0]

            comparison_data["Original (Mean)"] = comparison_data.get(
                "Original (Mean)", []
            ) + [mean_val]
            comparison_data["Original (Std)"] = comparison_data.get(
                "Original (Std)", []
            ) + [std_val]

    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)

    # Calculate improvement percentages
    if "Original (Mean)" in comparison_df.columns:
        comparison_df["Improvement (%)"] = np.nan

        for i, metric in enumerate(key_metrics):
            if metric in comparison_df["Metric"].values:
                idx = comparison_df.index[comparison_df["Metric"] == metric].tolist()[0]
                improved_val = comparison_df.loc[idx, "Improved (Mean)"]
                original_val = comparison_df.loc[idx, "Original (Mean)"]

                if not pd.isna(improved_val) and not pd.isna(original_val) and original_val != 0:
                    if "error" in metric:
                        # For error metrics, lower is better
                        improvement = ((original_val - improved_val) / original_val) * 100
                    else:
                        # For success metrics, higher is better
                        improvement = ((improved_val - original_val) / original_val) * 100

                    comparison_df.loc[idx, "Improvement (%)"] = improvement

    # Save comparison to CSV
    comparison_df.to_csv(
        f"improved_spatiotemporal_results/comparison_{config}.csv", index=False
    )

    # Create visualization
    plt.figure(figsize=(14, 8))

    # Filter to error metrics only
    error_df = comparison_df[comparison_df["Metric"].str.contains("error")]

    x = np.arange(len(error_df))
    width = 0.35

    # Plot bars for each model
    plt.bar(
        x - width/2,
        error_df["Improved (Mean)"],
        width,
        yerr=error_df["Improved (Std)"],
        label="Improved Model",
        color="royalblue",
        capsize=5,
    )

    if "Original (Mean)" in error_df.columns:
        plt.bar(
            x + width/2,
            error_df["Original (Mean)"],
            width,
            yerr=error_df["Original (Std)"],
            label="Original Model",
            color="firebrick",
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
        f"improved_spatiotemporal_results/comparison_{config}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Create success metrics visualization
    success_df = comparison_df[~comparison_df["Metric"].str.contains("error")]
    
    if not success_df.empty:
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(success_df))
        
        # Plot bars for each model
        plt.bar(
            x - width/2,
            success_df["Improved (Mean)"],
            width,
            yerr=success_df["Improved (Std)"],
            label="Improved Model",
            color="royalblue",
            capsize=5,
        )

        if "Original (Mean)" in success_df.columns:
            plt.bar(
                x + width/2,
                success_df["Original (Mean)"],
                width,
                yerr=success_df["Original (Std)"],
                label="Original Model",
                color="firebrick",
                capsize=5,
            )

        plt.xlabel("Metric")
        plt.ylabel("Success Rate (%)")
        plt.title("Success Metrics Comparison - Higher is Better")
        plt.xticks(x, success_df["Metric"], rotation=45)
        plt.legend()
        plt.grid(True, axis="y")

        plt.tight_layout()
        plt.savefig(
            f"improved_spatiotemporal_results/success_comparison_{config}.png",
            dpi=300,
            bbox_inches="tight",
        )

    # Create summary report
    with open(f"improved_spatiotemporal_results/comparison_summary_{config}.txt", "w") as f:
        f.write("==================================================\n")
        f.write("   IMPROVED vs. ORIGINAL MODEL COMPARISON        \n")
        f.write("==================================================\n\n")

        f.write(f"Improved Model Configuration: {config}\n\n")

        f.write("PERFORMANCE COMPARISON:\n")
        f.write("----------------------\n")

        for i, row in comparison_df.iterrows():
            metric = row["Metric"]
            improved_mean = row["Improved (Mean)"]
            improved_std = row["Improved (Std)"]

            f.write(f"{metric}:\n")
            f.write(
                f"  Improved Model:       {improved_mean:.4f} ± {improved_std:.4f}\n"
            )

            if "Original (Mean)" in row:
                original_mean = row["Original (Mean)"]
                original_std = row["Original (Std)"]
                f.write(
                    f"  Original Model: {original_mean:.4f} ± {original_std:.4f}\n"
                )

            if "Improvement (%)" in row and not pd.isna(row["Improvement (%)"]):
                improvement = row["Improvement (%)"]
                better = "better" if improvement > 0 else "worse"
                f.write(
                    f"  Improvement: {abs(improvement):.2f}% {better}\n"
                )

            f.write("\n")

        # Overall conclusion
        if "Improvement (%)" in comparison_df.columns:
            # Separate error and success metrics for calculation
            error_improvements = comparison_df[
                comparison_df["Metric"].str.contains("error")
            ]["Improvement (%)"].mean()
            
            success_improvements = comparison_df[
                ~comparison_df["Metric"].str.contains("error")
            ]["Improvement (%)"].mean()
            
            f.write("\nCONCLUSIONS:\n")
            f.write("-----------\n")
            
            # Report on error metrics (lower is better)
            if not np.isnan(error_improvements):
                if error_improvements > 0:
                    f.write(f"Error Reduction: The improved model reduces prediction errors by {error_improvements:.2f}% on average.\n")
                else:
                    f.write(f"Error Comparison: The improved model has {-error_improvements:.2f}% higher errors on average.\n")
            
            # Report on success metrics (higher is better)
            if not np.isnan(success_improvements):
                if success_improvements > 0:
                    f.write(f"Success Improvement: The improved model increases success rates by {success_improvements:.2f}% on average.\n")
                else:
                    f.write(f"Success Comparison: The improved model has {-success_improvements:.2f}% lower success rates on average.\n")
            
            # Add overall assessment
            if error_improvements > 0:
                f.write("\nThe improvements to the model architecture have resulted in significant performance gains.\n")
                f.write("Key improvements include:\n")
                f.write("- Proper edge feature handling in GNN layers\n")
                f.write("- Better angle representation using sin/cos\n")
                f.write("- Enhanced GNN layer types (GATv2Conv/TransformerConv)\n")
                f.write("- Weighted loss function prioritizing horizontal accuracy\n")
                f.write("- BatchNorm replacing LayerNorm for small graphs\n")
            else:
                f.write("\nThe original model currently outperforms the improved version.\n")
                f.write("Further parameter tuning and architecture adjustments may be needed:\n")
                f.write("- Try different learning rates or batch sizes\n")
                f.write("- Experiment with different loss function weightings\n")
                f.write("- Consider adding additional temporal scales\n")


def main():
    """Main function to run experiments."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Improved SpatiotemporalGNN experiments")
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
        "--use_transformer",
        type=bool,
        default=True,
        help="Whether to use TransformerConv instead of GATv2Conv",
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
        default=0.3,
        help="Weight for depth errors in loss function",
    )
    parser.add_argument(
        "--original_results",
        type=str,
        default="spatiotemporal_results/aggregate_stats_scales-1-10-50_stress-True.csv",
        help="Path to original results CSV file for comparison",
    )
    parser.add_argument(
        "--single_run", 
        action="store_true", 
        help="Run a single experiment with seed 42"
    )

    args = parser.parse_args()

    if args.single_run:
        # Run a single experiment with seed 42
        results = run_improved_spatiotemporal_experiment(
            42, 
            args.temporal_scales, 
            args.stress_encoding,
            args.use_transformer,
            args.horizontal_weight,
            args.depth_weight
        )
    else:
        # Run multiple experiments with different seeds
        results = run_multiple_seeds(
            args.seeds, 
            args.temporal_scales, 
            args.stress_encoding,
            args.use_transformer,
            args.horizontal_weight,
            args.depth_weight
        )

        # Compare with original results if provided
        if args.original_results and os.path.exists(args.original_results):
            compare_with_original(
                results, 
                args.original_results,
                args.temporal_scales,
                args.stress_encoding,
                args.use_transformer,
                args.horizontal_weight,
                args.depth_weight
            )

    print("Done!")


if __name__ == "__main__":
    main()