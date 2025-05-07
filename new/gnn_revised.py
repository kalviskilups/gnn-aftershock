#!/usr/bin/env python3
# gnn_aftershock_prediction.py - GNN-based models for aftershock location prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader 
from torch_geometric.utils import add_self_loops, degree
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
import pickle
from scipy import signal
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seisbench.data as sbd
from tqdm import tqdm
import warnings
import time
import argparse
import h5py

# Use same geographic to Cartesian functions as in XGBoost implementation
from xgboost_aftershock_prediction import XGBoostAfterShockPredictor


class GNNAfterShockPredictor:
    """
    Class for predicting aftershock locations using Graph Neural Network models
    with best-station and multi-station approaches
    """

    def __init__(
        self,
        data_file=None,
        validation_level="full",
        approach="multi_station",
        feature_type="all",  # "all", "physics", or "signal"
        gnn_type="gat",      # "gcn", "sage", or "gat"
    ):
        """
        Initialize the predictor with feature registry for consistency
        """
        # Reuse much of the initialization from XGBoost implementation
        self.data_dict = None
        self.aftershocks_df = None
        self.mainshock_key = None
        self.mainshock = None
        self.models = None
        self.feature_importances = None
        self.scaler = None
        self.validation_level = validation_level
        self.validation_results = {}
        self.approach = approach
        self.feature_type = feature_type
        self.gnn_type = gnn_type
        self.data_format = None
        self.data_file = data_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Add feature registry for consistency
        self.feature_registry = set()
        self.feature_list = None
        
        print(f"Validation level: {validation_level}")
        print(f"Analysis approach: {approach}")
        print(f"Feature type: {feature_type}")
        print(f"GNN type: {gnn_type}")
        print(f"Using device: {self.device}")

        # Load data - reusing the XGBoost implementation's loading functions
        if data_file and os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            # Initialize the XGBoost helper only once and store it
            self.xgb_helper = XGBoostAfterShockPredictor(
                data_file=data_file,
                validation_level="none",  # We'll validate separately
                approach=approach,
                feature_type=feature_type
            )
            self.data_dict = self.xgb_helper.data_dict
            self.data_format = self.xgb_helper.data_format
        else:
            print("No file found or invalid path provided.")
            return

        # Standardize waveform lengths if not already done
        if hasattr(self.xgb_helper, 'standardize_waveforms'):
            # Already standardized in XGBoost loader
            pass
        else:
            self.standardize_waveforms(target_length=14636)

        # Check data integrity if needed
        if validation_level != "none":
            self.validate_data_integrity()

    def standardize_waveforms(self, target_length=14636):
        """
        Standardize all waveforms to the same length by padding or trimming
        Reusing the implementation from XGBoost
        """
        # Use the existing XGBoost helper
        modified_count = self.xgb_helper.standardize_waveforms(target_length)
        # Update our data dictionary
        self.data_dict = self.xgb_helper.data_dict
        return modified_count

    def find_mainshock(self):
        """
        Identify the mainshock in the dataset
        """
        self.mainshock = {
            "origin_time": "2014-04-01T23:46:50.000000Z",
            "latitude": -19.642,
            "longitude": -70.817,
            "depth": 25.0,
        }
        
        self.mainshock_key = (
            self.mainshock["origin_time"],
            self.mainshock["latitude"],
            self.mainshock["longitude"],
            self.mainshock["depth"],
        )
        
        # Update mainshock in the XGBoost helper as well
        self.xgb_helper.mainshock = self.mainshock
        self.xgb_helper.mainshock_key = self.mainshock_key
        
        print(f"Mainshock: {self.mainshock}")
        return self.mainshock_key

    def create_relative_coordinate_dataframe(self):
        """
        Create a DataFrame with all events and their coordinates
        relative to the mainshock, supporting multi-station format.
        """
        if self.mainshock_key is None:
            self.find_mainshock()
        
        # Use the existing XGBoost helper
        self.xgb_helper.data_dict = self.data_dict
        self.xgb_helper.data_format = self.data_format
        
        # Call the XGBoost implementation
        self.aftershocks_df = self.xgb_helper.create_relative_coordinate_dataframe()
        
        return self.aftershocks_df

    def extract_waveform_features(self, waveform, metadata=None):
        """
        Extract features from the 3-component waveform data
        and update the feature registry for consistency
        """
        # Use the XGBoost helper to extract raw features
        features = self.xgb_helper.extract_waveform_features(waveform, metadata)
        
        # Register all features globally
        self.feature_registry.update(features.keys())
        
        return features

    def create_station_graph(self, event_data):
        """
        Create a graph representation from a single event with multiple stations
        using consistent feature dimensions
        """
        # Make sure feature list is initialized
        if self.feature_list is None:
            raise ValueError("Feature list not initialized. Run prepare_multi_station_graphs first.")
        
        # Extract features for each station
        station_features = []
        station_positions = []
        
        # Use the stations ordered by their selection_score
        for idx, row in event_data.iterrows():
            # Extract features from the waveform
            metadata = row.get("metadata", {})
            features = self.extract_waveform_features(row["waveform"], metadata)
            
            # Create feature vector with consistent dimensions
            feature_vector = []
            for feature_name in self.feature_list:
                if feature_name in features and pd.notna(features[feature_name]):
                    feature_vector.append(features[feature_name])
                else:
                    feature_vector.append(0.0)  # Default value for missing features
            
            station_features.append(feature_vector)
            
            # Store station position (x, y, z from mainshock)
            if "station_distance" in row:
                # Use station distance if available
                station_positions.append([row["station_distance"], 0, 0])  # Simplified
            else:
                # Default position (relative to mainshock)
                station_positions.append([0, 0, 0])
        
        # Create node features tensor
        if not station_features:
            # Handle empty case
            return None
        
        # Convert to torch tensors
        x = torch.tensor(station_features, dtype=torch.float)
        pos = torch.tensor(station_positions, dtype=torch.float)
        
        # Create a target tensor (3D coordinates relative to mainshock)
        y = torch.tensor(
            [
                event_data["relative_x"].iloc[0],
                event_data["relative_y"].iloc[0],
                event_data["relative_z"].iloc[0],
            ],
            dtype=torch.float,
        )
        
        # Create edges - fully connected graph
        num_nodes = len(station_features)
        
        # Create edges for a fully connected graph
        source_nodes = []
        target_nodes = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    source_nodes.append(i)
                    target_nodes.append(j)
        
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        # Add distance between stations as edge attributes
        edge_attr = []
        for i, j in zip(source_nodes, target_nodes):
            # Calculate Euclidean distance between stations
            dist = np.linalg.norm(
                np.array(station_positions[i]) - np.array(station_positions[j])
            )
            edge_attr.append([dist])
        
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)
        
        # Add additional info for interpretation
        data.num_nodes = num_nodes
        data.event_id = event_data["event_id"].iloc[0]
        
        return data

    def prepare_multi_station_graphs(self):
        """
        Prepare multi-station graph dataset for GNN training
        with consistent feature dimensions
        """
        if self.aftershocks_df is None:
            self.create_relative_coordinate_dataframe()
        
        if self.data_format != "multi_station":
            print("Using single-station format - falling back to best station approach")
            return self.prepare_best_station_graphs()
        
        print("Preparing multi-station graph dataset...")
        
        # First pass: extract all features to build the complete feature registry
        print("First pass: building feature registry...")
        for event_id, event_data in tqdm(self.aftershocks_df.groupby("event_id"), 
                                        desc="Building feature registry"):
            # Skip mainshock
            if event_data["is_mainshock"].any():
                continue
                
            for idx, row in event_data.iterrows():
                # Extract features to update the registry
                metadata = row.get("metadata", {})
                _ = self.extract_waveform_features(row["waveform"], metadata)
        
        # Finalize the feature list (sort for consistency)
        self.feature_list = sorted(self.feature_registry)
        print(f"Complete feature registry contains {len(self.feature_list)} features")
        
        # Second pass: create graphs with consistent feature dimensions
        graph_dataset = []
        errors = 0
        
        for event_id, event_data in tqdm(self.aftershocks_df.groupby("event_id"), 
                                        desc="Creating event graphs"):
            # Skip mainshock
            if event_data["is_mainshock"].any():
                continue
                
            try:
                # Create a graph for this event
                graph = self.create_station_graph(event_data)
                
                if graph is not None and graph.num_nodes > 0:
                    graph_dataset.append(graph)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error creating graph for event {event_id}: {e}")
                elif errors == 6:
                    print("Additional errors occurred but not printed...")
        
        print(f"Successfully created {len(graph_dataset)} event graphs with {errors} errors")
        
        return graph_dataset

    def prepare_best_station_graphs(self):
        """
        Prepare best-station dataset as individual graphs
        with consistent feature dimensions
        """
        if self.aftershocks_df is None:
            self.create_relative_coordinate_dataframe()
        
        print("Preparing best-station graph dataset...")
        
        # For multi-station format, select the best station for each event
        best_stations_df = self.aftershocks_df
        
        if self.data_format == "multi_station":
            print("Selecting best station for each event...")
            # Group by event_id and select the station with highest selection_score
            best_station_indices = self.aftershocks_df.groupby("event_id")[
                "selection_score"
            ].idxmax()
            best_stations_df = self.aftershocks_df.loc[
                best_station_indices
            ].reset_index(drop=True)
            print(
                f"Selected {len(best_stations_df)} best stations from {len(self.aftershocks_df)} total recordings"
            )
        
        # First pass: build feature registry
        print("First pass: building feature registry...")
        for idx, row in tqdm(best_stations_df.iterrows(), desc="Building feature registry"):
            # Skip mainshock
            if row["is_mainshock"]:
                continue
                
            # Extract features to update the registry
            metadata = row.get("metadata", {})
            _ = self.extract_waveform_features(row["waveform"], metadata)
        
        # Finalize the feature list
        self.feature_list = sorted(self.feature_registry)
        print(f"Complete feature registry contains {len(self.feature_list)} features")
        
        # Second pass: extract features with consistent dimensions
        print("Second pass: creating graphs with consistent features...")
        graph_dataset = []
        errors = 0
        
        for idx, row in tqdm(best_stations_df.iterrows(), desc="Creating graphs"):
            # Skip mainshock
            if row["is_mainshock"]:
                continue
                
            try:
                # Extract features
                metadata = row.get("metadata", {})
                features = self.extract_waveform_features(row["waveform"], metadata)
                
                # Create feature vector with consistent dimensions
                feature_vector = []
                for feature_name in self.feature_list:
                    if feature_name in features and pd.notna(features[feature_name]):
                        feature_vector.append(features[feature_name])
                    else:
                        feature_vector.append(0.0)  # Default value for missing features
                
                # Create target tensor
                target = torch.tensor(
                    [
                        row["relative_x"],
                        row["relative_y"],
                        row["relative_z"],
                    ],
                    dtype=torch.float,
                )
                
                # Create node features tensor
                x = torch.tensor([feature_vector], dtype=torch.float)
                
                # Create a single-node graph with self-loop
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                
                # Create data object
                data = Data(x=x, edge_index=edge_index, y=target)
                data.num_nodes = 1
                
                # Add event_id for GroupKFold
                if "event_id" in row:
                    data.event_id = row["event_id"]
                else:
                    data.event_id = f"event_{idx}"
                
                graph_dataset.append(data)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error processing waveform {idx}: {e}")
                elif errors == 6:
                    print("Additional errors occurred but not printed...")
        
        print(f"Created {len(graph_dataset)} graphs for best-station approach with {errors} errors")
        
        return graph_dataset

    def prepare_dataset(self):
        """
        Prepare dataset for GNN training with consistent feature dimensions
        """
        if self.mainshock_key is None:
            self.find_mainshock()
            
        if self.approach == "multi_station" and self.data_format == "multi_station":
            return self.prepare_multi_station_graphs()
        else:
            return self.prepare_best_station_graphs()

    class GNNModel(torch.nn.Module):
        """
        Graph Neural Network model for aftershock location prediction
        """
        def __init__(self, input_dim, hidden_dim=64, output_dim=3, gnn_type="gat", num_layers=3):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.gnn_type = gnn_type
            self.num_layers = num_layers
            
            # Feature importance tracking (similar to XGBoost)
            self.feature_importances_ = torch.zeros(input_dim)
            
            # Input layer
            if gnn_type == "gcn":
                self.conv_layers = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
                for _ in range(num_layers - 1):
                    self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "sage":
                self.conv_layers = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
                for _ in range(num_layers - 1):
                    self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.conv_layers = nn.ModuleList([GATConv(input_dim, hidden_dim // 4, heads=4)])
                for _ in range(num_layers - 1):
                    self.conv_layers.append(GATConv(hidden_dim, hidden_dim // 4, heads=4))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            # MLP for final prediction
            self.lin1 = nn.Linear(hidden_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # Store input for feature importance
            self.last_input = x.detach().clone()
            
            # Graph convolution layers
            for conv in self.conv_layers:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
            
            # Global pooling - aggregate node features for each graph
            x = torch_geometric.nn.global_mean_pool(x, batch)
            
            # MLP for prediction
            x = F.relu(self.lin1(x))
            x = self.lin2(x)
            
            return x
        
        def update_feature_importance(self, gradients):
            """
            Update feature importance based on gradients (similar to XGBoost)
            """
            if gradients is not None and self.last_input is not None:
                # Calculate feature importance as the mean of |gradient * input|
                importance = torch.abs(gradients * self.last_input).mean(dim=0)
                self.feature_importances_ += importance.cpu()

    def train_gnn_models(self, graph_dataset, perform_feature_importance=True):
        """
        Train GNN models to predict aftershock locations
        with proper batch handling
        """
        print("\n" + "=" * 50)
        print("TRAINING GNN MODEL")
        print("=" * 50)
        
        # Make sure we're using the proper DataLoader
        from torch_geometric.loader import DataLoader
        
        # Split data into training and testing sets
        if self.validation_level != "none":
            # Extract event_ids for GroupKFold
            event_ids = [data.event_id for data in graph_dataset]
            
            # Use GroupKFold to prevent data leakage
            gkf = GroupKFold(n_splits=5)
            # Convert to numpy array for GroupKFold
            event_ids_np = np.array(event_ids)
            
            # Get a single train/test split
            train_indices, test_indices = next(gkf.split(
                np.zeros(len(graph_dataset)), 
                np.zeros(len(graph_dataset)), 
                event_ids_np
            ))
            
            train_dataset = [graph_dataset[i] for i in train_indices]
            test_dataset = [graph_dataset[i] for i in test_indices]
        else:
            # Simple random split
            num_samples = len(graph_dataset)
            indices = list(range(num_samples))
            np.random.shuffle(indices)
            split = int(np.floor(0.8 * num_samples))
            
            train_indices, test_indices = indices[:split], indices[split:]
            train_dataset = [graph_dataset[i] for i in train_indices]
            test_dataset = [graph_dataset[i] for i in test_indices]
        
        print(f"Training dataset: {len(train_dataset)} samples")
        print(f"Testing dataset: {len(test_dataset)} samples")
        
        # Check if all graphs have the same number of features
        if len(train_dataset) == 0:
            raise ValueError("Empty training dataset!")
            
        input_dim = train_dataset[0].x.shape[1]
        print(f"Input dimension: {input_dim}")
        
        # Create dataloaders with smaller batch size initially
        train_loader = DataLoader(
            train_dataset, batch_size=4, shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=4, shuffle=False
        )
        
        # Initialize model
        model = self.GNNModel(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=3,  # x, y, z coordinates
            gnn_type=self.gnn_type,
            num_layers=3
        ).to(self.device)
        
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(1, 301):  # Maximum 300 epochs
            epoch_loss = 0
            for batch in train_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                out = model(batch)
                
                # Make sure y has the correct shape [batch_size, 3]
                if batch.y.dim() == 1:
                    # Reshape from [batch_size*3] to [batch_size, 3]
                    batch_size = out.size(0)
                    y_reshaped = batch.y.view(batch_size, 3)
                else:
                    y_reshaped = batch.y
                    
                # Loss calculation (MSE)
                loss = F.mse_loss(out, y_reshaped)
                
                # Update feature importance if needed
                if perform_feature_importance:
                    # Calculate gradients w.r.t input
                    if batch.x.requires_grad == False:
                        batch.x.requires_grad = True
                    
                    # We need to create gradients for input
                    gradients = torch.autograd.grad(
                        loss, batch.x, 
                        retain_graph=True, 
                        create_graph=False,
                        allow_unused=True
                    )[0]

                    if gradients is not None:
                        model.update_feature_importance(gradients)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validate on test set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    out = model(batch)
                    
                    # Ensure consistent shapes for validation loss
                    if batch.y.dim() == 1:
                        batch_size = out.size(0)
                        y_reshaped = batch.y.view(batch_size, 3)
                    else:
                        y_reshaped = batch.y
                        
                    val_loss += F.mse_loss(out, y_reshaped).item()
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Switch back to training mode
            model.train()
        
        # Load best model
        model.load_state_dict(best_model)
        
        # Switch to evaluation mode
        model.eval()
        
        # Evaluate model on test set
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch)
                
                # Collect predictions and ground truth
                y_true.append(batch.y.cpu().numpy())
                y_pred.append(out.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        
        # Calculate errors
        errors = {}
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            mse = np.mean((y_true[:, i] - y_pred[:, i]) ** 2)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            errors[coord] = rmse
            print(f"  {coord} RMSE: {rmse:.4f} km, R²: {r2:.4f}")
        
        # Calculate 3D distance error
        dist_3d = np.sqrt(
            (y_true[:, 0] - y_pred[:, 0]) ** 2 + 
            (y_true[:, 1] - y_pred[:, 1]) ** 2 + 
            (y_true[:, 2] - y_pred[:, 2]) ** 2
        )
        
        errors["3d_distance"] = np.mean(dist_3d)
        print(f"  3D Mean Error: {errors['3d_distance']:.4f} km")
        print(f"  3D Median Error: {np.median(dist_3d):.4f} km")
        
        # Extract feature importance if available
        if perform_feature_importance and hasattr(model, 'feature_importances_'):
            # Normalize importance
            feature_importances = model.feature_importances_.cpu().numpy()
            if np.sum(feature_importances) > 0:
                feature_importances = feature_importances / np.sum(feature_importances)
            
            # Store in dictionary with same structure as XGBoost
            self.feature_importances = {}
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
            
            for coord in ["relative_x", "relative_y", "relative_z"]:
                # Same importances for all coordinates in this simple model
                importance_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": feature_importances,
                }).sort_values("importance", ascending=False)
                
                self.feature_importances[coord] = importance_df
                
                print(f"\nTop features for {coord} prediction:")
                print(importance_df.head(10))
        
        # Store model and errors
        self.models = {
            "gnn": model,
            "type": f"gnn_{self.gnn_type}",
            "errors": errors,
        }
        
        return model, test_dataset

    def visualize_predictions_geographic(self, model, test_dataset):
        """
        Visualize prediction results on a geographic map
        """
        if self.models is None:
            raise ValueError("Models not trained yet")
        
        if self.mainshock is None:
            self.find_mainshock()
        
        # Create a DataLoader for the test dataset
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Make predictions
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch)
                
                # Add this reshaping logic for consistent dimensions
                if batch.y.dim() == 1:
                    # Reshape from [batch_size*3] to [batch_size, 3]
                    batch_size = out.size(0)
                    y_reshaped = batch.y.view(batch_size, 3)
                else:
                    y_reshaped = batch.y
                    
                # Use the reshaped tensor
                y_true.append(y_reshaped.cpu().numpy())
                y_pred.append(out.cpu().numpy())

                print(f"Batch y shape: {batch.y.shape}, Output shape: {out.shape}")

        
        # Convert to numpy arrays
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        
        # Create DataFrames for visualization
        test_indices = np.arange(len(y_true))
        y_test = pd.DataFrame(
            y_true, 
            columns=["relative_x", "relative_y", "relative_z"],
            index=test_indices,
        )
        
        y_pred_df = pd.DataFrame(
            y_pred,
            columns=["relative_x", "relative_y", "relative_z"],
            index=test_indices,
        )
        
        # Convert to absolute coordinates
        true_absolute = pd.DataFrame(index=test_indices)
        pred_absolute = pd.DataFrame(index=test_indices)
        
        # Use the XGBoost helper for conversion
        for i in range(len(y_test)):
            # True coordinates
            lat, lon, depth = self.xgb_helper.cartesian_to_geographic(
                y_test["relative_x"].iloc[i],
                y_test["relative_y"].iloc[i],
                y_test["relative_z"].iloc[i],
                self.mainshock["latitude"],
                self.mainshock["longitude"],
                self.mainshock["depth"],
            )
            true_absolute.loc[test_indices[i], "lat"] = lat
            true_absolute.loc[test_indices[i], "lon"] = lon
            true_absolute.loc[test_indices[i], "depth"] = depth

            # Predicted coordinates
            lat, lon, depth = self.xgb_helper.cartesian_to_geographic(
                y_pred_df["relative_x"].iloc[i],
                y_pred_df["relative_y"].iloc[i],
                y_pred_df["relative_z"].iloc[i],
                self.mainshock["latitude"],
                self.mainshock["longitude"],
                self.mainshock["depth"],
            )
            pred_absolute.loc[test_indices[i], "lat"] = lat
            pred_absolute.loc[test_indices[i], "lon"] = lon
            pred_absolute.loc[test_indices[i], "depth"] = depth
        
        # Create map visualization
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.Mercator())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        
        # Set extent
        buffer = 0.3
        min_lon = min(true_absolute["lon"].min(), pred_absolute["lon"].min()) - buffer
        max_lon = max(true_absolute["lon"].max(), pred_absolute["lon"].max()) + buffer
        min_lat = min(true_absolute["lat"].min(), pred_absolute["lat"].min()) - buffer
        max_lat = max(true_absolute["lat"].max(), pred_absolute["lat"].max()) + buffer
        
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Plot true locations
        ax.scatter(
            true_absolute["lon"],
            true_absolute["lat"],
            c="blue",
            s=30,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            label="True Locations",
        )
        
        # Plot predicted locations
        ax.scatter(
            pred_absolute["lon"],
            pred_absolute["lat"],
            c="red",
            s=30,
            alpha=0.7,
            marker="x",
            transform=ccrs.PlateCarree(),
            label="Predicted Locations",
        )
        
        # Connect true and predicted with lines
        for i in range(len(true_absolute)):
            ax.plot(
                [true_absolute["lon"].iloc[i], pred_absolute["lon"].iloc[i]],
                [true_absolute["lat"].iloc[i], pred_absolute["lat"].iloc[i]],
                "k-",
                alpha=0.2,
                transform=ccrs.PlateCarree(),
            )
        
        # Plot mainshock
        ax.scatter(
            self.mainshock["longitude"],
            self.mainshock["latitude"],
            c="yellow",
            s=200,
            marker="*",
            edgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=5,
            label="Mainshock",
        )
        
        # Add gridlines and legend
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        
        plt.title(
            f'GNN ({self.gnn_type.upper()}): True vs Predicted Aftershock Locations ({self.approach.replace("_", " ").title()})'
        )
        plt.legend(loc="lower left")
        
        # Save figure
        plt.savefig(
            f"gnn_prediction_results_geographic_{self.gnn_type}_{self.approach}.png",
            dpi=300,
            bbox_inches="tight",
        )
        
        # Calculate location errors
        earth_radius = 6371.0  # km
        
        # Calculate differences in degrees
        lat_diff_deg = np.abs(true_absolute["lat"] - pred_absolute["lat"])
        lon_diff_deg = np.abs(true_absolute["lon"] - pred_absolute["lon"])
        
        # Convert to approximate distances in km
        lat_diff_km = lat_diff_deg * (np.pi / 180) * earth_radius
        # Account for longitude convergence
        avg_lat = (true_absolute["lat"] + pred_absolute["lat"]) / 2
        lon_diff_km = (
            lon_diff_deg * (np.pi / 180) * earth_radius * np.cos(np.radians(avg_lat))
        )
        
        # Depth difference
        depth_diff_km = np.abs(true_absolute["depth"] - pred_absolute["depth"])
        
        # 3D distance
        distance_3d_km = np.sqrt(lat_diff_km**2 + lon_diff_km**2 + depth_diff_km**2)
        
        # Print statistics
        print("Prediction Error Statistics:")
        print(f"Mean latitude error: {lat_diff_km.mean():.2f} km")
        print(f"Mean longitude error: {lon_diff_km.mean():.2f} km")
        print(f"Mean depth error: {depth_diff_km.mean():.2f} km")
        print(f"Mean 3D error: {distance_3d_km.mean():.2f} km")
        print(f"Median 3D error: {distance_3d_km.median():.2f} km")
        
        # Create error histogram
        plt.figure(figsize=(10, 6))
        plt.hist(distance_3d_km, bins=20, alpha=0.7)
        plt.axvline(
            distance_3d_km.mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {distance_3d_km.mean():.2f} km",
        )
        plt.axvline(
            distance_3d_km.median(),
            color="g",
            linestyle="--",
            label=f"Median: {distance_3d_km.median():.2f} km",
        )
        plt.title(
            f'GNN ({self.gnn_type.upper()}): 3D Location Error Distribution ({self.approach.replace("_", " ").title()})'
        )
        plt.xlabel("Error (km)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(
            f"gnn_prediction_error_histogram_{self.gnn_type}_{self.approach}.png", 
            dpi=300
        )
        plt.close()
        
        # Return results
        return true_absolute, pred_absolute, {
            "3d": distance_3d_km,
            "lat": lat_diff_km,
            "lon": lon_diff_km,
            "depth": depth_diff_km,
        }

    def run_complete_workflow(self, perform_feature_importance=True):
        """
        Run the complete analysis workflow with GNN
        
        Args:
            perform_feature_importance: Whether to calculate feature importance
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Print header
        print("\n" + "=" * 70)
        print(
            f"GNN AFTERSHOCK ANALYSIS WITH {self.approach.upper()} APPROACH".center(70)
        )
        print(f"USING {self.feature_type.upper()} FEATURES AND {self.gnn_type.upper()} MODEL".center(70))
        print("=" * 70)
        
        # 1. Find the mainshock
        self.find_mainshock()
        
        # 2. Create relative coordinate dataframe
        self.create_relative_coordinate_dataframe()
        
        # 3. Prepare graph dataset
        graph_dataset = self.prepare_dataset()
        print(f"Prepared graph dataset with {len(graph_dataset)} samples")

        # Add the debug code here
        print("Checking feature dimensions in dataset...")
        feature_dims = [data.x.shape[1] for data in graph_dataset]
        unique_dims = set(feature_dims)
        if len(unique_dims) > 1:
            print(f"Warning: Found inconsistent feature dimensions: {unique_dims}")
            
            # Show how many samples have each dimension
            for dim in unique_dims:
                count = feature_dims.count(dim)
                print(f"  Dimension {dim}: {count} samples")
        
        # 4. Train GNN model
        model, test_dataset = self.train_gnn_models(
            graph_dataset, perform_feature_importance=perform_feature_importance
        )
        
        # 5. Visualize predictions
        true_abs, pred_abs, errors = self.visualize_predictions_geographic(
            model, test_dataset
        )
        
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"\nTotal execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)"
        )
        
        # Return results dictionary
        results = {
            "models": self.models,
            "feature_importances": self.feature_importances,
            "mainshock": self.mainshock,
            "aftershocks_df": self.aftershocks_df,
            "test_results": {
                "true_absolute": true_abs,
                "pred_absolute": pred_abs,
                "errors": errors,
            },
            "validation_results": self.validation_results,
        }
        
        return results

    def validate_data_integrity(self):
        """Validation method using the existing XGBoost helper"""
        validated, issues = self.xgb_helper.validate_data_integrity()
        self.validation_results["data_integrity"] = self.xgb_helper.validation_results["data_integrity"]
        return validated, issues


def compare_gnn_types(
    data_file,
    validation_level="full",
    approach="multi_station",
    results_dir="gnn_results",
    feature_type="all",
):
    """
    Compare different GNN architectures (GCN, GraphSAGE, GAT) on the same dataset
    
    Args:
        data_file: Path to data file
        validation_level: Level of validation to apply
        approach: Analysis approach to use
        results_dir: Directory to save results
        feature_type: Type of features to use
        
    Returns:
        Comparison DataFrame and results dictionary
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"COMPARING GNN ARCHITECTURES FOR {approach.upper()} APPROACH".center(70))
    print(f"USING {feature_type.upper()} FEATURES".center(70))
    print("=" * 70)
    
    results = {}
    error_metrics = {}
    gnn_types = ["gcn", "sage", "gat"]
    
    for gnn_type in gnn_types:
        print(f"\nRunning with {gnn_type.upper()} architecture...")
        
        predictor = GNNAfterShockPredictor(
            data_file=data_file,
            validation_level=validation_level,
            approach=approach,
            feature_type=feature_type,
            gnn_type=gnn_type,
        )
        
        results[gnn_type] = predictor.run_complete_workflow()
        
        # Extract error metrics
        error_metrics[gnn_type] = {
            "X Error": results[gnn_type]["test_results"]["errors"]["lon"].mean(),
            "Y Error": results[gnn_type]["test_results"]["errors"]["lat"].mean(),
            "Z Error": results[gnn_type]["test_results"]["errors"]["depth"].mean(),
            "3D Mean Error": results[gnn_type]["test_results"]["errors"]["3d"].mean(),
            "3D Median Error": np.median(results[gnn_type]["test_results"]["errors"]["3d"]),
        }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(error_metrics).T
    
    # Print comparison
    print("\n" + "=" * 70)
    print(f"GNN ARCHITECTURE COMPARISON SUMMARY".center(70))
    print("=" * 70)
    
    print("\nError Metrics by GNN Type:")
    print(comparison_df)
    
    # Visualize comparison
    plt.figure(figsize=(14, 8))
    bar_width = 0.25
    index = np.arange(5)  # 5 error metrics
    
    for i, gnn_type in enumerate(gnn_types):
        offset = (i - 1) * bar_width
        plt.bar(
            index + offset,
            comparison_df.loc[
                gnn_type,
                ["X Error", "Y Error", "Z Error", "3D Mean Error", "3D Median Error"],
            ],
            bar_width,
            label=f"{gnn_type.upper()}",
        )
    
    plt.xlabel("Error Metric")
    plt.ylabel("Error (km)")
    plt.title(f"GNN Architecture Comparison ({approach.replace('_', ' ').title()} Approach)")
    plt.xticks(index, ["X Error", "Y Error", "Z Error", "3D Mean", "3D Median"])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{results_dir}/gnn_architecture_comparison_{approach}.png", dpi=300)
    plt.tight_layout()
    
    # Create 3D error distribution comparison plot
    plt.figure(figsize=(12, 6))
    colors = ["blue", "red", "green"]
    
    for i, gnn_type in enumerate(gnn_types):
        sns.histplot(
            results[gnn_type]["test_results"]["errors"]["3d"],
            kde=True,
            color=colors[i],
            alpha=0.4,
            label=f"{gnn_type.upper()}",
        )
        plt.axvline(
            np.mean(results[gnn_type]["test_results"]["errors"]["3d"]),
            color=colors[i],
            linestyle="--",
            label=f"{gnn_type.upper()} Mean: {np.mean(results[gnn_type]['test_results']['errors']['3d']):.2f} km",
        )
    
    plt.xlabel("3D Error (km)")
    plt.ylabel("Frequency")
    plt.title(f"GNN: Error Distribution by Architecture ({approach.replace('_', ' ').title()} Approach)")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.7)
    plt.savefig(f"{results_dir}/gnn_error_distribution_comparison_{approach}.png", dpi=300)
    plt.tight_layout()
    
    print(f"\nGNN architecture comparison results saved to {results_dir}/")
    
    return comparison_df, results


def compare_with_xgboost(
    data_file,
    validation_level="full",
    approach="multi_station",
    results_dir="comparison_results",
    feature_type="all",
):
    """
    Compare GNN (GAT) with XGBoost on the same dataset
    
    Args:
        data_file: Path to data file
        validation_level: Level of validation to apply
        approach: Analysis approach to use
        results_dir: Directory to save results
        feature_type: Type of features to use
        
    Returns:
        Comparison DataFrame and results dictionary
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"COMPARING GNN AND XGBOOST FOR {approach.upper()} APPROACH".center(70))
    print(f"USING {feature_type.upper()} FEATURES".center(70))
    print("=" * 70)
    
    # Run XGBoost
    print("\nRunning XGBoost...")
    xgb_predictor = XGBoostAfterShockPredictor(
        data_file=data_file,
        validation_level=validation_level,
        approach=approach,
        feature_type=feature_type,
    )
    xgb_results = xgb_predictor.run_complete_workflow(perform_shap=False)
    
    # Run GNN (GAT)
    print("\nRunning GNN (GAT)...")
    gnn_predictor = GNNAfterShockPredictor(
        data_file=data_file,
        validation_level=validation_level,
        approach=approach,
        feature_type=feature_type,
        gnn_type="gat",
    )
    gnn_results = gnn_predictor.run_complete_workflow()
    
    # Extract error metrics
    error_metrics = {
        "XGBoost": {
            "X Error": xgb_results["test_results"]["errors"]["lon"].mean(),
            "Y Error": xgb_results["test_results"]["errors"]["lat"].mean(),
            "Z Error": xgb_results["test_results"]["errors"]["depth"].mean(),
            "3D Mean Error": xgb_results["test_results"]["errors"]["3d"].mean(),
            "3D Median Error": np.median(xgb_results["test_results"]["errors"]["3d"]),
        },
        "GNN (GAT)": {
            "X Error": gnn_results["test_results"]["errors"]["lon"].mean(),
            "Y Error": gnn_results["test_results"]["errors"]["lat"].mean(),
            "Z Error": gnn_results["test_results"]["errors"]["depth"].mean(),
            "3D Mean Error": gnn_results["test_results"]["errors"]["3d"].mean(),
            "3D Median Error": np.median(gnn_results["test_results"]["errors"]["3d"]),
        },
    }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(error_metrics).T
    
    # Calculate improvement
    comparison_df["Improvement (%)"] = (
        (comparison_df["3D Mean Error"]["XGBoost"] - comparison_df["3D Mean Error"]["GNN (GAT)"])
        / comparison_df["3D Mean Error"]["XGBoost"]
        * 100
    )
    
    # Print comparison
    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON SUMMARY".center(70))
    print("=" * 70)
    
    print("\nError Metrics by Model:")
    print(comparison_df)
    
    # Visualize comparison
    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(5)  # 5 error metrics
    
    plt.bar(
        index - bar_width / 2,
        comparison_df.loc["XGBoost", ["X Error", "Y Error", "Z Error", "3D Mean Error", "3D Median Error"]],
        bar_width,
        label="XGBoost",
    )
    plt.bar(
        index + bar_width / 2,
        comparison_df.loc["GNN (GAT)", ["X Error", "Y Error", "Z Error", "3D Mean Error", "3D Median Error"]],
        bar_width,
        label="GNN (GAT)",
    )
    
    plt.xlabel("Error Metric")
    plt.ylabel("Error (km)")
    plt.title(f"XGBoost vs GNN Comparison ({approach.replace('_', ' ').title()} Approach)")
    plt.xticks(index, ["X Error", "Y Error", "Z Error", "3D Mean", "3D Median"])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{results_dir}/xgboost_vs_gnn_comparison_{approach}.png", dpi=300)
    plt.tight_layout()
    
    # Create 3D error distribution comparison plot
    plt.figure(figsize=(12, 6))
    
    sns.histplot(
        xgb_results["test_results"]["errors"]["3d"],
        kde=True,
        color="blue",
        alpha=0.5,
        label="XGBoost",
    )
    sns.histplot(
        gnn_results["test_results"]["errors"]["3d"],
        kde=True,
        color="red",
        alpha=0.5,
        label="GNN (GAT)",
    )
    plt.axvline(
        np.mean(xgb_results["test_results"]["errors"]["3d"]),
        color="blue",
        linestyle="--",
        label=f"XGBoost Mean: {np.mean(xgb_results['test_results']['errors']['3d']):.2f} km",
    )
    plt.axvline(
        np.mean(gnn_results["test_results"]["errors"]["3d"]),
        color="red",
        linestyle="--",
        label=f"GNN Mean: {np.mean(gnn_results['test_results']['errors']['3d']):.2f} km",
    )
    
    plt.xlabel("3D Error (km)")
    plt.ylabel("Frequency")
    plt.title(f"XGBoost vs GNN: Error Distribution ({approach.replace('_', ' ').title()} Approach)")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.7)
    plt.savefig(f"{results_dir}/xgboost_vs_gnn_error_distribution_{approach}.png", dpi=300)
    plt.tight_layout()
    
    print(f"\nComparison results saved to {results_dir}/")
    
    return comparison_df, {"xgboost": xgb_results, "gnn": gnn_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GNN models for aftershock location prediction"
    )
    parser.add_argument(
        "--data", required=True, help="Path to pickle file with preprocessed data"
    )
    parser.add_argument(
        "--validation",
        choices=["none", "critical", "full"],
        default="full",
        help="Validation level (default: full)",
    )
    parser.add_argument(
        "--approach",
        choices=["best_station", "multi_station"],
        default="multi_station",
        help="Analysis approach (default: multi_station)",
    )
    parser.add_argument(
        "--feature-type",
        choices=["all", "physics", "signal"],
        default="all",
        help="Type of features to use (default: all)",
    )
    parser.add_argument(
        "--gnn-type",
        choices=["gcn", "sage", "gat", "compare"],
        default="gat",
        help="GNN architecture to use (default: gat, 'compare' to test all)",
    )
    parser.add_argument(
        "--compare-xgboost",
        action="store_true",
        help="Compare GNN with XGBoost",
    )
    parser.add_argument(
        "--results-dir",
        default="gnn_results",
        help="Directory to save results (default: gnn_results)",
    )
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.gnn_type == "compare":
        # Compare different GNN architectures
        comparison_df, results = compare_gnn_types(
            data_file=args.data,
            validation_level=args.validation,
            approach=args.approach,
            results_dir=args.results_dir,
            feature_type=args.feature_type,
        )
    elif args.compare_xgboost:
        # Compare GNN with XGBoost
        comparison_df, results = compare_with_xgboost(
            data_file=args.data,
            validation_level=args.validation,
            approach=args.approach,
            results_dir=args.results_dir,
            feature_type=args.feature_type,
        )
    else:
        # Run a single GNN model
        predictor = GNNAfterShockPredictor(
            data_file=args.data,
            validation_level=args.validation,
            approach=args.approach,
            feature_type=args.feature_type,
            gnn_type=args.gnn_type,
        )
        results = predictor.run_complete_workflow()
    
    print(f"All results saved to {args.results_dir}/")