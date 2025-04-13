import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from tqdm import tqdm
import seisbench.data as sbd
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
import os
from pathlib import Path
from scipy import signal

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class WaveformFeatureExtractor:
    """
    Class to extract features from seismic waveforms
    """

    def __init__(self, sampling_rate=100.0):
        self.sampling_rate = sampling_rate

    def extract_features(self, waveform, p_arrival=None, s_arrival=None):
        """
        Extract features from waveform data

        Parameters:
        -----------
        waveform : numpy.ndarray
            Waveform data with shape (num_components, num_samples)
        p_arrival : int, optional
            P-wave arrival sample
        s_arrival : int, optional
            S-wave arrival sample

        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        # Initialize feature dictionary
        features = {}

        # Basic time-domain features (overall signal)
        for i, component_name in enumerate(["Z", "N", "E"]):
            if i < waveform.shape[0]:  # Check if component exists
                component = waveform[i]

                # Calculate basic statistics
                features[f"{component_name}_max"] = np.max(np.abs(component))
                features[f"{component_name}_mean"] = np.mean(np.abs(component))
                features[f"{component_name}_std"] = np.std(component)
                features[f"{component_name}_rms"] = np.sqrt(np.mean(component**2))
                features[f"{component_name}_energy"] = np.sum(component**2)
                features[f"{component_name}_kurtosis"] = self._kurtosis(component)

                # Calculate frequency-domain features
                freq_features = self._compute_frequency_features(component)
                for feat_name, feat_value in freq_features.items():
                    features[f"{component_name}_{feat_name}"] = feat_value

        # Extract P-wave and S-wave specific features if arrivals are provided
        if p_arrival is not None and s_arrival is not None:
            p_window_size = 100  # 100 samples after P arrival
            s_window_size = 100  # 100 samples after S arrival

            # Ensure window sizes don't exceed waveform length
            p_window_size = min(p_window_size, waveform.shape[1] - p_arrival)
            s_window_size = min(s_window_size, waveform.shape[1] - s_arrival)

            if p_window_size > 0:
                for i, component_name in enumerate(["Z", "N", "E"]):
                    if i < waveform.shape[0]:  # Check if component exists
                        p_segment = waveform[i, p_arrival : p_arrival + p_window_size]

                        # Calculate P-wave features
                        features[f"P_{component_name}_max"] = np.max(np.abs(p_segment))
                        features[f"P_{component_name}_mean"] = np.mean(
                            np.abs(p_segment)
                        )
                        features[f"P_{component_name}_std"] = np.std(p_segment)
                        features[f"P_{component_name}_energy"] = np.sum(p_segment**2)

                        # P-wave frequency features
                        p_freq_features = self._compute_frequency_features(p_segment)
                        for feat_name, feat_value in p_freq_features.items():
                            features[f"P_{component_name}_{feat_name}"] = feat_value

            if s_window_size > 0:
                for i, component_name in enumerate(["Z", "N", "E"]):
                    if i < waveform.shape[0]:  # Check if component exists
                        s_segment = waveform[i, s_arrival : s_arrival + s_window_size]

                        # Calculate S-wave features
                        features[f"S_{component_name}_max"] = np.max(np.abs(s_segment))
                        features[f"S_{component_name}_mean"] = np.mean(
                            np.abs(s_segment)
                        )
                        features[f"S_{component_name}_std"] = np.std(s_segment)
                        features[f"S_{component_name}_energy"] = np.sum(s_segment**2)

                        # S-wave frequency features
                        s_freq_features = self._compute_frequency_features(s_segment)
                        for feat_name, feat_value in s_freq_features.items():
                            features[f"S_{component_name}_{feat_name}"] = feat_value

            # Calculate P/S amplitude ratios for each component
            for i, component_name in enumerate(["Z", "N", "E"]):
                if i < waveform.shape[0]:  # Check if component exists
                    if (
                        features.get(f"P_{component_name}_max", 0) > 0
                        and features.get(f"S_{component_name}_max", 0) > 0
                    ):
                        features[f"{component_name}_PS_ratio"] = (
                            features[f"P_{component_name}_max"]
                            / features[f"S_{component_name}_max"]
                        )

        return features

    def _compute_frequency_features(self, signal_segment):
        """
        Compute frequency domain features
        """
        features = {}

        # Check if signal segment is not empty
        if len(signal_segment) < 10:
            return {
                "dominant_freq": 0,
                "low_freq_energy": 0,
                "mid_freq_energy": 0,
                "high_freq_energy": 0,
                "low_high_ratio": 0,
            }

        try:
            # Calculate power spectral density
            f, Pxx = signal.welch(
                signal_segment,
                fs=self.sampling_rate,
                nperseg=min(256, len(signal_segment)),
            )

            # Dominant frequency
            features["dominant_freq"] = f[np.argmax(Pxx)]

            # Frequency band energies
            low_idx = f < 5
            mid_idx = (f >= 5) & (f < 15)
            high_idx = f >= 15

            # Handle empty frequency bands
            features["low_freq_energy"] = np.sum(Pxx[low_idx]) if np.any(low_idx) else 0
            features["mid_freq_energy"] = np.sum(Pxx[mid_idx]) if np.any(mid_idx) else 0
            features["high_freq_energy"] = (
                np.sum(Pxx[high_idx]) if np.any(high_idx) else 0
            )

            # Calculate frequency ratios
            if features["high_freq_energy"] > 0:
                features["low_high_ratio"] = (
                    features["low_freq_energy"] / features["high_freq_energy"]
                )
            else:
                features["low_high_ratio"] = 0

        except Exception as e:
            print(f"Error computing frequency features: {e}")
            features = {
                "dominant_freq": 0,
                "low_freq_energy": 0,
                "mid_freq_energy": 0,
                "high_freq_energy": 0,
                "low_high_ratio": 0,
            }

        return features

    def _kurtosis(self, x):
        """Calculate kurtosis of a signal"""
        n = len(x)
        if n < 4:
            return 0

        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            return 0

        kurt = np.mean(((x - mean) / std) ** 4) - 3
        return kurt


class AfterShockGNN(torch.nn.Module):
    """
    Enhanced Graph Attention Network for aftershock prediction
    that incorporates waveform features
    """

    def __init__(
        self,
        metadata_channels,
        waveform_channels,
        hidden_channels,
        num_layers,
        dropout=0.3,
    ):
        super(AfterShockGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Separate encoders for metadata and waveform features
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.waveform_encoder = nn.Sequential(
            nn.Linear(waveform_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combine metadata and waveform features
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(dropout)
        )

        # Graph Attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(hidden_channels, hidden_channels, heads=2, dropout=dropout)
        )

        # For subsequent layers, input is hidden_channels * 4 (from 4 attention heads)
        for i in range(1, num_layers):
            self.gat_layers.append(
                GATConv(hidden_channels * 2, hidden_channels, heads=2, dropout=dropout)
            )

        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_channels * 2)
            )  # *4 for the 4 attention heads

        # Output layers for predicting latitude and longitude
        self.lat_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        self.lon_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, metadata, waveform_features, edge_index, batch):
        """
        Forward pass of the GNN model with safety checks for batch tensor

        Parameters:
        -----------
        metadata : torch.Tensor
            Metadata features (lat, lon, depth, time)
        waveform_features : torch.Tensor
            Waveform features extracted from seismic signals
        edge_index : torch.Tensor
            Edge indices defining the graph structure
        batch : torch.Tensor
            Batch assignment for nodes

        Returns:
        --------
        lat, lon : torch.Tensor
            Predicted latitude and longitude
        """
        # Encode metadata features
        metadata_encoded = self.metadata_encoder(metadata)

        # Encode waveform features
        waveform_encoded = self.waveform_encoder(waveform_features)

        # Combine features
        x = torch.cat([metadata_encoded, waveform_encoded], dim=1)
        x = self.feature_combiner(x)

        # Apply GAT layers
        for i in range(self.num_layers):
            # Store original x for residual connection if shapes match
            x_res = x if i > 0 else None

            # Apply GAT layer
            x = self.gat_layers[i](x, edge_index)

            # Apply batch normalization
            x = self.batch_norms[i](x)

            # Apply residual connection if shapes match
            if x_res is not None and x.size() == x_res.size():
                x = x + x_res

            # Apply ReLU activation
            x = F.relu(x)

            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        # SAFETY CHECK: Ensure batch tensor has the right size
        if batch.size(0) != x.size(0):
            print(f"WARNING: Batch size mismatch. batch: {batch.size()}, x: {x.size()}")
            # Resize batch tensor or x to match
            if batch.size(0) < x.size(0):
                # Extend batch tensor by repeating the last element
                extension = batch[-1].repeat(x.size(0) - batch.size(0))
                batch = torch.cat([batch, extension])
            else:
                # Truncate batch tensor
                batch = batch[: x.size(0)]

        # Aggregate graph-level representation
        x_graph = global_mean_pool(x, batch)

        # Predict latitude and longitude
        lat = self.lat_predictor(x_graph)
        lon = self.lon_predictor(x_graph)

        return lat, lon


def load_aftershock_data_with_waveforms(max_waveforms=1000):
    """
    Load and preprocess aftershock data from the Iquique dataset,
    including waveform data
    """
    print("Loading Iquique dataset using SeisBench...")
    iquique = sbd.Iquique()

    # Get metadata
    metadata = iquique.metadata.copy()

    # Filter out rows with missing essential data
    metadata = metadata.dropna(
        subset=[
            "source_origin_time",
            "source_latitude_deg",
            "source_longitude_deg",
            "source_depth_km",
        ]
    )

    # Convert timestamps
    metadata["datetime"] = pd.to_datetime(metadata["source_origin_time"])

    # Sort by time
    metadata = metadata.sort_values("datetime")

    # Create a dictionary to store waveform features
    waveform_features_dict = {}

    # Initialize feature extractor
    feature_extractor = WaveformFeatureExtractor()

    # Limit the number of waveforms to process
    sample_indices = metadata.index[: min(max_waveforms, len(metadata))]

    print(f"Extracting waveform features for {len(sample_indices)} events...")
    for idx in tqdm(sample_indices):
        try:
            # Get waveform data
            waveform = iquique.get_waveforms(int(idx))

            # Get P and S arrival samples if available
            p_arrival = metadata.loc[idx, "trace_P_arrival_sample"]
            s_arrival = metadata.loc[idx, "trace_S_arrival_sample"]

            # Validate P and S arrivals
            if pd.isna(p_arrival) or pd.isna(s_arrival):
                p_arrival, s_arrival = None, None
            else:
                p_arrival = int(p_arrival)
                s_arrival = int(s_arrival)

            # Extract features
            features = feature_extractor.extract_features(
                waveform, p_arrival, s_arrival
            )

            # Store features
            waveform_features_dict[idx] = features

        except Exception as e:
            print(f"Error processing waveform {idx}: {e}")
            waveform_features_dict[idx] = {}

    print(
        f"Successfully extracted features for {len(waveform_features_dict)} waveforms"
    )

    return metadata, iquique, waveform_features_dict


def identify_mainshock_and_aftershocks(metadata):
    """
    Identify the mainshock and its associated aftershocks based on data patterns
    """
    # The known Iquique earthquake was in early April 2014
    # Let's create a date range for the expected period
    april_start = pd.Timestamp("2014-04-01", tz="UTC")
    april_end = pd.Timestamp("2014-04-05", tz="UTC")

    # Find events in early April 2014
    april_events = metadata[
        (metadata["datetime"] >= april_start) & (metadata["datetime"] <= april_end)
    ]

    if len(april_events) == 0:
        print(
            "No events found in early April 2014 timeframe. Using alternative approach."
        )
        # Alternative approach: find the earliest events in the dataset
        metadata_sorted = metadata.sort_values("datetime")
        earliest_date = metadata_sorted["datetime"].iloc[0]
        print(f"Earliest event in dataset: {earliest_date}")

        # Take the earliest events as potential mainshock candidates
        potential_mainshocks = metadata_sorted.iloc[:100]  # First 100 events

        # Find which event has most events in the following 24 hours
        best_event_idx = None
        most_followers = 0

        for idx, row in potential_mainshocks.iterrows():
            event_time = row["datetime"]
            next_24h = event_time + pd.Timedelta(hours=24)
            followers = sum(metadata["datetime"].between(event_time, next_24h))

            if followers > most_followers:
                most_followers = followers
                best_event_idx = idx

        mainshock = metadata.loc[best_event_idx]
    else:
        print(f"Found {len(april_events)} events in early April 2014.")
        # Within this timeframe, find the event with the most followers in 24 hours
        best_event_idx = None
        most_followers = 0

        for idx, row in april_events.iterrows():
            event_time = row["datetime"]
            next_24h = event_time + pd.Timedelta(hours=24)
            followers = sum(metadata["datetime"].between(event_time, next_24h))

            if followers > most_followers:
                most_followers = followers
                best_event_idx = idx

        mainshock = metadata.loc[best_event_idx]

    print(
        f"Identified potential mainshock at {mainshock['datetime']} at location "
        f"({mainshock['source_latitude_deg']}, {mainshock['source_longitude_deg']}), "
        f"depth {mainshock['source_depth_km']} km"
    )
    print(f"This event is followed by {most_followers} events in the next 24 hours")

    # Select events after the mainshock as aftershocks
    aftershocks = metadata[metadata["datetime"] > mainshock["datetime"]].copy()
    print(f"Found {len(aftershocks)} aftershocks")

    # Create a new feature: time since mainshock in hours
    aftershocks["hours_since_mainshock"] = (
        aftershocks["datetime"] - mainshock["datetime"]
    ).dt.total_seconds() / 3600

    # Create features: distance from mainshock (approximate using Haversine)
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c

        return distance

    aftershocks["distance_from_mainshock_km"] = aftershocks.apply(
        lambda row: haversine_distance(
            mainshock["source_latitude_deg"],
            mainshock["source_longitude_deg"],
            row["source_latitude_deg"],
            row["source_longitude_deg"],
        ),
        axis=1,
    )

    return mainshock, aftershocks


def create_aftershock_sequences_with_waveforms(
    aftershocks,
    waveform_features_dict,
    sequence_length=5,
    time_window_hours=72,
    max_sequences=5000,
):
    """
    Create aftershock sequences with waveform features using a sliding window approach
    """
    sequences = []
    total_aftershocks = len(aftershocks)

    # Check which aftershocks have waveform features
    aftershocks_with_features = aftershocks[
        aftershocks["event_id"].isin(waveform_features_dict.keys())
    ]

    print(f"Total aftershocks: {total_aftershocks}")
    print(f"Aftershocks with waveform features: {len(aftershocks_with_features)}")

    # Need at least sequence_length+1 events to create a sequence with a target
    if len(aftershocks_with_features) < sequence_length + 1:
        print(
            f"Not enough aftershocks with waveform features to create sequences (need {sequence_length + 1}, have {len(aftershocks_with_features)})"
        )
        return []

    # Sort aftershocks by time
    aftershocks_sorted = aftershocks_with_features.sort_values("hours_since_mainshock")

    # Use a sliding window approach with a smaller step size
    step_size = 1  # Create a sequence starting at each event

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

        # Check for sufficient spatial variation (with reduced threshold)
        lats = current_sequence["source_latitude_deg"].values
        lons = current_sequence["source_longitude_deg"].values

        lat_range = np.max(lats) - np.min(lats)
        lon_range = np.max(lons) - np.min(lons)

        # Reduced minimum variation threshold
        min_variation = 0.02  # ~2 km at this latitude
        if lat_range < min_variation and lon_range < min_variation:
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

        for _, row in current_sequence.iterrows():
            event_id = row["event_id"]  # Use event_id instead of index
            if event_id in waveform_features_dict:
                features = waveform_features_dict[event_id]

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

        # Target is the location of the next aftershock
        target = np.array(
            [
                target_aftershock["source_latitude_deg"],
                target_aftershock["source_longitude_deg"],
            ]
        )

        sequences.append((metadata_features, sequence_waveform_features, target))

        # Limit the number of sequences to avoid memory issues
        if len(sequences) >= max_sequences:
            print(f"Reached maximum number of sequences ({max_sequences})")
            break

    print(f"Created {len(sequences)} aftershock sequences with waveform features")
    return sequences


def consolidate_station_recordings(metadata, waveform_features_dict):
    """
    Consolidate multiple station recordings of the same event into a single representation
    """
    # Create event IDs based on source parameters
    metadata["lat_rounded"] = np.round(metadata["source_latitude_deg"], 4)
    metadata["lon_rounded"] = np.round(metadata["source_longitude_deg"], 4)
    metadata["depth_rounded"] = np.round(metadata["source_depth_km"], 1)

    metadata["event_id"] = metadata.groupby(
        ["source_origin_time", "lat_rounded", "lon_rounded", "depth_rounded"]
    ).ngroup()

    print(
        f"Original recordings: {len(metadata)}, Unique events: {metadata['event_id'].nunique()}"
    )

    # Create consolidated representations
    consolidated_metadata = []
    consolidated_features = {}

    # Track how many events actually have features
    events_with_features = 0

    for event_id, group in metadata.groupby("event_id"):
        # Find recordings with valid waveform features
        valid_recordings = []
        for idx in group.index:
            if idx in waveform_features_dict and waveform_features_dict[idx]:
                valid_recordings.append(idx)

        # If we have recordings with features, use one of them
        if valid_recordings:
            best_idx = valid_recordings[0]  # Just take the first valid one
            events_with_features += 1
        else:
            # If no recording has features, just take the first index
            best_idx = group.index[0]

        # Take metadata from best recording
        best_record = group.loc[best_idx].copy()
        best_record["station_count"] = len(group)
        best_record["event_id"] = event_id  # Keep event_id in the metadata
        consolidated_metadata.append(best_record)

        # Map waveform features to the event_id
        if best_idx in waveform_features_dict and waveform_features_dict[best_idx]:
            consolidated_features[event_id] = waveform_features_dict[best_idx]

    print(f"Events with valid waveform features: {events_with_features}")

    # Convert to DataFrame
    consolidated_metadata = pd.DataFrame(consolidated_metadata)
    return consolidated_metadata, consolidated_features


def train_with_spatial_regularization(
    train_dataset, 
    test_dataset,
    model_class,
    metadata_channels,
    waveform_channels,
    hidden_channels=128,
    num_layers=3,
    batch_size=4,
    learning_rate=0.0002,  # Reduced learning rate
    coverage_weight=0.2,   # Weight for spatial coverage regularization
    device="cpu"
):
    """
    Train with a custom loss function that encourages spatial diversity in predictions
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch_geometric.loader import DataLoader
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = model_class(
        metadata_channels=metadata_channels,
        waveform_channels=waveform_channels,
        edge_attr_channels=3,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
    ).to(device)
    
    # Initialize optimizer with reduced learning rate and increased weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Use a more aggressive learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-5
    )
    
    # Training loop
    num_epochs = 150  # Increased epochs
    patience = 20     # Increased patience
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    # Track metrics
    train_losses = []
    val_losses = []
    coverage_losses = []
    
    print(f"Starting training with spatial regularization...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_coverage_loss = 0
        batch_count = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            lat_pred, lon_pred = model(
                data.metadata, 
                data.waveform, 
                data.edge_index,
                data.edge_attr,
                data.batch
            )
            
            # Target values
            y = data.y.view(-1, 2)
            lat_true = y[:, 0]
            lon_true = y[:, 1]
            
            # Basic MSE loss
            lat_loss = torch.nn.functional.mse_loss(lat_pred.squeeze(), lat_true)
            lon_loss = torch.nn.functional.mse_loss(lon_pred.squeeze(), lon_true)
            mse_loss = lat_loss + lon_loss
            
            # Spatial coverage regularization - encourage diversity
            if lat_pred.size(0) > 1:  # Only if batch has multiple examples
                # Calculate variance of predictions vs. variance of targets
                pred_lat_var = torch.var(lat_pred.squeeze())
                pred_lon_var = torch.var(lon_pred.squeeze())
                true_lat_var = torch.var(lat_true)
                true_lon_var = torch.var(lon_true)
                
                # Penalize when prediction variance is less than target variance
                # This encourages model to match the spatial distribution
                coverage_loss = torch.abs(true_lat_var - pred_lat_var) + torch.abs(true_lon_var - pred_lon_var)
                
                # Combine losses
                loss = mse_loss + coverage_weight * coverage_loss
                epoch_coverage_loss += coverage_loss.item()
            else:
                # Just use MSE for single-example batches
                loss = mse_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += mse_loss.item()  # Track only the MSE part for comparison
            batch_count += 1
        
        # Calculate average loss
        avg_train_loss = epoch_loss / batch_count
        avg_coverage_loss = epoch_coverage_loss / batch_count if batch_count > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                
                # Forward pass
                lat_pred, lon_pred = model(
                    data.metadata, 
                    data.waveform, 
                    data.edge_index,
                    data.edge_attr,
                    data.batch
                )
                
                # Target values
                y = data.y.view(-1, 2)
                lat_true = y[:, 0]
                lon_true = y[:, 1]
                
                # Validation loss (MSE only)
                lat_loss = torch.nn.functional.mse_loss(lat_pred.squeeze(), lat_true)
                lon_loss = torch.nn.functional.mse_loss(lon_pred.squeeze(), lon_true)
                loss = lat_loss + lon_loss
                
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Track metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        coverage_losses.append(avg_coverage_loss)
        
        # Early stopping with increased patience
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Coverage Loss: {avg_coverage_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Plot training metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(coverage_losses, label='Coverage Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Coverage Loss')
    plt.title('Spatial Coverage Regularization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_metrics.png')
    plt.close()
    
    return model, (train_losses, val_losses, coverage_losses)


def evaluate_with_spatial_binning(model, test_dataset, mainshock, device="cpu"):
    """
    Enhanced evaluation with spatial binning analysis to better understand prediction patterns
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch_geometric.loader import DataLoader
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec
    
    # Create data loader with batch size 1 for clearer analysis
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Lists to store actual and predicted coordinates
    actual_lats = []
    actual_lons = []
    pred_lats = []
    pred_lons = []
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Forward pass
            lat_pred, lon_pred = model(
                data.metadata, 
                data.waveform, 
                data.edge_index,
                data.edge_attr,
                data.batch
            )
            
            # Extract predictions and targets
            lat_np = lat_pred.cpu().numpy().flatten()
            lon_np = lon_pred.cpu().numpy().flatten()
            
            # Get targets
            targets = data.y.cpu().numpy()
            
            # Store predictions and targets
            pred_lats.extend(lat_np)
            pred_lons.extend(lon_np)
            actual_lats.extend(targets[:, 0])
            actual_lons.extend(targets[:, 1])
    
    # Calculate error in kilometers
    errors_km = []
    for i in range(len(actual_lats)):
        # Haversine distance
        R = 6371  # Earth radius in kilometers
        lat1, lon1 = np.radians(actual_lats[i]), np.radians(actual_lons[i])
        lat2, lon2 = np.radians(pred_lats[i]), np.radians(pred_lons[i])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        errors_km.append(distance)
    
    mean_error = np.mean(errors_km)
    median_error = np.median(errors_km)
    percentile_90 = np.percentile(errors_km, 90)
    
    print(f"Enhanced Spatial Evaluation Results:")
    print(f"Mean Error: {mean_error:.2f} km")
    print(f"Median Error: {median_error:.2f} km")
    print(f"90th Percentile Error: {percentile_90:.2f} km")
    
    # Create comprehensive spatial analysis figure
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8])
    
    # 1. Basic map with predictions and actual points
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(
        mainshock["source_longitude_deg"],
        mainshock["source_latitude_deg"],
        s=200,
        c="red",
        marker="*",
        label="Mainshock",
        edgecolor="black",
        zorder=10
    )
    
    ax1.scatter(
        actual_lons,
        actual_lats,
        s=60,
        c="blue",
        alpha=0.7,
        label="Actual Aftershocks",
        edgecolor="black"
    )
    
    ax1.scatter(
        pred_lons,
        pred_lats,
        s=40,
        c="green",
        alpha=0.7,
        marker="x",
        label="Predicted Aftershocks"
    )
    
    # Connect actual to predicted with lines
    for i in range(len(actual_lats)):
        ax1.plot(
            [actual_lons[i], pred_lons[i]],
            [actual_lats[i], pred_lats[i]],
            "k-",
            alpha=0.2
        )
    
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Spatial Distribution: Actual vs Predicted Aftershocks")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Heatmap of errors
    ax2 = plt.subplot(gs[0, 1])
    sc = ax2.scatter(
        actual_lons,
        actual_lats,
        s=100,
        c=errors_km,
        cmap="viridis_r",
        alpha=0.8,
        edgecolor="black"
    )
    
    ax2.scatter(
        mainshock["source_longitude_deg"],
        mainshock["source_latitude_deg"],
        s=200,
        c="red",
        marker="*",
        label="Mainshock",
        edgecolor="black",
        zorder=10
    )
    
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Prediction Error (km)")
    
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Spatial Distribution of Prediction Errors")
    ax2.grid(True, alpha=0.3)
    
    # 3. Kernel density estimation (KDE) for actual points
    ax3 = plt.subplot(gs[1, 0])
    
    # Combine lat/lon for KDE
    actual_points = np.vstack([actual_lons, actual_lats])
    
    # Calculate KDE if we have enough points
    if len(actual_lats) > 3:
        kde = gaussian_kde(actual_points)
        
        # Create a grid for evaluation
        min_lon, max_lon = min(actual_lons) - 0.1, max(actual_lons) + 0.1
        min_lat, max_lat = min(actual_lats) - 0.1, max(actual_lats) + 0.1
        
        x_grid, y_grid = np.mgrid[min_lon:max_lon:100j, min_lat:max_lat:100j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        
        # Evaluate KDE
        z = kde(positions)
        z = z.reshape(x_grid.shape)
        
        # Plot KDE
        ax3.contourf(x_grid, y_grid, z, cmap='Blues', alpha=0.8)
        
    # Plot points on top
    ax3.scatter(
        actual_lons,
        actual_lats,
        s=30,
        c="blue",
        alpha=0.5,
        label="Actual Aftershocks"
    )
    
    ax3.scatter(
        mainshock["source_longitude_deg"],
        mainshock["source_latitude_deg"],
        s=200,
        c="red",
        marker="*",
        label="Mainshock",
        edgecolor="black",
        zorder=10
    )
    
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title("Density of Actual Aftershocks")
    ax3.grid(True, alpha=0.3)
    
    # 4. KDE for predicted points
    ax4 = plt.subplot(gs[1, 1])
    
    # Combine lat/lon for KDE
    pred_points = np.vstack([pred_lons, pred_lats])
    
    # Calculate KDE if we have enough points
    if len(pred_lats) > 3:
        kde = gaussian_kde(pred_points)
        
        # Create a grid for evaluation
        min_lon, max_lon = min(pred_lons) - 0.1, max(pred_lons) + 0.1
        min_lat, max_lat = min(pred_lats) - 0.1, max(pred_lats) + 0.1
        
        x_grid, y_grid = np.mgrid[min_lon:max_lon:100j, min_lat:max_lat:100j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        
        # Evaluate KDE
        z = kde(positions)
        z = z.reshape(x_grid.shape)
        
        # Plot KDE
        ax4.contourf(x_grid, y_grid, z, cmap='Greens', alpha=0.8)
    
    # Plot points on top
    ax4.scatter(
        pred_lons,
        pred_lats,
        s=30,
        c="green",
        alpha=0.5,
        label="Predicted Aftershocks"
    )
    
    ax4.scatter(
        mainshock["source_longitude_deg"],
        mainshock["source_latitude_deg"],
        s=200,
        c="red",
        marker="*",
        label="Mainshock",
        edgecolor="black",
        zorder=10
    )
    
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    ax4.set_title("Density of Predicted Aftershocks")
    ax4.grid(True, alpha=0.3)
    
    # 5. Error distribution histogram
    ax5 = plt.subplot(gs[2, :])
    ax5.hist(errors_km, bins=20, alpha=0.7, color='purple')
    ax5.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.2f} km')
    ax5.axvline(median_error, color='green', linestyle='--', label=f'Median: {median_error:.2f} km')
    ax5.axvline(percentile_90, color='blue', linestyle='--', label=f'90th Percentile: {percentile_90:.2f} km')
    
    ax5.set_xlabel('Error (km)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Prediction Errors')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/enhanced_spatial_analysis.png', dpi=300)
    plt.close()
    
    # Create a spatial binning analysis
    plt.figure(figsize=(12, 10))
    
    # Define grid for binning
    lat_bins = np.linspace(min(actual_lats) - 0.1, max(actual_lats) + 0.1, 10)
    lon_bins = np.linspace(min(actual_lons) - 0.1, max(actual_lons) + 0.1, 10)
    
    # Create 2D histogram
    h, _, _, _ = plt.hist2d(
        actual_lons,
        actual_lats,
        bins=[lon_bins, lat_bins],
        cmap='Blues',
        alpha=0.7,
        norm=LogNorm()
    )
    
    plt.colorbar(label='Actual Aftershock Count')
    
    # Overlay predicted points
    plt.scatter(
        pred_lons,
        pred_lats,
        s=30,
        c='green',
        alpha=0.7,
        marker='x',
        label='Predicted'
    )
    
    # Plot mainshock
    plt.scatter(
        mainshock["source_longitude_deg"],
        mainshock["source_latitude_deg"],
        s=200,
        c="red",
        marker="*",
        label="Mainshock",
        edgecolor="black",
        zorder=10
    )
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Binning Analysis: Aftershock Density vs Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/spatial_binning.png', dpi=300)
    plt.close()
    
    # Calculate and print additional statistics
    if len(pred_lats) >= 2:
        # Calculate spatial statistics
        pred_lat_std = np.std(pred_lats)
        pred_lon_std = np.std(pred_lons)
        actual_lat_std = np.std(actual_lats)
        actual_lon_std = np.std(actual_lons)
        
        print(f"\nSpatial Distribution Analysis:")
        print(f"  - Actual latitude std: {actual_lat_std:.4f}°")
        print(f"  - Predicted latitude std: {pred_lat_std:.4f}°")
        print(f"  - Actual longitude std: {actual_lon_std:.4f}°")
        print(f"  - Predicted longitude std: {pred_lon_std:.4f}°")
        
        # Calculate standard deviation ratio (how much of the actual variance is captured)
        lat_std_ratio = pred_lat_std / actual_lat_std if actual_lat_std > 0 else 0
        lon_std_ratio = pred_lon_std / actual_lon_std if actual_lon_std > 0 else 0
        
        print(f"  - Latitude std ratio: {lat_std_ratio:.2f}")
        print(f"  - Longitude std ratio: {lon_std_ratio:.2f}")
        
        # Calculate range coverage
        actual_lat_range = max(actual_lats) - min(actual_lats)
        actual_lon_range = max(actual_lons) - min(actual_lons)
        pred_lat_range = max(pred_lats) - min(pred_lats)
        pred_lon_range = max(pred_lons) - min(pred_lons)
        
        lat_coverage = pred_lat_range / actual_lat_range if actual_lat_range > 0 else 0
        lon_coverage = pred_lon_range / actual_lon_range if actual_lon_range > 0 else 0
        
        print(f"  - Latitude range coverage: {lat_coverage:.2f}")
        print(f"  - Longitude range coverage: {lon_coverage:.2f}")
        
        # Count predictions in actual high-density regions
        if h.size > 0:
            # Find bins with high aftershock density (top 30%)
            high_density_threshold = np.percentile(h[h > 0], 70)
            high_density_bins = h >= high_density_threshold
            
            # Count predictions in high density regions
            pred_in_high_density = 0
            for i in range(len(pred_lons)):
                lon_bin = np.digitize(pred_lons[i], lon_bins) - 1
                lat_bin = np.digitize(pred_lats[i], lat_bins) - 1
                
                # Check if bin is valid and is high density
                if 0 <= lon_bin < len(lon_bins)-1 and 0 <= lat_bin < len(lat_bins)-1:
                    if high_density_bins[lon_bin, lat_bin]:
                        pred_in_high_density += 1
            
            density_hit_rate = pred_in_high_density / len(pred_lons) if len(pred_lons) > 0 else 0
            print(f"  - Predictions in high-density regions: {density_hit_rate:.2f}")
    
    # Return error metrics and spatial data
    return {
        "mean_error": mean_error,
        "median_error": median_error,
        "percentile_90": percentile_90,
        "errors_km": errors_km,
        "spatial_data": {
            "actual_lats": actual_lats,
            "actual_lons": actual_lons,
            "pred_lats": pred_lats,
            "pred_lons": pred_lons
        }
    }

def custom_temporal_train_split(graph_dataset, train_ratio=0.8):
    """
    Split the dataset by time with an adjustment to ensure spatial diversity in both sets
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Get time info from each graph (last node's time feature)
    temporal_info = []
    for i, graph in enumerate(graph_dataset):
        last_time = graph.metadata[-1, 3].item()  # time column
        temporal_info.append((i, last_time))
    
    # Sort by time
    temporal_info.sort(key=lambda x: x[1])
    
    # Get indices in temporal order
    sorted_indices = [idx for idx, _ in temporal_info]
    
    # Split into chunks to ensure spatial diversity
    num_chunks = 10
    chunk_size = len(sorted_indices) // num_chunks
    
    train_indices = []
    test_indices = []
    
    # From each chunk, take train_ratio for training and the rest for testing
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else len(sorted_indices)
        
        chunk_indices = sorted_indices[start_idx:end_idx]
        
        # Split this chunk
        chunk_train_size = int(len(chunk_indices) * train_ratio)
        chunk_train = chunk_indices[:chunk_train_size]
        chunk_test = chunk_indices[chunk_train_size:]
        
        train_indices.extend(chunk_train)
        test_indices.extend(chunk_test)
    
    # Create train and test datasets
    train_dataset = [graph_dataset[i] for i in train_indices]
    test_dataset = [graph_dataset[i] for i in test_indices]
    
    print(f"Created improved temporal split: {len(train_dataset)} training, {len(test_dataset)} testing")
    
    return train_dataset, test_dataset


def evaluate_model_spatial(model, test_dataset, mainshock, device="cpu"):
    """
    Evaluate the model with spatial visualization
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch_geometric.loader import DataLoader
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=1)  # Batch size 1 for clear visualization
    
    # Lists to store actual and predicted coordinates
    actual_lats = []
    actual_lons = []
    pred_lats = []
    pred_lons = []
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Forward pass
            lat_pred, lon_pred = model(
                data.metadata, 
                data.waveform, 
                data.edge_index,
                data.edge_attr,
                data.batch
            )
            
            # Extract predictions and targets
            lat_np = lat_pred.cpu().numpy().flatten()
            lon_np = lon_pred.cpu().numpy().flatten()
            
            # Get targets
            targets = data.y.cpu().numpy()
            
            # Store predictions and targets
            pred_lats.extend(lat_np)
            pred_lons.extend(lon_np)
            actual_lats.extend(targets[:, 0])
            actual_lons.extend(targets[:, 1])
    
    # Calculate error in kilometers
    errors_km = []
    for i in range(len(actual_lats)):
        # Haversine distance
        R = 6371  # Earth radius in kilometers
        lat1, lon1 = np.radians(actual_lats[i]), np.radians(actual_lons[i])
        lat2, lon2 = np.radians(pred_lats[i]), np.radians(pred_lons[i])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        errors_km.append(distance)
    
    mean_error = np.mean(errors_km)
    median_error = np.median(errors_km)
    percentile_90 = np.percentile(errors_km, 90)
    
    print(f"Spatial Evaluation Results:")
    print(f"Mean Error: {mean_error:.2f} km")
    print(f"Median Error: {median_error:.2f} km")
    print(f"90th Percentile Error: {percentile_90:.2f} km")
    
    # Plot spatial predictions
    plt.figure(figsize=(12, 10))
    
    # Plot mainshock
    plt.scatter(
        mainshock["source_longitude_deg"],
        mainshock["source_latitude_deg"],
        s=200,
        c="red",
        marker="*",
        label="Mainshock",
        edgecolor="black",
        zorder=10
    )
    
    # Plot actual aftershocks
    plt.scatter(
        actual_lons,
        actual_lats,
        s=50,
        c="blue",
        alpha=0.7,
        label="Actual Aftershocks",
        edgecolor="black"
    )
    
    # Plot predicted aftershocks
    plt.scatter(
        pred_lons,
        pred_lats,
        s=30,
        c="green",
        alpha=0.7,
        marker="x",
        label="Predicted Aftershocks"
    )
    
    # Connect actual to predicted with lines
    for i in range(len(actual_lats)):
        plt.plot(
            [actual_lons[i], pred_lons[i]],
            [actual_lats[i], pred_lats[i]],
            "k-",
            alpha=0.2
        )
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial Distribution of Actual vs Predicted Aftershocks")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/spatial_predictions.png", dpi=300)
    plt.close()
    
    # Create heatmap of prediction errors
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot with error-based coloring
    sc = plt.scatter(
        actual_lons,
        actual_lats,
        s=100,
        c=errors_km,
        cmap="viridis_r",  # Reverse viridis: blue=good/low error, yellow=bad/high error
        alpha=0.8,
        edgecolor="black"
    )
    
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label("Prediction Error (km)")
    
    # Plot mainshock
    plt.scatter(
        mainshock["source_longitude_deg"],
        mainshock["source_latitude_deg"],
        s=200,
        c="red",
        marker="*",
        label="Mainshock",
        edgecolor="black",
        zorder=10
    )
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial Distribution of Prediction Errors")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/error_heatmap.png", dpi=300)
    plt.close()
    
    # Calculate spatial distribution statistics
    if len(pred_lats) >= 2:  # Need at least 2 points for std
        pred_std_lat = np.std(pred_lats)
        pred_std_lon = np.std(pred_lons)
        actual_std_lat = np.std(actual_lats)
        actual_std_lon = np.std(actual_lons)
        
        print(f"\nSpatial Distribution Analysis:")
        print(f"  - Actual latitude std: {actual_std_lat:.4f}°")
        print(f"  - Predicted latitude std: {pred_std_lat:.4f}°")
        print(f"  - Actual longitude std: {actual_std_lon:.4f}°")
        print(f"  - Predicted longitude std: {pred_std_lon:.4f}°")
        
        # Calculate coverage ratio (how much of the actual area is covered by predictions)
        actual_lat_range = np.max(actual_lats) - np.min(actual_lats)
        actual_lon_range = np.max(actual_lons) - np.min(actual_lons)
        pred_lat_range = np.max(pred_lats) - np.min(pred_lats)
        pred_lon_range = np.max(pred_lons) - np.min(pred_lons)
        
        lat_coverage = pred_lat_range / actual_lat_range if actual_lat_range > 0 else 0
        lon_coverage = pred_lon_range / actual_lon_range if actual_lon_range > 0 else 0
        
        print(f"  - Latitude range coverage: {lat_coverage:.2f}")
        print(f"  - Longitude range coverage: {lon_coverage:.2f}")
    
    return mean_error, median_error, errors_km, (pred_lats, pred_lons, actual_lats, actual_lons)





def analyze_feature_importance(graph_dataset, num_features=90):
    """
    Analyze waveform feature importance using a simple model to identify
    the most predictive features for aftershock location
    """
    import torch
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    import matplotlib.pyplot as plt

    print("Analyzing feature importance for waveform data...")

    # Extract features and targets from the graph dataset
    all_features = []
    all_targets = []

    for graph in graph_dataset:
        # For each node in the graph, create a sample
        for i in range(graph.num_nodes):
            # Extract waveform features for this node
            waveform_features = graph.waveform[i].numpy()

            # Skip if this is the last node (no next aftershock)
            if i < graph.num_nodes - 1:
                # Next node's position is the target
                target = graph.metadata[i + 1][:2].numpy()  # lat, lon

                all_features.append(waveform_features)
                all_targets.append(target)

    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_targets)

    print(f"Extracted {len(X)} samples with {X.shape[1]} features each")

    # 1. Random Forest feature importance
    print("Running Random Forest feature importance analysis...")
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X, y)

    rf_importances = rf_model.feature_importances_

    # 2. Mutual Information analysis for non-linear relationships
    print("Computing mutual information scores...")
    mi_scores = []

    # Calculate MI for both latitude and longitude prediction
    for target_idx in range(2):
        target = y[:, target_idx]
        mi = mutual_info_regression(X, target, random_state=42)
        mi_scores.append(mi)

    # Average MI across both targets
    mi_importances = (mi_scores[0] + mi_scores[1]) / 2

    # 3. Correlation analysis
    print("Computing feature correlations...")
    corr_matrix = np.corrcoef(X, rowvar=False)

    # Create a correlation score based on how redundant each feature is
    redundancy_scores = (
        np.sum(np.abs(corr_matrix), axis=1) - 1
    )  # Subtract self-correlation
    redundancy_scores = redundancy_scores / np.max(redundancy_scores)  # Normalize

    # Combine the importance scores (higher is better)
    # Normalize all scores to [0, 1] range
    rf_importances_norm = rf_importances / np.max(rf_importances)
    mi_importances_norm = mi_importances / np.max(mi_importances)

    # Low redundancy is good, so invert the score
    uniqueness_scores = 1 - redundancy_scores

    # Combine scores with weights
    combined_scores = (
        0.4 * rf_importances_norm + 0.4 * mi_importances_norm + 0.2 * uniqueness_scores
    )

    # Rank features
    feature_indices = np.argsort(combined_scores)[::-1]  # Sort in descending order

    # Extract feature names (if available)
    if hasattr(graph_dataset[0], "feature_names"):
        feature_names = graph_dataset[0].feature_names
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Create results dataframe
    results = []
    for rank, idx in enumerate(feature_indices):
        results.append(
            {
                "rank": rank + 1,
                "feature_idx": idx,
                "feature_name": (
                    feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                ),
                "rf_importance": rf_importances[idx],
                "mi_importance": mi_importances[idx],
                "uniqueness": uniqueness_scores[idx],
                "combined_score": combined_scores[idx],
            }
        )

    # Create a list of optimized feature indices
    selected_indices = feature_indices[:num_features]

    # Plot feature importance
    plt.figure(figsize=(14, 8))

    # Get top N features for plotting
    top_n = min(30, len(feature_indices))
    top_features = feature_indices[:top_n]

    # Features ordered by importance
    plt.subplot(2, 1, 1)
    plt.bar(range(top_n), combined_scores[top_features], color="cornflowerblue")
    plt.xticks(
        range(top_n),
        [
            feature_names[i] if i < len(feature_names) else f"feature_{i}"
            for i in top_features
        ],
        rotation=90,
    )
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.title("Top Feature Importance Scores")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot correlation heatmap of top features
    plt.subplot(2, 1, 2)
    top_feature_corr = corr_matrix[np.ix_(top_features[:15], top_features[:15])]
    plt.imshow(top_feature_corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(
        range(15),
        [
            feature_names[i] if i < len(feature_names) else f"feature_{i}"
            for i in top_features[:15]
        ],
        rotation=90,
    )
    plt.yticks(
        range(15),
        [
            feature_names[i] if i < len(feature_names) else f"feature_{i}"
            for i in top_features[:15]
        ],
    )
    plt.title("Correlation Between Top 15 Features")

    plt.tight_layout()
    plt.savefig("results/feature_importance.png")
    plt.close()

    print(f"Selected {len(selected_indices)} most important features")
    print(f"Top 10 features:")
    for i in range(min(10, len(results))):
        print(
            f"  {i+1}. {results[i]['feature_name']} (score: {results[i]['combined_score']:.4f})"
        )

    return selected_indices, results


def build_graphs_with_optimized_features(graph_dataset, selected_feature_indices):
    """
    Create a new graph dataset with only the selected features,
    avoiding the negative stride issue by creating new tensors
    """
    from torch_geometric.data import Data
    import torch
    
    # Sort indices to ensure consistent order
    selected_feature_indices = sorted(selected_feature_indices)
    
    optimized_dataset = []
    
    # Get original feature names if available
    original_feature_names = None
    if hasattr(graph_dataset[0], 'feature_names'):
        original_feature_names = graph_dataset[0].feature_names
        selected_feature_names = [original_feature_names[idx] if idx < len(original_feature_names) 
                                else f"feature_{idx}" for idx in selected_feature_indices]
    
    for graph in graph_dataset:
        # Explicitly create a new tensor with the selected features
        # This avoids the negative stride issue
        all_features = graph.waveform
        optimized_features = torch.zeros((all_features.size(0), len(selected_feature_indices)), 
                                        dtype=all_features.dtype)
        
        # Manually copy the selected features
        for i, idx in enumerate(selected_feature_indices):
            optimized_features[:, i] = all_features[:, idx]
        
        # Create a new graph with optimized features
        optimized_graph = Data(
            metadata=graph.metadata,
            waveform=optimized_features,  # Use the new tensor
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=graph.y,
            num_nodes=graph.num_nodes
        )
        
        # Save the selected feature names if available
        if original_feature_names is not None:
            optimized_graph.feature_names = selected_feature_names
        
        optimized_dataset.append(optimized_graph)
    
    print(f"Created optimized dataset with {len(optimized_dataset)} graphs")
    print(f"Original waveform feature size: {graph_dataset[0].waveform.shape[1]}")
    print(f"Optimized waveform feature size: {optimized_dataset[0].waveform.shape[1]}")
    
    return optimized_dataset


def apply_pca_to_features(graph_dataset, n_components=20):
    """
    Apply PCA to reduce the dimensionality of waveform features
    """
    import torch
    import numpy as np
    from sklearn.decomposition import PCA
    from torch_geometric.data import Data

    print(f"Applying PCA to reduce features to {n_components} components...")

    # Collect all waveform features
    all_features = []
    for graph in graph_dataset:
        all_features.append(graph.waveform.numpy())

    # Stack all features into a single array
    X = np.vstack(all_features)

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Create new dataset with PCA-transformed features
    pca_dataset = []

    for i, graph in enumerate(graph_dataset):
        # Transform features
        pca_features = pca.transform(graph.waveform.numpy())
        pca_features_tensor = torch.tensor(pca_features, dtype=torch.float)

        # Create a new graph
        pca_graph = Data(
            metadata=graph.metadata,
            waveform=pca_features_tensor,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=graph.y,
            num_nodes=graph.num_nodes,
        )

        pca_dataset.append(pca_graph)

    # Calculate explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(
        f"PCA with {n_components} components explains {explained_variance:.2f}% of variance"
    )

    return pca_dataset, pca


def create_domain_specific_features(graph_dataset, mainshock):
    """
    Create physics-informed domain-specific features for seismic prediction
    """
    import torch
    import numpy as np
    from torch_geometric.data import Data
    
    print("Creating domain-specific seismic features...")
    
    enhanced_dataset = []
    
    # Mainshock coordinates
    mainshock_lat = mainshock["source_latitude_deg"]
    mainshock_lon = mainshock["source_longitude_deg"]
    mainshock_depth = mainshock["source_depth_km"]
    
    for graph in graph_dataset:
        # Get basic features
        metadata = graph.metadata
        waveform = graph.waveform
        
        # Create new domain-specific features
        batch_size = metadata.shape[0]
        domain_features = torch.zeros((batch_size, 6), dtype=torch.float)
        
        for i in range(batch_size):
            # Extract basic metadata
            lat = metadata[i, 0].item()
            lon = metadata[i, 1].item()
            depth = metadata[i, 2].item()
            time = metadata[i, 3].item()
            
            # 1. Distance from mainshock (normalized)
            R = 6371  # Earth radius in km
            lat1, lon1, lat2, lon2 = map(np.radians, [mainshock_lat, mainshock_lon, lat, lon])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c
            domain_features[i, 0] = distance / 100.0  # Normalize by 100km
            
            # 2. Azimuth from mainshock (in radians - normalized by 2π)
            y = np.sin(lon2 - lon1) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
            azimuth = np.arctan2(y, x)
            domain_features[i, 1] = (azimuth + np.pi) / (2 * np.pi)  # Normalize to [0,1]
            
            # 3. Relative depth (normalized)
            rel_depth = (depth - mainshock_depth) / 50.0  # Normalize by 50km
            domain_features[i, 2] = rel_depth
            
            # 4. Log of time since mainshock (normalized)
            log_time = np.log1p(time) / 10.0  # Natural log, normalized by 10
            domain_features[i, 3] = log_time
            
            # 5-6. For nodes after first one, calculate migration vectors
            if i > 0:
                prev_lat = metadata[i-1, 0].item()
                prev_lon = metadata[i-1, 1].item()
                
                # Convert to radians
                lat1, lon1, lat2, lon2 = map(np.radians, [prev_lat, prev_lon, lat, lon])
                
                # Calculate vector components (simplified as we're just looking for direction)
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                # Normalize components to create a unit vector
                magnitude = np.sqrt(dlat**2 + dlon**2)
                if magnitude > 0:
                    domain_features[i, 4] = dlat / magnitude  # x-component of migration 
                    domain_features[i, 5] = dlon / magnitude  # y-component of migration
        
        # Combine original features with domain-specific features
        combined_features = torch.cat([waveform, domain_features], dim=1)
        
        # Create a new graph with combined features
        enhanced_graph = Data(
            metadata=metadata,
            waveform=combined_features,  # Use this as your feature set
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=graph.y,
            num_nodes=graph.num_nodes
        )
        
        # Preserve feature names if available
        if hasattr(graph, 'feature_names'):
            # Create extended feature names list
            domain_feature_names = [
                "distance_from_mainshock", 
                "azimuth_from_mainshock",
                "relative_depth", 
                "log_time",
                "migration_x",
                "migration_y"
            ]
            enhanced_graph.feature_names = list(graph.feature_names) + domain_feature_names
        
        enhanced_dataset.append(enhanced_graph)
    
    print(f"Created enhanced dataset with {len(enhanced_dataset)} graphs")
    print(f"Original feature size: {graph_dataset[0].waveform.shape[1]}")
    print(f"Enhanced feature size: {enhanced_dataset[0].waveform.shape[1]}")
    
    return enhanced_dataset
