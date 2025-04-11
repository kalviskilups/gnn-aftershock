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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        for i, component_name in enumerate(['Z', 'N', 'E']):
            if i < waveform.shape[0]:  # Check if component exists
                component = waveform[i]
                
                # Calculate basic statistics
                features[f'{component_name}_max'] = np.max(np.abs(component))
                features[f'{component_name}_mean'] = np.mean(np.abs(component))
                features[f'{component_name}_std'] = np.std(component)
                features[f'{component_name}_rms'] = np.sqrt(np.mean(component**2))
                features[f'{component_name}_energy'] = np.sum(component**2)
                features[f'{component_name}_kurtosis'] = self._kurtosis(component)
                
                # Calculate frequency-domain features
                freq_features = self._compute_frequency_features(component)
                for feat_name, feat_value in freq_features.items():
                    features[f'{component_name}_{feat_name}'] = feat_value
        
        # Extract P-wave and S-wave specific features if arrivals are provided
        if p_arrival is not None and s_arrival is not None:
            p_window_size = 100  # 100 samples after P arrival
            s_window_size = 100  # 100 samples after S arrival
            
            # Ensure window sizes don't exceed waveform length
            p_window_size = min(p_window_size, waveform.shape[1] - p_arrival)
            s_window_size = min(s_window_size, waveform.shape[1] - s_arrival)
            
            if p_window_size > 0:
                for i, component_name in enumerate(['Z', 'N', 'E']):
                    if i < waveform.shape[0]:  # Check if component exists
                        p_segment = waveform[i, p_arrival:p_arrival+p_window_size]
                        
                        # Calculate P-wave features
                        features[f'P_{component_name}_max'] = np.max(np.abs(p_segment))
                        features[f'P_{component_name}_mean'] = np.mean(np.abs(p_segment))
                        features[f'P_{component_name}_std'] = np.std(p_segment)
                        features[f'P_{component_name}_energy'] = np.sum(p_segment**2)
                        
                        # P-wave frequency features
                        p_freq_features = self._compute_frequency_features(p_segment)
                        for feat_name, feat_value in p_freq_features.items():
                            features[f'P_{component_name}_{feat_name}'] = feat_value
            
            if s_window_size > 0:
                for i, component_name in enumerate(['Z', 'N', 'E']):
                    if i < waveform.shape[0]:  # Check if component exists
                        s_segment = waveform[i, s_arrival:s_arrival+s_window_size]
                        
                        # Calculate S-wave features
                        features[f'S_{component_name}_max'] = np.max(np.abs(s_segment))
                        features[f'S_{component_name}_mean'] = np.mean(np.abs(s_segment))
                        features[f'S_{component_name}_std'] = np.std(s_segment)
                        features[f'S_{component_name}_energy'] = np.sum(s_segment**2)
                        
                        # S-wave frequency features
                        s_freq_features = self._compute_frequency_features(s_segment)
                        for feat_name, feat_value in s_freq_features.items():
                            features[f'S_{component_name}_{feat_name}'] = feat_value
            
            # Calculate P/S amplitude ratios for each component
            for i, component_name in enumerate(['Z', 'N', 'E']):
                if i < waveform.shape[0]:  # Check if component exists
                    if features.get(f'P_{component_name}_max', 0) > 0 and features.get(f'S_{component_name}_max', 0) > 0:
                        features[f'{component_name}_PS_ratio'] = features[f'P_{component_name}_max'] / features[f'S_{component_name}_max']
        
        return features
    
    def _compute_frequency_features(self, signal_segment):
        """
        Compute frequency domain features
        """
        features = {}
        
        # Check if signal segment is not empty
        if len(signal_segment) < 10:
            return {
                'dominant_freq': 0,
                'low_freq_energy': 0,
                'mid_freq_energy': 0,
                'high_freq_energy': 0,
                'low_high_ratio': 0
            }
        
        try:
            # Calculate power spectral density
            f, Pxx = signal.welch(signal_segment, fs=self.sampling_rate, nperseg=min(256, len(signal_segment)))
            
            # Dominant frequency
            features['dominant_freq'] = f[np.argmax(Pxx)]
            
            # Frequency band energies
            low_idx = f < 5
            mid_idx = (f >= 5) & (f < 15)
            high_idx = f >= 15
            
            # Handle empty frequency bands
            features['low_freq_energy'] = np.sum(Pxx[low_idx]) if np.any(low_idx) else 0
            features['mid_freq_energy'] = np.sum(Pxx[mid_idx]) if np.any(mid_idx) else 0
            features['high_freq_energy'] = np.sum(Pxx[high_idx]) if np.any(high_idx) else 0
            
            # Calculate frequency ratios
            if features['high_freq_energy'] > 0:
                features['low_high_ratio'] = features['low_freq_energy'] / features['high_freq_energy']
            else:
                features['low_high_ratio'] = 0
        
        except Exception as e:
            print(f"Error computing frequency features: {e}")
            features = {
                'dominant_freq': 0,
                'low_freq_energy': 0,
                'mid_freq_energy': 0,
                'high_freq_energy': 0,
                'low_high_ratio': 0
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
    def __init__(self, metadata_channels, waveform_channels, hidden_channels, num_layers, dropout=0.3):
        super(AfterShockGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        # Separate encoders for metadata and waveform features
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.waveform_encoder = nn.Sequential(
            nn.Linear(waveform_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combine metadata and waveform features
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(hidden_channels, hidden_channels, heads=2, dropout=dropout))
        
        # For subsequent layers, input is hidden_channels * 4 (from 4 attention heads)
        for i in range(1, num_layers):
            self.gat_layers.append(GATConv(hidden_channels * 2, hidden_channels, heads=2, dropout=dropout))
            
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * 2))  # *4 for the 4 attention heads
            
        # Output layers for predicting latitude and longitude
        self.lat_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        
        self.lon_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
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
                batch = batch[:x.size(0)]
        
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
    metadata = metadata.dropna(subset=[
        'source_origin_time', 
        'source_latitude_deg', 
        'source_longitude_deg', 
        'source_depth_km'
    ])
    
    # Convert timestamps
    metadata['datetime'] = pd.to_datetime(metadata['source_origin_time'])
    
    # Sort by time
    metadata = metadata.sort_values('datetime')
    
    # Create a dictionary to store waveform features
    waveform_features_dict = {}
    
    # Initialize feature extractor
    feature_extractor = WaveformFeatureExtractor()
    
    # Limit the number of waveforms to process
    sample_indices = metadata.index[:min(max_waveforms, len(metadata))]
    
    print(f"Extracting waveform features for {len(sample_indices)} events...")
    for idx in tqdm(sample_indices):
        try:
            # Get waveform data
            waveform = iquique.get_waveforms(int(idx))
            
            # Get P and S arrival samples if available
            p_arrival = metadata.loc[idx, 'trace_P_arrival_sample']
            s_arrival = metadata.loc[idx, 'trace_S_arrival_sample']
            
            # Validate P and S arrivals
            if pd.isna(p_arrival) or pd.isna(s_arrival):
                p_arrival, s_arrival = None, None
            else:
                p_arrival = int(p_arrival)
                s_arrival = int(s_arrival)
            
            # Extract features
            features = feature_extractor.extract_features(waveform, p_arrival, s_arrival)
            
            # Store features
            waveform_features_dict[idx] = features
        
        except Exception as e:
            print(f"Error processing waveform {idx}: {e}")
            waveform_features_dict[idx] = {}
    
    print(f"Successfully extracted features for {len(waveform_features_dict)} waveforms")
    
    return metadata, iquique, waveform_features_dict


def identify_mainshock_and_aftershocks(metadata):
    """
    Identify the mainshock and its associated aftershocks based on data patterns
    """
    # The known Iquique earthquake was in early April 2014
    # Let's create a date range for the expected period
    april_start = pd.Timestamp('2014-04-01', tz='UTC')
    april_end = pd.Timestamp('2014-04-05', tz='UTC')
    
    # Find events in early April 2014
    april_events = metadata[
        (metadata['datetime'] >= april_start) & 
        (metadata['datetime'] <= april_end)
    ]
    
    if len(april_events) == 0:
        print("No events found in early April 2014 timeframe. Using alternative approach.")
        # Alternative approach: find the earliest events in the dataset
        metadata_sorted = metadata.sort_values('datetime')
        earliest_date = metadata_sorted['datetime'].iloc[0]
        print(f"Earliest event in dataset: {earliest_date}")
        
        # Take the earliest events as potential mainshock candidates
        potential_mainshocks = metadata_sorted.iloc[:100]  # First 100 events
        
        # Find which event has most events in the following 24 hours
        best_event_idx = None
        most_followers = 0
        
        for idx, row in potential_mainshocks.iterrows():
            event_time = row['datetime']
            next_24h = event_time + pd.Timedelta(hours=24)
            followers = sum(metadata['datetime'].between(event_time, next_24h))
            
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
            event_time = row['datetime']
            next_24h = event_time + pd.Timedelta(hours=24)
            followers = sum(metadata['datetime'].between(event_time, next_24h))
            
            if followers > most_followers:
                most_followers = followers
                best_event_idx = idx
        
        mainshock = metadata.loc[best_event_idx]
    
    print(f"Identified potential mainshock at {mainshock['datetime']} at location "
          f"({mainshock['source_latitude_deg']}, {mainshock['source_longitude_deg']}), "
          f"depth {mainshock['source_depth_km']} km")
    print(f"This event is followed by {most_followers} events in the next 24 hours")
    
    # Select events after the mainshock as aftershocks
    aftershocks = metadata[metadata['datetime'] > mainshock['datetime']].copy()
    print(f"Found {len(aftershocks)} aftershocks")
    
    # Create a new feature: time since mainshock in hours
    aftershocks['hours_since_mainshock'] = (aftershocks['datetime'] - mainshock['datetime']).dt.total_seconds() / 3600
    
    # Create features: distance from mainshock (approximate using Haversine)
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    aftershocks['distance_from_mainshock_km'] = aftershocks.apply(
        lambda row: haversine_distance(
            mainshock['source_latitude_deg'], 
            mainshock['source_longitude_deg'],
            row['source_latitude_deg'], 
            row['source_longitude_deg']
        ), 
        axis=1
    )
    
    return mainshock, aftershocks


def create_aftershock_sequences_with_waveforms(aftershocks, waveform_features_dict, sequence_length=5, time_window_hours=72, max_sequences=5000):
    """
    Create aftershock sequences with waveform features using a sliding window approach
    """
    sequences = []
    total_aftershocks = len(aftershocks)
    
    # Check which aftershocks have waveform features
    aftershocks_with_features = aftershocks[aftershocks['event_id'].isin(waveform_features_dict.keys())]
    
    print(f"Total aftershocks: {total_aftershocks}")
    print(f"Aftershocks with waveform features: {len(aftershocks_with_features)}")
    
    # Need at least sequence_length+1 events to create a sequence with a target
    if len(aftershocks_with_features) < sequence_length + 1:
        print(f"Not enough aftershocks with waveform features to create sequences (need {sequence_length + 1}, have {len(aftershocks_with_features)})")
        return []
    
    # Sort aftershocks by time
    aftershocks_sorted = aftershocks_with_features.sort_values('hours_since_mainshock')
    
    # Use a sliding window approach with a smaller step size
    step_size = 1  # Create a sequence starting at each event
    
    for i in range(0, len(aftershocks_sorted) - sequence_length, step_size):
        # Get sequence of aftershocks
        current_sequence = aftershocks_sorted.iloc[i:i+sequence_length]
        target_aftershock = aftershocks_sorted.iloc[i+sequence_length]
        
        # Check if the sequence spans less than the time window
        seq_duration = current_sequence.iloc[-1]['hours_since_mainshock'] - current_sequence.iloc[0]['hours_since_mainshock']
        if seq_duration > time_window_hours:
            continue
            
        # Check for sufficient spatial variation (with reduced threshold)
        lats = current_sequence['source_latitude_deg'].values
        lons = current_sequence['source_longitude_deg'].values
        
        lat_range = np.max(lats) - np.min(lats)
        lon_range = np.max(lons) - np.min(lons)
        
        # Reduced minimum variation threshold
        min_variation = 0.02  # ~2 km at this latitude
        if lat_range < min_variation and lon_range < min_variation:
            continue
        
        # Extract metadata features for each aftershock in the sequence
        metadata_features = current_sequence[[
            'source_latitude_deg', 
            'source_longitude_deg', 
            'source_depth_km', 
            'hours_since_mainshock'
        ]].values
        
        # Extract waveform features for each aftershock in the sequence
        sequence_waveform_features = []
        valid_sequence = True
        
        for _, row in current_sequence.iterrows():
            event_id = row['event_id']  # Use event_id instead of index
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
        target = np.array([
            target_aftershock['source_latitude_deg'],
            target_aftershock['source_longitude_deg']
        ])
        
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
    metadata['lat_rounded'] = np.round(metadata['source_latitude_deg'], 4)
    metadata['lon_rounded'] = np.round(metadata['source_longitude_deg'], 4)
    metadata['depth_rounded'] = np.round(metadata['source_depth_km'], 1)
    
    metadata['event_id'] = metadata.groupby(['source_origin_time', 
                                            'lat_rounded',
                                            'lon_rounded',
                                            'depth_rounded']).ngroup()
    
    print(f"Original recordings: {len(metadata)}, Unique events: {metadata['event_id'].nunique()}")
    
    # Create consolidated representations
    consolidated_metadata = []
    consolidated_features = {}
    
    # Track how many events actually have features
    events_with_features = 0
    
    for event_id, group in metadata.groupby('event_id'):
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
        best_record['station_count'] = len(group)
        best_record['event_id'] = event_id  # Keep event_id in the metadata
        consolidated_metadata.append(best_record)
        
        # Map waveform features to the event_id
        if best_idx in waveform_features_dict and waveform_features_dict[best_idx]:
            consolidated_features[event_id] = waveform_features_dict[best_idx]
    
    print(f"Events with valid waveform features: {events_with_features}")
    
    # Convert to DataFrame
    consolidated_metadata = pd.DataFrame(consolidated_metadata)
    return consolidated_metadata, consolidated_features


def build_graphs_from_sequences_with_waveforms(sequences, distance_threshold_km=25):
    """
    Build graph representations from aftershock sequences with waveform features
    
    Parameters:
    -----------
    sequences : list
        List of tuples (metadata_features, waveform_features, target)
    distance_threshold_km : float
        Distance threshold for creating edges between events
        
    Returns:
    --------
    graph_dataset : list
        List of PyTorch Geometric Data objects
    feature_names : list
        List of feature names for the waveform features
    """
    from torch_geometric.data import Data
    import torch
    import numpy as np
    
    graph_dataset = []
    
    # Extract waveform feature names from the first sequence
    if len(sequences) > 0:
        first_sequence_waveform_features = sequences[0][1]
        feature_names = sorted(list(first_sequence_waveform_features[0].keys()))
        print(f"Using {len(feature_names)} waveform features: {feature_names[:5]}...")
    else:
        feature_names = []
        return graph_dataset, feature_names
    
    for i, (metadata_features, waveform_features_list, target) in enumerate(sequences):
        num_nodes = len(metadata_features)
        
        # Convert metadata features to torch tensors
        metadata_tensor = torch.tensor(metadata_features, dtype=torch.float)
        
        # Convert waveform features to torch tensors
        waveform_feature_matrix = []
        for waveform_features in waveform_features_list:
            # Extract features in consistent order
            feature_values = [waveform_features.get(name, 0.0) for name in feature_names]
            waveform_feature_matrix.append(feature_values)
        
        waveform_tensor = torch.tensor(waveform_feature_matrix, dtype=torch.float)
        
        # Convert target to torch tensor
        target_tensor = torch.tensor(target, dtype=torch.float).view(1, 2)
        
        # Create edges based on spatiotemporal proximity
        edge_list = []
        
        # Calculate distances between all pairs of events
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue  # Skip self-loops for now
                
                # Calculate distance between events
                lat1, lon1 = metadata_features[i][0], metadata_features[i][1]
                lat2, lon2 = metadata_features[j][0], metadata_features[j][1]
                
                # Approximate distance using Haversine
                R = 6371  # Earth radius in kilometers
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance = R * c
                
                # Calculate time difference
                time1 = metadata_features[i][3]  # hours since mainshock
                time2 = metadata_features[j][3]
                time_diff = abs(time2 - time1)
                
                # Add edge based on distance and time relationship
                if distance < distance_threshold_km and time_diff < 24:  # Within distance threshold and 24 hours
                    edge_list.append([i, j])
        
        # If no edges created, add connections to temporal neighbors
        if len(edge_list) == 0:
            for i in range(num_nodes - 1):
                edge_list.append([i, i+1])
                edge_list.append([i+1, i])
        
        # Convert edge list to torch tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data object
        # Note: We store metadata and waveform features separately
        # IMPORTANT FIX: Explicitly set num_nodes to avoid mismatch
        graph = Data(
            metadata=metadata_tensor,
            waveform=waveform_tensor,
            edge_index=edge_index,
            y=target_tensor,
            num_nodes=metadata_tensor.size(0)  # Explicitly set num_nodes
        )
        
        graph_dataset.append(graph)
    
    print(f"Built {len(graph_dataset)} graph representations with waveform features")
    return graph_dataset, feature_names


def normalize_waveform_features(graph_dataset, feature_names):
    """Enhanced normalization with robust scaling and outlier handling"""
    import torch
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    
    # Extract all waveform features
    all_waveform_features = []
    for graph in graph_dataset:
        all_waveform_features.append(graph.waveform.numpy())
    
    if not all_waveform_features:
        return graph_dataset
    
    # Reshape to 2D array
    stacked_features = np.vstack(all_waveform_features)
    
    # Check for NaN or infinite values
    invalid_mask = np.isnan(stacked_features) | np.isinf(stacked_features)
    if invalid_mask.any():
        print(f"WARNING: Found {np.sum(invalid_mask)} invalid values in features. Replacing with zeros.")
        stacked_features[invalid_mask] = 0

    mean = np.mean(stacked_features, axis=0)
    std = np.std(stacked_features, axis=0)
    
    # Clip values beyond 3 standard deviations
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    
    for i in range(stacked_features.shape[1]):
        stacked_features[:, i] = np.clip(
            stacked_features[:, i], 
            lower_bound[i], 
            upper_bound[i]
        )
    
    # Use RobustScaler instead of StandardScaler to handle outliers better
    scaler = RobustScaler()
    
    # Fit scaler and transform features
    normalized_features = scaler.fit_transform(stacked_features)
    
    # Print feature statistics to diagnose issues
    print(f"Feature min values: {np.min(normalized_features, axis=0)[:5]}...")
    print(f"Feature max values: {np.max(normalized_features, axis=0)[:5]}...")
    print(f"Feature median values: {np.median(normalized_features, axis=0)[:5]}...")
    
    # Replace features in the dataset
    start_idx = 0
    normalized_dataset = []
    
    for i, graph in enumerate(graph_dataset):
        num_nodes = graph.waveform.shape[0]
        graph_features = normalized_features[start_idx:start_idx + num_nodes]
        
        # Replace waveform features with normalized ones
        new_graph = graph.clone()
        new_graph.waveform = torch.tensor(graph_features, dtype=torch.float)
        
        normalized_dataset.append(new_graph)
        start_idx += num_nodes
    
    print(f"Normalized waveform features for {len(normalized_dataset)} graphs")
    return normalized_dataset


def train_waveform_gnn_model(graph_dataset, model, epochs=200, lr=0.001, batch_size=32, patience=15):
    """
    Train the GNN model using the graph dataset with waveform features
    
    Parameters:
    -----------
    graph_dataset : list
        List of PyTorch Geometric Data objects
    model : torch.nn.Module
        The GNN model
    epochs : int
        Maximum number of epochs to train
    lr : float
        Learning rate
    batch_size : int
        Batch size
    patience : int
        Early stopping patience
        
    Returns:
    --------
    model : torch.nn.Module
        The trained model
    train_losses : list
        Training loss history
    val_losses : list
        Validation loss history
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split
    import os
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Split into training and validation sets
    train_data, val_data = train_test_split(graph_dataset, test_size=0.2, random_state=42)
    
    # Set follow_batch to ensure proper batching of all tensors
    follow_batch = ['metadata', 'waveform']
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, follow_batch=follow_batch)
    val_loader = DataLoader(val_data, batch_size=batch_size, follow_batch=follow_batch)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Loss function - MSE for regression
    loss_fn = torch.nn.MSELoss()
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Explicitly print batch statistics for debugging on first epoch
            if epoch == 0 and train_loss == 0:
                print(f"Batch statistics:")
                print(f"  metadata shape: {batch.metadata.shape}")
                print(f"  waveform shape: {batch.waveform.shape}")
                print(f"  edge_index shape: {batch.edge_index.shape}")
                print(f"  batch tensor shape: {batch.batch.shape}")
                print(f"  y shape: {batch.y.shape}")
            
            # Forward pass (using both metadata and waveform features)
            try:
                lat_pred, lon_pred = model(batch.metadata, batch.waveform, batch.edge_index, batch.batch)
                
                # Combine predictions and compute loss
                pred = torch.cat([lat_pred, lon_pred], dim=1)  # Shape: [batch_size, 2]
                target = batch.y  # Should be shape [batch_size, 2]
                
                # Reshape target to ensure it has the shape [batch_size, 2]
                if target.dim() == 1:
                    target = target.view(-1, 2)
                elif target.shape[0] * 2 == target.numel():
                    # This handles the case where the target is flattened
                    target = target.view(pred.shape[0], 2)
                
                # Check for NaN values
                if torch.isnan(pred).any() or torch.isnan(target).any():
                    print("WARNING: NaN values detected in predictions or targets")
                    continue
                
                loss = loss_fn(pred, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch.num_graphs
                
            except RuntimeError as e:
                print(f"Error in batch processing: {e}")
                # Skip this batch and continue with training
                continue
        
        train_loss /= len(train_data)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                try:
                    # Forward pass
                    lat_pred, lon_pred = model(batch.metadata, batch.waveform, batch.edge_index, batch.batch)
                    
                    # Combine predictions and compute loss
                    pred = torch.cat([lat_pred, lon_pred], dim=1)
                    target = batch.y
                    
                    # Reshape target if needed
                    if target.dim() == 1:
                        target = target.view(-1, 2)
                    elif target.shape[0] * 2 == target.numel():
                        target = target.view(pred.shape[0], 2)
                    
                    # Check for NaN values
                    if torch.isnan(pred).any() or torch.isnan(target).any():
                        print("WARNING: NaN values detected in validation predictions or targets")
                        continue
                    
                    loss = loss_fn(pred, target)
                    
                    val_loss += loss.item() * batch.num_graphs
                
                except RuntimeError as e:
                    print(f"Error in validation batch processing: {e}")
                    # Skip this batch
                    continue
        
        val_loss /= len(val_data)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the best model
            torch.save(model.state_dict(), 'results/waveform_gnn_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    try:
        model.load_state_dict(torch.load('results/waveform_gnn_model.pt'))
    except Exception as e:
        print(f"Warning: Could not load best model: {e}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('results/waveform_training_history.png')
    plt.close()
    
    return model, train_losses, val_losses

def evaluate_waveform_model(model, graph_dataset, mainshock):
    """
    Evaluate the model with waveform features and generate visualizations
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained GNN model
    graph_dataset : list
        List of PyTorch Geometric Data objects
    mainshock : pandas.Series
        Information about the mainshock
        
    Returns:
    --------
    mean_error : float
        Mean prediction error in kilometers
    median_error : float
        Median prediction error in kilometers
    errors_km : list
        List of prediction errors in kilometers
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from torch_geometric.loader import DataLoader
    
    # Split into training and testing sets
    _, test_data = train_test_split(graph_dataset, test_size=0.2, random_state=42)
    
    # Create data loader
    test_loader = DataLoader(test_data, batch_size=32)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Model in evaluation mode
    model.eval()
    
    # Lists to store actual and predicted locations
    actual_lats = []
    actual_lons = []
    pred_lats = []
    pred_lons = []
    
    # Evaluate on test set
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            lat_pred, lon_pred = model(batch.metadata, batch.waveform, batch.edge_index, batch.batch)
            
            # Fix the dimensionality issue
            lat_np = lat_pred.cpu().numpy()
            lon_np = lon_pred.cpu().numpy()
            
            # Handle scalar case (when batch size is 1)
            if lat_np.ndim == 0 or (lat_np.ndim == 2 and lat_np.shape[0] == 1 and lat_np.shape[1] == 1):
                lat_np = np.array([float(lat_np)])
            else:
                lat_np = lat_np.squeeze()
                
            if lon_np.ndim == 0 or (lon_np.ndim == 2 and lon_np.shape[0] == 1 and lon_np.shape[1] == 1):
                lon_np = np.array([float(lon_np)])
            else:
                lon_np = lon_np.squeeze()
            
            actual_lats.extend(batch.y[:, 0].cpu().numpy())
            actual_lons.extend(batch.y[:, 1].cpu().numpy())
            pred_lats.extend(lat_np)
            pred_lons.extend(lon_np)
    
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
    
    print(f"Evaluation Results:")
    print(f"Mean Error: {mean_error:.2f} km")
    print(f"Median Error: {median_error:.2f} km")
    
    # Plot spatial predictions
    plt.figure(figsize=(12, 10))
    
    # Plot mainshock
    plt.scatter(
        mainshock['source_longitude_deg'], 
        mainshock['source_latitude_deg'], 
        s=200, 
        c='red', 
        marker='*', 
        label='Mainshock',
        edgecolor='black',
        zorder=10
    )
    
    # Plot actual aftershocks
    plt.scatter(
        actual_lons, 
        actual_lats, 
        s=50, 
        c='blue', 
        alpha=0.7, 
        label='Actual Aftershocks',
        edgecolor='black'
    )
    
    # Plot predicted aftershocks
    plt.scatter(
        pred_lons, 
        pred_lats, 
        s=30, 
        c='green', 
        alpha=0.7, 
        marker='x', 
        label='Predicted Aftershocks'
    )
    
    # Connect actual to predicted with lines
    for i in range(len(actual_lats)):
        plt.plot(
            [actual_lons[i], pred_lons[i]], 
            [actual_lats[i], pred_lats[i]], 
            'k-', 
            alpha=0.2
        )
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Distribution of Actual vs Predicted Aftershocks (with Waveform Features)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/waveform_spatial_predictions.png', dpi=300)
    plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors_km, bins=30, alpha=0.7, color='blue')
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_error:.2f} km')
    plt.axvline(median_error, color='green', linestyle='dashed', linewidth=2, label=f'Median Error: {median_error:.2f} km')
    plt.xlabel('Error (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (with Waveform Features)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/waveform_error_distribution.png', dpi=300)
    plt.close()
    
    return mean_error, median_error, errors_km


def compare_models(baseline_errors, waveform_errors):
    """
    Compare baseline model with waveform-enhanced model
    
    Parameters:
    -----------
    baseline_errors : list
        List of prediction errors from baseline model
    waveform_errors : list
        List of prediction errors from waveform-enhanced model
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Calculate statistics
    baseline_mean = np.mean(baseline_errors)
    baseline_median = np.median(baseline_errors)
    baseline_std = np.std(baseline_errors)
    
    waveform_mean = np.mean(waveform_errors)
    waveform_median = np.median(waveform_errors)
    waveform_std = np.std(waveform_errors)
    
    # Improvement percentages
    mean_improvement = ((baseline_mean - waveform_mean) / baseline_mean) * 100
    median_improvement = ((baseline_median - waveform_median) / baseline_median) * 100
    
    # Perform statistical test
    t_stat, p_value = stats.ttest_ind(baseline_errors, waveform_errors)
    
    print(f"Model Comparison:")
    print(f"Baseline Model - Mean Error: {baseline_mean:.2f} km, Median Error: {baseline_median:.2f} km, Std Dev: {baseline_std:.2f} km")
    print(f"Waveform Model - Mean Error: {waveform_mean:.2f} km, Median Error: {waveform_median:.2f} km, Std Dev: {waveform_std:.2f} km")
    print(f"Improvement - Mean: {mean_improvement:.2f}%, Median: {median_improvement:.2f}%")
    print(f"Statistical Test - t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot histograms
    plt.hist(baseline_errors, bins=30, alpha=0.5, color='blue', label='Baseline Model')
    plt.hist(waveform_errors, bins=30, alpha=0.5, color='green', label='Waveform Model')
    
    # Plot mean lines
    plt.axvline(baseline_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Baseline Mean: {baseline_mean:.2f} km')
    plt.axvline(waveform_mean, color='green', linestyle='dashed', linewidth=2, label=f'Waveform Mean: {waveform_mean:.2f} km')
    
    plt.xlabel('Error (km)')
    plt.ylabel('Frequency')
    plt.title('Comparison of Prediction Errors Between Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/model_comparison.png', dpi=300)
    plt.close()
    
    # Create summary table as a figure
    plt.figure(figsize=(10, 6))
    plt.axis('tight')
    plt.axis('off')
    
    table_data = [
        ['Metric', 'Baseline Model', 'Waveform Model', 'Improvement (%)'],
        ['Mean Error (km)', f'{baseline_mean:.2f}', f'{waveform_mean:.2f}', f'{mean_improvement:.2f}%'],
        ['Median Error (km)', f'{baseline_median:.2f}', f'{waveform_median:.2f}', f'{median_improvement:.2f}%'],
        ['Std. Deviation (km)', f'{baseline_std:.2f}', f'{waveform_std:.2f}', 'N/A'],
        ['p-value', 'N/A', 'N/A', f'{p_value:.4f}']
    ]
    
    table = plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.title('Model Comparison Summary', fontsize=16, pad=20)
    plt.savefig('results/model_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()