import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import seisbench.data as sbd
import matplotlib.pyplot as plt
from scipy import stats

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BaselineAfterShockGNN(torch.nn.Module):
    """
    Baseline Graph Attention Network for aftershock prediction
    that uses only metadata features (no waveform features)
    """
    def __init__(self, in_channels, hidden_channels, num_layers, dropout=0.3):
        super(BaselineAfterShockGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        # Encoder for metadata features
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(hidden_channels, hidden_channels, heads=2, dropout=dropout))
        
        # For subsequent layers, input is hidden_channels * 2 (from 2 attention heads)
        for i in range(1, num_layers):
            self.gat_layers.append(GATConv(hidden_channels * 2, hidden_channels, heads=2, dropout=dropout))
            
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * 2))  # *2 for the 2 attention heads
            
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
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GNN model with safety checks for batch tensor
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features (metadata only)
        edge_index : torch.Tensor
            Edge indices defining the graph structure
        batch : torch.Tensor
            Batch assignment for nodes
            
        Returns:
        --------
        lat, lon : torch.Tensor
            Predicted latitude and longitude
        """
        # Encode features
        x = self.encoder(x)
        
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


def load_aftershock_data():
    """
    Load and preprocess aftershock data from the Iquique dataset
    without waveform data (baseline approach)
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
    
    # Consolidate duplicate recordings (same as in waveform approach)
    consolidated_metadata = consolidate_recordings(metadata)
    
    print(f"Original dataset: {len(metadata)} recordings")
    print(f"Consolidated dataset: {len(consolidated_metadata)} unique events")
    
    return consolidated_metadata, iquique


def consolidate_recordings(metadata):
    """
    Consolidate multiple station recordings of the same event into a single representation.
    This is the baseline version that doesn't deal with waveform features.
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
    
    for event_id, group in metadata.groupby('event_id'):
        # Just take the first recording for each event
        best_record = group.iloc[0].copy()
        best_record['station_count'] = len(group)
        best_record['event_id'] = event_id  # Keep event_id in the metadata
        consolidated_metadata.append(best_record)
    
    # Convert to DataFrame
    consolidated_metadata = pd.DataFrame(consolidated_metadata)
    return consolidated_metadata


def create_aftershock_sequences(aftershocks, sequence_length=5, time_window_hours=72, max_sequences=5000):
    """
    Create aftershock sequences using a sliding window approach (baseline version)
    """
    sequences = []
    total_aftershocks = len(aftershocks)
    
    print(f"Total aftershocks: {total_aftershocks}")
    
    # Need at least sequence_length+1 events to create a sequence with a target
    if len(aftershocks) < sequence_length + 1:
        print(f"Not enough aftershocks to create sequences (need {sequence_length + 1}, have {len(aftershocks)})")
        return []
    
    # Sort aftershocks by time
    aftershocks_sorted = aftershocks.sort_values('hours_since_mainshock')
    
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
        
        # Extract features for each aftershock in the sequence
        metadata_features = current_sequence[[
            'source_latitude_deg', 
            'source_longitude_deg', 
            'source_depth_km', 
            'hours_since_mainshock'
        ]].values
        
        # Target is the location of the next aftershock
        target = np.array([
            target_aftershock['source_latitude_deg'],
            target_aftershock['source_longitude_deg']
        ])
        
        sequences.append((metadata_features, target))
        
        # Limit the number of sequences to avoid memory issues
        if len(sequences) >= max_sequences:
            print(f"Reached maximum number of sequences ({max_sequences})")
            break
    
    print(f"Created {len(sequences)} aftershock sequences")
    return sequences


def build_graphs_from_sequences(sequences, distance_threshold_km=25):
    """
    Build graph representations from aftershock sequences (baseline version without waveform features)
    
    Parameters:
    -----------
    sequences : list
        List of tuples (metadata_features, target)
    distance_threshold_km : float
        Distance threshold for creating edges between events
        
    Returns:
    --------
    graph_dataset : list
        List of PyTorch Geometric Data objects
    """
    from torch_geometric.data import Data
    import torch
    import numpy as np
    
    graph_dataset = []
    
    for i, (metadata_features, target) in enumerate(sequences):
        num_nodes = len(metadata_features)
        
        # Convert metadata features to torch tensors
        x = torch.tensor(metadata_features, dtype=torch.float)
        
        # Convert target to torch tensor
        y = torch.tensor(target, dtype=torch.float).view(1, 2)
        
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
        graph = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=x.size(0)  # Explicitly set num_nodes
        )
        
        graph_dataset.append(graph)
    
    print(f"Built {len(graph_dataset)} graph representations for baseline model")
    return graph_dataset


def train_gnn_model(graph_dataset, model, epochs=200, lr=0.001, batch_size=32, patience=15):
    """
    Train the baseline GNN model
    
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
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
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
                print(f"  x shape: {batch.x.shape}")
                print(f"  edge_index shape: {batch.edge_index.shape}")
                print(f"  batch tensor shape: {batch.batch.shape}")
                print(f"  y shape: {batch.y.shape}")
            
            # Forward pass
            try:
                lat_pred, lon_pred = model(batch.x, batch.edge_index, batch.batch)
                
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
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
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
                    lat_pred, lon_pred = model(batch.x, batch.edge_index, batch.batch)
                    
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
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the best model
            torch.save(model.state_dict(), 'results/baseline_gnn_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    try:
        model.load_state_dict(torch.load('results/baseline_gnn_model.pt'))
        print("Loaded best model from checkpoint")
    except Exception as e:
        print(f"Warning: Could not load best model: {e}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Baseline Model)')
    plt.legend()
    plt.savefig('results/baseline_training_history.png')
    plt.close()
    
    return model, train_losses, val_losses


def evaluate_model(model, graph_dataset, mainshock):
    """
    Evaluate the baseline model and generate visualizations
    
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
            lat_pred, lon_pred = model(batch.x, batch.edge_index, batch.batch)
            
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
    
    print(f"Baseline Model Evaluation Results:")
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
    plt.title('Spatial Distribution of Actual vs Predicted Aftershocks (Baseline Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/baseline_spatial_predictions.png', dpi=300)
    plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors_km, bins=30, alpha=0.7, color='blue')
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_error:.2f} km')
    plt.axvline(median_error, color='green', linestyle='dashed', linewidth=2, label=f'Median Error: {median_error:.2f} km')
    plt.xlabel('Error (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (Baseline Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/baseline_error_distribution.png', dpi=300)
    plt.close()
    
    return mean_error, median_error, errors_km


def run_baseline_comparison(args):
    """
    Run the baseline model and compare with waveform-enhanced model
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print("\n=== Running Baseline Model ===")

    # Load data for baseline
    metadata, iquique = load_aftershock_data()

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Identify mainshock and aftershocks
    mainshock, aftershocks = identify_mainshock_and_aftershocks(metadata)

    # Create aftershock sequences
    sequences = create_aftershock_sequences(
        aftershocks,
        sequence_length=args.sequence_length,
        time_window_hours=args.time_window,
    )

    # Build graph representations
    graph_dataset = build_graphs_from_sequences(
        sequences, distance_threshold_km=args.distance_threshold
    )

    if len(graph_dataset) == 0:
        print("No valid baseline graph representations created. Exiting.")
        return None

    # Create baseline model
    in_channels = 4  # latitude, longitude, depth, hours since mainshock
    baseline_model = BaselineAfterShockGNN(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(baseline_model)

    # Train baseline model
    baseline_model, train_losses, val_losses = train_gnn_model(
        graph_dataset,
        baseline_model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # Evaluate baseline model
    mean_error, median_error, errors_km = evaluate_model(
        baseline_model, graph_dataset, mainshock
    )
    
    return mean_error, median_error, errors_km


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