import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class SimplifiedAfterShockGNN(torch.nn.Module):
    """
    Simplified Graph Neural Network for aftershock prediction
    with relative position prediction capabilities
    """

    def __init__(
        self,
        metadata_channels,
        waveform_channels,
        hidden_channels,
        num_layers,
        dropout=0.3,
    ):
        super(SimplifiedAfterShockGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Separate encoders for metadata and waveform features
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.waveform_encoder = nn.Sequential(
            nn.Linear(waveform_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Feature combiner
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Graph Attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer takes hidden_channels, subsequent layers take hidden_channels * heads
            in_channels = hidden_channels if i == 0 else hidden_channels * 2
            self.gat_layers.append(
                GATConv(in_channels, hidden_channels, heads=2, dropout=dropout)
            )

        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * 2))

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
        Forward pass with separate processing of metadata and waveform features
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
            # Store original x for residual connection
            x_res = x if i > 0 else None

            # Apply GAT layer
            x = self.gat_layers[i](x, edge_index)

            # Apply batch normalization
            x = self.batch_norms[i](x)

            # Apply residual connection if shapes match
            if x_res is not None and x.size() == x_res.size():
                x = x + x_res

            # Apply ReLU activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x_graph = global_mean_pool(x, batch)

        # Predict latitude and longitude
        lat = self.lat_predictor(x_graph)
        lon = self.lon_predictor(x_graph)

        return lat, lon
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class SimplerAfterShockGNN(torch.nn.Module):
    """
    Simplified Graph Neural Network for aftershock prediction
    with better optimization properties
    """
    def __init__(self, metadata_channels, waveform_channels, hidden_channels=64, dropout=0.3, num_layers=3):
        super(SimplerAfterShockGNN, self).__init__()
        
        self.dropout = dropout
        
        # Simplified encoders
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.waveform_encoder = nn.Sequential(
            nn.Linear(waveform_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature combiner - simpler
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Use GCN instead of GAT - simpler and more stable
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Add skip connections and batch norms for stability
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        
        # Use both mean and max pooling for better feature extraction
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
        Forward pass with simplified architecture
        """
        # Encode metadata features
        metadata_encoded = self.metadata_encoder(metadata)
        
        # Encode waveform features
        waveform_encoded = self.waveform_encoder(waveform_features)
        
        # Combine features
        x = torch.cat([metadata_encoded, waveform_encoded], dim=1)
        x = self.feature_combiner(x)
        
        # First graph convolution with residual connection
        x1 = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Second graph convolution with skip connection from first
        x2 = self.conv2(x1, edge_index)
        x2 = self.batch_norm2(x2 + x1)  # Residual connection
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # Use both mean and max pooling for better features
        x_mean = global_mean_pool(x2, batch)
        x_max = global_max_pool(x2, batch)
        
        # Combine pooling methods
        x_combined = torch.cat([x_mean, x_max], dim=1)
        
        # Predict latitude and longitude
        lat = self.lat_predictor(x_combined)
        lon = self.lon_predictor(x_combined)
        
        return lat, lon


class BaselineRegressor(torch.nn.Module):
    """
    Even simpler baseline model that doesn't use graph structure
    to verify that learning is happening
    """
    def __init__(self, metadata_channels, waveform_channels, hidden_channels=64, dropout=0.3):
        super(BaselineRegressor, self).__init__()
        
        # Combined network for both metadata and waveform features
        self.network = nn.Sequential(
            nn.Linear(metadata_channels + waveform_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate output heads
        self.lat_head = nn.Linear(hidden_channels, 1)
        self.lon_head = nn.Linear(hidden_channels, 1)
        
    def forward(self, metadata, waveform_features, edge_index, batch):
        # Concatenate features - ignore edge_index
        x = torch.cat([metadata, waveform_features], dim=1)
        
        # Apply network
        x = self.network(x)
        
        # Apply pooling
        x = global_mean_pool(x, batch)
        
        # Get predictions
        lat = self.lat_head(x)
        lon = self.lon_head(x)
        
        return lat, lon