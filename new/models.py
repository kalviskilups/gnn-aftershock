import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn import global_mean_pool, global_max_pool

class EnhancedGNN(torch.nn.Module):
    """
    Enhanced Graph Neural Network for aftershock prediction
    with improved architecture for spatial coverage
    """

    def __init__(
        self,
        metadata_channels,
        waveform_channels,
        edge_attr_channels=3,
        hidden_channels=128,
        num_layers=3,
        dropout=0.2,
    ):
        super(EnhancedGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Layer normalization for more stable training
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_channels * 2))

        # More advanced metadata encoder with additional layers
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_channels, hidden_channels // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        # Waveform encoder with batch normalization
        self.waveform_encoder = nn.Sequential(
            nn.Linear(waveform_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        # Encoder for edge attributes
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_attr_channels, hidden_channels // 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels // 4, 1),
            nn.Sigmoid()
        )

        # Combine metadata and waveform features
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), 
            nn.LeakyReLU(), 
            nn.Dropout(dropout)
        )

        # Graph Attention layers with edge attributes
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(hidden_channels, hidden_channels, heads=2, dropout=dropout, edge_dim=edge_attr_channels)
        )

        # For subsequent layers
        for i in range(1, num_layers):
            self.gat_layers.append(
                GATConv(hidden_channels * 2, hidden_channels, heads=2, dropout=dropout, edge_dim=edge_attr_channels)
            )

        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * 2))
            
        # Global pooling with attention
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_channels * 2, 1),
            nn.Sigmoid()
        )
        
        # Diversity encoder - encourages output to have appropriate spatial variance
        self.diversity_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU()
        )

        # Output MLP with multiple heads for latitude and longitude
        self.lat_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 6, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.lon_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 6, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, metadata, waveform_features, edge_index, edge_attr, batch):
        """
        Forward pass of the enhanced GNN with multiple pooling strategies
        """
        # Encode metadata features
        metadata_encoded = self.metadata_encoder(metadata)

        # Encode waveform features
        waveform_encoded = self.waveform_encoder(waveform_features)

        # Combine features
        x = torch.cat([metadata_encoded, waveform_encoded], dim=1)
        x = self.feature_combiner(x)

        # Apply GAT layers with edge attributes
        for i in range(self.num_layers):
            # Store original x for residual connection
            x_res = x if i > 0 else None

            # Apply GAT layer
            x = self.gat_layers[i](x, edge_index, edge_attr=edge_attr)
            
            # Apply normalization
            x = self.batch_norms[i](x)
            
            # Apply residual connection if shapes match
            if x_res is not None and x.size() == x_res.size():
                x = x + x_res

            # Apply non-linearity
            x = F.leaky_relu(x)

            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Multiple pooling strategies for better coverage
        # 1. Mean pooling
        x_mean = global_mean_pool(x, batch)
        
        # 2. Max pooling - captures extreme values
        x_max = global_max_pool(x, batch)
        
        # 3. Attention-weighted pooling
        attention_weights = self.attention_pool(x)
        x_weighted = global_mean_pool(x * attention_weights, batch)
        
        # Combine pooling strategies
        x_combined = torch.cat([x_mean, x_max, x_weighted], dim=1)
        
        # Enhanced diversity features from graph-level representation
        diversity_features = self.diversity_mlp(torch.cat([x_mean, x_max], dim=1))
        
        # Predict latitude and longitude with diversity enhancement
        lat = self.lat_predictor(x_combined)
        lon = self.lon_predictor(x_combined)

        return lat, lon