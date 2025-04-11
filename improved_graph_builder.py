def build_enhanced_graphs(sequences, distance_threshold_km=25, time_threshold_hours=24,
                     use_edge_features=True, use_adaptive_thresholds=True):
    """
    Build enhanced graph representations with edge features and adaptive thresholds
    
    Parameters:
    -----------
    sequences : list
        List of tuples (metadata_features, waveform_features, target)
    distance_threshold_km : float
        Base distance threshold for creating edges between events
    time_threshold_hours : float
        Base time threshold for creating edges between events
    use_edge_features : bool
        Whether to include edge features (distances, time differences)
    use_adaptive_thresholds : bool
        Whether to use adaptive thresholds based on event properties
        
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
        edge_features_list = []
        
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
                
                # Calculate depth difference
                depth1 = metadata_features[i][2]  # depth in km
                depth2 = metadata_features[j][2]
                depth_diff = abs(depth2 - depth1)
                
                # Adjust thresholds if using adaptive thresholds
                curr_dist_threshold = distance_threshold_km
                curr_time_threshold = time_threshold_hours
                
                if use_adaptive_thresholds:
                    # Adjust distance threshold based on:
                    # 1. Time since mainshock (earlier events might have wider influence)
                    # 2. Depth (deeper events might influence wider areas)
                    # 3. Energy (from waveform features)
                    
                    # Get energy features if available (as a proxy for magnitude)
                    energy_feature_i = 0
                    energy_feature_j = 0
                    for feature_name in feature_names:
                        if "energy" in feature_name.lower() and not "ratio" in feature_name.lower():
                            try:
                                idx = feature_names.index(feature_name)
                                energy_feature_i += waveform_feature_matrix[i][idx]
                                energy_feature_j += waveform_feature_matrix[j][idx]
                            except (ValueError, IndexError):
                                pass
                    
                    # Earlier events have wider influence
                    time_factor = max(1.0, 2.0 - (time1 / 48.0))  # Normalize by 48 hours
                    
                    # Deeper events might influence wider areas
                    depth_factor = 1.0 + (depth1 / 50.0)  # Normalize by 50 km
                    
                    # Higher energy events have wider influence
                    energy_factor = 1.0
                    if energy_feature_i > 0:
                        # Logarithmic scaling for energy influence
                        energy_factor = 1.0 + min(2.0, np.log1p(energy_feature_i) / 10.0)
                    
                    # Combine factors
                    curr_dist_threshold = distance_threshold_km * time_factor * depth_factor * energy_factor
                    
                    # Adjust time threshold based on depth and energy
                    curr_time_threshold = time_threshold_hours * depth_factor * energy_factor
                
                # Add edge based on adjusted thresholds
                if distance < curr_dist_threshold and time_diff < curr_time_threshold:
                    edge_list.append([i, j])
                    
                    if use_edge_features:
                        # Normalize features
                        norm_distance = distance / distance_threshold_km
                        norm_time_diff = time_diff / time_threshold_hours
                        norm_depth_diff = depth_diff / 20.0  # Normalize by 20 km
                        
                        # Direction features (unit vector components)
                        direction_lat = np.sin(lat2 - lat1)
                        direction_lon = np.sin(lon2 - lon1)
                        
                        # Create edge feature vector
                        edge_features = [
                            norm_distance,
                            norm_time_diff,
                            norm_depth_diff,
                            direction_lat,
                            direction_lon
                        ]
                        
                        edge_features_list.append(edge_features)
        
        # If no edges created, add connections to temporal neighbors
        if len(edge_list) == 0:
            for i in range(num_nodes - 1):
                edge_list.append([i, i+1])
                edge_list.append([i+1, i])
                
                if use_edge_features:
                    # Default edge features for sequential connections
                    edge_features_list.append([0.5, 0.5, 0.5, 0, 0])
                    edge_features_list.append([0.5, 0.5, 0.5, 0, 0])
        
        # Convert edge list to torch tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data object
        if use_edge_features and edge_features_list:
            edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
            
            graph = Data(
                metadata=metadata_tensor,
                waveform=waveform_tensor,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=target_tensor,
                num_nodes=metadata_tensor.size(0)
            )
        else:
            graph = Data(
                metadata=metadata_tensor,
                waveform=waveform_tensor,
                edge_index=edge_index,
                y=target_tensor,
                num_nodes=metadata_tensor.size(0)
            )
        
        graph_dataset.append(graph)
    
    print(f"Built {len(graph_dataset)} enhanced graph representations with {'edge features' if use_edge_features else 'standard edges'}")
    return graph_dataset, feature_names