import matplotlib.pyplot as plt
import numpy as np

def debug_metadata(metadata, aftershocks, mainshock):
    """Debug function to check metadata and aftershocks dataframes"""
    print("\n=== Metadata and Aftershocks Debug ===")
    
    # Check basic properties
    print(f"Total metadata entries: {len(metadata)}")
    print(f"Total aftershocks: {len(aftershocks)}")
    
    # Check for missing values
    print("\nMissing values in metadata:")
    print(metadata.isnull().sum()[metadata.isnull().sum() > 0])
    
    print("\nMissing values in aftershocks:")
    print(aftershocks.isnull().sum()[aftershocks.isnull().sum() > 0])
    
    # Check value ranges
    print("\nMetadata value ranges:")
    for col in ['source_latitude_deg', 'source_longitude_deg', 'source_depth_km']:
        if col in metadata.columns:
            print(f"{col}: {metadata[col].min():.2f} to {metadata[col].max():.2f}")
    
    # Check temporal distribution
    if 'datetime' in aftershocks.columns:
        time_range = aftershocks['datetime'].max() - aftershocks['datetime'].min()
        print(f"\nAftershock time range: {time_range}")
        print(f"Hours since mainshock range: {aftershocks['hours_since_mainshock'].min():.2f} to {aftershocks['hours_since_mainshock'].max():.2f}")
    
    # Check spatial distribution relative to mainshock
    print("\nSpatial distribution relative to mainshock:")
    print(f"Mainshock location: ({mainshock['source_latitude_deg']:.4f}, {mainshock['source_longitude_deg']:.4f})")
    print(f"Aftershock latitude range: {aftershocks['source_latitude_deg'].min():.4f} to {aftershocks['source_latitude_deg'].max():.4f}")
    print(f"Aftershock longitude range: {aftershocks['source_longitude_deg'].min():.4f} to {aftershocks['source_longitude_deg'].max():.4f}")

def debug_waveform_features(waveform_features_dict, print_sample=True):
    """Debug function to check waveform features"""
    print("\n=== Waveform Features Debug ===")
    
    # Basic statistics
    num_features = len(waveform_features_dict)
    print(f"Total number of waveform features: {num_features}")
    
    # Check for empty or invalid features
    empty_features = sum(1 for features in waveform_features_dict.values() if not features)
    print(f"Number of empty feature sets: {empty_features}")
    
    # Get a sample of feature names
    if num_features > 0:
        sample_key = next(iter(waveform_features_dict))
        sample_features = waveform_features_dict[sample_key]
        if sample_features:
            print(f"\nFeature names ({len(sample_features)} total):")
            feature_names = sorted(list(sample_features.keys()))
            print(", ".join(feature_names[:5]) + "...")
            
            if print_sample:
                print("\nSample feature values:")
                for name, value in list(sample_features.items())[:5]:
                    print(f"{name}: {value}")
    
    # Check for NaN or infinite values
    nan_count = 0
    inf_count = 0
    for features in waveform_features_dict.values():
        if features:
            for value in features.values():
                if isinstance(value, (int, float)):
                    if np.isnan(value):
                        nan_count += 1
                    elif np.isinf(value):
                        inf_count += 1
    
    print(f"\nInvalid values found:")
    print(f"NaN values: {nan_count}")
    print(f"Infinite values: {inf_count}")

def debug_sequences(sequences, max_display=3):
    """Debug function to check created sequences"""
    print("\n=== Sequences Debug ===")
    
    # Basic statistics
    num_sequences = len(sequences)
    print(f"Total number of sequences: {num_sequences}")
    
    if num_sequences == 0:
        print("WARNING: No sequences created!")
        return
    
    # Check sequence structure
    print("\nSequence structure:")
    sample_seq = sequences[0]
    print(f"Metadata shape: {sample_seq[0].shape}")
    print(f"Number of waveform feature sets: {len(sample_seq[1])}")
    print(f"Target shape: {sample_seq[2].shape}")
    
    # Display sample sequences
    print(f"\nFirst {max_display} sequences:")
    for i, (metadata, waveform_features, target) in enumerate(sequences[:max_display]):
        print(f"\nSequence {i+1}:")
        print(f"Metadata range:")
        print(f"  Latitude: {metadata[:, 0].min():.4f} to {metadata[:, 0].max():.4f}")
        print(f"  Longitude: {metadata[:, 1].min():.4f} to {metadata[:, 1].max():.4f}")
        print(f"  Time range: {metadata[:, 3].min():.2f} to {metadata[:, 3].max():.2f} hours")
        print(f"Target location: ({target[0]:.4f}, {target[1]:.4f})")

def debug_graphs(graph_dataset, feature_names):
    """Debug function to check graph representations"""
    print("\n=== Graph Dataset Debug ===")
    
    # Basic statistics
    num_graphs = len(graph_dataset)
    print(f"Total number of graphs: {num_graphs}")
    
    if num_graphs == 0:
        print("WARNING: Empty graph dataset!")
        return
    
    # Check first few graphs in detail
    for i, graph in enumerate(graph_dataset[:3]):
        print(f"\nGraph {i+1}:")
        print(f"Number of nodes: {graph.num_nodes}")
        print(f"Number of edges: {graph.edge_index.size(1)}")
        print(f"Metadata tensor shape: {graph.metadata.shape}")
        print(f"Waveform tensor shape: {graph.waveform.shape}")
        print(f"Target shape: {graph.y.shape}")
        
        # Check for isolated nodes
        edge_set = set(graph.edge_index.view(-1).tolist())
        isolated_nodes = [n for n in range(graph.num_nodes) if n not in edge_set]
        if isolated_nodes:
            print(f"WARNING: Found {len(isolated_nodes)} isolated nodes!")
        
        # Check feature statistics
        if hasattr(graph, 'waveform'):
            print("\nWaveform feature statistics:")
            waveform_mean = graph.waveform.mean(dim=0)
            waveform_std = graph.waveform.std(dim=0)
            print(f"Mean range: {waveform_mean.min():.4f} to {waveform_mean.max():.4f}")
            print(f"Std range: {waveform_std.min():.4f} to {waveform_std.max():.4f}")
    
    # Overall dataset statistics
    total_nodes = sum(graph.num_nodes for graph in graph_dataset)
    total_edges = sum(graph.edge_index.size(1) for graph in graph_dataset)
    print(f"\nDataset statistics:")
    print(f"Total nodes: {total_nodes}")
    print(f"Total edges: {total_edges}")
    print(f"Average nodes per graph: {total_nodes/num_graphs:.2f}")
    print(f"Average edges per graph: {total_edges/num_graphs:.2f}")

def visualize_graph_structure(graph, index):
    """Create visualization of graph structure"""
    import networkx as nx
    
    # Convert to networkx graph
    G = nx.Graph()
    edge_index = graph.edge_index.numpy()
    
    # Add edges
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    
    # Create plot
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold')
    
    plt.title(f'Graph Structure - Example {index}')
    plt.savefig(f'results/graph_structure_{index}.png')
    plt.close()