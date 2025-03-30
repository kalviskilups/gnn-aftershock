#!/usr/bin/env python
"""
Main script to run the enhanced Aftershock GNN implementation with waveform features
for the Iquique dataset.

This script executes the full pipeline:
1. Data loading and preprocessing with waveform features
2. GNN model training with both metadata and waveform features
3. Evaluation and comparison with baseline model
4. Visualization of results

Example usage:
    python run_waveform_gnn.py --epochs 200 --batch_size 32 --sequence_length 10
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

# Import implementation modules
from waveform_gnn import (
    load_aftershock_data_with_waveforms,
    identify_mainshock_and_aftershocks,
    create_aftershock_sequences_with_waveforms,
    WaveformFeatureExtractor,
    AfterShockGNN
)

# Import graph building modules
from waveform_gnn import (
    build_graphs_from_sequences_with_waveforms,
    normalize_waveform_features,
    train_waveform_gnn_model,
    evaluate_waveform_model,
    compare_models
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Waveform-Enhanced Aftershock GNN Training')
    
    # Data parameters
    parser.add_argument('--time_window', type=int, default=12,
                        help='Time window for connecting events (hours)')
    parser.add_argument('--distance_threshold', type=int, default=25,
                        help='Distance threshold for connections (km)')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Number of events in each sequence')
    parser.add_argument('--max_waveforms', type=int, default=3000,
                        help='Maximum number of waveforms to process')
    
    # Model parameters
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Size of hidden layers')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    
    # Execution parameters
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline model training')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training (load from file)')
    parser.add_argument('--model_path', type=str, default='results/waveform_gnn_model.pt',
                        help='Path to save/load model')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Record start time
    start_time = datetime.now()
    print(f"Starting Waveform-Enhanced Aftershock GNN at {start_time}")
    
    # Step 1: Load and preprocess data with waveforms
    print("\n=== Loading and Preprocessing Data with Waveforms ===")
    metadata, iquique, waveform_features_dict = load_aftershock_data_with_waveforms(
        max_waveforms=args.max_waveforms
    )
    
    # Step 2: Identify mainshock and aftershocks
    mainshock, aftershocks = identify_mainshock_and_aftershocks(metadata)
    
    # Step 3: Create aftershock sequences with waveform features
    print("\n=== Creating Aftershock Sequences with Waveform Features ===")
    sequences = create_aftershock_sequences_with_waveforms(
        aftershocks,
        waveform_features_dict,
        sequence_length=args.sequence_length,
        time_window_hours=args.time_window
    )
    
    # Step 4: Build graph representations with waveform features
    print("\n=== Building Graph Representations with Waveform Features ===")
    graph_dataset, feature_names = build_graphs_from_sequences_with_waveforms(
        sequences,
        distance_threshold_km=args.distance_threshold
    )
    
    if len(graph_dataset) == 0:
        print("No valid graph representations created. Exiting.")
        return
    
    # Step 5: Normalize waveform features
    print("\n=== Normalizing Waveform Features ===")
    normalized_dataset = normalize_waveform_features(graph_dataset, feature_names)
    
    # Step 6: Create the GNN model for waveform features
    print("\n=== Creating Waveform-Enhanced GNN Model ===")
    metadata_channels = 4  # latitude, longitude, depth, hours since mainshock
    waveform_channels = len(feature_names)  # Number of waveform features
    
    waveform_model = AfterShockGNN(
        metadata_channels=metadata_channels,
        waveform_channels=waveform_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    print(waveform_model)
    
    # Step 7: Train the waveform-enhanced model (or load from file)
    if not args.skip_training:
        print("\n=== Training Waveform-Enhanced GNN Model ===")
        waveform_model, train_losses, val_losses = train_waveform_gnn_model(
            normalized_dataset,
            waveform_model,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience
        )
    else:
        print(f"\n=== Loading Waveform Model from {args.model_path} ===")
        try:
            waveform_model.load_state_dict(torch.load(args.model_path))
            print("Waveform model loaded successfully")
        except Exception as e:
            print(f"Error loading waveform model: {e}")
            return
    
    # Step 8: Evaluate the waveform-enhanced model
    print("\n=== Evaluating Waveform-Enhanced Model ===")
    waveform_mean_error, waveform_median_error, waveform_errors_km = evaluate_waveform_model(
        waveform_model, 
        normalized_dataset, 
        mainshock
    )
    
    # Step 9: If not skipping baseline, train and evaluate baseline model for comparison
    baseline_errors_km = []
    
    if not args.skip_baseline:
        # This would import and run the original implementation
        try:
            from waveform_gnn import (
                load_aftershock_data,
                identify_mainshock_and_aftershocks as original_identify_mainshock,
                create_aftershock_sequences,
                build_graphs_from_sequences,
                AfterShockGNN as OriginalAfterShockGNN,
                train_gnn_model,
                evaluate_model
            )
            
            print("\n=== Running Baseline Model for Comparison ===")
            
            # Load data for baseline
            baseline_metadata, baseline_iquique = load_aftershock_data()
            
            # Identify mainshock and aftershocks
            baseline_mainshock, baseline_aftershocks = original_identify_mainshock(baseline_metadata)
            
            # Create aftershock sequences
            baseline_sequences = create_aftershock_sequences(
                baseline_aftershocks,
                sequence_length=args.sequence_length,
                time_window_hours=args.time_window
            )
            
            # Build graph representations
            baseline_graph_dataset = build_graphs_from_sequences(
                baseline_sequences,
                distance_threshold_km=args.distance_threshold
            )
            
            if len(baseline_graph_dataset) > 0:
                # Create baseline model
                in_channels = 4  # latitude, longitude, depth, hours since mainshock
                baseline_model = OriginalAfterShockGNN(
                    in_channels=in_channels,
                    hidden_channels=args.hidden_channels,
                    num_layers=args.num_layers,
                    dropout=args.dropout
                ).to(device)
                
                # Train baseline model
                baseline_model, baseline_train_losses, baseline_val_losses = train_gnn_model(
                    baseline_graph_dataset,
                    baseline_model,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    patience=args.patience
                )
                
                # Evaluate baseline model
                baseline_mean_error, baseline_median_error, baseline_errors_km = evaluate_model(
                    baseline_model,
                    baseline_graph_dataset,
                    baseline_mainshock
                )
                
                # Compare models
                print("\n=== Comparing Baseline and Waveform-Enhanced Models ===")
                compare_models(baseline_errors_km, waveform_errors_km)
            else:
                print("No valid baseline graph representations created. Skipping comparison.")
                
        except ImportError as e:
            print(f"Could not import baseline model: {e}")
            print("Skipping baseline comparison")
        except Exception as e:
            print(f"Error running baseline comparison: {e}")
            print("Skipping baseline comparison")
    
    # Record end time and print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n=== Waveform-Enhanced Aftershock GNN Execution Summary ===")
    print(f"Started at: {start_time}")
    print(f"Completed at: {end_time}")
    print(f"Total duration: {duration}")
    print(f"\nDataset: Iquique Earthquake (2014)")
    print(f"Total sequences with waveform features: {len(normalized_dataset)}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Time window: {args.time_window} hours")
    print(f"Distance threshold: {args.distance_threshold} km")
    print(f"\nModel parameters:")
    print(f"  - Hidden channels: {args.hidden_channels}")
    print(f"  - Number of layers: {args.num_layers}")
    print(f"  - Dropout rate: {args.dropout}")
    print(f"\nTraining parameters:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Early stopping patience: {args.patience}")
    print(f"\nWaveform-Enhanced Model performance:")
    print(f"  - Mean error: {waveform_mean_error:.2f} km")
    print(f"  - Median error: {waveform_median_error:.2f} km")
    
    if baseline_errors_km:
        print(f"\nBaseline Model performance:")
        print(f"  - Mean error: {baseline_mean_error:.2f} km")
        print(f"  - Median error: {baseline_median_error:.2f} km")
        
        # Calculate improvement percentage
        mean_improvement = ((baseline_mean_error - waveform_mean_error) / baseline_mean_error) * 100
        median_improvement = ((baseline_median_error - waveform_median_error) / baseline_median_error) * 100
        
        print(f"\nImprovement with waveform features:")
        print(f"  - Mean error improvement: {mean_improvement:.2f}%")
        print(f"  - Median error improvement: {median_improvement:.2f}%")
    
    print(f"\nAll results saved in the 'results' directory")

if __name__ == "__main__":
    main()