#!/usr/bin/env python
"""
Script to run and compare both the baseline and waveform-enhanced versions
of the aftershock prediction GNN models.

This script executes the full pipeline:
1. Run baseline model (metadata only)
2. Run waveform-enhanced model
3. Compare results and generate visualizations

Example usage:
    python compare_models.py --epochs 300 --batch_size 16 --sequence_length 15
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import seaborn as sns
from scipy import stats

# Import waveform modules
from waveform_gnn import (
    load_aftershock_data_with_waveforms,
    identify_mainshock_and_aftershocks,
    create_aftershock_sequences_with_waveforms,
    WaveformFeatureExtractor,
    AfterShockGNN,
    consolidate_station_recordings,
    build_graphs_from_sequences_with_waveforms,
    normalize_waveform_features,
    train_waveform_gnn_model,
    evaluate_waveform_model,
)

# Import baseline modules
from baseline_aftershock_gnn import (
    load_aftershock_data,
    create_aftershock_sequences,
    build_graphs_from_sequences,
    BaselineAfterShockGNN,
    train_gnn_model,
    evaluate_model,
    run_baseline_comparison,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compare Baseline and Waveform-Enhanced Aftershock GNN Models"
    )

    # Data parameters
    parser.add_argument(
        "--time_window",
        type=int,
        default=24,
        help="Time window for connecting events (hours)",
    )
    parser.add_argument(
        "--distance_threshold",
        type=int,
        default=30,
        help="Distance threshold for connections (km)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=5,
        help="Number of events in each sequence",
    )
    parser.add_argument(
        "--max_waveforms",
        type=int,
        default=13400,
        help="Maximum number of waveforms to process",
    )

    # Model parameters
    parser.add_argument(
        "--hidden_channels", type=int, default=64, help="Size of hidden layers"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of GNN layers"
    )
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )

    # Execution parameters
    parser.add_argument(
        "--skip_waveform", action="store_true", help="Skip waveform model training"
    )
    parser.add_argument(
        "--skip_baseline", action="store_true", help="Skip baseline model training"
    )

    return parser.parse_args()


def compare_models(baseline_errors, waveform_errors):
    """
    Compare baseline model with waveform-enhanced model and generate visualizations

    Parameters:
    -----------
    baseline_errors : list
        List of prediction errors from baseline model
    waveform_errors : list
        List of prediction errors from waveform-enhanced model
    """
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

    print(f"\n=== Model Comparison ===")
    print(
        f"Baseline Model - Mean Error: {baseline_mean:.2f} km, Median Error: {baseline_median:.2f} km, Std Dev: {baseline_std:.2f} km"
    )
    print(
        f"Waveform Model - Mean Error: {waveform_mean:.2f} km, Median Error: {waveform_median:.2f} km, Std Dev: {waveform_std:.2f} km"
    )
    print(
        f"Improvement with waveform features - Mean: {mean_improvement:.2f}%, Median: {median_improvement:.2f}%"
    )
    print(f"Statistical Test - t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Plot comparison histograms
    plt.figure(figsize=(12, 8))

    # Plot histograms using kernel density estimation
    sns.histplot(
        baseline_errors,
        bins=30,
        alpha=0.5,
        color="blue",
        kde=True,
        label="Baseline Model",
    )
    sns.histplot(
        waveform_errors,
        bins=30,
        alpha=0.5,
        color="green",
        kde=True,
        label="Waveform Model",
    )

    # Plot mean lines
    plt.axvline(
        baseline_mean,
        color="blue",
        linestyle="dashed",
        linewidth=2,
        label=f"Baseline Mean: {baseline_mean:.2f} km",
    )
    plt.axvline(
        waveform_mean,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Waveform Mean: {waveform_mean:.2f} km",
    )

    plt.xlabel("Error (km)")
    plt.ylabel("Frequency")
    plt.title("Comparison of Prediction Errors Between Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/model_comparison_histogram.png", dpi=300)
    plt.close()

    # Create box plot comparison
    plt.figure(figsize=(10, 8))
    box_data = [baseline_errors, waveform_errors]
    box_labels = ["Baseline Model", "Waveform-Enhanced Model"]

    sns.boxplot(data=box_data, width=0.5)
    plt.xticks([0, 1], box_labels)
    plt.ylabel("Error (km)")
    plt.title("Error Distribution Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig("results/model_comparison_boxplot.png", dpi=300)
    plt.close()

    # Create scatter plot of errors
    if len(baseline_errors) == len(waveform_errors):
        plt.figure(figsize=(10, 8))
        plt.scatter(baseline_errors, waveform_errors, alpha=0.5)

        # Add diagonal line (x=y)
        max_val = max(max(baseline_errors), max(waveform_errors))
        plt.plot([0, max_val], [0, max_val], "r--", alpha=0.7)

        plt.xlabel("Baseline Model Error (km)")
        plt.ylabel("Waveform-Enhanced Model Error (km)")
        plt.title("Error Comparison by Sample")
        plt.grid(True, alpha=0.3)
        plt.savefig("results/model_comparison_scatter.png", dpi=300)
        plt.close()

    # Create improvement visualization
    plt.figure(figsize=(10, 6))
    plt.bar(
        ["Mean Error", "Median Error"],
        [mean_improvement, median_improvement],
        color=["#3498db", "#2ecc71"],
    )

    # Add value labels on top of bars
    for i, v in enumerate([mean_improvement, median_improvement]):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center")

    plt.axhline(0, color="black", linestyle="-", alpha=0.3)
    plt.ylabel("Improvement (%)")
    plt.title("Waveform Features Improvement")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig("results/model_improvement.png", dpi=300)
    plt.close()

    # Create summary table as a figure
    plt.figure(figsize=(10, 6))
    plt.axis("tight")
    plt.axis("off")

    table_data = [
        ["Metric", "Baseline Model", "Waveform Model", "Improvement (%)"],
        [
            "Mean Error (km)",
            f"{baseline_mean:.2f}",
            f"{waveform_mean:.2f}",
            f"{mean_improvement:.2f}%",
        ],
        [
            "Median Error (km)",
            f"{baseline_median:.2f}",
            f"{waveform_median:.2f}",
            f"{median_improvement:.2f}%",
        ],
        ["Std. Deviation (km)", f"{baseline_std:.2f}", f"{waveform_std:.2f}", "N/A"],
        ["Statistical Significance", "N/A", "N/A", f"p-value: {p_value:.4f}"],
    ]

    table = plt.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    plt.title("Model Comparison Summary", fontsize=16, pad=20)
    plt.savefig("results/model_comparison_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "baseline_mean": baseline_mean,
        "baseline_median": baseline_median,
        "waveform_mean": waveform_mean,
        "waveform_median": waveform_median,
        "mean_improvement": mean_improvement,
        "median_improvement": median_improvement,
        "p_value": p_value,
    }


def run_waveform_model(args):
    """
    Run the waveform-enhanced model

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    tuple
        (mean_error, median_error, errors_km)
    """
    print("\n=== Running Waveform-Enhanced Model ===")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load and preprocess data with waveforms
    print("\n=== Loading and Preprocessing Data with Waveforms ===")
    metadata, iquique, waveform_features_dict = load_aftershock_data_with_waveforms(
        max_waveforms=args.max_waveforms
    )

    # Consolidate station recordings
    print(f"Original dataset: {len(metadata)} recordings")
    metadata, waveform_features_dict = consolidate_station_recordings(
        metadata, waveform_features_dict
    )
    print(f"Consolidated dataset: {len(metadata)} unique events")

    # Identify mainshock and aftershocks
    mainshock, aftershocks = identify_mainshock_and_aftershocks(metadata)

    # Create aftershock sequences with waveform features
    print("\n=== Creating Aftershock Sequences with Waveform Features ===")
    sequences = create_aftershock_sequences_with_waveforms(
        aftershocks,
        waveform_features_dict,
        sequence_length=args.sequence_length,
        time_window_hours=args.time_window,
    )

    # Build graph representations with waveform features
    print("\n=== Building Graph Representations with Waveform Features ===")
    graph_dataset, feature_names = build_graphs_from_sequences_with_waveforms(
        sequences, distance_threshold_km=args.distance_threshold
    )

    if len(graph_dataset) == 0:
        print("No valid graph representations created. Exiting.")
        return None

    # Normalize waveform features
    print("\n=== Normalizing Waveform Features ===")
    normalized_dataset = normalize_waveform_features(graph_dataset, feature_names)

    # Create the GNN model for waveform features
    print("\n=== Creating Waveform-Enhanced GNN Model ===")
    metadata_channels = 4  # latitude, longitude, depth, hours since mainshock
    waveform_channels = len(feature_names)  # Number of waveform features

    waveform_model = AfterShockGNN(
        metadata_channels=metadata_channels,
        waveform_channels=waveform_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(waveform_model)

    # Train the waveform-enhanced model
    print("\n=== Training Waveform-Enhanced GNN Model ===")
    waveform_model, train_losses, val_losses = train_waveform_gnn_model(
        normalized_dataset,
        waveform_model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # Evaluate the waveform-enhanced model
    print("\n=== Evaluating Waveform-Enhanced Model ===")
    waveform_mean_error, waveform_median_error, waveform_errors_km = (
        evaluate_waveform_model(waveform_model, normalized_dataset, mainshock)
    )

    return waveform_mean_error, waveform_median_error, waveform_errors_km, mainshock


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Record start time
    start_time = datetime.now()
    print(f"Starting Model Comparison at {start_time}")

    # Run baseline model if not skipped
    baseline_results = None
    if not args.skip_baseline:
        baseline_mean_error, baseline_median_error, baseline_errors_km = (
            run_baseline_comparison(args)
        )
        baseline_results = (
            baseline_mean_error,
            baseline_median_error,
            baseline_errors_km,
        )

    # Run waveform-enhanced model if not skipped
    waveform_results = None
    if not args.skip_waveform:
        waveform_mean_error, waveform_median_error, waveform_errors_km, mainshock = (
            run_waveform_model(args)
        )
        waveform_results = (
            waveform_mean_error,
            waveform_median_error,
            waveform_errors_km,
        )

    # Compare models if both were run
    comparison_results = None
    if baseline_results and waveform_results:
        baseline_mean_error, baseline_median_error, baseline_errors_km = (
            baseline_results
        )
        waveform_mean_error, waveform_median_error, waveform_errors_km = (
            waveform_results
        )

        print("\n=== Comparing Baseline and Waveform-Enhanced Models ===")
        comparison_results = compare_models(baseline_errors_km, waveform_errors_km)

    # Record end time and print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n=== Model Comparison Execution Summary ===")
    print(f"Started at: {start_time}")
    print(f"Completed at: {end_time}")
    print(f"Total duration: {duration}")
    print(f"\nDataset: Iquique Earthquake (2014)")
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

    if baseline_results:
        baseline_mean_error, baseline_median_error, _ = baseline_results
        print(f"\nBaseline Model performance:")
        print(f"  - Mean error: {baseline_mean_error:.2f} km")
        print(f"  - Median error: {baseline_median_error:.2f} km")

    if waveform_results:
        waveform_mean_error, waveform_median_error, _ = waveform_results
        print(f"\nWaveform-Enhanced Model performance:")
        print(f"  - Mean error: {waveform_mean_error:.2f} km")
        print(f"  - Median error: {waveform_median_error:.2f} km")

    if comparison_results:
        print(f"\nImprovement with waveform features:")
        print(
            f"  - Mean error improvement: {comparison_results['mean_improvement']:.2f}%"
        )
        print(
            f"  - Median error improvement: {comparison_results['median_improvement']:.2f}%"
        )
        print(
            f"  - Statistical significance: p-value = {comparison_results['p_value']:.4f}"
        )

    print(f"\nAll results saved in the 'results' directory")


if __name__ == "__main__":
    main()
