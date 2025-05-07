# waveform_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import torch
import seaborn as sns
from scipy import stats
import pickle

# Import your existing functions
from relative_gnn import (
    read_data_from_pickle,
    create_causal_spatiotemporal_graph,
    RelativeGNNAftershockPredictor,
    plot_relative_results,
    plot_3d_aftershocks,
    extract_waveform_features,
)


def analyze_feature_importance(waveform_features, preprocessor):
    """Analyze which features were selected and their importance"""

    # Define feature names
    feature_names = []
    components = ["Vertical", "North-South", "East-West"]
    time_features = [
        "Max Amplitude",
        "Mean Amplitude",
        "StdDev",
        "Skewness",
        "Kurtosis",
    ]
    freq_features = [
        "Spectral Centroid",
        "Low Freq Energy",
        "Mid Freq Energy",
        "High Freq Energy",
    ]

    for comp in components:
        for feat in time_features:
            feature_names.append(f"{comp} {feat}")
        for feat in freq_features:
            feature_names.append(f"{comp} {feat}")

    # Get feature importance from preprocessor
    importance = preprocessor.feature_importance
    selected = preprocessor.selected_indices

    # Print mapping of feature names to importance
    print("Selected Features and their Importance:")
    for idx, imp in sorted(
        zip(selected, importance[selected]), key=lambda x: x[1], reverse=True
    ):
        print(f"{feature_names[idx]}: {imp:.6f}")

    return feature_names, importance, selected


class WaveformPreprocessor:
    """
    Specialized preprocessing for waveform features to improve their contribution
    """

    def __init__(self, log_transform=True, feature_selection=True, top_k_features=10):
        self.log_transform = log_transform
        self.feature_selection = feature_selection
        self.top_k = top_k_features
        self.selected_indices = None
        self.feature_importance = None

    def fit(self, waveform_features):
        """Find most informative waveform features"""
        # Apply log transform if needed
        if self.log_transform:
            # Add small constant to avoid log(0)
            features = self._safe_log_transform(waveform_features)
        else:
            features = waveform_features

        if self.feature_selection:
            # Calculate variance for each feature
            variances = np.var(features, axis=0)

            # Calculate coefficient of variation (normalized variance)
            means = np.mean(np.abs(features), axis=0)
            cv = np.zeros_like(variances)
            nonzero_means = means != 0
            cv[nonzero_means] = variances[nonzero_means] / means[nonzero_means]

            # Select top_k features with highest CV
            self.selected_indices = np.argsort(cv)[-self.top_k :]
            self.feature_importance = cv

            print(f"Selected {len(self.selected_indices)} waveform features")
            print(f"Top feature importance values: {cv[self.selected_indices]}")
        else:
            # Use all features
            self.selected_indices = np.arange(waveform_features.shape[1])

        return self

    def transform(self, waveform_features):
        """Apply preprocessing to waveform features"""
        if self.log_transform:
            features = self._safe_log_transform(waveform_features)
        else:
            features = waveform_features

        if self.selected_indices is not None:
            return features[:, self.selected_indices]
        return features

    def _safe_log_transform(self, data):
        """Apply log transform while handling negative values"""
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            col = data[:, i]
            min_val = np.min(col)

            if min_val < 0:
                # Shift to make all values positive
                shifted = col - min_val + 1e-6
                result[:, i] = np.log1p(shifted)
            else:
                # Add small constant to avoid log(0)
                result[:, i] = np.log1p(col + 1e-6)

        return result


def comparative_experiment(seed=42):
    """
    Run a controlled experiment comparing models with and without waveform features
    using identical train/val/test splits and random seeds for fair comparison.
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create the output directory
    os.makedirs("results", exist_ok=True)

    print("========== COMPARATIVE EXPERIMENT: WAVEFORM FEATURES ==========")
    print(
        "This experiment compares model performance with and without waveform features"
    )

    # Load the data
    if not os.path.exists("aftershock_data.pkl"):
        print("Error: aftershock_data.pkl not found")
        return

    # Load the data
    df = read_data_from_pickle("aftershock_data.pkl")

    # Sort data chronologically but DO NOT shuffle
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df["source_origin_time"])
    df_sorted = df_sorted.sort_values("timestamp").drop("timestamp", axis=1)

    # Just use the sorted dataframe as is - no shuffling!
    df = df_sorted[2:].reset_index(drop=True)
    print(f"Loaded data with {len(df)} events")

    # Extract and preprocess waveform features for all events
    print("\n1. EXTRACTING AND PREPROCESSING WAVEFORM FEATURES")
    waveform_features = []
    for i, waveform in enumerate(tqdm(df["waveform"])):
        features = extract_waveform_features(waveform)
        waveform_features.append(features)

    waveform_features = np.array(waveform_features)

    # Create and fit waveform preprocessor
    preprocessor = WaveformPreprocessor(
        log_transform=True, feature_selection=True, top_k_features=5
    )
    preprocessor.fit(waveform_features)

    # Analyze feature importance
    feature_names, importance, selected = analyze_feature_importance(
        waveform_features, preprocessor
    )

    # Save feature importance to file
    importance_df = pd.DataFrame(
        {
            "Feature": [feature_names[i] for i in selected],
            "Importance": importance[selected],
        }
    ).sort_values("Importance", ascending=False)

    importance_df.to_csv("results/feature_importance.csv", index=False)

    # Create a bar chart of feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance (Coefficient of Variation)")
    plt.title("Waveform Feature Importance")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=300, bbox_inches="tight")

    # Base parameters for both models
    base_params = {
        "time_window": 120,  # hours
        "spatial_threshold": 75,  # km
        "min_connections": 5,
        "model_type": "gat",
        "hidden_dim": 128,
        "num_layers": 3,
        "learning_rate": 0.0025,
        "batch_size": 8,
        "weight_decay": 5e-6,
        "epochs": 100,
        "patience": 20,
    }

    # Define functions for creating graphs with and without waveform features

    # VERSION 1: WITH waveform features (preprocessed)
    def modified_extract_with_waveforms(waveform):
        """Extract waveform features with preprocessing"""
        raw_features = extract_waveform_features(waveform)
        return preprocessor.transform(raw_features.reshape(1, -1)).flatten()

    print("\n2. CREATING GRAPHS FOR BOTH MODELS (SHARED DATA)")

    # Create common dataset for both models
    # This is important to ensure we're comparing models fairly
    print("\nCreating graphs with preprocessed waveform features...")

    # Store original function
    original_extract_waveform_features = extract_waveform_features

    # Temporarily replace it with the version that uses preprocessed features
    # We'll use this for both models to ensure identical graph structure
    import types
    import inspect

    module = inspect.getmodule(extract_waveform_features)
    module.extract_waveform_features = modified_extract_with_waveforms

    # Create graphs with processed waveform features
    graphs_with_waveforms, reference_coords = create_causal_spatiotemporal_graph(
        df,
        time_window=base_params["time_window"],
        spatial_threshold=base_params["spatial_threshold"],
        min_connections=base_params["min_connections"],
    )

    print(f"Created {len(graphs_with_waveforms)} causal graphs with waveform features")

    # Now create a copy of the graphs but with waveform features zeroed out
    print("\nCreating identical graphs with zeroed waveform features...")

    graphs_without_waveforms = []
    waveform_dim = len(preprocessor.selected_indices)

    # Make deep copies of graphs but zero out the waveform features
    for data in graphs_with_waveforms:
        # Create a copy of the graph
        new_data = data.clone()

        # Zero out waveform features for all nodes
        for i in range(new_data.num_nodes):
            # First waveform_dim features are waveform features
            new_data.x[i, :waveform_dim] = 0

        graphs_without_waveforms.append(new_data)

    print(
        f"Created {len(graphs_without_waveforms)} causal graphs without waveform features"
    )

    # Restore original function
    module.extract_waveform_features = original_extract_waveform_features

    # We'll create a function to train and evaluate a model to avoid code duplication
    def train_and_evaluate_model(graphs, model_name, params, reference_coords):
        print(f"\n===== TRAINING AND EVALUATING MODEL: {model_name} =====")

        # Create predictor
        predictor = RelativeGNNAftershockPredictor(
            graph_data_list=graphs,
            reference_coords=reference_coords,
            gnn_type=params["model_type"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            weight_decay=params["weight_decay"],
        )

        # Train the model
        train_losses, val_losses = predictor.train(
            num_epochs=params["epochs"], patience=params["patience"]
        )

        # Test the model
        metrics, y_true, y_pred, errors = predictor.test()

        # Print metrics
        print(f"\n{model_name} Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Plot results
        plot_relative_results(
            y_true,
            y_pred,
            errors,
            reference_coords=reference_coords,
            model_name=model_name,
        )

        plot_3d_aftershocks(
            y_true, y_pred, reference_coords=reference_coords, model_name=model_name
        )

        # Save metrics and predictions to file
        results = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        with open(f"results/{model_name}_results.pkl", "wb") as f:
            pickle.dump(results, f)

        return metrics, y_true, y_pred, train_losses, val_losses

    # Train and evaluate both models
    print("\n3. TRAINING AND EVALUATING BOTH MODELS")

    # Model WITH waveform features
    metrics_with, y_true_with, y_pred_with, train_losses_with, val_losses_with = (
        train_and_evaluate_model(
            graphs_with_waveforms, "with_waveforms", base_params, reference_coords
        )
    )

    # Model WITHOUT waveform features
    (
        metrics_without,
        y_true_without,
        y_pred_without,
        train_losses_without,
        val_losses_without,
    ) = train_and_evaluate_model(
        graphs_without_waveforms, "without_waveforms", base_params, reference_coords
    )

    # Now create comparison visualizations and metrics
    print("\n4. COMPARING MODEL PERFORMANCE")

    # 1. Compare key metrics
    key_metrics = [
        "mean_horizontal_error",
        "median_horizontal_error",
        "mean_depth_error",
        "median_depth_error",
        "mean_3d_error",
        "median_3d_error",
        "horizontal_5km",
        "horizontal_10km",
        "horizontal_15km",
        "depth_5km",
        "depth_10km",
        "depth_15km",
        "3d_5km",
        "3d_10km",
        "3d_15km",
    ]

    metrics_df = pd.DataFrame(
        {
            "Metric": key_metrics,
            "With Waveforms": [metrics_with.get(m, np.nan) for m in key_metrics],
            "Without Waveforms": [metrics_without.get(m, np.nan) for m in key_metrics],
        }
    )

    # Calculate improvement percentage
    metrics_df["Improvement"] = np.nan

    for i, metric in enumerate(key_metrics):
        with_val = metrics_with.get(metric, np.nan)
        without_val = metrics_without.get(metric, np.nan)

        if np.isnan(with_val) or np.isnan(without_val):
            continue

        if "error" in metric:
            # For error metrics, lower is better
            improvement = ((without_val - with_val) / without_val) * 100
        else:
            # For success rate metrics, higher is better
            improvement = ((with_val - without_val) / without_val) * 100

        metrics_df.loc[i, "Improvement"] = improvement

    # Save comparison to CSV
    metrics_df.to_csv("results/metrics_comparison.csv", index=False)

    # 2. Create comparison bar chart
    plt.figure(figsize=(15, 10))

    # Error metrics (lower is better)
    error_metrics = [m for m in key_metrics if "error" in m]
    error_df = metrics_df[metrics_df["Metric"].isin(error_metrics)]

    x = np.arange(len(error_df))
    width = 0.35

    plt.subplot(2, 1, 1)
    plt.bar(x - width / 2, error_df["With Waveforms"], width, label="With Waveforms")
    plt.bar(
        x + width / 2, error_df["Without Waveforms"], width, label="Without Waveforms"
    )
    plt.xlabel("Metric")
    plt.ylabel("Error (km)")
    plt.title("Error Metrics Comparison (lower is better)")
    plt.xticks(x, error_df["Metric"], rotation=45)
    plt.legend()

    # Success rate metrics (higher is better)
    success_metrics = [m for m in key_metrics if "km" in m and "error" not in m]
    success_df = metrics_df[metrics_df["Metric"].isin(success_metrics)]

    x = np.arange(len(success_df))

    plt.subplot(2, 1, 2)
    plt.bar(x - width / 2, success_df["With Waveforms"], width, label="With Waveforms")
    plt.bar(
        x + width / 2, success_df["Without Waveforms"], width, label="Without Waveforms"
    )
    plt.xlabel("Metric")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate Metrics Comparison (higher is better)")
    plt.xticks(x, success_df["Metric"], rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/metrics_comparison.png", dpi=300, bbox_inches="tight")

    # 3. Compare learning curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_with, label="With Waveforms")
    plt.plot(train_losses_without, label="Without Waveforms")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_losses_with, label="With Waveforms")
    plt.plot(val_losses_without, label="Without Waveforms")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/learning_curves_comparison.png", dpi=300, bbox_inches="tight")

    # 4. Create prediction error comparison histograms
    if len(y_true_with) == len(y_true_without):
        # Calculate horizontal errors
        horiz_errors_with = np.sqrt(
            (y_true_with[:, 0] - y_pred_with[:, 0]) ** 2
            + (y_true_with[:, 1] - y_pred_with[:, 1]) ** 2
        )

        horiz_errors_without = np.sqrt(
            (y_true_without[:, 0] - y_pred_without[:, 0]) ** 2
            + (y_true_without[:, 1] - y_pred_without[:, 1]) ** 2
        )

        # Calculate depth errors
        depth_errors_with = np.abs(y_true_with[:, 2] - y_pred_with[:, 2])
        depth_errors_without = np.abs(y_true_without[:, 2] - y_pred_without[:, 2])

        # Create histograms
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.hist(horiz_errors_with, bins=30, alpha=0.5, label="With Waveforms")
        plt.hist(horiz_errors_without, bins=30, alpha=0.5, label="Without Waveforms")
        plt.xlabel("Horizontal Error (km)")
        plt.ylabel("Count")
        plt.title("Horizontal Error Distribution Comparison")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.hist(depth_errors_with, bins=30, alpha=0.5, label="With Waveforms")
        plt.hist(depth_errors_without, bins=30, alpha=0.5, label="Without Waveforms")
        plt.xlabel("Depth Error (km)")
        plt.ylabel("Count")
        plt.title("Depth Error Distribution Comparison")
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            "results/error_distribution_comparison.png", dpi=300, bbox_inches="tight"
        )

        # Statistical significance test
        print("\n5. STATISTICAL SIGNIFICANCE TESTS")

        # Horizontal error significance test
        t_stat_horiz, p_val_horiz = stats.ttest_ind(
            horiz_errors_with, horiz_errors_without
        )
        print(f"Horizontal Error t-test: t={t_stat_horiz:.4f}, p={p_val_horiz:.4f}")

        # Depth error significance test
        t_stat_depth, p_val_depth = stats.ttest_ind(
            depth_errors_with, depth_errors_without
        )
        print(f"Depth Error t-test: t={t_stat_depth:.4f}, p={p_val_depth:.4f}")

        # Wilcoxon signed-rank test (for non-normal distributions)
        w_stat_horiz, p_val_w_horiz = stats.wilcoxon(
            horiz_errors_with, horiz_errors_without
        )
        print(
            f"Horizontal Error Wilcoxon test: W={w_stat_horiz:.4f}, p={p_val_w_horiz:.4f}"
        )

        w_stat_depth, p_val_w_depth = stats.wilcoxon(
            depth_errors_with, depth_errors_without
        )
        print(f"Depth Error Wilcoxon test: W={w_stat_depth:.4f}, p={p_val_w_depth:.4f}")

    # 5. Create summary report
    with open("results/waveform_comparison_summary.txt", "w") as f:
        f.write("================================================\n")
        f.write("   WAVEFORM FEATURES CONTRIBUTION EXPERIMENT    \n")
        f.write("================================================\n\n")

        f.write("This experiment compared two models:\n")
        f.write("1. WITH waveform features (preprocessed)\n")
        f.write("2. WITHOUT waveform features (zeroed out)\n\n")

        f.write("Top waveform features by importance:\n")
        for idx, imp in sorted(
            zip(selected, importance[selected]), key=lambda x: x[1], reverse=True
        ):
            f.write(f"- {feature_names[idx]}: {imp:.6f}\n")
        f.write("\n")

        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-----------------------\n")
        for i, row in metrics_df.iterrows():
            metric = row["Metric"]
            with_val = row["With Waveforms"]
            without_val = row["Without Waveforms"]
            improvement = row["Improvement"]

            if pd.isna(improvement):
                continue

            better = "better" if improvement > 0 else "worse"
            f.write(f"{metric}:\n")
            f.write(f"  With waveforms:    {with_val:.4f}\n")
            f.write(f"  Without waveforms: {without_val:.4f}\n")
            f.write(
                f"  Difference:        {abs(with_val - without_val):.4f} ({abs(improvement):.2f}% {better})\n\n"
            )

        if "p_val_horiz" in locals():
            f.write("STATISTICAL SIGNIFICANCE:\n")
            f.write("-------------------------\n")
            f.write(
                f"Horizontal Error t-test: t={t_stat_horiz:.4f}, p={p_val_horiz:.4f}"
            )
            f.write(
                " (statistically significant)\n"
                if p_val_horiz < 0.05
                else " (not statistically significant)\n"
            )

            f.write(f"Depth Error t-test: t={t_stat_depth:.4f}, p={p_val_depth:.4f}")
            f.write(
                " (statistically significant)\n"
                if p_val_depth < 0.05
                else " (not statistically significant)\n"
            )

            f.write(
                f"Horizontal Error Wilcoxon test: W={w_stat_horiz:.4f}, p={p_val_w_horiz:.4f}"
            )
            f.write(
                " (statistically significant)\n"
                if p_val_w_horiz < 0.05
                else " (not statistically significant)\n"
            )

            f.write(
                f"Depth Error Wilcoxon test: W={w_stat_depth:.4f}, p={p_val_w_depth:.4f}"
            )
            f.write(
                " (statistically significant)\n"
                if p_val_w_depth < 0.05
                else " (not statistically significant)\n"
            )

        f.write("\nCONCLUSION:\n")
        f.write("-----------\n")

        # Calculate overall improvement
        error_improvements = metrics_df[metrics_df["Metric"].str.contains("error")][
            "Improvement"
        ].mean()
        success_improvements = metrics_df[~metrics_df["Metric"].str.contains("error")][
            "Improvement"
        ].mean()

        if error_improvements > 0:
            f.write(
                f"The model WITH waveform features showed significant improvement in error metrics, "
            )
            f.write(f"with an average error reduction of {error_improvements:.2f}%.\n")
        else:
            f.write(
                f"The model WITHOUT waveform features performed better on error metrics, "
            )
            f.write(
                f"with {-error_improvements:.2f}% lower errors than the model with waveforms.\n"
            )

        if success_improvements > 0:
            f.write(f"The model WITH waveform features showed improved success rates, ")
            f.write(f"with an average improvement of {success_improvements:.2f}%.\n")
        else:
            f.write(f"The model WITHOUT waveform features had better success rates, ")
            f.write(
                f"with {-success_improvements:.2f}% higher success rates than the model with waveforms.\n"
            )

    print("\nExperiment complete. Results saved to results directory")
    print("See 'results/waveform_comparison_summary.txt' for detailed comparison")


if __name__ == "__main__":
    comparative_experiment(seed=42)
