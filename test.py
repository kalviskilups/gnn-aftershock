import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import time
import os
from datetime import datetime
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA


def read_data_from_pickle(file_path):
    """
    Load the data from pickle file.
    """
    # Load the pickle file that contains the data dictionary
    with open(file_path, "rb") as file:
        data_dict = pickle.load(file)

    # Extract the metadata from each event entry.
    data_list = [
        {**entry["metadata"], "waveform": entry["waveform"]}
        for entry in data_dict.values()
    ]
    # Convert the list of metadata dictionaries into a DataFrame
    df = pd.DataFrame(data_list)

    return df


def extract_waveform_features(waveform):
    """
    Extract relevant features from a seismic waveform with advanced processing.
    """
    features = []
    for component in range(waveform.shape[0]):
        # Basic time domain features
        features.append(np.max(np.abs(waveform[component])))  # Peak amplitude
        features.append(np.mean(np.abs(waveform[component])))  # Mean amplitude
        features.append(np.std(waveform[component]))  # Standard deviation
        features.append(np.percentile(waveform[component], 75))  # 75th percentile
        features.append(np.percentile(waveform[component], 25))  # 25th percentile

        # Signal energy features
        features.append(np.sum(waveform[component] ** 2))  # Energy
        features.append(np.sqrt(np.mean(waveform[component] ** 2)))  # RMS

        # Frequency domain features
        fft = np.abs(np.fft.rfft(waveform[component]))
        features.append(np.argmax(fft))  # Dominant frequency index
        features.append(np.max(fft))  # Maximum frequency amplitude
        features.append(np.mean(fft))  # Mean frequency amplitude
        features.append(np.sum(fft))  # Total spectral energy

        # Add spectral ratios (useful for identifying event characteristics)
        # Low frequency energy (0-10Hz approx)
        low_freq_energy = np.sum(fft[: min(10, len(fft))])
        # Mid frequency energy (10-20Hz approx)
        mid_freq_energy = np.sum(fft[min(10, len(fft)) : min(20, len(fft))])
        # High frequency energy (>20Hz approx)
        high_freq_energy = np.sum(fft[min(20, len(fft)) :])

        # Add ratios
        if low_freq_energy > 0:
            features.append(high_freq_energy / low_freq_energy)  # High/Low ratio
        else:
            features.append(0)

        if mid_freq_energy > 0:
            features.append(high_freq_energy / mid_freq_energy)  # High/Mid ratio
        else:
            features.append(0)

        # Add waveform shape features
        features.append(
            np.mean(np.abs(np.diff(waveform[component])))
        )  # Average absolute derivative
        features.append(np.max(np.abs(np.diff(waveform[component]))))  # Max derivative

        # Add zero crossing rate (indicates frequency characteristics)
        zero_crossings = np.where(np.diff(np.signbit(waveform[component])))[0]
        features.append(len(zero_crossings) / len(waveform[component]))

    return np.array(features)


def extract_combined_features(df):
    """
    Create a feature matrix combining waveform features with spatial-temporal and clustering information.
    """
    print("Extracting waveform features...")
    # Extract waveform features
    waveform_features = np.array(
        [extract_waveform_features(w) for w in tqdm(df["waveform"])]
    )

    # Create feature names for waveform features
    component_names = ["Z", "N", "E"]  # Assuming ZNE component order
    time_features = [
        "peak_amp",
        "mean_amp",
        "std",
        "percentile_75",
        "percentile_25",
        "energy",
        "rms",
    ]
    freq_features = [
        "dom_freq_idx",
        "max_freq_amp",
        "mean_freq_amp",
        "total_spectral_energy",
        "high_low_ratio",
        "high_mid_ratio",
        "avg_abs_deriv",
        "max_deriv",
        "zero_crossing_rate",
    ]

    waveform_feature_names = []
    for comp in component_names:
        for feat in time_features + freq_features:
            waveform_feature_names.append(f"{comp}_{feat}")

    # Temporal features - convert to hours since first event
    print("Adding temporal features...")
    times = pd.to_datetime(df["source_origin_time"])
    time_hours = (times - times.min()).dt.total_seconds() / 3600

    # Add station information
    print("Adding station location features...")
    station_features = []
    for idx, row in df.iterrows():
        # Calculate distance from station to event
        if pd.notna(row["station_latitude_deg"]) and pd.notna(
            row["station_longitude_deg"]
        ):
            station_lat = row["station_latitude_deg"]
            station_lon = row["station_longitude_deg"]
            event_lat = row["source_latitude_deg"]
            event_lon = row["source_longitude_deg"]

            # Calculate distance in degrees (approximate)
            dist = np.sqrt(
                (station_lat - event_lat) ** 2 + (station_lon - event_lon) ** 2
            )

            # Calculate azimuth if available
            if pd.notna(row["path_back_azimuth_deg"]):
                azimuth = row["path_back_azimuth_deg"]
            else:
                azimuth = 0

            station_features.append([dist, azimuth])
        else:
            station_features.append([0, 0])

    station_features = np.array(station_features)

    # Create spatial clusters of events
    print("Adding spatial clustering features...")
    # Extract spatial coordinates
    coords = df[
        ["source_latitude_deg", "source_longitude_deg", "source_depth_km"]
    ].values

    # Compute linkage for hierarchical clustering
    Z = linkage(coords, "ward")

    # Get cluster labels for different threshold levels
    n_clusters = [5, 10, 15]  # Try different numbers of clusters
    cluster_features = []

    for n in n_clusters:
        clusters = fcluster(Z, n, criterion="maxclust")
        cluster_features.append(clusters)

    cluster_features = np.column_stack(cluster_features)

    # Combine all features
    print("Combining all features...")
    all_features = np.hstack(
        (
            waveform_features,
            time_hours.values.reshape(-1, 1),
            station_features,
            cluster_features,
        )
    )

    # Create feature names for all features
    all_feature_names = (
        waveform_feature_names
        + ["time_hours"]
        + ["station_dist", "station_azimuth"]
        + [f"cluster_{n}" for n in n_clusters]
    )

    # Apply PCA to reduce dimensionality of waveform features
    print("Applying PCA to reduce feature dimensionality...")
    pca = PCA(n_components=min(20, waveform_features.shape[1]))
    waveform_pca = pca.fit_transform(waveform_features)

    # Create a feature set with PCA-reduced waveform features
    pca_features = np.hstack(
        (
            waveform_pca,
            time_hours.values.reshape(-1, 1),
            station_features,
            cluster_features,
        )
    )

    pca_feature_names = (
        [f"pca_{i}" for i in range(waveform_pca.shape[1])]
        + ["time_hours"]
        + ["station_dist", "station_azimuth"]
        + [f"cluster_{n}" for n in n_clusters]
    )

    # Create a feature set with neighboring event information
    print("Adding nearest neighbor event features...")
    # Compute pairwise distances between events
    event_dists = distance.pdist(coords)
    event_dists = distance.squareform(event_dists)

    # For each event, find the k nearest neighbors
    k = 5  # Number of neighbors to consider
    neighbor_features = []

    for i in range(len(df)):
        # Get distances to all other events
        dists = event_dists[i]

        # Find indices of k nearest neighbors (excluding self)
        indices = np.argsort(dists)[1 : k + 1]

        # Calculate mean location of neighbors
        neighbor_lats = df.iloc[indices]["source_latitude_deg"].values
        neighbor_lons = df.iloc[indices]["source_longitude_deg"].values
        neighbor_depths = df.iloc[indices]["source_depth_km"].values

        # Average location of nearest neighbors
        mean_lat = np.mean(neighbor_lats)
        mean_lon = np.mean(neighbor_lons)
        mean_depth = np.mean(neighbor_depths)

        # Standard deviation of neighbor locations
        std_lat = np.std(neighbor_lats)
        std_lon = np.std(neighbor_lons)
        std_depth = np.std(neighbor_depths)

        # Distance to nearest neighbor
        min_dist = dists[indices[0]]

        # Mean distance to k neighbors
        mean_dist = np.mean(dists[indices])

        neighbor_features.append(
            [
                mean_lat,
                mean_lon,
                mean_depth,
                std_lat,
                std_lon,
                std_depth,
                min_dist,
                mean_dist,
            ]
        )

    neighbor_features = np.array(neighbor_features)

    # Combine with PCA features
    pca_neighbor_features = np.hstack((pca_features, neighbor_features))

    pca_neighbor_feature_names = pca_feature_names + [
        "neighbor_mean_lat",
        "neighbor_mean_lon",
        "neighbor_mean_depth",
        "neighbor_std_lat",
        "neighbor_std_lon",
        "neighbor_std_depth",
        "min_neighbor_dist",
        "mean_neighbor_dist",
    ]

    return (
        all_features,
        all_feature_names,
        pca_features,
        pca_feature_names,
        pca_neighbor_features,
        pca_neighbor_feature_names,
    )


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points in km.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r


def calculate_prediction_errors(y_true, y_pred):
    """
    Calculate various error metrics for location predictions.
    """
    # Calculate distance errors in km
    horizontal_errors = np.array(
        [
            haversine_distance(y_true[i, 0], y_true[i, 1], y_pred[i, 0], y_pred[i, 1])
            for i in range(len(y_true))
        ]
    )

    # Depth errors in km
    depth_errors = np.abs(y_true[:, 2] - y_pred[:, 2])

    # 3D Euclidean error (approximation)
    euclidean_3d_errors = np.sqrt(horizontal_errors**2 + depth_errors**2)

    # Calculate success rates at different thresholds
    thresholds = [5, 10, 15, 20, 30]
    success_rates = {}
    for threshold in thresholds:
        success_rates[f"horizontal_{threshold}km"] = (
            np.mean(horizontal_errors < threshold) * 100
        )
        success_rates[f"depth_{threshold}km"] = np.mean(depth_errors < threshold) * 100
        success_rates[f"3d_{threshold}km"] = (
            np.mean(euclidean_3d_errors < threshold) * 100
        )

    # Combine all metrics
    metrics = {
        "mean_horizontal_error": np.mean(horizontal_errors),
        "median_horizontal_error": np.median(horizontal_errors),
        "mean_depth_error": np.mean(depth_errors),
        "median_depth_error": np.median(depth_errors),
        "mean_3d_error": np.mean(euclidean_3d_errors),
        "median_3d_error": np.median(euclidean_3d_errors),
        **success_rates,
    }

    return metrics, horizontal_errors, depth_errors, euclidean_3d_errors


def plot_results(
    df,
    y_true,
    y_pred,
    errors,
    feature_importances=None,
    feature_names=None,
    model_name="",
):
    """
    Create visualizations of prediction results.
    """
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Map of true vs predicted locations
    plt.figure(figsize=(12, 10))

    # Plot all events in gray first
    plt.scatter(
        df["source_longitude_deg"],
        df["source_latitude_deg"],
        c="lightgray",
        s=10,
        alpha=0.3,
        label="All events",
    )

    # Plot true test events
    plt.scatter(
        y_true[:, 1], y_true[:, 0], c="blue", s=30, alpha=0.6, label="True (test)"
    )

    # Plot predicted events
    plt.scatter(y_pred[:, 1], y_pred[:, 0], c="red", s=30, alpha=0.6, label="Predicted")

    # Draw lines connecting true and predicted points
    for i in range(len(y_true)):
        plt.plot(
            [y_true[i, 1], y_pred[i, 1]], [y_true[i, 0], y_pred[i, 0]], "k-", alpha=0.15
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"True vs Predicted Aftershock Locations - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"results/location_map_{model_name}_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Plot 2: Error distribution
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.hist(errors[1], bins=30, color="skyblue", edgecolor="black")
    plt.axvline(
        np.mean(errors[1]),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(errors[1]):.2f} km",
    )
    plt.axvline(
        np.median(errors[1]),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(errors[1]):.2f} km",
    )
    plt.xlabel("Horizontal Error (km)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.hist(errors[2], bins=30, color="lightgreen", edgecolor="black")
    plt.axvline(
        np.mean(errors[2]),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(errors[2]):.2f} km",
    )
    plt.axvline(
        np.median(errors[2]),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(errors[2]):.2f} km",
    )
    plt.xlabel("Depth Error (km)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.hist(errors[3], bins=30, color="salmon", edgecolor="black")
    plt.axvline(
        np.mean(errors[3]),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(errors[3]):.2f} km",
    )
    plt.axvline(
        np.median(errors[3]),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(errors[3]):.2f} km",
    )
    plt.xlabel("3D Error (km)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    # Success rate at different thresholds
    thresholds = [5, 10, 15, 20, 30]
    success_rates = [errors[0][f"horizontal_{t}km"] for t in thresholds]
    plt.bar(range(len(thresholds)), success_rates, color="lightblue")
    plt.xticks(range(len(thresholds)), [f"{t} km" for t in thresholds])
    plt.xlabel("Distance Threshold")
    plt.ylabel("Success Rate (%)")
    plt.title("Percentage of Predictions Within Distance Threshold")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(
        f"results/error_distribution_{model_name}_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Plot 3: Feature importances if available
    if feature_importances is not None and feature_names is not None:
        # Get top 30 features by importance (or fewer if there are less features)
        n_features = min(30, len(feature_names))
        top_indices = np.argsort(feature_importances)[-n_features:]

        plt.figure(figsize=(12, 14))
        plt.barh(
            range(len(top_indices)), feature_importances[top_indices], align="center"
        )
        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
        plt.xlabel("Feature Importance")
        plt.title(f"{model_name} - Top Feature Importances")
        plt.tight_layout()
        plt.savefig(
            f"results/feature_importances_{model_name}_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )

    # Spatial error distribution plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        y_true[:, 1],
        y_true[:, 0],
        c=errors[1],
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="k",
    )
    plt.colorbar(scatter, label="Horizontal Error (km)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Spatial Distribution of Prediction Errors - {model_name}")
    plt.grid(True)
    plt.savefig(
        f"results/spatial_error_distribution_{model_name}_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Save metrics to a text file
    with open(f"results/metrics_{model_name}_{timestamp}.txt", "w") as f:
        f.write(f"Prediction Error Metrics for {model_name}:\n")
        for key, value in errors[0].items():
            f.write(f"{key}: {value:.4f}\n")


def train_evaluate_models(X, y, feature_names, test_mask, df):
    """
    Train and evaluate multiple models for comparison.
    """
    # Split by train/test mask
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Models to evaluate
    base_models = {
        "RandomForest": {
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor,
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": 42,
            },
        },
        "KNeighbors": {
            "model": KNeighborsRegressor,
            "params": {"n_neighbors": 5, "weights": "distance"},
        },
    }

    # Add a spatially aware random forest if we have neighbor features
    if "neighbor_mean_lat" in feature_names:
        base_models["SpatialRandomForest"] = {
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
        }

    results = {}

    for name, model_info in base_models.items():
        print(f"\nTraining {name} model...")

        # For models that can't handle multi-output directly, use separate models for each coordinate
        if name in ["GradientBoosting", "KNeighbors"]:
            # Create and train separate models for lat, lon, and depth
            lat_model = model_info["model"](**model_info["params"])
            lon_model = model_info["model"](**model_info["params"])
            depth_model = model_info["model"](**model_info["params"])

            lat_model.fit(X_train_scaled, y_train[:, 0])
            lon_model.fit(X_train_scaled, y_train[:, 1])
            depth_model.fit(X_train_scaled, y_train[:, 2])

            # Predict each coordinate separately
            lat_pred = lat_model.predict(X_test_scaled)
            lon_pred = lon_model.predict(X_test_scaled)
            depth_pred = depth_model.predict(X_test_scaled)

            # Combine predictions
            y_pred = np.column_stack((lat_pred, lon_pred, depth_pred))

            # For feature importance, use the average of the three models if available
            if hasattr(lat_model, "feature_importances_"):
                feature_importances = (
                    lat_model.feature_importances_
                    + lon_model.feature_importances_
                    + depth_model.feature_importances_
                ) / 3
            else:
                feature_importances = None
        else:
            # Models that handle multi-output directly
            model = model_info["model"](**model_info["params"])
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Get feature importances if available
            if hasattr(model, "feature_importances_"):
                feature_importances = model.feature_importances_
            else:
                feature_importances = None

        # Calculate errors
        metrics, horizontal_errors, depth_errors, euclidean_3d_errors = (
            calculate_prediction_errors(y_test, y_pred)
        )

        # Print metrics
        print(f"\n{name} Prediction Error Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Plot results
        plot_results(
            df,
            y_test,
            y_pred,
            (metrics, horizontal_errors, depth_errors, euclidean_3d_errors),
            feature_importances,
            feature_names,
            name,
        )

        # Store results
        results[name] = {
            "metrics": metrics,
            "errors": (horizontal_errors, depth_errors, euclidean_3d_errors),
            "feature_importances": feature_importances,
        }

    return results


def experiment_with_train_size(X, y, feature_names, test_mask, df):
    """
    Experiment with different training set sizes to understand learning curve.
    """
    X_train_full, X_test = X[~test_mask], X[test_mask]
    y_train_full, y_test = y[~test_mask], y[test_mask]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled_full = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    # Different training set sizes to try (percentages)
    train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

    results = {}

    for size in train_sizes:
        if size < 1.0:
            # Subsample the training set
            indices = np.random.choice(
                len(X_train_scaled_full),
                size=int(size * len(X_train_scaled_full)),
                replace=False,
            )
            X_train_scaled = X_train_scaled_full[indices]
            y_train = y_train_full[indices]
        else:
            X_train_scaled = X_train_scaled_full
            y_train = y_train_full

        print(
            f"\nTraining with {int(size * 100)}% of training data ({len(X_train_scaled)} samples)..."
        )

        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )

        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Calculate errors
        metrics, horizontal_errors, depth_errors, euclidean_3d_errors = (
            calculate_prediction_errors(y_test, y_pred)
        )

        # Print metrics
        print(f"\nTraining Size {int(size * 100)}% - Prediction Error Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Store results
        results[f"train_size_{int(size * 100)}"] = {
            "metrics": metrics,
            "errors": (horizontal_errors, depth_errors, euclidean_3d_errors),
        }

    # Plot learning curve
    train_size_percents = [int(size * 100) for size in train_sizes]
    mean_errors = [
        results[f"train_size_{size}"]["metrics"]["mean_horizontal_error"]
        for size in train_size_percents
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(train_size_percents, mean_errors, "o-", linewidth=2)
    plt.xlabel("Training Set Size (%)")
    plt.ylabel("Mean Horizontal Error (km)")
    plt.title("Learning Curve: Effect of Training Set Size on Prediction Error")
    plt.grid(True)
    plt.savefig(
        f'results/learning_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
        dpi=300,
        bbox_inches="tight",
    )

    return results


def main():
    """
    Main function to run the improved aftershock prediction pipeline.
    """
    print("Starting Improved Aftershock Location Prediction...")

    # Check if data exists, otherwise notify
    if not os.path.exists("aftershock_data.pkl"):
        print(
            "Error: Data file 'aftershock_data.pkl' not found. Please run data loading first."
        )
        return

    # Read data
    print("Reading data from pickle file...")
    df = read_data_from_pickle("aftershock_data.pkl")

    # Display data info
    print(f"Dataset loaded. Total events: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Train/Test split: {df['split'].value_counts().to_dict()}")

    # Create test mask - combine 'test' and 'dev' into test set
    test_mask = (df["split"] == "test") | (df["split"] == "dev")

    # Extract all feature sets
    (
        all_features,
        all_feature_names,
        pca_features,
        pca_feature_names,
        pca_neighbor_features,
        pca_neighbor_feature_names,
    ) = extract_combined_features(df)

    # Target variables
    y = df[["source_latitude_deg", "source_longitude_deg", "source_depth_km"]].values

    # Train and evaluate with all features
    print("\n=== Training models with all features ===")
    all_results = train_evaluate_models(
        all_features, y, all_feature_names, test_mask, df
    )

    # Train and evaluate with PCA features
    print("\n=== Training models with PCA-reduced features ===")
    pca_results = train_evaluate_models(
        pca_features, y, pca_feature_names, test_mask, df
    )

    # Train and evaluate with PCA + neighbor features
    print("\n=== Training models with PCA + neighbor features ===")
    pca_neighbor_results = train_evaluate_models(
        pca_neighbor_features, y, pca_neighbor_feature_names, test_mask, df
    )

    # Experiment with training set size (learning curve)
    print("\n=== Experimenting with training set sizes ===")
    learning_curve_results = experiment_with_train_size(
        pca_neighbor_features, y, pca_neighbor_feature_names, test_mask, df
    )

    print("Prediction experiments complete. Results saved in the 'results' directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
