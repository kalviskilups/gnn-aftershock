import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance


def read_data_from_pickle(file_path):
    """
    Load the data from pickle file.
    """
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


def extract_enhanced_waveform_features(
    waveform, p_arrival_sample=None, s_arrival_sample=None, sampling_rate=100
):
    """
    Extract comprehensive features from a seismic waveform including P and S wave specific features.

    Args:
        waveform: numpy array with shape (components, samples)
        p_arrival_sample: Sample index for P-wave arrival
        s_arrival_sample: Sample index for S-wave arrival
        sampling_rate: Sampling rate in Hz

    Returns:
        numpy array of extracted features
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

        # Add zero crossing rate (indicates frequency characteristics)
        zero_crossings = np.where(np.diff(np.signbit(waveform[component])))[0]
        features.append(len(zero_crossings) / len(waveform[component]))

        # Waveform complexity features
        features.append(
            np.mean(np.abs(np.diff(waveform[component])))
        )  # Average absolute derivative
        features.append(np.max(np.abs(np.diff(waveform[component]))))  # Max derivative

        # Calculate entropy (measure of randomness in the signal)
        hist, _ = np.histogram(waveform[component], bins=20, density=True)
        hist = hist[hist > 0]  # Remove zeros to avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))
        features.append(entropy)

        # Frequency domain features
        fft = np.abs(np.fft.rfft(waveform[component]))
        features.append(np.argmax(fft))  # Dominant frequency index
        features.append(np.max(fft))  # Maximum frequency amplitude
        features.append(np.mean(fft))  # Mean frequency amplitude
        features.append(np.std(fft))  # Std of frequency amplitude
        features.append(np.sum(fft))  # Total spectral energy

        # Spectral ratios for different frequency bands
        # Divide the spectrum into low, mid, and high frequency bands
        freq_band_size = len(fft) // 3
        low_freq = np.sum(fft[:freq_band_size])
        mid_freq = np.sum(fft[freq_band_size : 2 * freq_band_size])
        high_freq = np.sum(fft[2 * freq_band_size :])

        # Calculate ratios (with safety for division by zero)
        if low_freq > 0:
            features.append(high_freq / low_freq)  # High/Low ratio
        else:
            features.append(0)

        if mid_freq > 0:
            features.append(high_freq / mid_freq)  # High/Mid ratio
            features.append(mid_freq / low_freq if low_freq > 0 else 0)  # Mid/Low ratio
        else:
            features.append(0)
            features.append(0)

        # P and S wave specific features if arrival times are available
        if p_arrival_sample is not None and s_arrival_sample is not None:
            # Ensure arrival indices are valid integers
            p_idx = int(p_arrival_sample)
            s_idx = int(s_arrival_sample)

            if 0 <= p_idx < len(waveform[component]) and 0 <= s_idx < len(
                waveform[component]
            ):
                # Extract P-wave window (small window around P arrival)
                p_window_start = max(0, p_idx - 20)
                p_window_end = min(len(waveform[component]), p_idx + 50)
                p_wave = waveform[component][p_window_start:p_window_end]

                # Extract S-wave window
                s_window_start = max(0, s_idx - 20)
                s_window_end = min(len(waveform[component]), s_idx + 80)
                s_wave = waveform[component][s_window_start:s_window_end]

                # Calculate P-wave features
                p_amp = np.max(np.abs(p_wave)) if len(p_wave) > 0 else 0
                p_energy = np.sum(p_wave**2) if len(p_wave) > 0 else 0

                # Calculate S-wave features
                s_amp = np.max(np.abs(s_wave)) if len(s_wave) > 0 else 0
                s_energy = np.sum(s_wave**2) if len(s_wave) > 0 else 0

                # Calculate P/S ratios (important for source characterization)
                ps_amp_ratio = p_amp / (s_amp + 1e-10)  # Avoid division by zero
                ps_energy_ratio = p_energy / (s_energy + 1e-10)

                # Add features
                features.extend(
                    [p_amp, p_energy, s_amp, s_energy, ps_amp_ratio, ps_energy_ratio]
                )

                # Calculate S-P time difference (related to distance)
                sp_time_diff = (s_idx - p_idx) / sampling_rate
                features.append(sp_time_diff)

                # Frequency content of P and S waves separately
                if len(p_wave) > 1:
                    p_fft = np.abs(np.fft.rfft(p_wave))
                    p_dom_freq = np.argmax(p_fft) if len(p_fft) > 0 else 0

                    # P-wave spectral ratios
                    if len(p_fft) >= 4:  # Ensure we have enough points to divide
                        p_high_freq = np.sum(p_fft[len(p_fft) // 2 :])
                        p_low_freq = np.sum(p_fft[: len(p_fft) // 2])
                        p_spectral_ratio = p_high_freq / (p_low_freq + 1e-10)
                    else:
                        p_spectral_ratio = 0
                else:
                    p_dom_freq = 0
                    p_spectral_ratio = 0

                if len(s_wave) > 1:
                    s_fft = np.abs(np.fft.rfft(s_wave))
                    s_dom_freq = np.argmax(s_fft) if len(s_fft) > 0 else 0

                    # S-wave spectral ratios
                    if len(s_fft) >= 4:  # Ensure we have enough points to divide
                        s_high_freq = np.sum(s_fft[len(s_fft) // 2 :])
                        s_low_freq = np.sum(s_fft[: len(s_fft) // 2])
                        s_spectral_ratio = s_high_freq / (s_low_freq + 1e-10)
                    else:
                        s_spectral_ratio = 0
                else:
                    s_dom_freq = 0
                    s_spectral_ratio = 0

                features.extend(
                    [p_dom_freq, s_dom_freq, p_spectral_ratio, s_spectral_ratio]
                )
            else:
                # If P or S indices are out of bounds, add placeholder values
                features.extend([0] * 12)  # 12 features for P/S waves
        else:
            # If P or S arrivals are missing, add placeholder values
            features.extend([0] * 12)  # 12 features for P/S waves

    return np.array(features)


def analyze_features(df):
    """
    Extract enhanced waveform features and analyze feature importance using RandomForest.
    """
    print("Extracting enhanced waveform features...")

    # Extract enhanced features from all waveforms
    features_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Get P and S arrival samples if available
        p_arrival = row.get("trace_P_arrival_sample", None)
        s_arrival = row.get("trace_S_arrival_sample", None)
        sampling_rate = row.get(
            "trace_sampling_rate_hz", 100
        )  # Default to 100 Hz if not available

        # Extract features
        features = extract_enhanced_waveform_features(
            row["waveform"],
            p_arrival_sample=p_arrival,
            s_arrival_sample=s_arrival,
            sampling_rate=sampling_rate,
        )
        features_list.append(features)

    # Convert to numpy array
    features_array = np.array(features_list)

    # Create feature names
    component_names = ["Z", "N", "E"]  # Assuming ZNE component order
    base_features = [
        "peak_amp",
        "mean_amp",
        "std",
        "percentile_75",
        "percentile_25",
        "energy",
        "rms",
        "zero_crossing_rate",
        "avg_abs_deriv",
        "max_deriv",
        "entropy",
        "dom_freq_idx",
        "max_freq_amp",
        "mean_freq_amp",
        "std_freq_amp",
        "total_spectral_energy",
        "high_low_ratio",
        "high_mid_ratio",
        "mid_low_ratio",
    ]
    ps_features = [
        "p_amp",
        "p_energy",
        "s_amp",
        "s_energy",
        "ps_amp_ratio",
        "ps_energy_ratio",
        "sp_time_diff",
        "p_dom_freq",
        "s_dom_freq",
        "p_spectral_ratio",
        "s_spectral_ratio",
    ]

    feature_names = []
    for comp in component_names:
        for feat in base_features:
            feature_names.append(f"{comp}_{feat}")
        for feat in ps_features:
            feature_names.append(f"{comp}_{feat}")

    # Add temporal and spatial features
    features_array = np.column_stack(
        (
            features_array,
            df["source_latitude_deg"].values,
            df["source_longitude_deg"].values,
            df["source_depth_km"].values,
        )
    )
    feature_names.extend(["latitude", "longitude", "depth"])

    # Create target values
    y = df[["source_latitude_deg", "source_longitude_deg", "source_depth_km"]].values

    # Split by train/test
    train_mask = df["split"] == "train"
    X_train, X_test = features_array[train_mask], features_array[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a RandomForest model for each coordinate
    lat_model = RandomForestRegressor(n_estimators=200, random_state=42)
    lon_model = RandomForestRegressor(n_estimators=200, random_state=42)
    depth_model = RandomForestRegressor(n_estimators=200, random_state=42)

    print("Training models to determine feature importance...")
    lat_model.fit(X_train_scaled, y_train[:, 0])
    lon_model.fit(X_train_scaled, y_train[:, 1])
    depth_model.fit(X_train_scaled, y_train[:, 2])

    # Calculate built-in feature importance
    lat_importance = lat_model.feature_importances_
    lon_importance = lon_model.feature_importances_
    depth_importance = depth_model.feature_importances_

    # Combine importances
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Latitude_Importance": lat_importance,
            "Longitude_Importance": lon_importance,
            "Depth_Importance": depth_importance,
            "Overall_Importance": (lat_importance + lon_importance + depth_importance)
            / 3,
        }
    )

    # Sort by overall importance
    importance_df = importance_df.sort_values("Overall_Importance", ascending=False)

    print("\nTop 20 Most Important Features:")
    print(importance_df.head(20))

    # Calculate permutation importance (more reliable than built-in feature importance)
    print("\nCalculating permutation importance (this may take a while)...")

    # Create a directory for results
    os.makedirs("feature_analysis", exist_ok=True)

    # Function to calculate permutation importance for each target
    def calc_perm_importance(model, X, y, feature_names):
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )

        perm_importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance_Mean": result.importances_mean,
                "Importance_Std": result.importances_std,
            }
        )
        return perm_importance_df.sort_values("Importance_Mean", ascending=False)

    # Calculate for each coordinate
    lat_perm_importance = calc_perm_importance(
        lat_model, X_test_scaled, y_test[:, 0], feature_names
    )
    lon_perm_importance = calc_perm_importance(
        lon_model, X_test_scaled, y_test[:, 1], feature_names
    )
    depth_perm_importance = calc_perm_importance(
        depth_model, X_test_scaled, y_test[:, 2], feature_names
    )

    # Save to CSV
    lat_perm_importance.to_csv(
        "feature_analysis/latitude_permutation_importance.csv", index=False
    )
    lon_perm_importance.to_csv(
        "feature_analysis/longitude_permutation_importance.csv", index=False
    )
    depth_perm_importance.to_csv(
        "feature_analysis/depth_permutation_importance.csv", index=False
    )

    # Plot feature importance for top features
    plt.figure(figsize=(12, 18))

    # Plot latitude importance
    plt.subplot(3, 1, 1)
    top_lat_features = lat_perm_importance.head(15)
    sns.barplot(
        x="Importance_Mean", y="Feature", data=top_lat_features, palette="viridis"
    )
    plt.title("Top 15 Features for Latitude Prediction")
    plt.tight_layout()

    # Plot longitude importance
    plt.subplot(3, 1, 2)
    top_lon_features = lon_perm_importance.head(15)
    sns.barplot(
        x="Importance_Mean", y="Feature", data=top_lon_features, palette="viridis"
    )
    plt.title("Top 15 Features for Longitude Prediction")
    plt.tight_layout()

    # Plot depth importance
    plt.subplot(3, 1, 3)
    top_depth_features = depth_perm_importance.head(15)
    sns.barplot(
        x="Importance_Mean", y="Feature", data=top_depth_features, palette="viridis"
    )
    plt.title("Top 15 Features for Depth Prediction")
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        "feature_analysis/feature_importance_plots.png", dpi=300, bbox_inches="tight"
    )

    # Create a combined importance plot showing feature groups
    # Group features by type
    feature_groups = {
        "Time Domain": [
            f
            for f in feature_names
            if any(x in f for x in ["amp", "std", "percentile", "rms", "energy"])
        ],
        "Frequency Domain": [
            f for f in feature_names if any(x in f for x in ["freq", "spectral"])
        ],
        "P-S Wave": [
            f
            for f in feature_names
            if any(x in f for x in ["p_", "s_", "ps_", "sp_time"])
        ],
        "Signal Complexity": [
            f
            for f in feature_names
            if any(x in f for x in ["deriv", "zero_crossing", "entropy"])
        ],
        "Spatial": ["latitude", "longitude", "depth"],
    }

    # Calculate average importance by group
    group_importance = {}
    for group, feats in feature_groups.items():
        lat_imp = lat_perm_importance[lat_perm_importance["Feature"].isin(feats)][
            "Importance_Mean"
        ].mean()
        lon_imp = lon_perm_importance[lon_perm_importance["Feature"].isin(feats)][
            "Importance_Mean"
        ].mean()
        depth_imp = depth_perm_importance[depth_perm_importance["Feature"].isin(feats)][
            "Importance_Mean"
        ].mean()
        group_importance[group] = {
            "Latitude": lat_imp,
            "Longitude": lon_imp,
            "Depth": depth_imp,
            "Overall": (lat_imp + lon_imp + depth_imp) / 3,
        }

    # Create dataframe
    group_imp_df = pd.DataFrame(group_importance).T.reset_index()
    group_imp_df = group_imp_df.rename(columns={"index": "Feature_Group"})

    # Plot grouped importance
    plt.figure(figsize=(14, 10))

    # Convert to long format for seaborn
    group_imp_long = pd.melt(
        group_imp_df,
        id_vars=["Feature_Group"],
        value_vars=["Latitude", "Longitude", "Depth", "Overall"],
        var_name="Target",
        value_name="Importance",
    )

    # Plot
    sns.barplot(
        x="Feature_Group",
        y="Importance",
        hue="Target",
        data=group_imp_long,
        palette="viridis",
    )
    plt.title("Feature Group Importance by Target")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        "feature_analysis/feature_group_importance.png", dpi=300, bbox_inches="tight"
    )

    # Return importance dataframes
    return (
        importance_df,
        lat_perm_importance,
        lon_perm_importance,
        depth_perm_importance,
    )


def main():
    """
    Main function to run feature analysis.
    """
    print("Starting Enhanced Waveform Feature Analysis...")

    # Check if data exists
    if not os.path.exists("aftershock_data.pkl"):
        print("Error: Data file 'aftershock_data.pkl' not found.")
        return

    # Read data
    print("Reading data from pickle file...")
    df = read_data_from_pickle("aftershock_data.pkl")

    # Display data info
    print(f"Dataset loaded. Total events: {len(df)}")

    # Analyze features
    analyze_features(df)

    print(
        "Feature analysis complete. Results saved in the 'feature_analysis' directory."
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
