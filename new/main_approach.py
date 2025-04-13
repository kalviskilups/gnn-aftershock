import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import seisbench.data as sbd
from scipy import signal
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import logging
import os
import datetime
import pywt

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(
    log_dir,
    f"aftershock_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)

# Set style for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper")


def load_aftershock_data_with_waveforms(max_waveforms):
    """
    Load and preprocess aftershock data from the Iquique dataset,
    including waveform data
    """
    logging.info("Loading Iquique dataset using SeisBench...")
    iquique = sbd.Iquique()

    # Get metadata
    metadata = iquique.metadata.copy()
    logging.info(f"Initial metadata shape: {metadata.shape}")

    # Filter out rows with missing essential data
    metadata = metadata.dropna(
        subset=[
            "source_origin_time",
            "source_latitude_deg",
            "source_longitude_deg",
            "source_depth_km",
        ]
    )
    logging.info(f"Metadata after filtering missing values: {metadata.shape}")

    # Convert timestamps
    metadata["datetime"] = pd.to_datetime(metadata["source_origin_time"])

    # Sort by time
    metadata = metadata.sort_values("datetime")

    # Create a dictionary to store waveform features
    waveform_features_dict = {}

    # Initialize feature extractor with a focused set of features
    feature_extractor = WaveformFeatureExtractor(focused=False)

    # Limit the number of waveforms to process
    sample_indices = metadata.index[: min(max_waveforms, len(metadata))]
    logging.info(
        f"Processing {len(sample_indices)} waveforms out of {len(metadata)} total"
    )

    logging.info("Extracting waveform features...")
    success_count = 0
    error_count = 0
    feature_stats = {}

    for idx in tqdm(sample_indices):
        try:
            # Get waveform data
            waveform = iquique.get_waveforms(int(idx))

            # Get P and S arrival samples if available
            p_arrival = metadata.loc[idx, "trace_P_arrival_sample"]
            s_arrival = metadata.loc[idx, "trace_S_arrival_sample"]

            # Validate P and S arrivals
            if pd.isna(p_arrival) or pd.isna(s_arrival):
                p_arrival, s_arrival = None, None
            else:
                p_arrival = int(p_arrival)
                s_arrival = int(s_arrival)

            # Extract features
            features = feature_extractor.extract_features(
                waveform, p_arrival, s_arrival
            )

            # Log feature statistics
            for feature_name, value in features.items():
                if feature_name not in feature_stats:
                    feature_stats[feature_name] = []
                feature_stats[feature_name].append(value)

            # Store features
            waveform_features_dict[idx] = features
            success_count += 1

        except Exception as e:
            logging.warning(f"Error processing waveform {idx}: {e}")
            waveform_features_dict[idx] = {}
            error_count += 1

    # Log feature statistics
    logging.info(f"Feature extraction summary:")
    logging.info(f"  Successful extractions: {success_count}")
    logging.info(f"  Failed extractions: {error_count}")

    # Log stats for key features
    for feature_name, values in feature_stats.items():
        if feature_name in [
            "Z_energy",
            "N_energy",
            "E_energy",
            "Z_dominant_freq",
            "Z_PS_ratio",
        ]:
            values_array = np.array(values)
            logging.info(
                f"  {feature_name}: min={np.min(values_array):.4f}, max={np.max(values_array):.4f}, mean={np.mean(values_array):.4f}, std={np.std(values_array):.4f}"
            )

    return metadata, iquique, waveform_features_dict


def identify_mainshock_and_aftershocks(metadata):
    """
    Identify the mainshock and its associated aftershocks based on data patterns
    """
    # The known Iquique earthquake was in early April 2014
    april_start = pd.Timestamp("2014-04-01", tz="UTC")
    april_end = pd.Timestamp("2014-04-05", tz="UTC")

    # Find events in early April 2014
    april_events = metadata[
        (metadata["datetime"] >= april_start) & (metadata["datetime"] <= april_end)
    ]

    if len(april_events) == 0:
        logging.info(
            "No events found in early April 2014 timeframe. Using alternative approach."
        )
        # Alternative approach: find the earliest events in the dataset
        metadata_sorted = metadata.sort_values("datetime")
        earliest_date = metadata_sorted["datetime"].iloc[0]
        logging.info(f"Earliest event in dataset: {earliest_date}")

        # Take the earliest events as potential mainshock candidates
        potential_mainshocks = metadata_sorted.iloc[:100]  # First 100 events

        # Find which event has most events in the following 24 hours
        best_event_idx = None
        most_followers = 0

        for idx, row in potential_mainshocks.iterrows():
            event_time = row["datetime"]
            next_24h = event_time + pd.Timedelta(hours=24)
            followers = sum(metadata["datetime"].between(event_time, next_24h))

            if followers > most_followers:
                most_followers = followers
                best_event_idx = idx

        mainshock = metadata.loc[best_event_idx]
    else:
        logging.info(f"Found {len(april_events)} events in early April 2014.")
        # Within this timeframe, find the event with the most followers in 24 hours
        best_event_idx = None
        most_followers = 0

        for idx, row in april_events.iterrows():
            event_time = row["datetime"]
            next_24h = event_time + pd.Timedelta(hours=24)
            followers = sum(metadata["datetime"].between(event_time, next_24h))

            if followers > most_followers:
                most_followers = followers
                best_event_idx = idx

        mainshock = metadata.loc[best_event_idx]

    logging.info(
        f"Identified potential mainshock at {mainshock['datetime']} at location "
        f"({mainshock['source_latitude_deg']}, {mainshock['source_longitude_deg']}), "
        f"depth {mainshock['source_depth_km']} km"
    )
    logging.info(
        f"This event is followed by {most_followers} events in the next 24 hours"
    )

    # Select events after the mainshock as aftershocks
    aftershocks = metadata[metadata["datetime"] > mainshock["datetime"]].copy()
    logging.info(f"Found {len(aftershocks)} aftershocks")

    # Create time since mainshock in hours
    aftershocks["hours_since_mainshock"] = (
        aftershocks["datetime"] - mainshock["datetime"]
    ).dt.total_seconds() / 3600

    # Calculate distance from mainshock
    aftershocks["distance_from_mainshock_km"] = aftershocks.apply(
        lambda row: haversine_distance(
            mainshock["source_latitude_deg"],
            mainshock["source_longitude_deg"],
            row["source_latitude_deg"],
            row["source_longitude_deg"],
        ),
        axis=1,
    )

    # Log some statistics on the aftershocks
    logging.info("Aftershock statistics:")

    # Time statistics
    hours = aftershocks["hours_since_mainshock"]
    logging.info(
        f"  Time since mainshock (hours): min={hours.min():.2f}, max={hours.max():.2f}, mean={hours.mean():.2f}"
    )

    # Distance statistics
    distances = aftershocks["distance_from_mainshock_km"]
    logging.info(
        f"  Distance from mainshock (km): min={distances.min():.2f}, max={distances.max():.2f}, mean={distances.mean():.2f}"
    )

    # Depth statistics
    depths = aftershocks["source_depth_km"]
    logging.info(
        f"  Depth (km): min={depths.min():.2f}, max={depths.max():.2f}, mean={depths.mean():.2f}"
    )

    # Count aftershocks by day
    days = (hours / 24).astype(int)
    day_counts = days.value_counts().sort_index()
    logging.info("  Aftershock counts by day:")
    for day, count in day_counts.items():
        if day < 10:  # Limit to first 10 days
            logging.info(f"    Day {day}: {count} events")

    return mainshock, aftershocks


def consolidate_station_recordings(metadata, waveform_features_dict):
    """
    Consolidate multiple station recordings of the same event into a single representation
    """
    # Create event IDs based on source parameters
    metadata["lat_rounded"] = np.round(metadata["source_latitude_deg"], 4)
    metadata["lon_rounded"] = np.round(metadata["source_longitude_deg"], 4)
    metadata["depth_rounded"] = np.round(metadata["source_depth_km"], 1)

    metadata["event_id"] = metadata.groupby(
        ["source_origin_time", "lat_rounded", "lon_rounded", "depth_rounded"]
    ).ngroup()

    logging.info(
        f"Original recordings: {len(metadata)}, Unique events: {metadata['event_id'].nunique()}"
    )

    # Create consolidated representations
    consolidated_metadata = []
    consolidated_features = {}

    # Track how many events actually have features
    events_with_features = 0
    stations_per_event = []

    # Count recordings per unique event
    recordings_per_event = metadata.groupby("event_id").size()

    # Log distribution of recordings per event
    logging.info("Distribution of recordings per event:")
    for count, num_events in recordings_per_event.value_counts().sort_index().items():
        logging.info(f"  {count} recordings: {num_events} events")

    logging.info(f"Max recordings per event: {recordings_per_event.max()}")
    logging.info(f"Median recordings per event: {recordings_per_event.median()}")

    for event_id, group in metadata.groupby("event_id"):
        # Find recordings with valid waveform features
        valid_recordings = []
        for idx in group.index:
            if idx in waveform_features_dict and waveform_features_dict[idx]:
                valid_recordings.append(idx)

        # If we have recordings with features, use one of them
        if valid_recordings:
            best_idx = valid_recordings[0]  # Just take the first valid one
            events_with_features += 1
        else:
            # If no recording has features, just take the first index
            best_idx = group.index[0]

        # Track stations per event
        stations_per_event.append(len(group))

        # Take metadata from best recording
        best_record = group.loc[best_idx].copy()
        best_record["station_count"] = len(group)
        best_record["event_id"] = event_id  # Keep event_id in the metadata
        consolidated_metadata.append(best_record)

        # Map waveform features to the event_id
        if best_idx in waveform_features_dict and waveform_features_dict[best_idx]:
            consolidated_features[event_id] = waveform_features_dict[best_idx]

    logging.info(f"Events with valid waveform features: {events_with_features}")
    logging.info(f"Average stations per event: {np.mean(stations_per_event):.2f}")

    # Convert to DataFrame
    consolidated_metadata = pd.DataFrame(consolidated_metadata)

    # Log some statistics about the consolidated metadata
    logging.info("Consolidated metadata statistics:")
    logging.info(f"  Unique events: {len(consolidated_metadata)}")
    logging.info(f"  Events with features: {len(consolidated_features)}")
    logging.info(
        f"  Feature coverage: {len(consolidated_features)/len(consolidated_metadata)*100:.2f}%"
    )

    return consolidated_metadata, consolidated_features


class WaveformFeatureExtractor:
    """Class to extract a focused set of features from seismic waveforms"""

    def __init__(self, sampling_rate=100.0, focused=False):
        self.sampling_rate = sampling_rate
        self.focused = focused  # Use a smaller feature set if True

    def extract_features(self, waveform, p_arrival=None, s_arrival=None):
        """Extract comprehensive features from waveform data"""
        # Initialize feature dictionary
        features = {}

        # Basic time-domain features (overall signal)
        for i, component_name in enumerate(["Z", "N", "E"]):
            if i < waveform.shape[0]:  # Check if component exists
                component = waveform[i]

                # Calculate basic statistics
                features[f"{component_name}_max"] = np.max(np.abs(component))
                features[f"{component_name}_rms"] = np.sqrt(np.mean(component**2))
                features[f"{component_name}_energy"] = np.sum(component**2)

                # Only include extended features if not in focused mode
                if not self.focused:
                    features[f"{component_name}_mean"] = np.mean(np.abs(component))
                    features[f"{component_name}_std"] = np.std(component)
                    features[f"{component_name}_kurtosis"] = self._kurtosis(component)

                # Calculate standard frequency-domain features
                freq_features = self._compute_frequency_features(component)
                for feat_name, feat_value in freq_features.items():
                    features[f"{component_name}_{feat_name}"] = feat_value

                # New advanced features - only compute when not in focused mode
                # or if you want them even in focused mode
                if not self.focused:
                    # Add wavelet features if signal is long enough
                    if len(component) >= 64:
                        wavelet_features = self._compute_wavelet_features(component)
                        for feat_name, feat_value in wavelet_features.items():
                            features[f"{component_name}_{feat_name}"] = feat_value

                    # Add spectrogram features if signal is long enough
                    if len(component) >= 128:
                        spec_features = self._compute_spectrogram_features(component)
                        for feat_name, feat_value in spec_features.items():
                            features[f"{component_name}_{feat_name}"] = feat_value

        # Extract P-wave and S-wave specific features if arrivals are provided
        if p_arrival is not None and s_arrival is not None:
            # Original P/S wave features
            p_window_size = 100  # 100 samples after P arrival
            s_window_size = 100  # 100 samples after S arrival

            # Ensure window sizes don't exceed waveform length
            p_window_size = min(p_window_size, waveform.shape[1] - p_arrival)
            s_window_size = min(s_window_size, waveform.shape[1] - s_arrival)

            if p_window_size > 0 and s_window_size > 0:
                # Basic P/S wave features (keep the original code)
                for i, component_name in enumerate(["Z", "N", "E"]):
                    if i < waveform.shape[0]:  # Check if component exists
                        p_segment = waveform[i, p_arrival : p_arrival + p_window_size]
                        s_segment = waveform[i, s_arrival : s_arrival + s_window_size]

                        # Key P-wave features
                        features[f"P_{component_name}_energy"] = np.sum(p_segment**2)

                        # Key S-wave features
                        features[f"S_{component_name}_energy"] = np.sum(s_segment**2)

                        # Calculate P/S ratio (important for location)
                        p_max = np.max(np.abs(p_segment))
                        s_max = np.max(np.abs(s_segment))
                        if s_max > 0:
                            features[f"{component_name}_PS_ratio"] = p_max / s_max

            # Add enhanced phase features if not in focused mode
            if not self.focused:
                # Add enhanced P/S wave features
                phase_features = self._extract_phase_features(
                    waveform, p_arrival, s_arrival
                )
                features.update(phase_features)

                # Add polarization features if we have 3-component data
                if waveform.shape[0] >= 3:
                    pol_features = self._compute_polarization_features(
                        waveform, p_arrival, s_arrival
                    )
                    features.update(pol_features)

        return features

    def _compute_frequency_features(self, signal_segment):
        """Compute focused frequency domain features"""
        features = {}

        # Check if signal segment is not empty
        if len(signal_segment) < 10:
            return {
                "dominant_freq": 0,
                "low_high_ratio": 0,
            }

        try:
            # Calculate power spectral density
            f, Pxx = signal.welch(
                signal_segment,
                fs=self.sampling_rate,
                nperseg=min(256, len(signal_segment)),
            )

            # Dominant frequency
            features["dominant_freq"] = f[np.argmax(Pxx)]

            # Frequency band energies
            low_idx = f < 5
            high_idx = f >= 15

            # Handle empty frequency bands
            low_energy = np.sum(Pxx[low_idx]) if np.any(low_idx) else 0
            high_energy = np.sum(Pxx[high_idx]) if np.any(high_idx) else 0

            # Calculate frequency ratio
            if high_energy > 0:
                features["low_high_ratio"] = low_energy / high_energy
            else:
                features["low_high_ratio"] = 0

        except Exception as e:
            print(f"Error computing frequency features: {e}")
            features = {
                "dominant_freq": 0,
                "low_high_ratio": 0,
            }

        return features

    def _kurtosis(self, x):
        """Calculate kurtosis of a signal"""
        n = len(x)
        if n < 4:
            return 0

        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            return 0

        kurt = np.mean(((x - mean) / std) ** 4) - 3
        return kurt

    def _compute_wavelet_features(self, signal_segment):
        """Extract features using wavelet decomposition"""
        # Handle short signals
        if len(signal_segment) < 64:
            return {"wavelet_energy_ratio_1_2": 0, "wavelet_band_0_energy": 0}

        try:
            # Use a standard wavelet family (e.g., 'db4' Daubechies wavelets)
            coeffs = pywt.wavedec(signal_segment, "db4", level=4)

            # Extract energy in different frequency bands from wavelet coefficients
            features = {}
            for i, coeff in enumerate(coeffs):
                band_energy = np.sum(coeff**2)
                features[f"wavelet_band_{i}_energy"] = band_energy

            # Calculate ratios between bands (can be informative for location)
            if len(coeffs) > 1:
                for i in range(len(coeffs) - 1):
                    band_ratio = np.sum(coeffs[i] ** 2) / (
                        np.sum(coeffs[i + 1] ** 2) + 1e-10
                    )
                    features[f"wavelet_band_ratio_{i}_{i+1}"] = band_ratio

            return features
        except Exception as e:
            print(f"Error computing wavelet features: {e}")
            return {"wavelet_energy_ratio_1_2": 0, "wavelet_band_0_energy": 0}

    def _compute_spectrogram_features(self, signal_segment):
        """Extract features from spectrogram of signal"""
        # Handle short signals
        if len(signal_segment) < 128:
            return {"spec_dom_freq_std": 0, "low_freq_decay_rate": 0}

        try:
            f, t, Sxx = signal.spectrogram(
                signal_segment,
                fs=self.sampling_rate,
                nperseg=min(128, len(signal_segment)),
            )

            features = {}

            # Time evolution of frequency content
            # For each time step, find dominant frequency
            if Sxx.shape[1] > 1:  # Ensure we have multiple time steps
                dom_freqs = f[np.argmax(Sxx, axis=0)]
                features["spec_dom_freq_std"] = np.std(
                    dom_freqs
                )  # Variability in dominant frequency
                features["spec_dom_freq_trend"] = (
                    np.polyfit(np.arange(len(dom_freqs)), dom_freqs, 1)[0]
                    if len(dom_freqs) > 2
                    else 0
                )  # Trend (slope)

            # Frequency band energies over time
            low_freq_idx = f < 5
            high_freq_idx = f > 15

            # Calculate how energy in different bands evolves over time
            if np.any(low_freq_idx) and Sxx.shape[1] > 1:
                low_energy_evolution = np.sum(Sxx[low_freq_idx, :], axis=0)
                features["low_freq_decay_rate"] = (
                    low_energy_evolution[0] - low_energy_evolution[-1]
                ) / (len(low_energy_evolution) + 1e-10)

            return features
        except Exception as e:
            print(f"Error computing spectrogram features: {e}")
            return {"spec_dom_freq_std": 0, "low_freq_decay_rate": 0}

    def _compute_polarization_features(self, waveform, p_arrival=None, s_arrival=None):
        """Calculate polarization features from three-component waveform data"""
        features = {}

        # Need all three components for polarization analysis
        if waveform.shape[0] < 3:
            return features

        # For P-wave window
        if p_arrival is not None:
            p_window_size = 100  # or adaptive as needed
            p_window_size = min(p_window_size, waveform.shape[1] - p_arrival)

            if p_window_size > 20:  # Ensure window is large enough
                try:
                    # Extract windows for all components
                    z_p = waveform[0, p_arrival : p_arrival + p_window_size]
                    n_p = waveform[1, p_arrival : p_arrival + p_window_size]
                    e_p = waveform[2, p_arrival : p_arrival + p_window_size]

                    # Covariance matrix of the three components
                    data_matrix = np.vstack((z_p, n_p, e_p)).T
                    cov_matrix = np.cov(data_matrix.T)

                    # Eigendecomposition to get polarization attributes
                    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

                    # Sort eigenvalues and eigenvectors in descending order
                    idx = eigvals.argsort()[::-1]
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]

                    # Linearity, planarity, and rectilinearity measures
                    if sum(eigvals) > 0:
                        features["P_linearity"] = 1 - (eigvals[1] + eigvals[2]) / (
                            2 * eigvals[0] + 1e-10
                        )
                        features["P_planarity"] = 1 - (2 * eigvals[2]) / (
                            eigvals[0] + eigvals[1] + 1e-10
                        )

                        # Direction of polarization (simplified)
                        # Convert first eigenvector to spherical coordinates
                        v = eigvecs[:, 0]
                        r = np.sqrt(np.sum(v**2))
                        if r > 0:
                            theta = np.arccos(v[0] / r)  # Angle from Z axis
                            phi = np.arctan2(v[2], v[1])  # Azimuth
                            features["P_pol_theta"] = theta
                            features["P_pol_phi"] = phi
                except Exception as e:
                    print(f"Error computing P-wave polarization: {e}")

        # Similar analysis for S-wave window if needed
        if s_arrival is not None:
            s_window_size = 100
            s_window_size = min(s_window_size, waveform.shape[1] - s_arrival)

            if s_window_size > 20:
                try:
                    # Similar calculation for S-waves
                    z_s = waveform[0, s_arrival : s_arrival + s_window_size]
                    n_s = waveform[1, s_arrival : s_arrival + s_window_size]
                    e_s = waveform[2, s_arrival : s_arrival + s_window_size]

                    data_matrix = np.vstack((z_s, n_s, e_s)).T
                    cov_matrix = np.cov(data_matrix.T)

                    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
                    idx = eigvals.argsort()[::-1]
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]

                    if sum(eigvals) > 0:
                        features["S_linearity"] = 1 - (eigvals[1] + eigvals[2]) / (
                            2 * eigvals[0] + 1e-10
                        )
                        features["S_planarity"] = 1 - (2 * eigvals[2]) / (
                            eigvals[0] + eigvals[1] + 1e-10
                        )
                except Exception as e:
                    print(f"Error computing S-wave polarization: {e}")

        return features

    def _extract_phase_features(self, waveform, p_arrival, s_arrival):
        """Extract more detailed features from P and S wave windows"""
        features = {}

        # Define adaptive window sizes based on expected separation of phases
        if p_arrival is not None and s_arrival is not None:
            s_p_time = s_arrival - p_arrival

            # P-wave window - use smaller of 100 samples or 80% of S-P time
            p_window_size = min(100, int(0.8 * s_p_time))

            # S-wave window - use window scaled by S-P time
            s_window_size = min(150, int(1.2 * s_p_time))

            # Ensure window sizes don't exceed waveform length
            p_window_size = min(p_window_size, waveform.shape[1] - p_arrival)
            s_window_size = min(s_window_size, waveform.shape[1] - s_arrival)

            if (
                p_window_size > 10 and s_window_size > 10
            ):  # Ensure windows are large enough
                for i, component_name in enumerate(["Z", "N", "E"]):
                    if i < waveform.shape[0]:
                        try:
                            # Extract appropriate windows
                            p_segment = waveform[
                                i, p_arrival : p_arrival + p_window_size
                            ]
                            s_segment = waveform[
                                i, s_arrival : s_arrival + s_window_size
                            ]

                            # Frequency content
                            f_p, Pxx_p = signal.welch(
                                p_segment,
                                fs=self.sampling_rate,
                                nperseg=min(64, len(p_segment)),
                            )
                            f_s, Pxx_s = signal.welch(
                                s_segment,
                                fs=self.sampling_rate,
                                nperseg=min(64, len(s_segment)),
                            )

                            # Dominant frequencies
                            if len(Pxx_p) > 0:
                                features[f"P_{component_name}_dom_freq"] = f_p[
                                    np.argmax(Pxx_p)
                                ]
                            if len(Pxx_s) > 0:
                                features[f"S_{component_name}_dom_freq"] = f_s[
                                    np.argmax(Pxx_s)
                                ]

                            # Spectral ratios
                            p_low_idx = f_p < 5 if len(f_p) > 0 else []
                            p_high_idx = f_p > 15 if len(f_p) > 0 else []
                            s_low_idx = f_s < 5 if len(f_s) > 0 else []
                            s_high_idx = f_s > 15 if len(f_s) > 0 else []

                            p_low_energy = (
                                np.sum(Pxx_p[p_low_idx])
                                if len(p_low_idx) > 0 and any(p_low_idx)
                                else 0
                            )
                            p_high_energy = (
                                np.sum(Pxx_p[p_high_idx])
                                if len(p_high_idx) > 0 and any(p_high_idx)
                                else 0
                            )
                            s_low_energy = (
                                np.sum(Pxx_s[s_low_idx])
                                if len(s_low_idx) > 0 and any(s_low_idx)
                                else 0
                            )
                            s_high_energy = (
                                np.sum(Pxx_s[s_high_idx])
                                if len(s_high_idx) > 0 and any(s_high_idx)
                                else 0
                            )

                            if p_high_energy > 0:
                                features[f"P_{component_name}_LH_ratio"] = (
                                    p_low_energy / p_high_energy
                                )
                            if s_high_energy > 0:
                                features[f"S_{component_name}_LH_ratio"] = (
                                    s_low_energy / s_high_energy
                                )
                        except Exception as e:
                            print(
                                f"Error extracting phase features for {component_name}: {e}"
                            )

        return features


def prepare_ml_dataset(aftershocks, consolidated_features):
    """
    Prepare dataset for machine learning by combining aftershock metadata
    with waveform features. Focus on location prediction.
    """
    # Create a list to store combined data
    ml_data = []

    # For each aftershock in chronological order
    aftershocks_sorted = aftershocks.sort_values("datetime")

    # Convert event_id to integer if it's not already
    if aftershocks_sorted["event_id"].dtype != "int64":
        aftershocks_sorted["event_id"] = aftershocks_sorted["event_id"].astype(int)

    logging.info(f"Processing {len(aftershocks_sorted)} aftershocks")

    # Count how many have matching features
    matching_features_count = 0
    feature_keys_found = set()

    for idx, row in aftershocks_sorted.iterrows():
        event_id = row["event_id"]

        # Skip if no waveform features available
        if event_id not in consolidated_features:
            continue

        matching_features_count += 1

        # Get features for this event
        features = consolidated_features[event_id].copy()

        # Track which feature keys we find
        feature_keys_found.update(features.keys())

        # Add metadata as features
        features["depth_km"] = row["source_depth_km"]
        features["hours_since_mainshock"] = row["hours_since_mainshock"]

        # Add target variables (location)
        features["latitude"] = row["source_latitude_deg"]
        features["longitude"] = row["source_longitude_deg"]

        # Add to dataset
        ml_data.append(features)

    logging.info(
        f"Found {matching_features_count} aftershocks with matching waveform features"
    )

    # Log the most common features found
    logging.info(f"Found {len(feature_keys_found)} unique feature keys")

    # Count feature occurrences
    feature_counts = {}
    for data_item in ml_data:
        for key in data_item.keys():
            if key not in feature_counts:
                feature_counts[key] = 0
            feature_counts[key] += 1

    # Log the top 20 most common features
    logging.info("Most common features:")
    for key, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[
        :20
    ]:
        if key not in [
            "latitude",
            "longitude",
            "depth_km",
            "hours_since_mainshock",
        ]:
            logging.info(
                f"  {key}: {count}/{matching_features_count} ({count/matching_features_count*100:.1f}%)"
            )

    # Convert to DataFrame
    ml_df = pd.DataFrame(ml_data)

    # Log dataset shape
    logging.info(f"Initial ML dataset shape: {ml_df.shape}")

    # Check for missing values
    missing_values = ml_df.isna().sum().sum()
    if missing_values > 0:
        logging.info(f"Dataset contains {missing_values} missing values")
        logging.info("Columns with missing values:")
        for col in ml_df.columns:
            missing_count = ml_df[col].isna().sum()
            if missing_count > 0:
                logging.info(
                    f"  {col}: {missing_count} missing values ({missing_count/len(ml_df)*100:.1f}%)"
                )

    # Handle missing values
    logging.info("Filling missing values with 0")
    ml_df = ml_df.fillna(0)

    # Check for constant columns
    constant_cols = []
    for col in ml_df.columns:
        if ml_df[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        logging.info(f"Found {len(constant_cols)} constant columns:")
        for col in constant_cols[:10]:  # List first 10
            logging.info(f"  {col}")
        if len(constant_cols) > 10:
            logging.info(f"  ... and {len(constant_cols) - 10} more")

    return ml_df


def safe_engineer_features(ml_df):
    """
    Engineer features using only historical data for each sample.
    For rolling averages, use a trailing window that excludes the current event.
    """
    logging.info("Engineering additional features (using safe trailing windows)...")

    # Sort by time
    ml_df = ml_df.sort_values("hours_since_mainshock").reset_index(drop=True)

    # Time-based features
    ml_df["log_hours"] = np.log1p(ml_df["hours_since_mainshock"])
    ml_df["day_number"] = ml_df["hours_since_mainshock"] // 24

    # Sequential features: when computing time-differences ensure that only past information is used.
    # Here, we compute the difference with the previous event.
    ml_df["hours_since_last"] = ml_df["hours_since_mainshock"].diff().fillna(0)

    # Use a trailing (past only) rolling window by setting closed='left'
    window_size = 5
    ml_df["depth_rolling_avg"] = (
        ml_df["depth_km"].rolling(window=window_size, closed="left").mean()
    )
    ml_df["depth_rolling_std"] = (
        ml_df["depth_km"].rolling(window=window_size, closed="left").std()
    )

    # Fill the NaN values that arise for the first few rows
    ml_df["depth_rolling_avg"].fillna(ml_df["depth_km"], inplace=True)
    ml_df["depth_rolling_std"].fillna(0, inplace=True)

    # Create energy ratio features (these typically do not leak location if computed only from waveform values)
    total_energy = ml_df["Z_energy"] + ml_df["N_energy"] + ml_df["E_energy"]
    ml_df["Z_energy_ratio"] = ml_df["Z_energy"] / (total_energy + 1e-10)
    ml_df["N_energy_ratio"] = ml_df["N_energy"] / (total_energy + 1e-10)
    ml_df["E_energy_ratio"] = ml_df["E_energy"] / (total_energy + 1e-10)

    logging.info(f"Feature engineering complete; dataset shape is now: {ml_df.shape}")
    return ml_df


def prepare_holdout_split(ml_df, holdout_ratio=0.2):
    """
    Split the ML dataset into training and holdout sets.
    Since the data are time series, we sort by the event time (hours_since_mainshock)
    and take the last holdout_ratio portion as the holdout set.
    """
    ml_df = ml_df.sort_values("hours_since_mainshock").reset_index(drop=True)
    holdout_index = int(len(ml_df) * (1 - holdout_ratio))
    train_df = ml_df.iloc[:holdout_index].copy()
    test_df = ml_df.iloc[holdout_index:].copy()
    logging.info(
        f"Training set shape: {train_df.shape}, Holdout set shape: {test_df.shape}"
    )
    return train_df, test_df


def train_location_prediction_model_holdout(ml_df):
    """
    Train a model to predict aftershock locations.
    Uses a holdout set (the last 20% of the events) to evaluate the model
    on unseen, future data.
    """
    # List the features you want to use.
    # Make sure none of these features include target data or are computed
    # using future values (we modified the rolling functions above).
    selected_features = [
        # Original features
        "Z_energy",
        "N_energy",
        "E_energy",
        "Z_dominant_freq",
        "N_dominant_freq",
        "E_dominant_freq",
        "Z_PS_ratio",
        "N_PS_ratio",
        "E_PS_ratio",
        "P_Z_energy",
        "P_N_energy",
        "P_E_energy",
        "S_Z_energy",
        "S_N_energy",
        "S_E_energy",
        "hours_since_mainshock",
        "log_hours",
        "day_number",
        "hours_since_last",
        "Z_energy_ratio",
        "N_energy_ratio",
        "E_energy_ratio",
        # New wavelet features
        "Z_wavelet_band_0_energy",
        "N_wavelet_band_0_energy",
        "E_wavelet_band_0_energy",
        "Z_wavelet_band_ratio_0_1",
        "N_wavelet_band_ratio_0_1",
        "E_wavelet_band_ratio_0_1",
        # New spectrogram features
        "Z_spec_dom_freq_std",
        "N_spec_dom_freq_std",
        "E_spec_dom_freq_std",
        "Z_low_freq_decay_rate",
        "N_low_freq_decay_rate",
        "E_low_freq_decay_rate",
        # New polarization features
        "P_linearity",
        "P_planarity",
        "S_linearity",
        "S_planarity",
        # New phase-specific features
        "P_Z_dom_freq",
        "P_N_dom_freq",
        "P_E_dom_freq",
        "S_Z_dom_freq",
        "S_N_dom_freq",
        "S_E_dom_freq",
        "P_Z_LH_ratio",
        "P_N_LH_ratio",
        "P_E_LH_ratio",
        "S_Z_LH_ratio",
        "S_N_LH_ratio",
        "S_E_LH_ratio",
        # "depth_km",
        # "depth_rolling_avg"
    ]
    # Ensure only available features are used
    selected_features = [feat for feat in selected_features if feat in ml_df.columns]
    logging.info(f"Selected {len(selected_features)} features for prediction.")

    # Split dataset into a training set and a holdout test set.
    train_df, test_df = prepare_holdout_split(ml_df, holdout_ratio=0.2)
    X_train = train_df[selected_features]
    y_train = train_df[["latitude", "longitude"]]
    X_test = test_df[selected_features]
    y_test = test_df[["latitude", "longitude"]]

    # Optional: Use TimeSeriesSplit on the training set for cross-validation.
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []
    for train_idx, val_idx in tscv.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline_cv = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )
        pipeline_cv.fit(X_train_cv, y_train_cv)
        y_val_pred = pipeline_cv.predict(X_val_cv)
        mae = mean_absolute_error(y_val_cv, y_val_pred)
        r2 = r2_score(y_val_cv, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_val_pred))
        cv_results.append({"MAE": mae, "R2": r2, "RMSE": rmse})
        logging.info(f"CV Fold: MAE={mae:.4f}, R2={r2:.4f}, RMSE={rmse:.4f}")

    # Train final model on full training set
    pipeline_final = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    pipeline_final.fit(X_train, y_train)

    # Evaluate on the holdout (test) set
    y_test_pred = pipeline_final.predict(X_test)

    # Compute error metrics based on the haversine distance
    errors_km = []
    for i in range(len(y_test)):
        true_lat, true_lon = y_test.iloc[i]
        pred_lat, pred_lon = y_test_pred[i]
        # Haversine formula to calculate distance between two lat/lon points
        distance = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
        errors_km.append(distance)
    median_error_km = np.median(errors_km)
    mean_error_km = np.mean(errors_km)

    logging.info(
        f"Holdout set evaluation: MAE={mean_absolute_error(y_test, y_test_pred):.4f}, "
        f"Median Error={median_error_km:.2f} km, Mean Error={mean_error_km:.2f} km"
    )

    # Optionally return the final pipeline and evaluation metrics
    return (
        pipeline_final,
        cv_results,
        {"median_error_km": median_error_km, "mean_error_km": mean_error_km},
        selected_features,
    )


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers."""
    R = 6371  # Earth radius in kilometers
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def visualize_results(ml_df, model, feature_importance, results, selected_features):
    """
    Visualize prediction results and feature importance.
    The 'results' parameter can now be a dictionary of aggregated holdout metrics.
    """
    # 1. Feature Importance Plot
    plt.figure(figsize=(10, 8))
    feature_importance.plot(kind="barh")
    plt.title("Feature Importance for Location Prediction")
    plt.tight_layout()
    plt.savefig("feature_importance_location.png")
    plt.close()

    # 2. Prepare test data - use last 20% for visualization based on holdout split.
    split_idx = int(len(ml_df) * 0.8)
    train_df = ml_df.iloc[:split_idx]
    test_df = ml_df.iloc[split_idx:]

    # Filter features
    X_test = test_df[selected_features]
    y_true = test_df[["latitude", "longitude"]]

    # Predict on the test set
    y_pred = model.predict(X_test)
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = pd.DataFrame(
            y_pred, columns=["latitude", "longitude"], index=y_true.index
        )

    # 3. Calculate errors in km using haversine_distance
    errors_km = []
    for i in range(len(y_true)):
        true_lat, true_lon = y_true.iloc[i]
        pred_lat, pred_lon = y_pred.iloc[i]
        distance = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
        errors_km.append(distance)

    # 4. Create Error Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(errors_km, bins=20)
    plt.xlabel("Location Error (km)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Location Prediction Errors")
    plt.axvline(
        np.median(errors_km),
        color="r",
        linestyle="--",
        label=f"Median Error: {np.median(errors_km):.2f} km",
    )
    plt.legend()
    plt.grid(True)
    plt.savefig("location_error_distribution.png")
    plt.close()

    # 5. Actual vs Predicted Map Plot
    logging.info("Verification statistics before plotting:")
    logging.info(f"Number of points: {len(y_true)}")
    logging.info(
        f"Error statistics: min={min(errors_km):.2f} km, max={max(errors_km):.2f} km, "
        f"median={np.median(errors_km):.2f} km, mean={np.mean(errors_km):.2f} km"
    )

    is_identical = np.allclose(y_true.values, y_pred.values)
    if is_identical:
        logging.warning(
            "WARNING: True and predicted coordinates are identical or very close!"
        )

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        y_true["longitude"],
        y_true["latitude"],
        c="blue",
        label="Actual",
        alpha=0.7,
        s=50,
    )
    ax.scatter(
        y_pred["longitude"],
        y_pred["latitude"],
        c="red",
        label="Predicted",
        alpha=0.5,
        s=30,
    )
    for i in range(len(y_true)):
        true_lat, true_lon = y_true.iloc[i]
        pred_lat, pred_lon = y_pred.iloc[i]
        ax.plot([true_lon, pred_lon], [true_lat, pred_lat], "k-", alpha=0.2)
        if i < 3:
            logging.info(
                f"Sample {i}: True: ({true_lat:.4f}, {true_lon:.4f}), "
                f"Pred: ({pred_lat:.4f}, {pred_lon:.4f}), Error: {errors_km[i]:.2f} km"
            )

    median_error = np.median(errors_km)
    mean_error = np.mean(errors_km)
    ax.set_title(
        f"Actual vs Predicted Locations\nMedian Error: {median_error:.2f} km, Mean Error: {mean_error:.2f} km"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    textstr = "\n".join(
        (
            f"Total points: {len(y_true)}",
            f"Median error: {median_error:.2f} km",
            f"Mean error: {mean_error:.2f} km",
            f"Min error: {min(errors_km):.2f} km",
            f"Max error: {max(errors_km):.2f} km",
        )
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    plt.tight_layout()
    plt.savefig("spatial_prediction.png", dpi=300)
    plt.close()
    logging.info("Saved location prediction map to spatial_prediction.png")

    # 6. Performance Metrics Summary
    # If results is a dictionary (holdout_results), then use it directly.
    if isinstance(results, dict):
        avg_results = results
    else:
        avg_results = {
            metric: np.mean([r[metric] for r in results])
            for metric in results[0].keys()
        }

    plt.figure(figsize=(10, 6))
    plt.bar(avg_results.keys(), avg_results.values())
    plt.title("Model Performance Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_performance.png")
    plt.close()

    # 7. Error vs Time Plot
    test_df.loc[:, "prediction_error_km"] = errors_km
    plt.figure(figsize=(12, 6))
    plt.scatter(
        test_df["hours_since_mainshock"],
        test_df["prediction_error_km"],
        alpha=0.7,
        c="purple",
    )
    plt.xlabel("Hours Since Mainshock")
    plt.ylabel("Prediction Error (km)")
    plt.title("Prediction Error vs Time Since Mainshock")
    plt.grid(True)
    plt.savefig("error_vs_time.png")
    plt.close()

    # Print summary metrics
    print("\nModel Performance Summary:")
    for metric, value in avg_results.items():
        print(f"  {metric}: {value:.4f}")


def create_lstm_model(X_train, y_train, X_test, y_test):
    """
    Create and train an LSTM model for location prediction.

    Args:
        X_train: Training features
        y_train: Training targets (lat, lon)
        X_test: Test features
        y_test: Test targets (lat, lon)

    Returns:
        Trained model, predictions, and scaler objects
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Scale features to [0,1] range
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale targets separately for each coordinate
    lat_scaler = MinMaxScaler()
    lon_scaler = MinMaxScaler()

    # Fit and transform latitude
    y_train_lat = lat_scaler.fit_transform(y_train["latitude"].values.reshape(-1, 1))
    y_test_lat = lat_scaler.transform(y_test["latitude"].values.reshape(-1, 1))

    # Fit and transform longitude
    y_train_lon = lon_scaler.fit_transform(y_train["longitude"].values.reshape(-1, 1))
    y_test_lon = lon_scaler.transform(y_test["longitude"].values.reshape(-1, 1))

    # Reshape input for LSTM [samples, time steps, features]
    # Since we're not using sequential data in the time dimension,
    # we'll use 1 for the time steps dimension
    X_train_reshaped = X_train_scaled.reshape(
        X_train_scaled.shape[0], 1, X_train_scaled.shape[1]
    )
    X_test_reshaped = X_test_scaled.reshape(
        X_test_scaled.shape[0], 1, X_test_scaled.shape[1]
    )

    # Create separate models for latitude and longitude

    # Latitude model
    lat_model = Sequential(
        [
            LSTM(64, input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    lat_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Train latitude model
    lat_model.fit(
        X_train_reshaped,
        y_train_lat,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )

    # Longitude model
    lon_model = Sequential(
        [
            LSTM(64, input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    lon_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Train longitude model
    lon_model.fit(
        X_train_reshaped,
        y_train_lon,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )

    # Make predictions
    lat_pred_scaled = lat_model.predict(X_test_reshaped, verbose=0)
    lon_pred_scaled = lon_model.predict(X_test_reshaped, verbose=0)

    # Inverse transform to get actual coordinates
    lat_pred = lat_scaler.inverse_transform(lat_pred_scaled)
    lon_pred = lon_scaler.inverse_transform(lon_pred_scaled)

    # Combine predictions
    y_pred = np.column_stack((lat_pred, lon_pred))

    return {
        "lat_model": lat_model,
        "lon_model": lon_model,
        "predictions": y_pred,
        "feature_scaler": feature_scaler,
        "lat_scaler": lat_scaler,
        "lon_scaler": lon_scaler,
    }


def compare_models(ml_df, selected_features):
    """
    Train and evaluate multiple models on the same dataset
    and return performance metrics for comparison.
    """
    # Prepare train/test split
    train_df, test_df = prepare_holdout_split(ml_df, holdout_ratio=0.2)
    X_train = train_df[selected_features]
    y_train = train_df[["latitude", "longitude"]]
    X_test = test_df[selected_features]
    y_test = test_df[["latitude", "longitude"]]

    # Define models to test - RandomForest supports multi-output directly
    models = {
        "RandomForest": {
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "multi_output": True,
        }
    }

    # Add models that require separate training for each output
    single_output_models = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(alpha=0.01, random_state=42),
    }

    # Store results
    results = {}

    # Train and evaluate each model
    for model_name, model_config in models.items():
        logging.info(f"Training {model_name}...")

        # Create pipeline with standardization
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", model_config["model"])]
        )

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_test_pred = pipeline.predict(X_test)

        # Calculate errors
        errors_km = calculate_distance_errors(y_test, y_test_pred)

        # Calculate metrics
        median_error_km = np.median(errors_km)
        mean_error_km = np.mean(errors_km)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Store results
        results[model_name] = {
            "median_error_km": median_error_km,
            "mean_error_km": mean_error_km,
            "MAE": mae,
            "R2": r2,
            "RMSE": rmse,
        }

        logging.info(
            f"{model_name} evaluation: MAE={mae:.4f}, "
            f"Median Error={median_error_km:.2f} km, "
            f"Mean Error={mean_error_km:.2f} km"
        )

    # Train and evaluate single-output models
    for model_name, model in single_output_models.items():
        logging.info(f"Training {model_name}...")

        # We need separate models for latitude and longitude
        lat_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", clone(model)),  # Use clone to create a fresh copy
            ]
        )

        lon_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", clone(model)),  # Use clone to create a fresh copy
            ]
        )

        # Train the models
        lat_pipeline.fit(X_train, y_train["latitude"])
        lon_pipeline.fit(X_train, y_train["longitude"])

        # Make predictions
        lat_pred = lat_pipeline.predict(X_test)
        lon_pred = lon_pipeline.predict(X_test)

        # Combine predictions
        y_test_pred = np.column_stack((lat_pred, lon_pred))

        # Calculate errors
        errors_km = calculate_distance_errors(y_test, y_test_pred)

        # Calculate metrics
        median_error_km = np.median(errors_km)
        mean_error_km = np.mean(errors_km)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Store results
        results[model_name] = {
            "median_error_km": median_error_km,
            "mean_error_km": mean_error_km,
            "MAE": mae,
            "R2": r2,
            "RMSE": rmse,
        }

        logging.info(
            f"{model_name} evaluation: MAE={mae:.4f}, "
            f"Median Error={median_error_km:.2f} km, "
            f"Mean Error={mean_error_km:.2f} km"
        )

    # Add LSTM model
    try:
        logging.info(f"Training LSTM...")
        lstm_results = create_lstm_model(X_train, y_train, X_test, y_test)
        y_test_pred = lstm_results["predictions"]

        # Calculate errors
        errors_km = calculate_distance_errors(y_test, y_test_pred)

        # Calculate metrics
        median_error_km = np.median(errors_km)
        mean_error_km = np.mean(errors_km)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Store results
        results["LSTM"] = {
            "median_error_km": median_error_km,
            "mean_error_km": mean_error_km,
            "MAE": mae,
            "R2": r2,
            "RMSE": rmse,
        }

        logging.info(
            f"LSTM evaluation: MAE={mae:.4f}, "
            f"Median Error={median_error_km:.2f} km, "
            f"Mean Error={mean_error_km:.2f} km"
        )
    except Exception as e:
        logging.error(f"Error training LSTM model: {e}")

    return results


def calculate_distance_errors(y_true, y_pred):
    """Calculate haversine distances between true and predicted coordinates"""
    errors_km = []
    for i in range(len(y_true)):
        if isinstance(y_true, pd.DataFrame):
            true_lat, true_lon = y_true.iloc[i]
        else:
            true_lat, true_lon = y_true[i]

        if isinstance(y_pred, pd.DataFrame):
            pred_lat, pred_lon = y_pred.iloc[i]
        else:
            pred_lat, pred_lon = y_pred[i]

        distance = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
        errors_km.append(distance)
    return errors_km


def visualize_model_comparison(results):
    """
    Create visualizations to compare model performances.
    """
    # Bar chart of median errors
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    median_errors = [results[model]["median_error_km"] for model in models]
    plt.bar(models, median_errors, color="skyblue")
    plt.title("Median Location Error by Model")
    plt.ylabel("Error (km)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for i, v in enumerate(median_errors):
        plt.text(i, v + 1, f"{v:.1f}", ha="center")
    plt.tight_layout()
    plt.savefig("model_comparison_median_error.png")
    plt.close()

    # Radar chart for multiple metrics
    metrics = ["median_error_km", "mean_error_km", "MAE", "RMSE"]

    # Prepare data
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Find max value for each metric to normalize
    max_values = {}
    for metric in metrics:
        max_values[metric] = max([results[model][metric] for model in models])

    for model in models:
        values = [(results[model][metric] / max_values[metric]) for metric in metrics]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)

    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    plt.legend(loc="upper right")
    plt.title("Model Comparison (normalized metrics)")
    plt.tight_layout()
    plt.savefig("model_comparison_radar.png")
    plt.close()

    # Results table
    print("\nModel Comparison Results:")
    print("-" * 80)
    headers = ["Model"] + [m for m in metrics]
    print(f"{headers[0]:<20} " + " ".join([f"{h:>15}" for h in headers[1:]]))
    print("-" * 80)

    for model in models:
        values = [f"{results[model][metric]:.2f}" for metric in metrics]
        print(f"{model:<20} " + " ".join([f"{v:>15}" for v in values]))


def match_aftershocks_to_events(
    aftershocks, consolidated_metadata, distance_threshold=2.0
):
    """
    Match aftershocks to events in consolidated metadata using a more robust approach.

    Args:
        aftershocks: DataFrame containing aftershock data
        consolidated_metadata: DataFrame containing consolidated event data with event_ids
        distance_threshold: Maximum distance in km to consider events as matching

    Returns:
        DataFrame with matched event_ids added to aftershocks
    """
    import numpy as np
    import pandas as pd
    from datetime import timedelta
    import logging

    logging.info("\nMatching aftershocks to events using improved method...")

    # Create a copy of the aftershocks DataFrame
    matched_aftershocks = aftershocks.copy()

    # Initialize event_id column
    matched_aftershocks["event_id"] = np.nan

    # Count of successful matches
    match_count = 0

    # Convert timestamps to datetime objects if they aren't already
    if not isinstance(aftershocks["datetime"].iloc[0], pd.Timestamp):
        aftershocks["datetime"] = pd.to_datetime(aftershocks["datetime"])
    if not isinstance(consolidated_metadata["datetime"].iloc[0], pd.Timestamp):
        consolidated_metadata["datetime"] = pd.to_datetime(
            consolidated_metadata["datetime"]
        )

    # Time threshold for matching (e.g., 1 second)
    time_threshold = timedelta(seconds=1)

    # For each aftershock, find the best matching event
    for i, aftershock in enumerate(aftershocks.itertuples()):
        # Filter potential matches by time first (for efficiency)
        time_matches = consolidated_metadata[
            (consolidated_metadata["datetime"] >= aftershock.datetime - time_threshold)
            & (
                consolidated_metadata["datetime"]
                <= aftershock.datetime + time_threshold
            )
        ]

        if len(time_matches) == 0:
            continue

        # For time matches, calculate spatial distance
        distances = []
        for event in time_matches.itertuples():
            dist = haversine_distance(
                aftershock.source_latitude_deg,
                aftershock.source_longitude_deg,
                event.source_latitude_deg,
                event.source_longitude_deg,
            )
            distances.append((event.event_id, dist))

        # Find the closest event within threshold
        closest_matches = [
            (event_id, dist)
            for event_id, dist in distances
            if dist <= distance_threshold
        ]

        if closest_matches:
            # Sort by distance and take the closest
            closest_matches.sort(key=lambda x: x[1])
            best_match_id, best_match_dist = closest_matches[0]

            # Assign the event_id
            matched_aftershocks.loc[aftershock.Index, "event_id"] = best_match_id
            match_count += 1

        # Provide progress updates
        if (i + 1) % 1000 == 0:
            logging.info(
                f"  Processed {i+1}/{len(aftershocks)} aftershocks, found {match_count} matches so far"
            )

    # Log final results
    missing_ids = matched_aftershocks["event_id"].isna().sum()
    match_percentage = (match_count / len(aftershocks)) * 100

    logging.info(
        f"Matching complete: Found {match_count}/{len(aftershocks)} matches ({match_percentage:.2f}%)"
    )
    logging.info(
        f"Aftershocks without matched event_id: {missing_ids}/{len(aftershocks)}"
    )

    return matched_aftershocks


def main():
    """Main execution function"""
    start_time = datetime.datetime.now()

    logging.info("=== Aftershock Location Prediction Model ===")
    logging.info(f"Started at: {start_time}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"NumPy version: {np.__version__}")
    logging.info(f"Pandas version: {pd.__version__}")
    logging.info(f"Scikit-learn version: {sklearn.__version__}")

    # 1. Load the dataset (limit to 5000 waveforms to keep runtime reasonable)
    logging.info("\nStep 1: Loading and preprocessing data...")
    metadata, iquique, waveform_features_dict = load_aftershock_data_with_waveforms(
        max_waveforms=13400
    )

    # 2. Identify mainshock and aftershocks
    logging.info("\nStep 2: Identifying mainshock and aftershocks...")
    mainshock, aftershocks = identify_mainshock_and_aftershocks(metadata)

    # 3. Consolidate station recordings
    logging.info("\nStep 3: Consolidating station recordings...")
    consolidated_metadata, consolidated_features = consolidate_station_recordings(
        metadata, waveform_features_dict
    )

    # 3.5. Match event_ids to aftershocks
    logging.info("\nStep 3.5: Matching event IDs to aftershocks...")
    # Create key columns in both DataFrames to match events
    aftershocks = match_aftershocks_to_events(
        aftershocks, consolidated_metadata, distance_threshold=2.0
    )

    # Check if merge was successful
    missing_ids = aftershocks["event_id"].isna().sum()
    logging.info(
        f"Aftershocks without matched event_id: {missing_ids}/{len(aftershocks)}"
    )

    # Filter to keep only aftershocks with event_id
    aftershocks = aftershocks[~aftershocks["event_id"].isna()]
    logging.info(f"Proceeding with {len(aftershocks)} aftershocks that have event_ids")

    # 4. Prepare ML dataset
    logging.info("\nStep 4: Preparing machine learning dataset...")
    ml_df = prepare_ml_dataset(aftershocks, consolidated_features)
    logging.info(f"Dataset shape: {ml_df.shape}")

    # Log some basic statistics about the dataset
    if len(ml_df) > 0:
        logging.info("\nDataset statistics:")
        for col in ["hours_since_mainshock", "distance_from_mainshock_km", "depth_km"]:
            if col in ml_df.columns:
                logging.info(
                    f"  {col}: min={ml_df[col].min():.2f}, max={ml_df[col].max():.2f}, mean={ml_df[col].mean():.2f}"
                )

        # Count how many events have each feature
        feature_counts = {
            col: ml_df[col].count()
            for col in ml_df.columns
            if col not in ["latitude", "longitude"]
        }
        logging.info("\nFeature availability counts:")
        for feature, count in sorted(
            feature_counts.items(), key=lambda x: x[1], reverse=True
        ):
            if count < len(ml_df):
                logging.info(
                    f"  {feature}: {count}/{len(ml_df)} ({count/len(ml_df)*100:.1f}%)"
                )
    else:
        logging.error("Empty dataset! Cannot proceed with training.")
        return

    # 5. Engineer features
    logging.info("\nStep 5: Engineering features...")
    ml_df = safe_engineer_features(ml_df)
    logging.info(f"Dataset shape after feature engineering: {ml_df.shape}")

    # 6. Train location prediction model
    # logging.info("\nStep 6: Training location prediction model...")
    # model, cv_results, holdout_results, selected_features = (
    #     train_location_prediction_model_holdout(ml_df)
    # )

    # 6. Define and select features
    selected_features = [
        # Your chosen features here
        "N_spec_dom_freq_std",
        "S_Z_LH_ratio",
        "S_E_LH_ratio",
        "log_hours",
        "hours_since_mainshock",
        "N_PS_ratio",
        "Z_energy_ratio",
        "Z_wavelet_band_ratio_0_1",
        "Z_spec_dom_freq_std",
        "Z_low_freq_decay_rate",
        "N_energy_ratio",
        "Z_PS_ratio",
        "P_E_LH_ratio",
        "day_number",
        "P_linearity",
        # "depth_km",
        # "depth_rolling_avg",
    ]
    # Ensure only available features are used
    selected_features = [feat for feat in selected_features if feat in ml_df.columns]
    logging.info(f"Selected {len(selected_features)} features for prediction.")

    logging.info("\nStep 7: Comparing different models...")
    model_results = compare_models(ml_df, selected_features)

    # 8. Visualize comparison results
    logging.info("\nStep 8: Visualizing comparison results...")
    visualize_model_comparison(model_results)

    # importances = model.named_steps["model"].feature_importances_
    # feature_importance = pd.Series(importances, index=selected_features).sort_values(
    #     ascending=False
    # )

    # print("\nFeature Importance:")
    # for feature, importance in feature_importance.items():
    #     print(f"{feature}: {importance:.4f}")

    # # 7. Visualize results
    # logging.info("\nStep 7: Visualizing results...")
    # visualize_results(
    #     ml_df, model, feature_importance, holdout_results, selected_features
    # )

    # Log execution time
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    logging.info(f"\nExecution completed at: {end_time}")
    logging.info(f"Total execution time: {execution_time}")

    logging.info("\nDone! Check the generated visualization files for results.")
    logging.info(f"Log file saved to: {log_filename}")


if __name__ == "__main__":
    # Ensure matplotlib doesn't use interactive backend
    plt.switch_backend("agg")

    # Make sure required directories exist
    for directory in ["logs", "plots", "results"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Import sys for version info
    import sys
    import sklearn

    try:
        main()
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        raise
