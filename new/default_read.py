import numpy as np
import pandas as pd
from tqdm import tqdm
import seisbench.data as sbd
from scipy import signal


def load_aftershock_data_with_waveforms(max_waveforms):
    """
    Load and preprocess aftershock data from the Iquique dataset,
    including waveform data
    """
    print("Loading Iquique dataset using SeisBench...")
    iquique = sbd.Iquique()

    # Get metadata
    metadata = iquique.metadata.copy()

    # Filter out rows with missing essential data
    metadata = metadata.dropna(
        subset=[
            "source_origin_time",
            "source_latitude_deg",
            "source_longitude_deg",
            "source_depth_km",
        ]
    )

    # Convert timestamps
    metadata["datetime"] = pd.to_datetime(metadata["source_origin_time"])

    # Sort by time
    metadata = metadata.sort_values("datetime")

    # Create a dictionary to store waveform features
    waveform_features_dict = {}

    # Initialize feature extractor
    feature_extractor = WaveformFeatureExtractor()

    # Limit the number of waveforms to process
    sample_indices = metadata.index[: min(max_waveforms, len(metadata))]

    print(f"Extracting waveform features for {len(sample_indices)} events...")
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

            # Store features
            waveform_features_dict[idx] = features

        except Exception as e:
            print(f"Error processing waveform {idx}: {e}")
            waveform_features_dict[idx] = {}

    print(
        f"Successfully extracted features for {len(waveform_features_dict)} waveforms"
    )

    return metadata, iquique, waveform_features_dict


def identify_mainshock_and_aftershocks(metadata):
    """
    Identify the mainshock and its associated aftershocks based on data patterns
    """
    # The known Iquique earthquake was in early April 2014
    # Let's create a date range for the expected period
    april_start = pd.Timestamp("2014-04-01", tz="UTC")
    april_end = pd.Timestamp("2014-04-05", tz="UTC")

    # Find events in early April 2014
    april_events = metadata[
        (metadata["datetime"] >= april_start) & (metadata["datetime"] <= april_end)
    ]

    if len(april_events) == 0:
        print(
            "No events found in early April 2014 timeframe. Using alternative approach."
        )
        # Alternative approach: find the earliest events in the dataset
        metadata_sorted = metadata.sort_values("datetime")
        earliest_date = metadata_sorted["datetime"].iloc[0]
        print(f"Earliest event in dataset: {earliest_date}")

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
        print(f"Found {len(april_events)} events in early April 2014.")
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

    print(
        f"Identified potential mainshock at {mainshock['datetime']} at location "
        f"({mainshock['source_latitude_deg']}, {mainshock['source_longitude_deg']}), "
        f"depth {mainshock['source_depth_km']} km"
    )
    print(f"This event is followed by {most_followers} events in the next 24 hours")

    # Select events after the mainshock as aftershocks
    aftershocks = metadata[metadata["datetime"] > mainshock["datetime"]].copy()
    print(f"Found {len(aftershocks)} aftershocks")

    # Create a new feature: time since mainshock in hours
    aftershocks["hours_since_mainshock"] = (
        aftershocks["datetime"] - mainshock["datetime"]
    ).dt.total_seconds() / 3600

    # Create features: distance from mainshock (approximate using Haversine)
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c

        return distance

    aftershocks["distance_from_mainshock_km"] = aftershocks.apply(
        lambda row: haversine_distance(
            mainshock["source_latitude_deg"],
            mainshock["source_longitude_deg"],
            row["source_latitude_deg"],
            row["source_longitude_deg"],
        ),
        axis=1,
    )

    return mainshock, aftershocks


def consolidate_station_recordings(metadata, waveform_features_dict):
    """
    Consolidate multiple station recordings of the same event into a single representation
    as there are 134000 recordings for 410 events
    """
    # Create event IDs based on source parameters
    metadata["lat_rounded"] = np.round(metadata["source_latitude_deg"], 4)
    metadata["lon_rounded"] = np.round(metadata["source_longitude_deg"], 4)
    metadata["depth_rounded"] = np.round(metadata["source_depth_km"], 1)

    metadata["event_id"] = metadata.groupby(
        ["source_origin_time", "lat_rounded", "lon_rounded", "depth_rounded"]
    ).ngroup()

    print(
        f"Original recordings: {len(metadata)}, Unique events: {metadata['event_id'].nunique()}"
    )

    # Create consolidated representations
    consolidated_metadata = []
    consolidated_features = {}

    # Track how many events actually have features
    events_with_features = 0

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

        # Take metadata from best recording
        best_record = group.loc[best_idx].copy()
        best_record["station_count"] = len(group)
        best_record["event_id"] = event_id  # Keep event_id in the metadata
        consolidated_metadata.append(best_record)

        # Map waveform features to the event_id
        if best_idx in waveform_features_dict and waveform_features_dict[best_idx]:
            consolidated_features[event_id] = waveform_features_dict[best_idx]

    print(f"Events with valid waveform features: {events_with_features}")

    # Convert to DataFrame
    consolidated_metadata = pd.DataFrame(consolidated_metadata)
    return consolidated_metadata, consolidated_features


class WaveformFeatureExtractor:
    """
    Class to extract features from seismic waveforms
    """

    def __init__(self, sampling_rate=100.0):
        self.sampling_rate = sampling_rate

    def extract_features(self, waveform, p_arrival=None, s_arrival=None):
        """
        Extract features from waveform data

        Parameters:
        -----------
        waveform : numpy.ndarray
            Waveform data with shape (num_components, num_samples)
        p_arrival : int, optional
            P-wave arrival sample
        s_arrival : int, optional
            S-wave arrival sample

        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        # Initialize feature dictionary
        features = {}

        # Basic time-domain features (overall signal)
        for i, component_name in enumerate(["Z", "N", "E"]):
            if i < waveform.shape[0]:  # Check if component exists
                component = waveform[i]

                # Calculate basic statistics
                features[f"{component_name}_max"] = np.max(np.abs(component))
                features[f"{component_name}_mean"] = np.mean(np.abs(component))
                features[f"{component_name}_std"] = np.std(component)
                features[f"{component_name}_rms"] = np.sqrt(np.mean(component**2))
                features[f"{component_name}_energy"] = np.sum(component**2)
                features[f"{component_name}_kurtosis"] = self._kurtosis(component)

                # Calculate frequency-domain features
                freq_features = self._compute_frequency_features(component)
                for feat_name, feat_value in freq_features.items():
                    features[f"{component_name}_{feat_name}"] = feat_value

        # Extract P-wave and S-wave specific features if arrivals are provided
        if p_arrival is not None and s_arrival is not None:
            p_window_size = 100  # 100 samples after P arrival
            s_window_size = 100  # 100 samples after S arrival

            # Ensure window sizes don't exceed waveform length
            p_window_size = min(p_window_size, waveform.shape[1] - p_arrival)
            s_window_size = min(s_window_size, waveform.shape[1] - s_arrival)

            if p_window_size > 0:
                for i, component_name in enumerate(["Z", "N", "E"]):
                    if i < waveform.shape[0]:  # Check if component exists
                        p_segment = waveform[i, p_arrival : p_arrival + p_window_size]

                        # Calculate P-wave features
                        features[f"P_{component_name}_max"] = np.max(np.abs(p_segment))
                        features[f"P_{component_name}_mean"] = np.mean(
                            np.abs(p_segment)
                        )
                        features[f"P_{component_name}_std"] = np.std(p_segment)
                        features[f"P_{component_name}_energy"] = np.sum(p_segment**2)

                        # P-wave frequency features
                        p_freq_features = self._compute_frequency_features(p_segment)
                        for feat_name, feat_value in p_freq_features.items():
                            features[f"P_{component_name}_{feat_name}"] = feat_value

            if s_window_size > 0:
                for i, component_name in enumerate(["Z", "N", "E"]):
                    if i < waveform.shape[0]:  # Check if component exists
                        s_segment = waveform[i, s_arrival : s_arrival + s_window_size]

                        # Calculate S-wave features
                        features[f"S_{component_name}_max"] = np.max(np.abs(s_segment))
                        features[f"S_{component_name}_mean"] = np.mean(
                            np.abs(s_segment)
                        )
                        features[f"S_{component_name}_std"] = np.std(s_segment)
                        features[f"S_{component_name}_energy"] = np.sum(s_segment**2)

                        # S-wave frequency features
                        s_freq_features = self._compute_frequency_features(s_segment)
                        for feat_name, feat_value in s_freq_features.items():
                            features[f"S_{component_name}_{feat_name}"] = feat_value

            # Calculate P/S amplitude ratios for each component
            for i, component_name in enumerate(["Z", "N", "E"]):
                if i < waveform.shape[0]:  # Check if component exists
                    if (
                        features.get(f"P_{component_name}_max", 0) > 0
                        and features.get(f"S_{component_name}_max", 0) > 0
                    ):
                        features[f"{component_name}_PS_ratio"] = (
                            features[f"P_{component_name}_max"]
                            / features[f"S_{component_name}_max"]
                        )

        return features

    def _compute_frequency_features(self, signal_segment):
        """
        Compute frequency domain features
        """
        features = {}

        # Check if signal segment is not empty
        if len(signal_segment) < 10:
            return {
                "dominant_freq": 0,
                "low_freq_energy": 0,
                "mid_freq_energy": 0,
                "high_freq_energy": 0,
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
            mid_idx = (f >= 5) & (f < 15)
            high_idx = f >= 15

            # Handle empty frequency bands
            features["low_freq_energy"] = np.sum(Pxx[low_idx]) if np.any(low_idx) else 0
            features["mid_freq_energy"] = np.sum(Pxx[mid_idx]) if np.any(mid_idx) else 0
            features["high_freq_energy"] = (
                np.sum(Pxx[high_idx]) if np.any(high_idx) else 0
            )

            # Calculate frequency ratios
            if features["high_freq_energy"] > 0:
                features["low_high_ratio"] = (
                    features["low_freq_energy"] / features["high_freq_energy"]
                )
            else:
                features["low_high_ratio"] = 0

        except Exception as e:
            print(f"Error computing frequency features: {e}")
            features = {
                "dominant_freq": 0,
                "low_freq_energy": 0,
                "mid_freq_energy": 0,
                "high_freq_energy": 0,
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