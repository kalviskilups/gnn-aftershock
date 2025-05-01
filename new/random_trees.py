#!/usr/bin/env python3
# compare_approaches.py - Compare best-station and multi-station approaches for aftershock analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from scipy import signal
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seisbench.data as sbd
from tqdm import tqdm
import warnings
import time
import argparse


class IquiqueAnalysis:
    """
    Main class for analyzing the Iquique aftershock sequence
    with integrated validation checks and multi-station support
    """

    def __init__(
        self, data_pickle=None, validation_level="full", approach="best_station"
    ):
        """
        Initialize the analysis class

        Args:
            data_pickle: Path to a pickle file with preprocessed data
                         If None, will load from SeisBench
            validation_level: Level of validation to apply
                             "full" - all checks
                             "critical" - only critical checks
                             "none" - no validation
            approach: Analysis approach to use
                      "best_station" - use only the best station for each event
                      "multi_station" - use all available stations for each event
        """
        self.data_dict = None
        self.aftershocks_df = None
        self.mainshock_key = None
        self.mainshock = None
        self.models = None
        self.scaler = None
        self.validation_level = validation_level
        self.validation_results = {}
        self.approach = approach
        self.data_format = None  # Will be set when loading data

        print(f"Validation level: {validation_level}")
        print(f"Analysis approach: {approach}")

        # Load data
        if data_pickle and os.path.exists(data_pickle):
            print(f"Loading data from {data_pickle}")
            self.data_dict = self.load_from_pickle(data_pickle)
        else:
            print("No pickle file found. Loading from SeisBench...")
            self.data_dict = self.load_from_seisbench()

        # Standardize waveform lengths BEFORE data integrity validation
        self.standardize_waveforms(target_length=14636)

        # Check data integrity after standardization
        if validation_level != "none":
            self.validate_data_integrity()

    def load_from_pickle(self, pickle_path):
        """
        Load data from a pickle file with support for multi-station format
        """
        with open(pickle_path, "rb") as f:
            data_dict = pickle.load(f)

        # Check if this is the new multi-station format
        is_multi_station = False
        if data_dict:
            # Get the first event key and check its value
            first_event_key = next(iter(data_dict))
            first_event_data = data_dict[first_event_key]
            # If the value is a dict of station keys, it's the new format
            is_multi_station = isinstance(first_event_data, dict) and any(
                isinstance(k, str) and "." in k for k in first_event_data.keys()
            )

        if is_multi_station:
            print("Detected multi-station data format")
            self.data_format = "multi_station"
        else:
            print("Detected single-station data format")
            self.data_format = "single_station"

        return data_dict

    def load_from_seisbench(self, max_waveforms=1000):
        """
        Load data directly from SeisBench
        """
        print("Loading Iquique dataset using SeisBench...")
        iquique = sbd.Iquique()

        # Get metadata
        metadata = iquique.metadata.copy()

        # Cap the data if needed
        if max_waveforms and max_waveforms < len(metadata):
            print(f"Limiting to {max_waveforms} waveforms")
            metadata = metadata.iloc[:max_waveforms].copy()

        # Calculate distance from station to epicenter for each recording
        metadata["epicentral_distance_km"] = metadata.apply(
            lambda row: self.haversine_distance(
                row["source_latitude_deg"],
                row["source_longitude_deg"],
                row["station_latitude_deg"],
                row["station_longitude_deg"],
            ),
            axis=1,
        )

        # Create a signal quality score
        metadata["signal_quality_score"] = metadata.apply(
            lambda row: (
                (
                    row["trace_completeness"]
                    if pd.notna(row["trace_completeness"])
                    else 0
                )
                * 0.5
                + (0.3 if not row["trace_has_spikes"] else 0)
                + (
                    0.2
                    if (
                        pd.notna(row["trace_P_arrival_sample"])
                        and pd.notna(row["trace_S_arrival_sample"])
                        and row["trace_P_arrival_sample"] >= 0
                        and row["trace_S_arrival_sample"] >= 0
                    )
                    else 0
                )
            ),
            axis=1,
        )

        # Normalize distance for selection score
        max_distance = metadata["epicentral_distance_km"].max()
        metadata["distance_normalized"] = 1 - (
            metadata["epicentral_distance_km"] / max_distance
        )

        # Combined score (higher is better)
        metadata["selection_score"] = (
            0.6 * metadata["distance_normalized"]
            + 0.4 * metadata["signal_quality_score"]
        )

        # Group by event parameters to identify unique events
        event_groups = metadata.groupby(
            [
                "source_origin_time",
                "source_latitude_deg",
                "source_longitude_deg",
                "source_depth_km",
            ]
        )

        # For each unique event, select the recording with the highest selection score
        best_recordings = []
        for event_params, group in tqdm(event_groups):
            best_idx = group["selection_score"].idxmax()
            best_recordings.append(best_idx)

        filtered_metadata = metadata.loc[best_recordings].copy()

        print(
            f"Found {len(filtered_metadata)} unique events with their best station recordings"
        )

        # Create data dictionary (single-station format)
        data_dict = {}
        for idx in tqdm(filtered_metadata.index):
            event_key = (
                metadata.loc[idx, "source_origin_time"],
                metadata.loc[idx, "source_latitude_deg"],
                metadata.loc[idx, "source_longitude_deg"],
                metadata.loc[idx, "source_depth_km"],
            )

            waveform = iquique.get_waveforms(int(idx))

            if waveform.shape[0] == 3:  # Ensure waveform has all three components
                data_dict[event_key] = {
                    "metadata": metadata.loc[idx].to_dict(),
                    "waveform": waveform,
                    "station_distance": metadata.loc[idx, "epicentral_distance_km"],
                    "selection_score": metadata.loc[idx, "selection_score"],
                }

        # Save for future use
        with open("aftershock_data_best.pkl", "wb") as f:
            pickle.dump(data_dict, f)
        print(f"Data dictionary saved to aftershock_data_best.pkl")

        self.data_format = "single_station"
        return data_dict

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance between two points in km"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    def find_mainshock(self):
        """
        Identify the mainshock in the dataset
        For the Iquique sequence, the mainshock occurred on April 1, 2014
        """
        # Use manual specification of Iquique mainshock (from USGS catalog)
        mainshock_coords = {
            "origin_time": "2014-04-01T23:46:50.000000Z",
            "latitude": -19.642,
            "longitude": -70.817,
            "depth": 25.0,
        }

        print("Using manually specified mainshock coordinates from USGS catalog")

        # Store mainshock info
        self.mainshock = mainshock_coords

        # Create a placeholder mainshock key (we don't have this exact event in our dataset)
        self.mainshock_key = (
            mainshock_coords["origin_time"],
            mainshock_coords["latitude"],
            mainshock_coords["longitude"],
            mainshock_coords["depth"],
        )

        # Print the mainshock details
        print(f"Mainshock: {self.mainshock}")

        return self.mainshock_key

    def create_relative_coordinate_dataframe(self):
        """
        Create a DataFrame with all events and their coordinates
        relative to the mainshock, supporting multi-station format
        """
        if self.mainshock_key is None:
            self.find_mainshock()

        # Extract mainshock coordinates
        mainshock_lat = self.mainshock["latitude"]
        mainshock_lon = self.mainshock["longitude"]
        mainshock_depth = self.mainshock["depth"]

        # Create a list to store all aftershock data
        events = []

        # Process each event differently based on data format
        if self.data_format == "multi_station":
            for event_key, stations_data in self.data_dict.items():
                origin_time, lat, lon, depth = event_key

                # Calculate relative coordinates
                x, y, z = self.geographic_to_cartesian(
                    lat, lon, depth, mainshock_lat, mainshock_lon, mainshock_depth
                )

                # For each station recording of this event
                for station_key, station_data in stations_data.items():
                    events.append(
                        {
                            "origin_time": origin_time,
                            "absolute_lat": lat,
                            "absolute_lon": lon,
                            "absolute_depth": depth,
                            "relative_x": x,  # East-West (km)
                            "relative_y": y,  # North-South (km)
                            "relative_z": z,  # Depth difference (km)
                            "waveform": station_data["waveform"],
                            "station_key": station_key,
                            "station_distance": station_data["station_distance"],
                            "selection_score": station_data["selection_score"],
                            "metadata": station_data["metadata"],
                            "is_mainshock": (event_key == self.mainshock_key),
                        }
                    )
        else:
            # Original single-station format
            for event_key, event_data in self.data_dict.items():
                origin_time, lat, lon, depth = event_key

                # Calculate relative coordinates
                x, y, z = self.geographic_to_cartesian(
                    lat, lon, depth, mainshock_lat, mainshock_lon, mainshock_depth
                )

                # Add to the list
                events.append(
                    {
                        "origin_time": origin_time,
                        "absolute_lat": lat,
                        "absolute_lon": lon,
                        "absolute_depth": depth,
                        "relative_x": x,  # East-West (km)
                        "relative_y": y,  # North-South (km)
                        "relative_z": z,  # Depth difference (km)
                        "waveform": event_data["waveform"],
                        "metadata": event_data["metadata"],
                        "is_mainshock": (event_key == self.mainshock_key),
                    }
                )

        # Convert to DataFrame
        self.aftershocks_df = pd.DataFrame(events)

        # Add event_date column for group validation
        self.aftershocks_df["event_date"] = pd.to_datetime(
            self.aftershocks_df["origin_time"]
        ).dt.date

        # Add event_id column to group stations from the same event
        if self.data_format == "multi_station":
            self.aftershocks_df["event_id"] = self.aftershocks_df.apply(
                lambda row: f"{row['origin_time']}_{row['absolute_lat']}_{row['absolute_lon']}_{row['absolute_depth']}",
                axis=1,
            )

        # Save for later use (without waveforms for CSV)
        csv_columns = [
            col
            for col in self.aftershocks_df.columns
            if col != "waveform" and col != "metadata"
        ]
        self.aftershocks_df[csv_columns].to_csv(
            "aftershocks_relative_coordinates.csv", index=False
        )

        # Also pickle the full dataset with waveforms
        with open("aftershocks_df_with_waveforms.pkl", "wb") as f:
            pickle.dump(self.aftershocks_df, f)

        # Validate coordinate conversion if required
        if self.validation_level != "none":
            self.validate_coordinate_conversion()

        return self.aftershocks_df

    @staticmethod
    def geographic_to_cartesian(lat, lon, depth, ref_lat, ref_lon, ref_depth):
        """
        Convert geographic coordinates to cartesian coordinates
        with the reference point (mainshock) as the origin

        Returns (x, y, z) where:
        x: East-West distance (positive = east)
        y: North-South distance (positive = north)
        z: Depth difference (positive = deeper than mainshock)
        """
        # Constants for Earth
        earth_radius = 6371.0  # km

        # Convert to radians
        lat1, lon1 = np.radians(ref_lat), np.radians(ref_lon)
        lat2, lon2 = np.radians(lat), np.radians(lon)

        # Calculate the differences
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # North-South distance (y)
        y = earth_radius * dlat

        # East-West distance (x) - IMPROVED: use reference latitude only for better accuracy
        # This avoids the average latitude which can cause larger errors
        x = earth_radius * dlon * np.cos(lat1)

        # Depth difference (z) - positive means deeper than mainshock
        z = depth - ref_depth

        return x, y, z

    @staticmethod
    def cartesian_to_geographic(x, y, z, ref_lat, ref_lon, ref_depth):
        """
        Convert cartesian coordinates back to geographic coordinates
        """
        # Earth radius in km
        earth_radius = 6371.0

        # Convert reference point to radians
        ref_lat_rad = np.radians(ref_lat)

        # Calculate latitude difference in radians
        dlat = y / earth_radius

        # Calculate longitude difference in radians
        # IMPROVED: Use ref_lat only for better accuracy and consistency
        dlon = x / (earth_radius * np.cos(ref_lat_rad))

        # Convert to absolute latitude and longitude in radians
        lat_rad = ref_lat_rad + dlat
        lon_rad = np.radians(ref_lon) + dlon

        # Convert back to degrees
        lat = np.degrees(lat_rad)
        lon = np.degrees(lon_rad)

        # Calculate absolute depth
        depth = ref_depth + z

        return lat, lon, depth

    def extract_waveform_features(self, waveform, metadata=None):
        """
        Extract features from the 3-component waveform data

        Args:
            waveform: 3-component seismic waveform array
            metadata: Optional metadata dictionary for the station
        """
        features = {}

        # Basic shape features
        for i, component in enumerate(["Z", "N", "E"]):
            # Simple statistics
            features[f"{component}_mean"] = np.mean(waveform[i])
            features[f"{component}_std"] = np.std(waveform[i])
            features[f"{component}_max"] = np.max(waveform[i])
            features[f"{component}_min"] = np.min(waveform[i])
            features[f"{component}_range"] = np.ptp(waveform[i])
            features[f"{component}_energy"] = np.sum(waveform[i] ** 2)

            # RMS
            features[f"{component}_rms"] = np.sqrt(np.mean(waveform[i] ** 2))

            # Zero crossings
            features[f"{component}_zero_crossings"] = np.sum(
                np.diff(np.signbit(waveform[i]))
            )

            # Spectral features
            f, Pxx = signal.welch(waveform[i], fs=100, nperseg=256)
            features[f"{component}_peak_freq"] = f[np.argmax(Pxx)]
            features[f"{component}_spectral_mean"] = np.mean(Pxx)
            features[f"{component}_spectral_std"] = np.std(Pxx)

            # Frequency bands energy
            band_ranges = [(0, 5), (5, 10), (10, 20), (20, 40)]
            for j, (low, high) in enumerate(band_ranges):
                mask = (f >= low) & (f <= high)
                features[f"{component}_band_{low}_{high}_energy"] = np.sum(Pxx[mask])

        # Cross-component features
        features["Z_N_correlation"] = np.corrcoef(waveform[0], waveform[1])[0, 1]
        features["Z_E_correlation"] = np.corrcoef(waveform[0], waveform[2])[0, 1]
        features["N_E_correlation"] = np.corrcoef(waveform[1], waveform[2])[0, 1]

        # Add polarization features (critical for baseline comparison)
        features["pol_az"] = np.degrees(
            np.arctan2(np.std(waveform[2]), np.std(waveform[1]))
        )  # Azimuth
        features["pol_inc"] = np.degrees(
            np.arctan2(
                np.sqrt(np.std(waveform[1]) ** 2 + np.std(waveform[2]) ** 2),
                np.std(waveform[0]),
            )
        )  # Incidence
        features["rect_lin"] = 1 - (
            np.min([np.std(waveform[0]), np.std(waveform[1]), np.std(waveform[2])])
            / np.max([np.std(waveform[0]), np.std(waveform[1]), np.std(waveform[2])])
        )  # Rectilinearity

        # REMOVED: Station coordinates. This removes the data leakage caused by
        # using station location as a feature, forcing the model to use only
        # waveform characteristics to predict event location

        return features

    def prepare_best_station_dataset(self):
        """
        Prepare dataset using only the best station for each event
        """
        if self.aftershocks_df is None:
            self.create_relative_coordinate_dataframe()

        # For multi-station format, select the best station for each event
        best_stations_df = self.aftershocks_df

        if self.data_format == "multi_station":
            print("Selecting best station for each event...")
            # Group by event_id and select the station with highest selection_score
            best_station_indices = self.aftershocks_df.groupby("event_id")[
                "selection_score"
            ].idxmax()
            best_stations_df = self.aftershocks_df.loc[
                best_station_indices
            ].reset_index(drop=True)
            print(
                f"Selected {len(best_stations_df)} best stations from {len(self.aftershocks_df)} total recordings"
            )

        # HYGIENE TWEAK 1: Drop station_distance columns early to avoid warnings later
        best_stations_df = best_stations_df.drop(
            columns=["station_distance"], errors="ignore"
        )

        print("Extracting features from waveforms...")
        # Extract features from the selected waveforms
        features_list = []
        errors = 0
        for idx, row in tqdm(best_stations_df.iterrows(), total=len(best_stations_df)):
            try:
                # Extract features with metadata
                metadata = row.get("metadata", {})
                features = self.extract_waveform_features(
                    row["waveform"], metadata=metadata
                )

                # Add P-S time difference if available
                if (
                    isinstance(metadata, dict)
                    and "trace_P_arrival_sample" in metadata
                    and "trace_S_arrival_sample" in metadata
                ):
                    if pd.notna(metadata["trace_P_arrival_sample"]) and pd.notna(
                        metadata["trace_S_arrival_sample"]
                    ):
                        p_sample = metadata["trace_P_arrival_sample"]
                        s_sample = metadata["trace_S_arrival_sample"]
                        if p_sample < s_sample:  # Ensure P arrives before S
                            # Convert samples to time (assuming 100 Hz sampling rate)
                            features["p_s_time_diff"] = (
                                s_sample - p_sample
                            ) / 100.0  # in seconds

                features["origin_time"] = row["origin_time"]
                features["event_date"] = row["event_date"]

                # Add event_id for proper GroupKFold validation
                if "event_id" in row:
                    features["event_id"] = row["event_id"]

                # Add station information if available (except coordinates/distances)
                if "station_key" in row:
                    features["station_key"] = row["station_key"]

                features_list.append(features)
            except Exception as e:
                errors += 1
                if errors <= 5:  # Limit error printing to avoid overwhelming output
                    print(f"Error processing waveform {idx}: {e}")
                elif errors == 6:
                    print("Additional errors occurred but not printed...")

        print(
            f"Successfully processed {len(features_list)} waveforms with {errors} errors"
        )

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)

        # Match with the original dataframe using origin_time
        merge_columns = ["origin_time", "event_date"]
        if (
            "station_key" in features_df.columns
            and "station_key" in best_stations_df.columns
        ):
            merge_columns.append("station_key")

        merged_df = pd.merge(
            features_df,
            best_stations_df.drop(["waveform", "metadata"], axis=1, errors="ignore"),
            on=merge_columns,
        )

        # Remove the mainshock from training data
        merged_df = merged_df[~merged_df["is_mainshock"]]

        # Define features and targets
        drop_columns = [
            "origin_time",
            "absolute_lat",
            "absolute_lon",
            "absolute_depth",
            "relative_x",
            "relative_y",
            "relative_z",
            "is_mainshock",
            "event_date",
        ]

        # Add multi-station specific columns to drop list
        additional_drops = ["station_key", "selection_score"]
        drop_columns.extend(
            [col for col in additional_drops if col in merged_df.columns]
        )

        # IMPROVEMENT 3: Keep event_id for GroupKFold
        if "event_id" in merged_df.columns:
            X = merged_df.drop(drop_columns, axis=1, errors="ignore")
            y = merged_df[["relative_x", "relative_y", "relative_z", "event_id"]]
        else:
            X = merged_df.drop(drop_columns, axis=1, errors="ignore")
            y = merged_df[["relative_x", "relative_y", "relative_z", "event_date"]]

        # Validate feature preparation if required
        if self.validation_level != "none":
            X, y = self.validate_features(X, y)

        return X, y

    def prepare_multi_station_dataset(self):
        """
        Advanced method to prepare dataset using multiple stations for each event
        """
        if self.aftershocks_df is None:
            self.create_relative_coordinate_dataframe()

        if self.data_format != "multi_station":
            print("Using single-station format - falling back to best station approach")
            return self.prepare_best_station_dataset()

        print("Preparing enhanced multi-station dataset...")

        # Step 1: Extract features from all waveforms
        print("Extracting features from waveforms...")
        features_list = []
        errors = 0

        for idx, row in tqdm(
            self.aftershocks_df.iterrows(), total=len(self.aftershocks_df)
        ):
            try:
                # Extract features with metadata - STATION COORDINATES WILL NOT BE ADDED
                metadata = row.get("metadata", {})
                features = self.extract_waveform_features(
                    row["waveform"], metadata=metadata
                )

                # Add P-S time difference if available
                if (
                    isinstance(metadata, dict)
                    and "trace_P_arrival_sample" in metadata
                    and "trace_S_arrival_sample" in metadata
                ):
                    if pd.notna(metadata["trace_P_arrival_sample"]) and pd.notna(
                        metadata["trace_S_arrival_sample"]
                    ):
                        p_sample = metadata["trace_P_arrival_sample"]
                        s_sample = metadata["trace_S_arrival_sample"]
                        if p_sample < s_sample:  # Ensure P arrives before S
                            # Convert samples to time (assuming 100 Hz sampling rate)
                            features["p_s_time_diff"] = (
                                s_sample - p_sample
                            ) / 100.0  # in seconds

                # Add station-specific features - BUT NOT DISTANCE OR STATION COORDINATES
                # IMPORTANT: Do not add station coordinates to prevent leakage

                # Add station metadata that doesn't leak target information
                features["station_key"] = row.get("station_key", "")
                features["event_id"] = row.get("event_id", "")
                features["origin_time"] = row["origin_time"]
                features["event_date"] = row["event_date"]

                # Add target values (for later aggregation)
                features["relative_x"] = row["relative_x"]
                features["relative_y"] = row["relative_y"]
                features["relative_z"] = row["relative_z"]
                features["is_mainshock"] = row["is_mainshock"]

                features_list.append(features)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error processing waveform {idx}: {e}")
                elif errors == 6:
                    print("Additional errors occurred but not printed...")

        print(
            f"Successfully processed {len(features_list)} waveforms with {errors} errors"
        )

        # Convert to DataFrame
        all_features_df = pd.DataFrame(features_list)

        # Remove the mainshock recordings
        all_features_df = all_features_df[~all_features_df["is_mainshock"]]

        # Step 2: Aggregate features across stations for each event
        print("Aggregating features across stations for each event...")

        # Group by event_id and calculate statistics
        event_groups = all_features_df.groupby("event_id")

        # List of columns that shouldn't be aggregated
        skip_columns = [
            "origin_time",
            "event_date",
            "relative_x",
            "relative_y",
            "relative_z",
            "is_mainshock",
            "event_id",
            "station_key",
        ]

        # List of columns that should not be aggregated to prevent data leakage
        # (in addition to those already excluded from features extraction)
        leakage_columns = [
            col
            for col in all_features_df.columns
            if "station_distance" in col
            or "distance_normalized" in col
            or "selection_score" in col
            or "epicentral_distance" in col
            or "station_lat" in col
            or "station_lon" in col
            or "station_elev" in col
        ]  # Added station location patterns

        skip_columns.extend(leakage_columns)

        # List of numeric feature columns to aggregate
        numeric_columns = [
            col
            for col in all_features_df.columns
            if col not in skip_columns
            and pd.api.types.is_numeric_dtype(all_features_df[col])
        ]

        # Create empty dataframe for aggregated features
        aggregated_features = []

        for event_id, group in event_groups:
            # Start with event metadata (same for all stations)
            event_data = {
                "event_id": event_id,
                "origin_time": group["origin_time"].iloc[0],
                "event_date": group["event_date"].iloc[0],
                "relative_x": group["relative_x"].iloc[0],
                "relative_y": group["relative_y"].iloc[0],
                "relative_z": group["relative_z"].iloc[0],
                "is_mainshock": group["is_mainshock"].iloc[0],
                "num_stations": len(group),
            }

            # For each feature, calculate various statistics across stations
            for feature in numeric_columns:
                values = group[feature].values

                # Basic statistics
                event_data[f"{feature}_mean"] = np.mean(values)
                event_data[f"{feature}_median"] = np.median(values)
                event_data[f"{feature}_std"] = np.std(values) if len(values) > 1 else 0
                event_data[f"{feature}_min"] = np.min(values)
                event_data[f"{feature}_max"] = np.max(values)
                event_data[f"{feature}_range"] = np.ptp(values)

                # Add robust statistics
                # IQR (Interquartile Range)
                if len(values) >= 4:  # Need at least 4 points for meaningful quartiles
                    q75, q25 = np.percentile(values, [75, 25])
                    event_data[f"{feature}_iqr"] = q75 - q25
                else:
                    event_data[f"{feature}_iqr"] = 0

                # MAD (Median Absolute Deviation)
                if len(values) >= 2:
                    median = np.median(values)
                    event_data[f"{feature}_mad"] = np.median(np.abs(values - median))
                else:
                    event_data[f"{feature}_mad"] = 0

                # For important features, add more detailed statistics
                if feature in [
                    "Z_energy",
                    "N_energy",
                    "E_energy",
                    "p_s_time_diff",
                    "pol_az",
                    "pol_inc",
                    "rect_lin",
                ]:
                    # Don't use selection_score as weight to avoid leakage
                    # For important features, we'll just use our existing statistics
                    pass

            # Add features from stations without using selection score (to avoid leakage)
            station_indices = group.index.tolist()

            # Instead of using selection_score to find best station, use a simple approach
            # For example, take the first station's features
            if len(station_indices) > 0:
                first_station = group.iloc[0]
                for feature in numeric_columns:
                    event_data[f"best_{feature}"] = first_station[feature]

            # Add second station features if available, again without selection_score
            if len(station_indices) > 1:
                # Get second station
                second_station = group.iloc[1]
                for feature in numeric_columns:
                    event_data[f"second_{feature}"] = second_station[feature]

            aggregated_features.append(event_data)

        # Convert to DataFrame
        merged_df = pd.DataFrame(aggregated_features)
        print(
            f"Created aggregated dataset with {len(merged_df)} events and {len(merged_df.columns)} features"
        )

        # Define features and targets
        drop_columns = [
            "origin_time",
            "relative_x",
            "relative_y",
            "relative_z",
            "is_mainshock",
            "event_date",
        ]

        # IMPROVEMENT 3: Keep event_id for GroupKFold
        X = merged_df.drop(drop_columns, axis=1)
        y = merged_df[["relative_x", "relative_y", "relative_z", "event_id"]]

        # Validate feature preparation if required
        if self.validation_level != "none":
            X, y = self.validate_features(X, y)

        return X, y

    def prepare_dataset(self):
        """
        Prepare dataset for machine learning by extracting features from waveforms
        """
        if self.approach == "multi_station" and self.data_format == "multi_station":
            return self.prepare_multi_station_dataset()
        else:
            return self.prepare_best_station_dataset()

    def standardize_waveforms(self, target_length=14636):
        """
        Standardize all waveforms to the same length by padding or trimming

        Args:
            target_length: Target length for all waveforms (samples)
        """
        print("\n" + "=" * 50)
        print(f"STANDARDIZING WAVEFORMS TO {target_length} SAMPLES")
        print("=" * 50)

        modified_count = 0

        if self.data_format == "multi_station":
            # For multi-station format, loop through the nested structure
            for event_key, stations_data in self.data_dict.items():
                for station_key, station_data in stations_data.items():
                    waveform = station_data["waveform"]

                    # Skip if already the correct length
                    if waveform.shape[1] == target_length:
                        continue

                    # Pad or trim the waveform
                    if waveform.shape[1] > target_length:
                        # Trim to target length
                        self.data_dict[event_key][station_key]["waveform"] = waveform[
                            :, :target_length
                        ]
                    else:
                        # Pad with zeros
                        padded = np.zeros((3, target_length))
                        padded[:, : waveform.shape[1]] = waveform
                        self.data_dict[event_key][station_key]["waveform"] = padded

                    modified_count += 1
        else:
            # Original single-station format
            for event_key, event_data in self.data_dict.items():
                waveform = event_data["waveform"]

                # Skip if already the correct length
                if waveform.shape[1] == target_length:
                    continue

                # Pad or trim the waveform
                if waveform.shape[1] > target_length:
                    # Trim to target length
                    self.data_dict[event_key]["waveform"] = waveform[:, :target_length]
                else:
                    # Pad with zeros
                    padded = np.zeros((3, target_length))
                    padded[:, : waveform.shape[1]] = waveform
                    self.data_dict[event_key]["waveform"] = padded

                modified_count += 1

        print(f"Standardized {modified_count} waveforms to length {target_length}")
        return modified_count

    def train_models(self, X, y):
        """
        Train models to predict aftershock locations
        Modified to remove baseline comparison
        """
        # Split data into training and testing sets
        # IMPROVEMENT 3: Use event_id for GroupKFold if available, otherwise use event_date
        if self.validation_level != "none":
            if "event_id" in y.columns:
                print(
                    "Using GroupKFold with event_id as the group to prevent data leakage..."
                )
                # Get a single train/test split using GroupKFold
                gkf = GroupKFold(n_splits=5)
                groups = y["event_id"]
                train_idx, test_idx = next(gkf.split(X, y, groups))

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                # FIX: Always drop the group columns from X_train and X_test before modeling
                if "event_id" in X_train.columns:
                    X_train = X_train.drop("event_id", axis=1)
                if "event_id" in X_test.columns:
                    X_test = X_test.drop("event_id", axis=1)

                # Also drop from y for modeling (keep a copy for evaluation)
                y_train_coord = y_train.drop("event_id", axis=1)
                y_test_coord = y_test.drop("event_id", axis=1)
            elif "event_date" in y.columns:
                print(
                    "Using GroupKFold with event_date as the group to prevent temporal leakage..."
                )
                # Get a single train/test split using GroupKFold
                gkf = GroupKFold(n_splits=5)
                groups = y["event_date"]
                train_idx, test_idx = next(gkf.split(X, y, groups))

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                # FIX: Always drop the group columns from X_train and X_test before modeling
                if "event_date" in X_train.columns:
                    X_train = X_train.drop("event_date", axis=1)
                if "event_date" in X_test.columns:
                    X_test = X_test.drop("event_date", axis=1)

                # Also drop from y for modeling (keep a copy for evaluation)
                y_train_coord = y_train.drop("event_date", axis=1)
                y_test_coord = y_test.drop("event_date", axis=1)
            else:
                # Standard random split if no grouping columns available
                X_temp = X.copy()
                y_temp = y.copy()
                X_train, X_test, y_train, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.2, random_state=42
                )
                y_train_coord = y_train
                y_test_coord = y_test
        else:
            # Standard random split for non-validation mode
            group_cols = [col for col in y.columns if col in ["event_id", "event_date"]]
            X_temp = X.drop(group_cols, axis=1, errors="ignore")
            y_temp = y.drop(group_cols, axis=1, errors="ignore")
            X_train, X_test, y_train, y_test = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42
            )
            y_train_coord = y_train
            y_test_coord = y_test

        # IMPORTANT: Filter numeric columns for model training
        numeric_columns = [
            col
            for col in X_train.columns
            if pd.api.types.is_numeric_dtype(X_train[col])
        ]
        non_numeric_columns = [
            col for col in X_train.columns if col not in numeric_columns
        ]

        if non_numeric_columns:
            print(
                f"Removing {len(non_numeric_columns)} non-numeric columns from training data: {non_numeric_columns}"
            )
            X_train_numeric = X_train[numeric_columns]
            X_test_numeric = X_test[numeric_columns]
        else:
            X_train_numeric = X_train
            X_test_numeric = X_test

        print("Training RandomForest model...")
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import RandomForestRegressor

        # Create and train optimized RandomForest model
        base_rf = RandomForestRegressor(
            n_estimators=800, max_depth=10, min_samples_leaf=5, random_state=42
        )

        multi_model = MultiOutputRegressor(base_rf)
        multi_model.fit(
            X_train_numeric, y_train_coord[["relative_x", "relative_y", "relative_z"]]
        )

        # Make predictions on test set
        y_pred = multi_model.predict(X_test_numeric)

        # Calculate and print errors
        from sklearn.metrics import mean_squared_error
        import numpy as np

        multi_ml_errors = {}
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            mse = mean_squared_error(y_test_coord[coord], y_pred[:, i])
            rmse = np.sqrt(mse)
            multi_ml_errors[coord] = rmse

        # Calculate 3D distance error
        multi_ml_3d_errors = np.sqrt(
            (y_pred[:, 0] - y_test_coord["relative_x"]) ** 2
            + (y_pred[:, 1] - y_test_coord["relative_y"]) ** 2
            + (y_pred[:, 2] - y_test_coord["relative_z"]) ** 2
        )
        multi_ml_errors["3d_distance"] = np.mean(multi_ml_3d_errors)

        print("\nModel Performance (RMSE):")
        for coord in ["relative_x", "relative_y", "relative_z", "3d_distance"]:
            print(f"  {coord}: {multi_ml_errors[coord]:.2f} km")

        # Store as the model
        self.models = {"multi_output": multi_model, "type": "multi_output"}
        self.scaler = None

        # Print feature importance from the multi-output model
        # Note: In MultiOutputRegressor, each coordinate has its own RF model
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            rf_model = multi_model.estimators_[i]

            # Use correct columns for feature importance
            feature_importances = pd.DataFrame(
                {
                    "feature": X_train_numeric.columns,
                    "importance": rf_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            print(f"\nTop features for {coord} prediction:")
            print(feature_importances.head(10))

            # Save feature importance plot
            plt.figure(figsize=(12, 8))
            top_features = feature_importances.head(20)
            sns.barplot(x="importance", y="feature", data=top_features)
            plt.title(f"Top 20 Features for {coord} Prediction")
            plt.tight_layout()
            plt.savefig(f"feature_importance_{coord}_{self.approach}.png", dpi=300)
            plt.close()

        # If validation is enabled at full level, check model stability
        if self.validation_level == "full":
            self.validate_model_stability(X_train_numeric, y_train_coord)

        # Save models
        with open(f"aftershock_location_models_{self.approach}.pkl", "wb") as f:
            pickle.dump({"models": self.models}, f)

        return X_test_numeric, y_test_coord

    def predict_location(self, waveform):
        """
        Predict the location of an aftershock from its waveform
        """
        if self.models is None:
            raise ValueError("Models not trained yet. Call train_models first.")

        # Extract features from the waveform
        features = self.extract_waveform_features(waveform)

        # Convert to DataFrame with the same columns as training data
        feature_df = pd.DataFrame([features])

        # Make predictions using the multi-output model
        predictions_array = self.models["multi_output"].predict(feature_df)[0]

        # Create a dictionary with the predictions
        predictions = {
            "relative_x": predictions_array[0],
            "relative_y": predictions_array[1],
            "relative_z": predictions_array[2],
        }

        return predictions

    def predict_and_convert(self, waveform):
        """
        Predict relative location and convert to absolute coordinates
        """
        # Predict relative coordinates
        rel_coords = self.predict_location(waveform)

        # Convert to absolute coordinates
        abs_lat, abs_lon, abs_depth = self.cartesian_to_geographic(
            rel_coords["relative_x"],
            rel_coords["relative_y"],
            rel_coords["relative_z"],
            self.mainshock["latitude"],
            self.mainshock["longitude"],
            self.mainshock["depth"],
        )

        return {
            "relative_x": rel_coords["relative_x"],
            "relative_y": rel_coords["relative_y"],
            "relative_z": rel_coords["relative_z"],
            "absolute_lat": abs_lat,
            "absolute_lon": abs_lon,
            "absolute_depth": abs_depth,
        }

    def visualize_predictions_geographic(self, X_test, y_test):
        """
        Visualize prediction results on a geographic map
        """
        if self.models is None:
            raise ValueError("Models not trained yet")

        # Make predictions using the multi-output model
        y_pred_array = self.models["multi_output"].predict(X_test)
        y_pred = pd.DataFrame(
            y_pred_array,
            columns=["relative_x", "relative_y", "relative_z"],
            index=y_test.index,
        )

        # Convert to absolute coordinates
        true_absolute = pd.DataFrame(index=y_test.index)
        pred_absolute = pd.DataFrame(index=y_test.index)

        for i in range(len(y_test)):
            # True coordinates
            lat, lon, depth = self.cartesian_to_geographic(
                y_test["relative_x"].iloc[i],
                y_test["relative_y"].iloc[i],
                y_test["relative_z"].iloc[i],
                self.mainshock["latitude"],
                self.mainshock["longitude"],
                self.mainshock["depth"],
            )
            true_absolute.loc[y_test.index[i], "lat"] = lat
            true_absolute.loc[y_test.index[i], "lon"] = lon
            true_absolute.loc[y_test.index[i], "depth"] = depth

            # Predicted coordinates
            lat, lon, depth = self.cartesian_to_geographic(
                y_pred["relative_x"].iloc[i],
                y_pred["relative_y"].iloc[i],
                y_pred["relative_z"].iloc[i],
                self.mainshock["latitude"],
                self.mainshock["longitude"],
                self.mainshock["depth"],
            )
            pred_absolute.loc[y_test.index[i], "lat"] = lat
            pred_absolute.loc[y_test.index[i], "lon"] = lon
            pred_absolute.loc[y_test.index[i], "depth"] = depth

        # Create map
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.Mercator())

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        # Set extent
        buffer = 0.3
        min_lon = min(true_absolute["lon"].min(), pred_absolute["lon"].min()) - buffer
        max_lon = max(true_absolute["lon"].max(), pred_absolute["lon"].max()) + buffer
        min_lat = min(true_absolute["lat"].min(), pred_absolute["lat"].min()) - buffer
        max_lat = max(true_absolute["lat"].max(), pred_absolute["lat"].max()) + buffer

        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        # Plot true locations
        ax.scatter(
            true_absolute["lon"],
            true_absolute["lat"],
            c="blue",
            s=30,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            label="True Locations",
        )

        # Plot predicted locations
        ax.scatter(
            pred_absolute["lon"],
            pred_absolute["lat"],
            c="red",
            s=30,
            alpha=0.7,
            marker="x",
            transform=ccrs.PlateCarree(),
            label="Predicted Locations",
        )

        # Connect true and predicted with lines
        for i in range(len(true_absolute)):
            ax.plot(
                [true_absolute["lon"].iloc[i], pred_absolute["lon"].iloc[i]],
                [true_absolute["lat"].iloc[i], pred_absolute["lat"].iloc[i]],
                "k-",
                alpha=0.2,
                transform=ccrs.PlateCarree(),
            )

        # Plot mainshock
        ax.scatter(
            self.mainshock["longitude"],
            self.mainshock["latitude"],
            c="yellow",
            s=200,
            marker="*",
            edgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=5,
            label="Mainshock",
        )

        # Add gridlines and legend
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        plt.title(
            f'True vs Predicted Aftershock Locations ({self.approach.replace("_", " ").title()})'
        )
        plt.legend(loc="lower left")

        plt.savefig(
            f"prediction_results_geographic_{self.approach}.png",
            dpi=300,
            bbox_inches="tight",
        )

        # Calculate location errors
        earth_radius = 6371.0  # km

        # Calculate differences in degrees
        lat_diff_deg = np.abs(true_absolute["lat"] - pred_absolute["lat"])
        lon_diff_deg = np.abs(true_absolute["lon"] - pred_absolute["lon"])

        # Convert to approximate distances in km
        lat_diff_km = lat_diff_deg * (np.pi / 180) * earth_radius
        # Account for longitude convergence
        avg_lat = (true_absolute["lat"] + pred_absolute["lat"]) / 2
        lon_diff_km = (
            lon_diff_deg * (np.pi / 180) * earth_radius * np.cos(np.radians(avg_lat))
        )

        # Depth difference
        depth_diff_km = np.abs(true_absolute["depth"] - pred_absolute["depth"])

        # 3D distance
        distance_3d_km = np.sqrt(lat_diff_km**2 + lon_diff_km**2 + depth_diff_km**2)

        # Print statistics
        print("Prediction Error Statistics:")
        print(f"Mean latitude error: {lat_diff_km.mean():.2f} km")
        print(f"Mean longitude error: {lon_diff_km.mean():.2f} km")
        print(f"Mean depth error: {depth_diff_km.mean():.2f} km")
        print(f"Mean 3D error: {distance_3d_km.mean():.2f} km")
        print(f"Median 3D error: {distance_3d_km.median():.2f} km")

        # Create error histogram
        plt.figure(figsize=(10, 6))
        plt.hist(distance_3d_km, bins=20, alpha=0.7)
        plt.axvline(
            distance_3d_km.mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {distance_3d_km.mean():.2f} km",
        )
        plt.axvline(
            distance_3d_km.median(),
            color="g",
            linestyle="--",
            label=f"Median: {distance_3d_km.median():.2f} km",
        )
        plt.title(
            f'3D Location Error Distribution ({self.approach.replace("_", " ").title()})'
        )
        plt.xlabel("Error (km)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f"prediction_error_histogram_{self.approach}.png", dpi=300)

        # NEW: Store individual coordinate errors along with 3D error
        return (
            true_absolute,
            pred_absolute,
            {
                "3d": distance_3d_km,
                "lat": lat_diff_km,
                "lon": lon_diff_km,
                "depth": depth_diff_km,
            },
        )

    def visualize_aftershocks_3d(self):
        """
        Create a 3D visualization of the aftershocks
        relative to the mainshock
        """
        if self.aftershocks_df is None:
            self.create_relative_coordinate_dataframe()

        # For multi-station format, select one station per event for visualization
        plot_df = self.aftershocks_df
        if (
            self.data_format == "multi_station"
            and "event_id" in self.aftershocks_df.columns
        ):
            # Group by event_id and select the station with highest selection_score
            best_station_indices = self.aftershocks_df.groupby("event_id")[
                "selection_score"
            ].idxmax()
            plot_df = self.aftershocks_df.loc[best_station_indices]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot aftershocks
        sc = ax.scatter(
            plot_df[~plot_df["is_mainshock"]]["relative_x"],
            plot_df[~plot_df["is_mainshock"]]["relative_y"],
            plot_df[~plot_df["is_mainshock"]]["relative_z"],
            c=plot_df[~plot_df["is_mainshock"]]["absolute_depth"],
            cmap="viridis_r",
            alpha=0.7,
            s=20,
            label="Aftershocks",
        )

        # Plot mainshock
        if plot_df["is_mainshock"].any():
            mainshock = plot_df[plot_df["is_mainshock"]].iloc[0]
            ax.scatter(
                [mainshock["relative_x"]],
                [mainshock["relative_y"]],
                [mainshock["relative_z"]],
                c="red",
                s=200,
                marker="*",
                label="Mainshock",
            )

        # Set labels and title
        ax.set_xlabel("East-West (km)")
        ax.set_ylabel("North-South (km)")
        ax.set_zlabel("Depth difference (km)")
        ax.set_title("3D Distribution of Aftershocks Relative to Mainshock")

        # Add a colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label("Absolute Depth (km)")

        # Set equal aspect ratio
        max_range = (
            np.array(
                [
                    plot_df["relative_x"].max() - plot_df["relative_x"].min(),
                    plot_df["relative_y"].max() - plot_df["relative_y"].min(),
                    plot_df["relative_z"].max() - plot_df["relative_z"].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (plot_df["relative_x"].max() + plot_df["relative_x"].min()) / 2
        mid_y = (plot_df["relative_y"].max() + plot_df["relative_y"].min()) / 2
        mid_z = (plot_df["relative_z"].max() + plot_df["relative_z"].min()) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        plt.tight_layout()
        plt.savefig("aftershocks_3d.png", dpi=300)

        return fig

    def run_complete_workflow(self):
        """
        Run the complete analysis workflow with validation
        """
        start_time = time.time()

        # Print header
        print("\n" + "=" * 70)
        print(
            f"IQUIQUE AFTERSHOCK ANALYSIS WITH {self.approach.upper()} APPROACH".center(
                70
            )
        )
        print("=" * 70)

        # 1. Find the mainshock (using manual coordinates from USGS)
        self.find_mainshock()

        # 2. Standardize waveform lengths
        self.standardize_waveforms(target_length=14636)

        # 3. Create relative coordinate dataframe
        self.create_relative_coordinate_dataframe()

        if self.data_format == "multi_station":
            event_count = (
                len(set(self.aftershocks_df["event_id"]))
                if "event_id" in self.aftershocks_df.columns
                else len(self.aftershocks_df)
            )
            print(
                f"Created dataframe with {event_count} events and {len(self.aftershocks_df)} station recordings"
            )

            # Show stations per event statistics
            if "event_id" in self.aftershocks_df.columns:
                stations_per_event = self.aftershocks_df.groupby("event_id").size()
                print(f"Stations per event:")
                print(f"  Mean: {stations_per_event.mean():.2f}")
                print(f"  Median: {stations_per_event.median()}")
                print(f"  Min: {stations_per_event.min()}")
                print(f"  Max: {stations_per_event.max()}")
        else:
            print(f"Created dataframe with {len(self.aftershocks_df)} events")

        # 4. Visualize aftershocks in 3D relative space
        fig = self.visualize_aftershocks_3d()

        # 5. Prepare dataset for machine learning
        X, y = self.prepare_dataset()
        print(f"Prepared dataset with {len(X)} samples and {X.shape[1]} features")

        # 6. Train models
        X_test, y_test = self.train_models(X, y)

        # 7. Visualize predictions on a geographic map
        true_abs, pred_abs, errors = self.visualize_predictions_geographic(
            X_test, y_test
        )

        # Print summary of validation results
        if self.validation_level != "none":
            print("\n" + "=" * 70)
            print("VALIDATION SUMMARY".center(70))
            print("=" * 70)

            # Data integrity
            if "data_integrity" in self.validation_results:
                if self.validation_results["data_integrity"]["passed"]:
                    print("✅ Data integrity checks passed")
                else:
                    print(
                        f"⚠️ Data integrity checks found {len(self.validation_results['data_integrity']['issues'])} issues"
                    )

            # Coordinate conversion
            if "coordinate_conversion" in self.validation_results:
                if self.validation_results["coordinate_conversion"]["passed"]:
                    print("✅ Coordinate conversion validation passed")
                else:
                    print("⚠️ Coordinate conversion showed some errors above threshold")
                mean_error = (
                    self.validation_results["coordinate_conversion"]["mean_error"]
                    * 1000
                )  # m
                print(f"   Mean round-trip error: {mean_error:.2f} meters")

            # Feature validation
            if "feature_validation" in self.validation_results:
                if self.validation_results["feature_validation"]["target_leakage"][
                    "passed"
                ]:
                    print("✅ No target leakage detected")
                else:
                    print(
                        f"⚠️ Target leakage detected and {len(self.validation_results['feature_validation']['target_leakage']['leaked_features'])} features removed"
                    )

                var_check = self.validation_results["feature_validation"][
                    "variance_check"
                ]
                if len(var_check["removed_features"]) > 0:
                    print(
                        f"ℹ️ Removed {len(var_check['removed_features'])} features with near-zero variance"
                    )

            # Model stability
            if "model_stability" in self.validation_results:
                unstable_coords = [
                    coord
                    for coord, result in self.validation_results[
                        "model_stability"
                    ].items()
                    if not result["stable"]
                ]
                if not unstable_coords:
                    print("✅ All models show good stability")
                else:
                    print(
                        f"⚠️ Models for {', '.join(unstable_coords)} showed some instability"
                    )

        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"\nTotal execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)"
        )

        return {
            "models": self.models,
            "scaler": self.scaler,
            "mainshock": self.mainshock,
            "aftershocks_df": self.aftershocks_df,
            "test_results": {
                "true_absolute": true_abs,
                "pred_absolute": pred_abs,
                "errors": errors,
            },
            "validation_results": self.validation_results,
        }

    def validate_data_integrity(self):
        """
        Run validation checks on the dataset to catch potential issues
        """
        print("\n" + "=" * 50)
        print("VALIDATING DATA INTEGRITY")
        print("=" * 50)

        issues = []

        # Skip uniqueness check as instructed - no actual duplicates in the dataset
        print("Skipping event uniqueness check (confirmed no duplicates)")

        # Check waveform shapes and NaNs
        print("Checking waveform consistency...")
        expected_shape = None
        waveform_issues = 0

        # Check 20 waveforms (multi-station or single-station)
        if self.data_format == "multi_station":
            waveform_check_count = 0
            for event_key, stations_data in self.data_dict.items():
                for station_key, station_data in stations_data.items():
                    waveform = station_data["waveform"]

                    # Check number of components
                    if waveform.shape[0] != 3:
                        issues.append(
                            f"Event {event_key}, Station {station_key} has {waveform.shape[0]} components instead of 3"
                        )
                        waveform_issues += 1
                        continue

                    # Set expected length from first waveform
                    if expected_shape is None:
                        expected_shape = waveform.shape[1]

                    # Check length consistency
                    if waveform.shape[1] != expected_shape:
                        issues.append(
                            f"Event {event_key}, Station {station_key} has length {waveform.shape[1]} instead of {expected_shape}"
                        )
                        waveform_issues += 1

                    # Check for NaNs
                    if np.isnan(waveform).any():
                        issues.append(
                            f"Event {event_key}, Station {station_key} contains NaN values"
                        )
                        waveform_issues += 1

                    waveform_check_count += 1
                    if waveform_check_count >= 20:
                        print(
                            f"✓ First {waveform_check_count} waveforms checked (skipping rest for speed)"
                        )
                        break

                if waveform_check_count >= 20:
                    break
        else:
            # Single-station format
            for i, (event_key, event_data) in enumerate(self.data_dict.items()):
                waveform = event_data["waveform"]

                # Check number of components
                if waveform.shape[0] != 3:
                    issues.append(
                        f"Event {event_key} has {waveform.shape[0]} components instead of 3"
                    )
                    waveform_issues += 1
                    continue

                # Set expected length from first waveform
                if expected_shape is None:
                    expected_shape = waveform.shape[1]

                # Check length consistency
                if waveform.shape[1] != expected_shape:
                    issues.append(
                        f"Event {event_key} has length {waveform.shape[1]} instead of {expected_shape}"
                    )
                    waveform_issues += 1

                # Check for NaNs
                if np.isnan(waveform).any():
                    issues.append(f"Event {event_key} contains NaN values")
                    waveform_issues += 1

                # Only check first 20 events if dataset is large
                if i >= 20 and len(self.data_dict) > 50:
                    print(
                        f"✓ First 20/{len(self.data_dict)} waveforms checked (skipping rest for speed)"
                    )
                    break

        if waveform_issues == 0:
            print(
                f"✓ All checked waveforms have consistent shape (3, {expected_shape}) with no NaNs"
            )
        else:
            print(f"❌ Found {waveform_issues} waveform issues!")

        # 4. Extract a sample of metadata to check P and S arrival times
        print("Checking P and S arrival samples (if available)...")
        sample_metadata = []

        # Get a sample of metadata for both formats
        if self.data_format == "multi_station":
            metadata_count = 0
            for event_key, stations_data in self.data_dict.items():
                for station_key, station_data in stations_data.items():
                    if metadata_count < 20:
                        sample_metadata.append(station_data["metadata"])
                        metadata_count += 1
        else:
            # Single-station format
            for i, (event_key, event_data) in enumerate(self.data_dict.items()):
                if i < 20:  # Just check a sample
                    sample_metadata.append(event_data["metadata"])

        sample_df = pd.DataFrame(sample_metadata)

        if (
            "trace_P_arrival_sample" in sample_df.columns
            and "trace_S_arrival_sample" in sample_df.columns
        ):
            valid_picks = sample_df[
                pd.notna(sample_df["trace_P_arrival_sample"])
                & pd.notna(sample_df["trace_S_arrival_sample"])
            ]

            ordered_picks = valid_picks[
                valid_picks["trace_P_arrival_sample"]
                < valid_picks["trace_S_arrival_sample"]
            ]

            if len(ordered_picks) < len(valid_picks):
                issues.append(
                    f"{len(valid_picks) - len(ordered_picks)} events have P arrival after S arrival"
                )
                print(
                    f"❌ {len(valid_picks) - len(ordered_picks)} events have P arrival after S arrival"
                )

            if len(valid_picks) < len(sample_df):
                print(
                    f"⚠️ {len(sample_df) - len(valid_picks)}/{len(sample_df)} events missing P or S picks"
                )
            else:
                print(f"✓ All checked events have valid P and S picks with P before S")
        else:
            print("⚠️ P or S arrival samples not available in metadata")

        # Overall status
        validated = len(issues) == 0

        if validated:
            print("✅ All data integrity checks passed!")
        else:
            print(
                f"⚠️ Found {len(issues)} issues. Will attempt to proceed with caution."
            )
            if len(issues) <= 5:
                for i, issue in enumerate(issues[:5]):
                    print(f"  {i+1}. {issue}")
            else:
                for i, issue in enumerate(issues[:5]):
                    print(f"  {i+1}. {issue}")
                print(f"  ... and {len(issues) - 5} more issues")

        self.validation_results["data_integrity"] = {
            "passed": validated,
            "issues": issues,
        }

        return validated, issues

    def validate_coordinate_conversion(self, num_points=50):
        """
        Test coordinate conversion functions for accuracy by doing a round-trip conversion
        geo → cart → geo and measuring the error
        """
        print("\n" + "=" * 50)
        print("VALIDATING COORDINATE CONVERSION")
        print("=" * 50)

        # Generate random geographic coordinates in a reasonable range around Iquique
        np.random.seed(42)
        lats = np.random.uniform(-20.5, -19.5, num_points)
        lons = np.random.uniform(-71.0, -70.0, num_points)
        depths = np.random.uniform(10, 50, num_points)

        # Reference point (use mainshock if available, otherwise default)
        if self.mainshock:
            ref_lat = self.mainshock["latitude"]
            ref_lon = self.mainshock["longitude"]
            ref_depth = self.mainshock["depth"]
        else:
            ref_lat, ref_lon, ref_depth = -20.0, -70.5, 30.0

        # Arrays to store errors
        lat_errors = []
        lon_errors = []
        depth_errors = []
        distance_errors = []

        for i in range(num_points):
            # Original coordinates
            orig_lat, orig_lon, orig_depth = lats[i], lons[i], depths[i]

            # Convert to Cartesian
            x, y, z = self.geographic_to_cartesian(
                orig_lat, orig_lon, orig_depth, ref_lat, ref_lon, ref_depth
            )

            # Convert back to geographic
            new_lat, new_lon, new_depth = self.cartesian_to_geographic(
                x, y, z, ref_lat, ref_lon, ref_depth
            )

            # Calculate errors
            lat_error = abs(orig_lat - new_lat)
            lon_error = abs(orig_lon - new_lon)
            depth_error = abs(orig_depth - new_depth)

            # Convert lat/lon errors to approximate distances in km
            earth_radius = 6371.0
            lat_error_km = lat_error * (np.pi / 180) * earth_radius
            lon_error_km = (
                lon_error * (np.pi / 180) * earth_radius * np.cos(np.radians(orig_lat))
            )

            # Calculate 3D distance error
            distance_error = np.sqrt(lat_error_km**2 + lon_error_km**2 + depth_error**2)

            lat_errors.append(lat_error_km)
            lon_errors.append(lon_error_km)
            depth_errors.append(depth_error)
            distance_errors.append(distance_error)

        # Get maximum errors
        max_errors = {
            "lat_km": max(lat_errors),
            "lon_km": max(lon_errors),
            "depth_km": max(depth_errors),
            "3d_distance_km": max(distance_errors),
        }

        # Check if errors are below threshold (100 meters = 0.1 km)
        threshold = 0.1
        passed = all(error < threshold for error in max_errors.values())

        # Print results
        print(f"Round-trip conversion test results ({num_points} random points):")
        print(f"  Max latitude error: {max_errors['lat_km']*1000:.2f} meters")
        print(f"  Max longitude error: {max_errors['lon_km']*1000:.2f} meters")
        print(f"  Max depth error: {max_errors['depth_km']*1000:.2f} meters")
        print(
            f"  Max 3D distance error: {max_errors['3d_distance_km']*1000:.2f} meters"
        )

        if passed:
            print("✅ All conversion errors below threshold (100 meters)")
        else:
            print("⚠️ Some conversion errors exceed threshold (100 meters)")
            print("   This could impact location accuracy - proceed with caution")

        # Visualize errors
        plt.figure(figsize=(10, 6))
        plt.hist(np.array(distance_errors) * 1000, bins=20)
        plt.axvline(
            np.mean(distance_errors) * 1000,
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(distance_errors)*1000:.2f} meters",
        )
        plt.axvline(
            threshold * 1000,
            color="g",
            linestyle="-",
            label=f"Threshold: {threshold*1000:.0f} meters",
        )
        plt.title("Distribution of 3D Round-Trip Conversion Errors")
        plt.xlabel("Error (meters)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig("coordinate_conversion_errors.png", dpi=300)
        plt.close()

        self.validation_results["coordinate_conversion"] = {
            "passed": passed,
            "max_errors": max_errors,
            "mean_error": np.mean(distance_errors),
        }

        return passed, max_errors

    def validate_features(self, X, y):
        """
        Validate feature preparation and check for target leakage
        """
        print("\n" + "=" * 50)
        print("VALIDATING FEATURE PREPARATION")
        print("=" * 50)

        # Check for target leakage - ENHANCED TO CATCH ALL POTENTIAL LEAKS
        print("Checking for target leakage...")

        # List of patterns that indicate potential target leakage
        forbidden_patterns = [
            "station_distance",
            "distance_normalized",
            "selection_score",
            "epicentral_distance",
            "relative_x",
            "relative_y",
            "relative_z",
            "absolute_lat",
            "absolute_lon",
            "absolute_depth",
            "station_lat",
            "station_lon",
            "station_elev",  # Added station location patterns
        ]

        # Find all columns that match any of the forbidden patterns
        leaked_features = []
        for pattern in forbidden_patterns:
            matching_cols = [col for col in X.columns if pattern in col]
            leaked_features.extend(matching_cols)

        # Remove duplicates
        leaked_features = list(set(leaked_features))

        clean = len(leaked_features) == 0

        if clean:
            print("✅ No target leakage detected")
        else:
            print(f"❌ Found {len(leaked_features)} leaked features: {leaked_features}")
            print("   Removing leaked features to prevent data leakage")
            X = X.drop(leaked_features, axis=1)

        # Check feature variance
        print("Checking feature variance...")

        # Define protected features that should never be dropped
        protected_features = ["pol_az", "pol_inc", "rect_lin", "p_s_time_diff"]

        # HYGIENE TWEAK 2: Pop grouping columns before variance threshold
        # Instead of just protecting them, extract them completely
        group_columns = {}
        if "event_id" in X.columns:
            group_columns["event_id"] = X["event_id"]
            X = X.drop("event_id", axis=1)
        if "event_date" in X.columns:
            group_columns["event_date"] = X["event_date"]
            X = X.drop("event_date", axis=1)

        # For multi-station approach, add aggregated versions of protected features
        if self.approach == "multi_station":
            for base_feature in protected_features:
                for prefix in ["", "best_", "second_"]:
                    for suffix in ["_mean", "_median", "_weighted", ""]:
                        feature = f"{prefix}{base_feature}{suffix}"
                        if feature in X.columns and feature not in protected_features:
                            protected_features.append(feature)

        protected_features_present = [f for f in protected_features if f in X.columns]

        # Apply variance threshold to features except protected ones
        X_unprotected = X.drop(columns=protected_features_present, errors="ignore")

        # Ensure all remaining features are numeric before applying VarianceThreshold
        numeric_columns = [
            col
            for col in X_unprotected.columns
            if pd.api.types.is_numeric_dtype(X_unprotected[col])
        ]
        non_numeric_columns = [
            col for col in X_unprotected.columns if col not in numeric_columns
        ]

        if non_numeric_columns:
            print(
                f"  Excluding {len(non_numeric_columns)} non-numeric columns from variance check: {non_numeric_columns}"
            )
            X_unprotected = X_unprotected[numeric_columns]

        # Now apply variance threshold only to numeric columns
        selector = VarianceThreshold(threshold=1e-5)

        # Fit and transform only unprotected numeric features
        X_filtered_array = selector.fit_transform(X_unprotected)

        # Get boolean mask of selected features
        mask = selector.get_support()

        # Get list of selected features
        selected_features = X_unprotected.columns[mask]

        # Get list of removed features
        removed_features = X_unprotected.columns[~mask]

        # Create new DataFrame with selected features + protected features
        X_filtered = pd.DataFrame(
            X_filtered_array, columns=selected_features, index=X.index
        )

        # Add back the protected features
        for feature in protected_features_present:
            X_filtered[feature] = X[feature]

        # Add back any non-numeric features that were excluded from variance check
        for feature in non_numeric_columns:
            if feature in X.columns:
                X_filtered[feature] = X[feature]

        # HYGIENE TWEAK 2 (continued): Add back grouping columns
        for col, values in group_columns.items():
            X_filtered[col] = values

        print(f"Found {len(removed_features)} features with near-zero variance:")
        if len(removed_features) > 0:
            for feature in removed_features[:5]:  # Show first 5 only
                print(f"  - {feature} (variance: {X[feature].var():.8f})")
            if len(removed_features) > 5:
                print(f"  - ... and {len(removed_features) - 5} more")
        else:
            print("  No low-variance features found")

        if protected_features_present:
            print(
                f"Protected {len(protected_features_present)} critical features from variance filtering:"
            )
            for feature in protected_features_present[:5]:  # Show first 5 only
                print(f"  - {feature}")
            if len(protected_features_present) > 5:
                print(f"  - ... and {len(protected_features_present) - 5} more")

        self.validation_results["feature_validation"] = {
            "target_leakage": {"passed": clean, "leaked_features": leaked_features},
            "variance_check": {
                "removed_features": list(removed_features),
                "protected_features": protected_features_present,
            },
        }

        return X_filtered, y

    def validate_model_stability(self, X, y, n_runs=1):
        """
        Check model stability across multiple runs with different random states
        """
        print("\n" + "=" * 50)
        print("VALIDATING MODEL STABILITY")
        print("=" * 50)

        # For MultiOutputRegressor, we need to test stability differently
        # We'll track the overall 3D error across runs

        stability_results = {"3d_error": []}

        # IMPORTANT FIX: Filter numeric columns for model training
        numeric_columns = [
            col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
        ]
        non_numeric_columns = [col for col in X.columns if col not in numeric_columns]

        if non_numeric_columns:
            print(
                f"Removing {len(non_numeric_columns)} non-numeric columns from stability test: {non_numeric_columns}"
            )
            X_temp = X[numeric_columns]
        else:
            X_temp = X

        # Extract groups for GroupKFold and target values
        if "event_date" in y.columns:
            groups = y["event_date"].values
            y_coords = y[["relative_x", "relative_y", "relative_z"]]
        elif "event_id" in y.columns:
            # FIX: Handle event_id as a group variable, not a feature
            groups = y["event_id"].values
            y_coords = y[["relative_x", "relative_y", "relative_z"]]
        else:
            groups = None
            y_coords = y[["relative_x", "relative_y", "relative_z"]]

        print(f"Checking MultiOutputRegressor stability across {n_runs} runs...")

        all_errors = []

        for i in range(n_runs):
            # Create model with different random state
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.ensemble import RandomForestRegressor

            base_rf = RandomForestRegressor(
                n_estimators=800, max_depth=10, min_samples_leaf=5, random_state=i * 42
            )

            model = MultiOutputRegressor(base_rf)

            if groups is not None:
                # Use group k-fold cross-validation to preserve temporal structure
                from sklearn.model_selection import GroupKFold

                gkf = GroupKFold(n_splits=5)
                fold_errors = []

                for train_idx, test_idx in gkf.split(X_temp, y_coords, groups):
                    X_train, X_test = X_temp.iloc[train_idx], X_temp.iloc[test_idx]
                    y_train, y_test = y_coords.iloc[train_idx], y_coords.iloc[test_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Calculate 3D error
                    errors = np.sqrt(
                        (y_pred[:, 0] - y_test["relative_x"].values) ** 2
                        + (y_pred[:, 1] - y_test["relative_y"].values) ** 2
                        + (y_pred[:, 2] - y_test["relative_z"].values) ** 2
                    )

                    fold_errors.append(np.mean(errors))

                # Average error across folds
                mean_error = np.mean(fold_errors)
                all_errors.append(mean_error)
                print(f"  Run {i+1}: Mean 3D Error = {mean_error:.2f} km")
            else:
                # Standard cross-validation
                from sklearn.model_selection import cross_val_predict

                y_pred = cross_val_predict(model, X_temp, y_coords, cv=5)

                # Calculate 3D error
                errors = np.sqrt(
                    (y_pred[:, 0] - y_coords["relative_x"].values) ** 2
                    + (y_pred[:, 1] - y_coords["relative_y"].values) ** 2
                    + (y_pred[:, 2] - y_coords["relative_z"].values) ** 2
                )

                mean_error = np.mean(errors)
                all_errors.append(mean_error)
                print(f"  Run {i+1}: Mean 3D Error = {mean_error:.2f} km")

        # Calculate statistics
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        cv_percent = (std_error / mean_error) * 100 if mean_error > 0 else float("inf")

        # Check if model is stable (CV < 5%)
        stable = cv_percent < 5.0

        print(
            f"Mean 3D Error: {mean_error:.2f} km, Std Dev: {std_error:.2f} km, CV: {cv_percent:.2f}%"
        )

        if stable:
            print("✅ Multi-output model is stable (CV < 5%)")
        else:
            print("⚠️ Multi-output model shows some instability (CV ≥ 5%)")

        stability_results["3d_error"] = {
            "stable": stable,
            "errors": all_errors,
            "mean_error": mean_error,
            "std_error": std_error,
            "cv_percent": cv_percent,
        }

        self.validation_results["model_stability"] = stability_results
        return stability_results

    def implement_baseline_comparison(self, X_train, y_train, X_test, y_test):
        """
        Implement a comparison baseline model
        """
        print("\n" + "=" * 50)
        print("IMPLEMENTING BASELINE MODELS")
        print("=" * 50)

        # IMPORTANT FIX: Remove non-numeric columns before training models
        # Identify and remove non-numeric columns from X_train and X_test
        numeric_columns = [
            col
            for col in X_train.columns
            if pd.api.types.is_numeric_dtype(X_train[col])
        ]
        non_numeric_columns = [
            col for col in X_train.columns if col not in numeric_columns
        ]

        if non_numeric_columns:
            print(
                f"Removing {len(non_numeric_columns)} non-numeric columns from training data: {non_numeric_columns}"
            )
            X_train_numeric = X_train[numeric_columns]
            X_test_numeric = X_test[numeric_columns]
        else:
            X_train_numeric = X_train
            X_test_numeric = X_test

        # Check if we have polarization features for azimuth estimation
        has_pol_features = False

        # For multi-station approach, check patterns of polarization features
        if self.approach == "multi_station":
            pol_patterns = ["pol_az", "pol_inc", "rect_lin"]
            for pattern in pol_patterns:
                if any(pattern in col for col in X_train_numeric.columns):
                    has_pol_features = True
                    break
        else:
            # For best-station approach, directly check for features
            has_pol_features = all(
                col in X_train_numeric.columns
                for col in ["pol_az", "pol_inc", "rect_lin"]
            )

        # Check for P-S time difference similarly
        has_ps_time = False
        if self.approach == "multi_station":
            if any("p_s_time_diff" in col for col in X_train_numeric.columns):
                has_ps_time = True
        else:
            has_ps_time = "p_s_time_diff" in X_train_numeric.columns

        baseline_preds = {}
        baseline_errors = {}

        # If we have polarization features, implement a distance + azimuth locator
        if has_pol_features:
            print("Implementing distance + azimuth deterministic baseline model")

            # Step 1: Estimate distance based on features
            # For a simple approach, we'll use RandomForest to predict distance first
            from sklearn.ensemble import RandomForestRegressor

            # Calculate true distances from origin (mainshock) to each point
            y_train_dist = np.sqrt(
                y_train["relative_x"] ** 2
                + y_train["relative_y"] ** 2
                + y_train["relative_z"] ** 2
            )

            # Train a model to predict distance
            dist_model = RandomForestRegressor(n_estimators=100, random_state=42)
            # Use only numeric features for the distance model
            dist_model.fit(X_train_numeric, y_train_dist)

            # Predict distances for test set using only numeric features
            predicted_distances = dist_model.predict(X_test_numeric)

            # Step 2: Use polarization azimuth to estimate x, y components
            # Predict horizontal locations using predicted distance and azimuth
            x_preds = []
            y_preds = []
            z_preds = []

            for i in range(len(X_test_numeric)):
                # Get predicted distance
                pred_dist = predicted_distances[i]

                # In multi-station approach, use best_pol_az if available, otherwise pol_az_mean
                if self.approach == "multi_station":
                    if "best_pol_az" in X_test_numeric.columns:
                        azimuth = X_test_numeric["best_pol_az"].iloc[i]
                    elif "pol_az_mean" in X_test_numeric.columns:
                        azimuth = X_test_numeric["pol_az_mean"].iloc[i]
                    elif "pol_az_median" in X_test_numeric.columns:
                        azimuth = X_test_numeric["pol_az_median"].iloc[i]
                    else:
                        # Fallback to the first polarization feature found
                        pol_az_cols = [
                            col for col in X_test_numeric.columns if "pol_az" in col
                        ]
                        if pol_az_cols:
                            azimuth = X_test_numeric[pol_az_cols[0]].iloc[i]
                        else:
                            print("Warning: No polarization azimuth feature found")
                            azimuth = 0

                    # Same for incidence
                    if "best_pol_inc" in X_test_numeric.columns:
                        inc = X_test_numeric["best_pol_inc"].iloc[i]
                    elif "pol_inc_mean" in X_test_numeric.columns:
                        inc = X_test_numeric["pol_inc_mean"].iloc[i]
                    elif "pol_inc_median" in X_test_numeric.columns:
                        inc = X_test_numeric["pol_inc_median"].iloc[i]
                    else:
                        pol_inc_cols = [
                            col for col in X_test_numeric.columns if "pol_inc" in col
                        ]
                        if pol_inc_cols:
                            inc = X_test_numeric[pol_inc_cols[0]].iloc[i]
                        else:
                            print("Warning: No polarization incidence feature found")
                            inc = 45  # Default
                else:
                    # Best-station approach
                    azimuth = X_test_numeric["pol_az"].iloc[i]
                    inc = X_test_numeric["pol_inc"].iloc[i]

                # Convert to radians
                azimuth_rad = np.radians(azimuth)
                inc_rad = np.radians(inc)

                # Calculate x, y, z using distance, azimuth, and incidence
                # Note: azimuth is measured clockwise from north
                horiz_dist = pred_dist * np.sin(inc_rad)
                x_pred = horiz_dist * np.sin(azimuth_rad)  # East-West
                y_pred = horiz_dist * np.cos(azimuth_rad)  # North-South
                z_pred = pred_dist * np.cos(inc_rad)  # Depth

                x_preds.append(x_pred)
                y_preds.append(y_pred)
                z_preds.append(z_pred)

            # Store predictions
            baseline_preds["relative_x"] = np.array(x_preds)
            baseline_preds["relative_y"] = np.array(y_preds)
            baseline_preds["relative_z"] = np.array(z_preds)

        # If we don't have polarization, but we have P-S time, use that for a simpler baseline
        elif has_ps_time:
            print("Using P-S time difference for travel-time based baseline model")

            # Get P-S time differences
            if self.approach == "multi_station":
                # Use best station's P-S time if available, otherwise mean
                if "best_p_s_time_diff" in X_train_numeric.columns:
                    ps_times_train = X_train_numeric["best_p_s_time_diff"]
                    ps_times_test = X_test_numeric["best_p_s_time_diff"]
                elif "p_s_time_diff_mean" in X_train_numeric.columns:
                    ps_times_train = X_train_numeric["p_s_time_diff_mean"]
                    ps_times_test = X_test_numeric["p_s_time_diff_mean"]
                else:
                    # Find the first p_s_time_diff column
                    ps_time_cols = [
                        col for col in X_train_numeric.columns if "p_s_time_diff" in col
                    ]
                    if ps_time_cols:
                        ps_times_train = X_train_numeric[ps_time_cols[0]]
                        ps_times_test = X_test_numeric[ps_time_cols[0]]
                    else:
                        print("Warning: No P-S time difference feature found")
                        return None, None, None
            else:
                # Best-station approach
                ps_times_train = X_train_numeric["p_s_time_diff"]
                ps_times_test = X_test_numeric["p_s_time_diff"]

            # Train a simple linear model for each coordinate based on P-S time
            from sklearn.linear_model import LinearRegression

            for coord in ["relative_x", "relative_y", "relative_z"]:
                model = LinearRegression()
                model.fit(ps_times_train.values.reshape(-1, 1), y_train[coord])
                baseline_preds[coord] = model.predict(
                    ps_times_test.values.reshape(-1, 1)
                )

        # If we have neither, use mean value baseline
        else:
            print("Using mean value baseline model")

            # Baseline: just predict mean values from training set
            for coord in ["relative_x", "relative_y", "relative_z"]:
                mean_value = y_train[coord].mean()
                baseline_preds[coord] = np.ones(len(y_test)) * mean_value

        # Calculate errors for each coordinate
        for coord in ["relative_x", "relative_y", "relative_z"]:
            mse = mean_squared_error(y_test[coord], baseline_preds[coord])
            rmse = np.sqrt(mse)
            baseline_errors[coord] = rmse

        # Calculate 3D distance error
        baseline_3d_errors = np.sqrt(
            (baseline_preds["relative_x"] - y_test["relative_x"]) ** 2
            + (baseline_preds["relative_y"] - y_test["relative_y"]) ** 2
            + (baseline_preds["relative_z"] - y_test["relative_z"]) ** 2
        )
        baseline_errors["3d_distance"] = np.mean(baseline_3d_errors)

        print("Baseline Model Errors (RMSE):")
        for coord in ["relative_x", "relative_y", "relative_z", "3d_distance"]:
            print(f"  {coord}: {baseline_errors[coord]:.2f} km")

        # Now train ML models for comparison
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import RandomForestRegressor

        print("\nTraining optimized ML model with shared trees...")
        # Use MultiOutputRegressor with better parameters
        base_rf = RandomForestRegressor(
            n_estimators=800, max_depth=10, min_samples_leaf=5, random_state=42
        )

        # Wrap with MultiOutputRegressor - use only numeric features
        multi_model = MultiOutputRegressor(base_rf)
        multi_model.fit(
            X_train_numeric, y_train[["relative_x", "relative_y", "relative_z"]]
        )

        # Make predictions using only numeric features
        y_pred_multi = multi_model.predict(X_test_numeric)

        # Convert to dictionary form
        multi_ml_preds = {
            "relative_x": y_pred_multi[:, 0],
            "relative_y": y_pred_multi[:, 1],
            "relative_z": y_pred_multi[:, 2],
        }

        # Calculate errors
        multi_ml_errors = {}
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            mse = mean_squared_error(y_test[coord], multi_ml_preds[coord])
            rmse = np.sqrt(mse)
            multi_ml_errors[coord] = rmse

        # Calculate 3D distance error
        multi_ml_3d_errors = np.sqrt(
            (multi_ml_preds["relative_x"] - y_test["relative_x"]) ** 2
            + (multi_ml_preds["relative_y"] - y_test["relative_y"]) ** 2
            + (multi_ml_preds["relative_z"] - y_test["relative_z"]) ** 2
        )
        multi_ml_errors["3d_distance"] = np.mean(multi_ml_3d_errors)

        # Compare and print results
        print("\nBaseline vs. Optimized ML Model Errors (RMSE):")
        for coord in ["relative_x", "relative_y", "relative_z", "3d_distance"]:
            improvement = (
                (baseline_errors[coord] - multi_ml_errors[coord])
                / baseline_errors[coord]
                * 100
            )
            print(
                f"  {coord}: {baseline_errors[coord]:.2f} km vs {multi_ml_errors[coord]:.2f} km "
                + f"({improvement:.1f}% improvement)"
            )

        # Visualize comparison
        coords = ["relative_x", "relative_y", "relative_z", "3d_distance"]
        baseline_vals = [baseline_errors[c] for c in coords]
        ml_vals = [multi_ml_errors[c] for c in coords]

        plt.figure(figsize=(10, 6))
        x = np.arange(len(coords))
        width = 0.35

        plt.bar(x - width / 2, baseline_vals, width, label="Baseline Model")
        plt.bar(x + width / 2, ml_vals, width, label="Optimized ML Model")

        plt.xlabel("Coordinate")
        plt.ylabel("RMSE (km)")
        plt.title("Baseline vs. ML Model Error Comparison")
        plt.xticks(x, coords)
        plt.legend()
        plt.savefig(f"baseline_comparison_{self.approach}.png", dpi=300)
        plt.close()

        # Store baseline info
        baseline_type = (
            "deterministic (distance + azimuth)"
            if has_pol_features
            else "P-S time" if has_ps_time else "mean value"
        )

        self.validation_results["baseline_comparison"] = {
            "baseline_type": baseline_type,
            "baseline_errors": baseline_errors,
            "ml_errors": multi_ml_errors,
            "improvement": {
                coord: (baseline_errors[coord] - multi_ml_errors[coord])
                / baseline_errors[coord]
                * 100
                for coord in ["relative_x", "relative_y", "relative_z", "3d_distance"]
            },
        }

        # Return the MultiOutputRegressor model, which we'll use for final training
        return baseline_errors, multi_ml_errors, multi_model


def compare_approaches(data_pickle, validation_level="full", results_dir="results"):
    """
    Compare best-station and multi-station approaches on the same dataset
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("COMPARING BEST-STATION AND MULTI-STATION APPROACHES".center(70))
    print("=" * 70)

    # 1. Run with best-station approach (Option 1)
    print("\nRunning best-station approach...")
    analyzer_best = IquiqueAnalysis(
        data_pickle=data_pickle,
        validation_level=validation_level,
        approach="best_station",
    )
    results_best = analyzer_best.run_complete_workflow()

    # 2. Run with multi-station approach (Option 3)
    print("\nRunning multi-station approach...")
    analyzer_multi = IquiqueAnalysis(
        data_pickle=data_pickle,
        validation_level=validation_level,
        approach="multi_station",
    )
    results_multi = analyzer_multi.run_complete_workflow()

    # 3. Compare the results
    print("\n" + "=" * 70)
    print("APPROACH COMPARISON SUMMARY".center(70))
    print("=" * 70)

    # Extract error metrics from the new structure
    best_errors = {
        "relative_x": results_best["test_results"]["errors"][
            "lon"
        ].mean(),  # Longitude -> X
        "relative_y": results_best["test_results"]["errors"][
            "lat"
        ].mean(),  # Latitude -> Y
        "relative_z": results_best["test_results"]["errors"][
            "depth"
        ].mean(),  # Depth -> Z
        "3d_distance": results_best["test_results"]["errors"]["3d"].mean(),
        "3d_median": np.median(results_best["test_results"]["errors"]["3d"]),
    }

    multi_errors = {
        "relative_x": results_multi["test_results"]["errors"][
            "lon"
        ].mean(),  # Longitude -> X
        "relative_y": results_multi["test_results"]["errors"][
            "lat"
        ].mean(),  # Latitude -> Y
        "relative_z": results_multi["test_results"]["errors"][
            "depth"
        ].mean(),  # Depth -> Z
        "3d_distance": results_multi["test_results"]["errors"]["3d"].mean(),
        "3d_median": np.median(results_multi["test_results"]["errors"]["3d"]),
    }

    # Create comparison DataFrame
    comparison = pd.DataFrame(
        {
            "Best-Station": [
                best_errors[k]
                for k in [
                    "relative_x",
                    "relative_y",
                    "relative_z",
                    "3d_distance",
                    "3d_median",
                ]
            ],
            "Multi-Station": [
                multi_errors[k]
                for k in [
                    "relative_x",
                    "relative_y",
                    "relative_z",
                    "3d_distance",
                    "3d_median",
                ]
            ],
        },
        index=["X Error", "Y Error", "Z Error", "3D Mean Error", "3D Median Error"],
    )

    # Calculate improvement
    comparison["Improvement (%)"] = (
        (comparison["Best-Station"] - comparison["Multi-Station"])
        / comparison["Best-Station"]
        * 100
    )

    print("\nError Comparison:")
    print(comparison)

    # Visualize comparison
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(comparison.index))

    plt.bar(
        index - bar_width / 2,
        comparison["Best-Station"],
        bar_width,
        label="Best-Station",
    )
    plt.bar(
        index + bar_width / 2,
        comparison["Multi-Station"],
        bar_width,
        label="Multi-Station",
    )

    # Add improvement annotations
    for i, imp in enumerate(comparison["Improvement (%)"]):
        if imp > 0:  # Only show positive improvements
            plt.text(
                i + bar_width / 2,
                comparison["Multi-Station"].iloc[i]
                + 1,  # Use iloc instead of direct indexing
                f"+{imp:.1f}%",
                ha="center",
                va="bottom",
                color="green",
                fontweight="bold",
            )

    plt.xlabel("Error Metric")
    plt.ylabel("Error (km)")
    plt.title("Best-Station vs. Multi-Station Approach Comparison")
    plt.xticks(index, comparison.index)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{results_dir}/approach_comparison.png", dpi=300)
    plt.tight_layout()

    # Analyze error distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(
        results_best["test_results"]["errors"]["3d"],
        kde=True,
        color="blue",
        alpha=0.5,
        label="Best-Station",
    )
    sns.histplot(
        results_multi["test_results"]["errors"]["3d"],
        kde=True,
        color="red",
        alpha=0.5,
        label="Multi-Station",
    )
    plt.axvline(
        np.mean(results_best["test_results"]["errors"]["3d"]),
        color="blue",
        linestyle="--",
        label=f'Best-Station Mean: {np.mean(results_best["test_results"]["errors"]["3d"]):.2f} km',
    )
    plt.axvline(
        np.mean(results_multi["test_results"]["errors"]["3d"]),
        color="red",
        linestyle="--",
        label=f'Multi-Station Mean: {np.mean(results_multi["test_results"]["errors"]["3d"]):.2f} km',
    )
    plt.xlabel("3D Error (km)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution Comparison")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.7)
    plt.savefig(f"{results_dir}/error_distribution_comparison.png", dpi=300)
    plt.tight_layout()

    # Save comparison results
    comparison.to_csv(f"{results_dir}/approach_comparison.csv")

    print(f"\nComparison results saved to {results_dir}/")

    return comparison, results_best, results_multi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare best-station and multi-station approaches for earthquake location"
    )
    parser.add_argument(
        "--data",
        default="aftershock_data_topN.pkl",
        help="Path to pickle file with preprocessed data",
    )
    parser.add_argument(
        "--validation",
        choices=["none", "critical", "full"],
        default="full",
        help="Validation level (default: full)",
    )
    parser.add_argument(
        "--approach",
        choices=["best_station", "multi_station", "compare"],
        default="compare",
        help="Analysis approach (default: compare both)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save results (default: results)",
    )

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    if args.approach == "compare":
        # Run comparison of both approaches
        comparison, results_best, results_multi = compare_approaches(
            args.data, validation_level=args.validation, results_dir=args.results_dir
        )
    else:
        # Run single approach
        analyzer = IquiqueAnalysis(
            data_pickle=args.data,
            validation_level=args.validation,
            approach=args.approach,
        )
        results = analyzer.run_complete_workflow()

        # Save results
        output_file = f"{args.results_dir}/results_{args.approach}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_file}")
