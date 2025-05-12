#!/usr/bin/env python3
# xgboost_aftershock_prediction.py - XGBoost-based models for aftershock location prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import fft
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
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


class XGBoostAfterShockPredictor:
    """
    Class for predicting aftershock locations using XGBoost models
    with best-station and multi-station approaches
    """

    def __init__(
        self,
        data_file=None,
        validation_level="full",
        approach="multi_station",
        feature_type="all",  # Modified: "all", "signal", "physics"
    ):
        """
        Initialize the predictor

        Args:
            data_file: Path to pickle or HDF5 file with preprocessed data
            approach: Analysis approach to use
                    "best_station" - use only the best station for each event
                    "multi_station" - use all available stations for each event
            feature_type: Type of features to use
                        "all" - use all features
                        "signal" - use only signal-based features (Tier A)
                        "physics" - use physics features (Tier C)
        """
        self.data_dict = None
        self.aftershocks_df = None
        self.mainshock_key = None
        self.mainshock = None
        self.models = None
        self.feature_importances = None
        self.scaler = None
        self.validation_level = "full"
        self.validation_results = {}
        self.approach = approach
        self.feature_type = feature_type
        self.data_format = None  # Will be set when loading data

        print(f"Validation level: {validation_level}")
        print(f"Analysis approach: {approach}")
        print(f"Feature type: {feature_type}")
        
        # Print more detailed feature tier description
        if feature_type == "signal":
            print("Using Tier A (signal statistics) features only")
        elif feature_type == "physics":
            print("Using Tier C (source physics) features only")
        elif feature_type in ["all"]:
            print("Using all features (Tiers A+C: signal + source physics)")

        # Load data
        if data_file and os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            # Determine file type by extension
            if data_file.lower().endswith(".pkl") or data_file.lower().endswith(
                ".pickle"
            ):
                self.data_dict = self.load_from_pickle(data_file)
            else:
                raise ValueError(
                    f"Unsupported file type: {data_file}. Must be .pkl, .pickle"
                )
        else:
            print("No file found or invalid path provided.")
            return

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
            # If the value is a dict of station keys, it's the multi-station format
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

        # Validate coordinate conversion if required
        if self.validation_level != "none":
            self.validate_coordinate_conversion()

        return self.aftershocks_df

    def remove_specific_features(self, X, features_to_remove=["pol_inc_min"]):
        """
        Remove specific features from the dataset

        Args:
            X: Feature dataframe
            features_to_remove: List of feature names to remove

        Returns:
            DataFrame with features removed
        """
        X_filtered = X.copy()

        # Find all columns that match any of the patterns to remove
        columns_to_remove = []
        for pattern in features_to_remove:
            matching_cols = [col for col in X.columns if pattern in col]
            columns_to_remove.extend(matching_cols)

        # Remove the columns if they exist
        existing_columns = [col for col in columns_to_remove if col in X.columns]
        if existing_columns:
            print(f"Removing {len(existing_columns)} features: {existing_columns}")
            X_filtered = X_filtered.drop(columns=existing_columns)
        else:
            print(f"No matching features found for patterns: {features_to_remove}")

        return X_filtered

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

        # East-West distance (x) - Use reference latitude for better accuracy
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
        Extract features from the 3-component waveform data based on selected feature type
        
        Feature types are organized in tiers:
        - "signal": Basic signal statistics (Tier A)
        - "physics": Source physics parameters requiring explicit models (Tier C)
        - "all": All feature types
        
        Args:
            waveform: 3-component seismic waveform array
            metadata: Optional metadata dictionary for the station
        """
        features = {}

        waveform_ms = waveform

        # Assume sampling rate of 100 Hz
        sampling_rate = 100.0
        g = 9.81  # Gravitational acceleration in m/s^2
        
        # ---- NEW HELPER FUNCTIONS FOR IMPROVED BRUNE MODEL FITTING ----
        def compute_multitaper_spectrum(signal, fs, nfft=None):
            """Multi-taper spectral estimation with consistent FFT length
            
            Args:
                signal: Time series data
                fs: Sampling frequency
                nfft: FFT length (if None, uses len(signal))
            """
            from scipy.signal import windows
            
            # Use specified FFT length or signal length
            N = len(signal)
            nfft = nfft if nfft is not None else N
            
            # Time-bandwidth parameter and number of tapers
            NW = 4.0
            K = int(2 * NW - 1)
            
            # Use Welch method for consistent length output
            return signal.welch(signal, fs=fs, nperseg=min(1024, N//4), nfft=nfft)
        
        def compute_spectrum_with_snr(signal, fs, pre_event_samples=500, nfft=None):
            """Compute spectrum with SNR-based quality filtering using consistent FFT length
            
            Args:
                signal: Time series data
                fs: Sampling frequency
                pre_event_samples: Number of samples to use for noise estimation
                nfft: FFT length (if None, uses next power of 2 >= len(signal))
            """
            # If nfft not specified, use next power of 2 >= signal length for efficiency
            if nfft is None:
                nfft = 2**int(np.ceil(np.log2(len(signal))))
            
            # Adjust pre-event window if needed
            pre_event_samples = min(pre_event_samples, len(signal)//10)
            
            # Extract pre-event (noise) window
            noise = signal[:pre_event_samples]
            
            # Compute spectra with consistent FFT length
            freqs, signal_psd = compute_multitaper_spectrum(signal, fs, nfft=nfft)
            _, noise_psd = compute_multitaper_spectrum(noise, fs, nfft=nfft)
            
            # Calculate SNR
            snr = signal_psd / (noise_psd + 1e-15)
            
            # Create mask for frequencies with good SNR
            snr_threshold = 3.0
            snr_mask = snr > snr_threshold
            
            # Ensure some low-frequency points for fitting
            low_freq_mask = freqs < 2.0
            if np.sum(snr_mask & low_freq_mask) < 5 and np.sum(low_freq_mask) >= 5:
                best_low_indices = np.argsort(snr[low_freq_mask])[-5:]
                low_indices = np.where(low_freq_mask)[0][best_low_indices]
                snr_mask[low_indices] = True
            
            return freqs, np.sqrt(signal_psd), snr_mask
        
        # ---- NEW HELPER FUNCTION FOR JOINT BRUNE + Q FIT IN LOG SPACE ----
        def fit_brune_model_with_Q(freqs, displ_spec, R_km, beta_km_s, snr_mask=None):
            """
            Joint fit of Brune spectrum and Q attenuation in log-log space
            
            Args:
                freqs: frequencies array (Hz)
                displ_spec: displacement spectrum array
                R_km: distance in km
                beta_km_s: shear wave velocity in km/s
                snr_mask: boolean mask for frequencies with good SNR (must match freqs length if provided)
                
            Returns:
                fc: corner frequency (Hz)
                Omega0: low-frequency plateau
                Q: quality factor
                success: whether the fit was successful
            """
            from scipy.optimize import curve_fit
            import numpy as np
            
            # Safety check for dimension mismatch
            if snr_mask is not None and len(snr_mask) != len(freqs):
                print(f"Warning: SNR mask length ({len(snr_mask)}) doesn't match frequency array length ({len(freqs)}). Using default frequency mask.")
                snr_mask = None
            
            # Apply SNR mask if provided (and verified)
            if snr_mask is not None and np.sum(snr_mask) > 10:
                f_fit = freqs[snr_mask]
                U_fit = displ_spec[snr_mask]
            else:
                # Default: use frequencies between 0.5 Hz and 80% of max
                mask = (freqs >= 0.5) & (freqs <= freqs.max() * 0.8)
                f_fit = freqs[mask]
                U_fit = displ_spec[mask]
            
            if len(f_fit) < 10:
                return 1.0, 1.0, 600, False
            
            # Ensure positive values for log transform
            U_fit = np.maximum(U_fit, 1e-15)
            
            # Convert to log space
            log_f = np.log10(f_fit)
            log_U = np.log10(U_fit)
            
            # Define Brune model with Q in log space
            def log_brune_Q(log_f, log_Omega0, log_fc, Q):
                f = 10**log_f
                Omega0, fc = 10**log_Omega0, 10**log_fc
                term = Omega0 / (1 + (f/fc)**2) * np.exp(np.pi*f*R_km/(Q*beta_km_s))
                return np.log10(term)
            
            # Initial parameter guess
            log_Omega0_guess = np.log10(np.max(U_fit))
            
            try:
                # Curve fit in log space
                popt, _ = curve_fit(
                    log_brune_Q, 
                    log_f, 
                    log_U,
                    p0=[log_Omega0_guess, np.log10(1.0), 600],
                    bounds=([log_Omega0_guess-2, -1.0, 50], 
                            [log_Omega0_guess+2, 1.5, 1500]),
                    maxfev=5000
                )
                
                # Convert back to linear space
                log_Omega0, log_fc, Q = popt
                Omega0, fc = 10**log_Omega0, 10**log_fc
                
                return fc, Omega0, Q, True
                
            except:
                # Fallback if fitting fails
                print("Fitting failed, returning default values")
                return 1.0, np.max(U_fit), 600, False
                
        # ---- LOOKUP FUNCTION FOR DEPTH-DEPENDENT VELOCITY AND DENSITY ----
        def get_velocity_density(depth_km, model="iasp91"):
            """Get shear velocity and density at a given depth using a 1D Earth model"""
            # Simple lookup table for IASP91 model (depth in km, vs in km/s)
            iasp91_vs = {
                0: 3.36,    # Upper crust
                20: 3.75,   # Mid-crust
                35: 4.47,   # Moho
                50: 4.49,   # Upper mantle
                100: 4.51,  # Upper mantle
                200: 4.56   # Upper mantle
            }
            
            # Find closest depth
            depths = np.array(list(iasp91_vs.keys()))
            idx = np.abs(depths - depth_km).argmin()
            
            # Get velocity in km/s
            beta = iasp91_vs[depths[idx]]
            
            # Estimate density using empirical relationship ρ ≈ 0.32 β^0.25 (g/cm³)
            rho = 0.32 * (beta*1000)**0.25  # Convert to g/cm³
            
            return beta, rho

        # Optional: Dynamic low-cut based on record length to ensure stability
        Fs  = sampling_rate              # 100 Hz in your data set
        ny  = Fs / 2.0
        T   = waveform_ms.shape[1] / Fs  # trace length ≈ 146.36 s

        # ------------------------------------------------------------------
        # 1.  Filters for engineering / IM-type features (Tiers A)
        #     0.30 – 35 Hz  4-pole Butterworth band-pass on *velocity*
        b_vel, a_vel = butter(4, [0.30/ny, 35.0/ny], btype="bandpass")

        # ------------------------------------------------------------------
        # 2.  Filters for source-physics features (Tier C)
        #     (i) integrate –> displacement
        #     (ii) *very* gentle high-pass at 0.05 Hz to kill DC drift only
        low_hp_disp  = max(0.05, 2.5/T)      #   2.5 / 146.36 ≈ 0.017 Hz  → 0.05 Hz
        b_hp_disp, a_hp_disp = butter(2, low_hp_disp/ny, btype="highpass")

        # Store filtered components for cross-component analysis
        velocity_filtered_components = []

        # Process each component
        for i, component in enumerate(["Z", "N", "E"]):
            velocity_filtered = filtfilt(b_vel, a_vel, waveform_ms[i])
            velocity_filtered_components.append(velocity_filtered)
            acceleration = np.gradient(velocity_filtered, 1/sampling_rate)
            displacement = scipy.integrate.cumulative_trapezoid(velocity_filtered, dx=1/Fs, initial=0)
            displacement_filtered = filtfilt(b_hp_disp, a_hp_disp, displacement)

            # Calculate spectrum on filtered velocity
            f, Pxx = signal.welch(velocity_filtered, fs=sampling_rate, nperseg=1024)

            # TIER A: Low-level signal statistics 
            if self.feature_type in ["all", "signal"]:
                # Simple statistics (now on filtered velocity)
                features[f"{component}_mean"] = np.mean(velocity_filtered)
                features[f"{component}_std"] = np.std(velocity_filtered)
                features[f"{component}_range"] = np.ptp(velocity_filtered)
                features[f"{component}_energy"] = np.sum(velocity_filtered ** 2)

                features[f"{component}_PGV"] = np.max(np.abs(velocity_filtered))
                features[f"{component}_PGA"] = np.max(np.abs(acceleration))
                features[f"{component}_PGD"] = np.max(np.abs(displacement_filtered))

                # RMS
                features[f"{component}_rms"] = np.sqrt(np.mean(velocity_filtered ** 2))

                # Zero crossings
                features[f"{component}_zero_crossings"] = np.sum(
                    np.diff(np.signbit(velocity_filtered))
                )

                # Spectral features
                features[f"{component}_peak_freq"] = f[np.argmax(Pxx)]
                features[f"{component}_spectral_mean"] = np.mean(Pxx)
                features[f"{component}_spectral_std"] = np.std(Pxx)

                # Frequency bands energy
                band_ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40)]
                for j, (low, high) in enumerate(band_ranges):
                    mask = (f >= low) & (f <= high)
                    features[f"{component}_band_{low}_{high}_energy"] = np.sum(
                        Pxx[mask]
                    )

            # TIER C: Source physics parameters (requiring physical model)
            if self.feature_type in ["all", "physics"]:
                # ---------- 1. SITE CORRECTION (KAPPA) ----------
                # Get kappa from Tier B (or use default if not available)
                kappa = 0.04
                
                # Compute displacement amplitude spectrum
                n = len(displacement_filtered)
                freqs = fft.rfftfreq(n, d=1/sampling_rate)
                displ_fft = fft.rfft(displacement_filtered * np.hanning(n))
                displacement_spectrum = np.abs(displ_fft)
                
                # Apply kappa correction to remove site effects
                displ_corr = displacement_spectrum * np.exp(np.pi * kappa * freqs)
                
                # ---------- 2. RADIATION PATTERN CORRECTION ----------
                # Calculate station distance from coordinates if not directly available
                if "station_distance" in metadata:        
                    R_km = metadata["station_distance"]
                    
                    # Add depth component if available
                    src_depth = metadata["metadata"]["source_depth_km"]
                    sta_elev = metadata["metadata"]["station_elevation_m"] / 1000.0  # Convert to km
                    depth_diff = src_depth + sta_elev  # Positive depth down, positive elevation up
                    
                    # Update R with 3D distance
                    R_km = np.sqrt(R_km**2 + depth_diff**2)
                
                # Radiation pattern coefficient
                G = 0.55  # Default average DC value
                
                # Apply radiation pattern correction (avoid division by zero)
                G = max(G, 0.1)  # Minimum value to avoid extreme amplification
                displ_corr_rad = displ_corr / G
                
                # ---------- 3. GEOMETRICAL SPREADING CORRECTION ----------
                # Apply R factor to correct for geometrical spreading
                displ_corr_rad_geo = displ_corr_rad * R_km
                
                # ---------- 4. JOINT BRUNE + Q FIT -------------
                # Ensure we use the same FFT length for spectral calculations
                nfft = len(displacement_filtered)
                
                # Use the same FFT for calculating spectrum with SNR mask to avoid dimension mismatch
                pre_event_samples = int(0.05 * len(displacement_filtered))  # Use first 5% as noise
                
                # Calculate displacement spectrum directly (without SNR mask for now)
                freqs = fft.rfftfreq(nfft, d=1/sampling_rate)
                
                # Skip SNR mask to avoid dimension mismatch error
                snr_mask = None
                
                # Get depth-dependent velocity and density
                source_depth_km = metadata["metadata"]["source_depth_km"]
                beta_km_s, rho_g_cm3 = get_velocity_density(source_depth_km)
                
                # Convert to correct units
                beta_m_s = beta_km_s * 1000  # km/s to m/s
                rho_kg_m3 = rho_g_cm3 * 1000  # g/cm³ to kg/m³
                
                # Joint Brune + Q fit in log-log space
                fc, Omega0, Q, fit_success = fit_brune_model_with_Q(
                    freqs, displ_corr_rad_geo, R_km, beta_km_s, None)
                
                # Store the calibrated parameters without "_cal" suffix (since we're removing proxy versions)
                features[f"{component}_corner_freq"] = fc
                features[f"{component}_Omega0"] = Omega0
                features[f"{component}_Q_path"] = Q
                features[f"{component}_G_radiation"] = G
                
                # ---------- 5. DERIVED SOURCE PARAMETERS ----------
                # Brune constant
                k = 0.37
                
                # Shear modulus (Pa)
                mu = 30e9
                
                # Calculate the moment (N·m) with correct units and scaling
                M0 = Omega0 * 4 * np.pi * rho_kg_m3 * (beta_m_s**3) * (R_km*1000) / G
                features[f"{component}_M0"] = M0
                
                # Moment magnitude (Hanks & Kanamori)
                Mw = (2.0/3.0) * np.log10(M0) - 6.06
                features[f"{component}_Mw"] = Mw
                
                # Source radius (m)
                r_brune = k * beta_m_s / fc
                features[f"{component}_source_radius"] = r_brune
                
                # Calibrated stress drop (Pa)
                stress_drop = (7./16.) * M0 / (r_brune**3)
                features[f"{component}_stress_drop"] = stress_drop
                
                # Rupture area (m²)
                rupture_area = np.pi * r_brune**2
                features[f"{component}_rupture_area"] = rupture_area
                
                # Average slip (m)
                avg_slip = M0 / (mu * rupture_area)
                features[f"{component}_avg_slip"] = avg_slip
                
                # Radiated seismic energy (J)
                # Compute velocity amplitude spectrum
                V = np.abs(fft.rfft(velocity_filtered * np.hanning(len(velocity_filtered))))
                V_freqs = fft.rfftfreq(len(velocity_filtered), d=1/sampling_rate)
                
                # Apply radiation and distance corrections
                V_corr = V / G * R_km
                
                # Integrate over frequency band with reliable SNR
                vel_mask = (V_freqs >= 0.5) & (V_freqs <= min(49, sampling_rate/2 * 0.8))
                Er = 4 * np.pi * rho_kg_m3 * beta_m_s**5 * np.trapz(V_corr[vel_mask]**2, V_freqs[vel_mask])
                features[f"{component}_Er"] = Er
                
                # Apparent stress (Pa)
                sigma_a = mu * Er / M0
                features[f"{component}_apparent_stress"] = sigma_a

        # TIER B: Polarization & site features (no source model)
        if self.feature_type in ["all", "physics"]:
            # Polarization features
            features["pol_az"] = np.degrees(
                np.arctan2(np.std(velocity_filtered_components[2]), np.std(velocity_filtered_components[1]))
            )  # Azimuth
            features["pol_inc"] = np.degrees(
                np.arctan2(
                    np.sqrt(np.std(velocity_filtered_components[1]) ** 2 + np.std(velocity_filtered_components[2]) ** 2),
                    np.std(velocity_filtered_components[0]),
                )
             )  # Incidence

                
        # TIER C: Corner-frequency anisotropy (source physics metric)
        if self.feature_type in ["all", "physics"]:
            fc_Z = features.get("Z_corner_freq", np.nan)
            fc_N = features.get("N_corner_freq", np.nan)
            fc_E = features.get("E_corner_freq", np.nan)
            fc_Hgeom = np.sqrt(fc_N * fc_E)  # geometric mean horizontal
            features["fc_anisotropy"] = fc_Z - fc_Hgeom  # Hz
            
            # Add average calibrated parameters across components
            # For stress drop, use geometric mean (log-normal distribution)
            stress_drops = [
                features.get("Z_stress_drop", np.nan),
                features.get("N_stress_drop", np.nan),
                features.get("E_stress_drop", np.nan)
            ]
            if all(~np.isnan(stress_drops)):
                features["avg_stress_drop"] = np.exp(np.nanmean(np.log(stress_drops)))
            
            # For moment, use median (more robust to outliers)
            moments = [
                features.get("Z_M0", np.nan),
                features.get("N_M0", np.nan),
                features.get("E_M0", np.nan)
            ]
            if all(~np.isnan(moments)):
                features["avg_M0"] = np.nanmedian(moments)
                # Add corresponding Mw
                features["avg_Mw"] = (2.0/3.0) * np.log10(features["avg_M0"]) - 6.06

        return features

    def perform_shap_analysis(self, X_test, y_test, max_display=20):
        """
        Perform SHAP analysis on the trained XGBoost models to interpret feature importance
        and feature contributions to individual predictions.

        Args:
            X_test: Test feature dataset
            y_test: Test target dataset
            max_display: Maximum number of features to display in SHAP plots

        Returns:
            Dictionary containing SHAP values and explanations
        """
        import shap
        import matplotlib.pyplot as plt
        import numpy as np

        if self.models is None:
            raise ValueError("Models not trained yet. Call train_xgboost_models first.")

        print("\n" + "=" * 50)
        print("PERFORMING SHAP ANALYSIS")
        print("=" * 50)

        # Prepare results dictionary
        shap_results = {"values": {}, "plots": {}, "feature_importance": {}}

        # Set up the explainer
        coords = ["relative_x", "relative_y", "relative_z"]

        for i, coord in enumerate(coords):
            print(f"\nAnalyzing SHAP values for {coord} prediction...")

            # Get the XGBoost model for this coordinate
            xgb_model = self.models["multi_output"].estimators_[i]

            # Initialize the explainer - use TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(xgb_model)

            # Calculate SHAP values for test set
            shap_values = explainer(X_test)

            # Store SHAP values
            shap_results["values"][coord] = shap_values

            # 1. Summary plot (overall feature importance based on SHAP)
            plt.figure(figsize=(12, 12))
            shap.summary_plot(
                shap_values.values,
                X_test,
                max_display=max_display,
                show=False,
                plot_size=(12, 12),
            )
            plt.title(f"SHAP Summary Plot for {coord} Prediction")
            plt.tight_layout()
            plt.savefig(
                f"shap_summary_{coord}_{self.approach}_{self.feature_type}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # 2. Bar plot (mean absolute SHAP values)
            plt.figure(figsize=(12, 10))
            shap.plots.bar(shap_values, max_display=max_display, show=False)
            plt.title(f"SHAP Feature Importance for {coord} Prediction")
            plt.tight_layout()
            plt.savefig(
                f"shap_importance_bar_{coord}_{self.approach}_{self.feature_type}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # # 3. Beeswarm plot (distribution of SHAP values)
            # plt.figure(figsize=(12, 12))
            # shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
            # plt.title(f"SHAP Value Distribution for {coord} Prediction")
            # plt.tight_layout()
            # plt.savefig(
            #     f"shap_beeswarm_{coord}_{self.approach}_{self.feature_type}.png",
            #     dpi=300,
            #     bbox_inches="tight",
            # )
            # plt.close()

            # 5. Waterfall plots for selected examples (show detailed feature contribution)
            # Select 3 examples: one with small error, one with average error, one with large error
            if (
                "relative_x" in y_test.columns
            ):  # Make sure we have actual coords to compare
                # Predict test samples
                y_pred = xgb_model.predict(X_test)

                # Calculate prediction errors
                errors = np.abs(y_test[coord].values - y_pred)

                # Get indices for examples with small, medium, and large errors
                small_error_idx = errors.argmin()
                large_error_idx = errors.argmax()

                # Sort errors and find the median error example
                sorted_indices = np.argsort(errors)
                medium_error_idx = sorted_indices[len(sorted_indices) // 2]

                example_indices = [small_error_idx, medium_error_idx, large_error_idx]
                error_types = ["small", "medium", "large"]

                # for idx, error_type in zip(example_indices, error_types):
                #     plt.figure(figsize=(12, 10))
                #     shap.plots.waterfall(
                #         shap_values[idx],
                #         max_display=15,  # Show more features for waterfall
                #         show=False,
                #     )
                #     plt.title(
                #         f"{error_type.capitalize()} Error Example: SHAP Waterfall Plot for {coord}"
                #     )
                #     plt.tight_layout()
                #     plt.savefig(
                #         f"shap_waterfall_{coord}_{error_type}_{self.approach}_{self.feature_type}.png",
                #         dpi=300,
                #         bbox_inches="tight",
                #     )
                #     plt.close()

                #     # Print actual vs predicted values for this example
                #     print(f"  {error_type.capitalize()} error example:")
                #     print(f"    True {coord}: {y_test[coord].iloc[idx]:.2f} km")
                #     print(f"    Predicted {coord}: {y_pred[idx]:.2f} km")
                #     print(f"    Error: {errors[idx]:.2f} km")

            # Store feature importance based on SHAP values
            feature_importance = pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance": np.abs(shap_values.values).mean(0),
                }
            ).sort_values("importance", ascending=False)

            shap_results["feature_importance"][coord] = feature_importance

            # Print top features based on SHAP
            print(f"\nTop features for {coord} prediction (SHAP-based):")
            print(feature_importance.head(10))

        # 6. Global Analysis - Compare SHAP vs. native XGBoost feature importance
        for coord in coords:
            plt.figure(figsize=(12, 8))

            # Get top 20 features from both methods
            xgb_importance = self.feature_importances[coord].head(20)
            shap_importance = shap_results["feature_importance"][coord].head(20)

            # Merge the two
            merged_importance = pd.merge(
                xgb_importance,
                shap_importance,
                on="feature",
                how="outer",
                suffixes=("_xgb", "_shap"),
            ).fillna(0)

            # Filter to get features that appear in either top 20
            top_features = merged_importance[
                (merged_importance["importance_xgb"] > 0)
                | (merged_importance["importance_shap"] > 0)
            ].sort_values("importance_shap", ascending=True)

            # Normalize for comparison
            top_features["importance_xgb"] = (
                top_features["importance_xgb"] / top_features["importance_xgb"].max()
            )
            top_features["importance_shap"] = (
                top_features["importance_shap"] / top_features["importance_shap"].max()
            )

            # # Plot
            # fig, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.3)))

            # x = np.arange(len(top_features))
            # width = 0.35

            # ax.barh(
            #     x - width / 2,
            #     top_features["importance_xgb"],
            #     width,
            #     label="XGBoost Native",
            # )
            # ax.barh(
            #     x + width / 2,
            #     top_features["importance_shap"],
            #     width,
            #     label="SHAP-based",
            # )

            # ax.set_yticks(x)
            # ax.set_yticklabels(top_features["feature"])
            # ax.set_xlabel("Normalized Importance")
            # ax.set_title(f"XGBoost vs. SHAP Feature Importance: {coord}")
            # ax.legend()

            # plt.tight_layout()
            # plt.savefig(
            #     f"importance_comparison_{coord}_{self.approach}_{self.feature_type}.png",
            #     dpi=300,
            #     bbox_inches="tight",
            # )
            # plt.close()

        # 7. Create a consolidated feature importance plot for all coordinates
        plt.figure(figsize=(15, 12))

        # Set up subplots
        fig, axes = plt.subplots(len(coords), 1, figsize=(15, 5 * len(coords)))

        for i, coord in enumerate(coords):
            # Get top 15 features based on SHAP
            top_features = shap_results["feature_importance"][coord].head(15)

            # Create bar plot
            axes[i].barh(top_features["feature"], top_features["importance"])
            axes[i].set_title(f"Top 15 Features for {coord} (SHAP Analysis)")
            axes[i].set_xlabel("Mean |SHAP Value|")

            # Add grid for readability
            axes[i].grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(
            f"shap_consolidated_{self.approach}_{self.feature_type}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"\nSHAP analysis completed. All visualizations saved.")

        return shap_results

    def prepare_multi_station_dataset(self):
        """
        Advanced method to prepare dataset using multiple stations for each event
        with improved station selection, distribution statistics, and reduced redundancy
        """
        if self.aftershocks_df is None:
            self.create_relative_coordinate_dataframe()

        print("Preparing enhanced multi-station dataset...")

        # Step 1: Extract features from all waveforms
        print("Extracting features from waveforms...")
        features_list = []
        errors = 0

        for idx, row in tqdm(
            self.aftershocks_df.iterrows(), total=len(self.aftershocks_df)
        ):
            # Extract features with metadata
            features = self.extract_waveform_features(
                row["waveform"], metadata=row
            )
            
            # Add necessary metadata needed for aggregation later
            features["station_key"] = row.get("station_key", "")
            features["event_id"] = row.get("event_id", "")
            features["origin_time"] = row["origin_time"]
            features["event_date"] = row["event_date"]
            features["relative_x"] = row["relative_x"]
            features["relative_y"] = row["relative_y"]
            features["relative_z"] = row["relative_z"]
            
            # Add station_distance ONLY for ranking purposes (prefixed to mark it)
            features["_ranking_station_distance"] = row["station_distance"]

            features_list.append(features)


        print(
            f"Successfully processed {len(features_list)} waveforms with {errors} errors"
        )

        # Convert to DataFrame
        all_features_df = pd.DataFrame(features_list)

        # Step 2: Aggregate features across stations for each event
        print("Aggregating features across stations for each event...")

        # Group by event_id
        event_groups = all_features_df.groupby("event_id")

        # List of columns that shouldn't be aggregated
        skip_columns = [
            "origin_time",
            "event_date",
            "relative_x",
            "relative_y",
            "relative_z",
            "event_id",
            "station_key",
        ]

        # List of columns that should not be aggregated to prevent data leakage
        leakage_columns = [
            col
            for col in all_features_df.columns
            if "selection_score" in col
            or "epicentral_distance" in col
            or "station_lat" in col
            or "station_lon" in col
            or "station_elev" in col
            or col.startswith("_ranking_")  # Exclude ranking metrics 
        ]

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
            }

            # For each feature, calculate various statistics across stations
            for feature in numeric_columns:
                values = group[feature].values
                valid_values = values[~np.isnan(values)]
                
                if len(valid_values) == 0:
                    # Handle case with no valid values
                    event_data[f"{feature}_mean"] = np.nan
                    event_data[f"{feature}_median"] = np.nan
                    event_data[f"{feature}_std"] = np.nan
                    event_data[f"{feature}_min"] = np.nan
                    event_data[f"{feature}_max"] = np.nan
                    event_data[f"{feature}_range"] = np.nan
                    event_data[f"{feature}_q25"] = np.nan
                    event_data[f"{feature}_q75"] = np.nan
                    continue
                    
                # Basic statistics
                event_data[f"{feature}_mean"] = np.mean(valid_values)
                event_data[f"{feature}_median"] = np.median(valid_values)
                # Use NaN for std when there's only one value
                event_data[f"{feature}_std"] = np.std(valid_values) if len(valid_values) > 1 else np.nan
                event_data[f"{feature}_min"] = np.min(valid_values)
                event_data[f"{feature}_max"] = np.max(valid_values)
                event_data[f"{feature}_range"] = np.ptp(valid_values)

                # Add direct quantile values for better distribution representation
                if len(valid_values) >= 4:  # Need enough points for meaningful quartiles
                    q25, q75 = np.percentile(valid_values, [25, 75])
                    event_data[f"{feature}_q25"] = q25
                    event_data[f"{feature}_q75"] = q75

            # Select representative stations using ONLY station_distance
            if "_ranking_station_distance" in group.columns and not all(np.isnan(group["_ranking_station_distance"])):
                # Sort by station_distance (ascending) - closest stations first
                quality_sorted = group.sort_values("_ranking_station_distance")
            else:
                # Fallback to original order if station_distance isn't available
                print(f"Warning: No station_distance available for event {event_id}. Using original order.")
                quality_sorted = group
                
            # Add features from best, second best, and third best stations
            station_indices = quality_sorted.index.tolist()
            
            # Best station (sorted by distance)
            if len(station_indices) > 0:
                best_station = quality_sorted.iloc[0]
                for feature in numeric_columns:
                    event_data[f"best_{feature}"] = best_station[feature]

            # Second best station
            if len(station_indices) > 1:
                second_station = quality_sorted.iloc[1]
                for feature in numeric_columns:
                    event_data[f"second_{feature}"] = second_station[feature]

            # Third best station
            if len(station_indices) > 2:
                third_station = quality_sorted.iloc[2]
                for feature in numeric_columns:
                    event_data[f"third_{feature}"] = third_station[feature]


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
            "event_date",
        ]

        # Keep event_id for GroupKFold
        X = merged_df.drop(drop_columns, axis=1)
        y = merged_df[["relative_x", "relative_y", "relative_z", "event_id"]]

        return X, y

    def prepare_dataset(self):
        """
        Prepare dataset for machine learning by extracting features from waveforms
        """
        if self.approach == "multi_station" and self.data_format == "multi_station":
            return self.prepare_multi_station_dataset()

    def train_xgboost_models(self, X, y, perform_shap=True):
        """
        Train XGBoost models to predict aftershock locations

        Args:
            X: Feature DataFrame
            y: Target DataFrame with coordinates
            perform_shap: Whether to perform SHAP analysis (default: True)

        Returns:
            X_test, y_test: Test data for evaluation
        """
        # Split data into training and testing sets using GroupKFold if available
        if self.validation_level != "none":
            if "event_id" in y.columns:
                print(
                    "Using GroupKFold with event_id as the group to prevent data leakage..."
                )
                # Get a single train/test split using GroupKFold
                gkf = GroupKFold(n_splits=10)
                groups = y["event_id"]
                train_idx, test_idx = next(gkf.split(X, y, groups))

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                # Drop the group columns from X_train and X_test before modeling
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
                gkf = GroupKFold(n_splits=10)
                groups = y["event_date"]
                train_idx, test_idx = next(gkf.split(X, y, groups))

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                # Drop the group columns from X_train and X_test before modeling
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

        # Filter numeric columns for model training
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

        # Fixed XGBoost parameters (best parameters from previous tuning)
        xgb_params = {
            "n_estimators": 800,
            "learning_rate": 0.017287387897853453,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.7980103868976611,
            "colsample_bytree": 0.8559520053515827,
            "reg_alpha": 0.231981075221874,
            "reg_lambda": 4.4044780906686425e-07,
            "gamma": 0.44340940726177724,
            "random_state": 42,
        }

        print(f"Training XGBoost model with parameters: {xgb_params}")

        # Create and train XGBoost model with fixed parameters
        base_xgb = XGBRegressor(**xgb_params)

        # Use MultiOutputRegressor to predict all three coordinates
        multi_model = MultiOutputRegressor(base_xgb)
        multi_model.fit(
            X_train_numeric, y_train_coord[["relative_x", "relative_y", "relative_z"]]
        )

        # Make predictions on test set
        y_pred = multi_model.predict(X_test_numeric)

        # Calculate and print errors
        multi_ml_errors = {}
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            mse = mean_squared_error(y_test_coord[coord], y_pred[:, i])
            rmse = np.sqrt(mse)
            multi_ml_errors[coord] = rmse

        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            r2 = r2_score(y_test_coord[coord], y_pred[:, i])
            print(f"  {coord} R²: {r2:.4f}")

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

        # Store the model
        self.models = {"multi_output": multi_model, "type": "xgboost"}
        self.scaler = None  # No scaler used with XGBoost

        # Extract feature importance
        self.feature_importances = {}
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            xgb_model = multi_model.estimators_[i]

            # Get feature importance
            importance_df = pd.DataFrame(
                {
                    "feature": X_train_numeric.columns,
                    "importance": xgb_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            self.feature_importances[coord] = importance_df

            print(f"\nTop features for {coord} prediction:")
            print(importance_df.head(100))

            # Save feature importance plot
            # plt.figure(figsize=(12, 8))
            # top_features = importance_df.head(100)
            # sns.barplot(x="importance", y="feature", data=top_features)
            # plt.title(f"Top 100 Features for {coord} Prediction (XGBoost)")
            # plt.tight_layout()
            # plt.savefig(
            #     f"xgboost_feature_importance_{coord}_{self.approach}_{self.feature_type}.png",
            #     dpi=300,
            # )
            # plt.close()

        # Perform SHAP analysis if requested
        if perform_shap:
            try:
                import shap

                self.shap_results = self.perform_shap_analysis(
                    X_test_numeric, y_test_coord
                )
            except ImportError:
                print("\nWARNING: SHAP library not found. Skipping SHAP analysis.")
                print("To install SHAP, run: pip install shap")
                self.shap_results = None
            except Exception as e:
                print(f"\nWARNING: Error during SHAP analysis: {str(e)}")
                print("Continuing without SHAP analysis.")
                self.shap_results = None

        return X_test_numeric, y_test_coord

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
            f'XGBoost: True vs Predicted Aftershock Locations ({self.approach.replace("_", " ").title()})'
        )
        plt.legend(loc="lower left")

        plt.savefig(
            f"xgboost_prediction_results_geographic_{self.approach}.png",
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
            f'XGBoost: 3D Location Error Distribution ({self.approach.replace("_", " ").title()})'
        )
        plt.xlabel("Error (km)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f"xgboost_prediction_error_histogram_{self.approach}.png", dpi=300)

        # Return coordinate errors
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

    def run_complete_workflow(self, perform_shap=True):
        """
        Run the complete analysis workflow with XGBoost

        Args:
            perform_shap: Whether to perform SHAP analysis

        Returns:
            Dictionary with results
        """
        start_time = time.time()

        # Print header
        print("\n" + "=" * 70)
        print(
            f"XGBOOST AFTERSHOCK ANALYSIS WITH {self.approach.upper()} APPROACH".center(
                70
            )
        )
        print(f"USING {self.feature_type.upper()} FEATURES".center(70))
        print("=" * 70)

        # 1. Find the mainshock
        self.find_mainshock()

        # 2. Create relative coordinate dataframe
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

        # 3. Prepare dataset for machine learning
        X, y = self.prepare_dataset()
        print(f"Prepared dataset with {len(X)} samples and {X.shape[1]} features")

        X.to_csv("feature_set.csv")

        # 4. Train XGBoost models (now includes SHAP analysis)
        X_test, y_test = self.train_xgboost_models(X, y, perform_shap=perform_shap)

        # 5. Visualize predictions on a geographic map
        true_abs, pred_abs, errors = self.visualize_predictions_geographic(
            X_test, y_test
        )

        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"\nTotal execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)"
        )

        # Return results dictionary including SHAP results if available
        results = {
            "models": self.models,
            "feature_importances": self.feature_importances,
            "mainshock": self.mainshock,
            "aftershocks_df": self.aftershocks_df,
            "test_results": {
                "true_absolute": true_abs,
                "pred_absolute": pred_abs,
                "errors": errors,
            },
            "validation_results": self.validation_results,
        }

        # Add SHAP results if they exist
        if hasattr(self, "shap_results") and self.shap_results is not None:
            results["shap_results"] = self.shap_results

        return results

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
        np.random.seed(44)
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

        # Check for target leakage
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
            "station_elev",
            "_ranking_station_distance"
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

        # Pop grouping columns before variance threshold
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

        # Apply variance threshold only to numeric columns
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

        # Add back grouping columns
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

    def visualize_untrained_predictions(self):
        """
        Visualize what predictions would look like with an untrained model
        using the mainshock location as baseline (no data leakage)
        """
        print("\n" + "=" * 50)
        print("VISUALIZING UNTRAINED MODEL PREDICTIONS")
        print("=" * 50)

        # Prepare dataset without training models
        self.find_mainshock()
        self.create_relative_coordinate_dataframe()

        print("Using mainshock location as baseline prediction for all aftershocks")
        # In relative coordinates, the mainshock is at (0,0,0)
        mainshock_relative_coords = np.array([0, 0, 0])

        # Get aftershocks (excluding the mainshock itself)
        aftershocks = self.aftershocks_df[~self.aftershocks_df["is_mainshock"]].copy()

        # Convert to absolute coordinates for visualization
        true_absolute = pd.DataFrame(index=aftershocks.index)
        pred_absolute = pd.DataFrame(index=aftershocks.index)

        for i in range(len(aftershocks)):
            # True coordinates (already in absolute form in the dataframe)
            true_absolute.loc[aftershocks.index[i], "lat"] = aftershocks[
                "absolute_lat"
            ].iloc[i]
            true_absolute.loc[aftershocks.index[i], "lon"] = aftershocks[
                "absolute_lon"
            ].iloc[i]
            true_absolute.loc[aftershocks.index[i], "depth"] = aftershocks[
                "absolute_depth"
            ].iloc[i]

            # Predicted coordinates (mainshock location for all)
            pred_absolute.loc[aftershocks.index[i], "lat"] = self.mainshock["latitude"]
            pred_absolute.loc[aftershocks.index[i], "lon"] = self.mainshock["longitude"]
            pred_absolute.loc[aftershocks.index[i], "depth"] = self.mainshock["depth"]

        # Create map visualization
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.Mercator())

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        # Set extent with buffer
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
            label="True Aftershock Locations",
        )

        # Plot untrained predictions (all at the mainshock location)
        ax.scatter(
            pred_absolute["lon"],
            pred_absolute["lat"],
            c="red",
            s=30,
            alpha=0.7,
            marker="x",
            transform=ccrs.PlateCarree(),
            label="Baseline Prediction (Mainshock)",
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

        plt.title("Untrained Model")
        plt.legend(loc="lower left")

        plt.savefig("untrained_model_predictions.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Calculate errors
        earth_radius = 6371.0  # km

        # Calculate differences in degrees
        lat_diff_deg = np.abs(true_absolute["lat"] - pred_absolute["lat"])
        lon_diff_deg = np.abs(true_absolute["lon"] - pred_absolute["lon"])

        # Convert to approximate distances in km
        lat_diff_km = lat_diff_deg * (np.pi / 180) * earth_radius
        avg_lat = (true_absolute["lat"] + pred_absolute["lat"]) / 2
        lon_diff_km = (
            lon_diff_deg * (np.pi / 180) * earth_radius * np.cos(np.radians(avg_lat))
        )

        # Depth difference
        depth_diff_km = np.abs(true_absolute["depth"] - pred_absolute["depth"])

        # 3D distance
        distance_3d_km = np.sqrt(lat_diff_km**2 + lon_diff_km**2 + depth_diff_km**2)

        # Print statistics
        print("\nMainshock Baseline Error Statistics:")
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
        plt.title("Mainshock Baseline: 3D Location Error Distribution")
        plt.xlabel("Error (km)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig("mainshock_baseline_error_histogram.png", dpi=300)
        plt.close()

        return {
            "true_absolute": true_absolute,
            "pred_absolute": pred_absolute,
            "errors": {
                "3d": distance_3d_km,
                "lat": lat_diff_km,
                "lon": lon_diff_km,
                "depth": depth_diff_km,
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for aftershock location prediction"
    )
    parser.add_argument(
        "--data", required=True, help="Path to pickle file with preprocessed data"
    )
    parser.add_argument(
        "--feature-type",
        choices=["all", "physics", "signal"],
        default="all",
        help="Type of features to use (default: all; 'compare-shap' performs SHAP analysis across feature types)",
    )
    parser.add_argument(
        "--results-dir",
        default="xgboost_results",
        help="Directory to save results (default: xgboost_results)",
    )
    parser.add_argument(
        "--shap", action="store_true", help="Perform SHAP analysis on trained models"
    )

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    predictor = XGBoostAfterShockPredictor(
        data_file=args.data,
        validation_level="full",
        approach="multi_station",
        feature_type=args.feature_type,
    )
    results = predictor.run_complete_workflow(perform_shap=args.shap)

    print(f"All results saved to {args.results_dir}/")

    # predictor = XGBoostAfterShockPredictor(
    #     data_file=args.data,
    #     validation_level='none',  # Faster with minimal validation
    #     approach='best_station'   # Either approach works for this test
    #     )

    # # Visualize untrained model predictions
    # untrained_results = predictor.visualize_untrained_predictions()

    # print(f"Results saved")
