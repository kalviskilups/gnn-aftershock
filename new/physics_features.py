#!/usr/bin/env python3
# physics_features.py - Analyze physics vs signal feature contributions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
import shap
import os
import time
import argparse
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon
from functools import partial
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import the XGBoostAfterShockPredictor class
# This assumes the class exists in the xgboost_aftershock_prediction.py file
from xgboost_aftershock_prediction import XGBoostAfterShockPredictor


class FeatureContributionAnalyzer:
    """
    Analyze the contribution of physics-based vs signal-based features
    to aftershock location prediction accuracy
    """
    
    def __init__(self, data_pickle, results_dir="feature_analysis_results"):
        """
        Initialize the analyzer
        
        Args:
            data_pickle: Path to pickle file with preprocessed data
            results_dir: Directory to save results
        """
        self.data_pickle = data_pickle
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize the aftershock predictor with multi-station approach
        self.predictor = XGBoostAfterShockPredictor(
            data_pickle=data_pickle,
            validation_level="full",  # Use critical validation to save time
            approach="multi_station"
        )
        
        # Feature categorization
        self.physics_features = self._get_physics_feature_patterns()
        self.signal_features = self._get_signal_feature_patterns()
        
        # Model results
        self.model_results = {}
        self.shap_values = {}
        self.feature_importance_df = None
        
        # Fixed XGBoost parameters (use the same as in the main script)
        self.xgb_params = {
            'n_estimators': 321,
            'learning_rate': 0.04428532496540518,
            'max_depth': 12,
            'min_child_weight': 5,
            'subsample': 0.7600184177542484,
            'colsample_bytree': 0.9051733564655277,
            'reg_alpha': 0.00683735048615785,
            'reg_lambda': 2.556018273438914e-08,
            'gamma': 4.4393088202742805e-05,
            'random_state': 42
        }
    
    def _get_physics_feature_patterns(self):
        """
        Define patterns for physics-based features
        """
        return [
            # Polarization features
            "_pol_az", "pol_az_", "pol_az$",
            "_pol_inc", "pol_inc_", "pol_inc$",
            "_rect_lin", "rect_lin_", "rect_lin$",
            "_p_s_time_diff", "p_s_time_diff_", "p_s_time_diff$",
            
            # Radiation pattern features
            "radiation_e1_e2_ratio", "radiation_e1_e3_ratio",
            
            # Magnitude-related physical features
            "_PGA", "_PGV", "_PGD", 
            "_arias_intensity", "_CAV",
            "_corner_freq", "_lf_spectral_level",
            
            # Stress-related features
            "_brune_stress", "_hf_lf_ratio", 
            "_p_s_energy_ratio", "_envelope_peak_count", "_envelope_duration"
        ]
    
    def _get_signal_feature_patterns(self):
        """
        Define patterns for signal-based features
        """
        return [
            "_mean", "mean_",
            "_std", "std_",
            "_max", "max_",
            "_min", "min_",
            "_range", "range_",
            "_energy", "energy_",
            "_rms", "rms_",
            "_zero_crossings", "zero_crossings_",
            "_peak_freq", "peak_freq_",
            "_spectral_mean", "spectral_mean_",
            "_spectral_std", "spectral_std_",
            "_band_", "band_",
            "_correlation", "correlation_"
        ]
    
    def _is_physics_feature(self, feature_name):
        """Check if a feature belongs to the physics category"""
        return any(pattern in feature_name for pattern in self.physics_features)
    
    def _is_signal_feature(self, feature_name):
        """Check if a feature belongs to the signal category"""
        return any(pattern in feature_name for pattern in self.signal_features)
    
    def _categorize_features(self, feature_names):
        """
        Categorize features into physics, signal, and other,
        with detection of features that match multiple patterns
        """
        physics = []
        signal = []
        other = []
        
        # Check for ambiguous features (matching both physics and signal patterns)
        ambiguous = []
        
        for feature in feature_names:
            is_physics = self._is_physics_feature(feature)
            is_signal = self._is_signal_feature(feature)
            
            if is_physics and is_signal:
                ambiguous.append(feature)
                # Default ambiguous features to physics category
                physics.append(feature)
            elif is_physics:
                physics.append(feature)
            elif is_signal:
                signal.append(feature)
            else:
                other.append(feature)
        
        # Check for duplicates in each category
        physics_duplicates = set([x for x in physics if physics.count(x) > 1])
        signal_duplicates = set([x for x in signal if signal.count(x) > 1])
        other_duplicates = set([x for x in other if other.count(x) > 1])
        
        # Remove duplicates if found
        physics = list(dict.fromkeys(physics))
        signal = list(dict.fromkeys(signal))
        other = list(dict.fromkeys(other))
        
        # Print warnings for any issues found
        if ambiguous:
            print(f"Warning: Found {len(ambiguous)} features matching both physics and signal patterns:")
            for i, feature in enumerate(ambiguous[:5]):
                print(f"  - {feature}")
            if len(ambiguous) > 5:
                print(f"  - ... and {len(ambiguous) - 5} more")
            print("  These are categorized as physics features by default.")
        
        if physics_duplicates or signal_duplicates or other_duplicates:
            print(f"Warning: Found duplicates in feature categories:")
            if physics_duplicates:
                print(f"  - Physics: {len(physics_duplicates)} duplicates")
            if signal_duplicates:
                print(f"  - Signal: {len(signal_duplicates)} duplicates")
            if other_duplicates:
                print(f"  - Other: {len(other_duplicates)} duplicates")
            print("  Duplicates have been removed.")
        
        print(f"Feature categorization complete:")
        print(f"  - Physics: {len(physics)} features")
        print(f"  - Signal: {len(signal)} features")
        print(f"  - Other: {len(other)} features")
        print(f"  - Total: {len(physics) + len(signal) + len(other)} features")
        
        return {
            "physics": physics,
            "signal": signal,
            "other": other,
        }
    
    def _filter_numeric_features(self, X):
        """
        Filter out non-numeric features that XGBoost can't handle
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with only numeric features
        """
        # Get numeric columns only
        numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
        
        # Remove event_id and other object columns that might cause issues
        non_numeric_columns = [col for col in X.columns if col not in numeric_columns]
        
        if non_numeric_columns:
            print(f"Removing {len(non_numeric_columns)} non-numeric columns: {non_numeric_columns}")
            X_numeric = X[numeric_columns].copy()
        else:
            X_numeric = X.copy()
            
        return X_numeric
    
    def prepare_datasets(self):
        """
        Prepare datasets for different feature sets
        
        Returns:
            Dictionary with X, y for different feature subsets
        """
        print("\n" + "=" * 70)
        print("PREPARING DATASETS WITH DIFFERENT FEATURE SUBSETS".center(70))
        print("=" * 70)
        
        # Initialize predictor for dataset preparation
        self.predictor.find_mainshock()
        self.predictor.create_relative_coordinate_dataframe()
        X, y = self.predictor.prepare_dataset()
        
        # Create a non-groupby copy of y for compatibility with our custom training function
        y_coords = y[["relative_x", "relative_y", "relative_z"]].copy()
        
        # Filter numeric features first to avoid categorization issues
        X_numeric = self._filter_numeric_features(X)
        
        # Get feature categories for numeric features only
        feature_categories = self._categorize_features(X_numeric.columns)
        
        print(f"Total numeric features: {len(X_numeric.columns)}")
        print(f"Physics features: {len(feature_categories['physics'])}")
        print(f"Signal features: {len(feature_categories['signal'])}")
        print(f"Other features: {len(feature_categories['other'])}")
        
        # Save the feature categorization for reference
        with open(f"{self.results_dir}/feature_categories.pkl", "wb") as f:
            pickle.dump(feature_categories, f)
        
        # Create dataset with all features
        X_all = X_numeric.copy()
        
        # Create dataset with only signal features
        X_signal = X_numeric[feature_categories['signal']].copy()
        
        # Create dataset with only physics features
        X_physics = X_numeric[feature_categories['physics']].copy()
        
        # Create dataset with signal and physics features
        signal_physics_columns = feature_categories['signal'] + feature_categories['physics']
        X_signal_physics = X_numeric[signal_physics_columns].copy()
        
        # Ensure group column is included for GroupKFold
        group_col = "event_id" if "event_id" in y.columns else "event_date"
        groups = y[group_col].copy()
        
        datasets = {
            "signal": (X_signal, y_coords, groups),
            "physics": (X_physics, y_coords, groups),
            "signal_physics": (X_signal_physics, y_coords, groups)
        }
        
        return datasets
    
    def train_evaluate_model(self, X, y, groups, model_name="all", n_splits=5):
        """
        Train and evaluate XGBoost model with specified features
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame with coordinates
            groups: GroupKFold groups
            model_name: Name for this model
            n_splits: Number of GroupKFold splits
        
        Returns:
            Dictionary with model results
        """
        print(f"\nTraining and evaluating model: {model_name}")
        print(f"Features: {X.shape[1]}")
        
        # Double-check for non-numeric features
        X = self._filter_numeric_features(X)
        
        # Initialize metrics storage
        results = {
            "model_name": model_name,
            "errors_x": [],  # Mean absolute error in X direction
            "errors_y": [],  # Mean absolute error in Y direction
            "errors_z": [],  # Mean absolute error in Z direction
            "errors_3d": [],  # Mean 3D Euclidean distance error
            "median_errors_3d": [],  # Median 3D Euclidean distance error
            "rmse_x": [],  # Keep RMSE for additional analysis
            "rmse_y": [],
            "rmse_z": [],
            "rmse_3d": [],
            "r2_x": [],
            "r2_y": [],
            "r2_z": [],
            "models": [],
            "feature_importance": defaultdict(list),
            "train_indices": [],
            "test_indices": []
        }
        
        # Check for empty dataset
        if X.shape[1] == 0:
            print(f"Warning: No features available for {model_name} model. Skipping.")
            return results
        
        # Initialize GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        
        # For SHAP analysis, store the first split's data
        first_split = True
        
        # Train and evaluate on each fold
        for fold, (train_idx, test_idx) in enumerate(tqdm(gkf.split(X, y, groups), 
                                                    total=n_splits, 
                                                    desc=f"Folds")):
            # Split the data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Record indices for possible later use
            results["train_indices"].append(train_idx)
            results["test_indices"].append(test_idx)
            
            # Create and train XGBoost model
            base_xgb = XGBRegressor(**self.xgb_params)
            multi_model = MultiOutputRegressor(base_xgb)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    multi_model.fit(X_train, y_train)
                
                # Store the models
                results["models"].append(multi_model)
                
                # Make predictions
                y_pred = multi_model.predict(X_test)
                
                # Calculate errors for each coordinate
                for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
                    # Mean absolute error
                    mae = np.mean(np.abs(y_test[coord].values - y_pred[:, i]))
                    results[f"errors_{coord[-1]}"].append(mae)
                    
                    # RMSE (keep for additional analysis)
                    mse = mean_squared_error(y_test[coord], y_pred[:, i])
                    rmse = np.sqrt(mse)
                    results[f"rmse_{coord[-1]}"].append(rmse)
                    
                    # R² score
                    ss_tot = np.sum((y_test[coord] - y_test[coord].mean())**2)
                    ss_res = np.sum((y_test[coord] - y_pred[:, i])**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    results[f"r2_{coord[-1]}"].append(r2)
                    
                    # Store feature importance for this coordinate
                    xgb_model = multi_model.estimators_[i]
                    for j, feature in enumerate(X.columns):
                        results["feature_importance"][coord].append({
                            "feature": feature,
                            "importance": xgb_model.feature_importances_[j],
                            "fold": fold
                        })
                
                # Calculate 3D Euclidean distance error
                error_3d = np.sqrt(
                    (y_pred[:, 0] - y_test["relative_x"])**2 +
                    (y_pred[:, 1] - y_test["relative_y"])**2 +
                    (y_pred[:, 2] - y_test["relative_z"])**2
                )
                
                # Mean 3D error
                results["errors_3d"].append(np.mean(error_3d))
                
                # Median 3D error
                results["median_errors_3d"].append(np.median(error_3d))
                
                # RMSE 3D (keep for additional analysis)
                results["rmse_3d"].append(np.sqrt(np.mean(error_3d**2)))
                
                # Store the data for the first fold for SHAP analysis
                if first_split and model_name != "physics":  # Skip physics-only model for SHAP to save time
                    self.shap_values[model_name] = {
                        "X_train": X_train.copy(),
                        "X_test": X_test.copy(),
                        "y_test": y_test.copy(),
                        "y_pred": y_pred.copy(),
                        "multi_model": multi_model
                    }
                    first_split = False
                    
            except Exception as e:
                print(f"Error in fold {fold} for {model_name} model: {e}")
                continue
        
        # Check if we have any valid results
        if not results["errors_3d"]:
            print(f"Warning: No valid results for {model_name} model.")
            return results
        
        # Calculate aggregate metrics - USING SAME NAMING AS ORIGINAL SCRIPT
        # This will make the comparison method match the original output format
        results["mean_errors_x"] = np.mean(results["errors_x"])
        results["std_errors_x"] = np.std(results["errors_x"])
        results["mean_errors_y"] = np.mean(results["errors_y"])
        results["std_errors_y"] = np.std(results["errors_y"])
        results["mean_errors_z"] = np.mean(results["errors_z"])
        results["std_errors_z"] = np.std(results["errors_z"])
        results["mean_errors_3d"] = np.mean(results["errors_3d"])
        results["std_errors_3d"] = np.std(results["errors_3d"])
        results["mean_median_errors_3d"] = np.mean(results["median_errors_3d"])
        results["std_median_errors_3d"] = np.std(results["median_errors_3d"])
        
        # Also keep the RMSE metrics
        results["mean_rmse_x"] = np.mean(results["rmse_x"])
        results["std_rmse_x"] = np.std(results["rmse_x"])
        results["mean_rmse_y"] = np.mean(results["rmse_y"])
        results["std_rmse_y"] = np.std(results["rmse_y"])
        results["mean_rmse_z"] = np.mean(results["rmse_z"])
        results["std_rmse_z"] = np.std(results["rmse_z"])
        results["mean_rmse_3d"] = np.mean(results["rmse_3d"])
        results["std_rmse_3d"] = np.std(results["rmse_3d"])
        
        # And the R² metrics
        results["mean_r2_x"] = np.mean(results["r2_x"])
        results["std_r2_x"] = np.std(results["r2_x"])
        results["mean_r2_y"] = np.mean(results["r2_y"])
        results["std_r2_y"] = np.std(results["r2_y"])
        results["mean_r2_z"] = np.mean(results["r2_z"])
        results["std_r2_z"] = np.std(results["r2_z"])
        
        # Print summary results
        print(f"\nResults for {model_name} model:")
        print(f"Mean X Error: {results['mean_errors_x']:.2f} km ± {results['std_errors_x']:.2f}")
        print(f"Mean Y Error: {results['mean_errors_y']:.2f} km ± {results['std_errors_y']:.2f}")
        print(f"Mean Z Error: {results['mean_errors_z']:.2f} km ± {results['std_errors_z']:.2f}")
        print(f"Mean 3D Error: {results['mean_errors_3d']:.2f} km ± {results['std_errors_3d']:.2f}")
        print(f"Median 3D Error: {results['mean_median_errors_3d']:.2f} km ± {results['std_median_errors_3d']:.2f}")
        print(f"Mean R² X: {results['mean_r2_x']:.4f} ± {results['std_r2_x']:.4f}")
        print(f"Mean R² Y: {results['mean_r2_y']:.4f} ± {results['std_r2_y']:.4f}")
        print(f"Mean R² Z: {results['mean_r2_z']:.4f} ± {results['std_r2_z']:.4f}")
        
        return results
    
    def calculate_shap_values(self, model_name="all", max_display=20):
        """
        Calculate SHAP values for the model to explain feature category contributions
        (simplified to focus on overall importance rather than individual features)
        
        Args:
            model_name: Name of model to analyze
            max_display: Maximum number of features to display in summary plot
        """
        if model_name not in self.shap_values:
            print(f"No data available for SHAP analysis of model: {model_name}")
            return None
        
        print(f"\nCalculating SHAP values for model: {model_name}")
        data = self.shap_values[model_name]
        
        # We'll calculate SHAP values for each target dimension
        shap_data = {}
        
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            # Get the estimator for this coordinate
            xgb_model = data["multi_model"].estimators_[i]
            
            # Calculate SHAP values - use a sample of the test data if it's large
            X_sample = data["X_test"]
            if len(X_sample) > 200:
                X_sample = X_sample.sample(200, random_state=42)
            
            # Create explainer
            try:
                explainer = shap.TreeExplainer(xgb_model)
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_sample)
                shap_data[coord] = {
                    "shap_values": shap_values,
                    "X_sample": X_sample,
                    "explainer": explainer
                }
                
                # Create SHAP summary plot (bar plot of feature importance)
                plt.figure(figsize=(10, 12))
                shap.summary_plot(
                    shap_values, 
                    X_sample, 
                    plot_type="bar", 
                    max_display=max_display,
                    show=False
                )
                plt.title(f"Feature SHAP Values for {coord} Prediction ({model_name} model)")
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/shap_summary_{model_name}_{coord}.png", dpi=300, bbox_inches="tight")
                plt.close()
                
                print(f"Saved SHAP summary plot for {coord}")
                    
            except Exception as e:
                print(f"Error calculating SHAP values for {coord}: {e}")
                continue
        
        return shap_data
    
    def analyze_feature_categories(self, model_name="all"):
        """
        Analyze contribution of feature categories based on SHAP values
        
        Args:
            model_name: Name of model to analyze
        """
        print(f"\nAnalyzing feature category contributions for model: {model_name}")
        
        # Ensure we have SHAP values for this model
        if model_name not in self.shap_values:
            print(f"No SHAP data available for model: {model_name}")
            return None
                
        # Get coordinate-specific SHAP data
        coord_results = {}
        
        # Process each coordinate
        for coord in ["relative_x", "relative_y", "relative_z"]:
            try:
                # Check if SHAP values exist for this coordinate
                if coord not in self.shap_values.get(model_name, {}):
                    print(f"No SHAP data available for {coord} in model: {model_name}")
                    continue
                    
                shap_data = self.shap_values[model_name][coord]
                
                if "shap_values" not in shap_data:
                    print(f"No SHAP values available for {coord}")
                    continue
                    
                # Get feature names
                features = shap_data["X_sample"].columns
                
                # Categorize features
                physics_idx = [i for i, feature in enumerate(features) if self._is_physics_feature(feature)]
                signal_idx = [i for i, feature in enumerate(features) if self._is_signal_feature(feature)]
                other_idx = [i for i, feature in enumerate(features) if not self._is_physics_feature(feature) and not self._is_signal_feature(feature)]
                
                # Calculate mean absolute SHAP values for each category
                abs_shap_values = np.abs(shap_data["shap_values"])
                
                # Get top 20 feature indices by mean absolute SHAP value
                top_indices = np.argsort(-np.mean(abs_shap_values, axis=0))[:20]
                
                # Count physics and signal features in top 20
                top_physics = sum(1 for idx in top_indices if idx in physics_idx)
                top_signal = sum(1 for idx in top_indices if idx in signal_idx)
                top_other = sum(1 for idx in top_indices if idx in other_idx)
                
                # Calculate total absolute SHAP contribution by category
                physics_contribution = np.sum(abs_shap_values[:, physics_idx]) if len(physics_idx) > 0 else 0
                signal_contribution = np.sum(abs_shap_values[:, signal_idx]) if len(signal_idx) > 0 else 0
                other_contribution = np.sum(abs_shap_values[:, other_idx]) if len(other_idx) > 0 else 0
                
                total_contribution = physics_contribution + signal_contribution + other_contribution
                
                # Calculate percentage contributions
                if total_contribution > 0:
                    physics_pct = (physics_contribution / total_contribution) * 100
                    signal_pct = (signal_contribution / total_contribution) * 100
                    other_pct = (other_contribution / total_contribution) * 100
                else:
                    physics_pct = signal_pct = other_pct = 0
                        
                # Store results
                coord_results[coord] = {
                    "physics_contribution": physics_contribution,
                    "signal_contribution": signal_contribution,
                    "other_contribution": other_contribution,
                    "physics_percentage": physics_pct,
                    "signal_percentage": signal_pct,
                    "other_percentage": other_pct,
                    "physics_count": len(physics_idx),
                    "signal_count": len(signal_idx),
                    "other_count": len(other_idx),
                    "top20_physics_count": top_physics,
                    "top20_signal_count": top_signal,
                    "top20_other_count": top_other
                }
                
                print(f"\nFeature category contributions for {coord}:")
                print(f"Physics features ({len(physics_idx)}): {physics_pct:.1f}% of total SHAP impact")
                print(f"Signal features ({len(signal_idx)}): {signal_pct:.1f}% of total SHAP impact")
                print(f"Other features ({len(other_idx)}): {other_pct:.1f}% of total SHAP impact")
                print(f"\nTop 20 features by importance:")
                print(f"Physics features: {top_physics}/20 ({top_physics/20*100:.1f}%)")
                print(f"Signal features: {top_signal}/20 ({top_signal/20*100:.1f}%)")
                print(f"Other features: {top_other}/20 ({top_other/20*100:.1f}%)")
                
                # Create a pie chart of total contributions
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.pie(
                    [physics_pct, signal_pct, other_pct],
                    labels=["Physics", "Signal", "Other"],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#ff9999','#66b3ff','#99ff99']
                )
                plt.title(f"Total SHAP Value Distribution\nfor {coord} Prediction")
                plt.axis('equal')
                
                # Create a pie chart of top 20 features
                plt.subplot(1, 2, 2)
                plt.pie(
                    [top_physics, top_signal, top_other],
                    labels=["Physics", "Signal", "Other"],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#ff9999','#66b3ff','#99ff99']
                )
                plt.title(f"Feature Types in Top 20\nfor {coord} Prediction")
                plt.axis('equal')
                
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/category_contribution_{model_name}_{coord}.png", dpi=300, bbox_inches="tight")
                plt.close()
                
                # Create stacked bar chart for feature category summary
                plt.figure(figsize=(10, 6))
                category_counts = [len(physics_idx), len(signal_idx), len(other_idx)]
                category_top20 = [top_physics, top_signal, top_other]
                category_importance = [physics_pct, signal_pct, other_pct]
                
                x = np.arange(3)  # Physics, Signal, Other
                width = 0.25
                
                plt.bar(x - width, [c/sum(category_counts)*100 for c in category_counts], width, 
                    label='% of Total Features', color='#dddddd')
                plt.bar(x, category_importance, width, 
                    label='% of Total Importance', color='#66b3ff')
                plt.bar(x + width, [c/20*100 for c in category_top20], width, 
                    label='% in Top 20 Features', color='#ff9999')
                
                plt.ylabel('Percentage')
                plt.title(f'Feature Category Analysis for {coord} Prediction')
                plt.xticks(x, ['Physics', 'Signal', 'Other'])
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/category_summary_{model_name}_{coord}.png", dpi=300, bbox_inches="tight")
                plt.close()
            
            except Exception as e:
                print(f"Error analyzing feature categories for {coord}: {e}")
                continue
                
        return coord_results
    
    def run_analysis(self, n_splits=5):
        """
        Run the complete analysis workflow
        
        Args:
            n_splits: Number of cross-validation splits
        """
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("FEATURE CONTRIBUTION ANALYSIS".center(70))
        print("=" * 70)
        
        try:
            # 1. Prepare datasets
            datasets = self.prepare_datasets()
            
            # 2. Train and evaluate models with different feature subsets
            for model_name, (X, y, groups) in datasets.items():
                try:
                    self.model_results[model_name] = self.train_evaluate_model(
                        X, y, groups, model_name=model_name, n_splits=n_splits
                    )
                except Exception as e:
                    print(f"Error training {model_name} model: {e}")
                    continue
            
            # 3. Calculate SHAP values for the models
            for model_name in ["all", "signal", "signal_physics"]:
                if model_name in self.model_results and self.model_results[model_name]["models"]:
                    try:
                        shap_data = self.calculate_shap_values(model_name=model_name)
                        if shap_data and model_name in self.shap_values:
                            self.shap_values[model_name].update(shap_data)
                    except Exception as e:
                        print(f"Error calculating SHAP values for {model_name} model: {e}")
                        continue
            
            # 4. Analyze feature category contributions
            category_contributions = {}
            for model_name in ["all", "signal_physics"]:
                if model_name in self.shap_values:
                    try:
                        category_contributions[model_name] = self.analyze_feature_categories(model_name=model_name)
                    except Exception as e:
                        print(f"Error analyzing feature categories for {model_name} model: {e}")
                        continue
            
            # 5. Compare model performances
            if self.model_results:
                self.compare_model_performances()
            
            # 6. Create summary of the most important physics features
            self.summarize_important_physics_features()
            
            # 7. Save all results
            with open(f"{self.results_dir}/feature_analysis_results.pkl", "wb") as f:
                results = {
                    "model_results": self.model_results,
                    "category_contributions": category_contributions
                }
                pickle.dump(results, f)
            
        except Exception as e:
            print(f"Error in analysis workflow: {e}")
        
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        
        return self.model_results
    
    def compare_model_performances(self):
        """
        Compare performances of models with different feature subsets
        using the same metrics as in the original XGBoost Error Comparison
        """
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE COMPARISON".center(70))
        print("=" * 70)
        
        # Extract metrics
        metrics = {}
        for model_name, results in self.model_results.items():
            if "mean_errors_3d" in results:
                metrics[model_name] = {
                    # These match the original metrics from XGBoost Error Comparison
                    "X Error": results["mean_errors_x"],
                    "Y Error": results["mean_errors_y"],
                    "Z Error": results["mean_errors_z"],
                    "3D Mean Error": results["mean_errors_3d"],
                    "3D Median Error": results["mean_median_errors_3d"],
                    
                    # Standard deviations
                    "std_x": results["std_errors_x"],
                    "std_y": results["std_errors_y"],
                    "std_z": results["std_errors_z"],
                    "std_3d": results["std_errors_3d"],
                    "std_3d_median": results["std_median_errors_3d"],
                    
                    # Additional metrics
                    "mean_r2_x": results.get("mean_r2_x", 0),
                    "mean_r2_y": results.get("mean_r2_y", 0),
                    "mean_r2_z": results.get("mean_r2_z", 0),
                    "feature_count": len(results["models"][0].estimators_[0].feature_names_in_) if results["models"] else 0
                }
        
        if not metrics:
            print("No valid model results to compare")
            return
        
        # Create comparison DataFrame
        comparison = pd.DataFrame(metrics).T
        
        # Sort by 3D Mean Error
        comparison = comparison.sort_values("3D Mean Error")
        
        # Calculate improvement percentages relative to signal-only
        if "signal" in comparison.index:
            signal_error = comparison.loc["signal", "3D Mean Error"]
            
            comparison["Improvement (%)"] = 0.0  # Initialize column
            
            for model in comparison.index:
                if model != "signal":
                    model_error = comparison.loc[model, "3D Mean Error"]
                    improvement = ((signal_error - model_error) / signal_error) * 100
                    comparison.loc[model, "Improvement (%)"] = improvement
            
            comparison.loc["signal", "Improvement (%)"] = 0.0
        
        # Print comparison in exactly the same format as the original XGBoost Error Comparison
        print("\nXGBoost Error Comparison:")
        
        # Select and reorder columns to match the original output
        display_cols = ["X Error", "Y Error", "Z Error", "3D Mean Error", "3D Median Error"]
        if "Improvement (%)" in comparison.columns:
            display_cols.append("Improvement (%)")
        
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(comparison[display_cols])
        
        # Save the detailed comparison table with all metrics
        full_comparison = comparison.copy()
        comparison.to_csv(f"{self.results_dir}/model_performance_comparison.csv")
        
        # Perform statistical test (Wilcoxon signed-rank test)
        if "signal" in self.model_results and "signal_physics" in self.model_results:
            if "errors_3d" in self.model_results["signal"] and "errors_3d" in self.model_results["signal_physics"]:
                signal_errors = self.model_results["signal"]["errors_3d"]
                physics_signal_errors = self.model_results["signal_physics"]["errors_3d"]
                
                # Ensure equal lengths and enough samples
                min_len = min(len(signal_errors), len(physics_signal_errors))
                
                if min_len > 4:  # Need at least 5 samples for meaningful test
                    stat, p_value = wilcoxon(
                        signal_errors[:min_len],
                        physics_signal_errors[:min_len]
                    )
                    
                    print(f"\nStatistical Test (Signal vs Signal+Physics):")
                    print(f"Wilcoxon signed-rank test: statistic={stat:.4f}, p-value={p_value:.4f}")
                    
                    if p_value < 0.05:
                        print("The difference is statistically significant (p < 0.05)")
                    else:
                        print("The difference is not statistically significant (p >= 0.05)")
                else:
                    print("Not enough samples for statistical testing")
        
        # Create bar chart of error values (matching the original metrics)
        plt.figure(figsize=(12, 8))
        
        models = comparison.index
        x = np.arange(len(models))
        width = 0.2
        
        plt.bar(x - width*2, comparison["X Error"], width, label='X Error', yerr=comparison["std_x"], capsize=5)
        plt.bar(x - width, comparison["Y Error"], width, label='Y Error', yerr=comparison["std_y"], capsize=5)
        plt.bar(x, comparison["Z Error"], width, label='Z Error', yerr=comparison["std_z"], capsize=5)
        plt.bar(x + width, comparison["3D Mean Error"], width, label='3D Mean Error', yerr=comparison["std_3d"], capsize=5)
        plt.bar(x + width*2, comparison["3D Median Error"], width, label='3D Median Error', yerr=comparison["std_3d_median"], capsize=5)
        
        plt.xlabel('Model')
        plt.ylabel('Error (km)')
        plt.title('Error Comparison Across Feature Subsets')
        plt.xticks(x, [f"{m}\n({full_comparison.loc[m, 'feature_count']} features)" for m in models])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add improvement annotations
        if "Improvement (%)" in comparison.columns:
            for i, model in enumerate(models):
                if model != "signal":
                    improvement = comparison.loc[model, "Improvement (%)"]
                    if pd.notna(improvement) and improvement > 0:
                        plt.text(
                            i + width, 
                            comparison.loc[model, "3D Mean Error"] - 2,
                            f"+{improvement:.1f}%",
                            ha='center',
                            va='bottom',
                            color='green',
                            fontweight='bold'
                        )
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/model_performance_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        return comparison
    
    def summarize_important_physics_features(self, n_top=10):
        """
        Create summary of the most important physics features
        
        Args:
            n_top: Number of top features to highlight
        """
        print("\n" + "=" * 70)
        print("TOP PHYSICS FEATURES".center(70))
        print("=" * 70)
        
        # Get the model results for the complete feature set
        if "all" not in self.model_results or not self.model_results["all"].get("models"):
            print("No results available for the complete feature set")
            return
            
        results = self.model_results["all"]
        
        # Collect all feature importances for each coordinate
        importance_dfs = {}
        for coord in ["relative_x", "relative_y", "relative_z"]:
            if coord not in results["feature_importance"]:
                continue
                
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(results["feature_importance"][coord])
            
            if df.empty:
                continue
                
            # Group by feature and average across folds
            avg_importance = df.groupby("feature")["importance"].mean().reset_index()
            avg_importance = avg_importance.sort_values("importance", ascending=False)
            
            # Categorize features
            avg_importance["category"] = avg_importance["feature"].apply(
                lambda f: "Physics" if self._is_physics_feature(f) else 
                         ("Signal" if self._is_signal_feature(f) else "Other")
            )
            
            importance_dfs[coord] = avg_importance
        
        if not importance_dfs:
            print("No feature importance data available")
            return
        
        # Create a combined feature importance table
        try:
            all_features_dfs = []
            
            if "relative_x" in importance_dfs:
                all_features_dfs.append(
                    importance_dfs["relative_x"].rename(columns={"importance": "importance_x"})[["feature", "importance_x", "category"]]
                )
            
            if "relative_y" in importance_dfs:
                all_features_dfs.append(
                    importance_dfs["relative_y"].rename(columns={"importance": "importance_y"})[["feature", "importance_y"]]
                )
            
            if "relative_z" in importance_dfs:
                all_features_dfs.append(
                    importance_dfs["relative_z"].rename(columns={"importance": "importance_z"})[["feature", "importance_z"]]
                )
            
            if not all_features_dfs:
                print("No feature importance data available for any coordinate")
                return
                
            # Start with the first DataFrame
            all_features = all_features_dfs[0].copy()
            
            # Merge with other DataFrames if available
            for df in all_features_dfs[1:]:
                all_features = pd.merge(all_features, df, on="feature", how="outer")
            
            # Fill NaN values with 0
            all_features = all_features.fillna(0)
            
            # Calculate mean importance across coordinates
            importance_columns = [col for col in all_features.columns if col.startswith("importance_")]
            if importance_columns:
                all_features["mean_importance"] = all_features[importance_columns].mean(axis=1)
                all_features = all_features.sort_values("mean_importance", ascending=False)
            
            # Store the feature importance DataFrame
            self.feature_importance_df = all_features.copy()
            all_features.to_csv(f"{self.results_dir}/all_feature_importances.csv", index=False)
            
            # Extract physics features
            physics_features = all_features[all_features["category"] == "Physics"].copy()
            physics_features = physics_features.sort_values("mean_importance", ascending=False)
            
            if physics_features.empty:
                print("No physics features found in importance data")
                return
                
            print("\nTop Physics Features (All Components):")
            pd.set_option('display.float_format', '{:.6f}'.format)
            display_columns = ["feature"]
            display_columns.extend([col for col in ["importance_x", "importance_y", "importance_z", "mean_importance"] if col in physics_features.columns])
            print(physics_features.head(n_top)[display_columns])
            
            # Create detailed bar chart of top physics features
            top_physics = physics_features.head(min(n_top, len(physics_features)))
            
            plt.figure(figsize=(12, max(6, len(top_physics)*0.4)))
            
            x = np.arange(len(top_physics))
            width = 0.25
            
            # Plot bars for each available coordinate
            if "importance_x" in top_physics.columns:
                plt.barh(x - width, top_physics["importance_x"], width, label='X Importance', color='#ff9999')
            if "importance_y" in top_physics.columns:
                plt.barh(x, top_physics["importance_y"], width, label='Y Importance', color='#66b3ff')
            if "importance_z" in top_physics.columns:
                plt.barh(x + width, top_physics["importance_z"], width, label='Z Importance', color='#99ff99')
            
            plt.xlabel('Feature Importance')
            plt.ylabel('Physics Feature')
            plt.title('Top Physics Features by Importance')
            plt.yticks(x, top_physics["feature"])
            plt.legend()
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/top_physics_features.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            # Compare physics feature counts and importance by coordinate
            physics_counts = {}
            physics_total_importance = {}
            
            for coord, col in [("relative_x", "importance_x"), ("relative_y", "importance_y"), ("relative_z", "importance_z")]:
                if col in physics_features.columns:
                    physics_counts[coord] = len(physics_features[physics_features[col] > 0])
                    physics_total_importance[coord] = physics_features[col].sum()
            
            if physics_counts and physics_total_importance:
                plt.figure(figsize=(10, 6))
                coords = list(physics_counts.keys())
                
                # Plot physics feature counts by coordinate
                ax1 = plt.subplot(1, 2, 1)
                ax1.bar(coords, [physics_counts[c] for c in coords], color='#66b3ff')
                ax1.set_title('Physics Features Used by Coordinate')
                ax1.set_ylabel('Count')
                ax1.set_xlabel('Coordinate')
                ax1.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Plot physics feature total importance by coordinate
                ax2 = plt.subplot(1, 2, 2)
                ax2.bar(coords, [physics_total_importance[c] for c in coords], color='#ff9999')
                ax2.set_title('Physics Feature Importance by Coordinate')
                ax2.set_ylabel('Total Importance')
                ax2.set_xlabel('Coordinate')
                ax2.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/physics_features_by_coordinate.png", dpi=300, bbox_inches="tight")
                plt.close()
        
        except Exception as e:
            print(f"Error summarizing important physics features: {e}")
        
        return physics_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the contribution of physics vs signal features in aftershock prediction"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to pickle file with preprocessed data"
    )
    parser.add_argument(
        "--results-dir", default="feature_analysis_results",
        help="Directory to save results (default: feature_analysis_results)"
    )
    parser.add_argument(
        "--splits", type=int, default=5,
        help="Number of cross-validation splits (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Run the analysis
    analyzer = FeatureContributionAnalyzer(
        data_pickle=args.data,
        results_dir=args.results_dir
    )
    
    results = analyzer.run_analysis(n_splits=args.splits)
    
    print("\nAnalysis complete! Results saved to:", args.results_dir)