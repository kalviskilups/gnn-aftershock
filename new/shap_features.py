import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score


class ShapFeatureSelector:
    """
    Feature selector based on SHAP values for aftershock location prediction
    """
    
    def __init__(self, n_features=30, min_features=10, abs_threshold=None, rel_threshold=0.02):
        """
        Initialize the feature selector
        
        Args:
            n_features: Number of top features to select (per coordinate)
            min_features: Minimum number of features to keep regardless of importance
            abs_threshold: Absolute threshold for importance score (features below this are dropped)
            rel_threshold: Relative threshold as fraction of max importance 
                          (features below max_importance * rel_threshold are dropped)
        """
        self.n_features = n_features
        self.min_features = min_features
        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold
        self.selected_features = {}
        self.shap_values = {}
        self.feature_importance = {}
        self.all_selected_features = None
    
    def fit(self, X, y, model, coordinate_names=None):
        """
        Calculate SHAP values and select important features for each coordinate
        
        Args:
            X: Feature dataframe
            y: Target dataframe with coordinates
            model: Trained model (single output or multi-output)
            coordinate_names: List of coordinate names (e.g., ['relative_x', 'relative_y', 'relative_z'])
            
        Returns:
            Set of unique selected features across all coordinates
        """
        # Default coordinate names if not provided
        if coordinate_names is None:
            coordinate_names = ['relative_x', 'relative_y', 'relative_z']
            
        # Initialize SHAP explainer based on model type
        if hasattr(model, 'estimators_'):  # MultiOutputRegressor
            # Handle each estimator separately
            for i, coord in enumerate(coordinate_names):
                estimator = model.estimators_[i]
                self._calculate_shap_for_estimator(estimator, X, coord)
        else:
            # Single output model
            self._calculate_shap_for_estimator(model, X, coordinate_names[0])
            
        # Create union of all selected features
        self.all_selected_features = set()
        for features in self.selected_features.values():
            self.all_selected_features.update(features)
            
        print(f"Total unique features selected across all coordinates: {len(self.all_selected_features)}")
        return self.all_selected_features
    
    def _calculate_shap_for_estimator(self, estimator, X, coordinate):
        """Calculate SHAP values for a single estimator"""
        try:
            print(f"Calculating SHAP values for {coordinate}...")
            # Create SHAP explainer
            explainer = shap.Explainer(estimator)
            
            # Calculate SHAP values
            shap_values = explainer(X)
            
            # Store raw SHAP values
            self.shap_values[coordinate] = shap_values
            
            # Calculate mean absolute SHAP value for each feature
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values.values).mean(0)
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[coordinate] = feature_importance
            
            # Select top features based on criteria
            self._select_features(feature_importance, coordinate)
            
            print(f"Selected {len(self.selected_features[coordinate])} features for {coordinate}")
            
        except Exception as e:
            print(f"Error calculating SHAP values for {coordinate}: {e}")
            # Fallback to feature importance from the model if SHAP fails
            print("Falling back to built-in feature importance")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': estimator.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[coordinate] = feature_importance
            self._select_features(feature_importance, coordinate)
    
    def _select_features(self, feature_importance, coordinate):
        """Select features based on importance thresholds"""
        # Apply criteria for feature selection
        selected = set()
        
        # Always include top min_features
        min_features_set = set(feature_importance.head(self.min_features)['feature'])
        selected.update(min_features_set)
        
        # Apply top N selection (may overlap with min_features)
        if self.n_features:
            top_n_set = set(feature_importance.head(self.n_features)['feature'])
            selected.update(top_n_set)
        
        # Apply absolute threshold
        if self.abs_threshold:
            threshold_set = set(feature_importance[
                feature_importance['importance'] >= self.abs_threshold
            ]['feature'])
            selected.update(threshold_set)
        
        # Apply relative threshold
        if self.rel_threshold and len(feature_importance) > 0:
            max_importance = feature_importance['importance'].max()
            rel_threshold_value = max_importance * self.rel_threshold
            rel_set = set(feature_importance[
                feature_importance['importance'] >= rel_threshold_value
            ]['feature'])
            selected.update(rel_set)
        
        self.selected_features[coordinate] = selected
    
    def transform(self, X):
        """
        Filter the feature dataframe to only include selected features
        
        Args:
            X: Feature dataframe
            
        Returns:
            Filtered dataframe with only selected features
        """
        if self.all_selected_features is None:
            raise ValueError("Must call fit() before transform()")
        
        # Convert set to list for column selection
        selected_cols = list(self.all_selected_features)
        
        # Check if all columns exist in X
        missing_cols = [col for col in selected_cols if col not in X.columns]
        if missing_cols:
            print(f"Warning: {len(missing_cols)} selected columns not found in data")
            selected_cols = [col for col in selected_cols if col in X.columns]
        
        return X[selected_cols]
    
    def fit_transform(self, X, y, model, coordinate_names=None):
        """
        Calculate SHAP values, select features, and transform the data
        
        Args:
            X: Feature dataframe
            y: Target dataframe with coordinates
            model: Trained model
            coordinate_names: List of coordinate names
            
        Returns:
            Filtered dataframe with only selected features
        """
        self.fit(X, y, model, coordinate_names)
        return self.transform(X)
    
    def plot_feature_importance(self, coordinate=None, top_n=20, save_path=None):
        """
        Plot feature importance based on SHAP values
        
        Args:
            coordinate: Specific coordinate to plot (if None, plots all)
            top_n: Number of top features to show
            save_path: Path to save the plot (if None, displays instead)
        """
        if coordinate:
            if coordinate not in self.feature_importance:
                print(f"No feature importance data for {coordinate}")
                return
            
            plt.figure(figsize=(12, top_n * 0.4))
            data = self.feature_importance[coordinate].head(top_n)
            sns.barplot(x='importance', y='feature', data=data)
            plt.title(f'Top {top_n} Features for {coordinate} (SHAP-based)')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_{coordinate}_shap_importance.png", dpi=300)
                plt.close()
        else:
            # Plot all coordinates
            for coord in self.feature_importance.keys():
                self.plot_feature_importance(coord, top_n, save_path)
    
    def plot_shap_summary(self, coordinate=None, max_display=20, save_path=None):
        """
        Create SHAP summary plot
        
        Args:
            coordinate: Specific coordinate to plot
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if coordinate not in self.shap_values:
            print(f"No SHAP values available for {coordinate}")
            return
            
        plt.figure(figsize=(12, 8))
        shap_obj = self.shap_values[coordinate]
        shap.summary_plot(
            shap_obj.values, 
            shap_obj.data, 
            feature_names=shap_obj.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(f'SHAP Summary for {coordinate}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_{coordinate}_shap_summary.png", dpi=300)
            plt.close()
        else:
            plt.show()


def add_shap_to_xgboost_predictor(XGBoostAfterShockPredictor):
    """
    Adds SHAP-based feature selection to the XGBoostAfterShockPredictor class
    
    This is a "monkey patching" approach that adds new methods to the existing class
    """
    
    def train_xgboost_models_with_shap(self, X, y, n_features=30, min_features=10):
        """
        Train XGBoost models with SHAP-based feature selection
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame with coordinates
            n_features: Number of top features to select per coordinate
            min_features: Minimum number of features to keep regardless of importance
            
        Returns:
            X_test, y_test: Test data for evaluation
        """
        print(f"Training XGBoost models with SHAP-based feature selection...")
        print(f"Target: select ~{n_features} features per coordinate (minimum {min_features})")
        
        # Split data into training and testing sets using GroupKFold if available
        if "event_id" in y.columns:
            print("Using GroupKFold with event_id as the group to prevent data leakage...")
            from sklearn.model_selection import GroupKFold
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
            print("Using GroupKFold with event_date as the group to prevent temporal leakage...")
            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=10)
            groups = y["event_date"]
            train_idx, test_idx = next(gkf.split(X, y, groups))
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Drop the group columns before modeling
            if "event_date" in X_train.columns:
                X_train = X_train.drop("event_date", axis=1)
            if "event_date" in X_test.columns:
                X_test = X_test.drop("event_date", axis=1)
                
            # Also drop from y for modeling (keep a copy for evaluation)
            y_train_coord = y_train.drop("event_date", axis=1)
            y_test_coord = y_test.drop("event_date", axis=1)
        else:
            # Standard random split if no grouping columns available
            from sklearn.model_selection import train_test_split
            X_temp = X.copy()
            y_temp = y.copy()
            X_train, X_test, y_train, y_test = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42
            )
            y_train_coord = y_train
            y_test_coord = y_test
        
        # Filter numeric columns for model training
        numeric_columns = [
            col for col in X_train.columns 
            if pd.api.types.is_numeric_dtype(X_train[col])
        ]
        non_numeric_columns = [
            col for col in X_train.columns if col not in numeric_columns
        ]
        
        if non_numeric_columns:
            print(f"Removing {len(non_numeric_columns)} non-numeric columns from training data: {non_numeric_columns}")
            X_train_numeric = X_train[numeric_columns]
            X_test_numeric = X_test[numeric_columns]
        else:
            X_train_numeric = X_train
            X_test_numeric = X_test
        
        # Fixed XGBoost parameters (best parameters from previous tuning)
        xgb_params = {
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
        
        # Step 1: Train an initial model to calculate SHAP values
        print("\nStep 1: Training initial model to calculate feature importance...")
        base_xgb = XGBRegressor(**xgb_params)
        multi_model_initial = MultiOutputRegressor(base_xgb)
        
        # Train the initial model
        multi_model_initial.fit(
            X_train_numeric, 
            y_train_coord[["relative_x", "relative_y", "relative_z"]]
        )
        
        # Step 2: Calculate SHAP values and select important features
        print("\nStep 2: Calculating SHAP values and selecting important features...")
        feature_selector = ShapFeatureSelector(
            n_features=n_features, 
            min_features=min_features, 
            rel_threshold=0.02
        )
        
        # Calculate SHAP values using the initial model
        coordinate_names = ["relative_x", "relative_y", "relative_z"]
        feature_selector.fit(
            X_train_numeric, 
            y_train_coord[coordinate_names], 
            multi_model_initial,
            coordinate_names
        )
        
        # Filter features based on SHAP importance
        X_train_selected = feature_selector.transform(X_train_numeric)
        X_test_selected = feature_selector.transform(X_test_numeric)
        
        print(f"\nFeature reduction: {X_train_numeric.shape[1]} → {X_train_selected.shape[1]} features")
        
        # Step 3: Train the final model with selected features
        print("\nStep 3: Training final model with selected features...")
        base_xgb_final = XGBRegressor(**xgb_params)
        multi_model_final = MultiOutputRegressor(base_xgb_final)
        
        # Train the final model
        multi_model_final.fit(
            X_train_selected, 
            y_train_coord[["relative_x", "relative_y", "relative_z"]]
        )
        
        # Store the feature selector for later use
        self.feature_selector = feature_selector
        
        # Compare performance: Initial model vs. SHAP-selected model
        print("\nComparing performance: Original model vs. SHAP-selected model")
        
        # Make predictions with both models
        y_pred_initial = multi_model_initial.predict(X_test_numeric)
        y_pred_final = multi_model_final.predict(X_test_selected)
        
        # Calculate errors for both models
        initial_ml_errors = {}
        final_ml_errors = {}
        
        print("\nModel Performance (R²):")
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            # R2 scores
            r2_initial = r2_score(y_test_coord[coord], y_pred_initial[:, i])
            r2_final = r2_score(y_test_coord[coord], y_pred_final[:, i])
            
            print(f"  {coord}:")
            print(f"    - Original model: {r2_initial:.4f}")
            print(f"    - SHAP-selected model: {r2_final:.4f}")
            
            # RMSE
            initial_mse = mean_squared_error(y_test_coord[coord], y_pred_initial[:, i])
            final_mse = mean_squared_error(y_test_coord[coord], y_pred_final[:, i])
            
            initial_ml_errors[coord] = np.sqrt(initial_mse)
            final_ml_errors[coord] = np.sqrt(final_mse)
        
        # Calculate 3D distance errors
        initial_ml_3d_errors = np.sqrt(
            (y_pred_initial[:, 0] - y_test_coord["relative_x"]) ** 2
            + (y_pred_initial[:, 1] - y_test_coord["relative_y"]) ** 2
            + (y_pred_initial[:, 2] - y_test_coord["relative_z"]) ** 2
        )
        initial_ml_errors["3d_distance"] = np.mean(initial_ml_3d_errors)
        
        final_ml_3d_errors = np.sqrt(
            (y_pred_final[:, 0] - y_test_coord["relative_x"]) ** 2
            + (y_pred_final[:, 1] - y_test_coord["relative_y"]) ** 2
            + (y_pred_final[:, 2] - y_test_coord["relative_z"]) ** 2
        )
        final_ml_errors["3d_distance"] = np.mean(final_ml_3d_errors)
        
        print("\nModel Performance (RMSE):")
        for coord in ["relative_x", "relative_y", "relative_z", "3d_distance"]:
            print(f"  {coord}:")
            print(f"    - Original model: {initial_ml_errors[coord]:.2f} km")
            print(f"    - SHAP-selected model: {final_ml_errors[coord]:.2f} km")
            
            # Calculate percent improvement
            percent_change = (initial_ml_errors[coord] - final_ml_errors[coord]) / initial_ml_errors[coord] * 100
            if percent_change > 0:
                print(f"    - Improvement: {percent_change:.2f}%")
            else:
                print(f"    - Change: {percent_change:.2f}%")
        
        # Plot feature importance for the final model
        output_dir = "xgboost_results"
        feature_selector.plot_feature_importance(
            save_path=f"{output_dir}/shap_feature_importance"
        )
        
        # Try to generate SHAP summary plots for each coordinate
        try:
            for coord in ["relative_x", "relative_y", "relative_z"]:
                feature_selector.plot_shap_summary(
                    coordinate=coord,
                    save_path=f"{output_dir}/shap_summary"
                )
        except Exception as e:
            print(f"Warning: Could not generate SHAP summary plots: {e}")
        
        # Store the models
        self.models = {
            "multi_output": multi_model_final, 
            "type": "xgboost_shap_selected",
            "initial_model": multi_model_initial
        }
        self.scaler = None  # No scaler used with XGBoost
        self.selected_features = feature_selector.all_selected_features
        
        # Store feature importance
        self.feature_importances = {}
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            self.feature_importances[coord] = feature_selector.feature_importance[coord]
        
        return X_test_selected, y_test_coord

    def run_complete_workflow_with_shap(self, n_features=30, min_features=10):
        """
        Run the complete analysis workflow with XGBoost and SHAP-based feature selection
        
        Args:
            n_features: Number of top features to select per coordinate
            min_features: Minimum number of features to keep regardless of importance
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Print header
        print("\n" + "=" * 70)
        title = f"XGBOOST AFTERSHOCK ANALYSIS WITH {self.approach.upper()} APPROACH AND SHAP SELECTION"
        print(title.center(70))
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
        
        # 4. Train XGBoost models with SHAP feature selection
        X_test, y_test = self.train_xgboost_models_with_shap(
            X, y, n_features=n_features, min_features=min_features
        )
        
        # 5. Visualize predictions on a geographic map
        true_abs, pred_abs, errors = self.visualize_predictions_geographic(X_test, y_test)
        
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"\nTotal execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)"
        )
        
        return {
            "models": self.models,
            "feature_importances": self.feature_importances,
            "feature_selector": getattr(self, "feature_selector", None),
            "selected_features": getattr(self, "selected_features", None),
            "mainshock": self.mainshock,
            "aftershocks_df": self.aftershocks_df,
            "test_results": {
                "true_absolute": true_abs,
                "pred_absolute": pred_abs,
                "errors": errors,
            },
            "validation_results": self.validation_results,
        }
    
    # Add the new methods to the class
    XGBoostAfterShockPredictor.train_xgboost_models_with_shap = train_xgboost_models_with_shap
    XGBoostAfterShockPredictor.run_complete_workflow_with_shap = run_complete_workflow_with_shap
    
    return XGBoostAfterShockPredictor


def compare_approaches_with_shap(data_file, validation_level="full", results_dir="xgboost_results"):
    """
    Compare best-station and multi-station approaches with SHAP-based feature selection
    
    Args:
        data_file: Path to pickle file with preprocessed data
        validation_level: Level of validation checks to perform
        results_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    # Create results directory if it doesn't exist
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("COMPARING APPROACHES WITH XGBOOST AND SHAP FEATURE SELECTION".center(70))
    print("=" * 70)
    
    # Import the original predictor class
    from xgboost_aftershock_prediction import XGBoostAfterShockPredictor
    
    # Enhance the class with SHAP methods
    XGBoostAfterShockPredictor = add_shap_to_xgboost_predictor(XGBoostAfterShockPredictor)
    
    # 1. Run with best-station approach
    print("\nRunning best-station approach with SHAP feature selection...")
    predictor_best = XGBoostAfterShockPredictor(
        data_file=data_file,
        validation_level=validation_level,
        approach="best_station",
    )
    results_best = predictor_best.run_complete_workflow_with_shap(n_features=25, min_features=5)
    
    # 2. Run with multi-station approach
    print("\nRunning multi-station approach with SHAP feature selection...")
    predictor_multi = XGBoostAfterShockPredictor(
        data_file=data_file,
        validation_level=validation_level,
        approach="multi_station",
    )
    results_multi = predictor_multi.run_complete_workflow_with_shap(n_features=25, min_features=5)
    
    # 3. Compare the results
    print("\n" + "=" * 70)
    print("XGBOOST WITH SHAP FEATURE SELECTION - COMPARISON SUMMARY".center(70))
    print("=" * 70)
    
    # Extract error metrics
    best_errors = {
        "relative_x": results_best["test_results"]["errors"]["lon"].mean(),
        "relative_y": results_best["test_results"]["errors"]["lat"].mean(),
        "relative_z": results_best["test_results"]["errors"]["depth"].mean(),
        "3d_distance": results_best["test_results"]["errors"]["3d"].mean(),
        "3d_median": np.median(results_best["test_results"]["errors"]["3d"]),
    }
    
    multi_errors = {
        "relative_x": results_multi["test_results"]["errors"]["lon"].mean(),
        "relative_y": results_multi["test_results"]["errors"]["lat"].mean(),
        "relative_z": results_multi["test_results"]["errors"]["depth"].mean(),
        "3d_distance": results_multi["test_results"]["errors"]["3d"].mean(),
        "3d_median": np.median(results_multi["test_results"]["errors"]["3d"]),
    }
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(
        {
            "Best-Station (SHAP)": [
                best_errors[k]
                for k in [
                    "relative_x",
                    "relative_y",
                    "relative_z",
                    "3d_distance",
                    "3d_median",
                ]
            ],
            "Multi-Station (SHAP)": [
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
        (comparison["Best-Station (SHAP)"] - comparison["Multi-Station (SHAP)"])
        / comparison["Best-Station (SHAP)"]
        * 100
    )
    
    print("\nXGBoost with SHAP Feature Selection - Error Comparison:")
    print(comparison)
    
    # Print feature reduction stats
    print("\nFeature reduction statistics:")
    print(f"  Best-station approach: {len(results_best['selected_features'])} selected features")
    print(f"  Multi-station approach: {len(results_multi['selected_features'])} selected features")
    
    # Visualize comparison
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(comparison.index))
    
    plt.bar(
        index - bar_width / 2,
        comparison["Best-Station (SHAP)"],
        bar_width,
        label="Best-Station (SHAP)",
    )
    plt.bar(
        index + bar_width / 2,
        comparison["Multi-Station (SHAP)"],
        bar_width,
        label="Multi-Station (SHAP)",
    )
    
    # Add improvement annotations
    for i, imp in enumerate(comparison["Improvement (%)"]):
        if imp > 0:  # Only show positive improvements
            plt.text(
                i + bar_width / 2,
                comparison["Multi-Station (SHAP)"].iloc[i] + 1,
                f"+{imp:.1f}%",
                ha="center",
                va="bottom",
                color="green",
                fontweight="bold",
            )
    
    plt.xlabel("Error Metric")
    plt.ylabel("Error (km)")
    plt.title("XGBoost with SHAP Feature Selection: Best-Station vs. Multi-Station Comparison")
    plt.xticks(index, comparison.index)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/xgboost_shap_approach_comparison.png", dpi=300)
    
    # Analyze error distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(
        results_best["test_results"]["errors"]["3d"],
        kde=True,
        color="blue",
        alpha=0.5,
        label="Best-Station (SHAP)",
    )
    sns.histplot(
        results_multi["test_results"]["errors"]["3d"],
        kde=True,
        color="red",
        alpha=0.5,
        label="Multi-Station (SHAP)",
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
    plt.title("XGBoost with SHAP: Error Distribution Comparison")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/xgboost_shap_error_distribution_comparison.png", dpi=300)
    
    print(f"\nComparison results saved to {results_dir}/")
    
    return comparison, results_best, results_multi


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train XGBoost models with SHAP feature selection"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to pickle file with preprocessed data"
    )
    parser.add_argument(
        "--validation",
        choices=["none", "critical", "full"],
        default="full",
        help="Validation level (default: full)"
    )
    parser.add_argument(
        "--approach",
        choices=["best_station", "multi_station", "compare"],
        default="compare",
        help="Analysis approach (default: compare both)"
    )
    parser.add_argument(
        "--results-dir",
        default="xgboost_results",
        help="Directory to save results (default: xgboost_results)"
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=25,
        help="Number of top features to select per coordinate (default: 25)"
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=5,
        help="Minimum number of features to keep (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Import the original predictor class
    from xgboost_aftershock_prediction import XGBoostAfterShockPredictor
    
    # Enhance the class with SHAP methods
    XGBoostAfterShockPredictor = add_shap_to_xgboost_predictor(XGBoostAfterShockPredictor)
    
    if args.approach == "compare":
        # Run comparison of both approaches
        comparison, results_best, results_multi = compare_approaches_with_shap(
            args.data,
            validation_level=args.validation,
            results_dir=args.results_dir
        )
    else:
        # Run single approach
        predictor = XGBoostAfterShockPredictor(
            data_file=args.data,
            validation_level=args.validation,
            approach=args.approach,
        )
        results = predictor.run_complete_workflow_with_shap(
            n_features=args.n_features,
            min_features=args.min_features
        )
    
    print("Analysis complete")