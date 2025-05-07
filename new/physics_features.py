#!/usr/bin/env python3
# physics_feature_analysis.py - Analyze contribution of physics-based features to aftershock prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle
import os
import shap
import time
import argparse
from tqdm import tqdm

# Import the XGBoostAfterShockPredictor class from the main script
from xgboost_aftershock_prediction import XGBoostAfterShockPredictor


class PhysicsFeatureAnalyzer:
    """
    Class for analyzing the contribution of physics-based features
    to aftershock location prediction accuracy
    """

    def __init__(self, data_file, validation_level="critical", approach="best_station", results_dir="physics_feature_analysis"):
        """
        Initialize the analyzer
        
        Args:
            data_file: Path to pickle or HDF5 file with preprocessed data
            validation_level: Level of validation to apply
            approach: Analysis approach to use ("best_station" or "multi_station")
            results_dir: Directory to save results
        """
        self.data_file = data_file
        self.validation_level = validation_level
        self.approach = approach
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize predictor
        self.predictor = XGBoostAfterShockPredictor(
            data_file=data_file,
            validation_level=validation_level,
            approach=approach
        )
        
        # Initialize results storage
        self.results = {
            'all_features': None,
            'no_physics': None,
            'feature_importance': None,
            'shap_values': None
        }
        
        # Define physics-based feature patterns
        self.physics_feature_patterns = [
            # Radiation pattern features
            "radiation_e1_e2_ratio", 
            "radiation_e1_e3_ratio",
            
            # Polarization features
            "pol_az", 
            "pol_inc", 
            "rect_lin",
            
            # Magnitude features
            "_PGA", 
            "_PGV", 
            "_PGD", 
            "_arias_intensity", 
            "_CAV", 
            "_corner_freq", 
            "_lf_spectral_level",
            
            # Stress features
            "_brune_stress", 
            "_hf_lf_ratio", 
            "_p_s_energy_ratio", 
            "_envelope_peak_count", 
            "_envelope_duration"
        ]
        
        print(f"Initialized PhysicsFeatureAnalyzer with approach: {approach}")
        print(f"Results will be saved to: {results_dir}")

    def identify_physics_features(self, X):
        """
        Identify physics-based features in the dataset
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List of physics-based feature names
        """
        physics_features = []
        
        for pattern in self.physics_feature_patterns:
            matching_features = [col for col in X.columns if pattern in col]
            physics_features.extend(matching_features)
        
        # Remove duplicates
        physics_features = list(set(physics_features))
        
        # Sort for better readability
        physics_features.sort()
        
        print(f"Identified {len(physics_features)} physics-based features")
        
        return physics_features

    def prepare_datasets(self):
        """
        Prepare datasets with all features and without physics features
        
        Returns:
            Tuple of (X_all, y_all, X_no_physics, y_no_physics)
        """
        print("\n" + "=" * 50)
        print("PREPARING DATASETS")
        print("=" * 50)
        
        # Find mainshock and prepare dataset
        self.predictor.find_mainshock()
        self.predictor.create_relative_coordinate_dataframe()
        X, y = self.predictor.prepare_dataset()
        
        print(f"Original dataset: {X.shape[1]} features")
        
        # Identify physics features
        physics_features = self.identify_physics_features(X)
        
        # Print some examples of physics features
        print("\nExamples of physics-based features:")
        for feature in physics_features[:10]:
            print(f"  - {feature}")
        if len(physics_features) > 10:
            print(f"  - ... and {len(physics_features) - 10} more")
        
        # Create dataset without physics features
        X_no_physics = X.copy()
        X_no_physics = X_no_physics.drop(columns=physics_features, errors='ignore')
        
        print(f"\nDataset without physics features: {X_no_physics.shape[1]} features")
        print(f"Removed {X.shape[1] - X_no_physics.shape[1]} physics-based features")
        
        return X, y, X_no_physics, y

    def train_models(self):
        """
        Train models with and without physics-based features
        """
        print("\n" + "=" * 50)
        print("TRAINING MODELS WITH AND WITHOUT PHYSICS FEATURES")
        print("=" * 50)
        
        start_time = time.time()
        
        # Prepare datasets
        X_all, y, X_no_physics, _ = self.prepare_datasets()
        
        # Save the list of physics features for future reference
        self.physics_features = self.identify_physics_features(X_all)
        
        # Train with all features
        print("\nTraining model with ALL features...")
        results_all = self.predictor.train_xgboost_models(X_all, y)
        
        # Train without physics features
        print("\nTraining model WITHOUT physics features...")
        
        # Use the original predictor's training method to avoid changing logic
        # But create a new predictor to avoid overwriting the original model
        predictor_no_physics = XGBoostAfterShockPredictor(
            data_file=None,  # No need to load data again
            validation_level=self.validation_level,
            approach=self.approach
        )
        
        # Copy necessary attributes
        predictor_no_physics.data_dict = self.predictor.data_dict
        predictor_no_physics.mainshock = self.predictor.mainshock
        predictor_no_physics.mainshock_key = self.predictor.mainshock_key
        predictor_no_physics.aftershocks_df = self.predictor.aftershocks_df
        predictor_no_physics.validation_results = self.predictor.validation_results
        predictor_no_physics.data_format = self.predictor.data_format
        
        # Train with no physics features
        results_no_physics = predictor_no_physics.train_xgboost_models(X_no_physics, y)
        
        # Store results
        self.results['all_features'] = {
            'predictor': self.predictor,
            'X_test': results_all[0],
            'y_test': results_all[1]
        }
        
        self.results['no_physics'] = {
            'predictor': predictor_no_physics,
            'X_test': results_no_physics[0],
            'y_test': results_no_physics[1]
        }
        
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTraining time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")

    def compare_performance(self):
        """
        Compare performance of the models with and without physics features
        """
        print("\n" + "=" * 50)
        print("COMPARING MODEL PERFORMANCE")
        print("=" * 50)
        
        if self.results['all_features'] is None or self.results['no_physics'] is None:
            print("Models need to be trained first. Running train_models()...")
            self.train_models()
        
        # Get test data and models
        X_test_all = self.results['all_features']['X_test']
        y_test_all = self.results['all_features']['y_test']
        X_test_no_physics = self.results['no_physics']['X_test']
        y_test_no_physics = self.results['no_physics']['y_test']
        
        predictor_all = self.results['all_features']['predictor']
        predictor_no_physics = self.results['no_physics']['predictor']
        
        # Make predictions
        y_pred_all = predictor_all.models['multi_output'].predict(X_test_all)
        y_pred_no_physics = predictor_no_physics.models['multi_output'].predict(X_test_no_physics)
        
        # Calculate errors for each model
        errors_all = {}
        errors_no_physics = {}
        
        # For all features
        for i, coord in enumerate(['relative_x', 'relative_y', 'relative_z']):
            mse = mean_squared_error(y_test_all[coord], y_pred_all[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_all[coord], y_pred_all[:, i])
            errors_all[coord] = {
                'rmse': rmse,
                'r2': r2
            }
        
        # Calculate 3D distance error
        all_3d_errors = np.sqrt(
            (y_pred_all[:, 0] - y_test_all['relative_x']) ** 2 +
            (y_pred_all[:, 1] - y_test_all['relative_y']) ** 2 +
            (y_pred_all[:, 2] - y_test_all['relative_z']) ** 2
        )
        errors_all['3d_distance'] = np.mean(all_3d_errors)
        errors_all['3d_median'] = np.median(all_3d_errors)
        
        # For no physics features
        for i, coord in enumerate(['relative_x', 'relative_y', 'relative_z']):
            mse = mean_squared_error(y_test_no_physics[coord], y_pred_no_physics[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_no_physics[coord], y_pred_no_physics[:, i])
            errors_no_physics[coord] = {
                'rmse': rmse,
                'r2': r2
            }
        
        # Calculate 3D distance error
        no_physics_3d_errors = np.sqrt(
            (y_pred_no_physics[:, 0] - y_test_no_physics['relative_x']) ** 2 +
            (y_pred_no_physics[:, 1] - y_test_no_physics['relative_y']) ** 2 +
            (y_pred_no_physics[:, 2] - y_test_no_physics['relative_z']) ** 2
        )
        errors_no_physics['3d_distance'] = np.mean(no_physics_3d_errors)
        errors_no_physics['3d_median'] = np.median(no_physics_3d_errors)
        
        # Create comparison table
        comparison = pd.DataFrame({
            'All Features': [
                errors_all['relative_x']['rmse'],
                errors_all['relative_y']['rmse'],
                errors_all['relative_z']['rmse'],
                errors_all['3d_distance'],
                errors_all['3d_median'],
                errors_all['relative_x']['r2'],
                errors_all['relative_y']['r2'],
                errors_all['relative_z']['r2']
            ],
            'No Physics': [
                errors_no_physics['relative_x']['rmse'],
                errors_no_physics['relative_y']['rmse'],
                errors_no_physics['relative_z']['rmse'],
                errors_no_physics['3d_distance'],
                errors_no_physics['3d_median'],
                errors_no_physics['relative_x']['r2'],
                errors_no_physics['relative_y']['r2'],
                errors_no_physics['relative_z']['r2']
            ]
        }, index=[
            'X Error (RMSE)', 
            'Y Error (RMSE)', 
            'Z Error (RMSE)', 
            '3D Mean Error',
            '3D Median Error',
            'X (R²)',
            'Y (R²)',
            'Z (R²)'
        ])
        
        # Calculate performance difference (improvement from physics features)
        comparison['Difference'] = comparison['No Physics'] - comparison['All Features']
        
        # For RMSE metrics, positive difference means worse performance without physics
        # For R² metrics, negative difference means worse performance without physics
        rmse_indices = [i for i in comparison.index if 'RMSE' in i or 'Error' in i]
        r2_indices = [i for i in comparison.index if 'R²' in i]
        
        comparison['Relative Impact (%)'] = 0
        
        # Calculate relative improvement for RMSE (positive means physics features improve model)
        for idx in rmse_indices:
            if comparison.loc[idx, 'No Physics'] > 0:  # Avoid division by zero
                comparison.loc[idx, 'Relative Impact (%)'] = (
                    comparison.loc[idx, 'Difference'] / comparison.loc[idx, 'No Physics'] * 100
                )
        
        # Calculate relative improvement for R² 
        for idx in r2_indices:
            if abs(comparison.loc[idx, 'No Physics']) > 0:  # Avoid division by zero
                comparison.loc[idx, 'Relative Impact (%)'] = (
                    -comparison.loc[idx, 'Difference'] / abs(comparison.loc[idx, 'No Physics']) * 100
                )
        
        # Store the comparison for later use
        self.performance_comparison = comparison
        
        # Print the comparison
        print("\nPerformance Comparison:")
        print(comparison)
        
        # Save the comparison to CSV
        comparison.to_csv(f"{self.results_dir}/performance_comparison.csv")
        
        # Visualize comparison of RMSE metrics
        plt.figure(figsize=(12, 8))
        bar_width = 0.35
        index = np.arange(len(rmse_indices))
        
        plt.bar(
            index - bar_width / 2,
            comparison.loc[rmse_indices, 'All Features'],
            bar_width,
            label='With Physics Features',
            color='forestgreen'
        )
        
        plt.bar(
            index + bar_width / 2,
            comparison.loc[rmse_indices, 'No Physics'],
            bar_width,
            label='Without Physics Features',
            color='firebrick'
        )
        
        # Add improvement annotations
        for i, idx in enumerate(rmse_indices):
            imp = comparison.loc[idx, 'Relative Impact (%)']
            if imp > 0:  # Positive impact means physics features help
                plt.text(
                    i,
                    comparison.loc[idx, 'All Features'] - 0.5,
                    f"+{imp:.1f}%",
                    ha='center',
                    va='top',
                    color='forestgreen',
                    fontweight='bold'
                )
            elif imp < 0:  # Negative impact means physics features make things worse
                plt.text(
                    i,
                    comparison.loc[idx, 'All Features'] + 0.5,
                    f"{imp:.1f}%",
                    ha='center',
                    va='bottom',
                    color='firebrick',
                    fontweight='bold'
                )
        
        plt.xlabel('Error Metric')
        plt.ylabel('Error (km)')
        plt.title('Impact of Physics-Based Features on Error Metrics')
        plt.xticks(index, [idx.split(' (')[0] for idx in rmse_indices])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.results_dir}/error_metrics_comparison.png", dpi=300)
        plt.close()
        
        # Visualize comparison of R² metrics
        plt.figure(figsize=(12, 6))
        index = np.arange(len(r2_indices))
        
        plt.bar(
            index - bar_width / 2,
            comparison.loc[r2_indices, 'All Features'],
            bar_width,
            label='With Physics Features',
            color='forestgreen'
        )
        
        plt.bar(
            index + bar_width / 2,
            comparison.loc[r2_indices, 'No Physics'],
            bar_width,
            label='Without Physics Features',
            color='firebrick'
        )
        
        # Add improvement annotations
        for i, idx in enumerate(r2_indices):
            imp = comparison.loc[idx, 'Relative Impact (%)']
            if imp > 0:  # Positive impact means physics features help
                plt.text(
                    i,
                    comparison.loc[idx, 'All Features'] + 0.02,
                    f"+{imp:.1f}%",
                    ha='center',
                    va='bottom',
                    color='forestgreen',
                    fontweight='bold'
                )
            elif imp < 0:  # Negative impact means physics features make things worse
                plt.text(
                    i,
                    comparison.loc[idx, 'All Features'] - 0.02,
                    f"{imp:.1f}%",
                    ha='center',
                    va='top',
                    color='firebrick',
                    fontweight='bold'
                )
        
        plt.xlabel('Coordinate')
        plt.ylabel('R² Score')
        plt.title('Impact of Physics-Based Features on R² Scores')
        plt.xticks(index, [idx.split(' (')[0] for idx in r2_indices])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.results_dir}/r2_scores_comparison.png", dpi=300)
        plt.close()
        
        # Compare error distributions
        plt.figure(figsize=(12, 6))
        sns.histplot(all_3d_errors, kde=True, color='forestgreen', alpha=0.6, label='With Physics Features')
        sns.histplot(no_physics_3d_errors, kde=True, color='firebrick', alpha=0.6, label='Without Physics Features')
        
        plt.axvline(errors_all['3d_distance'], color='forestgreen', linestyle='--', 
                   label=f'With Physics Mean: {errors_all["3d_distance"]:.2f} km')
        plt.axvline(errors_no_physics['3d_distance'], color='firebrick', linestyle='--',
                   label=f'Without Physics Mean: {errors_no_physics["3d_distance"]:.2f} km')
        
        plt.xlabel('3D Error (km)')
        plt.ylabel('Frequency')
        plt.title('Distribution of 3D Location Errors')
        plt.legend()
        plt.savefig(f"{self.results_dir}/error_distribution_comparison.png", dpi=300)
        plt.close()
        
        # Calculate and return the overall impact
        rmse_improvement = comparison.loc['3D Mean Error', 'Relative Impact (%)']
        return {
            'comparison': comparison,
            'overall_impact': rmse_improvement,
            'all_3d_errors': all_3d_errors,
            'no_physics_3d_errors': no_physics_3d_errors
        }

    def analyze_with_shap(self, max_display=20):
        """
        Use SHAP to analyze feature contributions
        
        Args:
            max_display: Maximum number of features to display in plots
        """
        print("\n" + "=" * 50)
        print("ANALYZING FEATURE CONTRIBUTIONS WITH SHAP")
        print("=" * 50)
        
        if self.results['all_features'] is None:
            print("Models need to be trained first. Running train_models()...")
            self.train_models()
        
        # Get test data and model
        X_test = self.results['all_features']['X_test']
        predictor_all = self.results['all_features']['predictor']
        
        # Define coordinate names
        coordinates = ['X (East-West)', 'Y (North-South)', 'Z (Depth)']
        
        # Create SHAP explainer
        shap_results = {}
        
        print("\nCalculating SHAP values for each coordinate prediction...")
        
        for i, coord in enumerate(['relative_x', 'relative_y', 'relative_z']):
            print(f"Analyzing SHAP values for {coord} prediction...")
            
            # Get the individual model for this coordinate
            xgb_model = predictor_all.models['multi_output'].estimators_[i]
            
            # Initialize the explainer
            explainer = shap.Explainer(xgb_model)
            
            # Calculate SHAP values
            shap_values = explainer(X_test)
            
            # Store the SHAP values
            shap_results[coord] = {
                'shap_values': shap_values,
                'explainer': explainer
            }
            
            # Mark physics features
            physics_features = self.identify_physics_features(X_test)
            
            # Create feature type mapping
            feature_types = pd.Series('Signal', index=X_test.columns)
            feature_types[physics_features] = 'Physics'
            
            # Sort features by SHAP value importance
            shap_df = pd.DataFrame({
                'feature': X_test.columns,
                'importance': np.abs(shap_values.values).mean(0),
                'type': feature_types.values
            })
            
            shap_df = shap_df.sort_values('importance', ascending=False)
            
            # Count physics vs. signal features in top N
            top_features = shap_df.head(max_display)
            physics_count = sum(top_features['type'] == 'Physics')
            signal_count = sum(top_features['type'] == 'Signal')
            
            print(f"Top {max_display} features for {coord} prediction:")
            print(f"  - Physics features: {physics_count} ({physics_count/max_display*100:.1f}%)")
            print(f"  - Signal features: {signal_count} ({signal_count/max_display*100:.1f}%)")
            
            # Save top feature importance to CSV
            top_features.to_csv(f"{self.results_dir}/top_features_{coord}.csv")
            
            # Create a custom plot to color by feature type
            fig, ax = plt.subplots(figsize=(12, 12))
            
            y_pos = np.arange(len(top_features))
            colors = ['#ff7f0e' if t == 'Physics' else '#1f77b4' for t in top_features['type']]
            
            ax.barh(y_pos, top_features['importance'], color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{feat} ({'P' if typ == 'Physics' else 'S'})" 
                               for feat, typ in zip(top_features['feature'], top_features['type'])])
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title(f'Top {max_display} Features Impact on {coordinates[i]} Prediction')
            
            # Add a legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff7f0e', label='Physics Feature'),
                Patch(facecolor='#1f77b4', label='Signal Feature')
            ]
            ax.legend(handles=legend_elements)
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/shap_top_features_{coord}.png", dpi=300)
            plt.close()
            
            # Create SHAP beeswarm plot
            plt.figure(figsize=(10, 12))
            shap.plots.beeswarm(
                shap_values,
                max_display=max_display,
                show=False
            )
            plt.title(f'SHAP Beeswarm for {coordinates[i]} Prediction')
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/shap_beeswarm_{coord}.png", dpi=300)
            plt.close()
            
            # Create SHAP waterfall plot for a sample instance
            plt.figure(figsize=(10, 12))
            sample_instance = 0  # Use first test instance
            shap.plots.waterfall(
                shap_values[sample_instance],
                max_display=max_display,
                show=False
            )
            plt.title(f'SHAP Waterfall for {coordinates[i]} (Sample Instance)')
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/shap_waterfall_{coord}.png", dpi=300)
            plt.close()
        
        # Store SHAP results
        self.results['shap_values'] = shap_results
        
        # Analyze overall contribution of physics vs. signal features
        self.analyze_feature_group_contribution()
        
        print(f"\nSHAP analysis complete. Results saved to {self.results_dir}/")

    def analyze_feature_group_contribution(self):
        """
        Analyze the overall contribution of physics vs. signal features
        """
        print("\nAnalyzing overall contribution of physics vs. signal features...")
        
        if 'shap_values' not in self.results or self.results['shap_values'] is None:
            print("SHAP analysis needs to be run first")
            return
        
        # Get test data
        X_test = self.results['all_features']['X_test']
        
        # Identify physics features
        physics_features = self.identify_physics_features(X_test)
        
        # Create a DataFrame to store contribution analysis
        contribution_df = pd.DataFrame(columns=[
            'Coordinate', 'Physics Feature Count', 'Signal Feature Count',
            'Physics Contribution (%)', 'Signal Contribution (%)'
        ])
        
        # Dictionary to store feature importance by type for the stacked bar chart
        feature_importance_by_type = {}
        
        for i, coord in enumerate(['relative_x', 'relative_y', 'relative_z']):
            shap_values = self.results['shap_values'][coord]['shap_values']
            
            # Calculate absolute SHAP values
            abs_shap = np.abs(shap_values.values)
            
            # Get feature importances
            feature_importances = np.mean(abs_shap, axis=0)
            
            # Create feature df with types
            feature_df = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': feature_importances,
                'Type': ['Physics' if f in physics_features else 'Signal' for f in X_test.columns]
            })
            
            # Sort by importance
            feature_df = feature_df.sort_values('Importance', ascending=False)
            
            # Store top 20 features by type for stacked bar chart
            top_features = feature_df.head(20)
            feature_importance_by_type[coord] = top_features
            
            # Calculate contribution by feature type
            physics_contribution = feature_df[feature_df['Type'] == 'Physics']['Importance'].sum()
            signal_contribution = feature_df[feature_df['Type'] == 'Signal']['Importance'].sum()
            total_contribution = physics_contribution + signal_contribution
            
            physics_count = sum(feature_df['Type'] == 'Physics')
            signal_count = sum(feature_df['Type'] == 'Signal')
            
            # Add to DataFrame
            contribution_df = contribution_df.append({
                'Coordinate': coord,
                'Physics Feature Count': physics_count,
                'Signal Feature Count': signal_count,
                'Physics Contribution (%)': physics_contribution / total_contribution * 100,
                'Signal Contribution (%)': signal_contribution / total_contribution * 100
            }, ignore_index=True)
            
            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(
                [physics_contribution, signal_contribution],
                labels=['Physics Features', 'Signal Features'],
                autopct='%1.1f%%',
                colors=['#ff7f0e', '#1f77b4'],
                startangle=90
            )
            plt.axis('equal')
            plt.title(f'Contribution to {coord} Prediction by Feature Type')
            plt.savefig(f"{self.results_dir}/feature_type_contribution_{coord}.png", dpi=300)
            plt.close()
            
            # Create bar chart of top 10 features colored by type
            top_features = feature_df.head(10)
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(
                range(len(top_features)),
                top_features['Importance'],
                color=['#ff7f0e' if t == 'Physics' else '#1f77b4' for t in top_features['Type']]
            )
            
            plt.xticks(
                range(len(top_features)),
                top_features['Feature'],
                rotation=45,
                ha='right'
            )
            
            # Add a legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff7f0e', label='Physics Feature'),
                Patch(facecolor='#1f77b4', label='Signal Feature')
            ]
            plt.legend(handles=legend_elements)
            
            plt.xlabel('Feature')
            plt.ylabel('Mean |SHAP Value|')
            plt.title(f'Top 10 Features for {coord} Prediction')
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/top10_features_{coord}.png", dpi=300)
            plt.close()
        
        # Add average row
        contribution_df = contribution_df.append({
            'Coordinate': 'Average',
            'Physics Feature Count': contribution_df['Physics Feature Count'].mean(),
            'Signal Feature Count': contribution_df['Signal Feature Count'].mean(),
            'Physics Contribution (%)': contribution_df['Physics Contribution (%)'].mean(),
            'Signal Contribution (%)': contribution_df['Signal Contribution (%)'].mean()
        }, ignore_index=True)
        
        # Print and save the contribution analysis
        print("\nFeature Type Contribution Analysis:")
        print(contribution_df)
        
        contribution_df.to_csv(f"{self.results_dir}/feature_type_contribution.csv")
        
        # Create overall contribution pie chart
        plt.figure(figsize=(10, 8))
        avg_physics = contribution_df.loc[contribution_df['Coordinate'] == 'Average', 'Physics Contribution (%)'].values[0]
        avg_signal = contribution_df.loc[contribution_df['Coordinate'] == 'Average', 'Signal Contribution (%)'].values[0]
        
        plt.pie(
            [avg_physics, avg_signal],
            labels=['Physics Features', 'Signal Features'],
            autopct='%1.1f%%',
            colors=['#ff7f0e', '#1f77b4'],
            startangle=90
        )
        plt.axis('equal')
        plt.title('Average Contribution by Feature Type')
        plt.savefig(f"{self.results_dir}/overall_feature_type_contribution.png", dpi=300)
        plt.close()
        
        # Create stacked bar chart for features by type
        for coord in ['relative_x', 'relative_y', 'relative_z']:
            top_features = feature_importance_by_type[coord]
            
            # Convert to long format for seaborn
            top_features_long = pd.DataFrame({
                'Feature': [],
                'Type': [],
                'Importance': []
            })
            
            for _, row in top_features.iterrows():
                top_features_long = top_features_long.append({
                    'Feature': row['Feature'],
                    'Type': row['Type'],
                    'Importance': row['Importance']
                }, ignore_index=True)
            
            plt.figure(figsize=(14, 8))
            sns.barplot(x='Feature', y='Importance', hue='Type', data=top_features_long, dodge=False)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Top 20 Features by Type for {coord} Prediction')
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/features_by_type_{coord}.png", dpi=300)
            plt.close()
        
        self.feature_contribution = contribution_df
        
        return contribution_df

    def run_complete_analysis(self):
        """
        Run the complete feature contribution analysis
        """
        print("\n" + "=" * 70)
        print("COMPLETE PHYSICS FEATURE CONTRIBUTION ANALYSIS".center(70))
        print("=" * 70)
        
        start_time = time.time()
        
        # 1. Train models with and without physics features
        self.train_models()
        
        # 2. Compare model performance
        performance = self.compare_performance()
        
        # 3. Analyze with SHAP
        self.analyze_with_shap()
        
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        
        # Print summary
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY".center(70))
        print("=" * 70)
        
        print(f"Overall impact of physics features on 3D error: {performance['overall_impact']:.1f}%")
        
        avg_physics_contribution = self.feature_contribution.loc[
            self.feature_contribution['Coordinate'] == 'Average', 
            'Physics Contribution (%)'
        ].values[0]
        
        print(f"Overall contribution of physics features to predictions: {avg_physics_contribution:.1f}%")
        
        # Create a summary file
        with open(f"{self.results_dir}/analysis_summary.txt", 'w') as f:
            f.write("PHYSICS FEATURE CONTRIBUTION ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Data file: {self.data_file}\n")
            f.write(f"Approach: {self.approach}\n")
            f.write(f"Validation level: {self.validation_level}\n\n")
            
            f.write("PERFORMANCE COMPARISON\n")
            f.write("-"*50 + "\n")
            f.write(f"Overall impact of physics features on 3D error: {performance['overall_impact']:.1f}%\n\n")
            f.write(str(self.performance_comparison) + "\n\n")
            
            f.write("FEATURE CONTRIBUTION ANALYSIS\n")
            f.write("-"*50 + "\n")
            f.write(f"Overall contribution of physics features to predictions: {avg_physics_contribution:.1f}%\n\n")
            f.write(str(self.feature_contribution) + "\n\n")
            
            f.write("PHYSICS FEATURES IDENTIFIED\n")
            f.write("-"*50 + "\n")
            for feature in self.physics_features:
                f.write(f"- {feature}\n")
        
        print(f"\nSummary saved to {self.results_dir}/analysis_summary.txt")
        
        return {
            'performance': performance,
            'feature_contribution': self.feature_contribution,
            'physics_features': self.physics_features
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze contribution of physics-based features to aftershock prediction"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to pickle or HDF5 file with preprocessed data"
    )
    parser.add_argument(
        "--validation",
        choices=["none", "critical", "full"],
        default="full",
        help="Validation level (default: critical)"
    )
    parser.add_argument(
        "--approach",
        choices=["best_station", "multi_station"],
        default="multi_station",
        help="Analysis approach (default: multi_station)"
    )
    parser.add_argument(
        "--results-dir",
        default="physics_feature_analysis",
        help="Directory to save results (default: physics_feature_analysis)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = PhysicsFeatureAnalyzer(
        data_file=args.data,
        validation_level=args.validation,
        approach=args.approach,
        results_dir=args.results_dir
    )
    
    results = analyzer.run_complete_analysis()