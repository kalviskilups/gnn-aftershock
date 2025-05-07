#!/usr/bin/env python3
# hyperparameters.py - Advanced hyperparameter tuning for XGBoost aftershock location prediction

import numpy as np
import pandas as pd
import optuna
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost_aftershock_prediction import XGBoostAfterShockPredictor
import time
import argparse


class XGBoostHyperparameterOptimizer:
    """
    Class for finding optimal XGBoost hyperparameters that minimize median 3D error
    for aftershock location prediction
    """

    def __init__(self, data_pickle, approach="best_station", n_trials=100, cv_folds=5):
        """
        Initialize the hyperparameter optimizer
        
        Args:
            data_pickle: Path to pickle file with preprocessed data
            approach: Analysis approach ("best_station" or "multi_station")
            n_trials: Number of optimization trials to run
            cv_folds: Number of cross-validation folds
        """
        self.data_pickle = data_pickle
        self.approach = approach
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = None
        self.study = None
        self.predictor = None
        self.X = None
        self.y = None
        
        print(f"Initializing hyperparameter optimizer for {approach} approach")
        print(f"Will run {n_trials} optimization trials with {cv_folds}-fold cross-validation")
        
        # Create predictor and load data
        self.predictor = XGBoostAfterShockPredictor(
            data_pickle=data_pickle,
            validation_level="critical",  # Only run critical validations for speed
            approach=approach
        )
        
    def prepare_data(self):
        """
        Prepare the dataset for hyperparameter optimization
        """
        print("Preparing dataset...")
        # Initialize mainshock and create coordinate dataframe
        self.predictor.find_mainshock()
        self.predictor.create_relative_coordinate_dataframe()
        
        # Get features and targets
        self.X, self.y = self.predictor.prepare_dataset()
        
        # Ensure we have grouping variable for proper CV
        self.grouping_var = None
        if "event_id" in self.y.columns:
            self.grouping_var = "event_id"
        elif "event_date" in self.y.columns:
            self.grouping_var = "event_date"
        
        print(f"Dataset prepared with {len(self.X)} samples and {self.X.shape[1]} features")
        if self.grouping_var:
            print(f"Using {self.grouping_var} for group-based cross-validation")
        
        return self.X, self.y
        
    def objective(self, trial):
        """
        Optimization objective function that evaluates a set of hyperparameters
        using cross-validation and returns the median 3D error
        
        Args:
            trial: Optuna trial object
        
        Returns:
            median_3d_error: Median 3D error across all CV folds
        """
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
            'random_state': 42
        }
        
        # Set up cross-validation
        cv_errors = []
        
        if self.grouping_var:
            # Group-based cross-validation to prevent data leakage
            gkf = GroupKFold(n_splits=self.cv_folds)
            groups = self.y[self.grouping_var]
            cv_splits = gkf.split(self.X, self.y, groups)
        else:
            # Standard k-fold if no grouping variable
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            cv_splits = kf.split(self.X, self.y)
        
        # Perform cross-validation
        for train_idx, test_idx in cv_splits:
            # Get train/test split
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]
            X_test = self.X.iloc[test_idx]
            y_test = self.y.iloc[test_idx]
            
            # Drop grouping variables if present
            if self.grouping_var:
                if self.grouping_var in X_train.columns:
                    X_train = X_train.drop(self.grouping_var, axis=1)
                if self.grouping_var in X_test.columns:
                    X_test = X_test.drop(self.grouping_var, axis=1)
                y_train_coord = y_train.drop(self.grouping_var, axis=1)
                y_test_coord = y_test.drop(self.grouping_var, axis=1)
            else:
                y_train_coord = y_train
                y_test_coord = y_test
            
            # Filter to numeric columns
            numeric_columns = [col for col in X_train.columns 
                              if pd.api.types.is_numeric_dtype(X_train[col])]
            X_train_numeric = X_train[numeric_columns]
            X_test_numeric = X_test[numeric_columns]
            
            # Train model on this fold
            base_xgb = XGBRegressor(**params)
            multi_model = MultiOutputRegressor(base_xgb)
            
            try:
                # Train on all three coordinates
                multi_model.fit(
                    X_train_numeric, 
                    y_train_coord[["relative_x", "relative_y", "relative_z"]]
                )
                
                # Make predictions
                y_pred = multi_model.predict(X_test_numeric)
                
                # Calculate 3D distance error
                errors_3d = np.sqrt(
                    (y_pred[:, 0] - y_test_coord["relative_x"].values) ** 2 +
                    (y_pred[:, 1] - y_test_coord["relative_y"].values) ** 2 +
                    (y_pred[:, 2] - y_test_coord["relative_z"].values) ** 2
                )
                
                # Get median of 3D errors for this fold
                median_error = np.median(errors_3d)
                cv_errors.append(median_error)
                
                # Report intermediate result
                trial.report(median_error, len(cv_errors) - 1)
                
                # Handle pruning (early stopping of unpromising trials)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
            except Exception as e:
                print(f"Error in trial: {e}")
                return float('inf')  # Return a high error if the trial fails
        
        # Return mean of median errors across folds
        return np.mean(cv_errors)
    
    def optimize(self):
        """
        Run the hyperparameter optimization process
        """
        print("\n" + "=" * 70)
        print(f"OPTIMIZING HYPERPARAMETERS FOR {self.approach.upper()} APPROACH".center(70))
        print("=" * 70)
        
        # Prepare data if not already done
        if self.X is None or self.y is None:
            self.prepare_data()
        
        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",  # We want to minimize the error
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        # Run optimization
        start_time = time.time()
        print(f"Starting optimization with {self.n_trials} trials...")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Store the study for later analysis
        self.study = study
        
        # Get best parameters
        self.best_params = study.best_params
        best_value = study.best_value
        
        # Print results
        print("\nOptimization complete!")
        print(f"Best median 3D error: {best_value:.2f} km")
        print("Best hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Total optimization time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        
        return self.best_params, best_value
    
    def save_results(self, output_dir="hyperparameter_results"):
        """
        Save optimization results and visualizations
        
        Args:
            output_dir: Directory to save results
        """
        if self.study is None:
            print("No optimization results to save. Run optimize() first.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best parameters to file
        params_file = f"{output_dir}/best_params_{self.approach}.pkl"
        with open(params_file, "wb") as f:
            pickle.dump(self.best_params, f)
        
        # Save best parameters as text file for easier reading
        with open(f"{output_dir}/best_params_{self.approach}.txt", "w") as f:
            f.write(f"Best median 3D error: {self.study.best_value:.4f} km\n\n")
            f.write("Best hyperparameters:\n")
            for param, value in self.best_params.items():
                f.write(f"{param}: {value}\n")
        
        # Create parameter importance visualization
        try:
            plt.figure(figsize=(10, 8))
            param_importances = optuna.importance.get_param_importances(self.study)
            params = list(param_importances.keys())
            importances = list(param_importances.values())
            
            # Sort by importance
            sorted_indices = np.argsort(importances)
            params = [params[i] for i in sorted_indices]
            importances = [importances[i] for i in sorted_indices]
            
            plt.barh(params, importances)
            plt.xlabel("Importance")
            plt.title(f"Hyperparameter Importance for {self.approach.title()} Approach")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/param_importance_{self.approach}.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating parameter importance plot: {e}")
        
        # Create optimization history visualization
        plt.figure(figsize=(10, 6))
        plt.plot(self.study.trials_dataframe()["value"])
        plt.xlabel("Trial")
        plt.ylabel("Median 3D Error (km)")
        plt.title(f"Optimization History for {self.approach.title()} Approach")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/optimization_history_{self.approach}.png", dpi=300)
        plt.close()
        
        print(f"Results saved to {output_dir}/")
    
    def validate_best_params(self):
        """
        Train a model with the best parameters and evaluate it
        on a held-out test set to validate performance
        """
        if self.best_params is None:
            print("No best parameters found. Run optimize() first.")
            return
        
        print("\n" + "=" * 70)
        print(f"VALIDATING BEST PARAMETERS FOR {self.approach.upper()} APPROACH".center(70))
        print("=" * 70)
        
        # Create a new predictor for validation
        predictor = XGBoostAfterShockPredictor(
            data_pickle=self.data_pickle,
            validation_level="critical",
            approach=self.approach
        )
        
        # Run workflow with best parameters
        predictor.find_mainshock()
        predictor.create_relative_coordinate_dataframe()
        X, y = predictor.prepare_dataset()
        
        # Set up GroupKFold for the test split
        if self.grouping_var and self.grouping_var in y.columns:
            print(f"Using GroupKFold with {self.grouping_var} to create validation split")
            gkf = GroupKFold(n_splits=5)
            groups = y[self.grouping_var]
            train_idx, test_idx = next(gkf.split(X, y, groups))
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Drop the group column
            if self.grouping_var in X_train.columns:
                X_train = X_train.drop(self.grouping_var, axis=1)
            if self.grouping_var in X_test.columns:
                X_test = X_test.drop(self.grouping_var, axis=1)
                
            y_train_coord = y_train.drop(self.grouping_var, axis=1)
            y_test_coord = y_test.drop(self.grouping_var, axis=1)
        else:
            # Standard train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_train_coord = y_train
            y_test_coord = y_test
        
        # Filter to numeric columns
        numeric_columns = [col for col in X_train.columns 
                          if pd.api.types.is_numeric_dtype(X_train[col])]
        X_train_numeric = X_train[numeric_columns]
        X_test_numeric = X_test[numeric_columns]
        
        # Create and train model with best parameters
        print("Training model with best parameters...")
        base_xgb = XGBRegressor(**self.best_params)
        multi_model = MultiOutputRegressor(base_xgb)
        
        # Train on all three coordinates
        multi_model.fit(
            X_train_numeric, 
            y_train_coord[["relative_x", "relative_y", "relative_z"]]
        )
        
        # Make predictions on test set
        y_pred = multi_model.predict(X_test_numeric)
        
        # Calculate 3D distance error
        errors_3d = np.sqrt(
            (y_pred[:, 0] - y_test_coord["relative_x"].values) ** 2 +
            (y_pred[:, 1] - y_test_coord["relative_y"].values) ** 2 +
            (y_pred[:, 2] - y_test_coord["relative_z"].values) ** 2
        )
        
        # Calculate statistics
        mean_error = np.mean(errors_3d)
        median_error = np.median(errors_3d)
        std_error = np.std(errors_3d)
        
        print("\nValidation Results:")
        print(f"Mean 3D error: {mean_error:.2f} km")
        print(f"Median 3D error: {median_error:.2f} km")
        print(f"Std Dev of 3D error: {std_error:.2f} km")
        
        # Visualize error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(errors_3d, bins=20, alpha=0.7)
        plt.axvline(
            mean_error,
            color="r",
            linestyle="--",
            label=f"Mean: {mean_error:.2f} km",
        )
        plt.axvline(
            median_error,
            color="g",
            linestyle="--",
            label=f"Median: {median_error:.2f} km",
        )
        plt.title(f"Optimized Model: 3D Error Distribution ({self.approach.title()})")
        plt.xlabel("Error (km)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f"optimized_model_error_histogram_{self.approach}.png", dpi=300)
        plt.close()
        
        return {
            "mean_error": mean_error,
            "median_error": median_error,
            "std_error": std_error,
            "errors_3d": errors_3d
        }


def main():
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for XGBoost aftershock location prediction"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to pickle file with preprocessed data"
    )
    parser.add_argument(
        "--approach",
        choices=["best_station", "multi_station", "both"],
        default="both",
        help="Analysis approach to optimize (default: best_station)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of optimization trials to run (default: 100)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        default="hyperparameter_results",
        help="Directory to save results (default: hyperparameter_results)"
    )

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.approach == "both":
        approaches = ["best_station", "multi_station"]
    else:
        approaches = [args.approach]
    
    results = {}
    
    for approach in approaches:
        print(f"\nOptimizing for {approach} approach...")
        optimizer = XGBoostHyperparameterOptimizer(
            data_pickle=args.data,
            approach=approach,
            n_trials=args.trials,
            cv_folds=args.cv_folds
        )
        
        # Prepare data
        optimizer.prepare_data()
        
        # Run optimization
        best_params, best_value = optimizer.optimize()
        
        # Save results
        optimizer.save_results(output_dir=args.output_dir)
        
        # Validate best parameters
        validation_results = optimizer.validate_best_params()
        
        # Store results
        results[approach] = {
            "best_params": best_params,
            "best_value": best_value,
            "validation": validation_results
        }
    
    # Compare approaches if both were optimized
    if len(approaches) > 1:
        print("\n" + "=" * 70)
        print("COMPARING OPTIMIZED APPROACHES".center(70))
        print("=" * 70)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame(
            {
                "Best-Station": [
                    results["best_station"]["validation"]["median_error"],
                    results["best_station"]["validation"]["mean_error"],
                    results["best_station"]["validation"]["std_error"],
                ],
                "Multi-Station": [
                    results["multi_station"]["validation"]["median_error"],
                    results["multi_station"]["validation"]["mean_error"],
                    results["multi_station"]["validation"]["std_error"],
                ],
            },
            index=["Median 3D Error (km)", "Mean 3D Error (km)", "Std Dev (km)"],
        )
        
        # Calculate improvement
        comparison["Improvement (%)"] = (
            (comparison["Best-Station"] - comparison["Multi-Station"])
            / comparison["Best-Station"]
            * 100
        )
        
        print("\nOptimized Approach Comparison:")
        print(comparison)

        
        # Visualize comparison
        plt.figure(figsize=(10, 6))
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
                    comparison["Multi-Station"].iloc[i] + 0.5,
                    f"+{imp:.1f}%",
                    ha="center",
                    va="bottom",
                    color="green",
                    fontweight="bold",
                )
        
        plt.xlabel("Metric")
        plt.ylabel("Error (km)")
        plt.title("Comparison of Optimized Approaches")
        plt.xticks(index, comparison.index)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(f"{args.output_dir}/optimized_approach_comparison.png", dpi=300)
        plt.tight_layout()
    
    print(f"\nAll optimization results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()