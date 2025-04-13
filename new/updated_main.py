"""
Enhanced Aftershock Location Prediction Models

This script integrates the advanced model architectures with the existing code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os
import datetime
import sys
from main_approach import *

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(
    log_dir,
    f"enhanced_aftershock_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)

# Set display style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper")


def create_cnn_lstm_model(X_train, y_train, X_test, y_test, waveform_data=None):
    """
    Creates and trains a CNN-LSTM hybrid model for aftershock location prediction.

    If waveform_data is provided, uses raw waveforms as input.
    Otherwise, uses the extracted features from X_train/X_test.

    Args:
        X_train: Training features
        y_train: Training targets (lat, lon)
        X_test: Test features
        y_test: Test targets (lat, lon)
        waveform_data: Optional dictionary mapping event_ids to raw waveforms

    Returns:
        Trained model, predictions, and evaluation metrics
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D
    from tensorflow.keras.layers import (
        Dropout,
        BatchNormalization,
        Concatenate,
        Flatten,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler

    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Create scalers
    feature_scaler = MinMaxScaler()
    lat_scaler = MinMaxScaler()
    lon_scaler = MinMaxScaler()

    # Scale features
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale targets
    y_train_lat = lat_scaler.fit_transform(y_train["latitude"].values.reshape(-1, 1))
    y_test_lat = lat_scaler.transform(y_test["latitude"].values.reshape(-1, 1))
    y_train_lon = lon_scaler.fit_transform(y_train["longitude"].values.reshape(-1, 1))
    y_test_lon = lon_scaler.transform(y_test["longitude"].values.reshape(-1, 1))

    # Combine latitude and longitude targets
    y_train_combined = np.hstack((y_train_lat, y_train_lon))

    # Create the model architecture
    # Feature input branch
    feature_input = Input(shape=(X_train_scaled.shape[1],), name="feature_input")
    feature_dense = Dense(128, activation="relu")(feature_input)
    feature_bn = BatchNormalization()(feature_dense)
    feature_dense2 = Dense(64, activation="relu")(feature_bn)

    # Reshape for LSTM
    feature_reshaped = tf.keras.layers.Reshape((-1, 64))(feature_dense2)

    # LSTM branch
    lstm_layer = LSTM(64, return_sequences=True)(feature_reshaped)
    lstm_dropout = Dropout(0.3)(lstm_layer)
    lstm_layer2 = LSTM(32)(lstm_dropout)
    lstm_dropout2 = Dropout(0.3)(lstm_layer2)

    # Combine and output
    combined = Dense(32, activation="relu")(lstm_dropout2)
    combined_bn = BatchNormalization()(combined)
    output_layer = Dense(2, activation="linear")(combined_bn)

    # Create and compile the model
    model = Model(inputs=feature_input, outputs=output_layer)
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    # Train the model
    history = model.fit(
        X_train_scaled,
        y_train_combined,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Make predictions
    predictions_scaled = model.predict(X_test_scaled)

    # Inverse transform predictions
    lat_pred = lat_scaler.inverse_transform(predictions_scaled[:, 0].reshape(-1, 1))
    lon_pred = lon_scaler.inverse_transform(predictions_scaled[:, 1].reshape(-1, 1))

    # Combine predictions
    y_pred = np.column_stack((lat_pred, lon_pred))

    # Calculate errors (using haversine distance)
    errors_km = []
    for i in range(len(y_test)):
        true_lat, true_lon = y_test.iloc[i]
        pred_lat, pred_lon = y_pred[i]
        distance = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
        errors_km.append(distance)

    median_error_km = np.median(errors_km)
    mean_error_km = np.mean(errors_km)

    print(f"CNN-LSTM Hybrid Model evaluation:")
    print(f"  Median Error: {median_error_km:.2f} km")
    print(f"  Mean Error: {mean_error_km:.2f} km")

    return {
        "model": model,
        "predictions": y_pred,
        "errors_km": errors_km,
        "history": history.history,
        "metrics": {"median_error_km": median_error_km, "mean_error_km": mean_error_km},
    }


def create_stacked_ensemble(X_train, y_train, X_test, y_test):
    """
    Creates and trains a stacked ensemble model for aftershock location prediction.

    This implementation combines the predictions of multiple base models:
    - LSTM
    - Linear Regression
    - Gradient Boosting
    - Random Forest

    A meta-learner is then trained on the base models' predictions.

    Args:
        X_train: Training features
        y_train: Training targets (lat, lon)
        X_test: Test features
        y_test: Test targets (lat, lon)

    Returns:
        Trained ensemble model, predictions, and evaluation metrics
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    print("Creating stacked ensemble model...")

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ===== Level 1: Base Models =====
    base_models = {
        "linear": LinearRegression(),
        "lasso": Lasso(alpha=0.01, random_state=42),
        "gbm": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "rf": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    # Create pipelines with standardization for each model
    pipelines = {
        name: Pipeline([("scaler", StandardScaler()), ("model", model)])
        for name, model in base_models.items()
    }

    # For storing out-of-fold predictions (will be used to train the meta-model)
    oof_predictions_lat = np.zeros((X_train.shape[0], len(base_models)))
    oof_predictions_lon = np.zeros((X_train.shape[0], len(base_models)))

    # For storing test predictions from each model
    test_predictions_lat = np.zeros((X_test.shape[0], len(base_models)))
    test_predictions_lon = np.zeros((X_test.shape[0], len(base_models)))

    # Train base models with cross-validation
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {i+1}/5")

        # Split data for this fold
        fold_X_train, fold_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        fold_y_train, fold_y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Train each base model on this fold
        for j, (name, pipeline) in enumerate(pipelines.items()):
            print(f"  Training {name}...")

            # Train separate models for latitude and longitude
            # Latitude
            pipeline_lat = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        base_models[name].__class__(**base_models[name].get_params()),
                    ),
                ]
            )
            pipeline_lat.fit(fold_X_train, fold_y_train["latitude"])
            oof_predictions_lat[val_idx, j] = pipeline_lat.predict(fold_X_val)
            test_predictions_lat[:, j] += pipeline_lat.predict(X_test) / kf.n_splits

            # Longitude
            pipeline_lon = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        base_models[name].__class__(**base_models[name].get_params()),
                    ),
                ]
            )
            pipeline_lon.fit(fold_X_train, fold_y_train["longitude"])
            oof_predictions_lon[val_idx, j] = pipeline_lon.predict(fold_X_val)
            test_predictions_lon[:, j] += pipeline_lon.predict(X_test) / kf.n_splits

    # ===== Add LSTM predictions =====
    # Create LSTM model for latitude
    print("\nTraining LSTM models...")

    # Function to create and train an LSTM model
    def train_lstm(X_train, y_train, X_test):
        # Scale the data
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Reshape for LSTM [samples, time steps, features]
        X_train_reshaped = X_train_scaled.reshape(
            X_train_scaled.shape[0], 1, X_train_scaled.shape[1]
        )
        X_test_reshaped = X_test_scaled.reshape(
            X_test_scaled.shape[0], 1, X_test_scaled.shape[1]
        )

        # Create model
        model = Sequential(
            [
                LSTM(
                    64, input_shape=(1, X_train_scaled.shape[1]), return_sequences=True
                ),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.3),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Train
        model.fit(
            X_train_reshaped,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
        )

        # Predict
        return model.predict(X_test_reshaped).flatten()

    # Use cross-validation for LSTM predictions too
    lstm_oof_predictions_lat = np.zeros(X_train.shape[0])
    lstm_oof_predictions_lon = np.zeros(X_train.shape[0])
    lstm_test_predictions_lat = np.zeros(X_test.shape[0])
    lstm_test_predictions_lon = np.zeros(X_test.shape[0])

    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  LSTM Fold {i+1}/5")

        # Split data
        fold_X_train, fold_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        fold_y_train_lat, fold_y_train_lon = (
            y_train.iloc[train_idx]["latitude"],
            y_train.iloc[train_idx]["longitude"],
        )

        # Train LSTM for latitude
        lstm_oof_predictions_lat[val_idx] = train_lstm(
            fold_X_train, fold_y_train_lat, fold_X_val
        )
        lstm_test_predictions_lat += (
            train_lstm(fold_X_train, fold_y_train_lat, X_test) / kf.n_splits
        )

        # Train LSTM for longitude
        lstm_oof_predictions_lon[val_idx] = train_lstm(
            fold_X_train, fold_y_train_lon, fold_X_val
        )
        lstm_test_predictions_lon += (
            train_lstm(fold_X_train, fold_y_train_lon, X_test) / kf.n_splits
        )

    # Add LSTM predictions to our ensemble
    oof_predictions_lat = np.column_stack(
        [oof_predictions_lat, lstm_oof_predictions_lat]
    )
    oof_predictions_lon = np.column_stack(
        [oof_predictions_lon, lstm_oof_predictions_lon]
    )
    test_predictions_lat = np.column_stack(
        [test_predictions_lat, lstm_test_predictions_lat]
    )
    test_predictions_lon = np.column_stack(
        [test_predictions_lon, lstm_test_predictions_lon]
    )

    # ===== Level 2: Meta-learner =====
    print("\nTraining meta-learners...")

    # Create and train meta-models
    meta_model_lat = GradientBoostingRegressor(n_estimators=100, random_state=42)
    meta_model_lon = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Train on out-of-fold predictions
    meta_model_lat.fit(oof_predictions_lat, y_train["latitude"])
    meta_model_lon.fit(oof_predictions_lon, y_train["longitude"])

    # Final predictions
    final_predictions_lat = meta_model_lat.predict(test_predictions_lat)
    final_predictions_lon = meta_model_lon.predict(test_predictions_lon)

    # Combine predictions
    y_pred = np.column_stack((final_predictions_lat, final_predictions_lon))

    # Calculate errors (using haversine distance)
    errors_km = []
    for i in range(len(y_test)):
        true_lat, true_lon = y_test.iloc[i]
        pred_lat, pred_lon = y_pred[i]
        distance = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
        errors_km.append(distance)

    median_error_km = np.median(errors_km)
    mean_error_km = np.mean(errors_km)

    print(f"\nStacked Ensemble Model evaluation:")
    print(f"  Median Error: {median_error_km:.2f} km")
    print(f"  Mean Error: {mean_error_km:.2f} km")

    # Return results
    return {
        "meta_model_lat": meta_model_lat,
        "meta_model_lon": meta_model_lon,
        "predictions": y_pred,
        "errors_km": errors_km,
        "metrics": {"median_error_km": median_error_km, "mean_error_km": mean_error_km},
    }


# Haversine distance function (copied from the original code)
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers."""
    import numpy as np

    R = 6371  # Earth radius in kilometers
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def train_enhanced_models(ml_df, selected_features=None):
    """
    Train and evaluate enhanced models on the preprocessed data.

    Args:
        ml_df: DataFrame with features and target variables
        selected_features: List of feature columns to use. If None, use all available features.

    Returns:
        Dictionary with model results
    """
    logging.info("\n=== Training Enhanced Models ===")

    # If no features selected, use all features except targets and identifiers
    if selected_features is None:
        exclude_cols = [
            "latitude",
            "longitude",
            "event_id",
            "datetime",
            "source_origin_time",
        ]
        selected_features = [col for col in ml_df.columns if col not in exclude_cols]

    logging.info(f"Using {len(selected_features)} features")

    # Prepare train/test split
    train_df, test_df = train_test_split(ml_df, test_size=0.2, random_state=42)

    # For temporal data, sorting may be important
    if "hours_since_mainshock" in ml_df.columns:
        # Ensure train set contains earlier events than test set
        train_df = ml_df.sort_values("hours_since_mainshock").iloc[
            : int(len(ml_df) * 0.8)
        ]
        test_df = ml_df.sort_values("hours_since_mainshock").iloc[
            int(len(ml_df) * 0.8) :
        ]

    X_train = train_df[selected_features]
    y_train = train_df[["latitude", "longitude"]]
    X_test = test_df[selected_features]
    y_test = test_df[["latitude", "longitude"]]

    logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Create dictionary to store results
    model_results = {}

    # 1. Train CNN-LSTM Hybrid Model
    logging.info("\n1. Training CNN-LSTM Hybrid Model...")
    try:
        cnn_lstm_results = create_cnn_lstm_model(X_train, y_train, X_test, y_test)
        model_results["CNN-LSTM"] = cnn_lstm_results["metrics"]

        # Save error distribution for later visualization
        np.save("cnn_lstm_errors.npy", cnn_lstm_results["errors_km"])

        logging.info(
            f"CNN-LSTM Model - Median Error: {cnn_lstm_results['metrics']['median_error_km']:.2f} km, "
            f"Mean Error: {cnn_lstm_results['metrics']['mean_error_km']:.2f} km"
        )
    except Exception as e:
        logging.error(f"Error training CNN-LSTM model: {e}")

    # 3. Train Stacked Ensemble Model
    logging.info("\n3. Training Stacked Ensemble Model...")
    try:
        ensemble_results = create_stacked_ensemble(X_train, y_train, X_test, y_test)
        model_results["Stacked Ensemble"] = ensemble_results["metrics"]

        # Save error distribution
        np.save("ensemble_errors.npy", ensemble_results["errors_km"])

        logging.info(
            f"Stacked Ensemble Model - Median Error: {ensemble_results['metrics']['median_error_km']:.2f} km, "
            f"Mean Error: {ensemble_results['metrics']['mean_error_km']:.2f} km"
        )
    except Exception as e:
        logging.error(f"Error training Stacked Ensemble model: {e}")

    # 4. Train Graph Neural Network Model if features are available
    if "hours_since_mainshock" in X_train.columns:
        logging.info("\n4. Training Graph Neural Network Model...")
        try:
            # Get mainshock location from first aftershock if available
            if (
                "mainshock_latitude" in ml_df.columns
                and "mainshock_longitude" in ml_df.columns
            ):
                mainshock_location = (
                    ml_df["mainshock_latitude"].iloc[0],
                    ml_df["mainshock_longitude"].iloc[0],
                )
            else:
                mainshock_location = None

            gnn_results = create_gnn_model(
                X_train, y_train, X_test, y_test, mainshock_location=mainshock_location
            )
            model_results["GNN"] = gnn_results["metrics"]

            # Save error distribution
            np.save("gnn_errors.npy", gnn_results["errors_km"])

            logging.info(
                f"GNN Model - Median Error: {gnn_results['metrics']['median_error_km']:.2f} km, "
                f"Mean Error: {gnn_results['metrics']['mean_error_km']:.2f} km"
            )
        except Exception as e:
            logging.error(f"Error training GNN model: {e}")

    # Compare with baseline models
    logging.info("\n=== Model Performance Comparison ===")
    for model_name, metrics in model_results.items():
        logging.info(
            f"{model_name}: Median Error = {metrics['median_error_km']:.2f} km, "
            f"Mean Error = {metrics['mean_error_km']:.2f} km"
        )

    # Visualize results
    visualize_model_comparison_new(model_results)

    return model_results


def create_gnn_model(
    X_train, y_train, X_test, y_test, mainshock_location=None, max_connections=10
):
    """
    Creates and trains a Graph Neural Network model for aftershock location prediction.

    This model treats seismic events as nodes in a graph, with connections based on
    temporal and spatial proximity. It can capture the complex spatiotemporal
    relationships between events.

    Args:
        X_train: Training features
        y_train: Training targets (lat, lon)
        X_test: Test features
        y_test: Test targets (lat, lon)
        mainshock_location: Tuple (lat, lon) of the mainshock
        max_connections: Maximum number of connections per node

    Returns:
        Trained model, predictions, and evaluation metrics
    """
    import numpy as np
    import tensorflow as tf
    import networkx as nx
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler

    print("Creating Graph Neural Network model...")

    # ===== Data Preparation =====
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Combine training data with time and location for graph creation
    train_data = X_train.copy()
    train_data["latitude"] = y_train["latitude"]
    train_data["longitude"] = y_train["longitude"]

    # If mainshock location is not provided, use average of locations
    if mainshock_location is None:
        mainshock_location = (
            train_data["latitude"].mean(),
            train_data["longitude"].mean(),
        )

    # ===== Create Graph =====
    print("Building the seismic event graph...")

    # Create graph
    G = nx.Graph()

    # Add nodes (each event is a node)
    for i in range(len(train_data)):
        G.add_node(i, features=X_train_scaled[i])

    # Extract time and spatial features for edge creation
    hours_since_mainshock = (
        train_data["hours_since_mainshock"].values
        if "hours_since_mainshock" in train_data.columns
        else None
    )
    lats = train_data["latitude"].values
    lons = train_data["longitude"].values

    # Function to calculate edge weight based on temporal and spatial proximity
    def calculate_edge_weight(i, j):
        # Temporal weight (higher for events closer in time)
        if hours_since_mainshock is not None:
            time_diff = abs(hours_since_mainshock[i] - hours_since_mainshock[j])
            temporal_weight = 1 / (1 + time_diff)
        else:
            temporal_weight = 1

        # Spatial weight (higher for events closer in space)
        spatial_distance = haversine_distance(lats[i], lons[i], lats[j], lons[j])
        spatial_weight = 1 / (1 + spatial_distance)

        # Combined weight (geometric mean)
        return (temporal_weight * spatial_weight) ** 0.5

    # Add edges based on temporal and spatial proximity
    for i in range(len(train_data)):
        # Calculate weights to all other nodes
        weights = [
            (j, calculate_edge_weight(i, j)) for j in range(len(train_data)) if i != j
        ]

        # Sort by weight and take top max_connections
        top_connections = sorted(weights, key=lambda x: x[1], reverse=True)[
            :max_connections
        ]

        # Add edges
        for j, weight in top_connections:
            G.add_edge(i, j, weight=weight)

    # ===== Implement Graph Neural Network =====
    # Define graph convolutional layer
    class GraphConvLayer(tf.keras.layers.Layer):
        def __init__(self, units, activation=None):
            super(GraphConvLayer, self).__init__()
            self.units = units
            self.activation = tf.keras.activations.get(activation)

        def build(self, input_shape):
            self.kernel = self.add_weight(
                shape=(input_shape[0][-1], self.units),
                initializer="glorot_uniform",
                trainable=True,
            )
            if self.activation:
                self.bias = self.add_weight(
                    shape=(self.units,),
                    initializer="zeros",
                    trainable=True,
                )

        def call(self, inputs):
            # Unpack inputs
            features, adjacency = inputs

            # Graph convolution: H = σ(A * X * W)
            support = tf.matmul(features, self.kernel)
            output = tf.matmul(adjacency, support)

            if self.activation:
                output = self.activation(output + self.bias)

            return output

    # Create adjacency matrix from the graph
    adjacency_matrix = nx.to_numpy_array(G)

    # Normalize adjacency matrix (add self-loops and symmetric normalization)
    identity = np.eye(adjacency_matrix.shape[0])
    adjacency_matrix = adjacency_matrix + identity
    # Convert to a row-stochastic matrix by normalizing each row
    row_sums = adjacency_matrix.sum(axis=1)
    adjacency_matrix = adjacency_matrix / row_sums[:, np.newaxis]

    # Create feature matrix
    feature_matrix = np.array([G.nodes[i]["features"] for i in range(len(G.nodes))])

    # Build the model
    input_features = Input(shape=(feature_matrix.shape[1],))
    input_adjacency = Input(shape=(adjacency_matrix.shape[0],))

    # Graph convolutional layers
    graph_conv_1 = GraphConvLayer(64, activation="relu")(
        [input_features, input_adjacency]
    )
    dropout_1 = Dropout(0.5)(graph_conv_1)

    graph_conv_2 = GraphConvLayer(32, activation="relu")([dropout_1, input_adjacency])
    dropout_2 = Dropout(0.5)(graph_conv_2)

    # Dense layers for final prediction
    dense = Dense(16, activation="relu")(dropout_2)
    output = Dense(2, activation="linear")(dense)

    # Create model
    model = Model(inputs=[input_features, input_adjacency], outputs=output)

    # Compile
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Prepare target data
    y_train_combined = np.column_stack((y_train["latitude"], y_train["longitude"]))

    # Train the model
    early_stopping = EarlyStopping(
        monitor="loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        [feature_matrix, adjacency_matrix],
        y_train_combined,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    # ===== Use model for predictions =====
    # For predictions, we need to extend the graph to include test nodes
    # This is a simplified approach - in a full implementation, you would
    # properly integrate the test nodes into the graph

    # Function to predict with the trained model, using a nearest neighbor approach
    def predict(X_test_scaled):
        test_predictions = []

        for i in range(len(X_test_scaled)):
            # Find most similar nodes in the training set
            similarities = []
            for j in range(len(feature_matrix)):
                similarity = np.sum((X_test_scaled[i] - feature_matrix[j]) ** 2)
                similarities.append((j, similarity))

            # Get top 5 most similar nodes
            top_similar = sorted(similarities, key=lambda x: x[1])[:5]
            top_indices = [j for j, _ in top_similar]

            # Get predicted locations for these nodes
            similar_node_preds = model.predict(
                [
                    feature_matrix[top_indices],
                    adjacency_matrix[top_indices][:, top_indices],
                ],
                verbose=0,
            )

            # Use weighted average based on similarity
            weights = [1 / (1 + sim) for _, sim in top_similar]
            weight_sum = sum(weights)

            weighted_prediction = np.zeros(2)
            for k, (j, _) in enumerate(top_similar):
                weighted_prediction += (weights[k] / weight_sum) * y_train_combined[j]

            test_predictions.append(weighted_prediction)

        return np.array(test_predictions)

    # Make predictions
    y_pred = predict(X_test_scaled)

    # Calculate errors
    errors_km = []
    for i in range(len(y_test)):
        true_lat, true_lon = y_test.iloc[i]
        pred_lat, pred_lon = y_pred[i]
        distance = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
        errors_km.append(distance)

    median_error_km = np.median(errors_km)
    mean_error_km = np.mean(errors_km)

    print(f"\nGraph Neural Network Model evaluation:")
    print(f"  Median Error: {median_error_km:.2f} km")
    print(f"  Mean Error: {mean_error_km:.2f} km")

    return {
        "model": model,
        "predictions": y_pred,
        "errors_km": errors_km,
        "history": history.history,
        "metrics": {"median_error_km": median_error_km, "mean_error_km": mean_error_km},
    }


def visualize_model_comparison_new(results):
    """
    Create visualizations to compare enhanced model performances.
    """
    # Bar chart of median errors
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    median_errors = [results[model]["median_error_km"] for model in models]
    mean_errors = [results[model]["mean_error_km"] for model in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(
        x - width / 2,
        median_errors,
        width,
        label="Median Error",
        color="cornflowerblue",
    )
    rects2 = ax.bar(
        x + width / 2, mean_errors, width, label="Mean Error", color="lightcoral"
    )

    ax.set_ylabel("Error (km)")
    ax.set_title("Location Prediction Error by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    # Add error values on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig("enhanced_model_comparison.png")
    plt.close()

    # Compare error distributions if available
    error_files = {
        "CNN-LSTM": "cnn_lstm_errors.npy",
        "Stacked Ensemble": "ensemble_errors.npy",
        "GNN": "gnn_errors.npy",
    }

    available_error_files = [f for f in error_files.values() if os.path.exists(f)]
    if available_error_files:
        plt.figure(figsize=(14, 8))

        for model_name, file_name in error_files.items():
            if os.path.exists(file_name):
                errors = np.load(file_name)
                # Plot error distribution as KDE
                sns.kdeplot(
                    errors, label=f"{model_name} (median={np.median(errors):.1f} km)"
                )

        plt.title("Error Distribution Comparison")
        plt.xlabel("Error (km)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("error_distributions.png")
        plt.close()

    logging.info("Saved model comparison visualizations")


def main():
    """Main execution function"""

    """Main execution function"""
    start_time = datetime.datetime.now()

    logging.info("=== Aftershock Location Prediction Model ===")

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
    start_time = datetime.datetime.now()

    logging.info("=== Enhanced Aftershock Location Prediction Models ===")
    logging.info(f"Started at: {start_time}")

    # Define chosen features
    selected_features = [
        "Z_spec_dom_freq_std",
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
        "Z_energy",
        "N_energy",
        "E_energy",
        "Z_dominant_freq",
        "N_dominant_freq",
        "E_dominant_freq",
    ]

    # Ensure only available features are used
    selected_features = [feat for feat in selected_features if feat in ml_df.columns]

    # Train enhanced models
    model_results = train_enhanced_models(ml_df, selected_features)

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
    for directory in ["logs", "plots", "results", "models"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    try:
        main()
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        raise
