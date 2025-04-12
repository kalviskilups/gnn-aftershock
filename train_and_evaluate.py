import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import math


def train_model_with_diagnostics(graph_dataset, model, epochs=300, lr=0.001, batch_size=32, patience=15):
    """Train with improved stability measures"""
    # Split dataset
    train_data, val_data = train_test_split(graph_dataset, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Initialize optimizer with lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Loss function with L2 regularization
    def loss_fn(pred, target, batch_size, diversity_weight=0.1):
        """Loss function with diversity regularization"""
        # Basic MSE loss
        mse_loss = F.mse_loss(pred, target)
        
        # Diversity regularization
        if batch_size > 1:
            # Calculate variance of predictions
            pred_var = torch.var(pred, dim=0)
            target_var = torch.var(target, dim=0)
            
            # Penalize if prediction variance is too small compared to target variance
            diversity_loss = F.mse_loss(pred_var, target_var)
            
            # Combined loss
            total_loss = mse_loss + diversity_weight * diversity_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            lat_pred, lon_pred = model(batch.metadata, batch.waveform, 
                                     batch.edge_index, batch.batch)
            pred = torch.cat([lat_pred, lon_pred], dim=1)
            
            # Compute loss
            loss = loss_fn(pred, batch.y, batch.num_graphs)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                lat_pred, lon_pred = model(batch.metadata, batch.waveform, 
                                         batch.edge_index, batch.batch)
                pred = torch.cat([lat_pred, lon_pred], dim=1)
                loss = F.mse_loss(pred, batch.y)
                val_loss += loss.item()
                
                predictions.append(pred)
                targets.append(batch.y)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Val Loss: {val_loss/len(val_loader):.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Print prediction statistics
            all_preds = torch.cat(predictions)
            all_targets = torch.cat(targets)
            print(f"Prediction std: {torch.std(all_preds, dim=0)}")
            print(f"Target std: {torch.std(all_targets, dim=0)}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'results/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    # Load best model
    model.load_state_dict(torch.load('results/best_model.pt'))
    return model


def convert_relative_to_absolute_predictions(pred_deltas, references):
    """
    Convert relative predictions (in km) back to absolute coordinates (lat/lon)
    """
    absolute_coords = np.zeros_like(references)

    for i in range(len(pred_deltas)):
        # Get reference point
        ref_lat = references[i, 0]
        ref_lon = references[i, 1]

        # Get predicted deltas (in km)
        delta_lat_km = pred_deltas[i, 0]
        delta_lon_km = pred_deltas[i, 1]

        # Convert km back to degrees
        lat_km_per_degree = 111.0
        lon_km_per_degree = 111.0 * np.cos(np.radians(ref_lat))

        delta_lat_deg = delta_lat_km / lat_km_per_degree
        delta_lon_deg = delta_lon_km / lon_km_per_degree

        # Calculate absolute coordinates
        absolute_coords[i, 0] = ref_lat + delta_lat_deg
        absolute_coords[i, 1] = ref_lon + delta_lon_deg

    return absolute_coords


def evaluate_model(model, graph_dataset, mainshock, reference_coords):
    """
    Evaluate the model with relative position predictions
    """
    # Handle the case where we have very few data points
    test_size = 0.2
    if len(graph_dataset) < 5:
        test_size = 0.4  # Use more for testing if we have limited data

    # Split into training and testing sets
    _, test_data = train_test_split(graph_dataset, test_size=test_size, random_state=42)

    # Create data loader - use batch size of 1 for very small datasets
    batch_size = 32
    if len(test_data) < 10:
        batch_size = 1

    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Model in evaluation mode
    model.eval()

    # Lists to store actual and predicted deltas
    actual_deltas = []
    pred_deltas = []
    test_indices = []

    # Evaluate on test set
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)

            # Forward pass
            lat_pred, lon_pred = model(
                batch.metadata, batch.waveform, batch.edge_index, batch.batch
            )

            # Extract predictions and targets
            lat_np = lat_pred.cpu().numpy().flatten()
            lon_np = lon_pred.cpu().numpy().flatten()

            # Combine into delta predictions
            batch_pred_deltas = np.column_stack((lat_np, lon_np))
            batch_actual_deltas = batch.y.cpu().numpy()

            # Store the batch indices for mapping back to reference coordinates
            batch_indices = list(
                range(i * batch_size, min((i + 1) * batch_size, len(test_data)))
            )
            test_indices.extend(batch_indices)

            actual_deltas.append(batch_actual_deltas)
            pred_deltas.append(batch_pred_deltas)

    # Concatenate results
    actual_deltas = np.vstack(actual_deltas)
    pred_deltas = np.vstack(pred_deltas)

    # Get the reference coordinates for the test set
    _, test_refs = train_test_split(
        reference_coords, test_size=test_size, random_state=42
    )

    # Ensure we're using the right indices if we have fewer test points than expected
    if len(test_indices) < len(test_refs):
        test_refs = test_refs[test_indices]
    else:
        # Make sure we don't exceed the length of pred_deltas
        test_refs = test_refs[: len(pred_deltas)]

    # Convert relative predictions to absolute coordinates
    absolute_pred = convert_relative_to_absolute_predictions(pred_deltas, test_refs)
    absolute_actual = convert_relative_to_absolute_predictions(
        actual_deltas[: len(test_refs)], test_refs
    )

    # Convert to individual arrays
    actual_lats = absolute_actual[:, 0]
    actual_lons = absolute_actual[:, 1]
    pred_lats = absolute_pred[:, 0]
    pred_lons = absolute_pred[:, 1]

    # After collecting predictions
    pred_lat_mean = np.mean(pred_lats)
    pred_lat_std = np.std(pred_lats)
    pred_lon_mean = np.mean(pred_lons)
    pred_lon_std = np.std(pred_lons)

    actual_lat_mean = np.mean(actual_lats)
    actual_lat_std = np.std(actual_lats)
    actual_lon_mean = np.mean(actual_lons)
    actual_lon_std = np.std(actual_lons)

    print(f"\nPrediction Distribution Analysis:")
    print(f"  Predicted latitude - Mean: {pred_lat_mean:.6f}, Std: {pred_lat_std:.6f}")
    print(f"  Actual latitude    - Mean: {actual_lat_mean:.6f}, Std: {actual_lat_std:.6f}")
    print(f"  Predicted longitude - Mean: {pred_lon_mean:.6f}, Std: {pred_lon_std:.6f}")
    print(f"  Actual longitude    - Mean: {actual_lon_mean:.6f}, Std: {actual_lon_std:.6f}")

    # Calculate diversity ratio (predicted std / actual std)
    lat_diversity = pred_lat_std / actual_lat_std if actual_lat_std > 0 else 0
    lon_diversity = pred_lon_std / actual_lon_std if actual_lon_std > 0 else 0
    print(f"  Latitude diversity ratio: {lat_diversity:.4f}")
    print(f"  Longitude diversity ratio: {lon_diversity:.4f}")

    # Calculate error in kilometers
    errors_km = []
    for i in range(len(actual_lats)):
        # Haversine distance
        R = 6371  # Earth radius in kilometers
        lat1, lon1 = np.radians(actual_lats[i]), np.radians(actual_lons[i])
        lat2, lon2 = np.radians(pred_lats[i]), np.radians(pred_lons[i])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c

        errors_km.append(distance)

    mean_error = np.mean(errors_km)
    median_error = np.median(errors_km)

    print(f"Evaluation Results:")
    print(f"Mean Error: {mean_error:.2f} km")
    print(f"Median Error: {median_error:.2f} km")

    # Plot spatial predictions
    plt.figure(figsize=(12, 10))

    # Plot mainshock
    plt.scatter(
        mainshock["source_longitude_deg"],
        mainshock["source_latitude_deg"],
        s=200,
        c="red",
        marker="*",
        label="Mainshock",
        edgecolor="black",
        zorder=10,
    )

    # Plot actual aftershocks
    plt.scatter(
        actual_lons,
        actual_lats,
        s=50,
        c="blue",
        alpha=0.7,
        label="Actual Aftershocks",
        edgecolor="black",
    )

    # Plot predicted aftershocks
    plt.scatter(
        pred_lons,
        pred_lats,
        s=30,
        c="green",
        alpha=0.7,
        marker="x",
        label="Predicted Aftershocks",
    )

    # Connect actual to predicted with lines
    for i in range(len(actual_lats)):
        plt.plot(
            [actual_lons[i], pred_lons[i]],
            [actual_lats[i], pred_lats[i]],
            "k-",
            alpha=0.2,
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(
        "Spatial Distribution of Actual vs Predicted Aftershocks (Relative Prediction Approach)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simplified_spatial_predictions.png", dpi=300)
    plt.close()

    # Calculate spatial distribution statistics
    if len(pred_lats) >= 2:  # Need at least 2 points for std
        pred_std_lat = np.std(pred_lats)
        pred_std_lon = np.std(pred_lons)
        actual_std_lat = np.std(actual_lats)
        actual_std_lon = np.std(actual_lons)

        print(f"\nSpatial Distribution Analysis:")
        print(f"  - Actual latitude std: {actual_std_lat:.4f}°")
        print(f"  - Predicted latitude std: {pred_std_lat:.4f}°")
        print(f"  - Actual longitude std: {actual_std_lon:.4f}°")
        print(f"  - Predicted longitude std: {pred_std_lon:.4f}°")

        # Calculate coverage ratio (how much of the actual area is covered by predictions)
        actual_lat_range = np.max(actual_lats) - np.min(actual_lats)
        actual_lon_range = np.max(actual_lons) - np.min(actual_lons)
        pred_lat_range = np.max(pred_lats) - np.min(pred_lats)
        pred_lon_range = np.max(pred_lons) - np.min(pred_lons)

        lat_coverage = pred_lat_range / actual_lat_range if actual_lat_range > 0 else 0
        lon_coverage = pred_lon_range / actual_lon_range if actual_lon_range > 0 else 0

        print(f"  - Latitude range coverage: {lat_coverage:.2f}")
        print(f"  - Longitude range coverage: {lon_coverage:.2f}")

    # Create heatmap comparison if we have enough data points
    if len(pred_lats) >= 4:  # Need enough points for a meaningful heatmap
        plt.figure(figsize=(16, 8))

        # Actual aftershocks heatmap
        plt.subplot(1, 2, 1)
        plt.hist2d(
            actual_lons,
            actual_lats,
            bins=min(10, len(actual_lons) // 2),
            cmap="Blues",
            alpha=0.7,
        )
        plt.colorbar(label="Count")
        plt.scatter(
            mainshock["source_longitude_deg"],
            mainshock["source_latitude_deg"],
            s=200,
            c="red",
            marker="*",
            label="Mainshock",
            edgecolor="black",
        )
        plt.title("Actual Aftershock Distribution")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()

        # Predicted aftershocks heatmap
        plt.subplot(1, 2, 2)
        plt.hist2d(
            pred_lons,
            pred_lats,
            bins=min(10, len(pred_lons) // 2),
            cmap="Greens",
            alpha=0.7,
        )
        plt.colorbar(label="Count")
        plt.scatter(
            mainshock["source_longitude_deg"],
            mainshock["source_latitude_deg"],
            s=200,
            c="red",
            marker="*",
            label="Mainshock",
            edgecolor="black",
        )
        plt.title("Predicted Aftershock Distribution")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/distribution_comparison.png", dpi=300)
        plt.close()

    return (
        mean_error,
        median_error,
        errors_km,
        (pred_lats, pred_lons, actual_lats, actual_lons),
    )
