import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


def train_model_with_diagnostics(
    graph_dataset, model, epochs=200, lr=0.001, batch_size=32, patience=15
):
    """
    Improved training function with diagnostics to monitor learning progress
    """
    import os
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from torch_geometric.loader import DataLoader
    import torch.nn.functional as F

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Split into training and validation sets
    train_data, val_data = train_test_split(
        graph_dataset, test_size=0.2, random_state=42
    )

    # Set follow_batch to ensure proper batching of all tensors
    follow_batch = ["metadata", "waveform"]

    # Adjust batch size if we have a small dataset
    batch_size = min(batch_size, len(train_data) // 2)
    if batch_size < 1:
        batch_size = 1

    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, follow_batch=follow_batch
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, follow_batch=follow_batch)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ===== IMPROVED OPTIMIZER =====
    # Use AdamW for better weight decay handling
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ===== IMPROVED LEARNING RATE SCHEDULER =====
    # Use OneCycleLR for better convergence
    from torch.optim.lr_scheduler import OneCycleLR

    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25.0,  # Initial LR = max_lr/25
        final_div_factor=10000.0,  # Final LR = max_lr/10000
    )

    # ===== IMPROVED LOSS FUNCTION =====
    def scaled_loss_function(pred, target, distribution_weight=0.3):
        """Loss function with proper scaling and L1 component"""
        # Scale down values for numerical stability
        scaled_pred = pred / 10.0  # Scale to tens of km
        scaled_target = target / 10.0

        # Combine L1 and L2 loss for better stability
        l1_loss = F.l1_loss(scaled_pred, scaled_target)
        l2_loss = F.mse_loss(scaled_pred, scaled_target)
        point_loss = 0.5 * l1_loss + 0.5 * l2_loss

        # Skip distribution component if batch size is too small
        if pred.size(0) <= 1:
            return point_loss

        # Distribution preservation with scaling
        pred_var = torch.var(scaled_pred, dim=0)
        target_var = torch.var(scaled_target, dim=0)

        # L1 loss for variance to avoid quadratic scaling
        variance_loss = F.l1_loss(pred_var, target_var)

        # Combine losses
        total_loss = (
            1 - distribution_weight
        ) * point_loss + distribution_weight * variance_loss

        return total_loss

    # For early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    # Training history
    train_losses = []
    val_losses = []

    # For monitoring predictions
    all_predictions = []

    # ===== DIAGNOSTIC FUNCTION =====
    def print_tensor_stats(prefix, tensor):
        """Print statistics about a tensor for diagnostics"""
        if tensor is not None:
            stats = {
                "mean": tensor.mean().item(),
                "std": tensor.std().item() if tensor.numel() > 1 else 0,
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "shape": tensor.shape,
            }
            print(f"{prefix}: {stats}")

    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        # Keep track of prediction changes
        epoch_predictions = []

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            try:
                # Forward pass
                lat_pred, lon_pred = model(
                    batch.metadata, batch.waveform, batch.edge_index, batch.batch
                )

                # Combine predictions
                pred = torch.cat([lat_pred, lon_pred], dim=1)

                # Save first batch predictions to monitor learning
                if batch_idx == 0:
                    epoch_predictions = pred.detach().cpu().numpy()

                # Print diagnostics every 50 epochs for the first batch
                if epoch % 50 == 0 and batch_idx == 0:
                    print(f"\nEpoch {epoch} diagnostics:")
                    print_tensor_stats("Predictions", pred)
                    print_tensor_stats("Targets", batch.y)

                    # Check gradients before backprop to see if they're flowing
                    pred_mean = pred.mean()
                    dummy_loss = (
                        pred_mean  # Compute a simple loss to check gradient flow
                    )
                    dummy_loss.backward(
                        retain_graph=True
                    )  # retain_graph=True keeps the computational graph

                    # Check if gradients are flowing to inputs
                    grad_norms = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_norms.append(grad_norm)

                    optimizer.zero_grad()  # Clear the dummy gradients

                    print(
                        f"Gradient stats - mean: {np.mean(grad_norms):.6f}, max: {np.max(grad_norms):.6f}, min: {np.min(grad_norms):.6f}"
                    )

                # Compute loss with distribution preservation
                loss = scaled_loss_function(pred, batch.y)

                # Backward pass
                loss.backward()

                # ===== IMPROVED GRADIENT CLIPPING =====
                # Use a more appropriate clipping value
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                optimizer.step()
                scheduler.step()  # Step LR scheduler every batch

                train_loss += loss.item() * batch.num_graphs

            except RuntimeError as e:
                print(f"Error in batch processing: {e}")
                # Skip this batch and continue with training
                continue

        train_loss /= len(train_data)
        train_losses.append(train_loss)

        # Store predictions to track changes
        all_predictions.append(epoch_predictions)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                try:
                    # Forward pass
                    lat_pred, lon_pred = model(
                        batch.metadata, batch.waveform, batch.edge_index, batch.batch
                    )

                    # Combine predictions
                    pred = torch.cat([lat_pred, lon_pred], dim=1)

                    # Compute loss
                    loss = scaled_loss_function(pred, batch.y)

                    val_loss += loss.item() * batch.num_graphs

                except RuntimeError as e:
                    print(f"Error in validation batch processing: {e}")
                    # Skip this batch
                    continue

        val_loss /= len(val_data)
        val_losses.append(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save the best model
            torch.save(model.state_dict(), "results/simplified_aftershock_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load the best model
    try:
        model.load_state_dict(torch.load("results/simplified_aftershock_model.pt"))
    except Exception as e:
        print(f"Warning: Could not load best model: {e}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simplified_training_history.png")
    plt.close()

    # ===== PREDICTION EVOLUTION ANALYSIS =====
    # Check if we have at least 3 epochs of predictions for analysis
    if len(all_predictions) >= 3:
        # Plot how predictions evolved over training
        plt.figure(figsize=(12, 8))
        colors = ["r", "g", "b", "c", "m", "y", "k"]

        # Plot predictions from selected epochs
        epochs_to_plot = [
            0,
            len(all_predictions) // 4,
            len(all_predictions) // 2,
            len(all_predictions) - 1,
        ]
        for i, epoch_idx in enumerate(epochs_to_plot):
            if epoch_idx < len(all_predictions) and len(all_predictions[epoch_idx]) > 0:
                epoch_preds = all_predictions[epoch_idx]
                plt.scatter(
                    epoch_preds[:, 1],  # Longitude predictions
                    epoch_preds[:, 0],  # Latitude predictions
                    color=colors[i % len(colors)],
                    marker="o",
                    s=50,
                    alpha=0.6,
                    label=f"Epoch {epoch_idx}",
                )

        plt.title("Evolution of Predictions During Training")
        plt.xlabel("Predicted Longitude Offset (km)")
        plt.ylabel("Predicted Latitude Offset (km)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("results/prediction_evolution.png")
        plt.close()

    return model, train_losses, val_losses


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
