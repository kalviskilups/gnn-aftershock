import numpy as np

def prepare_sequences_with_relative_targets(
    aftershocks,
    waveform_features_dict,
    sequence_length=5,
    time_window_hours=72,
    max_sequences=5000,
    use_relative_targets=True,
):
    """
    Create aftershock sequences with waveform features, using relative targets
    (deviations from previous events) instead of absolute coordinates
    """
    sequences = []
    total_aftershocks = len(aftershocks)

    # Check which aftershocks have waveform features
    aftershocks_with_features = aftershocks[
        aftershocks["event_id"].isin(waveform_features_dict.keys())
    ]

    print(f"Total aftershocks: {total_aftershocks}")
    print(f"Aftershocks with waveform features: {len(aftershocks_with_features)}")

    # Need at least sequence_length+1 events to create a sequence with a target
    if len(aftershocks_with_features) < sequence_length + 1:
        print(
            f"Not enough aftershocks with waveform features to create sequences (need {sequence_length + 1}, have {len(aftershocks_with_features)})"
        )
        return []

    # Sort aftershocks by time
    aftershocks_sorted = aftershocks_with_features.sort_values("hours_since_mainshock")

    # Use a sliding window approach
    step_size = 1  # Create a sequence starting at each event

    for i in range(0, len(aftershocks_sorted) - sequence_length, step_size):
        # Get sequence of aftershocks
        current_sequence = aftershocks_sorted.iloc[i : i + sequence_length]
        target_aftershock = aftershocks_sorted.iloc[i + sequence_length]

        # Check if the sequence spans less than the time window
        seq_duration = (
            current_sequence.iloc[-1]["hours_since_mainshock"]
            - current_sequence.iloc[0]["hours_since_mainshock"]
        )
        if seq_duration > time_window_hours:
            continue

        # Check for sufficient spatial variation (with reduced threshold)
        lats = current_sequence["source_latitude_deg"].values
        lons = current_sequence["source_longitude_deg"].values

        lat_range = np.max(lats) - np.min(lats)
        lon_range = np.max(lons) - np.min(lons)

        # Reduced minimum variation threshold
        min_variation = 0.02  # ~2 km at this latitude
        if lat_range < min_variation and lon_range < min_variation:
            continue

        # Extract metadata features for each aftershock in the sequence
        metadata_features = current_sequence[
            [
                "source_latitude_deg",
                "source_longitude_deg",
                "source_depth_km",
                "hours_since_mainshock",
            ]
        ].values

        # Extract waveform features for each aftershock in the sequence
        sequence_waveform_features = []
        valid_sequence = True

        for _, row in current_sequence.iterrows():
            event_id = row["event_id"]
            if event_id in waveform_features_dict:
                features = waveform_features_dict[event_id]

                # Check if we have valid features
                if features and len(features) > 0:
                    sequence_waveform_features.append(features)
                else:
                    valid_sequence = False
                    break
            else:
                valid_sequence = False
                break

        # Skip if any event in the sequence doesn't have valid waveform features
        if not valid_sequence:
            continue

        # Target is the location of the next aftershock
        if use_relative_targets:
            # Use relative target (delta from last event in sequence)
            last_event_lat = current_sequence.iloc[-1]["source_latitude_deg"]
            last_event_lon = current_sequence.iloc[-1]["source_longitude_deg"]

            target_lat = target_aftershock["source_latitude_deg"]
            target_lon = target_aftershock["source_longitude_deg"]

            # Convert to kilometers for more numerically stable targets
            # Approximate conversion at these latitudes
            lat_km_per_degree = 111.0  # Approximate km per degree latitude
            lon_km_per_degree = 111.0 * np.cos(
                np.radians(last_event_lat)
            )  # Varies with latitude

            delta_lat_km = (target_lat - last_event_lat) * lat_km_per_degree
            delta_lon_km = (target_lon - last_event_lon) * lon_km_per_degree

            # Store as relative target
            target = np.array([delta_lat_km, delta_lon_km])

            # Also store reference coordinates for conversion back to absolute
            reference = np.array([last_event_lat, last_event_lon])

            sequences.append(
                (metadata_features, sequence_waveform_features, target, reference)
            )
        else:
            # Use absolute target coordinates (original approach)
            target = np.array(
                [
                    target_aftershock["source_latitude_deg"],
                    target_aftershock["source_longitude_deg"],
                ]
            )

            sequences.append((metadata_features, sequence_waveform_features, target))

        # Limit the number of sequences to avoid memory issues
        if len(sequences) >= max_sequences:
            print(f"Reached maximum number of sequences ({max_sequences})")
            break

    print(
        f"Created {len(sequences)} aftershock sequences with {'relative' if use_relative_targets else 'absolute'} targets"
    )
    return sequences


def convert_relative_to_absolute_predictions(pred_deltas, references):
    """
    Convert relative predictions (in km) back to absolute coordinates (lat/lon)

    Parameters:
    -----------
    pred_deltas : numpy.ndarray
        Predicted deltas in km, shape [n_samples, 2]
    references : numpy.ndarray
        Reference coordinates (lat/lon), shape [n_samples, 2]

    Returns:
    --------
    absolute_coords : numpy.ndarray
        Absolute coordinates (lat/lon), shape [n_samples, 2]
    """
    import numpy as np

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


def evaluate_relative_model(model, graph_dataset, mainshock, reference_coords):
    """
    Evaluate the model with relative position predictions

    Parameters:
    -----------
    model : torch.nn.Module
        The trained GNN model
    graph_dataset : list
        List of PyTorch Geometric Data objects
    mainshock : pandas.Series
        Information about the mainshock
    reference_coords : numpy.ndarray
        Reference coordinates for converting relative to absolute

    Returns:
    --------
    mean_error : float
        Mean prediction error in kilometers
    median_error : float
        Median prediction error in kilometers
    errors_km : list
        List of prediction errors in kilometers
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from torch_geometric.loader import DataLoader

    # Split into training and testing sets
    _, test_data = train_test_split(graph_dataset, test_size=0.2, random_state=42)

    # Create data loader
    test_loader = DataLoader(test_data, batch_size=32)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Model in evaluation mode
    model.eval()

    # Lists to store actual and predicted deltas
    actual_deltas = []
    pred_deltas = []

    # Evaluate on test set
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)

            # Forward pass
            lat_pred, lon_pred = model(
                batch.metadata, batch.waveform, batch.edge_index, batch.batch
            )

            # Extract predictions and targets
            lat_np = lat_pred.cpu().numpy()
            lon_np = lon_pred.cpu().numpy()

            # Combine into delta predictions
            batch_pred_deltas = np.column_stack((lat_np, lon_np))
            batch_actual_deltas = batch.y.cpu().numpy()

            actual_deltas.append(batch_actual_deltas)
            pred_deltas.append(batch_pred_deltas)

    # Concatenate results
    actual_deltas = np.vstack(actual_deltas)
    pred_deltas = np.vstack(pred_deltas)

    # Convert relative predictions to absolute coordinates
    absolute_pred = convert_relative_to_absolute_predictions(
        pred_deltas, reference_coords
    )
    absolute_actual = convert_relative_to_absolute_predictions(
        actual_deltas, reference_coords
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
    plt.savefig("results/relative_spatial_predictions.png", dpi=300)
    plt.close()

    # Additional plots to better understand the distribution

    # Plot error vs. distance from mainshock
    distances_from_mainshock = []
    for i in range(len(actual_lats)):
        lat, lon = actual_lats[i], actual_lons[i]
        main_lat = mainshock["source_latitude_deg"]
        main_lon = mainshock["source_longitude_deg"]

        # Calculate distance
        lat1, lon1 = np.radians(lat), np.radians(lon)
        lat2, lon2 = np.radians(main_lat), np.radians(main_lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c

        distances_from_mainshock.append(distance)

    plt.figure(figsize=(10, 6))
    plt.scatter(distances_from_mainshock, errors_km, alpha=0.7)
    plt.xlabel("Distance from Mainshock (km)")
    plt.ylabel("Prediction Error (km)")
    plt.title("Prediction Error vs. Distance from Mainshock")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/error_vs_distance.png", dpi=300)
    plt.close()

    return (
        mean_error,
        median_error,
        errors_km,
        (pred_lats, pred_lons, actual_lats, actual_lons),
    )
