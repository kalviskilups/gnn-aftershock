import numpy as np
import pandas as pd
from tqdm import tqdm
import seisbench.data as sbd
from scipy import signal


def load_aftershock_data_with_waveforms(max_waveforms):
    """
    Load and preprocess aftershock data from the Iquique dataset,
    including waveform data, and add the waveform components before dropping duplicates.
    """
    print("Loading Iquique dataset using SeisBench...")
    iquique = sbd.Iquique()

    # Get metadata and preserve original indices for waveform extraction.
    metadata = iquique.metadata.copy()
    # (Optionally, store the original index if needed later)
    metadata["orig_idx"] = metadata.index

    # Define the columns that uniquely identify an event.

    # All metadata columns are:
    # source_origin_time,source_latitude_deg,source_longitude_deg,source_depth_km,
    # path_back_azimuth_deg,station_network_code,station_code,trace_channel,station_location_code,
    # station_latitude_deg,station_longitude_deg,station_elevation_m,trace_name,trace_sampling_rate_hz,
    # trace_completeness,trace_has_spikes,trace_start_time,trace_P_arrival_sample,trace_S_arrival_sample,
    # trace_name_original,trace_chunk,trace_component_order,split
    unique_event_columns = [
        "source_origin_time",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
    ]

    # Filter out rows with missing essential data.
    metadata = metadata.dropna(subset=unique_event_columns)

    # Optional: sort by datetime for a consistent ordering.
    metadata["datetime"] = pd.to_datetime(metadata["source_origin_time"])
    metadata = metadata.sort_values("datetime")

    # Limit to process only a subset, if desired.
    metadata = metadata.iloc[: min(max_waveforms, len(metadata))].copy()

    print(f"Extracting waveform features for {len(metadata)} events...")

    # Create an empty list to store waveform data for each row.
    waveforms = []
    # Extract waveforms row by row and add them to the list.
    for idx in tqdm(metadata.index):
        try:
            # Use the original index (or current index if they match) to retrieve waveforms.
            waveform = iquique.get_waveforms(int(metadata.loc[idx, "orig_idx"]))

            # Example output of get_waveforms(idx):
            # The array contains waveform recordings from a single seismic station.
            # Output component order not specified, defaulting to 'ZNE'.
            # [[-193.04892508 -196.04892508 -175.04892508 ...  314.95107492
            # 316.95107492  352.95107492]
            # [ 112.44318263  121.44318263  117.44318263 ...  195.44318263
            # 167.44318263  106.44318263]
            # [  -4.07892293    1.92107707    2.92107707 ...  154.92107707
            # 191.92107707  206.92107707]]

            waveforms.append(waveform)
        except Exception as e:
            print(f"Error processing waveform for row {idx}: {e}")
            waveforms.append(None)

    # Add waveform data as a new column.
    metadata["waveform"] = waveforms

    # Now drop duplicates based on unique event columns;
    # Note that this will drop duplicate events, retaining the first occurrence (and its waveform data).
    unique_metadata = metadata.drop_duplicates(subset=unique_event_columns).reset_index(
        drop=True
    )

    print(f"Total unique events in dataset: {len(unique_metadata)}")
    return unique_metadata, iquique


def identify_mainshock_and_aftershocks(metadata):
    """
    Identify the mainshock and its associated aftershocks based on data patterns
    """
    # The known Iquique earthquake was in early April 2014
    # Let's create a date range for the expected period
    april_start = pd.Timestamp("2014-04-01", tz="UTC")
    april_end = pd.Timestamp("2014-04-05", tz="UTC")

    # Find events in early April 2014
    april_events = metadata[
        (metadata["datetime"] >= april_start) & (metadata["datetime"] <= april_end)
    ]

    if len(april_events) == 0:
        print(
            "No events found in early April 2014 timeframe. Using alternative approach."
        )
        # Alternative approach: find the earliest events in the dataset
        metadata_sorted = metadata.sort_values("datetime")
        earliest_date = metadata_sorted["datetime"].iloc[0]
        print(f"Earliest event in dataset: {earliest_date}")

        # Take the earliest events as potential mainshock candidates
        potential_mainshocks = metadata_sorted.iloc[:100]  # First 100 events

        # Find which event has most events in the following 24 hours
        best_event_idx = None
        most_followers = 0

        for idx, row in potential_mainshocks.iterrows():
            event_time = row["datetime"]
            next_24h = event_time + pd.Timedelta(hours=24)
            followers = sum(metadata["datetime"].between(event_time, next_24h))

            if followers > most_followers:
                most_followers = followers
                best_event_idx = idx

        mainshock = metadata.loc[best_event_idx]
    else:
        print(f"Found {len(april_events)} events in early April 2014.")
        # Within this timeframe, find the event with the most followers in 24 hours
        best_event_idx = None
        most_followers = 0

        for idx, row in april_events.iterrows():
            event_time = row["datetime"]
            next_24h = event_time + pd.Timedelta(hours=24)
            followers = sum(metadata["datetime"].between(event_time, next_24h))

            if followers > most_followers:
                most_followers = followers
                best_event_idx = idx

        mainshock = metadata.loc[best_event_idx]

    print(
        f"Identified potential mainshock at {mainshock['datetime']} at location "
        f"({mainshock['source_latitude_deg']}, {mainshock['source_longitude_deg']}), "
        f"depth {mainshock['source_depth_km']} km"
    )
    print(f"This event is followed by {most_followers} events in the next 24 hours")

    # Select events after the mainshock as aftershocks
    aftershocks = metadata[metadata["datetime"] > mainshock["datetime"]].copy()
    print(f"Found {len(aftershocks)} aftershocks")

    # Create a new feature: time since mainshock in hours
    aftershocks["hours_since_mainshock"] = (
        aftershocks["datetime"] - mainshock["datetime"]
    ).dt.total_seconds() / 3600

    # Create features: distance from mainshock (approximate using Haversine)
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c

        return distance

    aftershocks["distance_from_mainshock_km"] = aftershocks.apply(
        lambda row: haversine_distance(
            mainshock["source_latitude_deg"],
            mainshock["source_longitude_deg"],
            row["source_latitude_deg"],
            row["source_longitude_deg"],
        ),
        axis=1,
    )

    return mainshock, aftershocks


if __name__ == "__main__":
    # Load the Iquique dataset and preprocess
    metadata, iquique = load_aftershock_data_with_waveforms(max_waveforms=13400)
