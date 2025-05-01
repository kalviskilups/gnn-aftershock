import numpy as np
import pandas as pd
from tqdm import tqdm
import seisbench.data as sbd
import pickle

def load_aftershock_data_with_waveforms(max_waveforms):
    """
    Load aftershock data and waveforms into a Python dictionary to reduce overhead.
    """
    print("Loading Iquique dataset using SeisBench...")
    iquique = sbd.Iquique()

    # All metadata columns are:
    # source_origin_time,source_latitude_deg,source_longitude_deg,source_depth_km,
    # path_back_azimuth_deg,station_network_code,station_code,trace_channel,station_location_code,
    # station_latitude_deg,station_longitude_deg,station_elevation_m,trace_name,trace_sampling_rate_hz,
    # trace_completeness,trace_has_spikes,trace_start_time,trace_P_arrival_sample,trace_S_arrival_sample,
    # trace_name_original,trace_chunk,trace_component_order,split
    metadata = iquique.metadata.copy()

    # Cap the data
    metadata = metadata.iloc[: min(max_waveforms, len(metadata))].copy()

    data_dict = {}
    for idx in tqdm(metadata.index):
        event_key = (
            metadata.loc[idx, "source_origin_time"],
            metadata.loc[idx, "source_latitude_deg"],
            metadata.loc[idx, "source_longitude_deg"],
            metadata.loc[idx, "source_depth_km"],
        )
        # Example output of get_waveforms(idx):
        # The array contains waveform recordings from a single seismic station.
        # Output component order not specified, defaulting to 'ZNE'.
        # [[-193.04892508 -196.04892508 -175.04892508 ...  314.95107492
        # 316.95107492  352.95107492]
        # [ 112.44318263  121.44318263  117.44318263 ...  195.44318263
        # 167.44318263  106.44318263]
        # [  -4.07892293    1.92107707    2.92107707 ...  154.92107707
        # 191.92107707  206.92107707]]
        waveform = iquique.get_waveforms(int(idx))
        data_dict[event_key] = {
            "metadata": metadata.loc[idx].to_dict(),
            "waveform": waveform,
        }

    with open("aftershock_data.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Data dictionary saved to aftershock_data.pkl")

    print(f"Total events stored: {len(data_dict)}")
    return data_dict, iquique


def load_aftershock_data_with_best_waveforms():
    """
    Load aftershock data and select the best waveform for each unique event,
    considering both distance and signal quality.
    """
    print("Loading Iquique dataset using SeisBench...")
    iquique = sbd.Iquique()
    
    # Get all metadata
    metadata = iquique.metadata.copy()
    
    # Calculate distance from each station to the event epicenter
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance between two points in km."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers

        return c * r
    
    # Calculate distance from station to epicenter for each recording
    metadata['epicentral_distance_km'] = metadata.apply(
        lambda row: haversine_distance(
            row['source_latitude_deg'], 
            row['source_longitude_deg'],
            row['station_latitude_deg'], 
            row['station_longitude_deg']
        ), 
        axis=1
    )
    
    # Create a signal quality score (higher is better)
    metadata['signal_quality_score'] = metadata.apply(
        lambda row: (
            (row['trace_completeness'] if pd.notna(row['trace_completeness']) else 0) * 0.5 +  # 50% weight for completeness
            (0.3 if not row['trace_has_spikes'] else 0) +  # 30% weight for no spikes
            (0.2 if (pd.notna(row['trace_P_arrival_sample']) and pd.notna(row['trace_S_arrival_sample']) and 
                    row['trace_P_arrival_sample'] >= 0 and row['trace_S_arrival_sample'] >= 0) else 0)  # 20% weight for P/S arrivals
        ),
        axis=1
    )
    
    # Combine distance and quality into a single selection metric
    # Normalize epicentral distance (lower is better)
    max_distance = metadata['epicentral_distance_km'].max()
    metadata['distance_normalized'] = 1 - (metadata['epicentral_distance_km'] / max_distance)
    
    # Combined score (higher is better) - 60% weight on distance, 40% on quality
    metadata['selection_score'] = 0.6 * metadata['distance_normalized'] + 0.4 * metadata['signal_quality_score']
    
    # Group by event parameters to identify unique events
    event_groups = metadata.groupby([
        'source_origin_time',
        'source_latitude_deg',
        'source_longitude_deg',
        'source_depth_km'
    ])
    
    # For each unique event, select the recording with the highest selection score
    best_recordings = []
    for event_params, group in tqdm(event_groups):
        # Find the index of the recording with the highest selection score
        best_idx = group['selection_score'].idxmax()
        best_recordings.append(best_idx)
    
    # Filter metadata to keep only the best recordings
    filtered_metadata = metadata.loc[best_recordings].copy()
    
    print(f"Found {len(filtered_metadata)} unique events with their best station recordings")
    
    # Create data dictionary with one entry per unique event
    data_dict = {}
    for idx in tqdm(filtered_metadata.index):
        event_key = (
            metadata.loc[idx, "source_origin_time"],
            metadata.loc[idx, "source_latitude_deg"],
            metadata.loc[idx, "source_longitude_deg"],
            metadata.loc[idx, "source_depth_km"],
        )
        
        # Get waveform from the best station
        waveform = iquique.get_waveforms(int(idx))
        
        # Ensure waveform has all three components
        if waveform.shape[0] == 3:
            # Store metadata and waveform
            data_dict[event_key] = {
                "metadata": metadata.loc[idx].to_dict(),
                "waveform": waveform,
                "station_distance": metadata.loc[idx, "epicentral_distance_km"],
                "selection_score": metadata.loc[idx, "selection_score"]
            }

    with open("aftershock_data_best.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Data dictionary saved to aftershock_data_best.pkl")

    print(f"Total unique events stored: {len(data_dict)}")
    return data_dict, iquique


def load_aftershock_data_with_best_waveforms_top_n(top_n=13):
    """
    Load aftershock data and select the best waveform for each unique event,
    considering both distance and signal quality.
    """
    print("Loading Iquique dataset using SeisBench...")
    iquique = sbd.Iquique()
    
    # Get all metadata
    metadata = iquique.metadata.copy()
    
    # Calculate distance from each station to the event epicenter
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance between two points in km."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers

        return c * r
    
    # Calculate distance from station to epicenter for each recording
    metadata['epicentral_distance_km'] = metadata.apply(
        lambda row: haversine_distance(
            row['source_latitude_deg'], 
            row['source_longitude_deg'],
            row['station_latitude_deg'], 
            row['station_longitude_deg']
        ), 
        axis=1
    )
    
    # Create a signal quality score (higher is better)
    metadata['signal_quality_score'] = metadata.apply(
        lambda row: (
            (row['trace_completeness'] if pd.notna(row['trace_completeness']) else 0) * 0.5 +  # 50% weight for completeness
            (0.3 if not row['trace_has_spikes'] else 0) +  # 30% weight for no spikes
            (0.2 if (pd.notna(row['trace_P_arrival_sample']) and pd.notna(row['trace_S_arrival_sample']) and 
                    row['trace_P_arrival_sample'] >= 0 and row['trace_S_arrival_sample'] >= 0) else 0)  # 20% weight for P/S arrivals
        ),
        axis=1
    )
    
    # Combine distance and quality into a single selection metric
    # Normalize epicentral distance (lower is better)
    max_distance = metadata['epicentral_distance_km'].max()
    metadata['distance_normalized'] = 1 - (metadata['epicentral_distance_km'] / max_distance)
    
    # Combined score (higher is better) - 60% weight on distance, 40% on quality
    metadata['selection_score'] = 0.6 * metadata['distance_normalized'] + 0.4 * metadata['signal_quality_score']
    
    # Group by the event parameters
    event_groups = metadata.groupby(
        ['source_origin_time', 'source_latitude_deg',
         'source_longitude_deg', 'source_depth_km']
    )

    data_dict = {}
    for event_key, group in tqdm(event_groups, desc="Events"):
        # 1. keep only recordings from different stations
        group_unique = (
            group.sort_values('selection_score', ascending=False)
                 .drop_duplicates(subset=['station_network_code', 'station_code'])
        )

        # 2. take the best `top_n` rows
        best_rows = group_unique.nlargest(top_n, 'selection_score')

        # 3. store them
        for _, row in best_rows.iterrows():
            idx = row.name  # row index in metadata
            station_key = f"{row['station_network_code']}.{row['station_code']}"

            # get the waveform
            waveform = iquique.get_waveforms(int(idx))
            if waveform.shape[0] != 3:
                continue  # skip if components missing

            data_dict.setdefault(event_key, {})[station_key] = {
                'metadata': row.to_dict(),
                'waveform': waveform,
                'station_distance': row['epicentral_distance_km'],
                'selection_score': row['selection_score']
            }

    with open("aftershock_data_topN13.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Saved {len(data_dict)} events (up to {top_n} stations each)")
    return data_dict, iquique


def read_data_from_pickle(file_path):
    """
    Read the multi-station format data from pickle file
    """
    # Load the pickle file that contains the data dictionary
    with open(file_path, "rb") as file:
        data_dict = pickle.load(file)
    
    # Check the first entry to determine data structure
    first_event_key = next(iter(data_dict))
    first_event_data = data_dict[first_event_key]
    
    # Check if this is multi-station format
    is_multi_station = isinstance(first_event_data, dict) and any(isinstance(k, str) and '.' in k for k in first_event_data.keys())
    
    data_list = []
    
    if is_multi_station:
        print(f"Detected multi-station format with {len(data_dict)} events")
        # For each event
        for event_key, stations_data in data_dict.items():
            origin_time, lat, lon, depth = event_key
            
            # For each station recording of this event
            for station_key, station_data in stations_data.items():
                # Extract metadata and add event location info
                record = {
                    **station_data["metadata"],
                    "waveform": station_data["waveform"],
                    "station_distance": station_data["station_distance"],
                    "selection_score": station_data["selection_score"],
                    "station_key": station_key,
                    "event_key": str(event_key),  # Convert tuple to string for DataFrame
                    "origin_time": origin_time,
                    "source_latitude_deg": lat,
                    "source_longitude_deg": lon,
                    "source_depth_km": depth
                }
                data_list.append(record)
        
        print(f"Total of {len(data_list)} station recordings")
    else:
        print("Detected single-station format")
        # Original format - each event has one recording
        for event_key, entry in data_dict.items():
            origin_time, lat, lon, depth = event_key
            record = {
                **entry["metadata"],
                "waveform": entry["waveform"],
                "event_key": str(event_key),
                "origin_time": origin_time,
                "source_latitude_deg": lat,
                "source_longitude_deg": lon,
                "source_depth_km": depth
            }
            data_list.append(record)
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data_list)
    
    return df


def load_aftershock_data_with_all_stations():
    """
    Load aftershock data with ALL available station recordings,
    processing in batches to avoid memory issues.
    """
    print("Loading Iquique dataset using SeisBench...")
    iquique = sbd.Iquique()
    
    # Get all metadata
    metadata = iquique.metadata.copy()
    
    # Calculate distance from each station to the event epicenter
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance between two points in km."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers

        return c * r
    
    # Calculate distance from station to epicenter for each recording
    print("Calculating epicentral distances...")
    metadata['epicentral_distance_km'] = metadata.apply(
        lambda row: haversine_distance(
            row['source_latitude_deg'], 
            row['source_longitude_deg'],
            row['station_latitude_deg'], 
            row['station_longitude_deg']
        ), 
        axis=1
    )
    
    # Create a signal quality score (higher is better)
    print("Calculating quality scores...")
    metadata['signal_quality_score'] = metadata.apply(
        lambda row: (
            (row['trace_completeness'] if pd.notna(row['trace_completeness']) else 0) * 0.5 +  # 50% weight for completeness
            (0.3 if not row['trace_has_spikes'] else 0) +  # 30% weight for no spikes
            (0.2 if (pd.notna(row['trace_P_arrival_sample']) and pd.notna(row['trace_S_arrival_sample']) and 
                    row['trace_P_arrival_sample'] >= 0 and row['trace_S_arrival_sample'] >= 0) else 0)  # 20% weight for P/S arrivals
        ),
        axis=1
    )
    
    # Combine distance and quality into a single selection metric
    # Normalize epicentral distance (lower is better)
    max_distance = metadata['epicentral_distance_km'].max()
    metadata['distance_normalized'] = 1 - (metadata['epicentral_distance_km'] / max_distance)
    
    # Combined score (higher is better) - 60% weight on distance, 40% on quality
    metadata['selection_score'] = 0.6 * metadata['distance_normalized'] + 0.4 * metadata['signal_quality_score']
    
    # Group by the event parameters
    event_groups = metadata.groupby(
        ['source_origin_time', 'source_latitude_deg',
         'source_longitude_deg', 'source_depth_km']
    )

    # Get list of unique event keys
    event_keys = list(event_groups.groups.keys())
    
    # Process in batches of 20 events
    BATCH_SIZE = 5
    output_file = "aftershock_data_all_stations.pkl"
    
    total_events = 0
    total_stations = 0
    
    # Create empty dict that will be populated and periodically saved
    master_data_dict = {}
    
    for batch_start in range(0, len(event_keys), BATCH_SIZE):
        # Get current batch of events
        batch_end = min(batch_start + BATCH_SIZE, len(event_keys))
        current_batch = event_keys[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1}/{(len(event_keys)-1)//BATCH_SIZE + 1} (events {batch_start+1}-{batch_end})")
        
        # Process this batch
        batch_data_dict = {}
        batch_events = 0
        batch_stations = 0
        
        for event_key in tqdm(current_batch, desc="Events in batch"):
            # Get group for this event
            group = event_groups.get_group(event_key)
            
            # Keep only recordings from different stations (avoid duplicates)
            group_unique = (
                group.sort_values('selection_score', ascending=False)
                     .drop_duplicates(subset=['station_network_code', 'station_code'])
            )
            
            # Store recordings for this event
            event_stations = 0
            for _, row in group_unique.iterrows():
                idx = row.name  # row index in metadata
                station_key = f"{row['station_network_code']}.{row['station_code']}"
                
                try:
                    # Get the waveform
                    waveform = iquique.get_waveforms(int(idx))
                    if waveform.shape[0] != 3:
                        continue  # skip if components missing
                    
                    # Add to batch dictionary
                    batch_data_dict.setdefault(event_key, {})[station_key] = {
                        'metadata': row.to_dict(),
                        'waveform': waveform,
                        'station_distance': row['epicentral_distance_km'],
                        'selection_score': row['selection_score']
                    }
                    event_stations += 1
                    batch_stations += 1
                except Exception as e:
                    print(f"Error loading waveform for idx {idx}: {e}")
            
            if event_stations > 0:
                batch_events += 1
                print(f"  Event {event_key[0]}: Loaded {event_stations} unique stations")
        
        # Update totals
        total_events += batch_events
        total_stations += batch_stations
        
        # Update master dictionary
        master_data_dict.update(batch_data_dict)
        
        # Save after each batch to prevent data loss
        print(f"Saving batch progress ({len(master_data_dict)} events, {total_stations} stations so far)...")
        with open(output_file, "wb") as f:
            pickle.dump(master_data_dict, f)
        
        # Clear batch dictionary to free memory
        del batch_data_dict
        
        print(f"Batch complete: Added {batch_events} events with {batch_stations} stations")
    
    print(f"\nAll batches complete!")
    print(f"Saved {len(master_data_dict)} events with {total_stations} total station recordings to {output_file}")
    
    return master_data_dict, iquique


if __name__ == "__main__":
    # Load the Iquique dataset and preprocess
    data, iq = load_aftershock_data_with_all_stations()
