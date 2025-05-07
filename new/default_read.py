# default_read.py

import numpy as np
import pandas as pd
from tqdm import tqdm
import seisbench.data as sbd
import pickle

def load_aftershock_data_with_CX_waveforms(top_n=15, min_stations=5):
    """
    Load aftershock data and select only CX stations for each unique event,
    considering both distance and signal quality.
    
    Args:
        top_n: Maximum number of stations to select per event
        min_stations: Minimum number of stations required to include an event
    """
    print("Loading Iquique dataset using SeisBench...")
    iquique = sbd.Iquique()
    
    # Get all metadata
    metadata = iquique.metadata.copy()
    
    cx_metadata = metadata[metadata['station_network_code'] == 'CX'].copy()
    
    if len(cx_metadata) == 0:
        print("No CX stations found in the dataset!")
        return {}, iquique
    
    print(f"Found {len(cx_metadata)} recordings from CX network out of {len(metadata)} total")
    
    # Group by full event parameters (not just origin_time)
    event_groups = cx_metadata.groupby(
        ['source_origin_time', 'source_latitude_deg', 'source_longitude_deg', 'source_depth_km']
    )
    
    # Calculate recordings per unique event
    recordings_per_event = event_groups.size()
    
    print(f"Recordings per unique event statistics:")
    print(f"  Total unique events: {len(recordings_per_event)}")
    print(f"  Average recordings per event: {recordings_per_event.mean():.2f}")
    print(f"  Median: {recordings_per_event.median()}")
    print(f"  Min: {recordings_per_event.min()}")
    print(f"  Max: {recordings_per_event.max()}")
    
    # Calculate events with at least min_stations recordings
    events_with_min_stations = sum(recordings_per_event >= min_stations)
    print(f"Events with at least {min_stations} station recordings: {events_with_min_stations}/{len(recordings_per_event)}")
    
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
    cx_metadata['epicentral_distance_km'] = cx_metadata.apply(
        lambda row: haversine_distance(
            row['source_latitude_deg'], 
            row['source_longitude_deg'],
            row['station_latitude_deg'], 
            row['station_longitude_deg']
        ), 
        axis=1
    )
    
    # Create a signal quality score (higher is better)
    cx_metadata['signal_quality_score'] = cx_metadata.apply(
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
    max_distance = cx_metadata['epicentral_distance_km'].max()
    cx_metadata['distance_normalized'] = 1 - (cx_metadata['epicentral_distance_km'] / max_distance)
    
    # Combined score (higher is better) - 60% weight on distance, 40% on quality
    cx_metadata['selection_score'] = 0.6 * cx_metadata['distance_normalized'] + 0.4 * cx_metadata['signal_quality_score']
    
    # Group by the event parameters
    event_groups = cx_metadata.groupby(
        ['source_origin_time', 'source_latitude_deg',
         'source_longitude_deg', 'source_depth_km']
    )

    data_dict = {}
    events_skipped = 0
    
    for event_key, group in tqdm(event_groups, desc="Events"):
        # 1. keep only recordings from different stations (though all are CX now)
        group_unique = (
            group.sort_values('selection_score', ascending=False)
                 .drop_duplicates(subset=['station_code'])  # Only need station_code since all are CX
        )

        # Check if we have at least min_stations unique stations
        if len(group_unique) < min_stations:
            events_skipped += 1
            continue  # Skip this event
            
        # 2. take the best `top_n` rows (or all if less than top_n)
        best_rows = group_unique.nlargest(min(top_n, len(group_unique)), 'selection_score')
        
        # Create temporary dict to collect station data for this event
        event_stations = {}

        # 3. store them
        for _, row in best_rows.iterrows():
            idx = row.name  # row index in metadata
            station_key = f"{row['station_network_code']}.{row['station_code']}"

            # get the waveform
            waveform = iquique.get_waveforms(int(idx))
            if waveform.shape[0] != 3:
                continue  # skip if components missing

            event_stations[station_key] = {
                'metadata': row.to_dict(),
                'waveform': waveform,
                'station_distance': row['epicentral_distance_km'],
                'selection_score': row['selection_score']
            }
        
        # Add the event only if we still have at least min_stations after checking waveforms
        if len(event_stations) >= min_stations:
            data_dict[event_key] = event_stations
        else:
            events_skipped += 1

    # Count how many events and stations were selected
    total_stations = sum(len(stations) for stations in data_dict.values())
    print(f"Selected {len(data_dict)} events with a total of {total_stations} CX station recordings")
    print(f"Skipped {events_skipped} events with fewer than {min_stations} valid stations")
    
    with open(f"aftershock_data_CX_only_8.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Saved {len(data_dict)} events with CX stations (min {min_stations}, max {top_n} stations each)")
    return data_dict, iquique



if __name__ == "__main__":
    # Load the Iquique dataset and preprocess
    data, iq = load_aftershock_data_with_CX_waveforms(14, 4)
