#!/usr/bin/env python3
# efficient_aftershock_processor.py - Memory-efficient version for large datasets

import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
from datetime import datetime

class MemoryEfficientAfterShockProcessor:
    """
    Memory-efficient processor for large aftershock datasets
    that avoids loading all data into memory at once
    """
    def __init__(self, input_file=None):
        """
        Initialize the processor
        
        Args:
            input_file: Path to HDF5 file with aftershock data
        """
        self.input_file = input_file
        self.output_file = None
        self.data_format = None
        self.mainshock_key = None
        self.mainshock = {
            "origin_time": "2014-04-01T23:46:50.000000Z",
            "latitude": -19.642,
            "longitude": -70.817,
            "depth": 25.0,
        }
        self.mainshock_key = (
            self.mainshock["origin_time"],
            self.mainshock["latitude"],
            self.mainshock["longitude"],
            self.mainshock["depth"],
        )
    
    def standardize_waveforms(self, output_file=None, target_length=14636, batch_size=10):
        """
        Memory-efficient standardization of waveforms that processes in small batches
        
        Args:
            output_file: Path to save the standardized HDF5 file
            target_length: Target length for all waveforms
            batch_size: Number of events to process at once
            
        Returns:
            Path to the standardized HDF5 file
        """
        if output_file is None:
            base_name = os.path.splitext(self.input_file)[0]
            output_file = f"{base_name}_standardized.h5"
        
        self.output_file = output_file
        
        print("\n" + "=" * 50)
        print(f"STANDARDIZING WAVEFORMS TO {target_length} SAMPLES")
        print("=" * 50)
        
        start_time = time.time()
        
        # Open the input file in read mode
        with h5py.File(self.input_file, "r") as source_file:
            # Get metadata
            if 'metadata' in source_file:
                total_events = source_file['metadata'].attrs.get('total_events', 0)
                total_stations = source_file['metadata'].attrs.get('total_stations', 0)
                print(f"Input file contains {total_events} events and {total_stations} station recordings")
            
            # Check if 'events' group exists
            if 'events' not in source_file:
                raise ValueError(f"Invalid HDF5 file structure: 'events' group not found")
                
            # Determine data format
            is_multi_station = False
            for event_key in list(source_file['events'].keys())[:1]:  # Check just the first event
                event_group = source_file['events'][event_key]
                if 'stations' in event_group:
                    is_multi_station = True
                    break
            
            self.data_format = "multi_station" if is_multi_station else "single_station"
            print(f"Detected {self.data_format} data format")
            
            # Create a new HDF5 file for the standardized data
            with h5py.File(output_file, "w") as target_file:
                # Copy metadata group
                if 'metadata' in source_file:
                    metadata_group = target_file.create_group("metadata")
                    for key, value in source_file['metadata'].attrs.items():
                        metadata_group.attrs[key] = value
                
                # Create events group
                events_group = target_file.create_group("events")
                
                # Get all event keys
                event_keys = list(source_file['events'].keys())
                
                # Process in batches
                modified_count = 0
                
                for batch_start in range(0, len(event_keys), batch_size):
                    batch_end = min(batch_start + batch_size, len(event_keys))
                    current_batch = event_keys[batch_start:batch_end]
                    
                    print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(event_keys)-1)//batch_size + 1} (events {batch_start+1}-{batch_end})")
                    
                    batch_modified = 0
                    
                    # Process each event in the batch
                    for event_key in tqdm(current_batch, desc="Events"):
                        source_event = source_file['events'][event_key]
                        
                        # Create corresponding event group in target file
                        target_event = events_group.create_group(event_key)
                        
                        # Copy event metadata
                        if 'event_metadata' in source_event:
                            target_event_meta = target_event.create_group("event_metadata")
                            for key, value in source_event['event_metadata'].attrs.items():
                                target_event_meta.attrs[key] = value
                        
                        if is_multi_station:
                            # Multi-station format
                            if 'stations' in source_event:
                                source_stations = source_event['stations']
                                target_stations = target_event.create_group("stations")
                                
                                # Process each station
                                for station_key in source_stations:
                                    source_station = source_stations[station_key]
                                    target_station = target_stations.create_group(station_key)
                                    
                                    # Get waveform
                                    if 'waveform' in source_station:
                                        waveform = source_station['waveform'][:]
                                        
                                        # Standardize waveform length
                                        if waveform.shape[1] != target_length:
                                            if waveform.shape[1] > target_length:
                                                # Trim
                                                waveform = waveform[:, :target_length]
                                            else:
                                                # Pad
                                                padded = np.zeros((3, target_length))
                                                padded[:, :waveform.shape[1]] = waveform
                                                waveform = padded
                                            
                                            batch_modified += 1
                                        
                                        # Save standardized waveform
                                        target_station.create_dataset("waveform", data=waveform)
                                    
                                    # Copy other data (metadata, attributes)
                                    if 'metadata' in source_station:
                                        target_meta = target_station.create_group("metadata")
                                        for key, value in source_station['metadata'].attrs.items():
                                            target_meta.attrs[key] = value
                                    
                                    # Copy attributes
                                    for key, value in source_station.attrs.items():
                                        target_station.attrs[key] = value
                        else:
                            # Single-station format
                            if 'waveform' in source_event:
                                waveform = source_event['waveform'][:]
                                
                                # Standardize waveform length
                                if waveform.shape[1] != target_length:
                                    if waveform.shape[1] > target_length:
                                        # Trim
                                        waveform = waveform[:, :target_length]
                                    else:
                                        # Pad
                                        padded = np.zeros((3, target_length))
                                        padded[:, :waveform.shape[1]] = waveform
                                        waveform = padded
                                    
                                    batch_modified += 1
                                
                                # Save standardized waveform
                                target_event.create_dataset("waveform", data=waveform)
                            
                            # Copy metadata
                            if 'metadata' in source_event:
                                target_meta = target_event.create_group("metadata")
                                for key, value in source_event['metadata'].attrs.items():
                                    target_meta.attrs[key] = value
                    
                    modified_count += batch_modified
                    print(f"Batch complete: Standardized {batch_modified} waveforms in this batch")
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f"\nStandardization complete!")
                print(f"Total waveforms standardized: {modified_count}")
                print(f"Standardized data saved to {output_file}")
                print(f"Processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        return output_file
    
    def create_filtered_dataset(self, output_file=None, max_stations_per_event=2, selection_method="best"):
        """
        Create a filtered dataset with a limited number of stations per event
        to reduce memory requirements for processing
        
        Args:
            output_file: Path to save the filtered HDF5 file
            max_stations_per_event: Maximum number of stations to keep per event
            selection_method: How to select stations ("best", "random", "first")
            
        Returns:
            Path to the filtered HDF5 file
        """
        if output_file is None:
            base_name = os.path.splitext(self.input_file)[0]
            output_file = f"{base_name}_filtered_{max_stations_per_event}stations.h5"
        
        input_file = self.output_file if self.output_file else self.input_file
        
        print("\n" + "=" * 50)
        print(f"CREATING FILTERED DATASET WITH {max_stations_per_event} STATIONS PER EVENT")
        print("=" * 50)
        
        start_time = time.time()
        
        # Open the input file in read mode
        with h5py.File(input_file, "r") as source_file:
            # Verify this is a multi-station dataset
            is_multi_station = False
            for event_key in list(source_file['events'].keys())[:1]:  # Check just first event
                event_group = source_file['events'][event_key]
                if 'stations' in event_group:
                    is_multi_station = True
                    break
            
            if not is_multi_station:
                print("This is a single-station dataset - no filtering needed")
                return input_file
            
            # Create a new HDF5 file for the filtered data
            with h5py.File(output_file, "w") as target_file:
                # Copy metadata group
                if 'metadata' in source_file:
                    metadata_group = target_file.create_group("metadata")
                    for key, value in source_file['metadata'].attrs.items():
                        metadata_group.attrs[key] = value
                
                # Create events group
                events_group = target_file.create_group("events")
                
                # Process each event
                total_events = 0
                total_stations = 0
                
                for event_key in tqdm(source_file['events'], desc="Filtering events"):
                    source_event = source_file['events'][event_key]
                    
                    # Skip events without stations group
                    if 'stations' not in source_event:
                        continue
                    
                    # Create corresponding event group in target file
                    target_event = events_group.create_group(event_key)
                    
                    # Copy event metadata
                    if 'event_metadata' in source_event:
                        target_event_meta = target_event.create_group("event_metadata")
                        for key, value in source_event['event_metadata'].attrs.items():
                            target_event_meta.attrs[key] = value
                    
                    # Get all station keys
                    station_keys = list(source_event['stations'].keys())
                    
                    if len(station_keys) <= max_stations_per_event:
                        # Keep all stations if there are fewer than the maximum
                        selected_keys = station_keys
                    else:
                        # Select stations based on the specified method
                        if selection_method == "random":
                            # Random selection
                            np.random.seed(42)  # For reproducibility
                            selected_keys = np.random.choice(station_keys, max_stations_per_event, replace=False)
                        elif selection_method == "best":
                            # Select stations with highest selection_score
                            scores = []
                            for station_key in station_keys:
                                score = source_event['stations'][station_key].attrs.get('selection_score', 0)
                                scores.append((station_key, score))
                            
                            # Sort by score (descending) and take top N
                            scores.sort(key=lambda x: x[1], reverse=True)
                            selected_keys = [pair[0] for pair in scores[:max_stations_per_event]]
                        else:
                            # Default to first N stations
                            selected_keys = station_keys[:max_stations_per_event]
                    
                    # Create stations group in target
                    target_stations = target_event.create_group("stations")
                    
                    # Copy selected stations
                    for station_key in selected_keys:
                        source_station = source_event['stations'][station_key]
                        target_station = target_stations.create_group(station_key)
                        
                        # Copy waveform
                        if 'waveform' in source_station:
                            waveform = source_station['waveform'][:]
                            target_station.create_dataset("waveform", data=waveform)
                        
                        # Copy metadata
                        if 'metadata' in source_station:
                            target_meta = target_station.create_group("metadata")
                            for key, value in source_station['metadata'].attrs.items():
                                target_meta.attrs[key] = value
                        
                        # Copy attributes
                        for key, value in source_station.attrs.items():
                            target_station.attrs[key] = value
                        
                        total_stations += 1
                    
                    # Update event attributes
                    target_event.attrs["num_stations"] = len(selected_keys)
                    total_events += 1
                
                # Update metadata
                metadata_group.attrs["total_events"] = total_events
                metadata_group.attrs["total_stations"] = total_stations
                metadata_group.attrs["data_format"] = "multi_station"
                metadata_group.attrs["max_stations_per_event"] = max_stations_per_event
                metadata_group.attrs["selection_method"] = selection_method
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f"\nFiltering complete!")
                print(f"Kept {total_events} events with {total_stations} station recordings")
                print(f"Average stations per event: {total_stations/total_events:.2f}")
                print(f"Filtered data saved to {output_file}")
                print(f"Processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        return output_file
    
    def create_best_station_dataset(self, output_file=None):
        """
        Create a simplified dataset with just the best station for each event
        (essentially converting multi-station to single-station format)
        
        Args:
            output_file: Path to save the best-station HDF5 file
            
        Returns:
            Path to the best-station HDF5 file
        """
        if output_file is None:
            base_name = os.path.splitext(self.input_file)[0]
            output_file = f"{base_name}_best_station.h5"
        
        input_file = self.output_file if self.output_file else self.input_file
        
        print("\n" + "=" * 50)
        print("CREATING BEST-STATION DATASET")
        print("=" * 50)
        
        start_time = time.time()
        
        # Open the input file in read mode
        with h5py.File(input_file, "r") as source_file:
            # Verify this is a multi-station dataset
            is_multi_station = False
            for event_key in list(source_file['events'].keys())[:1]:  # Check just first event
                event_group = source_file['events'][event_key]
                if 'stations' in event_group:
                    is_multi_station = True
                    break
            
            if not is_multi_station:
                print("This is already a single-station dataset - no conversion needed")
                return input_file
            
            # Create a new HDF5 file for the best-station data
            with h5py.File(output_file, "w") as target_file:
                # Create metadata group
                metadata_group = target_file.create_group("metadata")
                
                # Create events group
                events_group = target_file.create_group("events")
                
                # Process each event
                total_events = 0
                
                for event_key in tqdm(source_file['events'], desc="Processing events"):
                    source_event = source_file['events'][event_key]
                    
                    # Skip events without stations group
                    if 'stations' not in source_event:
                        continue
                    
                    # Create corresponding event group in target file
                    target_event = events_group.create_group(event_key)
                    
                    # Copy event metadata
                    if 'event_metadata' in source_event:
                        target_event_meta = target_event.create_group("event_metadata")
                        for key, value in source_event['event_metadata'].attrs.items():
                            target_event_meta.attrs[key] = value
                    
                    # Find the best station (highest selection_score)
                    best_station_key = None
                    best_score = -1
                    
                    for station_key in source_event['stations']:
                        station = source_event['stations'][station_key]
                        score = station.attrs.get('selection_score', 0)
                        
                        if score > best_score:
                            best_score = score
                            best_station_key = station_key
                    
                    if best_station_key is None:
                        # If no selection_score found, use the first station
                        best_station_key = list(source_event['stations'].keys())[0]
                    
                    # Get the best station data
                    best_station = source_event['stations'][best_station_key]
                    
                    # Copy waveform directly to event level (single-station format)
                    if 'waveform' in best_station:
                        waveform = best_station['waveform'][:]
                        target_event.create_dataset("waveform", data=waveform)
                    
                    # Create metadata at event level from station metadata
                    target_meta = target_event.create_group("metadata")
                    
                    # Copy original station metadata
                    if 'metadata' in best_station:
                        for key, value in best_station['metadata'].attrs.items():
                            target_meta.attrs[key] = value
                    
                    # Add station selection info as metadata
                    target_meta.attrs["original_station_key"] = best_station_key
                    target_meta.attrs["station_selection_score"] = best_score
                    
                    total_events += 1
                
                # Update metadata
                metadata_group.attrs["total_events"] = total_events
                metadata_group.attrs["total_stations"] = total_events  # One station per event
                metadata_group.attrs["data_format"] = "single_station"
                metadata_group.attrs["converted_from"] = "multi_station"
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f"\nConversion complete!")
                print(f"Created single-station dataset with {total_events} events")
                print(f"Best-station data saved to {output_file}")
                print(f"Processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        return output_file
    
    def run_full_preprocessing(self, standardize=True, filter_dataset=False, best_station=False,
                               target_length=14636, max_stations=3):
        """
        Run full preprocessing pipeline with memory-efficient operations
        
        Args:
            standardize: Whether to standardize waveform lengths
            filter_dataset: Whether to filter to limited stations per event
            best_station: Whether to create a single best-station dataset
            target_length: Target waveform length for standardization
            max_stations: Maximum stations per event for filtering
            
        Returns:
            Path to final processed file
        """
        start_time = time.time()
        print("\n" + "=" * 70)
        print("MEMORY-EFFICIENT AFTERSHOCK DATA PREPROCESSING".center(70))
        print("=" * 70)
        
        current_file = self.input_file
        
        # Step 1: Standardize waveforms
        if standardize:
            current_file = self.standardize_waveforms(target_length=target_length)
            self.output_file = current_file
        
        # Step 2: Filter to limited stations per event
        if filter_dataset:
            current_file = self.create_filtered_dataset(max_stations_per_event=max_stations)
            self.output_file = current_file
        
        # Step 3: Create best-station dataset
        if best_station:
            current_file = self.create_best_station_dataset()
            self.output_file = current_file
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE".center(70))
        print("=" * 70)
        print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Final dataset saved to: {current_file}")
        
        return current_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-efficient aftershock data preprocessing")
    parser.add_argument("--input", "-i", required=True, help="Input HDF5 file path")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--standardize", "-s", action="store_true", help="Standardize waveform lengths")
    parser.add_argument("--length", "-l", type=int, default=14636, help="Target waveform length")
    parser.add_argument("--filter", "-f", action="store_true", help="Filter to fewer stations per event")
    parser.add_argument("--max-stations", "-m", type=int, default=3, help="Maximum stations per event")
    parser.add_argument("--best-station", "-b", action="store_true", help="Create best-station dataset")
    
    args = parser.parse_args()
    
    processor = MemoryEfficientAfterShockProcessor(args.input)
    
    if args.output:
        output_file = args.output
        processor.run_full_preprocessing(
            standardize=args.standardize,
            filter_dataset=args.filter,
            best_station=args.best_station,
            target_length=args.length,
            max_stations=args.max_stations
        )
    else:
        processor.run_full_preprocessing(
            standardize=args.standardize,
            filter_dataset=args.filter,
            best_station=args.best_station,
            target_length=args.length,
            max_stations=args.max_stations
        )