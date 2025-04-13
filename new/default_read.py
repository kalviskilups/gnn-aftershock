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


def read_data_from_pickle(file_path):
    # Load the pickle file that contains the data dictionary
    with open(file_path, "rb") as file:
        data_dict = pickle.load(file)

    # Extract the metadata from each event entry.
    # Each value in data_dict is expected to have a "metadata" key.
    data_list = [
        {**entry["metadata"], "waveform": entry["waveform"]}
        for entry in data_dict.values()
    ]
    # Convert the list of metadata dictionaries into a DataFrame
    df = pd.DataFrame(data_list)

    return df


if __name__ == "__main__":
    # Load the Iquique dataset and preprocess
    data = read_data_from_pickle("aftershock_data.pkl")
