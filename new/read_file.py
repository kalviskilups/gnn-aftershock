import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# A vectorized version of the haversine formula for computing distance in km.
def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two sets of points.
    Assumes inputs are array-like (or pandas Series) and returns an array of distances in km.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in km
    return c * r

def plot_aftershock_spatiotemporal_differences(df):
    """
    Create a plot of time differences (hours) versus spatial differences (km)
    between consecutive aftershocks.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing at least the following columns:
            - 'source_origin_time': event timestamp (string or datetime)
            - 'source_latitude_deg': latitude of the event (in degrees)
            - 'source_longitude_deg': longitude of the event (in degrees)
    """
    # Convert the origin time to datetime and sort the DataFrame by time
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['source_origin_time'])
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate the time difference (in hours) between consecutive events
    df_sorted['time_diff'] = df_sorted['timestamp'].diff().dt.total_seconds() / 3600

    # Calculate the spatial (horizontal) distance difference between consecutive events
    # Create shifted columns for the previous event's latitude and longitude
    df_sorted['prev_lat'] = df_sorted['source_latitude_deg'].shift(1)
    df_sorted['prev_lon'] = df_sorted['source_longitude_deg'].shift(1)
    df_sorted['spatial_diff'] = haversine_distance_vectorized(
        df_sorted['source_latitude_deg'],
        df_sorted['source_longitude_deg'],
        df_sorted['prev_lat'],
        df_sorted['prev_lon']
    )
    
    # Remove the first row (which has NaNs because there is no previous event)
    diff_df = df_sorted.dropna(subset=['time_diff', 'spatial_diff'])
    
    # Plot using a hexbin plot (the color shows the number of event pairs in that bin)
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(diff_df['time_diff'], diff_df['spatial_diff'], gridsize=50, cmap='viridis', mincnt=1)
    cb = plt.colorbar(hb)
    cb.set_label('Count')
    plt.xlabel('Time Difference (hours)')
    plt.ylabel('Spatial Difference (km)')
    plt.title('Spatiotemporal Differences between Consecutive Aftershocks')
    plt.grid(True)
    plt.savefig("test.png")

    # Alternatively, you can use seaborn's jointplot to have marginal histograms
    # sns.jointplot(x='time_diff', y='spatial_diff', data=diff_df, kind='hex', color='blue')
    # plt.xlabel('Time Difference (hours)')
    # plt.ylabel('Spatial Difference (km)')
    # plt.suptitle('Aftershock Spatiotemporal Differences', y=1.02)
    # plt.show()

# Example usage:
# Assuming you have loaded your aftershock data into a DataFrame named `df`
with open("aftershock_data.pkl", "rb") as file:
    data_dict = pickle.load(file)

# Extract the metadata from each event entry.
# Each value in data_dict is expected to have a "metadata" key.
data_list = [
    {**entry["metadata"], "waveform": entry["waveform"]}
    for entry in data_dict.values()
]
# Convert the list of metadata dictionaries into a DataFrame
df = pd.DataFrame(data_list)
plot_aftershock_spatiotemporal_differences(df)
