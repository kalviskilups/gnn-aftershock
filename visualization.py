import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import timedelta
import matplotlib.cm as cm
import matplotlib.dates as mdates
# Replace Basemap with Cartopy for mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def visualize_iquique_data(metadata, mainshock=None):
    """
    Create an exploratory visualization of the Iquique dataset
    
    Parameters:
    -----------
    metadata : pandas.DataFrame
        The metadata from the Iquique dataset
    mainshock : pandas.Series, optional
        Information about the mainshock event
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 15))
    
    # 1. Spatial distribution of events
    # Define map boundaries
    min_lat = metadata['source_latitude_deg'].min() - 0.5
    max_lat = metadata['source_latitude_deg'].max() + 0.5
    min_lon = metadata['source_longitude_deg'].min() - 0.5
    max_lon = metadata['source_longitude_deg'].max() + 0.5
    
    # Create subplot with Cartopy projection
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    
    # Set map extent
    ax1.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Add map features
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.LAND, facecolor='tan')
    ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Add gridlines
    gl = ax1.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Color by time if mainshock is provided
    if mainshock is not None:
        # Calculate days since mainshock
        metadata['days_since_mainshock'] = (metadata['datetime'] - mainshock['datetime']).dt.total_seconds() / (24 * 3600)
        
        # Create a colormap for time
        scatter = ax1.scatter(
            metadata['source_longitude_deg'].values,
            metadata['source_latitude_deg'].values,
            c=metadata['days_since_mainshock'],
            cmap='viridis', 
            s=30, 
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            transform=ccrs.PlateCarree()
        )
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Days since mainshock')
        
        # Highlight the mainshock
        ax1.scatter(
            mainshock['source_longitude_deg'],
            mainshock['source_latitude_deg'],
            s=200, 
            c='red', 
            marker='*', 
            edgecolor='black',
            linewidth=1.5,
            label='Mainshock',
            transform=ccrs.PlateCarree()
        )
        ax1.legend(loc='upper right')
    else:
        # Simple scatter without time information
        ax1.scatter(
            metadata['source_longitude_deg'].values,
            metadata['source_latitude_deg'].values,
            s=30, 
            c='blue', 
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            transform=ccrs.PlateCarree()
        )
    
    ax1.set_title('Spatial Distribution of Seismic Events')
    
    # 2. Depth distribution
    ax2 = fig.add_subplot(222)
    
    # Plot depth vs. time if mainshock is provided
    if mainshock is not None:
        scatter = ax2.scatter(
            metadata['days_since_mainshock'],
            metadata['source_depth_km'],
            c=metadata['days_since_mainshock'],
            cmap='viridis',
            s=30,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Days since mainshock')
        
        # Highlight the mainshock
        ax2.scatter(
            0,  # 0 days since mainshock
            mainshock['source_depth_km'],
            s=200,
            c='red',
            marker='*',
            edgecolor='black',
            linewidth=1.5,
            label='Mainshock'
        )
        ax2.legend(loc='upper right')
        
        ax2.set_xlabel('Days since mainshock')
    else:
        # Simple histogram of depths
        ax2.hist(
            metadata['source_depth_km'],
            bins=30,
            color='blue',
            alpha=0.7,
            edgecolor='black'
        )
        ax2.set_xlabel('Depth (km)')
    
    ax2.set_ylabel('Depth (km)')
    ax2.invert_yaxis()  # Invert y-axis to show depth correctly
    ax2.set_title('Depth Distribution of Seismic Events')
    
    # 3. Temporal distribution
    ax3 = fig.add_subplot(223)
    
    # Create time bins
    if mainshock is not None:
        # Focus on aftershocks
        aftershocks = metadata[metadata['datetime'] > mainshock['datetime']]
        
        # Create time bins (1-day intervals)
        min_time = aftershocks['datetime'].min()
        max_time = aftershocks['datetime'].max()
        
        # Create daily bins
        time_bins = pd.date_range(
            start=min_time.floor('D'),
            end=max_time.ceil('D'),
            freq='D'
        )
        
        # Count events per day
        counts, bins = np.histogram(
            aftershocks['datetime'],
            bins=time_bins
        )
        
        # Convert bins to matplotlib dates
        bin_centers = time_bins[:-1] + pd.Timedelta(hours=12)
        
        # Plot aftershock frequency
        ax3.bar(
            bin_centers,
            counts,
            width=0.8,
            color='blue',
            alpha=0.7,
            edgecolor='black'
        )
        
        # Add reference line for mainshock
        ax3.axvline(
            mainshock['datetime'],
            color='red',
            linestyle='--',
            linewidth=2,
            label='Mainshock'
        )
        
        # Set date format
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        ax3.legend()
        ax3.set_title('Daily Frequency of Aftershocks')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Events')
    else:
        # Simple histogram of event times
        ax3.hist(
            metadata['datetime'],
            bins=30,
            color='blue',
            alpha=0.7,
            edgecolor='black'
        )
        
        ax3.set_title('Temporal Distribution of Seismic Events')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Events')
    
    # 4. Event distribution with distance from mainshock
    ax4 = fig.add_subplot(224)
    
    if mainshock is not None:
        # Calculate distance from mainshock (approximate using Haversine)
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in kilometers
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c
            
            return distance
        
        # Calculate distance for each event
        metadata['distance_from_mainshock_km'] = metadata.apply(
            lambda row: haversine_distance(
                mainshock['source_latitude_deg'], 
                mainshock['source_longitude_deg'],
                row['source_latitude_deg'], 
                row['source_longitude_deg']
            ), 
            axis=1
        )
        
        # Focus on aftershocks
        aftershocks = metadata[metadata['datetime'] > mainshock['datetime']]
        
        # Create scatter plot of distance vs. time
        scatter = ax4.scatter(
            aftershocks['days_since_mainshock'],
            aftershocks['distance_from_mainshock_km'],
            c=aftershocks['days_since_mainshock'],
            cmap='viridis',
            s=30,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Days since mainshock')
        
        ax4.set_xlabel('Days since mainshock')
        ax4.set_ylabel('Distance from mainshock (km)')
        ax4.set_title('Spatiotemporal Distribution of Aftershocks')
    else:
        ax4.text(
            0.5, 0.5,
            "Mainshock information required\nfor this visualization",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax4.transAxes,
            fontsize=12
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/iquique_data_visualization.png', dpi=300)
    plt.close()
    
    print("Visualization saved to results/iquique_data_visualization.png")

def visualize_omori_law(aftershocks, mainshock_time, time_bins=24):
    """
    Simple visualization of Omori's Law showing number of earthquakes vs. time
    """
    # Calculate hours since mainshock
    aftershocks['hours_since_mainshock'] = (aftershocks['datetime'] - mainshock_time).dt.total_seconds() / 3600
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create time bins (first for 24 hours, then daily for a month)
    hourly_bins = np.arange(0, 25, 1)  # Hourly bins for first day
    daily_bins = np.arange(24, 24*30+1, 24)  # Daily bins for the rest
    time_edges = np.concatenate([hourly_bins, daily_bins])
    
    # Count events in each bin
    counts, bin_edges = np.histogram(aftershocks['hours_since_mainshock'], bins=time_edges)
    
    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate bin widths (hourly for first day, then daily)
    bin_widths = np.diff(bin_edges)
    
    # Plot the count per time unit (hourly for first day, daily after)
    plt.plot(bin_centers, counts / bin_widths, 'bo-', linewidth=2, markersize=8)
    
    # Add labels and title
    plt.xlabel('Time since mainshock (hours)')
    plt.ylabel('Number of earthquakes per hour')
    plt.title('Omori\'s Law: Aftershock Frequency Over Time')
    
    # Add grid
    plt.grid(True, alpha=0.5)
    
    # Start y-axis from 0
    plt.ylim(bottom=0)
    
    # Save figure
    plt.savefig('results/omori_law_visualization.png', dpi=300)
    plt.close()
    
    print("Omori's Law visualization saved to results/omori_law_visualization.png")
    
    # Optional: Create a log-log plot as well, which often shows the power law more clearly
    plt.figure(figsize=(12, 8))
    
    # Create logarithmic time bins
    min_time = max(0.5, aftershocks['hours_since_mainshock'].min())  # Start from 0.5 hours
    max_time = aftershocks['hours_since_mainshock'].max()
    log_time_edges = np.logspace(np.log10(min_time), np.log10(max_time), time_bins)
    
    # Count events in logarithmic bins
    log_counts, _ = np.histogram(aftershocks['hours_since_mainshock'], bins=log_time_edges)
    
    # Calculate bin centers and widths
    log_bin_centers = np.sqrt(log_time_edges[:-1] * log_time_edges[1:])
    log_bin_widths = np.diff(log_time_edges)
    
    # Calculate rate (events per hour)
    log_rate = log_counts / log_bin_widths
    
    # Plot with log-log scales
    plt.loglog(log_bin_centers, log_rate, 'ro-', linewidth=2, markersize=8)
    
    # Add labels and title
    plt.xlabel('Time since mainshock (hours) - log scale')
    plt.ylabel('Number of earthquakes per hour - log scale')
    plt.title('Omori\'s Law: Log-Log Plot of Aftershock Frequency')
    
    # Add grid for both major and minor ticks
    plt.grid(True, which='both', alpha=0.5)
    
    # Save log-log figure
    plt.savefig('results/omori_law_loglog_visualization.png', dpi=300)
    plt.close()
    
    print("Log-log Omori's Law visualization saved to results/omori_law_loglog_visualization.png")

def visualize_graph_structure(graph_dataset, num_samples=3):
    """Modified function to better visualize graph structures"""
    # Select sample graphs
    if len(graph_dataset) <= num_samples:
        samples = graph_dataset
    else:
        # Choose random samples for diversity
        indices = np.random.choice(len(graph_dataset), num_samples, replace=False)
        samples = [graph_dataset[i] for i in indices]
    
    # Create figure
    fig, axes = plt.subplots(1, len(samples), figsize=(6*len(samples), 6))
    
    # If only one sample, axes is not a list
    if len(samples) == 1:
        axes = [axes]
    
    # Plot each sample
    for i, (graph, ax) in enumerate(zip(samples, axes)):
        # Extract node features
        node_pos = graph.x[:, :2].numpy()  # Latitude and longitude
        
        # Extract edge information
        edge_index = graph.edge_index.numpy()
        
        # Check if we have a valid graph
        if node_pos.shape[0] == 0:
            ax.text(0.5, 0.5, "Empty graph", ha='center', va='center')
            continue
            
        # Check if all nodes are at the same position (would result in a single dot)
        if np.all(node_pos == node_pos[0]):
            ax.text(0.5, 0.5, "All nodes at same position", ha='center', va='center')
            continue
        
        # Plot nodes
        scatter = ax.scatter(
            node_pos[:, 1],  # Longitude
            node_pos[:, 0],  # Latitude
            c=np.arange(len(node_pos)),  # Color by event order
            cmap='viridis',
            s=100,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Event order')
        
        # Plot edges
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[0, j], edge_index[1, j]
            # Only draw edges once (when src < dst)
            if src < dst:
                ax.plot(
                    [node_pos[src, 1], node_pos[dst, 1]],
                    [node_pos[src, 0], node_pos[dst, 0]],
                    'k-',
                    alpha=0.3,
                    linewidth=1
                )
        
        # Check if we have a target
        if hasattr(graph, 'y') and graph.y is not None:
            target = graph.y.numpy()
            if len(target.shape) > 1 and target.shape[1] == 2:
                # Plot target location
                ax.scatter(
                    target[0, 1],  # Longitude
                    target[0, 0],  # Latitude
                    c='red',
                    s=150,
                    marker='*',
                    edgecolor='black',
                    linewidth=1.5,
                    label='Target location'
                )
                ax.legend()
        
        # Add labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Graph Sample {i+1}')
        ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/graph_structure_visualization.png', dpi=300)
    plt.close()