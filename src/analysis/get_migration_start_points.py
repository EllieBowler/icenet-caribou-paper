"""
Script to compute migration start points from telemetry data and Victoria Island (VI) buffer polygon.

This script processes telemetry data from caribou (or other migratory species) and identifies migration start points
based on the crossing of a specific geographic boundary (Victoria Island, in this case). The script analyzes the movement
of tagged animals and determines the moment when they cross from water to land or vice versa, which is considered the
start of migration.

It outputs both a CSV file containing the full migration data and a GeoJSON file for the specific migration start points.

The script supports plotting of the trajectories and crossing points if specified.

Outputs:
- A CSV file with full migration start data.
- A GeoJSON and CSV file with only the identified migration start points (crossings).
"""

import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import argparse
import os

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import sys
sys.path.append("../../")
from src.data.load import load_tracking_data
from src.data.process import get_land_points, get_traj_set
from src.plotting.data_plots import plot_trajectory, plot_crossing_point
from src import config


def make_mig_start_df(full_gdf: object, poly_gdf: object, save_plot: bool=False) -> object:
    """
    Identifies migration start points from telemetry data by analyzing crossing points between land and water.

    The function processes the trajectory data of the migratory species and identifies when they cross from water to land
    or vice versa, using a given geographic boundary (in this case, Victoria Island). If crossing occurs, the start point
    is recorded, and relevant metadata (crossing time, crossing distance, etc.) are stored.

    Args:
        full_gdf (object): A GeoDataFrame containing the telemetry data (animal movement points) with temporal and spatial
                            information about each observation.
        poly_gdf (object): A GeoDataFrame containing the polygon geometry for the Victoria Island boundary.
        save_plot (bool, optional): Whether or not to save plots of the trajectories and crossing points. Defaults to False.

    Returns:
        object: A GeoDataFrame containing the migration start points, including crossing label, geometry, time, and distance
                of crossing events.
    """
    # Assign land/water status to each point based on the provided Victoria Island polygon
    full_gdf = get_land_points(full_gdf, poly_gdf)

    results = []  # List to store results for each trajectory

    # Process each year in the telemetry dataset
    for year in full_gdf.index.year.unique():
        print(f"Processing year: {year}")
        
        # Get trajectory set for the current year and autumn migration season
        traj_set = get_traj_set(full_gdf, year, "autumn")

        for example_traj in traj_set:
            print(f"Processing trajectory: {example_traj.id}")
            track_gdf = example_traj.to_point_gdf()

            # Count the number of points on land and water (Victoria Island and Gulf)
            num_land = len(track_gdf[track_gdf.land == 1])
            num_gulf = len(track_gdf[track_gdf.land == 0])

            # Determine crossing status based on land and water points
            if num_gulf == 0:
                print("NO CROSSING: No points on water")
                cross_label = "land_only"
            elif num_land == 0:
                print("NO CROSSING: No points on Victoria Island")
                cross_label = "water_only"
            else:
                # Identify the crossing point where the trajectory moves from water to land
                last_land = track_gdf.index[track_gdf['land'] == 1][-1]
                crossing_points = track_gdf[last_land:][:2]

                if len(crossing_points) == 1:
                    print("NO CROSSING: Crossing from gulf to Victoria Island")
                    cross_label = "cross_to_land"
                else:
                    print("CROSSING EXAMPLE!")
                    cross_label = "cross"

                    # Create a trajectory object from the crossing points
                    crossing_traj = mpd.Trajectory(crossing_points, example_traj.id, crs="epsg:4326")
                    cross_time = crossing_traj.get_duration().total_seconds() / 3600  # Duration in hours
                    cross_dist = crossing_traj.get_length()  # Distance traveled during the crossing

                    # Clip the trajectory to the Victoria Island polygon and identify the crossing point
                    inter_traj = crossing_traj.clip(poly_gdf.geometry[0])
                    cross_point = inter_traj.to_point_gdf().iloc[[-1]]

                    # Create a row of results for the crossing point
                    row = [example_traj.id, year, "cross", cross_point.index[0], cross_point.geometry[0],
                           cross_time, cross_dist]

                    if save_plot:
                        # Plot the crossing trajectory and save it if requested
                        plot_crossing_point(example_traj, crossing_traj, cross_point, poly_gdf, year, save_plot=save_plot)

            # If no crossing occurred, add the data for land-only, water-only, or cross-to-land
            if cross_label in ["land_only", "water_only", "cross_to_land"]:
                row = [example_traj.id, year, cross_label, np.nan, np.nan, np.nan, np.nan]
                if save_plot:
                    plot_trajectory(example_traj, poly_gdf, year, save_plot=save_plot)

            # Append the result to the list
            results.append(row)

    # Convert the result list to a DataFrame for further processing
    result_gdf = pd.DataFrame(results, columns=["FieldID", "year", "crossing", "t", "geometry", "full_time", "full_dist"])
    
    # Assign bull tag information based on the collar IDs
    result_gdf["bull_tag"] = [True if collar_id in config.BULL_IDS else False for collar_id in result_gdf.FieldID]

    return result_gdf


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Compute migration start points based on telemetry data.")
    parser.add_argument("--save_plot", type=bool, default=False, help="Whether to save plot images of the trajectories.")
    args = parser.parse_args()

    # Load the Victoria Island polygon (boundary for migration analysis)
    with open(config.ADJUSTED_VI_POLY) as file:
        vi_gdf = gpd.read_file(file)

    # Extract the name of the polygon file without the extension
    poly_name = os.path.basename(config.ADJUSTED_VI_POLY).split(".")[0]

    # Load the collar (telemetry) dataset
    caribou_gdf = load_tracking_data(config.PATH_TO_CSV)

    # Get the DataFrame of migration start points (crossings)
    crossing_df = make_mig_start_df(caribou_gdf, vi_gdf, args.save_plot)

    # Define path to save processed data
    save_path = os.path.join(config.PROCESSED_DATA_FOLDER, "migration_start")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save full migration start points data as CSV
    crossing_df.to_csv(f"{save_path}/{poly_name}_full_mig_start.csv")

    # Filter and save only crossing points as GeoJSON and CSV
    clean_df = crossing_df[crossing_df.crossing == "cross"]
    clean_df.set_index("t", inplace=True)
    clean_gdf = gpd.GeoDataFrame(clean_df, crs=4326, geometry="geometry")
    clean_gdf.to_file(f"{save_path}/{poly_name}_mig_start.geojson", driver="GeoJSON")
    clean_gdf.to_csv(f"{save_path}/{poly_name}_mig_start.csv")






