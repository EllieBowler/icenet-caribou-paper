"""
save_interp_gps.py

This script processes and interpolates GPS data for caribou trajectories, 
and saves the interpolated results. It reads caribou GPS data from a CSV file, 
performs interpolation for different years and seasons, and saves the interpolated GPS data.

Modules Required:
- process: Contains functions for interpolating GPS data and retrieving trajectory sets.
- load: Contains a function for reading CSV files and converting them into GeoDataFrames.
- src/config: Contains configuration settings like the path to the CSV file and season list.

Usage:
------
Run this script as a standalone program. It will:
1. Load caribou GPS data from a CSV file.
2. For each unique year in the dataset, process the data for the "spring" and "autumn" seasons.
3. Interpolate GPS data for the selected year and season, and save the results.

Parameters:
-----------
config.PATH_TO_CSV (str): Path to the CSV file containing caribou GPS data.
config.SEASON_LIST (list of str): A list containing the seasons to process (default includes "spring" and "autumn").

Main Steps:
-----------
1. The caribou GPS data is loaded using `read_csv_as_datetime_geometry`.
2. A loop is run over each year and season (spring, autumn), and the trajectories are retrieved using `get_traj_set`.
3. If there are no GPS points to interpolate for a given season, a message is printed and the script moves to the next iteration.
4. For valid trajectories, the GPS data is interpolated using `interpolate_gps_data`, and the results are saved.

Example:
--------
To run the script from the command line:
    python save_interp_gps.py
"""

from process import interpolate_gps_data, get_traj_set
from load import read_csv_as_datetime_geometry
import sys
sys.path.append("../../")
from src import config


if __name__ == "__main__":
    # Load the caribou GPS data from the provided CSV file
    caribou_gdf = read_csv_as_datetime_geometry(config.PATH_TO_CSV)
    
    # Iterate through each unique year in the data
    for year in caribou_gdf.index.year.unique():
        # For each year, iterate through the desired seasons (spring and autumn)
        for season in ["spring", "autumn"]:  # or use config.SEASON_LIST for flexibility
            # Retrieve the trajectory collection for the current year and season
            traj_collection = get_traj_set(caribou_gdf, year, season, plot=False)
            
            # If no trajectories are found, print a message and continue
            if len(traj_collection) == 0:
                print(f"No GPS points to interpolate for {year} {season}")
                pass
            else:
                # Interpolate GPS data for the found trajectories and save the result
                _, _ = interpolate_gps_data(traj_collection, year, season, save=True)