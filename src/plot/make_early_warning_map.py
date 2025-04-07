"""
Script to generate prediction map outputs for coastal grid cells based on migration thresholds from different datasets.

This script processes sea ice concentration (SIC) data from various sources (OSISAF, AMSR, IceNet) to generate a prediction map
showing the number of days after initialization when migration thresholds are reached. It interpolates SIC values for coastal grid
cells and applies smoothing techniques to the data before calculating migration predictions. The script can also add live tracking
data for caribou migration if specified.

Key steps:
1. Load the appropriate SIC data based on the specified data source (OSISAF, AMSR, or IceNet).
2. Interpolate missing SIC values for coastal grid cells.
3. Smooth the interpolated data using a rolling average.
4. Apply migration thresholds and calculate when the threshold is crossed.
5. Generate and save a plot showing the prediction map for migration.

Command-line Arguments:
    --data_source (str): The data source to use for analysis. Options: 'osisaf', 'amsr', 'icenet'.
    --init_date (str): The initialization date for the prediction in the format 'YYYY-MM-DD'.
    --add_traj (bool, optional): Whether to add live caribou tracking points and segments to the plot (default: True).
    --mig_percent (int, optional): The percentage of migration to use as the threshold for prediction (default: 10).

Outputs:
    A PNG file containing the prediction map for coastal grid cells, showing the number of days after initialization 
    when the migration threshold is crossed. The file is saved to the location specified in the config under 
    PRED_PLOT_FOLDER. If `--add_traj` is set to True, the map will also display caribou locations.

Example usage:
    python make_early_warning_map.py --data_source icenet --init_date 2022-10-01 --mig_percent 10 --add_traj True

"""

import argparse
import pandas as pd
import xarray as xr
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os

import sys
sys.path.append("../../")
from src import config
from src.data.load import load_osisaf_year
from src.data.process import interpolate_sic
from src.analysis.metric_utils import get_matching_index_array


if __name__ == "__main__":
    # Define commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str)
    parser.add_argument("--init_date", type=str)
    parser.add_argument("--add_traj", default=True, type=bool)
    parser.add_argument("--mig_percent", default=10, type=int)
    args = parser.parse_args()

    # Convert the initial date from string to datetime format
    init_date = pd.to_datetime(args.init_date, format="%Y-%m-%d")
    print_init_date = init_date.strftime("%dth %B")

    # Load the appropriate data source based on user input
    if args.data_source == "osisaf":
        print("Processing OSISAF data...")
        # Load the mapping file for the given data source (OSISAF)
        mapping_df = pd.read_csv(config.PERCENT_MIG_FILES["osisaf"], index_col=0)
        _, _, n_consec, _, smoothing_level, _ = os.path.basename(config.PERCENT_MIG_FILES["osisaf"]).split("_")
        n_consec = int(n_consec)  # Number of consecutive days to match
        smoothing_level = int(smoothing_level)  # Smoothing level for rolling mean
        print(f"Consecutive days: {n_consec}, Smoothing level: {smoothing_level}")

        # Load and process the OSISAF data for the given year
        xarray_year = load_osisaf_year(f"{config.PATH_TO_OSISAF}/{init_date.year}.nc")
        xarray_gulf = xarray_year.sel(
            time=slice(init_date, f"{init_date.year}-12-31")).sel(
            xc=slice(*config.osisaf_plot_config['crop_x']),
            yc=slice(*config.osisaf_plot_config['crop_y'])
        )

        print("Interpolating missing values in coastal grid cells...")
        # Interpolate missing values for coastal grid cells
        xarray_interp = xarray_gulf.groupby("time").apply(interpolate_sic, dataset="osisaf")
        xarray_smooth = xarray_interp.rolling(time=smoothing_level, center=True, min_periods=1).mean()

        # Load the coast grid cell mask and crop to the region
        mask_xr = xr.open_dataset(f"{config.OSISAF_COAST_CELLS}").load()
        mask_gulf = mask_xr.ice_conc.sel(xc=slice(*config.osisaf_plot_config['crop_x']),
                                         yc=slice(*config.osisaf_plot_config['crop_y']))

        # Select the threshold corresponding to the specified migration percentage
        print(f"Selecting threshold for {args.mig_percent}% migration...")
        chosen_sic = mapping_df[mapping_df.percent_interp == args.mig_percent].sic_interp.values[0]

        # Identify the grid cells that match the ice concentration threshold
        matching_array = np.apply_along_axis(get_matching_index_array, 0,
                                             xarray_smooth.ice_conc.to_numpy(),
                                             chosen_sic, n_consec)
        matching_array[np.isnan(matching_array)] = -1.0  # Replace NaNs with -1 for plotting
        data = xarray_gulf.copy()
        data["matching_gridcells"] = (("yc", "xc"), matching_array)
        data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")

    # Repeat similar processing for AMSR and IceNet data sources
    elif args.data_source == "amsr":
        print("Processing AMSR data...")
        mapping_df = pd.read_csv(config.PERCENT_MIG_FILES["amsr"], index_col=0)
        _, _, n_consec, _, smoothing_level, _ = os.path.basename(config.PERCENT_MIG_FILES["amsr"]).split("_")
        n_consec = int(n_consec)
        smoothing_level = int(smoothing_level)
        print(f"Consecutive days: {n_consec}, Smoothing level: {smoothing_level}")

        # Load and process AMSR data
        xarray_year = xr.open_dataset(f"{config.PATH_TO_AMSR2}/{init_date.year}.nc")
        xarray_gulf = xarray_year.sel(
            time=slice(init_date, f"{init_date.year}-12-31")).sel(
            x=slice(*config.amsr_plot_config['crop_x']),
            y=slice(*config.amsr_plot_config['crop_y'])
        )

        print("Interpolating missing values in coastal grid cells...")
        xarray_interp = xarray_gulf.groupby("time").apply(interpolate_sic, dataset="amsr")
        xarray_interp = xarray_interp.drop_vars("polar_stereographic")
        xarray_smooth = xarray_interp.rolling(time=smoothing_level, center=True, min_periods=1).mean()

        # Load coast grid cell mask for AMSR data
        mask_xr = xr.open_dataset(f"{config.AMSR_COAST_CELLS}").load()
        mask_gulf = mask_xr.z.sel(x=slice(*config.amsr_plot_config['crop_x']),
                                  y=slice(*config.amsr_plot_config['crop_y']))

        print(f"Selecting threshold for {args.mig_percent}% migration...")
        chosen_sic = mapping_df[mapping_df.percent_interp == args.mig_percent].sic_interp.values[0]

        matching_array = np.apply_along_axis(get_matching_index_array, 0,
                                             xarray_smooth.z.to_numpy() / 100,
                                             chosen_sic, n_consec)
        matching_array[np.isnan(matching_array)] = -1.0
        data = xarray_gulf.copy()
        data["matching_gridcells"] = (("y", "x"), matching_array)
        data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")

    elif args.data_source == "icenet":
        print("Processing IceNet data...")
        mapping_df = pd.read_csv(config.PERCENT_MIG_FILES["osisaf"], index_col=0)
        _, _, n_consec, _, smoothing_level, _ = os.path.basename(config.PERCENT_MIG_FILES["osisaf"]).split("_")
        n_consec = int(n_consec)
        smoothing_level = int(smoothing_level)
        print(f"Consecutive days: {n_consec}, Smoothing level: {smoothing_level}")

        # Load and process IceNet data
        xarray_year = xr.open_dataset(f"{config.PATH_TO_ICENET}/icenet_coronation_gulf_{init_date.year}.nc")
        xarray_gulf = xarray_year.sel(
            time=init_date).sel(
            xc=slice(*config.icenet_plot_config['crop_x']),
            yc=slice(*config.icenet_plot_config['crop_y'])
        )

        # Interpolate missing values for IceNet data
        xarray_interp = xarray_gulf.groupby("leadtime").apply(interpolate_sic, dataset="icenet")
        xarray_smooth = xarray_interp.rolling(leadtime=smoothing_level, center=True, min_periods=1).mean()

        # Load coast grid cell mask for IceNet data
        mask_xr = xr.open_mfdataset(f"{config.OSISAF_COAST_CELLS}").load()
        mask_gulf = mask_xr.ice_conc.sel(xc=slice(*config.osisaf_plot_config['crop_x']),
                                         yc=slice(*config.osisaf_plot_config['crop_y']))

        print(f"Selecting threshold for {args.mig_percent}% migration...")
        chosen_sic = mapping_df[mapping_df.percent_interp == args.mig_percent].sic_interp.values[0]

        matching_array = np.apply_along_axis(get_matching_index_array, 0,
                                             xarray_smooth.sic_mean.to_numpy(),
                                             chosen_sic, n_consec)
        matching_array[np.isnan(matching_array)] = -1.0
        data = xarray_gulf.copy()
        data["matching_gridcells"] = (("yc", "xc"), matching_array)
        data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")

    # ############## Plotting ##############
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=config.PROJECTION)

    # Define the colormap and its properties
    cm = mpl.cm.get_cmap("Spectral").copy()
    cm.set_under("darkred")  # Set color for negative values (NaNs will be white)
    cm.set_over("darkblue")
    colors = [cm(0), cm(0.075), cm(0.15), cm(0.2), cm(0.3), cm(0.35),
              cm(0.4), cm(0.55), cm(0.65), cm(0.8), cm(0.9), cm(1.0)]

    max_val = np.nanmax(data["matching_gridcells"].where(mask_gulf != 0).values)
    levels = [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

    # Determine the correct projection for the data
    if args.data_source in ["osisaf", "icenet"]:
        plot_transform = config.osisaf_crs
    elif args.data_source == "amsr":
        plot_transform = config.amsr_crs

    # If all grid cells passed the threshold, handle the edge case where max value is -1
    if max_val == -1:
        data["matching_gridcells"].where(mask_gulf != 0).plot(transform=plot_transform,
                                                              colors=colors, levels=levels,
                                                              cbar_kwargs={"label": f"Number of days after initialisation",
                                                                           "ticks": levels,
                                                                           "spacing": "proportional"},
                                                              vmin=0, vmax=28)
    else:
        data["matching_gridcells"].where(mask_gulf != 0).plot(transform=plot_transform,
                                                              colors=colors, levels=levels,
                                                              cbar_kwargs={"label": f"Number of days after initialisation",
                                                                           "ticks": levels,
                                                                           "spacing": "proportional"},
                                                              vmin=0, vmax=28)

    save_name = ""
    if args.add_traj:
        print("Adding live caribou locations...")
        print("Loading interpolated GPS data from 'processed' data folder")
        traj_point_gdf = gpd.read_file(f"{config.PROCESSED_DATA_FOLDER}/interp_gps/"
                                       f"{init_date.year}/autumn_point_{config.INTERP_FREQ}.geojson")
        traj_point_gdf.set_index("time", inplace=True)

        traj_seg_gdf = gpd.read_file(f"{config.PROCESSED_DATA_FOLDER}/interp_gps/"
                                     f"{init_date.year}/autumn_segs_{config.INTERP_FREQ}.geojson")
        traj_seg_gdf.set_index("time", inplace=True)

        # Get points and line segments for the specified date
        day_point = traj_point_gdf.loc[args.init_date + " 00:00:00"]
        day_seg = traj_seg_gdf.loc[args.init_date + " 00:00:00"]

        # Plot the tracking points and segments
        day_point.to_crs(config.PROJECTION, inplace=True)
        points = day_point[day_point.past_end == 0]
        points.plot(ax=ax, markersize=12, color="black", zorder=2)

        crosses = day_point[day_point.past_end == 1]
        crosses.plot(ax=ax, markersize=12, color="black", marker="x", zorder=2)

        # Add segments
        day_seg.to_crs(config.PROJECTION, inplace=True)
        day_seg.plot(ax=ax, linewidth=0.5, alpha=0.5, color="black", zorder=2)

        save_name = "_with_traj"

    # Set map extent and plot coastlines
    ax.set_extent(config.PLOT_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines("10m", color="dimgrey")
    ax.set_title(f"Initialised {print_init_date}")
    ax.gridlines(draw_labels=False)
    plt.tight_layout()

    # Save the plot to a file
    print(f"Saving plot to {config.PRED_PLOT_FOLDER}/{args.data_source}_{args.mig_percent}percent_"
          f"{args.init_date}{save_name}.png")
    plt.savefig(f"{config.PRED_PLOT_FOLDER}/{args.data_source}_{args.mig_percent}percent_"
                f"{args.init_date}{save_name}.png",
                bbox_inches="tight", dpi=600)
    plt.close()
