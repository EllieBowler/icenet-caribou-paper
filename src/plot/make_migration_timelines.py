"""
Script to make predicted timelines for migrations and compare them to observations.

This script loads predicted migration start times from various sources (OSISAF, AMSR, IceNet), 
compares them to observed migration start times, and generates a series of plots. The predicted 
migration start times are visualized as box-and-whisker plots, and the observed start times are 
plotted as points. 

Command-line arguments:
- `--init_day`: Day of the initial date for analysis (int).
- `--init_month`: Month of the initial date for analysis (int).
- `--start_year`: Start year for analysis (int, default 2015).
- `--end_year`: End year for analysis (int, default 2023).
- `--day_diff_lim`: Limit for the day difference for filtering observations (int, default 3).
- `--female_only`: Filter observations by female-only migrations (bool, default True).
"""

import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import xarray as xr
import os

import sys
sys.path.append("../../")
from src import config
from src.data.load import load_migration_start_points_gdf
from src.plot.plot_utils import get_crossing_box

if __name__ == "__main__":
    # Define command-line arguments for the script execution
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_day", type=int, help="Day of the migration initialization date")
    parser.add_argument("--init_month", type=int, help="Month of the migration initialization date")
    parser.add_argument("--start_year", type=int, default=2015, help="Starting year for the analysis")
    parser.add_argument("--end_year", type=int, default=2023, help="Ending year for the analysis")
    parser.add_argument("--day_diff_lim", type=int, default=3, help="Limit on day difference for filtering observations")
    parser.add_argument("--female_only", type=bool, default=True, help="Use only female migration data")
    args = parser.parse_args()

    result_list = []  # List to store the results for all years
    year_list = np.arange(args.start_year, args.end_year, 1)  # List of years to analyze

    # Loop over each year to get predictions
    for test_year in year_list:
        print(f"Getting results for {test_year}")

        # Get predicted crossing dates for different datasets
        osi_pred = get_crossing_box(f"{test_year}-{args.init_month}-{args.init_day}", "osisaf")
        amsr_pred = get_crossing_box(f"{test_year}-{args.init_month}-{args.init_day}", "amsr")
        icenet_pred = get_crossing_box(f"{test_year}-{args.init_month}-{args.init_day}", "icenet")

        # Append predictions for the current year
        result_list.append(osi_pred)
        result_list.append(amsr_pred)
        result_list.append(icenet_pred)

    # Concatenate all predictions into a single DataFrame
    result_df = pd.concat(result_list)

    # Load observed migration start points
    crossing_gdf = load_migration_start_points_gdf(config.PATH_TO_MIG_START_DF,
                                                   female_only=args.female_only,
                                                   start_year=args.start_year,
                                                   day_diff_lim=args.day_diff_lim)
    
    # Add day of year (DOY) and data source columns to the observed migration data
    crossing_gdf["doy"] = crossing_gdf.index.dayofyear + crossing_gdf.index.hour / 24.0
    crossing_gdf["data_source"] = "observed"

    # Define color and line style dictionaries for plotting
    colours = {25: "green", 75: "red"}
    color_dict = {"observed": "dimgrey", "icenet": "olive", "osisaf": "steelblue", "amsr": "mediumpurple"}
    ls_dict = {"observed": "solid", "icenet": "solid", "osisaf": ":", "amsr": "--"}

    # Set figure size based on the number of years being analyzed
    fig_height = len(year_list) * 2.5
    fig, ax = plt.subplots(nrows=len(year_list), ncols=1, figsize=(12, fig_height), sharex=True)

    # Loop through each year to create plots
    for index, year in enumerate(year_list):
        # Filter results and observed data for the specific year
        year_df = result_df[result_df.date.dt.year == year]
        year_obs = crossing_gdf[crossing_gdf.index.year == year][["data_source", "doy"]]
        
        # Add new rows for missing data sources in observed data
        new_row = [{'data_source': "osisaf", 'doy': np.nan},
                   {'data_source': "amsr", 'doy': np.nan},
                   {'data_source': "icenet", 'doy': np.nan}]
        year_obs = pd.concat([year_obs, pd.DataFrame(new_row)], ignore_index=True)

        # Plot observed migration start times as points
        sns.swarmplot(ax=ax[index], x=year_obs["doy"]-1, y=year_obs["data_source"], s=3.5, c="grey",
                      label="observed migration\nstart dates")

        # Loop through each prediction source (AMSR, IceNet, OSISAF) and plot the results
        for row_number, (data_source) in enumerate(["amsr", "icenet", "osisaf"]):
            data_df = year_df[year_df.data_source == data_source]

            # Get the day of year values for box-and-whisker plots
            low_whisk = np.min(data_df[data_df.plot_label == "low_whisk"].date.dt.dayofyear-1)
            low_box = np.min(data_df[data_df.plot_label == "low_box"].date.dt.dayofyear-1)
            up_box = np.max(data_df[data_df.plot_label == "up_box"].date.dt.dayofyear-1)
            up_whisk = np.max(data_df[data_df.plot_label == "up_whisk"].date.dt.dayofyear-1)

            # Plot the whiskers for the current data source
            ax[index].hlines(row_number + 1, low_whisk, up_whisk, lw=2, color=color_dict[data_source],
                             label=f"{data_source}", zorder=0)
            ax[index].scatter([low_whisk, up_whisk], [row_number + 1, row_number + 1],
                              marker="|", c=color_dict[data_source], s=100)
            
            # Plot the box for the current data source
            ax[index].hlines(row_number + 1, low_box, up_box, lw=10, facecolor="none",
                             edgecolor=color_dict[data_source], zorder=1)

        # Adjust plot aesthetics for each year
        ax[index].set_ylim([-1, 4])  # Set y-axis limits
        ax[index].set_xlim([284, 365])  # Set x-axis limits for days of the year

        # Alternate background colors for each year
        if index % 2 == 0:
            ax[index].set_facecolor("whitesmoke")
        else:
            ax[index].set_facecolor("white")

        # Set date formatting and tick parameters for the x-axis
        ax[index].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax[index].xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax[index].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax[index].tick_params(which='minor', length=2)
        ax[index].tick_params(axis="x", labelsize=12)

        # Set labels and grid
        ax[index].set_ylabel(f"{year}", rotation=90, fontsize=14)
        ax[index].grid(alpha=0.5, ls="--")
        ax[index].get_legend().set_visible(False)

    plt.xlabel("")  # Remove x-axis label
    plt.subplots_adjust(hspace=.0)  # Adjust subplot spacing

    # Save the generated plot as a PNG image
    print(f"Saving plot as range_compare_init-{args.init_month}-{args.init_day}_{args.start_year}-{args.end_year}.png"
          f" in {config.PRED_PLOT_FOLDER}")
    plt.savefig(f"{config.PRED_PLOT_FOLDER}/"
                f"range_compare_init-{args.init_month}-{args.init_day}_{args.start_year}-{args.end_year}.png",
                bbox_inches="tight", dpi=600)
    plt.close()
