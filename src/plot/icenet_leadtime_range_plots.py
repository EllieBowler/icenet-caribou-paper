"""
Script for generating lead-time analysis plots based on IceNet data and migration start points.

Key steps include:
1. Load and process IceNet SIC data for each year.
2. Interpolate and smooth the data.
3. Calculate the first observed migration crossing dates.
4. Generate lead-time plots for each year, showing the variation in predictions at different lead times.

### Command-line arguments:
- `--data_source`: The source of the data, default is "icenet".
- `--start_year`: The starting year for the analysis, default is 2020.
- `--end_year`: The ending year for the analysis, default is 2023.
- `--max_lead`: The maximum lead time (in days) to be considered for analysis, default is -40.

### Example usage:
To run the script and generate lead-time plots from 2020 to 2023:

```bash
python icenet_leadtime_range_plots.py --start_year 2020 --end_year 2023 --max_lead -40
```
"""

import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import argparse
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import os

import sys
sys.path.append("../../")
from src import config
from src.data.load import load_migration_start_points_gdf
from src.analysis.metric_utils import get_matching_index_array
from src.data.process import interpolate_sic


def day_of_year_to_date(day_num_list, year):
    """
    Convert a list of day numbers (1-based) to their corresponding date in the format 'MMM-DD'.
    
    Parameters:
    - day_num_list: List of day numbers (1-based)
    - year: The year to convert to dates
    
    Returns:
    - A list of formatted dates as strings in 'MMM-DD' format
    """
    res = [dt.datetime.strptime(str(year) + "-" + str(day_num), "%Y-%j").strftime("%b-%d") for day_num in day_num_list]
    return res


if __name__ == "__main__":
    # Define commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, default="icenet")
    parser.add_argument("--start_year", type=int, default=2020)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--max_lead", type=int, default=-40)
    args = parser.parse_args()

    # Define the percentiles for the box and whisker plots
    box_and_whisker = {"low_whisk": 10, "low_box": 25, "up_box": 75, "up_whisk": 90}

    # Load the migration start points dataframe
    crossing_gdf = load_migration_start_points_gdf(
        config.PATH_TO_MIG_START_DF, female_only=True, start_year=1990, day_diff_lim=3
    )
    
    # Add day of year (DOY) as a new column
    crossing_gdf["doy"] = crossing_gdf.index.dayofyear + crossing_gdf.index.hour / 24.0

    full_results = []  # List to store results for all years
    
    # Loop over the years for analysis
    for test_year in np.arange(args.start_year, args.end_year + 1):
        year_gdf = crossing_gdf[crossing_gdf.year == test_year]
        earliest_crossing = year_gdf.index.min().strftime('%Y-%m-%d')
        plot_start_date = f"{test_year}-07-01"
        
        print(f"Running {test_year} leadtime analysis")
        
        # Load and subset the IceNet dataset for the given year
        xarray_year = xr.open_dataset(f"{config.PATH_TO_ICENET}/icenet_coronation_gulf_{test_year}.nc")
        xarray_year = xarray_year.sel(time=slice(plot_start_date, earliest_crossing))

        # Loop through initialization dates
        for init_date in xarray_year.time.values:
            print(f"Processing initialization date: {init_date}")
            init_date = pd.to_datetime(init_date, format="%Y-%m-%d %H:%M")

            # Subset the data for the specific initialization date
            xarray_gulf = xarray_year.sel(
                time=init_date).sel(
                xc=slice(*config.icenet_plot_config["crop_x"]),
                yc=slice(*config.icenet_plot_config["crop_y"])
            )
            
            # Interpolate and smooth the data
            xarray_interp = xarray_gulf.groupby("leadtime").apply(interpolate_sic, dataset="icenet")
            xarray_smooth = xarray_interp.rolling(leadtime=smoothing_level, center=True, min_periods=1).mean()

            # Load the coast mask to filter out land areas
            coast_mask = xr.open_dataset(f"{config.OSISAF_CROSSING_CELLS}")
            mask = coast_mask.sel(time=f"{init_date.year}-01-01")
            mask_gulf = mask.sel(xc=slice(*config.osisaf_plot_config["crop_x"]),
                                 yc=slice(*config.osisaf_plot_config["crop_y"]))

            # Loop through the percentiles for box-and-whisker plots
            for key, chosen_percent in box_and_whisker.items():
                chosen_sic = mapping_df[mapping_df.percent_interp == chosen_percent].sic_interp.values[0]

                # Get the matching grid cells based on ice concentration threshold
                matching_array = np.apply_along_axis(
                    get_matching_index_array, 0, xarray_smooth.sic_mean.to_numpy(),
                    chosen_sic, n_consec
                )

                # Add the matching grid cells to the data
                data = xarray_gulf.copy()
                data["matching_gridcells"] = (("yc", "xc"), matching_array)
                data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")

                # Extract the result for valid grid cells (not masked)
                result = data_expanded["matching_gridcells"].where(mask_gulf != 0).ice_conc
                all_days = result.values[~np.isnan(result.values)]
                
                # If no valid days are found, create a default result
                if len(all_days) == 0:
                    result_df = pd.DataFrame(data={"pred_day_num": [0]}, dtype=np.int8)
                    result_df["init_date"] = init_date
                    result_df["date"] = init_date
                    result_df["year"] = init_date.year
                    result_df["label"] = "predicted"
                    result_df["plot_label"] = key
                    result_df["percent"] = chosen_percent
                    result_df["sic_thresh"] = chosen_sic
                    result_df["data_source"] = args.data_source
                    result_df["days_before_first_cross"] = init_date.day_of_year - year_gdf.index.min().day_of_year

                    full_results.append(result_df)
                else:
                    result_df = pd.DataFrame(data={"pred_day_num": all_days}, dtype=np.int8)
                    result_df["init_date"] = init_date
                    result_df["date"] = result_df.apply(
                        lambda x: x.init_date + np.timedelta64(x.pred_day_num, "D"), axis=1
                    )
                    result_df["year"] = init_date.year
                    result_df["label"] = "predicted"
                    result_df["plot_label"] = key
                    result_df["percent"] = chosen_percent
                    result_df["sic_thresh"] = chosen_sic
                    result_df["data_source"] = args.data_source
                    result_df["days_before_first_cross"] = init_date.day_of_year - year_gdf.index.min().day_of_year

                    full_results.append(result_df)

    # Concatenate all results into a single dataframe
    crossing_pred = pd.concat(full_results)
    crossing_pred["init_doy"] = crossing_pred.init_date.dt.day_of_year
    crossing_pred["pred_doy"] = crossing_pred.date.dt.day_of_year

    # Set up plotting for each year
    year_list = np.arange(args.start_year, args.end_year + 1)
    fig_height = len(year_list) * 2.5
    fig, axes = plt.subplots(nrows=len(year_list), ncols=1, figsize=(10, fig_height), sharex=True)

    # Loop through years and plot the results
    for index, PLOT_YEAR in enumerate(year_list):
        year_gdf = crossing_gdf[crossing_gdf.year == PLOT_YEAR]
        crossing_pred_year = crossing_pred[
            (crossing_pred.year == PLOT_YEAR) & (crossing_pred.days_before_first_cross >= args.max_lead)
        ]

        for init_date, data_df in crossing_pred_year.groupby("days_before_first_cross"):
            # Plot the whiskers and box plots
            low_whisk = np.min(data_df[data_df.plot_label == "low_whisk"].date)
            low_box = np.min(data_df[data_df.plot_label == "low_box"].date)
            up_box = np.max(data_df[data_df.plot_label == "up_box"].date)
            up_whisk = np.max(data_df[data_df.plot_label == "up_whisk"].date)

            axes[index].vlines(init_date, low_whisk, up_whisk, lw=2, color="grey")
            axes[index].scatter([init_date, init_date], [low_whisk, up_whisk], marker="_", s=100, color="grey")
            axes[index].vlines(init_date, low_box, up_box, lw=10, facecolor="none", color="silver")

        axes[index].set(ylim=[pd.to_datetime(f"{PLOT_YEAR}-10-14"), pd.to_datetime(f"{PLOT_YEAR}-12-20")])

        axes[index].hlines(year_gdf.index.min().date(), xmin=crossing_pred_year.days_before_first_cross.min() - 1,
                           xmax=1, color="black", ls="--")
        axes[index].hlines(year_gdf.index.max().date(), xmin=crossing_pred_year.days_before_first_cross.min() - 1,
                           xmax=1, color="black", ls="--")

        axes[index].yaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
        axes[index].yaxis.set_major_locator(mdates.DayLocator(interval=7))
        axes[index].yaxis.set_minor_locator(mdates.DayLocator(interval=1))

        axes[index].grid("on", which="major")
        axes[index].grid("on", which="minor", linestyle="--", alpha=0.5)

        axes[index].set_ylabel(f"{PLOT_YEAR}", fontsize=12)
        axes[index].set_xlim([args.max_lead, 1])

    axes[-1].set_xlabel("Days before first observed crossing", fontsize=12)

    # Save the plot
    print(f"Saving as {config.PRED_PLOT_FOLDER}/icenet_leadtime_analysis.png")
    plt.tight_layout()
    plt.savefig(f"{config.PRED_PLOT_FOLDER}/icenet_leadtime_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
