import numpy as np
from scipy.stats import sem
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr

import sys
sys.path.append("../../")
from src import config
from src.data.load import load_train_test_csv, load_osisaf_year, load_icenet_year


def rolling_window(arr, window_length):
    """
    Generate a sliding window view of the input array `arr` with a specified window length.

    This function returns a new view of the original array `arr`, where each element in the new array
    is a sliding window of length `window_length` over the original array along the last axis.

    Args:
        arr (np.ndarray): Input array for which the rolling window will be applied.
        window_length (int): Length of the rolling window.

    Returns:
        np.ndarray: A view of the original array with a rolling window applied along the last axis. 
                    The shape of the returned array will be (original shape with the last dimension 
                    reduced by the window length + 1, window_length).
    """
    # Calculate the shape of the new array, adjusting the last dimension for the window length
    shape = arr.shape[:-1] + (arr.shape[-1] - window_length + 1, window_length)
    
    # Create strides for the new array view: adding an extra stride for the rolling window
    strides = arr.strides + (arr.strides[-1],)
    
    # Use np.lib.stride_tricks.as_strided to create a view of the original array with the desired window
    arr_window = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    
    return arr_window


def get_matching_index_array(sic_array, sic_thresh, n_consec):
    """
    Find the first index in a sea ice concentration (SIC) time series where the concentration exceeds a specified threshold 
    for a consecutive number of time steps.

    This function searches for the first occurrence in the SIC time series where the value exceeds the given threshold 
    (`sic_thresh`) for a specified number of consecutive time steps (`n_consec`), and returns the first index of that sequence.

    Args:
        sic_array (np.ndarray): A 1D array representing the sea ice concentration (SIC) values over time.
        sic_thresh (float): The SIC threshold to be exceeded.
        n_consec (int): The number of consecutive time steps the SIC must exceed the threshold.

    Returns:
        int or np.nan: The index of the first occurrence where the SIC exceeds the threshold for `n_consec` consecutive 
                        time steps, or `np.nan` if no such sequence is found.
    """
    # Define the search pattern: first a 0 (below threshold), followed by `n_consec` ones (above threshold)
    search_arr = np.array([0] + [1] * n_consec)

    # Apply SIC threshold to the time series, converting it to a binary array (0 = below threshold, 1 = above threshold)
    arr_thresh = (sic_array > sic_thresh).astype(int)

    # Generate rolling windows over the thresholded array
    arr_window = rolling_window(arr_thresh, len(search_arr))

    # Find the indices where the rolling window matches the search pattern
    matching_indices = np.flatnonzero((arr_window == search_arr).all(axis=1))

    # If no matching indices were found, return NaN
    if matching_indices.size == 0:
        matching_indices = np.array([np.nan])

    # Return the first matching index (or NaN if no match was found)
    return matching_indices[0]


def get_matching_index(sic_timeseries, sic_thresh, n_consec, smoothing):
    """
    Find the first index in a sea ice concentration (SIC) time series where the concentration exceeds a specified threshold
    for a consecutive number of time steps, after applying smoothing to the time series.

    This function applies a rolling mean (smoothing) to the SIC time series and then searches for the first occurrence
    where the SIC exceeds the specified threshold (`sic_thresh`) for `n_consec` consecutive time steps.

    Args:
        sic_timeseries (pd.Series or np.ndarray): A time series of sea ice concentration values (SIC).
        sic_thresh (float): The SIC threshold to be exceeded.
        n_consec (int): The number of consecutive time steps the SIC must exceed the threshold.
        smoothing (int): The window size for applying a rolling mean (smoothing) to the SIC time series.

    Returns:
        int or np.nan: The index of the first occurrence where the SIC exceeds the threshold for `n_consec` consecutive 
                        time steps, after smoothing. If no such sequence is found, returns `np.nan`.
    """
    # Define the search pattern: first a 0 (below threshold), followed by `n_consec` ones (above threshold)
    search_arr = np.array([0] + [1] * n_consec)

    # Apply rolling mean (smoothing) to the SIC time series
    sic_timeseries = sic_timeseries.rolling(window=smoothing, center=True, min_periods=1).mean()

    # Apply SIC threshold to the smoothed time series, converting it to a binary array (0 = below threshold, 1 = above threshold)
    arr_thresh = (sic_timeseries > sic_thresh).values

    # Generate rolling windows over the thresholded array
    arr_window = rolling_window(arr_thresh, len(search_arr))

    # Find the indices where the rolling window matches the search pattern
    matching_indices = np.flatnonzero((arr_window == search_arr).all(axis=1))

    # If no matching indices were found, return NaN
    if matching_indices.size == 0:
        matching_indices = np.array([np.nan])

    # Return the first matching index (or NaN if no match was found)
    return matching_indices[0]


def get_percent_migrate_df(data_source, n_consec, smooth_level,
                           female_only=True, train_end=2019, day_diff_lim=3, plotting=False):
    """
    Calculate the percentage of migration events that are predicted based on sea ice concentration (SIC) thresholding.

    This function processes observed migration data and compares it to predicted migration dates determined by
    SIC threshold values. It calculates the percentage of observations where the predicted migration date occurs
    before the observed migration date, based on SIC data and a specified number of consecutive time steps.

    Args:
        data_source (str): The source of the SIC data (e.g., 'osisaf', 'amsr').
        n_consec (int): The number of consecutive time steps.
        smooth_level (int): The window size for applying a rolling mean (smoothing) to the SIC time series.
        female_only (bool, optional): If True, only female data is used. Defaults to True.
        train_end (int, optional): The year up to which the training data is considered. Defaults to 2019.
        day_diff_lim (int, optional): The maximum difference in days for a migration-start point.
        plotting (bool, optional): If True, plots will be generated (if applicable). Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the SIC threshold values and the corresponding percentage of migration events.
    """
    
    # Load training data
    train_df, _ = load_train_test_csv(data_source, female_only, train_end, day_diff_lim)
    
    # Get the column name for SIC data based on the data source
    sic_col_name = config.SIC_COL_REF[data_source]

    print(f"Getting percent migrated mapping for {data_source}...")

    # List to store results
    percent_migrate = []

    # Loop through different SIC threshold values from 0.5 to 1.0, in steps of 0.01
    for sic_thresh in np.round(np.arange(0.5, 1.01, 0.01), 2):
        result_df = []

        # Group data by observation ID
        for obs_id, obs_df in train_df.groupby("obs_id"):
            # Get the observed migration date
            cross_date = obs_df.mig_date.unique()
            if len(cross_date) == 1:
                cross_date = cross_date[0]
            else:
                print(f"Warning: Multiple migration dates found for observation {obs_id}, expected only one.")
                
            # Extract SIC time series for the observation
            sic_timeseries = obs_df[sic_col_name]
            
            # Get the predicted index based on SIC threshold and consecutive time steps
            pred_idx = get_matching_index(sic_timeseries, sic_thresh, n_consec, smooth_level)

            if not np.isnan(pred_idx):
                # If a valid prediction index is found, get the predicted migration date
                pred_date = obs_df.index[pred_idx]
                # Append the observed and predicted dates to the result dataframe
                result_df.append({"observed": cross_date, "predicted": pred_date})

        result_df = pd.DataFrame(result_df)

        # If no results are found for this SIC threshold, consider all observations as migrated
        if result_df.empty:
            percent_migrate.append({"sic_thresh": sic_thresh, "percent_mig": 1})
        else:
            # Calculate the percentage of correctly predicted migrations
            total_number = len(result_df)
            number_migrated = (result_df["observed"].dt.date < result_df["predicted"].dt.date).sum()
            percent_migrate.append({"sic_thresh": sic_thresh, "percent_mig": number_migrated / total_number})

    # Return a DataFrame with the SIC thresholds and the corresponding migration percentages
    return pd.DataFrame(percent_migrate)


def calculate_metric(result_df, pred_col_name="predicted", metric_name="made", alpha=1):
    """
    Calculate various metrics (Mean Absolute Deviation, Mean Squared Deviation, or Percent Migrated) 
    between observed and predicted migration dates.

    Args:
        result_df (pd.DataFrame): A DataFrame containing observed and predicted migration dates.
        pred_col_name (str, optional): The column name for predicted migration dates. Defaults to "predicted".
        metric_name (str, optional): The metric to calculate. Options are "made", "msde", and "percent_migrate". Defaults to "made".
        alpha (float, optional): A parameter that can be used for scaling in some metrics (currently unused). Defaults to 1.

    Returns:
        float: The calculated metric value.
        float (optional): The standard error of the mean (only for the "made" metric).
    """
    # Check if the metric is 'made' (Mean Absolute Deviation)
    if metric_name == "made":
        # Calculate the difference between observed and predicted dates
        days_diff = (result_df[pred_col_name].dt.date - result_df["observed"].dt.date).dt.days.values
        # Calculate Mean Absolute Deviation (MAD)
        made = np.sum(np.abs(days_diff)) / len(days_diff)
        # Calculate Standard Error of the Mean (SEM) for MAD
        made_sem = np.std(np.abs(days_diff), ddof=1) / np.sqrt(np.size(days_diff))
        return made, made_sem
    
    # Check if the metric is 'msde' (Mean Squared Deviation)
    if metric_name == "msde":
        # Calculate the squared difference between observed and predicted dates
        days_diff = (result_df[pred_col_name].dt.date - result_df["observed"].dt.date).dt.days.values
        # Calculate Mean Squared Deviation (MSD)
        msde = np.sum(np.square(days_diff)) / len(days_diff)
        return msde
    
    # Check if the metric is 'percent_migrate' (percentage of correctly predicted migration events)
    elif metric_name == "percent_migrate":
        # Calculate the total number of observations
        total_number = len(result_df)
        # Calculate the number of migrations correctly predicted (observed date < predicted date)
        number_migrated = (result_df["observed"].dt.date < result_df[pred_col_name].dt.date).sum()
        # Return the percentage of correctly predicted migrations
        return number_migrated / total_number
    
    else:
        # If an invalid metric_name is provided, print an error message
        print(f"Error - metric options are one of ['made', 'msde', 'percent_migrate']. Got input {metric_name}")
        return
    

def get_crossing_interval(init_date, data_source, percent_one, percent_two, smooth, n_consec):
    """
    Function to calculate predicted sea ice crossing intervals for specific thresholds (SIC) 
    based on sea ice concentration data from different data sources.
    
    Parameters:
    init_date (str): The initial date (in "YYYY-MM-DD" format) to start the prediction.
    data_source (str): The data source to use for the sea ice concentration ("amsr", "osisaf", "icenet_1w", "icenet_2w", "icenet_3w").
    percent_one (float): First threshold for sea ice concentration (SIC) as a percentage.
    percent_two (float): Second threshold for sea ice concentration (SIC) as a percentage.
    smooth (int): The rolling window size for smoothing the sea ice concentration data.
    n_consec (int): The number of consecutive days that the SIC must exceed the threshold to consider a crossing.

    Returns:
    pd.DataFrame: A DataFrame containing the predicted crossing dates for each grid cell where the SIC crosses the specified thresholds.
    """
    
    # Convert initial date to pandas datetime
    init_date = pd.to_datetime(init_date, format="%Y-%m-%d")
    print_init_date = init_date.strftime("%d %b %Y")
    
    # Load coast mask based on the data source
    if data_source in ("icenet_1w", "icenet_2w", "icenet_3w", "osisaf"):
        coast_mask = xr.open_mfdataset(f"{config.PROJECT_PATH}/data/masks/crossing_grid_cells.nc")
    elif data_source == "amsr":
        coast_mask = xr.open_mfdataset(f"{config.PROJECT_PATH}/data/masks/crossing_grid_cells_amsr.nc")
    
    # Select the coast mask for the initial date's year
    mask = coast_mask.sel(time=f"{init_date.year}-01-01")

    full_results = []  # List to store results for each threshold

    # Load and process data based on the source
    if data_source == "amsr":
        xarray_year = xr.open_dataset(f"{config.PATH_TO_AMSR2}/{init_date.year}.nc")
        xarray_gulf = xarray_year.sel(
            time=slice(init_date, f"{init_date.year}-12-31")).sel(
            x=slice(*config.amsr_plot_config['crop_x']),
            y=slice(*config.amsr_plot_config['crop_y']))
        xarray_smooth = xarray_gulf.z.rolling(time=smooth, center=True).mean()

    elif data_source == "osisaf":
        xarray_year = load_osisaf_year(f"{config.PATH_TO_OSISAF}/{init_date.year}.nc")
        xarray_gulf = xarray_year.sel(
            time=slice(init_date, f"{init_date.year}-12-31")).sel(
            xc=slice(*config.osisaf_plot_config['crop_x']),
            yc=slice(*config.osisaf_plot_config['crop_y']))
        xarray_smooth = xarray_gulf.rolling(time=smooth, center=True).mean()

    # Process different ICENET data sources
    elif data_source in ["icenet_1w", "icenet_2w", "icenet_3w"]:
        if init_date.year in [2021, 2022]:
            xarray_year = load_icenet_year(f"{config.PATH_TO_ICENET}/icenet_coronation_gulf_{init_date.year}.nc")
        else:
            xarray_year = load_icenet_year(f"{config.PATH_TO_ICENET}/icenet_coronation_gulf_{init_date.year}.nc")
        
        # Select SIC data based on the specific ICENET source and initial date
        if data_source == "icenet_1w":
            xarray_gulf = xarray_year.sic_mean.sel(time=f"{init_date.year}-10-24").sel(
                xc=slice(*config.icenet_plot_config['crop_x']),
                yc=slice(*config.icenet_plot_config['crop_y']))
        elif data_source == "icenet_2w":
            xarray_gulf = xarray_year.sic_mean.sel(time=f"{init_date.year}-10-17").sel(
                xc=slice(*config.icenet_plot_config['crop_x']),
                yc=slice(*config.icenet_plot_config['crop_y']))
        elif data_source == "icenet_3w":
            xarray_gulf = xarray_year.sic_mean.sel(time=f"{init_date.year}-10-10").sel(
                xc=slice(*config.icenet_plot_config['crop_x']),
                yc=slice(*config.icenet_plot_config['crop_y']))

        xarray_smooth = xarray_gulf  # No smoothing applied for ICENET sources

    # Loop through both SIC thresholds
    for i, chosen_percent in enumerate([percent_one, percent_two]):
        
        # Load SIC to concentration mapping data for the current data source
        mapping_df = load_mapping_df(data_source)
        chosen_sic = mapping_df[mapping_df.percent_interp == chosen_percent].sic_interp.values[0]

        # Process data based on the data source
        if data_source == "osisaf":
            mask_gulf = mask.sel(xc=slice(*config.osisaf_plot_config['crop_x']),
                                 yc=slice(*config.osisaf_plot_config['crop_y']))
            matching_array = np.apply_along_axis(get_matching_index_array, 0,
                                                 xarray_smooth.ice_conc.to_numpy(),
                                                 chosen_sic, n_consec)
            data = xarray_gulf.copy()
            data["matching_gridcells"] = (("yc", "xc"), matching_array)
            data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")
            result = data_expanded["matching_gridcells"].where(mask_gulf != 0).ice_conc

        elif data_source in ["icenet_1w", "icenet_2w", "icenet_3w"]:
            mask_gulf = mask.sel(xc=slice(*config.osisaf_plot_config['crop_x']),
                                 yc=slice(*config.osisaf_plot_config['crop_y']))
            matching_array = np.apply_along_axis(get_matching_index_array, 2,
                                                 xarray_smooth.to_numpy(),
                                                 chosen_sic, n_consec)
            data = xarray_gulf.sel(leadtime=1).copy()
            data["matching_gridcells"] = (("yc", "xc"), matching_array)
            data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")
            result = data_expanded["matching_gridcells"].where(mask_gulf != 0).ice_conc

        elif data_source == "amsr":
            mask_gulf = mask.sel(x=slice(*config.amsr_plot_config['crop_x']),
                                 y=slice(*config.amsr_plot_config['crop_y']))
            matching_array = np.apply_along_axis(get_matching_index_array, 0,
                                                 xarray_smooth.to_numpy() / 100,
                                                 chosen_sic, n_consec)
            data = xarray_gulf.copy()
            data["matching_gridcells"] = (("y", "x"), matching_array)
            data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")
            result = data_expanded["matching_gridcells"].where(mask_gulf != 0).z

        # Extract valid predicted crossing days (non-NaN values)
        all_days = result.values[~np.isnan(result.values)]

        # If no valid predictions, skip to next iteration
        if len(all_days) == 0:
            pass
        else:
            # Create a DataFrame for the valid predicted crossing days
            result_df = pd.DataFrame(data={"pred_day_num": all_days}, dtype=np.int8)
            result_df["init_date"] = init_date
            result_df["date"] = result_df.apply(lambda x: x.init_date + np.timedelta64(x.pred_day_num, "D"), axis=1)
            result_df["year"] = init_date.year
            result_df["label"] = "predicted"
            result_df["percent"] = chosen_percent
            result_df["sic_thresh"] = chosen_sic
            result_df["data_source"] = data_source

            # Append the result DataFrame to the full results list
            full_results.append(result_df)

    # Concatenate and return result
    return pd.concat(full_results)

