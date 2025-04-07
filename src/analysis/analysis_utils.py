"""
Module for analysing telemetry data with geospatial sea ice concentration datasets and IceNet forecasts
"""

import pandas as pd
import cartopy.crs as ccrs
from pyproj import Transformer
import datetime
import xarray as xr
import numpy as np
from sklearn.utils import resample

import sys
sys.path.append("..")
from src.data.load import load_osisaf_year
from src import config
from src.plotting.sic_plots import per_sample_sic_and_location
from src.data.process import interpolate_sic

pd.options.mode.chained_assignment = None  # default='warn'


def calc_mean_count_ci(_df, groupby_cols=None, stat_col=None, n_bootstrap=1000):
    """
    Calculate mean, count, and confidence intervals using bootstrapping.
 
    Parameters:
    - _df (DataFrame): Input DataFrame containing the data.
    - groupby_cols (list, optional): List of columns to group the data by. Default is None.
    - stat_col (str, optional): Column name for which to calculate mean, count, and confidence intervals.
    - n_bootstrap (int, optional): Number of bootstrap samples to generate. Default is 1000.
 
    Returns:
    - DataFrame: A DataFrame containing mean, count, and confidence interval information
                 for the specified groupby columns and statistical column.
 
    Example:
    >>> result_df = calc_mean_count_ci(df, ['A', 'B'], 'score').
    """
 
    # Define a function to calculate confidence interval using bootstrapping
    def calculate_ci_bootstrap(data):
        means = []
        for _ in range(n_bootstrap):
            sample = resample(data)  # Bootstrap resampling
            mean = np.mean(sample)  # Calculate mean of the resampled data
            means.append(mean)
 
        # Calculate the mean of the whole dataset and the confidence interval 
        # from the distribution of the sample means
        data_mean = np.mean(data)
        confidence_interval = np.percentile(means, [2.5, 97.5])
 
        return pd.DataFrame({
            'count': [len(data)],
            'mean_ci_lower': [confidence_interval[0]],
            'mean_value': [data_mean],
            'mean_ci_upper': [confidence_interval[1]],
        }).round(4)
 
    # Group by columns A and B, apply the calculate_ci_bootstrap function to 
    # the 'score' column, and merge the results
    result = (
        _df.groupby(groupby_cols)[stat_col]
        .apply(calculate_ci_bootstrap)
        .reset_index()
        .drop(columns=["level_2"])
    )
    return result


def get_sic_migration_timeseries(migration_gdf: object, sample_range: int = 45) -> object:
    """
    Generate a time series of Sea Ice Concentration (SIC) values for migration observations based on 
    a provided GeoDataFrame of migration data. The time series is sampled around each observation date.

    This function loops over each year in the migration GeoDataFrame and retrieves SIC data for 
    each observation within a specified sample range. The SIC values are taken from OSISAF data.

    Parameters:
    -----------
    migration_gdf : GeoDataFrame
        A GeoDataFrame containing migration data with columns for location (geometry) and year information.
    sample_range : int, optional
        The range (in days) to sample around each observation's date. Default is 45 days.

    Returns:
    --------
    DataFrame
        A Pandas DataFrame containing the migration data with additional columns for the SIC values, 
        time indices, and other relevant information.
    """

    full_df = []  # To store the results for all years and observations

    # Define the transformation from geographic coordinates (lat/lon) to Lambert Azimuthal Equal Area
    transform = ccrs.LambertAzimuthalEqualArea(0, 90)
    transformer = Transformer.from_crs('epsg:4326', transform)  # EPSG:4326 corresponds to WGS 84

    counter = 0  # Counter to keep track of observation IDs

    # Loop through each year in the migration GeoDataFrame
    for year, year_gdf in migration_gdf.groupby("year"):

        print(f"Processing year: {year}")
        
        # Load the OSISAF data for the current year and the following year
        year_xr = load_osisaf_year(f"{config.PATH_TO_OSISAF}/{year}.nc")
        year2_xr = load_osisaf_year(f"{config.PATH_TO_OSISAF}/{year + 1}.nc")
        
        # Merge the two years of OSISAF data to account for observations that span across the year-end
        full_xr = year_xr.merge(year2_xr, compat="override")

        # Loop through each observation in the migration GeoDataFrame for the current year
        for time_idx, row in year_gdf.iterrows():
            print(f"Processing observation at index: {time_idx}")

            # Define the time window around the current observation for SIC sampling
            start_date = time_idx - datetime.timedelta(days=sample_range + 1)
            end_date = time_idx + datetime.timedelta(days=sample_range)

            # Sample the SIC data for the specified date range
            sample_xr = full_xr.sel(time=slice(start_date, end_date))

            # Transform the observation location (latitude, longitude) to the target projection
            target_x, target_y = transformer.transform(row.geometry.y, row.geometry.x)

            # Extract the SIC value at the observation location using the nearest grid point
            sample_sic = sample_xr.sel(xc=target_x, yc=target_y, method="nearest")

            # Convert the extracted SIC data to a DataFrame, along with the migration information
            result_df = sample_sic.to_dataframe()[["lon", "lat", "ice_conc"]]
            
            # Insert additional information into the result DataFrame
            result_df.insert(0, "geometry", row.geometry)
            result_df.insert(0, "mig_date", time_idx)  # Original observation date
            result_df.insert(0, "FieldID", row.FieldID)
            result_df.insert(0, "full_time", row.full_time)
            result_df.insert(0, "full_dist", row.full_dist)
            result_df.insert(0, "obs_id", counter)  # Observation counter ID

            # Append the result to the full_df list
            full_df.append(result_df)

            # Increment the observation counter
            counter += 1

    # Concatenate all the results into a single DataFrame
    final_df = pd.concat(full_df)

    # Calculate the day index (number of days since the observation date)
    final_df["day_index"] = (final_df.index.date - final_df.mig_date.dt.date).dt.days

    return final_df


def get_osisaf_migration_timeseries(migration_gdf: object, sample_range: int = 45) -> object:
    """
    Generate a time series of Sea Ice Concentration (SIC) values for migration observations based on 
    the provided GeoDataFrame of migration data, including interpolation of missing values at the coastline.

    This function processes migration data over multiple years, retrieves SIC data from the OSISAF dataset,
    and interpolates missing SIC values. It also calculates additional migration-related columns for each
    observation and stores the results in a Pandas DataFrame.

    Parameters:
    -----------
    migration_gdf : GeoDataFrame
        A GeoDataFrame containing migration data, with columns such as 'geometry' (for location), 
        'year', 'FieldID', 'full_time', 'full_dist', 'bull_tag', 'diff_days', and 'days_int'.
    sample_range : int, optional
        The range (in days) to sample around each observation's date for SIC values. Default is 45 days.

    Returns:
    --------
    DataFrame
        A Pandas DataFrame containing the migration data with SIC values and other calculated fields, 
        including a "day_index" indicating the number of days since the migration date.
    """
    
    full_df = []  # List to store results for all observations

    # Define the projection for converting lat/lon to the target CRS
    transform = ccrs.LambertAzimuthalEqualArea(0, 90)
    transformer = Transformer.from_crs('epsg:4326', transform)  # WGS 84 to Lambert Azimuthal Equal Area

    counter = 0  # Counter to track unique observation IDs

    # Loop through each year in the migration GeoDataFrame
    for year, year_gdf in migration_gdf.groupby("year"):

        print(f"Processing year: {year}")
        
        # Load the OSISAF data for the current year and the following year
        year_xr = load_osisaf_year(f"{config.PATH_TO_OSISAF}/{year}.nc")
        year2_xr = load_osisaf_year(f"{config.PATH_TO_OSISAF}/{year + 1}.nc")
        
        # Concatenate the relevant months from both years to account for year-end transitions
        full_xr = xr.concat([year_xr.isel(time=year_xr.time.dt.month.isin([8, 9, 10, 11, 12])),
                             year2_xr.isel(time=year2_xr.time.dt.month.isin([1, 2, 3]))], dim="time")

        # Loop through each observation in the current year's migration GeoDataFrame
        for time_idx, row in year_gdf.iterrows():
            print(f"Processing observation at index: {time_idx}")

            # Define the start and end date for the sample range around the observation
            start_date = time_idx - datetime.timedelta(days=sample_range + 1)
            end_date = time_idx + datetime.timedelta(days=sample_range)

            # Sample the SIC data for the date range around the observation
            sample_xr = full_xr.sel(time=slice(start_date, end_date)).sel(
                xc=slice(*config.osisaf_plot_config['crop_x']),
                yc=slice(*config.osisaf_plot_config['crop_y']))

            # Transform the observation location (lat, lon) to the target projection
            target_x, target_y = transformer.transform(row.geometry.y, row.geometry.x)

            # Retrieve SIC data at the observation location using the nearest grid point
            sample_sic = sample_xr.sel(xc=target_x, yc=target_y, method="nearest")
            missing_values = np.isnan(sample_sic.ice_conc.values)

            # If there are missing SIC values, interpolate them
            if missing_values.any():
                print(f"{missing_values.sum()} days of records are NaN and require interpolation...")
                missing_dates = sample_sic.time.values[missing_values]

                interp_values = []
                # Interpolate missing SIC values
                for interp_date in missing_dates:
                    interp_day = interpolate_sic(sample_xr.sel(time=interp_date), method="linear", dataset="osisaf")
                    interp_val = interp_day.sel(xc=target_x, yc=target_y, method="nearest")
                    interp_values.append(interp_val.ice_conc.values)

                # Replace the missing values with the interpolated values
                interpolated_array = sample_sic.ice_conc.values
                interpolated_array[missing_values] = interp_values

                final_sic = sample_sic.copy()
                final_sic.ice_conc.values = interpolated_array
                print("Interpolation complete!")
            else:
                final_sic = sample_sic

            # Convert the resulting SIC data to a DataFrame
            result_df = final_sic.to_dataframe()[["lon", "lat", "ice_conc"]]
            
            # Insert additional migration-related data into the DataFrame
            result_df.insert(0, "geometry", row.geometry)
            result_df.insert(0, "interp", missing_values)
            result_df.insert(0, "bull_tag", row.bull_tag)
            result_df.insert(0, "mig_date", time_idx)
            result_df.insert(0, "FieldID", row.FieldID)
            result_df.insert(0, "full_time", row.full_time)
            result_df.insert(0, "full_dist", row.full_dist)
            result_df.insert(0, "diff_days", row.diff_days)
            result_df.insert(0, "days_int", row.days_int)
            result_df.insert(0, "year", row.year)
            result_df.insert(0, "doy", row.doy)
            result_df.insert(0, "obs_id", counter)

            # Append the results for this observation to the full list
            full_df.append(result_df)

            # Increment the observation counter
            counter += 1

    # Concatenate all the DataFrames from different observations
    final_df = pd.concat(full_df)

    # Calculate the day index, i.e., number of days since the migration date
    final_df["day_index"] = (final_df.index.date - final_df.mig_date.dt.date).dt.days

    return final_df


def get_amsr2_migration_timeseries(migration_gdf: object,
                                   sample_range: int = 45,
                                   coarsen: bool = False) -> object:
    """
    Generate a time series of Sea Ice Concentration (SIC) values for migration observations based on 
    the provided GeoDataFrame of migration data, using AMSR2 satellite data. Missing SIC values are 
    interpolated when necessary. The data can optionally be coarsened.

    This function processes migration data over multiple years (starting from 2012), retrieves SIC data 
    from the AMSR2 dataset, and interpolates missing SIC values if any. It calculates additional migration 
    related columns for each observation and stores the results in a Pandas DataFrame.

    Parameters:
    -----------
    migration_gdf : GeoDataFrame
        A GeoDataFrame containing migration data, with columns such as 'geometry' (for location), 
        'year', 'FieldID', 'full_time', 'full_dist', 'bull_tag', 'diff_days', and 'days_int'.
        
    sample_range : int, optional
        The range (in days) to sample around each observation's date for SIC values. Default is 45 days.
        
    coarsen : bool, optional
        Whether to coarsen the data before interpolation. Default is False (no coarsening).

    Returns:
    --------
    DataFrame
        A Pandas DataFrame containing the migration data with SIC values and other calculated fields, 
        including a "day_index" indicating the number of days since the migration date.
    """
    
    full_df = []  # List to store results for all observations

    # Define the projection for converting lat/lon to the target CRS (EPSG:3411)
    transform = ccrs.epsg(3411)
    transformer = Transformer.from_crs('epsg:4326', transform)  # WGS 84 to AMSR2 projection

    # Filter migration data for years starting from 2012 (AMSR2 data is available after 2012)
    migration_gdf = migration_gdf[migration_gdf.year >= 2012]

    counter = 0  # Counter to track unique observation IDs

    # Loop through each year in the migration GeoDataFrame
    for year, year_gdf in migration_gdf.groupby("year"):

        print(f"Processing {len(year_gdf)} records for year: {year}")
        
        # Load the AMSR2 data for the current year and the following year
        year_xr = xr.open_dataset(f"{config.PATH_TO_AMSR2}/{year}.nc")
        year2_xr = xr.open_dataset(f"{config.PATH_TO_AMSR2}/{year + 1}.nc")
        
        # Concatenate the relevant months from both years to account for year-end transitions
        full_xr = xr.concat([year_xr.isel(time=year_xr.time.dt.month.isin([8, 9, 10, 11, 12])),
                             year2_xr.isel(time=year2_xr.time.dt.month.isin([1, 2, 3]))], dim="time")

        # Loop through each observation in the current year's migration GeoDataFrame
        for time_idx, row in year_gdf.iterrows():
            print(f"Processing observation at index: {time_idx}")

            # Define the start and end date for the sample range around the observation
            start_date = time_idx - datetime.timedelta(days=sample_range + 1)
            end_date = time_idx + datetime.timedelta(days=sample_range)

            # Sample the SIC data for the date range around the observation
            sample_xr = full_xr.sel(time=slice(start_date, end_date)).sel(
                x=slice(*config.amsr_plot_config['crop_x']),
                y=slice(*config.amsr_plot_config['crop_y']))

            # Optionally coarsen the data (mean over 2x2 blocks)
            if coarsen:
                sample_xr = sample_xr.coarsen(x=2, y=2, boundary="pad").mean(skipna=True)

            # Transform the observation location (lat, lon) to the target projection (AMSR2)
            target_x, target_y = transformer.transform(row.geometry.y, row.geometry.x)

            # Retrieve SIC data at the observation location using the nearest grid point
            sample_sic = sample_xr.sel(x=target_x, y=target_y, method="nearest")
            missing_values = np.isnan(sample_sic.z.values)

            # If there are missing SIC values, interpolate them
            if missing_values.any():
                print(f"{missing_values.sum()} days of records are NaN and require interpolation...")
                missing_dates = sample_sic.time.values[missing_values]

                interp_values = []
                # Interpolate missing SIC values
                for interp_date in missing_dates:
                    if coarsen:
                        interp_day = interpolate_sic(sample_xr.sel(time=interp_date), method="linear", dataset="amsr_coarse")
                    else:
                        interp_day = interpolate_sic(sample_xr.sel(time=interp_date), method="linear", dataset="amsr")
                    interp_val = interp_day.sel(x=target_x, y=target_y, method="nearest")
                    interp_values.append(interp_val.z.values)

                # Replace the missing values with the interpolated values
                interpolated_array = sample_sic.z.values
                interpolated_array[missing_values] = interp_values

                final_sic = sample_sic.copy()
                final_sic.z.values = interpolated_array
                print("Interpolation complete!")
            else:
                final_sic = sample_sic

            # Convert the resulting SIC data to a DataFrame
            result_df = final_sic.to_dataframe()#[["lon", "lat", "ice_conc"]]
            
            # Insert additional migration-related data into the DataFrame
            result_df.insert(0, "geometry", row.geometry)
            result_df.insert(0, "interp", missing_values)
            result_df.insert(0, "bull_tag", row.bull_tag)
            result_df.insert(0, "mig_date", time_idx)
            result_df.insert(0, "FieldID", row.FieldID)
            result_df.insert(0, "full_time", row.full_time)
            result_df.insert(0, "full_dist", row.full_dist)
            result_df.insert(0, "diff_days", row.diff_days)
            result_df.insert(0, "days_int", row.days_int)
            result_df.insert(0, "year", row.year)
            result_df.insert(0, "doy", row.doy)
            result_df.insert(0, "obs_id", counter)

            # Append the results for this observation to the full list
            full_df.append(result_df)

            # Increment the observation counter
            counter += 1

    # Concatenate all the DataFrames from different observations
    final_df = pd.concat(full_df)

    # Calculate the day index, i.e., number of days since the migration date
    final_df["day_index"] = (final_df.index.date - final_df.mig_date.dt.date).dt.days

    return final_df


def get_icenet_migration_timeseries(migration_gdf: object, icenet_init_ref: str) -> object:
    """
    Generate a time series of Sea Ice Concentration (SIC) values for migration observations 
    using the IceNet model forecast data. This function handles missing SIC data by interpolating 
    when necessary and calculates additional migration-related columns.

    The function processes migration data over multiple years, retrieves SIC data from the IceNet 
    dataset for each migration event, interpolates missing SIC values, and combines the results 
    into a DataFrame.

    Parameters:
    -----------
    migration_gdf : GeoDataFrame
        A GeoDataFrame containing migration data, including columns such as 'geometry' (for location), 
        'year', 'FieldID', 'full_time', 'full_dist', 'bull_tag', 'diff_days', and 'days_int'.
        
    icenet_init_ref : str
        The reference for the IceNet forecast data files (e.g., 'icenet_init').

    Returns:
    --------
    DataFrame
        A Pandas DataFrame containing the migration data with SIC values and other calculated fields, 
        including a "day_index" indicating the number of days since the migration date.
    """
    
    full_df = []  # List to store results for all observations

    # Define the projection for transforming lat/lon to the target CRS
    transform = ccrs.LambertAzimuthalEqualArea(0, 90)
    transformer = Transformer.from_crs('epsg:4326', transform)  # WGS 84 to IceNet projection

    # Filter migration data for years starting from 2012 (IceNet data available from this year)
    migration_gdf = migration_gdf[migration_gdf.year >= 2012]

    counter = 0  # Counter to track unique observation IDs

    # Loop through each year in the migration GeoDataFrame
    for year, year_gdf in migration_gdf.groupby("year"):

        print(f"Processing {len(year_gdf)} records for year: {year}")
        
        # Load the IceNet forecast data for the current year
        year_xr = xr.open_mfdataset(f"{config.PROJECT_PATH}/forecasts/{icenet_init_ref}_{year}.nc")

        # Loop through each observation in the current year's migration GeoDataFrame
        for time_idx, row in year_gdf.iterrows():
            print(f"Processing observation at index: {time_idx}")

            # Sample the IceNet forecast data for the specified region (based on configuration)
            sample_xr = year_xr.sel(
                xc=slice(*config.icenet_plot_config['crop_x']),
                yc=slice(*config.icenet_plot_config['crop_y']))

            # Transform the observation location (lat, lon) to the target projection
            target_x, target_y = transformer.transform(row.geometry.y, row.geometry.x)

            # Retrieve SIC data at the observation location using the nearest grid point
            sample_sic = sample_xr.sel(xc=target_x, yc=target_y, method="nearest")
            missing_values = np.isnan(sample_sic.sic_mean.values)

            # If there are missing SIC values, interpolate them
            if missing_values.any():
                print(f"{missing_values.sum()} days of records are NaN and require interpolation...")
                missing_dates = sample_sic.leadtime.values[missing_values]

                interp_values = []
                # Interpolate missing SIC values for each missing date
                for interp_date in missing_dates:
                    interp_day = interpolate_sic(sample_xr.sel(leadtime=interp_date), method="linear", dataset="icenet")
                    interp_val = interp_day.sel(xc=target_x, yc=target_y, method="nearest")
                    interp_values.append(interp_val.sic_mean.values)

                # Replace the missing SIC values with the interpolated values
                interpolated_array = sample_sic.sic_mean.values
                interpolated_array[missing_values] = interp_values

                final_sic = sample_sic.copy()
                final_sic.sic_mean.values = interpolated_array
                print("Interpolation complete!")
            else:
                final_sic = sample_sic

            # Convert the resulting SIC data to a DataFrame
            result_df = final_sic.to_dataframe()[["lon", "lat", "sic_mean"]]
            
            # Insert additional migration-related data into the DataFrame
            result_df.insert(0, "time",
                             [sample_xr.time.values + pd.DateOffset(days=n_days) for n_days in result_df.index])
            result_df.insert(0, "geometry", row.geometry)
            result_df.insert(0, "interp", missing_values)
            result_df.insert(0, "bull_tag", row.bull_tag)
            result_df.insert(0, "mig_date", time_idx)
            result_df.insert(0, "FieldID", row.FieldID)
            result_df.insert(0, "full_time", row.full_time)
            result_df.insert(0, "full_dist", row.full_dist)
            result_df.insert(0, "diff_days", row.diff_days)
            result_df.insert(0, "days_int", row.days_int)
            result_df.insert(0, "year", row.year)
            result_df.insert(0, "doy", row.doy)
            result_df.insert(0, "obs_id", counter)
            
            # Set the 'time' column as the index for the DataFrame
            result_df.set_index("time", inplace=True)

            # Append the results for this observation to the full list
            full_df.append(result_df)

            # Increment the observation counter
            counter += 1

    # Concatenate all the DataFrames from different observations
    final_df = pd.concat(full_df)

    # Calculate the day index, i.e., number of days since the migration date
    final_df["day_index"] = (final_df.index.date - final_df.mig_date.dt.date).dt.days

    return final_df
