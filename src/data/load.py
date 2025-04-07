"""
Module for loading and preprocessing sea ice concentration and telemetry data
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from scipy import interpolate

import sys
sys.path.append("../../")
from src import config


def load_tracking_data(path_to_csv: str,
                       start_year: int = 1980,
                       date_col_name: str = "t",
                       source_crs: str = "epsg:4326") -> object:
    """
    Load tracking data from a CSV file and convert it into a GeoDataFrame.

    Args:
        path_to_csv (str): Path to the CSV file containing tracking data.
        start_year (int, optional): Minimum year of data to include. Defaults to 1980.
        date_col_name (str, optional): Name of the column containing datetime values. Defaults to "t".
        source_crs (str, optional): Coordinate reference system of the input data. Defaults to "epsg:4326".

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the filtered tracking data with geometry.
    """

    # Read data with datetime column
    raw_df = pd.read_csv(path_to_csv, parse_dates=[date_col_name])
    # Load geometry as for geopandas
    raw_df["geometry"] = raw_df["geometry"].apply(wkt.loads)
    # Filter data based on the specified year
    raw_df = raw_df[raw_df.year >= start_year]
    # Convert to GeoDataFrame
    raw_gdf = gpd.GeoDataFrame(raw_df, crs=source_crs)

    return raw_gdf


def load_migration_start_points(path_to_csv: str, 
                                female_only: bool = False, 
                                start_year: int = 1980, 
                                cross_only: bool = False) -> pd.DataFrame:
    """
    Load migration start points from a CSV file and apply filtering criteria.

    Args:
        path_to_csv (str): Path to the CSV file containing migration data.
        female_only (bool, optional): If True, filters out male (bull) migration data. Defaults to False.
        start_year (int, optional): Minimum year of migration data to include. Defaults to 1980.
        cross_only (bool, optional): If True, includes only records where migration involved a crossing event. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered migration start points.
    """
    # Load dataset from CSV file
    full_df = pd.read_csv(path_to_csv, index_col=0)
    # Filter data based on start year
    full_df = full_df[full_df.year >= start_year]
    # Remove records for male (bull) individuals if female_only is set to True
    if female_only:
        full_df = full_df[~full_df.bull_tag]
    # Retain only records where a crossing event occurred if cross_only is set to True
    if cross_only:
        full_df = full_df[full_df.crossing == "cross"]
    return full_df


def load_migration_start_points_gdf(path_to_geojson: str, 
                                    female_only: bool = False, 
                                    start_year: int = 1980, 
                                    day_diff_lim: float = np.inf) -> gpd.GeoDataFrame:
    """
    Load migration start points from a GeoJSON file and apply filtering criteria.

    Args:
        path_to_geojson (str): Path to the GeoJSON file containing migration data.
        female_only (bool, optional): If True, filters out male (bull) migration data. Defaults to False.
        start_year (int, optional): Minimum year of migration data to include. Defaults to 1980.
        day_diff_lim (float, optional): Maximum allowed difference in days for migration duration.
            Entries with `days_int` greater than this limit are removed. Defaults to infinity.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the filtered migration start points with additional derived columns.
    """

    # Load dataset from GeoJSON file
    full_gdf = gpd.read_file(path_to_geojson)

    # Filter data based on start year
    full_gdf = full_gdf[full_gdf.year >= start_year]

    # Remove records for male (bull) individuals if female_only is set to True
    if female_only:
        full_gdf = full_gdf[~full_gdf.bull_tag]

    # Set the time column as the index
    full_gdf.set_index("t", inplace=True)

    # Add additional derived information from time columns
    full_gdf["doy"] = full_gdf.index.dayofyear  # Day of the year
    full_gdf["diff_days"] = full_gdf.full_time / 24  # Convert hours to days
    full_gdf["days_int"] = np.ceil(full_gdf.diff_days).astype(int)  # Round up to nearest whole day

    # Filter data based on the maximum allowed migration duration
    full_gdf = full_gdf[full_gdf.days_int <= day_diff_lim]

    return full_gdf


def load_sic_trace_csv(path_to_csv: str, 
                       female_only: bool = False, 
                       start_year: int = 1980, 
                       end_year: int = 2023, 
                       day_diff_lim: float = np.inf) -> gpd.GeoDataFrame:
    """
    Load SIC (Sea Ice Concentration) trace data from a CSV file and apply filtering criteria.

    Args:
        path_to_csv (str): Path to the CSV file containing SIC trace data.
        female_only (bool, optional): If True, filters out male (bull) migration data. Defaults to False.
        start_year (int, optional): Minimum year of data to include. Defaults to 1980.
        end_year (int, optional): Maximum year of data to include. Defaults to 2023.
        day_diff_lim (float, optional): Maximum allowed difference in days for migration duration.
            Entries with `days_int` greater than this limit are removed. Defaults to infinity.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the filtered SIC trace data with geometry.
    """

    # Load dataset and parse datetime columns
    sic_df = pd.read_csv(path_to_csv, parse_dates=["time", "mig_date"])

    # Filter data by year range
    sic_df = sic_df[sic_df.year >= start_year]
    sic_df = sic_df[sic_df.year <= end_year]

    # Remove records for male (bull) individuals if female_only is set to True
    if female_only:
        sic_df = sic_df[~sic_df.bull_tag]

    # Set the time column as the index
    sic_df.set_index("time", inplace=True)

    # Filter data based on the maximum allowed migration duration
    sic_df = sic_df[sic_df.days_int <= day_diff_lim]

    # Convert geometry column to a proper geometry format for GeoPandas
    sic_df["geometry"] = sic_df.geometry.apply(wkt.loads)

    # Convert DataFrame to GeoDataFrame with WGS84 CRS (EPSG:4326)
    sic_gdf = gpd.GeoDataFrame(sic_df, geometry="geometry", crs=4326)

    return sic_gdf



def merge_osisaf_and_amsr(osi_filename: str, 
                           amsr_filename: str, 
                           female_only: bool = False, 
                           start_year: int = 1980, 
                           day_diff_lim: float = np.inf) -> pd.DataFrame:
    """
    Merge OSISAF and AMSR sea ice concentration (SIC) datasets based on common identifiers.

    Args:
        osi_filename (str): Path to the CSV file containing OSISAF SIC data.
        amsr_filename (str): Path to the CSV file containing AMSR SIC data.
        female_only (bool, optional): If True, filters out male (bull) migration data. Defaults to False.
        start_year (int, optional): Minimum year of data to include. Defaults to 1980.
        day_diff_lim (float, optional): Maximum allowed difference in days for migration duration.
            Entries with `days_int` greater than this limit are removed. Defaults to infinity.

    Returns:
        pd.DataFrame: A merged DataFrame containing OSISAF and AMSR SIC data with filtering applied.
    """

    # Load OSISAF and AMSR SIC data as GeoDataFrames
    osi_gdf = load_sic_trace_csv(osi_filename)
    amsr_gdf = load_sic_trace_csv(amsr_filename)

    # Restore 'time' column from index in OSISAF dataset
    osi_gdf["time"] = osi_gdf.index

    # Merge datasets on common identifiers
    merge_df = pd.merge(osi_gdf, 
                        amsr_gdf[['FieldID', 'mig_date', 'z', 'day_index']], 
                        on=['FieldID', 'mig_date', 'day_index'])

    # Convert OSISAF ice concentration to percentage
    merge_df['osi_sic'] = merge_df['ice_conc'] * 100

    # Rename AMSR SIC column
    merge_df.rename(columns={'z': 'amsr_sic'}, inplace=True)

    # Set time column as index
    merge_df = merge_df.set_index('time')

    # Filter data based on minimum year
    merge_df = merge_df[merge_df.year >= start_year]

    # Remove records for male (bull) individuals if female_only is set to True
    if female_only:
        merge_df = merge_df[~merge_df.bull_tag]

    # Filter data based on the maximum allowed migration duration
    merge_df = merge_df[merge_df.days_int <= day_diff_lim]

    return merge_df



def load_osm_for_aoi(path_to_aoi: str) -> gpd.GeoDataFrame:
    """
    Load OpenStreetMap (OSM) land data for a given area of interest (AOI) and merge land polygons.

    Args:
        path_to_aoi (str): Path to a GeoJSON file defining the area of interest (AOI).

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the merged OSM land polygons for the specified AOI.
    """

    # Open and read AOI GeoJSON file
    with open(path_to_aoi) as file:
        aoi_gdf = gpd.read_file(file)

    # Load OSM land polygons for the AOI
    land_gdf = gpd.read_file(config.PATH_TO_OSM, mask=aoi_gdf)

    # Merge individual land polygons into a single polygon
    land_polygon = land_gdf.unary_union

    # Convert merged polygon into a GeoDataFrame
    polygon_gdf = gpd.GeoDataFrame(pd.DataFrame([{'geometry': land_polygon, 'id': 1}]), crs=4326)

    return polygon_gdf


def read_csv_as_datetime_geometry(path_to_csv: str,
                                  date_format: str = "%Y-%m-%d %H:%M",
                                  source_crs: str = "epsg:4326",
                                  input_time_col: str = "t") -> gpd.GeoDataFrame:
    """
    Load caribou tracking data from a CSV file and convert it to a datetime-indexed GeoDataFrame.

    Args:
        path_to_csv (str): Path to the tracking data CSV file.
        date_format (str, optional): Format of the datetime column in the input CSV. Defaults to "%Y-%m-%d %H:%M".
        source_crs (str, optional): Coordinate reference system (CRS) of the tracking dataset. Defaults to "epsg:4326".
        input_time_col (str, optional): Name of the column containing datetime information. Defaults to "t".

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the tracking data with a datetime index, geometry points, and year.
    """

    # Read the CSV file
    raw_df = pd.read_csv(path_to_csv)

    # Convert the specified datetime column to a pandas datetime format and set it as the index
    raw_df["t"] = pd.to_datetime(raw_df[input_time_col], format=date_format)
    raw_df = raw_df.set_index("t")

    # Convert longitude and latitude columns into geometric point objects
    raw_gdf = gpd.GeoDataFrame(raw_df, crs=source_crs,
                               geometry=[Point(xy) for xy in zip(raw_df.Longitude, raw_df.Latitude)])

    # Extract the year from the datetime index
    raw_gdf["year"] = raw_gdf.index.year

    return raw_gdf[["FieldID", "geometry", "year"]]


def load_osisaf(path_to_data: str, date: str) -> xr.DataArray:
    """
    Load OSISAF sea ice concentration data for a given date.

    Args:
        path_to_data (str): Path to the OSISAF data folder containing NetCDF files.
        date (str): Date in "YYYY-MM-DD" format.

    Returns:
        xr.DataArray: Sea ice concentration data for the given date with appropriate attributes and filtering.
    """

    # Construct file path for the specified year
    file_path = f"{path_to_data}/{date.year}.nc"

    # Load dataset for the specified year
    sic_data = xr.open_mfdataset(file_path).load()

    # Extract sea ice concentration for the specific date
    sic_data = sic_data["ice_conc"].sel(time=date.strftime("%Y-%m-%d"))

    # Apply land mask to remove land regions
    mask_array = np.load(config.LAND_MASK_PATH)
    sic_data = sic_data.where(mask_array == 0)

    # Convert coordinate units from kilometers to meters
    sic_data = sic_data.assign_coords(xc=sic_data.xc * 1e3, yc=sic_data.yc * 1e3)

    # Scale sea ice concentration values to percentage
    sic_data = sic_data * 100

    # Assign metadata attributes to match the original OSISAF dataset format
    sic_data = sic_data.assign_attrs({
        'long_name': 'Fully filtered concentration of sea ice using atmospheric '
                     'correction of brightness temperatures and open water filters',
        'standard_name': 'sea_ice_area_fraction',
        'units': '%',
        'valid_min': 0,
        'valid_max': 10000,
        'grid_mapping': 'Lambert_Azimuthal_Grid',
        'ancillary_variables': 'total_standard_error status_flag',
        'comment': 'This field is the primary sea ice concentration '
                   'estimate for this climate data record',
        '_ChunkSizes': np.array([1, 432, 432])
    })

    return sic_data


def load_amsr2(path_to_data: str, date: str) -> xr.DataArray:
    """
    Load AMSR2 sea ice concentration data for a given date.

    Args:
        path_to_data (str): Path to the AMSR2 data folder containing NetCDF files.
        date (str): Date in "YYYY-MM-DD" format.

    Returns:
        xr.DataArray: Sea ice concentration data from the AMSR2 dataset for the specified date.
    """

    # Format the date to match the filename pattern
    format_date = f"{date.year}{date.month:02d}{date.day:02d}"

    # Construct the file path for the specified date
    file_path = f"{path_to_data}/asi-AMSR2-n6250-{format_date}-v5.4.nc"

    # Load the dataset
    sic_data = xr.open_dataset(file_path).load()

    return sic_data


def load_osisaf_year(path_to_data: str) -> xr.DataArray:
    """
    Load OSISAF sea ice concentration data for a specific year and apply land mask.

    Args:
        path_to_data (str): Path to the OSISAF dataset (NetCDF file) for the specific year.

    Returns:
        xr.DataArray: Processed sea ice concentration data with land areas masked out and coordinates converted.
    """

    # Load OSISAF dataset and land mask
    sic_data = xr.open_dataset(path_to_data)
    land_mask = np.load(config.LAND_MASK_PATH)

    # Apply land mask to exclude land regions
    sic_data = sic_data.where(land_mask == 0)

    # Convert coordinates from kilometers to meters
    sic_data = sic_data.assign_coords(xc=sic_data.xc * 1e3, yc=sic_data.yc * 1e3)

    return sic_data


def load_icenet(xr_forecast: object, year: int, season: str, date: pd.Timestamp) -> xr.DataArray:
    """
    Load sea ice concentration forecast for a specific date, season, and lead time.

    Args:
        xr_forecast (object): xarray object containing sea ice concentration forecast data.
        year (int): Year for which the forecast is required.
        season (str): Season to load data for, either 'spring' or 'autumn'.
        date (pd.Timestamp): Specific date for which the forecast is needed.

    Returns:
        xr.DataArray: Processed sea ice concentration forecast for the specified date and lead time, with coordinates in meters.
    """

    # Define start date based on the season
    if season == "spring":
        start_date = pd.Timestamp(f'{year}-03-01')
    elif season == "autumn":
        start_date = pd.Timestamp(f'{year}-10-01')
    else:
        print(f"Input invalid season <{season}>, should be either spring or autumn")
        return None  # Exit if season is invalid

    # Calculate the lead time in days
    leadtime = (date - start_date).days

    # Extract the sea ice concentration forecast for the specified date and lead time
    icenet_pred = xr_forecast["sic_mean"].sel(time=start_date.strftime("%Y-%m-%d")).sel(leadtime=leadtime)

    # Remove zero values
    icenet_pred = icenet_pred.where(icenet_pred != 0)

    # Convert coordinates from kilometers to meters
    icenet_pred = icenet_pred.assign_coords(xc=icenet_pred.xc * 1e3, yc=icenet_pred.yc * 1e3)

    return icenet_pred


def load_icenet_year(forecast_file: str) -> object:
    """
    Load a dataset containing sea ice concentration forecasts for a specific year.

    Args:
        forecast_file (str): Path to the NetCDF file containing the sea ice concentration forecast data.

    Returns:
        object: xarray dataset containing the sea ice concentration forecast for the specified year.
    """

    # Load the forecast data using xarray
    forecast = xr.open_mfdataset(forecast_file)

    return forecast


def load_train_test_csv(data_source: str, female_only: bool, train_end: int, day_diff_lim: int) -> tuple:
    """
    Load training and testing datasets for sea ice concentration based on the specified parameters.

    Args:
        data_source (str): The data source identifier (e.g., "amsr").
        female_only (bool): Whether to filter the data to include only female individuals (True or False).
        train_end (int): The end year for the training dataset.
        day_diff_lim (int): The maximum number of days to include in the dataset.

    Returns:
        tuple: A tuple containing the training and testing datasets as GeoDataFrames (data_train, data_test).
    """

    # Retrieve the file path for the specified data source
    sic_data_file = config.SIC_DATA_FILES[data_source]

    # Load the training and testing datasets using the specified parameters
    data_train = load_sic_trace_csv(sic_data_file, female_only=female_only, start_year=1990,
                                    end_year=train_end, day_diff_lim=day_diff_lim)
    data_test = load_sic_trace_csv(sic_data_file, female_only=female_only, start_year=train_end + 1,
                                   end_year=2023, day_diff_lim=day_diff_lim)

    # If the data source is "amsr", scale the "z" column (sea ice concentration) by 100
    if data_source == "amsr":
        data_train["z"] = data_train.z / 100
        data_test["z"] = data_test.z / 100

    return data_train, data_test
