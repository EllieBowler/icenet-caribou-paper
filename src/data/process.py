"""
Module for processing and interpolating the sea ice concentration and telemetry datasets
"""

import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import os
from shapely.geometry import Polygon
from scipy import interpolate
from pyproj import Transformer

import sys
sys.path.append("../../")
from src import config


def add_lat_lon_to_amsr(amsr_xarray):
    """
    Converts the x and y coordinates of an AMSR dataset to latitude and longitude.

    This function takes an xarray containing AMSR data with x and y coordinates 
    in a specified projection and transforms them to the WGS 84 (EPSG:4326) coordinate system.

    Parameters:
    ----------
    amsr_xarray : xarray.DataArray
        An xarray object containing AMSR data with 'x' and 'y' coordinates in a different projection.

    Returns:
    --------
    amsr_xarray : xarray.DataArray
        The input xarray with new latitude ('lat') and longitude ('lon') coordinates in EPSG:4326.
    """
    xv, yv = np.meshgrid(amsr_xarray.x, amsr_xarray.y)

    transformer = Transformer.from_crs(config.amsr_crs,
                                       "epsg:4326",
                                       always_xy=True,
                                       )

    lon, lat = transformer.transform(xv, yv)
    amsr_xarray.coords['lon'] = (('y', 'x'), lon)
    amsr_xarray.coords['lat'] = (('y', 'x'), lat)
    amsr_xarray.attrs['crs'] = 'epsg:4326'

    return amsr_xarray


def interpolate_gps_data(traj_collection: object,
                         year: int,
                         season: str,
                         interp_freq: str = "6H",
                         source_crs: str = "epsg:4326",
                         save: bool = False) -> object:
    """
    Interpolates GPS trajectory data at specified time intervals for a given year and season.

    This function interpolates GPS data of animal trajectories for the specified year and season, 
    generating a time series of interpolated points and segments. The interpolation is performed 
    using a specified frequency (e.g., 6 hours). Optionally, the results can be saved as GeoJSON files.

    Parameters:
    ----------
    traj_collection : movingpandas.TrajectoryCollection
        A collection of trajectories to be interpolated.
    year : int
        The year for which the data is being interpolated.
    season : str
        The season ('spring' or 'autumn') for which the data is being interpolated.
    interp_freq : str, optional
        The frequency for interpolation (default is "6H").
    source_crs : str, optional
        The coordinate reference system of the input data (default is "epsg:4326").
    save : bool, optional
        Whether to save the interpolated data as GeoJSON files (default is False).

    Returns:
    --------
    full_point_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the interpolated GPS points.
    full_segs_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the interpolated segments of the trajectories.
    """
    if season == "autumn":
        begin = pd.Timestamp(f'{year}-10-15-00-00-00')
        finish = pd.Timestamp(f'{year}-12-15-00-00-00')
    elif season == "spring":
        begin = pd.Timestamp(f'{year}-03-15-00-00-00')
        finish = pd.Timestamp(f'{year}-05-25-00-00-00')
    else:
        print(f"Invalid season - {season} - entered!")

    date_times = pd.date_range(begin, finish, freq=interp_freq, normalize=True).tz_convert(None)

    full_point_list = list()
    full_segs_list = list()
    print(f"{len(date_times)} dates to interpolate...")
    for index, date_time in enumerate(date_times):

        if index % 50 == 0:
            print(f"Completed {index} of {len(date_times)}")

        point_list = list()
        seg_list = list()

        for traj in traj_collection:
            traj_end = traj.get_end_time()
            traj_start = traj.get_start_time()

            #### POINTS ####
            interp_loc = traj.interpolate_position_at(date_time)
            if traj_start <= date_time <= traj_end:
                point_row = {"id": traj.id, "time": date_time, "geometry": interp_loc, "past_end": 0}
            elif date_time < traj_start:
                point_row = {"id": traj.id, "time": date_time, "geometry": interp_loc, "past_end": -1}
            elif date_time > traj_end:
                point_row = {"id": traj.id, "time": date_time, "geometry": interp_loc, "past_end": 1}

            point_list.append(point_row)

            #### SEGMENTS ####
            if date_time > traj_start:
                line_seg = traj.get_linestring_between(traj_start, date_time)
            else:
                line_seg = np.nan
            seg_row = {"id": traj.id, "time": date_time, "geometry": line_seg}
            seg_list.append(seg_row)

        point_gdf = gpd.GeoDataFrame(point_list, geometry="geometry", crs=source_crs)
        point_gdf = point_gdf.set_index("time")
        full_point_list.append(point_gdf)

        seg_gdf = gpd.GeoDataFrame(seg_list, geometry="geometry", crs=source_crs)
        seg_gdf = seg_gdf.set_index("time")
        full_segs_list.append(seg_gdf)

    full_point_gdf = pd.concat(full_point_list)
    full_segs_gdf = pd.concat(full_segs_list)

    if save:
        save_folder = f"{config.PROCESSED_DATA_FOLDER}/interp_gps/{year}"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        full_point_gdf.to_file(f"{save_folder}/{season}_point_{interp_freq}.geojson", driver="GeoJSON")
        full_segs_gdf.to_file(f"{save_folder}/{season}_segs_{interp_freq}.geojson", driver="GeoJSON")

    return full_point_gdf, full_segs_gdf


def get_year_season_df(full_gdf: object,
                       year: int,
                       season: str) -> object:
    """
    Filters a GeoDataFrame to extract data for a specific year and season.

    This function subsets a GeoDataFrame by year and season (spring or autumn), returning 
    only the relevant data for analysis.

    Parameters:
    ----------
    full_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing data for multiple years and seasons.
    year : int
        The year for which to filter the data.
    season : str
        The season ('spring' or 'autumn') for which to filter the data.

    Returns:
    --------
    sub_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the filtered data for the specified year and season.
    """
    season_dict = {"spring": [2, 3, 4, 5, 6], "autumn": [10, 11, 12]}

    sub_gdf = full_gdf[full_gdf.year == year]
    sub_gdf = sub_gdf[sub_gdf.index.month.isin(season_dict[season])]

    return sub_gdf


def get_traj_set(full_gdf: object,
                 year: int,
                 season: str = None,
                 plot: bool = False) -> object:
    """
    Extracts and returns a set of trajectories for a specified year and season.

    This function creates a `TrajectoryCollection` from a GeoDataFrame containing GPS tracking data, 
    filtered by the specified year and season. Optionally, the resulting collection can be plotted.

    Parameters:
    ----------
    full_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the GPS tracking data.
    year : int
        The year for which to extract the trajectories.
    season : str, optional
        The season ('spring' or 'autumn') for which to filter the data. If not specified, data from both seasons is used.
    plot : bool, optional
        Whether to plot the trajectories using the `movingpandas` plotting functionality (default is False).

    Returns:
    --------
    traj_collection : movingpandas.TrajectoryCollection
        A collection of trajectories filtered by the specified year and season.
    """
    season_dict = {"spring": [2, 3, 4, 5, 6], "autumn": [10, 11, 12]}

    sub_gdf = full_gdf[full_gdf.year == year]

    if season:
        sub_gdf = sub_gdf[sub_gdf.index.month.isin(season_dict[season])]
    else:
        season = "both seasons"

    print(f"Selecting trajectories for {year} ({season})")
    traj_collection = mpd.TrajectoryCollection(sub_gdf, "FieldID", t="t", crs="epsg:4326")
    print(traj_collection)
    if plot:
        traj_collection.hvplot(geo=True, hover_cols=["FieldID", "t"], tiles="OSM",
                               line_width=2, frame_width=500, frame_height=400)

    return traj_collection


def simplify_polygon(poly_gdf, projected_crs="epsg:32612", island_thresh=None, simplify_dist=0, buffer_dist=5000):
    """
    Simplifies and buffers a polygon, removing small islands and applying a buffer and simplification distance.

    This function reprojects a polygon to a specified projection, removes small interior islands (if requested),
    buffers the polygon by a given distance, simplifies it based on a distance threshold, and returns a simplified polygon.

    Parameters:
    ----------
    poly_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the polygon to be simplified.
    projected_crs : str, optional
        The Coordinate Reference System (CRS) to project the polygon to (default is "epsg:32612").
    island_thresh : float, optional
        The area threshold below which small interior polygons (islands) are removed (default is None).
    simplify_dist : float, optional
        The tolerance for simplifying the polygon (default is 0).
    buffer_dist : float, optional
        The distance in meters to buffer the polygon (default is 5000 meters).

    Returns:
    --------
    simple_poly_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the simplified polygon.
    """
    source_crs = poly_gdf.crs

    # Convert to area projection
    projected_poly = poly_gdf.to_crs(projected_crs)
    poly = projected_poly.geometry[0]

    # Remove islands
    if island_thresh:
        print(f"Removing islands with area < {island_thresh}")
        list_interiors = [interior for interior in poly.interiors if Polygon(interior).area > island_thresh]
        filtered_poly = Polygon(poly.exterior.coords, holes=list_interiors)
    else:
        print("Removing all islands")
        filtered_poly = max(poly, key=lambda a: a.area)

    # Erode and simplify polygon using specified thresholds/distances
    print(f"Buffering by {buffer_dist} meters and simplifying with threshold {simplify_dist}")
    simple_poly = filtered_poly.buffer(buffer_dist).simplify(simplify_dist)

    # May have created multipolygon with small extras, select only the biggest one
    if simple_poly.geom_type == "MultiPolygon":
        print("Converting multipolygon to simple poly")
        simple_poly = max(simple_poly, key=lambda a: a.area)

    # Convert to geopandas and reproject to original coord system
    simple_poly_gdf = gpd.GeoDataFrame(pd.DataFrame([{'geometry': simple_poly, 'id': 1}]), crs=projected_crs)
    simple_poly_gdf.to_crs(source_crs, inplace=True)

    return simple_poly_gdf


def get_land_points(point_gdf, land_poly):
    """
    Identifies whether each point in a GeoDataFrame is inside a land polygon.

    This function checks whether points from the `point_gdf` are contained within the `land_poly` polygon and 
    assigns a 'land' column indicating whether the point is land (1) or not (0).

    Parameters:
    ----------
    point_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing points to be checked.
    land_poly : geopandas.GeoDataFrame
        A GeoDataFrame containing land polygons.

    Returns:
    --------
    merge_gdf : geopandas.GeoDataFrame
        A GeoDataFrame with the original points and an additional 'land' column.
    """
    # Add column for water set to false
    land_poly.to_crs(4326, inplace=True)
    land_poly["land"] = 1

    # Join points with land polygon, check whether contained
    merge_gdf = gpd.sjoin(land_poly, point_gdf, how="right", predicate='contains')
    merge_gdf.drop(columns=["index_left"], inplace=True)
    merge_gdf["land"].fillna(0, inplace=True)

    return merge_gdf


def interpolate_sic(day_xr: xr.DataArray, method: str = "linear", dataset: str = "osisaf") -> xr.DataArray:
    """
    Interpolates sea ice concentration (SIC) data to fill missing values using a specified method.

    This function fills missing values (NaNs) in the sea ice concentration data from different datasets using
    the specified interpolation method.

    Parameters:
    ----------
    day_xr : xarray.DataArray
        The input xarray containing sea ice concentration data with missing values. The structure of the 
        data depends on the `dataset` parameter.
    method : str, optional
        The interpolation method to use. Options include 'linear', 'nearest', 'cubic', etc. Default is "linear".
    dataset : str, optional
        The dataset from which the sea ice concentration data is sourced. 
        Accepted values are 'osisaf', 'icenet', and 'amsr'. Default is 'osisaf'.

    Returns:
    --------
    interp_xr : xarray.DataArray
        The interpolated xarray with sea ice concentration data, where missing values have been filled.
    
    Raises:
    -------
    ValueError:
        If an invalid dataset is provided, a ValueError is raised with an appropriate message.
    """
    
    if dataset == "osisaf":
        day_xr = day_xr.astype(np.float32)
        xx, yy = np.meshgrid(np.arange(29), np.arange(33))
        valid = ~np.isnan(day_xr.ice_conc.values)

        x = xx[valid]
        y = yy[valid]

        x_interp = xx[~valid]
        y_interp = yy[~valid]

        values = day_xr.ice_conc.values[valid]
        interp_vals = interpolate.griddata((x, y), values, (x_interp, y_interp), method=method)

        interpolated_array = day_xr.ice_conc.values
        interpolated_array[~valid] = interp_vals

        interp_xr = day_xr.copy()
        interp_xr.ice_conc.values = interpolated_array

    elif dataset == "icenet":
        xx, yy = np.meshgrid(np.arange(29), np.arange(33))
        valid = ~np.isnan(day_xr.sic_mean.values)

        x = xx[valid]
        y = yy[valid]

        x_interp = xx[~valid]
        y_interp = yy[~valid]

        values = day_xr.sic_mean.values[valid]
        interp_vals = interpolate.griddata((x, y), values, (x_interp, y_interp), method=method)

        interpolated_array = day_xr.sic_mean.values
        interpolated_array[~valid] = interp_vals

        interp_xr = day_xr.copy()
        interp_xr.sic_mean.values = interpolated_array

    elif dataset == "amsr":
        xx, yy = np.meshgrid(np.arange(126), np.arange(135))
        valid = ~np.isnan(day_xr.z.values)

        x = xx[valid]
        y = yy[valid]

        x_interp = xx[~valid]
        y_interp = yy[~valid]

        values = day_xr.z.values[valid]
        interp_vals = interpolate.griddata((x, y), values, (x_interp, y_interp), method=method)

        interpolated_array = day_xr.z.values
        interpolated_array[~valid] = interp_vals

        interp_xr = day_xr.copy()
        interp_xr.z.values = interpolated_array

    else:
        raise ValueError(f"Invalid dataset '{dataset}' input. Should be either 'osisaf', 'icenet', or 'amsr'.")

    return interp_xr