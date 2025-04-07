"""
Script for downloading, processing, and saving AMSR2 sea ice concentration data.

This script:
1. Downloads daily AMSR2 sea ice concentration NetCDF files for a specified year range.
2. Concatenates the daily files into a single dataset for each year.
3. Interpolates missing days in the dataset.
4. Saves both the interpolated daily files and a full-year dataset.

Command-line Arguments:
-----------------------
--start_year : int (default: 2015)
    The starting year for downloading data.
--end_year : int (default: 2023)
    The ending year for downloading data.

Process:
--------
- Retrieves file URLs from the Bremen sea ice data repository.
- Downloads missing files into a local directory.
- Loads all daily NetCDF files for each year and combines them into a single dataset.
- Interpolates missing dates.
- Saves the interpolated daily files separately.
- Exports a full-year NetCDF file.

Example Usage:
--------------
Run the script for AMSR2 data from 2018 to 2022:
    python download_amsr2.py --start_year 2018 --end_year 2022
"""

import argparse
import shutil
import requests
from bs4 import BeautifulSoup
import glob
import os
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np

import sys
sys.path.append("../../")
from src import config


def get_url_paths(url, ext='', params={}):
    """
    Retrieve a list of file paths from a given URL based on a specified file extension.

    Parameters:
    ----------
    url : str
        The base URL to fetch the HTML content from.
    ext : str, optional
        The file extension to filter the links (e.g., '.pdf', '.txt'). Default is an empty string, 
        which retrieves all links.
    params : dict, optional
        Dictionary of query parameters to be passed in the HTTP request. Default is an empty dictionary.

    Returns:
    -------
    list of str
        A list of full URLs that match the specified file extension.
    
    Raises:
    ------
    requests.exceptions.HTTPError
        If the HTTP request fails.

    Example:
    -------
    >>> get_url_paths("https://example.com/files", ext=".pdf")
    ['https://example.com/files/document1.pdf', 'https://example.com/files/document2.pdf']
    """
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent
    


def download_file(url, save_dir):
    """
    Download a file from a given URL and save it to a specified directory.

    Parameters:
    ----------
    url : str
        The URL of the file to be downloaded.
    save_dir : str
        The directory where the downloaded file should be saved.

    Returns:
    -------
    None

    Notes:
    ------
    - If the file already exists in the specified directory, it is not downloaded again.
    - The function prints the file path if the download is successful.

    Example:
    -------
    >>> download_file("https://example.com/sample.pdf", "/path/to/save")
    /path/to/save/sample.pdf
    """
    local_filename = url.split('/')[-1]

    if os.path.exists(save_dir + "/" + local_filename):
        print(f"{save_dir}/{local_filename} already downloaded!")
        pass
    else:
        with requests.get(url, stream=True) as r:
            with open(save_dir + "/" + local_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

        print(save_dir + "/" + local_filename)
    return


if __name__ == "__main__":
    # Define commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2015)
    parser.add_argument("--end_year", type=int, default=2023)
    args = parser.parse_args()

    for year in range(args.start_year, args.end_year+1):
        print("-" * 20)
        print(f"Saving files for {year}...")
        all_files = get_url_paths(f"https://seaice.uni-bremen.de/data/amsr2/asi_daygrid_swath/n6250/netcdf/{year}/",
                                  ".nc")
        for file in all_files:
            download_file(file, config.PATH_TO_AMSR2)

        print(f"Concatenating files for {year}...")
        daily_files = glob.glob(f'{config.PATH_TO_AMSR2}/asi-AMSR2-n6250-{year}*-v5.4.nc')
        all_daily = []

        for file in daily_files:
            _, _, _, date, _ = os.path.basename(file).split("-")

            f = xr.open_dataset(file)
            f["time"] = pd.to_datetime(date)
            all_daily.append(f)

        year_file = xr.concat(all_daily, "time")

        ## interpolate missing days
        print("Interpolating missing dates:")
        dates_obs = year_file.time.values
        if year == 2023:
            dates_all = pd.date_range(dt.date(year, 1, 1), dt.date.today())
        else:
            dates_all = pd.date_range(dt.date(year, 1, 1), dt.date(year, 12, 31))
        missing_dates = dates_all.difference(dates_obs)
        print(missing_dates)

        if len(missing_dates) == 0:
            print(f"There are no missing dates in {year}!")
        else:
            interp_data = year_file.interp(time=missing_dates)
            interp_data["polar_stereographic"] = np.array(b'', dtype='|S1')

            # Save individual day files (for quicker plotting of AMSR2)
            for date in interp_data.time:
                print_date = pd.to_datetime(str(date.values)).strftime('%Y%m%d')
                print(f"Saving day file to {config.PATH_TO_AMSR2}/asi-AMSR2-n6250-{print_date}-v5.4.nc")
                interp_data.sel(time=date).to_netcdf(f"{config.PATH_TO_AMSR2}/asi-AMSR2-n6250-{print_date}-v5.4.nc")

            year_file = xr.concat([year_file,
                                   interp_data],
                                  dim="time")
            year_file = year_file.sortby("time")

        print(f"Saving full year file as {config.PATH_TO_AMSR2}/{year}.nc")
        year_file.to_netcdf(f"{config.PATH_TO_AMSR2}/{year}.nc")


