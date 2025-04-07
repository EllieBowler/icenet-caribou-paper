"""
Script to interpolate zero entry for 9/11/2022

This script loads an OSISAF dataset, removes erroneous zero records for 9/11/2022,
performs interpolation to estimate missing values, and saves the corrected dataset.
"""

import pandas as pd
import datetime as dt
import xarray as xr

import sys
sys.path.append("../../")
from src import config


if __name__ == "__main__":

    interp_date = pd.Timestamp(dt.date(2022, 11, 9))

    print("Loading osisaf for 2022...")
    da = xr.open_mfdataset(config.PATH_TO_OSISAF + "/2022_error_221109.nc")
    
    # Remove missing record for specified date 
    print("Delete and interpolate mis-record on 9/11/2022")
    da = da.drop_sel(time=interp_date)
    # Interpolate missing date and concatenate it back with dataset
    da = xr.concat([da,
                    da.interp(time=interp_date)],
                   dim="time")
    da = da.sortby("time") # ensure data remains sorted by time
    
    # Save the corrected dataset
    print(f"Save to {config.PATH_TO_OSISAF}/2022.nc")
    da.to_netcdf(path=config.PATH_TO_OSISAF + "/2022.nc", mode="w")

