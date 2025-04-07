import pandas as pd
import os
import xarray as xr
import numpy as np

import sys
sys.path.append("../../")
from src import config
from src.data.load import load_osisaf_year
from src.data.process import interpolate_sic
from src.analysis.metric_utils import get_matching_index_array


def get_crossing_box(init_date, data_source):
    box_and_whisker = {"low_whisk": 10, "low_box": 25, "up_box": 75, "up_whisk": 90}

    init_date = pd.to_datetime(init_date, format="%Y-%m-%d")
    print_init_date = init_date.strftime("%d %b %Y")

    if data_source == "osisaf":
        print("Processing osisaf data...")
        mapping_df = pd.read_csv(config.PERCENT_MIG_FILES["osisaf"], index_col=0)
        _, _, n_consec, _, smoothing_level, _ = os.path.basename(config.PERCENT_MIG_FILES["osisaf"]).split("_")
        n_consec = int(n_consec)
        smoothing_level = int(smoothing_level)
        print(n_consec, smoothing_level)
        xarray_year = load_osisaf_year(f"{config.PATH_TO_OSISAF}/{init_date.year}.nc")
        xarray_gulf = xarray_year.sel(
            time=slice(init_date, f"{init_date.year}-12-31")).sel(
            xc=slice(*config.osisaf_plot_config['crop_x']),
            yc=slice(*config.osisaf_plot_config['crop_y']))
        print("Interpolating at nan coastal gridcells...")
        xarray_interp = xarray_gulf.groupby("time").apply(interpolate_sic, dataset="osisaf")
        xarray_smooth = xarray_interp.rolling(time=smoothing_level, center=True, min_periods=1).mean()

        coast_mask = xr.open_mfdataset(f"{config.OSISAF_CROSSING_CELLS}")
        mask = coast_mask.sel(time=f"{init_date.year}-01-01")
        mask_gulf = mask.sel(xc=slice(*config.osisaf_plot_config['crop_x']),
                             yc=slice(*config.osisaf_plot_config['crop_y']))

    elif data_source == "amsr":
        print("Processing amsr data...")
        mapping_df = pd.read_csv(config.PERCENT_MIG_FILES["amsr"], index_col=0)

        _, _, n_consec, _, smoothing_level, _ = os.path.basename(config.PERCENT_MIG_FILES["amsr"]).split("_")
        n_consec = int(n_consec)
        smoothing_level = int(smoothing_level)
        print(n_consec, smoothing_level)

        # smoothing_level = 50
        # n_consec = 1
        xarray_year = xr.open_dataset(f"{config.PATH_TO_AMSR2}/{test_year}.nc")
        xarray_gulf = xarray_year.sel(
            time=slice(init_date, f"{test_year}-12-31")).sel(
            x=slice(*config.amsr_plot_config['crop_x']),
            y=slice(*config.amsr_plot_config['crop_y']))
        print("Interpolating at nan coastal gridcells...")
        xarray_interp = xarray_gulf.groupby("time").apply(interpolate_sic, dataset="amsr")
        xarray_interp = xarray_interp.drop_vars("polar_stereographic")
        #         xarray_interp = xarray_gulf.drop_vars("polar_stereographic")
        xarray_smooth = xarray_interp.rolling(time=smoothing_level, center=True, min_periods=1).mean()

        coast_mask = xr.open_mfdataset(f"{config.AMSR_CROSSING_CELLS}")
        mask = coast_mask.sel(time=f"{test_year}-01-01")
        mask_gulf = mask.sel(x=slice(*config.amsr_plot_config['crop_x']),
                             y=slice(*config.amsr_plot_config['crop_y']))

    elif data_source == "icenet":
        print("Processing IceNet data...")
        mapping_df = pd.read_csv(config.PERCENT_MIG_FILES["osisaf"], index_col=0)

        _, _, n_consec, _, smoothing_level, _ = os.path.basename(config.PERCENT_MIG_FILES["osisaf"]).split("_")
        n_consec = int(n_consec)
        smoothing_level = int(smoothing_level)
        print(n_consec, smoothing_level)

        xarray_year = xr.open_dataset(f"{config.PATH_TO_ICENET}/icenet_coronation_gulf_{test_year}.nc")
        xarray_gulf = xarray_year.sel(
            time=init_date).sel(
            xc=slice(*config.icenet_plot_config['crop_x']),
            yc=slice(*config.icenet_plot_config['crop_y']))
        xarray_interp = xarray_gulf.groupby("leadtime").apply(interpolate_sic, dataset="icenet")
        xarray_smooth = xarray_interp.rolling(leadtime=smoothing_level, center=True, min_periods=1).mean()

        coast_mask = xr.open_mfdataset(f"{config.OSISAF_CROSSING_CELLS}")
        mask = coast_mask.sel(time=f"{init_date.year}-01-01")
        mask_gulf = mask.sel(xc=slice(*config.osisaf_plot_config['crop_x']),
                             yc=slice(*config.osisaf_plot_config['crop_y']))

    full_results = []
    for key, chosen_percent in box_and_whisker.items():

        chosen_sic = mapping_df[mapping_df.percent_interp == chosen_percent].sic_interp.values[0]
        if data_source == "osisaf":
            matching_array = np.apply_along_axis(get_matching_index_array, 0,
                                                 xarray_smooth.ice_conc.to_numpy(),
                                                 chosen_sic, n_consec)
            data = xarray_gulf.copy()
            data["matching_gridcells"] = (("yc", "xc"), matching_array)
            data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")

            result = data_expanded["matching_gridcells"].where(mask_gulf != 0).ice_conc

        elif data_source == "amsr":
            matching_array = np.apply_along_axis(get_matching_index_array, 0,
                                                 xarray_smooth.z.to_numpy() / 100,
                                                 chosen_sic, n_consec)
            data = xarray_gulf.copy()
            data["matching_gridcells"] = (("y", "x"), matching_array)
            data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")

            result = data_expanded["matching_gridcells"].where(mask_gulf != 0).z

        elif data_source == "icenet":
            matching_array = np.apply_along_axis(get_matching_index_array, 0,
                                                 xarray_smooth.sic_mean.to_numpy(),
                                                 chosen_sic, n_consec)

            data = xarray_gulf.copy()
            data["matching_gridcells"] = (("yc", "xc"), matching_array)
            data_expanded = data.assign_coords(year=init_date.year).expand_dims("year")

            result = data_expanded["matching_gridcells"].where(mask_gulf != 0).ice_conc

        all_days = result.values[~np.isnan(result.values)]
        if len(all_days) == 0:
            result_df = pd.DataFrame(data={"pred_day_num": [0]}, dtype=np.int8)
            result_df["init_date"] = init_date
            result_df["date"] = init_date
            result_df["year"] = init_date.year
            result_df["label"] = "predicted"
            result_df["plot_label"] = key
            result_df["percent"] = chosen_percent
            result_df["sic_thresh"] = chosen_sic
            result_df["data_source"] = data_source

            full_results.append(result_df)
        else:
            result_df = pd.DataFrame(data={"pred_day_num": all_days}, dtype=np.int8)
            result_df["init_date"] = init_date
            result_df["date"] = result_df.apply(lambda x: x.init_date + np.timedelta64(x.pred_day_num, "D"), axis=1)
            result_df["year"] = init_date.year
            result_df["label"] = "predicted"
            result_df["plot_label"] = key
            result_df["percent"] = chosen_percent
            result_df["sic_thresh"] = chosen_sic
            result_df["data_source"] = data_source

            full_results.append(result_df)

    return pd.concat(full_results)
