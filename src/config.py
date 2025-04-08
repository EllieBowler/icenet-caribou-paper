import os
from pyproj import Transformer
import numpy as np
import cartopy.io.img_tiles as cimgt
import matplotlib as mpl
import geopandas as gpd
import cartopy.crs as ccrs

constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)


###############################
#### Paths to example data ####
###############################

# Path to demonstration caribou data csv. Please note - this data is not publicly available, data requests should be made to the Government of Nunavut.
PATH_TO_CSV = PROJECT_PATH + "/data/tracking_data/example_migration_data_format.csv"

# Paths to static data sources - these should be downloaded according to readme instructions
PATH_TO_OSISAF = PROJECT_PATH + "/data/osisaf"
PATH_TO_ICENET = PROJECT_PATH + "/data/forecasts"
PATH_TO_AMSR2 = PROJECT_PATH + "/data/amsr2"
PATH_TO_OSM = PROJECT_PATH + "/data/osm/land-polygons-split-4326/land_polygons.shp"

# Path to post-processed VI-buffer and gulf area for plotting
ADJUSTED_VI_POLY = PROJECT_PATH + "/data/polygons/final_vi_buffer.shp"

# Paths to land and coast masks
LAND_MASK_PATH = PROJECT_PATH + "/data/masks/land_mask.npy"
OSISAF_COAST_CELLS = PROJECT_PATH + "/data/masks/osisaf_coastal_gridcells.nc"
AMSR_COAST_CELLS = PROJECT_PATH + "/data/masks/amsr_coastal_gridcells.nc"

# Paths to results files to make plots for paper
MAE_COMPARE_ARCTIC = PROJECT_PATH + "/data/results/seas_compare/whole_arctic_bias_correct.csv"
MAE_COMPARE_GULF = PROJECT_PATH + "/data/results/seas_compare/coronation_gulf_bias_correct.csv"

PERCENT_MIG_MAPPING = {"osisaf": PROJECT_PATH + "/data/results/percent_migrate/osisaf_nconsec_1_smooth_40_mapping.csv", 
                       "amsr": PROJECT_PATH + "/data/results/percent_migrate/amsr_nconsec_1_smooth_50_mapping.csv"}

PROCESSED_SIC_DATA_FILES = {"osisaf": f"{PROJECT_PATH}/data/results/sic_observations/osisaf_sic_data.csv",
                            "amsr": f"{PROJECT_PATH}/data/results/sic_observations/amsr_sic_data.csv"}


############################################
#### Paths defined for main src scripts ####
############################################

PATH_TO_MIG_START_DF = PROJECT_PATH + "/data/migration_start/final_vi_buffer_mig_start.geojson"

# Rectangular Area Of Interest for Victoria Island and wider gulf
VI_AOI_PATH = PROJECT_PATH + "/data/polygons/victoria_island_aoi.geojson"
GULF_AOI_PATH = PROJECT_PATH + "/data/polygons/coronation_gulf_aoi.geojson"

# Path to post-processed VI-buffer and gulf area for plotting
ADJUSTED_VI_POLY = PROJECT_PATH + "/data/polygons/final_vi_buffer.geojson"
GULF_COASTLINE_PATH = PROJECT_PATH + "/data/polygons/coronation_gulf_land_poly.geojson"

# path to migration start date csv
PATH_TO_MIG_START_CSV = PROJECT_PATH + "/processed/migration_start/final_vi_buffer_full_mig_start_clean.csv"

PERCENT_MIG_FILES = {"osisaf": PROJECT_PATH + "/processed/osisaf_nconsec_1_smooth_40_mapping.csv",
                     "amsr": PROJECT_PATH + "/processed/amsr_nconsec_1_smooth_50_mapping.csv"}

BULL_IDS = ["DU-145-18", "DU-143-18", "DU-168-18"] # FieldID's of the bulls in the dataset
SOURCE_CRS = "epsg:4326"  # reference system of the caribou tracking data
INTERP_FREQ = "6H"  # interpolation frequency for tracking data
TIME_COL = "FixDateTime"  # time column name
SEASON_LIST = ["spring", "autumn"]

# buffer/simplify parameters for VI land polygon
BUFFER_DIST = 5000
SIMP_DIST = 100

# pre set paths to data save folders
PATH_TO_POLYGONS = PROJECT_PATH + "/data/polygons"
TEMP_IMG_FOLDER = PROJECT_PATH + "/TEMP_IMG_FOLDER"
VIS_SAVE_FOLDER = PROJECT_PATH + "/visualisations_dev2"
PROCESSED_DATA_FOLDER = PROJECT_PATH + "/processed"
SIC_PLOTS_FOLDER = PROJECT_PATH + "/plots/sic_plots"
TRAJ_PLOTS_FOLDER = PROJECT_PATH + "/plots/traj_plots_test"
PRED_PLOT_FOLDER = PROJECT_PATH + "/plots/predictions"

# processed sic data files used in paper
SIC_DATA_FILES = {"osisaf": f"{PROCESSED_DATA_FOLDER}/sic_observations/osisaf_sic_data_45_clean.csv",
                  "amsr": f"{PROCESSED_DATA_FOLDER}/sic_observations/amsr_sic_data_45_clean.csv"}
SIC_COL_REF = {"osisaf": "ice_conc", "amsr": "z",
               "icenet_1w": "sic_mean", "icenet_2w": "sic_mean", "icenet_3w": "sic_mean"}


###############################
### BOUNDS OF PLOTTING DATA ###
###############################

central_lat = 68
central_lon = -110   
lat_span = 5  # degrees
lon_span = 16  # degrees

# Determine crop corners
lon_left = central_lon - lon_span / 2
lon_right = central_lon + lon_span / 2
lat_upper = central_lat + lat_span / 2
lat_lower = central_lat - lat_span / 2

lats = [lat_upper, lat_upper, lat_lower, lat_lower]
lons = [lon_left, lon_right, lon_left, lon_right]

osisaf_crs = ccrs.LambertAzimuthalEqualArea(0, 90)
amsr_crs = ccrs.epsg(3411)

# Transform from lat/lon to EASE2 (EPSG:6932)
osisaf_transformer = Transformer.from_crs('epsg:4326', osisaf_crs)
amsr_transformer = Transformer.from_crs('epsg:4326', amsr_crs)

osisaf_x, osisaf_y = osisaf_transformer.transform(lats, lons)
amsr_x, amsr_y = amsr_transformer.transform(lats, lons)

PROJECTION = ccrs.LambertAzimuthalEqualArea(central_lon, 90)
PLOT_EXTENT = [lon_left+1, lon_right-1, lat_lower+1, lat_upper-1]

c_map = mpl.cm.get_cmap("viridis").copy()
bg_map = cimgt.Stamen('terrain-background')
bg_map = None
land_gdf = gpd.read_file(GULF_COASTLINE_PATH)
fill_land = True
caribou_colour = "red"

osisaf_plot_config = {
    "caribou_colour": caribou_colour,
    "c_map": c_map,
    "bg_map": bg_map,
    "coastline": land_gdf,
    "fill_land": fill_land,
    "crop_x": [np.min(osisaf_x), np.max(osisaf_x)],
    "crop_y": [np.max(osisaf_y), np.min(osisaf_y)],
    "central_lon": central_lon,
    "full_name": "OSI-SAF",
    "save_name": "osisaf",
    "vmin": 0,
    "vmax": 100,
    "transform": osisaf_crs
}

icenet_plot_config = {
    "caribou_colour": caribou_colour,
    "c_map": c_map,
    "bg_map": bg_map,
    "coastline": land_gdf,
    "fill_land": fill_land,
    "crop_x": [np.min(osisaf_x), np.max(osisaf_x)],
    "crop_y": [np.max(osisaf_y), np.min(osisaf_y)],
    "central_lon": central_lon,
    "full_name": "IceNet",
    "save_name": "icenet",
    "vmin": 0,
    "vmax": 1,
}

amsr_plot_config = {
    "caribou_colour": caribou_colour,
    "c_map": c_map,
    "bg_map": bg_map,
    "coastline": land_gdf,
    "fill_land": fill_land,
    "crop_x": [np.min(amsr_x), np.max(amsr_x)],
    "crop_y": [np.min(amsr_y), np.max(amsr_y)],
    "central_lon": central_lon,
    "full_name": "AMSR-2",
    "save_name": "amsr2",
    "vmin": 0,
    "vmax": 100,
    "transform": amsr_crs,
}