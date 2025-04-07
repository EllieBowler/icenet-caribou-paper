"""
Script to extract Sea Ice Concentration (SIC) timeseries at migration start points.

This script calculates the SIC timeseries for migration start points using different data sources (OSISAF or AMSR2).
It loads migration start points, processes the data, and outputs the SIC data for those points over a specified sample range.

The script supports different data sources:
- OSISAF: Sea Ice Concentration data from the OSISAF dataset.
- AMSR2: Sea Ice Concentration data from the AMSR2 satellite data.

Outputs:
- A CSV file containing the SIC timeseries data for each migration start point.
"""

import geopandas as gpd
import argparse

import sys
sys.path.append("../../")
from src import config
from src.data.load import load_migration_start_points_gdf
from src.analysis.analysis_utils import get_osisaf_migration_timeseries, get_amsr2_migration_timeseries



if __name__ == "__main__":
    # Define commandline arguments
    parser = argparse.ArgumentParser(description="Extract SIC timeseries at migration start points.")
    parser.add_argument("--sample_range", type=int, default=45, help="Number of days before/after migration start to sample.")
    parser.add_argument("--data_source", type=str, choices=["osisaf", "amsr"], default="osisaf", 
                        help="Data source to use for SIC timeseries. Choose from 'osisaf' or 'amsr'.")
    parser.add_argument("--save_plots", type=bool, default=False, help="Whether to save plots of the SIC timeseries.")
    args = parser.parse_args()

    # Load migration start data (GeoDataFrame containing migration start points)
    crossing_gdf = load_migration_start_points_gdf(config.PATH_TO_MIG_START_DF)

    # Process the SIC timeseries based on the chosen data source
    if args.data_source == "osisaf":
        print("Processing OSISAF SIC timeseries data...")
        # Get the SIC timeseries from the OSISAF dataset for each migration start point
        osisaf_df = get_osisaf_migration_timeseries(crossing_gdf, args.sample_range)
        print(f"Saving OSISAF SIC data to {config.PROCESSED_DATA_FOLDER}/sic_observations/osisaf_sic_data_{args.sample_range}_clean.csv")
        osisaf_df.to_csv(f"{config.PROCESSED_DATA_FOLDER}/sic_observations/osisaf_sic_data_{args.sample_range}_clean.csv")
        
    elif args.data_source == "amsr":
        print("Processing AMSR2 SIC timeseries data...")
        # Get the SIC timeseries from the AMSR2 dataset for each migration start point
        amsr_df = get_amsr2_migration_timeseries(crossing_gdf, args.sample_range)
        print(f"Saving AMSR2 SIC data to {config.PROCESSED_DATA_FOLDER}/sic_observations/amsr_sic_data_{args.sample_range}_clean.csv")
        amsr_df.to_csv(f"{config.PROCESSED_DATA_FOLDER}/sic_observations/amsr_sic_data_{args.sample_range}_clean.csv")

    else:
        print(f"Error: Invalid data source '{args.data_source}' provided. Please choose 'osisaf' or 'amsr.")






