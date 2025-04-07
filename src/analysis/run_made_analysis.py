"""
Run metric analysis to determine the best hyperparameters (SIC threshold, consecutive days, smoothing level)
for predicting migration dates based on observed sea ice concentration (SIC) time series.

The results are saved as a CSV file containing the metrics for each combination of parameters.
"""

import numpy as np
import pandas as pd
import argparse
import sys

# Append the parent directory to the system path to allow importing from src
sys.path.append("../../")
from src import config
from src.analysis.metric_utils import get_matching_index, calculate_metric
from src.data.load import load_train_test_csv

if __name__ == "__main__":
    # Define commandline arguments for specifying the data source
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, help="Data source for SIC time series.")
    args = parser.parse_args()

    # Load training and test data for the given data source, filtering for female-only and applying other filters
    df, df_test = load_train_test_csv(args.data_source, female_only=True, train_end=2019, day_diff_lim=3)

    # List to store results of the analysis
    result_df = []

    # Loop through different combinations of hyperparameters (SIC threshold, consecutive days, smoothing level)
    for sic_thresh in np.round(np.arange(0.7, 0.99, 0.05), 2):
        for n_consec in [1, 3, 5, 10]:
            for smooth_level in [1, 5, 10, 20, 30, 40, 50, 60]:

                # Print current parameter combination for tracking
                print("sic_thresh: {:.2f}, n_consec: {}, smoothing: {}".format(sic_thresh, n_consec, smooth_level))

                # Lists to store observed and predicted migration dates
                cross_list = []
                pred_list = []

                # Iterate through each observation (grouped by 'obs_id')
                for obs_id, obs_df in df.groupby("obs_id"):

                    # Get the observed migration date, assuming only one migration date per observation
                    cross_date = obs_df.mig_date.unique()
                    if len(cross_date) == 1:
                        cross_date = cross_date[0]
                    else:
                        print("Warning: Multiple migration dates found for observation, should be only one.")

                    # Get the SIC time series for the current observation
                    sic_timeseries = obs_df[config.SIC_COL_REF[args.data_source]]

                    # Get the predicted migration date by applying the 'get_matching_index' function
                    pred_idx = get_matching_index(sic_timeseries, sic_thresh, n_consec, smooth_level)

                    # If a valid index is found (i.e., not NaN), store the observed and predicted dates
                    if not np.isnan(pred_idx):
                        pred_date = obs_df.index[pred_idx]
                        cross_list.append(cross_date)
                        pred_list.append(pred_date)

                # Create a DataFrame with the observed and predicted migration dates
                final_df = pd.DataFrame(list(zip(cross_list, pred_list)), columns=["observed", "predicted"])

                # Calculate the Mean Absolute Date Error (MADE) for the current hyperparameters
                (made, made_sem) = calculate_metric(final_df, metric_name="made")

                # Store the results of this parameter combination in the result list
                row = {"sic_thresh": sic_thresh, "n_consec": n_consec, "smooth_level": smooth_level,
                       "n_valid": len(final_df), "made": made}
                result_df.append(row)

                # Print the current MADE value and the number of valid predictions
                print("MADE: {:.2f}, (n_valid: {})".format(made, len(final_df)))

    # Convert the result list into a DataFrame and save it as a CSV file
    result_df = pd.DataFrame(result_df)
    print(f"Saving results to {config.PROCESSED_DATA_FOLDER}/{args.data_source}_made_search.csv")
    result_df.to_csv(f"{config.PROCESSED_DATA_FOLDER}/{args.data_source}_made_search.csv")