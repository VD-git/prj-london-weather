import argparse
import json
import pandas as pd
import os
import nannyml as nml

def multivariate_analysis(cols, date_col, reference_df, analysis_df):

    features_column_names = cols.split(",")

    mv_calculator = nml.DataReconstructionDriftCalculator(
        column_names = features_column_names,
        timestamp_column_name = date_col,
        chunk_period = 'm',
        threshold = nml.thresholds.ConstantThreshold(lower=0, upper=1.5)
    )

    mv_calculator.fit(reference_df)
    mv_results = mv_calculator.calculate(analysis_df)

    mv_results.filter(period = 'analysis').to_df().to_csv('output/results_multivariate.csv', index = False)

    mv_figure = mv_results.filter(period = 'analysis').plot()
    mv_figure.write_image('output/multivariate.png')

    return mv_results

def univariate_analysis(cols, date_col, reference_df, analysis_df):

    features_column_names = cols.split(",")
    features_column_names = [col for col in features_column_names if col not in [date_col]]
    
    uv_calculator = nml.UnivariateDriftCalculator(
        continuous_methods = ['wasserstein', 'hellinger'],
        categorical_methods = ['jensen_shannon', 'l_infinity', 'chi2'],
        column_names = features_column_names,
        timestamp_column_name = date_col,
        chunk_period = 'm'
    )

    uv_calculator.fit(reference_df)
    uv_results = uv_calculator.calculate(analysis_df)

    uv_results.filter(period = 'analysis').to_df().to_csv('output/results_univariate.csv', index = False)
    
    uv_figure = uv_results.filter(period = 'analysis').plot()
    uv_figure.write_image('output/univariate.png')

    return uv_results

def go(args):

    reference = pd.read_csv(os.path.join("data", args.reference_csv))
    analysis = pd.read_csv(os.path.join("data", args.analysis_csv))
    reference[args.date_col] = pd.to_datetime(reference[args.date_col], format='%Y%m%d').astype(str)
    analysis[args.date_col] = pd.to_datetime(analysis[args.date_col], format='%Y%m%d').astype(str)

    mv_results = multivariate_analysis(args.cols, args.date_col, reference, analysis)
    uv_results = univariate_analysis(args.cols, args.date_col, reference, analysis)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script For Model Training"
    )

    parser.add_argument(
        "--cols",
        type=str,
        help="Columns of the archive",
        required=False,
        default="cloud_cover,sunshine,global_radiation,max_temp,min_temp,precipitation,pressure,snow_depth,date"
    )

    parser.add_argument(
        "--date_col",
        type=str,
        help="Date column",
        required=False,
        default="date"
    )

    parser.add_argument(
        "--reference_csv",
        type=str,
        help="Reference csv",
        required=False,
        default="old_london_weather.csv"
    )

    parser.add_argument(
        "--analysis_csv",
        type=str,
        help="Analysis csv",
        required=False,
        default="new_london_weather.csv"
    )

    args = parser.parse_args()

    go(args)