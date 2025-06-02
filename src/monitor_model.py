import argparse
import json
import pandas as pd
import os
import nannyml as nml
import pickle

def multivariate_analysis(cols, date_col, reference_df, analysis_df):
    """
    Function to measure multivariate drift
    Args:
        cols: list
            Columns that will be used on the dataframe given
        date_col: str
            Name of the column referencing timestamp for the drift
        reference_df: pd.DataFrame
            Reference dataset
        analysis_df: pd.DataFrame
            Analysis dataset
    Outputs:
        mv_calculator: nannyml.drift.multivariate.data_reconstruction.calculator.DataReconstructionDriftCalculator
            Class with the role of measuring multivariate drift
        mv_results: nannyml.drift.multivariate.data_reconstruction.result.Result
            Dataset with the results of the analysis period given
    """

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

    return mv_calculator, mv_results

def univariate_analysis(cols, date_col, reference_df, analysis_df):
    """
    Function to measure univariate drift.
    Common to be used after the multivariate drift detection
    Args:
        cols: list
            Columns that will be used on the dataframe given
        date_col: str
            Name of the column referencing timestamp for the drift
        reference_df: pd.DataFrame
            Reference dataset
        analysis_df: pd.DataFrame
            Analysis dataset
    Outputs:
        uv_calculator: nannyml.drift.univariate.calculator.UnivariateDriftCalculator
            Class with the role of measuring multivariate drift
        uv_results: nannyml.drift.univariate.result.Result
            Dataset with the results of the analysis period given
    """

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

    return uv_calculator, uv_results

def rank_analysis(uv_results, method):
    """
    Function to rank features by the number of the times a drift was detected.
    Args:
        uv_results: nannyml.drift.univariate.result.Result
            Dataset with the results of the analysis period given
        method: str
            Method chosen to measure the univariate shift (e.g. wasserstein for regression, chi2 for classification
    Ouputs
        alert_count_ranker_results: csv
            Csv with the ranking
    """

    alert_count_ranker = nml.AlertCountRanker()
    alert_count_ranker_results = alert_count_ranker.rank(
        uv_results.filter(methods = method).filter(period = 'analysis'),
    	only_drifting = False # Case True, it will only apper the features with detected drift, usually want to see all of them
    )
    
    return alert_count_ranker_results.to_csv('output/feature_drift_ranker.csv', index = False)

def drift_analysis(uv_results, reference, analysis, model_path, y_label, date_col, method):
    """
    Function to measure correlated drift
    Gives more precision with p_value to detect a true drift and avoid false alerts, predictions and true values are needed
    Args:
        uv_results: nannyml.drift.univariate.result.Result
            Dataset with the results of the analysis period given, output of `univariate_analysis()`
        reference: pd.DataFrame
            Reference dataset
        analysis: pd.DataFrame
            Analysis dataset
        model_path: str
            Path of the model
        y_label: str
            Name of the column with the `y_true`, the true value
        date_col: str
            Name of the column referencing timestamp for the drift
        method: str
            Method chosen to measure the univariate shift (e.g. wasserstein for regression, chi2 for classification
    Outputs:
        realized_perf_results: csv
            Csv with true performance results
        realized_perf_figure: png
            Png plot with the performance results
        correlation_ranked_results: csv
            Csv with the correlation of the p-value of drifts detected
    """

    # For performance calculator it is needed the predicted and real labels
    # Performance calculator is needed for the correlation rank
    y_label_pred = y_label + '_pred'

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(analysis.drop(['date', 'mean_temp'], axis = 1))
    analysis['mean_temp_pred'] = y_pred
    
    y_pred = model.predict(reference.drop(['date', 'mean_temp'], axis = 1))
    reference['mean_temp_pred'] = y_pred

    realized_calculator = nml.PerformanceCalculator(
        y_pred=y_label_pred,
        y_true=y_label,
        timestamp_column_name=date_col,
        problem_type='regression',
        metrics=['rmse'],
        chunk_period='m'
    )

    realized_calculator.fit(reference)
    realized_perf_results = realized_calculator.calculate(analysis)
    realized_perf_results.to_df().to_csv('output/performance_results.csv', index = False)
    realized_perf_figure = realized_perf_results.plot()
    realized_perf_figure.write_image('output/performance.png')

    # In order to measure the correlation it is needed to chose one method of covariate shift like wasserstein
    correlation_ranker = nml.CorrelationRanker()
    correlation_ranker.fit(
    	realized_perf_results.filter(period = 'analysis')
    )
    correlation_ranked_results = correlation_ranker.rank(uv_results.filter(methods = method).filter(period = 'analysis'), realized_perf_results.filter(period = 'analysis'))
    correlation_ranked_results.to_csv('output/feature_drift_correlation_ranker.csv', index = False)
    

def go(args):

    reference = pd.read_csv(os.path.join("data", args.reference_csv))
    analysis = pd.read_csv(os.path.join("data", args.analysis_csv))
    reference[args.date_col] = pd.to_datetime(reference[args.date_col], format='%Y%m%d').astype(str)
    analysis[args.date_col] = pd.to_datetime(analysis[args.date_col], format='%Y%m%d').astype(str)

    mv_calculator, mv_results = multivariate_analysis(args.cols, args.date_col, reference, analysis)
    uv_calculator, uv_results = univariate_analysis(args.cols, args.date_col, reference, analysis)

    rank_analysis(uv_results, args.method)
    drift_analysis(
        uv_results = uv_results,
        reference = reference,
        analysis = analysis,
        model_path = "output/model.pkl",
        y_label = args.y_label,
        date_col = args.date_col,
        method = args.method
    )


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
        "--y_label",
        type=str,
        help="Output column",
        required=False,
        default="mean_temp"
    )

    parser.add_argument(
        "--method",
        type=str,
        help="Drift method for correlation rank",
        required=False,
        default="wasserstein"
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