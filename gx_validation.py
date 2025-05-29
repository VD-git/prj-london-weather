import argparse
import json
import pandas as pd
import os
import great_expectations as gx

def define_batch():
    
    context = gx.get_context()
    data_source = context.data_sources.add_pandas(name="dataframe")
    data_asset = data_source.add_dataframe_asset(name="data_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name="batch")

    return context, batch_definition

def define_suite():

    suite = gx.ExpectationSuite(name = "suite")

    list_expectations = [
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set = ["date", "cloud_cover", "sunshine", "global_radiation", "max_temp", "mean_temp", "min_temp", "precipitation", "pressure", "snow_depth"],
            exact_match = True
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column = "mean_temp",
            min_value = 0,
            condition_parser = "pandas",
            row_condition = 'sunshine > 0'
        )
    ]

    for expect in list_expectations:
        suite.add_expectation(expectation=expect)

    return suite

def go(args):

    df = pd.read_csv(os.path.join("data", args.csv))
    context, batch_definition = define_batch()
    
    suite = define_suite()
    suite = context.suites.add(suite=suite)
    
    validation_definition = gx.ValidationDefinition(name = "validation", data = batch_definition, suite = suite)
    validation_definition = context.validation_definitions.add(validation_definition)
    
    checkpoint = gx.Checkpoint(name = "checkpoint", validation_definitions = [validation_definition], actions=[])
    checkpoint_results = checkpoint.run(batch_parameters={"dataframe": df})

    with open("output/metrics.json", "w+") as f:
        json.dump({"status": checkpoint_results.success, "logs": checkpoint_results.describe()}, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script For Model Training"
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="Data file",
        required=False,
        default="london_weather.csv"
    )

    args = parser.parse_args()

    go(args)