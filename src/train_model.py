import argparse
import json
import pandas as pd
import os
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def prepare_dataset(df:pd.DataFrame):
    """
    Preprocessing of the dataset for training the model
    Args:
        df: pd.DataFrame
    Outputs:
        X_train: np.array
        X_test: np.array
        y_train: np.array
        y_test: np.array
    """

    df.dropna(subset=['mean_temp'], inplace = True)
    X = df.drop(['mean_temp', 'date'], axis = 1)
    y = df['mean_temp']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test
    

def build_pipeline(params:dict):
    """
    Build of the pipeline to train the model
    Args:
        params: dict
            Parameters to be inputed for the gridsearch
    Outputs:
        pipeline_cv: Pipeline
    """

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("preprocessor", StandardScaler()),
            ("model", RandomForestRegressor())
        ]
    )

    params = {
        "model__max_depth": [5, 10, 15],
        "model__n_estimators": [100]
    }

    pipeline_cv = GridSearchCV(pipeline, param_grid=params, cv = 3)

    return pipeline_cv


def go(args):

    df = pd.read_csv(os.path.join("data", args.csv))
    params = json.loads(args.params)
    
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    pipeline_cv = build_pipeline(params)
    
    pipeline_cv.fit(X_train, y_train)
    y_pred = pipeline_cv.predict(X_test)
    

    with open("output/model_metrics.json", "w+") as f:
        json.dump({"rmse": np.sqrt(mean_squared_error(y_test, y_pred))}, f)

    with open("output/model.pkl", "wb") as f:
        pickle.dump(pipeline_cv, f)

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
    
    parser.add_argument(
        "--params",
        type=str,
        help="Data file",
        required=False,
        default='{"model__max_depth": [5, 10, 15], "model__n_estimators": [100]}'
    )

    args = parser.parse_args()

    go(args)