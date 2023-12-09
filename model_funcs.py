"""
This script contains the functions to run the baseline and deep neural network models
"""

import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def split_to_test_dev_train(dataset: pd.DataFrame, dev_size, test_size, prop=True):
    """
    This function splits the dataframe into training, dev and test
    datasets based in the values provided. prop is a boolean toggle
    for whether arguments are interpreted as absolute size, or
    proportion of the total dataset
    """
    if not prop:
        # convert absolute values to proportions
        total_obs = len(dataset)
        dev_size /= total_obs 
        test_size /= total_obs 
    
    train, remaining = train_test_split(dataset, test_size=(dev_size + test_size))
    dev, test = train_test_split(remaining, test_size = (test_size / test_size + dev_size))

    return train, dev, test


def extract_target_var(df:pd.DataFrame, target_col: str, out_cols: List[str]):
    """
    This function isolates the target variable as a seperate object
    """
    target = df[target_col]
    input = df.drop(columns=[target_col, out_cols], axis=1)
    return input, target


def random_forest_reg(x_train,x_dev,y_train,y_dev, tuning, tuning_params: dict = None):
    """
    This function creates the random forest regression
    model used as a baseline model.

    Parameters:
        tuning (boolean): set to True to activate parameter tuning
        tuning_params (dict): dictionary containing tuning parameters.
        Should include a 'grid' that contains dimensions of grid to tune
        with.
    """

    if tuning:
        assert tuning_params is not None, "if 'tuning=True, tuning_params is required"

        grid = tuning_params['grid']

        tuning = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=grid
        )

        rfr = tuning.fit(x_train, y_train)

    else:
        rfr = RandomForestRegressor().fit(x_train, y_train)
        
    rfr_pred = rfr.predict(x_dev)
    mse = mean_squared_error(y_dev, rfr_pred)
    rmse = mse ** 0.5

    metrics = {
        'mse': mse,
        'rmse': rmse
    }
    
    return rfr, rfr_pred, metrics
