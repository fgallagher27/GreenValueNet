"""
This script contains the functions to run the baseline and deep neural network models
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def create_x_y_arr(dataset: pd.DataFrame, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function splits the dataset into an input array and an output array
    Each row corresponds to an example, and each column of x to a feature.
    """
    df = dataset.drop(columns=params['cols_out'])

    x = df.drop(columns=params['target_var']).to_numpy()
    y = df[params['target_var']].to_numpy()

    return x,y


def split_to_test_dev_train(
        x: np.ndarray,
        y: np.ndarray,
        dev_size:float,
        test_size:float,
        prop: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function splits the dataframe into training, dev and test
    datasets based in the values provided. prop is a boolean toggle
    for whether arguments are interpreted as absolute size, or
    proportion of the total dataset
    """
    if not prop:
        # convert absolute values to proportions
        total_obs = len(x)
        dev_size /= total_obs 
        test_size /= total_obs

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=(dev_size + test_size)
    )

    x_dev, x_test, y_dev, y_test = train_test_split(
        x_temp,
        y_temp, 
        test_size=(test_size / test_size + dev_size)
    )

    # TODO create a toggle that allows sampling based on TimeSeriesSplit() from sklearn

    return x_train, x_dev, x_test, y_train, y_dev, y_test


def random_forest_reg(x_train,x_dev,y_train,y_dev, tuning, tuning_params: dict = None) -> tuple():
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


def baseline_nn(x_train, y_train, batch_size):
    """
    This function creates the baseline single layer neural network model
    """
    # TODO make this flexible so that n_layers becomes a hyperparameter 
    # and can be iterated over as an argument in baseline_nn()
    mean_squared_error = tf.keras.metrics.mean_squared_error
    # this normalises over the last axis?
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)

    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(from_logits=True),
        metrics=['mean_squared_error']
    )

    model.fit(x_train, y_train, epochs=15, batch_size = batch_size)

    pred = None
    metrics = None

    return model, pred, metrics