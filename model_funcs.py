"""
This script contains the functions to run the baseline and deep neural network models
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras import layers, models


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
        test_size=(test_size / (test_size + dev_size))
    )

    # TODO create a toggle that allows sampling based on TimeSeriesSplit() from sklearn

    return x_train, x_dev, x_test, y_train, y_dev, y_test


def random_forest_reg(
        x_train: np.ndarray,
        y_train: np.ndarray,
        tuning: bool,
        tuning_params: dict = None) -> RandomForestRegressor:
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
    
    return rfr


def boosted_grad_reg(x_train, y_train) -> GradientBoostingRegressor:
    """
    This function trains a boosted regression model
    to be used in model benchmarking
    """

    reg = GradientBoostingRegressor()
    xgb = reg.fit(x_train, y_train)

    return xgb


def neural_net(
        x_train: tf.tensor,
        y_train: tf.tensor,
        n_layers: int = 1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        hidden_activation: str = 'relu',
        output_activation: str = 'linear',
        loss: str = 'mean_squared_error'
    ) -> tf.keras.Model:
    """
    This function creates a neural network model with n_layers hidden layers
    and an output layer using the activations specified.
    """

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=x_train.shape[1:]))
    
    # Adding hidden layers
    for _ in range(n_layers):
        model.add(layers.Dense(128, activation=hidden_activation))
    # Adding output layer
    model.add(layers.Dense(units=1, activation=output_activation))
    
    model.compile(
        optimizer=tf._optimizers.Adam(learning_rate = learning_rate),
        loss=loss,
        metrics=[loss]
    )
    model.fit(x_train, y_train, epochs=5, batch_size=batch_size) 

    return model


def generate_pred_metric(model, metric: function, x_dev, y_dev):
    """
    This function takes model which has a method predict
    and generates predictions and accuracy according to a 
    metric function that takes in two arrays of numbers
    """
    pred = model.predict(x_dev)
    metric_calc = metric(y_dev, pred)

    return pred, metric_calc