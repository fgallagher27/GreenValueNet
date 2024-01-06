"""
This script contains the functions to run the baseline and deep neural network models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras import layers, models, initializers


def create_x_y_arr(dataset: pd.DataFrame, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function splits the dataset into an input array and an output array
    Each row corresponds to an example, and each column of x to a feature.
    """
    df = dataset.drop(columns=params['cols_out'])

    derivative_cols = params['derivative_cols']
    derivative_index = [df.columns.get_loc(col) for col in derivative_cols]
    zipped_index = list(zip(derivative_cols, derivative_index))

    x = df.drop(columns=params['target_var']).to_numpy()
    y = df[params['target_var']].to_numpy()

    return x,y, zipped_index


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
        tuning_params: dict = None,
        **kwargs) -> RandomForestRegressor:
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

        tuning = GridSearchCV(
            estimator=RandomForestRegressor(**kwargs),
            param_grid=tuning_params
        )

        rfr = tuning.fit(x_train, y_train)

    else:
        rfr = RandomForestRegressor(**kwargs).fit(x_train, y_train)
    
    return rfr


def boosted_grad_reg(x_train, y_train, **kwargs) -> GradientBoostingRegressor:
    """
    This function trains a boosted regression model
    to be used in model benchmarking
    """

    reg = GradientBoostingRegressor(**kwargs)
    xgb = reg.fit(x_train, y_train)

    return xgb


def neural_net(
        x_train: Union[tf.Tensor, np.ndarray],
        y_train: Union[tf.Tensor, np.ndarray],
        n_hidden_units: int,
        n_layers: int = 1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 50,
        hidden_activation: str = 'relu',
        output_activation: str = 'linear',
        loss: str = 'mean_squared_error',
        **kwargs
    ) -> tf.keras.Model:
    """
    This function creates a neural network model with n_layers hidden layers
    and an output layer using the activations specified.
    """

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=x_train.shape[1:]))
    
    # Adding hidden layers
    for _ in range(n_layers):
        model.add(
            layers.Dense(
                n_hidden_units,
                kernel_initializer=initializers.he_normal(),
                trainable=True)
            )
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(hidden_activation))
    
    # Adding output layer
    model.add(layers.Dense(
        units=1,
        activation=output_activation,
        kernel_initializer=initializers.glorot_normal(),
        trainable=True))
    
    model.compile(
        optimizer=tf._optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['mae']
    )
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, **kwargs) 

    return model


def generate_pred_metric(model, metric, x_dev, y_dev):
    """
    This function takes model which has a method predict
    and generates predictions and accuracy according to a 
    metric function that takes in two arrays of numbers
    """
    pred = model.predict(x_dev)
    metric_calc = metric(y_dev, pred)

    return pred, metric_calc


def generate_plot(nn_dict: dict, baseline_dict: dict):
    """
    This function plots the loss of the baseline models
    and neural networks over epochs
    """

    for model, loss in nn_dict.items():
        plt.plot(
            range(1, len(loss) + 1),
            loss,
            label = model
        )
    
    for model, loss in baseline_dict.items():
        plt.axhline(y=loss, linestyle='--', label=model)

    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of performance of models')
    plt.legend()
    plt.show()


def calc_partial_grad(
        model: tf.keras.Model,
        x_train: np.ndarray,
        derivative_index: zip,
        points_to_eval: np.ndarray):
    """
    This function evaluates the partial derivative 
    w.r.t specified features in derivative_index.
    It does so at the feature values specified in
    points_to_eval.
    """

    def partial_derivative(model, x_values, feature_index):
        x_values = tf.constant(x_values, dtype=tf.float32)
        with tf.GradientTape() as tape:

            prediction = model(x_values)

        # Calculate the partial derivative
        gradients = tape.gradient(prediction, x_values)
        partial_deriv = gradients[:, feature_index]

        return partial_deriv.numpy()

    # for each feature in gradient calculations
    derivative_vals = {}
    x_mean = np.mean(x_train, axis=0)
    for name, index in zip(*derivative_index):

        # Create input features, setting other features to their mean
        x_values = np.tile(x_mean, (len(points_to_eval), 1))
        x_values[:, 0] = points_to_eval

        pd_values = partial_derivative(model, x_values, index)

        derivative_vals[name] = pd_values

    return derivative_vals
