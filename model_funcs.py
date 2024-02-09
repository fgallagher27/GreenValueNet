"""
This script contains the functions to run the baseline and deep neural network models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kerastuner as kt
from pathlib import Path
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from tensorflow.keras import layers, models, initializers

cwd = Path.cwd()

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
    arr_path = cwd / 'data'/ 'interim_files' / 'test_dev_train_arr.npz'
    if os.path.exists(arr_path):
        arrays = np.load(arr_path)
        x_train, x_dev, x_test, y_train, y_dev, y_test = arrays.values()
    else:
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

        np.savez(
            arr_path,
            x_train=x_train,
            x_dev=x_dev,
            x_test=x_test,
            y_train=y_train,
            y_dev=y_dev,
            y_test=y_test
        )
        # TODO create a toggle that allows sampling based on TimeSeriesSplit() from sklearn
    return x_train, x_dev, x_test, y_train, y_dev, y_test


def baseline_model(
        x_train: np.ndarray,
        y_train: np.ndarray,
        model_func: Union[HistGradientBoostingRegressor, RandomForestRegressor],
        tuning: bool,
        tuning_params: dict = None,
        tuning_iter: int = 10,
        **kwargs) -> Union[HistGradientBoostingRegressor, RandomForestRegressor]:
    """
    This function creates the baseline model.

    Parameters:
        tuning (boolean): set to True to activate parameter tuning
        tuning_params (dict): dictionary containing tuning parameters.
        Should include a 'grid' that contains dimensions of grid to tune
        with.
    """

    if tuning:
        assert tuning_params is not None, "if 'tuning=True, tuning_params is required"

        tuning = RandomizedSearchCV(
            estimator=model_func(**kwargs),
            param_distributions=tuning_params,
            n_iter = tuning_iter
        )

        model = tuning.fit(x_train, y_train)
        print("Optimal parameters based on hyperparameter tuning: ", tuning.best_params_)

    else:
        model = model_func(**kwargs).fit(x_train, y_train)
    
    return model


def build_neural_net(
        input_shape: int,
        n_hidden_units: int,
        n_layers: int = 1,
        learning_rate: float = 0.01,
        hidden_activation: str = 'relu',
        output_activation: str = 'linear',
        optimizer: Union[tf.keras.optimizers.Optimizer, str] = tf.keras.optimizers.Adam,
        loss: str = 'mean_squared_error',
        tuning: bool = False,
    ) -> tf.keras.Model:
    """
    This function creates a neural network model with n_layers hidden layers
    and an output layer using the activations specified.
    """

    if tuning:
        model = build_tuned_model(
            kt.HyperParameters(),
            input_shape=input_shape,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            optimizer=optimizer,
            loss=loss
        )

    else:
        model = build_model(
            input_shape,
            n_layers,
            n_hidden_units,
            hidden_activation,
            output_activation,
            optimizer=optimizer,
            loss=loss,
            lr=learning_rate
        )

    return model

def build_model(
        input_shape: int,
        n_units: int,
        n_layers: int = 1,
        hidden_activation: str = 'relu', 
        output_activation: str = 'linear',
        optimizer: Union[tf.keras.optimizers.Optimizer, str] = tf.keras.optimizers.Adam,
        loss: str = 'mean_squared_error',
        lr: float = 0.01,
        dropout: bool = False,
        d_rate: float = 0.25,
    ) -> tf.keras.Model:
    """
    This function builds the model architecture
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_shape,)))

    # Add hidden layers
    for _ in range(n_layers):
        model.add(
            tf.keras.layers.Dense(
                n_units,
                activation=hidden_activation
            )
        )
        if dropout:
            model.add(layers.Dropout(rate=d_rate))

    # Add output layer
    model.add(
        tf.keras.layers.Dense(
            1,
            activation=output_activation
        )
    )

    # Compile the model
    model.compile(optimizer=optimizer(learning_rate=lr), loss=loss)

    return model


def build_tuned_model(hp, **kwargs):
    """
    This function creates a model class with a hyperparameter
    search space that can be tuned by running run_hp_search.
    """
    # TODO adjust model functions to tune number of units in different layers seperately
    # TODO read in hp search space from config file
    n_hidden_units = hp.Int('n_units', min_value=16, max_value=128, step=32)
    n_layers = hp.Choice('n_layers', [5, 8, 10])
    learning_rate = hp.Choice('lr', [0.01, 0.05, 0.1])

    model = build_model(
        n_layers = n_layers,
        n_units = n_hidden_units,
        lr=learning_rate,
        **kwargs,
    )
    return model


def run_hp_search(
        x_train: Union[np.ndarray, tf.Tensor],
        y_train: Union[np.ndarray, tf.Tensor],
        validation_set: Tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]],
        search_name: str,
        n_models_return: int = 1,
        search_epochs: int = 5,
        algorithm: kt.HyperModel = kt.BayesianOptimization,
        print_summaries: bool = False,
        **kwargs
    ):
    """
    This function executes the hyperparamter search algorithms. 
    The algorithm used is determined by the algorithm argument and
    defaults to Bayesian Optimization
    """
    tuner = algorithm(
        hypermodel=lambda hp, **hp_kwargs: build_tuned_model(
            hp,
            input_shape=x_train.shape[1:],
            **hp_kwargs
        ),
        objective="mse",
        max_trials=10,
        executions_per_trial=2,
        overwrite=True,
        directory= cwd / 'outputs' / 'models' / 'tuning',
        project_name=search_name,
        **kwargs
    )
    if print_summaries:
        tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=search_epochs, validation_data=validation_set)
    deep_nn = tuner.get_best_models(num_models=n_models_return)

    return deep_nn


def generate_pred_metric(model, metric, x_dev, y_dev):
    """
    This function takes model which has a method predict
    and generates predictions and accuracy according to a 
    metric function that takes in two arrays of numbers
    """
    pred = model.predict(x_dev)
    metric_calc = metric(y_dev, pred)

    return pred, metric_calc


def generate_plot(nn_dict: dict, baseline_dict: dict, save: bool = False, name: str = ''):
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
    if save:
        plt.savefig(cwd / "outputs" / "images" / name, format=name[-3:])
        plt.close()
    plt.show()


def calc_partial_grad(
        model: tf.keras.Model,
        input_values: np.ndarray,
        derivative_index: zip,
        points_to_eval: np.ndarray):
    """
    This function loops over the derivative index and
    calculates partial derivative of each feature holding
    all other variables at their mean. It does so evaluating
    the partial derivative for x_i equal to each value in 
    points_to_eval.
    """
    def calc_partial_derivatives(model, x_values):
        x_tf = tf.constant(x_values, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            predictions = model(x_tf)
            target_var = predictions[0,0]

        gradients = tape.gradient(target_var, x_tf)

        return gradients.numpy()

    # Initialise empty array to store gradients
    partial_derivs = np.zeros((len(points_to_eval), input_values.shape[1]))

    # Loop through each feature
    keep = []
    for key, i in derivative_index:
        keep.append(i-1)

        # Loop over values of x_i to calculate partial derivative
        for j, n in enumerate(points_to_eval):

            # reset values to their mean
            vals = input_values.copy()
            vals = vals.mean(axis=0)

            # adjust value of x_i
            vals[i - 1] = n
            vals = np.expand_dims(vals, axis=0)
            partial_derivatives = calc_partial_derivatives(model, vals)
            partial_derivs[j, i-1] = partial_derivatives[0, i-1]

    partial_derivs = partial_derivs[:, keep]
    return partial_derivs


def plot_partial_grads(
        gradients: np.ndarray,
        points_to_eval: np.ndarray,
        derivative_index: zip,
        save: bool,
        name: str
    ):
    """
    This function takes the partial gradient
    array and plots each features partial gradient
    curve over the range of points to eval
    """
    for col, (label, old_index) in enumerate(derivative_index):
        y_values = gradients[:, col]
        plt.plot(points_to_eval, y_values, label=label)
    plt.xlabel('Change in x_i')
    plt.ylabel('Change in ln(price)')
    plt.title('Partial derivative curves for selected features')
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    if save:
        plt.savefig(cwd / "outputs" / "images" / name, format=name[-3:])
        plt.close()
    plt.show()
        