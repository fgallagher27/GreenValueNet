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
from tensorflow.keras.callbacks import ModelCheckpoint
from processing_funcs import normalise_arr

cwd = Path.cwd()

def create_x_y_arr(dataset: pd.DataFrame, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function splits the dataset into an input array and an output array
    Each row corresponds to an example, and each column of x to a feature.
    """
    df = dataset.drop(columns=params['cols_out'])
    x = df.drop(columns=params['target_var'])
    
    derivative_cols = params['derivative_cols']
    derivative_index = [x.columns.get_loc(col) for col in derivative_cols]
    zipped_index = list(zip(derivative_cols, derivative_index))

    
    # get indexes to normalise
    norm_cols = [col for col in dataset.columns if col not in params['non_norm_cols']]
    norm_index = [x.columns.get_loc(col) for col in norm_cols]
    x = x.to_numpy()
    y = df[params['target_var']].to_numpy()

    return x,y, zipped_index, norm_index


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
        loss: str = 'mse',
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
            n_hidden_units,
            n_layers,
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
        loss: str = 'mse',
        lr: float = 0.01,
        dropout: bool = False,
        d_rate: float = 0.25,
    ) -> tf.keras.Model:
    """
    This function builds the model architecture
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

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
    model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics=[loss])

    return model


def build_tuned_model(hp, **kwargs):
    """
    This function creates a model class with a hyperparameter
    search space that can be tuned by running run_hp_search.
    """
    # TODO adjust model functions to tune number of units in different layers seperately
    # TODO read in hp search space from config file
    n_hidden_units = hp.Int('n_units', min_value=48, max_value=144, step=16)
    n_layers = hp.Choice('n_layers', [8, 10, 12])
    learning_rate = hp.Choice('lr', [0.001, 0.01])

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
        search_epochs: int = 3,
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
        max_trials=20,
        executions_per_trial=1,
        overwrite=True,
        directory= cwd / 'outputs' / 'models' / 'tuning',
        project_name=search_name,
        **kwargs
    )
    if print_summaries:
        tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=search_epochs, validation_data=validation_set)

    return tuner


def get_checkpoint(name:str) -> ModelCheckpoint:
    name += '.keras'
    model_dir = str(cwd / "outputs" / "models" / name)
    return ModelCheckpoint(
        model_dir,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=0
    )


def generate_pred_metric(model, metric, x_dev, y_dev):
    """
    This function takes model which has a method predict
    and generates predictions and accuracy according to a 
    metric function that takes in two arrays of numbers
    """
    pred = model.predict(x_dev)
    metric_calc = metric(y_dev, pred)

    return pred, metric_calc


def generate_plot(
        model_dict: dict,
        baseline_dict: dict,
        cut_off_epoch: int = 0,
        save: bool = False,
        name: str = ''):
    """
    This function plots the loss of the baseline models
    and neural networks over epochs
    """
    colours = ['blue', 'orange', 'green', 'purple']
    nn_dict = model_dict.copy()

    for i, (model, history) in enumerate(nn_dict.items()):
        if cut_off_epoch == 0:
            limit = len(history['loss'])
        else:
            limit = cut_off_epoch
        colour=colours[i]
        loss_arr = history['loss'][:limit]
        val_loss_arr = history['val_loss'][:limit]
        plt.plot(
            range(1, len(loss_arr) + 1),
            loss_arr,
            label = model + ' - train set',
            color=colour
        )
        plt.plot(
            range(1, len(val_loss_arr) + 1),
            val_loss_arr,
            label = model + ' - dev set',
            linestyle='--',
            color=colour
        )
    
    for model, loss in baseline_dict.items():
        plt.axhline(y=loss, color='red', linestyle='--', label=model)

    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of performance of models')
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    if save:
        plt.savefig(cwd / "outputs" / "images" / name, format=name[-3:])
        plt.show()
        plt.close()
    else:
        plt.show()


def calc_partial_grad(
        model: tf.keras.Model,
        dataset: np.ndarray,
        derivative_index: zip,
        norm_index: List[int],
        pop_mean: np.ndarray,
        pop_std: np.ndarray,
        num_points_to_eval: np.ndarray):
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

        gradients = tape.gradient(predictions, x_tf)

        return gradients.numpy()
    
    # initialise output dictionaries
    gradients = {}
    synthetic_data = {}

    dataset_arr = dataset.copy()
    med_vals = np.median(dataset_arr, axis=0)
    sampled_index = np.linspace(0, 100, num_points_to_eval + 1)

    # Loop through each feature
    for feature, i in derivative_index:
        # create synthetic data points
        arr = np.zeros((num_points_to_eval + 1, dataset_arr.shape[1]))
        sorted_vals = np.sort(dataset_arr[:, i])
        sampled_vals = np.percentile(sorted_vals, sampled_index)
        # set all values to median
        arr[:,:] = med_vals
        # overwrite column i with percentile values
        arr[:, i] = sampled_vals

        norm_arr, _, _ = normalise_arr(arr, norm_index, pop_mean, pop_std)

        # calculate gradients using backward propagation
        partial_derivs = calc_partial_derivatives(model, norm_arr)
        gradients[feature] = partial_derivs[:, i]
        synthetic_data[feature] = arr

    return gradients, synthetic_data

def calc_partial_grad_temp(
        model: Union[tf.keras.Model, RandomForestRegressor, HistGradientBoostingRegressor],
        dataset: np.ndarray,
        derivative_index: zip,
        norm_index: List[str],
        pop_mean: np.ndarray,
        pop_std: np.ndarray,
        num_points_to_eval: np.ndarray = 100):
    """
    This function loops over the derivative index and
    calculates partial derivative of each feature holding
    all other variables at their median. It does so by sampling 
    num_points_to_eval quantiles from the array, and then
    approximating the gradient at this point using a very
    simple rise / run calculation.
    """
    dataset_arr = dataset.copy()
    med_vals = np.median(dataset_arr, axis=0)
    sampled_index = np.linspace(0, 100, num_points_to_eval + 1)
    
    gradients = {}
    synthetic_data = {}
    for feature, i in derivative_index:
        # create synthetic data points
        arr = np.zeros((num_points_to_eval + 1, dataset_arr.shape[1]))
        sorted_vals = np.sort(dataset_arr[:, i])
        sampled_vals = np.percentile(sorted_vals, sampled_index)
        # set all values to median
        arr[:,:] = med_vals
        # overwrite column i with percentile values
        arr[:, i] = sampled_vals

        norm_arr, _, _ = normalise_arr(arr, norm_index, pop_mean, pop_std)

        # generate synthetic predictions
        predictions = model.predict(norm_arr)
        predictions = predictions.flatten()
        output_diff = np.diff(predictions)
        input_diff = np.diff(sampled_vals)
        # approximate gradients as rise/run
        gradients[feature] = (output_diff / input_diff)
        synthetic_data[feature] = arr

    return gradients, synthetic_data




def plot_partial_grads(
        gradients: dict,
        x_points: np.ndarray,
        save: bool = False,
        name: str = ''
    ):
    """
    This function takes the partial gradient
    array and plots each features partial gradient
    curve over the range of points to eval
    """
    for label, grads in gradients.items():
        plt.plot(x_points, grads, label=label)
    plt.xlabel('Percentile rank of x_i')
    plt.ylabel('Change in ln(price)')
    plt.title('Partial derivative curves for selected features')
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    if save:
        plt.savefig(cwd / "outputs" / "images" / name, format=name[-3:])
        plt.show()
        plt.close()
    else:
        plt.show()


def plot_loss(model, validation_data, metric):
    """
    This function plots the loss of a scikit learn
    gradient boosting regression over boosting iterations
    """
    params = model.get_params()
    test_score = np.zeros((params["max_iter"],), dtype=np.float64)
    for i, y_pred in enumerate(model.staged_predict(validation_data[0])):
        test_score[i] = metric(validation_data[1], y_pred)
    
    plt.plot(
        np.arange(params['max_iter']),
        abs(model.train_score_[1:]),
        "b-",
        label="Training set error"
    )
    plt.plot(
        np.arange(params['max_iter']),
        test_score,
        "r-",
        label="Test set error"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Loss")
    plt.show()