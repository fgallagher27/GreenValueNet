"""
This script contains the functions to run the baseline and deep neural network models
"""

import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split

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


def random_forest_reg():
    """
    This function creates the random forest regression
    model used as a baseline model.
    """

    return model