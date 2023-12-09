"""
This script contains the functions to run the baseline and deep neural network models
"""

import pandas as pd
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
