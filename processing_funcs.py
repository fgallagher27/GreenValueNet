"""
This script contains the functions to process the input data
Data processing can be conducted by executing process_data()
The following functions are defined:

    * process_data
    * process_housing_data
    * process_spatial_attr
"""

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from data_load_funcs import get_params, get_file_path, load_data_catalogue

cwd = Path.cwd()
def process_data(catalogue, params):
    """
    This function processes the input data
    """
    hp = process_housing_data(catalogue, params)
    spa = process_spatial_attr(catalogue, params)
    df = pd.merge(hp, spa)

    df.to_csv()

    return df


def process_housing_data(catalogue, params):
    """
    This function cleans the raw house prices csv
    """
    hp_catalogue = catalogue['inputs']['house_prices']
    clean_file_path = cwd / "data" / "processed_data" / params['house_prices']['processed_file']
    raw_file_path = cwd / "data" / hp_catalogue['location'] / hp_catalogue['file_name']

    if os.path.exists(clean_file_path):
        hp = pd.read_csv(clean_file_path)
    
    elif os.path.exists(raw_file_path):
        hp_raw = pd.read_csv(raw_file_path)
        # do analysis
        hp.to_csv(clean_file_path, index=False)
    
    else:
        raise FileNotFoundError("House price data specified in data catalogue and parameters not found")
    
    return hp


def process_spatial_attr(catalgoue, params):
    ...


def clean_glud():
    # read in postcodes - ONS map
    # read in GLUD, drop first 6 rows and name columns
    # join dfs keeping all unique postcodes and duplicating ward rows
    # convert all to proportion of total area
    # drop boring columns
    # normalise values whilst we're here
    # write out as csv to processed file
    ...


if __name__ == "__main__":
    params = get_params()
    data_catalogue = load_data_catalogue()
    process_data(data_catalogue, params)
