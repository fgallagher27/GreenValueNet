"""
This script contains the functions to process the input data
Data processing can be conducted by executing process_data()
The following functions are defined:

    * process_data
    * process_housing_data
    * process_spatial_attr
"""


import pandas as pd
from pathlib import Path
from data_load_funcs import get_params, get_file_path

cwd = Path.cwd()
def process_data():
    """
    This function processes the input data
    """
    hp = process_housing_data()
    spa = process_spatial_attr()
    df = pd.merge(hp, spa)

    df.to_csv()

    return df


def process_housing_data():
    ...

def process_spatial_attr():
    ...