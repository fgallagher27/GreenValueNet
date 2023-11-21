"""
This script downloads and cleans the data needed for GreenValueNet
Given the size of the data required, this script can take #X mins to run
"""

import pandas as pd
import time
from processing_funcs import *

# initialise script run time
tic = time.time()

download_data()

# here write function to write for rest of files

catalogue = load_data_catalogue()

hp_path = get_file_path(catalogue, 'inputs', 'house_prices')

# write a function to load a file from its foldder and path based on an entry into the catalogue
# this should also have a dynamic switch to recognise the file type and use the right function

raw_hp_data = pd.read_csv(hp_path)
print(raw_hp_data.head)

# first clean the housing data, save a subset to test on

# drop unneccessary columns

# join to environmental rasters


toc = time.time()
print('data_processing.py runtime: ', round((toc-tic)/60),' minutes')