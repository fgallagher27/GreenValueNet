"""
This script downloads and cleans the data needed for GreenValueNet
"""

import pandas as pd
from processing_funcs import *

download_data()
catalogue = load_data_catalogue()
hp_path = get_file_path(catalogue, 'inputs', 'house_prices')

raw_hp_data = pd.read_csv(hp_path)
print(raw_hp_data.head)

# drop unneccessary columns

# join to environmental rasters

# for now lets do air quality, maybe water quality