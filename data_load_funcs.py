"""
This script contains the functions needed to download the GreenValueNet inputs and
interact with the data catalogue.
It can be imported as a module into other scripts.
It contains the following functions:

    * download_zip_from_url 
    * extract_zip
    * download_data
    * load_data_catalogue
    * get_file_path

"""

import os
import pandas as pd
import geopandas as gpd
import zipfile
import yaml
import requests
from pathlib import Path
from typing import List, Union

cwd = Path.cwd()

### Download files ----

def download_zip_from_url(url: str, file_path: str):
    """
    This function downloads a zip file from a url and places into file_path

    Args:
        url (str): url address of the zip folder to download
        file_path (str): folder path to save the zip file to including the name of the zip

    """
    print("Downloading data...")
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    response = requests.get(url)

    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File '{file_path}' downloaded successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def extract_zip(zip_file_path: str, extract_location: str):
    """
    This function extracts the contents of a zipfile to extract_location.
    """
    print(f"Extracting data from {zip_file_path}")
    try:
        if not os.path.isfile(zip_file_path):
            raise FileNotFoundError(f"The file '{zip_file_path}' does not exist.")
        
        # Create the extraction directory if it doesn't exist
        os.makedirs(extract_location, exist_ok=True)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_location)

        print(f"Successfully extracted {zip_file_path} to {extract_location}")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# maybe at some point refactor as a module/class structure
def download_data():
    """
    This function checks if the required data has already been downloaded.
    If it has not, it downloads and extracts it.
    """

    data_catalogue = load_data_catalogue()

    for input in data_catalogue['downloaded_inputs']:

        print(f"Accessing information for {input} input...")
        catalogue = data_catalogue['inputs'][input]
        folder_path = catalogue['location']
        file_name = catalogue['file_name']
        zip = catalogue['zip_folder']
        url = catalogue['url']
        input_path = os.path.join(folder_path, file_name)

        if os.path.exists(input_path):
            print(f"{file_name} is already downloaded in the subdirectory 'data'")
        elif os.path.exists(zip):
            print(f"Extracting data from {folder_path}...")
            extract_zip(zip, folder_path)
        elif url is not None:
            print(f"Downloading and extracting data from {url}...")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            download_zip_from_url(url, zip)
            extract_zip(zip, folder_path)
        else:
            raise ValueError("Invalid paths specified in data catalogue")

def check_inputs():
    """
    This function ensures that all inputs are present before processing.
    """
    catalogue = load_data_catalogue()


### Data Catalogue ----

def load_data_catalogue() -> dict:
    """
    Loads data catalogue
    """
    # Load the YAML data from the file
    with open('data_catalogue.yml', 'r') as file:
        data_catalogue = yaml.safe_load(file)
    return data_catalogue


def get_file_path(catalogue: dict, level_1: str, level_2: str) -> str:
    """
    Extracts the file path from the data catalogue

    Args:
    catalogue (dict): a dictionary catalogue with 3 layers
    level_1 (str): the first level of the catalogue to search
    level_2 (str): the second level of the catalogue to search

    Returns:
    string containing file path
    """
    
    print(f"Extracting file path from catalogue: {level_1}/{level_2} ...")
    folder = catalogue[level_1][level_2]['location']
    file = catalogue[level_1][level_2]['file_name']
    return os.path.join(folder, file)

### Parameters ----

def get_params(model_run: str = 'default') -> dict:
    """
    Loads config.yml containing parameters
    """
    # Load the YAML data from the file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    if model_run in config:
        params = config[model_run]
    else:
        raise ValueError(f"No configuration found for model_run {model_run}")
    
    return params


def process_spatial_dict(params: dict) -> dict:
    """
    This function processes the folder and file
    names in the spatial dictionary in the config
    file into system agnostic filepaths to be
    used in the creation of the spatial
    attributes file
    """
    spatial_dict = params['spatial_dict']

    new = {}
    for key, values in spatial_dict.items():
        path = cwd / "data" / values['folder'] / values['file']
        
        # Update the dictionary with the new file path
        new[key] = path

    return new

