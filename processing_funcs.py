"""
This script contains the functions to process the input data
Data processing can be conducted by executing process_data()
The following functions are defined:

    * process_data
    * process_housing_data
    * process_spatial_attr
    * process_glud
    * process_school_data
    * calc_dist_to_nearest
    * calc_share
    * normalise_values
    * integer_encoding
    
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.impute import KNNImputer
from pathlib import Path
from typing import List, Union
from data_load_funcs import get_params, get_file_path, load_data_catalogue, process_spatial_dict

cwd = Path.cwd()
def process_data(catalogue: dict, params: dict) -> pd.DataFrame:
    """
    This function processes the input data
    """

    folder = cwd / "data" / "interim_files"
    path = folder / catalogue['interim_files']['dataset']['file_name']

    if os.path.exists(path):
        print("Processed dataset already exists.\nLoading exisitng dataset...")
        df = pd.read_csv(path)
    else:
        print("Processing data to create dataset...")
        hp = process_housing_data(catalogue, params)
        spa = process_spatial_attr(catalogue, params)
        df = pd.merge(hp, spa, on='postcode', how='left')

        if not os.path.exists(folder):
            os.makedirs(folder)

        df.to_csv(path, index=False)

    return df


def process_housing_data(catalogue: dict, params: dict):
    """
    This function cleans the raw house prices csv. Specifically it:
    - Normalised numerical variables
    - encodes strings as integers
    - converts house prices to 2022 GBP and takes log
    - drops excess columns
    - saves the cleaned file to processed_inputs
    """

    clean_file_path = cwd / "data" / "processed_inputs" / params['house_prices']['processed_file']
    raw_file_path = cwd / "data" / get_file_path(catalogue, "inputs", "house_prices")

    if os.path.exists(clean_file_path):
        print("Processed house price file exists.\nLoading data...")
        hp_full = pd.read_csv(clean_file_path)
    
    elif os.path.exists(raw_file_path):

        print("Processing raw house price data...")
        # load inflation index
        ppp_path = get_file_path(catalogue, "inputs", "inflation")
        ppp = pd.read_csv(
            cwd / "data" / ppp_path,
            skiprows=8,
            header=None
        ).head(36) # get annual figures only
        # rebase to 2022
        ppp.columns = ['year', 'index']
        ppp['year'] = pd.to_numeric(ppp['year'])
        ppp['index'] = ppp['index'] / 100
        base_2022 = ppp[ppp['year'] == 2022]['index'].tolist()
        ppp['index'] = ppp['index'] / base_2022

        # load raw house prices
        chunked_list = []
        chunk_size = params['chunksize']
        hp_chunks = pd.read_csv(raw_file_path,chunksize=chunk_size)

        for hp in hp_chunks:
            hp['year'] = pd.to_datetime(hp['dateoftransfer'], format=r"%Y-%m-%d", errors='coerce')
            hp['year'] = hp['year'].dt.year
            hp = hp.merge(ppp, on='year', how='left')
            
            # convert prices to 2022 GBP and take log
            hp['ln_price'] = np.log(hp['price'] / hp['index'])

            # convert potential energy efficiency to proportion of potential
            # where potential < current, overwrite current with potential
            # ensures current as proportion of potential is capped at 1
            en_err_mask = hp['CURRENT_ENERGY_EFFICIENCY'] > hp['POTENTIAL_ENERGY_EFFICIENCY']
            hp.loc[en_err_mask, 'CURRENT_ENERGY_EFFICIENCY'] = hp.loc[en_err_mask, 'POTENTIAL_ENERGY_EFFICIENCY']
            hp['POTENTIAL_ENERGY_EFFICIENCY'] = hp['CURRENT_ENERGY_EFFICIENCY'] / hp['POTENTIAL_ENERGY_EFFICIENCY']

            # TODO move to parameter config
            keep = [
                'transactionid', 'ln_price', 'postcode', 'propertytype',
                'oldnew', 'duration', 'current_energy_efficiency', 'potential_energy_efficiency',
                'total_floor_area', 'extension_count', 'number_habitable_rooms',
                'number_heated_rooms', 'construction_age_band'
            ]
            hp.columns = [item.lower() for item in hp.columns]
            hp = hp.loc[:, keep]

            encode_list = ['propertytype', 'oldnew', 'duration', 'construction_age_band']
            mapping = {key: None for key in encode_list}
            for col in encode_list:
                # TODO replace with one-hot encoding
                hp[col], mapping[col] = integer_encoding(
                    hp[col],
                    exclude_strings=['', 'NO DATA!', 'INVALID!']
                )

            chunked_list.append(hp)

        hp_full = pd.concat(chunked_list, ignore_index=True)

        if params['impute_missing_vals']:

            print("Imputing missing values using nearest neighbours")
            # now we do imputation on missing numeric values using nearest neighbours
            cols_to_impute = hp_full.select_dtypes(include=['int', 'float']).columns[hp_full.select_dtypes(include=['int', 'float']).isna().any()]
            imputer = KNNImputer(weights='distance')
            hp_full[cols_to_impute] = imputer.fit_transform(hp_full[cols_to_impute])
        
        hp_full.to_csv(clean_file_path, index=False)
        pd.DataFrame(mapping).to_csv(
            cwd / "data" / "interim_files" / "encoding_mappings.csv",
            index=False
        )
    
    else:
        raise FileNotFoundError("House price data specified in data catalogue and parameters not found")
    
    return hp_full


def process_spatial_attr(catalogue: dict, params: dict):
    """
    Takes in spatial characteristics from params and calculates the distance
    to the nearest instance of the characterisitic at the postcode level.
    Also joins the generalised land use database and land use catalogue
    characteristics of an area, drops excess columns and stores in 
    processed inputs.
    """

    # check if processed file exists
    file_name = params['spatial_attributes']['processed_file']
    clean_path = cwd / "data" / "processed_inputs" / file_name
    if os.path.exists(clean_path):
        print("Processed spatial attributes file exists.\nLoading data...")
        spatial_attributes = gpd.read_file(clean_path)
    else:
        print("Creating spatial attributes dataset...")
        spatial_mapping = gpd.read_file(
            cwd / "data" / "processed_inputs" / "mapped_postcodes.shp"
        )

        ward_chars = process_glud(catalogue)
        process_school_data(catalogue)
        spatial_dict = process_spatial_dict(params)

        # Apply function over all shapefiles in the feature dict
        spatial_attributes = calc_dist_to_nearest(
            spatial_mapping,
            spatial_dict
        )

        spatial_attributes = spatial_attributes.merge(
            ward_chars,
            left_on='statsward',
            right_on = 'ward_code',
            how='left'
        )

        spatial_attributes = spatial_attributes.drop(
            ['pcd', 'ward_code', 'statsward', 'geometry'],
            axis=1
        )

        spatial_attributes.to_csv(clean_path)

    return spatial_attributes



def process_glud(catalogue: dict):
    """
    This function processes the data from the Generalised Land Use Database
    It drops excess columns, calculates variable as a share of ward area
    and normalises values
    """

    print("Processing Generalised Land Use Database...")
    glud_vars = catalogue['inputs']['ward_characteristics']
    glud_folder = glud_vars['location']
    glud_file = glud_vars['file_name']
    glud = pd.read_csv(
        cwd / "data" / glud_folder / glud_file,
        skiprows=6,
        header=None
    )
    glud = glud.iloc[:, [0,6,7,8,11,12,13,16]]
    cols = [
        'ward_code',
        'dom_builds',
        'garden',
        'non_dom_builds',
        'path',
        'greenspace',
        'water',
        'total_area'
    ]
    glud.columns=cols

    value_cols = [col for col in cols if col not in ['total_area', 'ward_code']]
    glud[value_cols] = (glud[value_cols].apply(pd.to_numeric, errors='coerce')
                        .fillna(0.0)
                        .astype(float))
    for col in value_cols:
        glud[col + '_share'] = glud[col].combine(glud['total_area'], calc_share)
    glud.drop(value_cols + ['total_area'], axis=1, inplace=True)

    glud.to_csv(
        cwd / "data" / "processed_inputs" / "ward_chars.csv",
        index=False
    )

    return glud

def process_school_data(catalogue: dict):
    """
    This function extracts schools as geo points based
    on postcode. Schools are also split into different
    dataframes based on primary vs secondary school
    """

    print("Processing school locations...")
    path = get_file_path(catalogue, 'inputs', 'schools')
    schools_raw = pd.read_csv(cwd / "data" / path)
    schools = schools_raw.loc[:, ['EstablishmentNumber', 'Postcode', 'PhaseOfEducation (name)']]
    schools.rename(columns={
        'EstablishmentNumber': 'school_id',
        'Postcode': 'postcode',
        'PhaseOfEducation (name)': 'sch_type'}, inplace=True)
    schools = schools[schools['sch_type'].isin(['Primary', 'Secondary'])]

    # now convert to a shapefile
    postcodes_path = get_file_path(catalogue, 'inputs', 'postcodes')
    postcodes = gpd.read_file(cwd / "data" / postcodes_path)
    schools = schools.merge(postcodes, on='postcode', how='left')
    schools_shp = gpd.GeoDataFrame(schools, geometry='geometry')
    
    primary_df = schools_shp[schools_shp['sch_type'] == 'Primary']
    secondary_df = schools_shp[schools_shp['sch_type'] == 'Secondary']

    primary_df.to_file(
        cwd / "data" / "processed_inputs" / "primary_school.shp",
        driver = "ESRI Shapefile"
    )

    secondary_df.to_file(
        cwd / "data" / "processed_inputs" / "secondary_school.shp",
        driver = "ESRI Shapefile"
    )

    return primary_df, secondary_df


def calc_dist_to_nearest(points_gdf, feature_dict):
    """
    This function reads in the feature shapefile and conducts
    the nearest point analysis on all points in points_gdf
    """

    for feature, feature_path in feature_dict.items():
        print(f"Caclulating distance to nearest {feature}")
        feature_gdf = gpd.read_file(feature_path)
        feature_gdf = feature_gdf.to_crs(points_gdf.crs)
        # calculate distance and convert from m to km
        points_gdf.loc[:, f'{feature}_dist'] = points_gdf.distance(feature_gdf.unary_union) / 1_000

    return points_gdf

def calc_share(values, totals):
    """
    This function divides a vector of values by a vector of totals to get the proportion.
    """

    return values / totals


def normalise_values(numbers):
    """
    This function takes in a list of numbers and normalises the data to lie between 0 and 1

    Args:
        numbers (list): A list of numbers to normalise.

    Returns:
        list: list of numbers normalised between 0 and 1
    """

    min_val = min(numbers)
    max_val = max(numbers)

    # avoid division by zero if all numbers are the same
    if min_val == max_val:
        return [0.0] * len(numbers)
    else:
        return [(x - min_val) / (max_val - min_val) for x in numbers]
    
def integer_encoding(strings, exclude_strings=[]):
    """
    Encodes a list of strings as integer values based on unique values
    exclude_strings is a list that can contain any values that should
    not be encoded and will be replaced with NaN
    """

    strings = strings.apply(lambda x: x if x not in exclude_strings else np.nan)
    encoded, mapping = strings.factorize()
    return encoded, mapping


if __name__ == "__main__":
    params = get_params()
    data_catalogue = load_data_catalogue()
    process_data(data_catalogue, params)
