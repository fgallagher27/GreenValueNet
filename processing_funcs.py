"""
This script contains the functions to process the input data
Data processing can be conducted by executing process_data()
The following functions are defined:

    * process_data
    * process_housing_data
    * process_spatial_attr
    
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from typing import List, Union
from shapely.ops import nearest_points
from data_load_funcs import get_params, get_file_path, load_data_catalogue, process_spatial_dict

cwd = Path.cwd()
def process_data(catalogue: dict, params: dict):
    """
    This function processes the input data
    """
    path = cwd / "data" / "interim_files" / catalogue['interim_files']['dataset']
    if os.path.exists(path):
        print("Processed dataset already exists.\nLoading exisitng dataset...")
        df = pd.read_csv(path)
    else:
        print("Processing data to create dataset...")
        hp = process_housing_data(catalogue, params)
        spa = process_spatial_attr(catalogue, params)
        df = pd.merge(hp, spa, on='postcode', how='left')

        folder = cwd / "data" / "interim_files"
        if not os.path.exists(folder):
            os.makedirs(folder)

        df.to_csv(folder / "dataset.csv")

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
            
            for col in ['propertytype', 'oldnew', 'duration', 'construction_age_band']:
                hp[col] = integer_encoding(hp[col])
            chunked_list.append(hp)

        hp_full = pd.concat(chunked_list, ignore_index=True)

        hp_full.to_csv(clean_file_path, index=False)
    
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
    clean_path = cwd / "data" / "processed_inputs" / params['spatial_attributes']['processed_file']
    if os.path.exists(clean_path):
        spatial_attributes = gpd.read_file(clean_path)
    else:
        spatial_mapping = gpd.read_file(
            cwd / "data" / "processed_data" / "mapped_postcodes.shp"
        )

        ward_chars = process_glud(catalogue)

        process_school_data(catalogue)

        spatial_dict = process_spatial_dict(params)
        # Apply function over all shapefiles in the feature dict
        spatial_attributes = calc_dist_to_nearest(spatial_mapping, spatial_dict)

        keep = []
        spatial_attributes = spatial_attributes.loc[:, keep]
        spatial_attributes = spatial_attributes.merge(ward_chars, on='', how='left')

        lu_props = process_lu_rast()

        spatial_attributes.to_file(
            clean_path,
            driver = "ESRI Shapefile"
        )


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
        cwd / "data" / "processed_data" / "ward_chars.csv",
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
        'PhaseOfEducation (name)': 'school_type'}, inplace=True)
    schools = schools[schools['school_type'].isin(['Primary', 'Secondary'])]

    # now convert to a shapefile
    postcodes_path = get_file_path(catalogue, 'inputs', 'postcodes')
    postcodes = gpd.read_file(postcodes_path)
    schools_shp = schools.merge(postcodes, on='postcode', how='left')

    primary_df = schools_shp[schools_shp['school_type'] == 'Primary']
    secondary_df = schools_shp[schools_shp['school_type'] == 'Secondary']

    primary_df.to_file(
        cwd / "data" / "processed_inputs" / "primary_school.shp",
        driver = "ESRI Shapefile"
    )

    secondary_df.to_file(
        cwd / "data" / "processed_inputs" / "secondary_school.shp",
        driver = "ESRI Shapefile"
    )

    return primary_df, secondary_df

def process_lu_rast(catalogue: dict, pararms:dict):

    # now process land use raster
    lu_rast = get_file_path(catalogue, 'inputs', 'land_cover')
    clean_path = cwd / "data" / "processed_inputs" / params['land_use']['processed_file']
    if os.path.exists(clean_path):
        lu_props = pd.read_file(clean_path)
    else:
        # TODO update and move to Params
        legend = {1: 'Urban', 2: 'Forest', 3: 'Water', 4: 'Agriculture', 5: '...'}
        target_grid_size = params['target_grid']

        with rasterio.open(lu_rast) as src:
            land_use = src.read(1)
            pix_w, pix_h = src.transofrm.a, src.transform.e
        pixel_dim = {
            'height': pix_h,
            'width': pix_w
        }
        
        lu_props = calc_rast_props(land_use, legend, target_grid_size, pixel_dim)
    
    return lu_props_out



    

def nearest_point(point, spatial_index, other_gdf):
    """
    This function takes a point and a spatial indx from
    other_gdf and calculates the nearest point in other_gdf to point"""
    possible_matches_index = list(spatial_index.intersection(point.bounds))
    possible_matches = other_gdf.iloc[possible_matches_index]
    nearest_geom = nearest_points(point, possible_matches.unary_union)[1]
    nearest = possible_matches.geometry == nearest_geom
    return possible_matches[nearest]


def calc_dist_to_nearest(points_gdf, feature_dict):
    """
    This function reads in the feature shapefile and conducts
    the nearest point analysis on all points in points_gdf
    """
    for feature, feature_path in feature_dict.items():
        feature_gdf = gpd.read_file(feature_path)

        # check for matching CRS
        if feature_gdf.crs != points_gdf.crs:
            feature_gdf = feature_gdf.to_crs(points_gdf.crs)
        feature_spatial_index = feature_gdf.sindex

        nearest_points = points_gdf.geometry.apply(lambda x: nearest_point(x, feature_spatial_index, feature_gdf))
        nearest_points = nearest_points.geometry
        points_gdf[f'{feature}_dist'] = points_gdf.geometry.distance(nearest_points)


def calc_rast_props(raster_data, legend, target_grid_size, pixel_dim: dict):
    """
    This function takes in a raster as a numpy array and uses the legend and pixel_dim
    to calculate the proportion of each pixel in target grid size that is each legend
    category
    """
    pixel_w, pixel_h = pixel_dim['width'], pixel_dim['height']
    target_grid_pixels = int(target_grid_size / pixel_w)
    # Reshape the raster data into non-overlapping target grid cells
    reshaped_data = raster_data.reshape(
        raster_data.shape[0] // target_grid_pixels, target_grid_pixels,
        raster_data.shape[1] // target_grid_pixels, target_grid_pixels
    )

    # Count the occurrences of each land use category in each grid cell
    counts = np.zeros((len(legend), reshaped_data.shape[0], reshaped_data.shape[2]), dtype=int)
    for value, category in legend.items():
        counts[value - 1] = (reshaped_data == value).sum(axis=(1, 3))

 
    total_pixels = counts.sum(axis=0)
    proportions_grid_cell = counts / total_pixels
    proportions = proportions_grid_cell.sum(axis=(1, 2))
    total_cells = reshaped_data.shape[0] * reshaped_data.shape[2]
    proportions /= total_cells

    return proportions

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
    
def integer_encoding(strings):
    """
    Encodes a list of strings as integer values based on unique values
    """
    encoded = strings.factorize()[0]
    return encoded


if __name__ == "__main__":
    params = get_params()
    data_catalogue = load_data_catalogue()
    process_data(data_catalogue, params)
