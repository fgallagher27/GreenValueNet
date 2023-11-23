"""
This script contains the functions needed to combine downloads that are multiple files into one file.
It can be imported as a module into the data download and clean scripts
It contains the following functions:

    * concat_postcodes
    * concat_roads
    * make_coastline

"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from data_load_funcs import load_data_catalogue

cwd = Path.cwd()

def pre_processing(processed_files_lib: dict):
    """
    This function checks a dictionary of files and functions.
    It checks if the file exists, and if not executes the associated
    pre-processing function.
    """

    print("Beginning pre-processing of data...")
    # check for existing files
    for file, func in processed_files_lib.items():
        if not os.path.exists(cwd / "data" / "processed_inputs" / file):
            func()
        else:
            print(f"{file} has already been pre-processed")
    print("Pre-processing completed.")


def concat_postcodes():
    """
    This function combines the postcode csv files into one file.
    It also drops excess columns and filter for english postcodes only
    """
    print("Preparing postcode data...")
    postcode_folder = cwd / "data" / "raw_inputs" / "postcodes"

    csv_files = [f for f in os.listdir(postcode_folder) if f.endswith('.csv')]
    dfs = []
    cols = ['postcode', 'eastings', 'northings']

    # TODO refactor using ThreadPoolExecutor to speed up
    for file in csv_files:
        file_path = os.path.join(postcode_folder, file)

        if file == "Code-Point_Open_Column_Headers.csv":
            pass
        else:
            df = pd.read_csv(file_path, header=None).iloc[:, [0,2,3,4]]
            eng = df[df.iloc[:, -1] == 'E92000001']
            eng = eng.drop(eng.columns[-1], axis=1)
            dfs.append(eng)

    postcodes_full = pd.concat(dfs, ignore_index=True, axis=0)
    postcodes_full.columns = cols

    # convert to shapefile
    geometry = [Point(xy) for xy in zip(postcodes_full['eastings'], postcodes_full['northings'])]
    gdf = gpd.GeoDataFrame(postcodes_full, geometry=geometry, crs="EPSG:27700")
    gdf = gdf.to_crs("EPSG:3857").drop(columns=['eastings', 'northings'])
    gdf.to_file(
        cwd / "data" / "processed_inputs" / "postcodes_c.shp",
        driver="ESRI Shapefile"
    )


def concat_roads():
    """
    This function binds together the road shapefiles.
    It also filters for main roads and drops excess columns.
    """
    print("Preparing road data...")
    road_folder = cwd / "data" / "raw_inputs" / "roads"

    road_files = [f for f in os.listdir(road_folder) if f.endswith('RoadLink.shp')]
    main_roads = ['A road', 'Motorway']
    cols = ['identifier', 'class', 'geometry']

    def clean_road_subset(file_name, folder, main_roads, cols):
        """
        Cleans an individual road shapefile
        """
        file_path = os.path.join(folder, file_name)
        road_shp = gpd.read_file(file_path)
        road_shp = road_shp[road_shp['class'].isin(main_roads)]
        road_shp = road_shp[cols]
        return road_shp

    partial_subset = partial(
        clean_road_subset,
        folder=road_folder,
        main_roads=main_roads,
        cols=cols
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        shps = list(
            executor.map(
                partial_subset,
                road_files
            )
        )

    road_shp_full = pd.concat(shps, ignore_index=True, axis=0)
    road_shp_full.to_file(
        cwd / "data" / "processed_inputs" / "roads_c.shp",
        driver="ESRI Shapefile"
    )


def make_coastline():
    """
    This function creates a boundary outline of the UK coastline from the regional polygons
    """
    print("Preparing coastline data...")
    regional_boundary = gpd.read_file(cwd /"data" / "raw_inputs" / "english_region_region.shp")
    england = regional_boundary.dissolve().to_crs("EPSG:3857")
    coastline = england['geometry'].boundary
    coastline.to_file(
        cwd / "data" / "processed_inputs" / "coastline.shp",
        driver="ESRI Shapefile"
    )


def match_ons_postcode():
    """
    This function maps ONS codes used in GLUD to postcodes.
    It used the ON postcode directory file to map
    """
    print("Mapping postcodes to census wards...")
    catalogue = load_data_catalogue()
    postcode_path = (
        cwd / 
        "data" /
        catalogue['inputs']['postcodes']['location'] /
        catalogue['inputs']['postcodes']['file_name']
    )

    mapping_path = (
        cwd / 
        "data" /
        catalogue['inputs']['glud_mapping']['location'] /
        catalogue['inputs']['glud_mapping']['file_name']
    ) 
    postcodes = gpd.read_file(postcode_path)
    mapping = pd.read_csv(mapping_path)
    mapping = mapping.loc[:, ['pcd', 'statsward']]
    mapped = pd.merge(postcodes, mapping, left_on = "postcode", right_on = "pcd", how="left").dropna()
    mapped.to_csv(cwd / "data" / "processed_data" / "mapped_postcodes.shp")


if __name__ == "__main__":
    # TODO move this to a yml file to hold parameters
    processed_files_lib = {
        "postcodes_c.shp": concat_postcodes,
        "roads_c.shp": concat_roads,
        "coastline.shp": make_coastline,
        "mapped_postcodes.shp": match_ons_postcode
    }
    pre_processing(processed_files_lib)
