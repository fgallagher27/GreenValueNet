# GreenValueNet

GreenValueNet is my attempt to use a neural network to value environmental ammenities using a hedonic pricing approach.  
This approach draws on the hedonic pricing model outlined in *'The Amenity Value of English Nature: A Hedonic Price Approach' (Gibbons, Mourato & Resende; 2013) (DOI: 10.1007/s10640-013-9664-9).*

## Model

### Structure

### Potential improvements

**Improvements to the NN architecture**

**Improvements to the dataset**

The dataset could be further modified to explain more of the variation in house prices and therefore hopefully improve the accuracy of the model. Computational power acts as a major constraint here. Potential modifications include:
- Widening the dataset from England to UK given many of the inputs cover regions of the UK besides England
- The house price dataset contained lots of variables relating to energy efficiency but indicators such as garden (or garden size) could be particularly relevant for valuing natural ammenities
- Using school catchment areas rather than striaght line distances for closest school, and including school quality explicitly (i.e. latest Ofsted inspection) as a measure of school quality

## Dataset

### Data Sources and processing

The input downloads can be found in `data_catalogue.yml` with more details below. Not all datasets can be included in the `download_data()` function so have been downloaded seperately. All are publicly available and links have been included below. These should be placed into a folder with filepath `data/raw_inputs` in order to run the model. If you want the processed data directly please get in touch. Any use of interim files should comply with the license agreements of all datasets below. The following datasets need to be downloaded seperately:
- Land cover
- National Parks
- Green spaces
- National Trust
- Rivers
- Roads
- Postcodes
- Schools
- Coastline

**Land cover**

For the land cover raster, an account with the UK Centre for Ecology and Hydrology is needed to download the map. I used the Land Cover Map 2017 (GB) 20m rasterised map.  
For licensing reasons I am not able to incorporate the downloading of this dataset within the data_processing script but it can be accessed by requesting download via the following link `https://order-eidc.ceh.ac.uk/orders/DS6HRVEC`.  
*Morton, R. D., Marston, C. G., O’Neil, A. W., & Rowland, C. S. (2020). Land Cover Map 2017 (20m classified pixels, GB) [Data set]. NERC Environmental Information Data Centre. https://doi.org/10.5285/F6F86B1A-AF6D-4ED8-85AF-21EE97EC5333*

**National Parks**

The national parks shapefile was downloaded from the Natural England Open Data Publication, a Defra group ArcGIS Online Organisation. It contains all the national parks of England as polygons.  
*https://naturalengland-defra.opendata.arcgis.com/datasets/national-parks-england/explore*

**National Trust**

The National Trust shapefile was downloaded from the National Trust Open Data site, with the Always open version being selected for use as a better measure of natural amendity than ticketed estates. These areas are stored as polygons.  
*https://open-data-national-trust.hub.arcgis.com/datasets/3511d41489ae442c877db40698b3b0c9_0/explore?location=52.824441%2C-2.103324%2C6.78 (Downloaded: 21/11/2023)*

**Roads**

The roads shapefiles are from the OS Data Hub and covers all raods in the UK. The shapefile is split into 100km^2 tiles.
*https://osdatahub.os.uk/downloads/open/OpenRoads*

**Postcodes**

The coordinates of the centre of each postcode are extracted from the OS Data Hub across 120 csv files in in EPSG:27700 projection.
*https://osdatahub.os.uk/downloads/open/CodePointOpen*

**Schools**

School location and ratings data was downloaded from the UK government data download portal page. This file should be saved with UTF-8 encoding.
*https://get-information-schools.service.gov.uk/Downloads*

**Coastline**

A UK coastline polygon is constructed from boundary lines polygons downloaded from the OS data hub.  
*https://osdatahub.os.uk/downloads/open/BoundaryLine*

**Ward characteristics**

The Generalised Land Use Database is used to get information at the ward level on:
- Domestic building density
- Non domestic building density
- Garden density
- Greenspace density
- Water density (i.e. proportion of land that is river/lake)
- Path density

`ONS_code` is matched to postcode using ONS UPRN Directory (May 2018) *Office for National Statistics licensed under the Open Government Licence v.3.0*  
*https://www.gov.uk/government/statistics/generalised-land-use-database-statistics-for-england-2005*

**House Prices**

House price data was obtained from the UK Data Service House Price per sq metre dataset.  
*Chi, Bin and Dennett, Adam and Oléron-Evans, Thomas and Morphet, Robin (2021). House Price per Square Metre in England and Wales, 1995-2021. [Data Collection]. Colchester, Essex: UK Data Service. 10.5255/UKDA-SN-855033*

**Travel To Work Areas (TTWAs)**

TTWAs are areas where >75% of the economically active population work in the area, and 75% of the working population live in the area. This shapefile contains TTWAs from the ONS geography geoportal.
*https://geoportal.statistics.gov.uk/datasets/ons::ttwa-dec-2001-generalised-clipped-boundaries-in-the-uk/explore*

**Train Stations**

This shapefile contains the locations of all train stations in the UK as points. Due to difficutlies assigning the missing CRS, this variable has been excluded for now
*Pope, Addy. (2017). GB Railways and stations. University of Edinburgh. https://doi.org/10.7488/ds/1773.*
