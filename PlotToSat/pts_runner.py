# Author: James William Chamberlain
# Date: 12-06-2025 

import ee
import os
import time
import papermill as pm
from pts_check4usage import wait_until_idle

ee.Authenticate()
ee.Initialize(project="jameswilliamchamberlain")

# ===================================== FILL THIS IN TO YOUR SPECIFIC REGION AND YEARS =====================================
# Define study region / countries MUST BE DEFINED IN THE NOTEBOOK AS IMPASSABLE PARAMETER
# countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')     # POLYGON DEFINED DIRECLTY IN FILE
# polygon = countries.filter(ee.Filter.eq('country_na', 'Iraq')) # POLYGON DEFINED DIRECLTY IN FILE
# Define time range (years)
year_start = 2017
year_end = 2025
# Constants (papermill outputs)
# notebook_input = "F:/0-Projects/GEE-UNESCO/COMM514_A_12_202425/PlotToSat/pts_samarra.ipynb" # MAIN PC FILES
notebook_input = "D:/1_Personal_Projects/COMM514_Diss/PlotToSat/pts_samarra.ipynb" # LAPTOP FILES 
output_dir = "executed_logs"
assets_folder = "projects/jameswilliamchamberlain/assets/activechunks_06"   
# =========================================================================================================================
index_start = 3     # if your using the same chunk_{index} naming structure else you will need to adjust to your needs!
year_start = 2019   # if you need to resume from a specific year, else you can set this to the `year_start` or earlier 
# =========================================================================================================================



# Load asset list
assets = ee.data.listAssets({'parent': assets_folder}).get('assets', [])

# Ensure log dir
os.makedirs(output_dir, exist_ok=True)

# Iterate
for i, asset in enumerate(assets):
    # skip if index is below i 
    if i < index_start:
        print(f"Skipping asset {i} as it is below the index start of {index_start}.")
        continue

    # Standard behaviour:
    for year in range(year_start, year_end):
        # Skip if year is below the year_start
        if year_start > year:
            print(f"Skipping year {year} as it is below the year start of {year_start}.")
            continue

        log_file = os.path.join(output_dir, f"chunk_{i}_{year}.ipynb")

        print(f"Running: {notebook_input} for asset {i}, year {year}. Log File: {log_file}")

        # ============================== FILL THIS IN TO YOUR SPECIFC LOCATION(S) ==============================
        # asset does contain the shpfilename so can be extracted from the asset - just unessusary in my case
        shp = {
            "shpfilename"        : f"projects/jameswilliamchamberlain/assets/activechunks_06/chunk_{i}",
            "polygonKeyColumn"   : "file_name" # column that ids of polygons are stored
        }
        # ====================================================================================================== 

        print(f"Using SHP: {shp['shpfilename']}. for year {year}")

        pm.execute_notebook(
            input_path=notebook_input,
            output_path=log_file,
            parameters={"shp": shp, "year": year, "asset_index": i}, 
            log_output=True
        )

        # expected earliest finish 

        # remaining wait time 
        wait_until_idle(150, legacy=True, logtime=True) # 5 min between checks
        print(f"Notebook complete. Log saved to {log_file}") 