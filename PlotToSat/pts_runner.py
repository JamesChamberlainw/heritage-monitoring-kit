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
year_start = 2019
year_end = 2025 # year after******  so to get say 2019 to 2024 you would set year_start to 2019 and end to 2026 
# Constants (papermill outputs)
notebook_input = "F:/0-Projects/GEE-UNESCO/COMM514_A_12_202425/PlotToSat/pts_samarra.ipynb" # MAIN PC FILES
# notebook_input = "D:/1_Personal_Projects/COMM514_Diss/PlotToSat/pts_samarra.ipynb" # LAPTOP FILES 
output_dir = "executed_logs"
assets_folder = "projects/jameswilliamchamberlain/assets/activechunks_reduced"   
# =========================================================================================================================
index_start = 6     # if your using the same chunk_{index} naming structure else you will need to adjust to your needs!
offset_first = 2000 # if you need to resume  
# =========================================================================================================================



# Load asset list
assets = ee.data.listAssets({'parent': assets_folder}).get('assets', [])

# Ensure log dir
os.makedirs(output_dir, exist_ok=True)

# Iterate
for i, asset in enumerate(assets):
    # skip if index is below i 
    if (i + 1) < index_start:
        print(f"Skipping asset {(i + 1)} as it is below the index start of {index_start}.")
        continue

    # Standard behaviour:
    for year in range(year_start, year_end):

        if year < offset_first:
            print(f"Skipping year {year} for asset {(i + 1)} as it is below the offset first year of {offset_first}.")
            continue
        elif year == offset_first:
            print(f"Offset Correction: Starting from year {year} for asset {(i + 1)}.")
            print(f"Offset disabled for future assets, so will not skip any years after this point.")
            offset_first = -1 # Disables offset correction for future assets
        
        log_file = os.path.join(output_dir, f"chunk_{(i + 1)}_{year}.ipynb")

        print(f"Running: {notebook_input} for asset {(i + 1)}, year {year}. Log File: {log_file}")

        # ============================== FILL THIS IN TO YOUR SPECIFC LOCATION(S) ==============================
        # asset does contain the shpfilename so can be extracted from the asset - just unessusary in my case
        shp = {
            "shpfilename"        : f"projects/jameswilliamchamberlain/assets/activechunks_reduced/chunk_10{(i + 1)}",
            "polygonKeyColumn"   : "file_name" # column that ids of polygons are stored
        } 
        # ====================================================================================================== 

        print(f"Using SHP: {shp['shpfilename']}. for year {year}")

        pm.execute_notebook(
            input_path=notebook_input,
            output_path=log_file,
            parameters={"shp": shp, "year": year, "asset_index": (i + 1)}, 
            log_output=True
        )

        # remaining wait time 
        wait_until_idle(150, legacy=True, logtime=True) # 5 min between checks
        print(f"Notebook complete. Log saved to {log_file}") 