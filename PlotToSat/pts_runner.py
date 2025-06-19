# Author: James William Chamberlain
# Date: 12-06-2025 

import ee
import os
import papermill as pm
from pts_check4usage import wait_until_idle

ee.Authenticate()
ee.Initialize(project="jameswilliamchamberlain")

# ===================================== FILL THIS IN TO YOUR SPECIFC REGION AND YEARS =====================================
# Define study region / countries MUST BE DEFINED IN THE NOTEBOOK AS IMPASSABLE PARAMETER
# countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
# polygon = countries.filter(ee.Filter.eq('country_na', 'Iraq'))
# Define time range (years)
year_start = 2017
year_end = 2025
# Constants (papermill outputs)
notebook_input = "F:/0-Projects/GEE-UNESCO/COMM514_A_12_202425/PlotToSat/pts_samarra.ipynb"
output_dir = "executed_logs"
assets_folder = "projects/jameswilliamchamberlain/assets/activechunks_06"   
# =========================================================================================================================

# Load asset list
assets = ee.data.listAssets({'parent': assets_folder}).get('assets', [])

# Ensure log dir
os.makedirs(output_dir, exist_ok=True)

# Iterate
for i, asset in enumerate(assets):
    for year in range(year_start, year_end):
        log_file = os.path.join(output_dir, f"chunk_{i+1}_{year}.ipynb")

        print(f"Running: {notebook_input} for asset {i+1}, year {year}. Log File: {log_file}")

        # ============================== FILL THIS IN TO YOUR SPECIFC LOCATION(S) ==============================
        # asset does contain the shpfilename so can be extracted from the asset - just unessusary in my case
        shp = {
            "shpfilename"        : f"projects/jameswilliamchamberlain/assets/activechunks_06/chunk_{i+1}",
            "polygonKeyColumn"   : "file_name" # column that ids of polygons are stored
        }
        # ====================================================================================================== 

        print(f"Using SHP: {shp['shpfilename']}. for year {year}")

        pm.execute_notebook(
            input_path=notebook_input,
            output_path=log_file,
            parameters={"shp": shp, "year": year, "asset_index": i + 1}, 
            log_output=True
        )

        wait_until_idle(900, legacy=True) # 15 min between checks
        print(f"Notebook complete. Log saved to {log_file}") 