import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape
import json

# tif file creation
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import re



def prep_data(dir):
    """Creates two dataframes from a CSV file, one ready for clustering with additional columns removed and the other with all columns left intact."""
    df1 = pd.read_csv(dir)
    df1 = df1.dropna()
    df2_clear = df1.copy()
    df2_clear = df2_clear.drop(columns=["system:index", ".geo"])
    df2_clear = df2_clear.set_index("file_name")
    df1 = df1.set_index("file_name")
    df2_clear = df2_clear.apply(pd.to_numeric, errors='coerce')

    return df1, df2_clear


def extract_band_time_series_array(df):
    """
    Convert a DataFrame with 'month_band' columns (e.g., '0_B1') into a 2D NumPy array
    of shape (n_bands, 12), where each row is a time series (12 months) for one band.
    
    Assumes input DataFrame contains clean data in a {month_index}_{band_name} format and nothing else! 
    """
    col_names = df.columns.tolist()
    
    # Extract unique bands and months
    bands = sorted(set(name.split('_')[1] for name in col_names if '_' in name))
    months = sorted(set(int(name.split('_')[0]) for name in col_names if '_' in name))

    print("Bands:", bands)
    print("Months:", months)

    df_final = df.copy()
    # add empty column 
    df_final['ts_matrix'] = None

    for row in df.index:
        row_df = df.loc[[row]]
        arr_bands = np.zeros((len(bands), len(months)))
        
        # Extract the time series for each band
        for band in bands:
            # temp_df = df.filter(like=f"_{band}")
            temp_df = row_df.filter(regex=fr"_{band}$")
            for month in months:
                col_name = f"{month}_{band}"
                if col_name in temp_df.columns:
                    arr_bands[bands.index(band), months.index(month)] = temp_df[col_name].values[0]
                else:
                    arr_bands[bands.index(band), months.index(month)] = np.nan
                    print(f"Column {col_name} not found in temp_df, setting to NaN")

        # save bkac into origina 
        df_final.at[row, 'ts_matrix'] = arr_bands
        
    return df_final


def disp_labels(df, labels, output_img="plt.png", output_tif="plt.tif", UTM_EPSG="EPSG:32638", EPSG="EPSG:4326", FLAG_SHOW=True):
    # GeoDataFrame creation
    gdf = gpd.GeoDataFrame({
        "clusters": labels,
        "geometry": df[".geo"].apply(lambda x: shape(json.loads(x)))
    }, crs="EPSG:4326")
    gdf = gdf.to_crs(UTM_EPSG) # for some reason does not work at ESPG:4326 instead need ot use UTM_EPSG for Iraq 

    # Plot
    fig, ax = plt.subplots(figsize=(30, 30), dpi=300)
    gdf.plot(column="clusters", ax=ax, cmap="tab10", edgecolor="none", linewidth=0, antialiased=False)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_img, dpi=600)

    if FLAG_SHOW:
        plt.show()
    else:
        plt.close()

def os_path_chcker(output_dir, postfix=".tif", NAME_CODE_LIM=8, FLAG_APPEND_POSTFIX=True):
    """
        Ensures that the output directory is valid, exists and can be written to.

        This is a simple function that does three things:
            1. Checks if the directory exists, if not creates it.
            2. Checks if the file name is valid (and not empty, else creates a unique filename).
            3. (Optional) Appends a postfix to the file name if it does not already end with it.

    Args:
        output_dir (str): The directory path to check.
        postfix (str, optional): The postfix to append to the file name if it does not already end with it. Default is ".tif".
        NAME_CODE_LIM (int, optional): The length of the hex code to generate for filename. Default is 8.
        FLAG_APPEND_POSTFIX (bool, optional): Whether to append the postfix if the file name does not end with it. Default is True.
    """


    # path existance check
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
        print(f"Created directory: {os.path.dirname(output_dir)}")


    # file name existance check
    flag_filename = not os.path.basename(output_dir)
    while flag_filename:
        print("Generating new file name...")
        # assign unique name to the tif file
        new_file_name = f"{os.urandom(NAME_CODE_LIM).hex()}{postfix}"
        if not os.path.exists(output_dir + new_file_name):
            output_dir += new_file_name
            flag_filename = False
            print(f"New File Name: {output_dir}")


    # check labelled correctly 
    if FLAG_APPEND_POSTFIX and not output_dir.endswith(postfix):
        print(f"Output directory {output_dir} does not end with .tif, appending .tif")
        output_dir += {postfix}
        
    return output_dir


def export_to_tif(df, bands, output_dir, res=50, UTM_ESPG=32638, EPSG=4326):
    """
        Exports a DataFrame generated by PlotToSat to a GeoTIFF file. 

        Note: df files MUST CONTAIN:
        - `.geo` column with GeoJSON geometries.
        - `file_name` column with unique identifiers for each row.
        - 1 Unique band labelled in `bands` list - this CAN BE LABELLED CLUSTERS. 

        Expected formats:
            Time-Series data should be in a DataFrame with 'month_band' columns (e.g., '0_B1', '1_B2', etc.).
            if a band is provideda e.g.m B1 but there are no pre-fix values this data will be assumed to be a single band AS-IS, and will be included in the output just as that band alone. 

        Args:
            df (pd.DataFrame): PlotToSat Style pandas DataFrame containing time-series data or single band data. Must have file_name and `.geo` geometry columns.
            bands (list): List of band names to include in the output. e.g., ['B1', 'B2', 'B3'] or ['SingleBand', etc.] can be mixed with single band data.
            output_dir (str): Output file directory for the GeoTIFF.
            res (int, optional): Resolution of the output raster in meters. Default is 50m.
            UTM_ESPG (int, optional): EPSG code for the UTM coordinate reference system. Default is 32638.
            


            # EPSG (int, optional): EPSG code for the coordinate reference system.
    """

    # Ensure output directory is valid, exists and can be written to.
    file = os_path_chcker(output_dir, postfix=".tif", NAME_CODE_LIM=8, FLAG_APPEND_POSTFIX=True)


    # Sort the Bands into single and time-series band data. 
    column_heads = df.columns.tolist()

    
    # all available bands in the data
    band_columns = [col for col in column_heads if any(col.endswith(band) for band in bands)]


    # Acceptable bands to process 
    acceptable_lst = []
    for col in bands:
        if any(col.startswith(f"{i}_") for i in range(12)):
            # must be non-prefix and only one band of that ts 
            if col not in acceptable_lst and any(coli for coli in band_columns if coli == col):
                acceptable_lst.append(col)
            # acceptable_lst.append(col)
        else:
            # must exist in band_columns and take all 
            appended = False
            for i in range(12):
                if any(coli for coli in band_columns if coli == f"{i}_{col}"):
                    if f"{i}_{col}" not in acceptable_lst:
                        # add only if not already in list     
                        acceptable_lst.append(f"{i}_{col}")
                        appended = True

            if appended == False:
                # if not appended must be a unique column 
                acceptable_lst.append(col)
    
    band_columns = acceptable_lst

 
    geometry = df[".geo"].apply(lambda x: shape(json.loads(x)))
    gdf = gpd.GeoDataFrame(df[band_columns].copy(), geometry=geometry, crs=f"EPSG:{EPSG}") # EPSG not UTM_ESPG else will raise an error

    gdf_utm = gdf.to_crs(epsg=UTM_ESPG)

    minx, miny, maxx, maxy = gdf_utm.total_bounds
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)

    transform = from_origin(minx, maxy, res, res)

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster dimensions (width={width}, height={height}). Check CRS and resolution.")

    # Build Raster Stack
    rasters = []
    for band in band_columns:
        values = gdf_utm[band]
        shapes = ((geom, val) for geom, val in zip(gdf_utm.geometry, values))
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            dtype="float32",
            fill=np.NaN,  # Fill with NaN values for NO FUCKING DATA
        )

        rasters.append(raster)

    raster_stack = raster_stack = np.stack(rasters, axis=0) if len(rasters) >= 2 else rasters

    # Save rasters to GeoTIFF 
    with rasterio.open(
        file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(band_columns),
        dtype="float32",
        crs=f"EPSG:{UTM_ESPG}",
        transform=transform,
        nodata=np.NaN,  # Set NoData value to NaN
    ) as dst:
        for i, band in enumerate(band_columns):
            dst.write(raster_stack[i], i + 1)
            dst.set_band_description(i + 1, band) # Keep band discription!!!


def clip(polygon, df, UTM_ESPG=32638, EPSG=4326):
    """
        Clips the DataFrame based on the .geo column and a given polygon.   
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to clip the DataFrame to.
        df (pd.DataFrame): The DataFrame containing the geometries to be clipped.
        
    Returns:
        pd.DataFrame: The clipped DataFrame.
    """

    # based on .geo column drop all rows that do not intersect with the polygon
    df['geometry'] = df['.geo'].apply(lambda x: shape(json.loads(x)))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=UTM_ESPG)
    gdf = gdf[gdf.geometry.within(polygon)]
    gdf = gdf.drop(columns=['geometry'])

    # convert back to DataFrame
    df_clipped = pd.DataFrame(gdf)

    return df_clipped


def clip_to_polygon(polygon, df, utm_epsg=32638, wgs_epsg=4326):
    """
    Clips the input DataFrame to rows whose `.geo` geometry intersects the given polygon.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to clip to.
        df (pd.DataFrame): DataFrame with a '.geo' column (GeoJSON format).
        utm_epsg (int): EPSG code of the UTM projection used for filtering.
        wgs_epsg (int): EPSG code of the incoming geometry (usually EPSG:4326).

    Returns:
        pd.DataFrame: Subset of df with geometries intersecting the polygon.
    """
    # Reproject the polygon to UTM
    polygon_proj = gpd.GeoSeries([polygon], crs=f"EPSG:{utm_epsg}").to_crs(epsg=utm_epsg).iloc[0]

    # Convert .geo strings into Shapely geometries
    df = df.copy()
    df['geometry'] = df['.geo'].apply(lambda x: shape(json.loads(x)))

    # Create GeoDataFrame in UTM projection
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=f"EPSG:{utm_epsg}")

    mask = gdf.geometry.intersects(polygon_proj)

    gdf_in = gdf[mask].copy()
    gdf_in['file_name'] = gdf_in.index

    gdf_out = gdf[~mask].copy()
    gdf_out['file_name'] = gdf_out.index

    # # Spatial filtering: keep rows that intersect the polygon
    # gdf_clipped_in = gdf[gdf.geometry.intersects(polygon_proj)].copy()
    # gdf_clipped_in['file_name'] = gdf_clipped_in.index  # Keep the original index as a column

    # # any gdf_clipped_in != gdf_clipped_out file_name
    # gdf_clipped_out = gdf[~gdf.geometry.intersects(polygon_proj)].copy()
    # gdf_clipped_out['file_name'] = gdf_clipped_out.index
    
    return gdf_in, gdf_out

def summarise(df):
    """
        Creates a summary Dataframe for all columns in the Dataframe. 

    Args:
        df (pd.DataFrame): The DataFrame to summarise.

    Returns:
        pd.DataFrame: A summary DataFrame with the summary statistics mean, median, and standard deviation. 
    """

    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')

    # Compute summary statistics
    summary = numeric_df.agg(['mean', 'median', 'std']).T

    # Flatten the DataFrame to a single row with renamed columns
    summary = summary.T.stack().to_frame().T
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]

    return summary

def create_figs(df):
    """
        Creates figures to summarise the DataFrame. 

        Assumes the DataFrame is contains summary statistics only, each must contain a mean, median and standard deviation columns.
    """

    months = [f"{i}" for i in range(12)] # 0 to 11

    # extract only band name and sort
    bands = set(re.sub(r'^(mean_|std_|median_)?\d{1,2}_', '', col) for col in df.columns)       # Split into bands 
    bands = sorted(set(bands), key=lambda b: (int(re.search(r'\d+', b).group()), b))            # Sort 

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    axes = axes.flatten()

    z_matrix = np.zeros((len(bands), len(bands))) 
    global_mean_max = 0 
    global_std_max = 0

    for i, band in enumerate(bands):
        means = []
        stds = []
        valid_months = []

        for j, month in enumerate(months):
            mean_col = f"mean_{month}_{band}"
            std_col = f"std_{month}_{band}"
            
            if mean_col in df.columns and std_col in df.columns:
                means.append(df[mean_col].values[0])
                stds.append(df[std_col].values[0])
                valid_months.append(int(month))

                # max values for global max
                global_std_max = max(global_std_max, df[std_col].values[0])
                global_mean_max = max(global_mean_max, df[mean_col].values[0]) 

                z_matrix[i][j] = df[mean_col].values[0]  
            else:
                z_matrix[i][j] = 0 # np.nan
            
        if not means:
            continue

        means = np.array(means)
        stds = np.array(stds)
        
        ax = axes[i]
        ax.plot(valid_months, means, marker='o', label="Mean", color=cm.viridis(0.1))
        ax.set_ylim(0, global_mean_max + global_std_max * 1.1)  # Set y-limits based on global mean max and std
        ax.fill_between(valid_months, means - stds, means + stds, alpha=0.2, label="±1 Std Dev", color=cm.viridis(0.1))
        ax.set_title(band)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months)
        ax.set_xlabel("Months")
        ax.set_xlim(0, 11)


    Z = np.array(z_matrix)
    X = np.arange(Z.shape[1])
    Y = np.arange(Z.shape[0])

    X, Y = np.meshgrid(X, Y)
    fig_3d = plt.figure(figsize=(12, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    surf = ax_3d.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none')
    ax_3d.set_xlabel('Months')
    ax_3d.set_xticks(range(len(months)))
    ax_3d.set_xticklabels(months)
    ax_3d.set_yticks(range(len(bands)))
    ax_3d.set_yticklabels(bands)
    ax_3d.set_ylabel('Bands')
    ax_3d.set_zlabel('Mean Values')
    # ax_3d.set_title("3D Pane showing mean values across bands and months")
    fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5)
    plt.show()