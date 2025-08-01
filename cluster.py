import os
import json
import fiona
import geemap
import numpy as np
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET

from collections import Counter

from shapely.geometry import shape
from shapely.geometry import Point

# Clustering (best for testing** due to speed)
from sklearn.cluster import KMeans

# tif file creation
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize

# Plotting and Vis 
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import re


def kmeans_clustering(df, k=10):
    """sklearn kmeans"""
    df_clean = df.copy()
    print(len(df_clean))
    print(f"Clustering {len(df_clean)} rows with {k} clusters.")
    print(df_clean.columns)
    df_clean = df_clean.dropna()  # May Crash if Empty Cells 
    print(len(df_clean))

    kmeans = KMeans(n_clusters=k, random_state=42) # TODO: make random_state random
    kmeans.fit(df_clean)
    
    return kmeans.labels_, df_clean.index.tolist()


class Cluster:
    """
        Takes a set of polygons and a set of points with attached geometry 

        and runs clustering over all points in the polygons, and returns a set of clusters based on the points.
    """

    sparse_matrix = pd.DataFrame()

    def __init__(self, subregions, df_data, mapping, passes=6, aoi=None, cluster_fn=kmeans_clustering, index_column="file_name", points=gpd.GeoDataFrame(), supress_warnings=False):
        """
            Initialises the cluster object with subregions, data, number of subregion passes, the area of interest (aoi), index column, and points.
        """
        # execution variables 
        self.passes = passes 
        self.cluster_fn = cluster_fn

        # Data + Ensure correct index 
        self.data = df_data
        try:
            self.data = self.data.set_index(index_column)
        except KeyError:
            if not supress_warnings:
                print(f"KeyError: {index_column} not found in DataFrame columns. Assuming default index is correct; this will not affect processing.")
        except Exception as e:
            if not supress_warnings:
                print(f"Error: {e}, this may affect and break processing. Please check your input DataFrame columns and index_column.")
        self.points = points
        self.mapping = mapping

        # Subregions 
        if aoi is None:
            aoi = subregions.dissolve()

        # subregions_poly = create_subregions(chunks)
        # subregions_poly = clip(subregions, aoi=aoi)
        self.subregions = subregions

        self.index_column = index_column
        self.UTM_ESPG = 32638
        self.EPSG = 4326

        self.labels = None
        self.sparse_matrix = pd.DataFrame()
    
    def clip_dataframe(self, polygon, df):
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
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=self.UTM_ESPG)
        gdf = gdf[gdf.geometry.within(polygon)]
        gdf = gdf.drop(columns=['geometry'])

        # convert back to DataFrame
        df_clipped = pd.DataFrame(gdf)

        return df_clipped
    
    def clip_polygon(chunks, aoi):
        """
            Clips the chunks or subregions to the area of interest (aoi).

            Parameters:
                chunks (GeoDataFrame):      The chunks or subregions to be clipped.
                aoi (GeoDataFrame):         The area of interest (aoi) to clip the chunks against.
        """

        # clip to aoi 
        if chunks.empty or aoi.empty:
            return gpd.GeoDataFrame(columns=chunks.columns.tolist(), crs=chunks.crs)
        clipped = gpd.clip(chunks, aoi)
        clipped = clipped[clipped.geometry.notnull()]  # Remove any null geometries

        return clipped.reset_index(drop=True)
    
    def split_df(self, df):
        """
            Creates two dataframes from a CSV file, one ready for clustering with additional columns removed and the other with all columns left intact.
        """

        df1 = df.copy()

        df1 = df1.dropna()
        df2_clear = df1.copy()
        df2_clear = df2_clear.drop(columns=["system:index", ".geo"])
        df2_clear = df2_clear.set_index(self.index_column)
        df1 = df1.set_index(self.index_column)
        df2_clear = df2_clear.apply(pd.to_numeric, errors='coerce')

        return df1, df2_clear
    
    def create_subregions(chunks, sift_percentage=0.5):
        """
            TODO: Replace with updated version above
        """

        if chunks.empty:
            return {}
        
        first_polygon = chunks.geometry.iloc[0]

        # take top two points of the polygon and get the length between them
        top_points = first_polygon.exterior.coords[:2]
        length_lon = abs(top_points[0][0] - top_points[1][0])

        # calculate the shift amount
        shift_amount = length_lon * sift_percentage

        shift_directions = {
            "left": (-shift_amount, 0),
            "right": (shift_amount, 0),
            "up": (0, shift_amount),
            "down": (0, -shift_amount),
            "top_left": (-shift_amount, shift_amount),
            "top_right": (shift_amount, shift_amount),
            "bottom_left": (-shift_amount, -shift_amount),
            "bottom_right": (shift_amount, -shift_amount),
        }

        subregions = []

        # Create subregions by shifting the geometries in all directions
        for _, (dx, dy) in shift_directions.items():
            gdf_shifted = chunks.copy()
            gdf_shifted["geometry"] = gdf_shifted["geometry"].translate(dx, dy)
            subregions.append(gdf_shifted)

        # Combine all into one GeoDataFrame
        subregions = pd.concat(subregions, ignore_index=True)
        return gpd.GeoDataFrame(subregions, crs=chunks.crs)
    
    @staticmethod
    def convert_df_to_geodf(df, geo_col='.geo', crs="EPSG:32638"):
        """
        Converts a DataFrame with a '.geo' column (GeoJSON strings) to a GeoDataFrame.
        """
        df = df.copy()

        if df.empty:
            return gpd.GeoDataFrame(columns=df.columns.tolist(), crs=crs)
        if df is type(gpd.DataFrame):
            return df 

        # Only parse if the entry is a string
        def safe_parse(x):
            if isinstance(x, str):
                try:
                    return shape(json.loads(x))
                except Exception as e:
                    print(f"[WARNING] Bad geometry skipped: {x[:30]}... ({e})")
            return None

        df['geometry'] = df[geo_col].apply(safe_parse)
        df = df[df['geometry'].notnull()]  # drop invalid rows

        return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
    
    ##################################################################################################################################################################
    # ================================================================== SAVE AND RESTORE ========================================================================== #
    ##################################################################################################################################################################

    def save_state(self, filname_prefix="temp_test/test2", filename_postfix="_state"):
        """
            Saves the current state of the cluster object to files for later use, only preserving class specifc data and not the spectral/time-series data.
        """


        # Save the Sparse Matrix (if exists)
        if self.sparse_matrix.empty:
            print("No sparse matrix to save.")
        else:
            df_sparse = self.sparse_matrix.copy()
            df_sparse = df_sparse.reset_index(drop=False) # Preserve index as its a uniuqe identifier external to class
            df_sparse.to_csv(f"{filname_prefix}{filename_postfix}_sparse_matrix.csv", index=False)


        # Save the Points and Labelled Data
        if self.points.empty:
            print("No points to save.")
        else:
            self.points.to_file(f"{filname_prefix}{filename_postfix}_points.geojson", driver='GeoJSON')


        # Save Labelled Data 
        if self.labels.empty:
            print("No assigned labels to save.")
        else:
            df_labels = self.labels.copy()
            df_labels = df_labels.reset_index(drop=False)
            df_labels = df_labels[[self.index_column, 'label']]  # Keep only core columns (.geo comes back later with data)
            df_labels.to_csv(f"{filname_prefix}{filename_postfix}_labels.csv", index=False) 


        # Save Subregions
        subregions = self.subregions.copy()
        if subregions.empty:
            print("No subregions to save.")
        else:
            subregions.to_file(f"{filname_prefix}{filename_postfix}_subregions.geojson", driver='GeoJSON')  

    def load_sparse_matrix(self, filename):
        """Loads a sparse matrix from a CSV file into the cluster object."""

        if not os.path.exists(filename):
            print(f"No sparse matrix file found at {filename}. Assuming no sparse matrix to load.")
            return

        df_sparse = pd.read_csv(filename)
        df_sparse = df_sparse.set_index(self.index_column)
        df_sparse = df_sparse.dropna()
        self.sparse_matrix = df_sparse


    def load_points(self, filename):
        """Loads Points from a GeoJSON file into the cluster object."""

        if not os.path.exists(filename):
            print(f"No points file found at {filename}. Assuming no points to load.")
            return

        with fiona.open(filename) as src:
            points = gpd.GeoDataFrame.from_features(src, crs=src.crs)
            points = points.dropna()
        self.points = points


    def load_labels_df(self, df):
        """Loads labels from a GeoDataFrame file into the cluster object."""

        data = self.data.copy()
        data = data.reset_index(drop=False)
        gdf_labels = df
        gdf_labels = gdf_labels.merge(data, on=self.index_column, how='inner') # ensures re-useabilty of labells on other data 
        gdf_labels['label'] = gdf_labels['label'].replace({np.NaN: None, "NaN": None, "None": None, None: None})
        gdf_labels = gdf_labels.set_index(self.index_column) 
        gdf_labels = gdf_labels.dropna()
        self.labels = gdf_labels


    def load_labels(self, filename):
        """Loads labels from a CSV file into the cluster object."""

        if not os.path.exists(filename):
            print(f"No labels file found at {filename}. Assuming no labels to load.")
            return

        df_labels = pd.read_csv(filename)
        df_labels = df_labels.dropna()
        self.load_labels_df(df_labels)
    

    def load_subregions(self, filename):
        """Loads subregions from a GeoJSON file into the cluster object."""

        if not os.path.exists(filename):
            print(f"No subregions file found at {filename}. Assuming no subregions to load or already loaded.")
            return

        with fiona.open(filename) as src:
            subregions = gpd.GeoDataFrame.from_features(src, crs=src.crs)
        self.subregions = subregions.dropna()

    def reload_state(self, filname_prefix="temp_test/test2", filename_postfix="_state"):
        """
            Reloads the state of the cluster object from files.
        """

    
        # Load Sparse Matrix (if exists)
        self.load_sparse_matrix(f"{filname_prefix}{filename_postfix}_sparse_matrix.csv")
        # if not os.path.exists(f"{filname_prefix}{filename_postfix}_sparse_matrix.csv"):
        #     print("No sparse matrix file found, no edits made. Assuming no sparse matrix to load.")
        # else:
        #     df_sparse = pd.read_csv(f"{filname_prefix}{filename_postfix}_sparse_matrix.csv")
        #     df_sparse = df_sparse.set_index("file_name")
        #     self.sparse_matrix = df_sparse
        

        # Load Points (if exists)
        self.load_points(f"{filname_prefix}{filename_postfix}_points.geojson")
        # if not os.path.exists(f"{filname_prefix}{filename_postfix}_points.geojson"):
        #     print("No points file found, no edits made. Assuming no points to load.")
        # else:
        #     dir_points = f"{filname_prefix}{filename_postfix}_points.geojson"
        #     with fiona.open(dir_points) as src:
        #             points = gpd.GeoDataFrame.from_features(src, crs=src.crs)
        #     self.points = points

        # Load Labels (if exists)
        self.load_labels(f"{filname_prefix}{filename_postfix}_labels.csv")
        # if not os.path.exists(f"{filname_prefix}{filename_postfix}_labels.csv"):
        #     print("No labels file found, no edits made. Assuming no labels to load.")
        #     # self.labels = pd.DataFrame(columns=["file_name", "label"])
        # else: 
        #     data = self.data.copy()
        #     data = data.reset_index(drop=False)
        #     gdf_labels = pd.read_csv(f"{filname_prefix}{filename_postfix}_labels.csv")
        #     gdf_labels = gdf_labels.merge(data, on='file_name', how='inner') # ensures re-useabilty of labells on other data 
        #     gdf_labels['label'] = gdf_labels['label'].replace({np.NaN: None, "NaN": None, "None": None})
        #     gdf_labels = gdf_labels.set_index("file_name") 
        #     self.labels = gdf_labels

        
        # Load Subregions (if exists)
        self.load_subregions(f"{filname_prefix}{filename_postfix}_subregions.geojson")
        # if not os.path.exists(f"{filname_prefix}{filename_postfix}_subregions.geojson"):
        #     print("No subregions file found, no edits made. Assuming no subregions to load or already loaded.")
        # else:
        #     dir_subregions = f"{filname_prefix}{filename_postfix}_subregions.geojson"
        #     with fiona.open(dir_subregions) as src:
        #         subregions = gpd.GeoDataFrame.from_features(src, crs=src.crs)
        #     self.subregions = subregions
        

    ##################################################################################################################################################################
    # ================================================================== EXPORT FUNCTIONS ========================================================================== #
    ##################################################################################################################################################################

    # generate tif 
    @staticmethod
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
        flag_filename = not os.path.basename(output_dir) or output_dir.endswith("/")
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


    def export_to_tif(self, df, bands, output_dir, res=50, UTM_ESPG=32638, EPSG=4326):
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
        file = self.os_path_chcker(output_dir, postfix=".tif", NAME_CODE_LIM=8, FLAG_APPEND_POSTFIX=True)


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
                fill=np.NaN,  # Fill with NaN values 
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


    def create_predictions(self):
        """
        Converts the sparse DataFrame of multiple cluster label columns to a single-column label prediction
        based on majority voting from known labels.

        Args:
            df_sparse (pd.DataFrame): DataFrame with 'cluster_label_*' columns and index as file_name.
            df_data (pd.DataFrame): Original data containing '.geo' and any other metadata.
            gdf_labelled (GeoDataFrame): GeoDataFrame with true labels and index as file_name.

        Returns:
            pd.DataFrame: DataFrame indexed by file_name, with predicted label (or None if unassignable).
        """

        df_sparse = self.sparse_matrix.copy()
        df_data = self.data.copy()
        gdf_labelled = self.labels.copy()

        # Ensure labels column exists
        if 'label' not in gdf_labelled.columns:
            raise ValueError("gdf_labelled must contain a 'label' column with ground truth labels.")

        predictions = pd.Series(index=df_sparse.index, dtype=object)

        for col in df_sparse.columns:
            if not col.startswith("cluster_label_"):
                continue

            # Get cluster IDs and their associated true labels
            cluster_ids = df_sparse[col]
            known_labels = gdf_labelled['label']

            # Build mapping: cluster_id -> list of known labels
            cluster_to_labels = {}
            for file_name, cluster_id in cluster_ids.items():
                if file_name in known_labels and pd.notna(known_labels[file_name]):
                    cluster_to_labels.setdefault(cluster_id, []).append(known_labels[file_name])

            # Compute majority label for each cluster
            cluster_to_majority = {}
            for cluster_id, labels in cluster_to_labels.items():
                if not labels:
                    cluster_to_majority[cluster_id] = None
                else:
                    label_counts = Counter(labels)
                    most_common = label_counts.most_common()
                    top_label = most_common[0][0] if len(most_common) == 1 or most_common[0][1] != most_common[1][1] \
                        else np.random.choice([l for l, c in most_common if c == most_common[0][1]])
                    cluster_to_majority[cluster_id] = top_label

            # Assign predicted label per row
            for idx in df_sparse.index:
                cluster_id = df_sparse.at[idx, col]
                label = cluster_to_majority.get(cluster_id, None)
                if pd.isna(predictions.at[idx]) and label is not None:
                    predictions.at[idx] = label

        # Create output DataFrame
        predictions = pd.DataFrame({'predicted_label': predictions})

        # Join metadata like `.geo` if needed
        if '.geo' in df_data.columns:
            predictions = predictions.join(df_data['.geo'])

        return predictions


    def create_map(self, filename=""):
        predictions = self.create_predictions()

        # tif creation
        predictions['numeric_label'] = predictions['predicted_label'].map(self.mapping)
        output_path = self.os_path_chcker(filename)
        self.export_to_tif(predictions, bands=['numeric_label'], output_dir=output_path)

    ##################################################################################################################################################################
    # =============================================================== PRE-PROCESSING =============================================================================== #
    ##################################################################################################################################################################

    def update_row_labels(self, labels_gdf, label_row='label'):
        """Simple QoL function to call build_row_labels and be more intuitive."""

        return self.build_row_labels(label_row=label_row, labels_gdf=labels_gdf, update=True)


    def build_row_labels(self, label_row='label', additional_labels=None, update=False):
        """
        Builds a DataFrame with labelled polygons based on the points in the GeoDataFrame.

        Args:
            df (GeoDataFrame): The GeoDataFrame containing the points.
            labels (GeoDataFrame): The GeoDataFrame containing the polygons and their labels.
            additional_labels (GeoDataFrame, optional): Additional labels to be added, will be merged with the main labels. # MUST BE POINTS # 
            update (bool, optional): If True, will use additional_labels to update existing set, rather than regenerating from scratch. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame with the labels for each point.
        """

        # Prepare and Ensure in GeoDataFrame format
        labels_gdf = gpd.GeoDataFrame(columns=["geometry", "label"], geometry="geometry")
        df_gdf = self.data.copy()
        if df_gdf is type(pd.DataFrame):
            df_gdf = self.convert_df_to_geodf(df_gdf, geo_col='.geo', crs=f"EPSG:{self.UTM_ESPG}")

        # Load in Points and Labels
        if not update:
            labels_gdf = self.points.copy()
            df_gdf['label'] = None
            # merge with additional labels if provided

        # Prepare additional_labels 
        if additional_labels is not None and not additional_labels.empty:
            additional_points_copy = additional_labels.copy()
            additional_points_copy = additional_points_copy.to_crs(self.points.crs)
            labels_gdf = pd.concat([labels_gdf, additional_points_copy], ignore_index=True)

            # merges points as these are now additional labels
            self.points = pd.concat([self.points, additional_points_copy], ignore_index=True)

            # reload already existing labels to ensure they are not lost
            df_gdf = self.labels.copy()

        # check for intersections and assign labels
        for _, label_row in labels_gdf.iterrows():
            print(f"progress: {_+1}/{len(labels_gdf)} labels processed.")
            label = label_row['label']
            
            # Check each point until assigned or dropped
            for idx, point in df_gdf.iterrows():
                if label_row.geometry.intersects(point.geometry):
                    df_gdf.at[idx, 'label'] = label
                    break

        self.labels = df_gdf

        return df_gdf
    
    ##################################################################################################################################################################
    # ================================================================ Label Recomendations ======================================================================== #
    ##################################################################################################################################################################

    def create_recommendations(self, filename="/"):
        """
            Based on the Points Creates a map of recommendations for labelling the points in the subregions. 

            The higher the value the higher the recommendation to label the point. 
        """

        # resolution (m)
        res = 50

        # labels 
        labels = self.points.copy()
        labelled_data = self.labels.copy()
        data = self.data.copy()

        # # height map from truth labels 
        # max_height = 100.0
        # data['distance_value'] = 0.0 # old is 0.0

        # for _, label_geometry in labels.iterrows():
        #     print(f"progress: {_}/{len(labels)} labels processed.")
        #     label = label_geometry['label']
        #     if pd.isna(label):
        #         continue
            
        #     # Calculate distance map where lowest values are the points closest to the label geometry
        #     distances = data.geometry.distance(label_geometry.geometry)
        #     # data['distance_value'] += (max_height - distances) / max_height
        #     data['distance_value'] += distances / max_height

        # # normalise by the number of labels
        # if len(labels) > 0:
        #     data['distance_value'] /= len(labels)

        # # flip vlaues by max to create a recommendation map
        # data['distance_value'] = max_height - data['distance_value']

        max_height = 100.0
        data['distance_value'] = 0.0 

        for _, label_geometry in labels.iterrows():
            print(f"progress: {_}/{len(labels)} labels processed.")

            distances = data.geometry.distance(label_geometry.geometry) 

            # if distance is greater than 500m away ignore so set to 0 
            distances[distances > 500] = 0.0

            # normal around the point from 100 to 0.0 at edge
            data['distance_value'] += (max_height - distances) / max_height

            # take max value when compared between data['distance_value'] and the new value
            data['distance_value'] = data['distance_value'].combine(data['distance_value'], max)  
            
        try:
            self.export_to_tif(data, bands=['distance_value'], output_dir=self.os_path_chcker(filename, postfix=".tif", NAME_CODE_LIM=8, FLAG_APPEND_POSTFIX=True), res=50, UTM_ESPG=self.UTM_ESPG, EPSG=self.EPSG)
        except Exception as e:
            print(f"Error: {e}, \n Could not export recommendations to TIF, Returing DataFrame, so you can try again (This may be filename related!)")

        return data
            
    ##################################################################################################################################################################
    # ========================================================================== CLUSTERING ======================================================================== #
    ##################################################################################################################################################################
        
    def fit(self):
        """
            Fits the clustering model to the data.
            This method should be implemented in subclasses.
        """

        subregions_data = []

        for _, subregion in self.subregions.iterrows():
            # clip the data to the subregion
            # based on .geo column drop all rows that do not intersect with the polygon

            df_clipped = self.clip_dataframe(subregion.geometry, self.data)
            df_clipped = df_clipped.dropna()
            df_clipped = df_clipped.drop(columns=["system:index", ".geo"]) # unessusary columns for clustering
            # df_clipped = df_clipped.set_index("file_name")
            df_clipped = df_clipped.apply(pd.to_numeric, errors='coerce')

            if df_clipped.empty:
                print("No data points found in this subregion.")
                continue

            subregions_data.append(df_clipped)

        matrix = pd.DataFrame()
        index_number = 0

        for subregion_df in subregions_data:

            for i in range(self.passes):
                # Perform clustering
                labels, indecies = self.cluster_fn(subregion_df)

                col_name = f'cluster_{index_number}'

                # Temporary DataFrame
                temp_df = pd.DataFrame({col_name: labels, self.index_column: indecies})

                # Group in case of duplicates
                temp_grouped = temp_df.groupby(self.index_column)[col_name].first()

                # Convert to DataFrame for merging
                temp_grouped = temp_grouped.to_frame()

                print(f"temp columns: {temp_grouped.columns}")

                # Merge with main matrix
                matrix = matrix.join(temp_grouped, how='outer') if not matrix.empty else temp_grouped

                index_number += 1
                print(len(matrix.columns), "columns in the matrix after clustering.")
                print(len(matrix), "rows in the matrix after clustering.")

        matrix = matrix.join(self.data[[".geo"]], how='outer')
        self.sparse_matrix = matrix

        return matrix
    