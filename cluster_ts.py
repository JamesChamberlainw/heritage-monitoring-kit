import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

import cluster as cl
from cluster import kmeans_clustering

class cluster_ts():

    cluster_list = []

    yearly_labels_dict = []
    common_labels = pd.DataFrame(columns=["year_start", "year_end", "label", "geometry"])
    
    data_dir = pd.DataFrame(columns=["dir", "year"])

    start_year = 2019
    end_year = 2024

    def __init__(self, ts_point_labels, data_dir, mapping, start_year, end_year, subregions, cluster_class=cl.Cluster, index_column='file_name', passes=6, full_data=True):
        """
            Initializes the ts_cluster object with time-series point labels, data directory, mapping, and number of passes.
        """
        
        self.cluster_class = cluster_class
        
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.full_data = full_data
        self.index_column = index_column

        self.data_dir = data_dir
        self.subregions = subregions

        if start_year > end_year:
            raise ValueError("start_year must be less than or equal to end_year")
        
        common_labels, uncommon_labels = self.__reduce_to_common__(ts_point_labels)
        self.common_labels = common_labels
        self.yearly_labels_dict = {
            year: self.__split_year__(year, uncommon_labels)
            for year in range(start_year, end_year + 1)
        }

        # runtime vars 
        self.mapping = mapping
        self.passes = passes

    def __reduce_to_year__(year, label_df):
        """
            Returns a DataFrame with only entries that overlap with the specified year.

        Args:
            year (int):                The year to filter the labels DataFrame.
            label_df (GeoDataFrame):   The labels DataFrame to filter.

        Returns:
            GeoDataFrame:              A DataFrame containing only the labels that overlap with the specified year.
        """

        year_df = label_df[(label_df['start_year'] <= year) & (label_df['end_year'] >= year)]
        return year_df.reset_index(drop=True)


    def __reduce_to_common__(self, labels_df):
        """
        Splits the labels DataFrame into two parts:
        1. Common labels that fall within the specified start and end years.
        2. Uncommon labels that do not fall within the specified range, and may be year specific.
        
        Args:
            labels_df (GeoDataFrame):   The labels DataFrame to reduce.
            start_year (int):           The starting year for the reduction.
            end_year (int):             The ending year for the reduction.
        
        Returns:
            GeoDataFrame:               The Reduced labels DataFrame.
        """
        common_df = labels_df[(labels_df['start_year'] <= self.start_year) & (labels_df['end_year'] >= self.end_year)]
        uncommon_df = labels_df[(labels_df['start_year'] > self.start_year) | (labels_df['end_year'] < self.end_year)]
        return common_df.reset_index(drop=True), uncommon_df.reset_index(drop=True)  


    @staticmethod
    def __split_year__(year, label_df):
        """
            Returns a DataFrame of where any label that overlaps a given year is returned.
        """

        year_df = label_df[(label_df['start_year'] <= year) & (label_df['end_year'] >= year)]
        # year_df['year'] = year
        return year_df.reset_index(drop=True)

    def __build_all_geometries__(self):
        """
            Builds a all Geometries DataFrame from the data labels
        """
        uniques = pd.DataFrame(columns=[self.index_column, '.geo'])
        for i in self.data_dir.index:
            df_dir = self.data_dir.loc[i, 'dir']
            # print(f"Loading Geometries {df_dir}...")

            df = pd.read_csv(df_dir)
            df = df[[self.index_column, '.geo']]
            print(f"Loaded {len(df)} geometries from {df_dir}.")

            df = df.dropna(subset=[self.index_column, '.geo'])
            df = df.reset_index(drop=True)
            
            uniques = pd.concat([uniques, df], ignore_index=True)
            uniques = uniques.drop_duplicates(subset=[self.index_column])

            # no need to check all if we know the data IS FULL!
            if self.full_data:
                return uniques
            
        return uniques

    def __build_common_labels__(self):
        """
            Builds a common cluster from the common labels DataFrame. That ALL years share to avoid redundant processing.
        """

        if self.common_labels.empty:
            return pd.DataFrame(columns=["file_name", "label"]) # empty DataFrame 

        uniuqes_gdf = self.__build_all_geometries__()
        uniuqes_gdf['geometry']  = uniuqes_gdf['.geo'].apply(lambda x: shape(json.loads(x)))
        uniuqes_gdf = gpd.GeoDataFrame(uniuqes_gdf, geometry='geometry', crs="EPSG:4326")

        cl_cluster = cl.Cluster(subregions=self.subregions, df_data=None, mapping=None, index_column=self.index_column)
        cl_cluster.data = uniuqes_gdf.set_index("file_name")
        # cl_cluster.points = common_df.drop(columns=['start_year', 'end_year'])    
        cl_cluster.points = self.common_labels.drop(columns=['start_year', 'end_year'])

        cl_cluster.build_row_labels()

        # Select only the necessary columns
        base_labels = cl_cluster.labels.copy()
        base_labels = base_labels.reset_index(drop=False)
        self.base_labels = pd.DataFrame(base_labels, columns=['file_name', 'label'])
        return self.base_labels

    def instansiate_clusters(self, cluster_class, aoi=None, index_column="file_name"):
        """
            Instantiates clusters for each year in the data directory.

            Args:
                cluster_class (type):           cluster class of the cluster to instantiate.
                aoi (GeoDataFrame, optional):   The area of interest to use for the cluster. If None, the entire area is used.
                index_column (str, optional):   The name of the column to use as the index. Default is "file_name".
        """
        
        base_labelles = self.__build_common_labels__()
        
        cluster_list = []

        for year in range(self.start_year, self.end_year + 1):
            print(f"Instantiating cluster for year {year}...")
            data = pd.read_csv(self.data_dir.loc[self.data_dir['year'] == year, 'dir'].values[0])

            # (self, subregions, df_data, mapping, passes=6, aoi=None, index_column="file_name", points=gpd.GeoDataFrame()):
            cl_cluster = cluster_class(subregions=self.subregions, 
                                          df_data=data, 
                                          mapping=self.mapping, 
                                          points=self.common_labels,
                                          aoi=aoi,
                                          index_column=index_column,
                                          passes=self.passes,
                                          supress_warnings=True) # ONLY warnings that exist at the current time are errors that should only occur outside of this
            
            # load in the common shared labels 
            df_copy = base_labelles.copy()
            cl_cluster.load_labels_df(df=df_copy)
            
            # update labels with the non-shared labels 
            uncommon_year_df = self.yearly_labels_dict[year]
            uncommon_year_df = uncommon_year_df.drop(columns=['start_year', 'end_year'])
            print(f"TRYING row labels for year {year} with {len(uncommon_year_df)} uncommon labels. columns: {uncommon_year_df.columns.tolist()}")
            cl_cluster.build_row_labels(self.index_column, additional_labels=uncommon_year_df, update=True)

            cluster_list.append(cl_cluster)

        self.cluster_list = cluster_list


    def fit(self, save_state=""):
        """
            Fits all clusters in the cluster list.

            Args:
                save_state (str):         The prefix for the filename to save the state of the clusters. If empty, no state is saved.
        """

        for i, cl_cluster in enumerate(self.cluster_list):
            print(f"Fittingitted cluster for year {self.start_year + i}")

            cl_cluster.fit()

        if save_state is not "":
            self.save_states(filename_prefix=save_state)

    def update_labels(self, labels_df):
        """
            Updates the labels of all clusters in the cluster list with the provided labels DataFrame.

            Args:
                labels_df (GeoDataFrame):  The DataFrame containing the new labels to update.
        """

        for i, cl_cluster in enumerate(self.cluster_list):
            print(f"Updating labels for cluster {self.start_year + i}")
            cl_cluster.update_row_labels(labels_gdf=labels_df, label_row=self.index_column)
                
    def create_map(self, filename_prefix="/cluster/clusters", formats=["tif"]):
        """
            Predicts the labels for all clusters in the cluster list.

            Args:
                save_state (str):         The prefix for the filename to save the state of the clusters. If empty, no state is saved.
        """

        for i, cl_cluster in enumerate(self.cluster_list):
            print(f"Predicting / Creating Map's based on Clusters for year {self.start_year + i}")

            if filename_prefix is not "":
                if "tif" in formats or ".tif" in formats:
                    cl_cluster.create_map(filename=f"{filename_prefix}_{self.start_year + i}.tif")
            else: 
                raise ValueError("filename_prefix must be a non-empty string to save the state of the clusters. e.g. 'cluster_' to create 'cluster_2024.tif'.")
            
    def build_recomendation(self, filename_prefix="clusters"):
        """Builds recommendations for all clusters"""

        for i, cl_cluster in enumerate(self.cluster_list):
            print(f"Building recommendations for cluster {self.start_year + i} with {len(cl_cluster.labels)} tiles.")

            # build the recommendations
            if cl_cluster.predictions.empty:
                print("Creating predictions for cluster first...")
                cl_cluster.create_predictions() # MUST be done first else will not work 
            cl_cluster.create_recommendations(export_filename=f"/{filename_prefix}_{self.start_year + i}_recomendation.tif")


    def save_states(self, filename_prefix="clusters/cluster_"):
        """
            Saves the state of all clusters in the cluster list to files.
        """
        
        for i, cl_cluster in enumerate(self.cluster_list):
            print(f"Saving state of cluster {i + self.start_year} with {len(cl_cluster.labels)} labels.")
            cl_cluster.save_state(filname_prefix=filename_prefix, filename_postfix=f"{self.start_year + i}_state")


    def load_states(self, filename_prefix="clusters/cluster", aoi=None):
        """
            Reloads clusters from saved states. 
        """

        cluster_list = []

        for year in range(self.start_year, self.end_year + 1):
            print(f"Instantiating cluster for year {year}...")
            data = pd.read_csv(self.data_dir.loc[self.data_dir['year'] == year, 'dir'].values[0])

            # (self, subregions, df_data, mapping, passes=6, aoi=None, index_column="file_name", points=gpd.GeoDataFrame()):
            cl_cluster = self.cluster_class(subregions=self.subregions, 
                                            df_data=data, 
                                            mapping=self.mapping, 
                                            points=self.common_labels,
                                            aoi=aoi,
                                            index_column=self.index_column,
                                            passes=self.passes,
                                            supress_warnings=True)
            
            # reload the data from the state files 
            cl_cluster.reload_state(filname_prefix=filename_prefix, filename_postfix=f"{year}_state")

            cluster_list.append(cl_cluster)

        self.cluster_list = cluster_list
