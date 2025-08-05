# Author: James William Chamberlain
# Date: 09/07/2025

"""
    Simple file for merining a full folder and sets of folder into a single csv file uising pandas
"""

import os
import pandas as pd



def merge_csv_files(folder_path, output_file, save=True, flag_overwrite=False, flag_raise_errors=True):
    """
    Merges all CSV files in the specified folder into a single CSV file.

    Args:
        folder_path (str):          Path to the input folder path containing CSV files.
        output_file (str):          Path to the output folder path containing the merged CSV file. 
                                    Ensure this contains a vailid file name, e.g. "merged_data.csv" 
                                    else this will raise an error. 
        save (bool):                If True, saves the merged DataFrame to the output file.
                                    Else returns the merged DataFrame without saving.
        flag_overwrite (bool):      If True, will overwrite the output file if it already exists. 
                                    Else raises an error if the output file already exisits and will not concatenate the files.
        flag_raise_errors (bool):   If True, raises errors for any issues encountered. Excluding overwrite errors.

    Returns:
        output_file_path (str):     Path to the output folder path containing the merged CSV file. 
    """

    if flag_overwrite is False and os.path.exists(output_file): raise FileExistsError(f"The output file '{output_file}' already exists. Set flag_overwrite=True to overwrite it.")
    
    # list of all files to concat 
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files: raise ValueError("No CSV files found in the specified folder. Please check the folder path.")

    # Read each CSV file and append its DataFrame to the list
    df = pd.DataFrame()
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        
        try:
            df_new = pd.read_csv(file_path)
            df = pd.concat([df, df_new], ignore_index=True)
        except Exception as e:
            if flag_raise_errors:
                raise ValueError(f"Error reading file '{file_path}': {e}")
            else:
                print(f"Warning: Could not read file '{file_path}'. Skipping this file.")

    if save:
        df.to_csv(output_file, index=False)
    else:
        return df
    return output_file


def merge_csv_files_from_multiple_folders(folder_list, output_file, save=True, flag_overwrite=False, flag_raise_errors=True, flag_mean_nan=False):
    """
    Merges all CSV files from multiple folders into a single CSV file.

    Args:
        folder_list (list):         List of folder paths containing CSV files.
        output_file (str):          Path to the output folder path containing the merged CSV file. 
                                    Ensure this contains a vailid file name, e.g. "merged_data.csv" 
                                    else this will raise an error. 
        flag_overwrite (bool):      If True, will overwrite the output file if it already exists        . 
                                    If False, raises an error if the output file already exisits and will not concatenate the files.
        flag_raise_errors (bool):   If True, raises errors for any issues encountered. Excluding overwrite errors.

    Returns:
        output_file_path (str): Path to the output folder path containing the merged CSV file. 
    """

    df = pd.DataFrame()

    # Collect all CSV files and merge them
    for folder_path in folder_list:
        try:
            df = pd.concat([df, merge_csv_files(folder_path, output_file, save=False, flag_overwrite=flag_overwrite, flag_raise_errors=flag_raise_errors)], ignore_index=True)#
        except Exception as e:
            print(f"Warning: Could not merge files from folder '{folder_path}'. Error: {e}, skipping this folder. This may result in an incomplete merged file.")

    # Overwrite cehck to avoid overwriting existing files
    if flag_overwrite is False and os.path.exists(output_file):
        raise FileExistsError(f"The output file '{output_file}' already exists. Set flag_overwrite=True to overwrite it.")
    
    # fill NaN values with the mean of the column if flag_mean_nan is True
    if flag_mean_nan:
        for col in df.columns:
            if df[col].isnull().any():
                avg = df[col].mean()
                df[col].fillna(avg, inplace=True)
    
    return df.to_csv(output_file, index=False) if output_file else df