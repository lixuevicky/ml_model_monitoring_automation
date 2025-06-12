import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    all_dfs = []
    os.makedirs(output_folder_path, exist_ok=True)
    ingested_list_path = os.path.join(output_folder_path, "ingestedfiles.txt")
    # (re)create the ingestedfiles.txt

    for filename in os.listdir(input_folder_path):
        if filename.lower().endswith('.csv'):
            csv_path = os.path.join(input_folder_path, filename)
            data = pd.read_csv(csv_path)
            all_dfs.append(data)
            # record which files we ingested
            with open(ingested_list_path, "a") as f:
                f.write(filename + '\n')

    if not all_dfs:
        print(f"No CSV files found in {input_folder_path}.")
        return
    
    df = pd.concat(all_dfs, ignore_index=True)
    df_unique = df.drop_duplicates()
    df_unique.to_csv(output_folder_path + '/' + 'finaldata.csv')
    return df_unique
    

if __name__ == '__main__':
    merge_multiple_dataframe()
