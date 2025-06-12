# from flask import Flask, session, jsonify, request
# import pandas as pd
# import numpy as np
# import pickle
import os
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path'])

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    os.makedirs(prod_deployment_path, exist_ok=True)

    for file in ['/trainedmodel.pkl', '/latestscore.txt']:
        shutil.copy(output_model_path + file, prod_deployment_path + file)

    for file in ['/ingestedfiles.txt']:
        shutil.copy(dataset_csv_path + file, prod_deployment_path + file)

    return True

if __name__ == '__main__':
    store_model_into_pickle()




        
        
        

