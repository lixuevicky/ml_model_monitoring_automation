
import pandas as pd
import numpy as np
import timeit
import os
import pickle
import json
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(data = None):
    #read the deployed model and a test dataset, calculate predictions
    if data is None:
        for filename in os.listdir(test_data_path):
            if filename[-4:] == '.csv':
                data = pd.read_csv(test_data_path + '/' + filename)

    X_data = data[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)

    for filename in os.listdir(prod_deployment_path):
        if filename[-4:] == '.pkl':
            with open(prod_deployment_path + '/' + filename, 'rb') as f:
                model = pickle.load(f)
    
    return list(model.predict(X_data)) #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    res = []
    for filename in os.listdir(dataset_csv_path):
        if filename[-4:] == '.csv':
            data = pd.read_csv(dataset_csv_path + '/' + filename)
    data = data.drop(columns='Unnamed: 0').select_dtypes('number')

    res.append(['medians: ', data.median(axis=0)])
    res.append(['means: ', data.mean(axis=0)])
    res.append(['stds: ', data.std(axis=0)])

    return res #return value should be a list containing all summary statistics

def missing_data():
    #calculate missing val percentages here
    for filename in os.listdir(dataset_csv_path):
        if filename[-4:] == '.csv':
            data = pd.read_csv(dataset_csv_path + '/' + filename)

    rows = len(data)
    return [nans/rows for nans in data.isna().sum()] #return value should be a list containing all missing val percentages

    
##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    res = []
    for file in ['ingestion.py', 'training.py']:
        starttime = timeit.default_timer()
        os.system('python3 ' + 'ingestion.py')
        timing=timeit.default_timer() - starttime
        res.append(timing)

    return res #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    with open('outdated_packages.txt', 'wb') as f:
        f.write(outdated)
    return outdated


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
