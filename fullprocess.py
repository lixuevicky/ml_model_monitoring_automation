
import os
import json
import sys
import pickle
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import logging
import pandas as pd
logging.basicConfig(filename='logs.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = os.path.join(config['output_folder_path']) 
##################Check and read new data
#first, read ingestedfiles.txt
with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
f.close()
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_files = os.listdir(os.path.join(input_folder_path))
print(input_files)
if (lines == input_files):
    logging.info('No new data found - exiting')
    sys.exit(0)

new_data = ingestion.merge_multiple_dataframe()
##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r', encoding='utf-8') as f:
    lastest_score = float(f.read())

with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as m:
    model = pickle.load(m)

f1score = scoring.score_model(testdata = new_data, model = model)
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if f1score >= lastest_score:
    logging.info("No model drift found.")
    sys.exit(0)

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
logging.info('Training new model')
model = training.train_model()

logging.info('Deploying new model')
deployment.store_model_into_pickle()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
logging.info('Diagnostics')
os.system("python diagnostics.py")
logging.info('Reporting')
os.system("python reporting.py")
logging.info('Apicalls')
os.system("python apicalls.py")






