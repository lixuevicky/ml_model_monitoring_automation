# from flask import Flask, session, jsonify, request
import pandas as pd
# import numpy as np
import pickle
import os
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
import json
from sklearn.metrics import f1_score



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model(testdata = None, model = None):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    os.makedirs(output_model_path, exist_ok=True)
    if testdata is None:
        for filename in os.listdir(test_data_path):
            if filename[-4:] == '.csv':
                testdata = pd.read_csv(test_data_path + '/' + filename)
    if model is None:
        for filename in os.listdir(output_model_path):
            if filename[-4:] == '.pkl':
                with open(output_model_path + '/' + filename, 'rb') as f:
                    model = pickle.load(f)
    
    X = testdata[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y = testdata['exited'].values.reshape(-1, 1).ravel()

    predicted = model.predict(X)
    f1score = f1_score(predicted,y)

    with open(output_model_path + '/' + "latestscore.txt", "w") as f:
        f.write(str(f1score))

    print(f"F1 score {f1score:.7f} written to {output_model_path}")
    return f1score

if __name__ == '__main__':
    score_model()
