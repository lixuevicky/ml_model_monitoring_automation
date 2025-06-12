import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 
output_model_path = os.path.join(config['output_model_path']) 
dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    for filename in os.listdir(test_data_path):
        if filename[-4:] == '.csv':
            data = pd.read_csv(test_data_path + '/' + filename)
    y = data['exited'].values.reshape(-1, 1)
    preds = model_predictions()

    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.savefig(output_model_path + '/confusionmatrix.png')
    return None

if __name__ == '__main__':
    score_model()
