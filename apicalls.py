import requests
import os
import json


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json','r') as f:
    config = json.load(f) 
 
test_data_path = os.path.join(config['test_data_path']) 
filepath = os.path.join(test_data_path,'testdata.csv')
output_model_path = os.path.join(config['output_model_path']) 

#Call each API endpoint and store the responses
response1 = requests.post(f"{URL}/prediction", files={'filename': open(filepath, 'rb')})
# response1 = requests.post(URL + '/prediction' + f'?filename={filepath}') #put an API call here
response2 = requests.get(f"{URL}/scoring") #put an API call here
response3 = requests.get(f"{URL}/summarystats") #put an API call here
response4 = requests.get(f"{URL}/diagnostics") #put an API call here

#combine all API responses
responses = {'Predictions': response1.text,
             'Scoring': response2.text,
             'Stats': response3.text,
             'Diagnostics': response4.json()}#combine reponses here

#write the responses to your workspace
output_name = 'apireturns.txt'
with open(os.path.join(output_model_path, output_name),'w') as f:
    json.dump(responses, f, indent=2)

