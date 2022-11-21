import os
from flask import Flask, render_template, request
import numpy as np
import pickle
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "bsWv_lABYnj3nfgIHYWsuEomzdpXEVt8_UWMnvWO2j2d"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__)
model = pickle.load(open('CKD.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('Home.html')

@app.route('/index',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        bu = float(request.form['bu'])
        bgr = float(request.form['bgr'])
        cad = float(request.form['cad'])
        ane = float(request.form['ane'])
        pc = float(request.form['pc'])
        rbc = float(request.form['rbc'])
        dm = float(request.form['dm'])
        pe = float(request.form['pe'])

        values = np.array([[rbc,pc,bgr,bu,pe,ane,dm,cad]])
        payload_scoring = {"input_data": [{"fields": ["bu","bgr","cad","ane","pc","rbc","dm","pe"], "values": values}]}

        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/489bfc2d-47c4-46e3-a46a-1e5901b7d816/predictions?version=2022-11-21', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})
        print("Scoring response")

        prediction=response_scoring.json()
        print(prediction)
        print('Final prediction Result',prediction['prediction'][0]['values'][0][0])
        #prediction = model.predict(values)

        return render_template('result.html', showcase='Result '+str(prediction))


if __name__ == "__main__":
    app.run(debug=True)


# NOTE: manually define and pass the array(s) of values to be scored in the next line




