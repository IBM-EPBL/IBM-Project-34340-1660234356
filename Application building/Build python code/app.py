import numpy as np
import pandas as pd 
from flask import Flask,request,render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('CKD.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/Prediction',methods=['Post','Get'])
def prediction():
    return render_template('predictor.html')
@app.route('/Home',methods=['Post','Get'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=['Post'])
def predict():
    input_feature = [float(x) for x in request.form.values()]
    features_value = [np.array(input_feature)]

    features_name = ['blood_urea','blood glucose random','coronary_artery_disease',
    'anemia','pus_cell','red_blood_cells','diabetesmellitus','pedal_edema']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    return render_template('result.html',prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)