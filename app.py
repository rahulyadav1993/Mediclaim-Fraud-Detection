import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import pandas as pd



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./temp_uploads/"
app.config['DOWNLOAD_FOLDER'] = "./temp_downloads/"
model=pickle.load(open('fraud_model.pkl','rb'))
model_imp = pickle.load(open('fraud_detect.pkl', 'rb'))

required_columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'RenalDiseaseIndicator',
       'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician',
       'AdmitForDays', 'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
       'ChronicCond_Cancer', 'ChronicCond_KidneyDisease',
       'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
       'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
       'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
       'ChronicCond_stroke', 'IPAnnualReimbursementAmt',
       'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
       'OPAnnualDeductibleAmt', 'WhetherDead', 'N_Types_Physicians',
       'IsDiagnosisCode', 'N_Procedure', 'N_UniqueDiagnosis_Claims']
y_col = "PotentialFraud"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model_imp.predict(final_features)

    output = ["Yes" if prediction==1 else "No"]

    return render_template('index.html', prediction_text='Fraud is:  {}'.format(output[0]))


@app.route('/file_predict', methods = ['POST'])  

def file_predict():  
    if request.method == 'POST':  
        f = request.files['file']
        df = pd.read_csv(f)
        time_tag = datetime.now().isoformat().split(".")[0].replace(":","_")
        df.to_csv(app.config['UPLOAD_FOLDER'] +  "input_file_"+ time_tag +".csv") 
        if (set(list(df.columns)) & set(required_columns)) == set(required_columns):
        	predict_df = df[required_columns]
        	x= predict_df.values
        	y_pred = model.predict(x)
            

        	x_df = pd.DataFrame(x,columns = required_columns)
        	x_df[y_col+"_predicted"] = list(y_pred)
        	
        	x_df.to_csv(app.config['DOWNLOAD_FOLDER'] +  "predicted_"+ time_tag +".csv")
        	return send_from_directory(app.config['DOWNLOAD_FOLDER'],"predicted_"+ time_tag +".csv", as_attachment=True)
        else:
        	print("some required columns not present in the file,")
        	print("required cols", str(required_columns))
        	error_string = "some required columns not present in the file,required cols: " + str(required_columns)
        	return render_template('index.html', prediction_text=error_string)
 


if __name__ == "__main__":
    app.run(debug=False)