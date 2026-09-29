from flask import Flask, render_template, request, jsonify
import sqlite3, pickle, json, os, pandas as pd

app=Flask(__name__)
MODEL_PATH='models/model.pkl'; DB='claims.db'
with open(MODEL_PATH,'rb') as f: model=pickle.load(f)
with open('models/metrics.json') as f: metrics=json.load(f)

FIELDS=['Month','WeekOfMonth','DayOfWeek','Make','AccidentArea','DayOfWeekClaimed','MonthClaimed','WeekOfMonthClaimed','Sex','MaritalStatus','Age','Fault','PolicyType','VehicleCategory','VehiclePrice','PolicyNumber','RepNumber','Deductible','DriverRating','Days_Policy_Accident','Days_Policy_Claim','PastNumberOfClaims','AgeOfVehicle','AgeOfPolicyHolder','PoliceReportFiled','WitnessPresent','AgentType','NumberOfSuppliments','AddressChange_Claim','NumberOfCars','Year','BasePolicy']

def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    c.execute('''CREATE TABLE IF NOT EXISTS claims(id INTEGER PRIMARY KEY AUTOINCREMENT, claim_name TEXT, age INTEGER, vehicle_category TEXT, policy_type TEXT, fraud_probability REAL, prediction INTEGER, status TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)'''); c.commit(); return c

def make_record(form):
    defaults={'Month':'Jan','WeekOfMonth':2,'DayOfWeek':'Monday','Make':'Honda','AccidentArea':'Urban','DayOfWeekClaimed':'Monday','MonthClaimed':'Jan','WeekOfMonthClaimed':2,'Sex':'Male','MaritalStatus':'Married','Age':35,'Fault':'Policy Holder','PolicyType':'Sedan - Collision','VehicleCategory':'Sedan','VehiclePrice':'20000 to 29000','PolicyNumber':99999,'RepNumber':5,'Deductible':400,'DriverRating':2,'Days_Policy_Accident':'1 to 7','Days_Policy_Claim':'8 to 15','PastNumberOfClaims':'none','AgeOfVehicle':'3 years','AgeOfPolicyHolder':'31 to 35','PoliceReportFiled':'No','WitnessPresent':'No','AgentType':'External','NumberOfSuppliments':'none','AddressChange_Claim':'no change','NumberOfCars':'1 vehicle','Year':1996,'BasePolicy':'Collision'}
    for k in defaults:
        if k in form and form[k] != '': defaults[k]=form[k]
    for k in ['WeekOfMonth','WeekOfMonthClaimed','Age','PolicyNumber','RepNumber','Deductible','DriverRating','Year']: defaults[k]=int(defaults[k])
    return pd.DataFrame([{k:defaults[k] for k in FIELDS}])

@app.route('/')
def home(): return render_template('index.html', metrics=metrics)

@app.route('/predict',methods=['POST'])
def predict():
    form=request.form if request.form else request.get_json(force=True)
    row=make_record(form); p=float(model.predict_proba(row)[0,1]); pred=int(p>=0.5)
    status='Flagged for Further Review' if pred else 'Likely Genuine'
    name=form.get('claim_name','Demo Claim')
    c=db(); c.execute('INSERT INTO claims(claim_name,age,vehicle_category,policy_type,fraud_probability,prediction,status) VALUES(?,?,?,?,?,?,?)',(name,int(row.iloc[0]['Age']),row.iloc[0]['VehicleCategory'],row.iloc[0]['PolicyType'],p,pred,status)); c.commit(); c.close()
    result={'prediction':pred,'probability':round(p*100,2),'status':status}
    if request.is_json: return jsonify(result)
    return render_template('result.html', result=result, row=row.iloc[0].to_dict())

@app.route('/api/predict',methods=['POST'])
def api_predict():
    return predict()

@app.route('/dashboard')
def dashboard():
    c=db(); rows=c.execute('SELECT * FROM claims ORDER BY id DESC').fetchall(); total=len(rows); flagged=sum(r['prediction'] for r in rows); genuine=total-flagged
    return render_template('dashboard.html',rows=rows,total=total,flagged=flagged,genuine=genuine,metrics=metrics)

@app.route('/api/metrics')
def api_metrics(): return jsonify(metrics)

if __name__=='__main__':
    db(); app.run(debug=True,host='0.0.0.0',port=int(os.getenv('PORT',5000)))
