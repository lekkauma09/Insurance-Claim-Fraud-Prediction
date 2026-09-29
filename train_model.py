import os, json, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

DATA = 'data/fraud_oracle.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORICAL = ['Month','DayOfWeek','Make','AccidentArea','DayOfWeekClaimed','MonthClaimed','Sex','MaritalStatus','Fault','PolicyType','VehicleCategory','VehiclePrice','Days_Policy_Accident','Days_Policy_Claim','PastNumberOfClaims','AgeOfVehicle','AgeOfPolicyHolder','PoliceReportFiled','WitnessPresent','AgentType','NumberOfSuppliments','AddressChange_Claim','NumberOfCars','BasePolicy']
NUMERIC = ['WeekOfMonth','WeekOfMonthClaimed','Age','PolicyNumber','RepNumber','Deductible','DriverRating','Year']
TARGET='FraudFound_P'

# This demo generator is ONLY a local fallback when the Kaggle CSV is not yet downloaded.
def make_demo(n=2500, seed=42):
    rng=np.random.default_rng(seed)
    choices=lambda vals: rng.choice(vals,n)
    df=pd.DataFrame({
      'Month':choices(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
      'WeekOfMonth':rng.integers(1,6,n),'DayOfWeek':choices(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']),
      'Make':choices(['Honda','Toyota','Ford','Mazda','Chevrolet','Pontiac','BMW','Mercedes','Nissan']),
      'AccidentArea':choices(['Urban','Rural']),'DayOfWeekClaimed':choices(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']),
      'MonthClaimed':choices(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),'WeekOfMonthClaimed':rng.integers(1,6,n),
      'Sex':choices(['Male','Female']),'MaritalStatus':choices(['Single','Married','Widow','Divorced']), 'Age':rng.integers(18,81,n),
      'Fault':choices(['Policy Holder','Third Party']),'PolicyType':choices(['Sedan - Liability','Sedan - Collision','Sedan - All Perils','Sport - Collision','Sport - Liability','Utility - All Perils']),
      'VehicleCategory':choices(['Sedan','Sport','Utility']),'VehiclePrice':choices(['less than 20000','20000 to 29000','30000 to 39000','40000 to 59000','60000 to 69000','more than 69000']),
      'PolicyNumber':np.arange(1,n+1),'RepNumber':rng.integers(1,17,n),'Deductible':choices([300,400,500,700]),'DriverRating':rng.integers(1,5,n),
      'Days_Policy_Accident':choices(['none','1 to 7','8 to 15','15 to 30','more than 30']), 'Days_Policy_Claim':choices(['none','8 to 15','15 to 30','more than 30']),
      'PastNumberOfClaims':choices(['none','1','2 to 4','more than 4']), 'AgeOfVehicle':choices(['new','2 years','3 years','4 years','5 years','6 years','7 years','more than 7']),
      'AgeOfPolicyHolder':choices(['16 to 17','18 to 20','21 to 25','26 to 30','31 to 35','36 to 40','41 to 50','51 to 65','over 65']),
      'PoliceReportFiled':choices(['No','Yes']),'WitnessPresent':choices(['No','Yes']),'AgentType':choices(['External','Internal']),
      'NumberOfSuppliments':choices(['none','1 to 2','3 to 5','more than 5']), 'AddressChange_Claim':choices(['no change','under 6 months','1 year','2 to 3 years','4 to 8 years']),
      'NumberOfCars':choices(['1 vehicle','2 vehicles','3 to 4','5 to 8','more than 8']),'Year':rng.integers(1994,1997,n),'BasePolicy':choices(['Liability','Collision','All Perils'])
    })
    # synthetic demonstration label: intentionally imbalanced and based on claim-risk signals
    risk=(df['Fault'].eq('Third Party').astype(int)*1.0 + df['PoliceReportFiled'].eq('No').astype(int)*0.5 + df['WitnessPresent'].eq('No').astype(int)*0.35 + df['NumberOfSuppliments'].eq('more than 5').astype(int)*0.8 + df['PastNumberOfClaims'].eq('more than 4').astype(int)*0.7 + df['AgeOfVehicle'].eq('more than 7').astype(int)*0.25 + df['AgentType'].eq('External').astype(int)*0.2)
    p=0.02 + 0.65*(1/(1+np.exp(-3*(risk-1.6))))
    df[TARGET]=(rng.random(n)<p).astype(int)
    return df

def load_data():
    if os.path.exists(DATA):
        df=pd.read_csv(DATA)
        source='Kaggle Vehicle Insurance Claim Fraud Detection dataset'
    else:
        df=make_demo(); os.makedirs('data',exist_ok=True); df.to_csv('data/demo_insurance_claims.csv',index=False)
        source='LOCAL DEMO DATA (synthetic fallback)'
    if TARGET not in df.columns:
        raise ValueError(f'Missing target column {TARGET}. Expected the Kaggle fraud_oracle.csv file.')
    return df, source

df, source=load_data()
X=df[CATEGORICAL+NUMERIC].copy(); y=df[TARGET].astype(int)
cat_pipe=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
num_pipe=Pipeline([('imputer',SimpleImputer(strategy='median'))])
prep=ColumnTransformer([('cat',cat_pipe,CATEGORICAL),('num',num_pipe,NUMERIC)])
models={
 'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1, min_samples_leaf=2)
}
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
results=[]
for name,clf in models.items():
    pipe=Pipeline([('prep',prep),('model',clf)])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1]
    results.append({'model':name,'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0),'roc_auc':roc_auc_score(yte,proba)})
    fn=name.lower().replace(' ','_')+'.pkl'
    with open(os.path.join(MODEL_DIR,fn),'wb') as f: pickle.dump(pipe,f)
    if name=='Random Forest':
        with open(os.path.join(MODEL_DIR,'model.pkl'),'wb') as f: pickle.dump(pipe,f)
        cm=confusion_matrix(yte,pred).tolist()

with open(os.path.join(MODEL_DIR,'metrics.json'),'w') as f: json.dump({'source':source,'rows':len(df),'fraud_count':int(y.sum()),'legitimate_count':int((1-y).sum()),'results':results,'confusion_matrix':cm},f,indent=2)
print(json.dumps({'source':source,'rows':len(df),'results':results},indent=2))
