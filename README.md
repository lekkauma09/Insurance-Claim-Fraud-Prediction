# ClaimGuard AI — Insurance Claim Fraud Prediction

A complete mini-project for predicting whether an insurance claim should be flagged for further fraud review.

## Dataset
Use Kaggle's **Vehicle Insurance Claim Fraud Detection** dataset and place its `fraud_oracle.csv` file at:

`data/fraud_oracle.csv`

Target column: `FraudFound_P` (0 = legitimate, 1 = fraudulent).

The project currently includes a clearly labelled synthetic demo fallback so the application can be tested before downloading the Kaggle CSV. **Do not report demo metrics as Kaggle results.**

## Features
- Data preprocessing and one-hot encoding
- Logistic Regression and Random Forest classification
- Accuracy, precision, recall, F1 and ROC-AUC comparison
- Flask backend API
- Claim entry web form
- Fraud probability output
- SQLite prediction history
- Claims analytics dashboard

## Run locally
```bash
python -m pip install -r requirements.txt
python train_model.py
python app.py
```
Open `http://127.0.0.1:5000`.

## API
POST JSON to `/api/predict`. The endpoint accepts the same claim fields as the web form. Example:
```json
{"claim_name":"CLM-1001","Age":35,"Sex":"Male","AccidentArea":"Urban","Fault":"Policy Holder","PolicyType":"Sedan - Collision","VehicleCategory":"Sedan","VehiclePrice":"20000 to 29000","DriverRating":2,"PoliceReportFiled":"No","WitnessPresent":"No","AgentType":"External","PastNumberOfClaims":"none","NumberOfSuppliments":"none","AgeOfVehicle":"3 years","BasePolicy":"Collision","Deductible":400}
```

## Academic note
The model predicts fraud risk, not an insurer's final legal/financial claim decision. A human claims investigator should review flagged cases.
