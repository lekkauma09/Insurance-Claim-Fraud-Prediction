# Insurance Claim Fraud Prediction System

## 1. Introduction
Insurance claim fraud can increase losses and investigation workload. This project develops a machine-learning based decision-support system that estimates the probability that an insurance claim contains fraud-related patterns.

## 2. Objective
- Collect and preprocess insurance claim data.
- Perform exploratory data analysis.
- Train at least two classification algorithms.
- Compare their performance using standard metrics.
- Build a Flask API and web form for prediction.
- Store prediction history and display claim analytics.

## 3. Dataset
The project is designed for Kaggle's Vehicle Insurance Claim Fraud Detection dataset. It contains 15,420 records and 33 columns, with `FraudFound_P` as the binary target. The target identifies legitimate and fraudulent claims. Source: Kaggle dataset/notebooks.

## 4. Methodology
Data Collection → Data Cleaning → Encoding → Train/Test Split → Model Training → Evaluation → API Integration → Web Prediction → Database Storage → Dashboard.

## 5. Algorithms
### Logistic Regression
Used as a linear classification baseline. Class weighting is enabled to handle the unequal class distribution.

### Random Forest
An ensemble of decision trees that can model non-linear relationships between claim attributes. Class weighting is enabled.

## 6. System Modules
1. Dataset preprocessing
2. EDA and visualization
3. ML model training
4. Model comparison
5. Flask prediction API
6. Claim entry frontend
7. SQLite prediction storage
8. Analytics dashboard

## 7. Output
The application returns a fraud probability and a status. If probability is at least 50%, the demo system labels the claim “Flagged for Further Review”; otherwise it labels it “Likely Genuine”. This threshold is a project decision and should be calibrated using validation data for real deployment.

## 8. Limitations
The current bundled model is trained on a synthetic demo fallback because the execution environment cannot directly download the Kaggle file. Before final academic submission, download the Kaggle `fraud_oracle.csv`, place it in `data/`, run `python train_model.py`, and use the resulting real-dataset metrics in the report.

## 9. Future Enhancements
- Hyperparameter tuning
- Explainable AI with SHAP
- User authentication
- Cloud deployment
- Automated model monitoring
- Role-based investigator dashboard

## 10. Conclusion
The project demonstrates an end-to-end insurance fraud prediction workflow, connecting machine learning with a practical web application and analytics dashboard.
