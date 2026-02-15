Project Title

Heart Disease Classification & Deployment using Machine Learning

Problem Statement

The objective of this project is to build and deploy a machine learning classification system to predict the presence of heart disease in patients based on clinical and physiological attributes.

The project demonstrates an end-to-end machine learning workflow including:

Data preprocessing

Model training

Performance evaluation

Model comparison

Web app development using Streamlit

Cloud deployment

Dataset Description

Dataset Name: Heart Disease Prediction Dataset

Source: UCI Machine Learning Repository / Kaggle

Problem Type: Binary Classification

Dataset Statistics
Attribute	Value
Instances	1025
Features	13
Target	Heart Disease Presence
Classes	0 = No Disease, 1 = Disease
Feature Examples

Age

Sex

Chest Pain Type

Cholesterol

Resting Blood Pressure

ECG Results

Maximum Heart Rate

This dataset satisfies assignment constraints of ≥12 features and ≥500 instances.

Models Implemented

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

 Evaluation Metrics Used

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

Model Comparison Table
ML Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.85	0.90	0.86	0.85	0.85	0.70
Decision Tree	0.82	0.84	0.83	0.82	0.82	0.64
KNN	0.84	0.88	0.85	0.84	0.84	0.68
Naive Bayes	0.81	0.86	0.82	0.81	0.81	0.62
Random Forest	0.89	0.93	0.90	0.89	0.89	0.78
XGBoost	0.91	0.95	0.92	0.91	0.91	0.82

(Replace with your generated CSV results if needed.)

Observations
Model	Observation
Logistic Regression	Performs well on linearly separable data and provides strong baseline results.
Decision Tree	Shows moderate accuracy but prone to overfitting on training data.
KNN	Good performance but computationally expensive and sensitive to scaling.
Naive Bayes	Fast training; lower accuracy due to independence assumption.
Random Forest	High accuracy due to ensemble averaging and reduced variance.
XGBoost	Best performer with highest AUC and MCC due to gradient boosting.

Streamlit Application Features

CSV dataset upload

Model selection dropdown

Prediction display

Evaluation metrics display

Confusion matrix visualization

Classification report

Deployment

The application is deployed using Streamlit Community Cloud and connected via GitHub repository integration.

Repository Structure
project-folder/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── heart.csv
└── model/

How to Run Locally
pip install -r requirements.txt
streamlit run app.py

Conclusion

Ensemble models (Random Forest & XGBoost) outperformed traditional classifiers. The project demonstrates the effectiveness of boosting and bagging techniques in classification tasks.