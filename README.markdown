**Project Title**

**Heart Disease Classification & Deployment using Machine Learning**

---

**Problem Statement**

The objective of this project is to build and deploy a machine learning classification system to predict the presence of heart disease in patients based on clinical and physiological attributes.

The project demonstrates an end-to-end machine learning workflow including:

* Data preprocessing
* Model training
* Performance evaluation
* Model comparison
* Web app development using Streamlit
* Cloud deployment

---

**Dataset Description**

**Dataset Name:** Heart Disease Prediction Dataset
**Source:** UCI Machine Learning Repository / Kaggle
**Problem Type:** Binary Classification

**Dataset Statistics**

| Attribute | Value                       |
| --------- | --------------------------- |
| Instances | 1025                        |
| Features  | 13                          |
| Target    | Heart Disease Presence      |
| Classes   | 0 = No Disease, 1 = Disease |

**Feature Examples**

* Age
* Sex
* Chest Pain Type
* Cholesterol
* Resting Blood Pressure
* ECG Results
* Maximum Heart Rate

This dataset satisfies assignment constraints of ≥12 features and ≥500 instances.

---

**Models Implemented**

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors
* Naive Bayes (Gaussian)
* Random Forest (Ensemble)
* XGBoost (Ensemble)

---

**Evaluation Metrics Used**

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

**Model Comparison Table**

| Model               | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------- | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.795122 | 0.878736 | 0.756303  | 0.873786 | 0.810811 | 0.597255 |
| Decision Tree       | 0.985366 | 0.985437 | 1.000000  | 0.970874 | 0.985222 | 0.971151 |
| KNN                 | 0.834146 | 0.948553 | 0.800000  | 0.893204 | 0.844037 | 0.672727 |
| Naive Bayes         | 0.800000 | 0.870550 | 0.754098  | 0.893204 | 0.817778 | 0.610224 |
| Random Forest       | 0.985366 | 1.000000 | 1.000000  | 0.970874 | 0.985222 | 0.971151 |
| XGBoost             | 0.985366 | 0.989435 | 1.000000  | 0.970874 | 0.985222 | 0.971151 |

---

**Observations**

| Model               | Observation                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| Logistic Regression | Provides a good baseline but lower performance compared to ensemble models. |
| Decision Tree       | Very high accuracy but risk of overfitting due to single-tree structure.    |
| KNN                 | Moderate performance; sensitive to scaling and computationally expensive.   |
| Naive Bayes         | Fast but assumes feature independence, leading to lower accuracy.           |
| Random Forest       | Excellent performance due to bagging and variance reduction.                |
| XGBoost             | Matches top performance with strong boosting optimization.                  |

---

**Streamlit Application Features**

* CSV dataset upload
* Model selection dropdown
* Prediction display
* Evaluation metrics display
* Confusion matrix visualization
* Classification report

---

**Deployment**

The application is deployed using Streamlit Community Cloud and connected via GitHub repository integration.

---

**Repository Structure**

```
project-folder/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── heart.csv
└── model/
```

---

**How to Run Locally**

```
pip install -r requirements.txt
streamlit run app.py
```

---

**Conclusion**

Ensemble models such as Random Forest and XGBoost achieved the highest performance across all evaluation metrics.

The project highlights the effectiveness of bagging and boosting techniques in improving classification accuracy and model robustness.
