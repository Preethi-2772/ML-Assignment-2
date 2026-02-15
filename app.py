import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

st.title("Heart Disease Prediction App")

# Upload CSV
file = st.file_uploader("Upload Test CSV", type=["csv"])

if file:

    data = pd.read_csv(file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["Logistic","Decision Tree","KNN","Naive Bayes","Random Forest","XGBoost"]
    )

    model_files = {
        "Logistic":"model/logistic.pkl",
        "Decision Tree":"model/dt.pkl",
        "KNN":"model/knn.pkl",
        "Naive Bayes":"model/nb.pkl",
        "Random Forest":"model/rf.pkl",
        "XGBoost":"model/xgb.pkl"
    }

    # Load scaler & model
    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(model_files[model_name])

    
    if "target" in data.columns:
        X = data.drop("target", axis=1)
        y_true = data["target"]
    else:
        X = data.copy()
        y_true = None

    
    feature_order = [
        'age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal'
    ]

    try:
        X = X[feature_order]
    except:
        st.error("Uploaded CSV does not match required feature format.")
        st.stop()

    
    X_scaled = scaler.transform(X)

    # Predictions
    preds = model.predict(X_scaled)

    st.subheader("Predictions")
    st.write(preds)

    # Confusion Matrix & Report (if target exists)
    if y_true is not None:

        cm = confusion_matrix(y_true, preds)

        fig, ax = plt.subplots()
        ax.matshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_true, preds)
        st.text(report)
