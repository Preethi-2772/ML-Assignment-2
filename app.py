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

    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(model_files[model_name])

    X_scaled = scaler.transform(data)
    preds = model.predict(X_scaled)

    st.subheader("Predictions")
    st.write(preds)

    # If target column present
    if "target" in data.columns:

        y_true = data["target"]

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
