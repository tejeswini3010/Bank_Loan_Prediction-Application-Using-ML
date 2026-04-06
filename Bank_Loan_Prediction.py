import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("BANK LOAN PREDICTION")

try:
    
    data = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

    st.write("Dataset loaded successfully ")

    
    if "Loan_ID" in data.columns:
        data.drop("Loan_ID", axis=1, inplace=True)

    np.random.seed(42)
    data["Loan_Status"] = np.random.randint(0, 2, size=len(data))

    
    data.fillna(method='ffill', inplace=True)

    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le


    X = data.drop("Loan_Status", axis=1)
    y = data["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    st.success("Model trained successfully ")


    st.header("Enter Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    app_income = st.number_input("Applicant Income", value=5000)
    co_income = st.number_input("Coapplicant Income", value=0)
    loan_amount = st.number_input("Loan Amount", value=150)
    loan_term = st.number_input("Loan Term", value=360)
    credit_history = st.selectbox("Credit History", [1, 0])

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [app_income],
        "CoapplicantIncome": [co_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })

    # Encode input
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    if st.button("Predict"):
        pred = model.predict(input_data)
        if pred[0] == 1:
            st.success("Loan Approved ")
        else:
            st.error("Loan Rejected ")

except Exception as e:
    st.error(f"Error occurred: {e}")