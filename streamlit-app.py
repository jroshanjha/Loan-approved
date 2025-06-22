# app.py
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

st.title("🏦 Loan Approval Prediction - Deep Learning")

# Load model and preprocessors
model = load_model("loan_model.h5")
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))  # Save your ColumnTransformer as well

# Input fields
age = st.number_input("Age", 18, 100)
income = st.number_input("Income", 1000, 100000)
loan_amount = st.number_input("Loan Amount", 1000, 50000)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "PERSONAL", "MEDICAL", "VENTURE", "HOME IMPROVEMENT", "DEBT CONSOLIDATION"])
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Doctorate", "Associate"])
gender = st.selectbox("Gender", ["Male", "Female"])
ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"])

# Prepare DataFrame
input_df = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'loan_amnt': [loan_amount],
    'loan_intent': [loan_intent],
    'person_education': [education],
    'person_gender': [gender],
    'person_home_ownership': [ownership]
    # Add all required features used in training
})

# Transform
input_processed = preprocessor.transform(input_df)
input_processed = input_processed.toarray() if hasattr(input_processed, 'toarray') else input_processed

# Predict
if st.button("Predict Loan Approval"):
    result = model.predict(input_processed)
    approved = "✅ Approved" if result[0][0] > 0.5 else "❌ Not Approved"
    st.success(f"Prediction: {approved} (Confidence: {result[0][0]:.2f})")
