import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model from the file
with open('xgboost_model.pkl', 'rb') as file:
    model_xgb = pickle.load(file)

# Define the feature names
feature_names = [
    'StockOptionLevel',
    'OverTime',
    'TotalWorkingYears',
    'Age',
    'JobLevel',
    'YearsInCurrentRole',
    'MonthlyIncome',
    'YearsWithCurrManager',
    'YearsAtCompany',
    'SalarySlab',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'JobInvolvement'
]

# Streamlit app
st.title("Employee Attrition Prediction")

# Create input fields for each feature
inputs = {}
for feature in feature_names:
    if feature in ['StockOptionLevel', 'OverTime', 'JobLevel', 'SalarySlab', 'JobSatisfaction', 'EnvironmentSatisfaction', 'JobInvolvement']:
        # For categorical features, use selectbox and convert to numerical values
        options = [0, 1] if feature != 'OverTime' else ['Yes', 'No']
        value = st.selectbox(f"Select {feature}", options)
        inputs[feature] = 1 if value == 'Yes' else (0 if value == 'No' else value)
    else:
        # For numerical features, use number input
        inputs[feature] = st.number_input(f"Enter {feature}", value=0)

# Convert inputs to a DataFrame with the correct feature names
input_data = pd.DataFrame([inputs], columns=feature_names)

# Check the data types of the input data
st.write("Input Data Types:", input_data.dtypes)

# Button to make prediction
if st.button("Predict"):
    try:
        # Ensure data types are consistent with what the model expects
        for feature in feature_names:
            input_data[feature] = pd.to_numeric(input_data[feature], errors='coerce')
        
        # Make prediction
        prediction = model_xgb.predict(input_data)
        st.write("Prediction: ", "Attrition" if prediction[0] == 1 else "No Attrition")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
