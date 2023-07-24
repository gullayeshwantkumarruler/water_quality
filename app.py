import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import pickle

# Load the trained model from a pickle file
model = pickle.load(open('rf_final_model_maternity.pkl', 'rb'))

# Streamlit app
st.title("Maternity Health Prediction")
st.image("image_indoor.jpg", use_column_width=True)

# Input features
age = st.number_input("Age")
systolic_bp = st.number_input("Systolic Blood Pressure")
diastolic_bp = st.number_input("Diastolic Blood Pressure")
bs = st.number_input("Blood Sugar Level")
body_temp = st.number_input("Body Temperature")
heart_rate = st.number_input("Heart Rate")

# Predict button
if st.button("Predict"):
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'BS': [bs],
        'BodyTemp': [body_temp],
        'HeartRate': [heart_rate]
    })

    # Make the prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Display the predicted risk level
    st.header("Risk Level Prediction:")
    st.write(prediction)
