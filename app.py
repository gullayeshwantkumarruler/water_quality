import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the trained model from a pickle file
model = pickle.load(open('rf_model_water_quality.pkl', 'rb'))

# Streamlit app
st.title("Water Quality Predictor")
st.sidebar.title("Input Features")
st.sidebar.markdown("Enter the values for the following features:")
st.image("river_water_quality.jpg", use_column_width=True)

# Input features
station_name = st.text_input("Station Name")
primary_basin = st.text_input("Primary Basin")
depth = st.number_input("Depth")
Site_Status_21Oct2020 = st.text_input("Site_Status_21Oct2020")


# Predict button
if st.button("Predict"):
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame({
        'station_name': [station_name],
        'primary_basin': [primary_basin],
        'depth': [depth],
        'site_status': [Site_Status_21Oct2020]
    })

    # Load the label encoder for 'Station_Name' and 'Primary_Basin' from a saved file
    label_encoder = LabelEncoder()
    # label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)
    
    # Perform label encoding on 'Station_Name' and 'Primary_Basin' columns
    input_data['station_name'] = label_encoder.fit_transform(input_data['station_name'])
    input_data['primary_basin'] = label_encoder.fit_transform(input_data['primary_basin'])
    input_data['site_status'] = label_encoder.fit_transform(input_data['site_status'])
    
    # Make the prediction using the loaded model
    prediction = model.predict(input_data)[0]
    
    # Display the predicted risk level
    st.header("Prediction")
    st.write(prediction)
    # Display the prediction
    if (prediction >= 7.76) and (prediction<=9.0) :
        st.success("The water quality is predicted to be Good.")
    else:
        st.error("The water quality is predicted to be Bad.")
