import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load data function
def load_data(file_path):
    data = pd.read_csv('Final_EMG-EEG-ML.csv')
    return data

# Function to apply thresholds to create labels
def apply_thresholds(data):
    conditions = [
        (data['EMG Rest (µV)'] > 0.05) | 
        (data['EMG Flexion (µV)'] > 1.25) | 
        (data['EMG Extension (µV)'] > 1.6) |
        (data['EEG Rest (µV)'] > 1.5) | 
        (data['EEG Flexion (µV)'] > 3.5) | 
        (data['EEG Extension (µV)'] > 4.5),
        (data['EMG Rest (µV)'] <= 0.02) & 
        (data['EMG Flexion (µV)'] <= 0.7) & 
        (data['EMG Extension (µV)'] <= 0.8) &
        (data['EEG Rest (µV)'] <= 0.5) & 
        (data['EEG Flexion (µV)'] <= 1.5) & 
        (data['EEG Extension (µV)'] <= 2.0)
    ]
    
    choices = ['Pain', 'No Pain']
    
    data['Category'] = np.select(conditions, choices, default='Check Values')
    return data

# Set up the Streamlit interface
st.title('Pain Detection System')
data = load_data('/mnt/data/Final_EMG-EEG-ML.csv')

# Show data and allow user interactions
if st.button('Show Data'):
    st.write(data)

# Apply thresholds to the data
data_with_pain_status = apply_thresholds(data)

# Check for missing values
if data_with_pain_status.isnull().values.any():
    st.write("Warning: There are missing values in the dataset. Please handle them before training the model.")
else:
    # Ensure all feature values are numeric
    feature_columns = ['EMG Rest (µV)', 'EMG Flexion (µV)', 'EMG Extension (µV)', 'EEG Rest (µV)', 'EEG Flexion (µV)', 'EEG Extension (µV)']
    data_with_pain_status[feature_columns] = data_with_pain_status[feature_columns].apply(pd.to_numeric, errors='coerce')

    # Check for any remaining non-numeric values
    if data_with_pain_status[feature_columns].isnull().values.any():
        st.write("Warning: There are non-numeric values in the features. Please clean the data.")
    else:
        # Machine learning model training
        model = LogisticRegression()
        features = data_with_pain_status[feature_columns]
        labels = (data_with_pain_status['Category'] == 'Pain').astype(int)
        
        try:
            model.fit(features, labels)
        except ValueError as e:
            st.write(f"Error in model fitting: {e}")

        # User input for prediction
        st.subheader('Predict Pain Status')
        input_data = st.text_input('Enter EMG and EEG values separated by comma (in the order: EMG Rest, EMG Flexion, EMG Extension, EEG Rest, EEG Flexion, EEG Extension):')
        if st.button('Predict'):
            input_list = list(map(float, input_data.split(',')))
            prediction = model.predict([input_list])
            st.write('Pain' if prediction[0] == 1 else 'No Pain')

st.sidebar.header('About')
st.sidebar.info('This is a Streamlit app for detecting pain based on EMG and EEG readings.')

# Display processed data
if st.button('Show Processed Data'):
    st.write(data_with_pain_status)
