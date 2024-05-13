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
    # Ensure column names match exactly
    emg_rest = 'EMG Rest (µV)'
    emg_flexion = 'EMG Flexion (µV)'
    emg_extension = 'EMG Extension (µV)'
    eeg_rest = 'EEG Rest (µV)'
    eeg_flexion = 'EEG Flexion (µV)'
    eeg_extension = 'EEG Extension (µV)'

    conditions = [
        (data[emg_rest] > 0.05) | 
        (data[emg_flexion] > 1.25) | 
        (data[emg_extension] > 1.6) |
        (data[eeg_rest] > 1.5) | 
        (data[eeg_flexion] > 3.5) | 
        (data[eeg_extension] > 4.5),
        (data[emg_rest] <= 0.02) & 
        (data[emg_flexion] <= 0.7) & 
        (data[emg_extension] <= 0.8) &
        (data[eeg_rest] <= 0.5) & 
        (data[eeg_flexion] <= 1.5) & 
        (data[eeg_extension] <= 2.0)
    ]
    
    choices = ['Pain', 'No Pain']
    
    data['Category'] = np.select(conditions, choices, default='Check Values')
    return data

# Set up the Streamlit interface
st.title('Pain Detection System')

# Load and categorize data
data = load_data('/mnt/data/Final_EMG-EEG-ML.csv')
st.write("Column names in the dataset:", data.columns.tolist())  # Display column names for debugging

data_with_pain_status = apply_thresholds(data)

# Filter out rows with 'Check Values' in 'Category'
filtered_data = data_with_pain_status[data_with_pain_status['Category'] != 'Check Values']

# Show data and allow user interactions
if st.button('Show Data'):
    st.write(filtered_data)

# Check if there are at least two classes present
class_counts = filtered_data['Category'].value_counts()
st.write("Class distribution:", class_counts)

if len(class_counts) < 2:
    st.write("Error: The dataset contains only one class. The model cannot be trained.")
else:
    # Machine learning model training
    model = LogisticRegression()
    feature_columns = [emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension]
    features = filtered_data[feature_columns]
    labels = (filtered_data['Category'] == 'Pain').astype(int)

    model.fit(features, labels)
    st.write("Model training completed successfully.")

    # User input for prediction
    st.subheader('Predict Pain Status')
    input_data = st.text_input('Enter EMG and EEG values separated by comma (in the order: EMG Rest, EMG Flexion, EMG Extension, EEG Rest, EEG Flexion, EEG Extension):')
    if st.button('Predict'):
        try:
            input_list = list(map(float, input_data.split(',')))
            if len(input_list) != 6:
                st.write("Error: Please enter exactly 6 values.")
            else:
                input_array = np.array(input_list).reshape(1, -1)
                prediction = model.predict(input_array)
                st.write('Pain' if prediction[0] == 1 else 'No Pain')
        except ValueError:
            st.write("Error: Please enter valid numeric values.")

st.sidebar.header('About')
st.sidebar.info('This is a Streamlit app for detecting pain based on EMG and EEG readings.')

# Display processed data
if st.button('Show Processed Data'):
    st.write(data_with_pain_status)
