import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load data function
def load_data(file_path):
    data = pd.read_csv('DataSet_Exo-MP.csv')
    return data

# Function to apply thresholds to create labels
def apply_thresholds(data):
    conditions = [
        (data['EMG Rest (ÂµV)'] > 0.05) | 
        (data['EMG Flexion (ÂµV)'] > 1.25) | 
        (data['EMG Extension (ÂµV)'] > 1.6) |
        (data['EEG Rest (ÂµV)'] > 1.5) | 
        (data['EEG Flexion (ÂµV)'] > 3.5) | 
        (data['EEG Extension (ÂµV)'] > 4.5),
        (data['EMG Rest (ÂµV)'] <= 0.02) & 
        (data['EMG Flexion (ÂµV)'] <= 0.7) & 
        (data['EMG Extension (ÂµV)'] <= 0.8) &
        (data['EEG Rest (ÂµV)'] <= 0.5) & 
        (data['EEG Flexion (ÂµV)'] <= 1.5) & 
        (data['EEG Extension (ÂµV)'] <= 2.0)
    ]
    
    choices = ['Pain', 'No Pain']
    
    data['Category'] = np.select(conditions, choices, default='Check Values')
    return data

# Function to classify the level of pain
def classify_pain(emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension):
    if (emg_rest > 0.1 or emg_flexion > 2.5 or emg_extension > 3.2 or
        eeg_rest > 3.0 or eeg_flexion > 7.0 or eeg_extension > 9.0):
        return 'Very High Pain'
    elif (emg_rest > 0.05 or emg_flexion > 1.25 or emg_extension > 1.6 or
          eeg_rest > 1.5 or eeg_flexion > 3.5 or eeg_extension > 4.5):
        return 'Pain'
    elif (emg_rest <= 0.02 and emg_flexion <= 0.7 and emg_extension <= 0.8 and
          eeg_rest <= 0.5 and eeg_flexion <= 1.5 and eeg_extension <= 2.0):
        return 'No Pain'
    else:
        return 'Check Values'

# Set up the Streamlit interface
st.title('Pain Detection System')

# Load and categorize data
file_path = '/mnt/data/DataSet_Exo-MP.csv'
data = load_data(file_path)

# Display the column names
st.write("Column names in the dataset:", data.columns.tolist())

# Apply thresholds to the data
data_with_pain_status = apply_thresholds(data)

# Filter out rows with 'Check Values' in 'Category'
filtered_data = data_with_pain_status[data_with_pain_status['Category'] != 'Check Values']

# Show data and allow user interactions
if st.button('Show Data'):
    st.write(filtered_data)

# Machine learning model training
model = LogisticRegression()
feature_columns = ['EMG Rest (ÂµV)', 'EMG Flexion (ÂµV)', 'EMG Extension (ÂµV)', 'EEG Rest (ÂµV)', 'EEG Flexion (ÂµV)', 'EEG Extension (ÂµV)']
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
            
            # Additional classification based on ranges
            emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension = input_list
            pain_level = classify_pain(emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension)
            st.write(f'Pain Level: {pain_level}')
    except ValueError:
        st.write("Error: Please enter valid numeric values.")

# Patient data visualization
st.subheader('Patient Data Details')
patient_names = filtered_data['Name'].unique()
selected_patient = st.selectbox('Select a patient to view details:', patient_names)

if selected_patient:
    patient_data = data[data['Name'] == selected_patient]
    if not patient_data.empty:
        st.write(f"Details for patient {selected_patient}:")
        st.write(patient_data)

st.sidebar.header('About')
st.sidebar.info('This is a Streamlit app for detecting pain based on EMG and EEG readings.')

# Display processed data
if st.button('Show Processed Data'):
    st.write(data_with_pain_status)
