import streamlit as st
import pandas as pd
import numpy as np

# Load data function
def load_data(file_path):
    data = pd.read_csv('DataSet_Exo-MP.csv')
    data.columns = data.columns.str.strip()  # Strip any extra spaces from column names
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

# Function to classify pain status based on provided ranges
def classify_pain_status(emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension):
    if (emg_rest > 0.05 or 
        0.7 <= emg_flexion <= 1.25 or 
        1.0 <= emg_extension <= 1.6 or 
        eeg_rest > 1.5 or 
        2.5 <= eeg_flexion <= 3.5 or 
        3.0 <= eeg_extension <= 4.5):
        return 'Pain'
    elif (emg_rest <= 0.02 and 
          0.3 <= emg_flexion <= 0.7 and 
          0.4 <= emg_extension <= 0.8 and 
          eeg_rest <= 0.5 and 
          1.0 <= eeg_flexion <= 1.5 and 
          1.0 <= eeg_extension <= 2.0):
        return 'No Pain'
    else:
        return 'Check Values'

# Set up the Streamlit interface
st.markdown("""
    <h1 style='text-align: center; font-weight: bold;'>EMG - EEG Sensor-Based Exoskeleton for Knee Injury Rehabilitation</h1>
    <h3 style='text-align: center;'>BMS COLLEGE OF ENGINEERING</h3>
    <h3 style='text-align: center;'>MAJOR PROJECT VIII SEM</h3>
    <h4 style='text-align: center;'>By - Prathiksha Harish, Krithik Raj K, Prajval Prakash, and Dr. Jisha P</h4>
    """, unsafe_allow_html=True)

# Load and categorize data
file_path = 'DataSet_Exo-MP.csv'
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

# User input for prediction
st.subheader('Predict Pain Status')

# Define input fields for EMG and EEG values
emg_rest = st.number_input('EMG Rest (µV)', min_value=0.0, max_value=5.0, step=0.01)
emg_flexion = st.number_input('EMG Flexion (µV)', min_value=0.0, max_value=5.0, step=0.01)
emg_extension = st.number_input('EMG Extension (µV)', min_value=0.0, max_value=5.0, step=0.01)
eeg_rest = st.number_input('EEG Rest (µV)', min_value=0.0, max_value=10.0, step=0.01)
eeg_flexion = st.number_input('EEG Flexion (µV)', min_value=0.0, max_value=10.0, step=0.01)
eeg_extension = st.number_input('EEG Extension (µV)', min_value=0.0, max_value=10.0, step=0.01)

if st.button('Predict'):
    input_list = [emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension]
    pain_status = classify_pain_status(*input_list)
    st.write(f'Pain Status: {pain_status}')

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
