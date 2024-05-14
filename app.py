import streamlit as st
import pandas as pd
import numpy as np

# Load data function
def load_data(file_path):
    data = pd.read_csv('EDITED DATA SET - Sheet1.csv')
    data.columns = data.columns.str.strip()  # Strip any extra spaces from column names
    return data

# Function to classify pain status based on provided ranges
def classify_pain_status(emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension):
    if (0.7501276039 <= emg_rest <= 1.501316884 or 
        0.997254081 <= emg_flexion <= 3.998466906 or 
        1.798109079 <= emg_extension <= 3.801380313 or 
        5.000450261 <= eeg_rest <= 50.18350703 or 
        60.04197402 <= eeg_flexion <= 86.95405508 or 
        69.82874585 <= eeg_extension <= 81.92500559):
        return 'Pain'
    elif (0.497758633 <= emg_rest <= 0.501794274 and 
          0.8001190203 <= emg_flexion <= 1.999715615 and 
          0.798152212 <= emg_extension <= 0.803510512 and 
          2.02509753 <= eeg_rest <= 4.089443689 and 
          19.87502164 <= eeg_flexion <= 50.08350199 and 
          49.84258228 <= eeg_extension <= 50.17559819):
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
file_path = '/mnt/data/EDITED DATA SET - Sheet1.csv'
data = load_data(file_path)

# Display the column names
st.write("Column names in the dataset:", data.columns.tolist())

# Show data and allow user interactions
if st.button('Show Data'):
    st.write(data)

# User input for prediction
st.subheader('Predict Pain Status')

# Define input fields for EMG and EEG values with validation
emg_rest = st.number_input('EMG Rest (µV)', min_value=0.0, max_value=1.501316884, step=0.01)
emg_flexion = st.number_input('EMG Flexion (µV)', min_value=0.0, max_value=3.998466906, step=0.01)
emg_extension = st.number_input('EMG Extension (µV)', min_value=0.0, max_value=3.801380313, step=0.01)
eeg_rest = st.number_input('EEG Rest (µV)', min_value=0.0, max_value=50.18350703, step=0.01)
eeg_flexion = st.number_input('EEG Flexion (µV)', min_value=0.0, max_value=86.95405508, step=0.01)
eeg_extension = st.number_input('EEG Extension (µV)', min_value=0.0, max_value=81.92500559, step=0.01)

if st.button('Predict'):
    input_list = [emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension]
    if any([
        emg_rest > 1.501316884,
        emg_flexion > 3.998466906,
        emg_extension > 3.801380313,
        eeg_rest > 50.18350703,
        eeg_flexion > 86.95405508,
        eeg_extension > 81.92500559
    ]):
        st.write('Error: One or more input values exceed the maximum allowed value.')
    else:
        pain_status = classify_pain_status(*input_list)
        st.write(f'Pain Status: {pain_status}')

# Patient data visualization
st.subheader('Patient Data Details')
patient_ids = data['Patient_ID'].unique()
selected_patient = st.selectbox('Select a patient to view details:', patient_ids)

if selected_patient:
    patient_data = data[data['Patient_ID'] == selected_patient]
    if not patient_data.empty:
        st.write(f"Details for patient {selected_patient}:")
        st.write(patient_data)

st.sidebar.header('About')
st.sidebar.info('This model will be running real-time very soon! Stay tuned :D')

# Display processed data
if st.button('Show Processed Data'):
    st.write(data)
