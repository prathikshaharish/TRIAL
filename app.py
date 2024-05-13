import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

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

# Initialize model variable
model = None

# Check if there are at least two classes present
class_counts = filtered_data['Category'].value_counts()
st.write("Class distribution:", class_counts)

if len(class_counts) < 2:
    st.write("Error: The dataset contains only one class. The model cannot be trained.")
else:
    # Machine learning model training
    model = LogisticRegression()
    feature_columns = ['EMG Rest (µV)', 'EMG Flexion (µV)', 'EMG Extension (µV)', 'EEG Rest (µV)', 'EEG Flexion (µV)', 'EEG Extension (µV)']
    features = filtered_data[feature_columns]
    labels = (filtered_data['Category'] == 'Pain').astype(int)

    # Check for missing values
    if features.isnull().values.any() or labels.isnull().values.any():
        st.write("Error: The dataset contains missing values. Please clean the data.")
    else:
        try:
            model.fit(features, labels)
            st.write("Model training completed successfully.")
        except ValueError as e:
            st.write(f"Error in model fitting: {e}")
            model = None

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
    if model is None:
        st.write("Error: The model has not been trained. Please check the dataset and ensure the model is trained.")
    else:
        input_list = [emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension]
        input_array = np.array(input_list).reshape(1, -1)
        prediction = model.predict(input_array)
        st.write('Pain' if prediction[0] == 1 else 'No Pain')
        
        # Additional classification based on ranges
        pain_level = classify_pain(emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension)
        st.write(f'Pain Level: {pain_level}')

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
