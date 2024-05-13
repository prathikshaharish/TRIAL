import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys

# Install seaborn and matplotlib if not already installed
try:
    import seaborn as sns
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

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
    except ValueError:
        st.write("Error: Please enter valid numeric values.")

# Plot EEG and EMG data over time for each patient
st.subheader('Patient Data Visualization')
patient_names = filtered_data['Name'].unique()
selected_patient = st.selectbox('Select a patient:', patient_names)

if selected_patient:
    patient_data = data[data['Name'] == selected_patient]
    if not patient_data.empty:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax2 = ax1.twinx()
        sns.lineplot(data=patient_data, x='Time (s)', y='EMG Rest (ÂµV)', ax=ax1, label='EMG Rest (ÂµV)', color='r')
        sns.lineplot(data=patient_data, x='Time (s)', y='EEG Rest (ÂµV)', ax=ax2, label='EEG Rest (ÂµV)', color='b')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('EMG Rest (ÂµV)', color='r')
        ax2.set_ylabel('EEG Rest (ÂµV)', color='b')

        fig.tight_layout()
        st.pyplot(fig)

st.sidebar.header('About')
st.sidebar.info('This is a Streamlit app for detecting pain based on EMG and EEG readings.')

# Display processed data
if st.button('Show Processed Data'):
    st.write(data_with_pain_status)
