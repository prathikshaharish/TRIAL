import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data function
def load_data(file_path):
    data = pd.read_excel('Final_EMG-EEG-ML.csv')
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
    
    data['Category'] = pd.select(conditions, choices, default='Check Values')
    return data

# Set up the Streamlit interface
st.title('Pain Detection System')
data = load_data('Final_EMG-EEG-ML.csv')

# Show data and allow user interactions
if st.button('Show Data'):
    st.write(data)

# Apply thresholds to the data
data_with_pain_status = apply_thresholds(data)

# Machine learning model training
model = LogisticRegression()
features = data[['EMG Rest (ÂµV)', 'EMG Flexion (ÂµV)', 'EMG Extension (ÂµV)', 'EEG Rest (ÂµV)', 'EEG Flexion (ÂµV)', 'EEG Extension (ÂµV)']]
labels = (data['Category'] == 'Pain').astype(int)
model.fit(features, labels)

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
