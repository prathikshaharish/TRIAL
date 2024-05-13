import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

# Load data function (dummy data in this example)
def load_data():
    # Normally you would load your data from a file
    data = pd.read_csv('DataSet_Exo-MP.csv')
    return data

# Function to apply thresholds to create labels
def apply_thresholds(data):
    conditions = [
        (data['EMG Rest (µV)'] > 0.05) | (data['EMG Flexion (µV)'] > 1.25) | (data['EMG Extension (µV)'] > 1.6) |
        (data['EEG Rest (µV)'] > 1.5) | (data['EEG Flexion (µV)'] > 3.5) | (data['EEG Extension (µV)'] > 4.5),
        (data['EMG Rest (µV)'] <= 0.02) & (data['EMG Flexion (µV)'] <= 0.7) & (data['EMG Extension (µV)'] <= 0.8) &
        (data['EEG Rest (µV)'] <= 0.5) & (data['EEG Flexion (µV)'] <= 1.5) & (data['EEG Extension (µV)'] <= 2.0)
    ]
    choices = ['Pain', 'No Pain']
    data['Pain_Status'] = pd.np.select(conditions, choices, default='Check Values')
    return data

# Set up the Streamlit interface
st.title('Pain Detection System')
data = load_data()
def apply_thresholds(data):
    # Define conditions for "Pain" based on provided thresholds
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
    
    # Define the category for each condition
    choices = ['Pain', 'No Pain']
    
    # Apply conditions to the DataFrame
    data['Pain_Status'] = pd.np.select(conditions, choices, default='Check Values')
    return data


# Show data and allow user interactions
if st.button('Show Data'):
    st.write(data)

# Machine learning model training (Dummy example)
model = LogisticRegression()
features = data[['EMG Rest (µV)', 'EMG Flexion (µV)', 'EMG Extension (µV)', 'EEG Rest (µV)', 'EEG Flexion (µV)', 'EEG Extension (µV)']]
labels = (data['Pain_Status'] == 'Pain').astype(int)
model.fit(features, labels)

# User input for prediction
st.subheader('Predict Pain Status')
input_data = st.text_input('Enter EMG and EEG values separated by comma:')
if st.button('Predict'):
    input_list = list(map(float, input_data.split(',')))
    prediction = model.predict([input_list])
    st.write('Pain' if prediction[0] == 1 else 'No Pain')

st.sidebar.header('About')
st.sidebar.info('This is a Streamlit app for detecting pain based on EMG and EEG readings.')

# After defining apply_thresholds function
if __name__ == '__main__':
    data = load_data()
    data_with_pain_status = apply_thresholds(data)
    st.dataframe(data_with_pain_status)  # Display data with pain status on the webpage

    # Any further functionality you want to add can be written below
    st.title('Pain Detection System')
    if st.button('Show Processed Data'):
        st.write(data_with_pain_status)
