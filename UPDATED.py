import streamlit as st
import pandas as pd

# Load dataset
uploaded_file = st.file_uploader("EDITED DATA SET - EXO.xlsx", type="xlsx")
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Column names in the dataset:", df.columns.tolist())

# Define the threshold ranges
thresholds = {
    "EMG Rest": {"Pain": (0.05, float("inf")), "No Pain": (-float("inf"), 0.02)},
    "EMG Flexion": {"Pain": (0.7, 1.25), "No Pain": (0.3, 0.7)},
    "EMG Extension": {"Pain": (1.0, 1.6), "No Pain": (0.4, 0.8)},
    "EEG Rest": {"Pain": (1.5, float("inf")), "No Pain": (-float("inf"), 0.5)},
    "EEG Flexion": {"Pain": (2.5, 3.5), "No Pain": (1.0, 1.5)},
    "EEG Extension": {"Pain": (3.0, 4.5), "No Pain": (1.0, 2.0)},
}

def classify_value(value, threshold):
    if threshold["Pain"][0] <= value <= threshold["Pain"][1]:
        return "Pain"
    elif threshold["No Pain"][0] <= value <= threshold["No Pain"][1]:
        return "No Pain"
    else:
        return "Unknown"

def classify_emg_eeg(emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension):
    classifications = {
        "EMG Rest": classify_value(emg_rest, thresholds["EMG Rest"]),
        "EMG Flexion": classify_value(emg_flexion, thresholds["EMG Flexion"]),
        "EMG Extension": classify_value(emg_extension, thresholds["EMG Extension"]),
        "EEG Rest": classify_value(eeg_rest, thresholds["EEG Rest"]),
        "EEG Flexion": classify_value(eeg_flexion, thresholds["EEG Flexion"]),
        "EEG Extension": classify_value(eeg_extension, thresholds["EEG Extension"]),
    }
    return classifications

# Streamlit app layout
st.markdown("""
    <h1 style='text-align: center; font-weight: bold;'>EMG - EEG Sensor-Based Exoskeleton for Knee Injury Rehabilitation</h1>
    <h3 style='text-align: center;'>BMS COLLEGE OF ENGINEERING</h3>
    <h3 style='text-align: center;'>MAJOR PROJECT VIII SEM</h3>
    <h4 style='text-align: center;'>By - Prathiksha Harish, Krithik Raj K, Prajval Prakash, and Dr. Jisha P</h4>
    """, unsafe_allow_html=True)

st.write("### Enter EMG and EEG values to classify pain/no pain:")

# Input fields for EMG and EEG values
emg_rest = st.number_input("EMG Rest (µV)", value=0.0)
emg_flexion = st.number_input("EMG Flexion (µV)", value=0.0)
emg_extension = st.number_input("EMG Extension (µV)", value=0.0)
eeg_rest = st.number_input("EEG Rest (µV)", value=0.0)
eeg_flexion = st.number_input("EEG Flexion (µV)", value=0.0)
eeg_extension = st.number_input("EEG Extension (µV)", value=0.0)

if st.button("Predict"):
    classifications = classify_emg_eeg(emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension)
    st.write("### Classification Results:")
    for key, value in classifications.items():
        st.write(f"{key}: {value}")

if uploaded_file:
    st.write("### Uploaded Dataset:")
    st.dataframe(df)

    st.subheader('Patient Data Details')
    patient_ids = df['Patient_ID'].unique()
    selected_patient = st.selectbox('Select a patient to view details:', patient_ids)

    if selected_patient:
        patient_data = df[df['Patient_ID'] == selected_patient]
        if not patient_data.empty:
            st.write(f"Details for patient {selected_patient}:")
            st.write(patient_data)
