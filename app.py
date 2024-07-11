import streamlit as st
import pandas as pd

# Load dataset
uploaded_file = st.file_uploader("EDITED DATA SET - EXO.xlsx", type="xlsx")
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Column names in the dataset:", df.columns.tolist())

# Define the threshold ranges
thresholds = {
    "EMG Rest": {"Pain": (0.7501276039, 1.501316884), "No Pain": (0.497758633, 0.501794274)},
    "EMG Flexion": {"Pain": (0.997254081, 3.998466906), "No Pain": (0.8001190203, 1.999715615)},
    "EMG Extension": {"Pain": (1.798109079, 3.801380313), "No Pain": (0.798152212, 0.803510512)},
    "EEG Rest": {"Pain": (5.000450261, 50.18350703), "No Pain": (2.02509753, 4.089443689)},
    "EEG Flexion": {"Pain": (60.04197402, 86.95405508), "No Pain": (19.87502164, 50.08350199)},
    "EEG Extension": {"Pain": (69.82874585, 81.92500559), "No Pain": (49.84258228, 50.17559819)},
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

    pain_count = sum(1 for key in classifications if classifications[key] == "Pain")
    no_pain_count = sum(1 for key in classifications if classifications[key] == "No Pain")

    if pain_count > no_pain_count:
        overall_classification = "Pain"
    elif no_pain_count > pain_count:
        overall_classification = "No Pain"
    else:
        overall_classification = "Uncertain"

    return classifications, overall_classification

# Streamlit app layout
st.markdown(
    """
    <h1 style='text-align: center; font-weight: bold;'>EMG - EEG Sensor-Based Exoskeleton for Knee Injury Rehabilitation</h1>
    <h4 style='text-align: center;'>By - Prakruthi Harish , Prathiksha Harish, Krithik Raj K, Prajval Prakash, and Dr. Jisha P</h4>
    """,
    unsafe_allow_html=True,
)

st.write("### Enter EMG and EEG values to classify pain/no pain:")

# Input fields for EMG and EEG values
emg_rest = st.number_input("EMG Rest (µV)", value=0.0)
emg_flexion = st.number_input("EMG Flexion (µV)", value=0.0)
emg_extension = st.number_input("EMG Extension (µV)", value=0.0)
eeg_rest = st.number_input("EEG Rest (µV)", value=0.0)
eeg_flexion = st.number_input("EEG Flexion (µV)", value=0.0)
eeg_extension = st.number_input("EEG Extension (µV)", value=0.0)

if st.button("Predict"):
    classifications, overall_classification = classify_emg_eeg(
        emg_rest, emg_flexion, emg_extension, eeg_rest, eeg_flexion, eeg_extension
    )
    st.write("### Classification Results:")
    for key, value in classifications.items():
        st.write(f"{key}: {value}")
    st.write(f"### Overall Classification: {overall_classification}")

if uploaded_file:
    st.write("uploaded_file")
    st.dataframe(df)

    st.subheader("Patient Data Details")
    patient_ids = df["Patient_ID"].unique()
    selected_patient = st.selectbox("Select a patient to view details:", patient_ids)

    if selected_patient:
        patient_data = df[df["Patient_ID"] == selected_patient]
        if not patient_data.empty:
            st.write(f"Details for patient {selected_patient}:")
            st.write(patient_data)

st.sidebar.header("About")
st.sidebar.info("This model will be running real-time very soon! Stay tuned")
