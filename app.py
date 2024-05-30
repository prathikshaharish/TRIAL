import streamlit as st
import pandas as pd

# Load dataset
uploaded_file = st.file_uploader("EDITED DATA SET.xlsx", type="xlsx")
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
    st.write("EDITED DATA SET.xlsx")
    st.dataframe(df)

    st.subheader('Patient Data Details')
    patient_ids = df['Patient_ID'].unique()
    selected_patient = st.selectbox('Select a patient to view details:', patient_ids)

    if selected_patient:
        patient_data = df[df['Patient_ID'] == selected_patient]
        if not patient_data.empty:
            st.write(f"Details for patient {selected_patient}:")
            st.write(patient_data)

st.sidebar.header('About')
st.sidebar.info('This model will be running real-time very soon! Stay tuned')

st.sidebar.markdown("""
### _**NO PAIN RANGES:**_

- **EMG Rest (µV):** 0.497758633 to 0.501794274
- **EMG Flexion (µV):** 0.8001190203 to 1.999715615
- **EMG Extension (µV):** 0.798152212 to 0.803510512
- **EEG Rest (µV):** 2.02509753 to 4.089443689
- **EEG Flexion (µV):** 19.87502164 to 50.08350199
- **EEG Extension (µV):** 49.84258228 to 50.17559819

### _**PAIN RANGES:**_

- **EMG Rest (µV):** 0.7501276039 to 1.501316884
- **EMG Flexion (µV):** 0.997254081 to 3.998466906
- **EMG Extension (µV):** 1.798109079 to 3.801380313
- **EEG Rest (µV):** 5.000450261 to 50.18350703
- **EEG Flexion (µV):** 60.04197402 to 86.95405508
- **EEG Extension (µV):** 69.82874585 to 81.92500559

### _**Benefits for Doctors:**_

1. _**Objective Pain Assessment:**_
   - **Consistent Monitoring:** By using EMG and EEG sensors to continuously monitor muscle and brain activity, the model provides consistent and objective assessments of the patient's pain levels. This removes subjectivity from the process and allows for more accurate tracking of the patient's progress.
   - **Real-Time Feedback:** Doctors can get real-time feedback on the patient’s pain levels during rehabilitation exercises. This allows for immediate adjustments to the rehabilitation program to ensure that exercises are beneficial and not causing excessive pain or stress.

2. _**Personalized Rehabilitation Programs:**_
   - **Tailored Interventions:** The data collected and analyzed by the model can help doctors create personalized rehabilitation programs. By understanding the specific pain thresholds and responses of each patient, doctors can design exercises that are both effective and comfortable.
   - **Adaptive Therapy:** The model can help identify which exercises cause pain and which ones are more tolerable. This information can be used to adapt the rehabilitation program dynamically, ensuring that the patient remains engaged and motivated without experiencing undue pain.

3. _**Enhanced Patient Monitoring:**_
   - **Detailed Tracking:** The model provides a detailed record of the patient's EMG and EEG readings over time. This can be used to track improvements or setbacks in the patient’s condition, offering a comprehensive view of their rehabilitation journey.
   - **Early Detection of Complications:** By continuously monitoring pain levels and muscle activity, the model can help detect any signs of complications early. This can prompt timely interventions and potentially prevent more severe issues from developing.

4. _**Data-Driven Decisions:**_
   - **Evidence-Based Adjustments:** The model provides data-driven insights that can support clinical decisions. Doctors can use the collected data to adjust rehabilitation protocols based on empirical evidence rather than intuition alone.
   - **Outcome Measurement:** The effectiveness of different rehabilitation techniques can be quantitatively measured, allowing for evidence-based evaluations of various approaches. This can lead to the refinement of best practices in post-surgery knee rehabilitation.

5. _**Patient Engagement and Motivation:**_
   - **Transparent Progress:** Patients can be shown their progress through visualizations of their EMG and EEG data. Seeing concrete evidence of their improvement can be highly motivating and encourage adherence to the rehabilitation program.
   - **Empowerment:** Patients are empowered by understanding the impact of their efforts. This can lead to increased compliance with prescribed exercises and a more proactive approach to their own recovery.

6. _**Research and Development:**_
   - **New Insights:** The data collected can contribute to research in the field of rehabilitation. Analyzing trends and patterns across multiple patients can lead to new insights into pain management and recovery processes.
   - **Innovation:** Continuous monitoring and data collection can drive innovation in rehabilitation techniques and technologies. Insights gained can inform the development of new therapeutic devices and protocols.

### _**Implementation in Clinical Settings:**_

To effectively implement this model in clinical settings, the following steps can be taken:
- **Integration with Existing Systems:** Ensure the model and sensors can integrate seamlessly with the hospital’s electronic health records (EHR) and other monitoring systems.
- **Training for Medical Staff:** Provide training for doctors, physiotherapists, and other medical staff on how to use the system, interpret the data, and incorporate it into patient care.
- **Patient Education:** Educate patients on the importance of continuous monitoring and how it will benefit their recovery. 