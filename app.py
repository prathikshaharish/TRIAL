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
- **Feedback Loop:** Establish a feedback loop where data from the model is regularly reviewed and used to make informed decisions about the patient’s rehabilitation plan.
""", unsafe_allow_html=True)

# Display processed data
if st.button('Show Processed Data'):
    st.write(data)
