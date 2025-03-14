import streamlit as st
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title='Prediction')

# Load the classifier
model = joblib.load('saved_models/random_forest_classifier.pkl')

# Define labels for encoding
Binary = ['No', 'Yes']
Gender = ['Male', 'Female']
Ethnicity = ['Caucasian', 'African American', 'Asian', 'Other']
EducationLevel = ['None', 'High School', "Bachelor's", 'Higher']

# One-hot encoding
def encode(input, labels):
    for i in range(len(labels)):
        if input == labels[i]:
            return i

# Alzheimer classifier
def predict_alzheimer(input_data, model):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    return prediction, probability

def LIME(input_data, model):
    class_names = ['Not Alzheimer', 'Alzheimer']
    feature_names = ['Age', 'Gender', 'Ethnicity', 'EducationLevel',
                        'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 
                        'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 
                        'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
                        'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
                        'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']

    # Fit the Explainer on the training data set using the LimeTabularExplainer
    explainer = LimeTabularExplainer(input_data, feature_names=feature_names, class_names=class_names, mode='classification')

    # Predict function should return probability estimates
    predict_fn = model.predict_proba

    # Generate explanation for the input data
    explanation = explainer.explain_instance(input_data.flatten(), predict_fn, num_features=5)

    # Show LIME explaination
    fig = explanation.as_pyplot_figure()

    st.pyplot(fig)

# Streamlit page title
st.title("Predict Alzheimers")

# Input form
st.header("Demographics")
age = st.number_input("Age", min_value=0, max_value=122, step=1, value=65)
gender = st.radio("Gender", ("Male", "Female"))
ethnicity = st.selectbox("Ethnicity", ('Caucasian', 'African American', 'Asian', 'Other'))
education_level = st.selectbox("Education level", ('None', 'High School', "Bachelor's", 'Higher'))

st.write("<br>", unsafe_allow_html=True)

st.header("Life Style")
bmi = st.number_input("BMI", min_value=15, max_value=40, step=1, value=27)
smoking = st.radio("Smoking", ("No", "Yes"))
alchohol = st.number_input('Weekly alcohol consumption (0-20)', min_value=0, max_value=20, step=1) 
actitvity = st.number_input('Weekly physical activity in hours (0-20)', min_value=0, max_value=20, step=1) 
diet = st.number_input('Diet quality score (0-10)', min_value=0, max_value=10, step=1) 
sleep = st.number_input('Sleep quality score (0-10)', min_value=0, max_value=10, step=1) 

st.write("<br>", unsafe_allow_html=True)

st.header("Medical History")
family_history = st.radio("Family history of Alzheimer's Disease", ("No", "Yes"))
cardiovascular = st.radio("Cardiovascular disease", ("No", "Yes"))
diabetes = st.radio("Diabetes", ("No", "Yes"))
depression = st.radio("Depression", ("No", "Yes"))
headInjury = st.radio("History of head injury", ("No", "Yes"))
hypertension = st.radio("Hypertension", ("No", "Yes"))

st.write("<br>", unsafe_allow_html=True)

st.header("Medical History")
systolicBP = st.number_input('Systolic blood pressure (mmHg)', min_value=90, max_value=180, step=1, value=134) 
diastolicBP = st.number_input('Diastolic blood pressure (mmHg)', min_value=60, max_value=120, step=1, value=90) 
cholesterolTotal = st.number_input('Total cholesterol levels (mg/dL)', min_value=150, max_value=300, step=1, value=225) 
cholesterolLDL = st.number_input('Low-density lipoprotein cholesterol levels (mg/dL)', min_value=50, max_value=200, step=1, value=124) 
cholesterolHDL = st.number_input('High-density lipoprotein cholesterol levels (mg/dL)', min_value=20, max_value=100, step=1, value=59) 
cholesterolTriglyceride = st.number_input('Triglycerides levels (mg/dL)', min_value=50, max_value=400, step=1, value=228) 

st.write("<br>", unsafe_allow_html=True)

st.header("Cognitive and Functional Assessment")
mmse = st.number_input("MMSE Score (0-30). Lower scores indicate cognitive impairment.", min_value=0, max_value=30, step=1)
functional = st.number_input("Functional assessment score (0-10). Lower scores indicate greater impairment.", min_value=0, max_value=10, step=1)
memory = st.radio("Presence of memory complaints", ("No", "Yes"))
behavioral = st.radio("Presence of behavioral problems", ("No", "Yes"))
adl = st.number_input("Activities of Daily Living score (0-10). Lower scores indicate greater impairment.", min_value=0, max_value=10, step=1)

st.write("<br>", unsafe_allow_html=True)

st.header("Symptoms")
confusion = st.radio("Presence of confusion", ("No", "Yes"))
disorientation = st.radio("Presence of disorientation", ("No", "Yes"))
personality_changes = st.radio("Presence of personality changes", ("No", "Yes"))
difficulty_completing_tasks = st.radio("Presence of difficulty completing tasks", ("No", "Yes"))
forgetfulness = st.radio("Presence of forgetfulness", ("No", "Yes"))

st.write("<br>", unsafe_allow_html=True)

predict_button = st.button("Predict")

# Encode input data
if predict_button:
    gender = encode(gender, Gender)
    ethnicity = encode(ethnicity, Ethnicity)
    education_level = encode(education_level, EducationLevel)
    smoking = encode(smoking, Binary)
    family_history = encode(family_history, Binary)
    cardiovascular = encode(cardiovascular, Binary)
    diabetes = encode(diabetes, Binary)
    depression = encode(depression, Binary)
    headInjury = encode(headInjury, Binary)
    hypertension = encode(hypertension, Binary)
    memory = encode(memory, Binary)
    behavioral = encode(behavioral, Binary)
    confusion = encode(confusion, Binary)
    disorientation = encode(disorientation, Binary)
    personality_changes = encode(personality_changes, Binary)
    difficulty_completing_tasks = encode(difficulty_completing_tasks, Binary)
    forgetfulness = encode(forgetfulness, Binary)

    # Predict
    input_data = [age, gender, ethnicity, education_level, 
                   bmi, smoking, alchohol, actitvity, diet, sleep, 
                   family_history, cardiovascular, diabetes, depression, headInjury, hypertension, 
                   systolicBP, diastolicBP, cholesterolTotal, cholesterolLDL, cholesterolHDL, cholesterolTriglyceride, 
                   mmse, functional, memory, behavioral, adl, 
                   confusion, disorientation, personality_changes, difficulty_completing_tasks, forgetfulness]
    input_data = np.array(input_data).reshape(1, -1)
    prediction, proba = predict_alzheimer(input_data, model)

    # Show result
    if prediction == 0:
        st.write(f'Not Alzheimer: {proba[0][0] * 100:.2f}%')
    else:
        st.write(f'Alzheimer: {proba[0][1] * 100:.2f}%')

    # show LIME explaination
    st.title("LIME Explaination")
    LIME(input_data, model)
