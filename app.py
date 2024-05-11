import streamlit as st
import numpy as np
from joblib import load

# Load the model
model = load('C:\\Users\\msssg\\Desktop\\data analysis with python\\model.pkl')

st.title('Healthcare Cost Prediction')

# Create inputs for the features
age = st.number_input('Age', min_value=18, max_value=100, value=30)
sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input('Children', min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox('Smoker', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
region = st.selectbox('Region', [0, 1, 2, 3], format_func=lambda x: f'Region {x+1}')

# Calculate BMI-Age Interaction on the fly
bmi_age_interaction = bmi * age

# Button to make prediction
if st.button('Predict'):
    # Collect all features in the correct order
    features = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(features)
    st.write(f'Estimated Healthcare Cost: ${prediction[0]:,.2f}')
