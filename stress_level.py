import streamlit as st

# Import necessary libraries for the model here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Stress Level Prediction",
    page_icon="😩"
)

# Model function
@st.cache_resource
def mlpModel():
    ...

# Prediction function
def predictStress(gender, age, sleep_dur, bmi, heart_rate, daily_steps, systolic_bp, diastolic_bp):
    ... # Work in progress

# Main page
st.title('Stress Level Prediction')
st.write("This machine learning project aims to predict the stress level being experienced by an individual given their sleep health and various lifestyle habits. The project uses a Multilayer Perceptron (MLP) as the machine learning algorithm.")

st.divider()    # Add a divider

st.header("Input Data")
st.write("To predict your stress level, please input the following data:")
gender_col, age_col, sleep_dur_col = st.columns(3)
bmi_col, heart_rate_col, steps_col = st.columns(3)
systolic_col, diastolic_col = st.columns(2)

## Gender, Age, Sleep duration
with gender_col:
    st.subheader('Gender')
    gender = st.radio("Please select your gender.", ["Male", "Female"], index=None)

with age_col:
    st.subheader('Age')
    age = st.number_input("Please enter your age.", min_value=27, max_value=60, step=1)

with sleep_dur_col:
    st.subheader('Sleep Duration')
    sleep_dur = st.slider("Use the slider to input your average sleep duration (in hours)", min_value=5.8, max_value=8.5, step=0.1)

## Body Mass Index (BMI), Heart rate (beats/min), Steps (per day)
with bmi_col:
    st.subheader('Body Mass Index (BMI)')
    bmi = st.radio("Please select your BMI category.", ["Normal", "Overweight", "Obese"], index=None)

with heart_rate_col:
    st.subheader('Heart Rate')
    heart_rate = st.number_input("Please enter your heart rate (in beats per minute).", min_value=65, max_value=86, step=1)

with steps_col:
    st.subheader('Daily Steps')
    daily_steps = st.number_input("Please enter the number of steps you take daily.", min_value=1000, max_value=10000)

## Systolic and diastolic blood pressure
with systolic_col:
    st.subheader('Systolic Blood Pressure')
    systolic_bp = st.number_input("Please enter your systolic blood pressure.", min_value=100, max_value=140, step=1)

with diastolic_col:
    st.subheader('Diastolic Blood Pressure')
    diastolic_bp = st.number_input("Please enter your diastolic blood pressure.", min_value=60, max_value=90, step=1)

st.divider()    # Add a divider

## Predict button
if st.button('Predict Your Stress Level'):
    stress_pred = predictStress(gender, age, sleep_dur, bmi, heart_rate, daily_steps, systolic_bp, diastolic_bp)
    st.success(f"Your predicted stress level is: {stress_pred}")