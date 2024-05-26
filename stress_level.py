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
st.title('Stress Level Prediction')
# Model function
@st.cache_resource
def mlpModel():
    ...

st.write('This machine learning project aims to predict the stress level being experienced by an individual given their sleep health and various lifestyle habits. The project uses a Multilayer Perceptron (MLP) as the machine learning algorithm.')

st.divider()    # Add a divider

st.header('Input data here')
gender_col, age_col, sleep_dur_col, bmi_col = st.columns(4)
heart_rate_col, steps_col, systolic_col, diastolic_col = st.columns(4)

with gender_col:
    st.subheader('Gender')
    sex = st.radio("Please select your gender.", ["Male", "Female"], index=None)

    if sex:
        st.write(f'You selected {sex}')
    else:
        st.write('You have not selected your gender.')
