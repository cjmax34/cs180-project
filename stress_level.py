from helper.func import *   # Import the necessary libraries and functions from helper/func.py

st.set_page_config(
    page_title="Stress Level Prediction",
    page_icon="😩"
)

# Main page
st.title('Stress Level Prediction')
st.write("This machine learning project aims to predict the stress level being experienced by an individual given their sleep health and various lifestyle habits. The project uses a Multilayer Perceptron (MLP) as the machine learning algorithm.")

st.divider()    # Add a divider

st.header("Input Data")
st.write("To predict your stress level, please input the following data:")
gender_col, age_col, bmi_col = st.columns(3)
sleep_dur_col, sleep_qual_col, sleep_disorder_col = st.columns(3)
heart_rate_col, systolic_col, diastolic_col = st.columns(3)
steps_col, physical_act_col = st.columns(2)


## Gender, Age (years), Body Mass Index (BMI)
with gender_col:
    st.subheader('Gender')
    gender = st.radio("Please select your gender.", ["Male", "Female"], index=None)

with age_col:
    st.subheader('Age')
    age = st.number_input("Please enter your age.", min_value=18, max_value=60, step=1)

with bmi_col:
    st.subheader('Body Mass Index (BMI)')
    bmi = st.radio("Please select your BMI category.", ["Normal", "Overweight", "Obese"], index=None)

## Sleep duration (hours), Sleep quality, Sleep disorder
with sleep_dur_col:
    st.subheader('Sleep Duration')
    sleep_dur = st.slider("Use the slider to input your average sleep duration (in hours).", min_value=5.8, max_value=8.5, step=0.1)

with sleep_qual_col:
    st.subheader('Sleep Quality')
    sleep_qual = st.number_input("Please type your subjective rating of sleep quality.", min_value=1, max_value=10, step=1)

with sleep_disorder_col:
    st.subheader('Sleep Disorder')
    sleep_disorder = st.radio("What sleep disorder are you currently diagnosed with?", ["None", "Sleep Apnea", "Insomnia"], index=None)

## Heart rate (bpm), and systolic and diastolic blood pressure 
with heart_rate_col:
    st.subheader('Heart Rate')
    heart_rate = st.number_input("Please enter your heart rate (in beats per minute).", min_value=65, max_value=86, step=1)

with systolic_col:
    st.subheader('Systolic Blood Pressure')
    systolic_bp = st.number_input("Please enter your systolic blood pressure.", min_value=100, max_value=140, step=1)

with diastolic_col:
    st.subheader('Diastolic Blood Pressure')
    diastolic_bp = st.number_input("Please enter your diastolic blood pressure.", min_value=60, max_value=90, step=1)

## Steps (per day), Physical activity level
with steps_col:
    st.subheader('Daily Steps')
    daily_steps = st.number_input("Please enter the number of steps you take daily.", min_value=1000, max_value=10000)

with physical_act_col:
    st.subheader('Physical Activity Level')
    phys_act_level = st.number_input("Please enter the amount of your physical activity (in minutes).", min_value=30, max_value=90, step=1)

st.divider()    # Add a divider

## Predict button
if st.button('Predict Your Stress Level'):
    stress_pred = predictStress(gender, age, sleep_dur, sleep_qual, phys_act_level, bmi, heart_rate, daily_steps, sleep_disorder, systolic_bp, diastolic_bp)
    st.success(f"Your predicted stress level is: {stress_pred}")