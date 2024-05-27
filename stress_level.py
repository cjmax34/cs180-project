import streamlit as st

# Import necessary libraries for the model here
import pandas as pd
import numpy as np

# For data preprocessing
from sklearn.preprocessing import LabelEncoder

# For generating the training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split

# For evaluating the model's performance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# For the main model (Multilayer Perceptron)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler  # Import the scaler to be used in scaling the features

st.set_page_config(
    page_title="Stress Level Prediction",
    page_icon="😩"
)

# Model function
@st.cache_resource
def mlpModel():
    # Load the dataset
    dataset = pd.read_csv('Sleep_Data_Sampled.csv')

    # Rename Normal Weight to Normal
    dataset['BMI Category'] = dataset['BMI Category'].str.replace('Normal Weight', 'Normal')

    # Split the blood pressure feature into systolic blood pressure and diastolic blood pressure
    dataset[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = dataset['Blood Pressure'].str.split('/', n=1, expand=True)

    # Convert the data type of the newly created columns to numeric
    dataset[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = dataset[['Systolic Blood Pressure', 'Diastolic Blood Pressure']].apply(pd.to_numeric)

    # Move the target column (stress level) to the last column
    dataset['Stress Level'] = dataset.pop('Stress Level')

    # Remove some features because they are unnecessary for analysis
    features_to_remove = ['Person ID', 'Occupation', 'Blood Pressure']
    dataset.drop(features_to_remove, axis=1, inplace=True)

    # Use LabelEncoder to encode the categorical features
    label_encoder = LabelEncoder()
    categorical_features = dataset.select_dtypes(include=['object']).columns.tolist()
    for cat in categorical_features:
        dataset[cat] = label_encoder.fit_transform(dataset[cat])

    # Initializing the features and target variables
    X = dataset.drop('Stress Level', axis=1)
    y = dataset['Stress Level']

    # Generating the training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
    
    # Instantiate the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1,1))   # tanh activation function

    # Identify the numerical columns
    numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Heart Rate', 'Daily Steps', 'Systolic Blood Pressure', 'Diastolic Blood Pressure']

    # Create a copy of the testing and training sets
    X_train_mlp = X_train.copy()  
    X_test_mlp = X_test.copy()  

    # Store in the copies the scaled training and testing sets (to be used for the MLP)
    X_train_mlp[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_mlp[numerical_cols] = scaler.transform(X_test[numerical_cols])

    mlp = MLPClassifier(hidden_layer_sizes=20, max_iter=5000, learning_rate_init=0.001, activation='tanh', solver='lbfgs', random_state=22) # Optimal parameters
    mlp.fit(X_train_mlp, y_train)

    mlp_predictions = mlp.predict(X_test_mlp)
    print(mlp_predictions)  # For debugging purposes

    utils = {
        "model": mlp,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }

    return utils    # Return the model and the encoders (to be used for the prediction functioon)
        
# Prediction function
def predictStress(gender, age, sleep_dur, bmi, heart_rate, daily_steps, systolic_bp, diastolic_bp):
    ... # Work in progress

# Main page
st.title('Stress Level Prediction')
st.write("This machine learning project aims to predict the stress level being experienced by an individual given their sleep health and various lifestyle habits. The project uses a Multilayer Perceptron (MLP) as the machine learning algorithm.")

st.divider()    # Add a divider

st.header("Input Data")
st.write("To predict your stress level, please input the following data:")
gender_col, age_col, bmi_col = st.columns(3)
sleep_dur_col, sleep_disorder_col, steps_col = st.columns(3)
heart_rate_col, systolic_col, diastolic_col = st.columns(3)

## Gender, Age (years), Body Mass Index (BMI)
with gender_col:
    st.subheader('Gender')
    gender = st.radio("Please select your gender.", ["Male", "Female"], index=None)

with age_col:
    st.subheader('Age')
    age = st.number_input("Please enter your age.", min_value=27, max_value=60, step=1)

with bmi_col:
    st.subheader('Body Mass Index (BMI)')
    bmi = st.radio("Please select your BMI category.", ["Normal", "Overweight", "Obese"], index=None)

## Sleep duration (hours), Sleep disorder, Steps (per day)
with sleep_dur_col:
    st.subheader('Sleep Duration')
    sleep_dur = st.slider("Use the slider to input your average sleep duration (in hours).", min_value=5.8, max_value=8.5, step=0.1)

with sleep_disorder_col:
    st.subheader('Sleep Disorder')
    sleep_disorder = st.radio("What sleep disorder are you currently diagnosed with?", ["None", "Sleep Apnea", "Insomnia"], index=None)

with steps_col:
    st.subheader('Daily Steps')
    daily_steps = st.number_input("Please enter the number of steps you take daily.", min_value=1000, max_value=10000)

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

st.divider()    # Add a divider

## Predict button
if st.button('Predict Your Stress Level'):
    stress_pred = predictStress(gender, age, sleep_dur, bmi, heart_rate, daily_steps, systolic_bp, diastolic_bp)
    st.success(f"Your predicted stress level is: {stress_pred}")