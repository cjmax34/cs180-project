import streamlit as st

# Import necessary libraries for the model here
import pandas as pd
import numpy as np
import sklearn

# For data preprocessing
from sklearn.preprocessing import LabelEncoder

# For generating the training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split

# For the main model (Multilayer Perceptron)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler  # Import the scaler to be used in scaling the features

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
    label_encoder_gender, label_encoder_bmi, label_encoder_sleep_disorder = LabelEncoder(), LabelEncoder(), LabelEncoder()
    dataset['Gender'] = label_encoder_gender.fit_transform(dataset['Gender'])
    dataset['BMI Category'] = label_encoder_bmi.fit_transform(dataset['BMI Category'])
    dataset['Sleep Disorder'] = label_encoder_sleep_disorder.fit_transform(dataset['Sleep Disorder'])

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
        "label_encoder_gender": label_encoder_gender,
        "label_encoder_bmi": label_encoder_bmi,
        "label_encoder_sleep_disorder": label_encoder_sleep_disorder,
    }

    return utils    # Return the model and the encoders (to be used for the prediction functioon)
        
# Prediction function
def predictStress(gender, age, sleep_dur, sleep_qual, phys_act_level, bmi, heart_rate, daily_steps, sleep_disorder, systolic_bp, diastolic_bp):
    # Convert "None" of sleep_disorder to "Healthy"
    if sleep_disorder == "None":
        sleep_disorder = "Healthy"
    
    utils = mlpModel() # Load the model and the encoders
    mlp = utils['model']
    scaler = utils['scaler']
    label_encoder_gender = utils['label_encoder_gender']
    label_encoder_bmi = utils['label_encoder_bmi']
    label_encoder_sleep_disorder = utils['label_encoder_sleep_disorder']

    # Perform label encoding on Gender, BMI, and Sleep Disorder
    gender_encoded = label_encoder_gender.transform(np.array([gender]))
    bmi_encoded = label_encoder_bmi.transform(np.array([bmi]))
    sleep_disorder_encoded = label_encoder_sleep_disorder.transform(np.array([sleep_disorder]))

    # Perform normalization on the numerical features
    numerical_features = np.array([age, sleep_dur, sleep_qual, phys_act_level, heart_rate, daily_steps, systolic_bp, diastolic_bp]).reshape(1, -1)
    numerical_features = scaler.transform(numerical_features)

    # Combine the encoded and normalized features
    input_features = np.array([gender_encoded[0], numerical_features[0][0], numerical_features[0][1], numerical_features[0][2], numerical_features[0][3], bmi_encoded[0], numerical_features[0][4], numerical_features[0][5], sleep_disorder_encoded[0], numerical_features[0][6], numerical_features[0][7]]).reshape(1, -1)
    stress_prediction = mlp.predict(input_features)

    return stress_prediction[0]  # Return the predicted stress level