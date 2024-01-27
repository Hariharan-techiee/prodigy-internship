import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Get the absolute path to the model file
current_directory = os.path.dirname(__file__)  # Assuming this script is in the same directory as the model file
model_path = os.path.join(current_directory, 'linear_regression_model.pkl')

# Load the trained model
try:
    model = joblib.load(model_path)
    print("Model Loaded Successfully!")
except Exception as e:
    print("Error Loading Model:", e)
    st.error("Error loading the model. Please check the model file path.")

# Feature Selection
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']

# Load the test data
file_path = "tests.csv"  # Provide the correct file path
test_data = pd.read_csv(file_path)

# Display the test data

# User Input for Key Metrics
st.sidebar.header('Input Features')

# Add sliders for each feature in the selected features list
user_inputs = {}
for feature in features:
    # Ensure that min_value and max_value are of the same type
    feature_min = float(test_data[feature].min())
    feature_max = float(test_data[feature].max())
    user_inputs[feature] = st.sidebar.slider(
        feature,
        min_value=feature_min,
        max_value=feature_max,
        value=(feature_min + feature_max) / 2  # Set initial value to the midpoint
    )

# Check if the model is loaded successfully before making predictions
if 'model' in globals():
    # Preprocess the user input data
    # This should include handling missing values and scaling features using the same scaler
    scaler = StandardScaler()  # Use the same scaler used during training
    user_inputs_scaled = scaler.fit_transform(pd.DataFrame(user_inputs, index=[0]))

    # Make predictions on user input
    prediction = model.predict(user_inputs_scaled)

    
    # Additional interactive section to dynamically update predictions
    st.sidebar.header('Adjust Input Features for Real-time Prediction')
    for feature in features:
        user_inputs[feature] = st.sidebar.slider(
            f"{feature} (Current Value: {user_inputs[feature]})",
            min_value=float(test_data[feature].min()),
            max_value=float(test_data[feature].max()),
            value=user_inputs[feature]
        )

    # Update predictions based on the adjusted user inputs
    user_inputs_scaled = scaler.transform(pd.DataFrame(user_inputs, index=[0]))
    updated_prediction = model.predict(user_inputs_scaled)

    # Display the updated prediction
    st.subheader('Updated Predicted House Price')
    st.write(f"The updated predicted house price is ${updated_prediction[0]:,.2f}")

else:
    st.error("Error: Model not loaded.")