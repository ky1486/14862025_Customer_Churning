import streamlit as st
import pickle
import numpy as np

# Load the preprocessing object
with open('churn_assgignment_3.pkl', 'rb') as file:
    sc = pickle.load(file)

# Load the model
with open('churn_assign (1).h5', 'rb') as file:
    model = pickle.load(file)

def preprocess_input(features):
    # Add preprocessing steps here based on your specific preprocessing logic
    # For example, use the loaded preprocessing object to scale features
    features_scaled = sc.transform(features)
    return features_scaled

def predict_churn(features):
    # Preprocess the input
    features_preprocessed = preprocess_input(features)
    # Make predictions using the loaded model
    prediction = model.predict(features_preprocessed)
    return prediction

def main():
    st.title("Churn Prediction App")

    # Add input fields for user input
    contract_options = ['Month-to-month', 'One year', 'Two year']
    contract = st.selectbox("Contract", contract_options)

    tenure = st.slider("Tenure (months)", min_value=0, max_value=100, step=1)

    online_security_options = ['No', 'Yes', 'Unknown']
    online_security = st.selectbox("Online Security", online_security_options)

    tech_support_options = ['No', 'Yes', 'Unknown']
    tech_support = st.selectbox("Tech Support", tech_support_options)

    total_charges = st.number_input("Total Charges")

    online_backup_options = ['No', 'Yes', 'Unknown']
    online_backup = st.selectbox("Online Backup", online_backup_options)

    monthly_charges = st.number_input("Monthly Charges")

    # Collect user input into a list
    user_input = [contract, tenure, online_security, tech_support, total_charges, online_backup, monthly_charges]

    # Convert user input to a NumPy array
    input_array = np.array(user_input).reshape(1, -1)

    # Button to make a prediction
    if st.button("Predict"):
        # Make prediction
        prediction = predict_churn(input_array)

        # Display result
        st.write(f"Churn Prediction: {prediction}")

if __name__ == '__main__':
    main()
