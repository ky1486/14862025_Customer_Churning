import streamlit as st
import pickle as pkl
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
model = tf.keras.models.load_model('churn_assign.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pkl.load(file)


# Function to preprocess input data
def preprocess_input(input_data):
    input_df = pd.DataFrame(input_data, columns=['Contract', 'tenure', 'OnlineSecurity', 'TechSupport', 'TotalCharges',
                                                 'OnlineBackup', 'MonthlyCharges'])

    # Fit and transform a new label encoder for each categorical column
    for column in input_df.columns:
        if input_df[column].dtype == 'object':
            encoder = LabelEncoder()
            input_df[column] = encoder.fit_transform(input_df[column])

    input_scaled = scaler.transform(input_df)
    return input_scaled


# Streamlit web app
def main():
    st.title("Churn Prediction Web App")

    st.sidebar.header("User Input")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.sidebar.slider("Tenure", min_value=0, max_value=100, step=1)
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, step=1.0)
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=1.0)

    input_data = [[contract, tenure, online_security, tech_support, total_charges, online_backup, monthly_charges]]
    input_scaled = preprocess_input(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        churn_prob = prediction[0][0]
        churn_confidence = churn_prob * 100  # Convert probability to percentage

        st.subheader("Prediction:")
        if churn_prob > 0.5:
            st.write(f"The customer is likely to churn with a probability of {churn_prob:.2f} and a confidence of {churn_confidence:.2f}%.")
        else:
            st.write(f"The customer is not likely to churn with a probability of {1 - churn_prob:.2f} and a confidence of {100 - churn_confidence:.2f}%.")


if __name__ == '__main__':
    main()
