import streamlit as st
import pandas as pd
import joblib

# Load the logistic regression model and column transformer
model_logistic = joblib.load("logistic_reg.pkl")
col_transform = joblib.load("column_transformer.pkl")

st.title("Credit Card Fraud Detection")

# Centered layout for input fields
st.markdown("<h3 style='text-align: center;'>Enter Transaction Details</h3>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    
    distance_from_home = st.number_input("Distance From Home", min_value=0.0, format="%.2f")
    distance_from_last_transaction = st.number_input("Distance From Last Transaction", min_value=0.0, format="%.2f")
    ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0, format="%.2f")
    used_chip = st.selectbox("Used Chip", [0, 1])
    used_pin_number = st.selectbox("Used Pin Number", [0, 1])
    online_order = st.selectbox("Online Order", [0, 1])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Predict Fraud"):
        # Validate user input
        if any(pd.isnull([distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price])):
            st.error("Please fill in all required fields.")
        else:
            # Creating input DataFrame
            input_data = pd.DataFrame({
                'distance_from_home': [distance_from_home],
                'distance_from_last_transaction': [distance_from_last_transaction],
                'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
                'used_chip': [used_chip],
                'used_pin_number': [used_pin_number],
                'online_order': [online_order]
            })

            # Transforming the input data using column transformer
            input_data_transformed = col_transform.transform(input_data)

            # Predicting the output using the logistic regression model
            y_pred = model_logistic.predict(input_data_transformed)
            y_prob = model_logistic.predict_proba(input_data_transformed)[0]

            # Displaying the result
            if y_pred == 1:  # Fraudulent transaction
                st.error("Fraudulent Transaction Detected! Alert!")
            else:
                st.success("Transaction is Legitimate.")

            # Display probabilities
            st.write(f"Probability of Legitimate Transaction: {y_prob[0]:.4f}")
            st.write(f"Probability of Fraudulent Transaction: {y_prob[1]:.4f}")

