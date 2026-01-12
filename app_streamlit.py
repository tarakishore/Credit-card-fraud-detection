import streamlit as st
import pandas as pd
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('fraud_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'fraud_model.pkl' not found!")
        return None

model = load_model()

# App title
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it's fraudulent or legitimate.")

# Create input form
with st.form("fraud_detection_form"):
    st.subheader("Transaction Details")
    
    # Time and Amount
    col1, col2 = st.columns(2)
    with col1:
        time = st.number_input("Time (seconds)", min_value=0.0, value=0.0)
    with col2:
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
    
    st.subheader("PCA Features (V1-V28)")
    
    # Create 4 columns for V1-V28 inputs
    v_features = {}
    cols = st.columns(4)
    
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with cols[col_idx]:
            v_features[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.6f")
    
    # Submit button
    submitted = st.form_submit_button("🔍 Predict")
    
    if submitted:
        if model is None:
            st.error("Model not loaded. Cannot make predictions.")
        else:
            # FIXED: Prepare input data in CORRECT column order
            # Model expects: Time, V1-V28, Amount (Amount at the END)
            input_data = [[time] + [v_features[f'V{i}'] for i in range(1, 29)] + [amount]]
            input_df = pd.DataFrame(input_data, columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            # Display result
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"🚨 **FRAUD DETECTED!**")
                st.write(f"Fraud Probability: **{probability[1]*100:.2f}%**")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**")
                st.write(f"Fraud Probability: **{probability[1]*100:.2f}%**")
            
            # Show confidence bar
            st.progress(float(probability[prediction]))
            
            # Additional details
            with st.expander("📊 View Input Data"):
                st.dataframe(input_df)

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app uses a machine learning model to detect fraudulent credit card transactions.

**Features:**
- Time: Seconds elapsed
- V1-V28: PCA-transformed features
- Amount: Transaction amount (at the end)
""")

st.sidebar.success("Model Status: Loaded" if model else "Model Status: Not Loaded")