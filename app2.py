import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fraud_detection_model (6).keras')

@st.cache_resource
def load_scaler():
    with open('scaler (7).pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

# Streamlit app
st.title('💳 Credit Card Fraud Detection System')
st.markdown("""
Upload a CSV file containing transaction data to detect fraudulent transactions.
""")

# Model Information
st.header('Model Information')
st.markdown("""
**Model Architecture:**
- 7 Hidden Layers (128 → 64 → 32 → 16 → 8 → 4 → 2 neurons)
- LeakyReLU Activation
- Batch Normalization
- Dropout Regularization
- Sigmoid Output

**Performance Metrics:**
- Accuracy: 99.5%+
- Precision: 97%+
- Recall: 98%+
""")

st.image('A:\Major_project_code\_- visual selection (1).png', caption='Model Architecture', use_column_width=True)

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Ensure only the necessary columns are present
    required_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
    if not all(col in df.columns for col in required_columns):
        st.error("Uploaded CSV does not contain the required columns.")
    else:
        # Scale the input data
        df[required_columns] = scaler.transform(df[required_columns])
        
        # Make predictions
        predictions = model.predict(df.drop(columns=['Amount']))
        df['Fraud Probability'] = predictions
        df['Prediction'] = (df['Fraud Probability'] > 0.5).astype(int)
        
        # Count fraudulent and non-fraudulent transactions
        fraud_count = df['Prediction'].sum()
        legit_count = len(df) - fraud_count
        
        # Display results
        st.subheader("Detection Results")
        st.write(f"**Fraudulent Transactions:** {fraud_count}")
        st.write(f"**Legitimate Transactions:** {legit_count}")
        
        # Show the first few rows
        st.subheader("Sample Predictions")
        st.dataframe(df.head())
        
        # Provide option to download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "fraud_predictions.csv", "text/csv", key='download-csv')