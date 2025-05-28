import streamlit as st
import pandas as pd
import numpy as np
from model import FraudDetectionModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import io
from PIL import Image

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

def load_model():
    model = FraudDetectionModel()
    if model.load_model():
        return model
    return None

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def main():
    # Sidebar with model information
    with st.sidebar:
        st.title("Model Information")
        st.markdown("""
        ### Model Architecture
        - 7 Hidden Layers (128 → 64 → 32 → 16 → 8 → 4 → 2 neurons)
        - LeakyReLU Activation
        - Batch Normalization
        - Dropout Regularization
        - Sigmoid Output
        
        ### Performance Metrics
        Accuracy: 0.999663
        
        Macro Average:
        
        Precision: 0.918367
        
        Recall: 0.999831
        
        F1-Score: 0.955471

        Weighted Average:
        
        Precision: 0.999718
        
        Recall: 0.999663
        
        F1-Score: 0.999678
        """)
        
        # Add a placeholder for the model architecture image
        st.image("model_architecture.png", caption="BiLSTM Model Architecture", use_column_width=True)

    # Main content
    st.title("Credit Card Fraud Detection System")
    st.write("Upload a CSV file containing credit card transaction data for fraud detection.")
    
    # Initialize model
    model = load_model()
    if model is None:
        st.error("Model not found. Please train the model first.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the data
            df = pd.read_csv(uploaded_file)
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Check if data has the correct number of features
            expected_features = 30  # V1-V28, Time, Amount
            if len(df.columns) != expected_features and 'Class' not in df.columns:
                st.error(f"Input data must have exactly {expected_features} features (V1-V28, Time, Amount)")
                return
            
            # Remove Class column if it exists
            if 'Class' in df.columns:
                y_true = df['Class']
                df = df.drop('Class', axis=1)
            else:
                y_true = None
            
            # Create two columns for threshold and visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Threshold slider
                threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.01)
            
            with col2:
                # Add a placeholder for real-time visualization
                st.write("Real-time Fraud Detection Visualization")
                # You can add a real-time chart here if needed
            
            # Make predictions
            if st.button("Detect Fraud"):
                with st.spinner("Processing..."):
                    # Get predictions
                    predictions_prob = model.predict(df.values)
                    predictions = (predictions_prob > threshold).astype(int)
                    
                    # Create results dataframe
                    results_df = df.copy()
                    results_df['Fraud_Probability'] = predictions_prob
                    results_df['Fraud_Prediction'] = predictions
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["Results", "Evaluation", "Download"])
                    
                    with tab1:
                        st.subheader("Detection Results")
                        st.dataframe(results_df)
                        
                        # Add a pie chart of fraud vs non-fraud predictions
                        fraud_counts = results_df['Fraud_Prediction'].value_counts()
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.pie(fraud_counts, labels=['Normal', 'Fraud'], autopct='%1.1f%%')
                        ax.set_title('Distribution of Predictions')
                        st.pyplot(fig)
                    
                    with tab2:
                        if y_true is not None:
                            st.subheader("Model Evaluation")
                            
                            # Plot confusion matrix
                            fig = plot_confusion_matrix(y_true, predictions)
                            st.pyplot(fig)
                            
                            # Display classification report
                            report = classification_report(y_true, predictions, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                        else:
                            st.info("No ground truth labels available for evaluation")
                    
                    with tab3:
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 
