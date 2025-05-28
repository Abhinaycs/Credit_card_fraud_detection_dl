import streamlit as st
import pandas as pd
import numpy as np
from model import FraudDetectionModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

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
    with st.sidebar:
        st.title("Model Information")
        st.markdown("""
        ### Model Architecture
        - BiLSTM Layers with Dropout
        - Dense Final Layers
        - Sigmoid Output

        ### Performance Metrics
        Accuracy: 0.999663

        Macro Avg:
        - Precision: 0.918
        - Recall: 0.999
        - F1: 0.955

        Weighted Avg:
        - Precision: 0.9997
        - Recall: 0.9996
        - F1: 0.9997
        """)
        st.image("model_architecture.png", caption="BiLSTM Model Architecture", use_column_width=True)

    st.title("Credit Card Fraud Detection System")
    st.write("Upload a CSV file with features like Time, Amount, V1-V28 (without 'Class').")

    model = load_model()
    if model is None:
        st.error("Model not found. Please train and save the model first.")
        return

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())

            expected_features = 30  # V1-V28, Time, Amount
            if len(df.columns) != expected_features and 'Class' not in df.columns:
                st.error(f"Data must have exactly {expected_features} columns (V1-V28, Time, Amount).")
                return

            if 'Class' in df.columns:
                y_true = df['Class']
                df = df.drop('Class', axis=1)
            else:
                y_true = None

            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.01)

            with col2:
                st.write("Real-time Fraud Detection Visualization")

            if st.button("Detect Fraud"):
                with st.spinner("Processing..."):
                    predictions_prob = model.predict(df.values)
                    predictions = (predictions_prob > threshold).astype(int)

                    results_df = df.copy()
                    results_df['Fraud_Probability'] = predictions_prob
                    results_df['Fraud_Prediction'] = predictions

                    tab1, tab2, tab3 = st.tabs(["Results", "Evaluation", "Download"])

                    with tab1:
                        st.subheader("Detection Results")
                        st.dataframe(results_df)
                        fraud_counts = results_df['Fraud_Prediction'].value_counts()
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.pie(fraud_counts, labels=['Normal', 'Fraud'], autopct='%1.1f%%')
                        ax.set_title('Prediction Distribution')
                        st.pyplot(fig)

                    with tab2:
                        if y_true is not None:
                            st.subheader("Model Evaluation")
                            fig = plot_confusion_matrix(y_true, predictions)
                            st.pyplot(fig)
                            report = classification_report(y_true, predictions, output_dict=True)
                            st.dataframe(pd.DataFrame(report).transpose())
                        else:
                            st.info("Ground truth labels not provided for evaluation.")

                    with tab3:
                        csv = results_df.to_csv(index=False)
                        st.download_button("Download Results", data=csv, file_name="fraud_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
