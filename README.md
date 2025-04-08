# Credit Card Fraud Detection using BiLSTM

This project implements a credit card fraud detection system using a Bidirectional LSTM (BiLSTM) neural network. The system can process credit card transaction data and predict whether a transaction is fraudulent or not.

## Features

- Preprocessing of credit card transaction data
- SMOTE-based class balancing
- BiLSTM model for fraud detection
- Streamlit web interface for easy interaction
- Model evaluation metrics and visualization
- Export results to CSV

## Project Structure

```
fraud_detection/
├── model.py           # BiLSTM model implementation
├── app.py            # Streamlit web application
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model (if not already trained):
```python
from model import FraudDetectionModel
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Initialize and train the model
model = FraudDetectionModel()
X, y = model.preprocess_data(df)
X_balanced, y_balanced = model.balance_data(X, y)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Build and train the model
model.build_model(input_shape=(1, 30))
model.train(X_train, y_train, X_val, y_val)
model.save_model()
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the web interface at `http://localhost:8501`

## Input Data Format

The input CSV file should contain the following features:
- V1 to V28: Anonymized features
- Time: Transaction time
- Amount: Transaction amount
- Class (optional): 1 for fraud, 0 for normal

## Model Architecture

- Input shape: (samples, time_steps=1, features=30)
- Bidirectional LSTM layers with dropout
- Dense layers with ReLU and sigmoid activation
- Binary cross-entropy loss function
- Adam optimizer

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- AUC
- Confusion Matrix

## License

This project is licensed under the MIT License - see the LICENSE file for details. 