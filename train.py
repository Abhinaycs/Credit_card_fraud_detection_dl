from model import FraudDetectionModel
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('creditcard.csv')

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