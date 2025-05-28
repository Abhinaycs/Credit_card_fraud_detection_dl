import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_loaded = False

    def load_model(self, model_path="bilstm_fraud_detection.h5", scaler_path="scaler.npy"):
        try:
            self.model = load_model(model_path)
            scaler_data = np.load(scaler_path, allow_pickle=True)
            self.scaler.mean_, self.scaler.scale_ = scaler_data
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            return False

    def predict(self, input_data):
        if not self.model_loaded:
            raise Exception("Model not loaded. Call load_model() first.")
        
        input_scaled = self.scaler.transform(input_data)
        input_reshaped = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1]))
        predictions_prob = self.model.predict(input_reshaped)
        return predictions_prob.flatten()
