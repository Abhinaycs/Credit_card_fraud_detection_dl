import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

class FraudDetectionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        
    def preprocess_data(self, df):
        """Preprocess the input dataframe."""
        # Drop duplicates and null values
        df = df.drop_duplicates()
        df = df.dropna()
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape for BiLSTM (samples, time_steps, features)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        return X_reshaped, y
    
    def balance_data(self, X, y):
        """Apply SMOTE to balance the dataset."""
        smote = SMOTE(random_state=42)
        X_reshaped = X.reshape(X.shape[0], X.shape[2])
        X_balanced, y_balanced = smote.fit_resample(X_reshaped, y)
        return X_balanced.reshape(X_balanced.shape[0], 1, X_balanced.shape[1]), y_balanced
    
    def build_model(self, input_shape):
        """Build the BiLSTM model."""
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model with callbacks."""
        callbacks = [
            ModelCheckpoint(
                'bilstm_fraud_detection.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Ensure input is properly shaped
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Scale the input
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], X.shape[2]))
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        return self.model.predict(X_reshaped)
    
    def save_model(self, path='bilstm_fraud_detection.keras'):
        """Save the model and scaler."""
        if self.model is not None:
            self.model.save(path)
            np.save('scaler.npy', self.scaler)
    
    def load_model(self, path='bilstm_fraud_detection.keras'):
        """Load the model and scaler."""
        if os.path.exists(path):
            self.model = load_model(path)
            self.scaler = np.load('scaler.npy', allow_pickle=True).item()
            return True
        return False 