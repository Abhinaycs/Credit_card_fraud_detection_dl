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
import h5py

class FraudDetectionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        
    def preprocess_data(self, df):
        df = df.drop_duplicates().dropna()
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_scaled = self.scaler.fit_transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        return X_reshaped, y
    
    def balance_data(self, X, y):
        smote = SMOTE(random_state=42)
        X_flattened = X.reshape(X.shape[0], X.shape[2])
        X_balanced, y_balanced = smote.fit_resample(X_flattened, y)
        X_balanced = X_balanced.reshape(X_balanced.shape[0], 1, X_balanced.shape[1])
        return X_balanced, y_balanced
    
    def build_model(self, input_shape):
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
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        callbacks = [
            ModelCheckpoint('bilstm_fraud_detection.h5', monitor='val_loss', save_best_only=True, mode='min'),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                              epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Scale and reshape input
        if len(X.shape) == 2:  # (samples, features)
            X_scaled = self.scaler.transform(X)
            X = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        elif len(X.shape) == 3:  # (samples, 1, features)
            X_flattened = X.reshape(X.shape[0], X.shape[2])
            X_scaled = self.scaler.transform(X_flattened)
            X = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        else:
            raise ValueError("Input data has incorrect shape")

        return self.model.predict(X)
    
    def save_model(self, path='bilstm_fraud_detection.h5'):
        if self.model:
            self.model.save(path, save_format='h5')
            np.save('scaler.npy', self.scaler)

    def load_model(self, path='bilstm_fraud_detection.h5'):
        try:
            if os.path.exists(path):
                self.model = load_model(path)
                scaler_path = 'scaler.npy'
                if os.path.exists(scaler_path):
                    self.scaler = np.load(scaler_path, allow_pickle=True).item()
                else:
                    print("Warning: Scaler file not found")
                return True
            else:
                print(f"Model file not found at {path}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
