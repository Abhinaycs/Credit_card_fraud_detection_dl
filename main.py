# fraud-detection/
# ├── app/
# │   ├── main.py
# │   ├── models/
# │   │   ├── voting_classifier_model.pkl
# │   │   ├── scaler.pkl
# │   │   └── fraud_detection_model.h5
# │   └── requirements.txt
# ├── Dockerfile
# └── README.md


from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from fastapi.responses import JSONResponse

app = FastAPI()

# Load models and scaler on startup
models = {}
scaler = None

@app.on_event("startup")
async def load_models():
    global models, scaler
    
    # Load ensemble model and scaler
    with open('models/voting_classifier_model.pkl', 'rb') as f:
        models['ensemble'] = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load deep learning model
    models['dl'] = load_model('models/fraud_detection_model.h5')
    
    # Load meta-classifier
    models['meta'] = LogisticRegression(random_state=42)
    # Note: You need to train and save the meta-classifier during development

@app.post("/predict")
async def predict_fraud(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        df = pd.read_csv(file.file)
        
        # Store original data for output
        original_df = df.copy()
        
        # Preprocess data
        if 'Class' in df.columns:
            df = df.drop(['Class'], axis=1)
            
        X = scaler.transform(df)
        
        # Get predictions from both models
        ml_probs = models['ensemble'].predict_proba(X)[:, 1]
        dl_probs = models['dl'].predict(X).flatten()
        
        # Combine predictions
        combined_probs = np.vstack((ml_probs, dl_probs)).T
        
        # Final prediction with meta-classifier
        final_pred = models['meta'].predict(combined_probs)
        
        # Add predictions to original data
        original_df['Prediction'] = final_pred
        
        # Filter and return fraudulent transactions
        frauds = original_df[original_df['Prediction'] == 1]
        return JSONResponse(content=frauds.to_dict(orient='records'))
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing file: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)