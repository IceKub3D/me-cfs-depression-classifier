
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from utils import FeatureEngineer

app = FastAPI()

# Load artifacts
artifacts_dir = "artifacts"
model = joblib.load(f"{artifacts_dir}/model.pkl")
preprocessor = joblib.load(f"{artifacts_dir}/preprocessor.pkl")
label_encoders = joblib.load(f"{artifacts_dir}/label_encoders.pkl")
target_encoder = joblib.load(f"{artifacts_dir}/target_encoder.pkl")


# Define input schema
class PredictionInput(BaseModel):
    age: float
    sleep_quality_index: float
    brain_fog_level: float
    physical_pain_score: float
    stress_level: float
    depression_phq9_score: float
    fatigue_severity_scale_score: float
    pem_duration_hours: float
    hours_of_sleep_per_night: float
    pem_present: float
    gender: str
    work_status: str
    social_activity_level: str
    exercise_frequency: str
    meditation_or_mindfulness: str


@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([input_data.dict()])

        # Encode categorical variables
        categorical_cols = ['gender', 'work_status', 'social_activity_level', 'exercise_frequency',
                            'meditation_or_mindfulness']
        for col in categorical_cols:
            data[col] = data[col].fillna('Unknown')
            if data[col].dtype not in ['int64', 'int32', 'float64', 'float32']:
                data[col] = label_encoders[col].transform(data[col].astype(str))

        # Preprocess and predict
        data_preprocessed = preprocessor.transform(data)
        prediction_encoded = model.predict(data_preprocessed)
        prediction = target_encoder.inverse_transform(prediction_encoded)[0]

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
