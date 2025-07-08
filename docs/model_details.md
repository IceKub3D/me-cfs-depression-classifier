Model Details
Overview

Model Type: Random Forest Classifier (scikit-learn, optimized via GridSearchCV)
Task: Multi-class classification
Classes: Depression, ME/CFS, Both
Framework: scikit-learn
Version: 1.0
Hyperparameters: 
class_weight: balanced
max_depth: 10
min_samples_leaf: 1
min_samples_split: 2
n_estimators: 100


Best CV F1 Score: 0.9913

Input

Format: DataFrame with 15 features (before encoding and feature engineering)
Numerical (10): age, sleep_quality_index, brain_fog_level, physical_pain_score, stress_level, depression_phq9_score, fatigue_severity_scale_score, pem_duration_hours, hours_of_sleep_per_night, pem_present
Categorical (5): gender, work_status, social_activity_level, exercise_frequency, meditation_or_mindfulness


Preprocessing:
Categorical Encoding: Apply LabelEncoder to each categorical column
Feature Engineering:
sleep_stress_interaction: sleep_quality_index * stress_level
fatigue_pem_interaction: fatigue_severity_scale_score * pem_duration_hours
depression_fatigue_ratio: depression_phq9_score / (fatigue_severity_scale_score + 1e-10)
pain_stress_sum: physical_pain_score + stress_level


Numerical (19, after encoding and feature engineering): Impute missing values with mean, apply StandardScaler
Target: Encode diagnosis using LabelEncoder



Output

Format: Predicted class labels (Depression, ME/CFS, Both)
Alternative Output: Probabilities available via model.predict_proba()

Dependencies

scikit-learn==1.2.2
joblib==1.2.0
numpy==1.24.3
pandas==2.0.3
fastapi==0.103.0
uvicorn==0.23.2

Artifacts

model.pkl: Trained Random Forest Classifier
preprocessor.pkl: Pipeline with feature engineering, imputation, and scaling
label_encoders.pkl: Dictionary of LabelEncoders for categorical columns
target_encoder.pkl: LabelEncoder for decoding predictions
export_model.py: Script for exporting and testing model

Usage
Run scripts/export_model.py to save artifacts to artifacts/. Use scripts/test_model.py to validate the model. Deploy using scripts/api.py with FastAPI.