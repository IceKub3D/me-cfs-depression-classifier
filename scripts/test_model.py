import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os

# Load artifacts
artifacts_dir = "artifacts"
try:
    model = joblib.load(os.path.join(artifacts_dir, "model.pkl"))
    preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.pkl"))
    label_encoders = joblib.load(os.path.join(artifacts_dir, "label_encoders.pkl"))
    target_encoder = joblib.load(os.path.join(artifacts_dir, "target_encoder.pkl"))
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise

# Load test data
test_data_path = "data/test_data.csv"
try:
    test_df = pd.read_csv(test_data_path)
except FileNotFoundError:
    print(f"Error: {test_data_path} not found")
    raise

# Verify columns
expected_columns = ['diagnosis', 'age', 'sleep_quality_index', 'brain_fog_level', 'physical_pain_score',
                    'stress_level', 'depression_phq9_score', 'fatigue_severity_scale_score',
                    'pem_duration_hours', 'hours_of_sleep_per_night', 'pem_present',
                    'gender', 'work_status', 'social_activity_level', 'exercise_frequency',
                    'meditation_or_mindfulness']
if not all(col in test_df.columns for col in expected_columns):
    print(f"Error: Missing columns in test_data.csv. Expected: {expected_columns}")
    raise ValueError("Invalid columns in test data")

X_test = test_df.drop(columns=['diagnosis'])
y_test = test_df['diagnosis']

# Encode categorical variables
categorical_cols = ['gender', 'work_status', 'social_activity_level', 'exercise_frequency', 'meditation_or_mindfulness']
for col in categorical_cols:
    if X_test[col].dtype not in ['int64', 'int32', 'float64', 'float32']:
        try:
            X_test[col] = label_encoders[col].transform(X_test[col].astype(str))
        except ValueError as e:
            print(f"Error encoding {col}: {e}")
            print(f"Unique values in {col}: {X_test[col].unique()}")
            print(f"Expected classes: {label_encoders[col].classes_}")
            raise

# Preprocess and predict
try:
    X_test_preprocessed = preprocessor.transform(X_test)
    y_pred_encoded = model.predict(X_test_preprocessed)
    y_pred = target_encoder.inverse_transform(y_pred_encoded)
    y_test_encoded = target_encoder.transform(y_test)
except Exception as e:
    print(f"Error during preprocessing or prediction: {e}")
    raise

# Calculate F1 score
f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
print(f"Test F1 Score: {f1}")

# Assert the model performs well
assert f1 > 0.9, f"Test F1 score {f1} is below threshold 0.9"