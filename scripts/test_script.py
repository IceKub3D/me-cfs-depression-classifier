import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os

# Load artifacts
artifacts_dir = "artifacts"
model = joblib.load(os.path.join(artifacts_dir, "model.pkl"))
preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.pkl"))
label_encoders = joblib.load(os.path.join(artifacts_dir, "label_encoders.pkl"))
target_encoder = joblib.load(os.path.join(artifacts_dir, "target_encoder.pkl"))

# Load test data (use a small subset of the dataset for CI/CD)
# For this example, assume a small CSV with 10 rows is included in the repo
test_data_path = "data/test_data.csv"  # You’ll create this
test_df = pd.read_csv(test_data_path)
X_test = test_df.drop(columns=['diagnosis'])
y_test = test_df['diagnosis']

# Encode categorical variables
categorical_cols = ['gender', 'work_status', 'social_activity_level', 'exercise_frequency', 'meditation_or_mindfulness']
for col in categorical_cols:
    if X_test[col].dtype not in ['int64', 'int32', 'float64', 'float32']:
        X_test[col] = label_encoders[col].transform(X_test[col].astype(str))

# Preprocess and predict
X_test_preprocessed = preprocessor.transform(X_test)
y_pred_encoded = model.predict(X_test_preprocessed)
y_pred = target_encoder.inverse_transform(y_pred_encoded)
y_test_encoded = target_encoder.transform(y_test)

# Calculate F1 score
f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
print(f"Test F1 Score: {f1}")

# Assert the model performs well
assert f1 > 0.9, f"Test F1 score {f1} is below threshold 0.9"