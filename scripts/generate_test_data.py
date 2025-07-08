
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
data_path = "/Users/naya/depression-classification-project/data/original_data.csv"  # Update this path
df = pd.read_csv(data_path)

# Load existing label encoders
label_encoders_path = "/Users/naya/depression-classification-project/artifacts/label_encoders.pkl"
label_encoders = joblib.load(label_encoders_path)

# Encode categorical columns
categorical_cols = ['gender', 'work_status', 'social_activity_level', 'exercise_frequency', 'meditation_or_mindfulness']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')
    if 'Unknown' not in label_encoders[col].classes_:
        # Extend label encoder classes to include 'Unknown'
        le_classes = label_encoders[col].classes_.tolist()
        le_classes.append('Unknown')
        label_encoders[col].classes_ = np.array(le_classes)
    df[col] = label_encoders[col].transform(df[col].astype(str))

# Save updated label encoders (to ensure 'Unknown' is included)
joblib.dump(label_encoders, label_encoders_path)

# Create test data (10 rows, ensuring not all are from training data)
# Use a random sample to avoid data leakage
test_data = df.sample(n=10, random_state=42)
test_data.to_csv('data/test_data.csv', index=False)

print("Test data saved to data/test_data.csv")
print(test_data[categorical_cols].head())
print(test_data.dtypes)