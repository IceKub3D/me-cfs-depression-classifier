
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
        le_classes = label_encoders[col].classes_.tolist()
        le_classes.append('Unknown')
        label_encoders[col].classes_ = np.array(le_classes)
    df[col] = label_encoders[col].transform(df[col].astype(str))

# Save updated label encoders
joblib.dump(label_encoders, label_encoders_path)

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df = test_df.head(10)  # Take 10 rows for testing
test_df.to_csv('data/test_data.csv', index=False)

print("Test data saved to data/test_data.csv")
print(test_df[categorical_cols].head())
print(test_df.dtypes)
