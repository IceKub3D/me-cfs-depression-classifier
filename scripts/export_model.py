import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os


# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['sleep_stress_interaction'] = X_copy['sleep_quality_index'] * X_copy['stress_level']
        X_copy['fatigue_pem_interaction'] = X_copy['fatigue_severity_scale_score'] * X_copy['pem_duration_hours']
        X_copy['depression_fatigue_ratio'] = X_copy['depression_phq9_score'] / (
                    X_copy['fatigue_severity_scale_score'] + 1e-10)
        X_copy['pain_stress_sum'] = X_copy['physical_pain_score'] + X_copy['stress_level']
        return X_copy


# Load dataset
data_path = "/kaggle/input/mecfs-vs-depression-classification-dataset/me_cfs_vs_depression_dataset.csv"
df = pd.read_csv(data_path)

# Encode categorical variables
categorical_cols = ['gender', 'work_status', 'social_activity_level', 'exercise_frequency', 'meditation_or_mindfulness']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Ensure string input for LabelEncoder
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Encode target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Define numerical columns (including encoded categorical and engineered features)
numerical_cols = ['age', 'sleep_quality_index', 'brain_fog_level', 'physical_pain_score',
                  'stress_level', 'depression_phq9_score', 'fatigue_severity_scale_score',
                  'pem_duration_hours', 'hours_of_sleep_per_night', 'pem_present',
                  'gender', 'work_status', 'social_activity_level', 'exercise_frequency',
                  'meditation_or_mindfulness', 'sleep_stress_interaction',
                  'fatigue_pem_interaction', 'depression_fatigue_ratio', 'pain_stress_sum']

# Preprocessing pipeline
feature_engineer = FeatureEngineer()
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
preprocessor = Pipeline(steps=[
    ('feature_engineer', feature_engineer),
    ('numerical_transformer', numerical_transformer)
])

# Fit preprocessor (comment out if already fitted in your notebook)
preprocessor.fit(X)

# Train best_rf with best results of cross-validation:
# {'class_weight': 'balanced', 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

from sklearn.ensemble import RandomForestClassifier

best_rf = RandomForestClassifier(class_weight='balanced', max_depth=10, min_samples_leaf=1,
                                 min_samples_split=2, n_estimators=100, random_state=42)
best_rf.fit(preprocessor.transform(X), y_encoded)

# Save model, preprocessor, label encoders, and target encoder
output_dir = "/kaggle/working/artifacts"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(best_rf, os.path.join(output_dir, "model.pkl"))
joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
joblib.dump(label_encoders, os.path.join(output_dir, "label_encoders.pkl"))
joblib.dump(target_encoder, os.path.join(output_dir, "target_encoder.pkl"))


# Inference function
def predict(input_data):
    """
    Make predictions using the trained Random Forest Classifier.
    input_data: DataFrame with same columns as training data (excluding diagnosis)
    Returns: Predicted class labels (Depression, ME/CFS, Both)
    """
    # Load model, preprocessor, and encoders
    loaded_model = joblib.load(os.path.join(output_dir, "model.pkl"))
    loaded_preprocessor = joblib.load(os.path.join(output_dir, "preprocessor.pkl"))
    loaded_label_encoders = joblib.load(os.path.join(output_dir, "label_encoders.pkl"))
    loaded_target_encoder = joblib.load(os.path.join(output_dir, "target_encoder.pkl"))

    # Encode categorical variables only if they are not already numerical
    input_data_copy = input_data.copy()
    for col in categorical_cols:
        if input_data_copy[col].dtype not in ['int64', 'int32', 'float64', 'float32']:  # Check if not numerical
            try:
                input_data_copy[col] = loaded_label_encoders[col].transform(input_data_copy[col].astype(str))
            except ValueError as e:
                print(f"Error encoding column {col}: {e}")
                print(f"Unique values in {col}: {input_data_copy[col].unique()}")
                print(f"Known classes for {col}: {loaded_label_encoders[col].classes_}")
                raise

    # Preprocess input
    input_preprocessed = loaded_preprocessor.transform(input_data_copy)

    # Predict
    predictions_encoded = loaded_model.predict(input_preprocessed)

    # Decode predictions
    predictions = loaded_target_encoder.inverse_transform(predictions_encoded)
    return predictions


# Test inference
if __name__ == "__main__":
    # Sample test data (first 5 rows of input data)
    test_data = df.drop(columns=['diagnosis']).head(5)
    predictions = predict(test_data)
    print("Sample predictions:", predictions)
    print("Actual labels:", df['diagnosis'].head(5).values)