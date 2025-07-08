import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utils import FeatureEngineer
import os

# Load dataset
data_path = "/kaggle/input/mecfs-vs-depression-classification-dataset/me_cfs_vs_depression_dataset.csv"
df = pd.read_csv(data_path)

# Encode categorical variables
categorical_cols = ['gender', 'work_status', 'social_activity_level', 'exercise_frequency', 'meditation_or_mindfulness']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df['diagnosis'] = target_encoder.fit_transform(df['diagnosis'])

# Split features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Define preprocessing pipeline
preprocessor = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define model
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(preprocessor.fit_transform(X), y)

# Best model
best_rf = grid_search.best_estimator_

# Save artifacts
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)
joblib.dump(best_rf, os.path.join(artifacts_dir, "model.pkl"))
joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.pkl"))
joblib.dump(label_encoders, os.path.join(artifacts_dir, "label_encoders.pkl"))
joblib.dump(target_encoder, os.path.join(artifacts_dir, "target_encoder.pkl"))

# Test predictions
test_data = df.head(5)
X_test = test_data.drop(columns=['diagnosis'])
y_test = test_data['diagnosis']
X_test_preprocessed = preprocessor.transform(X_test)
y_pred_encoded = best_rf.predict(X_test_preprocessed)
y_pred = target_encoder.inverse_transform(y_pred_encoded)
print("Sample predictions:", y_pred)
print("Actual labels:", target_encoder.inverse_transform(y_test))
