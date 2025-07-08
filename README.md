ME/CFS vs Depression Classifier
This repository contains a Random Forest Classifier for classifying patients into Depression, ME/CFS, or Both, based on a dataset from Kaggle. The model is trained using scikit-learn and optimized with GridSearchCV.
Dataset

Source: ME/CFS vs Depression Classification Dataset
Features: 15 original features (10 numerical, 5 categorical) plus 4 engineered features
Target: diagnosis (Depression, ME/CFS, Both)

Repository Structure

artifacts/: Model (model.pkl), preprocessor (preprocessor.pkl), label encoders (label_encoders.pkl), target encoder (target_encoder.pkl)
data/: Test data (test_data.csv)
scripts/: export_model.py for exporting, test_model.py for testing, api.py for deployment
docs/: model_details.md with model and preprocessing details
requirements.txt: Python dependencies
.gitignore: Excludes temporary files and dataset

Usage

Install dependencies: pip install -r requirements.txt
Run scripts/test_model.py to validate the model.
Deploy using scripts/api.py with FastAPI: uvicorn scripts.api:app --host 0.0.0.0 --port 8000

Dependencies

scikit-learn==1.2.2
joblib==1.2.0
numpy==1.24.3
pandas==2.0.3
fastapi==0.103.0
uvicorn==0.23.2

Model Details
See docs/model_details.md for hyperparameters, preprocessing, and input/output formats.
CI/CD

GitHub Actions: .github/workflows/ci.yml runs tests on push/pull requests.
Deployment: FastAPI app deployed on Render.
