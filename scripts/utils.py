from sklearn.base import BaseEstimator, TransformerMixin


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


