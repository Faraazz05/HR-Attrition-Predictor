import pandas as pd
import numpy as np
import pickle
from datetime import datetime

class HRAttritionPredictor:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model_package = pickle.load(f)
        self.model = self.model_package['model']
        self.scaler = self.model_package['scaler']
        self.feature_names = self.model_package['feature_names']
        self.label_encoders = self.model_package['label_encoders']

    def predict(self, data):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Apply encodings
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except:
                    df[col] = 0

        # Ensure all features
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        X = df[self.feature_names].values
        if self.scaler:
            X = self.scaler.transform(X)

        prediction = self.model.predict(X)[0]
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X)[0, 1]
        else:
            probability = 0.5

        risk_level = "High" if probability >= 0.7 else "Medium" if probability >= 0.3 else "Low"

        return {
            'prediction': 'Attrition' if prediction == 1 else 'No Attrition',
            'probability': float(probability),
            'risk_level': risk_level,
            'model_used': self.model_package['model_name'],
            'prediction_date': datetime.now().isoformat()
        }

# Usage example:
# predictor = HRAttritionPredictor("model_file.pkl")
# result = predictor.predict({'Age': 35, 'Department': 'Engineering', 'MonthlyIncome': 8000})
# print(f"Risk: {result['risk_level']} ({result['probability']:.1%})")
