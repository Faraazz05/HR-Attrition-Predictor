
# Deployment-ready SHAP prediction function
# Generated on: 2025-09-14T01:28:19.951388

import pandas as pd
import numpy as np
import pickle
import shap
from datetime import datetime

class HRAttritionSHAPPredictor:
    def __init__(self, model_path, explainer_path):
        """
        Initialize the SHAP-enabled attrition predictor.

        Args:
            model_path: Path to the trained model pickle file
            explainer_path: Path to the SHAP explainer pickle file
        """
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.label_encoders = model_data.get('label_encoders', {})

        # Load SHAP explainer
        with open(explainer_path, 'rb') as f:
            explainer_data = pickle.load(f)
            self.explainer = explainer_data.get('explainer')
            if self.explainer is None:
                print("Warning: No explainer found, predictions will not include SHAP values")

    def predict_with_explanation(self, employee_data):
        """
        Predict attrition probability with SHAP explanation.

        Args:
            employee_data: Dictionary or DataFrame with employee features

        Returns:
            Dictionary with prediction and SHAP explanation
        """
        # Preprocess data
        if isinstance(employee_data, dict):
            df = pd.DataFrame([employee_data])
        else:
            df = employee_data.copy()

        # Apply label encoding
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except:
                    df[col] = 0  # Default value if encoding fails

        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value

        # Select features in correct order
        X = df[self.feature_names]

        # Make prediction
        try:
            probability = self.model.predict_proba(X)[0, 1]
        except:
            probability = self.model.predict(X)[0] if len(self.model.predict(X)) > 0 else 0.5

        # Generate SHAP explanation
        explanation = {
            'attrition_probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'prediction_timestamp': datetime.now().isoformat()
        }

        # Add SHAP explanation if available
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X)

                # Handle different SHAP value formats
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]  # Positive class
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 1]  # Positive class

                # Ensure 2D array
                if shap_values.ndim > 2:
                    shap_values = shap_values[0]

                explanation['feature_contributions'] = {
                    feature: float(shap_val) 
                    for feature, shap_val in zip(self.feature_names, shap_values[0])
                }

                explanation['top_risk_factors'] = [
                    {
                        'feature': feature,
                        'impact': float(shap_val),
                        'direction': 'increases' if shap_val > 0 else 'decreases'
                    }
                    for feature, shap_val in sorted(
                        zip(self.feature_names, shap_values[0]), 
                        key=lambda x: abs(x[1]), reverse=True
                    )[:5]
                ]
            except Exception as e:
                explanation['shap_error'] = f"Could not generate SHAP explanation: {e}"

        return explanation

# Example usage:
# predictor = HRAttritionSHAPPredictor('trained_model_[timestamp].pkl', 'shap_explainer_[timestamp].pkl')
# result = predictor.predict_with_explanation({'Age': 35, 'MonthlyIncome': 5000, 'JobSatisfaction': 3})
# print(f"Attrition Risk: {result['risk_level']} ({result['attrition_probability']:.1%})")
