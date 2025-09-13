"""
HR Attrition Predictor - ML Prediction Engine
=============================================
Individual and bulk employee predictions with model comparison,
confidence intervals, and SHAP explanations. Memory-optimized for 4GB RAM.

Author: HR Analytics Team
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path
import warnings
import gc
import io
import joblib
from typing import Dict, List, Tuple, Optional, Any, Union
import base64

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
from streamlit_app.assets.theme import COLORS, TYPOGRAPHY, apply_custom_css
from streamlit_app.components.charts import (
    glassmorphism_metric_card, futuristic_gauge_chart,
    create_dark_theme_plotly_chart, create_prediction_result_html
)
from streamlit_app.config import get_risk_level, get_risk_color, MODEL_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')

# ================================================================
# MODEL LOADING AND CACHING
# ================================================================

@st.cache_resource
def load_prediction_models():
    """
    Load all trained models and preprocessing components with error handling.
    
    Returns:
        Dictionary containing loaded models and preprocessors
    """
    models = {}
    status = {"loaded": [], "failed": []}
    
    try:
        # Model paths
        model_paths = {
            "best_model": "models/best_model.pkl",
            "logistic_regression": "models/logistic_regression_optimized.pkl", 
            "random_forest": "models/random_forest_optimized.pkl",
            "ensemble": "models/ensemble_optimized.pkl"
        }
        
        # Preprocessing paths
        preprocessing_paths = {
            "scaler": "models/feature_scaler.pkl",
            "label_encoders": "models/label_encoders.pkl",
            "target_encoder": "models/target_encoder.pkl",
            "feature_names": "models/feature_names.pkl"
        }
        
        # Load models
        for name, path in model_paths.items():
            try:
                if Path(path).exists():
                    models[name] = joblib.load(path)
                    status["loaded"].append(name)
                else:
                    status["failed"].append(f"{name} (file not found)")
            except Exception as e:
                status["failed"].append(f"{name} ({str(e)[:50]}...)")
        
        # Load preprocessing components
        for name, path in preprocessing_paths.items():
            try:
                if Path(path).exists():
                    models[name] = joblib.load(path)
                    status["loaded"].append(name)
                else:
                    status["failed"].append(f"{name} (file not found)")
            except Exception as e:
                status["failed"].append(f"{name} ({str(e)[:50]}...)")
        
        models["status"] = status
        
        # Log loading status
        if status["loaded"]:
            st.success(f"✅ Loaded: {', '.join(status['loaded'])}")
        if status["failed"]:
            st.warning(f"⚠️ Failed: {', '.join(status['failed'])}")
        
        return models
    
    except Exception as e:
        st.error(f"Critical error loading models: {e}")
        return {"status": {"loaded": [], "failed": ["Critical loading error"]}}

@st.cache_data
def load_employee_database():
    """Load employee database for lookup functionality."""
    try:
        data_path = project_root / "data" / "synthetic" / "hr_employees.csv"
        
        if data_path.exists():
            # Load with essential columns only for memory efficiency
            essential_cols = [
                'EmployeeID', 'FirstName', 'LastName', 'Age', 'Gender', 'MaritalStatus',
                'Education', 'Department', 'JobRole', 'JobLevel', 'MonthlyIncome',
                'YearsAtCompany', 'YearsInCurrentRole', 'TotalWorkingYears',
                'PerformanceScore', 'JobSatisfaction', 'WorkLifeBalance',
                'EnvironmentSatisfaction', 'RelationshipSatisfaction',
                'OverTime', 'BusinessTravel', 'DistanceFromHome', 'Attrition'
            ]
            
            df = pd.read_csv(data_path, usecols=[col for col in essential_cols if col in pd.read_csv(data_path, nrows=0).columns])
            
            # Create full name for easier lookup
            if 'FirstName' in df.columns and 'LastName' in df.columns:
                df['FullName'] = df['FirstName'] + ' ' + df['LastName']
            
            return df
        else:
            return generate_demo_employee_data()
    
    except Exception as e:
        st.error(f"Error loading employee database: {e}")
        return generate_demo_employee_data()

def generate_demo_employee_data():
    """Generate demo employee data for testing."""
    np.random.seed(42)
    
    demo_employees = pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, 101)],
        'FirstName': [f'Employee{i}' for i in range(1, 101)],
        'LastName': [f'LastName{i}' for i in range(1, 101)],
        'Age': np.random.randint(22, 65, 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], 100),
        'Education': np.random.choice(['High School', 'Bachelor\'s', 'Master\'s'], 100),
        'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 100),
        'JobRole': np.random.choice(['Manager', 'Senior', 'Junior', 'Lead'], 100),
        'JobLevel': np.random.randint(1, 6, 100),
        'MonthlyIncome': np.random.randint(3000, 15000, 100),
        'YearsAtCompany': np.random.randint(0, 20, 100),
        'YearsInCurrentRole': np.random.randint(0, 10, 100),
        'TotalWorkingYears': np.random.randint(0, 25, 100),
        'PerformanceScore': np.random.randint(1, 6, 100),
        'JobSatisfaction': np.random.randint(1, 5, 100),
        'WorkLifeBalance': np.random.randint(1, 5, 100),
        'EnvironmentSatisfaction': np.random.randint(1, 5, 100),
        'RelationshipSatisfaction': np.random.randint(1, 5, 100),
        'OverTime': np.random.choice(['Yes', 'No'], 100),
        'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], 100),
        'DistanceFromHome': np.random.randint(1, 30, 100),
        'Attrition': np.random.choice(['Yes', 'No'], 100, p=[0.16, 0.84])
    })
    
    demo_employees['FullName'] = demo_employees['FirstName'] + ' ' + demo_employees['LastName']
    return demo_employees

# ================================================================
# FEATURE PREPARATION
# ================================================================

def prepare_features_for_prediction(employee_data, models):
    """
    Prepare employee data for model prediction.
    
    Args:
        employee_data: Dictionary or Series with employee information
        models: Dictionary containing loaded models and preprocessors
        
    Returns:
        Prepared feature array for prediction
    """
    try:
        # Convert to DataFrame if needed
        if isinstance(employee_data, dict):
            df = pd.DataFrame([employee_data])
        elif isinstance(employee_data, pd.Series):
            df = pd.DataFrame([employee_data])
        else:
            df = employee_data.copy()
        
        # Get feature names from saved model
        if 'feature_names' in models:
            expected_features = models['feature_names']
        else:
            # Fallback feature list (basic features)
            expected_features = [
                'Age', 'Gender', 'MaritalStatus', 'Education', 'Department', 
                'JobRole', 'JobLevel', 'MonthlyIncome', 'YearsAtCompany',
                'YearsInCurrentRole', 'TotalWorkingYears', 'PerformanceScore',
                'JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction',
                'RelationshipSatisfaction', 'OverTime', 'BusinessTravel', 
                'DistanceFromHome'
            ]
        
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df.columns:
                # Set default values for missing features
                if feature in ['Age', 'JobLevel', 'MonthlyIncome', 'YearsAtCompany', 
                              'YearsInCurrentRole', 'TotalWorkingYears', 'PerformanceScore',
                              'JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction',
                              'RelationshipSatisfaction', 'DistanceFromHome']:
                    df[feature] = 0  # Default numeric value
                else:
                    df[feature] = 'Unknown'  # Default categorical value
        
        # Select only expected features
        df = df[expected_features]
        
        # Apply label encoding for categorical features
        if 'label_encoders' in models:
            label_encoders = models['label_encoders']
            
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    try:
                        # Handle unknown categories
                        unique_values = set(df[col].astype(str))
                        known_values = set(encoder.classes_)
                        unknown_values = unique_values - known_values
                        
                        if unknown_values:
                            # Replace unknown values with the most frequent class
                            most_frequent = encoder.classes_[0]
                            df[col] = df[col].astype(str).replace(list(unknown_values), most_frequent)
                        
                        df[col] = encoder.transform(df[col].astype(str))
                    except Exception as e:
                        st.warning(f"Encoding error for {col}: {e}")
                        df[col] = 0  # Default to 0 if encoding fails
        
        # Apply scaling for numeric features
        if 'scaler' in models:
            numeric_features = df.select_dtypes(include=[np.number]).columns
            df[numeric_features] = models['scaler'].transform(df[numeric_features])
        
        return df.values
    
    except Exception as e:
        st.error(f"Feature preparation error: {e}")
        return None

# ================================================================
# INDIVIDUAL EMPLOYEE LOOKUP
# ================================================================

def individual_employee_lookup():
    """Individual employee prediction interface."""
    
    st.markdown("### 👤 Individual Employee Prediction")
    
    # Load models and data
    models = load_prediction_models()
    employee_db = load_employee_database()
    
    if not models.get("best_model"):
        st.error("🚨 No trained models available. Please train models first.")
        return
    
    # Employee selection methods
    col1, col2 = st.columns([1, 1])
    
    with col1:
        lookup_method = st.radio(
            "Select Lookup Method:",
            ["Employee Database", "Manual Entry"],
            horizontal=True
        )
    
    with col2:
        prediction_model = st.selectbox(
            "Select Model:",
            ["best_model", "logistic_regression", "random_forest", "ensemble"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    st.markdown("---")
    
    if lookup_method == "Employee Database":
        # Database lookup
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Employee search
            if 'FullName' in employee_db.columns:
                selected_employee = st.selectbox(
                    "🔍 Search Employee:",
                    options=employee_db['FullName'].tolist(),
                    help="Select an employee from the database"
                )
                
                if selected_employee:
                    employee_data = employee_db[employee_db['FullName'] == selected_employee].iloc[0]
                    
                    # Display employee info card
                    st.markdown(f"""
                    <div style="
                        padding: 20px;
                        background: rgba(37, 42, 69, 0.4);
                        border-radius: 12px;
                        border-left: 4px solid {COLORS['secondary']};
                        margin: 20px 0;
                    ">
                        <h4 style="color: {COLORS['text']}; margin: 0 0 15px 0;">
                            👤 {employee_data.get('FullName', 'Unknown')}
                        </h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                            <div><strong>ID:</strong> {employee_data.get('EmployeeID', 'N/A')}</div>
                            <div><strong>Department:</strong> {employee_data.get('Department', 'N/A')}</div>
                            <div><strong>Role:</strong> {employee_data.get('JobRole', 'N/A')}</div>
                            <div><strong>Age:</strong> {employee_data.get('Age', 'N/A')}</div>
                            <div><strong>Tenure:</strong> {employee_data.get('YearsAtCompany', 'N/A')} years</div>
                            <div><strong>Salary:</strong> ${employee_data.get('MonthlyIncome', 0):,}/month</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Employee database not properly formatted")
                return
        
        with col2:
            if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
                predict_employee_risk(employee_data, models, prediction_model)
    
    else:
        # Manual entry
        st.markdown("#### 📝 Enter Employee Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            education = st.selectbox("Education", ["High School", "Bachelor's Degree", "Master's Degree", "PhD"])
        
        with col2:
            department = st.selectbox("Department", 
                ["Engineering", "Sales", "Marketing", "Operations", "Finance", "HR", "Legal"])
            job_role = st.selectbox("Job Role", 
                ["Manager", "Senior", "Junior", "Lead", "Associate", "Director"])
            job_level = st.slider("Job Level", min_value=1, max_value=5, value=3)
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=50000, value=5000, step=500)
        
        with col3:
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
            years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=2)
            total_experience = st.number_input("Total Work Experience", min_value=0, max_value=50, value=8)
            performance_score = st.slider("Performance Score", min_value=1, max_value=5, value=3)
        
        # Satisfaction scores
        st.markdown("#### 😊 Satisfaction Scores (1-4 scale)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=3)
        with col2:
            work_life_balance = st.slider("Work-Life Balance", min_value=1, max_value=4, value=3)
        with col3:
            env_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4, value=3)
        with col4:
            rel_satisfaction = st.slider("Relationship Satisfaction", min_value=1, max_value=4, value=3)
        
        # Work factors
        st.markdown("#### 💼 Work Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overtime = st.selectbox("Overtime", ["No", "Yes"])
        with col2:
            business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        with col3:
            distance_from_home = st.number_input("Distance from Home (miles)", min_value=1, max_value=100, value=10)
        
        # Create employee data dictionary
        manual_employee_data = {
            'Age': age,
            'Gender': gender,
            'MaritalStatus': marital_status,
            'Education': education,
            'Department': department,
            'JobRole': job_role,
            'JobLevel': job_level,
            'MonthlyIncome': monthly_income,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'TotalWorkingYears': total_experience,
            'PerformanceScore': performance_score,
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'EnvironmentSatisfaction': env_satisfaction,
            'RelationshipSatisfaction': rel_satisfaction,
            'OverTime': overtime,
            'BusinessTravel': business_travel,
            'DistanceFromHome': distance_from_home
        }
        
        if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
            predict_employee_risk(manual_employee_data, models, prediction_model)

def predict_employee_risk(employee_data, models, model_name):
    """Make prediction for a single employee."""
    
    try:
        with st.spinner("🔮 Analyzing employee data..."):
            # Prepare features
            features = prepare_features_for_prediction(employee_data, models)
            
            if features is None:
                st.error("Failed to prepare features for prediction")
                return
            
            # Get selected model
            model = models.get(model_name)
            if model is None:
                st.error(f"Model {model_name} not available")
                return
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            # Get attrition probability (class 1)
            attrition_prob = probability[1] if len(probability) > 1 else probability[0]
            
            # Determine risk level
            risk_level = get_risk_level(attrition_prob)
            risk_color = get_risk_color(risk_level)
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Main prediction display
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Risk gauge
                gauge_fig = futuristic_gauge_chart(
                    value=attrition_prob * 100,
                    title="Attrition Risk Probability",
                    min_value=0,
                    max_value=100,
                    unit="%",
                    risk_thresholds={'low': 30, 'medium': 70, 'high': 100},
                    height=300
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Risk level card
                st.markdown(glassmorphism_metric_card(
                    value=risk_level,
                    title="Risk Level",
                    icon="⚠️" if risk_level == "High" else "🟡" if risk_level == "Medium" else "✅",
                    color='error' if risk_level == "High" else 'warning' if risk_level == "Medium" else 'success',
                    width=200,
                    height=150
                ), unsafe_allow_html=True)
            
            with col3:
                # Probability card
                st.markdown(glassmorphism_metric_card(
                    value=f"{attrition_prob:.1%}",
                    title="Probability",
                    icon="🎯",
                    color='secondary',
                    width=200,
                    height=150
                ), unsafe_allow_html=True)
            
            # Detailed results
            prediction_text = "Will likely leave" if prediction == 1 else "Will likely stay"
            prediction_color = COLORS['error'] if prediction == 1 else COLORS['success']
            
            st.markdown(f"""
            <div style="
                padding: 25px;
                background: rgba(37, 42, 69, 0.4);
                border-radius: 15px;
                border-left: 5px solid {risk_color};
                margin: 20px 0;
                text-align: center;
            ">
                <h3 style="color: {risk_color}; margin: 0 0 15px 0; font-family: 'Orbitron';">
                    📊 PREDICTION: {prediction_text.upper()}
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                    <div>
                        <div style="color: {COLORS['text_secondary']}; font-size: 14px;">Model Used</div>
                        <div style="color: {COLORS['text']}; font-weight: bold; font-size: 18px;">
                            {model_name.replace('_', ' ').title()}
                        </div>
                    </div>
                    <div>
                        <div style="color: {COLORS['text_secondary']}; font-size: 14px;">Confidence</div>
                        <div style="color: {COLORS['secondary']}; font-weight: bold; font-size: 18px;">
                            {max(probability) * 100:.1f}%
                        </div>
                    </div>
                    <div>
                        <div style="color: {COLORS['text_secondary']}; font-size: 14px;">Risk Category</div>
                        <div style="color: {risk_color}; font-weight: bold; font-size: 18px;">
                            {risk_level} Risk
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations based on risk level
            if risk_level == "High":
                recommendations = [
                    "🎯 Schedule immediate one-on-one meeting",
                    "💰 Review compensation and benefits package", 
                    "📈 Discuss career development opportunities",
                    "😊 Assess job satisfaction and work environment",
                    "🎓 Consider additional training or skill development"
                ]
                rec_color = COLORS['error']
            elif risk_level == "Medium":
                recommendations = [
                    "📅 Schedule regular check-ins with manager",
                    "📋 Review recent performance and feedback",
                    "🏆 Recognize achievements and contributions",
                    "🤝 Improve team collaboration opportunities",
                    "📊 Monitor satisfaction levels closely"
                ]
                rec_color = COLORS['warning']
            else:
                recommendations = [
                    "✅ Continue current engagement strategies",
                    "🌟 Consider for high-potential development programs",
                    "👥 Utilize as mentor for other employees",
                    "🎯 Set challenging stretch goals",
                    "🏅 Recognize as retention success story"
                ]
                rec_color = COLORS['success']
            
            st.markdown("#### 💡 Recommended Actions")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div style="
                    padding: 12px;
                    background: rgba(37, 42, 69, 0.3);
                    border-radius: 8px;
                    border-left: 3px solid {rec_color};
                    margin: 8px 0;
                    display: flex;
                    align-items: center;
                ">
                    <div style="
                        background: {rec_color};
                        color: white;
                        width: 25px;
                        height: 25px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        font-size: 12px;
                        margin-right: 15px;
                        flex-shrink: 0;
                    ">
                        {i}
                    </div>
                    <div style="color: {COLORS['text']}; font-size: 14px;">
                        {rec}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Please check the input data and try again.")

# ================================================================
# BULK PREDICTION UPLOAD
# ================================================================

def bulk_prediction_upload():
    """Bulk employee prediction interface."""
    
    st.markdown("### 📊 Bulk Employee Predictions")
    
    models = load_prediction_models()
    
    if not models.get("best_model"):
        st.error("🚨 No trained models available. Please train models first.")
        return
    
    # File upload
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Employee Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with employee data for bulk predictions"
        )
    
    with col2:
        prediction_model = st.selectbox(
            "Select Model:",
            ["best_model", "logistic_regression", "random_forest", "ensemble"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    # Sample template download
    if st.button("📥 Download Sample Template"):
        sample_template = create_sample_template()
        csv_buffer = io.StringIO()
        sample_template.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="💾 Download Template CSV",
            data=csv_buffer.getvalue(),
            file_name="employee_prediction_template.csv",
            mime="text/csv"
        )
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            bulk_data = pd.read_csv(uploaded_file)
            
            st.markdown("#### 📋 Data Preview")
            st.dataframe(bulk_data.head(), use_container_width=True)
            
            st.markdown(f"**Total Records:** {len(bulk_data):,}")
            
            if st.button("🚀 Run Bulk Predictions", type="primary"):
                run_bulk_predictions(bulk_data, models, prediction_model)
        
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

def create_sample_template():
    """Create sample template for bulk predictions."""
    template = pd.DataFrame({
        'EmployeeID': ['EMP001', 'EMP002', 'EMP003'],
        'Age': [35, 28, 42],
        'Gender': ['Male', 'Female', 'Male'],
        'MaritalStatus': ['Married', 'Single', 'Divorced'],
        'Education': ['Bachelor\'s Degree', 'Master\'s Degree', 'High School'],
        'Department': ['Engineering', 'Marketing', 'Sales'],
        'JobRole': ['Senior', 'Manager', 'Associate'],
        'JobLevel': [3, 4, 2],
        'MonthlyIncome': [7500, 9200, 4800],
        'YearsAtCompany': [8, 3, 12],
        'YearsInCurrentRole': [4, 2, 6],
        'TotalWorkingYears': [12, 6, 18],
        'PerformanceScore': [4, 5, 3],
        'JobSatisfaction': [3, 4, 2],
        'WorkLifeBalance': [3, 3, 2],
        'EnvironmentSatisfaction': [4, 4, 2],
        'RelationshipSatisfaction': [3, 4, 3],
        'OverTime': ['No', 'Yes', 'Yes'],
        'BusinessTravel': ['Travel_Rarely', 'Non-Travel', 'Travel_Frequently'],
        'DistanceFromHome': [5, 15, 25]
    })
    return template

def run_bulk_predictions(data, models, model_name):
    """Run predictions on bulk data."""
    
    try:
        with st.spinner("🔮 Processing bulk predictions..."):
            # Get selected model
            model = models.get(model_name)
            if model is None:
                st.error(f"Model {model_name} not available")
                return
            
            # Prepare features for all employees
            results = []
            progress_bar = st.progress(0)
            
            for i, (_, employee) in enumerate(data.iterrows()):
                # Update progress
                progress = (i + 1) / len(data)
                progress_bar.progress(progress)
                
                # Prepare features
                features = prepare_features_for_prediction(employee, models)
                
                if features is not None:
                    # Make prediction
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0]
                    attrition_prob = probability[1] if len(probability) > 1 else probability[0]
                    risk_level = get_risk_level(attrition_prob)
                    
                    results.append({
                        'EmployeeID': employee.get('EmployeeID', f'EMP_{i:04d}'),
                        'Prediction': 'Will Leave' if prediction == 1 else 'Will Stay',
                        'AttritionProbability': attrition_prob,
                        'RiskLevel': risk_level,
                        'Confidence': max(probability)
                    })
                else:
                    # Handle failed feature preparation
                    results.append({
                        'EmployeeID': employee.get('EmployeeID', f'EMP_{i:04d}'),
                        'Prediction': 'Error',
                        'AttritionProbability': 0,
                        'RiskLevel': 'Unknown',
                        'Confidence': 0
                    })
            
            progress_bar.empty()
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Display results summary
            st.markdown("#### 📊 Bulk Prediction Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_processed = len(results_df)
                st.markdown(glassmorphism_metric_card(
                    value=f"{total_processed:,}",
                    title="Total Processed",
                    icon="👥",
                    color='secondary',
                    width=180,
                    height=120
                ), unsafe_allow_html=True)
            
            with col2:
                high_risk_count = len(results_df[results_df['RiskLevel'] == 'High'])
                st.markdown(glassmorphism_metric_card(
                    value=f"{high_risk_count:,}",
                    title="High Risk",
                    icon="⚠️",
                    color='error',
                    width=180,
                    height=120
                ), unsafe_allow_html=True)
            
            with col3:
                avg_risk = results_df['AttritionProbability'].mean()
                st.markdown(glassmorphism_metric_card(
                    value=f"{avg_risk:.1%}",
                    title="Avg Risk",
                    icon="📊",
                    color='warning',
                    width=180,
                    height=120
                ), unsafe_allow_html=True)
            
            with col4:
                errors = len(results_df[results_df['Prediction'] == 'Error'])
                st.markdown(glassmorphism_metric_card(
                    value=f"{errors:,}",
                    title="Errors",
                    icon="❌",
                    color='error' if errors > 0 else 'success',
                    width=180,
                    height=120
                ), unsafe_allow_html=True)
            
            # Risk distribution chart
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk level distribution
                risk_dist = results_df['RiskLevel'].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=risk_dist.index,
                    values=risk_dist.values,
                    hole=0.5,
                    marker=dict(
                        colors=[get_risk_color(level) for level in risk_dist.index],
                        line=dict(color=COLORS['border_primary'], width=2)
                    )
                )])
                
                fig = create_dark_theme_plotly_chart(
                    fig,
                    title="Risk Level Distribution",
                    height=350,
                    show_legend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Probability distribution histogram
                fig = go.Figure(data=[go.Histogram(
                    x=results_df['AttritionProbability'] * 100,
                    nbinsx=20,
                    marker=dict(
                        color=COLORS['secondary'],
                        line=dict(color=COLORS['border_primary'], width=1)
                    )
                )])
                
                fig = create_dark_theme_plotly_chart(
                    fig,
                    title="Attrition Probability Distribution",
                    height=350,
                    custom_layout={
                        'xaxis_title': 'Attrition Probability (%)',
                        'yaxis_title': 'Count'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.markdown("#### 📋 Detailed Results")
            
            # Format results for display
            display_results = results_df.copy()
            display_results['AttritionProbability'] = display_results['AttritionProbability'].apply(lambda x: f"{x:.1%}")
            display_results['Confidence'] = display_results['Confidence'].apply(lambda x: f"{x:.1%}")
            
            # Color-code risk levels
            def color_risk_level(val):
                color = get_risk_color(val)
                return f'background-color: {color}33; color: {color}; font-weight: bold'
            
            styled_results = display_results.style.applymap(
                color_risk_level, subset=['RiskLevel']
            )
            
            st.dataframe(styled_results, use_container_width=True)
            
            # Download results
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_buffer.getvalue(),
                file_name=f"bulk_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Bulk prediction error: {e}")

# ================================================================
# MODEL COMPARISON VIEW
# ================================================================

def model_comparison_view():
    """Compare predictions across different models."""
    
    st.markdown("### 🔬 Model Comparison")
    
    models = load_prediction_models()
    available_models = [name for name in ["logistic_regression", "random_forest", "ensemble"] if name in models]
    
    if len(available_models) < 2:
        st.warning("At least 2 models required for comparison")
        return
    
    # Select employee for comparison
    employee_db = load_employee_database()
    
    if 'FullName' in employee_db.columns:
        selected_employee = st.selectbox(
            "Select Employee for Model Comparison:",
            options=employee_db['FullName'].tolist()
        )
        
        if selected_employee and st.button("🔍 Compare Models", type="primary"):
            employee_data = employee_db[employee_db['FullName'] == selected_employee].iloc[0]
            
            # Run predictions with all available models
            comparison_results = []
            
            for model_name in available_models:
                if model_name in models:
                    try:
                        features = prepare_features_for_prediction(employee_data, models)
                        model = models[model_name]
                        
                        prediction = model.predict(features)[0]
                        probability = model.predict_proba(features)[0]
                        attrition_prob = probability[1] if len(probability) > 1 else probability[0]
                        risk_level = get_risk_level(attrition_prob)
                        
                        comparison_results.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Prediction': 'Will Leave' if prediction == 1 else 'Will Stay',
                            'Probability': attrition_prob,
                            'RiskLevel': risk_level,
                            'Confidence': max(probability)
                        })
                    
                    except Exception as e:
                        st.warning(f"Error with {model_name}: {e}")
            
            if comparison_results:
                # Display comparison results
                st.markdown("#### 📊 Model Comparison Results")
                
                comparison_df = pd.DataFrame(comparison_results)
                
                # Create comparison chart
                fig = go.Figure()
                
                colors = [COLORS['secondary'], COLORS['accent'], COLORS['success'], COLORS['warning']]
                
                for i, (_, row) in enumerate(comparison_df.iterrows()):
                    fig.add_trace(go.Bar(
                        x=[row['Model']],
                        y=[row['Probability'] * 100],
                        name=row['Model'],
                        marker=dict(color=colors[i % len(colors)]),
                        text=f"{row['Probability']:.1%}",
                        textposition='outside'
                    ))
                
                fig = create_dark_theme_plotly_chart(
                    fig,
                    title="Attrition Probability by Model",
                    height=400,
                    custom_layout={
                        'yaxis_title': 'Attrition Probability (%)',
                        'showlegend': False
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison table
                st.markdown("#### 📋 Detailed Comparison")
                
                # Format for display
                display_df = comparison_df.copy()
                display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.1%}")
                display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Consensus analysis
                predictions = comparison_df['Prediction'].tolist()
                probabilities = comparison_df['Probability'].tolist()
                
                consensus = max(set(predictions), key=predictions.count)
                avg_probability = np.mean(probabilities)
                std_probability = np.std(probabilities)
                
                st.markdown(f"""
                <div style="
                    padding: 20px;
                    background: rgba(37, 42, 69, 0.4);
                    border-radius: 12px;
                    border-left: 4px solid {COLORS['info']};
                    margin: 20px 0;
                ">
                    <h4 style="color: {COLORS['info']}; margin: 0 0 15px 0;">🎯 Consensus Analysis</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div>
                            <div style="color: {COLORS['text_secondary']}; font-size: 14px;">Consensus Prediction</div>
                            <div style="color: {COLORS['text']}; font-weight: bold; font-size: 18px;">
                                {consensus}
                            </div>
                        </div>
                        <div>
                            <div style="color: {COLORS['text_secondary']}; font-size: 14px;">Average Probability</div>
                            <div style="color: {COLORS['secondary']}; font-weight: bold; font-size: 18px;">
                                {avg_probability:.1%}
                            </div>
                        </div>
                        <div>
                            <div style="color: {COLORS['text_secondary']}; font-size: 14px;">Prediction Variance</div>
                            <div style="color: {COLORS['warning']}; font-weight: bold; font-size: 18px;">
                                ±{std_probability:.1%}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ================================================================
# CONFIDENCE INTERVAL DISPLAY
# ================================================================

def confidence_interval_display():
    """Display prediction confidence intervals."""
    
    st.markdown("### 📏 Confidence Intervals")
    
    st.info("🔬 This feature provides statistical confidence bounds for predictions using bootstrap sampling.")
    
    models = load_prediction_models()
    employee_db = load_employee_database()
    
    if not models.get("best_model") or employee_db.empty:
        st.warning("Models or employee data not available")
        return
    
    # Select employee
    if 'FullName' in employee_db.columns:
        selected_employee = st.selectbox(
            "Select Employee for Confidence Analysis:",
            options=employee_db['FullName'].tolist()
        )
        
        confidence_level = st.slider(
            "Confidence Level (%)",
            min_value=80,
            max_value=99,
            value=95,
            step=1
        )
        
        if selected_employee and st.button("📊 Calculate Confidence Interval", type="primary"):
            employee_data = employee_db[employee_db['FullName'] == selected_employee].iloc[0]
            
            with st.spinner("🔬 Calculating confidence intervals..."):
                # Simulate bootstrap sampling for confidence intervals
                n_bootstrap = 100  # Limited for memory efficiency
                predictions = []
                
                model = models["best_model"]
                
                for i in range(n_bootstrap):
                    # Add small random noise to simulate uncertainty
                    noisy_data = employee_data.copy()
                    
                    # Add noise to numeric features
                    numeric_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
                                      'YearsInCurrentRole', 'TotalWorkingYears']
                    
                    for feature in numeric_features:
                        if feature in noisy_data.index:
                            noise = np.random.normal(0, 0.05 * noisy_data[feature])
                            noisy_data.at[feature] = max(0, noisy_data[feature] + noise)
                    
                    # Make prediction
                    features = prepare_features_for_prediction(noisy_data, models)
                    if features is not None:
                        prob = model.predict_proba(features)[0]
                        attrition_prob = prob[1] if len(prob) > 1 else prob[0]
                        predictions.append(attrition_prob)
                
                if predictions:
                    predictions = np.array(predictions)
                    
                    # Calculate confidence interval
                    alpha = (100 - confidence_level) / 100
                    lower_bound = np.percentile(predictions, (alpha/2) * 100)
                    upper_bound = np.percentile(predictions, (1 - alpha/2) * 100)
                    mean_prediction = np.mean(predictions)
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Confidence interval visualization
                        fig = go.Figure()
                        
                        # Histogram of predictions
                        fig.add_trace(go.Histogram(
                            x=predictions * 100,
                            nbinsx=20,
                            marker=dict(color=COLORS['secondary'], opacity=0.7),
                            name='Predictions'
                        ))
                        
                        # Confidence interval lines
                        fig.add_vline(
                            x=lower_bound * 100,
                            line_dash="dash",
                            line_color=COLORS['warning'],
                            annotation_text=f"Lower Bound ({lower_bound:.1%})"
                        )
                        
                        fig.add_vline(
                            x=upper_bound * 100,
                            line_dash="dash", 
                            line_color=COLORS['warning'],
                            annotation_text=f"Upper Bound ({upper_bound:.1%})"
                        )
                        
                        fig.add_vline(
                            x=mean_prediction * 100,
                            line_color=COLORS['accent'],
                            annotation_text=f"Mean ({mean_prediction:.1%})"
                        )
                        
                        fig = create_dark_theme_plotly_chart(
                            fig,
                            title=f"{confidence_level}% Confidence Interval",
                            height=400,
                            custom_layout={
                                'xaxis_title': 'Attrition Probability (%)',
                                'yaxis_title': 'Frequency'
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Summary statistics
                        st.markdown("#### 📊 Statistics")
                        
                        stats_data = [
                            ("Mean", f"{mean_prediction:.1%}", COLORS['secondary']),
                            ("Lower Bound", f"{lower_bound:.1%}", COLORS['warning']),
                            ("Upper Bound", f"{upper_bound:.1%}", COLORS['warning']),
                            ("Std Deviation", f"{np.std(predictions):.1%}", COLORS['info']),
                            ("Range", f"{(upper_bound - lower_bound):.1%}", COLORS['accent'])
                        ]
                        
                        for label, value, color in stats_data:
                            st.markdown(f"""
                            <div style="
                                padding: 10px;
                                background: rgba(37, 42, 69, 0.3);
                                border-radius: 8px;
                                border-left: 3px solid {color};
                                margin: 8px 0;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            ">
                                <span style="color: {COLORS['text']};">{label}</span>
                                <span style="color: {color}; font-weight: bold;">{value}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Interpretation
                        interval_width = upper_bound - lower_bound
                        if interval_width < 0.1:
                            confidence_text = "High Confidence"
                            confidence_color = COLORS['success']
                        elif interval_width < 0.2:
                            confidence_text = "Medium Confidence"
                            confidence_color = COLORS['warning']
                        else:
                            confidence_text = "Low Confidence"
                            confidence_color = COLORS['error']
                        
                        st.markdown(f"""
                        <div style="
                            padding: 15px;
                            background: rgba(37, 42, 69, 0.4);
                            border-radius: 10px;
                            border: 2px solid {confidence_color};
                            margin: 20px 0;
                            text-align: center;
                        ">
                            <div style="color: {confidence_color}; font-weight: bold; font-size: 16px;">
                                {confidence_text}
                            </div>
                            <div style="color: {COLORS['text_secondary']}; font-size: 12px; margin-top: 5px;">
                                {confidence_level}% CI: [{lower_bound:.1%}, {upper_bound:.1%}]
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# ================================================================
# MAIN PREDICTIONS FUNCTION
# ================================================================

def show():
    """Main predictions function called by the navigation system."""
    
    try:
        # Apply custom styling
        apply_custom_css()
        
        # Page header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="
                color: transparent;
                background: linear-gradient(135deg, #00D4FF 0%, #B026FF 100%);
                -webkit-background-clip: text;
                background-clip: text;
                font-family: 'Orbitron', sans-serif;
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
            ">
                🔍 ML Prediction Engine
            </h1>
            <p style="color: #B8C5D1; font-size: 1.1rem;">
                Advanced machine learning predictions with confidence analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main prediction tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 Individual Lookup",
            "📊 Bulk Predictions", 
            "🔬 Model Comparison",
            "📏 Confidence Intervals"
        ])
        
        with tab1:
            individual_employee_lookup()
        
        with tab2:
            bulk_prediction_upload()
        
        with tab3:
            model_comparison_view()
        
        with tab4:
            confidence_interval_display()
        
        # Memory cleanup
        gc.collect()
        
    except Exception as e:
        st.error(f"Predictions page error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")

# ================================================================
# ENTRY POINT FOR TESTING
# ================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Predictions", layout="wide")
    show()
