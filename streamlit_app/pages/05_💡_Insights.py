"""
HR Attrition Predictor - AI Insights & Explainability Page
==========================================================
Comprehensive SHAP-powered insights with individual explanations, retention strategies,
what-if scenarios, and success stories. Memory-optimized for 4GB RAM systems.

Author: HR Analytics Team
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import warnings
import gc
import json
import io
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import base64

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
from streamlit_app.assets.theme import COLORS, TYPOGRAPHY, apply_custom_css
from streamlit_app.components.charts import (
    glassmorphism_metric_card, create_dark_theme_plotly_chart,
    futuristic_gauge_chart
)
from streamlit_app.config import get_risk_level, get_risk_color

# Import explainability engine
try:
    from src.ml_pipeline.explainability import (
        SHAPExplainer, create_explainer_for_model, quick_explanation,
        IndividualExplanation, FeatureImportance, SHAP_AVAILABLE
    )
    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False
    st.error("🚨 SHAP explainability module not available. Please install SHAP.")

# Import visualization utilities
try:
    from src.utils.visualizations import create_risk_distribution_chart, get_color
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# ================================================================
# DATA LOADING AND CACHING
# ================================================================

@st.cache_resource
def load_model_and_explainer():
    """Load trained model and create SHAP explainer with caching."""
    
    if not EXPLAINER_AVAILABLE:
        return None, None, "SHAP not available"
    
    try:
        import joblib
        
        # Load best model
        model_path = "models/best_model.pkl"
        if not Path(model_path).exists():
            return None, None, "Model file not found"
        
        model = joblib.load(model_path)
        
        # Load training data for explainer background
        data_path = "data/synthetic/hr_employees.csv"
        if Path(data_path).exists():
            # Load subset for memory efficiency
            train_data = pd.read_csv(data_path).sample(200, random_state=42)
            
            # Prepare features (simplified version)
            feature_cols = [
                'Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction',
                'WorkLifeBalance', 'EnvironmentSatisfaction', 'PerformanceScore',
                'YearsInCurrentRole', 'TotalWorkingYears', 'DistanceFromHome'
            ]
            
            # Use only available numeric columns
            available_features = [col for col in feature_cols if col in train_data.columns]
            X_train = train_data[available_features].fillna(0)
            
            # Create explainer
            explainer = SHAPExplainer(model, X_train, max_background_size=50)
            
            return model, explainer, "Success"
        
        else:
            return model, None, "Training data not found"
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"

@st.cache_data
def load_employee_insights_data():
    """Load employee data for insights analysis."""
    
    try:
        data_path = "data/synthetic/hr_employees.csv"
        
        if Path(data_path).exists():
            df = pd.read_csv(data_path)
            
            # Add risk scores if not present (simulate)
            if 'AttritionProbability' not in df.columns:
                np.random.seed(42)
                df['AttritionProbability'] = np.random.uniform(0, 1, len(df))
            
            df['RiskLevel'] = df['AttritionProbability'].apply(
                lambda x: get_risk_level(x)
            )
            
            return df
        else:
            # Generate demo data
            return generate_demo_insights_data()
    
    except Exception as e:
        st.error(f"Error loading insights data: {e}")
        return generate_demo_insights_data()

def generate_demo_insights_data():
    """Generate demo data for insights."""
    
    np.random.seed(42)
    n_employees = 300
    
    demo_data = pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n_employees + 1)],
        'FirstName': [f'Employee{i}' for i in range(1, n_employees + 1)],
        'LastName': [f'Last{i}' for i in range(1, n_employees + 1)],
        'Age': np.random.randint(22, 65, n_employees),
        'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'Operations', 'Finance'], n_employees),
        'JobRole': np.random.choice(['Manager', 'Senior', 'Junior', 'Lead', 'Associate'], n_employees),
        'YearsAtCompany': np.random.randint(0, 20, n_employees),
        'MonthlyIncome': np.random.randint(3000, 15000, n_employees),
        'JobSatisfaction': np.random.randint(1, 5, n_employees),
        'WorkLifeBalance': np.random.randint(1, 5, n_employees),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n_employees),
        'PerformanceScore': np.random.randint(1, 6, n_employees),
        'YearsInCurrentRole': np.random.randint(0, 10, n_employees),
        'TotalWorkingYears': np.random.randint(0, 25, n_employees),
        'DistanceFromHome': np.random.randint(1, 30, n_employees),
        'Attrition': np.random.choice(['Yes', 'No'], n_employees, p=[0.16, 0.84]),
        'AttritionProbability': np.random.uniform(0, 1, n_employees)
    })
    
    demo_data['FullName'] = demo_data['FirstName'] + ' ' + demo_data['LastName']
    demo_data['RiskLevel'] = demo_data['AttritionProbability'].apply(get_risk_level)
    
    return demo_data

# ================================================================
# INDIVIDUAL SHAP EXPLANATIONS
# ================================================================

def individual_shap_explanations():
    """Generate and display individual SHAP explanations."""
    
    st.markdown("### 🧠 Individual Employee AI Explanations")
    
    # Load model and explainer
    model, explainer, status = load_model_and_explainer()
    employee_data = load_employee_insights_data()
    
    if not EXPLAINER_AVAILABLE or explainer is None:
        st.warning("🔧 SHAP explainer not available. Using simulated explanations.")
        _show_simulated_explanations(employee_data)
        return
    
    # Employee selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'FullName' in employee_data.columns:
            selected_employee = st.selectbox(
                "🔍 Select Employee for AI Analysis:",
                options=employee_data['FullName'].tolist(),
                help="Choose an employee to see detailed AI explanations"
            )
        else:
            st.error("Employee data not properly formatted")
            return
    
    with col2:
        analysis_depth = st.selectbox(
            "Analysis Depth:",
            ["Quick Overview", "Detailed Analysis", "Deep Dive"],
            help="Choose the level of detail for explanations"
        )
    
    if selected_employee and st.button("🚀 Generate AI Explanation", type="primary"):
        
        with st.spinner("🧠 AI is analyzing employee data..."):
            # Get employee data
            employee_row = employee_data[employee_data['FullName'] == selected_employee].iloc[0]
            
            try:
                # Generate SHAP explanation
                explanation = _generate_shap_explanation(employee_row, model, explainer)
                
                if explanation:
                    _display_individual_explanation(explanation, employee_row, analysis_depth)
                else:
                    st.error("Failed to generate explanation")
            
            except Exception as e:
                st.error(f"Explanation error: {e}")
                _show_simulated_explanation_for_employee(employee_row)

def _generate_shap_explanation(employee_row, model, explainer):
    """Generate SHAP explanation for individual employee."""
    
    try:
        # Prepare feature data
        feature_cols = [
            'Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction',
            'WorkLifeBalance', 'EnvironmentSatisfaction', 'PerformanceScore',
            'YearsInCurrentRole', 'TotalWorkingYears', 'DistanceFromHome'
        ]
        
        # Create feature vector
        features = {}
        for col in feature_cols:
            if col in employee_row.index:
                features[col] = employee_row[col]
            else:
                features[col] = 0  # Default value
        
        feature_df = pd.DataFrame([features])
        
        # Generate explanation
        explanations = explainer.generate_individual_explanations(
            feature_df, 
            [employee_row.get('EmployeeID', 'SAMPLE')],
            max_samples=1
        )
        
        return explanations[0] if explanations else None
    
    except Exception as e:
        st.error(f"SHAP generation error: {e}")
        return None

def _display_individual_explanation(explanation: IndividualExplanation, 
                                  employee_row: pd.Series, 
                                  analysis_depth: str):
    """Display comprehensive individual explanation."""
    
    # Main prediction display
    st.markdown("### 🎯 AI Prediction Results")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Risk gauge
        gauge_fig = futuristic_gauge_chart(
            value=explanation.prediction * 100,
            title="Attrition Risk Assessment",
            min_value=0,
            max_value=100,
            unit="%",
            height=300
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        # Risk level card
        risk_level = get_risk_level(explanation.prediction)
        risk_color = get_risk_color(risk_level)
        
        st.markdown(glassmorphism_metric_card(
            value=risk_level,
            title="Risk Category",
            icon="⚠️" if risk_level == "High" else "🟡" if risk_level == "Medium" else "✅",
            color='error' if risk_level == "High" else 'warning' if risk_level == "Medium" else 'success',
            width=200,
            height=150
        ), unsafe_allow_html=True)
    
    with col3:
        # Confidence card
        st.markdown(glassmorphism_metric_card(
            value=explanation.confidence_level,
            title="AI Confidence",
            subtitle=f"{explanation.prediction:.1%}",
            icon="🎯",
            color='info',
            width=200,
            height=150
        ), unsafe_allow_html=True)
    
    # Natural language explanation
    st.markdown("### 💬 AI Explanation")
    
    st.markdown(f"""
    <div style="
        padding: 25px;
        background: rgba(37, 42, 69, 0.4);
        border-radius: 15px;
        border-left: 5px solid {risk_color};
        margin: 20px 0;
    ">
        <h4 style="color: {risk_color}; margin: 0 0 15px 0;">
            🤖 AI Analysis for {employee_row.get('FullName', 'Employee')}
        </h4>
        <div style="color: {COLORS['text']}; line-height: 1.6; font-size: 14px;">
            {explanation.natural_language_explanation}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if analysis_depth in ["Detailed Analysis", "Deep Dive"]:
        _show_detailed_feature_analysis(explanation, analysis_depth)
    
    if analysis_depth == "Deep Dive":
        _show_deep_dive_analysis(explanation, employee_row)

def _show_detailed_feature_analysis(explanation: IndividualExplanation, depth: str):
    """Show detailed feature-by-feature analysis."""
    
    st.markdown("### 📊 Detailed Factor Analysis")
    
    # Feature importance chart
    top_features = explanation.feature_contributions[:10]
    
    if top_features:
        # Create feature importance chart
        fig = go.Figure()
        
        feature_names = [fc.feature_name for fc in top_features]
        importance_values = [fc.importance_score * (1 if fc.impact_direction == 'positive' else -1) 
                           for fc in top_features]
        colors = [COLORS['error'] if val > 0 else COLORS['success'] for val in importance_values]
        
        fig.add_trace(go.Bar(
            y=feature_names,
            x=importance_values,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{abs(val):.3f}" for val in importance_values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<br><extra></extra>'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Feature Impact on Attrition Risk",
            height=400,
            custom_layout={
                'xaxis_title': 'Impact Score (Positive = Increases Risk)',
                'yaxis_title': 'Features'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature details table
        st.markdown("#### 📋 Feature Impact Details")
        
        for i, fc in enumerate(top_features[:5], 1):
            impact_color = COLORS['error'] if fc.impact_direction == 'positive' else COLORS['success']
            impact_icon = "📈" if fc.impact_direction == 'positive' else "📉"
            
            st.markdown(f"""
            <div style="
                padding: 15px;
                background: rgba(37, 42, 69, 0.3);
                border-radius: 8px;
                border-left: 3px solid {impact_color};
                margin: 10px 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 1;">
                        <h5 style="color: {COLORS['text']}; margin: 0 0 8px 0;">
                            {impact_icon} {fc.feature_name.replace('_', ' ').title()}
                        </h5>
                        <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 13px;">
                            {fc.description}
                        </p>
                    </div>
                    <div style="text-align: right; margin-left: 20px;">
                        <div style="color: {impact_color}; font-weight: bold; font-size: 16px;">
                            Rank #{fc.rank}
                        </div>
                        <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                            Impact: {fc.importance_score:.3f}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def _show_deep_dive_analysis(explanation: IndividualExplanation, employee_row: pd.Series):
    """Show deep dive analysis with actionable insights."""
    
    st.markdown("### 🔬 Deep Dive Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Risk Factors")
        
        if explanation.risk_factors:
            for i, factor in enumerate(explanation.risk_factors[:5], 1):
                st.markdown(f"""
                <div style="
                    padding: 12px;
                    background: rgba(255, 45, 117, 0.1);
                    border-radius: 8px;
                    border-left: 3px solid {COLORS['error']};
                    margin: 8px 0;
                    display: flex;
                    align-items: center;
                ">
                    <div style="
                        background: {COLORS['error']};
                        color: white;
                        width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        font-size: 12px;
                        margin-right: 12px;
                        flex-shrink: 0;
                    ">
                        {i}
                    </div>
                    <div style="color: {COLORS['text']}; font-size: 14px;">
                        {factor.replace('_', ' ').title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant risk factors identified")
    
    with col2:
        st.markdown("#### ✅ Protective Factors")
        
        if explanation.protective_factors:
            for i, factor in enumerate(explanation.protective_factors[:5], 1):
                st.markdown(f"""
                <div style="
                    padding: 12px;
                    background: rgba(0, 255, 136, 0.1);
                    border-radius: 8px;
                    border-left: 3px solid {COLORS['success']};
                    margin: 8px 0;
                    display: flex;
                    align-items: center;
                ">
                    <div style="
                        background: {COLORS['success']};
                        color: white;
                        width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        font-size: 12px;
                        margin-right: 12px;
                        flex-shrink: 0;
                    ">
                        {i}
                    </div>
                    <div style="color: {COLORS['text']}; font-size: 14px;">
                        {factor.replace('_', ' ').title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong protective factors identified")
    
    # Scenario analysis
    st.markdown("#### 🎭 What-If Scenario Impact")
    
    scenarios = [
        {"factor": "Job Satisfaction", "change": "+1 point", "impact": -0.12},
        {"factor": "Monthly Income", "change": "+$1000", "impact": -0.08},
        {"factor": "Work Life Balance", "change": "+1 point", "impact": -0.15},
        {"factor": "Years at Company", "change": "+1 year", "impact": -0.05}
    ]
    
    scenario_df = pd.DataFrame(scenarios)
    
    fig = go.Figure()
    
    colors = [COLORS['success'] if impact < 0 else COLORS['error'] for impact in scenario_df['impact']]
    
    fig.add_trace(go.Bar(
        x=scenario_df['factor'],
        y=scenario_df['impact'] * 100,  # Convert to percentage
        marker=dict(color=colors),
        text=[f"{impact*100:+.1f}%" for impact in scenario_df['impact']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Risk Change: %{y:.1f}%<br><extra></extra>'
    ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title="Estimated Risk Change from Interventions",
        height=300,
        custom_layout={
            'yaxis_title': 'Risk Change (%)',
            'xaxis_title': 'Intervention'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _show_simulated_explanations(employee_data: pd.DataFrame):
    """Show simulated explanations when SHAP is not available."""
    
    st.info("🔧 Using simulated AI explanations (SHAP not available)")
    
    if 'FullName' in employee_data.columns:
        selected_employee = st.selectbox(
            "🔍 Select Employee:",
            options=employee_data['FullName'].tolist()
        )
        
        if selected_employee and st.button("🚀 Generate Simulated Explanation"):
            employee_row = employee_data[employee_data['FullName'] == selected_employee].iloc[0]
            _show_simulated_explanation_for_employee(employee_row)

def _show_simulated_explanation_for_employee(employee_row: pd.Series):
    """Show simulated explanation for a specific employee."""
    
    # Simulate prediction and explanation
    risk_prob = employee_row.get('AttritionProbability', np.random.random())
    risk_level = get_risk_level(risk_prob)
    risk_color = get_risk_color(risk_level)
    
    # Main display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        gauge_fig = futuristic_gauge_chart(
            value=risk_prob * 100,
            title="Simulated Risk Assessment",
            height=300
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        st.markdown(glassmorphism_metric_card(
            value=risk_level,
            title="Risk Level",
            icon="⚠️" if risk_level == "High" else "🟡" if risk_level == "Medium" else "✅",
            color='error' if risk_level == "High" else 'warning' if risk_level == "Medium" else 'success'
        ), unsafe_allow_html=True)
    
    # Simulated explanation
    explanation_text = f"""
    **Employee {employee_row.get('FullName', 'Unknown')}** has a **{risk_level.lower()} risk** of attrition 
    (probability: {risk_prob:.1%}).
    
    **Key Factors (Simulated):**
    - Age: {employee_row.get('Age', 'Unknown')} years
    - Years at Company: {employee_row.get('YearsAtCompany', 'Unknown')} years  
    - Job Satisfaction: {employee_row.get('JobSatisfaction', 'Unknown')}/5
    - Monthly Income: ${employee_row.get('MonthlyIncome', 0):,}
    
    **Note:** This is a simulated explanation. Install SHAP for real AI insights.
    """
    
    st.markdown(f"""
    <div style="
        padding: 20px;
        background: rgba(37, 42, 69, 0.4);
        border-radius: 12px;
        border-left: 4px solid {risk_color};
        margin: 20px 0;
    ">
        <div style="color: {COLORS['text']}; line-height: 1.6;">
            {explanation_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# RETENTION STRATEGY RECOMMENDATIONS
# ================================================================

def retention_strategy_recommendations():
    """Generate personalized retention strategies."""
    
    st.markdown("### 💼 AI-Powered Retention Strategies")
    
    employee_data = load_employee_insights_data()
    
    # Strategy selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        strategy_type = st.selectbox(
            "Strategy Focus:",
            [
                "High-Risk Employee Interventions",
                "Department-Specific Strategies", 
                "Performance-Based Retention",
                "Satisfaction Improvement Plans",
                "Career Development Programs"
            ]
        )
    
    with col2:
        urgency_level = st.selectbox(
            "Urgency Level:",
            ["Immediate (0-30 days)", "Short-term (1-3 months)", "Long-term (3-12 months)"]
        )
    
    if st.button("🚀 Generate Retention Strategies", type="primary"):
        _generate_retention_strategies(employee_data, strategy_type, urgency_level)

def _generate_retention_strategies(data: pd.DataFrame, strategy_type: str, urgency: str):
    """Generate specific retention strategies based on data analysis."""
    
    st.markdown("### 📋 Recommended Retention Strategies")
    
    # Analyze data for insights
    high_risk_employees = data[data['RiskLevel'] == 'High']
    avg_satisfaction = data['JobSatisfaction'].mean() if 'JobSatisfaction' in data.columns else 3
    
    strategies = []
    
    if strategy_type == "High-Risk Employee Interventions":
        strategies = [
            {
                'title': 'Emergency One-on-One Sessions',
                'description': f'Schedule immediate meetings with {len(high_risk_employees)} high-risk employees',
                'priority': 'Critical',
                'timeline': '1-2 weeks',
                'expected_impact': '25-40% risk reduction',
                'cost': 'Low',
                'resources': ['Manager time', 'HR support'],
                'success_metrics': ['Risk score improvement', 'Engagement survey results']
            },
            {
                'title': 'Personalized Retention Packages',
                'description': 'Custom benefits and incentives for high-risk employees',
                'priority': 'High',
                'timeline': '2-4 weeks', 
                'expected_impact': '30-50% retention improvement',
                'cost': 'Medium',
                'resources': ['HR budget', 'Benefits team'],
                'success_metrics': ['Retention rate', 'Employee satisfaction']
            },
            {
                'title': 'Rapid Career Development Track',
                'description': 'Accelerated promotion and skill development opportunities',
                'priority': 'High',
                'timeline': '1-3 months',
                'expected_impact': '20-35% risk reduction',
                'cost': 'Medium',
                'resources': ['Training budget', 'Mentoring program'],
                'success_metrics': ['Career progression', 'Skill assessments']
            }
        ]
    
    elif strategy_type == "Department-Specific Strategies":
        # Analyze department-wise attrition
        dept_analysis = data.groupby('Department')['RiskLevel'].apply(
            lambda x: (x == 'High').mean()
        ).sort_values(ascending=False)
        
        worst_dept = dept_analysis.index[0] if len(dept_analysis) > 0 else "Engineering"
        worst_rate = dept_analysis.iloc[0] if len(dept_analysis) > 0 else 0.3
        
        strategies = [
            {
                'title': f'{worst_dept} Department Intervention',
                'description': f'Targeted program for {worst_dept} with {worst_rate:.1%} high-risk rate',
                'priority': 'Critical',
                'timeline': '2-6 weeks',
                'expected_impact': '40-60% department risk reduction',
                'cost': 'High',
                'resources': ['Department head', 'External consultants'],
                'success_metrics': ['Department retention rate', 'Team satisfaction']
            },
            {
                'title': 'Cross-Department Best Practices',
                'description': 'Share successful retention practices across departments',
                'priority': 'Medium',
                'timeline': '1-2 months',
                'expected_impact': '15-25% overall improvement',
                'cost': 'Low',
                'resources': ['Internal knowledge sharing'],
                'success_metrics': ['Best practice adoption', 'Cross-department metrics']
            }
        ]
    
    elif strategy_type == "Satisfaction Improvement Plans":
        if avg_satisfaction < 3.5:
            satisfaction_gap = 4.0 - avg_satisfaction
            
            strategies = [
                {
                    'title': 'Organization-Wide Satisfaction Initiative',
                    'description': f'Address {satisfaction_gap:.1f} point satisfaction gap',
                    'priority': 'High',
                    'timeline': '3-6 months',
                    'expected_impact': '20-40% attrition reduction',
                    'cost': 'Medium',
                    'resources': ['HR team', 'Employee engagement tools'],
                    'success_metrics': ['Satisfaction scores', 'Engagement metrics']
                },
                {
                    'title': 'Manager Training Program',
                    'description': 'Improve management quality to boost satisfaction',
                    'priority': 'High',
                    'timeline': '2-4 months',
                    'expected_impact': '25-35% satisfaction improvement',
                    'cost': 'Medium',
                    'resources': ['Training budget', 'External trainers'],
                    'success_metrics': ['Manager effectiveness', 'Team satisfaction']
                }
            ]
    
    # Display strategies
    for i, strategy in enumerate(strategies, 1):
        priority_color = {
            'Critical': COLORS['error'],
            'High': COLORS['warning'],
            'Medium': COLORS['secondary'],
            'Low': COLORS['success']
        }.get(strategy['priority'], COLORS['secondary'])
        
        st.markdown(f"""
        <div style="
            padding: 25px;
            background: rgba(37, 42, 69, 0.4);
            border-radius: 15px;
            border-left: 5px solid {priority_color};
            margin: 20px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
                <h4 style="color: {COLORS['text']}; margin: 0; flex: 1;">
                    {i}. {strategy['title']}
                </h4>
                <div style="
                    background: {priority_color};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 15px;
                    font-size: 12px;
                    font-weight: bold;
                ">
                    {strategy['priority']} Priority
                </div>
            </div>
            
            <p style="color: {COLORS['text_secondary']}; margin: 0 0 20px 0; line-height: 1.6;">
                {strategy['description']}
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div>
                    <strong style="color: {COLORS['secondary']};">Timeline:</strong><br>
                    <span style="color: {COLORS['text']};">{strategy['timeline']}</span>
                </div>
                <div>
                    <strong style="color: {COLORS['success']};">Expected Impact:</strong><br>
                    <span style="color: {COLORS['text']};">{strategy['expected_impact']}</span>
                </div>
                <div>
                    <strong style="color: {COLORS['warning']};">Cost:</strong><br>
                    <span style="color: {COLORS['text']};">{strategy['cost']}</span>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <div style="margin-bottom: 10px;">
                    <strong style="color: {COLORS['accent']};">Required Resources:</strong>
                    <span style="color: {COLORS['text']}; margin-left: 10px;">
                        {', '.join(strategy['resources'])}
                    </span>
                </div>
                <div>
                    <strong style="color: {COLORS['info']};">Success Metrics:</strong>
                    <span style="color: {COLORS['text']}; margin-left: 10px;">
                        {', '.join(strategy['success_metrics'])}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Implementation timeline
    st.markdown("### 📅 Implementation Timeline")
    
    timeline_fig = go.Figure()
    
    for i, strategy in enumerate(strategies):
        timeline_fig.add_trace(go.Bar(
            name=strategy['title'],
            x=[strategy['title']],
            y=[i + 1],
            orientation='v',
            marker=dict(
                color=COLORS['secondary'],
                opacity=0.7
            ),
            text=strategy['timeline'],
            textposition='outside',
            hovertemplate=f"<b>{strategy['title']}</b><br>Timeline: {strategy['timeline']}<br><extra></extra>"
        ))
    
    timeline_fig = create_dark_theme_plotly_chart(
        timeline_fig,
        title="Strategy Implementation Timeline",
        height=300,
        custom_layout={
            'yaxis_title': 'Strategy Priority',
            'showlegend': False
        }
    )
    
    st.plotly_chart(timeline_fig, use_container_width=True)

# ================================================================
# WHAT-IF SCENARIO ANALYSIS
# ================================================================

def what_if_scenario_analysis():
    """Interactive what-if scenario analysis."""
    
    st.markdown("### 🎭 What-If Scenario Analysis")
    
    st.info("🔬 Explore how changes in key factors affect attrition risk")
    
    employee_data = load_employee_insights_data()
    
    # Scenario setup
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Select Base Employee")
        
        if 'FullName' in employee_data.columns:
            base_employee = st.selectbox(
                "Base Employee:",
                options=employee_data['FullName'].tolist(),
                help="Choose an employee as the baseline for scenario analysis"
            )
        else:
            st.error("Employee data not available")
            return
    
    with col2:
        st.markdown("#### ⚙️ Scenario Parameters")
        
        scenario_type = st.selectbox(
            "Scenario Type:",
            [
                "Salary Adjustment",
                "Satisfaction Improvement", 
                "Role Change",
                "Work-Life Balance",
                "Multi-Factor Intervention"
            ]
        )
    
    if base_employee:
        employee_row = employee_data[employee_data['FullName'] == base_employee].iloc[0]
        
        st.markdown("---")
        _run_scenario_analysis(employee_row, scenario_type)

def _run_scenario_analysis(employee_row: pd.Series, scenario_type: str):
    """Run specific scenario analysis."""
    
    st.markdown("### 📊 Scenario Results")
    
    # Current baseline
    current_risk = employee_row.get('AttritionProbability', 0.5)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📋 Current State")
        
        current_metrics = {
            'Risk Level': get_risk_level(current_risk),
            'Probability': f"{current_risk:.1%}",
            'Job Satisfaction': f"{employee_row.get('JobSatisfaction', 3)}/5",
            'Monthly Income': f"${employee_row.get('MonthlyIncome', 5000):,}",
            'Work-Life Balance': f"{employee_row.get('WorkLifeBalance', 3)}/5"
        }
        
        for metric, value in current_metrics.items():
            st.metric(metric, value)
    
    with col2:
        st.markdown("#### 🎭 Scenario Simulation")
        
        scenarios = []
        
        if scenario_type == "Salary Adjustment":
            # Salary scenarios
            salary_changes = [1000, 2000, 3000, 5000]
            current_salary = employee_row.get('MonthlyIncome', 5000)
            
            for change in salary_changes:
                new_salary = current_salary + change
                # Simulate risk reduction (simplified model)
                risk_reduction = min(0.15, change / 10000 * 0.10)  # Max 15% reduction
                new_risk = max(0.05, current_risk - risk_reduction)
                
                scenarios.append({
                    'name': f'+${change:,}',
                    'risk': new_risk,
                    'change': new_risk - current_risk,
                    'details': f'Salary: ${new_salary:,}'
                })
        
        elif scenario_type == "Satisfaction Improvement":
            # Satisfaction scenarios
            satisfaction_changes = [0.5, 1.0, 1.5, 2.0]
            current_satisfaction = employee_row.get('JobSatisfaction', 3)
            
            for change in satisfaction_changes:
                new_satisfaction = min(5, current_satisfaction + change)
                # Simulate risk reduction
                risk_reduction = change * 0.08  # 8% reduction per satisfaction point
                new_risk = max(0.05, current_risk - risk_reduction)
                
                scenarios.append({
                    'name': f'+{change} points',
                    'risk': new_risk,
                    'change': new_risk - current_risk,
                    'details': f'Satisfaction: {new_satisfaction:.1f}/5'
                })
        
        elif scenario_type == "Multi-Factor Intervention":
            # Combined interventions
            interventions = [
                {'name': 'Basic Package', 'salary': 1500, 'satisfaction': 0.5, 'wlb': 0.5},
                {'name': 'Standard Package', 'salary': 3000, 'satisfaction': 1.0, 'wlb': 1.0},
                {'name': 'Premium Package', 'salary': 5000, 'satisfaction': 1.5, 'wlb': 1.5},
                {'name': 'Executive Package', 'salary': 8000, 'satisfaction': 2.0, 'wlb': 2.0}
            ]
            
            for intervention in interventions:
                # Combined effect
                salary_effect = min(0.12, intervention['salary'] / 10000 * 0.08)
                satisfaction_effect = intervention['satisfaction'] * 0.06
                wlb_effect = intervention['wlb'] * 0.05
                
                total_reduction = salary_effect + satisfaction_effect + wlb_effect
                new_risk = max(0.05, current_risk - total_reduction)
                
                scenarios.append({
                    'name': intervention['name'],
                    'risk': new_risk,
                    'change': new_risk - current_risk,
                    'details': f"Salary +${intervention['salary']:,}, Satisfaction +{intervention['satisfaction']}, WLB +{intervention['wlb']}"
                })
        
        # Display scenario results
        if scenarios:
            # Create scenario comparison chart
            scenario_names = [s['name'] for s in scenarios]
            scenario_risks = [s['risk'] * 100 for s in scenarios]  # Convert to percentage
            scenario_changes = [s['change'] * 100 for s in scenarios]  # Convert to percentage
            
            fig = go.Figure()
            
            # Current risk line
            fig.add_hline(
                y=current_risk * 100,
                line_dash="dash",
                line_color=COLORS['warning'],
                annotation_text="Current Risk"
            )
            
            # Scenario bars
            colors = [COLORS['success'] if change < 0 else COLORS['error'] for change in scenario_changes]
            
            fig.add_trace(go.Bar(
                x=scenario_names,
                y=scenario_risks,
                marker=dict(color=colors),
                text=[f"{risk:.1f}%" for risk in scenario_risks],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Risk: %{y:.1f}%<br>Change: %{customdata:+.1f}%<br><extra></extra>',
                customdata=scenario_changes
            ))
            
            fig = create_dark_theme_plotly_chart(
                fig,
                title=f"Scenario Impact Analysis - {scenario_type}",
                height=400,
                custom_layout={
                    'yaxis_title': 'Attrition Risk (%)',
                    'xaxis_title': 'Scenario'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario details table
            st.markdown("#### 📋 Scenario Details")
            
            for i, scenario in enumerate(scenarios, 1):
                change_color = COLORS['success'] if scenario['change'] < 0 else COLORS['error']
                change_icon = "📉" if scenario['change'] < 0 else "📈"
                
                st.markdown(f"""
                <div style="
                    padding: 15px;
                    background: rgba(37, 42, 69, 0.3);
                    border-radius: 8px;
                    border-left: 3px solid {change_color};
                    margin: 10px 0;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h5 style="color: {COLORS['text']}; margin: 0 0 5px 0;">
                                {i}. {scenario['name']}
                            </div>
                            <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 12px;">
                                {scenario['details']}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: {change_color}; font-weight: bold;">
                                {change_icon} {scenario['change']:+.1%}
                            </div>
                            <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                                New Risk: {scenario['risk']:.1%}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ================================================================
# SUCCESS STORY CASE STUDIES
# ================================================================

def success_story_case_studies():
    """Display success stories and case studies."""
    
    st.markdown("### 🏆 Success Stories & Case Studies")
    
    # Case study selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        case_study_type = st.selectbox(
            "Case Study Type:",
            [
                "High-Risk Turnarounds",
                "Department Transformations",
                "Satisfaction Improvements", 
                "Retention Best Practices",
                "AI-Driven Interventions"
            ]
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period:",
            ["Last 6 months", "Last year", "All time"]
        )
    
    # Generate case studies based on selection
    _display_case_studies(case_study_type, time_period)

def _display_case_studies(case_type: str, time_period: str):
    """Display specific case studies."""
    
    # Simulated case studies (in real implementation, these would come from data)
    case_studies = []
    
    if case_type == "High-Risk Turnarounds":
        case_studies = [
            {
                'title': 'Software Engineer Retention Success',
                'employee': 'Sarah Chen - Senior Developer',
                'initial_risk': 0.85,
                'final_risk': 0.25,
                'timeframe': '3 months',
                'interventions': [
                    'Salary increase: $8,000 annually',
                    'Flexible work arrangement',
                    'Technical lead promotion path',
                    'Conference attendance budget'
                ],
                'outcomes': [
                    '60-point risk reduction',
                    'Satisfaction score: 2.5 → 4.2',
                    'Team productivity +15%',
                    'Became team mentor'
                ],
                'cost': '$12,000',
                'roi': '350% (vs $45,000 replacement cost)'
            },
            {
                'title': 'Sales Manager Recovery',
                'employee': 'Michael Rodriguez - Sales Manager',
                'initial_risk': 0.78,
                'final_risk': 0.20,
                'timeframe': '4 months',
                'interventions': [
                    'Commission structure improvement',
                    'Team autonomy increase',
                    'Executive coaching program',
                    'Work-from-home flexibility'
                ],
                'outcomes': [
                    '58-point risk reduction',
                    'Team retention improved to 95%',
                    'Sales performance +22%',
                    'Leadership score: 3.1 → 4.6'
                ],
                'cost': '$8,500',
                'roi': '425% (vs $38,000 replacement cost)'
            }
        ]
    
    elif case_type == "Department Transformations":
        case_studies = [
            {
                'title': 'Marketing Department Turnaround',
                'employee': 'Marketing Team (15 people)',
                'initial_risk': 0.42,
                'final_risk': 0.18,
                'timeframe': '6 months',
                'interventions': [
                    'New department head with modern approach',
                    'Creative freedom policy implementation',
                    'Cross-functional collaboration tools',
                    'Professional development budget increase'
                ],
                'outcomes': [
                    'Department attrition: 28% → 8%',
                    'Project completion +35%',
                    'Inter-department satisfaction +40%',
                    'Brand awareness campaigns +60%'
                ],
                'cost': '$45,000',
                'roi': '280% (vs $165,000 replacement costs)'
            }
        ]
    
    elif case_type == "AI-Driven Interventions":
        case_studies = [
            {
                'title': 'Predictive Intervention Program',
                'employee': 'Company-wide Initiative',
                'initial_risk': 0.16,
                'final_risk': 0.11,
                'timeframe': '12 months',
                'interventions': [
                    'AI risk scoring implementation',
                    'Proactive manager alerts',
                    'Personalized retention plans',
                    'Real-time satisfaction monitoring'
                ],
                'outcomes': [
                    'Overall attrition: 16% → 11%',
                    'Early intervention success: 73%',
                    'Manager response time: 2 weeks → 3 days',
                    'Employee satisfaction +18%'
                ],
                'cost': '$85,000',
                'roi': '420% (vs $380,000 in prevented turnover)'
            }
        ]
    
    # Display case studies
    for i, case in enumerate(case_studies, 1):
        initial_risk_color = COLORS['error'] if case['initial_risk'] > 0.7 else COLORS['warning']
        final_risk_color = COLORS['success'] if case['final_risk'] < 0.3 else COLORS['warning']
        improvement = case['initial_risk'] - case['final_risk']
        
        st.markdown(f"""
        <div style="
            padding: 25px;
            background: rgba(37, 42, 69, 0.4);
            border-radius: 15px;
            border-left: 5px solid {COLORS['success']};
            margin: 25px 0;
        ">
            <div style="display: flex; justify-content: between; align-items: flex-start; margin-bottom: 20px;">
                <div style="flex: 1;">
                    <h3 style="color: {COLORS['text']}; margin: 0 0 10px 0;">
                        🏆 {case['title']}
                    </h3>
                    <p style="color: {COLORS['secondary']}; margin: 0; font-size: 16px; font-weight: 500;">
                        {case['employee']}
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="color: {COLORS['success']}; font-size: 24px; font-weight: bold;">
                        -{improvement:.0%}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                        Risk Reduction
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin-bottom: 25px;">
                <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <div style="color: {initial_risk_color}; font-size: 20px; font-weight: bold;">
                        {case['initial_risk']:.0%}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                        Initial Risk
                    </div>
                </div>
                <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <div style="color: {final_risk_color}; font-size: 20px; font-weight: bold;">
                        {case['final_risk']:.0%}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                        Final Risk
                    </div>
                </div>
                <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <div style="color: {COLORS['info']}; font-size: 20px; font-weight: bold;">
                        {case['timeframe']}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                        Timeframe
                    </div>
                </div>
                <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <div style="color: {COLORS['accent']}; font-size: 20px; font-weight: bold;">
                        {case['roi']}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                        ROI
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
                <div>
                    <h4 style="color: {COLORS['warning']}; margin: 0 0 15px 0; font-size: 16px;">
                        🎯 Interventions Applied
                    </h4>
                    <ul style="color: {COLORS['text']}; margin: 0; padding-left: 20px; line-height: 1.8;">
                        {''.join([f'<li>{intervention}</li>' for intervention in case['interventions']])}
                    </ul>
                </div>
                <div>
                    <h4 style="color: {COLORS['success']}; margin: 0 0 15px 0; font-size: 16px;">
                        ✅ Outcomes Achieved
                    </h4>
                    <ul style="color: {COLORS['text']}; margin: 0; padding-left: 20px; line-height: 1.8;">
                        {''.join([f'<li>{outcome}</li>' for outcome in case['outcomes']])}
                    </ul>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid {COLORS['border_primary']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: {COLORS['text_secondary']};">Investment:</span>
                        <span style="color: {COLORS['text']}; font-weight: bold; margin-left: 8px;">{case['cost']}</span>
                    </div>
                    <div>
                        <span style="color: {COLORS['text_secondary']};">Return on Investment:</span>
                        <span style="color: {COLORS['success']}; font-weight: bold; margin-left: 8px;">{case['roi']}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Success metrics summary
    if case_studies:
        st.markdown("### 📊 Success Metrics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        avg_improvement = np.mean([case['initial_risk'] - case['final_risk'] for case in case_studies])
        avg_timeframe = "3.5 months"  # Simulated average
        total_roi = "385%"  # Simulated average ROI
        success_rate = "87%"  # Simulated success rate
        
        with col1:
            st.markdown(glassmorphism_metric_card(
                value=f"{avg_improvement:.0%}",
                title="Avg Risk Reduction",
                icon="📉",
                color='success'
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(glassmorphism_metric_card(
                value=avg_timeframe,
                title="Avg Timeframe",
                icon="⏱️",
                color='info'
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(glassmorphism_metric_card(
                value=total_roi,
                title="Average ROI",
                icon="💰",
                color='success'
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(glassmorphism_metric_card(
                value=success_rate,
                title="Success Rate",
                icon="🎯",
                color='secondary'
            ), unsafe_allow_html=True)

# ================================================================
# MAIN INSIGHTS FUNCTION
# ================================================================

def show():
    """Main insights function called by the navigation system."""
    
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
                💡 AI Insights & Explainability
            </h1>
            <p style="color: #B8C5D1; font-size: 1.1rem;">
                Advanced AI-powered insights with SHAP explanations and retention strategies
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # SHAP availability check
        if not EXPLAINER_AVAILABLE:
            st.warning("""
            🔧 **SHAP Explainability Not Available**
            
            To enable full AI insights, install SHAP:
            ```
            pip install shap
            ```
            
            Currently showing simulated insights for demonstration.
            """)
        
        # Main insights tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧠 Individual Explanations",
            "💼 Retention Strategies",
            "🎭 What-If Analysis", 
            "🏆 Success Stories"
        ])
        
        with tab1:
            individual_shap_explanations()
        
        with tab2:
            retention_strategy_recommendations()
        
        with tab3:
            what_if_scenario_analysis()
        
        with tab4:
            success_story_case_studies()
        
        # Memory cleanup
        gc.collect()
        
    except Exception as e:
        st.error(f"Insights page error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")

# ================================================================
# ENTRY POINT FOR TESTING
# ================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="AI Insights", layout="wide")
    show()
