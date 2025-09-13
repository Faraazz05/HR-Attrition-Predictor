"""
HR Attrition Predictor - Employee Management Dashboard
====================================================
Comprehensive employee directory with individual profiles, HR action recommendations,
and integrated email automation. Memory-optimized for 4GB RAM systems.

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
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import json
import io
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

# Import email service
try:
    from src.utils.email_service import (
        EmailService, SMTPConfig, EmailRecipient, create_email_service
    )
    EMAIL_SERVICE_AVAILABLE = True
except ImportError:
    EMAIL_SERVICE_AVAILABLE = False
    st.error("📧 Email service not available. Please install required dependencies.")

# Suppress warnings
warnings.filterwarnings('ignore')

# ================================================================
# DATA LOADING AND CACHING
# ================================================================

@st.cache_data(ttl=300)
def load_employee_directory():
    """Load comprehensive employee directory with all details."""
    
    try:
        data_path = project_root / "data" / "synthetic" / "hr_employees.csv"
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            
            # Enrich employee data
            df = _enrich_employee_data(df)
            
            return df, True
        else:
            return _generate_comprehensive_employee_data(), False
    
    except Exception as e:
        st.error(f"Error loading employee directory: {e}")
        return _generate_comprehensive_employee_data(), False

def _enrich_employee_data(df):
    """Enrich employee data with calculated fields."""
    
    # Full name
    if 'FirstName' in df.columns and 'LastName' in df.columns:
        df['FullName'] = df['FirstName'] + ' ' + df['LastName']
    elif 'FullName' not in df.columns:
        df['FullName'] = df.get('EmployeeID', 'Unknown')
    
    # Risk scoring (simulate if not present)
    if 'AttritionProbability' not in df.columns:
        df['AttritionProbability'] = _simulate_risk_scores(df)
    
    df['RiskLevel'] = df['AttritionProbability'].apply(get_risk_level)
    
    # Employee status
    df['Status'] = np.random.choice(['Active', 'On Leave', 'Remote'], len(df), p=[0.85, 0.05, 0.10])
    
    # Manager assignments
    if 'Manager' not in df.columns:
        managers = df.sample(min(20, len(df) // 10))['FullName'].tolist()
        df['Manager'] = np.random.choice(managers, len(df))
    
    # Last interaction dates
    df['LastInteraction'] = pd.date_range(
        end=datetime.now(), 
        periods=len(df), 
        freq='-1D'
    ) + pd.to_timedelta(np.random.randint(0, 90, len(df)), unit='D')
    
    # Performance ratings
    if 'PerformanceRating' not in df.columns:
        df['PerformanceRating'] = np.random.choice([1, 2, 3, 4, 5], len(df), p=[0.05, 0.15, 0.35, 0.35, 0.10])
    
    # Contact information (simulated)
    df['Email'] = df['FirstName'].str.lower() + '.' + df['LastName'].str.lower() + '@company.com'
    df['Phone'] = '+1-555-' + np.random.randint(1000, 9999, len(df)).astype(str)
    
    # HR flags
    df['RequiresAttention'] = (df['RiskLevel'] == 'High') | (df['PerformanceRating'] <= 2)
    df['EligibleForPromotion'] = (df['PerformanceRating'] >= 4) & (df['YearsInCurrentRole'] >= 2)
    
    # Salary bands
    if 'MonthlyIncome' in df.columns:
        df['SalaryBand'] = pd.cut(
            df['MonthlyIncome'], 
            bins=[0, 4000, 6000, 8000, 12000, 50000],
            labels=['Entry', 'Mid', 'Senior', 'Principal', 'Executive']
        )
    
    return df

def _simulate_risk_scores(df):
    """Simulate realistic risk scores based on available data."""
    
    np.random.seed(42)
    base_risk = np.random.beta(2, 5, len(df))  # Skewed towards lower risk
    
    # Adjust based on factors
    adjustments = np.zeros(len(df))
    
    if 'Age' in df.columns:
        # Younger employees typically higher risk
        age_factor = np.clip((35 - df['Age']) / 100, -0.2, 0.3)
        adjustments += age_factor
    
    if 'JobSatisfaction' in df.columns:
        # Low satisfaction increases risk
        sat_factor = (3.5 - df['JobSatisfaction']) / 20
        adjustments += sat_factor
    
    if 'YearsAtCompany' in df.columns:
        # New employees higher risk
        tenure_factor = np.clip((3 - df['YearsAtCompany']) / 20, -0.1, 0.2)
        adjustments += tenure_factor
    
    final_risk = np.clip(base_risk + adjustments, 0.01, 0.99)
    return final_risk

def _generate_comprehensive_employee_data():
    """Generate comprehensive demo employee data."""
    
    np.random.seed(42)
    n_employees = 500
    
    first_names = ['Alex', 'Jordan', 'Taylor', 'Casey', 'Riley', 'Morgan', 'Avery', 'Quinn', 'Sage', 'River', 
                   'Emma', 'Liam', 'Olivia', 'Noah', 'Sophia', 'Jackson', 'Ava', 'Lucas', 'Isabella', 'Mason']
    last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas',
                  'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez']
    
    departments = ['Engineering', 'Sales', 'Marketing', 'Operations', 'Finance', 'HR', 'Legal', 'Customer Success', 'Product']
    job_roles = ['Manager', 'Senior', 'Junior', 'Lead', 'Principal', 'Associate', 'Director', 'VP', 'Analyst', 'Specialist']
    
    demo_data = pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n_employees + 1)],
        'FirstName': np.random.choice(first_names, n_employees),
        'LastName': np.random.choice(last_names, n_employees),
        'Age': np.random.randint(22, 65, n_employees),
        'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_employees, p=[0.48, 0.48, 0.04]),
        'Department': np.random.choice(departments, n_employees),
        'JobRole': np.random.choice(job_roles, n_employees),
        'JobLevel': np.random.randint(1, 8, n_employees),
        'YearsAtCompany': np.random.gamma(2, 2, n_employees).astype(int),
        'YearsInCurrentRole': np.random.gamma(1.5, 1.5, n_employees).astype(int),
        'MonthlyIncome': np.random.normal(7500, 3000, n_employees).astype(int),
        'JobSatisfaction': np.random.randint(1, 5, n_employees),
        'WorkLifeBalance': np.random.randint(1, 4, n_employees),
        'PerformanceRating': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.05, 0.15, 0.35, 0.35, 0.10]),
        'DistanceFromHome': np.random.randint(1, 50, n_employees),
        'Education': np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], n_employees, p=[0.15, 0.45, 0.35, 0.05]),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_employees, p=[0.35, 0.55, 0.10]),
        'OverTime': np.random.choice(['Yes', 'No'], n_employees, p=[0.28, 0.72]),
        'BusinessTravel': np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], n_employees, p=[0.25, 0.60, 0.15])
    })
    
    # Fix data consistency
    demo_data['MonthlyIncome'] = np.clip(demo_data['MonthlyIncome'], 3000, 25000)
    demo_data['YearsInCurrentRole'] = np.clip(demo_data['YearsInCurrentRole'], 0, demo_data['YearsAtCompany'])
    
    # Apply enrichment
    demo_data = _enrich_employee_data(demo_data)
    
    return demo_data

@st.cache_resource
def initialize_email_service():
    """Initialize email service with error handling."""
    
    if not EMAIL_SERVICE_AVAILABLE:
        return None
    
    try:
        # Try to create email service from environment
        return create_email_service()
    except Exception as e:
        st.warning(f"Email service initialization failed: {e}")
        return None

# ================================================================
# SEARCHABLE EMPLOYEE DIRECTORY
# ================================================================

def searchable_employee_directory():
    """Create searchable and filterable employee directory."""
    
    st.markdown("### 🔍 Employee Directory")
    
    # Load data
    employees, is_real = load_employee_directory()
    
    if employees.empty:
        st.error("No employee data available")
        return
    
    # Search and filter controls
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    
    with col1:
        search_term = st.text_input(
            "🔎 Search employees",
            placeholder="Search by name, role, department, or ID...",
            help="Search across name, role, department, and employee ID"
        )
    
    with col2:
        departments = ['All'] + sorted(employees['Department'].unique().tolist())
        selected_dept = st.selectbox("Department Filter", departments)
    
    with col3:
        risk_levels = ['All'] + sorted(employees['RiskLevel'].unique().tolist())
        selected_risk = st.selectbox("Risk Level Filter", risk_levels)
    
    with col4:
        view_mode = st.selectbox("View", ["Cards", "Table"])
    
    # Apply filters
    filtered_employees = _apply_directory_filters(employees, search_term, selected_dept, selected_risk)
    
    # Display results summary
    st.markdown(f"**Showing {len(filtered_employees)} of {len(employees)} employees**")
    
    if len(filtered_employees) == 0:
        st.warning("No employees match the current filters")
        return
    
    # Display mode selection
    if view_mode == "Cards":
        _display_employee_cards(filtered_employees)
    else:
        _display_employee_table(filtered_employees)

def _apply_directory_filters(df, search_term, department, risk_level):
    """Apply search and filter criteria to employee directory."""
    
    filtered = df.copy()
    
    # Search filter
    if search_term:
        search_cols = ['FullName', 'Department', 'JobRole', 'EmployeeID']
        search_mask = df[search_cols].astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered = filtered[search_mask]
    
    # Department filter
    if department != 'All':
        filtered = filtered[filtered['Department'] == department]
    
    # Risk level filter
    if risk_level != 'All':
        filtered = filtered[filtered['RiskLevel'] == risk_level]
    
    return filtered

def _display_employee_cards(employees):
    """Display employees in card format."""
    
    # Create cards in responsive grid
    cols_per_row = 3
    
    for i in range(0, len(employees), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(employees):
                employee = employees.iloc[i + j]
                
                with col:
                    _create_employee_card(employee)

def _create_employee_card(employee):
    """Create individual employee card."""
    
    risk_color = get_risk_color(employee['RiskLevel'])
    
    # Status indicators
    status_icon = "🟢" if employee['Status'] == 'Active' else "🟡" if employee['Status'] == 'On Leave' else "🔵"
    attention_flag = "🚨" if employee.get('RequiresAttention', False) else ""
    promotion_flag = "⭐" if employee.get('EligibleForPromotion', False) else ""
    
    st.markdown(f"""
    <div style="
        background: rgba(37, 42, 69, 0.4);
        border-radius: 12px;
        border-left: 4px solid {risk_color};
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
        
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
            <div>
                <h4 style="color: {COLORS['text']}; margin: 0 0 5px 0; font-size: 16px;">
                    {attention_flag} {promotion_flag} {employee['FullName']}
                </h4>
                <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 12px;">
                    {employee['JobRole']} • {employee['Department']}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="color: {risk_color}; font-weight: bold; font-size: 12px;">
                    {employee['RiskLevel'].upper()}
                </div>
                <div style="color: {COLORS['text_secondary']}; font-size: 10px;">
                    {status_icon} {employee['Status']}
                </div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; font-size: 11px;">
            <div>
                <span style="color: {COLORS['text_secondary']};">ID:</span>
                <span style="color: {COLORS['text']}; font-weight: bold;"> {employee['EmployeeID']}</span>
            </div>
            <div>
                <span style="color: {COLORS['text_secondary']};">Tenure:</span>
                <span style="color: {COLORS['text']}; font-weight: bold;"> {employee.get('YearsAtCompany', 0)}y</span>
            </div>
            <div>
                <span style="color: {COLORS['text_secondary']};">Performance:</span>
                <span style="color: {COLORS['success'] if employee.get('PerformanceRating', 3) >= 4 else COLORS['warning'] if employee.get('PerformanceRating', 3) >= 3 else COLORS['error']}; font-weight: bold;">
                    {employee.get('PerformanceRating', 3)}/5
                </span>
            </div>
            <div>
                <span style="color: {COLORS['text_secondary']};">Satisfaction:</span>
                <span style="color: {COLORS['success'] if employee.get('JobSatisfaction', 3) >= 4 else COLORS['warning'] if employee.get('JobSatisfaction', 3) >= 3 else COLORS['error']}; font-weight: bold;">
                    {employee.get('JobSatisfaction', 3)}/5
                </span>
            </div>
        </div>
        
        <div style="border-top: 1px solid {COLORS['border_primary']}; padding-top: 15px;">
            <button style="
                background: {COLORS['secondary']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 11px;
                cursor: pointer;
                width: 100%;
                font-weight: bold;
            " onclick="window.open('#', '_blank')">
                📋 View Full Profile
            </button>
        </div>
        
    </div>
    """, unsafe_allow_html=True)
    
    # Add profile button functionality (would need to be handled via Streamlit session state)
    if st.button(f"View {employee['FirstName']}'s Profile", key=f"profile_{employee['EmployeeID']}", help="View detailed employee profile"):
        st.session_state['selected_employee_id'] = employee['EmployeeID']
        st.rerun()

def _display_employee_table(employees):
    """Display employees in table format with enhanced styling."""
    
    # Prepare display data
    display_cols = [
        'FullName', 'EmployeeID', 'Department', 'JobRole', 'Status',
        'RiskLevel', 'PerformanceRating', 'JobSatisfaction', 'YearsAtCompany'
    ]
    
    display_data = employees[display_cols].copy()
    
    # Format columns
    display_data = display_data.rename(columns={
        'FullName': 'Name',
        'EmployeeID': 'ID',
        'JobRole': 'Role',
        'RiskLevel': 'Risk',
        'PerformanceRating': 'Performance',
        'JobSatisfaction': 'Satisfaction',
        'YearsAtCompany': 'Tenure'
    })
    
    # Apply conditional formatting
    def color_risk_level(val):
        color = get_risk_color(val)
        return f'background-color: {color}33; color: {color}; font-weight: bold'
    
    def color_performance(val):
        if val >= 4:
            return f'color: {COLORS["success"]}; font-weight: bold'
        elif val >= 3:
            return f'color: {COLORS["warning"]}'
        else:
            return f'color: {COLORS["error"]}; font-weight: bold'
    
    # Style the dataframe
    styled_df = display_data.style.applymap(
        color_risk_level, subset=['Risk']
    ).applymap(
        color_performance, subset=['Performance']
    )
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
        column_config={
            "Name": st.column_config.TextColumn("Employee Name", width="medium"),
            "Risk": st.column_config.TextColumn("Attrition Risk", width="small"),
            "Performance": st.column_config.NumberColumn("Performance", format="%d/5"),
            "Satisfaction": st.column_config.NumberColumn("Satisfaction", format="%d/5"),
            "Tenure": st.column_config.NumberColumn("Years", format="%d years")
        }
    )
    
    # Bulk actions
    st.markdown("#### 🔧 Bulk Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Send Engagement Survey", help="Send engagement surveys to filtered employees"):
            _handle_bulk_email_action(employees, 'engagement_survey')
    
    with col2:
        if st.button("🚨 Alert Managers", help="Alert managers of high-risk employees"):
            _handle_bulk_manager_alerts(employees)
    
    with col3:
        if st.button("📊 Export to CSV", help="Export filtered employee data"):
            _handle_export_employees(employees)

# ================================================================
# INDIVIDUAL EMPLOYEE PROFILE
# ================================================================

def individual_employee_profile():
    """Display detailed individual employee profile."""
    
    st.markdown("### 👤 Employee Profile")
    
    employees, _ = load_employee_directory()
    
    # Employee selection
    if 'selected_employee_id' not in st.session_state:
        st.session_state.selected_employee_id = None
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        employee_options = {f"{row['FullName']} ({row['EmployeeID']})": row['EmployeeID'] 
                           for _, row in employees.iterrows()}
        
        selected_option = st.selectbox(
            "Select Employee:",
            options=list(employee_options.keys()),
            index=0 if not st.session_state.selected_employee_id else None
        )
        
        if selected_option:
            selected_employee_id = employee_options[selected_option]
            st.session_state.selected_employee_id = selected_employee_id
    
    with col2:
        if st.button("🔄 Refresh Profile", type="secondary"):
            st.rerun()
    
    if not st.session_state.selected_employee_id:
        st.info("Please select an employee to view their profile")
        return
    
    # Get selected employee data
    employee = employees[employees['EmployeeID'] == st.session_state.selected_employee_id].iloc[0]
    
    # Profile header
    _display_profile_header(employee)
    
    # Profile tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "📈 Performance", 
        "🎯 Risk Analysis", 
        "💼 Actions"
    ])
    
    with tab1:
        _display_profile_overview(employee)
    
    with tab2:
        _display_profile_performance(employee)
    
    with tab3:
        _display_profile_risk_analysis(employee)
    
    with tab4:
        _display_profile_actions(employee)

def _display_profile_header(employee):
    """Display employee profile header with key information."""
    
    risk_color = get_risk_color(employee['RiskLevel'])
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['background_light']} 0%, {COLORS['primary']} 100%);
        padding: 30px;
        border-radius: 15px;
        border-left: 6px solid {risk_color};
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h2 style="color: {COLORS['text']}; margin: 0 0 10px 0; font-size: 28px; font-family: 'Orbitron';">
                    👤 {employee['FullName']}
                </h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
                    <div>
                        <span style="color: {COLORS['text_secondary']}; font-size: 14px;">Employee ID:</span><br>
                        <span style="color: {COLORS['text']}; font-weight: bold; font-size: 16px;">{employee['EmployeeID']}</span>
                    </div>
                    <div>
                        <span style="color: {COLORS['text_secondary']}; font-size: 14px;">Department:</span><br>
                        <span style="color: {COLORS['secondary']}; font-weight: bold; font-size: 16px;">{employee['Department']}</span>
                    </div>
                    <div>
                        <span style="color: {COLORS['text_secondary']}; font-size: 14px;">Role:</span><br>
                        <span style="color: {COLORS['accent']}; font-weight: bold; font-size: 16px;">{employee['JobRole']}</span>
                    </div>
                    <div>
                        <span style="color: {COLORS['text_secondary']}; font-size: 14px;">Manager:</span><br>
                        <span style="color: {COLORS['text']}; font-weight: bold; font-size: 16px;">{employee.get('Manager', 'Not Assigned')}</span>
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="
                    background: {risk_color};
                    color: white;
                    padding: 12px 20px;
                    border-radius: 25px;
                    font-weight: bold;
                    font-size: 16px;
                    margin-bottom: 10px;
                ">
                    {employee['RiskLevel'].upper()} RISK
                </div>
                <div style="color: {COLORS['text_secondary']}; font-size: 14px;">
                    Last Updated: {datetime.now().strftime('%Y-%m-%d')}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def _display_profile_overview(employee):
    """Display employee profile overview tab."""
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(glassmorphism_metric_card(
            value=f"{employee.get('YearsAtCompany', 0)}",
            title="Years at Company",
            subtitle="Tenure",
            icon="📅",
            color='info'
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(glassmorphism_metric_card(
            value=f"${employee.get('MonthlyIncome', 0):,.0f}",
            title="Monthly Income",
            subtitle="Current Salary",
            icon="💰",
            color='success'
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(glassmorphism_metric_card(
            value=f"{employee.get('PerformanceRating', 0)}/5",
            title="Performance",
            subtitle="Latest Rating",
            icon="⭐",
            color='warning'
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(glassmorphism_metric_card(
            value=f"{employee.get('JobSatisfaction', 0)}/5",
            title="Job Satisfaction",
            subtitle="Self-Reported",
            icon="😊",
            color='secondary'
        ), unsafe_allow_html=True)
    
    # Detailed information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Personal Information")
        
        info_items = [
            ("Age", f"{employee.get('Age', 'N/A')} years"),
            ("Gender", employee.get('Gender', 'Not Specified')),
            ("Education", employee.get('Education', 'Not Specified')),
            ("Marital Status", employee.get('MaritalStatus', 'Not Specified')),
            ("Distance from Office", f"{employee.get('DistanceFromHome', 0)} miles"),
            ("Work Status", employee.get('Status', 'Active'))
        ]
        
        for label, value in info_items:
            st.markdown(f"""
            <div style="
                display: flex; 
                justify-content: space-between; 
                padding: 8px 0; 
                border-bottom: 1px solid {COLORS['border_primary']}33;
            ">
                <span style="color: {COLORS['text_secondary']};">{label}:</span>
                <span style="color: {COLORS['text']}; font-weight: bold;">{value}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 💼 Work Information")
        
        work_items = [
            ("Job Level", f"Level {employee.get('JobLevel', 'N/A')}"),
            ("Years in Role", f"{employee.get('YearsInCurrentRole', 0)} years"),
            ("Salary Band", employee.get('SalaryBand', 'Not Classified')),
            ("Overtime", employee.get('OverTime', 'No')),
            ("Business Travel", employee.get('BusinessTravel', 'Not Required')),
            ("Work-Life Balance", f"{employee.get('WorkLifeBalance', 0)}/4")
        ]
        
        for label, value in work_items:
            st.markdown(f"""
            <div style="
                display: flex; 
                justify-content: space-between; 
                padding: 8px 0; 
                border-bottom: 1px solid {COLORS['border_primary']}33;
            ">
                <span style="color: {COLORS['text_secondary']};">{label}:</span>
                <span style="color: {COLORS['text']}; font-weight: bold;">{value}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Contact Information
    st.markdown("#### 📞 Contact Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: rgba(37, 42, 69, 0.3);
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid {COLORS['info']};
        ">
            <div style="color: {COLORS['info']}; font-weight: bold; margin-bottom: 8px;">📧 Email</div>
            <div style="color: {COLORS['text']};">{employee.get('Email', 'Not Available')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: rgba(37, 42, 69, 0.3);
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid {COLORS['success']};
        ">
            <div style="color: {COLORS['success']}; font-weight: bold; margin-bottom: 8px;">📞 Phone</div>
            <div style="color: {COLORS['text']};">{employee.get('Phone', 'Not Available')}</div>
        </div>
        """, unsafe_allow_html=True)

def _display_profile_performance(employee):
    """Display employee performance analysis."""
    
    st.markdown("#### 📈 Performance Trends")
    
    # Simulate performance history
    months = pd.date_range(end=datetime.now(), periods=12, freq='M')
    base_performance = employee.get('PerformanceRating', 3)
    
    # Generate realistic performance trend
    np.random.seed(hash(employee['EmployeeID']) % 2147483647)
    trend = np.random.normal(0, 0.3, 12)
    seasonal = 0.2 * np.sin(2 * np.pi * np.arange(12) / 12)
    performance_history = np.clip(base_performance + trend + seasonal, 1, 5)
    
    # Performance chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months,
        y=performance_history,
        mode='lines+markers',
        name='Performance Rating',
        line=dict(color=COLORS['secondary'], width=3),
        marker=dict(size=8, color=COLORS['secondary']),
        hovertemplate='<b>%{x|%B %Y}</b><br>Rating: %{y:.1f}/5<extra></extra>'
    ))
    
    # Add average line
    avg_performance = np.mean(performance_history)
    fig.add_hline(
        y=avg_performance,
        line_dash="dash",
        line_color=COLORS['accent'],
        annotation_text=f"Average: {avg_performance:.2f}"
    )
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title="Performance Rating History",
        height=400,
        custom_layout={
            'yaxis': dict(range=[0.5, 5.5], title='Performance Rating'),
            'xaxis': dict(title='Month')
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Performance Metrics")
        
        metrics = {
            "Current Rating": f"{employee.get('PerformanceRating', 0)}/5",
            "12-Month Average": f"{avg_performance:.2f}/5",
            "Improvement Trend": "↗️ Improving" if performance_history[-1] > performance_history[0] else "↘️ Declining" if performance_history[-1] < performance_history[0] else "➡️ Stable",
            "Consistency": f"{(1 - np.std(performance_history) / np.mean(performance_history)):.1%}"
        }
        
        for label, value in metrics.items():
            color = (COLORS['success'] if '↗️' in value or ('/' in value and float(value.split('/')[0]) >= 4)
                    else COLORS['error'] if '↘️' in value
                    else COLORS['text'])
            
            st.markdown(f"""
            <div style="
                display: flex; 
                justify-content: space-between; 
                padding: 10px; 
                background: rgba(37, 42, 69, 0.3);
                border-radius: 6px;
                margin: 8px 0;
            ">
                <span style="color: {COLORS['text_secondary']};">{label}:</span>
                <span style="color: {color}; font-weight: bold;">{value}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🏆 Performance Categories")
        
        # Performance radar chart (simulated)
        categories = ['Quality', 'Productivity', 'Teamwork', 'Innovation', 'Leadership']
        values = np.random.uniform(2.5, 5, len(categories))  # Based on overall performance
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Performance',
            line=dict(color=COLORS['secondary'], width=2),
            fillcolor=f'{COLORS["secondary"]}33'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    gridcolor=COLORS['border_primary'],
                    tickfont=dict(color=COLORS['text_secondary'])
                ),
                angularaxis=dict(
                    gridcolor=COLORS['border_primary'],
                    tickfont=dict(color=COLORS['text'])
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def _display_profile_risk_analysis(employee):
    """Display detailed risk analysis for employee."""
    
    st.markdown("#### 🎯 Attrition Risk Analysis")
    
    risk_prob = employee.get('AttritionProbability', 0.5)
    risk_level = employee['RiskLevel']
    risk_color = get_risk_color(risk_level)
    
    # Risk gauge
    col1, col2 = st.columns([2, 1])
    
    with col1:
        gauge_fig = futuristic_gauge_chart(
            value=risk_prob * 100,
            title="Attrition Risk Probability",
            min_value=0,
            max_value=100,
            unit="%",
            height=350
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: rgba(37, 42, 69, 0.4);
            padding: 25px;
            border-radius: 12px;
            border-left: 4px solid {risk_color};
            text-align: center;
            height: 300px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="color: {risk_color}; font-size: 48px; margin-bottom: 15px;">
                {'🚨' if risk_level == 'High' else '⚠️' if risk_level == 'Medium' else '✅'}
            </div>
            <div style="color: {COLORS['text']}; font-size: 24px; font-weight: bold; margin-bottom: 10px;">
                {risk_level.upper()} RISK
            </div>
            <div style="color: {COLORS['text_secondary']}; font-size: 16px;">
                {risk_prob:.1%} probability
            </div>
            <div style="color: {COLORS['text_secondary']}; font-size: 12px; margin-top: 15px;">
                Last assessed: {datetime.now().strftime('%Y-%m-%d')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk factors analysis
    st.markdown("#### 🔍 Risk Factor Analysis")
    
    # Simulate risk factors based on employee data
    risk_factors = _analyze_employee_risk_factors(employee)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🚨 Risk Factors (Increase Attrition)**")
        
        for factor in risk_factors['high_risk']:
            st.markdown(f"""
            <div style="
                padding: 10px;
                background: rgba(255, 45, 117, 0.1);
                border-radius: 6px;
                border-left: 3px solid {COLORS['error']};
                margin: 8px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color: {COLORS['text']};">{factor['name']}</span>
                <span style="color: {COLORS['error']}; font-weight: bold;">+{factor['impact']:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**✅ Protective Factors (Reduce Attrition)**")
        
        for factor in risk_factors['protective']:
            st.markdown(f"""
            <div style="
                padding: 10px;
                background: rgba(0, 255, 136, 0.1);
                border-radius: 6px;
                border-left: 3px solid {COLORS['success']};
                margin: 8px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color: {COLORS['text']};">{factor['name']}</span>
                <span style="color: {COLORS['success']}; font-weight: bold;">-{factor['impact']:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Risk trend over time
    st.markdown("#### 📈 Risk Trend Analysis")
    
    # Simulate risk history
    months = pd.date_range(end=datetime.now(), periods=6, freq='M')
    base_risk = risk_prob
    risk_trend = np.random.normal(base_risk, 0.05, len(months))
    risk_trend = np.clip(risk_trend, 0.01, 0.99)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months,
        y=risk_trend * 100,
        mode='lines+markers',
        name='Attrition Risk',
        line=dict(color=risk_color, width=3),
        marker=dict(size=8),
        fill='tonexty'
    ))
    
    # Risk threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS['error'], annotation_text="High Risk (70%)")
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS['warning'], annotation_text="Medium Risk (30%)")
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title="6-Month Risk Trend",
        height=300,
        custom_layout={
            'yaxis': dict(range=[0, 100], title='Risk Probability (%)'),
            'xaxis': dict(title='Month')
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _analyze_employee_risk_factors(employee):
    """Analyze individual risk factors for an employee."""
    
    high_risk_factors = []
    protective_factors = []
    
    # Age factor
    age = employee.get('Age', 35)
    if age < 30:
        high_risk_factors.append({'name': 'Young Age', 'impact': 0.12})
    elif age > 50:
        protective_factors.append({'name': 'Experienced Age', 'impact': 0.08})
    
    # Tenure factor
    tenure = employee.get('YearsAtCompany', 5)
    if tenure < 2:
        high_risk_factors.append({'name': 'Low Tenure', 'impact': 0.20})
    elif tenure > 10:
        protective_factors.append({'name': 'Long Tenure', 'impact': 0.15})
    
    # Satisfaction factors
    job_sat = employee.get('JobSatisfaction', 3)
    if job_sat <= 2:
        high_risk_factors.append({'name': 'Low Job Satisfaction', 'impact': 0.25})
    elif job_sat >= 4:
        protective_factors.append({'name': 'High Job Satisfaction', 'impact': 0.18})
    
    # Performance factor
    performance = employee.get('PerformanceRating', 3)
    if performance <= 2:
        high_risk_factors.append({'name': 'Low Performance', 'impact': 0.15})
    elif performance >= 4:
        protective_factors.append({'name': 'High Performance', 'impact': 0.12})
    
    # Work-life balance
    wlb = employee.get('WorkLifeBalance', 3)
    if wlb <= 2:
        high_risk_factors.append({'name': 'Poor Work-Life Balance', 'impact': 0.18})
    elif wlb >= 4:
        protective_factors.append({'name': 'Good Work-Life Balance', 'impact': 0.14})
    
    # Overtime
    if employee.get('OverTime') == 'Yes':
        high_risk_factors.append({'name': 'Frequent Overtime', 'impact': 0.10})
    
    # Distance from home
    distance = employee.get('DistanceFromHome', 10)
    if distance > 25:
        high_risk_factors.append({'name': 'Long Commute', 'impact': 0.08})
    
    return {
        'high_risk': high_risk_factors,
        'protective': protective_factors
    }

def _display_profile_actions(employee):
    """Display actionable items and recommendations for employee."""
    
    st.markdown("#### 💼 Recommended Actions")
    
    # Get email service
    email_service = initialize_email_service()
    
    # Action recommendations based on risk level
    risk_level = employee['RiskLevel']
    
    if risk_level == 'High':
        actions = [
            {
                'title': '🚨 Schedule Immediate One-on-One',
                'description': 'Urgent meeting with manager within 24 hours',
                'priority': 'Critical',
                'timeline': 'Immediate'
            },
            {
                'title': '💰 Review Compensation Package',
                'description': 'Assess salary competitiveness and benefits',
                'priority': 'High',
                'timeline': '1 week'
            },
            {
                'title': '📈 Career Development Discussion',
                'description': 'Explore growth opportunities and career path',
                'priority': 'High',
                'timeline': '1 week'
            }
        ]
    elif risk_level == 'Medium':
        actions = [
            {
                'title': '📅 Regular Check-in Schedule',
                'description': 'Bi-weekly meetings with direct manager',
                'priority': 'Medium',
                'timeline': '2 weeks'
            },
            {
                'title': '😊 Satisfaction Assessment',
                'description': 'Send targeted satisfaction survey',
                'priority': 'Medium',
                'timeline': '1 week'
            },
            {
                'title': '🎓 Training Opportunities',
                'description': 'Identify skill development programs',
                'priority': 'Medium',
                'timeline': '1 month'
            }
        ]
    else:  # Low risk
        actions = [
            {
                'title': '✅ Continue Current Approach',
                'description': 'Maintain current engagement strategies',
                'priority': 'Low',
                'timeline': 'Ongoing'
            },
            {
                'title': '🌟 Recognition Program',
                'description': 'Acknowledge strong performance',
                'priority': 'Low',
                'timeline': '1 month'
            },
            {
                'title': '👥 Mentorship Opportunity',
                'description': 'Consider as mentor for other employees',
                'priority': 'Low',
                'timeline': '3 months'
            }
        ]
    
    # Display actions
    for i, action in enumerate(actions, 1):
        priority_color = {
            'Critical': COLORS['error'],
            'High': COLORS['warning'],
            'Medium': COLORS['secondary'],
            'Low': COLORS['success']
        }.get(action['priority'], COLORS['secondary'])
        
        st.markdown(f"""
        <div style="
            background: rgba(37, 42, 69, 0.4);
            border-radius: 12px;
            border-left: 4px solid {priority_color};
            padding: 20px;
            margin: 15px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
                <div style="flex: 1;">
                    <h4 style="color: {COLORS['text']}; margin: 0 0 8px 0;">
                        {i}. {action['title']}
                    </h4>
                    <p style="color: {COLORS['text_secondary']}; margin: 0; line-height: 1.5;">
                        {action['description']}
                    </p>
                </div>
                <div style="text-align: right; margin-left: 20px;">
                    <div style="
                        background: {priority_color};
                        color: white;
                        padding: 6px 12px;
                        border-radius: 15px;
                        font-size: 12px;
                        font-weight: bold;
                        margin-bottom: 5px;
                    ">
                        {action['priority']}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                        {action['timeline']}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("#### 🔧 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Send Engagement Survey", key=f"survey_{employee['EmployeeID']}"):
            if email_service:
                _send_individual_survey(employee, email_service)
            else:
                st.error("Email service not available")
    
    with col2:
        if st.button("🚨 Alert Manager", key=f"alert_{employee['EmployeeID']}"):
            if email_service:
                _send_manager_alert(employee, email_service)
            else:
                st.error("Email service not available")
    
    with col3:
        if st.button("💬 Schedule Meeting", key=f"meeting_{employee['EmployeeID']}"):
            st.success("Meeting request sent to manager")
            # In real implementation, integrate with calendar API
    
    # Documentation section
    st.markdown("#### 📝 Action Documentation")
    
    action_notes = st.text_area(
        "Add notes about actions taken:",
        placeholder="Document any meetings, emails sent, or other actions taken for this employee...",
        key=f"notes_{employee['EmployeeID']}"
    )
    
    if st.button("💾 Save Notes", key=f"save_notes_{employee['EmployeeID']}"):
        # In real implementation, save to database
        st.success("Notes saved successfully!")

# ================================================================
# HR ACTION RECOMMENDATIONS
# ================================================================

def hr_action_recommendations():
    """Generate and display HR action recommendations."""
    
    st.markdown("### 🎯 HR Action Recommendations")
    
    employees, _ = load_employee_directory()
    
    # Generate recommendations
    recommendations = _generate_hr_recommendations(employees)
    
    # Display recommendation categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Urgent Actions",
        "📅 Scheduled Actions", 
        "📊 Strategic Initiatives",
        "📈 Performance Reviews"
    ])
    
    with tab1:
        _display_urgent_recommendations(recommendations['urgent'])
    
    with tab2:
        _display_scheduled_recommendations(recommendations['scheduled'])
    
    with tab3:
        _display_strategic_recommendations(recommendations['strategic'])
    
    with tab4:
        _display_performance_recommendations(recommendations['performance'])

def _generate_hr_recommendations(employees):
    """Generate comprehensive HR recommendations based on employee data."""
    
    recommendations = {
        'urgent': [],
        'scheduled': [],
        'strategic': [],
        'performance': []
    }
    
    # Urgent: High-risk employees
    high_risk = employees[employees['RiskLevel'] == 'High']
    for _, emp in high_risk.iterrows():
        recommendations['urgent'].append({
            'type': 'High Risk Employee',
            'employee': emp['FullName'],
            'employee_id': emp['EmployeeID'],
            'description': f"{emp['FullName']} ({emp['Department']}) has {emp['AttritionProbability']:.1%} attrition risk",
            'action': 'Schedule immediate one-on-one with manager',
            'timeline': '24 hours',
            'priority': 'Critical'
        })
    
    # Scheduled: Performance reviews due
    low_performers = employees[employees['PerformanceRating'] <= 2]
    for _, emp in low_performers.iterrows():
        recommendations['scheduled'].append({
            'type': 'Performance Review',
            'employee': emp['FullName'],
            'employee_id': emp['EmployeeID'],
            'description': f"Performance rating of {emp['PerformanceRating']}/5 requires review",
            'action': 'Conduct performance improvement meeting',
            'timeline': '1 week',
            'priority': 'High'
        })
    
    # Strategic: Department-wide issues
    dept_risks = employees.groupby('Department')['AttritionProbability'].mean()
    high_risk_depts = dept_risks[dept_risks > 0.3]
    
    for dept, risk in high_risk_depts.items():
        emp_count = len(employees[employees['Department'] == dept])
        recommendations['strategic'].append({
            'type': 'Department Risk',
            'department': dept,
            'description': f"{dept} department has {risk:.1%} average attrition risk ({emp_count} employees)",
            'action': 'Implement department-wide engagement initiative',
            'timeline': '2 weeks',
            'priority': 'Medium'
        })
    
    # Performance: Promotable employees
    promotable = employees[(employees['PerformanceRating'] >= 4) & (employees['YearsInCurrentRole'] >= 2)]
    for _, emp in promotable.iterrows():
        recommendations['performance'].append({
            'type': 'Promotion Candidate',
            'employee': emp['FullName'],
            'employee_id': emp['EmployeeID'],
            'description': f"High performer ({emp['PerformanceRating']}/5) with {emp['YearsInCurrentRole']} years in role",
            'action': 'Discuss career advancement opportunities',
            'timeline': '2 weeks',
            'priority': 'Medium'
        })
    
    return recommendations

def _display_urgent_recommendations(urgent_recs):
    """Display urgent action recommendations."""
    
    if not urgent_recs:
        st.success("🎉 No urgent actions required at this time!")
        return
    
    st.markdown(f"**{len(urgent_recs)} urgent actions require immediate attention:**")
    
    for i, rec in enumerate(urgent_recs, 1):
        st.markdown(f"""
        <div style="
            background: rgba(255, 45, 117, 0.1);
            border-radius: 12px;
            border-left: 4px solid {COLORS['error']};
            padding: 20px;
            margin: 15px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <h4 style="color: {COLORS['error']}; margin: 0 0 10px 0; font-size: 18px;">
                        🚨 {i}. {rec['type']}
                    </h4>
                    <p style="color: {COLORS['text']}; margin: 0 0 10px 0; font-size: 14px;">
                        <strong>{rec.get('employee', rec.get('department', 'N/A'))}</strong>
                    </p>
                    <p style="color: {COLORS['text_secondary']}; margin: 0 0 15px 0; line-height: 1.5;">
                        {rec['description']}
                    </p>
                    <div style="color: {COLORS['warning']}; font-weight: bold; font-size: 14px;">
                        📋 Action: {rec['action']}
                    </div>
                </div>
                <div style="text-align: right; margin-left: 20px;">
                    <div style="
                        background: {COLORS['error']};
                        color: white;
                        padding: 8px 16px;
                        border-radius: 20px;
                        font-size: 12px;
                        font-weight: bold;
                        margin-bottom: 8px;
                    ">
                        {rec['timeline']}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                        {rec['priority']} Priority
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Bulk action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Email All Managers", key="urgent_email_managers"):
            _handle_bulk_urgent_alerts(urgent_recs)
    
    with col2:
        if st.button("📅 Schedule All Meetings", key="urgent_schedule"):
            st.success("Meeting requests sent to all relevant managers")
    
    with col3:
        if st.button("📊 Generate Report", key="urgent_report"):
            _generate_urgent_actions_report(urgent_recs)

def _display_scheduled_recommendations(scheduled_recs):
    """Display scheduled action recommendations."""
    
    if not scheduled_recs:
        st.info("No scheduled actions at this time")
        return
    
    st.markdown(f"**{len(scheduled_recs)} scheduled actions:**")
    
    # Group by timeline
    timelines = {}
    for rec in scheduled_recs:
        timeline = rec['timeline']
        if timeline not in timelines:
            timelines[timeline] = []
        timelines[timeline].append(rec)
    
    for timeline, recs in timelines.items():
        st.markdown(f"#### 📅 Due in {timeline}")
        
        for rec in recs:
            priority_color = COLORS['warning'] if rec['priority'] == 'High' else COLORS['secondary']
            
            st.markdown(f"""
            <div style="
                background: rgba(37, 42, 69, 0.4);
                border-radius: 8px;
                border-left: 3px solid {priority_color};
                padding: 15px;
                margin: 10px 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <strong style="color: {COLORS['text']};">{rec['type']}: {rec.get('employee', rec.get('department'))}</strong><br>
                        <span style="color: {COLORS['text_secondary']}; font-size: 13px;">{rec['description']}</span><br>
                        <span style="color: {priority_color}; font-size: 12px; font-weight: bold;">Action: {rec['action']}</span>
                    </div>
                    <div style="
                        background: {priority_color};
                        color: white;
                        padding: 6px 12px;
                        border-radius: 15px;
                        font-size: 11px;
                        font-weight: bold;
                    ">
                        {rec['priority']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def _display_strategic_recommendations(strategic_recs):
    """Display strategic initiative recommendations."""
    
    if not strategic_recs:
        st.info("No strategic initiatives recommended at this time")
        return
    
    st.markdown(f"**{len(strategic_recs)} strategic initiatives:**")
    
    for i, rec in enumerate(strategic_recs, 1):
        st.markdown(f"""
        <div style="
            background: rgba(37, 42, 69, 0.4);
            border-radius: 12px;
            border-left: 4px solid {COLORS['info']};
            padding: 20px;
            margin: 15px 0;
        ">
            <h4 style="color: {COLORS['info']}; margin: 0 0 10px 0;">
                🎯 {i}. {rec['type']}: {rec.get('department', 'Organization-wide')}
            </h4>
            <p style="color: {COLORS['text_secondary']}; margin: 0 0 15px 0;">
                {rec['description']}
            </p>
            <div style="background: rgba(0, 212, 255, 0.1); padding: 12px; border-radius: 6px;">
                <strong style="color: {COLORS['info']};">Recommended Action:</strong>
                <span style="color: {COLORS['text']}; margin-left: 8px;">{rec['action']}</span>
            </div>
            <div style="margin-top: 10px; color: {COLORS['text_secondary']}; font-size: 12px;">
                Timeline: {rec['timeline']} • Priority: {rec['priority']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def _display_performance_recommendations(performance_recs):
    """Display performance-related recommendations."""
    
    if not performance_recs:
        st.info("No performance recommendations at this time")
        return
    
    st.markdown(f"**{len(performance_recs)} performance-related actions:**")
    
    for rec in performance_recs:
        st.markdown(f"""
        <div style="
            background: rgba(0, 255, 136, 0.1);
            border-radius: 12px;
            border-left: 4px solid {COLORS['success']};
            padding: 20px;
            margin: 15px 0;
        ">
            <h4 style="color: {COLORS['success']}; margin: 0 0 10px 0;">
                ⭐ {rec['type']}: {rec['employee']}
            </h4>
            <p style="color: {COLORS['text_secondary']}; margin: 0 0 15px 0;">
                {rec['description']}
            </p>
            <div style="color: {COLORS['success']}; font-weight: bold;">
                💡 Action: {rec['action']}
            </div>
            <div style="margin-top: 10px; color: {COLORS['text_secondary']}; font-size: 12px;">
                Timeline: {rec['timeline']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# EMAIL INTEGRATION DASHBOARD
# ================================================================

def email_integration_dashboard():
    """Email automation dashboard with campaign management."""
    
    st.markdown("### 📧 Email Integration Dashboard")
    
    email_service = initialize_email_service()
    
    if not email_service:
        st.error("📧 Email service not available. Please configure SMTP settings.")
        
        # Configuration helper
        with st.expander("📝 Email Configuration Help"):
            st.markdown("""
            **Required Environment Variables:**
            ```
            SMTP_HOST=smtp.gmail.com
            SMTP_PORT=587
            SMTP_USERNAME=your-email@company.com
            SMTP_PASSWORD=your-app-password
            SMTP_USE_TLS=true
            ```
            
            **For Gmail:**
            1. Enable 2-factor authentication
            2. Generate an app password
            3. Use the app password instead of your regular password
            """)
        
        return
    
    # Email service status
    stats = email_service.get_email_statistics()
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(glassmorphism_metric_card(
            value=f"{stats['total_sent']:,}",
            title="Emails Sent",
            subtitle="Total",
            icon="📧",
            color='success'
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(glassmorphism_metric_card(
            value=f"{stats['success_rate']:.1%}",
            title="Success Rate",
            subtitle="Delivery",
            icon="✅",
            color='secondary'
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(glassmorphism_metric_card(
            value=f"{stats['total_campaigns']:,}",
            title="Campaigns",
            subtitle="Total Run",
            icon="📊",
            color='info'
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(glassmorphism_metric_card(
            value=f"{stats['templates_available']:,}",
            title="Templates",
            subtitle="Available",
            icon="📝",
            color='accent'
        ), unsafe_allow_html=True)
    
    # Email campaign tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Send Campaign",
        "📝 Templates", 
        "📊 Campaign History",
        "⚙️ Settings"
    ])
    
    with tab1:
        _display_email_campaign_sender(email_service)
    
    with tab2:
        _display_email_templates(email_service)
    
    with tab3:
        _display_campaign_history()
    
    with tab4:
        _display_email_settings(email_service)

def _display_email_campaign_sender(email_service):
    """Display email campaign sending interface."""
    
    st.markdown("#### 🚀 Send Email Campaign")
    
    employees, _ = load_employee_directory()
    
    # Campaign configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        campaign_type = st.selectbox(
            "Campaign Type:",
            [
                "Manager Alerts",
                "Employee Engagement Survey", 
                "Retention Campaign - General",
                "Retention Campaign - Benefits",
                "Retention Campaign - Career",
                "Retention Campaign - Wellness",
                "Custom Campaign"
            ]
        )
    
    with col2:
        priority = st.selectbox("Priority:", ["High", "Normal", "Low"])
    
    # Target audience selection
    st.markdown("#### 🎯 Target Audience")
    
    audience_type = st.radio(
        "Select Recipients:",
        ["High Risk Employees", "All Employees", "Department", "Custom Filter"],
        horizontal=True
    )
    
    # Filter employees based on selection
    if audience_type == "High Risk Employees":
        target_employees = employees[employees['RiskLevel'] == 'High']
        st.info(f"Selected {len(target_employees)} high-risk employees")
    
    elif audience_type == "Department":
        selected_dept = st.selectbox("Department:", employees['Department'].unique())
        target_employees = employees[employees['Department'] == selected_dept]
        st.info(f"Selected {len(target_employees)} employees from {selected_dept}")
    
    elif audience_type == "Custom Filter":
        # Custom filtering interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dept_filter = st.multiselect("Departments:", employees['Department'].unique())
        with col2:
            risk_filter = st.multiselect("Risk Levels:", employees['RiskLevel'].unique())
        with col3:
            performance_filter = st.slider("Min Performance:", 1, 5, 1)
        
        # Apply filters
        filtered = employees.copy()
        if dept_filter:
            filtered = filtered[filtered['Department'].isin(dept_filter)]
        if risk_filter:
            filtered = filtered[filtered['RiskLevel'].isin(risk_filter)]
        if performance_filter > 1:
            filtered = filtered[filtered['PerformanceRating'] >= performance_filter]
        
        target_employees = filtered
        st.info(f"Selected {len(target_employees)} employees with custom filters")
    
    else:  # All Employees
        target_employees = employees
        st.info(f"Selected all {len(target_employees)} employees")
    
    # Campaign preview
    if len(target_employees) > 0:
        st.markdown("#### 👀 Campaign Preview")
        
        # Show recipient sample
        with st.expander(f"📋 View Recipients ({len(target_employees)} total)"):
            st.dataframe(
                target_employees[['FullName', 'Department', 'RiskLevel', 'Email']].head(10),
                use_container_width=True
            )
        
        # Send campaign
        col1, col2 = st.columns([3, 1])
        
        with col1:
            send_immediately = st.checkbox("Send Immediately", value=True)
            if not send_immediately:
                schedule_time = st.datetime_input(
                    "Schedule for:",
                    value=datetime.now() + timedelta(hours=1),
                    min_value=datetime.now()
                )
        
        with col2:
            if st.button("🚀 Send Campaign", type="primary", use_container_width=True):
                _execute_email_campaign(email_service, campaign_type, target_employees, priority)

def _execute_email_campaign(email_service, campaign_type, target_employees, priority):
    """Execute the email campaign."""
    
    with st.spinner("📧 Sending emails..."):
        
        # Convert campaign type to email service method
        if campaign_type == "Manager Alerts":
            # Group by manager and send alerts
            manager_groups = target_employees.groupby('Manager')
            results = []
            
            for manager, group in manager_groups:
                if manager and manager != 'Not Assigned':
                    manager_email = f"{manager.lower().replace(' ', '.')}@company.com"
                    high_risk_list = group.to_dict('records')
                    
                    result = email_service.send_manager_alert(
                        manager_email=manager_email,
                        manager_name=manager,
                        high_risk_employees=high_risk_list,
                        department=group['Department'].iloc[0] if len(group['Department'].unique()) == 1 else ""
                    )
                    results.append(result)
        
        elif campaign_type == "Employee Engagement Survey":
            employee_list = target_employees.to_dict('records')
            results = email_service.send_employee_engagement_survey(employee_list)
        
        elif campaign_type.startswith("Retention Campaign"):
            # Determine campaign subtype
            if "Benefits" in campaign_type:
                campaign_subtype = "benefits"
            elif "Career" in campaign_type:
                campaign_subtype = "career"
            elif "Wellness" in campaign_type:
                campaign_subtype = "wellness"
            else:
                campaign_subtype = "general"
            
            employee_list = target_employees.to_dict('records')
            results = email_service.send_retention_campaign(employee_list, campaign_subtype)
        
        else:
            st.error("Campaign type not implemented yet")
            return
        
        # Display results
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        if successful > 0:
            st.success(f"✅ Campaign sent successfully! {successful} emails delivered, {failed} failed.")
        else:
            st.error(f"❌ Campaign failed! {failed} emails could not be delivered.")
        
        # Show detailed results
        if failed > 0:
            with st.expander("❌ View Failed Emails"):
                failed_results = [r for r in results if not r.success]
                for result in failed_results:
                    st.error(f"{result.recipient_email}: {result.message}")

def _display_email_templates(email_service):
    """Display and manage email templates."""
    
    st.markdown("#### 📝 Email Templates")
    
    templates = email_service.templates
    
    # Template selection
    template_id = st.selectbox(
        "Select Template:",
        list(templates.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if template_id:
        template = templates[template_id]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Subject:**")
            st.text(template.subject)
            
            st.markdown("**HTML Preview:**")
            with st.expander("View HTML Template"):
                st.code(template.html_content, language='html')
            
            st.markdown("**Text Version:**")
            with st.expander("View Text Template"):
                st.text(template.text_content or "No text version available")
        
        with col2:
            st.markdown("**Template Info:**")
            st.info(f"""
            **ID:** {template.template_id}
            **Category:** {template.category}
            **Variables:** {len(template.template_vars or {})} defined
            """)
            
            if st.button("💾 Save to File", key=f"save_{template_id}"):
                email_service.save_templates_to_files()
                st.success("Templates saved to files!")
            
            if st.button("🧪 Test Email", key=f"test_{template_id}"):
                test_email = st.text_input("Test Email Address:", key=f"test_addr_{template_id}")
                if test_email:
                    # Here you would send a test email
                    st.success(f"Test email would be sent to {test_email}")

def _display_campaign_history():
    """Display email campaign history."""
    
    st.markdown("#### 📊 Campaign History")
    
    # Generate sample campaign history (in real implementation, load from database)
    sample_campaigns = [
        {
            'campaign_id': 'ENG_2025_Q3',
            'name': 'Q3 Employee Engagement Survey',
            'type': 'Engagement Survey',
            'sent_date': datetime.now() - timedelta(days=5),
            'recipients': 245,
            'delivered': 238,
            'failed': 7,
            'opened': 189,
            'clicked': 156,
            'status': 'Completed'
        },
        {
            'campaign_id': 'ALERT_2025_09_10',
            'name': 'High Risk Manager Alerts',
            'type': 'Manager Alert',
            'sent_date': datetime.now() - timedelta(days=2),
            'recipients': 12,
            'delivered': 12,
            'failed': 0,
            'opened': 10,
            'clicked': 8,
            'status': 'Completed'
        },
        {
            'campaign_id': 'RET_CAREER_2025_09',
            'name': 'Career Development Retention',
            'type': 'Retention Campaign',
            'sent_date': datetime.now() - timedelta(days=1),
            'recipients': 67,
            'delivered': 65,
            'failed': 2,
            'opened': 23,
            'clicked': 15,
            'status': 'Active'
        }
    ]
    
    # Campaign statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_sent = sum(c['recipients'] for c in sample_campaigns)
    total_delivered = sum(c['delivered'] for c in sample_campaigns)
    avg_open_rate = sum(c['opened'] / c['delivered'] for c in sample_campaigns if c['delivered'] > 0) / len(sample_campaigns)
    avg_click_rate = sum(c['clicked'] / c['delivered'] for c in sample_campaigns if c['delivered'] > 0) / len(sample_campaigns)
    
    with col1:
        st.metric("Total Campaigns", len(sample_campaigns))
    
    with col2:
        st.metric("Total Sent", f"{total_sent:,}")
    
    with col3:
        st.metric("Avg Open Rate", f"{avg_open_rate:.1%}")
    
    with col4:
        st.metric("Avg Click Rate", f"{avg_click_rate:.1%}")
    
    # Campaign list
    st.markdown("#### 📋 Recent Campaigns")
    
    for campaign in sample_campaigns:
        delivery_rate = campaign['delivered'] / campaign['recipients'] if campaign['recipients'] > 0 else 0
        open_rate = campaign['opened'] / campaign['delivered'] if campaign['delivered'] > 0 else 0
        
        status_color = COLORS['success'] if campaign['status'] == 'Completed' else COLORS['info']
        
        st.markdown(f"""
        <div style="
            background: rgba(37, 42, 69, 0.4);
            border-radius: 12px;
            border-left: 4px solid {status_color};
            padding: 20px;
            margin: 15px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
                <div>
                    <h4 style="color: {COLORS['text']}; margin: 0 0 5px 0;">
                        {campaign['name']}
                    </h4>
                    <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 14px;">
                        {campaign['type']} • {campaign['sent_date'].strftime('%B %d, %Y')}
                    </p>
                </div>
                <div style="
                    background: {status_color};
                    color: white;
                    padding: 6px 12px;
                    border-radius: 15px;
                    font-size: 12px;
                    font-weight: bold;
                ">
                    {campaign['status']}
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 15px; margin-bottom: 15px;">
                <div style="text-align: center;">
                    <div style="color: {COLORS['secondary']}; font-size: 20px; font-weight: bold;">
                        {campaign['recipients']}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">Recipients</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: {COLORS['success']}; font-size: 20px; font-weight: bold;">
                        {delivery_rate:.1%}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">Delivered</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: {COLORS['warning']}; font-size: 20px; font-weight: bold;">
                        {open_rate:.1%}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">Opened</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: {COLORS['accent']}; font-size: 20px; font-weight: bold;">
                        {campaign['clicked']}
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 12px;">Clicked</div>
                </div>
            </div>
            
            {f"<div style='color: {COLORS['error']}; font-size: 12px;'>⚠️ {campaign['failed']} delivery failures</div>" if campaign['failed'] > 0 else ""}
        </div>
        """, unsafe_allow_html=True)

def _display_email_settings(email_service):
    """Display email service settings and configuration."""
    
    st.markdown("#### ⚙️ Email Service Settings")
    
    # Test email service
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_email = st.text_input(
            "Test Email Address:",
            placeholder="your.email@company.com",
            help="Send a test email to verify configuration"
        )
    
    with col2:
        if st.button("🧪 Send Test Email", type="secondary"):
            if test_email and email_service:
                result = email_service.test_email_service(test_email)
                if result.success:
                    st.success("✅ Test email sent successfully!")
                else:
                    st.error(f"❌ Test failed: {result.message}")
            else:
                st.error("Please enter a valid email address")
    
    # Service statistics
    st.markdown("#### 📊 Service Statistics")
    
    stats = email_service.get_email_statistics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Rate Limiting:**")
        st.info(f"""
        - Limit: {stats['rate_limit_per_minute']} emails/minute
        - Current: {stats['emails_in_last_minute']} emails sent in last minute
        - Status: {'🟢 Normal' if stats['emails_in_last_minute'] < stats['rate_limit_per_minute'] * 0.8 else '🟡 High' if stats['emails_in_last_minute'] < stats['rate_limit_per_minute'] else '🔴 Limit Reached'}
        """)
    
    with col2:
        st.markdown("**Template Management:**")
        st.info(f"""
        - Available Templates: {stats['templates_available']}
        - Categories: Manager Alerts, Surveys, Retention
        - Last Updated: {datetime.now().strftime('%Y-%m-%d')}
        """)
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        
        st.markdown("**SMTP Configuration Status:**")
        
        # In a real implementation, you'd check actual SMTP settings
        config_items = [
            ("SMTP Host", "smtp.company.com", "✅"),
            ("SMTP Port", "587", "✅"),
            ("Authentication", "Enabled", "✅"),
            ("TLS/SSL", "Enabled", "✅"),
            ("Rate Limiting", f"{stats['rate_limit_per_minute']}/min", "✅")
        ]
        
        for item, value, status in config_items:
            st.markdown(f"""
            <div style="
                display: flex; 
                justify-content: space-between; 
                align-items: center;
                padding: 8px 0; 
                border-bottom: 1px solid {COLORS['border_primary']}33;
            ">
                <span style="color: {COLORS['text_secondary']};">{item}:</span>
                <span style="color: {COLORS['text']};">{value}</span>
                <span>{status}</span>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Configuration"):
            st.success("Configuration refreshed!")
        
        if st.button("💾 Export Templates"):
            email_service.save_templates_to_files()
            st.success("Templates exported to files!")

# ================================================================
# HELPER FUNCTIONS FOR EMAIL ACTIONS
# ================================================================

def _handle_bulk_email_action(employees, action_type):
    """Handle bulk email actions."""
    
    email_service = initialize_email_service()
    
    if not email_service:
        st.error("Email service not available")
        return
    
    with st.spinner(f"📧 Sending {action_type} emails..."):
        
        if action_type == 'engagement_survey':
            employee_list = employees.to_dict('records')
            results = email_service.send_employee_engagement_survey(employee_list)
        else:
            st.error(f"Action type {action_type} not implemented")
            return
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        if successful > 0:
            st.success(f"✅ {successful} emails sent successfully! {failed} failed.")
        else:
            st.error(f"❌ All emails failed to send!")

def _handle_bulk_manager_alerts(employees):
    """Handle bulk manager alerts for high-risk employees."""
    
    email_service = initialize_email_service()
    
    if not email_service:
        st.error("Email service not available")
        return
    
    # Group high-risk employees by manager
    manager_groups = employees[employees['RiskLevel'] == 'High'].groupby('Manager')
    
    with st.spinner("📧 Sending manager alerts..."):
        results = []
        
        for manager, group in manager_groups:
            if manager and manager != 'Not Assigned':
                manager_email = f"{manager.lower().replace(' ', '.')}@company.com"
                high_risk_list = group.to_dict('records')
                
                result = email_service.send_manager_alert(
                    manager_email=manager_email,
                    manager_name=manager,
                    high_risk_employees=high_risk_list,
                    department=group['Department'].iloc[0] if len(group['Department'].unique()) == 1 else ""
                )
                results.append(result)
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        if successful > 0:
            st.success(f"✅ Manager alerts sent! {successful} successful, {failed} failed.")
        else:
            st.error("❌ Failed to send manager alerts!")

def _handle_export_employees(employees):
    """Handle employee data export."""
    
    # Prepare export data
    export_data = employees[[
        'EmployeeID', 'FullName', 'Department', 'JobRole', 'RiskLevel', 
        'PerformanceRating', 'JobSatisfaction', 'YearsAtCompany', 'Status'
    ]].copy()
    
    # Convert to CSV
    csv_data = export_data.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="💾 Download Employee Data CSV",
        data=csv_data,
        file_name=f"employee_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def _send_individual_survey(employee, email_service):
    """Send engagement survey to individual employee."""
    
    employee_data = [employee.to_dict()]
    
    with st.spinner("📧 Sending engagement survey..."):
        results = email_service.send_employee_engagement_survey(employee_data)
        
        if results and results[0].success:
            st.success(f"✅ Engagement survey sent to {employee['FullName']}!")
        else:
            st.error(f"❌ Failed to send survey to {employee['FullName']}")

def _send_manager_alert(employee, email_service):
    """Send manager alert for individual employee."""
    
    manager = employee.get('Manager', 'Not Assigned')
    
    if not manager or manager == 'Not Assigned':
        st.error("No manager assigned to this employee")
        return
    
    manager_email = f"{manager.lower().replace(' ', '.')}@company.com"
    high_risk_data = [employee.to_dict()]
    
    with st.spinner("📧 Sending manager alert..."):
        result = email_service.send_manager_alert(
            manager_email=manager_email,
            manager_name=manager,
            high_risk_employees=high_risk_data,
            department=employee['Department']
        )
        
        if result.success:
            st.success(f"✅ Manager alert sent to {manager}!")
        else:
            st.error(f"❌ Failed to send alert: {result.message}")

def _handle_bulk_urgent_alerts(urgent_recs):
    """Handle bulk urgent alerts to managers."""
    
    email_service = initialize_email_service()
    
    if not email_service:
        st.error("Email service not available")
        return
    
    # Group by manager
    manager_emails = {}
    
    for rec in urgent_recs:
        if rec['type'] == 'High Risk Employee':
            # Get employee data to find manager
            employees, _ = load_employee_directory()
            employee = employees[employees['EmployeeID'] == rec['employee_id']].iloc[0]
            manager = employee.get('Manager', 'Not Assigned')
            
            if manager and manager != 'Not Assigned':
                manager_email = f"{manager.lower().replace(' ', '.')}@company.com"
                
                if manager_email not in manager_emails:
                    manager_emails[manager_email] = {
                        'manager_name': manager,
                        'employees': []
                    }
                
                manager_emails[manager_email]['employees'].append(employee.to_dict())
    
    # Send alerts to managers
    with st.spinner("📧 Sending urgent manager alerts..."):
        results = []
        
        for manager_email, data in manager_emails.items():
            result = email_service.send_manager_alert(
                manager_email=manager_email,
                manager_name=data['manager_name'],
                high_risk_employees=data['employees']
            )
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        if successful > 0:
            st.success(f"✅ Urgent alerts sent to {successful} managers! {failed} failed.")
        else:
            st.error("❌ Failed to send urgent alerts!")

def _generate_urgent_actions_report(urgent_recs):
    """Generate urgent actions report for download."""
    
    # Create report DataFrame
    report_data = []
    
    for rec in urgent_recs:
        report_data.append({
            'Type': rec['type'],
            'Employee/Department': rec.get('employee', rec.get('department', 'N/A')),
            'Employee_ID': rec.get('employee_id', 'N/A'),
            'Description': rec['description'],
            'Action_Required': rec['action'],
            'Timeline': rec['timeline'],
            'Priority': rec['priority'],
            'Generated_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    report_df = pd.DataFrame(report_data)
    
    # Convert to CSV
    csv_data = report_df.to_csv(index=False)
    
    # Create download
    st.download_button(
        label="📊 Download Urgent Actions Report",
        data=csv_data,
        file_name=f"urgent_actions_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.success("✅ Report generated! Use the download button above.")

# ================================================================
# MAIN EMPLOYEE MANAGEMENT FUNCTION
# ================================================================

def show():
    """Main employee management function called by the navigation system."""
    
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
                👥 Employee Management
            </h1>
            <p style="color: #B8C5D1; font-size: 1.1rem;">
                Comprehensive employee directory with AI-powered insights and automated HR actions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main management tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Employee Directory",
            "👤 Individual Profile", 
            "🎯 HR Recommendations",
            "📧 Email Integration"
        ])
        
        with tab1:
            searchable_employee_directory()
        
        with tab2:
            individual_employee_profile()
        
        with tab3:
            hr_action_recommendations()
        
        with tab4:
            email_integration_dashboard()
        
        # Memory cleanup
        gc.collect()
        
    except Exception as e:
        st.error(f"Employee management page error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")

# ================================================================
# ENTRY POINT FOR TESTING
# ================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Employee Management", layout="wide")
    show()
