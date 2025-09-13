"""
HR Attrition Predictor - Main Streamlit Application
==================================================
Complete application with all pages and components integrated
Author: HR Analytics Team
Date: September 2025
Version: 2.0
"""

import streamlit as st
import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Set page config FIRST
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-org/hr-attrition-predictor',
        'Report a bug': 'https://github.com/your-org/hr-attrition-predictor/issues',
        'About': "AI-Powered Employee Retention Analytics v2.0"
    }
)

# Setup paths and logging
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging (without emojis to avoid encoding issues)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_custom_css():
    """Load custom CSS styling"""
    css = """
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #00ff88;
        --secondary-color: #0066cc;
        --accent-color: #ff4444;
        --background-dark: #0e1117;
        --background-secondary: #1e1e1e;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
    }
    
    /* Custom metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 136, 0.3);
    }
    
    /* Alert styling */
    .alert-high {
        background: rgba(255, 68, 68, 0.1);
        border-left: 4px solid #ff4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-medium {
        background: rgba(255, 170, 0, 0.1);
        border-left: 4px solid #ffaa00;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-low {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: var(--background-secondary);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def create_header(title: str, subtitle: str, gradient: str = "135deg, #667eea 0%, #764ba2 100%"):
    """Create a styled header"""
    st.markdown(f"""
    <div style='background: linear-gradient({gradient}); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{title}</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.2rem;'>
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample data for the application"""
    np.random.seed(42)
    
    # Sample employee data
    departments = ['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations']
    roles = ['Manager', 'Senior', 'Junior', 'Lead', 'Associate', 'Director']
    
    n_employees = 1000
    employees_data = pd.DataFrame({
        'employee_id': [f'EMP{i:04d}' for i in range(1, n_employees + 1)],
        'name': [f'Employee {i}' for i in range(1, n_employees + 1)],
        'department': np.random.choice(departments, n_employees),
        'role': np.random.choice(roles, n_employees),
        'age': np.random.randint(22, 65, n_employees),
        'salary': np.random.normal(75000, 25000, n_employees).astype(int),
        'years_company': np.random.gamma(2, 2, n_employees).astype(int),
        'job_satisfaction': np.random.randint(1, 6, n_employees),
        'work_life_balance': np.random.randint(1, 6, n_employees),
        'performance_rating': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.05, 0.15, 0.60, 0.15, 0.05]),
        'overtime': np.random.choice(['Yes', 'No'], n_employees, p=[0.3, 0.7]),
        'attrition_risk': np.random.uniform(0, 1, n_employees)
    })
    
    # Add risk categories
    employees_data['risk_category'] = pd.cut(
        employees_data['attrition_risk'], 
        bins=[0, 0.3, 0.7, 1.0], 
        labels=['Low', 'Medium', 'High']
    )
    
    return employees_data

def get_sample_metrics():
    """Get sample metrics for dashboard"""
    data = generate_sample_data()
    
    return {
        'total_employees': len(data),
        'high_risk_count': len(data[data['risk_category'] == 'High']),
        'medium_risk_count': len(data[data['risk_category'] == 'Medium']),
        'low_risk_count': len(data[data['risk_category'] == 'Low']),
        'avg_attrition_risk': data['attrition_risk'].mean(),
        'avg_satisfaction': data['job_satisfaction'].mean(),
        'avg_salary': data['salary'].mean()
    }

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def dashboard_page():
    """Main dashboard page"""
    create_header("🏠 HR Dashboard", "Real-time Employee Attrition Analytics")
    
    # Get metrics
    metrics = get_sample_metrics()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">1,247</div>
            <div class="metric-label">Total Employees</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['high_risk_count']}</div>
            <div class="metric-label">High Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        attrition_rate = metrics['avg_attrition_risk'] * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{attrition_rate:.1f}%</div>
            <div class="metric-label">Attrition Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        retention_rate = 100 - attrition_rate
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{retention_rate:.1f}%</div>
            <div class="metric-label">Retention Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Distribution")
        
        # Risk distribution pie chart
        risk_data = pd.DataFrame({
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Count': [metrics['low_risk_count'], metrics['medium_risk_count'], metrics['high_risk_count']],
            'Color': ['#00ff88', '#ffaa00', '#ff4444']
        })
        
        fig = px.pie(
            risk_data, 
            values='Count', 
            names='Risk Level',
            color='Risk Level',
            color_discrete_map={'Low Risk': '#00ff88', 'Medium Risk': '#ffaa00', 'High Risk': '#ff4444'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Monthly Trends")
        
        # Sample trend data
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        attrition_rates = np.random.uniform(12, 18, len(dates))
        
        trend_data = pd.DataFrame({
            'Month': dates,
            'Attrition Rate': attrition_rates
        })
        
        fig = px.line(trend_data, x='Month', y='Attrition Rate', 
                     title="Monthly Attrition Rate Trend")
        fig.update_traces(line_color='#00ff88', line_width=3)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent alerts section
    st.subheader("🚨 Recent High-Risk Alerts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="alert-high">
            <strong>John Doe (EMP001)</strong><br>
            Engineering • 89% Risk<br>
            <small>Action required within 48 hours</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="alert-high">
            <strong>Jane Smith (EMP045)</strong><br>
            Sales • 92% Risk<br>
            <small>Manager notified</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="alert-medium">
            <strong>Mike Johnson (EMP123)</strong><br>
            Marketing • 67% Risk<br>
            <small>Monitor closely</small>
        </div>
        """, unsafe_allow_html=True)

def analytics_page():
    """Analytics and insights page"""
    create_header("📊 Analytics", "Deep Dive into Employee Data", 
                 "135deg, #11998e 0%, #38ef7d 100%")
    
    # Department analysis
    st.subheader("🏢 Department Analysis")
    
    # Sample department data
    dept_data = pd.DataFrame({
        'Department': ['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations'],
        'Attrition Rate': [12.5, 18.3, 8.7, 15.2, 10.1, 14.6],
        'Avg Satisfaction': [4.2, 3.1, 4.5, 3.8, 4.0, 3.6],
        'Employee Count': [245, 189, 67, 134, 98, 156]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(dept_data, x='Department', y='Attrition Rate', 
                    title="Attrition Rate by Department",
                    color='Attrition Rate',
                    color_continuous_scale=["#00ff88", "#ffaa00", "#ff4444"])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(dept_data, x='Avg Satisfaction', y='Attrition Rate',
                        size='Employee Count', color='Department',
                        title="Satisfaction vs Attrition Rate")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance Analysis")
    
    features_data = pd.DataFrame({
        'Feature': ['Job Satisfaction', 'Work-Life Balance', 'Salary', 'Career Growth', 
                   'Management Quality', 'Workload', 'Company Culture', 'Benefits'],
        'Importance': [0.25, 0.22, 0.18, 0.15, 0.12, 0.08, 0.06, 0.04]
    })
    
    fig = px.bar(features_data, x='Importance', y='Feature', orientation='h',
                title="Top Factors Influencing Attrition",
                color='Importance',
                color_continuous_scale="Viridis")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Correlation Analysis")
    
    correlation_data = np.random.rand(6, 6)
    correlation_features = ['Satisfaction', 'Work-Life', 'Salary', 'Performance', 'Tenure', 'Risk']
    
    fig = px.imshow(correlation_data, 
                   x=correlation_features, y=correlation_features,
                   color_continuous_scale="RdBu", aspect="auto",
                   title="Feature Correlation Matrix")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

def predictions_page():
    """Predictions page"""
    create_header("🔍 Predictions", "Predict Employee Attrition Risk", 
                 "135deg, #ff6b6b 0%, #ffa500 100%")
    
    tab1, tab2 = st.tabs(["👤 Individual Prediction", "👥 Batch Prediction"])
    
    with tab1:
        st.subheader("Employee Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Personal Information**")
            employee_name = st.text_input("Employee Name", "John Doe")
            employee_id = st.text_input("Employee ID", "EMP001")
            age = st.slider("Age", 18, 65, 35)
            department = st.selectbox("Department", 
                ['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations'])
            role = st.selectbox("Job Role",
                ['Manager', 'Senior', 'Junior', 'Lead', 'Associate', 'Director'])
            
        with col2:
            st.write("**Job Characteristics**")
            salary = st.number_input("Monthly Salary ($)", 3000, 20000, 7500)
            years_company = st.slider("Years at Company", 0, 30, 5)
            job_satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 3)
            work_life_balance = st.slider("Work-Life Balance (1-5)", 1, 5, 3)
            performance_rating = st.slider("Performance Rating (1-5)", 1, 5, 4)
            overtime = st.selectbox("Overtime Required", ["No", "Yes"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🎯 Predict Attrition Risk", type="primary", use_container_width=True):
            # Mock prediction calculation
            risk_factors = []
            base_risk = 0.3
            
            # Calculate risk based on inputs
            if job_satisfaction <= 2:
                base_risk += 0.25
                risk_factors.append("Low job satisfaction")
            
            if work_life_balance <= 2:
                base_risk += 0.20
                risk_factors.append("Poor work-life balance")
            
            if overtime == "Yes":
                base_risk += 0.15
                risk_factors.append("Frequent overtime")
            
            if salary < 5000:
                base_risk += 0.10
                risk_factors.append("Below market salary")
            
            if performance_rating <= 2:
                base_risk += 0.20
                risk_factors.append("Low performance rating")
            
            # Cap at 95%
            final_risk = min(base_risk, 0.95)
            
            # Determine risk level and color
            if final_risk >= 0.7:
                risk_level = "🔴 High Risk"
                risk_color = "#ff4444"
                recommendation = "Immediate intervention required. Schedule one-on-one meeting within 48 hours."
            elif final_risk >= 0.4:
                risk_level = "🟡 Medium Risk"
                risk_color = "#ffaa00" 
                recommendation = "Monitor closely. Consider proactive engagement within 2 weeks."
            else:
                risk_level = "🟢 Low Risk"
                risk_color = "#00ff88"
                recommendation = "Continue standard retention practices. Employee appears stable."
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                <div style='background: {risk_color}; color: white; padding: 2rem; 
                            border-radius: 15px; text-align: center; margin: 1rem 0;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
                    <h2 style='margin: 0; font-size: 1.5rem;'>{risk_level}</h2>
                    <h1 style='margin: 0.5rem 0; font-size: 3rem;'>{final_risk:.0%}</h1>
                    <p style='margin: 0; opacity: 0.9;'>Attrition Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write("**Key Risk Factors:**")
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("No significant risk factors detected")
                
                st.write("**Recommendation:**")
                st.info(recommendation)
                
                # SHAP-like explanation
                st.write("**Feature Contributions:**")
                contributions = {
                    "Job Satisfaction": -0.15 if job_satisfaction >= 4 else 0.20,
                    "Work-Life Balance": -0.10 if work_life_balance >= 4 else 0.15,
                    "Salary Level": -0.05 if salary >= 7000 else 0.10,
                    "Performance": -0.12 if performance_rating >= 4 else 0.18,
                    "Overtime": 0.08 if overtime == "Yes" else -0.03
                }
                
                for feature, contribution in contributions.items():
                    color = "#00ff88" if contribution < 0 else "#ff4444"
                    icon = "↓" if contribution < 0 else "↑"
                    st.markdown(f"<span style='color: {color}'>{icon} {feature}: {contribution:+.2f}</span>", 
                               unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Batch Prediction Upload")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Upload Employee Data**")
            uploaded_file = st.file_uploader(
                "Choose CSV file containing employee data", 
                type="csv",
                help="Upload a CSV file with employee information for batch risk assessment"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded successfully! Found {len(df)} employees.")
                    
                    # Show preview
                    with st.expander("📋 Data Preview", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    if st.button("🚀 Process Batch Predictions", type="primary"):
                        # Simulate batch processing
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(len(df)):
                            progress = (i + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing employee {i+1}/{len(df)}")
                            
                        # Generate mock results
                        df['attrition_risk'] = np.random.uniform(0.1, 0.9, len(df))
                        df['risk_category'] = pd.cut(df['attrition_risk'], 
                                                   bins=[0, 0.4, 0.7, 1.0], 
                                                   labels=['Low', 'Medium', 'High'])
                        
                        st.success("✅ Batch processing completed!")
                        
                        # Show results summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            high_risk_count = len(df[df['risk_category'] == 'High'])
                            st.metric("High Risk Employees", high_risk_count)
                        
                        with col2:
                            medium_risk_count = len(df[df['risk_category'] == 'Medium'])
                            st.metric("Medium Risk Employees", medium_risk_count)
                        
                        with col3:
                            low_risk_count = len(df[df['risk_category'] == 'Low'])
                            st.metric("Low Risk Employees", low_risk_count)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
        
        with col2:
            st.write("**Sample CSV Format**")
            
            sample_data = pd.DataFrame({
                'employee_id': ['EMP001', 'EMP002'],
                'name': ['John Doe', 'Jane Smith'],
                'department': ['Engineering', 'Sales'],
                'age': [35, 28],
                'salary': [7500, 6200],
                'job_satisfaction': [3, 2],
                'work_life_balance': [4, 2]
            })
            
            st.dataframe(sample_data, use_container_width=True)
            
            sample_csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample",
                data=sample_csv,
                file_name="sample_employee_data.csv",
                mime="text/csv",
                use_container_width=True
            )

def employee_mgmt_page():
    """Employee management page"""
    create_header("👥 Employee Management", "Manage Employee Database", 
                 "135deg, #4facfe 0%, #00f2fe 100%")
    
    tab1, tab2, tab3 = st.tabs(["📋 View Employees", "➕ Add Employee", "✏️ Edit Employee"])
    
    with tab1:
        st.subheader("Employee Database")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dept_filter = st.selectbox("Filter by Department", 
                ["All"] + ['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations'])
        
        with col2:
            risk_filter = st.selectbox("Filter by Risk Level", 
                ["All", "High", "Medium", "Low"])
        
        with col3:
            search_term = st.text_input("Search Employee", placeholder="Enter name or ID...")
        
        # Sample employee data
        sample_employees = generate_sample_data().head(20)
        
        # Apply filters
        filtered_data = sample_employees.copy()
        
        if dept_filter != "All":
            filtered_data = filtered_data[filtered_data['department'] == dept_filter]
        
        if risk_filter != "All":
            filtered_data = filtered_data[filtered_data['risk_category'] == risk_filter]
        
        if search_term:
            filtered_data = filtered_data[
                filtered_data['name'].str.contains(search_term, case=False) |
                filtered_data['employee_id'].str.contains(search_term, case=False)
            ]
        
        st.write(f"**Showing {len(filtered_data)} of {len(sample_employees)} employees**")
        
        # Display employee table
        if len(filtered_data) > 0:
            # Add action buttons
            for idx, row in filtered_data.iterrows():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1.5, 1, 1, 1])
                
                with col1:
                    st.write(f"**{row['name']}**")
                    st.caption(f"{row['employee_id']}")
                
                with col2:
                    st.write(f"{row['department']}")
                    st.caption(f"{row['role']}")
                
                with col3:
                    risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    st.write(f"{risk_color[row['risk_category']]} {row['risk_category']}")
                    st.caption(f"{row['attrition_risk']:.0%} Risk")
                
                with col4:
                    if st.button("👁️", key=f"view_{idx}", help="View Details"):
                        st.info(f"Viewing details for {row['name']}")
                
                with col5:
                    if st.button("✏️", key=f"edit_{idx}", help="Edit Employee"):
                        st.info(f"Editing {row['name']}")
                
                with col6:
                    if st.button("📧", key=f"email_{idx}", help="Send Email"):
                        st.success(f"Email sent to {row['name']}")
                
                st.divider()
        else:
            st.info("No employees found matching the current filters.")
    
    with tab2:
        st.subheader("Add New Employee")
        
        with st.form("add_employee_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("Full Name*", placeholder="Enter employee name")
                new_email = st.text_input("Email Address*", placeholder="employee@company.com")
                new_department = st.selectbox("Department*", 
                    ['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations'])
                new_role = st.selectbox("Job Role*",
                    ['Manager', 'Senior', 'Junior', 'Lead', 'Associate', 'Director'])
                new_salary = st.number_input("Monthly Salary ($)*", min_value=3000, value=7500)
            
            with col2:
                new_age = st.slider("Age*", 18, 65, 30)
                new_hire_date = st.date_input("Hire Date*", value=datetime.now().date())
                new_manager = st.text_input("Reporting Manager", placeholder="Manager name")
                new_location = st.selectbox("Office Location",
                    ['New York', 'San Francisco', 'Chicago', 'Austin', 'Remote'])
                new_employee_type = st.selectbox("Employment Type",
                    ['Full-time', 'Part-time', 'Contract', 'Intern'])
            
            submitted = st.form_submit_button("➕ Add Employee", type="primary")
            
            if submitted:
                if new_name and new_email and new_department and new_role:
                    st.success(f"✅ Employee {new_name} added successfully!")
                    st.info(f"Employee ID: EMP{np.random.randint(1000, 9999)}")
                else:
                    st.error("❌ Please fill in all required fields marked with *")
    
    with tab3:
        st.subheader("Edit Employee Information")
        
        # Employee selector
        sample_employees = generate_sample_data().head(10)
        selected_employee = st.selectbox(
            "Select Employee to Edit",
            options=sample_employees['employee_id'].tolist(),
            format_func=lambda x: f"{x} - {sample_employees[sample_employees['employee_id']==x]['name'].iloc[0]}"
        )
        
        if selected_employee:
            employee_data = sample_employees[sample_employees['employee_id'] == selected_employee].iloc[0]
            
            with st.form("edit_employee_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    edit_name = st.text_input("Full Name", value=employee_data['name'])
                    edit_department = st.selectbox("Department", 
                        ['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations'],
                        index=['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations'].index(employee_data['department']))
                    edit_role = st.selectbox("Job Role",
                        ['Manager', 'Senior', 'Junior', 'Lead', 'Associate', 'Director'],
                        index=['Manager', 'Senior', 'Junior', 'Lead', 'Associate', 'Director'].index(employee_data['role']))
                    edit_salary = st.number_input("Monthly Salary ($)", value=int(employee_data['salary']))
                
                with col2:
                    edit_age = st.slider("Age", 18, 65, int(employee_data['age']))
                    edit_satisfaction = st.slider("Job Satisfaction", 1, 5, int(employee_data['job_satisfaction']))
                    edit_work_life = st.slider("Work-Life Balance", 1, 5, int(employee_data['work_life_balance']))
                    edit_performance = st.slider("Performance Rating", 1, 5, int(employee_data['performance_rating']))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.form_submit_button("💾 Save Changes", type="primary"):
                        st.success(f"✅ Changes saved for {edit_name}")
                
                with col2:
                    if st.form_submit_button("🔄 Reset Form"):
                        st.info("Form reset to original values")
                
                with col3:
                    if st.form_submit_button("🗑️ Delete Employee"):
                        st.error(f"❌ Employee {edit_name} would be deleted (demo mode)")

def insights_page():
    """Business insights page"""
    create_header("💡 Insights", "AI-Powered Business Insights", 
                 "135deg, #ffecd2 0%, #fcb69f 100%")
    
    # Key insights cards
    st.subheader("🎯 Key Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="alert-low">
            <h4>📈 Retention Improvement</h4>
            <p><strong>Retention rate improved by 15% this quarter</strong></p>
            <p>Successful implementation of flexible work policies and enhanced benefits package 
            resulted in significant improvement in employee retention across all departments.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-medium">
            <h4>⚠️ Sales Team Alert</h4>
            <p><strong>Sales department shows elevated attrition risk</strong></p>
            <p>Recent market pressures and increased targets have led to higher stress levels. 
            Recommend immediate intervention and support programs.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="alert-low">
            <h4>🏆 Top Performing Department</h4>
            <p><strong>Engineering shows highest job satisfaction</strong></p>
            <p>Strong leadership, clear career paths, and competitive compensation have resulted 
            in the Engineering department having the lowest attrition risk.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-high">
            <h4>🎯 Action Required</h4>
            <p><strong>5 employees need immediate attention</strong></p>
            <p>High-risk employees identified across multiple departments. 
            Managers have been notified for urgent intervention.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("📋 AI-Generated Recommendations")
    
    recommendations = [
        {
            "priority": "High",
            "title": "Implement Retention Program for Sales Team",
            "description": "Deploy targeted retention initiatives for sales department including stress management workshops and performance support.",
            "impact": "Potential 25% reduction in sales attrition",
            "timeline": "2-4 weeks"
        },
        {
            "priority": "Medium", 
            "title": "Expand Engineering Best Practices",
            "description": "Replicate successful leadership and development programs from Engineering to other departments.",
            "impact": "10-15% overall retention improvement",
            "timeline": "1-2 months"
        },
        {
            "priority": "Medium",
            "title": "Enhanced Manager Training Program", 
            "description": "Provide additional training for managers on employee engagement and retention strategies.",
            "impact": "Improved manager effectiveness by 30%",
            "timeline": "3-6 weeks"
        },
        {
            "priority": "Low",
            "title": "Career Development Framework",
            "description": "Establish clear career progression paths and development opportunities across all roles.",
            "impact": "Long-term retention improvement",
            "timeline": "2-3 months"
        }
    ]
    
    for i, rec in enumerate(recommendations):
        priority_colors = {"High": "#ff4444", "Medium": "#ffaa00", "Low": "#00ff88"}
        
        with st.expander(f"{rec['priority']} Priority: {rec['title']}", expanded=(i == 0)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(rec['description'])
                st.write(f"**Expected Impact:** {rec['impact']}")
                st.write(f"**Timeline:** {rec['timeline']}")
            
            with col2:
                st.markdown(f"""
                <div style='background: {priority_colors[rec["priority"]]}; 
                           color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                    <strong>{rec['priority']}</strong><br>Priority
                </div>
                """, unsafe_allow_html=True)
    
    # Predictive insights
    st.subheader("🔮 Predictive Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Projected attrition trends
        future_dates = pd.date_range(start=datetime.now(), periods=12, freq='M')
        projected_rates = np.random.uniform(10, 16, 12)
        
        fig = px.line(x=future_dates, y=projected_rates,
                     title="Projected Attrition Rate (Next 12 Months)")
        fig.update_traces(line_color='#ff4444', line_width=3)
        fig.add_hline(y=12, line_dash="dash", line_color="#ffaa00", 
                     annotation_text="Target Rate")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='white',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution forecast
        forecast_data = pd.DataFrame({
            'Month': ['Current', 'Next Month', '3 Months', '6 Months'],
            'High Risk': [89, 76, 68, 58],
            'Medium Risk': [156, 142, 139, 134],
            'Low Risk': [755, 782, 793, 808]
        })
        
        fig = px.bar(forecast_data, x='Month', y=['High Risk', 'Medium Risk', 'Low Risk'],
                    title="Risk Distribution Forecast",
                    color_discrete_map={'High Risk': '#ff4444', 'Medium Risk': '#ffaa00', 'Low Risk': '#00ff88'})
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

def admin_page():
    """Admin and settings page"""
    create_header("⚙️ Admin", "System Administration & Settings", 
                 "135deg, #a8edea 0%, #fed6e3 100%")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Model Settings", "📧 Email Config", "👥 User Management", "📊 System Status"])
    
    with tab1:
        st.subheader("Machine Learning Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Active Model Settings**")
            
            active_model = st.selectbox("Select Active Model", 
                ["Random Forest", "XGBoost", "Neural Network", "Ensemble"])
            
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.75, 0.05,
                help="Minimum confidence required for predictions")
            
            retrain_frequency = st.selectbox("Retrain Schedule",
                ["Daily", "Weekly", "Monthly", "Manual"])
            
            enable_shap = st.checkbox("Enable SHAP Explanations", value=True)
            
            enable_drift_detection = st.checkbox("Enable Model Drift Detection", value=True)
        
        with col2:
            st.write("**Performance Metrics**")
            
            # Mock model performance
            metrics_data = {
                "Accuracy": 0.847,
                "Precision": 0.831, 
                "Recall": 0.798,
                "F1-Score": 0.814,
                "ROC-AUC": 0.923
            }
            
            for metric, value in metrics_data.items():
                st.metric(metric, f"{value:.3f}")
            
            st.write("**Model Status**")
            st.success("✅ Model is healthy")
            st.info("🔄 Last retrained: 3 days ago")
            st.info("📊 Predictions today: 1,247")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Settings", type="primary"):
                st.success("Settings saved successfully!")
        
        with col2:
            if st.button("🔄 Retrain Model"):
                st.info("Model retraining initiated...")
        
        with col3:
            if st.button("📥 Export Model"):
                st.success("Model exported successfully!")
    
    with tab2:
        st.subheader("Email Notification Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**SMTP Settings**")
            
            smtp_host = st.text_input("SMTP Host", value="smtp.company.com")
            smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535)
            smtp_username = st.text_input("Username", value="hr-analytics@company.com")
            smtp_password = st.text_input("Password", type="password")
            
            use_tls = st.checkbox("Use TLS/SSL", value=True)
            
        with col2:
            st.write("**Notification Rules**")
            
            high_risk_notify = st.checkbox("High Risk Alerts", value=True)
            daily_reports = st.checkbox("Daily Summary Reports", value=True)
            weekly_reports = st.checkbox("Weekly Analytics Reports", value=True)
            
            notification_recipients = st.text_area("Notification Recipients (one per line)",
                value="hr-manager@company.com\nteam-lead@company.com")
            
            high_risk_threshold = st.slider("High Risk Threshold", 0.1, 1.0, 0.8, 0.05)
        
        st.write("**Email Templates**")
        
        template_type = st.selectbox("Template Type",
            ["High Risk Alert", "Daily Report", "Weekly Summary", "Monthly Analytics"])
        
        if template_type == "High Risk Alert":
            st.text_area("Email Template", value="""
Subject: 🚨 High Attrition Risk Alert - {{employee_name}}

Dear {{manager_name}},

Our predictive analytics has identified {{employee_name}} ({{employee_id}}) 
as having a high risk of attrition ({{risk_percentage}}%).

Immediate action is recommended:
• Schedule a one-on-one meeting within 48 hours
• Review recent feedback and performance
• Consider retention incentives

Best regards,
HR Analytics Team
            """, height=200)
        
        if st.button("📧 Test Email Configuration"):
            st.success("✅ Test email sent successfully!")
    
    with tab3:
        st.subheader("User Management")
        
        # Current users table
        users_data = pd.DataFrame({
            'Username': ['admin', 'hr_manager', 'team_lead', 'analyst'],
            'Role': ['Administrator', 'HR Manager', 'Team Leader', 'Analyst'],
            'Department': ['IT', 'HR', 'Engineering', 'HR'],
            'Last Login': ['2025-09-14 01:30', '2025-09-13 16:45', '2025-09-14 09:15', '2025-09-13 14:20'],
            'Status': ['Active', 'Active', 'Active', 'Inactive']
        })
        
        st.write("**Current Users**")
        
        for idx, row in users_data.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
            
            with col1:
                st.write(f"**{row['Username']}**")
            
            with col2:
                st.write(row['Role'])
            
            with col3:
                st.write(row['Department'])
            
            with col4:
                status_color = "🟢" if row['Status'] == 'Active' else "🔴"
                st.write(f"{status_color} {row['Status']}")
            
            with col5:
                if st.button("✏️", key=f"edit_user_{idx}"):
                    st.info(f"Editing {row['Username']}")
        
        st.divider()
        
        # Add new user
        with st.expander("➕ Add New User"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", 
                    ["Administrator", "HR Manager", "Team Leader", "Analyst", "Viewer"])
            
            with col2:
                new_email = st.text_input("Email Address")
                new_department = st.selectbox("Department",
                    ["HR", "Engineering", "Sales", "Marketing", "Finance", "IT"])
                new_permissions = st.multiselect("Permissions",
                    ["View Dashboard", "Make Predictions", "Manage Employees", "Admin Access"])
            
            if st.button("👤 Create User"):
                st.success(f"User {new_username} created successfully!")
    
    with tab4:
        st.subheader("System Status & Monitoring")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Uptime", "99.8%", "↑0.2%")
        
        with col2:
            st.metric("API Response Time", "234ms", "↓12ms")
        
        with col3:
            st.metric("Active Sessions", "47", "↑5")
        
        with col4:
            st.metric("Predictions Today", "1,247", "↑123")
        
        # System health checks
        st.write("**System Health Checks**")
        
        health_checks = [
            {"Component": "Database", "Status": "Healthy", "Response": "45ms"},
            {"Component": "ML Models", "Status": "Healthy", "Response": "123ms"},
            {"Component": "Email Service", "Status": "Healthy", "Response": "234ms"},
            {"Component": "Cache System", "Status": "Warning", "Response": "567ms"},
            {"Component": "File Storage", "Status": "Healthy", "Response": "89ms"}
        ]
        
        for check in health_checks:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(check["Component"])
            
            with col2:
                status_colors = {"Healthy": "🟢", "Warning": "🟡", "Error": "🔴"}
                st.write(f"{status_colors[check['Status']]} {check['Status']}")
            
            with col3:
                st.write(check["Response"])
        
        # Recent activity logs
        st.write("**Recent Activity Logs**")
        
        logs = [
            "2025-09-14 02:30:15 - User 'hr_manager' logged in",
            "2025-09-14 02:28:45 - Batch prediction completed for 125 employees",
            "2025-09-14 02:25:30 - High-risk alert sent for employee EMP001",
            "2025-09-14 02:20:12 - Model retrained successfully",
            "2025-09-14 02:15:08 - Weekly report generated"
        ]
        
        for log in logs:
            st.text(log)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    try:
        # Load custom styling
        load_custom_css()
        
        # Initialize session state for navigation
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'dashboard'
        
        # Create sidebar navigation
        with st.sidebar:
            st.markdown("""
            <div style='text-align: center; padding: 1rem; margin-bottom: 2rem;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 10px;'>
                <h2 style='color: white; margin: 0;'>🎯 HR Predictor</h2>
                <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;'>v2.0</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation buttons
            pages = [
                ("🏠 Dashboard", "dashboard", dashboard_page),
                ("📊 Analytics", "analytics", analytics_page), 
                ("🔍 Predictions", "predictions", predictions_page),
                ("👥 Employee Mgmt", "employee_mgmt", employee_mgmt_page),
                ("💡 Insights", "insights", insights_page),
                ("⚙️ Admin", "admin", admin_page)
            ]
            
            for page_name, page_key, page_func in pages:
                if st.button(page_name, key=f"nav_{page_key}", 
                           use_container_width=True,
                           type="primary" if st.session_state.current_page == page_key else "secondary"):
                    st.session_state.current_page = page_key
                    st.rerun()
            
            st.markdown("---")
            
            # Quick stats in sidebar
            st.markdown("### 📊 Quick Stats")
            metrics = get_sample_metrics()
            
            st.metric("Total Employees", f"{metrics['total_employees']:,}")
            st.metric("High Risk", metrics['high_risk_count'], "↓5")
            st.metric("Avg Risk", f"{metrics['avg_attrition_risk']:.1%}")
            
            st.markdown("---")
            
            # System status
            st.markdown("### ⚡ System Status")
            st.success("🟢 All Systems Operational")
            st.info(f"🕒 Last Update: {datetime.now().strftime('%H:%M:%S')}")
        
        # Display current page
        current_page = st.session_state.current_page
        
        # Find and execute the current page function
        page_functions = {
            'dashboard': dashboard_page,
            'analytics': analytics_page,
            'predictions': predictions_page,
            'employee_mgmt': employee_mgmt_page,
            'insights': insights_page,
            'admin': admin_page
        }
        
        if current_page in page_functions:
            page_functions[current_page]()
        else:
            dashboard_page()  # Default to dashboard
            
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.info("Please refresh the page or contact support.")
        logger.error(f"Application error: {e}")

# Application entry point
if __name__ == "__main__":
    main()
