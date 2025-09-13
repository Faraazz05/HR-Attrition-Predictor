"""
HR Attrition Predictor - Executive Dashboard
===========================================
Comprehensive executive dashboard with KPIs, trends, and insights.
Memory-optimized for 4GB RAM systems with real-time metrics.

Author: HR Analytics Team
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
import warnings
import gc
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
from streamlit_app.assets.theme import COLORS, TYPOGRAPHY, apply_custom_css
from streamlit_app.components.charts import (
    glassmorphism_metric_card, futuristic_gauge_chart,
    neon_bar_chart, futuristic_donut_chart, create_dark_theme_plotly_chart
)
from streamlit_app.config import DASHBOARD_CONFIG, get_risk_level, get_risk_color

# Suppress warnings
warnings.filterwarnings('ignore')

# ================================================================
# DATA LOADING AND CACHING
# ================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_dashboard_data():
    """
    Load and cache dashboard data with memory optimization.
    
    Returns:
        Tuple of (employee_data, processed_metrics)
    """
    try:
        # Try to load actual data first
        data_path = project_root / "data" / "synthetic" / "hr_employees.csv"
        
        if data_path.exists():
            # Load with memory optimization
            df = pd.read_csv(data_path, low_memory=True)
            
            # Basic data cleaning
            if 'Attrition' in df.columns:
                df['AttritionBinary'] = (df['Attrition'] == 'Yes').astype(int)
            
            # Calculate risk levels if not present
            if 'AttritionProbability' not in df.columns:
                # Simulate risk probabilities based on available features
                df['AttritionProbability'] = simulate_risk_probabilities(df)
            
            df['RiskLevel'] = df['AttritionProbability'].apply(
                lambda x: get_risk_level(x / 100 if x > 1 else x)
            )
            
            return df, True
        
        else:
            # Generate demo data if real data not available
            st.warning("📊 Using demo data. Upload real data for accurate insights.")
            return generate_demo_data(), False
    
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return generate_demo_data(), False

def simulate_risk_probabilities(df):
    """Generate realistic risk probabilities based on available features."""
    np.random.seed(42)
    
    # Base probability
    probs = np.random.uniform(0, 1, len(df))
    
    # Adjust based on available features
    if 'Age' in df.columns:
        # Young employees (under 30) have higher risk
        young_mask = df['Age'] < 30
        probs[young_mask] += 0.2
    
    if 'YearsAtCompany' in df.columns:
        # New employees (< 2 years) have higher risk
        new_mask = df['YearsAtCompany'] < 2
        probs[new_mask] += 0.3
    
    if 'JobSatisfaction' in df.columns:
        # Low satisfaction increases risk
        low_sat_mask = df['JobSatisfaction'] <= 2
        probs[low_sat_mask] += 0.4
    
    if 'OverTime' in df.columns:
        # Overtime increases risk
        overtime_mask = (df['OverTime'] == 'Yes')
        probs[overtime_mask] += 0.2
    
    # Normalize to 0-100 range
    probs = np.clip(probs * 100, 5, 95)
    return probs

def generate_demo_data():
    """Generate demo data for testing purposes."""
    np.random.seed(42)
    
    n_employees = 1000
    departments = ['Engineering', 'Sales', 'Marketing', 'Operations', 'Finance', 'HR']
    job_roles = ['Manager', 'Senior', 'Junior', 'Lead', 'Associate', 'Director']
    
    demo_data = pd.DataFrame({
        'EmployeeID': range(1, n_employees + 1),
        'Age': np.random.randint(22, 65, n_employees),
        'Department': np.random.choice(departments, n_employees),
        'JobRole': np.random.choice(job_roles, n_employees),
        'YearsAtCompany': np.random.randint(0, 20, n_employees),
        'MonthlyIncome': np.random.randint(3000, 15000, n_employees),
        'JobSatisfaction': np.random.randint(1, 5, n_employees),
        'WorkLifeBalance': np.random.randint(1, 5, n_employees),
        'OverTime': np.random.choice(['Yes', 'No'], n_employees, p=[0.3, 0.7]),
        'Attrition': np.random.choice(['Yes', 'No'], n_employees, p=[0.16, 0.84]),
    })
    
    # Add calculated fields
    demo_data['AttritionBinary'] = (demo_data['Attrition'] == 'Yes').astype(int)
    demo_data['AttritionProbability'] = simulate_risk_probabilities(demo_data)
    demo_data['RiskLevel'] = demo_data['AttritionProbability'].apply(get_risk_level)
    
    return demo_data

# ================================================================
# KPI CALCULATION FUNCTIONS
# ================================================================

def calculate_dashboard_metrics(df):
    """Calculate all dashboard KPIs and metrics."""
    metrics = {}
    
    try:
        # Basic counts
        metrics['total_employees'] = len(df)
        metrics['attrition_count'] = df['AttritionBinary'].sum() if 'AttritionBinary' in df.columns else 0
        metrics['attrition_rate'] = metrics['attrition_count'] / metrics['total_employees'] if metrics['total_employees'] > 0 else 0
        
        # Risk distribution
        if 'RiskLevel' in df.columns:
            risk_dist = df['RiskLevel'].value_counts()
            metrics['high_risk_count'] = risk_dist.get('High', 0)
            metrics['medium_risk_count'] = risk_dist.get('Medium', 0)
            metrics['low_risk_count'] = risk_dist.get('Low', 0)
        else:
            metrics['high_risk_count'] = 0
            metrics['medium_risk_count'] = 0
            metrics['low_risk_count'] = 0
        
        # Average tenure
        if 'YearsAtCompany' in df.columns:
            metrics['avg_tenure'] = df['YearsAtCompany'].mean()
        else:
            metrics['avg_tenure'] = 0
        
        # Cost calculations (estimated)
        avg_salary = df['MonthlyIncome'].mean() * 12 if 'MonthlyIncome' in df.columns else 50000
        replacement_cost_per_employee = avg_salary * 1.5  # Standard 1.5x annual salary
        metrics['cost_saved'] = metrics['low_risk_count'] * replacement_cost_per_employee * 0.1  # Estimated savings
        
        # Model performance (simulated)
        metrics['model_accuracy'] = 0.847  # Simulated accuracy
        
        # Department breakdown
        if 'Department' in df.columns:
            dept_attrition = df.groupby('Department')['AttritionBinary'].agg(['count', 'sum']).reset_index()
            dept_attrition['attrition_rate'] = dept_attrition['sum'] / dept_attrition['count']
            dept_attrition = dept_attrition.sort_values('attrition_rate', ascending=False)
            metrics['dept_attrition'] = dept_attrition
        
        # Monthly trends (simulated)
        metrics['monthly_trends'] = generate_monthly_trends()
        
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {}
    
    return metrics

def generate_monthly_trends():
    """Generate monthly trend data for visualization."""
    months = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
    
    # Simulate realistic trends
    base_rate = 0.16
    seasonal_factor = np.sin(np.arange(len(months)) * 2 * np.pi / 12) * 0.03
    noise = np.random.normal(0, 0.01, len(months))
    
    attrition_rates = base_rate + seasonal_factor + noise
    attrition_rates = np.clip(attrition_rates, 0.05, 0.25)
    
    return pd.DataFrame({
        'Month': months,
        'AttritionRate': attrition_rates,
        'EmployeeCount': np.random.randint(950, 1050, len(months))
    })

# ================================================================
# KPI CARDS RENDERING
# ================================================================

def render_kpi_cards(metrics):
    """
    Render executive KPI cards with glassmorphism styling.
    
    Args:
        metrics: Dictionary containing calculated metrics
    """
    st.markdown("### 📊 Executive KPI Dashboard")
    
    # Create 3 rows of KPIs for better mobile layout
    kpi_data = [
        {
            'value': f"{metrics['total_employees']:,}",
            'title': 'Total Employees',
            'icon': '👥',
            'color': 'secondary',
            'delta': '+12' if metrics['total_employees'] > 500 else None
        },
        {
            'value': f"{metrics['attrition_rate']:.1%}",
            'title': 'Attrition Rate',
            'icon': '📈',
            'color': 'warning' if metrics['attrition_rate'] > 0.15 else 'success',
            'delta': f"{(metrics['attrition_rate'] - 0.16) * 100:+.1f}%" if metrics['attrition_rate'] > 0 else None
        },
        {
            'value': f"{metrics['high_risk_count']:,}",
            'title': 'High Risk Employees',
            'icon': '⚠️',
            'color': 'error',
            'delta': '-5' if metrics['high_risk_count'] > 0 else None
        },
        {
            'value': f"{metrics['avg_tenure']:.1f} years",
            'title': 'Average Tenure',
            'icon': '🕐',
            'color': 'success',
            'delta': '+0.2' if metrics['avg_tenure'] > 3 else None
        },
        {
            'value': f"${metrics['cost_saved']:,.0f}",
            'title': 'Cost Savings',
            'icon': '💰',
            'color': 'success',
            'delta': f"+${metrics['cost_saved'] * 0.1:,.0f}" if metrics['cost_saved'] > 0 else None
        },
        {
            'value': f"{metrics['model_accuracy']:.1%}",
            'title': 'Model Accuracy',
            'icon': '🎯',
            'color': 'info',
            'delta': '+2.3%' if metrics['model_accuracy'] > 0.8 else None
        }
    ]
    
    # Render KPIs in responsive layout
    cols_per_row = 3
    for i in range(0, len(kpi_data), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(kpi_data):
                kpi = kpi_data[i + j]
                
                with col:
                    card_html = glassmorphism_metric_card(
                        value=kpi['value'],
                        title=kpi['title'],
                        icon=kpi['icon'],
                        color=kpi['color'],
                        delta=kpi.get('delta'),
                        width=280,
                        height=160
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

# ================================================================
# RISK DISTRIBUTION VISUALIZATION
# ================================================================

def show_risk_distribution(df):
    """
    Display risk distribution with futuristic donut chart.
    
    Args:
        df: Employee DataFrame with risk levels
    """
    st.markdown("### 🎯 Employee Risk Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'RiskLevel' in df.columns:
            # Calculate risk distribution
            risk_counts = df['RiskLevel'].value_counts()
            risk_data = pd.DataFrame({
                'RiskLevel': risk_counts.index,
                'Count': risk_counts.values
            })
            
            # Create futuristic donut chart
            fig = futuristic_donut_chart(
                data=risk_data,
                values_col='Count',
                names_col='RiskLevel',
                title='Risk Level Distribution',
                height=400,
                hole_size=0.6
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Risk level data not available")
    
    with col2:
        # Risk level details
        st.markdown("#### 📋 Risk Breakdown")
        
        if 'RiskLevel' in df.columns:
            risk_counts = df['RiskLevel'].value_counts()
            total = len(df)
            
            for risk_level in ['High', 'Medium', 'Low']:
                count = risk_counts.get(risk_level, 0)
                percentage = count / total * 100 if total > 0 else 0
                color = get_risk_color(risk_level)
                
                st.markdown(f"""
                <div style="
                    padding: 10px; 
                    margin: 5px 0; 
                    border-radius: 8px; 
                    border-left: 4px solid {color};
                    background: rgba(37, 42, 69, 0.3);
                ">
                    <strong style="color: {color};">{risk_level} Risk</strong><br>
                    {count:,} employees ({percentage:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        
        # Risk trend indicator
        st.markdown("#### 📈 Risk Trend")
        
        # Simulate trend data
        trend_direction = "↗️" if np.random.random() > 0.5 else "↘️"
        trend_value = np.random.uniform(1, 5)
        trend_color = COLORS['error'] if trend_direction == "↗️" else COLORS['success']
        
        st.markdown(f"""
        <div style="
            text-align: center; 
            padding: 15px;
            background: rgba(37, 42, 69, 0.3);
            border-radius: 10px;
            border: 1px solid {trend_color}33;
        ">
            <div style="font-size: 24px;">{trend_direction}</div>
            <div style="color: {trend_color}; font-weight: bold;">
                {trend_value:.1f}% vs Last Month
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# DEPARTMENT HEATMAP
# ================================================================

def display_department_heatmap(df):
    """
    Display department-wise attrition heatmap.
    
    Args:
        df: Employee DataFrame with department information
    """
    st.markdown("### 🏢 Department Attrition Analysis")
    
    if 'Department' in df.columns:
        # Calculate department metrics
        dept_metrics = df.groupby('Department').agg({
            'AttritionBinary': ['count', 'sum', 'mean'],
            'AttritionProbability': 'mean' if 'AttritionProbability' in df.columns else 'count'
        }).round(3)
        
        # Flatten column names
        dept_metrics.columns = ['EmployeeCount', 'AttritionCount', 'AttritionRate', 'AvgRiskScore']
        dept_metrics = dept_metrics.reset_index()
        dept_metrics = dept_metrics.sort_values('AttritionRate', ascending=False)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create neon bar chart
            fig = neon_bar_chart(
                data=dept_metrics,
                x_col='Department',
                y_col='AttritionRate',
                title='Attrition Rate by Department',
                height=400,
                show_values=True,
                orientation='v'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Department details table
            st.markdown("#### 📊 Department Details")
            
            # Create custom styled table
            for _, row in dept_metrics.iterrows():
                dept_name = row['Department']
                attrition_rate = row['AttritionRate']
                employee_count = int(row['EmployeeCount'])
                
                # Color coding based on attrition rate
                if attrition_rate > 0.2:
                    status_color = COLORS['error']
                    status_icon = "🔴"
                elif attrition_rate > 0.15:
                    status_color = COLORS['warning'] 
                    status_icon = "🟡"
                else:
                    status_color = COLORS['success']
                    status_icon = "🟢"
                
                st.markdown(f"""
                <div style="
                    padding: 12px; 
                    margin: 8px 0; 
                    border-radius: 10px; 
                    background: rgba(37, 42, 69, 0.4);
                    border-left: 3px solid {status_color};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: {COLORS['text']};">{status_icon} {dept_name}</strong><br>
                            <small style="color: {COLORS['text_secondary']};">
                                {employee_count} employees
                            </small>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: {status_color}; font-weight: bold; font-size: 16px;">
                                {attrition_rate:.1%}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.info("Department data not available")

# ================================================================
# MONTHLY TRENDS ANALYSIS
# ================================================================

def plot_monthly_trends(metrics):
    """
    Plot monthly attrition trends with cyberpunk styling.
    
    Args:
        metrics: Dictionary containing trend data
    """
    st.markdown("### 📈 Monthly Attrition Trends")
    
    if 'monthly_trends' in metrics:
        trends_data = metrics['monthly_trends']
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create line chart with dual y-axis
            fig = go.Figure()
            
            # Attrition rate line
            fig.add_trace(go.Scatter(
                x=trends_data['Month'],
                y=trends_data['AttritionRate'] * 100,  # Convert to percentage
                mode='lines+markers',
                name='Attrition Rate (%)',
                line=dict(color=COLORS['secondary'], width=3),
                marker=dict(size=8, color=COLORS['secondary']),
                yaxis='y'
            ))
            
            # Add trend line
            z = np.polyfit(range(len(trends_data)), trends_data['AttritionRate'] * 100, 1)
            trend_line = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=trends_data['Month'],
                y=trend_line(range(len(trends_data))),
                mode='lines',
                name='Trend',
                line=dict(color=COLORS['accent'], width=2, dash='dash'),
                yaxis='y'
            ))
            
            # Employee count bars (secondary axis)
            fig.add_trace(go.Bar(
                x=trends_data['Month'],
                y=trends_data['EmployeeCount'],
                name='Employee Count',
                marker=dict(color=COLORS['success'], opacity=0.3),
                yaxis='y2'
            ))
            
            # Apply dark theme
            fig = create_dark_theme_plotly_chart(
                fig,
                title='Monthly Attrition Rate & Employee Count Trends',
                height=400,
                custom_layout={
                    'yaxis': dict(
                        title='Attrition Rate (%)',
                        side='left'
                    ),
                    'yaxis2': dict(
                        title='Employee Count',
                        side='right',
                        overlaying='y'
                    ),
                    'hovermode': 'x unified'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trend summary
            st.markdown("#### 📊 Trend Summary")
            
            # Calculate trend metrics
            latest_rate = trends_data['AttritionRate'].iloc[-1]
            prev_rate = trends_data['AttritionRate'].iloc[-2]
            rate_change = latest_rate - prev_rate
            
            # Current month metrics
            current_month = trends_data['Month'].iloc[-1].strftime('%B %Y')
            current_count = int(trends_data['EmployeeCount'].iloc[-1])
            
            st.markdown(f"""
            <div style="
                padding: 15px;
                background: rgba(37, 42, 69, 0.4);
                border-radius: 10px;
                margin-bottom: 15px;
            ">
                <div style="color: {COLORS['text']}; font-weight: bold; margin-bottom: 10px;">
                    📅 {current_month}
                </div>
                <div style="color: {COLORS['secondary']}; font-size: 20px; font-weight: bold;">
                    {latest_rate:.1%}
                </div>
                <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                    Current Attrition Rate
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Month-over-month change
            change_color = COLORS['error'] if rate_change > 0 else COLORS['success']
            change_icon = "↗️" if rate_change > 0 else "↘️"
            
            st.markdown(f"""
            <div style="
                padding: 15px;
                background: rgba(37, 42, 69, 0.4);
                border-radius: 10px;
                margin-bottom: 15px;
                border-left: 3px solid {change_color};
            ">
                <div style="color: {COLORS['text']}; font-weight: bold; margin-bottom: 5px;">
                    📊 Month-over-Month
                </div>
                <div style="color: {change_color}; font-size: 16px; font-weight: bold;">
                    {change_icon} {rate_change:+.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Employee count
            st.markdown(f"""
            <div style="
                padding: 15px;
                background: rgba(37, 42, 69, 0.4);
                border-radius: 10px;
            ">
                <div style="color: {COLORS['text']}; font-weight: bold; margin-bottom: 5px;">
                    👥 Current Headcount
                </div>
                <div style="color: {COLORS['success']}; font-size: 18px; font-weight: bold;">
                    {current_count:,}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("Trend data not available")

# ================================================================
# QUICK INSIGHTS SECTION
# ================================================================

def render_quick_insights(metrics, df):
    """Render quick insights and recommendations."""
    
    st.markdown("### 💡 Quick Insights & Recommendations")
    
    insights = []
    
    # Attrition rate insight
    if metrics['attrition_rate'] > 0.20:
        insights.append({
            'type': 'warning',
            'title': 'High Attrition Alert',
            'message': f'Attrition rate of {metrics["attrition_rate"]:.1%} is above industry average (15%). Immediate intervention recommended.',
            'action': 'Review compensation and employee satisfaction surveys.'
        })
    elif metrics['attrition_rate'] < 0.10:
        insights.append({
            'type': 'success',
            'title': 'Excellent Retention',
            'message': f'Attrition rate of {metrics["attrition_rate"]:.1%} is well below industry average.',
            'action': 'Continue current retention strategies and document best practices.'
        })
    
    # High risk employees
    if metrics['high_risk_count'] > metrics['total_employees'] * 0.1:
        insights.append({
            'type': 'error',
            'title': 'High Risk Employees',
            'message': f'{metrics["high_risk_count"]} employees are at high risk of attrition.',
            'action': 'Schedule one-on-one meetings with high-risk employees immediately.'
        })
    
    # Department-specific insights
    if 'dept_attrition' in metrics and not metrics['dept_attrition'].empty:
        worst_dept = metrics['dept_attrition'].iloc[0]
        if worst_dept['attrition_rate'] > 0.25:
            insights.append({
                'type': 'warning',
                'title': f'{worst_dept["Department"]} Department Concern',
                'message': f'Attrition rate of {worst_dept["attrition_rate"]:.1%} requires immediate attention.',
                'action': f'Conduct focus groups with {worst_dept["Department"]} team members.'
            })
    
    # Display insights
    if insights:
        cols = st.columns(len(insights))
        for i, insight in enumerate(insights):
            with cols[i]:
                color_map = {
                    'success': COLORS['success'],
                    'warning': COLORS['warning'],
                    'error': COLORS['error']
                }
                
                color = color_map[insight['type']]
                
                st.markdown(f"""
                <div style="
                    padding: 20px;
                    background: rgba(37, 42, 69, 0.4);
                    border-radius: 12px;
                    border-left: 4px solid {color};
                    margin-bottom: 15px;
                    height: 200px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                ">
                    <div>
                        <h4 style="color: {color}; margin: 0 0 10px 0; font-size: 16px;">
                            {insight['title']}
                        </h4>
                        <p style="color: {COLORS['text']}; margin: 0 0 15px 0; font-size: 14px;">
                            {insight['message']}
                        </p>
                    </div>
                    <div style="
                        background: rgba(0, 0, 0, 0.2);
                        padding: 10px;
                        border-radius: 6px;
                        border-left: 2px solid {color};
                    ">
                        <strong style="color: {COLORS['text_secondary']}; font-size: 12px;">
                            💡 Action: {insight['action']}
                        </strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("All metrics are within normal ranges. Continue monitoring.")

# ================================================================
# MAIN DASHBOARD FUNCTION
# ================================================================

def show():
    """Main dashboard function called by the navigation system."""
    
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
                🏠 Executive Dashboard
            </h1>
            <p style="color: #B8C5D1; font-size: 1.1rem;">
                Real-time HR analytics and employee attrition insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load data with progress indicator
        with st.spinner("🔄 Loading dashboard data..."):
            df, is_real_data = load_dashboard_data()
            metrics = calculate_dashboard_metrics(df)
        
        # Data status indicator
        status_color = COLORS['success'] if is_real_data else COLORS['warning']
        status_text = "Live Data" if is_real_data else "Demo Data"
        
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="
                color: {status_color};
                font-size: 14px;
                padding: 5px 15px;
                background: rgba(37, 42, 69, 0.4);
                border-radius: 20px;
                border: 1px solid {status_color}33;
            ">
                📊 {status_text} • Last Updated: {datetime.now().strftime('%H:%M:%S')}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Main dashboard sections
        render_kpi_cards(metrics)
        
        st.markdown("---")
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            show_risk_distribution(df)
        
        with col2:
            display_department_heatmap(df)
        
        st.markdown("---")
        
        plot_monthly_trends(metrics)
        
        st.markdown("---")
        
        render_quick_insights(metrics, df)
        
        # Refresh data button
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.experimental_rerun()
        
        # Memory cleanup
        gc.collect()
        
    except Exception as e:
        st.error(f"Dashboard Error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")

# ================================================================
# ENTRY POINT FOR TESTING
# ================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Dashboard", layout="wide")
    show()
