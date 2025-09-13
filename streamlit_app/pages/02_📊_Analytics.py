"""
HR Attrition Predictor - Deep Analytics Dashboard
================================================
Comprehensive data analytics with interactive visualizations, correlation analysis,
demographic insights, and satisfaction analysis. Memory-optimized for 4GB RAM.

Author: HR Analytics Team
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import warnings
import gc
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
from streamlit_app.assets.theme import COLORS, TYPOGRAPHY, apply_custom_css
from streamlit_app.components.charts import (
    glassmorphism_metric_card, create_dark_theme_plotly_chart,
    futuristic_donut_chart, neon_bar_chart, futuristic_gauge_chart
)
from streamlit_app.config import get_risk_level, get_risk_color

# Import visualization utilities
try:
    from src.utils.visualizations import (
        create_risk_distribution_chart, plot_department_heatmap,
        generate_salary_distribution, create_satisfaction_radar_chart,
        get_color
    )
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('dark_background')

# ================================================================
# DATA LOADING AND CACHING
# ================================================================

@st.cache_data(ttl=300)
def load_analytics_data():
    """Load and cache analytics data with comprehensive preprocessing."""
    
    try:
        data_path = project_root / "data" / "synthetic" / "hr_employees.csv"
        
        if data_path.exists():
            # Load full dataset
            df = pd.read_csv(data_path)
            
            # Data enrichment for analytics
            df = _enrich_analytics_data(df)
            
            return df, True
        else:
            # Generate comprehensive demo data
            return _generate_comprehensive_demo_data(), False
    
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
        return _generate_comprehensive_demo_data(), False

def _enrich_analytics_data(df):
    """Enrich data with calculated fields for analytics."""
    
    # Age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                           labels=['<25', '25-34', '35-44', '45-54', '55+'])
    
    # Tenure groups
    if 'YearsAtCompany' in df.columns:
        df['TenureGroup'] = pd.cut(df['YearsAtCompany'], bins=[0, 1, 3, 5, 10, 50],
                                  labels=['0-1 years', '1-3 years', '3-5 years', '5-10 years', '10+ years'])
    
    # Salary groups
    if 'MonthlyIncome' in df.columns:
        df['SalaryGroup'] = pd.cut(df['MonthlyIncome'], bins=[0, 3000, 5000, 8000, 12000, 100000],
                                  labels=['<$3K', '$3-5K', '$5-8K', '$8-12K', '$12K+'])
    
    # Performance tiers
    if 'PerformanceRating' in df.columns:
        df['PerformanceTier'] = pd.cut(df['PerformanceRating'], bins=[0, 2, 3, 4, 5],
                                      labels=['Below Avg', 'Average', 'Good', 'Excellent'])
    
    # Satisfaction composite score
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'RelationshipSatisfaction']
    available_sat_cols = [col for col in satisfaction_cols if col in df.columns]
    
    if available_sat_cols:
        df['SatisfactionComposite'] = df[available_sat_cols].mean(axis=1)
        df['SatisfactionTier'] = pd.cut(df['SatisfactionComposite'], bins=[0, 2, 3, 4, 5],
                                       labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Attrition risk probability (simulate if not present)
    if 'AttritionProbability' not in df.columns:
        df['AttritionProbability'] = _simulate_attrition_probability(df)
    
    df['RiskLevel'] = df['AttritionProbability'].apply(get_risk_level)
    
    # Work-life indicators
    if 'OverTime' in df.columns:
        df['WorkIntensity'] = df['OverTime'].map({'Yes': 'High', 'No': 'Normal'})
    
    if 'BusinessTravel' in df.columns:
        df['TravelFrequency'] = df['BusinessTravel'].map({
            'Non-Travel': 'None',
            'Travel_Rarely': 'Occasional', 
            'Travel_Frequently': 'Frequent'
        })
    
    return df

def _simulate_attrition_probability(df):
    """Simulate realistic attrition probabilities."""
    np.random.seed(42)
    
    # Base probability
    base_prob = np.random.uniform(0.05, 0.95, len(df))
    
    # Adjust based on available features
    adjustments = np.zeros(len(df))
    
    # Age factor (younger employees higher risk)
    if 'Age' in df.columns:
        age_factor = (40 - df['Age'].clip(22, 65)) / 100
        adjustments += age_factor
    
    # Tenure factor (new employees higher risk)
    if 'YearsAtCompany' in df.columns:
        tenure_factor = (5 - df['YearsAtCompany'].clip(0, 20)) / 20
        adjustments += tenure_factor
    
    # Satisfaction factor
    satisfaction_cols = ['JobSatisfaction', 'WorkLifeBalance']
    for col in satisfaction_cols:
        if col in df.columns:
            sat_factor = (3 - df[col].clip(1, 5)) / 10
            adjustments += sat_factor
    
    # Overtime factor
    if 'OverTime' in df.columns:
        overtime_factor = (df['OverTime'] == 'Yes') * 0.2
        adjustments += overtime_factor
    
    # Apply adjustments
    final_prob = base_prob + adjustments
    return np.clip(final_prob, 0.02, 0.98)

def _generate_comprehensive_demo_data():
    """Generate comprehensive demo data for analytics."""
    
    np.random.seed(42)
    n_employees = 1500  # Larger dataset for analytics
    
    # Demographics
    ages = np.random.normal(38, 12, n_employees).astype(int)
    ages = np.clip(ages, 22, 65)
    
    departments = np.random.choice([
        'Engineering', 'Sales', 'Marketing', 'Operations', 
        'Finance', 'HR', 'Legal', 'Customer Service'
    ], n_employees, p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.05])
    
    job_roles = np.random.choice([
        'Manager', 'Senior', 'Junior', 'Lead', 'Associate', 
        'Director', 'Analyst', 'Specialist'
    ], n_employees, p=[0.12, 0.18, 0.25, 0.15, 0.15, 0.05, 0.08, 0.02])
    
    # Generate correlated features
    demo_data = pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n_employees + 1)],
        'Age': ages,
        'Gender': np.random.choice(['Male', 'Female'], n_employees, p=[0.55, 0.45]),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_employees, p=[0.35, 0.55, 0.10]),
        'Education': np.random.choice([
            'High School', "Bachelor's", "Master's", 'PhD'
        ], n_employees, p=[0.15, 0.45, 0.35, 0.05]),
        
        'Department': departments,
        'JobRole': job_roles,
        'JobLevel': np.random.randint(1, 6, n_employees),
        
        'YearsAtCompany': np.random.gamma(2, 2, n_employees).astype(int),
        'YearsInCurrentRole': np.random.gamma(1.5, 1.5, n_employees).astype(int),
        'TotalWorkingYears': ages - 22 + np.random.randint(-3, 8, n_employees),
        
        'MonthlyIncome': np.random.normal(6500, 2500, n_employees).astype(int),
        'PercentSalaryHike': np.random.normal(12, 5, n_employees),
        'PerformanceRating': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.05, 0.15, 0.45, 0.30, 0.05]),
        
        'JobSatisfaction': np.random.randint(1, 5, n_employees),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n_employees),
        'WorkLifeBalance': np.random.randint(1, 4, n_employees),
        'RelationshipSatisfaction': np.random.randint(1, 5, n_employees),
        
        'OverTime': np.random.choice(['Yes', 'No'], n_employees, p=[0.28, 0.72]),
        'BusinessTravel': np.random.choice([
            'Non-Travel', 'Travel_Rarely', 'Travel_Frequently'
        ], n_employees, p=[0.25, 0.60, 0.15]),
        'DistanceFromHome': np.random.gamma(2, 5, n_employees).astype(int),
        
        'TrainingTimesLastYear': np.random.poisson(2, n_employees),
        'WorkAccident': np.random.choice(['Yes', 'No'], n_employees, p=[0.03, 0.97]),
        'Promotion': np.random.choice(['Yes', 'No'], n_employees, p=[0.12, 0.88]),
        
        'Attrition': np.random.choice(['Yes', 'No'], n_employees, p=[0.16, 0.84])
    })
    
    # Fix data consistency
    demo_data['MonthlyIncome'] = np.clip(demo_data['MonthlyIncome'], 2500, 25000)
    demo_data['YearsInCurrentRole'] = np.clip(demo_data['YearsInCurrentRole'], 0, demo_data['YearsAtCompany'])
    demo_data['TotalWorkingYears'] = np.clip(demo_data['TotalWorkingYears'], demo_data['YearsAtCompany'], 50)
    demo_data['DistanceFromHome'] = np.clip(demo_data['DistanceFromHome'], 1, 50)
    
    # Apply enrichment
    demo_data = _enrich_analytics_data(demo_data)
    
    return demo_data

# ================================================================
# CORRELATION MATRIX ANALYSIS
# ================================================================

def correlation_matrix():
    """Generate comprehensive correlation matrix analysis."""
    
    st.markdown("### 🔗 Correlation Matrix Analysis")
    
    # Load data
    data, is_real = load_analytics_data()
    
    # Configuration options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        correlation_type = st.selectbox(
            "Correlation Type:",
            ["Pearson", "Spearman", "Kendall"],
            help="Choose correlation method"
        )
    
    with col2:
        min_correlation = st.slider(
            "Min Correlation:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Filter correlations below threshold"
        )
    
    with col3:
        matrix_style = st.selectbox(
            "Matrix Style:",
            ["Heatmap", "Network", "Clustermap"]
        )
    
    # Select numeric columns for correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns
    numeric_cols = [col for col in numeric_cols if not col.lower().endswith('id')]
    
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns for correlation analysis")
        return
    
    # Calculate correlations
    if correlation_type == "Pearson":
        corr_matrix = data[numeric_cols].corr()
    elif correlation_type == "Spearman":
        corr_matrix = data[numeric_cols].corr(method='spearman')
    else:  # Kendall
        corr_matrix = data[numeric_cols].corr(method='kendall')
    
    # Filter correlations
    mask = np.abs(corr_matrix) >= min_correlation
    corr_matrix_filtered = corr_matrix.where(mask)
    
    # Display correlation matrix
    if matrix_style == "Heatmap":
        _display_correlation_heatmap(corr_matrix, corr_matrix_filtered, min_correlation)
    elif matrix_style == "Network":
        _display_correlation_network(corr_matrix, min_correlation)
    else:  # Clustermap
        _display_correlation_clustermap(corr_matrix, min_correlation)
    
    # Key insights
    _display_correlation_insights(corr_matrix, data)

def _display_correlation_heatmap(corr_matrix, filtered_matrix, threshold):
    """Display interactive correlation heatmap."""
    
    fig = go.Figure()
    
    # Create heatmap
    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
            [0, get_color('error')],
            [0.5, get_color('background_light')],
            [1, get_color('success')]
        ],
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(color=get_color('text')),
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title="Correlation",
            titlefont=dict(color=get_color('text')),
            tickfont=dict(color=get_color('text'))
        )
    ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title=f"Feature Correlation Matrix (threshold: {threshold})",
        height=600,
        show_legend=False,
        custom_layout={
            'xaxis': dict(tickangle=45),
            'yaxis': dict(tickangle=0)
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _display_correlation_network(corr_matrix, threshold):
    """Display correlation network graph."""
    
    st.info("🔗 Network visualization shows features connected by correlations above threshold")
    
    # Create network data
    edges = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                edges.append({
                    'source': corr_matrix.columns[i],
                    'target': corr_matrix.columns[j], 
                    'weight': abs(corr_val),
                    'correlation': corr_val
                })
    
    if not edges:
        st.warning("No correlations above threshold found")
        return
    
    # Create network visualization using plotly
    # Simplified network layout
    nodes = list(corr_matrix.columns)
    
    # Calculate node positions (circular layout)
    n_nodes = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    fig = go.Figure()
    
    # Add edges
    for edge in edges:
        source_idx = nodes.index(edge['source'])
        target_idx = nodes.index(edge['target'])
        
        color = get_color('success') if edge['correlation'] > 0 else get_color('error')
        width = edge['weight'] * 5  # Scale width by correlation strength
        
        fig.add_trace(go.Scatter(
            x=[x_pos[source_idx], x_pos[target_idx], None],
            y=[y_pos[source_idx], y_pos[target_idx], None],
            mode='lines',
            line=dict(color=color, width=width),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=20,
            color=get_color('secondary'),
            line=dict(color=get_color('border_primary'), width=2)
        ),
        text=nodes,
        textposition='middle center',
        textfont=dict(color=get_color('text'), size=10),
        hovertemplate='<b>%{text}</b><extra></extra>',
        showlegend=False
    ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title=f"Correlation Network (threshold: {threshold})",
        height=600,
        show_legend=False,
        custom_layout={
            'xaxis': dict(showgrid=False, showticklabels=False, zeroline=False),
            'yaxis': dict(showgrid=False, showticklabels=False, zeroline=False)
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Network statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Connections", len(edges))
    with col2:
        avg_correlation = np.mean([abs(e['correlation']) for e in edges])
        st.metric("Avg Correlation", f"{avg_correlation:.3f}")
    with col3:
        max_correlation = max([abs(e['correlation']) for e in edges])
        st.metric("Max Correlation", f"{max_correlation:.3f}")

def _display_correlation_clustermap(corr_matrix, threshold):
    """Display clustered correlation matrix."""
    
    st.info("🗂️ Clustered view groups similar features together")
    
    try:
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Perform clustering
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        
        # Get cluster order
        dendro = dendrogram(linkage_matrix, no_plot=True)
        cluster_order = dendro['leaves']
        
        # Reorder correlation matrix
        corr_clustered = corr_matrix.iloc[cluster_order, cluster_order]
        
        # Display clustered heatmap
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=corr_clustered.values,
            x=corr_clustered.columns,
            y=corr_clustered.columns,
            colorscale=[
                [0, get_color('error')],
                [0.5, get_color('background_light')],
                [1, get_color('success')]
            ],
            zmid=0,
            text=np.round(corr_clustered.values, 2),
            texttemplate='%{text}',
            textfont=dict(color=get_color('text')),
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>',
            showscale=True
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Clustered Correlation Matrix",
            height=600,
            custom_layout={
                'xaxis': dict(tickangle=45),
                'yaxis': dict(tickangle=0)
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        st.warning("Clustering requires scipy. Showing regular heatmap instead.")
        _display_correlation_heatmap(corr_matrix, corr_matrix, threshold)

def _display_correlation_insights(corr_matrix, data):
    """Display key correlation insights."""
    
    st.markdown("### 🧠 Key Correlation Insights")
    
    # Find strongest correlations
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })
    
    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # Display top correlations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔗 Strongest Positive Correlations")
        positive_corrs = [cp for cp in corr_pairs if cp['correlation'] > 0][:5]
        
        for i, cp in enumerate(positive_corrs, 1):
            st.markdown(f"""
            <div style="
                padding: 12px;
                background: rgba(0, 255, 136, 0.1);
                border-radius: 8px;
                border-left: 3px solid {get_color('success')};
                margin: 8px 0;
            ">
                <div style="color: {get_color('text')}; font-weight: bold;">
                    {i}. {cp['feature1']} ↔ {cp['feature2']}
                </div>
                <div style="color: {get_color('success')}; font-size: 14px;">
                    Correlation: {cp['correlation']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⚡ Strongest Negative Correlations")
        negative_corrs = [cp for cp in corr_pairs if cp['correlation'] < 0][:5]
        
        for i, cp in enumerate(negative_corrs, 1):
            st.markdown(f"""
            <div style="
                padding: 12px;
                background: rgba(255, 45, 117, 0.1);
                border-radius: 8px;
                border-left: 3px solid {get_color('error')};
                margin: 8px 0;
            ">
                <div style="color: {get_color('text')}; font-weight: bold;">
                    {i}. {cp['feature1']} ↔ {cp['feature2']}
                </div>
                <div style="color: {get_color('error')}; font-size: 14px;">
                    Correlation: {cp['correlation']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Correlation with attrition (if available)
    if 'AttritionProbability' in data.columns:
        st.markdown("#### 🎯 Correlations with Attrition Risk")
        
        attrition_corrs = corr_matrix['AttritionProbability'].sort_values(key=abs, ascending=False)[1:6]  # Exclude self-correlation
        
        for feature, corr in attrition_corrs.items():
            color = get_color('error') if corr > 0 else get_color('success')
            arrow = "↗️" if corr > 0 else "↘️"
            
            st.markdown(f"""
            <div style="
                padding: 10px;
                background: rgba(37, 42, 69, 0.3);
                border-radius: 6px;
                border-left: 2px solid {color};
                margin: 5px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color: {get_color('text')};">{arrow} {feature}</span>
                <span style="color: {color}; font-weight: bold;">{corr:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

# ================================================================
# SALARY DISTRIBUTION ANALYSIS
# ================================================================

def salary_distribution_analysis():
    """Comprehensive salary distribution analysis."""
    
    st.markdown("### 💰 Salary Distribution Analysis")
    
    # Load data
    data, is_real = load_analytics_data()
    
    if 'MonthlyIncome' not in data.columns:
        st.warning("Salary data not available")
        return
    
    # Analysis options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        group_by_options = ['None', 'Department', 'JobRole', 'JobLevel', 'Gender', 'AgeGroup', 'TenureGroup']
        group_by = st.selectbox("Group Analysis By:", group_by_options)
    
    with col2:
        chart_type = st.selectbox("Chart Type:", ["Distribution", "Box Plot", "Violin Plot", "Statistical"])
    
    with col3:
        show_outliers = st.checkbox("Show Outliers", value=True)
    
    # Salary statistics
    salary_stats = data['MonthlyIncome'].describe()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(glassmorphism_metric_card(
            value=f"${salary_stats['mean']:,.0f}",
            title="Average Salary",
            icon="💰",
            color='secondary'
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(glassmorphism_metric_card(
            value=f"${salary_stats['50%']:,.0f}",
            title="Median Salary",
            icon="📊",
            color='info'
        ), unsafe_allow_html=True)
    
    with col3:
        salary_range = salary_stats['max'] - salary_stats['min']
        st.markdown(glassmorphism_metric_card(
            value=f"${salary_range:,.0f}",
            title="Salary Range",
            icon="📈",
            color='warning'
        ), unsafe_allow_html=True)
    
    with col4:
        cv = (salary_stats['std'] / salary_stats['mean']) * 100
        st.markdown(glassmorphism_metric_card(
            value=f"{cv:.1f}%",
            title="Coefficient of Variation",
            icon="📏",
            color='accent'
        ), unsafe_allow_html=True)
    
    # Main visualization
    if chart_type == "Distribution":
        _display_salary_distribution(data, group_by, show_outliers)
    elif chart_type == "Box Plot":
        _display_salary_boxplot(data, group_by, show_outliers)
    elif chart_type == "Violin Plot":
        _display_salary_violinplot(data, group_by)
    else:  # Statistical
        _display_salary_statistical_analysis(data, group_by)
    
    # Salary equity analysis
    _display_salary_equity_analysis(data)

def _display_salary_distribution(data, group_by, show_outliers):
    """Display salary distribution histogram."""
    
    if group_by == 'None':
        # Single distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data['MonthlyIncome'],
            nbinsx=30,
            marker=dict(
                color=get_color('secondary'),
                line=dict(color=get_color('border_primary'), width=1)
            ),
            opacity=0.8,
            hovertemplate='Salary Range: $%{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add statistical lines
        mean_salary = data['MonthlyIncome'].mean()
        median_salary = data['MonthlyIncome'].median()
        
        fig.add_vline(x=mean_salary, line_dash="dash", line_color=get_color('warning'),
                     annotation_text=f"Mean: ${mean_salary:,.0f}")
        fig.add_vline(x=median_salary, line_dash="dot", line_color=get_color('success'),
                     annotation_text=f"Median: ${median_salary:,.0f}")
        
        # Outliers
        if show_outliers:
            Q1 = data['MonthlyIncome'].quantile(0.25)
            Q3 = data['MonthlyIncome'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            fig.add_vline(x=lower_bound, line_dash="longdash", line_color=get_color('error'),
                         annotation_text=f"Lower Outlier: ${lower_bound:,.0f}")
            fig.add_vline(x=upper_bound, line_dash="longdash", line_color=get_color('error'),
                         annotation_text=f"Upper Outlier: ${upper_bound:,.0f}")
    
    else:
        # Grouped distribution
        if group_by not in data.columns:
            st.error(f"Column {group_by} not found")
            return
        
        fig = go.Figure()
        
        groups = data[group_by].unique()
        colors = px.colors.qualitative.Set3[:len(groups)]
        
        for i, group in enumerate(groups):
            group_data = data[data[group_by] == group]['MonthlyIncome']
            
            fig.add_trace(go.Histogram(
                x=group_data,
                name=str(group),
                nbinsx=20,
                opacity=0.7,
                marker=dict(color=colors[i % len(colors)]),
                hovertemplate=f'<b>{group}</b><br>Salary: $%{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title=f"Salary Distribution {'by ' + group_by if group_by != 'None' else ''}",
        height=500,
        custom_layout={
            'xaxis_title': 'Monthly Income ($)',
            'yaxis_title': 'Number of Employees',
            'barmode': 'overlay' if group_by != 'None' else None
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _display_salary_boxplot(data, group_by, show_outliers):
    """Display salary box plot analysis."""
    
    fig = go.Figure()
    
    if group_by == 'None':
        fig.add_trace(go.Box(
            y=data['MonthlyIncome'],
            name='All Employees',
            marker=dict(color=get_color('secondary')),
            boxpoints='outliers' if show_outliers else False,
            hovertemplate='Salary: $%{y:,.0f}<extra></extra>'
        ))
    else:
        if group_by not in data.columns:
            st.error(f"Column {group_by} not found")
            return
        
        groups = data[group_by].unique()
        colors = px.colors.qualitative.Set3[:len(groups)]
        
        for i, group in enumerate(groups):
            group_data = data[data[group_by] == group]['MonthlyIncome']
            
            fig.add_trace(go.Box(
                y=group_data,
                name=str(group),
                marker=dict(color=colors[i % len(colors)]),
                boxpoints='outliers' if show_outliers else False,
                hovertemplate=f'<b>{group}</b><br>Salary: $%{{y:,.0f}}<extra></extra>'
            ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title=f"Salary Box Plot {'by ' + group_by if group_by != 'None' else ''}",
        height=500,
        custom_layout={
            'yaxis_title': 'Monthly Income ($)',
            'xaxis_title': group_by if group_by != 'None' else None
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _display_salary_violinplot(data, group_by):
    """Display salary violin plot."""
    
    fig = go.Figure()
    
    if group_by == 'None':
        fig.add_trace(go.Violin(
            y=data['MonthlyIncome'],
            name='All Employees',
            marker=dict(color=get_color('secondary')),
            box_visible=True,
            meanline_visible=True,
            hovertemplate='Salary: $%{y:,.0f}<extra></extra>'
        ))
    else:
        if group_by not in data.columns:
            st.error(f"Column {group_by} not found")
            return
        
        groups = data[group_by].unique()
        colors = px.colors.qualitative.Set3[:len(groups)]
        
        for i, group in enumerate(groups):
            group_data = data[data[group_by] == group]['MonthlyIncome']
            
            fig.add_trace(go.Violin(
                y=group_data,
                name=str(group),
                marker=dict(color=colors[i % len(colors)]),
                box_visible=True,
                meanline_visible=True,
                hovertemplate=f'<b>{group}</b><br>Salary: $%{{y:,.0f}}<extra></extra>'
            ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title=f"Salary Distribution (Violin Plot) {'by ' + group_by if group_by != 'None' else ''}",
        height=500,
        custom_layout={
            'yaxis_title': 'Monthly Income ($)',
            'xaxis_title': group_by if group_by != 'None' else None
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _display_salary_statistical_analysis(data, group_by):
    """Display statistical analysis of salaries."""
    
    if group_by == 'None':
        # Overall statistics
        st.markdown("#### 📊 Statistical Summary")
        
        salary_stats = data['MonthlyIncome'].describe()
        
        # Create statistics chart
        fig = go.Figure()
        
        stats_names = ['Min', 'Q1', 'Median', 'Q3', 'Max', 'Mean']
        stats_values = [
            salary_stats['min'],
            salary_stats['25%'],
            salary_stats['50%'],
            salary_stats['75%'],
            salary_stats['max'],
            salary_stats['mean']
        ]
        
        fig.add_trace(go.Bar(
            x=stats_names,
            y=stats_values,
            marker=dict(color=get_color('secondary')),
            text=[f'${val:,.0f}' for val in stats_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Value: $%{y:,.0f}<extra></extra>'
        ))
        
    else:
        # Group statistics
        if group_by not in data.columns:
            st.error(f"Column {group_by} not found")
            return
        
        st.markdown(f"#### 📊 Statistical Summary by {group_by}")
        
        group_stats = data.groupby(group_by)['MonthlyIncome'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(0)
        
        # Display as table
        st.dataframe(
            group_stats.style.format({
                'mean': '${:,.0f}',
                'median': '${:,.0f}',
                'std': '${:,.0f}',
                'min': '${:,.0f}',
                'max': '${:,.0f}'
            }),
            use_container_width=True
        )
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=group_stats.index,
            y=group_stats['mean'],
            name='Mean',
            marker=dict(color=get_color('secondary')),
            text=[f'${val:,.0f}' for val in group_stats['mean']],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            x=group_stats.index,
            y=group_stats['median'],
            name='Median',
            marker=dict(color=get_color('accent')),
            text=[f'${val:,.0f}' for val in group_stats['median']],
            textposition='outside'
        ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title=f"Salary Statistics {'by ' + group_by if group_by != 'None' else ''}",
        height=400,
        custom_layout={
            'yaxis_title': 'Salary ($)',
            'barmode': 'group' if group_by != 'None' else None
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _display_salary_equity_analysis(data):
    """Analyze salary equity across different groups."""
    
    st.markdown("#### ⚖️ Salary Equity Analysis")
    
    equity_groups = ['Gender', 'Department', 'JobLevel']
    
    for group in equity_groups:
        if group not in data.columns:
            continue
        
        # Calculate statistics by group
        group_stats = data.groupby(group)['MonthlyIncome'].agg(['mean', 'median', 'count']).round(0)
        
        if len(group_stats) < 2:
            continue
        
        # Calculate pay gaps
        if group == 'Gender' and 'Male' in group_stats.index and 'Female' in group_stats.index:
            male_avg = group_stats.loc['Male', 'mean']
            female_avg = group_stats.loc['Female', 'mean']
            pay_gap = ((male_avg - female_avg) / male_avg) * 100
            
            st.markdown(f"""
            <div style="
                padding: 15px;
                background: rgba(37, 42, 69, 0.4);
                border-radius: 10px;
                border-left: 4px solid {'red' if pay_gap > 5 else 'orange' if pay_gap > 0 else 'green'};
                margin: 10px 0;
            ">
                <h5 style="color: {get_color('text')}; margin: 0 0 10px 0;">Gender Pay Gap Analysis</h5>
                <div style="color: {get_color('text_secondary')};">
                    Male Average: ${male_avg:,.0f}<br>
                    Female Average: ${female_avg:,.0f}<br>
                    Pay Gap: {pay_gap:.1f}% {'(Male higher)' if pay_gap > 0 else '(Female higher)'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show distribution for this group
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = go.Figure()
            
            for category in group_stats.index:
                category_data = data[data[group] == category]['MonthlyIncome']
                
                fig.add_trace(go.Box(
                    y=category_data,
                    name=str(category),
                    hovertemplate=f'<b>{category}</b><br>Salary: $%{{y:,.0f}}<extra></extra>'
                ))
            
            fig = create_dark_theme_plotly_chart(
                fig,
                title=f"Salary Distribution by {group}",
                height=300,
                custom_layout={'yaxis_title': 'Monthly Income ($)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"**{group} Statistics**")
            for category, stats in group_stats.iterrows():
                st.metric(
                    f"{category} Avg",
                    f"${stats['mean']:,.0f}",
                    f"{stats['count']} employees"
                )

# ================================================================
# DEMOGRAPHIC BREAKDOWNS
# ================================================================

def demographic_breakdowns():
    """Comprehensive demographic analysis."""
    
    st.markdown("### 👥 Demographic Analysis")
    
    # Load data
    data, is_real = load_analytics_data()
    
    # Demographic analysis options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        demo_analysis = st.selectbox(
            "Analysis Type:",
            [
                "Age Distribution", 
                "Department Composition",
                "Education Levels",
                "Tenure Analysis",
                "Gender Distribution",
                "Cross-Demographic Analysis"
            ]
        )
    
    with col2:
        include_attrition = st.checkbox("Include Attrition Analysis", value=True)
    
    # Execute selected analysis
    if demo_analysis == "Age Distribution":
        _analyze_age_demographics(data, include_attrition)
    elif demo_analysis == "Department Composition":
        _analyze_department_demographics(data, include_attrition)
    elif demo_analysis == "Education Levels":
        _analyze_education_demographics(data, include_attrition)
    elif demo_analysis == "Tenure Analysis":
        _analyze_tenure_demographics(data, include_attrition)
    elif demo_analysis == "Gender Distribution":
        _analyze_gender_demographics(data, include_attrition)
    else:  # Cross-Demographic Analysis
        _analyze_cross_demographics(data, include_attrition)

def _analyze_age_demographics(data, include_attrition):
    """Analyze age-based demographics."""
    
    st.markdown("#### 🎂 Age Distribution Analysis")
    
    # Age statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_age = data['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    with col2:
        median_age = data['Age'].median()
        st.metric("Median Age", f"{median_age:.0f} years")
    
    with col3:
        age_range = data['Age'].max() - data['Age'].min()
        st.metric("Age Range", f"{age_range} years")
    
    with col4:
        if 'AgeGroup' in data.columns:
            modal_age_group = data['AgeGroup'].mode().iloc[0]
            st.metric("Most Common Group", modal_age_group)
    
    # Age distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data['Age'],
            nbinsx=15,
            marker=dict(color=get_color('secondary')),
            hovertemplate='Age Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Age Distribution",
            height=400,
            custom_layout={
                'xaxis_title': 'Age (years)',
                'yaxis_title': 'Number of Employees'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age groups pie chart
        if 'AgeGroup' in data.columns:
            age_group_counts = data['AgeGroup'].value_counts()
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=age_group_counts.index,
                values=age_group_counts.values,
                marker=dict(colors=px.colors.qualitative.Set3),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ))
            
            fig = create_dark_theme_plotly_chart(
                fig,
                title="Age Group Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Attrition by age
    if include_attrition and 'Attrition' in data.columns:
        st.markdown("#### 📊 Attrition by Age")
        
        # Age vs attrition
        age_attrition = data.groupby(pd.cut(data['Age'], bins=8))['Attrition'].apply(
            lambda x: (x == 'Yes').mean() if len(x) > 0 else 0
        ).reset_index()
        
        age_attrition['AgeRange'] = age_attrition['Age'].astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=age_attrition['AgeRange'],
            y=age_attrition['Attrition'] * 100,  # Convert to percentage
            marker=dict(color=get_color('warning')),
            text=[f'{val:.1f}%' for val in age_attrition['Attrition'] * 100],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Attrition Rate: %{y:.1f}%<extra></extra>'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Attrition Rate by Age Range",
            height=400,
            custom_layout={
                'xaxis_title': 'Age Range',
                'yaxis_title': 'Attrition Rate (%)',
                'xaxis': dict(tickangle=45)
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)

def _analyze_department_demographics(data, include_attrition):
    """Analyze department-based demographics."""
    
    st.markdown("#### 🏢 Department Composition Analysis")
    
    # Department statistics
    dept_stats = data['Department'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Departments", len(dept_stats))
    
    with col2:
        largest_dept = dept_stats.index[0]
        st.metric("Largest Department", largest_dept)
    
    with col3:
        largest_size = dept_stats.iloc[0]
        st.metric("Largest Dept Size", f"{largest_size} employees")
    
    with col4:
        avg_dept_size = dept_stats.mean()
        st.metric("Average Dept Size", f"{avg_dept_size:.0f} employees")
    
    # Department visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=dept_stats.values,
            y=dept_stats.index,
            orientation='h',
            marker=dict(color=get_color('secondary')),
            text=dept_stats.values,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Employees: %{x}<extra></extra>'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Department Sizes",
            height=400,
            custom_layout={
                'xaxis_title': 'Number of Employees',
                'yaxis_title': 'Department'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Treemap
        fig = go.Figure()
        
        fig.add_trace(go.Treemap(
            labels=dept_stats.index,
            values=dept_stats.values,
            parents=[""] * len(dept_stats),
            marker=dict(colors=px.colors.qualitative.Set3),
            hovertemplate='<b>%{label}</b><br>Employees: %{value}<br>Percentage: %{percentParent}<extra></extra>'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Department Distribution (Treemap)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Department demographics details
    if include_attrition and 'Attrition' in data.columns:
        st.markdown("#### 📊 Department Attrition Analysis")
        
        dept_analysis = data.groupby('Department').agg({
            'Attrition': lambda x: (x == 'Yes').mean(),
            'Age': 'mean',
            'MonthlyIncome': 'mean' if 'MonthlyIncome' in data.columns else lambda x: 0,
            'JobSatisfaction': 'mean' if 'JobSatisfaction' in data.columns else lambda x: 0
        }).round(3)
        
        dept_analysis.columns = ['AttritionRate', 'AvgAge', 'AvgSalary', 'AvgSatisfaction']
        dept_analysis = dept_analysis.sort_values('AttritionRate', ascending=False)
        
        # Display as styled dataframe
        styled_df = dept_analysis.style.format({
            'AttritionRate': '{:.1%}',
            'AvgAge': '{:.1f}',
            'AvgSalary': '${:,.0f}',
            'AvgSatisfaction': '{:.2f}'
        }).background_gradient(subset=['AttritionRate'], cmap='Reds')
        
        st.dataframe(styled_df, use_container_width=True)

def _analyze_education_demographics(data, include_attrition):
    """Analyze education-based demographics."""
    
    st.markdown("#### 🎓 Education Level Analysis")
    
    if 'Education' not in data.columns:
        st.warning("Education data not available")
        return
    
    # Education distribution
    edu_counts = data['Education'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=edu_counts.index,
            y=edu_counts.values,
            marker=dict(color=get_color('accent')),
            text=edu_counts.values,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Education Level Distribution",
            height=400,
            custom_layout={
                'xaxis_title': 'Education Level',
                'yaxis_title': 'Number of Employees',
                'xaxis': dict(tickangle=45)
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Education vs salary (if available)
        if 'MonthlyIncome' in data.columns:
            edu_salary = data.groupby('Education')['MonthlyIncome'].mean().sort_values(ascending=False)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=edu_salary.index,
                y=edu_salary.values,
                marker=dict(color=get_color('success')),
                text=[f'${val:,.0f}' for val in edu_salary.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Salary: $%{y:,.0f}<extra></extra>'
            ))
            
            fig = create_dark_theme_plotly_chart(
                fig,
                title="Average Salary by Education",
                height=400,
                custom_layout={
                    'xaxis_title': 'Education Level',
                    'yaxis_title': 'Average Monthly Income ($)',
                    'xaxis': dict(tickangle=45)
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Education insights
    if include_attrition and 'Attrition' in data.columns:
        edu_attrition = data.groupby('Education')['Attrition'].apply(
            lambda x: (x == 'Yes').mean()
        ).sort_values(ascending=False)
        
        st.markdown("#### 📊 Attrition by Education Level")
        
        for edu_level, rate in edu_attrition.items():
            color = get_color('error') if rate > 0.2 else get_color('warning') if rate > 0.15 else get_color('success')
            
            st.markdown(f"""
            <div style="
                padding: 10px;
                background: rgba(37, 42, 69, 0.3);
                border-radius: 6px;
                border-left: 3px solid {color};
                margin: 5px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color: {get_color('text')};">{edu_level}</span>
                <span style="color: {color}; font-weight: bold;">{rate:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

def _analyze_tenure_demographics(data, include_attrition):
    """Analyze tenure-based demographics."""
    
    st.markdown("#### ⏱️ Tenure Analysis")
    
    if 'YearsAtCompany' not in data.columns:
        st.warning("Tenure data not available")
        return
    
    # Tenure statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_tenure = data['YearsAtCompany'].mean()
        st.metric("Average Tenure", f"{avg_tenure:.1f} years")
    
    with col2:
        median_tenure = data['YearsAtCompany'].median()
        st.metric("Median Tenure", f"{median_tenure:.0f} years")
    
    with col3:
        new_employees = (data['YearsAtCompany'] <= 1).sum()
        st.metric("New Employees (≤1 year)", new_employees)
    
    with col4:
        veterans = (data['YearsAtCompany'] >= 10).sum()
        st.metric("Veterans (≥10 years)", veterans)
    
    # Tenure distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data['YearsAtCompany'],
            nbinsx=20,
            marker=dict(color=get_color('info')),
            hovertemplate='Tenure Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Tenure Distribution",
            height=400,
            custom_layout={
                'xaxis_title': 'Years at Company',
                'yaxis_title': 'Number of Employees'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tenure groups
        if 'TenureGroup' in data.columns:
            tenure_counts = data['TenureGroup'].value_counts()
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=tenure_counts.index,
                values=tenure_counts.values,
                marker=dict(colors=px.colors.qualitative.Pastel),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ))
            
            fig = create_dark_theme_plotly_chart(
                fig,
                title="Tenure Group Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tenure vs attrition
    if include_attrition and 'Attrition' in data.columns:
        st.markdown("#### 📊 Attrition Risk by Tenure")
        
        # Create tenure bins for analysis
        tenure_bins = pd.cut(data['YearsAtCompany'], bins=[0, 1, 3, 5, 10, 50], 
                           labels=['0-1', '1-3', '3-5', '5-10', '10+'])
        
        tenure_attrition = data.groupby(tenure_bins)['Attrition'].apply(
            lambda x: (x == 'Yes').mean()
        )
        
        fig = go.Figure()
        
        colors = [get_color('error') if rate > 0.25 else get_color('warning') if rate > 0.15 else get_color('success') 
                 for rate in tenure_attrition.values]
        
        fig.add_trace(go.Bar(
            x=tenure_attrition.index.astype(str),
            y=tenure_attrition.values * 100,
            marker=dict(color=colors),
            text=[f'{val:.1f}%' for val in tenure_attrition.values * 100],
            textposition='outside',
            hovertemplate='<b>%{x} years</b><br>Attrition Rate: %{y:.1f}%<extra></extra>'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Attrition Rate by Tenure Range",
            height=400,
            custom_layout={
                'xaxis_title': 'Years at Company',
                'yaxis_title': 'Attrition Rate (%)'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)

def _analyze_gender_demographics(data, include_attrition):
    """Analyze gender-based demographics."""
    
    st.markdown("#### 👥 Gender Distribution Analysis")
    
    if 'Gender' not in data.columns:
        st.warning("Gender data not available")
        return
    
    # Gender distribution
    gender_counts = data['Gender'].value_counts()
    gender_pct = data['Gender'].value_counts(normalize=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Gender Categories", len(gender_counts))
    
    with col2:
        majority_gender = gender_counts.index[0]
        majority_pct = gender_pct.iloc[0]
        st.metric(f"{majority_gender} Employees", f"{majority_pct:.1%}")
    
    with col3:
        if len(gender_counts) >= 2:
            minority_gender = gender_counts.index[1]
            minority_pct = gender_pct.iloc[1]
            st.metric(f"{minority_gender} Employees", f"{minority_pct:.1%}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = go.Figure()
        
        colors = [get_color('secondary'), get_color('accent'), get_color('success')]
        
        fig.add_trace(go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            marker=dict(colors=colors[:len(gender_counts)]),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Gender Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender by department
        if 'Department' in data.columns:
            gender_dept = pd.crosstab(data['Department'], data['Gender'], normalize='index') * 100
            
            fig = go.Figure()
            
            for i, gender in enumerate(gender_dept.columns):
                fig.add_trace(go.Bar(
                    name=gender,
                    x=gender_dept.index,
                    y=gender_dept[gender],
                    marker=dict(color=colors[i % len(colors)]),
                    text=[f'{val:.1f}%' for val in gender_dept[gender]],
                    textposition='inside' if gender_dept[gender].mean() > 20 else 'outside'
                ))
            
            fig = create_dark_theme_plotly_chart(
                fig,
                title="Gender Distribution by Department",
                height=400,
                custom_layout={
                    'xaxis_title': 'Department',
                    'yaxis_title': 'Percentage',
                    'barmode': 'stack',
                    'xaxis': dict(tickangle=45)
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Gender equity analysis
    if 'MonthlyIncome' in data.columns:
        st.markdown("#### ⚖️ Gender Pay Equity")
        
        gender_salary = data.groupby('Gender')['MonthlyIncome'].agg(['mean', 'median', 'count'])
        
        # Display as metrics
        cols = st.columns(len(gender_salary))
        
        for i, (gender, stats) in enumerate(gender_salary.iterrows()):
            with cols[i]:
                st.metric(
                    f"{gender} Avg Salary",
                    f"${stats['mean']:,.0f}",
                    f"Median: ${stats['median']:,.0f}"
                )
    
    # Attrition by gender
    if include_attrition and 'Attrition' in data.columns:
        st.markdown("#### 📊 Attrition by Gender")
        
        gender_attrition = data.groupby('Gender')['Attrition'].apply(
            lambda x: (x == 'Yes').mean()
        )
        
        for gender, rate in gender_attrition.items():
            color = get_color('error') if rate > 0.2 else get_color('warning') if rate > 0.15 else get_color('success')
            
            st.markdown(f"""
            <div style="
                padding: 15px;
                background: rgba(37, 42, 69, 0.3);
                border-radius: 8px;
                border-left: 3px solid {color};
                margin: 10px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div>
                    <span style="color: {get_color('text')}; font-weight: bold;">{gender}</span>
                    <br>
                    <span style="color: {get_color('text_secondary')}; font-size: 12px;">
                        {gender_counts[gender]} employees
                    </span>
                </div>
                <div style="text-align: right;">
                    <div style="color: {color}; font-weight: bold; font-size: 18px;">
                        {rate:.1%}
                    </div>
                    <div style="color: {get_color('text_secondary')}; font-size: 12px;">
                        Attrition Rate
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def _analyze_cross_demographics(data, include_attrition):
    """Cross-demographic analysis."""
    
    st.markdown("#### 🔄 Cross-Demographic Analysis")
    
    # Select dimensions for analysis
    col1, col2 = st.columns(2)
    
    available_dims = ['Department', 'Gender', 'AgeGroup', 'Education', 'TenureGroup', 'SalaryGroup']
    available_dims = [dim for dim in available_dims if dim in data.columns]
    
    with col1:
        dim1 = st.selectbox("Primary Dimension:", available_dims, index=0 if available_dims else None)
    
    with col2:
        dim2_options = [dim for dim in available_dims if dim != dim1]
        dim2 = st.selectbox("Secondary Dimension:", dim2_options, index=0 if dim2_options else None)
    
    if not dim1 or not dim2:
        st.warning("Need at least 2 dimensions for cross-analysis")
        return
    
    # Cross-tabulation
    crosstab = pd.crosstab(data[dim1], data[dim2])
    
    # Heatmap visualization
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=crosstab.values,
        x=crosstab.columns,
        y=crosstab.index,
        colorscale='Blues',
        text=crosstab.values,
        texttemplate='%{text}',
        textfont=dict(color=get_color('text')),
        hovertemplate='<b>%{y} × %{x}</b><br>Count: %{z}<extra></extra>',
        showscale=True
    ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title=f"Cross-Analysis: {dim1} × {dim2}",
        height=500,
        custom_layout={
            'xaxis_title': dim2,
            'yaxis_title': dim1
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical insights
    chi2_stat, p_value = stats.chi2_contingency(crosstab)[:2]
    
    st.markdown(f"""
    <div style="
        padding: 15px;
        background: rgba(37, 42, 69, 0.4);
        border-radius: 10px;
        border-left: 4px solid {get_color('info')};
        margin: 20px 0;
    ">
        <h5 style="color: {get_color('info')}; margin: 0 0 10px 0;">Statistical Independence Test</h5>
        <div style="color: {get_color('text_secondary')};">
            Chi-square statistic: {chi2_stat:.2f}<br>
            P-value: {p_value:.4f}<br>
            {'<strong style="color:' + get_color('error') + ';">Significant relationship detected</strong>' if p_value < 0.05 else '<strong style="color:' + get_color('success') + ';">No significant relationship</strong>'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# SATISFACTION ANALYSIS
# ================================================================

def satisfaction_analysis():
    """Comprehensive satisfaction analysis."""
    
    st.markdown("### 😊 Employee Satisfaction Analysis")
    
    # Load data
    data, is_real = load_analytics_data()
    
    # Check for satisfaction columns
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'RelationshipSatisfaction']
    available_sat_cols = [col for col in satisfaction_cols if col in data.columns]
    
    if not available_sat_cols:
        st.warning("No satisfaction data available")
        return
    
    # Satisfaction analysis options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type:",
            [
                "Overall Satisfaction Overview",
                "Multi-Dimensional Satisfaction",
                "Satisfaction Correlations",
                "Satisfaction Trends",
                "Department Satisfaction Analysis"
            ]
        )
    
    with col2:
        primary_satisfaction = st.selectbox(
            "Primary Satisfaction Metric:",
            available_sat_cols,
            index=0
        )
    
    # Execute analysis based on type
    if analysis_type == "Overall Satisfaction Overview":
        _analyze_overall_satisfaction(data, primary_satisfaction)
    elif analysis_type == "Multi-Dimensional Satisfaction":
        _analyze_multi_dimensional_satisfaction(data, available_sat_cols)
    elif analysis_type == "Satisfaction Correlations":
        _analyze_satisfaction_correlations(data, available_sat_cols)
    elif analysis_type == "Satisfaction Trends":
        _analyze_satisfaction_trends(data, available_sat_cols)
    else:  # Department Satisfaction Analysis
        _analyze_department_satisfaction(data, available_sat_cols)

def _analyze_overall_satisfaction(data, primary_satisfaction):
    """Analyze overall satisfaction patterns."""
    
    # Satisfaction overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_satisfaction = data[primary_satisfaction].mean()
        st.markdown(glassmorphism_metric_card(
            value=f"{avg_satisfaction:.2f}",
            title="Average Score",
            subtitle="Out of 5.0",
            icon="😊",
            color='secondary'
        ), unsafe_allow_html=True)
    
    with col2:
        high_satisfaction = (data[primary_satisfaction] >= 4).mean()
        st.markdown(glassmorphism_metric_card(
            value=f"{high_satisfaction:.1%}",
            title="High Satisfaction",
            subtitle="Rating ≥ 4",
            icon="😍",
            color='success'
        ), unsafe_allow_html=True)
    
    with col3:
        low_satisfaction = (data[primary_satisfaction] <= 2).mean()
        st.markdown(glassmorphism_metric_card(
            value=f"{low_satisfaction:.1%}",
            title="Low Satisfaction",
            subtitle="Rating ≤ 2",
            icon="😞",
            color='error'
        ), unsafe_allow_html=True)
    
    with col4:
        satisfaction_std = data[primary_satisfaction].std()
        st.markdown(glassmorphism_metric_card(
            value=f"{satisfaction_std:.2f}",
            title="Variability",
            subtitle="Standard Dev",
            icon="📊",
            color='info'
        ), unsafe_allow_html=True)
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution histogram
        fig = go.Figure()
        
        # Create histogram with custom colors based on satisfaction level
        satisfaction_counts = data[primary_satisfaction].value_counts().sort_index()
        colors = []
        for rating in satisfaction_counts.index:
            if rating >= 4:
                colors.append(get_color('success'))
            elif rating >= 3:
                colors.append(get_color('warning'))
            else:
                colors.append(get_color('error'))
        
        fig.add_trace(go.Bar(
            x=satisfaction_counts.index,
            y=satisfaction_counts.values,
            marker=dict(color=colors),
            text=satisfaction_counts.values,
            textposition='outside',
            hovertemplate='<b>Rating: %{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=(satisfaction_counts.values / satisfaction_counts.sum() * 100)
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title=f"{primary_satisfaction.replace('Satisfaction', '')} Distribution",
            height=400,
            custom_layout={
                'xaxis_title': 'Satisfaction Rating',
                'yaxis_title': 'Number of Employees'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Satisfaction gauge
        gauge_fig = futuristic_gauge_chart(
            value=avg_satisfaction * 20,  # Convert 1-5 scale to 0-100
            title="Satisfaction Level",
            min_value=0,
            max_value=100,
            unit="",
            risk_thresholds={'low': 40, 'medium': 70, 'high': 100},
            height=400
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Satisfaction vs Attrition Analysis
    if 'Attrition' in data.columns:
        st.markdown("#### 📊 Satisfaction Impact on Attrition")
        
        satisfaction_attrition = data.groupby(primary_satisfaction).agg({
            'Attrition': lambda x: (x == 'Yes').mean(),
            primary_satisfaction: 'count'
        }).reset_index()
        satisfaction_attrition.columns = [primary_satisfaction, 'AttritionRate', 'Count']
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Attrition rate line
        fig.add_trace(
            go.Scatter(
                x=satisfaction_attrition[primary_satisfaction],
                y=satisfaction_attrition['AttritionRate'] * 100,
                mode='lines+markers',
                name='Attrition Rate (%)',
                line=dict(color=get_color('error'), width=3),
                marker=dict(size=10)
            ),
            secondary_y=False,
        )
        
        # Employee count bars
        fig.add_trace(
            go.Bar(
                x=satisfaction_attrition[primary_satisfaction],
                y=satisfaction_attrition['Count'],
                name='Employee Count',
                marker=dict(color=get_color('secondary'), opacity=0.6),
                yaxis='y2'
            ),
            secondary_y=True,
        )
        
        fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Employees", secondary_y=True)
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title=f"Attrition Rate vs {primary_satisfaction}",
            height=400,
            custom_layout={
                'xaxis_title': 'Satisfaction Rating',
                'hovermode': 'x unified'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation coefficient
        correlation = data[primary_satisfaction].corr(data['Attrition'].map({'Yes': 1, 'No': 0}))
        
        st.markdown(f"""
        <div style="
            padding: 15px;
            background: rgba(37, 42, 69, 0.4);
            border-radius: 10px;
            border-left: 4px solid {get_color('info')};
            margin: 20px 0;
        ">
            <h5 style="color: {get_color('info')}; margin: 0 0 10px 0;">Satisfaction-Attrition Correlation</h5>
            <div style="color: {get_color('text')};">
                Correlation coefficient: <strong>{correlation:.3f}</strong><br>
                <span style="color: {get_color('text_secondary')}; font-size: 14px;">
                    {'Strong negative correlation - higher satisfaction = lower attrition' if correlation < -0.5 else 
                     'Moderate negative correlation' if correlation < -0.3 else 
                     'Weak correlation' if abs(correlation) < 0.3 else 
                     'Positive correlation - investigate further'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Satisfaction vs Salary Analysis (your existing code enhanced)
    if 'MonthlyIncome' in data.columns:
        st.markdown("#### 💰 Satisfaction vs Salary Relationship")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot (your existing code)
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data[primary_satisfaction],
                y=data['MonthlyIncome'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data['MonthlyIncome'],
                    colorscale='Viridis',
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(title="Salary ($)")
                ),
                hovertemplate='Satisfaction: %{x}<br>Salary: $%{y:,.0f}<extra></extra>'
            ))
            
            # Add trendline
            z = np.polyfit(data[primary_satisfaction], data['MonthlyIncome'], 1)
            trendline = np.poly1d(z)
            x_trend = np.linspace(data[primary_satisfaction].min(), data[primary_satisfaction].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=trendline(x_trend),
                mode='lines',
                name='Trend',
                line=dict(color=get_color('accent'), width=3, dash='dash')
            ))
            
            fig = create_dark_theme_plotly_chart(
                fig,
                title="Job Satisfaction vs Monthly Income",
                height=400,
                custom_layout={
                    'xaxis_title': f'{primary_satisfaction} (1-5)',
                    'yaxis_title': 'Monthly Income ($)'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average salary by satisfaction level
            salary_by_satisfaction = data.groupby(primary_satisfaction)['MonthlyIncome'].agg(['mean', 'median', 'std']).round(0)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=salary_by_satisfaction.index,
                y=salary_by_satisfaction['mean'],
                name='Mean Salary',
                marker=dict(color=get_color('success')),
                text=[f'${val:,.0f}' for val in salary_by_satisfaction['mean']],
                textposition='outside'
            ))
            
            fig.add_trace(go.Scatter(
                x=salary_by_satisfaction.index,
                y=salary_by_satisfaction['median'],
                mode='lines+markers',
                name='Median Salary',
                line=dict(color=get_color('warning'), width=3),
                marker=dict(size=8)
            ))
            
            fig = create_dark_theme_plotly_chart(
                fig,
                title="Average Salary by Satisfaction Level",
                height=400,
                custom_layout={
                    'xaxis_title': 'Satisfaction Rating',
                    'yaxis_title': 'Monthly Income ($)'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)

def _analyze_multi_dimensional_satisfaction(data, satisfaction_cols):
    """Analyze multiple satisfaction dimensions."""
    
    st.markdown("#### 🎯 Multi-Dimensional Satisfaction Analysis")
    
    if len(satisfaction_cols) < 2:
        st.warning("Need at least 2 satisfaction dimensions for multi-dimensional analysis")
        return
    
    # Calculate composite satisfaction score
    if 'SatisfactionComposite' not in data.columns:
        data['SatisfactionComposite'] = data[satisfaction_cols].mean(axis=1)
    
    # Satisfaction radar chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create radar chart for overall satisfaction
        if VISUALIZATIONS_AVAILABLE:
            radar_fig = create_satisfaction_radar_chart(
                data, satisfaction_cols, height=500,
                title="Overall Satisfaction Profile"
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            # Fallback bar chart
            avg_satisfactions = data[satisfaction_cols].mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[col.replace('Satisfaction', '').replace('WorkLife', 'Work-Life') for col in avg_satisfactions.index],
                y=avg_satisfactions.values,
                marker=dict(color=get_color('secondary')),
                text=[f'{val:.2f}' for val in avg_satisfactions.values],
                textposition='outside'
            ))
            
            fig = create_dark_theme_plotly_chart(
                fig,
                title="Average Satisfaction Scores",
                height=500,
                custom_layout={
                    'xaxis_title': 'Satisfaction Dimension',
                    'yaxis_title': 'Average Score (1-5)',
                    'xaxis': dict(tickangle=45)
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Satisfaction statistics
        st.markdown("**Satisfaction Metrics**")
        
        for col in satisfaction_cols:
            avg_score = data[col].mean()
            color = get_color('success') if avg_score >= 3.5 else get_color('warning') if avg_score >= 3 else get_color('error')
            
            st.markdown(f"""
            <div style="
                padding: 10px;
                background: rgba(37, 42, 69, 0.3);
                border-radius: 6px;
                border-left: 3px solid {color};
                margin: 8px 0;
            ">
                <div style="color: {get_color('text')}; font-size: 14px; font-weight: bold;">
                    {col.replace('Satisfaction', '').replace('WorkLife', 'Work-Life')}
                </div>
                <div style="color: {color}; font-size: 18px; font-weight: bold;">
                    {avg_score:.2f} / 5.0
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Composite score
        composite_avg = data['SatisfactionComposite'].mean()
        composite_color = get_color('success') if composite_avg >= 3.5 else get_color('warning') if composite_avg >= 3 else get_color('error')
        
        st.markdown(f"""
        <div style="
            padding: 15px;
            background: rgba(37, 42, 69, 0.4);
            border-radius: 10px;
            border: 2px solid {composite_color};
            margin: 15px 0;
            text-align: center;
        ">
            <div style="color: {get_color('text')}; font-size: 16px; margin-bottom: 5px;">
                <strong>Overall Satisfaction</strong>
            </div>
            <div style="color: {composite_color}; font-size: 24px; font-weight: bold;">
                {composite_avg:.2f} / 5.0
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Satisfaction correlation matrix
    st.markdown("#### 🔗 Satisfaction Dimension Correlations")
    
    satisfaction_corr = data[satisfaction_cols].corr()
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=satisfaction_corr.values,
        x=[col.replace('Satisfaction', '') for col in satisfaction_corr.columns],
        y=[col.replace('Satisfaction', '') for col in satisfaction_corr.columns],
        colorscale='RdYlBu_r',
        zmid=0,
        text=np.round(satisfaction_corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(color=get_color('text')),
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>',
        showscale=True
    ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title="Satisfaction Dimensions Correlation Matrix",
        height=400,
        custom_layout={
            'xaxis': dict(tickangle=45),
            'yaxis': dict(tickangle=0)
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _analyze_satisfaction_correlations(data, satisfaction_cols):
    """Analyze correlations between satisfaction and other factors."""
    
    st.markdown("#### 🔗 Satisfaction Correlations Analysis")
    
    # Select columns for correlation analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove satisfaction columns from correlation targets
    other_factors = [col for col in numeric_cols if col not in satisfaction_cols and not col.lower().endswith('id')]
    
    if not other_factors:
        st.warning("No other numeric factors available for correlation analysis")
        return
    
    # Calculate correlations
    correlation_results = []
    
    for sat_col in satisfaction_cols:
        for factor in other_factors:
            correlation = data[sat_col].corr(data[factor])
            if not np.isnan(correlation):
                correlation_results.append({
                    'Satisfaction_Type': sat_col.replace('Satisfaction', ''),
                    'Factor': factor,
                    'Correlation': correlation,
                    'Abs_Correlation': abs(correlation)
                })
    
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df = correlation_df.sort_values('Abs_Correlation', ascending=False)
    
    # Display top correlations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔗 Strongest Positive Correlations**")
        
        positive_corr = correlation_df[correlation_df['Correlation'] > 0].head(5)
        
        for _, row in positive_corr.iterrows():
            st.markdown(f"""
            <div style="
                padding: 10px;
                background: rgba(0, 255, 136, 0.1);
                border-radius: 6px;
                border-left: 3px solid {get_color('success')};
                margin: 5px 0;
            ">
                <div style="color: {get_color('text')}; font-size: 14px;">
                    <strong>{row['Satisfaction_Type']}</strong> ↔ {row['Factor']}
                </div>
                <div style="color: {get_color('success')}; font-weight: bold;">
                    +{row['Correlation']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**⚡ Strongest Negative Correlations**")
        
        negative_corr = correlation_df[correlation_df['Correlation'] < 0].head(5)
        
        for _, row in negative_corr.iterrows():
            st.markdown(f"""
            <div style="
                padding: 10px;
                background: rgba(255, 45, 117, 0.1);
                border-radius: 6px;
                border-left: 3px solid {get_color('error')};
                margin: 5px 0;
            ">
                <div style="color: {get_color('text')}; font-size: 14px;">
                    <strong>{row['Satisfaction_Type']}</strong> ↔ {row['Factor']}
                </div>
                <div style="color: {get_color('error')}; font-weight: bold;">
                    {row['Correlation']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Correlation heatmap
    if len(other_factors) <= 10:  # Limit for readability
        pivot_corr = correlation_df.pivot(index='Satisfaction_Type', columns='Factor', values='Correlation')
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=pivot_corr.values,
            x=pivot_corr.columns,
            y=pivot_corr.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(pivot_corr.values, 2),
            texttemplate='%{text}',
            textfont=dict(color=get_color('text')),
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>',
            showscale=True
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Satisfaction vs Other Factors Correlation Heatmap",
            height=400,
            custom_layout={
                'xaxis': dict(tickangle=45),
                'yaxis': dict(tickangle=0)
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)

def _analyze_satisfaction_trends(data, satisfaction_cols):
    """Analyze satisfaction trends and patterns."""
    
    st.markdown("#### 📈 Satisfaction Trends Analysis")
    
    # Simulate time-based trends (in real implementation, you'd have date columns)
    st.info("📅 Simulated trends - in production, connect to actual date data")
    
    # Create synthetic monthly data
    months = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
    
    # Generate realistic trends for each satisfaction dimension
    trend_data = pd.DataFrame({'Month': months})
    
    for sat_col in satisfaction_cols:
        base_satisfaction = data[sat_col].mean()
        
        # Add seasonal trend and noise
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(len(months)) / 12)
        trend = np.linspace(-0.1, 0.1, len(months))  # Slight improvement over year
        noise = np.random.normal(0, 0.05, len(months))
        
        satisfaction_trend = base_satisfaction + seasonal + trend + noise
        satisfaction_trend = np.clip(satisfaction_trend, 1, 5)
        
        trend_data[sat_col] = satisfaction_trend
    
    # Plot trends
    fig = go.Figure()
    
    colors = [get_color('secondary'), get_color('accent'), get_color('success'), get_color('warning')]
    
    for i, sat_col in enumerate(satisfaction_cols):
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data[sat_col],
            mode='lines+markers',
            name=sat_col.replace('Satisfaction', ''),
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=6)
        ))
    
    fig = create_dark_theme_plotly_chart(
        fig,
        title="Satisfaction Trends Over Time",
        height=500,
        custom_layout={
            'xaxis_title': 'Month',
            'yaxis_title': 'Satisfaction Score (1-5)',
            'hovermode': 'x unified'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Overall trend direction
        overall_trend = "📈 Improving" if trend_data[satisfaction_cols].iloc[-1].mean() > trend_data[satisfaction_cols].iloc[0].mean() else "📉 Declining"
        st.metric("Overall Trend", overall_trend)
    
    with col2:
        # Most volatile dimension
        volatilities = trend_data[satisfaction_cols].std()
        most_volatile = volatilities.idxmax().replace('Satisfaction', '')
        st.metric("Most Volatile", most_volatile)
    
    with col3:
        # Most stable dimension
        most_stable = volatilities.idxmin().replace('Satisfaction', '')
        st.metric("Most Stable", most_stable)

def _analyze_department_satisfaction(data, satisfaction_cols):
    """Analyze satisfaction patterns by department."""
    
    st.markdown("#### 🏢 Department Satisfaction Analysis")
    
    if 'Department' not in data.columns:
        st.warning("Department data not available")
        return
    
    # Department satisfaction summary
    dept_satisfaction = data.groupby('Department')[satisfaction_cols].mean().round(2)
    
    # Overall satisfaction by department
    dept_satisfaction['Overall'] = dept_satisfaction.mean(axis=1)
    dept_satisfaction_sorted = dept_satisfaction.sort_values('Overall', ascending=False)
    
    # Display department rankings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Heatmap of all satisfaction dimensions by department
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=dept_satisfaction_sorted[satisfaction_cols].values,
            x=[col.replace('Satisfaction', '') for col in satisfaction_cols],
            y=dept_satisfaction_sorted.index,
            colorscale='RdYlGn',
            zmin=1,
            zmax=5,
            text=np.round(dept_satisfaction_sorted[satisfaction_cols].values, 2),
            texttemplate='%{text}',
            textfont=dict(color=get_color('text')),
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>',
            showscale=True,
            colorbar=dict(title="Satisfaction Score")
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title="Department Satisfaction Heatmap",
            height=500,
            custom_layout={
                'xaxis': dict(tickangle=45),
                'yaxis': dict(tickangle=0)
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Department rankings
        st.markdown("**🏆 Department Rankings**")
        
        for i, (dept, scores) in enumerate(dept_satisfaction_sorted.iterrows(), 1):
            overall_score = scores['Overall']
            
            if overall_score >= 4:
                color = get_color('success')
                icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "✅"
            elif overall_score >= 3.5:
                color = get_color('warning')
                icon = "🟡"
            else:
                color = get_color('error')
                icon = "🔴"
            
            st.markdown(f"""
            <div style="
                padding: 12px;
                background: rgba(37, 42, 69, 0.3);
                border-radius: 8px;
                border-left: 3px solid {color};
                margin: 8px 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: {get_color('text')}; font-weight: bold;">
                            {icon} #{i} {dept}
                        </div>
                        <div style="color: {get_color('text_secondary')}; font-size: 12px;">
                            {data[data['Department'] == dept].shape[0]} employees
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {color}; font-weight: bold; font-size: 16px;">
                            {overall_score:.2f}
                        </div>
                        <div style="color: {get_color('text_secondary')}; font-size: 10px;">
                            / 5.0
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Best and worst performing departments analysis
    st.markdown("#### 📊 Department Performance Analysis")
    
    best_dept = dept_satisfaction_sorted.index[0]
    worst_dept = dept_satisfaction_sorted.index[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**🏆 Best Performing: {best_dept}**")
        
        best_scores = dept_satisfaction_sorted.loc[best_dept, satisfaction_cols]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[col.replace('Satisfaction', '') for col in best_scores.index],
            y=best_scores.values,
            marker=dict(color=get_color('success')),
            text=[f'{val:.2f}' for val in best_scores.values],
            textposition='outside'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title=f"{best_dept} Satisfaction Breakdown",
            height=350,
            custom_layout={
                'xaxis_title': 'Satisfaction Dimension',
                'yaxis_title': 'Score (1-5)',
                'xaxis': dict(tickangle=45)
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"**🎯 Needs Improvement: {worst_dept}**")
        
        worst_scores = dept_satisfaction_sorted.loc[worst_dept, satisfaction_cols]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[col.replace('Satisfaction', '') for col in worst_scores.index],
            y=worst_scores.values,
            marker=dict(color=get_color('warning')),
            text=[f'{val:.2f}' for val in worst_scores.values],
            textposition='outside'
        ))
        
        fig = create_dark_theme_plotly_chart(
            fig,
            title=f"{worst_dept} Satisfaction Breakdown",
            height=350,
            custom_layout={
                'xaxis_title': 'Satisfaction Dimension',
                'yaxis_title': 'Score (1-5)',
                'xaxis': dict(tickangle=45)
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Improvement recommendations
    st.markdown("#### 💡 Satisfaction Improvement Recommendations")
    
    # Find lowest satisfaction areas across all departments
    lowest_scores = dept_satisfaction[satisfaction_cols].min()
    improvement_priorities = lowest_scores.sort_values().head(3)
    
    for i, (dimension, score) in enumerate(improvement_priorities.items(), 1):
        worst_dept_for_dimension = dept_satisfaction[dimension].idxmin()
        
        st.markdown(f"""
        <div style="
            padding: 15px;
            background: rgba(37, 42, 69, 0.4);
            border-radius: 10px;
            border-left: 4px solid {get_color('warning')};
            margin: 10px 0;
        ">
            <h5 style="color: {get_color('warning')}; margin: 0 0 10px 0;">
                Priority #{i}: {dimension.replace('Satisfaction', ' Satisfaction')}
            </h5>
            <div style="color: {get_color('text')};">
                Lowest score: <strong>{score:.2f}/5.0</strong> in <strong>{worst_dept_for_dimension}</strong><br>
                <span style="color: {get_color('text_secondary')}; font-size: 14px;">
                    Recommended actions: 
                    {
                        'Focus on job role clarity and career development opportunities' if 'Job' in dimension else
                        'Improve workplace environment and facilities' if 'Environment' in dimension else
                        'Implement flexible work arrangements and wellness programs' if 'WorkLife' in dimension else
                        'Enhance team dynamics and communication channels'
                    }
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# MAIN ANALYTICS FUNCTION
# ================================================================

def show():
    """Main analytics function called by the navigation system."""
    
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
                📊 Deep Analytics Dashboard
            </h1>
            <p style="color: #B8C5D1; font-size: 1.1rem;">
                Comprehensive data analysis with interactive visualizations and statistical insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main analytics tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔗 Correlation Analysis",
            "💰 Salary Analysis", 
            "👥 Demographics",
            "😊 Satisfaction Analysis"
        ])
        
        with tab1:
            correlation_matrix()
        
        with tab2:
            salary_distribution_analysis()
        
        with tab3:
            demographic_breakdowns()
        
        with tab4:
            satisfaction_analysis()
        
        # Memory cleanup
        gc.collect()
        
    except Exception as e:
        st.error(f"Analytics page error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")

# ================================================================
# ENTRY POINT FOR TESTING
# ================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Analytics", layout="wide")
    show()
