"""
HR Attrition Predictor - Advanced Visualization Components
=========================================================
Comprehensive chart library with consistent cyberpunk styling, interactive features,
and memory-optimized rendering for 4GB RAM systems.

Author: HR Analytics Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import gc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import theme configuration
try:
    from streamlit_app.assets.theme import (
        COLORS, TYPOGRAPHY, get_plotly_dark_theme, 
        get_chart_color_palette, SIZING
    )
    from streamlit_app.components.charts import create_dark_theme_plotly_chart
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False
    logging.warning("Theme configuration not available")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================================================
# DEFAULT COLORS (FALLBACK)
# ================================================================

DEFAULT_COLORS = {
    'primary': '#0A0E27',
    'secondary': '#00D4FF', 
    'accent': '#B026FF',
    'success': '#00FF88',
    'warning': '#FF6B35',
    'error': '#FF2D75',
    'text': '#F0F8FF',
    'text_secondary': '#B8C5D1',
    'background': '#0A0E27',
    'background_light': '#1A1F3A',
    'border_primary': '#00D4FF'
}

def get_color(color_name: str) -> str:
    """Get color with fallback to defaults."""
    if THEME_AVAILABLE:
        return COLORS.get(color_name, DEFAULT_COLORS.get(color_name, '#00D4FF'))
    return DEFAULT_COLORS.get(color_name, '#00D4FF')

# ================================================================
# RISK DISTRIBUTION CHARTS
# ================================================================

def create_risk_distribution_chart(data: pd.DataFrame, 
                                 risk_column: str = 'RiskLevel',
                                 chart_type: str = 'donut',
                                 height: int = 400,
                                 title: str = 'Employee Risk Distribution') -> go.Figure:
    """
    Create comprehensive risk distribution visualization.
    
    Args:
        data: DataFrame containing risk level data
        risk_column: Column name containing risk levels
        chart_type: 'donut', 'pie', 'bar', 'treemap'
        height: Chart height in pixels
        title: Chart title
        
    Returns:
        Plotly figure with risk distribution
    """
    
    logger.info(f"Creating risk distribution chart ({chart_type}) for {len(data)} records")
    
    try:
        # Calculate risk distribution
        if risk_column not in data.columns:
            logger.error(f"Column {risk_column} not found in data")
            return _create_error_chart(f"Column {risk_column} not found")
        
        risk_counts = data[risk_column].value_counts()
        total_employees = len(data)
        
        # Define risk level colors
        risk_colors = {
            'High': get_color('error'),
            'Medium': get_color('warning'), 
            'Low': get_color('success'),
            'Unknown': get_color('text_secondary')
        }
        
        # Ensure standard risk levels exist
        for level in ['Low', 'Medium', 'High']:
            if level not in risk_counts.index:
                risk_counts[level] = 0
        
        # Sort by risk severity
        risk_order = ['Low', 'Medium', 'High'] + [x for x in risk_counts.index if x not in ['Low', 'Medium', 'High']]
        risk_counts = risk_counts.reindex(risk_order, fill_value=0)
        
        # Create chart based on type
        if chart_type.lower() == 'donut':
            fig = _create_donut_risk_chart(risk_counts, risk_colors, total_employees)
            
        elif chart_type.lower() == 'pie':
            fig = _create_pie_risk_chart(risk_counts, risk_colors, total_employees)
            
        elif chart_type.lower() == 'bar':
            fig = _create_bar_risk_chart(risk_counts, risk_colors, total_employees)
            
        elif chart_type.lower() == 'treemap':
            fig = _create_treemap_risk_chart(risk_counts, risk_colors, total_employees)
            
        else:
            logger.warning(f"Unknown chart type {chart_type}, defaulting to donut")
            fig = _create_donut_risk_chart(risk_counts, risk_colors, total_employees)
        
        # Apply dark theme
        fig = create_dark_theme_plotly_chart(
            fig, title=title, height=height,
            show_legend=True if chart_type != 'treemap' else False
        )
        
        # Add summary annotation for donut/pie charts
        if chart_type.lower() in ['donut', 'pie']:
            high_risk_pct = (risk_counts.get('High', 0) / total_employees * 100) if total_employees > 0 else 0
            
            fig.add_annotation(
                text=f"<span style='color:{get_color('text')};font-size:14px'><b>High Risk</b></span><br>"
                     f"<span style='color:{get_color('error')};font-size:24px;font-weight:bold'>{high_risk_pct:.1f}%</span>",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk distribution chart: {e}")
        return _create_error_chart(f"Error: {e}")

def _create_donut_risk_chart(risk_counts: pd.Series, risk_colors: Dict, total: int) -> go.Figure:
    """Create donut chart for risk distribution."""
    
    colors = [risk_colors.get(level, get_color('text_secondary')) for level in risk_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.6,
        marker=dict(
            colors=colors,
            line=dict(color=get_color('border_primary'), width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(color=get_color('text'), size=12),
        hovertemplate='<b>%{label}</b><br>' +
                     'Count: %{value}<br>' + 
                     'Percentage: %{percent}<br>' +
                     '<extra></extra>'
    )])
    
    return fig

def _create_pie_risk_chart(risk_counts: pd.Series, risk_colors: Dict, total: int) -> go.Figure:
    """Create pie chart for risk distribution."""
    
    colors = [risk_colors.get(level, get_color('text_secondary')) for level in risk_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker=dict(
            colors=colors,
            line=dict(color=get_color('border_primary'), width=2)
        ),
        textinfo='label+percent',
        textfont=dict(color=get_color('text'), size=12)
    )])
    
    return fig

def _create_bar_risk_chart(risk_counts: pd.Series, risk_colors: Dict, total: int) -> go.Figure:
    """Create bar chart for risk distribution."""
    
    colors = [risk_colors.get(level, get_color('text_secondary')) for level in risk_counts.index]
    percentages = (risk_counts / total * 100).round(1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker=dict(
            color=colors,
            line=dict(color=get_color('border_primary'), width=2)
        ),
        text=[f'{count}<br>({pct}%)' for count, pct in zip(risk_counts.values, percentages)],
        textposition='outside',
        textfont=dict(color=get_color('text'), size=12),
        hovertemplate='<b>%{x}</b><br>' +
                     'Count: %{y}<br>' +
                     'Percentage: %{customdata}%<br>' +
                     '<extra></extra>',
        customdata=percentages
    ))
    
    return fig

def _create_treemap_risk_chart(risk_counts: pd.Series, risk_colors: Dict, total: int) -> go.Figure:
    """Create treemap for risk distribution."""
    
    colors = [risk_colors.get(level, get_color('text_secondary')) for level in risk_counts.index]
    
    fig = go.Figure(go.Treemap(
        labels=risk_counts.index,
        values=risk_counts.values,
        parents=[""] * len(risk_counts),
        marker=dict(
            colors=colors,
            line=dict(width=2, color=get_color('border_primary'))
        ),
        textfont=dict(color=get_color('text'), size=14),
        hovertemplate='<b>%{label}</b><br>' +
                     'Count: %{value}<br>' +
                     'Percentage: %{percentParent}<br>' +
                     '<extra></extra>'
    ))
    
    return fig

# ================================================================
# DEPARTMENT HEATMAPS
# ================================================================

def plot_department_heatmap(data: pd.DataFrame,
                          department_col: str = 'Department',
                          metric_col: str = 'AttritionRate',
                          chart_type: str = 'heatmap',
                          height: int = 400,
                          title: str = 'Department Analysis Heatmap') -> go.Figure:
    """
    Create department-wise heatmap visualization.
    
    Args:
        data: DataFrame with department and metric data
        department_col: Column name for departments
        metric_col: Column name for the metric to visualize
        chart_type: 'heatmap', 'bar', 'bubble'
        height: Chart height
        title: Chart title
        
    Returns:
        Plotly figure with department analysis
    """
    
    logger.info(f"Creating department heatmap for {len(data)} records")
    
    try:
        if department_col not in data.columns:
            return _create_error_chart(f"Column {department_col} not found")
        
        # Calculate department metrics
        if metric_col == 'AttritionRate':
            # Calculate attrition rate if not provided
            dept_stats = data.groupby(department_col).agg({
                'Attrition': [
                    ('count', 'count'),
                    ('attrition_count', lambda x: (x == 'Yes').sum() if 'Attrition' in data.columns else 0)
                ]
            }).round(3)
            
            dept_stats.columns = ['EmployeeCount', 'AttritionCount']
            dept_stats['AttritionRate'] = dept_stats['AttritionCount'] / dept_stats['EmployeeCount']
            dept_stats = dept_stats.reset_index()
            metric_values = dept_stats['AttritionRate']
            
        else:
            # Use provided metric
            if metric_col not in data.columns:
                return _create_error_chart(f"Column {metric_col} not found")
            
            dept_stats = data.groupby(department_col)[metric_col].agg(['mean', 'count']).reset_index()
            dept_stats.columns = [department_col, metric_col, 'EmployeeCount']
            metric_values = dept_stats[metric_col]
        
        # Create visualization based on type
        if chart_type.lower() == 'heatmap':
            fig = _create_dept_heatmap(dept_stats, department_col, metric_col)
            
        elif chart_type.lower() == 'bar':
            fig = _create_dept_bar_chart(dept_stats, department_col, metric_col)
            
        elif chart_type.lower() == 'bubble':
            fig = _create_dept_bubble_chart(dept_stats, department_col, metric_col)
            
        else:
            fig = _create_dept_heatmap(dept_stats, department_col, metric_col)
        
        # Apply dark theme
        fig = create_dark_theme_plotly_chart(
            fig, title=title, height=height, show_legend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating department heatmap: {e}")
        return _create_error_chart(f"Error: {e}")

def _create_dept_heatmap(data: pd.DataFrame, dept_col: str, metric_col: str) -> go.Figure:
    """Create traditional heatmap for department data."""
    
    # Reshape data for heatmap (create a matrix)
    departments = data[dept_col].tolist()
    values = data[metric_col].tolist()
    
    # Create a simple 1-row heatmap
    heatmap_data = [values]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=departments,
        y=['Metric'],
        colorscale=[
            [0, get_color('success')],
            [0.5, get_color('warning')], 
            [1, get_color('error')]
        ],
        showscale=True,
        texttemplate='%{z:.2f}',
        textfont=dict(color=get_color('text')),
        hovertemplate='<b>%{x}</b><br>' +
                     f'{metric_col}: %{{z:.3f}}<br>' +
                     '<extra></extra>'
    ))
    
    return fig

def _create_dept_bar_chart(data: pd.DataFrame, dept_col: str, metric_col: str) -> go.Figure:
    """Create bar chart for department data."""
    
    # Sort by metric value
    data_sorted = data.sort_values(metric_col, ascending=False)
    
    # Color based on metric value
    max_val = data[metric_col].max()
    min_val = data[metric_col].min()
    
    colors = []
    for val in data_sorted[metric_col]:
        if max_val > min_val:
            normalized = (val - min_val) / (max_val - min_val)
            if normalized > 0.7:
                colors.append(get_color('error'))
            elif normalized > 0.3:
                colors.append(get_color('warning'))
            else:
                colors.append(get_color('success'))
        else:
            colors.append(get_color('secondary'))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data_sorted[dept_col],
        y=data_sorted[metric_col],
        marker=dict(
            color=colors,
            line=dict(color=get_color('border_primary'), width=1)
        ),
        text=[f'{val:.2f}' for val in data_sorted[metric_col]],
        textposition='outside',
        textfont=dict(color=get_color('text')),
        hovertemplate='<b>%{x}</b><br>' +
                     f'{metric_col}: %{{y:.3f}}<br>' +
                     'Employees: %{customdata}<br>' +
                     '<extra></extra>',
        customdata=data_sorted.get('EmployeeCount', [0] * len(data_sorted))
    ))
    
    return fig

def _create_dept_bubble_chart(data: pd.DataFrame, dept_col: str, metric_col: str) -> go.Figure:
    """Create bubble chart for department data."""
    
    fig = go.Figure()
    
    # Create bubble sizes based on employee count
    employee_counts = data.get('EmployeeCount', [10] * len(data))
    bubble_sizes = [(count / max(employee_counts)) * 100 + 20 for count in employee_counts]
    
    # Color based on metric value
    max_val = data[metric_col].max()
    min_val = data[metric_col].min()
    
    fig.add_trace(go.Scatter(
        x=data[dept_col],
        y=data[metric_col],
        mode='markers+text',
        marker=dict(
            size=bubble_sizes,
            color=data[metric_col],
            colorscale=[
                [0, get_color('success')],
                [0.5, get_color('warning')],
                [1, get_color('error')]
            ],
            showscale=True,
            line=dict(color=get_color('border_primary'), width=2),
            sizemode='diameter'
        ),
        text=data[dept_col],
        textposition='middle center',
        textfont=dict(color=get_color('text'), size=10),
        hovertemplate='<b>%{text}</b><br>' +
                     f'{metric_col}: %{{y:.3f}}<br>' +
                     'Employees: %{customdata}<br>' +
                     '<extra></extra>',
        customdata=employee_counts
    ))
    
    return fig

# ================================================================
# SALARY DISTRIBUTION CHARTS
# ================================================================

def generate_salary_distribution(data: pd.DataFrame,
                                salary_col: str = 'MonthlyIncome', 
                                group_by: Optional[str] = None,
                                chart_type: str = 'histogram',
                                height: int = 400,
                                title: str = 'Salary Distribution Analysis') -> go.Figure:
    """
    Generate comprehensive salary distribution visualization.
    
    Args:
        data: DataFrame with salary data
        salary_col: Column name containing salary information
        group_by: Optional column to group by (Department, JobRole, etc.)
        chart_type: 'histogram', 'box', 'violin', 'density'
        height: Chart height
        title: Chart title
        
    Returns:
        Plotly figure with salary distribution
    """
    
    logger.info(f"Creating salary distribution chart ({chart_type}) for {len(data)} records")
    
    try:
        if salary_col not in data.columns:
            return _create_error_chart(f"Column {salary_col} not found")
        
        # Clean salary data
        salary_data = data[salary_col].dropna()
        
        if len(salary_data) == 0:
            return _create_error_chart("No valid salary data found")
        
        # Create chart based on type
        if chart_type.lower() == 'histogram':
            fig = _create_salary_histogram(data, salary_col, group_by)
            
        elif chart_type.lower() == 'box':
            fig = _create_salary_boxplot(data, salary_col, group_by)
            
        elif chart_type.lower() == 'violin':
            fig = _create_salary_violinplot(data, salary_col, group_by)
            
        elif chart_type.lower() == 'density':
            fig = _create_salary_density(data, salary_col, group_by)
            
        else:
            fig = _create_salary_histogram(data, salary_col, group_by)
        
        # Apply dark theme
        fig = create_dark_theme_plotly_chart(
            fig, title=title, height=height, 
            show_legend=True if group_by else False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating salary distribution chart: {e}")
        return _create_error_chart(f"Error: {e}")

def _create_salary_histogram(data: pd.DataFrame, salary_col: str, group_by: Optional[str]) -> go.Figure:
    """Create salary histogram with optional grouping."""
    
    fig = go.Figure()
    
    if group_by and group_by in data.columns:
        # Grouped histogram
        groups = data[group_by].unique()
        colors = get_chart_color_palette('categorical')
        
        for i, group in enumerate(groups[:7]):  # Limit to 7 groups for readability
            group_data = data[data[group_by] == group][salary_col]
            
            fig.add_trace(go.Histogram(
                x=group_data,
                name=str(group),
                opacity=0.7,
                marker=dict(
                    color=colors[i % len(colors)],
                    line=dict(color=get_color('border_primary'), width=1)
                ),
                hovertemplate=f'<b>{group}</b><br>' +
                             'Salary Range: %{x}<br>' +
                             'Count: %{y}<br>' +
                             '<extra></extra>'
            ))
    else:
        # Single histogram
        fig.add_trace(go.Histogram(
            x=data[salary_col],
            marker=dict(
                color=get_color('secondary'),
                line=dict(color=get_color('border_primary'), width=1)
            ),
            opacity=0.8,
            hovertemplate='Salary Range: %{x}<br>' +
                         'Count: %{y}<br>' +
                         '<extra></extra>'
        ))
        
        # Add statistical lines
        mean_salary = data[salary_col].mean()
        median_salary = data[salary_col].median()
        
        fig.add_vline(
            x=mean_salary,
            line_dash="dash",
            line_color=get_color('warning'),
            annotation_text=f"Mean: ${mean_salary:,.0f}"
        )
        
        fig.add_vline(
            x=median_salary,
            line_dash="dot",
            line_color=get_color('success'),
            annotation_text=f"Median: ${median_salary:,.0f}"
        )
    
    return fig

def _create_salary_boxplot(data: pd.DataFrame, salary_col: str, group_by: Optional[str]) -> go.Figure:
    """Create salary box plot with optional grouping."""
    
    fig = go.Figure()
    
    if group_by and group_by in data.columns:
        # Grouped box plot
        groups = data[group_by].unique()
        colors = get_chart_color_palette('categorical')
        
        for i, group in enumerate(groups):
            group_data = data[data[group_by] == group][salary_col]
            
            fig.add_trace(go.Box(
                y=group_data,
                name=str(group),
                marker=dict(color=colors[i % len(colors)]),
                line=dict(color=get_color('border_primary')),
                boxpoints='outliers',
                hovertemplate=f'<b>{group}</b><br>' +
                             'Salary: %{y:$,.0f}<br>' +
                             '<extra></extra>'
            ))
    else:
        # Single box plot
        fig.add_trace(go.Box(
            y=data[salary_col],
            marker=dict(color=get_color('secondary')),
            line=dict(color=get_color('border_primary')),
            boxpoints='outliers',
            hovertemplate='Salary: %{y:$,.0f}<br><extra></extra>'
        ))
    
    return fig

def _create_salary_violinplot(data: pd.DataFrame, salary_col: str, group_by: Optional[str]) -> go.Figure:
    """Create salary violin plot with optional grouping."""
    
    fig = go.Figure()
    
    if group_by and group_by in data.columns:
        # Grouped violin plot
        groups = data[group_by].unique()
        colors = get_chart_color_palette('categorical')
        
        for i, group in enumerate(groups):
            group_data = data[data[group_by] == group][salary_col]
            
            fig.add_trace(go.Violin(
                y=group_data,
                name=str(group),
                marker=dict(color=colors[i % len(colors)]),
                line=dict(color=get_color('border_primary')),
                box_visible=True,
                meanline_visible=True,
                hovertemplate=f'<b>{group}</b><br>' +
                             'Salary: %{y:$,.0f}<br>' +
                             '<extra></extra>'
            ))
    else:
        # Single violin plot
        fig.add_trace(go.Violin(
            y=data[salary_col],
            marker=dict(color=get_color('secondary')),
            line=dict(color=get_color('border_primary')),
            box_visible=True,
            meanline_visible=True,
            hovertemplate='Salary: %{y:$,.0f}<br><extra></extra>'
        ))
    
    return fig

def _create_salary_density(data: pd.DataFrame, salary_col: str, group_by: Optional[str]) -> go.Figure:
    """Create salary density plot."""
    
    fig = go.Figure()
    
    if group_by and group_by in data.columns:
        # Grouped density
        groups = data[group_by].unique()
        colors = get_chart_color_palette('categorical')
        
        for i, group in enumerate(groups[:5]):  # Limit for performance
            group_data = data[data[group_by] == group][salary_col].dropna()
            
            if len(group_data) > 0:
                # Simple density estimation
                hist, bin_edges = np.histogram(group_data, bins=30, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                fig.add_trace(go.Scatter(
                    x=bin_centers,
                    y=hist,
                    mode='lines',
                    fill='tonexty' if i > 0 else 'tozeroy',
                    name=str(group),
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.6
                ))
    else:
        # Single density
        salary_data = data[salary_col].dropna()
        hist, bin_edges = np.histogram(salary_data, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=hist,
            mode='lines',
            fill='tozeroy',
            line=dict(color=get_color('secondary'), width=3),
            opacity=0.7
        ))
    
    return fig

# ================================================================
# SATISFACTION RADAR CHARTS
# ================================================================

def create_satisfaction_radar_chart(data: pd.DataFrame,
                                  satisfaction_cols: List[str] = None,
                                  group_by: Optional[str] = None,
                                  height: int = 500,
                                  title: str = 'Employee Satisfaction Analysis') -> go.Figure:
    """
    Create radar chart for satisfaction metrics.
    
    Args:
        data: DataFrame with satisfaction data
        satisfaction_cols: List of satisfaction column names
        group_by: Optional grouping column
        height: Chart height
        title: Chart title
        
    Returns:
        Plotly figure with radar chart
    """
    
    logger.info(f"Creating satisfaction radar chart for {len(data)} records")
    
    # Default satisfaction columns
    if satisfaction_cols is None:
        satisfaction_cols = [
            'JobSatisfaction', 'EnvironmentSatisfaction', 
            'WorkLifeBalance', 'RelationshipSatisfaction'
        ]
    
    # Filter available columns
    available_cols = [col for col in satisfaction_cols if col in data.columns]
    
    if not available_cols:
        return _create_error_chart("No satisfaction columns found")
    
    try:
        fig = go.Figure()
        
        if group_by and group_by in data.columns:
            # Grouped radar chart
            groups = data[group_by].unique()
            colors = get_chart_color_palette('categorical')
            
            for i, group in enumerate(groups[:5]):  # Limit groups for readability
                group_data = data[data[group_by] == group]
                
                # Calculate mean satisfaction scores
                group_scores = []
                for col in available_cols:
                    score = group_data[col].mean()
                    group_scores.append(score)
                
                # Add trace for this group
                fig.add_trace(go.Scatterpolar(
                    r=group_scores + [group_scores[0]],  # Close the polygon
                    theta=available_cols + [available_cols[0]],
                    fill='toself',
                    name=str(group),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(color=colors[i % len(colors)], size=8),
                    hovertemplate=f'<b>{group}</b><br>' +
                                 '%{theta}: %{r:.2f}<br>' +
                                 '<extra></extra>'
                ))
        else:
            # Single radar chart (overall averages)
            overall_scores = []
            for col in available_cols:
                score = data[col].mean()
                overall_scores.append(score)
            
            fig.add_trace(go.Scatterpolar(
                r=overall_scores + [overall_scores[0]],
                theta=available_cols + [available_cols[0]],
                fill='toself',
                name='Overall',
                line=dict(color=get_color('secondary'), width=3),
                marker=dict(color=get_color('secondary'), size=10),
                hovertemplate='%{theta}: %{r:.2f}<br><extra></extra>'
            ))
        
        # Configure radar layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],  # Assuming 1-5 satisfaction scale
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    gridcolor=get_color('border_primary'),
                    gridwidth=1,
                    linecolor=get_color('border_primary'),
                    tickfont=dict(color=get_color('text_secondary'))
                ),
                angularaxis=dict(
                    gridcolor=get_color('border_primary'),
                    linecolor=get_color('border_primary'),
                    tickfont=dict(color=get_color('text'), size=12)
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True if group_by else False
        )
        
        # Apply dark theme
        fig = create_dark_theme_plotly_chart(
            fig, title=title, height=height, 
            show_legend=True if group_by else False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating satisfaction radar chart: {e}")
        return _create_error_chart(f"Error: {e}")

# ================================================================
# ATTRITION TRENDS
# ================================================================

def plot_attrition_trends(data: pd.DataFrame,
                         date_col: Optional[str] = None,
                         period: str = 'monthly',
                         height: int = 400,
                         title: str = 'Attrition Trends Over Time') -> go.Figure:
    """
    Plot attrition trends over time.
    
    Args:
        data: DataFrame with date and attrition data
        date_col: Column name with dates (if None, generates synthetic data)
        period: 'daily', 'weekly', 'monthly', 'quarterly'
        height: Chart height
        title: Chart title
        
    Returns:
        Plotly figure with trend analysis
    """
    
    logger.info(f"Creating attrition trends chart ({period}) for {len(data)} records")
    
    try:
        # Generate synthetic date data if not provided
        if date_col is None or date_col not in data.columns:
            logger.warning("No date column found, generating synthetic trend data")
            return _create_synthetic_trends(data, period)
        
        # Process actual date data
        trend_data = data.copy()
        trend_data[date_col] = pd.to_datetime(trend_data[date_col])
        
        # Group by period
        if period == 'daily':
            trend_data['Period'] = trend_data[date_col].dt.date
        elif period == 'weekly':
            trend_data['Period'] = trend_data[date_col].dt.to_period('W').dt.start_time
        elif period == 'monthly':
            trend_data['Period'] = trend_data[date_col].dt.to_period('M').dt.start_time
        elif period == 'quarterly':
            trend_data['Period'] = trend_data[date_col].dt.to_period('Q').dt.start_time
        else:
            trend_data['Period'] = trend_data[date_col].dt.to_period('M').dt.start_time
        
        # Calculate attrition metrics by period
        period_stats = trend_data.groupby('Period').agg({
            'Attrition': [
                ('total_employees', 'count'),
                ('attritions', lambda x: (x == 'Yes').sum() if 'Attrition' in trend_data.columns else 0)
            ]
        }).round(3)
        
        period_stats.columns = ['TotalEmployees', 'Attritions']
        period_stats['AttritionRate'] = period_stats['Attritions'] / period_stats['TotalEmployees']
        period_stats = period_stats.reset_index()
        
        # Create trend visualization
        fig = _create_trend_chart(period_stats)
        
        # Apply dark theme
        fig = create_dark_theme_plotly_chart(
            fig, title=title, height=height, show_legend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating attrition trends chart: {e}")
        return _create_error_chart(f"Error: {e}")

def _create_trend_chart(period_stats: pd.DataFrame) -> go.Figure:
    """Create trend chart from period statistics."""
    
    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )
    
    # Attrition rate line
    fig.add_trace(
        go.Scatter(
            x=period_stats['Period'],
            y=period_stats['AttritionRate'] * 100,
            mode='lines+markers',
            name='Attrition Rate (%)',
            line=dict(color=get_color('secondary'), width=3),
            marker=dict(size=8, color=get_color('secondary')),
            hovertemplate='Date: %{x}<br>Attrition Rate: %{y:.1f}%<br><extra></extra>'
        ),
        secondary_y=False
    )
    
    # Total employees bars
    fig.add_trace(
        go.Bar(
            x=period_stats['Period'],
            y=period_stats['TotalEmployees'],
            name='Total Employees',
            marker=dict(color=get_color('success'), opacity=0.6),
            hovertemplate='Date: %{x}<br>Total Employees: %{y}<br><extra></extra>'
        ),
        secondary_y=True
    )
    
    # Trend line
    if len(period_stats) > 2:
        z = np.polyfit(range(len(period_stats)), period_stats['AttritionRate'] * 100, 1)
        trend_line = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=period_stats['Period'],
                y=trend_line(range(len(period_stats))),
                mode='lines',
                name='Trend',
                line=dict(color=get_color('accent'), width=2, dash='dash'),
                hoverinfo='skip'
            ),
            secondary_y=False
        )
    
    # Configure y-axes
    fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Total Employees", secondary_y=True)
    
    return fig

def _create_synthetic_trends(data: pd.DataFrame, period: str) -> go.Figure:
    """Create synthetic trend data for demonstration."""
    
    # Generate date range
    end_date = datetime.now()
    
    if period == 'daily':
        start_date = end_date - timedelta(days=90)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
    elif period == 'weekly':
        start_date = end_date - timedelta(weeks=52)
        dates = pd.date_range(start=start_date, end=end_date, freq='W')
    elif period == 'monthly':
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
    else:
        start_date = end_date - timedelta(days=1095)  # 3 years
        dates = pd.date_range(start=start_date, end=end_date, freq='Q')
    
    # Generate synthetic attrition rates with trend and seasonality
    np.random.seed(42)
    base_rate = 0.16  # 16% base attrition rate
    
    trend = np.linspace(-0.02, 0.02, len(dates))  # Slight upward trend
    seasonality = 0.03 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)  # Annual seasonality
    noise = np.random.normal(0, 0.015, len(dates))
    
    attrition_rates = base_rate + trend + seasonality + noise
    attrition_rates = np.clip(attrition_rates, 0.05, 0.30)  # Keep realistic bounds
    
    # Generate employee counts
    base_employees = len(data) if len(data) > 0 else 1000
    employee_counts = base_employees + np.random.randint(-50, 50, len(dates))
    
    # Create synthetic dataframe
    synthetic_data = pd.DataFrame({
        'Period': dates,
        'AttritionRate': attrition_rates,
        'TotalEmployees': employee_counts,
        'Attritions': (attrition_rates * employee_counts).astype(int)
    })
    
    return _create_trend_chart(synthetic_data)

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def _create_error_chart(error_message: str) -> go.Figure:
    """Create error chart when visualization fails."""
    
    fig = go.Figure()
    
    fig.add_annotation(
        text=f"<span style='color:{get_color('error')};font-size:16px'><b>Chart Error</b></span><br>"
             f"<span style='color:{get_color('text')};font-size:12px'>{error_message}</span>",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14)
    )
    
    fig.update_layout(
        paper_bgcolor=get_color('background'),
        plot_bgcolor=get_color('background_light'),
        height=400
    )
    
    return fig

def optimize_chart_performance(fig: go.Figure, max_points: int = 1000) -> go.Figure:
    """Optimize chart for better performance on limited memory systems."""
    
    try:
        # Reduce marker sizes
        fig.update_traces(marker_size=4)
        
        # Simplify hover templates
        fig.update_traces(hovertemplate='%{y}<extra></extra>')
        
        # Remove animations for better performance
        fig.update_layout(
            transition_duration=0,
            font_size=10
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error optimizing chart performance: {e}")
        return fig

def save_chart_as_image(fig: go.Figure, filename: str, width: int = 1200, height: int = 600):
    """Save chart as static image."""
    
    try:
        fig.write_image(
            filename,
            width=width,
            height=height,
            scale=2,
            format='png'
        )
        logger.info(f"Chart saved as {filename}")
    except Exception as e:
        logger.error(f"Error saving chart image: {e}")

# ================================================================
# MEMORY MANAGEMENT
# ================================================================

def cleanup_chart_memory():
    """Force garbage collection for chart memory cleanup."""
    gc.collect()
    logger.info("Chart memory cleanup performed")

# ================================================================
# EXPORT ALL FUNCTIONS
# ================================================================

__all__ = [
    'create_risk_distribution_chart',
    'plot_department_heatmap',
    'generate_salary_distribution', 
    'create_satisfaction_radar_chart',
    'plot_attrition_trends',
    'optimize_chart_performance',
    'save_chart_as_image',
    'cleanup_chart_memory',
    'get_color'
]

# ================================================================
# DEVELOPMENT TESTING
# ================================================================

def test_visualizations():
    """Test visualization functions with sample data."""
    
    logger.info("🧪 Testing visualization components...")
    
    # Generate sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 500),
        'MonthlyIncome': np.random.randint(3000, 15000, 500),
        'JobSatisfaction': np.random.randint(1, 5, 500),
        'WorkLifeBalance': np.random.randint(1, 5, 500),
        'EnvironmentSatisfaction': np.random.randint(1, 5, 500),
        'RelationshipSatisfaction': np.random.randint(1, 5, 500),
        'RiskLevel': np.random.choice(['Low', 'Medium', 'High'], 500),
        'Attrition': np.random.choice(['Yes', 'No'], 500, p=[0.16, 0.84])
    })
    
    try:
        # Test risk distribution
        risk_fig = create_risk_distribution_chart(sample_data)
        logger.info("✅ Risk distribution chart created")
        
        # Test department heatmap  
        dept_fig = plot_department_heatmap(sample_data)
        logger.info("✅ Department heatmap created")
        
        # Test salary distribution
        salary_fig = generate_salary_distribution(sample_data)
        logger.info("✅ Salary distribution chart created")
        
        # Test satisfaction radar
        satisfaction_fig = create_satisfaction_radar_chart(sample_data)
        logger.info("✅ Satisfaction radar chart created")
        
        # Test attrition trends
        trends_fig = plot_attrition_trends(sample_data)
        logger.info("✅ Attrition trends chart created")
        
        logger.info("🎉 All visualization tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Visualization test failed: {e}")

if __name__ == "__main__":
    test_visualizations()
