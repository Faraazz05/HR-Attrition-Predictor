"""
HR Attrition Predictor - Custom UI Widgets & Interactive Components
==================================================================
Advanced Streamlit widgets with enhanced UX, real-time filtering, and
responsive design. Optimized for the cyberpunk theme integration.

Author: HR Analytics Team
Date: September 2025
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import re
from dataclasses import dataclass, asdict
from enum import Enum
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import theme components
try:
    from streamlit_app.assets.theme import COLORS, get_chart_color_palette
    from streamlit_app.config import get_risk_level, get_risk_color
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

# ================================================================
# WIDGET DATA CLASSES
# ================================================================

@dataclass
class FilterState:
    """State management for filter widgets."""
    departments: List[str] = None
    risk_levels: List[str] = None
    date_range: Tuple[date, date] = None
    search_term: str = ""
    employee_ids: List[str] = None
    active_filters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.departments is None:
            self.departments = []
        if self.risk_levels is None:
            self.risk_levels = []
        if self.active_filters is None:
            self.active_filters = {}

@dataclass
class WidgetStyle:
    """Styling configuration for widgets."""
    primary_color: str = "#00D4FF"
    secondary_color: str = "#B026FF"
    success_color: str = "#00FF88"
    warning_color: str = "#FF6B35"
    error_color: str = "#FF2D75"
    text_color: str = "#F0F8FF"
    background_color: str = "rgba(26, 31, 58, 0.4)"
    border_radius: str = "12px"
    box_shadow: str = "0 8px 32px rgba(0, 0, 0, 0.37)"

class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "Low"
    MEDIUM = "Medium"  
    HIGH = "High"
    CRITICAL = "Critical"

# ================================================================
# THEME INTEGRATION
# ================================================================

def get_widget_colors():
    """Get colors for widgets with fallback."""
    if THEME_AVAILABLE:
        return {
            'primary': COLORS.get('secondary', '#00D4FF'),
            'secondary': COLORS.get('accent', '#B026FF'),
            'success': COLORS.get('success', '#00FF88'),
            'warning': COLORS.get('warning', '#FF6B35'),
            'error': COLORS.get('error', '#FF2D75'),
            'text': COLORS.get('text', '#F0F8FF'),
            'background': COLORS.get('background_light', 'rgba(26, 31, 58, 0.4)')
        }
    else:
        return {
            'primary': '#00D4FF',
            'secondary': '#B026FF', 
            'success': '#00FF88',
            'warning': '#FF6B35',
            'error': '#FF2D75',
            'text': '#F0F8FF',
            'background': 'rgba(26, 31, 58, 0.4)'
        }

def apply_widget_styling():
    """Apply custom CSS for enhanced widget styling."""
    
    colors = get_widget_colors()
    
    st.markdown(f"""
    <style>
    /* Enhanced Widget Styling */
    .custom-widget {{
        background: {colors['background']};
        backdrop-filter: blur(15px) saturate(180%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }}
    
    .custom-widget:hover {{
        border-color: {colors['primary']};
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37), 0 0 20px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
    }}
    
    .widget-title {{
        color: {colors['text']};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .risk-badge {{
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0.25rem;
        display: inline-block;
        transition: all 0.2s ease;
        cursor: pointer;
        border: 2px solid transparent;
    }}
    
    .risk-badge:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }}
    
    .risk-badge.selected {{
        border-color: white;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }}
    
    .risk-low {{
        background: linear-gradient(135deg, {colors['success']}, #34D399);
        color: white;
    }}
    
    .risk-medium {{
        background: linear-gradient(135deg, {colors['warning']}, #FBBF24);
        color: white;
    }}
    
    .risk-high {{
        background: linear-gradient(135deg, {colors['error']}, #F87171);
        color: white;
    }}
    
    .risk-critical {{
        background: linear-gradient(135deg, #DC2626, #7F1D1D);
        color: white;
    }}
    
    .filter-chip {{
        background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: none;
    }}
    
    .filter-chip:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
    }}
    
    .filter-chip .remove-btn {{
        background: rgba(255, 255, 255, 0.2);
        border: none;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 12px;
        color: white;
    }}
    
    .search-container {{
        position: relative;
        margin: 1rem 0;
    }}
    
    .search-icon {{
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        color: {colors['text']};
        opacity: 0.6;
        font-size: 1.2rem;
        z-index: 2;
    }}
    
    .clear-filters-btn {{
        background: linear-gradient(135deg, {colors['error']}, #F87171);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 1rem;
    }}
    
    .clear-filters-btn:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }}
    
    .widget-stats {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .stat-item {{
        text-align: center;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        transition: all 0.2s ease;
    }}
    
    .stat-item:hover {{
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }}
    
    .stat-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {colors['primary']};
        margin-bottom: 0.25rem;
    }}
    
    .stat-label {{
        font-size: 0.875rem;
        color: {colors['text']};
        opacity: 0.8;
    }}
    </style>
    """, unsafe_allow_html=True)

# ================================================================
# RISK LEVEL SELECTOR WIDGET
# ================================================================

def risk_level_selector(
    key: str = "risk_selector",
    default_selection: List[str] = None,
    show_stats: bool = True,
    multi_select: bool = True,
    data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Advanced risk level selector with visual indicators and statistics.
    
    Args:
        key: Unique key for the widget
        default_selection: Default selected risk levels
        show_stats: Whether to show statistics for each risk level
        multi_select: Allow multiple selections
        data: Optional dataframe for statistics calculation
        
    Returns:
        Dictionary with selected risk levels and metadata
    """
    
    # Apply styling
    apply_widget_styling()
    
    # Initialize state
    if f"{key}_selected" not in st.session_state:
        st.session_state[f"{key}_selected"] = default_selection or []
    
    # Widget container
    st.markdown('<div class="custom-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-title">🎯 Risk Level Filter</div>', unsafe_allow_html=True)
    
    # Risk level options
    risk_levels = [level.value for level in RiskLevel]
    risk_colors = {
        'Low': 'risk-low',
        'Medium': 'risk-medium', 
        'High': 'risk-high',
        'Critical': 'risk-critical'
    }
    
    # Calculate statistics if data provided
    risk_stats = {}
    if data is not None and 'RiskLevel' in data.columns:
        for level in risk_levels:
            count = len(data[data['RiskLevel'] == level])
            percentage = (count / len(data)) * 100 if len(data) > 0 else 0
            risk_stats[level] = {'count': count, 'percentage': percentage}
    
    # Create risk level buttons
    cols = st.columns(len(risk_levels))
    selected_risks = st.session_state[f"{key}_selected"].copy()
    
    for i, (level, col) in enumerate(zip(risk_levels, cols)):
        with col:
            # Risk badge with stats
            stats_text = ""
            if level in risk_stats:
                stats_text = f"<br><small>{risk_stats[level]['count']} ({risk_stats[level]['percentage']:.1f}%)</small>"
            
            is_selected = level in selected_risks
            selected_class = " selected" if is_selected else ""
            
            badge_html = f"""
            <div class="risk-badge {risk_colors[level]}{selected_class}" 
                 onclick="toggleRisk('{key}', '{level}')" 
                 title="Click to {'remove' if is_selected else 'add'} {level} risk filter">
                {level}{stats_text}
            </div>
            """
            
            st.markdown(badge_html, unsafe_allow_html=True)
            
            # Handle selection
            if st.button(f"Toggle {level}", key=f"{key}_{level}_btn", help=f"Toggle {level} risk level"):
                if multi_select:
                    if level in selected_risks:
                        selected_risks.remove(level)
                    else:
                        selected_risks.append(level)
                else:
                    selected_risks = [level] if level not in selected_risks else []
                
                st.session_state[f"{key}_selected"] = selected_risks
                st.rerun()
    
    # Display selected filters
    if selected_risks:
        st.markdown("**Selected Risk Levels:**")
        chips_html = ""
        for risk in selected_risks:
            chips_html += f"""
            <span class="filter-chip">
                {risk}
                <button class="remove-btn" onclick="removeRisk('{key}', '{risk}')" title="Remove filter">×</button>
            </span>
            """
        st.markdown(chips_html, unsafe_allow_html=True)
    
    # Clear all button
    if selected_risks:
        if st.button("🗑️ Clear All", key=f"{key}_clear", help="Clear all risk level filters"):
            st.session_state[f"{key}_selected"] = []
            st.rerun()
    
    # Statistics summary
    if show_stats and data is not None and risk_stats:
        st.markdown("**Risk Distribution:**")
        stats_html = '<div class="widget-stats">'
        
        for level in risk_levels:
            if level in risk_stats:
                stats_html += f"""
                <div class="stat-item">
                    <div class="stat-value">{risk_stats[level]['count']}</div>
                    <div class="stat-label">{level}</div>
                </div>
                """
        
        stats_html += '</div>'
        st.markdown(stats_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # JavaScript for interactivity
    st.markdown(f"""
    <script>
    function toggleRisk(key, level) {{
        // This would be handled by Streamlit's component system in production
        console.log('Toggle risk:', key, level);
    }}
    
    function removeRisk(key, level) {{
        // This would be handled by Streamlit's component system in production
        console.log('Remove risk:', key, level);
    }}
    </script>
    """, unsafe_allow_html=True)
    
    return {
        'selected_risks': selected_risks,
        'risk_stats': risk_stats,
        'total_selected': len(selected_risks),
        'filter_active': len(selected_risks) > 0
    }

# ================================================================
# DEPARTMENT FILTER WIDGET
# ================================================================

def department_filter(
    departments: List[str],
    key: str = "dept_filter",
    default_selection: List[str] = None,
    show_employee_counts: bool = True,
    data: Optional[pd.DataFrame] = None,
    layout: str = "grid"  # "grid", "list", "dropdown"
) -> Dict[str, Any]:
    """
    Advanced department filter with multiple display modes and statistics.
    
    Args:
        departments: List of available departments
        key: Unique key for the widget
        default_selection: Default selected departments
        show_employee_counts: Show employee count per department
        data: Optional dataframe for statistics
        layout: Display layout ("grid", "list", "dropdown")
        
    Returns:
        Dictionary with selected departments and metadata
    """
    
    # Apply styling
    apply_widget_styling()
    
    # Initialize state
    if f"{key}_selected" not in st.session_state:
        st.session_state[f"{key}_selected"] = default_selection or []
    
    # Widget container
    st.markdown('<div class="custom-widget">', unsafe_allow_html=True)
    
    # Header with layout selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="widget-title">🏢 Department Filter</div>', unsafe_allow_html=True)
    with col2:
        layout = st.selectbox("Layout", ["grid", "list", "dropdown"], 
                             index=["grid", "list", "dropdown"].index(layout),
                             key=f"{key}_layout")
    
    # Calculate department statistics
    dept_stats = {}
    if data is not None and 'Department' in data.columns:
        for dept in departments:
            count = len(data[data['Department'] == dept])
            avg_risk = 0
            if count > 0 and 'AttritionProbability' in data.columns:
                avg_risk = data[data['Department'] == dept]['AttritionProbability'].mean()
            dept_stats[dept] = {
                'count': count,
                'avg_risk': avg_risk,
                'risk_level': get_risk_level(avg_risk) if avg_risk > 0 else 'Low'
            }
    
    selected_depts = st.session_state[f"{key}_selected"].copy()
    
    # Display based on layout
    if layout == "dropdown":
        # Dropdown multi-select
        selected_depts = st.multiselect(
            "Select Departments:",
            options=departments,
            default=selected_depts,
            key=f"{key}_multiselect",
            help="Select one or more departments to filter"
        )
        st.session_state[f"{key}_selected"] = selected_depts
        
    elif layout == "list":
        # List with checkboxes
        st.markdown("**Select Departments:**")
        
        # Search within departments
        search_term = st.text_input(
            "Search departments...",
            key=f"{key}_search",
            placeholder="Type to filter departments"
        )
        
        filtered_depts = [d for d in departments if search_term.lower() in d.lower()] if search_term else departments
        
        for dept in filtered_depts:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                is_selected = st.checkbox(
                    dept,
                    value=dept in selected_depts,
                    key=f"{key}_check_{dept}"
                )
                
                if is_selected and dept not in selected_depts:
                    selected_depts.append(dept)
                elif not is_selected and dept in selected_depts:
                    selected_depts.remove(dept)
            
            with col2:
                if dept in dept_stats and show_employee_counts:
                    count = dept_stats[dept]['count']
                    risk_level = dept_stats[dept]['risk_level']
                    risk_color = get_risk_color(risk_level) if THEME_AVAILABLE else '#00D4FF'
                    st.markdown(f"""
                    <div style="text-align: right; font-size: 0.875rem;">
                        <span style="color: {get_widget_colors()['text']};">{count} emp</span><br>
                        <span style="color: {risk_color}; font-size: 0.75rem;">{risk_level} risk</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.session_state[f"{key}_selected"] = selected_depts
        
    else:  # grid layout
        # Grid of department cards
        st.markdown("**Select Departments:**")
        
        # Arrange in responsive grid
        cols_per_row = min(3, len(departments))
        if cols_per_row == 0:
            cols_per_row = 1
            
        rows = (len(departments) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                dept_idx = row * cols_per_row + col_idx
                
                if dept_idx < len(departments):
                    dept = departments[dept_idx]
                    
                    with cols[col_idx]:
                        # Department card
                        is_selected = dept in selected_depts
                        card_style = "border: 2px solid #00D4FF;" if is_selected else "border: 1px solid rgba(255,255,255,0.2);"
                        
                        stats_html = ""
                        if dept in dept_stats and show_employee_counts:
                            count = dept_stats[dept]['count']
                            risk_level = dept_stats[dept]['risk_level']
                            risk_color = get_risk_color(risk_level) if THEME_AVAILABLE else '#00D4FF'
                            stats_html = f"""
                            <div style="margin-top: 0.5rem; font-size: 0.875rem;">
                                <div style="color: {get_widget_colors()['text']};">{count} employees</div>
                                <div style="color: {risk_color}; font-size: 0.75rem;">{risk_level} avg risk</div>
                            </div>
                            """
                        
                        card_html = f"""
                        <div style="
                            {card_style}
                            background: rgba(255,255,255,0.05);
                            border-radius: 8px;
                            padding: 1rem;
                            text-align: center;
                            cursor: pointer;
                            transition: all 0.2s ease;
                            margin-bottom: 0.5rem;
                        " onclick="toggleDept('{key}', '{dept}')">
                            <div style="font-weight: 600; color: {get_widget_colors()['text']};">{dept}</div>
                            {stats_html}
                        </div>
                        """
                        
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Handle selection with button
                        if st.button(f"Toggle {dept}", key=f"{key}_toggle_{dept}", help=f"Toggle {dept} selection"):
                            if dept in selected_depts:
                                selected_depts.remove(dept)
                            else:
                                selected_depts.append(dept)
                            st.session_state[f"{key}_selected"] = selected_depts
                            st.rerun()
    
    # Display active filters
    if selected_depts:
        st.markdown("**Active Filters:**")
        chips_html = ""
        for dept in selected_depts:
            chips_html += f"""
            <span class="filter-chip">
                🏢 {dept}
                <button class="remove-btn" onclick="removeDept('{key}', '{dept}')" title="Remove filter">×</button>
            </span>
            """
        st.markdown(chips_html, unsafe_allow_html=True)
        
        # Clear all button
        if st.button("🗑️ Clear All Departments", key=f"{key}_clear_all"):
            st.session_state[f"{key}_selected"] = []
            st.rerun()
    
    # Summary statistics
    if show_employee_counts and dept_stats:
        total_selected_employees = sum(dept_stats.get(d, {}).get('count', 0) for d in selected_depts)
        total_employees = sum(stats['count'] for stats in dept_stats.values())
        
        st.markdown("**Selection Summary:**")
        summary_html = f"""
        <div class="widget-stats">
            <div class="stat-item">
                <div class="stat-value">{len(selected_depts)}</div>
                <div class="stat-label">Departments</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_selected_employees}</div>
                <div class="stat-label">Employees</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{(total_selected_employees/total_employees*100):.1f}%</div>
                <div class="stat-label">Coverage</div>
            </div>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'selected_departments': selected_depts,
        'department_stats': dept_stats,
        'total_selected': len(selected_depts),
        'filter_active': len(selected_depts) > 0,
        'layout': layout
    }

# ================================================================
# DATE RANGE PICKER WIDGET
# ================================================================

def date_range_picker(
    key: str = "date_picker",
    default_range: Optional[Tuple[date, date]] = None,
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
    preset_ranges: bool = True,
    show_stats: bool = True,
    data: Optional[pd.DataFrame] = None,
    date_column: str = "Date"
) -> Dict[str, Any]:
    """
    Advanced date range picker with preset ranges and data filtering.
    
    Args:
        key: Unique key for the widget
        default_range: Default date range (start, end)
        min_date: Minimum selectable date
        max_date: Maximum selectable date
        preset_ranges: Show preset range buttons
        show_stats: Show statistics for selected range
        data: Optional dataframe for statistics
        date_column: Column name containing dates
        
    Returns:
        Dictionary with selected date range and metadata
    """
    
    # Apply styling
    apply_widget_styling()
    
    # Default date range
    if default_range is None:
        end_date = max_date or date.today()
        start_date = min_date or (end_date - timedelta(days=30))
        default_range = (start_date, end_date)
    
    # Initialize state
    if f"{key}_start" not in st.session_state:
        st.session_state[f"{key}_start"] = default_range[0]
    if f"{key}_end" not in st.session_state:
        st.session_state[f"{key}_end"] = default_range[1]
    
    # Widget container
    st.markdown('<div class="custom-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-title">📅 Date Range Filter</div>', unsafe_allow_html=True)
    
    # Preset range buttons
    if preset_ranges:
        st.markdown("**Quick Select:**")
        
        presets = [
            ("Last 7 Days", 7),
            ("Last 30 Days", 30),
            ("Last 90 Days", 90),
            ("Last 6 Months", 180),
            ("Last Year", 365),
            ("All Time", None)
        ]
        
        cols = st.columns(len(presets))
        
        for i, (label, days) in enumerate(presets):
            with cols[i]:
                if st.button(label, key=f"{key}_preset_{i}", help=f"Select {label.lower()}"):
                    if days is None:
                        # All time
                        start_date = min_date or date(2020, 1, 1)
                        end_date = max_date or date.today()
                    else:
                        end_date = max_date or date.today()
                        start_date = max(min_date or date(2020, 1, 1), end_date - timedelta(days=days))
                    
                    st.session_state[f"{key}_start"] = start_date
                    st.session_state[f"{key}_end"] = end_date
                    st.rerun()
    
    # Date input controls
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state[f"{key}_start"],
            min_value=min_date,
            max_value=st.session_state[f"{key}_end"],
            key=f"{key}_start_input",
            help="Select the start date for the range"
        )
        st.session_state[f"{key}_start"] = start_date
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=st.session_state[f"{key}_end"],
            min_value=st.session_state[f"{key}_start"],
            max_value=max_date,
            key=f"{key}_end_input",
            help="Select the end date for the range"
        )
        st.session_state[f"{key}_end"] = end_date
    
    # Calculate range statistics
    selected_range = (start_date, end_date)
    range_days = (end_date - start_date).days + 1
    
    # Data statistics if provided
    range_stats = {}
    if data is not None and date_column in data.columns:
        try:
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                data_dates = pd.to_datetime(data[date_column])
            else:
                data_dates = data[date_column]
            
            # Filter data by date range
            mask = (data_dates.dt.date >= start_date) & (data_dates.dt.date <= end_date)
            filtered_data = data[mask]
            
            range_stats = {
                'total_records': len(filtered_data),
                'date_coverage': len(filtered_data) / len(data) * 100 if len(data) > 0 else 0,
                'daily_average': len(filtered_data) / max(range_days, 1),
                'data_available': len(filtered_data) > 0
            }
            
            # Additional stats if risk/attrition columns available
            if 'AttritionProbability' in filtered_data.columns:
                range_stats['avg_risk'] = filtered_data['AttritionProbability'].mean()
                range_stats['high_risk_count'] = len(filtered_data[filtered_data['AttritionProbability'] > 0.7])
            
        except Exception as e:
            st.warning(f"Error calculating date range statistics: {e}")
            range_stats = {'error': str(e)}
    
    # Display range information
    st.markdown("**Selected Range:**")
    range_info_html = f"""
    <div style="
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="color: {get_widget_colors()['primary']};">
                    {start_date.strftime('%B %d, %Y')} → {end_date.strftime('%B %d, %Y')}
                </strong>
                <div style="color: {get_widget_colors()['text']}; opacity: 0.8; font-size: 0.875rem; margin-top: 0.25rem;">
                    {range_days} day{'s' if range_days != 1 else ''} selected
                </div>
            </div>
            <div style="text-align: right;">
                <div style="color: {get_widget_colors()['success']}; font-size: 1.2rem; font-weight: bold;">
                    {range_stats.get('total_records', 'N/A')}
                </div>
                <div style="color: {get_widget_colors()['text']}; opacity: 0.8; font-size: 0.75rem;">
                    records
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(range_info_html, unsafe_allow_html=True)
    
    # Statistics display
    if show_stats and range_stats and 'error' not in range_stats:
        st.markdown("**Range Statistics:**")
        
        stats_html = '<div class="widget-stats">'
        
        # Basic statistics
        if 'total_records' in range_stats:
            stats_html += f"""
            <div class="stat-item">
                <div class="stat-value">{range_stats['total_records']:,}</div>
                <div class="stat-label">Total Records</div>
            </div>
            """
        
        if 'date_coverage' in range_stats:
            stats_html += f"""
            <div class="stat-item">
                <div class="stat-value">{range_stats['date_coverage']:.1f}%</div>
                <div class="stat-label">Data Coverage</div>
            </div>
            """
        
        if 'daily_average' in range_stats:
            stats_html += f"""
            <div class="stat-item">
                <div class="stat-value">{range_stats['daily_average']:.1f}</div>
                <div class="stat-label">Daily Average</div>
            </div>
            """
        
        if 'avg_risk' in range_stats:
            stats_html += f"""
            <div class="stat-item">
                <div class="stat-value">{range_stats['avg_risk']:.2f}</div>
                <div class="stat-label">Avg Risk Score</div>
            </div>
            """
        
        stats_html += '</div>'
        st.markdown(stats_html, unsafe_allow_html=True)
    
    # Clear/Reset button
    if st.button("🔄 Reset to Default", key=f"{key}_reset", help="Reset to default date range"):
        st.session_state[f"{key}_start"] = default_range[0]
        st.session_state[f"{key}_end"] = default_range[1]
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'range_days': range_days,
        'range_stats': range_stats,
        'filter_active': True,
        'date_range': selected_range
    }

# ================================================================
# EMPLOYEE SEARCH BOX WIDGET
# ================================================================

def employee_search_box(
    employees: List[str],
    key: str = "employee_search",
    placeholder: str = "Search employees by name, ID, or department...",
    search_fields: List[str] = None,
    max_results: int = 10,
    show_suggestions: bool = True,
    data: Optional[pd.DataFrame] = None,
    quick_filters: bool = True
) -> Dict[str, Any]:
    """
    Advanced employee search box with autocomplete, filters, and smart suggestions.
    
    Args:
        employees: List of employee names or IDs
        key: Unique key for the widget
        placeholder: Placeholder text for search input
        search_fields: Fields to search in (if data provided)
        max_results: Maximum number of results to show
        show_suggestions: Show search suggestions
        data: Optional dataframe for advanced search
        quick_filters: Show quick filter buttons
        
    Returns:
        Dictionary with search results and metadata
    """
    
    # Apply styling
    apply_widget_styling()
    
    # Initialize state
    if f"{key}_query" not in st.session_state:
        st.session_state[f"{key}_query"] = ""
    if f"{key}_selected" not in st.session_state:
        st.session_state[f"{key}_selected"] = []
    if f"{key}_filters" not in st.session_state:
        st.session_state[f"{key}_filters"] = {}
    
    # Widget container
    st.markdown('<div class="custom-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-title">🔍 Employee Search</div>', unsafe_allow_html=True)
    
    # Search input with custom styling
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown('<div class="search-icon">🔍</div>', unsafe_allow_html=True)
    
    search_query = st.text_input(
        "",
        placeholder=placeholder,
        key=f"{key}_input",
        label_visibility="collapsed",
        help="Search by name, employee ID, department, or role"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update session state
    st.session_state[f"{key}_query"] = search_query
    
    # Quick filters
    if quick_filters and data is not None:
        st.markdown("**Quick Filters:**")
        
        filter_cols = st.columns(4)
        quick_filter_options = {}
        
        # Department filter
        if 'Department' in data.columns:
            with filter_cols[0]:
                departments = ['All'] + sorted(data['Department'].unique().tolist())
                selected_dept = st.selectbox(
                    "Department",
                    departments,
                    key=f"{key}_dept_filter"
                )
                if selected_dept != 'All':
                    quick_filter_options['Department'] = selected_dept
        
        # Risk level filter
        if 'RiskLevel' in data.columns:
            with filter_cols[1]:
                risk_levels = ['All'] + sorted(data['RiskLevel'].unique().tolist())
                selected_risk = st.selectbox(
                    "Risk Level",
                    risk_levels,
                    key=f"{key}_risk_filter"
                )
                if selected_risk != 'All':
                    quick_filter_options['RiskLevel'] = selected_risk
        
        # Performance filter
        if 'PerformanceRating' in data.columns:
            with filter_cols[2]:
                perf_options = ['All', 'High (4-5)', 'Medium (3)', 'Low (1-2)']
                selected_perf = st.selectbox(
                    "Performance",
                    perf_options,
                    key=f"{key}_perf_filter"
                )
                if selected_perf != 'All':
                    quick_filter_options['Performance'] = selected_perf
        
        # Status filter
        if 'Status' in data.columns:
            with filter_cols[3]:
                statuses = ['All'] + sorted(data['Status'].unique().tolist())
                selected_status = st.selectbox(
                    "Status",
                    statuses,
                    key=f"{key}_status_filter"
                )
                if selected_status != 'All':
                    quick_filter_options['Status'] = selected_status
        
        st.session_state[f"{key}_filters"] = quick_filter_options
    
    # Perform search
    search_results = []
    filtered_data = data.copy() if data is not None else pd.DataFrame()
    
    if data is not None and not data.empty:
        # Apply quick filters first
        for filter_key, filter_value in st.session_state[f"{key}_filters"].items():
            if filter_key == 'Performance':
                if filter_value == 'High (4-5)':
                    filtered_data = filtered_data[filtered_data['PerformanceRating'] >= 4]
                elif filter_value == 'Medium (3)':
                    filtered_data = filtered_data[filtered_data['PerformanceRating'] == 3]
                elif filter_value == 'Low (1-2)':
                    filtered_data = filtered_data[filtered_data['PerformanceRating'] <= 2]
            else:
                filtered_data = filtered_data[filtered_data[filter_key] == filter_value]
        
        # Perform text search
        if search_query:
            search_fields = search_fields or ['FullName', 'EmployeeID', 'Department', 'JobRole']
            available_fields = [f for f in search_fields if f in filtered_data.columns]
            
            if available_fields:
                # Create search mask
                search_mask = pd.Series([False] * len(filtered_data))
                
                for field in available_fields:
                    field_mask = filtered_data[field].astype(str).str.contains(
                        search_query, case=False, na=False, regex=False
                    )
                    search_mask = search_mask | field_mask
                
                search_results_df = filtered_data[search_mask].head(max_results)
                search_results = search_results_df.to_dict('records')
        else:
            # No search query, return filtered results
            search_results = filtered_data.head(max_results).to_dict('records')
    else:
        # Simple string search in employee list
        if search_query:
            search_results = [emp for emp in employees 
                            if search_query.lower() in emp.lower()][:max_results]
    
    # Display search results
    if search_query or st.session_state[f"{key}_filters"]:
        st.markdown(f"**Search Results ({len(search_results)} found):**")
        
        if search_results:
            # Display results based on data type
            if isinstance(search_results[0], dict):
                # DataFrame results with rich information
                for i, result in enumerate(search_results):
                    # Employee card
                    emp_name = result.get('FullName', result.get('Name', f"Employee {i+1}"))
                    emp_id = result.get('EmployeeID', 'N/A')
                    dept = result.get('Department', 'N/A')
                    role = result.get('JobRole', 'N/A')
                    risk_level = result.get('RiskLevel', 'Unknown')
                    
                    # Risk color
                    risk_color = get_risk_color(risk_level) if THEME_AVAILABLE else get_widget_colors()['warning']
                    
                    # Selection state
                    is_selected = emp_id in st.session_state[f"{key}_selected"]
                    selection_style = "border: 2px solid #00D4FF;" if is_selected else "border: 1px solid rgba(255,255,255,0.2);"
                    
                    result_html = f"""
                    <div style="
                        {selection_style}
                        background: rgba(255,255,255,0.05);
                        border-radius: 8px;
                        padding: 1rem;
                        margin: 0.5rem 0;
                        cursor: pointer;
                        transition: all 0.2s ease;
                    " onclick="toggleEmployee('{key}', '{emp_id}')">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-weight: 600; color: {get_widget_colors()['text']}; font-size: 1.1rem;">
                                    {emp_name}
                                </div>
                                <div style="color: {get_widget_colors()['text']}; opacity: 0.8; font-size: 0.875rem;">
                                    ID: {emp_id} • {role} • {dept}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="
                                    background: {risk_color};
                                    color: white;
                                    padding: 0.25rem 0.75rem;
                                    border-radius: 12px;
                                    font-size: 0.75rem;
                                    font-weight: 600;
                                    text-transform: uppercase;
                                ">
                                    {risk_level}
                                </div>
                                {'<div style="color: #00D4FF; font-size: 0.875rem; margin-top: 0.25rem;">✓ Selected</div>' if is_selected else ''}
                            </div>
                        </div>
                    </div>
                    """
                    
                    st.markdown(result_html, unsafe_allow_html=True)
                    
                    # Selection button
                    button_label = f"Remove {emp_name}" if is_selected else f"Select {emp_name}"
                    if st.button(button_label, key=f"{key}_select_{emp_id}"):
                        if is_selected:
                            st.session_state[f"{key}_selected"].remove(emp_id)
                        else:
                            st.session_state[f"{key}_selected"].append(emp_id)
                        st.rerun()
            
            else:
                # Simple string results
                for result in search_results:
                    is_selected = result in st.session_state[f"{key}_selected"]
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{'✅' if is_selected else '◻️'} {result}")
                    with col2:
                        button_label = "Remove" if is_selected else "Select"
                        if st.button(button_label, key=f"{key}_select_{result}"):
                            if is_selected:
                                st.session_state[f"{key}_selected"].remove(result)
                            else:
                                st.session_state[f"{key}_selected"].append(result)
                            st.rerun()
        else:
            st.info("No employees found matching your search criteria.")
    
    # Display selected employees
    if st.session_state[f"{key}_selected"]:
        st.markdown("**Selected Employees:**")
        
        chips_html = ""
        for emp_id in st.session_state[f"{key}_selected"]:
            # Find employee name
            emp_name = emp_id
            if data is not None and 'EmployeeID' in data.columns:
                emp_row = data[data['EmployeeID'] == emp_id]
                if not emp_row.empty:
                    emp_name = emp_row.iloc[0].get('FullName', emp_id)
            
            chips_html += f"""
            <span class="filter-chip">
                👤 {emp_name}
                <button class="remove-btn" onclick="removeEmployee('{key}', '{emp_id}')" title="Remove employee">×</button>
            </span>
            """
        
        st.markdown(chips_html, unsafe_allow_html=True)
        
        # Clear selection button
        if st.button("🗑️ Clear Selection", key=f"{key}_clear_selection"):
            st.session_state[f"{key}_selected"] = []
            st.rerun()
    
    # Search suggestions
    if show_suggestions and search_query and len(search_query) >= 2:
        suggestions = []
        
        if data is not None:
            # Generate smart suggestions
            suggestion_fields = ['Department', 'JobRole']
            for field in suggestion_fields:
                if field in data.columns:
                    unique_values = data[field].unique()
                    matching_values = [v for v in unique_values 
                                     if search_query.lower() in str(v).lower()]
                    suggestions.extend(matching_values[:3])
        
        if suggestions:
            st.markdown("**Suggestions:**")
            suggestion_cols = st.columns(min(len(suggestions), 4))
            
            for i, suggestion in enumerate(suggestions[:4]):
                with suggestion_cols[i]:
                    if st.button(f"🔍 {suggestion}", key=f"{key}_suggest_{i}"):
                        st.session_state[f"{key}_query"] = str(suggestion)
                        st.rerun()
    
    # Search statistics
    total_employees = len(data) if data is not None else len(employees)
    search_stats = {
        'total_results': len(search_results),
        'selected_count': len(st.session_state[f"{key}_selected"]),
        'search_coverage': len(search_results) / total_employees * 100 if total_employees > 0 else 0,
        'has_query': bool(search_query),
        'has_filters': bool(st.session_state[f"{key}_filters"])
    }
    
    # Statistics display
    if search_query or st.session_state[f"{key}_filters"]:
        stats_html = f"""
        <div class="widget-stats">
            <div class="stat-item">
                <div class="stat-value">{search_stats['total_results']}</div>
                <div class="stat-label">Found</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{search_stats['selected_count']}</div>
                <div class="stat-label">Selected</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{search_stats['search_coverage']:.1f}%</div>
                <div class="stat-label">Coverage</div>
            </div>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # JavaScript for interactivity (placeholder)
    st.markdown(f"""
    <script>
    function toggleEmployee(key, empId) {{
        console.log('Toggle employee:', key, empId);
    }}
    
    function removeEmployee(key, empId) {{
        console.log('Remove employee:', key, empId);
    }}
    </script>
    """, unsafe_allow_html=True)
    
    return {
        'search_query': search_query,
        'search_results': search_results,
        'selected_employees': st.session_state[f"{key}_selected"],
        'active_filters': st.session_state[f"{key}_filters"],
        'search_stats': search_stats,
        'filtered_data': filtered_data if data is not None else None
    }

# ================================================================
# WIDGET UTILITIES AND HELPERS
# ================================================================

def clear_all_filters(widget_keys: List[str]):
    """Clear all filters for specified widgets."""
    
    for key in widget_keys:
        # Clear different widget types
        if f"{key}_selected" in st.session_state:
            st.session_state[f"{key}_selected"] = []
        if f"{key}_query" in st.session_state:
            st.session_state[f"{key}_query"] = ""
        if f"{key}_filters" in st.session_state:
            st.session_state[f"{key}_filters"] = {}
        if f"{key}_start" in st.session_state:
            st.session_state[f"{key}_start"] = date.today() - timedelta(days=30)
        if f"{key}_end" in st.session_state:
            st.session_state[f"{key}_end"] = date.today()
    
    st.rerun()

def get_combined_filter_state(*widget_results) -> FilterState:
    """Combine multiple widget results into a single filter state."""
    
    combined_state = FilterState()
    
    for result in widget_results:
        if isinstance(result, dict):
            # Risk level selector
            if 'selected_risks' in result:
                combined_state.risk_levels.extend(result['selected_risks'])
            
            # Department filter
            if 'selected_departments' in result:
                combined_state.departments.extend(result['selected_departments'])
            
            # Date range picker
            if 'date_range' in result:
                combined_state.date_range = result['date_range']
            
            # Employee search
            if 'selected_employees' in result:
                combined_state.employee_ids.extend(result['selected_employees'])
            if 'search_query' in result:
                combined_state.search_term = result['search_query']
            if 'active_filters' in result:
                combined_state.active_filters.update(result['active_filters'])
    
    return combined_state

def apply_filters_to_dataframe(df: pd.DataFrame, filter_state: FilterState) -> pd.DataFrame:
    """Apply combined filter state to a dataframe."""
    
    filtered_df = df.copy()
    
    # Apply department filter
    if filter_state.departments:
        if 'Department' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Department'].isin(filter_state.departments)]
    
    # Apply risk level filter
    if filter_state.risk_levels:
        if 'RiskLevel' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['RiskLevel'].isin(filter_state.risk_levels)]
    
    # Apply date range filter
    if filter_state.date_range:
        date_columns = ['Date', 'HireDate', 'LastInteraction']
        for date_col in date_columns:
            if date_col in filtered_df.columns:
                try:
                    df_dates = pd.to_datetime(filtered_df[date_col])
                    start_date, end_date = filter_state.date_range
                    mask = (df_dates.dt.date >= start_date) & (df_dates.dt.date <= end_date)
                    filtered_df = filtered_df[mask]
                    break
                except:
                    continue
    
    # Apply employee ID filter
    if filter_state.employee_ids:
        if 'EmployeeID' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['EmployeeID'].isin(filter_state.employee_ids)]
    
    # Apply search term filter
    if filter_state.search_term:
        search_columns = ['FullName', 'EmployeeID', 'Department', 'JobRole']
        available_cols = [c for c in search_columns if c in filtered_df.columns]
        
        if available_cols:
            search_mask = pd.Series([False] * len(filtered_df))
            for col in available_cols:
                col_mask = filtered_df[col].astype(str).str.contains(
                    filter_state.search_term, case=False, na=False, regex=False
                )
                search_mask = search_mask | col_mask
            filtered_df = filtered_df[search_mask]
    
    return filtered_df

# ================================================================
# WIDGET TESTING AND EXAMPLES
# ================================================================

def test_widgets():
    """Test all custom widgets with sample data."""
    
    st.title("🧪 Custom Widgets Testing")
    
    # Generate sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, 101)],
        'FullName': [f'Employee {i}' for i in range(1, 101)],
        'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], 100),
        'JobRole': np.random.choice(['Manager', 'Senior', 'Junior', 'Lead'], 100),
        'RiskLevel': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.6, 0.3, 0.1]),
        'AttritionProbability': np.random.uniform(0, 1, 100),
        'PerformanceRating': np.random.randint(1, 6, 100),
        'Status': np.random.choice(['Active', 'On Leave', 'Remote'], 100),
        'Date': pd.date_range(start='2024-01-01', periods=100, freq='D')
    })
    
    st.markdown("## Sample Data Generated")
    st.dataframe(sample_data.head(), use_container_width=True)
    
    # Test risk level selector
    st.markdown("## Risk Level Selector")
    risk_result = risk_level_selector(
        key="test_risk",
        data=sample_data,
        show_stats=True
    )
    st.json(risk_result)
    
    # Test department filter
    st.markdown("## Department Filter")
    departments = sample_data['Department'].unique().tolist()
    dept_result = department_filter(
        departments=departments,
        key="test_dept",
        data=sample_data,
        layout="grid"
    )
    st.json(dept_result)
    
    # Test date range picker
    st.markdown("## Date Range Picker")
    date_result = date_range_picker(
        key="test_date",
        data=sample_data,
        date_column="Date"
    )
    st.json(date_result)
    
    # Test employee search
    st.markdown("## Employee Search")
    employees = sample_data['FullName'].tolist()
    search_result = employee_search_box(
        employees=employees,
        key="test_search",
        data=sample_data
    )
    st.json(search_result)
    
    # Combined filter state
    st.markdown("## Combined Filter State")
    combined_state = get_combined_filter_state(risk_result, dept_result, date_result, search_result)
    st.json(asdict(combined_state))
    
    # Apply filters
    st.markdown("## Filtered Data")
    filtered_data = apply_filters_to_dataframe(sample_data, combined_state)
    st.dataframe(filtered_data, use_container_width=True)
    st.success(f"Filtered from {len(sample_data)} to {len(filtered_data)} records")

# ================================================================
# EXPORT ALL FUNCTIONS
# ================================================================

__all__ = [
    'risk_level_selector',
    'department_filter', 
    'date_range_picker',
    'employee_search_box',
    'FilterState',
    'WidgetStyle',
    'RiskLevel',
    'get_widget_colors',
    'apply_widget_styling',
    'clear_all_filters',
    'get_combined_filter_state',
    'apply_filters_to_dataframe',
    'test_widgets'
]

# ================================================================
# MAIN EXECUTION
# ================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Custom Widgets Test",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    test_widgets()
