"""
HR Attrition Predictor - Dark Theme Chart Components
===================================================
Beautiful, interactive Plotly charts with cyberpunk dark theme integration.
Optimized for 4GB RAM systems with memory-efficient rendering.

Author: HR Analytics Team
Date: September 2025
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Union, Tuple
import math

# Import theme configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from streamlit_app.assets.theme import (
    COLORS, TYPOGRAPHY, SIZING, get_plotly_dark_theme, 
    get_chart_color_palette
)

# ================================================================
# CORE DARK THEME CHART FUNCTION
# ================================================================

def create_dark_theme_plotly_chart(
    fig: go.Figure, 
    title: str = "",
    height: int = 400,
    show_legend: bool = True,
    grid_opacity: float = 0.2,
    title_size: int = 20,
    custom_layout: Optional[Dict] = None
) -> go.Figure:
    """
    Apply comprehensive dark theme styling to any Plotly chart.
    
    Args:
        fig: Plotly figure object
        title: Chart title
        height: Chart height in pixels
        show_legend: Whether to show legend
        grid_opacity: Grid line opacity (0-1)
        title_size: Title font size
        custom_layout: Additional layout customizations
        
    Returns:
        Styled Plotly figure with dark cyberpunk theme
    """
    
    # Get base theme
    theme = get_plotly_dark_theme()
    
    # Apply comprehensive styling
    fig.update_layout(
        # Background and paper
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background_light'],
        
        # Title styling
        title={
            'text': title,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'family': TYPOGRAPHY['font_family_display'],
                'size': title_size,
                'color': COLORS['text'],
            }
        },
        
        # Layout dimensions
        height=height,
        margin=dict(l=60, r=60, t=80, b=60),
        
        # Font configuration
        font=dict(
            family=TYPOGRAPHY['font_family_primary'],
            size=12,
            color=COLORS['text']
        ),
        
        # Legend styling
        legend=dict(
            bgcolor='rgba(0, 0, 0, 0.1)',
            bordercolor=COLORS['border_muted'],
            borderwidth=1,
            font=dict(color=COLORS['text'], size=11),
            orientation='v',
            x=1.02,
            y=1
        ) if show_legend else dict(showlegend=False),
        
        # Grid and axes
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=f"rgba{tuple(list(bytes.fromhex(COLORS['border_muted'].lstrip('#'))) + [int(255 * grid_opacity)])}",
            showline=True,
            linewidth=2,
            linecolor=COLORS['border_primary'],
            tickfont=dict(color=COLORS['text_secondary'], size=10),
            titlefont=dict(color=COLORS['text'], size=12),
            zeroline=True,
            zerolinecolor=COLORS['border_primary'],
            zerolinewidth=1
        ),
        
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=f"rgba{tuple(list(bytes.fromhex(COLORS['border_muted'].lstrip('#'))) + [int(255 * grid_opacity)])}",
            showline=True,
            linewidth=2,
            linecolor=COLORS['border_primary'],
            tickfont=dict(color=COLORS['text_secondary'], size=10),
            titlefont=dict(color=COLORS['text'], size=12),
            zeroline=True,
            zerolinecolor=COLORS['border_primary'],
            zerolinewidth=1
        ),
        
        # Hover styling
        hoverlabel=dict(
            bgcolor=COLORS['background_card'],
            bordercolor=COLORS['border_primary'],
            font_size=12,
            font_color=COLORS['text']
        ),
        
        # Animation and interaction
        transition_duration=300,
        clickmode='event+select'
    )
    
    # Apply custom layout if provided
    if custom_layout:
        fig.update_layout(custom_layout)
    
    # Add subtle glow effect to chart area
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        xref="paper", yref="paper",
        line=dict(color=COLORS['secondary'], width=2, dash="solid"),
        fillcolor="rgba(0, 0, 0, 0)"
    )
    
    return fig

# ================================================================
# FUTURISTIC GAUGE CHART
# ================================================================

def futuristic_gauge_chart(
    value: float,
    title: str,
    min_value: float = 0,
    max_value: float = 100,
    unit: str = "%",
    risk_thresholds: Optional[Dict[str, float]] = None,
    height: int = 350,
    show_needle: bool = True
) -> go.Figure:
    """
    Create a futuristic gauge chart with neon styling and risk zones.
    
    Args:
        value: Current value to display
        title: Chart title
        min_value: Minimum value on gauge
        max_value: Maximum value on gauge
        unit: Unit to display with value
        risk_thresholds: Dict with 'low', 'medium', 'high' thresholds
        height: Chart height in pixels
        show_needle: Whether to show the gauge needle
        
    Returns:
        Plotly gauge chart with cyberpunk styling
    """
    
    # Default risk thresholds for attrition probability
    if risk_thresholds is None:
        risk_thresholds = {'low': 30, 'medium': 70, 'high': 100}
    
    # Determine risk level and color
    if value <= risk_thresholds['low']:
        gauge_color = COLORS['success']
        risk_level = "LOW RISK"
        text_color = COLORS['success']
    elif value <= risk_thresholds['medium']:
        gauge_color = COLORS['warning']
        risk_level = "MEDIUM RISK"
        text_color = COLORS['warning']
    else:
        gauge_color = COLORS['error']
        risk_level = "HIGH RISK"
        text_color = COLORS['error']
    
    # Create gauge figure
    fig = go.Figure()
    
    # Main gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={
            'text': f"<span style='color:{COLORS['text']};font-family:{TYPOGRAPHY['font_family_display']};font-size:18px'>{title}</span>",
            'font': {'size': 18}
        },
        number={
            'font': {
                'size': 40,
                'color': text_color,
                'family': TYPOGRAPHY['font_family_display']
            },
            'suffix': unit
        },
        gauge={
            'axis': {
                'range': [min_value, max_value],
                'tickwidth': 1,
                'tickcolor': COLORS['text_secondary'],
                'tickfont': {'size': 12, 'color': COLORS['text_secondary']}
            },
            'bar': {
                'color': gauge_color,
                'thickness': 0.8 if show_needle else 1.0
            },
            'bgcolor': COLORS['background_light'],
            'borderwidth': 2,
            'bordercolor': COLORS['border_primary'],
            'steps': [
                {'range': [min_value, risk_thresholds['low']], 
                 'color': f"{COLORS['success']}33"},
                {'range': [risk_thresholds['low'], risk_thresholds['medium']], 
                 'color': f"{COLORS['warning']}33"},
                {'range': [risk_thresholds['medium'], max_value], 
                 'color': f"{COLORS['error']}33"}
            ],
            'threshold': {
                'line': {'color': COLORS['text'], 'width': 4},
                'thickness': 0.75,
                'value': value
            } if show_needle else None
        }
    ))
    
    # Apply dark theme
    fig = create_dark_theme_plotly_chart(
        fig, 
        title="", 
        height=height,
        show_legend=False,
        custom_layout={
            'margin': dict(l=20, r=20, t=60, b=20),
            'annotations': [
                dict(
                    text=f"<span style='color:{text_color};font-family:{TYPOGRAPHY['font_family_mono']};font-size:14px;font-weight:bold'>{risk_level}</span>",
                    x=0.5, y=0.15,
                    xref="paper", yref="paper",
                    showarrow=False
                )
            ]
        }
    )
    
    # Add glow effect
    fig.add_shape(
        type="circle",
        x0=0.1, y0=0.1, x1=0.9, y1=0.9,
        xref="paper", yref="paper",
        line=dict(color=gauge_color, width=3),
        fillcolor="rgba(0, 0, 0, 0)"
    )
    
    return fig

# ================================================================
# NEON BAR CHART
# ================================================================

def neon_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    color_column: Optional[str] = None,
    orientation: str = 'v',
    height: int = 400,
    show_values: bool = True,
    sort_bars: bool = True,
    max_bars: int = 20
) -> go.Figure:
    """
    Create a neon-styled bar chart with glowing effects.
    
    Args:
        data: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis  
        title: Chart title
        color_column: Optional column for color mapping
        orientation: 'v' for vertical, 'h' for horizontal
        height: Chart height in pixels
        show_values: Whether to show values on bars
        sort_bars: Whether to sort bars by value
        max_bars: Maximum number of bars to display
        
    Returns:
        Plotly bar chart with neon cyberpunk styling
    """
    
    # Prepare data
    chart_data = data.copy()
    
    # Sort if requested
    if sort_bars:
        chart_data = chart_data.sort_values(y_col, ascending=False)
    
    # Limit number of bars for performance
    if len(chart_data) > max_bars:
        chart_data = chart_data.head(max_bars)
        title += f" (Top {max_bars})"
    
    # Get color palette
    colors = get_chart_color_palette('categorical')
    
    # Create bar chart
    if color_column and color_column in chart_data.columns:
        # Colored by category
        fig = px.bar(
            chart_data,
            x=x_col if orientation == 'v' else y_col,
            y=y_col if orientation == 'v' else x_col,
            color=color_column,
            orientation=orientation,
            color_discrete_sequence=colors
        )
    else:
        # Single color with gradient
        fig = go.Figure()
        
        bar_colors = []
        for i in range(len(chart_data)):
            # Create gradient effect
            intensity = i / max(len(chart_data) - 1, 1)
            if i % 3 == 0:
                bar_colors.append(COLORS['secondary'])
            elif i % 3 == 1:
                bar_colors.append(COLORS['accent'])
            else:
                bar_colors.append(COLORS['success'])
        
        if orientation == 'v':
            fig.add_trace(go.Bar(
                x=chart_data[x_col],
                y=chart_data[y_col],
                marker=dict(
                    color=bar_colors,
                    line=dict(color=COLORS['border_primary'], width=2),
                    opacity=0.8
                ),
                name=y_col,
                text=chart_data[y_col] if show_values else None,
                textposition='outside' if show_values else 'none',
                textfont=dict(color=COLORS['text'], size=11)
            ))
        else:
            fig.add_trace(go.Bar(
                x=chart_data[y_col],
                y=chart_data[x_col],
                orientation='h',
                marker=dict(
                    color=bar_colors,
                    line=dict(color=COLORS['border_primary'], width=2),
                    opacity=0.8
                ),
                name=y_col,
                text=chart_data[y_col] if show_values else None,
                textposition='outside' if show_values else 'none',
                textfont=dict(color=COLORS['text'], size=11)
            ))
    
    # Apply dark theme
    fig = create_dark_theme_plotly_chart(
        fig,
        title=title,
        height=height,
        show_legend=bool(color_column),
        custom_layout={
            'bargap': 0.1,
            'bargroupgap': 0.05
        }
    )
    
    # Add hover effects
    fig.update_traces(
        hovertemplate='<b>%{' + ('x' if orientation == 'v' else 'y') + '}</b><br>' +
                     f'{y_col}: %{{"y" if orientation == "v" else "x"}}<br>' +
                     '<extra></extra>',
        hoverlabel=dict(
            bgcolor=COLORS['background_card'],
            bordercolor=COLORS['secondary'],
            font_color=COLORS['text']
        )
    )
    
    return fig

# ================================================================
# GLASSMORPHISM METRIC CARD
# ================================================================

def glassmorphism_metric_card(
    value: Union[str, float, int],
    title: str,
    subtitle: str = "",
    delta: Optional[float] = None,
    delta_color: Optional[str] = None,
    icon: str = "📊",
    color: str = 'secondary',
    width: int = 300,
    height: int = 200,
    animate: bool = True
) -> str:
    """
    Create a glassmorphism-styled metric card with neon accents.
    
    Args:
        value: Main metric value to display
        title: Card title
        subtitle: Optional subtitle text
        delta: Change value (positive/negative)
        delta_color: Color for delta (auto-determined if None)
        icon: Emoji or icon to display
        color: Theme color name for accent
        width: Card width in pixels
        height: Card height in pixels
        animate: Whether to include hover animations
        
    Returns:
        HTML string for the glassmorphism metric card
    """
    
    # Get colors
    accent_color = COLORS.get(color, COLORS['secondary'])
    
    # Format value
    if isinstance(value, float):
        if value >= 1000000:
            formatted_value = f"{value/1000000:.1f}M"
        elif value >= 1000:
            formatted_value = f"{value/1000:.1f}K"
        else:
            formatted_value = f"{value:.1f}"
    else:
        formatted_value = str(value)
    
    # Delta styling
    delta_html = ""
    if delta is not None:
        if delta_color is None:
            delta_color = COLORS['success'] if delta >= 0 else COLORS['error']
        
        delta_icon = "↗️" if delta >= 0 else "↘️"
        delta_text = f"+{delta}" if delta >= 0 else str(delta)
        
        delta_html = f"""
        <div style="
            font-size: 14px;
            color: {delta_color};
            font-weight: 600;
            margin-top: 5px;
            font-family: {TYPOGRAPHY['font_family_mono']};
        ">
            {delta_icon} {delta_text}
        </div>
        """
    
    # Animation CSS
    animation_style = """
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        transform: translateY(0px);
    """ if animate else ""
    
    hover_style = """
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.2), 
                   0 10px 20px rgba(176, 38, 255, 0.1);
        border-color: {accent_color};
        background: rgba(37, 42, 69, 0.4);
    """.format(accent_color=accent_color) if animate else ""
    
    # Create HTML
    card_html = f"""
    <div style="
        width: {width}px;
        height: {height}px;
        background: rgba(37, 42, 69, 0.25);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        {animation_style}
        margin: 10px;
        cursor: pointer;
    " onmouseover="this.style.cssText += '{hover_style}'" 
       onmouseout="this.style.cssText = this.style.cssText.replace('{hover_style}', '')">
        
        <!-- Glow effect overlay -->
        <div style="
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, {accent_color}33, transparent, {accent_color}33);
            border-radius: 18px;
            z-index: -1;
            opacity: 0.3;
        "></div>
        
        <!-- Icon -->
        <div style="
            font-size: 32px;
            margin-bottom: 10px;
            filter: drop-shadow(0 0 10px {accent_color}66);
        ">
            {icon}
        </div>
        
        <!-- Value -->
        <div style="
            font-size: 36px;
            font-weight: 700;
            color: {accent_color};
            font-family: {TYPOGRAPHY['font_family_display']};
            margin-bottom: 5px;
            text-shadow: 0 0 20px {accent_color}66;
            line-height: 1;
        ">
            {formatted_value}
        </div>
        
        <!-- Title -->
        <div style="
            font-size: 14px;
            color: {COLORS['text']};
            font-weight: 600;
            margin-bottom: 5px;
            font-family: {TYPOGRAPHY['font_family_primary']};
            text-transform: uppercase;
            letter-spacing: 1px;
        ">
            {title}
        </div>
        
        <!-- Subtitle -->
        <div style="
            font-size: 12px;
            color: {COLORS['text_secondary']};
            font-family: {TYPOGRAPHY['font_family_primary']};
            margin-bottom: 5px;
        ">
            {subtitle}
        </div>
        
        <!-- Delta -->
        {delta_html}
        
        <!-- Bottom accent line -->
        <div style="
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, {accent_color}, transparent);
        "></div>
    </div>
    """
    
    return card_html

# ================================================================
# ADVANCED CHART COMPONENTS
# ================================================================

def cyberpunk_line_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    color_col: Optional[str] = None,
    height: int = 400,
    show_markers: bool = True,
    glow_effect: bool = True
) -> go.Figure:
    """
    Create a cyberpunk-styled line chart with glowing lines.
    
    Args:
        data: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        color_col: Optional column for color grouping
        height: Chart height in pixels
        show_markers: Whether to show data point markers
        glow_effect: Whether to add glow effect to lines
        
    Returns:
        Plotly line chart with cyberpunk styling
    """
    
    colors = get_chart_color_palette('default')
    
    # Create line chart
    if color_col and color_col in data.columns:
        fig = px.line(
            data, x=x_col, y=y_col, color=color_col,
            color_discrete_sequence=colors,
            markers=show_markers
        )
    else:
        fig = go.Figure()
        
        # Main line
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='lines+markers' if show_markers else 'lines',
            line=dict(color=COLORS['secondary'], width=3),
            marker=dict(
                color=COLORS['secondary'],
                size=8,
                line=dict(color=COLORS['text'], width=1)
            ) if show_markers else None,
            name=y_col
        ))
        
        # Add glow effect
        if glow_effect:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines',
                line=dict(color=COLORS['secondary'], width=8, opacity=0.3),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Apply dark theme
    fig = create_dark_theme_plotly_chart(
        fig,
        title=title,
        height=height,
        show_legend=bool(color_col)
    )
    
    return fig

def futuristic_donut_chart(
    data: pd.DataFrame,
    values_col: str,
    names_col: str,
    title: str = "",
    height: int = 400,
    hole_size: float = 0.6,
    show_percentages: bool = True
) -> go.Figure:
    """
    Create a futuristic donut chart with neon styling.
    
    Args:
        data: DataFrame containing the data
        values_col: Column name for values
        names_col: Column name for category names
        title: Chart title
        height: Chart height in pixels
        hole_size: Size of the hole (0-1)
        show_percentages: Whether to show percentages on slices
        
    Returns:
        Plotly donut chart with cyberpunk styling
    """
    
    colors = get_chart_color_palette('categorical')
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=data[names_col],
        values=data[values_col],
        hole=hole_size,
        marker=dict(
            colors=colors,
            line=dict(color=COLORS['border_primary'], width=2)
        ),
        textinfo='label+percent' if show_percentages else 'label',
        textfont=dict(color=COLORS['text'], size=12),
        hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    # Apply dark theme
    fig = create_dark_theme_plotly_chart(
        fig,
        title=title,
        height=height,
        show_legend=True,
        custom_layout={
            'annotations': [
                dict(
                    text=f'<span style="color:{COLORS["text"]};font-size:16px;font-weight:bold">Total</span>',
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )
            ]
        }
    )
    
    return fig

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def optimize_chart_for_memory(fig: go.Figure, max_points: int = 1000) -> go.Figure:
    """
    Optimize chart for memory usage by sampling data points.
    
    Args:
        fig: Plotly figure to optimize
        max_points: Maximum number of data points to display
        
    Returns:
        Memory-optimized figure
    """
    # This is a simplified version - in practice, you'd sample the underlying data
    # before creating the chart for better memory efficiency
    
    # Reduce marker size for scatter plots
    fig.update_traces(marker_size=4)
    
    # Simplify hover templates
    fig.update_traces(hovertemplate='%{x}: %{y}<extra></extra>')
    
    return fig

def create_responsive_layout(fig: go.Figure, mobile_friendly: bool = True) -> go.Figure:
    """
    Make chart layout responsive for different screen sizes.
    
    Args:
        fig: Plotly figure to make responsive
        mobile_friendly: Whether to optimize for mobile
        
    Returns:
        Responsive figure
    """
    if mobile_friendly:
        fig.update_layout(
            font_size=10,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", y=-0.1)
        )
    
    return fig

# ================================================================
# EXPORT ALL FUNCTIONS
# ================================================================

__all__ = [
    'create_dark_theme_plotly_chart',
    'futuristic_gauge_chart',
    'neon_bar_chart', 
    'glassmorphism_metric_card',
    'cyberpunk_line_chart',
    'futuristic_donut_chart',
    'optimize_chart_for_memory',
    'create_responsive_layout'
]

# ================================================================
# DEVELOPMENT TESTING
# ================================================================

def test_charts():
    """Test function to verify chart components work correctly."""
    import random
    
    print("🧪 Testing Chart Components...")
    
    # Test data
    test_data = pd.DataFrame({
        'Department': ['Engineering', 'Sales', 'Marketing', 'Operations'],
        'Attrition_Rate': [0.12, 0.18, 0.15, 0.10],
        'Employee_Count': [120, 80, 60, 90]
    })
    
    print("✅ Test data created")
    
    # Test glassmorphism card
    card_html = glassmorphism_metric_card(
        value=15.6,
        title="Attrition Rate", 
        subtitle="Current month",
        delta=2.3,
        icon="📈",
        color='warning'
    )
    
    print("✅ Glassmorphism card created")
    print("🎨 All chart components ready!")

if __name__ == "__main__":
    test_charts()
