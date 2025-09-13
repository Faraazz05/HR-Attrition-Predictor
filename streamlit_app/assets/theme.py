"""
HR Attrition Predictor - Dark Cyberpunk Theme Configuration
==========================================================
Comprehensive dark theme with cyberpunk aesthetics, glassmorphism effects,
and professional data visualization styling for Streamlit applications.

Author: HR Analytics Team
Date: September 2025
"""

from typing import Dict, Any, List
import streamlit as st

# ================================================================
# COLOR PALETTE - CYBERPUNK DARK THEME
# ================================================================

COLORS = {
    # Primary Colors - Deep Space Theme
    'primary': '#0A0E27',        # Deep space blue (main background)
    'secondary': '#00D4FF',      # Electric blue (highlights)
    'accent': '#B026FF',         # Neon purple (interactive elements)
    
    # Status Colors - Cyberpunk Vibes
    'success': '#00FF88',        # Matrix green (positive indicators)
    'warning': '#FF6B35',        # Cyber orange (attention needed)
    'error': '#FF2D75',          # Neon pink (errors/high risk)
    'info': '#00E5FF',           # Bright cyan (information)
    
    # Text Colors - High Contrast
    'text': '#F0F8FF',           # Ice white (primary text)
    'text_secondary': '#B8C5D1', # Muted blue-white (secondary text)
    'text_muted': '#6C7B8A',     # Dark grey-blue (muted text)
    
    # Background Variations
    'background': '#0A0E27',     # Main background
    'background_light': '#1A1F3A', # Lighter background (cards)
    'background_card': '#252A45', # Card backgrounds
    'background_hover': '#2D3552', # Hover states
    
    # Interactive Elements
    'button_primary': '#00D4FF',  # Primary buttons
    'button_secondary': '#B026FF', # Secondary buttons
    'button_success': '#00FF88',  # Success buttons
    'button_danger': '#FF2D75',   # Danger buttons
    
    # Data Visualization
    'chart_positive': '#00FF88',  # Positive data points
    'chart_negative': '#FF2D75',  # Negative data points
    'chart_neutral': '#00D4FF',   # Neutral data points
    'chart_accent': '#B026FF',    # Accent data points
    
    # Gradients (for advanced styling)
    'gradient_primary': 'linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%)',
    'gradient_card': 'linear-gradient(135deg, #252A45 0%, #2D3552 100%)',
    'gradient_glow': 'linear-gradient(135deg, #00D4FF 0%, #B026FF 100%)',
    
    # Borders and Lines
    'border_primary': '#00D4FF',  # Primary borders
    'border_secondary': '#B026FF', # Secondary borders
    'border_muted': '#3A4158',    # Muted borders
    
    # Special Effects
    'glow_blue': '#00D4FF33',     # Blue glow (with transparency)
    'glow_purple': '#B026FF33',   # Purple glow
    'glow_green': '#00FF8833',    # Green glow
}

# ================================================================
# TYPOGRAPHY CONFIGURATION
# ================================================================

TYPOGRAPHY = {
    'font_family_primary': "'Roboto', 'Segoe UI', 'Helvetica Neue', sans-serif",
    'font_family_mono': "'Fira Code', 'Monaco', 'Consolas', monospace",
    'font_family_display': "'Orbitron', 'Roboto', sans-serif",
    
    'font_sizes': {
        'xs': '0.75rem',    # 12px
        'sm': '0.875rem',   # 14px
        'base': '1rem',     # 16px
        'lg': '1.125rem',   # 18px
        'xl': '1.25rem',    # 20px
        '2xl': '1.5rem',    # 24px
        '3xl': '1.875rem',  # 30px
        '4xl': '2.25rem',   # 36px
        '5xl': '3rem',      # 48px
    },
    
    'font_weights': {
        'light': '300',
        'normal': '400',
        'medium': '500',
        'semibold': '600',
        'bold': '700',
        'extrabold': '800',
    }
}

# ================================================================
# COMPONENT SIZING AND SPACING
# ================================================================

SPACING = {
    'xs': '0.25rem',   # 4px
    'sm': '0.5rem',    # 8px
    'md': '1rem',      # 16px
    'lg': '1.5rem',    # 24px
    'xl': '2rem',      # 32px
    '2xl': '3rem',     # 48px
    '3xl': '4rem',     # 64px
}

SIZING = {
    'border_radius': {
        'sm': '0.25rem',   # 4px
        'md': '0.5rem',    # 8px
        'lg': '0.75rem',   # 12px
        'xl': '1rem',      # 16px
        '2xl': '1.5rem',   # 24px
        'full': '9999px',  # Fully rounded
    },
    
    'shadows': {
        'sm': '0 1px 2px rgba(0, 212, 255, 0.1)',
        'md': '0 4px 6px rgba(0, 212, 255, 0.1), 0 2px 4px rgba(176, 38, 255, 0.06)',
        'lg': '0 10px 15px rgba(0, 212, 255, 0.1), 0 4px 6px rgba(176, 38, 255, 0.05)',
        'xl': '0 20px 25px rgba(0, 212, 255, 0.1), 0 10px 10px rgba(176, 38, 255, 0.04)',
        'glow': '0 0 20px rgba(0, 212, 255, 0.3), 0 0 40px rgba(176, 38, 255, 0.2)',
    }
}

# ================================================================
# PLOTLY DARK THEME CONFIGURATION
# ================================================================

def get_plotly_dark_theme() -> Dict[str, Any]:
    """
    Generate comprehensive Plotly dark theme configuration.
    
    Returns:
        Dictionary containing Plotly theme settings
    """
    return {
        'layout': {
            'paper_bgcolor': COLORS['background'],
            'plot_bgcolor': COLORS['background_light'],
            'font': {
                'color': COLORS['text'],
                'family': TYPOGRAPHY['font_family_primary'],
                'size': 14
            },
            'title': {
                'font': {
                    'color': COLORS['text'],
                    'size': 24,
                    'family': TYPOGRAPHY['font_family_display']
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            'colorway': [
                COLORS['secondary'],
                COLORS['accent'], 
                COLORS['success'],
                COLORS['warning'],
                COLORS['error'],
                COLORS['info'],
                COLORS['chart_accent'],
                COLORS['text_secondary']
            ],
            'grid': {
                'xside': 'bottom',
                'yside': 'left'
            },
            'xaxis': {
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': COLORS['border_muted'],
                'showline': True,
                'linewidth': 1,
                'linecolor': COLORS['border_primary'],
                'tickfont': {
                    'color': COLORS['text_secondary'],
                    'size': 12
                },
                'titlefont': {
                    'color': COLORS['text'],
                    'size': 14
                }
            },
            'yaxis': {
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': COLORS['border_muted'],
                'showline': True,
                'linewidth': 1,
                'linecolor': COLORS['border_primary'],
                'tickfont': {
                    'color': COLORS['text_secondary'],
                    'size': 12
                },
                'titlefont': {
                    'color': COLORS['text'],
                    'size': 14
                }
            },
            'legend': {
                'font': {
                    'color': COLORS['text'],
                    'size': 12
                },
                'bgcolor': 'rgba(0, 0, 0, 0)',
                'bordercolor': COLORS['border_muted'],
                'borderwidth': 1
            },
            'hoverlabel': {
                'bgcolor': COLORS['background_card'],
                'bordercolor': COLORS['border_primary'],
                'font': {
                    'color': COLORS['text'],
                    'size': 12
                }
            }
        },
        
        # Specific chart type configurations
        'bar': {
            'marker': {
                'line': {
                    'color': COLORS['border_primary'],
                    'width': 1
                }
            }
        },
        
        'scatter': {
            'marker': {
                'line': {
                    'color': COLORS['border_primary'],
                    'width': 0.5
                }
            }
        },
        
        'heatmap': {
            'colorscale': [
                [0, COLORS['background_light']],
                [0.5, COLORS['secondary']],
                [1, COLORS['accent']]
            ]
        }
    }

# ================================================================
# GLASSMORPHISM CSS STYLING
# ================================================================

def get_glassmorphism_css() -> str:
    """
    Generate glassmorphism CSS for modern, translucent UI elements.
    
    Returns:
        CSS string for glassmorphism effects
    """
    return f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Orbitron:wght@400;700;900&family=Fira+Code:wght@300;400;500&display=swap');
    
    /* Root Variables */
    :root {{
        --primary-color: {COLORS['primary']};
        --secondary-color: {COLORS['secondary']};
        --accent-color: {COLORS['accent']};
        --success-color: {COLORS['success']};
        --warning-color: {COLORS['warning']};
        --error-color: {COLORS['error']};
        --text-color: {COLORS['text']};
        --text-secondary: {COLORS['text_secondary']};
        --background: {COLORS['background']};
        --background-light: {COLORS['background_light']};
        --background-card: {COLORS['background_card']};
        --border-primary: {COLORS['border_primary']};
        --glow-blue: {COLORS['glow_blue']};
        --glow-purple: {COLORS['glow_purple']};
    }}
    
    /* Global Styles */
    .stApp {{
        background: {COLORS['gradient_primary']};
        font-family: {TYPOGRAPHY['font_family_primary']};
        color: {COLORS['text']};
    }}
    
    /* Glassmorphism Card */
    .glassmorphism-card {{
        background: rgba(37, 42, 69, 0.25);
        backdrop-filter: blur(10px);
        border-radius: {SIZING['border_radius']['lg']};
        border: 1px solid rgba(0, 212, 255, 0.2);
        box-shadow: {SIZING['shadows']['lg']};
        padding: {SPACING['lg']};
        margin: {SPACING['md']} 0;
        transition: all 0.3s ease;
    }}
    
    .glassmorphism-card:hover {{
        background: rgba(37, 42, 69, 0.35);
        border-color: rgba(0, 212, 255, 0.4);
        box-shadow: {SIZING['shadows']['glow']};
        transform: translateY(-2px);
    }}
    
    /* Header Styling */
    .main-header {{
        background: {COLORS['gradient_glow']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: {TYPOGRAPHY['font_family_display']};
        font-size: {TYPOGRAPHY['font_sizes']['4xl']};
        font-weight: {TYPOGRAPHY['font_weights']['bold']};
        text-align: center;
        margin-bottom: {SPACING['xl']};
        text-shadow: 0 0 30px {COLORS['glow_blue']};
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: rgba(26, 31, 58, 0.6);
        backdrop-filter: blur(8px);
        border-radius: {SIZING['border_radius']['xl']};
        border: 1px solid {COLORS['border_muted']};
        padding: {SPACING['lg']};
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: {SIZING['shadows']['md']};
    }}
    
    .metric-card:hover {{
        border-color: {COLORS['secondary']};
        box-shadow: 0 0 25px {COLORS['glow_blue']};
        transform: scale(1.02);
    }}
    
    .metric-value {{
        font-size: {TYPOGRAPHY['font_sizes']['3xl']};
        font-weight: {TYPOGRAPHY['font_weights']['bold']};
        color: {COLORS['secondary']};
        text-shadow: 0 0 10px {COLORS['glow_blue']};
    }}
    
    .metric-label {{
        font-size: {TYPOGRAPHY['font_sizes']['sm']};
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: {SPACING['xs']};
    }}
    
    /* Button Styling */
    .stButton > button {{
        background: {COLORS['gradient_glow']} !important;
        color: {COLORS['text']} !important;
        border: 1px solid {COLORS['border_primary']} !important;
        border-radius: {SIZING['border_radius']['lg']} !important;
        font-family: {TYPOGRAPHY['font_family_primary']} !important;
        font-weight: {TYPOGRAPHY['font_weights']['medium']} !important;
        padding: {SPACING['md']} {SPACING['lg']} !important;
        transition: all 0.3s ease !important;
        box-shadow: {SIZING['shadows']['md']} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }}
    
    .stButton > button:hover {{
        background: {COLORS['secondary']} !important;
        box-shadow: {SIZING['shadows']['glow']} !important;
        transform: translateY(-2px) !important;
        border-color: {COLORS['accent']} !important;
    }}
    
    /* Sidebar Styling */
    .css-1d391kg {{
        background: {COLORS['gradient_card']} !important;
        border-right: 1px solid {COLORS['border_primary']} !important;
    }}
    
    .sidebar-header {{
        color: {COLORS['secondary']};
        font-family: {TYPOGRAPHY['font_family_display']};
        font-size: {TYPOGRAPHY['font_sizes']['xl']};
        font-weight: {TYPOGRAPHY['font_weights']['bold']};
        text-align: center;
        padding: {SPACING['lg']};
        border-bottom: 1px solid {COLORS['border_muted']};
        margin-bottom: {SPACING['lg']};
    }}
    
    /* Select Box Styling */
    .stSelectbox > div > div {{
        background: rgba(37, 42, 69, 0.8) !important;
        color: {COLORS['text']} !important;
        border: 1px solid {COLORS['border_muted']} !important;
        border-radius: {SIZING['border_radius']['md']} !important;
    }}
    
    .stSelectbox > div > div:hover {{
        border-color: {COLORS['secondary']} !important;
        box-shadow: 0 0 10px {COLORS['glow_blue']} !important;
    }}
    
    /* Input Field Styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background: rgba(37, 42, 69, 0.8) !important;
        color: {COLORS['text']} !important;
        border: 1px solid {COLORS['border_muted']} !important;
        border-radius: {SIZING['border_radius']['md']} !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {COLORS['secondary']} !important;
        box-shadow: 0 0 10px {COLORS['glow_blue']} !important;
    }}
    
    /* Progress Bar Styling */
    .stProgress > div > div > div {{
        background: {COLORS['gradient_glow']} !important;
        border-radius: {SIZING['border_radius']['full']} !important;
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: {SPACING['md']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(37, 42, 69, 0.6) !important;
        color: {COLORS['text_secondary']} !important;
        border: 1px solid {COLORS['border_muted']} !important;
        border-radius: {SIZING['border_radius']['md']} !important;
        padding: {SPACING['md']} {SPACING['lg']} !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(37, 42, 69, 0.8) !important;
        border-color: {COLORS['secondary']} !important;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {COLORS['secondary']} !important;
        color: {COLORS['primary']} !important;
        border-color: {COLORS['secondary']} !important;
        box-shadow: 0 0 15px {COLORS['glow_blue']} !important;
    }}
    
    /* Alert Styling */
    .alert-success {{
        background: rgba(0, 255, 136, 0.1) !important;
        color: {COLORS['success']} !important;
        border: 1px solid {COLORS['success']} !important;
        border-radius: {SIZING['border_radius']['lg']} !important;
        padding: {SPACING['md']} !important;
        margin: {SPACING['md']} 0 !important;
    }}
    
    .alert-warning {{
        background: rgba(255, 107, 53, 0.1) !important;
        color: {COLORS['warning']} !important;
        border: 1px solid {COLORS['warning']} !important;
        border-radius: {SIZING['border_radius']['lg']} !important;
        padding: {SPACING['md']} !important;
        margin: {SPACING['md']} 0 !important;
    }}
    
    .alert-error {{
        background: rgba(255, 45, 117, 0.1) !important;
        color: {COLORS['error']} !important;
        border: 1px solid {COLORS['error']} !important;
        border-radius: {SIZING['border_radius']['lg']} !important;
        padding: {SPACING['md']} !important;
        margin: {SPACING['md']} 0 !important;
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader {{
        background: rgba(37, 42, 69, 0.6) !important;
        color: {COLORS['text']} !important;
        border: 1px solid {COLORS['border_muted']} !important;
        border-radius: {SIZING['border_radius']['md']} !important;
    }}
    
    .streamlit-expanderContent {{
        background: rgba(26, 31, 58, 0.4) !important;
        border: 1px solid {COLORS['border_muted']} !important;
        border-top: none !important;
        border-radius: 0 0 {SIZING['border_radius']['md']} {SIZING['border_radius']['md']} !important;
    }}
    
    /* Data Frame Styling */
    .stDataFrame {{
        background: rgba(37, 42, 69, 0.6) !important;
        border-radius: {SIZING['border_radius']['lg']} !important;
        border: 1px solid {COLORS['border_muted']} !important;
        overflow: hidden !important;
    }}
    
    /* Custom Classes for Special Elements */
    .prediction-result-high {{
        background: rgba(255, 45, 117, 0.2) !important;
        border: 2px solid {COLORS['error']} !important;
        border-radius: {SIZING['border_radius']['xl']} !important;
        padding: {SPACING['lg']} !important;
        text-align: center !important;
        box-shadow: 0 0 30px {COLORS['error']}33 !important;
        animation: pulse-red 2s infinite !important;
    }}
    
    .prediction-result-medium {{
        background: rgba(255, 107, 53, 0.2) !important;
        border: 2px solid {COLORS['warning']} !important;
        border-radius: {SIZING['border_radius']['xl']} !important;
        padding: {SPACING['lg']} !important;
        text-align: center !important;
        box-shadow: 0 0 30px {COLORS['warning']}33 !important;
    }}
    
    .prediction-result-low {{
        background: rgba(0, 255, 136, 0.2) !important;
        border: 2px solid {COLORS['success']} !important;
        border-radius: {SIZING['border_radius']['xl']} !important;
        padding: {SPACING['lg']} !important;
        text-align: center !important;
        box-shadow: 0 0 30px {COLORS['success']}33 !important;
    }}
    
    /* Animations */
    @keyframes pulse-red {{
        0%, 100% {{ box-shadow: 0 0 30px {COLORS['error']}33; }}
        50% {{ box-shadow: 0 0 50px {COLORS['error']}66; }}
    }}
    
    @keyframes glow {{
        0%, 100% {{ box-shadow: 0 0 20px {COLORS['glow_blue']}; }}
        50% {{ box-shadow: 0 0 40px {COLORS['glow_purple']}; }}
    }}
    
    .glow-animation {{
        animation: glow 3s ease-in-out infinite !important;
    }}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['background_light']};
        border-radius: {SIZING['border_radius']['full']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['secondary']};
        border-radius: {SIZING['border_radius']['full']};
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['accent']};
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    </style>
    """

# ================================================================
# CHART COLOR PALETTES
# ================================================================

def get_chart_color_palette(chart_type: str = 'default') -> List[str]:
    """
    Get color palette for different chart types.
    
    Args:
        chart_type: Type of chart ('default', 'categorical', 'sequential', 'diverging')
        
    Returns:
        List of hex color codes
    """
    palettes = {
        'default': [
            COLORS['secondary'],
            COLORS['accent'],
            COLORS['success'], 
            COLORS['warning'],
            COLORS['error'],
            COLORS['info'],
            COLORS['text_secondary']
        ],
        
        'categorical': [
            COLORS['secondary'],
            COLORS['accent'],
            COLORS['success'],
            COLORS['warning'],
            COLORS['chart_accent'],
            COLORS['info'],
            COLORS['error']
        ],
        
        'sequential': [
            COLORS['background_light'],
            COLORS['secondary'] + '66',
            COLORS['secondary'] + 'CC',
            COLORS['secondary']
        ],
        
        'diverging': [
            COLORS['error'],
            COLORS['warning'], 
            COLORS['text_secondary'],
            COLORS['success'],
            COLORS['secondary']
        ],
        
        'risk_levels': [
            COLORS['success'],    # Low risk
            COLORS['warning'],    # Medium risk  
            COLORS['error']       # High risk
        ]
    }
    
    return palettes.get(chart_type, palettes['default'])

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def apply_custom_css():
    """Apply the custom CSS to Streamlit app."""
    st.markdown(get_glassmorphism_css(), unsafe_allow_html=True)

def get_color(color_name: str) -> str:
    """
    Get color value by name.
    
    Args:
        color_name: Name of the color from COLORS dictionary
        
    Returns:
        Hex color code
    """
    return COLORS.get(color_name, COLORS['text'])

def create_metric_html(value: str, label: str, color: str = 'secondary') -> str:
    """
    Create HTML for a metric display card.
    
    Args:
        value: The metric value to display
        label: The metric label
        color: Color theme for the metric
        
    Returns:
        HTML string for the metric card
    """
    color_value = COLORS.get(color, COLORS['secondary'])
    
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color_value};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def create_alert_html(message: str, alert_type: str = 'info') -> str:
    """
    Create HTML for alert messages.
    
    Args:
        message: Alert message text
        alert_type: Type of alert ('success', 'warning', 'error', 'info')
        
    Returns:
        HTML string for the alert
    """
    return f"""
    <div class="alert-{alert_type}">
        {message}
    </div>
    """

def create_prediction_result_html(risk_level: str, probability: float) -> str:
    """
    Create HTML for prediction results with appropriate styling.
    
    Args:
        risk_level: Risk level ('Low', 'Medium', 'High')
        probability: Attrition probability (0-1)
        
    Returns:
        HTML string for prediction result
    """
    risk_class = f"prediction-result-{risk_level.lower()}"
    percentage = f"{probability * 100:.1f}%"
    
    return f"""
    <div class="{risk_class}">
        <h2 style="margin: 0; font-family: {TYPOGRAPHY['font_family_display']};">
            🎯 Attrition Risk: {risk_level.upper()}
        </h2>
        <h3 style="margin: 10px 0 0 0; font-size: {TYPOGRAPHY['font_sizes']['2xl']};">
            Probability: {percentage}
        </h3>
    </div>
    """

# ================================================================
# EXPORT ALL CONFIGURATIONS
# ================================================================

__all__ = [
    'COLORS',
    'TYPOGRAPHY', 
    'SPACING',
    'SIZING',
    'get_plotly_dark_theme',
    'get_glassmorphism_css',
    'get_chart_color_palette',
    'apply_custom_css',
    'get_color',
    'create_metric_html',
    'create_alert_html',
    'create_prediction_result_html'
]

# ================================================================
# THEME PREVIEW (FOR DEVELOPMENT)
# ================================================================

def preview_theme():
    """Preview the theme configuration (for development use)."""
    print("🎨 HR Attrition Predictor - Dark Cyberpunk Theme")
    print("=" * 50)
    print(f"Primary Color: {COLORS['primary']}")
    print(f"Secondary Color: {COLORS['secondary']}")
    print(f"Accent Color: {COLORS['accent']}")
    print(f"Text Color: {COLORS['text']}")
    print(f"Success Color: {COLORS['success']}")
    print(f"Warning Color: {COLORS['warning']}")
    print(f"Error Color: {COLORS['error']}")
    print("\n✅ Theme configuration loaded successfully!")

if __name__ == "__main__":
    preview_theme()
