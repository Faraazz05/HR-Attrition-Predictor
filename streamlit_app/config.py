"""
HR Attrition Predictor - Streamlit Application Configuration
===========================================================
Centralized configuration for Streamlit app settings, page configuration,
navigation, and application-wide constants.

Author: HR Analytics Team
Date: September 2025
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional
import os
import sys
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from streamlit_app.assets.theme import COLORS, TYPOGRAPHY

# ================================================================
# STREAMLIT PAGE CONFIGURATION
# ================================================================

PAGE_CONFIG = {
    "page_title": "HR Attrition Predictor",
    "page_icon": "🏢",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        'Get Help': 'https://github.com/your-repo/hr-attrition-predictor',
        'Report a bug': 'https://github.com/your-repo/hr-attrition-predictor/issues',
        'About': """
        # HR Attrition Predictor 🏢
        
        **Enterprise-grade AI-powered employee attrition prediction system**
        
        ### Features:
        - 🤖 Machine Learning predictions using XGBoost and ensemble methods
        - 🔍 SHAP explainability for transparent decision-making
        - 📊 Interactive dashboards and analytics
        - 👥 Individual employee risk assessment
        - 💡 Actionable business insights and recommendations
        - 📧 Automated HR intervention workflows
        
        ### Built with:
        - Streamlit for the web interface
        - scikit-learn & XGBoost for ML models
        - SHAP for model explainability
        - Plotly for interactive visualizations
        
        **Version:** 1.0.0  
        **Author:** HR Analytics Team  
        **Date:** September 2025
        """
    }
}

# ================================================================
# APPLICATION NAVIGATION STRUCTURE
# ================================================================

NAVIGATION_PAGES = {
    "🏠 Dashboard": {
        "title": "Executive Dashboard",
        "description": "High-level KPIs and attrition overview",
        "icon": "🏠",
        "file": "01_🏠_Dashboard.py",
        "category": "overview"
    },
    "📊 Analytics": {
        "title": "Data Analytics",
        "description": "Deep-dive analysis and insights",
        "icon": "📊", 
        "file": "02_📊_Analytics.py",
        "category": "analysis"
    },
    "🔍 Predictions": {
        "title": "ML Predictions",
        "description": "Individual and batch predictions",
        "icon": "🔍",
        "file": "03_🔍_Predictions.py",
        "category": "ml"
    },
    "👥 Employee Management": {
        "title": "Employee Directory",
        "description": "Employee profiles and risk management",
        "icon": "👥",
        "file": "04_👥_Employee_Management.py",
        "category": "management"
    },
    "💡 Insights": {
        "title": "AI Insights",
        "description": "SHAP explanations and recommendations",
        "icon": "💡",
        "file": "05_💡_Insights.py",
        "category": "insights"
    },
    "⚙️ Admin": {
        "title": "System Administration",
        "description": "Model performance and system health",
        "icon": "⚙️",
        "file": "06_⚙️_Admin.py",
        "category": "admin"
    }
}

# ================================================================
# MODEL CONFIGURATION
# ================================================================

MODEL_CONFIG = {
    "model_paths": {
        "best_model": "models/best_model.pkl",
        "logistic_regression": "models/logistic_regression_optimized.pkl",
        "random_forest": "models/random_forest_optimized.pkl",
        "xgboost": "models/xgboost_optimized.pkl",
        "ensemble": "models/ensemble_optimized.pkl"
    },
    
    "preprocessing_paths": {
        "scaler": "models/feature_scaler.pkl",
        "label_encoders": "models/label_encoders.pkl",
        "target_encoder": "models/target_encoder.pkl",
        "feature_names": "models/feature_names.pkl"
    },
    
    "model_info": {
        "primary_metric": "roc_auc",
        "threshold": 0.5,
        "risk_thresholds": {
            "low": 0.3,
            "medium": 0.7,
            "high": 1.0
        },
        "feature_importance_top_n": 15,
        "shap_sample_size": 100  # For memory optimization
    }
}

# ================================================================
# DATA CONFIGURATION
# ================================================================

DATA_CONFIG = {
    "data_paths": {
        "employee_data": "data/synthetic/hr_employees.csv",
        "processed_data": "data/processed/",
        "reports": "reports/",
        "exports": "exports/"
    },
    
    "display_columns": {
        "employee_overview": [
            'EmployeeID', 'FullName', 'Department', 'JobRole', 
            'Age', 'YearsAtCompany', 'MonthlyIncome', 'Attrition'
        ],
        "risk_assessment": [
            'EmployeeID', 'FullName', 'Department', 'JobRole',
            'RiskLevel', 'AttritionProbability', 'RiskFactors'
        ],
        "performance_metrics": [
            'EmployeeID', 'FullName', 'PerformanceRating', 
            'JobSatisfaction', 'WorkLifeBalance', 'TrainingTimesLastYear'
        ]
    },
    
    "filters": {
        "departments": "All Departments",
        "job_roles": "All Job Roles", 
        "risk_levels": ["All", "Low", "Medium", "High"],
        "age_ranges": ["All", "Under 30", "30-40", "41-50", "Over 50"],
        "tenure_ranges": ["All", "0-1 years", "1-3 years", "3-5 years", "5+ years"]
    }
}

# ================================================================
# UI CONFIGURATION
# ================================================================

UI_CONFIG = {
    "theme": {
        "primary_color": COLORS['secondary'],
        "background_color": COLORS['background'],
        "secondary_background_color": COLORS['background_light'],
        "text_color": COLORS['text'],
        "font": TYPOGRAPHY['font_family_primary']
    },
    
    "layout": {
        "sidebar_width": 300,
        "main_content_max_width": 1200,
        "chart_height": 400,
        "table_height": 500,
        "card_height": 200
    },
    
    "animations": {
        "enable_animations": True,
        "transition_speed": "0.3s",
        "hover_effects": True,
        "loading_spinners": True
    },
    
    "responsiveness": {
        "mobile_breakpoint": 768,
        "tablet_breakpoint": 1024,
        "desktop_breakpoint": 1200
    }
}

# ================================================================
# DASHBOARD KPIs CONFIGURATION
# ================================================================

DASHBOARD_CONFIG = {
    "kpis": [
        {
            "name": "Total Employees",
            "key": "total_employees",
            "format": "{:,}",
            "color": "secondary",
            "icon": "👥",
            "description": "Total active employees in the system"
        },
        {
            "name": "Attrition Rate",
            "key": "attrition_rate", 
            "format": "{:.1%}",
            "color": "warning",
            "icon": "📈",
            "description": "Current employee attrition percentage",
            "benchmark": 0.15  # 15% benchmark
        },
        {
            "name": "High Risk Employees",
            "key": "high_risk_count",
            "format": "{:,}",
            "color": "error",
            "icon": "⚠️",
            "description": "Employees with high attrition risk"
        },
        {
            "name": "Average Tenure",
            "key": "avg_tenure",
            "format": "{:.1f} years",
            "color": "success",
            "icon": "🕐",
            "description": "Average years of service"
        },
        {
            "name": "Retention Cost Saved",
            "key": "cost_saved",
            "format": "${:,.0f}",
            "color": "success",
            "icon": "💰",
            "description": "Estimated savings from retention efforts"
        },
        {
            "name": "Model Accuracy",
            "key": "model_accuracy",
            "format": "{:.1%}",
            "color": "info",
            "icon": "🎯",
            "description": "Current ML model performance"
        }
    ],
    
    "charts": {
        "attrition_by_department": {
            "type": "bar",
            "title": "Attrition Rate by Department",
            "height": 400
        },
        "risk_distribution": {
            "type": "pie", 
            "title": "Employee Risk Distribution",
            "height": 400
        },
        "tenure_analysis": {
            "type": "histogram",
            "title": "Employee Tenure Distribution", 
            "height": 400
        },
        "salary_vs_risk": {
            "type": "scatter",
            "title": "Salary vs Attrition Risk",
            "height": 400
        }
    }
}

# ================================================================
# EMAIL AND NOTIFICATIONS
# ================================================================

EMAIL_CONFIG = {
    "templates": {
        "high_risk_alert": {
            "subject": "🚨 High Attrition Risk Alert - {employee_name}",
            "priority": "high",
            "auto_send": True
        },
        "monthly_report": {
            "subject": "📊 Monthly HR Analytics Report - {month} {year}",
            "priority": "medium",
            "auto_send": False
        },
        "retention_recommendation": {
            "subject": "💡 Employee Retention Recommendations",
            "priority": "medium", 
            "auto_send": False
        }
    },
    
    "notification_settings": {
        "enable_notifications": True,
        "high_risk_threshold": 0.8,
        "batch_size": 50,
        "frequency": "daily"
    }
}

# ================================================================
# EXPORT AND REPORTING
# ================================================================

EXPORT_CONFIG = {
    "formats": ["CSV", "Excel", "PDF"],
    "default_format": "Excel",
    
    "report_types": {
        "executive_summary": {
            "name": "Executive Summary Report",
            "description": "High-level KPIs and insights",
            "template": "executive_template.html"
        },
        "detailed_analysis": {
            "name": "Detailed Analytics Report", 
            "description": "Comprehensive data analysis",
            "template": "detailed_template.html"
        },
        "employee_list": {
            "name": "Employee Risk Assessment",
            "description": "Individual employee risk scores",
            "template": "employee_template.html"
        },
        "model_performance": {
            "name": "Model Performance Report",
            "description": "ML model metrics and evaluation",
            "template": "model_template.html"
        }
    },
    
    "file_naming": {
        "prefix": "HR_Attrition_",
        "include_timestamp": True,
        "timestamp_format": "%Y%m%d_%H%M%S"
    }
}

# ================================================================
# SECURITY AND PRIVACY
# ================================================================

SECURITY_CONFIG = {
    "data_privacy": {
        "mask_sensitive_data": True,
        "sensitive_fields": ["Email", "Phone", "Address", "EmergencyContact"],
        "access_logging": True
    },
    
    "user_roles": {
        "admin": {
            "permissions": ["view_all", "export_data", "manage_models", "system_admin"],
            "description": "Full system access"
        },
        "hr_manager": {
            "permissions": ["view_all", "export_data", "view_individual"],
            "description": "HR management access"
        },
        "analyst": {
            "permissions": ["view_analytics", "view_predictions"],
            "description": "Analytics and prediction access"
        },
        "viewer": {
            "permissions": ["view_dashboard"],
            "description": "Dashboard view only"
        }
    },
    
    "session_config": {
        "timeout_minutes": 120,
        "max_concurrent_sessions": 10,
        "require_authentication": False  # Set to True in production
    }
}

# ================================================================
# PERFORMANCE AND CACHING
# ================================================================

PERFORMANCE_CONFIG = {
    "caching": {
        "enable_caching": True,
        "cache_timeout": 3600,  # 1 hour
        "max_cache_size": "100MB"
    },
    
    "data_loading": {
        "chunk_size": 10000,
        "lazy_loading": True,
        "compression": True
    },
    
    "visualization": {
        "max_data_points": 5000,
        "sampling_method": "random",
        "render_timeout": 30
    }
}

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def configure_streamlit_page():
    """Configure the Streamlit page with optimal settings."""
    st.set_page_config(**PAGE_CONFIG)

def get_navigation_info(page_key: str) -> Dict[str, Any]:
    """
    Get navigation information for a specific page.
    
    Args:
        page_key: Key of the navigation page
        
    Returns:
        Dictionary containing page information
    """
    return NAVIGATION_PAGES.get(page_key, {})

def get_model_path(model_name: str) -> str:
    """
    Get the file path for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        File path to the model
    """
    return MODEL_CONFIG["model_paths"].get(model_name, "")

def get_risk_level(probability: float) -> str:
    """
    Determine risk level based on attrition probability.
    
    Args:
        probability: Attrition probability (0-1)
        
    Returns:
        Risk level string
    """
    thresholds = MODEL_CONFIG["model_info"]["risk_thresholds"]
    
    if probability <= thresholds["low"]:
        return "Low"
    elif probability <= thresholds["medium"]:
        return "Medium"
    else:
        return "High"

def get_risk_color(risk_level: str) -> str:
    """
    Get color for risk level display.
    
    Args:
        risk_level: Risk level string
        
    Returns:
        Hex color code
    """
    colors = {
        "Low": COLORS['success'],
        "Medium": COLORS['warning'], 
        "High": COLORS['error']
    }
    return colors.get(risk_level, COLORS['text'])

def format_kpi_value(value: Any, kpi_config: Dict[str, Any]) -> str:
    """
    Format KPI value according to its configuration.
    
    Args:
        value: Raw KPI value
        kpi_config: KPI configuration dictionary
        
    Returns:
        Formatted string value
    """
    format_string = kpi_config.get("format", "{}")
    try:
        return format_string.format(value)
    except (ValueError, TypeError):
        return str(value)

def get_user_permissions(user_role: str = "viewer") -> List[str]:
    """
    Get permissions for a user role.
    
    Args:
        user_role: User role name
        
    Returns:
        List of permissions
    """
    return SECURITY_CONFIG["user_roles"].get(user_role, {}).get("permissions", ["view_dashboard"])

def is_mobile_device() -> bool:
    """
    Check if the user is on a mobile device.
    
    Returns:
        Boolean indicating if mobile
    """
    # This would need additional implementation with JavaScript
    # For now, return False as default
    return False

def get_export_filename(report_type: str) -> str:
    """
    Generate filename for export based on type and configuration.
    
    Args:
        report_type: Type of report being exported
        
    Returns:
        Generated filename
    """
    from datetime import datetime
    
    config = EXPORT_CONFIG
    prefix = config["file_naming"]["prefix"]
    
    if config["file_naming"]["include_timestamp"]:
        timestamp = datetime.now().strftime(config["file_naming"]["timestamp_format"])
        return f"{prefix}{report_type}_{timestamp}"
    else:
        return f"{prefix}{report_type}"

def validate_model_files() -> Dict[str, bool]:
    """
    Validate that all required model files exist.
    
    Returns:
        Dictionary mapping model names to existence status
    """
    model_status = {}
    
    # Check model files
    for model_name, path in MODEL_CONFIG["model_paths"].items():
        model_status[f"model_{model_name}"] = os.path.exists(path)
    
    # Check preprocessing files
    for prep_name, path in MODEL_CONFIG["preprocessing_paths"].items():
        model_status[f"preprocessing_{prep_name}"] = os.path.exists(path)
    
    return model_status

def get_app_info() -> Dict[str, Any]:
    """
    Get comprehensive application information.
    
    Returns:
        Dictionary containing app metadata
    """
    return {
        "name": "HR Attrition Predictor",
        "version": "1.0.0",
        "description": "AI-powered employee attrition prediction system",
        "authors": ["HR Analytics Team"],
        "created_date": "September 2025",
        "page_count": len(NAVIGATION_PAGES),
        "model_count": len(MODEL_CONFIG["model_paths"]),
        "theme": "Dark Cyberpunk",
        "framework": "Streamlit",
        "ml_backend": "scikit-learn, XGBoost"
    }

# ================================================================
# SESSION STATE MANAGEMENT
# ================================================================

def initialize_session_state():
    """Initialize Streamlit session state with default values."""
    default_values = {
        "current_page": "🏠 Dashboard",
        "user_role": "viewer",
        "selected_employee": None,
        "filter_department": "All Departments",
        "filter_risk_level": "All",
        "model_loaded": False,
        "data_loaded": False,
        "last_prediction": None,
        "notification_count": 0,
        "session_start": datetime.now()
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ================================================================
# CONSTANTS AND ENUMS
# ================================================================

class RiskLevel:
    """Risk level constants."""
    LOW = "Low"
    MEDIUM = "Medium" 
    HIGH = "High"

class UserRole:
    """User role constants."""
    ADMIN = "admin"
    HR_MANAGER = "hr_manager"
    ANALYST = "analyst"
    VIEWER = "viewer"

class ModelType:
    """Model type constants."""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"
    BEST = "best_model"

# ================================================================
# EXPORT ALL CONFIGURATIONS
# ================================================================

__all__ = [
    'PAGE_CONFIG',
    'NAVIGATION_PAGES',
    'MODEL_CONFIG', 
    'DATA_CONFIG',
    'UI_CONFIG',
    'DASHBOARD_CONFIG',
    'EMAIL_CONFIG',
    'EXPORT_CONFIG',
    'SECURITY_CONFIG',
    'PERFORMANCE_CONFIG',
    'configure_streamlit_page',
    'get_navigation_info',
    'get_model_path',
    'get_risk_level',
    'get_risk_color',
    'format_kpi_value',
    'get_user_permissions',
    'get_export_filename',
    'validate_model_files',
    'get_app_info',
    'initialize_session_state',
    'RiskLevel',
    'UserRole', 
    'ModelType'
]

# ================================================================
# DEVELOPMENT PREVIEW
# ================================================================

def preview_config():
    """Preview the configuration (for development use)."""
    print("⚙️ HR Attrition Predictor - Streamlit Configuration")
    print("=" * 60)
    print(f"📱 App: {PAGE_CONFIG['page_title']}")
    print(f"🏗️  Layout: {PAGE_CONFIG['layout']}")
    print(f"📄 Pages: {len(NAVIGATION_PAGES)}")
    print(f"🤖 Models: {len(MODEL_CONFIG['model_paths'])}")
    print(f"📊 KPIs: {len(DASHBOARD_CONFIG['kpis'])}")
    print(f"🎨 Theme: Dark Cyberpunk")
    print(f"🔐 Security: {'Enabled' if SECURITY_CONFIG['data_privacy']['mask_sensitive_data'] else 'Disabled'}")
    print("\n✅ Configuration loaded successfully!")

if __name__ == "__main__":
    preview_config()
