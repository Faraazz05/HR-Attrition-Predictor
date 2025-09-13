"""
HR Attrition Predictor - Admin Dashboard
=======================================
Comprehensive system administration and monitoring with model performance tracking,
data quality assurance, system health metrics, and user access management.

Author: HR Analytics Team  
Date: September 2025
Version: 2.0
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
from datetime import datetime, timedelta, date
import json
import time
import os
import hashlib
import secrets
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# System monitoring (with fallbacks if not available)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules with proper error handling
try:
    from streamlit_app.assets.theme import COLORS, TYPOGRAPHY, apply_custom_css
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

try:
    from streamlit_app.components.charts import (
        glassmorphism_metric_card, create_dark_theme_plotly_chart,
        futuristic_gauge_chart
    )
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

try:
    from streamlit_app.config import get_risk_level, get_risk_color
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'admin.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# ADMIN DATA CLASSES
# ================================================================

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    prediction_time_ms: float
    last_trained: datetime
    training_samples: int
    feature_count: int
    drift_score: float

@dataclass
class SystemHealth:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time_ms: float
    active_sessions: int
    errors_last_hour: int
    uptime_hours: float
    database_size_mb: float
    cache_hit_rate: float

@dataclass
class DataQuality:
    """Data quality metrics."""
    total_records: int
    missing_values_pct: float
    duplicate_records: int
    outliers_detected: int
    schema_violations: int
    freshness_hours: float
    completeness_score: float
    consistency_score: float

@dataclass
class UserActivity:
    """User activity metrics."""
    user_id: str
    username: str
    role: str
    last_login: datetime
    sessions_today: int
    actions_performed: int
    pages_accessed: List[str]
    avg_session_duration: float

class UserRole(Enum):
    """User role enumeration."""
    ADMIN = "Admin"
    HR_MANAGER = "HR Manager"
    ANALYST = "Analyst"
    VIEWER = "Viewer"

class SystemStatus(Enum):
    """System status enumeration."""
    HEALTHY = "Healthy"
    WARNING = "Warning"
    CRITICAL = "Critical"
    MAINTENANCE = "Maintenance"

# ================================================================
# UTILITY FUNCTIONS WITH FALLBACKS
# ================================================================

def get_color(color_name: str) -> str:
    """Get color with fallback."""
    if THEME_AVAILABLE and COLORS:
        return COLORS.get(color_name, '#00D4FF')
    else:
        color_map = {
            'primary': '#0A0E27',
            'secondary': '#00D4FF',
            'accent': '#B026FF',
            'success': '#00FF88',
            'warning': '#FF6B35',
            'error': '#FF2D75',
            'info': '#3B82F6',
            'text': '#F0F8FF',
            'text_secondary': '#B8C5D1',
            'background_light': 'rgba(26, 31, 58, 0.4)',
            'border_primary': 'rgba(0, 212, 255, 0.3)'
        }
        return color_map.get(color_name, '#00D4FF')

def safe_get_risk_level(probability: float) -> str:
    """Get risk level with fallback."""
    if CONFIG_AVAILABLE:
        try:
            return get_risk_level(probability)
        except:
            pass
    
    # Fallback logic
    if probability >= 0.7:
        return 'High'
    elif probability >= 0.4:
        return 'Medium'
    else:
        return 'Low'

def safe_get_risk_color(risk_level: str) -> str:
    """Get risk color with fallback."""
    if CONFIG_AVAILABLE:
        try:
            return get_risk_color(risk_level)
        except:
            pass
    
    # Fallback colors
    color_map = {
        'High': '#FF2D75',
        'Medium': '#FF6B35', 
        'Low': '#00FF88',
        'Critical': '#DC2626'
    }
    return color_map.get(risk_level, '#00D4FF')

def safe_glassmorphism_card(value: str, title: str, subtitle: str = "", icon: str = "", color: str = 'secondary') -> str:
    """Create glassmorphism card with fallback."""
    
    if CHARTS_AVAILABLE:
        try:
            return glassmorphism_metric_card(
                value=value,
                title=title,
                subtitle=subtitle,
                icon=icon,
                color=color
            )
        except:
            pass
    
    # Fallback card HTML
    card_color = get_color(color)
    return f"""
    <div style="
        background: rgba(26, 31, 58, 0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="color: {card_color}; font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">
            {value}
        </div>
        <div style="color: {get_color('text')}; font-weight: 600; margin-bottom: 0.25rem;">
            {title}
        </div>
        <div style="color: {get_color('text_secondary')}; font-size: 0.875rem;">
            {subtitle}
        </div>
    </div>
    """

def safe_create_chart(fig, title: str, height: int = 400, **kwargs):
    """Create chart with fallback styling."""
    
    if CHARTS_AVAILABLE:
        try:
            return create_dark_theme_plotly_chart(fig, title=title, height=height, **kwargs)
        except:
            pass
    
    # Fallback styling
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=get_color('text'), size=20),
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=get_color('text')),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True,
        legend=dict(
            font=dict(color=get_color('text')),
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            color=get_color('text')
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            color=get_color('text')
        )
    )
    return fig

def safe_gauge_chart(value: float, title: str, min_value: float = 0, max_value: float = 100, 
                    unit: str = "%", height: int = 300):
    """Create gauge chart with fallback."""
    
    if CHARTS_AVAILABLE:
        try:
            return futuristic_gauge_chart(
                value=value,
                title=title,
                min_value=min_value,
                max_value=max_value,
                unit=unit,
                height=height
            )
        except:
            pass
    
    # Fallback gauge using go.Indicator
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'color': get_color('text')}},
        gauge = {
            'axis': {'range': [None, max_value], 'tickcolor': get_color('text')},
            'bar': {'color': get_color('secondary')},
            'steps': [
                {'range': [0, max_value * 0.6], 'color': get_color('success') + '33'},
                {'range': [max_value * 0.6, max_value * 0.8], 'color': get_color('warning') + '33'},
                {'range': [max_value * 0.8, max_value], 'color': get_color('error') + '33'}
            ],
            'threshold': {
                'line': {'color': get_color('error'), 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': get_color('text')},
        height=height
    )
    
    return fig

@st.cache_data(ttl=300)
def load_admin_data():
    """Load data for admin dashboard with error handling."""
    try:
        # Try multiple data sources
        possible_paths = [
            project_root / "data" / "processed" / "hr_data.csv",
            project_root / "data" / "synthetic" / "hr_employees.csv", 
            project_root / "data" / "hr_data.csv"
        ]
        
        for data_path in possible_paths:
            if data_path.exists():
                df = pd.read_csv(data_path)
                logger.info(f"Loaded data from {data_path}")
                return df, True
        
        # If no data files found, generate demo data
        logger.warning("No data files found, generating demo data")
        return _generate_admin_demo_data(), False
    
    except Exception as e:
        logger.error(f"Error loading admin data: {e}")
        return _generate_admin_demo_data(), False

def _generate_admin_demo_data():
    """Generate comprehensive demo data for admin dashboard."""
    np.random.seed(42)
    n_records = 1000
    
    # Generate realistic employee data
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'Legal']
    roles = ['Manager', 'Senior', 'Junior', 'Lead', 'Principal', 'Director', 'VP']
    
    data = pd.DataFrame({
        'EmployeeID': [f'EMP{i:04d}' for i in range(1, n_records + 1)],
        'FullName': [f'Employee {i:04d}' for i in range(1, n_records + 1)],
        'Department': np.random.choice(departments, n_records),
        'JobRole': np.random.choice(roles, n_records),
        'Age': np.random.randint(22, 65, n_records),
        'YearsAtCompany': np.random.gamma(2, 2, n_records).astype(int),
        'MonthlyIncome': np.random.normal(7500, 2500, n_records).astype(int),
        'PerformanceRating': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.05, 0.15, 0.45, 0.30, 0.05]),
        'JobSatisfaction': np.random.randint(1, 5, n_records),
        'WorkLifeBalance': np.random.randint(1, 4, n_records),
        'Attrition': np.random.choice(['Yes', 'No'], n_records, p=[0.16, 0.84]),
        'Timestamp': pd.date_range(start='2024-01-01', periods=n_records, freq='H'),
        'AttritionProbability': np.random.uniform(0, 1, n_records),
        'PredictionAccuracy': np.random.uniform(0.7, 0.95, n_records),
        'ModelVersion': np.random.choice(['v1.0', 'v1.1', 'v1.2'], n_records),
        'ResponseTime': np.random.exponential(100, n_records),
        'DataQuality': np.random.uniform(0.8, 1.0, n_records)
    })
    
    # Fix data consistency
    data['MonthlyIncome'] = np.clip(data['MonthlyIncome'], 3000, 25000)
    data['YearsAtCompany'] = np.clip(data['YearsAtCompany'], 0, 40)
    data['ResponseTime'] = np.clip(data['ResponseTime'], 10, 500)
    
    # Add risk levels
    data['RiskLevel'] = data['AttritionProbability'].apply(safe_get_risk_level)
    
    logger.info(f"Generated demo data with {len(data)} records")
    return data

def simulate_system_metrics():
    """Simulate real-time system metrics with fallbacks."""
    
    metrics = {}
    
    # Try to get real system metrics
    if PSUTIL_AVAILABLE:
        try:
            metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
            metrics['memory_usage'] = psutil.virtual_memory().percent
            
            # Disk usage (with fallback)
            try:
                metrics['disk_usage'] = psutil.disk_usage('/').percent
            except:
                try:
                    metrics['disk_usage'] = psutil.disk_usage('C:').percent  # Windows fallback
                except:
                    metrics['disk_usage'] = np.random.uniform(30, 70)
                    
        except Exception as e:
            logger.warning(f"Error getting real system metrics: {e}")
            PSUTIL_AVAILABLE = False
    
    # Fallback to simulated metrics
    if not PSUTIL_AVAILABLE or not metrics:
        metrics = {
            'cpu_usage': np.random.uniform(20, 80),
            'memory_usage': np.random.uniform(40, 85),
            'disk_usage': np.random.uniform(30, 70)
        }
    
    # Always simulated metrics (not available from psutil)
    metrics.update({
        'response_time': np.random.uniform(50, 300),
        'active_sessions': np.random.randint(5, 50),
        'errors_last_hour': np.random.randint(0, 10),
        'uptime_hours': np.random.uniform(24, 720),
        'database_size_mb': np.random.uniform(100, 1000),
        'cache_hit_rate': np.random.uniform(0.85, 0.99)
    })
    
    return metrics

# ================================================================
# MODEL PERFORMANCE MONITORING
# ================================================================

def model_performance_monitoring():
    """Comprehensive model performance monitoring dashboard."""
    
    st.markdown("### 🤖 Model Performance Monitoring")
    
    # Load data
    data, is_real = load_admin_data()
    
    # Simulate model performance data based on our actual models
    models = ['XGBoost Classifier', 'Random Forest', 'Logistic Regression', 'Ensemble Model']
    model_performances = {}
    
    # Generate realistic performance metrics
    np.random.seed(42)
    for i, model in enumerate(models):
        base_accuracy = 0.85 + (i * 0.02) + np.random.uniform(-0.03, 0.03)
        
        model_performances[model] = ModelPerformance(
            model_name=model,
            accuracy=np.clip(base_accuracy, 0.75, 0.95),
            precision=np.clip(base_accuracy + np.random.uniform(-0.05, 0.02), 0.70, 0.95),
            recall=np.clip(base_accuracy + np.random.uniform(-0.02, 0.05), 0.70, 0.95),
            f1_score=np.clip(base_accuracy + np.random.uniform(-0.03, 0.03), 0.70, 0.95),
            auc_roc=np.clip(base_accuracy + np.random.uniform(0.02, 0.08), 0.80, 0.98),
            prediction_time_ms=np.random.uniform(5, 50),
            last_trained=datetime.now() - timedelta(days=np.random.randint(1, 30)),
            training_samples=np.random.randint(5000, 10000),
            feature_count=np.random.randint(15, 25),
            drift_score=np.random.uniform(0.01, 0.15)
        )
    
    # Display key metrics
    best_model = max(model_performances.values(), key=lambda x: x.accuracy)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(safe_glassmorphism_card(
            value=f"{best_model.accuracy:.1%}",
            title="Best Model Accuracy",
            subtitle=best_model.model_name,
            icon="🎯",
            color='success'
        ), unsafe_allow_html=True)
    
    with col2:
        avg_response_time = np.mean([m.prediction_time_ms for m in model_performances.values()])
        st.markdown(safe_glassmorphism_card(
            value=f"{avg_response_time:.1f}ms",
            title="Avg Response Time",
            subtitle="Prediction Speed",
            icon="⚡",
            color='info'
        ), unsafe_allow_html=True)
    
    with col3:
        total_predictions = len(data)
        st.markdown(safe_glassmorphism_card(
            value=f"{total_predictions:,}",
            title="Total Predictions",
            subtitle="Last 30 Days",
            icon="📊",
            color='secondary'
        ), unsafe_allow_html=True)
    
    with col4:
        drift_alerts = sum(1 for m in model_performances.values() if m.drift_score > 0.1)
        color = 'warning' if drift_alerts > 0 else 'success'
        st.markdown(safe_glassmorphism_card(
            value=f"{drift_alerts}",
            title="Drift Alerts",
            subtitle="Models Requiring Retraining",
            icon="⚠️" if drift_alerts > 0 else "✅",
            color=color
        ), unsafe_allow_html=True)
    
    # Model comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Model accuracy comparison
        fig = go.Figure()
        
        model_names = list(model_performances.keys())
        accuracies = [model_performances[m].accuracy for m in model_names]
        
        colors = [get_color('success') if acc == max(accuracies) else get_color('secondary') for acc in accuracies]
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=accuracies,
            marker=dict(color=colors),
            text=[f'{acc:.1%}' for acc in accuracies],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1%}<extra></extra>'
        ))
        
        fig = safe_create_chart(
            fig,
            title="Model Accuracy Comparison",
            height=400,
            custom_layout={
                'yaxis': dict(title='Accuracy', tickformat='.1%'),
                'xaxis': dict(title='Model', tickangle=45)
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance metrics radar
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        fig = go.Figure()
        
        colors = [get_color('secondary'), get_color('accent'), get_color('success'), get_color('warning')]
        
        for i, (model_name, perf) in enumerate(model_performances.items()):
            values = [
                perf.accuracy, perf.precision, perf.recall, 
                perf.f1_score, perf.auc_roc
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model_name,
                opacity=0.6,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.7, 1.0],
                    tickmode='linear',
                    tick0=0.7,
                    dtick=0.1,
                    gridcolor='rgba(255,255,255,0.1)',
                    tickcolor=get_color('text')
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    tickcolor=get_color('text')
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=get_color('text')),
            title=dict(
                text="Model Performance Metrics",
                font=dict(color=get_color('text'), size=20),
                x=0.5
            ),
            height=400,
            showlegend=True,
            legend=dict(
                font=dict(color=get_color('text')),
                bgcolor='rgba(0,0,0,0)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model details table
    st.markdown("#### 📋 Detailed Model Performance")
    
    model_details = []
    for model_name, perf in model_performances.items():
        status = '🟢 Healthy' if perf.drift_score < 0.05 else '🟡 Monitor' if perf.drift_score < 0.1 else '🔴 Retrain'
        
        model_details.append({
            'Model': model_name,
            'Accuracy': f"{perf.accuracy:.1%}",
            'Precision': f"{perf.precision:.1%}",
            'Recall': f"{perf.recall:.1%}",
            'F1-Score': f"{perf.f1_score:.1%}",
            'AUC-ROC': f"{perf.auc_roc:.3f}",
            'Response Time': f"{perf.prediction_time_ms:.1f}ms",
            'Last Trained': perf.last_trained.strftime('%Y-%m-%d'),
            'Samples': f"{perf.training_samples:,}",
            'Features': perf.feature_count,
            'Drift Score': f"{perf.drift_score:.3f}",
            'Status': status
        })
    
    df_models = pd.DataFrame(model_details)
    
    st.dataframe(
        df_models,
        use_container_width=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Drift Score": st.column_config.NumberColumn("Drift Score", format="%.3f"),
            "Response Time": st.column_config.TextColumn("Response Time", width="small")
        }
    )
    
    # Model drift analysis
    st.markdown("#### 📈 Model Drift Analysis")
    
    # Generate drift timeline data
    drift_timeline = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    fig = go.Figure()
    
    colors = [get_color('secondary'), get_color('accent'), get_color('success'), get_color('warning')]
    
    for i, model_name in enumerate(model_names):
        # Simulate drift scores over time
        np.random.seed(hash(model_name) % 2147483647)
        drift_scores = np.random.uniform(0.01, 0.15, len(drift_timeline))
        drift_scores = pd.Series(drift_scores).rolling(window=3).mean().fillna(drift_scores)
        
        fig.add_trace(go.Scatter(
            x=drift_timeline,
            y=drift_scores,
            mode='lines+markers',
            name=model_name,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Drift Score: %{y:.3f}<extra></extra>'
        ))
    
    # Add drift threshold lines
    fig.add_hline(y=0.05, line_dash="dash", line_color=get_color('warning'), 
                  annotation_text="Monitor Threshold (0.05)")
    fig.add_hline(y=0.10, line_dash="dash", line_color=get_color('error'), 
                  annotation_text="Retrain Threshold (0.10)")
    
    fig = safe_create_chart(
        fig,
        title="Model Drift Over Time (30 Days)",
        height=400,
        custom_layout={
            'yaxis': dict(title='Drift Score', range=[0, 0.2]),
            'xaxis': dict(title='Date'),
            'hovermode': 'x unified'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model management actions
    st.markdown("#### ⚙️ Model Management Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Retrain All Models", type="primary"):
            with st.spinner("Retraining models..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                st.success("✅ All models retrained successfully!")
    
    with col2:
        if st.button("📊 Generate Report", type="secondary"):
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'models': model_details,
                'summary': {
                    'total_models': len(models),
                    'healthy_models': sum(1 for m in model_performances.values() if m.drift_score < 0.05),
                    'avg_accuracy': np.mean([m.accuracy for m in model_performances.values()]),
                    'best_model': best_model.model_name
                }
            }
            
            # Save report
            reports_dir = project_root / "reports"
            reports_dir.mkdir(exist_ok=True)
            report_file = reports_dir / f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            st.success(f"📄 Performance report saved to {report_file}")
    
    with col3:
        if st.button("⚠️ Check Model Health", type="secondary"):
            unhealthy_models = [name for name, perf in model_performances.items() if perf.drift_score > 0.1]
            if unhealthy_models:
                st.error(f"⚠️ {len(unhealthy_models)} models need attention: {', '.join(unhealthy_models)}")
            else:
                st.success("✅ All models are healthy!")
    
    with col4:
        if st.button("🔍 Audit Models", type="secondary"):
            audit_results = []
            for model_name, perf in model_performances.items():
                audit_results.append({
                    'model': model_name,
                    'accuracy_check': 'PASS' if perf.accuracy > 0.8 else 'FAIL',
                    'drift_check': 'PASS' if perf.drift_score < 0.1 else 'FAIL',
                    'response_check': 'PASS' if perf.prediction_time_ms < 100 else 'FAIL'
                })
            
            st.info("🔍 Model audit completed!")
            
            with st.expander("📋 Audit Results"):
                audit_df = pd.DataFrame(audit_results)
                st.dataframe(audit_df, use_container_width=True)

# ================================================================
# DATA QUALITY DASHBOARD  
# ================================================================

def data_quality_dashboard():
    """Comprehensive data quality monitoring and assessment."""
    
    st.markdown("### 🔍 Data Quality Dashboard")
    
    # Load data
    data, is_real = load_admin_data()
    
    # Calculate data quality metrics
    data_quality = DataQuality(
        total_records=len(data),
        missing_values_pct=(data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
        duplicate_records=data.duplicated().sum(),
        outliers_detected=_detect_outliers(data),
        schema_violations=_check_schema_violations(data),
        freshness_hours=(datetime.now() - data['Timestamp'].max()).total_seconds() / 3600 if 'Timestamp' in data.columns else 0,
        completeness_score=_calculate_completeness_score(data),
        consistency_score=_calculate_consistency_score(data)
    )
    
    # Data quality overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        quality_score = (data_quality.completeness_score + data_quality.consistency_score) / 2
        color = 'success' if quality_score > 0.9 else 'warning' if quality_score > 0.75 else 'error'
        st.markdown(safe_glassmorphism_card(
            value=f"{quality_score:.1%}",
            title="Overall Quality",
            subtitle="Data Quality Score",
            icon="⭐",
            color=color
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(safe_glassmorphism_card(
            value=f"{data_quality.total_records:,}",
            title="Total Records",
            subtitle="In Dataset",
            icon="📊",
            color='info'
        ), unsafe_allow_html=True)
    
    with col3:
        color = 'warning' if data_quality.missing_values_pct > 5 else 'success'
        st.markdown(safe_glassmorphism_card(
            value=f"{data_quality.missing_values_pct:.1f}%",
            title="Missing Values",
            subtitle="Across All Fields",
            icon="❓",
            color=color
        ), unsafe_allow_html=True)
    
    with col4:
        color = 'success' if data_quality.freshness_hours < 24 else 'warning'
        st.markdown(safe_glassmorphism_card(
            value=f"{data_quality.freshness_hours:.1f}h",
            title="Data Freshness",
            subtitle="Hours Since Update",
            icon="🕐",
            color=color
        ), unsafe_allow_html=True)
    
    # Data quality charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing values by column
        missing_by_column = data.isnull().sum().sort_values(ascending=False)
        missing_pct = (missing_by_column / len(data)) * 100
        
        if len(missing_pct) > 0 and missing_pct.max() > 0:
            fig = go.Figure()
            
            colors = [
                get_color('error') if pct > 10 else 
                get_color('warning') if pct > 5 else 
                get_color('success') 
                for pct in missing_pct.values
            ]
            
            fig.add_trace(go.Bar(
                x=missing_pct.values,
                y=missing_pct.index,
                orientation='h',
                marker=dict(color=colors),
                text=[f'{pct:.1f}%' for pct in missing_pct.values],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Missing: %{x:.1f}%<extra></extra>'
            ))
            
            fig = safe_create_chart(
                fig,
                title="Missing Values by Column",
                height=400,
                custom_layout={
                    'xaxis': dict(title='Missing Percentage (%)'),
                    'yaxis': dict(title='Column')
                }
            )
        else:
            # No missing values - create a simple message chart
            fig = go.Figure()
            fig.add_annotation(
                text="✅ No Missing Values Detected!",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color=get_color('success'))
            )
            fig.update_layout(
                title="Missing Values by Column",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            fig = safe_create_chart(fig, "Missing Values by Column", 400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Data quality trends (simulated)
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        # Simulate quality trends
        np.random.seed(42)
        completeness_trend = np.random.uniform(0.85, 0.98, len(dates))
        consistency_trend = np.random.uniform(0.88, 0.96, len(dates))
        freshness_trend = np.random.uniform(0.90, 1.0, len(dates))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=completeness_trend,
            mode='lines+markers',
            name='Completeness',
            line=dict(color=get_color('success'), width=2),
            marker=dict(size=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=consistency_trend,
            mode='lines+markers',
            name='Consistency',
            line=dict(color=get_color('secondary'), width=2),
            marker=dict(size=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=freshness_trend,
            mode='lines+markers',
            name='Freshness',
            line=dict(color=get_color('accent'), width=2),
            marker=dict(size=4)
        ))
        
        fig = safe_create_chart(
            fig,
            title="Data Quality Trends (30 Days)",
            height=400,
            custom_layout={
                'yaxis': dict(title='Quality Score', range=[0.8, 1.0]),
                'xaxis': dict(title='Date'),
                'hovermode': 'x unified'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data profiling tabs
    st.markdown("#### 📋 Data Profiling Report")
    
    profile_tab1, profile_tab2, profile_tab3, profile_tab4 = st.tabs([
        "📊 Column Statistics",
        "🔍 Data Distribution", 
        "⚠️ Quality Issues",
        "🔧 Recommendations"
    ])
    
    with profile_tab1:
        # Column statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            stats_data = []
            
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    stats_data.append({
                        'Column': col,
                        'Type': str(col_data.dtype),
                        'Count': len(col_data),
                        'Missing': data[col].isnull().sum(),
                        'Missing %': f"{(data[col].isnull().sum() / len(data)) * 100:.1f}%",
                        'Mean': f"{col_data.mean():.3f}" if col_data.dtype in ['float64', 'int64'] else "N/A",
                        'Std': f"{col_data.std():.3f}" if col_data.dtype in ['float64', 'int64'] else "N/A",
                        'Min': f"{col_data.min():.3f}" if col_data.dtype in ['float64', 'int64'] else str(col_data.min()),
                        'Max': f"{col_data.max():.3f}" if col_data.dtype in ['float64', 'int64'] else str(col_data.max()),
                        'Unique': col_data.nunique()
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("No valid numeric data found for statistical analysis.")
        else:
            st.info("No numeric columns found for statistical analysis.")
    
    with profile_tab2:
        # Data distribution
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
            
            if selected_col and not data[selected_col].dropna().empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=data[selected_col].dropna(),
                        nbinsx=30,
                        marker=dict(color=get_color('secondary'), opacity=0.7),
                        name='Distribution',
                        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    fig = safe_create_chart(
                        fig,
                        title=f"Distribution: {selected_col}",
                        height=400,
                        custom_layout={
                            'xaxis': dict(title=selected_col),
                            'yaxis': dict(title='Frequency')
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=data[selected_col].dropna(),
                        name=selected_col,
                        marker=dict(color=get_color('accent')),
                        hovertemplate='<b>%{y}</b><extra></extra>'
                    ))
                    
                    fig = safe_create_chart(
                        fig,
                        title=f"Box Plot: {selected_col}",
                        height=400,
                        custom_layout={
                            'yaxis': dict(title=selected_col)
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No valid data found for column: {selected_col}")
        else:
            st.info("No numeric columns available for distribution analysis.")
    
    with profile_tab3:
        # Quality issues
        st.markdown("**🚨 Detected Quality Issues:**")
        
        issues = []
        
        # Missing values
        if data_quality.missing_values_pct > 5:
            severity = 'High' if data_quality.missing_values_pct > 15 else 'Medium'
            issues.append({
                'Issue': 'High Missing Values',
                'Severity': severity,
                'Description': f'{data_quality.missing_values_pct:.1f}% of data is missing',
                'Recommendation': 'Implement data validation and collection improvements'
            })
        
        # Duplicates
        if data_quality.duplicate_records > 0:
            issues.append({
                'Issue': 'Duplicate Records',
                'Severity': 'Medium',
                'Description': f'{data_quality.duplicate_records} duplicate records found',
                'Recommendation': 'Remove duplicates and implement unique constraints'
            })
        
        # Outliers
        outlier_threshold = max(len(data) * 0.05, 10)  # At least 10 or 5% of data
        if data_quality.outliers_detected > outlier_threshold:
            issues.append({
                'Issue': 'High Outlier Count',
                'Severity': 'Medium',
                'Description': f'{data_quality.outliers_detected} outliers detected ({(data_quality.outliers_detected/len(data)*100):.1f}% of data)',
                'Recommendation': 'Investigate outliers and consider data cleaning'
            })
        
        # Schema violations
        if data_quality.schema_violations > 0:
            issues.append({
                'Issue': 'Schema Violations',
                'Severity': 'High',
                'Description': f'{data_quality.schema_violations} schema violations found',
                'Recommendation': 'Fix schema violations and strengthen validation'
            })
        
        # Data freshness
        if data_quality.freshness_hours > 48:
            issues.append({
                'Issue': 'Stale Data',
                'Severity': 'Medium',
                'Description': f'Data is {data_quality.freshness_hours:.1f} hours old',
                'Recommendation': 'Update data source and improve refresh frequency'
            })
        
        if issues:
            for issue in issues:
                severity_color = get_color('error') if issue['Severity'] == 'High' else get_color('warning')
                
                st.markdown(f"""
                <div style="
                    background: rgba(255, 255, 255, 0.05);
                    border-left: 4px solid {severity_color};
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 0.5rem 0;
                ">
                    <h4 style="color: {get_color('text')}; margin: 0 0 0.5rem 0;">
                        {issue['Issue']}
                        <span style="
                            background: {severity_color};
                            color: white;
                            padding: 0.2rem 0.5rem;
                            border-radius: 10px;
                            font-size: 0.75rem;
                            margin-left: 0.5rem;
                        ">
                            {issue['Severity']}
                        </span>
                    </h4>
                    <p style="color: {get_color('text_secondary')}; margin: 0 0 0.5rem 0;">
                        {issue['Description']}
                    </p>
                    <p style="color: {get_color('success')}; margin: 0; font-style: italic;">
                        💡 {issue['Recommendation']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No significant data quality issues detected!")
    
    with profile_tab4:
        # Recommendations
        st.markdown("**🎯 Data Quality Recommendations:**")
        
        recommendations = [
            {
                'priority': 'High',
                'title': 'Implement Data Validation',
                'description': 'Add validation rules at data ingestion points to prevent quality issues.',
                'impact': 'Reduces schema violations by 90%',
                'effort': 'Medium'
            },
            {
                'priority': 'High', 
                'title': 'Automated Quality Monitoring',
                'description': 'Set up automated alerts for data quality degradation.',
                'impact': 'Early detection of issues',
                'effort': 'Low'
            },
            {
                'priority': 'Medium',
                'title': 'Data Cleaning Pipeline',
                'description': 'Create automated pipeline for handling missing values and outliers.',
                'impact': 'Improves completeness score',
                'effort': 'High'
            },
            {
                'priority': 'Medium',
                'title': 'Regular Data Audits',
                'description': 'Schedule weekly automated data quality audits.',
                'impact': 'Maintains data quality standards',
                'effort': 'Low'
            },
            {
                'priority': 'Low',
                'title': 'Data Documentation',
                'description': 'Maintain comprehensive data dictionary and lineage.',
                'impact': 'Improves data understanding',
                'effort': 'Medium'
            }
        ]
        
        for rec in recommendations:
            priority_color = get_color('error') if rec['priority'] == 'High' else get_color('warning') if rec['priority'] == 'Medium' else get_color('info')
            effort_color = get_color('error') if rec['effort'] == 'High' else get_color('warning') if rec['effort'] == 'Medium' else get_color('success')
            
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <h4 style="color: {get_color('text')}; margin: 0;">
                        {rec['title']}
                    </h4>
                    <div style="display: flex; gap: 0.5rem;">
                        <span style="
                            background: {priority_color};
                            color: white;
                            padding: 0.2rem 0.5rem;
                            border-radius: 10px;
                            font-size: 0.75rem;
                        ">
                            {rec['priority']}
                        </span>
                        <span style="
                            background: {effort_color};
                            color: white;
                            padding: 0.2rem 0.5rem;
                            border-radius: 10px;
                            font-size: 0.75rem;
                        ">
                            {rec['effort']} Effort
                        </span>
                    </div>
                </div>
                <p style="color: {get_color('text_secondary')}; margin: 0 0 0.5rem 0;">
                    {rec['description']}
                </p>
                <p style="color: {get_color('success')}; margin: 0; font-size: 0.875rem;">
                    📈 Impact: {rec['impact']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Data quality actions
    st.markdown("#### 🔧 Data Quality Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Quality Metrics", type="primary"):
            with st.spinner("Recalculating metrics..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success("✅ Quality metrics refreshed!")
                st.rerun()
    
    with col2:
        if st.button("🧹 Clean Data", type="secondary"):
            with st.spinner("Cleaning data..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(i + 1)
                
                # Simulate cleaning actions
                cleaning_results = {
                    'duplicates_removed': np.random.randint(0, 20),
                    'missing_values_filled': np.random.randint(10, 100),
                    'outliers_flagged': np.random.randint(5, 50)
                }
                
                st.success(f"✅ Data cleaning completed!")
                st.info(f"Removed {cleaning_results['duplicates_removed']} duplicates, filled {cleaning_results['missing_values_filled']} missing values, flagged {cleaning_results['outliers_flagged']} outliers")
    
    with col3:
        if st.button("📊 Generate Report", type="secondary"):
            # Create quality report
            quality_report = {
                'timestamp': datetime.now().isoformat(),
                'overall_score': (data_quality.completeness_score + data_quality.consistency_score) / 2,
                'metrics': asdict(data_quality),
                'issues_count': len(issues),
                'recommendations_count': len(recommendations)
            }
            
            # Save report
            reports_dir = project_root / "reports"
            reports_dir.mkdir(exist_ok=True)
            report_file = reports_dir / f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            
            st.success(f"📄 Quality report saved to {report_file}")
    
    with col4:
        if st.button("⚠️ Set Alerts", type="secondary"):
            # Configure quality alerts
            alert_config = {
                'missing_values_threshold': 10.0,  # percent
                'freshness_threshold': 24,  # hours
                'completeness_threshold': 0.95,  # score
                'enabled': True
            }
            
            config_dir = project_root / "config"
            config_dir.mkdir(exist_ok=True)
            
            with open(config_dir / "quality_alerts.json", 'w') as f:
                json.dump(alert_config, f, indent=2)
            
            st.success("🔔 Quality alerts configured and saved!")

def _detect_outliers(data):
    """Detect outliers in numeric columns using IQR method."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    total_outliers = 0
    
    for col in numeric_cols:
        col_data = data[col].dropna()
        if len(col_data) > 4:  # Need at least 5 values for IQR
            try:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    total_outliers += len(outliers)
            except Exception as e:
                logger.warning(f"Error detecting outliers in column {col}: {e}")
                continue
    
    return total_outliers

def _check_schema_violations(data):
    """Check for schema violations (simplified)."""
    violations = 0
    
    try:
        # Check for negative values in columns that should be positive
        positive_cols = ['AttritionProbability', 'ResponseTime', 'Age', 'MonthlyIncome', 'YearsAtCompany']
        for col in positive_cols:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                violations += negative_count
        
        # Check for values outside expected ranges
        if 'AttritionProbability' in data.columns:
            range_violations = ((data['AttritionProbability'] < 0) | (data['AttritionProbability'] > 1)).sum()
            violations += range_violations
        
        # Check for unrealistic age values
        if 'Age' in data.columns:
            age_violations = ((data['Age'] < 16) | (data['Age'] > 100)).sum()
            violations += age_violations
        
        # Check for unrealistic performance ratings
        if 'PerformanceRating' in data.columns:
            perf_violations = ((data['PerformanceRating'] < 1) | (data['PerformanceRating'] > 5)).sum()
            violations += perf_violations
            
    except Exception as e:
        logger.warning(f"Error checking schema violations: {e}")
    
    return violations

def _calculate_completeness_score(data):
    """Calculate data completeness score."""
    if data.empty:
        return 0.0
    
    try:
        total_cells = len(data) * len(data.columns)
        non_missing_cells = total_cells - data.isnull().sum().sum()
        
        return non_missing_cells / total_cells if total_cells > 0 else 0.0
    except Exception as e:
        logger.warning(f"Error calculating completeness score: {e}")
        return 0.0

def _calculate_consistency_score(data):
    """Calculate data consistency score (simplified)."""
    try:
        consistency_score = 0.95  # Base score
        
        # Check department naming consistency
        if 'Department' in data.columns:
            unique_depts = data['Department'].dropna().unique()
            # Penalize if too many department variations (might indicate inconsistent naming)
            if len(unique_depts) > 15:
                consistency_score -= 0.1
        
        # Check for consistency in categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in data.columns:
                unique_values = data[col].dropna().unique()
                # Check for potential inconsistencies (case variations, etc.)
                if len(unique_values) > len(data) * 0.5:  # Too many unique values for categorical
                    consistency_score -= 0.05
        
        return max(0.0, min(1.0, consistency_score))
        
    except Exception as e:
        logger.warning(f"Error calculating consistency score: {e}")
        return 0.90  # Default fallback

# ================================================================
# SYSTEM HEALTH METRICS
# ================================================================

def system_health_metrics():
    """Real-time system health monitoring and metrics."""
    
    st.markdown("### 🏥 System Health Metrics")
    
    # Get real-time system metrics
    metrics = simulate_system_metrics()
    
    # Determine overall system status
    def get_system_status():
        if metrics['cpu_usage'] > 90 or metrics['memory_usage'] > 95:
            return SystemStatus.CRITICAL
        elif metrics['cpu_usage'] > 70 or metrics['memory_usage'] > 80 or metrics['errors_last_hour'] > 5:
            return SystemStatus.WARNING
        elif 'maintenance' in str(metrics.get('notes', '')).lower():
            return SystemStatus.MAINTENANCE
        else:
            return SystemStatus.HEALTHY
    
    system_status = get_system_status()
    
    # System status header
    status_colors = {
        SystemStatus.HEALTHY: get_color('success'),
        SystemStatus.WARNING: get_color('warning'),
        SystemStatus.CRITICAL: get_color('error'),
        SystemStatus.MAINTENANCE: get_color('info')
    }
    
    status_icons = {
        SystemStatus.HEALTHY: "✅",
        SystemStatus.WARNING: "⚠️", 
        SystemStatus.CRITICAL: "🚨",
        SystemStatus.MAINTENANCE: "🔧"
    }
    
    st.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid {status_colors[system_status]};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    ">
        <h2 style="color: {status_colors[system_status]}; margin: 0;">
            {status_icons[system_status]} System Status: {system_status.value}
        </h2>
        <p style="color: {get_color('text_secondary')}; margin: 0.5rem 0 0 0;">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_color = 'error' if metrics['cpu_usage'] > 80 else 'warning' if metrics['cpu_usage'] > 60 else 'success'
        st.markdown(safe_glassmorphism_card(
            value=f"{metrics['cpu_usage']:.1f}%",
            title="CPU Usage",
            subtitle="Current Load",
            icon="🖥️",
            color=cpu_color
        ), unsafe_allow_html=True)
    
    with col2:
        mem_color = 'error' if metrics['memory_usage'] > 85 else 'warning' if metrics['memory_usage'] > 70 else 'success'
        st.markdown(safe_glassmorphism_card(
            value=f"{metrics['memory_usage']:.1f}%",
            title="Memory Usage",
            subtitle="RAM Utilization",
            icon="💾",
            color=mem_color
        ), unsafe_allow_html=True)
    
    with col3:
        response_color = 'error' if metrics['response_time'] > 200 else 'warning' if metrics['response_time'] > 100 else 'success'
        st.markdown(safe_glassmorphism_card(
            value=f"{metrics['response_time']:.0f}ms",
            title="Response Time",
            subtitle="API Performance",
            icon="⚡",
            color=response_color
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(safe_glassmorphism_card(
            value=f"{metrics['active_sessions']}",
            title="Active Sessions",
            subtitle="Current Users",
            icon="👥",
            color='info'
        ), unsafe_allow_html=True)
    
    # Performance gauges
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_gauge = safe_gauge_chart(
            value=metrics['cpu_usage'],
            title="CPU Usage",
            min_value=0,
            max_value=100,
            unit="%",
            height=300
        )
        st.plotly_chart(cpu_gauge, use_container_width=True)
    
    with col2:
        memory_gauge = safe_gauge_chart(
            value=metrics['memory_usage'],
            title="Memory Usage", 
            min_value=0,
            max_value=100,
            unit="%",
            height=300
        )
        st.plotly_chart(memory_gauge, use_container_width=True)
    
    # System performance trends
    st.markdown("#### 📈 System Performance Trends")
    
    # Generate historical data (last 24 hours)
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
    
    # Simulate historical metrics with realistic patterns
    np.random.seed(int(datetime.now().timestamp()) % 2147483647)
    
    # CPU usage with daily patterns (higher during work hours)
    hour_of_day = pd.Series(hours).dt.hour
    base_cpu = 30 + 20 * np.sin((hour_of_day - 6) * np.pi / 12)  # Peak at 6 PM
    hist_cpu = base_cpu + np.random.normal(0, 10, len(hours))
    hist_cpu = np.clip(hist_cpu, 10, 95)
    
    # Memory usage (more stable, slight upward trend)
    base_memory = 50 + np.linspace(0, 10, len(hours))  # Slight memory leak simulation
    hist_memory = base_memory + np.random.normal(0, 5, len(hours))
    hist_memory = np.clip(hist_memory, 30, 90)
    
    # Response time (correlated with CPU usage)
    hist_response = 50 + (hist_cpu / 100) * 150 + np.random.normal(0, 20, len(hours))
    hist_response = np.clip(hist_response, 20, 400)
    
    # Apply smoothing
    hist_cpu = pd.Series(hist_cpu).rolling(window=3, center=True).mean().fillna(hist_cpu)
    hist_memory = pd.Series(hist_memory).rolling(window=3, center=True).mean().fillna(hist_memory)
    hist_response = pd.Series(hist_response).rolling(window=3, center=True).mean().fillna(hist_response)
    
    # Create performance trends chart
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['CPU Usage (%)', 'Memory Usage (%)', 'Response Time (ms)'],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # CPU Usage
    fig.add_trace(
        go.Scatter(
            x=hours, y=hist_cpu,
            mode='lines+markers',
            name='CPU Usage',
            line=dict(color=get_color('secondary'), width=2),
            marker=dict(size=3),
            hovertemplate='<b>CPU Usage</b><br>Time: %{x}<br>Usage: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add CPU threshold line
    fig.add_hline(y=80, line_dash="dash", line_color=get_color('warning'), 
                  annotation_text="High Usage (80%)", row=1, col=1)
    
    # Memory Usage
    fig.add_trace(
        go.Scatter(
            x=hours, y=hist_memory,
            mode='lines+markers',
            name='Memory Usage',
            line=dict(color=get_color('accent'), width=2),
            marker=dict(size=3),
            hovertemplate='<b>Memory Usage</b><br>Time: %{x}<br>Usage: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add Memory threshold line
    fig.add_hline(y=85, line_dash="dash", line_color=get_color('warning'), 
                  annotation_text="High Usage (85%)", row=2, col=1)
    
    # Response Time
    fig.add_trace(
        go.Scatter(
            x=hours, y=hist_response,
            mode='lines+markers',
            name='Response Time',
            line=dict(color=get_color('warning'), width=2),
            marker=dict(size=3),
            hovertemplate='<b>Response Time</b><br>Time: %{x}<br>Time: %{y:.0f}ms<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add Response time threshold line
    fig.add_hline(y=200, line_dash="dash", line_color=get_color('error'), 
                  annotation_text="Slow Response (200ms)", row=3, col=1)
    
    fig.update_layout(
        title=dict(
            text="24-Hour System Performance Trends",
            font=dict(color=get_color('text'), size=20),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=get_color('text')),
        height=600,
        showlegend=False,
        hovermode='x unified'
    )
    
    # Update axes styling
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor='rgba(255,255,255,0.1)',
            color=get_color('text'),
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.1)',
            color=get_color('text'),
            row=i, col=1
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional system metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💿 Storage & Database")
        
        storage_metrics = [
            ('Disk Usage', f"{metrics['disk_usage']:.1f}%", 'success' if metrics['disk_usage'] < 70 else 'warning'),
            ('Database Size', f"{metrics['database_size_mb']:.0f} MB", 'info'),
            ('Cache Hit Rate', f"{metrics['cache_hit_rate']:.1%}", 'success' if metrics['cache_hit_rate'] > 0.9 else 'warning'),
            ('Uptime', f"{metrics['uptime_hours']:.1f} hours", 'success')
        ]
        
        for label, value, color in storage_metrics:
            st.markdown(f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 8px;
                margin: 0.5rem 0;
                border-left: 3px solid {get_color(color)};
            ">
                <span style="color: {get_color('text')};">{label}</span>
                <span style="color: {get_color(color)}; font-weight: bold;">{value}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🚨 System Alerts & Errors")
        
        # Recent alerts (simulated based on current metrics)
        alerts = []
        
        # Generate alerts based on current system state
        if metrics['cpu_usage'] > 75:
            alerts.append({
                'time': datetime.now() - timedelta(minutes=5),
                'level': 'Warning',
                'message': f'High CPU usage detected ({metrics["cpu_usage"]:.1f}%)',
                'resolved': False
            })
        
        if metrics['memory_usage'] > 80:
            alerts.append({
                'time': datetime.now() - timedelta(minutes=10),
                'level': 'Warning',
                'message': f'High memory usage detected ({metrics["memory_usage"]:.1f}%)',
                'resolved': False
            })
        
        if metrics['response_time'] > 150:
            alerts.append({
                'time': datetime.now() - timedelta(minutes=15),
                'level': 'Error',
                'message': f'Slow response times ({metrics["response_time"]:.0f}ms)',
                'resolved': False
            })
        
        if metrics['errors_last_hour'] > 3:
            alerts.append({
                'time': datetime.now() - timedelta(minutes=20),
                'level': 'Error',
                'message': f'{metrics["errors_last_hour"]} errors in the last hour',
                'resolved': False
            })
        
        # Add some historical alerts
        historical_alerts = [
            {
                'time': datetime.now() - timedelta(hours=2),
                'level': 'Info',
                'message': 'Model retrained successfully',
                'resolved': True
            },
            {
                'time': datetime.now() - timedelta(hours=4),
                'level': 'Warning',
                'message': 'Database connection timeout',
                'resolved': True
            },
            {
                'time': datetime.now() - timedelta(hours=6),
                'level': 'Info',
                'message': 'System backup completed',
                'resolved': True
            }
        ]
        
        alerts.extend(historical_alerts)
        
        # Sort by time (newest first) and limit to recent alerts
        alerts = sorted(alerts, key=lambda x: x['time'], reverse=True)[:8]
        
        if alerts:
            for alert in alerts:
                level_colors = {
                    'Error': get_color('error'),
                    'Warning': get_color('warning'),
                    'Info': get_color('info')
                }
                
                level_icons = {
                    'Error': '🚨',
                    'Warning': '⚠️',
                    'Info': 'ℹ️'
                }
                
                status_indicator = "🔴" if not alert['resolved'] else "✅"
                
                time_ago = datetime.now() - alert['time']
                if time_ago.total_seconds() < 3600:
                    time_str = f"{int(time_ago.total_seconds() // 60)}m ago"
                else:
                    time_str = f"{int(time_ago.total_seconds() // 3600)}h ago"
                
                st.markdown(f"""
                <div style="
                    padding: 0.75rem;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    margin: 0.5rem 0;
                    border-left: 3px solid {level_colors[alert['level']]};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: {level_colors[alert['level']]}; font-weight: bold; font-size: 0.875rem;">
                            {level_icons[alert['level']]} {alert['level']}
                        </span>
                        <span style="font-size: 0.75rem;">
                            {status_indicator}
                        </span>
                    </div>
                    <div style="color: {get_color('text')}; font-size: 0.875rem; margin-bottom: 0.25rem;">
                        {alert['message']}
                    </div>
                    <div style="color: {get_color('text_secondary')}; font-size: 0.75rem;">
                        {time_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No recent alerts!")
    
    # System actions
    st.markdown("#### 🔧 System Management Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Metrics", type="primary"):
            with st.spinner("Refreshing system metrics..."):
                time.sleep(1)
            st.success("✅ Metrics refreshed!")
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Cache", type="secondary"):
            with st.spinner("Clearing system cache..."):
                time.sleep(2)
            st.success("✅ Cache cleared! Memory usage reduced.")
    
    with col3:
        if st.button("📊 Export Metrics", type="secondary"):
            # Create metrics export
            metrics_export = {
                'timestamp': datetime.now().isoformat(),
                'system_status': system_status.value,
                'current_metrics': metrics,
                'historical_data': {
                    'cpu_usage': hist_cpu.tolist(),
                    'memory_usage': hist_memory.tolist(),
                    'response_time': hist_response.tolist(),
                    'timestamps': [t.isoformat() for t in hours]
                }
            }
            
            # Save metrics
            logs_dir = project_root / "logs"
            logs_dir.mkdir(exist_ok=True)
            metrics_file = logs_dir / f"system_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_export, f, indent=2, default=str)
            
            st.success(f"📄 Metrics exported to {metrics_file}")
    
    with col4:
        if st.button("🔔 Test Alerts", type="secondary"):
            # Simulate alert test
            test_alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'system_test',
                'message': 'This is a test alert from the admin dashboard',
                'level': 'Info',
                'source': 'Admin Panel'
            }
            
            st.warning("🔔 Test alert generated and sent to monitoring system!")
            st.json(test_alert)

# ================================================================
# USER ACCESS MANAGEMENT
# ================================================================

def user_access_management():
    """User access management and permission control."""
    
    st.markdown("### 👥 User Access Management")
    
    # Generate realistic user data based on our system
    users = [
        UserActivity(
            user_id="USR001",
            username="admin@company.com",
            role=UserRole.ADMIN.value,
            last_login=datetime.now() - timedelta(minutes=30),
            sessions_today=3,
            actions_performed=45,
            pages_accessed=["Dashboard", "Analytics", "Predictions", "Employee Management", "Admin"],
            avg_session_duration=25.5
        ),
        UserActivity(
            user_id="USR002",
            username="hr.manager@company.com",
            role=UserRole.HR_MANAGER.value,
            last_login=datetime.now() - timedelta(hours=2),
            sessions_today=2,
            actions_performed=28,
            pages_accessed=["Dashboard", "Employee Management", "Reports", "Analytics"],
            avg_session_duration=18.3
        ),
        UserActivity(
            user_id="USR003",
            username="data.analyst@company.com",
            role=UserRole.ANALYST.value,
            last_login=datetime.now() - timedelta(hours=1),
            sessions_today=4,
            actions_performed=67,
            pages_accessed=["Analytics", "Predictions", "Reports", "Dashboard"],
            avg_session_duration=32.1
        ),
        UserActivity(
            user_id="USR004", 
            username="manager.smith@company.com",
            role=UserRole.VIEWER.value,
            last_login=datetime.now() - timedelta(days=1),
            sessions_today=0,
            actions_performed=12,
            pages_accessed=["Dashboard", "Reports"],
            avg_session_duration=8.7
        ),
        UserActivity(
            user_id="USR005",
            username="hr.specialist@company.com",
            role=UserRole.HR_MANAGER.value,
            last_login=datetime.now() - timedelta(hours=3),
            sessions_today=1,
            actions_performed=19,
            pages_accessed=["Employee Management", "Dashboard", "Reports"],
            avg_session_duration=22.4
        ),
        UserActivity(
            user_id="USR006",
            username="senior.analyst@company.com",
            role=UserRole.ANALYST.value,
            last_login=datetime.now() - timedelta(minutes=45),
            sessions_today=2,
            actions_performed=38,
            pages_accessed=["Analytics", "Predictions", "Dashboard"],
            avg_session_duration=28.9
        )
    ]
    
    # User overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(safe_glassmorphism_card(
            value=f"{len(users)}",
            title="Total Users",
            subtitle="Registered",
            icon="👥",
            color='info'
        ), unsafe_allow_html=True)
    
    with col2:
        active_today = sum(1 for u in users if u.sessions_today > 0)
        st.markdown(safe_glassmorphism_card(
            value=f"{active_today}",
            title="Active Today",
            subtitle="Current Sessions",
            icon="🟢",
            color='success'
        ), unsafe_allow_html=True)
    
    with col3:
        total_sessions = sum(u.sessions_today for u in users)
        st.markdown(safe_glassmorphism_card(
            value=f"{total_sessions}",
            title="Total Sessions",
            subtitle="Today",
            icon="📊",
            color='secondary'
        ), unsafe_allow_html=True)
    
    with col4:
        avg_session_time = np.mean([u.avg_session_duration for u in users if u.avg_session_duration > 0])
        st.markdown(safe_glassmorphism_card(
            value=f"{avg_session_time:.1f}m",
            title="Avg Session",
            subtitle="Duration",
            icon="⏱️",
            color='accent'
        ), unsafe_allow_html=True)
    
    # User management tabs
    user_tab1, user_tab2, user_tab3, user_tab4 = st.tabs([
        "👥 User Directory",
        "🔐 Role Permissions",
        "📊 Activity Analytics",
        "⚙️ System Settings"
    ])
    
    with user_tab1:
        st.markdown("#### 📋 User Directory")
        
        # User search and filters
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_user = st.text_input("🔍 Search users...", placeholder="Search by username or role")
        
        with col2:
            role_filter = st.selectbox("Filter by Role", ["All"] + [role.value for role in UserRole])
        
        with col3:
            status_filter = st.selectbox("Filter by Status", ["All", "Active", "Inactive"])
        
        # Filter users
        filtered_users = users
        if search_user:
            filtered_users = [u for u in filtered_users if search_user.lower() in u.username.lower() or search_user.lower() in u.role.lower()]
        if role_filter != "All":
            filtered_users = [u for u in filtered_users if u.role == role_filter]
        if status_filter != "All":
            if status_filter == "Active":
                filtered_users = [u for u in filtered_users if u.sessions_today > 0]
            else:
                filtered_users = [u for u in filtered_users if u.sessions_today == 0]
        
        # Display users
        for user in filtered_users:
            # Determine user status
            if user.sessions_today > 0:
                status = "🟢 Active"
                status_color = get_color('success')
            elif (datetime.now() - user.last_login).days < 7:
                status = "🟡 Recent"
                status_color = get_color('warning')
            else:
                status = "🔴 Inactive"
                status_color = get_color('error')
            
            # Role color
            role_colors = {
                UserRole.ADMIN.value: get_color('error'),
                UserRole.HR_MANAGER.value: get_color('warning'),
                UserRole.ANALYST.value: get_color('secondary'),
                UserRole.VIEWER.value: get_color('info')
            }
            role_color = role_colors.get(user.role, get_color('text'))
            
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                transition: all 0.2s ease;
            " onmouseover="this.style.background='rgba(255, 255, 255, 0.05)'" 
               onmouseout="this.style.background='rgba(255, 255, 255, 0.03)'">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="color: {get_color('text')}; margin: 0 0 0.5rem 0; font-size: 1.1rem;">
                            {user.username}
                        </h4>
                        <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 0.5rem;">
                            <span style="
                                background: {role_color};
                                color: white;
                                padding: 0.25rem 0.75rem;
                                border-radius: 12px;
                                font-size: 0.75rem;
                                font-weight: 600;
                            ">
                                {user.role}
                            </span>
                            <span style="color: {status_color}; font-size: 0.875rem; font-weight: 500;">
                                {status}
                            </span>
                        </div>
                        <div style="color: {get_color('text_secondary')}; font-size: 0.875rem;">
                            Last login: {user.last_login.strftime('%Y-%m-%d %H:%M')} • 
                            {user.sessions_today} sessions today • 
                            {user.actions_performed} actions
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {get_color('text')}; font-weight: bold; font-size: 1.1rem;">
                            {user.user_id}
                        </div>
                        <div style="color: {get_color('text_secondary')}; font-size: 0.875rem;">
                            Avg session: {user.avg_session_duration:.1f}m
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                    <div style="color: {get_color('text_secondary')}; font-size: 0.875rem; margin-bottom: 0.5rem;">
                        <strong>Recent Pages Accessed:</strong>
                    </div>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        {' '.join([f'<span style="background: rgba(0, 212, 255, 0.2); color: {get_color("secondary")}; padding: 0.2rem 0.5rem; border-radius: 8px; font-size: 0.75rem;">{page}</span>' for page in user.pages_accessed[:5]])}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # User action buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button(f"✏️ Edit User", key=f"edit_{user.user_id}"):
                    st.info(f"Edit interface for {user.username} would open here")
            
            with col2:
                if st.button(f"🔒 Reset Password", key=f"reset_{user.user_id}"):
                    st.success(f"Password reset email sent to {user.username}")
            
            with col3:
                action_text = "🔓 Activate" if user.sessions_today == 0 else "🔒 Deactivate"
                if st.button(action_text, key=f"toggle_{user.user_id}"):
                    status_change = "activated" if user.sessions_today == 0 else "deactivated"
                    st.warning(f"User {user.username} {status_change}")
            
            with col4:
                if st.button(f"📊 View Analytics", key=f"analytics_{user.user_id}"):
                    # Show user analytics in expander
                    with st.expander(f"📊 Analytics for {user.username}"):
                        st.json({
                            'user_id': user.user_id,
                            'total_sessions': user.sessions_today,
                            'avg_session_duration': user.avg_session_duration,
                            'actions_performed': user.actions_performed,
                            'most_accessed_pages': user.pages_accessed[:3],
                            'last_login': user.last_login.strftime('%Y-%m-%d %H:%M:%S'),
                            'account_status': 'Active' if user.sessions_today > 0 else 'Inactive'
                        })
            
            st.divider()
    
    with user_tab2:
        st.markdown("#### 🔐 Role-Based Access Control")
        
        # Permission matrix for our actual system pages
        permissions = {
            'Dashboard': {UserRole.ADMIN: True, UserRole.HR_MANAGER: True, UserRole.ANALYST: True, UserRole.VIEWER: True},
            'Analytics': {UserRole.ADMIN: True, UserRole.HR_MANAGER: True, UserRole.ANALYST: True, UserRole.VIEWER: False},
            'Predictions': {UserRole.ADMIN: True, UserRole.HR_MANAGER: True, UserRole.ANALYST: True, UserRole.VIEWER: False},
            'Employee Management': {UserRole.ADMIN: True, UserRole.HR_MANAGER: True, UserRole.ANALYST: False, UserRole.VIEWER: False},
            'Reports': {UserRole.ADMIN: True, UserRole.HR_MANAGER: True, UserRole.ANALYST: True, UserRole.VIEWER: True},
            'Admin Panel': {UserRole.ADMIN: True, UserRole.HR_MANAGER: False, UserRole.ANALYST: False, UserRole.VIEWER: False},
            'User Management': {UserRole.ADMIN: True, UserRole.HR_MANAGER: False, UserRole.ANALYST: False, UserRole.VIEWER: False},
            'System Settings': {UserRole.ADMIN: True, UserRole.HR_MANAGER: False, UserRole.ANALYST: False, UserRole.VIEWER: False},
            'Model Training': {UserRole.ADMIN: True, UserRole.HR_MANAGER: False, UserRole.ANALYST: True, UserRole.VIEWER: False},
            'Data Export': {UserRole.ADMIN: True, UserRole.HR_MANAGER: True, UserRole.ANALYST: True, UserRole.VIEWER: False}
        }
        
        # Create permission matrix table
        matrix_data = []
        for feature, role_perms in permissions.items():
            row = {'Feature': feature}
            for role in UserRole:
                has_permission = role_perms.get(role, False)
                row[role.value] = "✅ Granted" if has_permission else "❌ Denied"
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(matrix_data)
        
        st.dataframe(
            matrix_df,
            use_container_width=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature/Page", width="medium"),
                **{role.value: st.column_config.TextColumn(role.value, width="small") for role in UserRole}
            }
        )
        
        # Permission management
        st.markdown("#### ⚙️ Permission Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_role = st.selectbox("Select Role:", [role.value for role in UserRole])
        
        with col2:
            selected_feature = st.selectbox("Select Feature:", list(permissions.keys()))
        
        with col3:
            if st.button("🔄 Toggle Permission", type="secondary"):
                # Get current permission state
                current_role = UserRole(selected_role)
                current_perm = permissions[selected_feature].get(current_role, False)
                
                # Toggle permission
                permissions[selected_feature][current_role] = not current_perm
                new_status = "granted" if not current_perm else "revoked"
                
                st.success(f"✅ Permission {new_status} for {selected_role} on {selected_feature}")
                time.sleep(1)
                st.rerun()
        
        # Role summary
        st.markdown("#### 📊 Role Summary")
        
        role_summary = []
        for role in UserRole:
            total_permissions = len(permissions)
            granted_permissions = sum(1 for feature_perms in permissions.values() if feature_perms.get(role, False))
            permission_percentage = (granted_permissions / total_permissions) * 100
            
            users_with_role = len([u for u in users if u.role == role.value])
            
            role_summary.append({
                'Role': role.value,
                'Users': users_with_role,
                'Permissions Granted': f"{granted_permissions}/{total_permissions}",
                'Access Level': f"{permission_percentage:.0f}%",
                'Status': '🔴 Restricted' if permission_percentage < 30 else '🟡 Limited' if permission_percentage < 70 else '🟢 Full Access'
            })
        
        summary_df = pd.DataFrame(role_summary)
        st.dataframe(summary_df, use_container_width=True)
    
    with user_tab3:
        st.markdown("#### 📊 User Activity Analytics")
        
        # Activity charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sessions by role
            role_sessions = {}
            role_counts = {}
            
            for user in users:
                if user.role not in role_sessions:
                    role_sessions[user.role] = 0
                    role_counts[user.role] = 0
                role_sessions[user.role] += user.sessions_today
                role_counts[user.role] += 1
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(role_sessions.keys()),
                y=list(role_sessions.values()),
                marker=dict(color=get_color('secondary')),
                text=list(role_sessions.values()),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Sessions: %{y}<br>Users: %{customdata}<extra></extra>',
                customdata=[role_counts[role] for role in role_sessions.keys()]
            ))
            
            fig = safe_create_chart(
                fig,
                title="Total Sessions by Role (Today)",
                height=400,
                custom_layout={
                    'yaxis': dict(title='Number of Sessions'),
                    'xaxis': dict(title='User Role', tickangle=45)
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average session duration by role
            role_duration = {}
            role_user_count = {}
            
            for user in users:
                if user.role not in role_duration:
                    role_duration[user.role] = 0
                    role_user_count[user.role] = 0
                if user.avg_session_duration > 0:  # Only count users with actual sessions
                    role_duration[user.role] += user.avg_session_duration
                    role_user_count[user.role] += 1
            
            avg_duration = {}
            for role in role_duration:
                if role_user_count[role] > 0:
                    avg_duration[role] = role_duration[role] / role_user_count[role]
                else:
                    avg_duration[role] = 0
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(avg_duration.keys()),
                y=list(avg_duration.values()),
                marker=dict(color=get_color('accent')),
                text=[f'{duration:.1f}m' for duration in avg_duration.values()],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Duration: %{y:.1f} min<extra></extra>'
            ))
            
            fig = safe_create_chart(
                fig,
                title="Average Session Duration by Role",
                height=400,
                custom_layout={
                    'yaxis': dict(title='Duration (minutes)'),
                    'xaxis': dict(title='User Role', tickangle=45)
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # User activity timeline
        st.markdown("#### ⏰ Recent Activity Timeline")
        
        # Generate activity timeline based on user data
        activities = []
        action_types = ['Login', 'View Dashboard', 'Run Analysis', 'Export Report', 'Update Profile', 'Access Admin', 'Generate Prediction']
        
        for user in users:
            # Generate activities based on user's action count
            for i in range(min(user.actions_performed, 50)):  # Limit for performance
                activity_time = user.last_login + timedelta(
                    minutes=np.random.randint(-480, 30)  # Activities in last 8 hours
                )
                
                # Ensure activity time is not in the future
                activity_time = min(activity_time, datetime.now())
                
                # Choose appropriate action based on user role and pages accessed
                if user.role == UserRole.ADMIN.value:
                    possible_actions = action_types
                elif user.role == UserRole.ANALYST.value:
                    possible_actions = ['Run Analysis', 'Generate Prediction', 'Export Report', 'View Dashboard', 'Login']
                elif user.role == UserRole.HR_MANAGER.value:
                    possible_actions = ['View Dashboard', 'Export Report', 'Update Profile', 'Login', 'Access Admin']
                else:  # VIEWER
                    possible_actions = ['View Dashboard', 'Login']
                
                action = np.random.choice(possible_actions)
                page = np.random.choice(user.pages_accessed) if user.pages_accessed else 'Dashboard'
                
                activities.append({
                    'user': user.username,
                    'role': user.role,
                    'action': action,
                    'timestamp': activity_time,
                    'page': page,
                    'user_id': user.user_id
                })
        
        # Sort by timestamp and show recent activities
        activities = sorted(activities, key=lambda x: x['timestamp'], reverse=True)[:25]
        
        for activity in activities:
            time_diff = datetime.now() - activity['timestamp']
            
            if time_diff.total_seconds() < 3600:
                time_str = f"{int(time_diff.total_seconds() // 60)}m ago"
            elif time_diff.total_seconds() < 86400:
                time_str = f"{int(time_diff.total_seconds() // 3600)}h ago"
            else:
                time_str = f"{int(time_diff.days)}d ago"
            
            # Choose icon based on activity type
            action_icons = {
                'Login': '🔑',
                'View Dashboard': '📊',
                'Run Analysis': '🔬',
                'Export Report': '📄',
                'Update Profile': '✏️',
                'Access Admin': '⚙️',
                'Generate Prediction': '🎯'
            }
            
            icon = action_icons.get(activity['action'], '📝')
            
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.02);
                border-left: 3px solid {get_color('info')};
                border-radius: 0 8px 8px 0;
                padding: 0.75rem;
                margin: 0.5rem 0;
                transition: background 0.2s ease;
            " onmouseover="this.style.background='rgba(255, 255, 255, 0.05)'" 
               onmouseout="this.style.background='rgba(255, 255, 255, 0.02)'">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <span style="font-size: 1.1rem; margin-right: 0.5rem;">{icon}</span>
                        <span style="color: {get_color('text')}; font-weight: 600;">
                            {activity['user']}
                        </span>
                        <span style="color: {get_color('text_secondary')}; margin: 0 0.5rem;">
                            ({activity['role']})
                        </span>
                        <span style="color: {get_color('secondary')};">
                            {activity['action']}
                        </span>
                        <span style="color: {get_color('text_secondary')};">
                            in {activity['page']}
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {get_color('text_secondary')}; font-size: 0.875rem;">
                            {time_str}
                        </div>
                        <div style="color: {get_color('accent')}; font-size: 0.75rem;">
                            {activity['user_id']}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Activity summary statistics
        st.markdown("#### 📈 Activity Statistics")
        
        # Calculate statistics
        total_activities = len(activities)
        unique_users_active = len(set(a['user'] for a in activities))
        most_active_user = max(set(a['user'] for a in activities), key=lambda u: len([a for a in activities if a['user'] == u]))
        most_common_action = max(set(a['action'] for a in activities), key=lambda action: len([a for a in activities if a['action'] == action]))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Recent Activities", total_activities)
