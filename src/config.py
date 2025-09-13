"""
HR Attrition Predictor - Configuration Management System
========================================================
Centralized configuration management with support for YAML files,
environment variables, and dynamic parameter loading.

Author: HR Analytics Team
Date: September 2025
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelParams:
    """Data class for model hyperparameters"""
    # Logistic Regression
    logistic_C: float = 1.0
    logistic_max_iter: int = 1000
    logistic_solver: str = 'liblinear'
    
    # XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    
    # General ML Settings
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    scoring_metric: str = 'roc_auc'


@dataclass  
class DataPaths:
    """Data class for file and directory paths"""
    # Base directories
    project_root: Path
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    reports_dir: Path
    
    # Data subdirectories
    raw_data_dir: Path
    processed_data_dir: Path
    synthetic_data_dir: Path
    
    # Model subdirectories
    saved_models_dir: Path
    checkpoints_dir: Path
    
    # Specific files
    synthetic_employees_file: Path
    processed_train_file: Path
    processed_test_file: Path
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path) and 'file' not in field_name:
                field_value.mkdir(parents=True, exist_ok=True)


@dataclass
class EmailConfig:
    """Email service configuration"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    sender_email: Optional[str] = None
    sender_password: Optional[str] = None
    default_subject: str = "HR Attrition Alert"
    
    @property
    def is_configured(self) -> bool:
        """Check if email is properly configured"""
        return bool(self.sender_email and self.sender_password)


@dataclass
class StreamlitConfig:
    """Streamlit application configuration"""
    page_title: str = "HR Attrition Predictor"
    page_icon: str = "🏢"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Theme colors (Dark Cyberpunk)
    primary_color: str = "#00D4FF"      # Electric Blue
    background_color: str = "#0A0E27"   # Deep Space Blue
    secondary_color: str = "#B026FF"    # Neon Purple
    text_color: str = "#F0F8FF"         # Ice White
    success_color: str = "#00FF88"      # Matrix Green
    warning_color: str = "#FF6B35"      # Cyber Orange


class Config:
    """
    Main configuration management class for HR Attrition Predictor.
    
    Supports loading from YAML files, environment variables, and 
    provides centralized access to all application settings.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to main config.yaml file
        """
        # Determine project root
        self.project_root = self._find_project_root()
        
        # Initialize configuration dictionaries
        self._config_data: Dict[str, Any] = {}
        self._model_params: Optional[ModelParams] = None
        self._data_paths: Optional[DataPaths] = None
        self._email_config: Optional[EmailConfig] = None
        self._streamlit_config: Optional[StreamlitConfig] = None
        
        # Load configurations
        self.config_path = config_path or self.project_root / "config.yaml"
        self.load_config()
        
        logger.info(f"Configuration loaded from: {self.config_path}")
    
    def _find_project_root(self) -> Path:
        """
        Find the project root directory by looking for specific files.
        
        Returns:
            Path to project root directory
        """
        current_path = Path(__file__).resolve()
        
        # Look for project indicators
        indicators = ["requirements.txt", "config.yaml", ".git", "README.md"]
        
        for parent in [current_path] + list(current_path.parents):
            if any((parent / indicator).exists() for indicator in indicators):
                return parent
        
        # Fallback to parent of src directory
        return current_path.parent.parent
    
    def load_config(self) -> None:
        """
        Load configuration from YAML file and environment variables.
        Implements layered configuration with environment overrides.
        """
        try:
            # Load from YAML file if exists
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {self.config_path}")
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                self._config_data = {}
            
            # Override with environment variables
            self._load_env_overrides()
            
            # Initialize data classes
            self._initialize_configs()
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Initialize with defaults
            self._config_data = {}
            self._initialize_configs()
    
    def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables"""
        env_mappings = {
            'HR_DATA_PATH': ['data', 'base_path'],
            'HR_MODELS_PATH': ['models', 'base_path'],
            'HR_EMAIL_SENDER': ['email', 'sender_email'],
            'HR_EMAIL_PASSWORD': ['email', 'sender_password'],
            'HR_SMTP_SERVER': ['email', 'smtp_server'],
            'HR_LOG_LEVEL': ['logging', 'level'],
            'HR_RANDOM_SEED': ['model', 'random_state']
        }
        
        for env_var, config_path in env_mappings.items():
            if env_value := os.getenv(env_var):
                self._set_nested_config(config_path, env_value)
    
    def _set_nested_config(self, path: List[str], value: Any) -> None:
        """Set nested configuration value"""
        current = self._config_data
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value
    
    def _initialize_configs(self) -> None:
        """Initialize configuration data classes"""
        # Initialize data paths
        self._data_paths = DataPaths(
            project_root=self.project_root,
            data_dir=self.project_root / "data",
            models_dir=self.project_root / "models",
            logs_dir=self.project_root / "logs",
            reports_dir=self.project_root / "reports",
            raw_data_dir=self.project_root / "data" / "raw",
            processed_data_dir=self.project_root / "data" / "processed", 
            synthetic_data_dir=self.project_root / "data" / "synthetic",
            saved_models_dir=self.project_root / "models" / "saved_models",
            checkpoints_dir=self.project_root / "models" / "checkpoints",
            synthetic_employees_file=self.project_root / "data" / "synthetic" / "hr_employees.csv",
            processed_train_file=self.project_root / "data" / "processed" / "train_data.pkl",
            processed_test_file=self.project_root / "data" / "processed" / "test_data.pkl"
        )
        
        # Initialize model parameters
        model_config = self._config_data.get('model', {})
        self._model_params = ModelParams(
            logistic_C=model_config.get('logistic_C', 1.0),
            logistic_max_iter=model_config.get('logistic_max_iter', 1000),
            xgb_n_estimators=model_config.get('xgb_n_estimators', 100),
            xgb_max_depth=model_config.get('xgb_max_depth', 6),
            xgb_learning_rate=model_config.get('xgb_learning_rate', 0.1),
            rf_n_estimators=model_config.get('rf_n_estimators', 100),
            rf_max_depth=model_config.get('rf_max_depth', 10),
            test_size=model_config.get('test_size', 0.2),
            random_state=model_config.get('random_state', 42),
            cv_folds=model_config.get('cv_folds', 5),
            scoring_metric=model_config.get('scoring_metric', 'roc_auc')
        )
        
        # Initialize email configuration
        email_config = self._config_data.get('email', {})
        self._email_config = EmailConfig(
            smtp_server=email_config.get('smtp_server', 'smtp.gmail.com'),
            smtp_port=email_config.get('smtp_port', 587),
            sender_email=email_config.get('sender_email'),
            sender_password=email_config.get('sender_password'),
            default_subject=email_config.get('default_subject', 'HR Attrition Alert')
        )
        
        # Initialize Streamlit configuration
        ui_config = self._config_data.get('streamlit', {})
        self._streamlit_config = StreamlitConfig(
            page_title=ui_config.get('page_title', 'HR Attrition Predictor'),
            page_icon=ui_config.get('page_icon', '🏢'),
            layout=ui_config.get('layout', 'wide'),
            primary_color=ui_config.get('primary_color', '#00D4FF'),
            background_color=ui_config.get('background_color', '#0A0E27')
        )
    
    def get_model_params(self) -> ModelParams:
        """
        Get model hyperparameters configuration.
        
        Returns:
            ModelParams dataclass with all ML model settings
        """
        return self._model_params
    
    def get_data_paths(self) -> DataPaths:
        """
        Get data file and directory paths.
        
        Returns:
            DataPaths dataclass with all file/directory paths
        """
        return self._data_paths
    
    def get_email_config(self) -> EmailConfig:
        """
        Get email service configuration.
        
        Returns:
            EmailConfig dataclass with SMTP settings
        """
        return self._email_config
    
    def get_streamlit_config(self) -> StreamlitConfig:
        """
        Get Streamlit UI configuration.
        
        Returns:
            StreamlitConfig dataclass with UI settings
        """
        return self._streamlit_config
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with dot notation support.
        
        Args:
            key: Configuration key (supports 'section.subsection.key')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        current = self._config_data
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(base_dict: dict, update_dict: dict) -> dict:
            """Recursively update nested dictionaries"""
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    base_dict[key] = deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict
        
        self._config_data = deep_update(self._config_data, updates)
        self._initialize_configs()  # Re-initialize data classes
        
        logger.info("Configuration updated")
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Optional custom output path
        """
        output_path = output_path or self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate configuration completeness and correctness.
        
        Returns:
            Dictionary with validation results (errors/warnings)
        """
        issues = {'errors': [], 'warnings': []}
        
        # Check critical paths exist
        if not self._data_paths.project_root.exists():
            issues['errors'].append(f"Project root not found: {self._data_paths.project_root}")
        
        # Check model parameters are reasonable
        if self._model_params.test_size <= 0 or self._model_params.test_size >= 1:
            issues['errors'].append(f"Invalid test_size: {self._model_params.test_size}")
        
        # Check email configuration
        if not self._email_config.is_configured:
            issues['warnings'].append("Email not configured - notifications disabled")
        
        # Check Streamlit config
        if self._streamlit_config.layout not in ['centered', 'wide']:
            issues['warnings'].append(f"Unknown layout: {self._streamlit_config.layout}")
        
        return issues
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"Config(project_root='{self.project_root}', config_path='{self.config_path}')"


# Global configuration instance (singleton pattern)
_global_config: Optional[Config] = None

def get_config() -> Config:
    """
    Get global configuration instance (singleton).
    
    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Reload global configuration from file.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Reloaded Config instance
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config


# Convenience functions for quick access
def get_model_params() -> ModelParams:
    """Quick access to model parameters"""
    return get_config().get_model_params()

def get_data_paths() -> DataPaths:
    """Quick access to data paths"""
    return get_config().get_data_paths()

def get_email_config() -> EmailConfig:
    """Quick access to email config"""
    return get_config().get_email_config()

def get_streamlit_config() -> StreamlitConfig:
    """Quick access to Streamlit config"""
    return get_config().get_streamlit_config()


if __name__ == "__main__":
    # Test configuration loading
    config = Config()
    print(f"Project root: {config.get_data_paths().project_root}")
    print(f"Model parameters: {config.get_model_params()}")
    print(f"Email configured: {config.get_email_config().is_configured}")
    
    # Validate configuration
    issues = config.validate_config()
    if issues['errors']:
        print("Configuration errors:", issues['errors'])
    if issues['warnings']:
        print("Configuration warnings:", issues['warnings'])
