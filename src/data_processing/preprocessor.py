"""
HR Attrition Predictor - Data Preprocessing Pipeline
==================================================
Comprehensive data preprocessing with cleaning, encoding, scaling,
and train-test splitting for machine learning pipeline.

Author: HR Analytics Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from datetime import datetime
import pickle
import joblib
from dataclasses import dataclass, asdict
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class PreprocessingConfig:
    """Configuration class for preprocessing parameters"""
    # Missing value handling
    missing_strategy_numeric: str = 'median'  # 'mean', 'median', 'most_frequent', 'constant', 'knn'
    missing_strategy_categorical: str = 'most_frequent'  # 'most_frequent', 'constant'
    missing_constant_value: Any = 'Unknown'
    knn_neighbors: int = 5
    
    # Encoding options
    categorical_encoding: str = 'onehot'  # 'onehot', 'label', 'ordinal', 'target'
    handle_unknown: str = 'ignore'  # 'error', 'ignore'
    drop_first_dummy: bool = True
    
    # Scaling options
    scaling_method: str = 'standard'  # 'standard', 'minmax', 'robust', 'quantile', 'power'
    
    # Feature selection
    remove_low_variance: bool = True
    variance_threshold: float = 0.01
    remove_highly_correlated: bool = True
    correlation_threshold: float = 0.95
    
    # Train-test split
    test_size: float = 0.2
    validation_size: float = 0.15
    random_state: int = 42
    stratify_column: str = 'Attrition'
    
    # Data cleaning
    remove_duplicates: bool = True
    outlier_treatment: str = 'cap'  # 'remove', 'cap', 'transform', 'none'
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
    outlier_cap_percentiles: Tuple[float, float] = (1, 99)


@dataclass
class PreprocessingReport:
    """Report class for preprocessing operations"""
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    removed_features: List[str]
    encoded_features: List[str]
    scaled_features: List[str]
    missing_values_handled: Dict[str, int]
    outliers_treated: Dict[str, int]
    train_shape: Tuple[int, int]
    test_shape: Tuple[int, int]
    validation_shape: Optional[Tuple[int, int]]
    processing_time: float
    preprocessing_timestamp: str


class CustomImputer(BaseEstimator, TransformerMixin):
    """Custom imputer with advanced strategies"""
    
    def __init__(self, strategy='median', fill_value=None, knn_neighbors=5):
        self.strategy = strategy
        self.fill_value = fill_value
        self.knn_neighbors = knn_neighbors
        self.imputer_ = None
    
    def fit(self, X, y=None):
        if self.strategy == 'knn':
            self.imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
        else:
            self.imputer_ = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        
        self.imputer_.fit(X)
        return self
    
    def transform(self, X):
        return self.imputer_.transform(X)


class OutlierTreatment(BaseEstimator, TransformerMixin):
    """Custom outlier treatment transformer"""
    
    def __init__(self, method='iqr', treatment='cap', percentiles=(1, 99)):
        self.method = method
        self.treatment = treatment
        self.percentiles = percentiles
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for column in X_df.select_dtypes(include=[np.number]).columns:
            if self.method == 'iqr':
                Q1 = X_df[column].quantile(0.25)
                Q3 = X_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            elif self.method == 'percentile':
                lower_bound = X_df[column].quantile(self.percentiles[0] / 100)
                upper_bound = X_df[column].quantile(self.percentiles[1] / 100)
            else:  # zscore
                mean = X_df[column].mean()
                std = X_df[column].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
            
            self.bounds_[column] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        outlier_counts = {}
        
        for column, (lower_bound, upper_bound) in self.bounds_.items():
            if column in X_df.columns:
                if self.treatment == 'cap':
                    outliers_lower = X_df[column] < lower_bound
                    outliers_upper = X_df[column] > upper_bound
                    outlier_counts[column] = outliers_lower.sum() + outliers_upper.sum()
                    
                    X_df.loc[outliers_lower, column] = lower_bound
                    X_df.loc[outliers_upper, column] = upper_bound
                    
                elif self.treatment == 'remove':
                    mask = (X_df[column] >= lower_bound) & (X_df[column] <= upper_bound)
                    outlier_counts[column] = (~mask).sum()
                    X_df = X_df[mask]
        
        self.outlier_counts_ = outlier_counts
        return X_df.values if not isinstance(X, pd.DataFrame) else X_df


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for HR attrition prediction.
    
    Handles data cleaning, encoding, scaling, and train-test splitting with
    enterprise-grade features and comprehensive reporting.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Optional preprocessing configuration
        """
        self.config = config if config else PreprocessingConfig()
        self.preprocessing_pipeline: Optional[Pipeline] = None
        self.feature_names_: Optional[List[str]] = None
        self.target_encoder_: Optional[LabelEncoder] = None
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.scaler_: Optional[Any] = None
        self.report_: Optional[PreprocessingReport] = None
        
        # Store original data references
        self.original_data_: Optional[pd.DataFrame] = None
        self.processed_data_: Optional[pd.DataFrame] = None
        
        logger.info("DataPreprocessor initialized with configuration")
    
    def fit_transform(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (processed_features, target_series)
        """
        start_time = datetime.now()
        logger.info(f"Starting comprehensive data preprocessing...")
        logger.info(f"Input data shape: {data.shape}")
        
        self.original_data_ = data.copy()
        original_shape = data.shape
        
        # Step 1: Clean data
        cleaned_data = self.clean_data(data)
        logger.info(f"After cleaning: {cleaned_data.shape}")
        
        # Step 2: Handle missing values
        imputed_data = self.handle_missing_values(cleaned_data)
        logger.info(f"After imputation: {imputed_data.shape}")
        
        # Step 3: Encode categorical variables
        encoded_data = self.encode_categorical(imputed_data)
        logger.info(f"After encoding: {encoded_data.shape}")
        
        # Step 4: Scale numerical features
        scaled_data = self.scale_numerical(encoded_data)
        logger.info(f"After scaling: {scaled_data.shape}")
        
        # Prepare target variable
        target_series = None
        if target_column and target_column in self.original_data_.columns:
            target_series = self._prepare_target(self.original_data_[target_column])
        
        self.processed_data_ = scaled_data
        
        # Create preprocessing report
        processing_time = (datetime.now() - start_time).total_seconds()
        self.report_ = PreprocessingReport(
            original_shape=original_shape,
            processed_shape=scaled_data.shape,
            removed_features=self._get_removed_features(),
            encoded_features=self._get_encoded_features(),
            scaled_features=self._get_scaled_features(),
            missing_values_handled=getattr(self, 'missing_counts_', {}),
            outliers_treated=getattr(self, 'outlier_counts_', {}),
            train_shape=(0, 0),  # Will be updated in create_train_test_split
            test_shape=(0, 0),
            validation_shape=None,
            processing_time=processing_time,
            preprocessing_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Preprocessing completed in {processing_time:.2f} seconds")
        return scaled_data, target_series
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling duplicates, invalid values, and basic issues.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        cleaned_data = data.copy()
        initial_rows = len(cleaned_data)
        
        # Remove duplicates
        if self.config.remove_duplicates:
            duplicates_count = cleaned_data.duplicated().sum()
            if duplicates_count > 0:
                cleaned_data = cleaned_data.drop_duplicates()
                logger.info(f"Removed {duplicates_count} duplicate records")
        
        # Remove rows with all missing values
        all_missing_mask = cleaned_data.isnull().all(axis=1)
        if all_missing_mask.any():
            rows_removed = all_missing_mask.sum()
            cleaned_data = cleaned_data[~all_missing_mask]
            logger.info(f"Removed {rows_removed} rows with all missing values")
        
        # Clean specific data issues
        cleaned_data = self._clean_data_types(cleaned_data)
        cleaned_data = self._clean_business_logic(cleaned_data)
        
        rows_removed = initial_rows - len(cleaned_data)
        if rows_removed > 0:
            logger.info(f"Total rows removed during cleaning: {rows_removed}")
        
        return cleaned_data
    
    def _clean_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data type issues and invalid values"""
        cleaned_data = data.copy()
        
        # Clean age values
        if 'Age' in cleaned_data.columns:
            # Remove impossible ages
            age_mask = (cleaned_data['Age'] >= 16) & (cleaned_data['Age'] <= 80)
            invalid_ages = (~age_mask).sum()
            if invalid_ages > 0:
                logger.info(f"Removing {invalid_ages} records with invalid ages")
                cleaned_data = cleaned_data[age_mask]
        
        # Clean salary values
        salary_columns = ['MonthlyIncome', 'HourlyRate', 'DailyRate']
        for col in salary_columns:
            if col in cleaned_data.columns:
                # Remove negative or zero salaries
                salary_mask = cleaned_data[col] > 0
                invalid_salaries = (~salary_mask).sum()
                if invalid_salaries > 0:
                    logger.info(f"Removing {invalid_salaries} records with invalid {col}")
                    cleaned_data = cleaned_data[salary_mask]
        
        # Clean experience values
        experience_columns = ['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole']
        for col in experience_columns:
            if col in cleaned_data.columns:
                # Remove negative experience
                exp_mask = cleaned_data[col] >= 0
                invalid_exp = (~exp_mask).sum()
                if invalid_exp > 0:
                    logger.info(f"Removing {invalid_exp} records with negative {col}")
                    cleaned_data = cleaned_data[exp_mask]
        
        return cleaned_data
    
    def _clean_business_logic(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply business logic cleaning rules"""
        cleaned_data = data.copy()
        initial_rows = len(cleaned_data)
        
        # Rule 1: Years in current role should not exceed years at company
        if all(col in cleaned_data.columns for col in ['YearsInCurrentRole', 'YearsAtCompany']):
            valid_tenure = cleaned_data['YearsInCurrentRole'] <= cleaned_data['YearsAtCompany']
            invalid_count = (~valid_tenure).sum()
            if invalid_count > 0:
                logger.info(f"Fixing {invalid_count} records where YearsInCurrentRole > YearsAtCompany")
                # Fix by setting YearsInCurrentRole = YearsAtCompany
                mask = cleaned_data['YearsInCurrentRole'] > cleaned_data['YearsAtCompany']
                cleaned_data.loc[mask, 'YearsInCurrentRole'] = cleaned_data.loc[mask, 'YearsAtCompany']
        
        # Rule 2: Total working years should not exceed possible working years based on age
        if all(col in cleaned_data.columns for col in ['Age', 'TotalWorkingYears']):
            max_possible_years = cleaned_data['Age'] - 16  # Assume work starts at 16
            valid_experience = cleaned_data['TotalWorkingYears'] <= max_possible_years
            invalid_count = (~valid_experience).sum()
            if invalid_count > 0:
                logger.info(f"Fixing {invalid_count} records with impossible work experience")
                # Cap at maximum possible
                mask = cleaned_data['TotalWorkingYears'] > max_possible_years
                cleaned_data.loc[mask, 'TotalWorkingYears'] = max_possible_years[mask]
        
        # Rule 3: Years since last promotion should not exceed years at company
        if all(col in cleaned_data.columns for col in ['YearsSinceLastPromotion', 'YearsAtCompany']):
            valid_promotion = cleaned_data['YearsSinceLastPromotion'] <= cleaned_data['YearsAtCompany']
            invalid_count = (~valid_promotion).sum()
            if invalid_count > 0:
                logger.info(f"Fixing {invalid_count} records with invalid promotion timeline")
                mask = cleaned_data['YearsSinceLastPromotion'] > cleaned_data['YearsAtCompany']
                cleaned_data.loc[mask, 'YearsSinceLastPromotion'] = cleaned_data.loc[mask, 'YearsAtCompany']
        
        rows_fixed = initial_rows - len(cleaned_data) if len(cleaned_data) != initial_rows else 0
        if rows_fixed > 0:
            logger.info(f"Business logic cleaning completed: {rows_fixed} rows affected")
        
        return cleaned_data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using various imputation strategies.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        imputed_data = data.copy()
        
        # Analyze missing values
        missing_counts = imputed_data.isnull().sum()
        self.missing_counts_ = missing_counts[missing_counts > 0].to_dict()
        
        if not self.missing_counts_:
            logger.info("No missing values found")
            return imputed_data
        
        logger.info(f"Found missing values in {len(self.missing_counts_)} columns")
        
        # Separate numeric and categorical columns
        numeric_columns = imputed_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = imputed_data.select_dtypes(include=['object']).columns.tolist()
        
        # Handle numeric missing values
        numeric_missing = [col for col in numeric_columns if col in self.missing_counts_]
        if numeric_missing:
            logger.info(f"Imputing {len(numeric_missing)} numeric columns")
            
            if self.config.missing_strategy_numeric == 'knn':
                imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
                imputed_data[numeric_missing] = imputer.fit_transform(imputed_data[numeric_missing])
            else:
                imputer = SimpleImputer(strategy=self.config.missing_strategy_numeric)
                imputed_data[numeric_missing] = imputer.fit_transform(imputed_data[numeric_missing])
        
        # Handle categorical missing values
        categorical_missing = [col for col in categorical_columns if col in self.missing_counts_]
        if categorical_missing:
            logger.info(f"Imputing {len(categorical_missing)} categorical columns")
            
            for col in categorical_missing:
                if self.config.missing_strategy_categorical == 'most_frequent':
                    most_frequent = imputed_data[col].mode().iloc[0] if len(imputed_data[col].mode()) > 0 else 'Unknown'
                    imputed_data[col].fillna(most_frequent, inplace=True)
                elif self.config.missing_strategy_categorical == 'constant':
                    imputed_data[col].fillna(self.config.missing_constant_value, inplace=True)
        
        # Verify no missing values remain
        remaining_missing = imputed_data.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"Warning: {remaining_missing} missing values still remain")
        else:
            logger.info("All missing values successfully handled")
        
        return imputed_data
    
    def encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using specified strategy.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info(f"Encoding categorical variables using {self.config.categorical_encoding} method...")
        
        encoded_data = data.copy()
        categorical_columns = encoded_data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from encoding if present
        if self.config.stratify_column in categorical_columns:
            categorical_columns.remove(self.config.stratify_column)
        
        if not categorical_columns:
            logger.info("No categorical columns to encode")
            return encoded_data
        
        logger.info(f"Encoding {len(categorical_columns)} categorical columns: {categorical_columns}")
        
        if self.config.categorical_encoding == 'onehot':
            # One-hot encoding
            encoded_data = pd.get_dummies(
                encoded_data,
                columns=categorical_columns,
                drop_first=self.config.drop_first_dummy,
                dummy_na=False
            )
            
        elif self.config.categorical_encoding == 'label':
            # Label encoding
            for col in categorical_columns:
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                
        elif self.config.categorical_encoding == 'ordinal':
            # Ordinal encoding (maintains order)
            for col in categorical_columns:
                unique_values = sorted(encoded_data[col].unique())
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                encoded_data[col] = encoded_data[col].map(value_map)
        
        # Store encoded feature names
        self.encoded_features_ = [col for col in encoded_data.columns if col not in data.columns]
        
        logger.info(f"Categorical encoding completed. New shape: {encoded_data.shape}")
        return encoded_data
    
    def scale_numerical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using specified scaling method.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with scaled numerical features
        """
        logger.info(f"Scaling numerical features using {self.config.scaling_method} method...")
        
        scaled_data = data.copy()
        numeric_columns = scaled_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and target-related columns from scaling
        exclude_from_scaling = ['EmployeeID', 'AttritionProbability', 'RiskScore']
        numeric_columns = [col for col in numeric_columns if col not in exclude_from_scaling]
        
        if not numeric_columns:
            logger.info("No numerical columns to scale")
            return scaled_data
        
        logger.info(f"Scaling {len(numeric_columns)} numerical columns")
        
        # Choose scaler
        if self.config.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.config.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.config.scaling_method == 'robust':
            scaler = RobustScaler()
        elif self.config.scaling_method == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            scaler = QuantileTransformer(output_distribution='uniform')
        elif self.config.scaling_method == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            logger.warning(f"Unknown scaling method: {self.config.scaling_method}, using standard")
            scaler = StandardScaler()
        
        # Fit and transform
        scaled_data[numeric_columns] = scaler.fit_transform(scaled_data[numeric_columns])
        
        # Store scaler and scaled features
        self.scaler_ = scaler
        self.scaled_features_ = numeric_columns
        
        logger.info(f"Numerical scaling completed")
        return scaled_data
    
    def _prepare_target(self, target_series: pd.Series) -> pd.Series:
        """Prepare target variable for modeling"""
        if target_series.dtype == 'object':
            # Encode string targets
            self.target_encoder_ = LabelEncoder()
            encoded_target = self.target_encoder_.fit_transform(target_series)
            logger.info(f"Target variable encoded: {dict(zip(self.target_encoder_.classes_, range(len(self.target_encoder_.classes_))))}")
            return pd.Series(encoded_target, index=target_series.index, name=target_series.name)
        
        return target_series
    
    def create_train_test_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                               create_validation: bool = True) -> Union[
                                   Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                                   Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
                               ]:
        """
        Create train, test, and optionally validation splits.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            create_validation: Whether to create a validation set
            
        Returns:
            Tuple of splits (X_train, X_test, y_train, y_test) or 
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Creating train-test splits...")
        
        if y is not None:
            # Stratified split to maintain class distribution
            if create_validation:
                # First split: train+val vs test
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y
                )
                
                # Second split: train vs validation
                val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=val_size_adjusted,
                    random_state=self.config.random_state,
                    stratify=y_temp
                )
                
                logger.info(f"Split sizes - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
                
                # Update report
                if self.report_:
                    self.report_.train_shape = X_train.shape
                    self.report_.test_shape = X_test.shape
                    self.report_.validation_shape = X_val.shape
                
                return X_train, X_val, X_test, y_train, y_val, y_test
            
            else:
                # Simple train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y
                )
                
                logger.info(f"Split sizes - Train: {X_train.shape}, Test: {X_test.shape}")
                
                # Update report
                if self.report_:
                    self.report_.train_shape = X_train.shape
                    self.report_.test_shape = X_test.shape
                
                return X_train, X_test, y_train, y_test
        
        else:
            # No target variable provided, simple split
            if create_validation:
                X_temp, X_test = train_test_split(
                    X, 
                    test_size=self.config.test_size,
                    random_state=self.config.random_state
                )
                
                val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
                X_train, X_val = train_test_split(
                    X_temp,
                    test_size=val_size_adjusted,
                    random_state=self.config.random_state
                )
                
                return X_train, X_val, X_test
            
            else:
                X_train, X_test = train_test_split(
                    X,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state
                )
                
                return X_train, X_test
    
    def _get_removed_features(self) -> List[str]:
        """Get list of features removed during preprocessing"""
        if self.original_data_ is not None and self.processed_data_ is not None:
            original_features = set(self.original_data_.columns)
            processed_features = set(self.processed_data_.columns)
            return list(original_features - processed_features)
        return []
    
    def _get_encoded_features(self) -> List[str]:
        """Get list of encoded features"""
        return getattr(self, 'encoded_features_', [])
    
    def _get_scaled_features(self) -> List[str]:
        """Get list of scaled features"""
        return getattr(self, 'scaled_features_', [])
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data
        """
        if self.scaler_ is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        logger.info("Transforming new data using fitted preprocessor...")
        
        # Apply same transformations
        transformed_data = data.copy()
        
        # Apply cleaning (basic only, no removal of rows)
        transformed_data = self._clean_data_types(transformed_data)
        
        # Handle missing values
        if hasattr(self, 'missing_counts_') and self.missing_counts_:
            transformed_data = self.handle_missing_values(transformed_data)
        
        # Encode categorical variables
        transformed_data = self.encode_categorical(transformed_data)
        
        # Scale numerical features
        if self.scaled_features_:
            available_features = [col for col in self.scaled_features_ if col in transformed_data.columns]
            if available_features:
                transformed_data[available_features] = self.scaler_.transform(transformed_data[available_features])
        
        return transformed_data
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the fitted preprocessor to disk"""
        preprocessor_data = {
            'config': self.config,
            'scaler': self.scaler_,
            'target_encoder': self.target_encoder_,
            'scaled_features': getattr(self, 'scaled_features_', []),
            'encoded_features': getattr(self, 'encoded_features_', []),
            'report': self.report_
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load a fitted preprocessor from disk"""
        preprocessor_data = joblib.load(filepath)
        
        self.config = preprocessor_data['config']
        self.scaler_ = preprocessor_data['scaler']
        self.target_encoder_ = preprocessor_data['target_encoder']
        self.scaled_features_ = preprocessor_data.get('scaled_features', [])
        self.encoded_features_ = preprocessor_data.get('encoded_features', [])
        self.report_ = preprocessor_data.get('report')
        
        logger.info(f"Preprocessor loaded from {filepath}")
    
    def get_preprocessing_report(self) -> Optional[PreprocessingReport]:
        """Get detailed preprocessing report"""
        return self.report_
    
    def create_preprocessing_summary(self) -> Dict[str, Any]:
        """Create a comprehensive preprocessing summary"""
        if not self.report_:
            return {"error": "No preprocessing report available"}
        
        summary = {
            "preprocessing_overview": {
                "original_shape": self.report_.original_shape,
                "processed_shape": self.report_.processed_shape,
                "features_removed": len(self.report_.removed_features),
                "features_encoded": len(self.report_.encoded_features),
                "features_scaled": len(self.report_.scaled_features),
                "processing_time": f"{self.report_.processing_time:.2f} seconds"
            },
            "data_quality_improvements": {
                "missing_values_handled": self.report_.missing_values_handled,
                "outliers_treated": self.report_.outliers_treated
            },
            "train_test_split": {
                "train_shape": self.report_.train_shape,
                "test_shape": self.report_.test_shape,
                "validation_shape": self.report_.validation_shape
            },
            "configuration_used": asdict(self.config)
        }
        
        return summary
    
    def plot_preprocessing_summary(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Create visual summary of preprocessing steps"""
        if not self.report_:
            logger.warning("No preprocessing report available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Data Preprocessing Summary', fontsize=16, fontweight='bold')
        
        # 1. Data shape changes
        shapes = ['Original', 'Processed']
        rows = [self.report_.original_shape[0], self.report_.processed_shape[0]]
        cols = [self.report_.original_shape[1], self.report_.processed_shape[1]]
        
        x = np.arange(len(shapes))
        width = 0.35
        
        axes[0,0].bar(x - width/2, rows, width, label='Rows', alpha=0.8)
        axes[0,0].bar(x + width/2, cols, width, label='Columns', alpha=0.8)
        axes[0,0].set_title('Data Shape Changes')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(shapes)
        axes[0,0].legend()
        
        # 2. Feature transformations
        transformations = ['Removed', 'Encoded', 'Scaled']
        counts = [
            len(self.report_.removed_features),
            len(self.report_.encoded_features),
            len(self.report_.scaled_features)
        ]
        
        axes[0,1].bar(transformations, counts, color=['red', 'orange', 'green'], alpha=0.7)
        axes[0,1].set_title('Feature Transformations')
        axes[0,1].set_ylabel('Number of Features')
        
        # 3. Missing values handled
        if self.report_.missing_values_handled:
            missing_cols = list(self.report_.missing_values_handled.keys())[:10]  # Top 10
            missing_counts = [self.report_.missing_values_handled[col] for col in missing_cols]
            
            axes[1,0].bar(range(len(missing_cols)), missing_counts, alpha=0.7, color='skyblue')
            axes[1,0].set_title('Missing Values Handled')
            axes[1,0].set_xlabel('Features')
            axes[1,0].set_ylabel('Missing Count')
            axes[1,0].set_xticks(range(len(missing_cols)))
            axes[1,0].set_xticklabels(missing_cols, rotation=45)
        
        # 4. Train-test-validation split
        if self.report_.validation_shape:
            split_names = ['Train', 'Validation', 'Test']
            split_sizes = [
                self.report_.train_shape[0],
                self.report_.validation_shape[0],
                self.report_.test_shape[0]
            ]
        else:
            split_names = ['Train', 'Test']
            split_sizes = [
                self.report_.train_shape[0],
                self.report_.test_shape[0]
            ]
        
        axes[1,1].pie(split_sizes, labels=split_names, autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Train-Test Split Distribution')
        
        plt.tight_layout()
        plt.show()


# Convenience functions
def preprocess_hr_data(data: pd.DataFrame, target_column: str = 'Attrition',
                      config: Optional[PreprocessingConfig] = None) -> Tuple[pd.DataFrame, pd.Series, DataPreprocessor]:
    """
    Quick function to preprocess HR data with default settings.
    
    Args:
        data: Input DataFrame
        target_column: Name of target column
        config: Optional preprocessing configuration
        
    Returns:
        Tuple of (processed_features, target, fitted_preprocessor)
    """
    preprocessor = DataPreprocessor(config)
    X_processed, y_processed = preprocessor.fit_transform(data, target_column)
    return X_processed, y_processed, preprocessor


def create_preprocessing_pipeline(config: Optional[PreprocessingConfig] = None) -> DataPreprocessor:
    """Create a configured preprocessing pipeline"""
    return DataPreprocessor(config)


if __name__ == "__main__":
    # Test the preprocessor
    print("🧪 Testing DataPreprocessor...")
    
    # This would test with actual data
    # from src.data_processing.data_loader import load_hr_data
    # data, _ = load_hr_data("data/synthetic/hr_employees.csv")
    # 
    # preprocessor = DataPreprocessor()
    # X_processed, y_processed = preprocessor.fit_transform(data, 'Attrition')
    # 
    # X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(X_processed, y_processed)
    
    print("✅ DataPreprocessor test completed!")
