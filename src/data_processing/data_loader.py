"""
HR Attrition Predictor - Data Loading & Validation System
========================================================
Comprehensive data loading with quality validation, outlier detection,
and data integrity checks for the HR attrition prediction pipeline.

Author: Mohd Faraz
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from datetime import datetime, timedelta
import json
import pickle
import yaml
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class DataQualityReport:
    """Data class for comprehensive data quality assessment"""
    total_records: int
    total_features: int
    missing_values_count: int
    missing_percentage: float
    duplicate_records: int
    outlier_records: int
    data_types: Dict[str, int]
    memory_usage_mb: float
    quality_score: float
    issues_found: List[str]
    recommendations: List[str]
    validation_timestamp: str


@dataclass
class OutlierAnalysis:
    """Data class for outlier detection results"""
    feature_name: str
    outlier_method: str
    outlier_count: int
    outlier_percentage: float
    outlier_indices: List[int]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    outlier_values: List[float]


class DataLoader:
    """
    Comprehensive data loading and validation system for HR datasets.
    
    Supports multiple file formats, performs extensive quality checks,
    detects outliers, and provides detailed validation reports for
    enterprise-grade data pipeline operations.
    """
    
    def __init__(self, data_path: Union[str, Path], config_path: Optional[str] = None):
        """
        Initialize the DataLoader with data path and optional configuration.
        
        Args:
            data_path: Path to the data file or directory
            config_path: Optional path to configuration file
        """
        self.data_path = Path(data_path)
        self.config_path = config_path
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.quality_report: Optional[DataQualityReport] = None
        self.outlier_analyses: Dict[str, OutlierAnalysis] = {}
        
        # Load configuration if provided
        self.config = self._load_config() if config_path else {}
        
        # Define expected schema for HR data
        self._define_expected_schema()
        
        logger.info(f"DataLoader initialized with path: {self.data_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
    
    def _define_expected_schema(self) -> None:
        """Define expected data schema for validation"""
        self.expected_schema = {
            # Personal Information
            'EmployeeID': {'type': 'object', 'nullable': False, 'unique': True},
            'FirstName': {'type': 'object', 'nullable': False},
            'LastName': {'type': 'object', 'nullable': False},
            'Age': {'type': 'int64', 'range': (18, 70), 'nullable': False},
            'Gender': {'type': 'object', 'values': ['Male', 'Female', 'Non-Binary']},
            'MaritalStatus': {'type': 'object', 'values': ['Single', 'Married', 'Divorced']},
            'Education': {'type': 'object', 'nullable': False},
            
            # Professional Information
            'Department': {'type': 'object', 'nullable': False},
            'JobRole': {'type': 'object', 'nullable': False},
            'JobLevel': {'type': 'int64', 'range': (1, 5)},
            'MonthlyIncome': {'type': 'int64', 'range': (1000, 50000)},
            'YearsAtCompany': {'type': 'int64', 'range': (0, 50)},
            'YearsInCurrentRole': {'type': 'int64', 'range': (0, 50)},
            
            # Performance Metrics
            'PerformanceScore': {'type': 'int64', 'range': (1, 5)},
            'JobSatisfaction': {'type': 'int64', 'range': (1, 4)},
            'WorkLifeBalance': {'type': 'int64', 'range': (1, 4)},
            'EnvironmentSatisfaction': {'type': 'int64', 'range': (1, 4)},
            
            # Target Variable
            'Attrition': {'type': 'object', 'values': ['Yes', 'No'], 'nullable': False},
            'AttritionProbability': {'type': 'float64', 'range': (0.0, 1.0)},
            'RiskLevel': {'type': 'object', 'values': ['Low', 'Medium', 'High']}
        }
    
    def load_raw_data(self, file_format: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw data from various file formats with error handling.
        
        Args:
            file_format: Specific file format ('csv', 'excel', 'pickle', 'json')
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If unsupported file format
        """
        logger.info(f"Loading data from: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Auto-detect file format if not specified
        if file_format is None:
            file_format = self._detect_file_format()
        
        try:
            # Load data based on format
            if file_format == 'csv':
                self.raw_data = self._load_csv()
            elif file_format == 'excel':
                self.raw_data = self._load_excel()
            elif file_format == 'pickle':
                self.raw_data = self._load_pickle()
            elif file_format == 'json':
                self.raw_data = self._load_json()
            elif file_format == 'parquet':
                self.raw_data = self._load_parquet()
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Log loading success
            logger.info(f"Successfully loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
            logger.info(f"Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Basic data info
            self._log_basic_info()
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _detect_file_format(self) -> str:
        """Auto-detect file format from extension"""
        extension = self.data_path.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel', 
            '.xls': 'excel',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.json': 'json',
            '.parquet': 'parquet'
        }
        
        if extension in format_map:
            detected_format = format_map[extension]
            logger.info(f"Auto-detected file format: {detected_format}")
            return detected_format
        else:
            logger.warning(f"Unknown file extension: {extension}, defaulting to CSV")
            return 'csv'
    
    def _load_csv(self) -> pd.DataFrame:
        """Load CSV file with optimized parameters"""
        try:
            # Try different encodings if needed
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        self.data_path,
                        encoding=encoding,
                        low_memory=False,
                        parse_dates=True,
                        infer_datetime_format=True
                    )
                    logger.info(f"CSV loaded successfully with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            df = pd.read_csv(self.data_path, encoding='utf-8', errors='replace')
            logger.warning("Loaded CSV with error replacement for encoding issues")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def _load_excel(self) -> pd.DataFrame:
        """Load Excel file"""
        try:
            # Try to load first sheet or specific sheet if configured
            sheet_name = self.config.get('excel_sheet', 0)
            df = pd.read_excel(self.data_path, sheet_name=sheet_name, engine='openpyxl')
            logger.info(f"Excel file loaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise
    
    def _load_pickle(self) -> pd.DataFrame:
        """Load pickle file"""
        try:
            df = pd.read_pickle(self.data_path)
            logger.info("Pickle file loaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error loading pickle: {e}")
            raise
    
    def _load_json(self) -> pd.DataFrame:
        """Load JSON file"""
        try:
            df = pd.read_json(self.data_path, orient='records')
            logger.info("JSON file loaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def _load_parquet(self) -> pd.DataFrame:
        """Load Parquet file"""
        try:
            df = pd.read_parquet(self.data_path)
            logger.info("Parquet file loaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error loading Parquet: {e}")
            raise
    
    def _log_basic_info(self) -> None:
        """Log basic information about the loaded dataset"""
        if self.raw_data is not None:
            logger.info(f"Dataset shape: {self.raw_data.shape}")
            logger.info(f"Columns: {list(self.raw_data.columns)[:10]}{'...' if len(self.raw_data.columns) > 10 else ''}")
            logger.info(f"Data types: {dict(self.raw_data.dtypes.value_counts())}")
    
    def validate_data_quality(self, perform_deep_analysis: bool = True) -> DataQualityReport:
        """
        Perform comprehensive data quality validation.
        
        Args:
            perform_deep_analysis: Whether to perform expensive quality checks
            
        Returns:
            DataQualityReport with detailed quality assessment
        """
        logger.info("Starting comprehensive data quality validation...")
        
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        start_time = datetime.now()
        issues_found = []
        recommendations = []
        
        # Basic metrics
        total_records = len(self.raw_data)
        total_features = len(self.raw_data.columns)
        
        # 1. Missing value analysis
        missing_analysis = self._analyze_missing_values()
        missing_count = missing_analysis['total_missing']
        missing_percentage = (missing_count / (total_records * total_features)) * 100
        
        # 2. Duplicate analysis
        duplicate_records = self.raw_data.duplicated().sum()
        if duplicate_records > 0:
            issues_found.append(f"Found {duplicate_records} duplicate records")
            recommendations.append("Remove duplicate records before modeling")
        
        # 3. Data type analysis
        data_types = dict(self.raw_data.dtypes.value_counts())
        
        # 4. Memory usage
        memory_usage_mb = self.raw_data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # 5. Schema validation
        schema_issues = self._validate_schema()
        issues_found.extend(schema_issues)
        
        # 6. Outlier detection (if deep analysis enabled)
        outlier_count = 0
        if perform_deep_analysis:
            outlier_count = self._detect_all_outliers()
        
        # 7. Business rule validation
        business_issues = self._validate_business_rules()
        issues_found.extend(business_issues)
        
        # 8. Calculate quality score (0-100)
        quality_score = self._calculate_quality_score(
            missing_percentage, duplicate_records, len(schema_issues), 
            len(business_issues), outlier_count
        )
        
        # 9. Generate recommendations
        if missing_percentage > 5:
            recommendations.append("High missing value percentage - consider imputation strategies")
        if quality_score < 80:
            recommendations.append("Data quality below threshold - review and clean before modeling")
        if outlier_count > total_records * 0.05:
            recommendations.append("High outlier percentage - review outlier treatment strategies")
        
        # Create quality report
        self.quality_report = DataQualityReport(
            total_records=total_records,
            total_features=total_features,
            missing_values_count=missing_count,
            missing_percentage=missing_percentage,
            duplicate_records=duplicate_records,
            outlier_records=outlier_count,
            data_types=data_types,
            memory_usage_mb=memory_usage_mb,
            quality_score=quality_score,
            issues_found=issues_found,
            recommendations=recommendations,
            validation_timestamp=datetime.now().isoformat()
        )
        
        # Log results
        validation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Data quality validation completed in {validation_time:.2f} seconds")
        logger.info(f"Quality Score: {quality_score:.1f}/100")
        
        if issues_found:
            logger.warning(f"Found {len(issues_found)} data quality issues")
            for issue in issues_found[:5]:  # Show first 5 issues
                logger.warning(f"  - {issue}")
        
        return self.quality_report
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values in detail"""
        missing_counts = self.raw_data.isnull().sum()
        missing_percentages = (missing_counts / len(self.raw_data)) * 100
        
        missing_summary = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': (missing_counts > 0).sum(),
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
        }
        
        return missing_summary
    
    def _validate_schema(self) -> List[str]:
        """Validate data against expected schema"""
        issues = []
        
        for column, schema in self.expected_schema.items():
            if column in self.raw_data.columns:
                # Check data type
                expected_type = schema.get('type')
                actual_type = str(self.raw_data[column].dtype)
                
                # Flexible type checking
                if expected_type == 'object' and actual_type not in ['object', 'string']:
                    issues.append(f"Column '{column}' expected {expected_type}, got {actual_type}")
                elif expected_type in ['int64', 'float64'] and not pd.api.types.is_numeric_dtype(self.raw_data[column]):
                    issues.append(f"Column '{column}' expected numeric, got {actual_type}")
                
                # Check value ranges
                if 'range' in schema and pd.api.types.is_numeric_dtype(self.raw_data[column]):
                    min_val, max_val = schema['range']
                    actual_min = self.raw_data[column].min()
                    actual_max = self.raw_data[column].max()
                    
                    if actual_min < min_val or actual_max > max_val:
                        issues.append(f"Column '{column}' values outside expected range {schema['range']}")
                
                # Check allowed values
                if 'values' in schema:
                    invalid_values = set(self.raw_data[column].unique()) - set(schema['values'])
                    if invalid_values:
                        issues.append(f"Column '{column}' contains invalid values: {invalid_values}")
                
                # Check uniqueness
                if schema.get('unique', False):
                    if self.raw_data[column].duplicated().any():
                        issues.append(f"Column '{column}' should be unique but contains duplicates")
            
            else:
                # Column missing
                if not schema.get('optional', False):
                    issues.append(f"Required column '{column}' is missing")
        
        return issues
    
    def _validate_business_rules(self) -> List[str]:
        """Validate business logic rules"""
        issues = []
        
        # Rule 1: Years in current role should not exceed years at company
        if 'YearsInCurrentRole' in self.raw_data.columns and 'YearsAtCompany' in self.raw_data.columns:
            invalid_tenure = self.raw_data['YearsInCurrentRole'] > self.raw_data['YearsAtCompany']
            if invalid_tenure.any():
                issues.append(f"Found {invalid_tenure.sum()} records where YearsInCurrentRole > YearsAtCompany")
        
        # Rule 2: Salary should be reasonable for job level
        if all(col in self.raw_data.columns for col in ['MonthlyIncome', 'JobLevel']):
            # Very basic salary validation - could be enhanced
            min_salaries = {1: 2000, 2: 3000, 3: 4000, 4: 6000, 5: 8000}
            for level, min_salary in min_salaries.items():
                level_data = self.raw_data[self.raw_data['JobLevel'] == level]
                if len(level_data) > 0:
                    low_salary_count = (level_data['MonthlyIncome'] < min_salary).sum()
                    if low_salary_count > 0:
                        issues.append(f"Found {low_salary_count} Level {level} employees with unusually low salaries")
        
        # Rule 3: Age should be consistent with experience
        if all(col in self.raw_data.columns for col in ['Age', 'TotalWorkingYears']):
            # Assume people start working at 18-22
            max_possible_experience = self.raw_data['Age'] - 18
            invalid_experience = self.raw_data['TotalWorkingYears'] > max_possible_experience
            if invalid_experience.any():
                issues.append(f"Found {invalid_experience.sum()} records with impossible experience vs age")
        
        # Rule 4: Attrition consistency
        if 'Attrition' in self.raw_data.columns:
            attrition_rate = (self.raw_data['Attrition'] == 'Yes').mean()
            if attrition_rate < 0.05 or attrition_rate > 0.50:
                issues.append(f"Attrition rate {attrition_rate:.1%} is outside normal range (5-50%)")
        
        return issues
    
    def _calculate_quality_score(self, missing_pct: float, duplicates: int, 
                                schema_issues: int, business_issues: int, 
                                outliers: int) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct points for issues
        score -= min(missing_pct * 2, 30)  # Max 30 points for missing values
        score -= min(duplicates / len(self.raw_data) * 100, 20)  # Max 20 points for duplicates
        score -= min(schema_issues * 5, 20)  # Max 20 points for schema issues
        score -= min(business_issues * 5, 15)  # Max 15 points for business rule violations
        score -= min(outliers / len(self.raw_data) * 100, 15)  # Max 15 points for outliers
        
        return max(0, score)
    
    def check_missing_values(self, visualization: bool = False) -> Dict[str, Any]:
        """
        Comprehensive missing value analysis with optional visualization.
        
        Args:
            visualization: Whether to create missing value plots
            
        Returns:
            Dictionary with missing value analysis results
        """
        logger.info("Performing detailed missing value analysis...")
        
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        # Calculate missing values
        missing_counts = self.raw_data.isnull().sum()
        missing_percentages = (missing_counts / len(self.raw_data)) * 100
        
        # Identify patterns
        columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
        complete_rows = self.raw_data.dropna().shape[0]
        complete_percentage = (complete_rows / len(self.raw_data)) * 100
        
        # Missing value patterns
        missing_patterns = {}
        if columns_with_missing:
            # Analyze combinations of missing values
            missing_combinations = self.raw_data[columns_with_missing].isnull().value_counts()
            missing_patterns = missing_combinations.head(10).to_dict()
        
        # Create result dictionary
        missing_analysis = {
            'total_missing': missing_counts.sum(),
            'missing_percentage': (missing_counts.sum() / self.raw_data.size) * 100,
            'columns_with_missing': len(columns_with_missing),
            'complete_rows': complete_rows,
            'complete_percentage': complete_percentage,
            'missing_by_column': {
                col: {'count': int(missing_counts[col]), 
                     'percentage': round(missing_percentages[col], 2)}
                for col in columns_with_missing
            },
            'missing_patterns': missing_patterns
        }
        
        # Log results
        logger.info(f"Missing value analysis complete:")
        logger.info(f"  Total missing values: {missing_analysis['total_missing']:,}")
        logger.info(f"  Overall missing percentage: {missing_analysis['missing_percentage']:.2f}%")
        logger.info(f"  Columns with missing values: {missing_analysis['columns_with_missing']}")
        logger.info(f"  Complete rows: {complete_rows:,} ({complete_percentage:.1f}%)")
        
        # Create visualization if requested
        if visualization and columns_with_missing:
            self._create_missing_value_plots(missing_counts, missing_percentages)
        
        return missing_analysis
    
    def _create_missing_value_plots(self, missing_counts: pd.Series, 
                                   missing_percentages: pd.Series) -> None:
        """Create missing value visualization plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Missing Value Analysis', fontsize=16, fontweight='bold')
            
            # 1. Missing count by column
            missing_nonzero = missing_counts[missing_counts > 0]
            if len(missing_nonzero) > 0:
                missing_nonzero.plot(kind='bar', ax=axes[0,0], color='lightcoral')
                axes[0,0].set_title('Missing Values by Column')
                axes[0,0].set_ylabel('Count')
                axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. Missing percentage by column
            missing_pct_nonzero = missing_percentages[missing_percentages > 0]
            if len(missing_pct_nonzero) > 0:
                missing_pct_nonzero.plot(kind='bar', ax=axes[0,1], color='orange')
                axes[0,1].set_title('Missing Percentage by Column')
                axes[0,1].set_ylabel('Percentage (%)')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. Missing value heatmap
            if len(missing_nonzero) > 0:
                missing_cols = missing_nonzero.index[:20]  # Top 20 columns
                sample_data = self.raw_data[missing_cols].head(100)  # First 100 rows
                sns.heatmap(sample_data.isnull(), cbar=True, ax=axes[1,0], 
                           cmap='viridis', yticklabels=False)
                axes[1,0].set_title('Missing Value Pattern (Sample)')
            
            # 4. Complete vs incomplete records
            complete_data = [
                len(self.raw_data.dropna()),
                len(self.raw_data) - len(self.raw_data.dropna())
            ]
            axes[1,1].pie(complete_data, labels=['Complete', 'Incomplete'], 
                         autopct='%1.1f%%', startangle=90)
            axes[1,1].set_title('Complete vs Incomplete Records')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.warning(f"Could not create missing value plots: {e}")
    
    def detect_outliers(self, method: str = 'iqr', columns: Optional[List[str]] = None,
                       contamination: float = 0.1) -> Dict[str, OutlierAnalysis]:
        """
        Detect outliers using multiple methods.
        
        Args:
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            columns: Specific columns to analyze (None for all numeric)
            contamination: Expected outlier proportion for isolation forest
            
        Returns:
            Dictionary mapping column names to OutlierAnalysis objects
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        # Select numeric columns
        if columns is None:
            numeric_columns = self.raw_data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [col for col in columns if col in self.raw_data.columns]
        
        outlier_results = {}
        
        for column in numeric_columns:
            try:
                if method == 'iqr':
                    analysis = self._detect_outliers_iqr(column)
                elif method == 'zscore':
                    analysis = self._detect_outliers_zscore(column)
                elif method == 'isolation_forest':
                    analysis = self._detect_outliers_isolation_forest(column, contamination)
                else:
                    logger.warning(f"Unknown outlier method: {method}")
                    continue
                
                outlier_results[column] = analysis
                
                if analysis.outlier_count > 0:
                    logger.info(f"  {column}: Found {analysis.outlier_count} outliers "
                              f"({analysis.outlier_percentage:.1f}%)")
                
            except Exception as e:
                logger.warning(f"Error detecting outliers in {column}: {e}")
        
        self.outlier_analyses.update(outlier_results)
        return outlier_results
    
    def _detect_outliers_iqr(self, column: str) -> OutlierAnalysis:
        """Detect outliers using Interquartile Range method"""
        data = self.raw_data[column].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_values = data[outlier_mask].tolist()
        
        return OutlierAnalysis(
            feature_name=column,
            outlier_method='IQR',
            outlier_count=len(outlier_indices),
            outlier_percentage=(len(outlier_indices) / len(data)) * 100,
            outlier_indices=outlier_indices,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            outlier_values=outlier_values
        )
    
    def _detect_outliers_zscore(self, column: str, threshold: float = 3) -> OutlierAnalysis:
        """Detect outliers using Z-score method"""
        data = self.raw_data[column].dropna()
        z_scores = np.abs(stats.zscore(data))
        
        outlier_mask = z_scores > threshold
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_values = data[outlier_mask].tolist()
        
        return OutlierAnalysis(
            feature_name=column,
            outlier_method=f'Z-Score (threshold={threshold})',
            outlier_count=len(outlier_indices),
            outlier_percentage=(len(outlier_indices) / len(data)) * 100,
            outlier_indices=outlier_indices,
            lower_bound=None,
            upper_bound=None,
            outlier_values=outlier_values
        )
    
    def _detect_outliers_isolation_forest(self, column: str, 
                                         contamination: float) -> OutlierAnalysis:
        """Detect outliers using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            data = self.raw_data[column].dropna().values.reshape(-1, 1)
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_predictions = iso_forest.fit_predict(data)
            
            outlier_mask = outlier_predictions == -1
            outlier_indices = self.raw_data[column].dropna().index[outlier_mask].tolist()
            outlier_values = data[outlier_mask].flatten().tolist()
            
            return OutlierAnalysis(
                feature_name=column,
                outlier_method=f'Isolation Forest (contamination={contamination})',
                outlier_count=len(outlier_indices),
                outlier_percentage=(len(outlier_indices) / len(data)) * 100,
                outlier_indices=outlier_indices,
                lower_bound=None,
                upper_bound=None,
                outlier_values=outlier_values
            )
        
        except ImportError:
            logger.warning("scikit-learn not available for Isolation Forest")
            # Fallback to IQR method
            return self._detect_outliers_iqr(column)
    
    def _detect_all_outliers(self) -> int:
        """Detect outliers in all numeric columns and return total count"""
        numeric_columns = self.raw_data.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        
        for column in numeric_columns:
            try:
                analysis = self._detect_outliers_iqr(column)
                total_outliers += analysis.outlier_count
            except Exception:
                continue
        
        return total_outliers
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        if self.raw_data is None:
            return {"error": "No data loaded"}
        
        summary = {
            'basic_info': {
                'rows': len(self.raw_data),
                'columns': len(self.raw_data.columns),
                'memory_usage_mb': self.raw_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'file_path': str(self.data_path)
            },
            'data_types': dict(self.raw_data.dtypes.value_counts()),
            'missing_values': dict(self.raw_data.isnull().sum()),
            'numeric_summary': self.raw_data.describe().to_dict() if len(self.raw_data.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {
                col: self.raw_data[col].value_counts().head().to_dict()
                for col in self.raw_data.select_dtypes(include=['object']).columns[:5]
            }
        }
        
        if self.quality_report:
            summary['quality_report'] = asdict(self.quality_report)
        
        return summary
    
    def save_quality_report(self, output_path: Optional[str] = None) -> str:
        """Save quality report to JSON file"""
        if self.quality_report is None:
            raise ValueError("No quality report available. Run validate_data_quality() first.")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/data_quality_report_{timestamp}.json"
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(asdict(self.quality_report), f, indent=2, default=str)
        
        logger.info(f"Quality report saved to: {output_path}")
        return output_path


# Convenience functions for quick usage
def load_hr_data(file_path: str, validate: bool = True) -> Tuple[pd.DataFrame, Optional[DataQualityReport]]:
    """
    Quick function to load and validate HR data.
    
    Args:
        file_path: Path to data file
        validate: Whether to perform quality validation
        
    Returns:
        Tuple of (DataFrame, QualityReport)
    """
    loader = DataLoader(file_path)
    data = loader.load_raw_data()
    
    quality_report = None
    if validate:
        quality_report = loader.validate_data_quality()
    
    return data, quality_report


def quick_data_check(file_path: str) -> None:
    """Quick data quality check with summary output"""
    loader = DataLoader(file_path)
    data = loader.load_raw_data()
    
    print(f"📊 Quick Data Check: {file_path}")
    print(f"   Shape: {data.shape}")
    print(f"   Missing values: {data.isnull().sum().sum()}")
    print(f"   Duplicates: {data.duplicated().sum()}")
    print(f"   Memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Quick outlier check on first numeric column
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        outliers = loader.detect_outliers('iqr', [numeric_cols[0]])
        if numeric_cols[0] in outliers:
            print(f"   Outliers in {numeric_cols[0]}: {outliers[numeric_cols[0]].outlier_count}")


if __name__ == "__main__":
    # Test the data loader
    try:
        # Test with sample data
        print("🧪 Testing DataLoader...")
        
        # This would test with actual data file
        loader = DataLoader("data/synthetic/hr_employees.csv")
        data = loader.load_raw_data()
        quality_report = loader.validate_data_quality()
        missing_analysis = loader.check_missing_values()
        outliers = loader.detect_outliers()
        
        print("✅ DataLoader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
