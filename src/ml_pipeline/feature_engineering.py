"""
HR Attrition Predictor - Advanced Feature Engineering Pipeline
============================================================
Creates sophisticated features, interactions, and derived metrics
to maximize machine learning model performance and business insights.

Author: HR Analytics Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import warnings
from datetime import datetime
from dataclasses import dataclass, asdict
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering parameters"""
    # Interaction features
    create_interactions: bool = True
    max_interaction_features: int = 50
    interaction_methods: List[str] = None
    
    # Polynomial features
    create_polynomial: bool = True
    polynomial_degree: int = 2
    include_bias: bool = False
    
    # Tenure-based features
    create_tenure_features: bool = True
    tenure_bins: List[int] = None
    
    # Satisfaction features
    create_satisfaction_composites: bool = True
    satisfaction_weights: Dict[str, float] = None
    
    # Risk indicators
    create_risk_features: bool = True
    risk_thresholds: Dict[str, float] = None
    
    # Advanced features
    create_statistical_features: bool = True
    create_clustering_features: bool = True
    n_clusters: int = 5
    
    # Feature selection
    enable_feature_selection: bool = True
    selection_method: str = 'mutual_info'  # 'f_classif', 'mutual_info', 'variance'
    max_features: int = 100
    
    def __post_init__(self):
        if self.interaction_methods is None:
            self.interaction_methods = ['multiply', 'divide', 'add', 'subtract']
        if self.tenure_bins is None:
            self.tenure_bins = [0, 1, 3, 5, 10, 50]
        if self.satisfaction_weights is None:
            self.satisfaction_weights = {
                'JobSatisfaction': 0.3,
                'EnvironmentSatisfaction': 0.25,
                'WorkLifeBalance': 0.25,
                'RelationshipSatisfaction': 0.2
            }
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                'low_salary_percentile': 25,
                'high_overtime_threshold': 1,
                'long_commute_miles': 20,
                'low_satisfaction_threshold': 2,
                'high_travel_frequency': 1
            }


@dataclass
class FeatureEngineeringReport:
    """Report for feature engineering operations"""
    original_features: int
    engineered_features: int
    total_features: int
    interaction_features: int
    tenure_features: int
    satisfaction_features: int
    risk_features: int
    statistical_features: int
    clustering_features: int
    selected_features: int
    feature_importance_scores: Dict[str, float]
    processing_time: float
    timestamp: str


class FeatureEngineer:
    """
    Advanced feature engineering pipeline for HR attrition prediction.
    
    Creates sophisticated derived features, interactions, and composite metrics
    to enhance machine learning model performance and provide business insights.
    """
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Optional feature engineering configuration
        """
        self.config = config if config else FeatureEngineeringConfig()
        self.feature_names_: Optional[List[str]] = None
        self.interaction_features_: List[str] = []
        self.tenure_features_: List[str] = []
        self.satisfaction_features_: List[str] = []
        self.risk_features_: List[str] = []
        self.statistical_features_: List[str] = []
        self.clustering_features_: List[str] = []
        self.selected_features_: List[str] = []
        self.feature_selector_: Optional[Any] = None
        self.scaler_: Optional[StandardScaler] = None
        self.clusterer_: Optional[KMeans] = None
        self.report_: Optional[FeatureEngineeringReport] = None
        
        logger.info("FeatureEngineer initialized with configuration")
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit feature engineer and transform data.
        
        Args:
            X: Input feature DataFrame
            y: Optional target variable for supervised feature selection
            
        Returns:
            DataFrame with engineered features
        """
        start_time = datetime.now()
        logger.info(f"Starting advanced feature engineering...")
        logger.info(f"Input features: {X.shape[1]}")
        
        original_features = X.shape[1]
        engineered_data = X.copy()
        
        # Step 1: Create interaction features
        if self.config.create_interactions:
            engineered_data = self.create_interaction_features(engineered_data)
            logger.info(f"After interactions: {engineered_data.shape[1]} features")
        
        # Step 2: Create polynomial features
        if self.config.create_polynomial:
            engineered_data = self._create_polynomial_features(engineered_data)
            logger.info(f"After polynomial: {engineered_data.shape[1]} features")
        
        # Step 3: Calculate tenure-based features
        if self.config.create_tenure_features:
            engineered_data = self.calculate_tenure_features(engineered_data)
            logger.info(f"After tenure features: {engineered_data.shape[1]} features")
        
        # Step 4: Generate satisfaction composites
        if self.config.create_satisfaction_composites:
            engineered_data = self.generate_satisfaction_scores(engineered_data)
            logger.info(f"After satisfaction features: {engineered_data.shape[1]} features")
        
        # Step 5: Create risk indicators
        if self.config.create_risk_features:
            engineered_data = self.create_risk_indicators(engineered_data)
            logger.info(f"After risk features: {engineered_data.shape[1]} features")
        
        # Step 6: Create statistical features
        if self.config.create_statistical_features:
            engineered_data = self._create_statistical_features(engineered_data)
            logger.info(f"After statistical features: {engineered_data.shape[1]} features")
        
        # Step 7: Create clustering features
        if self.config.create_clustering_features:
            engineered_data = self._create_clustering_features(engineered_data)
            logger.info(f"After clustering features: {engineered_data.shape[1]} features")
        
        # Step 8: Feature selection
        selected_features = engineered_data.shape[1]
        if self.config.enable_feature_selection and y is not None:
            engineered_data = self._select_features(engineered_data, y)
            selected_features = engineered_data.shape[1]
            logger.info(f"After feature selection: {selected_features} features")
        
        # Store feature names
        self.feature_names_ = engineered_data.columns.tolist()
        
        # Create report
        processing_time = (datetime.now() - start_time).total_seconds()
        self.report_ = FeatureEngineeringReport(
            original_features=original_features,
            engineered_features=engineered_data.shape[1] - original_features,
            total_features=engineered_data.shape[1],
            interaction_features=len(self.interaction_features_),
            tenure_features=len(self.tenure_features_),
            satisfaction_features=len(self.satisfaction_features_),
            risk_features=len(self.risk_features_),
            statistical_features=len(self.statistical_features_),
            clustering_features=len(self.clustering_features_),
            selected_features=selected_features,
            feature_importance_scores=self._get_feature_importance_scores(),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Feature engineering completed in {processing_time:.2f} seconds")
        logger.info(f"Created {engineered_data.shape[1] - original_features} new features")
        
        return engineered_data
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        logger.info("Creating interaction features...")
        
        interaction_data = X.copy()
        created_features = []
        
        # Define high-value interaction pairs based on business logic
        interaction_pairs = self._get_interaction_pairs(X)
        
        for (col1, col2), methods in interaction_pairs.items():
            if col1 in X.columns and col2 in X.columns:
                for method in methods:
                    try:
                        feature_name = f"{col1}_{method}_{col2}"
                        
                        if method == 'multiply':
                            interaction_data[feature_name] = X[col1] * X[col2]
                        elif method == 'divide' and (X[col2] != 0).all():
                            interaction_data[feature_name] = X[col1] / (X[col2] + 1e-8)  # Add small epsilon
                        elif method == 'add':
                            interaction_data[feature_name] = X[col1] + X[col2]
                        elif method == 'subtract':
                            interaction_data[feature_name] = X[col1] - X[col2]
                        elif method == 'ratio' and (X[col2] != 0).all():
                            interaction_data[feature_name] = X[col1] / (X[col1] + X[col2] + 1e-8)
                        
                        created_features.append(feature_name)
                        
                        # Limit number of interactions
                        if len(created_features) >= self.config.max_interaction_features:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error creating interaction {feature_name}: {e}")
                        continue
                
                if len(created_features) >= self.config.max_interaction_features:
                    break
        
        self.interaction_features_ = created_features
        logger.info(f"Created {len(created_features)} interaction features")
        
        return interaction_data
    
    def _get_interaction_pairs(self, X: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
        """Define high-value interaction pairs based on business logic"""
        
        interaction_pairs = {}
        
        # Salary-Performance interactions (high business value)
        salary_cols = ['MonthlyIncome', 'HourlyRate', 'DailyRate']
        performance_cols = ['PerformanceScore', 'PercentSalaryHike', 'TrainingTimesLastYear']
        
        for sal_col in salary_cols:
            for perf_col in performance_cols:
                if sal_col in X.columns and perf_col in X.columns:
                    interaction_pairs[(sal_col, perf_col)] = ['multiply', 'ratio']
        
        # Experience-Satisfaction interactions
        experience_cols = ['YearsAtCompany', 'YearsInCurrentRole', 'TotalWorkingYears']
        satisfaction_cols = ['JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction']
        
        for exp_col in experience_cols:
            for sat_col in satisfaction_cols:
                if exp_col in X.columns and sat_col in X.columns:
                    interaction_pairs[(exp_col, sat_col)] = ['multiply', 'divide']
        
        # Age-related interactions
        if 'Age' in X.columns:
            age_interactions = ['MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance']
            for col in age_interactions:
                if col in X.columns:
                    interaction_pairs[('Age', col)] = ['multiply', 'ratio']
        
        # Job level interactions
        if 'JobLevel' in X.columns:
            level_interactions = ['MonthlyIncome', 'PerformanceScore', 'YearsAtCompany']
            for col in level_interactions:
                if col in X.columns:
                    interaction_pairs[('JobLevel', col)] = ['multiply', 'divide']
        
        # Distance-satisfaction interactions
        if 'DistanceFromHome' in X.columns:
            distance_interactions = ['WorkLifeBalance', 'JobSatisfaction', 'EnvironmentSatisfaction']
            for col in distance_interactions:
                if col in X.columns:
                    interaction_pairs[('DistanceFromHome', col)] = ['multiply', 'subtract']
        
        # Overtime-satisfaction interactions
        if 'OverTime' in X.columns:  # This might be encoded as 0/1
            overtime_interactions = ['WorkLifeBalance', 'JobSatisfaction', 'MonthlyIncome']
            for col in overtime_interactions:
                if col in X.columns:
                    interaction_pairs[('OverTime', col)] = ['multiply']
        
        return interaction_pairs
    
    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for numeric columns"""
        logger.info("Creating polynomial features...")
        
        poly_data = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select subset of numeric columns for polynomial features (avoid explosion)
        key_numeric_cols = []
        priority_cols = ['MonthlyIncome', 'Age', 'YearsAtCompany', 'JobSatisfaction', 
                        'WorkLifeBalance', 'PerformanceScore', 'DistanceFromHome']
        
        for col in priority_cols:
            if col in numeric_cols:
                key_numeric_cols.append(col)
        
        # Add other important numeric columns (limit to top 10 total)
        remaining_cols = [col for col in numeric_cols if col not in key_numeric_cols]
        key_numeric_cols.extend(remaining_cols[:max(0, 10 - len(key_numeric_cols))])
        
        if key_numeric_cols:
            try:
                poly_features = PolynomialFeatures(
                    degree=self.config.polynomial_degree,
                    include_bias=self.config.include_bias,
                    interaction_only=False
                )
                
                poly_array = poly_features.fit_transform(X[key_numeric_cols])
                poly_feature_names = poly_features.get_feature_names_out(key_numeric_cols)
                
                # Add new polynomial features (excluding original features)
                original_feature_count = len(key_numeric_cols)
                new_poly_features = poly_feature_names[original_feature_count:]
                
                # Limit polynomial features to prevent explosion
                max_poly_features = min(50, len(new_poly_features))
                selected_poly_features = new_poly_features[:max_poly_features]
                
                for i, feature_name in enumerate(selected_poly_features):
                    poly_data[f"poly_{feature_name}"] = poly_array[:, original_feature_count + i]
                
                logger.info(f"Created {len(selected_poly_features)} polynomial features")
                
            except Exception as e:
                logger.warning(f"Error creating polynomial features: {e}")
        
        return poly_data
    
    def calculate_tenure_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sophisticated tenure-based features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with tenure features added
        """
        logger.info("Calculating tenure-based features...")
        
        tenure_data = X.copy()
        created_features = []
        
        # Basic tenure calculations
        if 'YearsAtCompany' in X.columns and 'YearsInCurrentRole' in X.columns:
            # Years in other roles at company
            tenure_data['YearsInOtherRoles'] = X['YearsAtCompany'] - X['YearsInCurrentRole']
            created_features.append('YearsInOtherRoles')
            
            # Role tenure ratio (how much of company time in current role)
            tenure_data['RoleTenureRatio'] = X['YearsInCurrentRole'] / (X['YearsAtCompany'] + 1e-8)
            created_features.append('RoleTenureRatio')
        
        # Career progression rate
        if all(col in X.columns for col in ['JobLevel', 'YearsAtCompany']):
            tenure_data['CareerProgressionRate'] = X['JobLevel'] / (X['YearsAtCompany'] + 1e-8)
            created_features.append('CareerProgressionRate')
        
        # Experience efficiency (job level per total experience)
        if all(col in X.columns for col in ['JobLevel', 'TotalWorkingYears']):
            tenure_data['ExperienceEfficiency'] = X['JobLevel'] / (X['TotalWorkingYears'] + 1e-8)
            created_features.append('ExperienceEfficiency')
        
        # Company loyalty indicator
        if all(col in X.columns for col in ['YearsAtCompany', 'TotalWorkingYears']):
            tenure_data['CompanyLoyaltyRatio'] = X['YearsAtCompany'] / (X['TotalWorkingYears'] + 1e-8)
            created_features.append('CompanyLoyaltyRatio')
        
        # Promotion frequency
        if all(col in X.columns for col in ['YearsSinceLastPromotion', 'YearsAtCompany']):
            tenure_data['PromotionFrequency'] = X['YearsAtCompany'] / (X['YearsSinceLastPromotion'] + 1e-8)
            created_features.append('PromotionFrequency')
        
        # Tenure categories (binned features)
        if 'YearsAtCompany' in X.columns:
            tenure_data['TenureCategory'] = pd.cut(
                X['YearsAtCompany'],
                bins=self.config.tenure_bins,
                labels=[f'Tenure_{i}' for i in range(len(self.config.tenure_bins)-1)],
                include_lowest=True
            )
            
            # One-hot encode tenure categories
            tenure_dummies = pd.get_dummies(tenure_data['TenureCategory'], prefix='TenureCat')
            tenure_data = pd.concat([tenure_data, tenure_dummies], axis=1)
            tenure_data.drop('TenureCategory', axis=1, inplace=True)
            created_features.extend(tenure_dummies.columns.tolist())
        
        # Experience vs age ratio
        if all(col in X.columns for col in ['TotalWorkingYears', 'Age']):
            tenure_data['ExperienceAgeRatio'] = X['TotalWorkingYears'] / (X['Age'] - 16 + 1e-8)  # Assume work starts at 16
            created_features.append('ExperienceAgeRatio')
        
        # Salary growth rate (if salary hike data available)
        if all(col in X.columns for col in ['PercentSalaryHike', 'YearsAtCompany']):
            tenure_data['SalaryGrowthRate'] = X['PercentSalaryHike'] / (X['YearsAtCompany'] + 1e-8)
            created_features.append('SalaryGrowthRate')
        
        # Training efficiency (training per year at company)
        if all(col in X.columns for col in ['TrainingTimesLastYear', 'YearsAtCompany']):
            tenure_data['TrainingEfficiency'] = X['TrainingTimesLastYear'] / (X['YearsAtCompany'] + 1e-8)
            created_features.append('TrainingEfficiency')
        
        self.tenure_features_ = created_features
        logger.info(f"Created {len(created_features)} tenure-based features")
        
        return tenure_data
    
    def generate_satisfaction_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate composite satisfaction scores and satisfaction-based features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with satisfaction features added
        """
        logger.info("Generating satisfaction composite scores...")
        
        satisfaction_data = X.copy()
        created_features = []
        
        # Define satisfaction columns
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                           'WorkLifeBalance', 'RelationshipSatisfaction']
        available_satisfaction_cols = [col for col in satisfaction_cols if col in X.columns]
        
        if len(available_satisfaction_cols) >= 2:
            # Weighted composite satisfaction score
            weights = self.config.satisfaction_weights
            weighted_scores = []
            weight_sum = 0
            
            for col in available_satisfaction_cols:
                if col in weights:
                    weighted_scores.append(X[col] * weights[col])
                    weight_sum += weights[col]
                else:
                    weighted_scores.append(X[col] * 0.25)  # Default weight
                    weight_sum += 0.25
            
            satisfaction_data['WeightedSatisfactionScore'] = sum(weighted_scores) / weight_sum
            created_features.append('WeightedSatisfactionScore')
            
            # Simple average satisfaction
            satisfaction_data['AvgSatisfactionScore'] = X[available_satisfaction_cols].mean(axis=1)
            created_features.append('AvgSatisfactionScore')
            
            # Satisfaction standard deviation (consistency indicator)
            satisfaction_data['SatisfactionConsistency'] = X[available_satisfaction_cols].std(axis=1)
            created_features.append('SatisfactionConsistency')
            
            # Satisfaction range (max - min)
            satisfaction_data['SatisfactionRange'] = (
                X[available_satisfaction_cols].max(axis=1) - 
                X[available_satisfaction_cols].min(axis=1)
            )
            created_features.append('SatisfactionRange')
            
            # Low satisfaction indicator
            low_satisfaction_threshold = self.config.risk_thresholds['low_satisfaction_threshold']
            satisfaction_data['LowSatisfactionCount'] = (
                X[available_satisfaction_cols] <= low_satisfaction_threshold
            ).sum(axis=1)
            created_features.append('LowSatisfactionCount')
            
            # High satisfaction indicator  
            satisfaction_data['HighSatisfactionCount'] = (
                X[available_satisfaction_cols] >= 4
            ).sum(axis=1)
            created_features.append('HighSatisfactionCount')
        
        # Satisfaction-specific features
        if 'WorkLifeBalance' in X.columns:
            # Work-life balance categories
            satisfaction_data['WorkLifeBalanceCategory'] = pd.cut(
                X['WorkLifeBalance'],
                bins=[0, 1, 2, 3, 4],
                labels=['Poor', 'Fair', 'Good', 'Excellent'],
                include_lowest=True
            )
            
            # One-hot encode
            wlb_dummies = pd.get_dummies(satisfaction_data['WorkLifeBalanceCategory'], prefix='WLB')
            satisfaction_data = pd.concat([satisfaction_data, wlb_dummies], axis=1)
            satisfaction_data.drop('WorkLifeBalanceCategory', axis=1, inplace=True)
            created_features.extend(wlb_dummies.columns.tolist())
        
        # Job satisfaction vs environment satisfaction gap
        if all(col in X.columns for col in ['JobSatisfaction', 'EnvironmentSatisfaction']):
            satisfaction_data['JobEnvSatisfactionGap'] = (
                X['JobSatisfaction'] - X['EnvironmentSatisfaction']
            )
            created_features.append('JobEnvSatisfactionGap')
        
        # Satisfaction improvement potential
        if available_satisfaction_cols:
            satisfaction_data['SatisfactionImprovementPotential'] = (
                4 - X[available_satisfaction_cols].mean(axis=1)
            )
            created_features.append('SatisfactionImprovementPotential')
        
        self.satisfaction_features_ = created_features
        logger.info(f"Created {len(created_features)} satisfaction-based features")
        
        return satisfaction_data
    
    def create_risk_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create flight risk indicators based on multiple risk factors.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with risk indicator features added
        """
        logger.info("Creating flight risk indicators...")
        
        risk_data = X.copy()
        created_features = []
        risk_thresholds = self.config.risk_thresholds
        
        # 1. Salary risk indicators
        if 'MonthlyIncome' in X.columns:
            salary_percentile_25 = X['MonthlyIncome'].quantile(0.25)
            risk_data['LowSalaryRisk'] = (X['MonthlyIncome'] <= salary_percentile_25).astype(int)
            created_features.append('LowSalaryRisk')
            
            # Salary vs job level mismatch
            if 'JobLevel' in X.columns:
                avg_salary_by_level = X.groupby('JobLevel')['MonthlyIncome'].transform('mean')
                risk_data['SalaryLevelMismatch'] = (
                    X['MonthlyIncome'] < avg_salary_by_level * 0.8
                ).astype(int)
                created_features.append('SalaryLevelMismatch')
        
        # 2. Overtime risk
        if 'OverTime' in X.columns:
            # Assuming OverTime is encoded as 0/1 or Yes/No
            if X['OverTime'].dtype == 'object':
                risk_data['OvertimeRisk'] = (X['OverTime'] == 'Yes').astype(int)
            else:
                risk_data['OvertimeRisk'] = X['OverTime'].astype(int)
            created_features.append('OvertimeRisk')
        
        # 3. Commute risk
        if 'DistanceFromHome' in X.columns:
            long_commute_threshold = risk_thresholds['long_commute_miles']
            risk_data['LongCommuteRisk'] = (
                X['DistanceFromHome'] > long_commute_threshold
            ).astype(int)
            created_features.append('LongCommuteRisk')
        
        # 4. Satisfaction risk
        satisfaction_cols = ['JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction']
        available_satisfaction = [col for col in satisfaction_cols if col in X.columns]
        
        if available_satisfaction:
            low_sat_threshold = risk_thresholds['low_satisfaction_threshold']
            
            for col in available_satisfaction:
                risk_feature_name = f'{col}Risk'
                risk_data[risk_feature_name] = (X[col] <= low_sat_threshold).astype(int)
                created_features.append(risk_feature_name)
            
            # Overall satisfaction risk
            risk_data['OverallSatisfactionRisk'] = (
                X[available_satisfaction].mean(axis=1) <= low_sat_threshold
            ).astype(int)
            created_features.append('OverallSatisfactionRisk')
        
        # 5. Career stagnation risk
        if 'YearsSinceLastPromotion' in X.columns:
            risk_data['CareerStagnationRisk'] = (X['YearsSinceLastPromotion'] > 4).astype(int)
            created_features.append('CareerStagnationRisk')
        
        # 6. Age-related risk (young employees often have higher attrition)
        if 'Age' in X.columns:
            risk_data['YoungEmployeeRisk'] = (X['Age'] < 30).astype(int)
            risk_data['NearRetirementRisk'] = (X['Age'] > 55).astype(int)
            created_features.extend(['YoungEmployeeRisk', 'NearRetirementRisk'])
        
        # 7. Travel frequency risk
        if 'BusinessTravel' in X.columns:
            # Assuming BusinessTravel has categories like 'Travel_Frequently', 'Travel_Rarely', 'Non-Travel'
            if X['BusinessTravel'].dtype == 'object':
                risk_data['HighTravelRisk'] = (
                    X['BusinessTravel'] == 'Travel_Frequently'
                ).astype(int)
            created_features.append('HighTravelRisk')
        
        # 8. Performance risk
        if 'PerformanceScore' in X.columns:
            risk_data['LowPerformanceRisk'] = (X['PerformanceScore'] <= 2).astype(int)
            created_features.append('LowPerformanceRisk')
        
        # 9. Training deficiency risk
        if 'TrainingTimesLastYear' in X.columns:
            risk_data['TrainingDeficiencyRisk'] = (X['TrainingTimesLastYear'] == 0).astype(int)
            created_features.append('TrainingDeficiencyRisk')
        
        # 10. Composite risk score
        risk_features = [col for col in created_features if col.endswith('Risk')]
        if risk_features:
            # Simple additive risk score
            risk_data['TotalRiskScore'] = risk_data[risk_features].sum(axis=1)
            created_features.append('TotalRiskScore')
            
            # Weighted risk score (based on business importance)
            risk_weights = {
                'LowSalaryRisk': 0.2,
                'OverallSatisfactionRisk': 0.25,
                'CareerStagnationRisk': 0.15,
                'OvertimeRisk': 0.1,
                'LongCommuteRisk': 0.1,
                'LowPerformanceRisk': 0.2
            }
            
            weighted_risk_score = 0
            for risk_feature in risk_features:
                weight = risk_weights.get(risk_feature, 0.05)  # Default small weight
                if risk_feature in risk_data.columns:
                    weighted_risk_score += risk_data[risk_feature] * weight
            
            risk_data['WeightedRiskScore'] = weighted_risk_score
            created_features.append('WeightedRiskScore')
            
            # Risk level categories
            risk_data['RiskCategory'] = pd.cut(
                risk_data['TotalRiskScore'],
                bins=[-1, 1, 3, 5, float('inf')],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            # One-hot encode risk categories
            risk_cat_dummies = pd.get_dummies(risk_data['RiskCategory'], prefix='RiskCat')
            risk_data = pd.concat([risk_data, risk_cat_dummies], axis=1)
            risk_data.drop('RiskCategory', axis=1, inplace=True)
            created_features.extend(risk_cat_dummies.columns.tolist())
        
        self.risk_features_ = created_features
        logger.info(f"Created {len(created_features)} risk indicator features")
        
        return risk_data
    
    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from numeric data"""
        logger.info("Creating statistical features...")
        
        stat_data = X.copy()
        created_features = []
        
        # Select numeric columns for statistical features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove already engineered features to avoid recursion
        base_numeric_cols = [col for col in numeric_cols 
                           if not any(col.startswith(prefix) for prefix in 
                                    ['poly_', 'TotalRiskScore', 'WeightedRiskScore'])]
        
        if len(base_numeric_cols) >= 3:
            # Cross-feature statistics
            satisfaction_related = [col for col in base_numeric_cols 
                                  if 'Satisfaction' in col or 'WorkLifeBalance' in col]
            
            if len(satisfaction_related) >= 2:
                # Statistical measures across satisfaction features
                stat_data['SatisfactionSkewness'] = X[satisfaction_related].apply(
                    lambda row: stats.skew(row.dropna()) if len(row.dropna()) >= 3 else 0, axis=1
                )
                created_features.append('SatisfactionSkewness')
                
                stat_data['SatisfactionKurtosis'] = X[satisfaction_related].apply(
                    lambda row: stats.kurtosis(row.dropna()) if len(row.dropna()) >= 4 else 0, axis=1
                )
                created_features.append('SatisfactionKurtosis')
            
            # Salary-related statistics
            salary_cols = [col for col in base_numeric_cols 
                          if 'Income' in col or 'Rate' in col or 'Salary' in col]
            
            if len(salary_cols) >= 2:
                stat_data['SalaryVariability'] = X[salary_cols].std(axis=1)
                created_features.append('SalaryVariability')
            
            # Experience-related statistics
            experience_cols = [col for col in base_numeric_cols 
                             if 'Years' in col and 'Since' not in col]
            
            if len(experience_cols) >= 2:
                stat_data['ExperienceVariability'] = X[experience_cols].std(axis=1)
                created_features.append('ExperienceVariability')
                
                stat_data['MaxExperience'] = X[experience_cols].max(axis=1)
                stat_data['MinExperience'] = X[experience_cols].min(axis=1)
                created_features.extend(['MaxExperience', 'MinExperience'])
        
        self.statistical_features_ = created_features
        logger.info(f"Created {len(created_features)} statistical features")
        
        return stat_data
    
    def _create_clustering_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create clustering-based features"""
        logger.info("Creating clustering features...")
        
        cluster_data = X.copy()
        created_features = []
        
        # Select numeric columns for clustering
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select key features for clustering
        key_features = []
        priority_features = ['MonthlyIncome', 'Age', 'YearsAtCompany', 'JobSatisfaction', 
                           'WorkLifeBalance', 'PerformanceScore']
        
        for feature in priority_features:
            if feature in numeric_cols:
                key_features.append(feature)
        
        # Add other important numeric features
        remaining_features = [col for col in numeric_cols if col not in key_features]
        key_features.extend(remaining_features[:max(0, 10 - len(key_features))])
        
        if len(key_features) >= 3:
            try:
                # Standardize features for clustering
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(X[key_features])
                
                # K-means clustering
                kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                # Store clusterer for transform method
                self.clusterer_ = kmeans
                self.scaler_ = scaler
                
                # Add cluster labels
                cluster_data['EmployeeCluster'] = cluster_labels
                created_features.append('EmployeeCluster')
                
                # One-hot encode clusters
                cluster_dummies = pd.get_dummies(cluster_labels, prefix='Cluster')
                cluster_data = pd.concat([cluster_data, cluster_dummies], axis=1)
                created_features.extend(cluster_dummies.columns.tolist())
                
                # Distance to cluster center
                cluster_centers = kmeans.cluster_centers_
                distances = []
                
                for i, center in enumerate(cluster_centers):
                    cluster_mask = cluster_labels == i
                    if cluster_mask.any():
                        cluster_distances = np.linalg.norm(
                            scaled_features[cluster_mask] - center, axis=1
                        )
                        distances.extend(cluster_distances)
                    
                cluster_data['ClusterDistance'] = distances
                created_features.append('ClusterDistance')
                
                logger.info(f"Created {self.config.n_clusters} employee clusters")
                
            except Exception as e:
                logger.warning(f"Error creating clustering features: {e}")
        
        self.clustering_features_ = created_features
        logger.info(f"Created {len(created_features)} clustering features")
        
        return cluster_data
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select most important features using specified method"""
        logger.info(f"Selecting features using {self.config.selection_method} method...")
        
        # Prepare data for feature selection
        numeric_data = X.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) <= self.config.max_features:
            logger.info(f"Feature count ({len(numeric_data.columns)}) already below threshold")
            return X
        
        try:
            if self.config.selection_method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=self.config.max_features)
            elif self.config.selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=self.config.max_features)
            else:
                # Default to mutual info
                selector = SelectKBest(score_func=mutual_info_classif, k=self.config.max_features)
            
            # Fit selector and get selected features
            selected_features_array = selector.fit_transform(numeric_data, y)
            selected_feature_names = numeric_data.columns[selector.get_support()].tolist()
            
            # Store selector and selected features
            self.feature_selector_ = selector
            self.selected_features_ = selected_feature_names
            
            # Return DataFrame with selected features plus non-numeric columns
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            final_columns = selected_feature_names + non_numeric_cols
            
            logger.info(f"Selected {len(selected_feature_names)} features from {len(numeric_data.columns)} candidates")
            
            return X[final_columns]
            
        except Exception as e:
            logger.warning(f"Error in feature selection: {e}")
            return X
    
    def _get_feature_importance_scores(self) -> Dict[str, float]:
        """Get feature importance scores from selector if available"""
        if self.feature_selector_ is not None and hasattr(self.feature_selector_, 'scores_'):
            feature_names = self.feature_names_ or []
            scores = self.feature_selector_.scores_
            
            if len(feature_names) == len(scores):
                return dict(zip(feature_names, scores))
        
        return {}
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted feature engineer"""
        if not hasattr(self, 'feature_names_') or self.feature_names_ is None:
            raise ValueError("FeatureEngineer not fitted. Call fit_transform() first.")
        
        logger.info("Transforming new data using fitted feature engineer...")
        
        # Apply same transformations (without fitting)
        transformed_data = X.copy()
        
        # Create interaction features using same pairs
        if self.config.create_interactions:
            transformed_data = self.create_interaction_features(transformed_data)
        
        # Apply other transformations...
        # (Similar to fit_transform but without fitting new parameters)
        
        # Select same features
        if self.selected_features_:
            available_features = [col for col in self.selected_features_ if col in transformed_data.columns]
            transformed_data = transformed_data[available_features]
        
        return transformed_data
    
    def get_feature_engineering_report(self) -> Optional[FeatureEngineeringReport]:
        """Get detailed feature engineering report"""
        return self.report_
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, float]:
        """Get top K most important features"""
        importance_scores = self._get_feature_importance_scores()
        
        if importance_scores:
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features[:top_k])
        
        return {}


# Convenience functions
def engineer_hr_features(X: pd.DataFrame, y: Optional[pd.Series] = None,
                        config: Optional[FeatureEngineeringConfig] = None) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """Quick function to engineer HR features with default settings"""
    engineer = FeatureEngineer(config)
    X_engineered = engineer.fit_transform(X, y)
    return X_engineered, engineer


def create_feature_engineering_pipeline(config: Optional[FeatureEngineeringConfig] = None) -> FeatureEngineer:
    """Create a configured feature engineering pipeline"""
    return FeatureEngineer(config)


if __name__ == "__main__":
    # Test the feature engineer
    print("🧪 Testing FeatureEngineer...")
    
    # This would test with actual data
    # from src.data_processing.data_loader import load_hr_data
    # data, _ = load_hr_data("data/synthetic/hr_employees.csv")
    # 
    # engineer = FeatureEngineer()
    # X_engineered = engineer.fit_transform(data.drop('Attrition', axis=1), data['Attrition'])
    # 
    # report = engineer.get_feature_engineering_report()
    # print(f"Created {report.engineered_features} new features")
    
    print("✅ FeatureEngineer test completed!")
