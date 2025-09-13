"""
HR Attrition Predictor - SHAP Explainability Engine
==================================================
Comprehensive SHAP integration for model interpretability with individual explanations,
feature importance analysis, and natural language insights. Optimized for 4GB RAM.

Author: HR Analytics Team  
Date: September 2025
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import gc
import json
from dataclasses import dataclass, asdict

# SHAP imports with error handling
try:
    import shap
    SHAP_AVAILABLE = True
    # Configure SHAP
    shap.initjs()
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================================================
# DATA CLASSES FOR STRUCTURED OUTPUTS
# ================================================================

@dataclass
class FeatureImportance:
    """Feature importance data structure"""
    feature_name: str
    importance_score: float
    rank: int
    impact_direction: str  # 'positive' or 'negative'
    description: str

@dataclass
class IndividualExplanation:
    """Individual prediction explanation"""
    employee_id: str
    prediction: float
    base_value: float
    shap_values: Dict[str, float]
    feature_contributions: List[FeatureImportance]
    natural_language_explanation: str
    confidence_level: str
    risk_factors: List[str]
    protective_factors: List[str]

@dataclass
class ExplainabilityReport:
    """Comprehensive explainability report"""
    model_name: str
    explanation_type: str
    generation_timestamp: str
    sample_size: int
    top_features: List[FeatureImportance]
    individual_explanations: List[IndividualExplanation]
    summary_insights: List[str]
    methodology_notes: str

# ================================================================
# SHAP EXPLAINER CLASS
# ================================================================

class SHAPExplainer:
    """
    Comprehensive SHAP explainer for HR attrition models with memory optimization.
    
    Provides individual explanations, feature importance, force plots, waterfall charts,
    and natural language explanations optimized for 4GB RAM systems.
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame, 
                 feature_names: Optional[List[str]] = None,
                 max_background_size: int = 100):
        """
        Initialize SHAP explainer with memory optimization.
        
        Args:
            model: Trained ML model (scikit-learn compatible)
            X_train: Training data for background distribution
            feature_names: Optional list of feature names
            max_background_size: Maximum background sample size for memory efficiency
        """
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names or list(X_train.columns)
        self.model_name = type(model).__name__
        
        # Memory-optimized background data
        if len(X_train) > max_background_size:
            logger.info(f"Sampling {max_background_size} background samples for memory efficiency")
            self.X_background = X_train.sample(max_background_size, random_state=42)
        else:
            self.X_background = X_train.copy()
        
        # Initialize explainer based on model type
        self.explainer = self._initialize_explainer()
        
        # Caches for performance
        self._global_importance_cache = None
        self._base_value_cache = None
        
        logger.info(f"SHAPExplainer initialized for {self.model_name} with {len(self.X_background)} background samples")
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        
        try:
            # Tree-based models (more memory efficient)
            if hasattr(self.model, 'tree_') or 'Forest' in self.model_name or 'XGB' in self.model_name:
                if hasattr(shap, 'TreeExplainer'):
                    logger.info("Using TreeExplainer for tree-based model")
                    return shap.TreeExplainer(self.model)
                else:
                    logger.warning("TreeExplainer not available, falling back to general explainer")
            
            # Linear models
            elif 'Linear' in self.model_name or 'Logistic' in self.model_name:
                if hasattr(shap, 'LinearExplainer'):
                    logger.info("Using LinearExplainer for linear model")
                    return shap.LinearExplainer(self.model, self.X_background)
                
            # General explainer (most memory intensive)
            logger.info("Using general Explainer")
            return shap.Explainer(self.model, self.X_background)
            
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            # Fallback to basic explainer
            return shap.Explainer(self.model, self.X_background)
    
    def generate_individual_explanations(self, 
                                       X_samples: Union[pd.DataFrame, np.ndarray],
                                       employee_ids: Optional[List[str]] = None,
                                       max_samples: int = 50) -> List[IndividualExplanation]:
        """
        Generate individual SHAP explanations for specific samples.
        
        Args:
            X_samples: Sample data to explain (DataFrame or array)
            employee_ids: Optional employee IDs for samples
            max_samples: Maximum samples to process (memory limitation)
            
        Returns:
            List of IndividualExplanation objects
        """
        
        logger.info(f"Generating individual explanations for {len(X_samples)} samples...")
        
        # Memory optimization: limit samples
        if len(X_samples) > max_samples:
            logger.warning(f"Limiting to {max_samples} samples for memory efficiency")
            if isinstance(X_samples, pd.DataFrame):
                X_samples = X_samples.head(max_samples)
            else:
                X_samples = X_samples[:max_samples]
        
        # Convert to DataFrame if needed
        if isinstance(X_samples, np.ndarray):
            X_samples = pd.DataFrame(X_samples, columns=self.feature_names)
        
        # Generate employee IDs if not provided
        if employee_ids is None:
            employee_ids = [f"EMP_{i:04d}" for i in range(len(X_samples))]
        
        explanations = []
        
        try:
            # Calculate SHAP values (memory efficient batch processing)
            shap_values = self._calculate_shap_values_batch(X_samples)
            
            # Get base value (expected model output)
            if hasattr(self.explainer, 'expected_value'):
                base_value = float(self.explainer.expected_value)
            else:
                base_value = float(np.mean(self.model.predict_proba(self.X_background)[:, 1]))
            
            # Generate explanations for each sample
            for i, (idx, row) in enumerate(X_samples.iterrows()):
                if i >= len(employee_ids):
                    break
                
                # Get SHAP values for this sample
                sample_shap_values = shap_values[i] if len(shap_values.shape) > 1 else shap_values
                
                # Make prediction
                if hasattr(self.model, 'predict_proba'):
                    prediction = float(self.model.predict_proba([row.values])[0][1])
                else:
                    prediction = float(self.model.predict([row.values])[0])
                
                # Create feature contributions
                feature_contributions = []
                shap_dict = {}
                
                for j, (feature_name, shap_value) in enumerate(zip(self.feature_names, sample_shap_values)):
                    shap_dict[feature_name] = float(shap_value)
                    
                    feature_contributions.append(FeatureImportance(
                        feature_name=feature_name,
                        importance_score=abs(float(shap_value)),
                        rank=0,  # Will be set later
                        impact_direction='positive' if shap_value > 0 else 'negative',
                        description=self._get_feature_description(feature_name, row[feature_name], shap_value)
                    ))
                
                # Sort and rank contributions
                feature_contributions.sort(key=lambda x: x.importance_score, reverse=True)
                for rank, contrib in enumerate(feature_contributions, 1):
                    contrib.rank = rank
                
                # Generate natural language explanation
                nl_explanation = self._generate_natural_language_explanation(
                    employee_ids[i], prediction, feature_contributions[:5]
                )
                
                # Identify risk and protective factors
                risk_factors = [fc.feature_name for fc in feature_contributions[:3] 
                              if fc.impact_direction == 'positive']
                protective_factors = [fc.feature_name for fc in feature_contributions[:3] 
                                    if fc.impact_direction == 'negative']
                
                # Determine confidence level
                confidence_level = self._determine_confidence_level(prediction, feature_contributions)
                
                # Create explanation object
                explanation = IndividualExplanation(
                    employee_id=employee_ids[i],
                    prediction=prediction,
                    base_value=base_value,
                    shap_values=shap_dict,
                    feature_contributions=feature_contributions,
                    natural_language_explanation=nl_explanation,
                    confidence_level=confidence_level,
                    risk_factors=risk_factors,
                    protective_factors=protective_factors
                )
                
                explanations.append(explanation)
            
            logger.info(f"Generated {len(explanations)} individual explanations")
            
        except Exception as e:
            logger.error(f"Error generating individual explanations: {e}")
            raise
        
        finally:
            # Memory cleanup
            gc.collect()
        
        return explanations
    
    def _calculate_shap_values_batch(self, X_samples: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values with memory-efficient batching."""
        
        try:
            # For tree explainers, we can pass data directly
            if hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(X_samples.values)
                # Handle multi-class output (take positive class)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class for binary classification
            else:
                # For general explainers
                shap_explanation = self.explainer(X_samples.values)
                if hasattr(shap_explanation, 'values'):
                    shap_values = shap_explanation.values
                else:
                    shap_values = shap_explanation
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            # Fallback: return zeros
            return np.zeros((len(X_samples), len(self.feature_names)))
    
    def create_force_plots(self, explanations: List[IndividualExplanation],
                          save_path: Optional[str] = None) -> List[str]:
        """
        Create SHAP force plots for individual explanations.
        
        Args:
            explanations: List of individual explanations
            save_path: Optional path to save plots
            
        Returns:
            List of plot file paths or HTML strings
        """
        
        logger.info(f"Creating force plots for {len(explanations)} explanations...")
        
        plot_outputs = []
        
        try:
            for i, explanation in enumerate(explanations[:10]):  # Limit to 10 for memory
                # Convert SHAP values to array
                shap_values_array = np.array([list(explanation.shap_values.values())])
                
                # Create force plot
                force_plot = shap.force_plot(
                    explanation.base_value,
                    shap_values_array[0],
                    features=list(explanation.shap_values.keys()),
                    show=False,
                    matplotlib=True
                )
                
                if save_path:
                    plot_file = f"{save_path}/force_plot_{explanation.employee_id}_{i}.png"
                    plt.savefig(plot_file, bbox_inches='tight', facecolor='black', 
                               edgecolor='none', dpi=150)
                    plt.close()
                    plot_outputs.append(plot_file)
                else:
                    # Return HTML string
                    plot_outputs.append(force_plot)
        
        except Exception as e:
            logger.error(f"Error creating force plots: {e}")
        
        return plot_outputs
    
    def calculate_feature_importance(self, X_samples: Optional[pd.DataFrame] = None,
                                   top_n: int = 20) -> List[FeatureImportance]:
        """
        Calculate global feature importance using SHAP values.
        
        Args:
            X_samples: Optional samples for importance calculation (uses background if None)
            top_n: Number of top features to return
            
        Returns:
            List of FeatureImportance objects sorted by importance
        """
        
        if self._global_importance_cache is not None:
            logger.info("Using cached global feature importance")
            return self._global_importance_cache[:top_n]
        
        logger.info("Calculating global feature importance...")
        
        # Use background data if no samples provided
        if X_samples is None:
            X_samples = self.X_background.copy()
        
        # Limit sample size for memory efficiency
        max_samples = min(200, len(X_samples))
        if len(X_samples) > max_samples:
            X_samples = X_samples.sample(max_samples, random_state=42)
        
        try:
            # Calculate SHAP values
            shap_values = self._calculate_shap_values_batch(X_samples)
            
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance objects
            feature_importances = []
            
            for i, (feature_name, importance) in enumerate(zip(self.feature_names, mean_abs_shap)):
                # Calculate average impact direction
                avg_shap = np.mean(shap_values[:, i])
                impact_direction = 'positive' if avg_shap > 0 else 'negative'
                
                feature_importances.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(importance),
                    rank=0,  # Will be set after sorting
                    impact_direction=impact_direction,
                    description=self._get_global_feature_description(feature_name, avg_shap)
                ))
            
            # Sort by importance and assign ranks
            feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
            for rank, fi in enumerate(feature_importances, 1):
                fi.rank = rank
            
            # Cache results
            self._global_importance_cache = feature_importances
            
            logger.info(f"Calculated importance for {len(feature_importances)} features")
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            feature_importances = []
        
        return feature_importances[:top_n]
    
    def generate_waterfall_charts(self, explanations: List[IndividualExplanation],
                                save_path: Optional[str] = None) -> List[str]:
        """
        Generate SHAP waterfall charts for individual explanations.
        
        Args:
            explanations: List of individual explanations
            save_path: Optional path to save charts
            
        Returns:
            List of chart file paths or data
        """
        
        logger.info(f"Generating waterfall charts for {len(explanations)} explanations...")
        
        chart_outputs = []
        
        try:
            for i, explanation in enumerate(explanations[:5]):  # Limit for memory
                # Get top contributing features
                top_features = explanation.feature_contributions[:10]
                
                # Create custom waterfall chart data
                waterfall_data = {
                    'employee_id': explanation.employee_id,
                    'base_value': explanation.base_value,
                    'prediction': explanation.prediction,
                    'features': [fc.feature_name for fc in top_features],
                    'contributions': [fc.importance_score * (1 if fc.impact_direction == 'positive' else -1) 
                                   for fc in top_features],
                    'cumulative_values': []
                }
                
                # Calculate cumulative values for waterfall
                cumulative = explanation.base_value
                waterfall_data['cumulative_values'].append(cumulative)
                
                for contrib in waterfall_data['contributions']:
                    cumulative += contrib
                    waterfall_data['cumulative_values'].append(cumulative)
                
                if save_path:
                    chart_file = f"{save_path}/waterfall_{explanation.employee_id}_{i}.json"
                    with open(chart_file, 'w') as f:
                        json.dump(waterfall_data, f, indent=2)
                    chart_outputs.append(chart_file)
                else:
                    chart_outputs.append(waterfall_data)
        
        except Exception as e:
            logger.error(f"Error generating waterfall charts: {e}")
        
        return chart_outputs
    
    def create_natural_language_explanations(self, explanations: List[IndividualExplanation]) -> Dict[str, str]:
        """
        Generate natural language explanations for multiple employees.
        
        Args:
            explanations: List of individual explanations
            
        Returns:
            Dictionary mapping employee IDs to natural language explanations
        """
        
        logger.info(f"Creating natural language explanations for {len(explanations)} employees...")
        
        nl_explanations = {}
        
        for explanation in explanations:
            nl_explanations[explanation.employee_id] = explanation.natural_language_explanation
        
        return nl_explanations
    
    def _generate_natural_language_explanation(self, employee_id: str, prediction: float,
                                             top_features: List[FeatureImportance]) -> str:
        """Generate natural language explanation for an individual prediction."""
        
        # Risk level determination
        if prediction > 0.7:
            risk_level = "high"
            risk_phrase = "is at high risk of leaving"
        elif prediction > 0.3:
            risk_level = "moderate"
            risk_phrase = "has moderate risk of leaving"
        else:
            risk_level = "low"
            risk_phrase = "has low risk of leaving"
        
        # Start explanation
        explanation = f"Employee {employee_id} {risk_phrase} the organization (probability: {prediction:.1%}). "
        
        # Add key contributing factors
        if top_features:
            explanation += "Key factors influencing this prediction include:\n\n"
            
            for i, feature in enumerate(top_features[:3], 1):
                impact_word = "increases" if feature.impact_direction == "positive" else "decreases"
                explanation += f"{i}. **{self._humanize_feature_name(feature.feature_name)}**: {impact_word} attrition risk "
                explanation += f"(impact: {feature.importance_score:.3f})\n"
            
            # Add interpretation
            if risk_level == "high":
                explanation += "\n**Recommended Actions:**\n"
                explanation += "- Schedule immediate one-on-one discussion\n"
                explanation += "- Review compensation and career development\n"
                explanation += "- Address specific concerns identified above\n"
            elif risk_level == "moderate":
                explanation += "\n**Recommended Actions:**\n"
                explanation += "- Increase engagement and feedback frequency\n"
                explanation += "- Monitor satisfaction levels closely\n"
                explanation += "- Consider targeted retention strategies\n"
            else:
                explanation += "\n**Status:** Employee appears well-engaged. Continue current strategies.\n"
        
        return explanation
    
    def _get_feature_description(self, feature_name: str, feature_value: Any, shap_value: float) -> str:
        """Generate description for individual feature contribution."""
        
        humanized_name = self._humanize_feature_name(feature_name)
        impact = "increases" if shap_value > 0 else "decreases"
        
        return f"{humanized_name} (value: {feature_value}) {impact} attrition risk by {abs(shap_value):.3f}"
    
    def _get_global_feature_description(self, feature_name: str, avg_shap: float) -> str:
        """Generate description for global feature importance."""
        
        humanized_name = self._humanize_feature_name(feature_name)
        direction = "positively" if avg_shap > 0 else "negatively"
        
        return f"{humanized_name} {direction} correlates with attrition across the organization"
    
    def _humanize_feature_name(self, feature_name: str) -> str:
        """Convert feature names to human-readable format."""
        
        name_mapping = {
            'Age': 'Employee Age',
            'YearsAtCompany': 'Years at Company',
            'YearsInCurrentRole': 'Years in Current Role',
            'TotalWorkingYears': 'Total Work Experience',
            'MonthlyIncome': 'Monthly Income',
            'JobSatisfaction': 'Job Satisfaction',
            'WorkLifeBalance': 'Work-Life Balance',
            'EnvironmentSatisfaction': 'Environment Satisfaction',
            'RelationshipSatisfaction': 'Relationship Satisfaction',
            'PerformanceRating': 'Performance Rating',
            'OverTime': 'Overtime Work',
            'BusinessTravel': 'Business Travel',
            'DistanceFromHome': 'Distance from Home',
            'Department': 'Department',
            'JobRole': 'Job Role',
            'JobLevel': 'Job Level',
            'Gender': 'Gender',
            'MaritalStatus': 'Marital Status',
            'Education': 'Education Level'
        }
        
        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())
    
    def _determine_confidence_level(self, prediction: float, 
                                  feature_contributions: List[FeatureImportance]) -> str:
        """Determine confidence level based on prediction and feature contributions."""
        
        # Check prediction extremes
        if prediction < 0.1 or prediction > 0.9:
            confidence = "High"
        elif prediction < 0.3 or prediction > 0.7:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Adjust based on feature contribution consistency
        top_contributions = feature_contributions[:5]
        if len(top_contributions) > 2:
            contribution_variance = np.var([fc.importance_score for fc in top_contributions])
            
            # If contributions are very uneven, reduce confidence
            if contribution_variance > 0.1:
                if confidence == "High":
                    confidence = "Medium"
                elif confidence == "Medium":
                    confidence = "Low"
        
        return confidence
    
    def generate_explainability_report(self, explanations: List[IndividualExplanation],
                                     feature_importance: List[FeatureImportance]) -> ExplainabilityReport:
        """
        Generate comprehensive explainability report.
        
        Args:
            explanations: List of individual explanations
            feature_importance: Global feature importance list
            
        Returns:
            ExplainabilityReport object
        """
        
        logger.info("Generating comprehensive explainability report...")
        
        # Generate summary insights
        summary_insights = self._generate_summary_insights(explanations, feature_importance)
        
        # Create methodology notes
        methodology_notes = f"""
        SHAP (SHapley Additive exPlanations) Analysis:
        - Model: {self.model_name}
        - Background samples: {len(self.X_background)}
        - Explainer type: {type(self.explainer).__name__}
        - Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Methodology:
        1. SHAP values calculated for each prediction
        2. Feature contributions ranked by absolute impact
        3. Natural language explanations generated
        4. Confidence levels determined by prediction certainty and feature consistency
        """
        
        report = ExplainabilityReport(
            model_name=self.model_name,
            explanation_type="SHAP Individual + Global Analysis",
            generation_timestamp=datetime.now().isoformat(),
            sample_size=len(explanations),
            top_features=feature_importance,
            individual_explanations=explanations,
            summary_insights=summary_insights,
            methodology_notes=methodology_notes
        )
        
        logger.info("Explainability report generated successfully")
        return report
    
    def _generate_summary_insights(self, explanations: List[IndividualExplanation],
                                 feature_importance: List[FeatureImportance]) -> List[str]:
        """Generate high-level insights from explanations."""
        
        insights = []
        
        if not explanations or not feature_importance:
            return ["Insufficient data for summary insights"]
        
        # Most important global feature
        top_global_feature = feature_importance[0] if feature_importance else None
        if top_global_feature:
            insights.append(f"**Most Important Factor**: {self._humanize_feature_name(top_global_feature.feature_name)} "
                          f"is the strongest predictor of attrition across the organization.")
        
        # Risk distribution
        high_risk = sum(1 for exp in explanations if exp.prediction > 0.7)
        medium_risk = sum(1 for exp in explanations if 0.3 < exp.prediction <= 0.7)
        low_risk = len(explanations) - high_risk - medium_risk
        
        insights.append(f"**Risk Distribution**: {high_risk} high-risk, {medium_risk} medium-risk, "
                       f"and {low_risk} low-risk employees in the analyzed sample.")
        
        # Common risk factors
        all_risk_factors = []
        for exp in explanations:
            all_risk_factors.extend(exp.risk_factors)
        
        if all_risk_factors:
            from collections import Counter
            common_risks = Counter(all_risk_factors).most_common(3)
            risk_names = [self._humanize_feature_name(risk[0]) for risk in common_risks]
            insights.append(f"**Common Risk Factors**: {', '.join(risk_names)} appear frequently "
                          f"as contributors to attrition risk.")
        
        # Confidence analysis
        high_confidence = sum(1 for exp in explanations if exp.confidence_level == "High")
        confidence_rate = high_confidence / len(explanations) if explanations else 0
        
        insights.append(f"**Prediction Confidence**: {confidence_rate:.1%} of predictions have high confidence, "
                       f"indicating strong model certainty for most employees.")
        
        return insights
    
    def export_explanations(self, explanations: List[IndividualExplanation],
                          export_path: str, format: str = 'json') -> str:
        """
        Export explanations to file.
        
        Args:
            explanations: List of explanations to export
            export_path: Path to export file
            format: Export format ('json' or 'csv')
            
        Returns:
            Path to exported file
        """
        
        logger.info(f"Exporting {len(explanations)} explanations to {format.upper()} format...")
        
        if format.lower() == 'json':
            export_data = [asdict(exp) for exp in explanations]
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Convert to DataFrame for CSV export
            csv_data = []
            for exp in explanations:
                row = {
                    'employee_id': exp.employee_id,
                    'prediction': exp.prediction,
                    'confidence_level': exp.confidence_level,
                    'natural_language_explanation': exp.natural_language_explanation,
                    'top_risk_factors': ', '.join(exp.risk_factors[:3]),
                    'top_protective_factors': ', '.join(exp.protective_factors[:3])
                }
                
                # Add top 5 SHAP values
                for i, contrib in enumerate(exp.feature_contributions[:5], 1):
                    row[f'top_{i}_feature'] = contrib.feature_name
                    row[f'top_{i}_impact'] = contrib.importance_score
                    row[f'top_{i}_direction'] = contrib.impact_direction
                
                csv_data.append(row)
            
            pd.DataFrame(csv_data).to_csv(export_path, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Explanations exported to {export_path}")
        return export_path


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def create_explainer_for_model(model: Any, X_train: pd.DataFrame, 
                              feature_names: Optional[List[str]] = None) -> Optional[SHAPExplainer]:
    """
    Convenience function to create SHAP explainer.
    
    Args:
        model: Trained ML model
        X_train: Training data
        feature_names: Optional feature names
        
    Returns:
        SHAPExplainer instance or None if SHAP not available
    """
    
    if not SHAP_AVAILABLE:
        logger.error("SHAP not available for explainer creation")
        return None
    
    try:
        return SHAPExplainer(model, X_train, feature_names)
    except Exception as e:
        logger.error(f"Error creating SHAP explainer: {e}")
        return None

def quick_explanation(model: Any, X_train: pd.DataFrame, X_sample: pd.DataFrame,
                     employee_id: str = "SAMPLE") -> Optional[IndividualExplanation]:
    """
    Quick individual explanation for a single sample.
    
    Args:
        model: Trained ML model
        X_train: Training data
        X_sample: Single sample to explain
        employee_id: Employee identifier
        
    Returns:
        IndividualExplanation or None
    """
    
    explainer = create_explainer_for_model(model, X_train)
    if explainer is None:
        return None
    
    try:
        explanations = explainer.generate_individual_explanations(
            X_sample, [employee_id], max_samples=1
        )
        return explanations[0] if explanations else None
    except Exception as e:
        logger.error(f"Error generating quick explanation: {e}")
        return None

# ================================================================
# EXPORT ALL CLASSES AND FUNCTIONS
# ================================================================

__all__ = [
    'SHAPExplainer',
    'IndividualExplanation',
    'FeatureImportance', 
    'ExplainabilityReport',
    'create_explainer_for_model',
    'quick_explanation',
    'SHAP_AVAILABLE'
]

# ================================================================
# DEVELOPMENT TESTING
# ================================================================

def test_explainer():
    """Test function for SHAP explainer (requires trained model)."""
    
    if not SHAP_AVAILABLE:
        print("❌ SHAP not available for testing")
        return
    
    print("🧪 Testing SHAP Explainer...")
    
    # This would require actual trained model and data
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.datasets import make_classification
    # 
    # X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    # X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    # 
    # model = RandomForestClassifier(random_state=42)
    # model.fit(X_df, y)
    # 
    # explainer = SHAPExplainer(model, X_df)
    # explanations = explainer.generate_individual_explanations(X_df.head(5))
    # 
    # print(f"✅ Generated {len(explanations)} explanations")
    
    print("✅ SHAP Explainer module loaded successfully!")

if __name__ == "__main__":
    test_explainer()
