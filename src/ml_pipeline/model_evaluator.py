"""
HR Attrition Predictor - Model Evaluator
========================================
Comprehensive model evaluation with advanced metrics, visualizations,
and performance analysis for ML models.

Author: HR Analytics Team
Date: September 2025
Version: 2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
from datetime import datetime
import pickle

# ML Libraries
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss
)
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, learning_curve,
    validation_curve
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import shap

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
try:
    from src.models.predictor import AttritionPredictor # type: ignore 
    from src.models.ensemble import EnsemblePredictor # type: ignore
    from src.models.explainer import SHAPExplainer # type: ignore
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Model imports not available: {e}")
    MODELS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================================================
# EVALUATION DATA CLASSES
# ================================================================

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    precision_recall_auc: float
    matthews_corrcoef: float
    cohen_kappa: float
    log_loss: float
    brier_score: float
    specificity: float
    npv: float  # Negative Predictive Value
    balanced_accuracy: float

@dataclass
class CrossValidationResults:
    """Container for cross-validation results."""
    cv_scores: np.ndarray
    mean_score: float
    std_score: float
    fold_scores: List[Dict[str, float]]
    best_fold: int
    worst_fold: int

@dataclass
class FeatureImportanceResults:
    """Container for feature importance results."""
    feature_names: List[str]
    importance_scores: np.ndarray
    importance_type: str  # 'gain', 'permutation', 'shap'
    model_name: str
    ranking: List[int]

# ================================================================
# MODEL EVALUATOR CLASS
# ================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation with advanced metrics and visualizations.
    
    This class provides extensive evaluation capabilities including:
    - Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
    - Confusion matrix analysis
    - ROC and Precision-Recall curves
    - Feature importance analysis
    - Cross-validation with detailed statistics
    - Model calibration analysis
    - Learning curves
    - Model comparison utilities
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 cv_folds: int = 5,
                 n_jobs: int = -1,
                 verbose: bool = True):
        """
        Initialize the ModelEvaluator.
        
        Args:
            random_state: Random state for reproducibility
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all processors)
            verbose: Whether to print detailed information
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Storage for evaluation results
        self.evaluation_results = {}
        self.comparison_results = {}
        
        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if MODELS_AVAILABLE:
            try:
                self.shap_explainer = SHAPExplainer()
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer: {e}")
        
        # Create output directories
        self.output_dir = project_root / "results" / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info("ModelEvaluator initialized successfully")
    
    def calculate_accuracy_metrics(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 y_pred_proba: Optional[np.ndarray] = None,
                                 model_name: str = "Unknown") -> EvaluationMetrics:
        """
        Calculate comprehensive accuracy metrics for binary classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model being evaluated
            
        Returns:
            EvaluationMetrics object containing all metrics
        """
        
        if self.verbose:
            logger.info(f"Calculating accuracy metrics for {model_name}")
        
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            # Confusion matrix for additional metrics
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate specificity and NPV
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Balanced accuracy
            balanced_accuracy = (recall + specificity) / 2
            
            # Advanced metrics
            matthews_cc = matthews_corrcoef(y_true, y_pred)
            cohen_kappa = cohen_kappa_score(y_true, y_pred)
            
            # Probability-based metrics (if probabilities available)
            if y_pred_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_true, y_pred_proba)
                    pr_auc = average_precision_score(y_true, y_pred_proba)
                    logloss = log_loss(y_true, y_pred_proba)
                    brier_score = brier_score_loss(y_true, y_pred_proba)
                except Exception as e:
                    logger.warning(f"Error calculating probability-based metrics: {e}")
                    roc_auc = pr_auc = logloss = brier_score = 0.0
            else:
                roc_auc = pr_auc = logloss = brier_score = 0.0
            
            # Create metrics object
            metrics = EvaluationMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                precision_recall_auc=pr_auc,
                matthews_corrcoef=matthews_cc,
                cohen_kappa=cohen_kappa,
                log_loss=logloss,
                brier_score=brier_score,
                specificity=specificity,
                npv=npv,
                balanced_accuracy=balanced_accuracy
            )
            
            # Store results
            self.evaluation_results[model_name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': classification_report(y_true, y_pred, output_dict=True),
                'timestamp': datetime.now()
            }
            
            if self.verbose:
                self._print_metrics_summary(metrics, model_name)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics for {model_name}: {e}")
            raise
    
    def generate_confusion_matrix(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                model_name: str = "Unknown",
                                normalize: Optional[str] = None,
                                save_plot: bool = True) -> Tuple[np.ndarray, go.Figure]:
        """
        Generate and visualize confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            normalize: Normalization mode ('true', 'pred', 'all', or None)
            save_plot: Whether to save the plot
            
        Returns:
            Tuple of (confusion_matrix_array, plotly_figure)
        """
        
        if self.verbose:
            logger.info(f"Generating confusion matrix for {model_name}")
        
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred, normalize=normalize)
            
            # Create labels
            if normalize == 'true':
                title = f"Normalized Confusion Matrix (by True Label) - {model_name}"
                text_suffix = "%"
            elif normalize == 'pred':
                title = f"Normalized Confusion Matrix (by Prediction) - {model_name}"
                text_suffix = "%"
            elif normalize == 'all':
                title = f"Normalized Confusion Matrix (Overall) - {model_name}"
                text_suffix = "%"
            else:
                title = f"Confusion Matrix - {model_name}"
                text_suffix = ""
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['No Attrition', 'Attrition'],
                y=['No Attrition', 'Attrition'],
                colorscale='Blues',
                showscale=True,
                text=[[f"{val:.2f}{text_suffix}" if normalize else f"{int(val)}" for val in row] for row in cm],
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"},
                hovertemplate='<b>True: %{y}</b><br>Predicted: %{x}<br>Value: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'font': {'size': 18}
                },
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                width=500,
                height=400,
                font=dict(size=12)
            )
            
            # Add performance annotations
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            annotations_text = [
                f"True Negatives: {tn}",
                f"False Positives: {fp}",
                f"False Negatives: {fn}",
                f"True Positives: {tp}",
                f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.3f}",
                f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.3f}",
                f"Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.3f}"
            ]
            
            fig.add_annotation(
                x=1.15, y=0.5,
                xref="paper", yref="paper",
                text="<br>".join(annotations_text),
                showarrow=False,
                font=dict(size=10),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
            
            # Save plot if requested
            if save_plot:
                plot_path = self.plots_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.html"
                fig.write_html(str(plot_path))
                
                if self.verbose:
                    logger.info(f"Confusion matrix saved to {plot_path}")
            
            return cm, fig
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix for {model_name}: {e}")
            raise
    
    def plot_roc_curves(self, 
                       models_data: Dict[str, Dict[str, np.ndarray]],
                       save_plot: bool = True) -> go.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_data: Dictionary with model_name -> {'y_true': array, 'y_pred_proba': array}
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure with ROC curves
        """
        
        if self.verbose:
            logger.info(f"Plotting ROC curves for {len(models_data)} models")
        
        try:
            fig = go.Figure()
            
            # Color palette for different models
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            auc_scores = {}
            
            for i, (model_name, data) in enumerate(models_data.items()):
                y_true = data['y_true']
                y_pred_proba = data['y_pred_proba']
                
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
                auc_score = roc_auc_score(y_true, y_pred_proba)
                auc_scores[model_name] = auc_score
                
                # Add ROC curve
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc_score:.3f})',
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{model_name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
                ))
            
            # Add diagonal line (random classifier)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=2, dash='dash'),
                hovertemplate='Random Classifier<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': 'ROC Curves Comparison',
                    'x': 0.5,
                    'font': {'size': 20}
                },
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=800,
                height=600,
                legend=dict(
                    yanchor="bottom",
                    y=0.02,
                    xanchor="right",
                    x=0.98
                ),
                font=dict(size=12)
            )
            
            # Save plot if requested
            if save_plot:
                plot_path = self.plots_dir / "roc_curves_comparison.html"
                fig.write_html(str(plot_path))
                
                if self.verbose:
                    logger.info(f"ROC curves saved to {plot_path}")
            
            # Store AUC scores
            self.comparison_results['roc_auc_scores'] = auc_scores
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting ROC curves: {e}")
            raise
    
    def plot_precision_recall_curves(self,
                                    models_data: Dict[str, Dict[str, np.ndarray]],
                                    save_plot: bool = True) -> go.Figure:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            models_data: Dictionary with model_name -> {'y_true': array, 'y_pred_proba': array}
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure with PR curves
        """
        
        if self.verbose:
            logger.info(f"Plotting Precision-Recall curves for {len(models_data)} models")
        
        try:
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            ap_scores = {}
            
            for i, (model_name, data) in enumerate(models_data.items()):
                y_true = data['y_true']
                y_pred_proba = data['y_pred_proba']
                
                # Calculate PR curve
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
                ap_score = average_precision_score(y_true, y_pred_proba)
                ap_scores[model_name] = ap_score
                
                # Add PR curve
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f'{model_name} (AP = {ap_score:.3f})',
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{model_name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>'
                ))
            
            # Add baseline (random classifier)
            baseline = sum(data['y_true']) / len(data['y_true'])  # Positive class ratio
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[baseline, baseline],
                mode='lines',
                name=f'Random Classifier (AP = {baseline:.3f})',
                line=dict(color='gray', width=2, dash='dash'),
                hovertemplate=f'Random Classifier<br>Recall: %{{x:.3f}}<br>Precision: {baseline:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': 'Precision-Recall Curves Comparison',
                    'x': 0.5,
                    'font': {'size': 20}
                },
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=800,
                height=600,
                legend=dict(
                    yanchor="top",
                    y=0.98,
                    xanchor="right",
                    x=0.98
                ),
                font=dict(size=12)
            )
            
            if save_plot:
                plot_path = self.plots_dir / "precision_recall_curves.html"
                fig.write_html(str(plot_path))
                
                if self.verbose:
                    logger.info(f"PR curves saved to {plot_path}")
            
            self.comparison_results['ap_scores'] = ap_scores
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting Precision-Recall curves: {e}")
            raise
    
    def calculate_feature_importance(self,
                                   model: Any,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_names: List[str],
                                   model_name: str = "Unknown",
                                   importance_type: str = "auto") -> FeatureImportanceResults:
        """
        Calculate feature importance using various methods.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            model_name: Name of the model
            importance_type: Type of importance ('gain', 'permutation', 'shap', 'auto')
            
        Returns:
            FeatureImportanceResults object
        """
        
        if self.verbose:
            logger.info(f"Calculating feature importance for {model_name} using {importance_type}")
        
        try:
            importance_scores = None
            actual_importance_type = importance_type
            
            # Auto-detect best importance method
            if importance_type == "auto":
                if hasattr(model, 'feature_importances_'):
                    actual_importance_type = "gain"
                else:
                    actual_importance_type = "permutation"
            
            # Calculate importance based on type
            if actual_importance_type == "gain" and hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                
            elif actual_importance_type == "permutation":
                perm_importance = permutation_importance(
                    model, X, y, 
                    n_repeats=10, 
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                importance_scores = perm_importance.importances_mean
                
            elif actual_importance_type == "shap" and self.shap_explainer:
                try:
                    shap_values = self.shap_explainer.calculate_shap_values(model, X)
                    importance_scores = np.abs(shap_values).mean(axis=0)
                except Exception as e:
                    logger.warning(f"SHAP calculation failed: {e}. Falling back to permutation importance.")
                    perm_importance = permutation_importance(
                        model, X, y,
                        n_repeats=10,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs
                    )
                    importance_scores = perm_importance.importances_mean
                    actual_importance_type = "permutation"
            
            else:
                # Fallback to permutation importance
                perm_importance = permutation_importance(
                    model, X, y,
                    n_repeats=10,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                importance_scores = perm_importance.importances_mean
                actual_importance_type = "permutation"
            
            # Create ranking
            ranking = np.argsort(importance_scores)[::-1]  # Descending order
            
            # Create results object
            results = FeatureImportanceResults(
                feature_names=feature_names,
                importance_scores=importance_scores,
                importance_type=actual_importance_type,
                model_name=model_name,
                ranking=ranking.tolist()
            )
            
            if self.verbose:
                self._print_feature_importance_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating feature importance for {model_name}: {e}")
            raise
    
    def plot_feature_importance(self,
                              importance_results: FeatureImportanceResults,
                              top_n: int = 20,
                              save_plot: bool = True) -> go.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_results: Feature importance results
            top_n: Number of top features to display
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        
        try:
            # Get top N features
            top_indices = importance_results.ranking[:top_n]
            top_features = [importance_results.feature_names[i] for i in top_indices]
            top_scores = [importance_results.importance_scores[i] for i in top_indices]
            
            # Create horizontal bar plot
            fig = go.Figure(go.Bar(
                x=top_scores[::-1],  # Reverse for ascending order
                y=top_features[::-1],
                orientation='h',
                marker=dict(
                    color=top_scores[::-1],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance Score")
                ),
                text=[f'{score:.4f}' for score in top_scores[::-1]],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': f'Top {top_n} Feature Importance - {importance_results.model_name}<br>({importance_results.importance_type.title()} Importance)',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title='Importance Score',
                yaxis_title='Features',
                width=800,
                height=max(400, top_n * 25),
                margin=dict(l=200),
                font=dict(size=10)
            )
            
            if save_plot:
                plot_path = self.plots_dir / f"feature_importance_{importance_results.model_name.lower().replace(' ', '_')}.html"
                fig.write_html(str(plot_path))
                
                if self.verbose:
                    logger.info(f"Feature importance plot saved to {plot_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            raise
    
    def cross_validation_scores(self,
                              model: Any,
                              X: np.ndarray,
                              y: np.ndarray,
                              scoring: Union[str, List[str]] = 'accuracy',
                              model_name: str = "Unknown") -> CrossValidationResults:
        """
        Perform comprehensive cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric(s) to use
            model_name: Name of the model
            
        Returns:
            CrossValidationResults object
        """
        
        if self.verbose:
            logger.info(f"Performing {self.cv_folds}-fold cross-validation for {model_name}")
        
        try:
            # Setup cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # Define multiple scoring metrics
            if isinstance(scoring, str):
                scoring_metrics = [scoring]
            else:
                scoring_metrics = scoring
            
            # Default comprehensive scoring
            if scoring == 'accuracy':
                scoring_dict = {
                    'accuracy': 'accuracy',
                    'precision': 'precision',
                    'recall': 'recall',
                    'f1': 'f1',
                    'roc_auc': 'roc_auc'
                }
            else:
                scoring_dict = {metric: metric for metric in scoring_metrics}
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring_dict,
                n_jobs=self.n_jobs,
                return_train_score=True,
                return_estimator=True
            )
            
            # Process results
            primary_metric = list(scoring_dict.keys())[0]
            test_scores = cv_results[f'test_{primary_metric}']
            
            # Calculate fold-wise detailed scores
            fold_scores = []
            for fold in range(self.cv_folds):
                fold_result = {}
                for metric in scoring_dict.keys():
                    fold_result[f'test_{metric}'] = cv_results[f'test_{metric}'][fold]
                    fold_result[f'train_{metric}'] = cv_results[f'train_{metric}'][fold]
                fold_scores.append(fold_result)
            
            # Find best and worst folds
            best_fold = int(np.argmax(test_scores))
            worst_fold = int(np.argmin(test_scores))
            
            # Create results object
            results = CrossValidationResults(
                cv_scores=test_scores,
                mean_score=np.mean(test_scores),
                std_score=np.std(test_scores),
                fold_scores=fold_scores,
                best_fold=best_fold,
                worst_fold=worst_fold
            )
            
            # Store detailed CV results
            self.evaluation_results[f"{model_name}_cv"] = {
                'cv_results': cv_results,
                'processed_results': results,
                'scoring_metrics': scoring_dict,
                'timestamp': datetime.now()
            }
            
            if self.verbose:
                self._print_cv_summary(results, model_name, primary_metric)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing cross-validation for {model_name}: {e}")
            raise
    
    def plot_cross_validation_results(self,
                                     cv_results: CrossValidationResults,
                                     model_name: str,
                                     metric_name: str = "accuracy",
                                     save_plot: bool = True) -> go.Figure:
        """
        Plot cross-validation results.
        
        Args:
            cv_results: Cross-validation results
            model_name: Name of the model
            metric_name: Name of the metric to plot
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        
        try:
            scores = cv_results.cv_scores
            folds = list(range(1, len(scores) + 1))
            
            fig = go.Figure()
            
            # Add individual fold scores
            fig.add_trace(go.Scatter(
                x=folds,
                y=scores,
                mode='markers+lines',
                name=f'{metric_name.title()} Scores',
                marker=dict(size=10, color='blue'),
                line=dict(color='blue', width=2),
                hovertemplate=f'<b>Fold %{{x}}</b><br>{metric_name.title()}: %{{y:.4f}}<extra></extra>'
            ))
            
            # Add mean line
            mean_score = cv_results.mean_score
            fig.add_hline(
                y=mean_score,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_score:.4f}"
            )
            
            # Add standard deviation band
            std_score = cv_results.std_score
            fig.add_trace(go.Scatter(
                x=folds + folds[::-1],
                y=[mean_score + std_score] * len(folds) + [mean_score - std_score] * len(folds),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'±1 Std Dev ({std_score:.4f})',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title={
                    'text': f'Cross-Validation Results - {model_name}<br>{metric_name.title()} across {len(folds)} folds',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                xaxis_title='Fold',
                yaxis_title=metric_name.title(),
                width=800,
                height=500,
                showlegend=True,
                font=dict(size=12)
            )
            
            if save_plot:
                plot_path = self.plots_dir / f"cv_results_{model_name.lower().replace(' ', '_')}.html"
                fig.write_html(str(plot_path))
                
                if self.verbose:
                    logger.info(f"CV results plot saved to {plot_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting cross-validation results: {e}")
            raise
    
    def plot_learning_curves(self,
                           model: Any,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_name: str = "Unknown",
                           scoring: str = 'accuracy',
                           train_sizes: Optional[np.ndarray] = None,
                           save_plot: bool = True) -> go.Figure:
        """
        Plot learning curves to analyze model performance vs training set size.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            model_name: Name of the model
            scoring: Scoring metric
            train_sizes: Training set sizes to evaluate
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        
        if self.verbose:
            logger.info(f"Generating learning curves for {model_name}")
        
        try:
            # Default training sizes
            if train_sizes is None:
                train_sizes = np.linspace(0.1, 1.0, 10)
            
            # Calculate learning curves
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=self.cv_folds,
                scoring=scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig = go.Figure()
            
            # Training scores
            fig.add_trace(go.Scatter(
                x=train_sizes_abs,
                y=train_mean,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
                hovertemplate='<b>Training</b><br>Size: %{x}<br>Score: %{y:.4f}<extra></extra>'
            ))
            
            # Training confidence interval
            fig.add_trace(go.Scatter(
                x=train_sizes_abs,
                y=train_mean + train_std,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes_abs,
                y=train_mean - train_std,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,0,255,0.2)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Validation scores
            fig.add_trace(go.Scatter(
                x=train_sizes_abs,
                y=val_mean,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='red', width=2),
                marker=dict(size=6),
                hovertemplate='<b>Validation</b><br>Size: %{x}<br>Score: %{y:.4f}<extra></extra>'
            ))
            
            # Validation confidence interval
            fig.add_trace(go.Scatter(
                x=train_sizes_abs,
                y=val_mean + val_std,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes_abs,
                y=val_mean - val_std,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title={
                    'text': f'Learning Curves - {model_name}',
                    'x': 0.5,
                    'font': {'size': 18}
                },
                xaxis_title='Training Set Size',
                yaxis_title=scoring.title(),
                width=800,
                height=600,
                showlegend=True,
                font=dict(size=12)
            )
            
            if save_plot:
                plot_path = self.plots_dir / f"learning_curves_{model_name.lower().replace(' ', '_')}.html"
                fig.write_html(str(plot_path))
                
                if self.verbose:
                    logger.info(f"Learning curves saved to {plot_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating learning curves for {model_name}: {e}")
            raise
    
    def model_calibration_plot(self,
                             y_true: np.ndarray,
                             y_pred_proba: np.ndarray,
                             model_name: str = "Unknown",
                             n_bins: int = 10,
                             save_plot: bool = True) -> go.Figure:
        """
        Plot model calibration (reliability diagram).
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            n_bins: Number of bins for calibration
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        
        try:
            # Calculate calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins
            )
            
            fig = go.Figure()
            
            # Calibration plot
            fig.add_trace(go.Scatter(
                x=mean_predicted_value,
                y=fraction_of_positives,
                mode='markers+lines',
                name=f'{model_name}',
                marker=dict(size=10, color='blue'),
                line=dict(color='blue', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Mean Predicted: %{x:.3f}<br>Fraction Positive: %{y:.3f}<extra></extra>'
            ))
            
            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Perfect Calibration<br>x = y<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': f'Calibration Plot - {model_name}',
                    'x': 0.5,
                    'font': {'size': 18}
                },
                xaxis_title='Mean Predicted Probability',
                yaxis_title='Fraction of Positives',
                width=600,
                height=600,
                showlegend=True,
                font=dict(size=12)
            )
            
            if save_plot:
                plot_path = self.plots_dir / f"calibration_{model_name.lower().replace(' ', '_')}.html"
                fig.write_html(str(plot_path))
                
                if self.verbose:
                    logger.info(f"Calibration plot saved to {plot_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating calibration plot for {model_name}: {e}")
            raise
    
    def compare_models(self,
                      models_results: Dict[str, EvaluationMetrics],
                      save_plot: bool = True) -> go.Figure:
        """
        Compare multiple models across different metrics.
        
        Args:
            models_results: Dictionary of model_name -> EvaluationMetrics
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure with model comparison
        """
        
        if self.verbose:
            logger.info(f"Comparing {len(models_results)} models")
        
        try:
            # Prepare data for comparison
            model_names = list(models_results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'balanced_accuracy']
            
            # Create radar chart for comparison
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, (model_name, results) in enumerate(models_results.items()):
                values = [
                    getattr(results, metric) for metric in metrics
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[m.replace('_', ' ').title() for m in metrics],
                    fill='toself',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.7
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title={
                    'text': 'Model Performance Comparison',
                    'x': 0.5,
                    'font': {'size': 20}
                },
                showlegend=True,
                width=800,
                height=800,
                font=dict(size=12)
            )
            
            # Also create bar chart comparison
            fig_bar = make_subplots(
                rows=2, cols=3,
                subplot_titles=[m.replace('_', ' ').title() for m in metrics],
                specs=[[{"type": "bar"}] * 3] * 2
            )
            
            positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
            
            for i, metric in enumerate(metrics):
                row, col = positions[i]
                values = [getattr(models_results[name], metric) for name in model_names]
                
                fig_bar.add_trace(
                    go.Bar(x=model_names, y=values, name=metric.title(), showlegend=False),
                    row=row, col=col
                )
            
            fig_bar.update_layout(
                title_text="Detailed Model Metrics Comparison",
                height=800,
                font=dict(size=10)
            )
            
            if save_plot:
                # Save radar chart
                radar_path = self.plots_dir / "model_comparison_radar.html"
                fig.write_html(str(radar_path))
                
                # Save bar chart
                bar_path = self.plots_dir / "model_comparison_bars.html"
                fig_bar.write_html(str(bar_path))
                
                if self.verbose:
                    logger.info(f"Model comparison plots saved to {self.plots_dir}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise
    
    def save_evaluation_report(self, filepath: Optional[str] = None) -> str:
        """
        Save comprehensive evaluation report to JSON file.
        
        Args:
            filepath: Path to save the report (optional)
            
        Returns:
            Path where the report was saved
        """
        
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.output_dir / f"evaluation_report_{timestamp}.json"
            
            # Prepare report data
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'evaluation_config': {
                    'cv_folds': self.cv_folds,
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs
                },
                'evaluation_results': {},
                'comparison_results': self.comparison_results
            }
            
            # Convert results to serializable format
            for model_name, results in self.evaluation_results.items():
                report_data['evaluation_results'][model_name] = {}
                
                for key, value in results.items():
                    if key == 'metrics' and hasattr(value, '__dict__'):
                        report_data['evaluation_results'][model_name][key] = value.__dict__
                    elif key == 'timestamp':
                        report_data['evaluation_results'][model_name][key] = value.isoformat()
                    elif isinstance(value, np.ndarray):
                        report_data['evaluation_results'][model_name][key] = value.tolist()
                    else:
                        report_data['evaluation_results'][model_name][key] = value
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            if self.verbose:
                logger.info(f"Evaluation report saved to {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
            raise
    
    # ================================================================
    # HELPER METHODS
    # ================================================================
    
    def _print_metrics_summary(self, metrics: EvaluationMetrics, model_name: str):
        """Print a summary of evaluation metrics."""
        print(f"\n{'='*50}")
        print(f"EVALUATION SUMMARY - {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy:          {metrics.accuracy:.4f}")
        print(f"Precision:         {metrics.precision:.4f}")
        print(f"Recall:            {metrics.recall:.4f}")
        print(f"F1-Score:          {metrics.f1_score:.4f}")
        print(f"ROC-AUC:           {metrics.roc_auc:.4f}")
        print(f"PR-AUC:            {metrics.precision_recall_auc:.4f}")
        print(f"Matthews Corr:     {metrics.matthews_corrcoef:.4f}")
        print(f"Cohen Kappa:       {metrics.cohen_kappa:.4f}")
        print(f"Balanced Accuracy: {metrics.balanced_accuracy:.4f}")
        print(f"Specificity:       {metrics.specificity:.4f}")
        print(f"NPV:               {metrics.npv:.4f}")
        if metrics.log_loss > 0:
            print(f"Log Loss:          {metrics.log_loss:.4f}")
            print(f"Brier Score:       {metrics.brier_score:.4f}")
        print(f"{'='*50}\n")
    
    def _print_feature_importance_summary(self, results: FeatureImportanceResults):
        """Print feature importance summary."""
        print(f"\nTOP 10 IMPORTANT FEATURES - {results.model_name} ({results.importance_type})")
        print(f"{'='*60}")
        
        for i in range(min(10, len(results.ranking))):
            idx = results.ranking[i]
            feature_name = results.feature_names[idx]
            importance = results.importance_scores[idx]
            print(f"{i+1:2d}. {feature_name:<30} {importance:.6f}")
        print(f"{'='*60}\n")
    
    def _print_cv_summary(self, results: CrossValidationResults, model_name: str, metric: str):
        """Print cross-validation summary."""
        print(f"\nCROSS-VALIDATION SUMMARY - {model_name}")
        print(f"{'='*50}")
        print(f"Metric: {metric.title()}")
        print(f"Mean Score:    {results.mean_score:.4f}")
        print(f"Std Deviation: {results.std_score:.4f}")
        print(f"Best Fold:     {results.best_fold + 1} ({results.cv_scores[results.best_fold]:.4f})")
        print(f"Worst Fold:    {results.worst_fold + 1} ({results.cv_scores[results.worst_fold]:.4f})")
        print(f"All Scores:    {[f'{score:.4f}' for score in results.cv_scores]}")
        print(f"{'='*50}\n")

# ================================================================
# EVALUATION RUNNER CLASS
# ================================================================

class EvaluationRunner:
    """
    High-level class to run comprehensive model evaluation pipeline.
    """
    
    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)
    
    def run_complete_evaluation(self,
                              models: Dict[str, Any],
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              feature_names: List[str]) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline for multiple models.
        
        Args:
            models: Dictionary of model_name -> trained_model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            
        Returns:
            Dictionary containing all evaluation results
        """
        
        self.logger.info("Starting complete model evaluation pipeline")
        
        results = {
            'metrics': {},
            'cross_validation': {},
            'feature_importance': {},
            'plots': {}
        }
        
        # Prepare models data for ROC/PR curves
        models_data = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test)
                    # Normalize to [0, 1]
                    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
                else:
                    y_pred_proba = None
                
                # Calculate metrics
                metrics = self.evaluator.calculate_accuracy_metrics(
                    y_test, y_pred, y_pred_proba, model_name
                )
                results['metrics'][model_name] = metrics
                
                # Generate confusion matrix
                cm, cm_fig = self.evaluator.generate_confusion_matrix(
                    y_test, y_pred, model_name
                )
                results['plots'][f'{model_name}_confusion_matrix'] = cm_fig
                
                # Cross-validation
                cv_results = self.evaluator.cross_validation_scores(
                    model, X_train, y_train, model_name=model_name
                )
                results['cross_validation'][model_name] = cv_results
                
                # Cross-validation plot
                cv_fig = self.evaluator.plot_cross_validation_results(
                    cv_results, model_name
                )
                results['plots'][f'{model_name}_cv'] = cv_fig
                
                # Feature importance
                importance_results = self.evaluator.calculate_feature_importance(
                    model, X_train, y_train, feature_names, model_name
                )
                results['feature_importance'][model_name] = importance_results
                
                # Feature importance plot
                importance_fig = self.evaluator.plot_feature_importance(
                    importance_results
                )
                results['plots'][f'{model_name}_feature_importance'] = importance_fig
                
                # Learning curves
                learning_fig = self.evaluator.plot_learning_curves(
                    model, X_train, y_train, model_name
                )
                results['plots'][f'{model_name}_learning_curves'] = learning_fig
                
                # Store data for ROC/PR curves
                if y_pred_proba is not None:
                    models_data[model_name] = {
                        'y_true': y_test,
                        'y_pred_proba': y_pred_proba
                    }
                
                # Model calibration
                if y_pred_proba is not None:
                    calibration_fig = self.evaluator.model_calibration_plot(
                        y_test, y_pred_proba, model_name
                    )
                    results['plots'][f'{model_name}_calibration'] = calibration_fig
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        # Generate comparison plots
        if len(models_data) > 1:
            # ROC curves
            roc_fig = self.evaluator.plot_roc_curves(models_data)
            results['plots']['roc_comparison'] = roc_fig
            
            # PR curves
            pr_fig = self.evaluator.plot_precision_recall_curves(models_data)
            results['plots']['pr_comparison'] = pr_fig
        
        # Model comparison
        if len(results['metrics']) > 1:
            comparison_fig = self.evaluator.compare_models(results['metrics'])
            results['plots']['model_comparison'] = comparison_fig
        
        # Save comprehensive report
        report_path = self.evaluator.save_evaluation_report()
        results['report_path'] = report_path
        
        self.logger.info("Complete model evaluation pipeline finished")
        
        return results

# ================================================================
# EXAMPLE USAGE AND TESTING
# ================================================================

def example_usage():
    """Example of how to use the ModelEvaluator."""
    
    # This would typically be imported and used in other scripts
    print("ModelEvaluator Example Usage:")
    print("="*50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(cv_folds=5, verbose=True)
    
    # Example with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train models
    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(random_state=42)
    
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    models = {
        'Random Forest': rf_model,
        'Logistic Regression': lr_model
    }
    
    # Run complete evaluation
    runner = EvaluationRunner(evaluator)
    results = runner.run_complete_evaluation(
        models, X_train, y_train, X_test, y_test, feature_names
    )
    
    print(f"Evaluation completed! Results saved to: {results['report_path']}")
    print(f"Generated {len(results['plots'])} visualization plots")
    
    return results

if __name__ == "__main__":
    example_usage()
