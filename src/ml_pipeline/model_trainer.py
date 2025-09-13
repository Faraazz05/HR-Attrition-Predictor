"""
HR Attrition Predictor - Core ML Training Engine
===============================================
Comprehensive machine learning model training pipeline with multiple algorithms,
hyperparameter optimization, ensemble methods, and production-ready model persistence.

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
import joblib
import pickle
from dataclasses import dataclass, asdict
import json

# Core ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# LightGBM  
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Hyperparameter optimization
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    log_loss, average_precision_score
)

# Advanced optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainingConfig:
    """Configuration for model training parameters"""
    # General training settings
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    scoring_metric: str = 'roc_auc'
    
    # Models to train
    train_logistic: bool = True
    train_xgboost: bool = True
    train_random_forest: bool = True
    train_svm: bool = True
    train_gradient_boosting: bool = True
    train_neural_network: bool = True
    train_ensemble: bool = True
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = True
    tuning_method: str = 'random'  # 'grid', 'random', 'bayesian'
    n_iter: int = 50
    tuning_cv_folds: int = 3
    
    # Model persistence
    save_models: bool = True
    model_save_path: str = "models/trained_models"
    save_preprocessing_info: bool = True
    
    # Performance thresholds
    min_accuracy: float = 0.75
    min_roc_auc: float = 0.80


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    log_loss: float
    average_precision: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float


@dataclass
class ModelTrainingReport:
    """Comprehensive training report"""
    total_models_trained: int
    best_model_name: str
    best_model_score: float
    model_performances: List[ModelPerformance]
    ensemble_performance: Optional[ModelPerformance]
    hyperparameter_results: Dict[str, Any]
    training_dataset_info: Dict[str, Any]
    total_training_time: float
    timestamp: str


class ModelTrainer:
    """
    Comprehensive ML model training engine for HR attrition prediction.
    
    Supports multiple algorithms, hyperparameter optimization, ensemble methods,
    and production-ready model persistence with extensive performance tracking.
    """
    
    def __init__(self, config: Optional[ModelTrainingConfig] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Optional model training configuration
        """
        self.config = config if config else ModelTrainingConfig()
        self.trained_models_: Dict[str, Any] = {}
        self.model_performances_: Dict[str, ModelPerformance] = {}
        self.best_model_: Optional[Tuple[str, Any]] = None
        self.ensemble_model_: Optional[VotingClassifier] = None
        self.hyperparameter_results_: Dict[str, Any] = {}
        self.training_report_: Optional[ModelTrainingReport] = None
        
        # Training data storage
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None
        
        logger.info("ModelTrainer initialized with configuration")
    
    def fit(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
            y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train all configured models and return trained models dictionary.
        
        Args:
            X_train: Training features
            X_test: Testing features  
            y_train: Training targets
            y_test: Testing targets
            
        Returns:
            Dictionary of trained models
        """
        start_time = datetime.now()
        logger.info("Starting comprehensive model training pipeline...")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Testing set shape: {X_test.shape}")
        
        # Store training data
        self.X_train_ = X_train
        self.X_test_ = X_test
        self.y_train_ = y_train
        self.y_test_ = y_test
        
        # Train individual models
        if self.config.train_logistic:
            self.train_logistic_regression()
        
        if self.config.train_xgboost and XGBOOST_AVAILABLE:
            self.train_xgboost_classifier()
        
        if self.config.train_random_forest:
            self.train_random_forest()
        
        if self.config.train_svm:
            self.train_svm_classifier()
        
        if self.config.train_gradient_boosting:
            self.train_gradient_boosting()
        
        if self.config.train_neural_network:
            self.train_neural_network()
        
        # Train ensemble models
        if self.config.train_ensemble and len(self.trained_models_) >= 2:
            self.train_ensemble_models()
        
        # Determine best model
        self._select_best_model()
        
        # Create training report
        total_time = (datetime.now() - start_time).total_seconds()
        self._create_training_report(total_time)
        
        # Save models if configured
        if self.config.save_models:
            self.save_trained_models()
        
        logger.info(f"Model training completed in {total_time:.2f} seconds")
        logger.info(f"Trained {len(self.trained_models_)} models")
        logger.info(f"Best model: {self.best_model_[0] if self.best_model_ else 'None'}")
        
        return self.trained_models_
    
    def train_logistic_regression(self) -> LogisticRegression:
        """
        Train Logistic Regression classifier with hyperparameter tuning.
        
        Returns:
            Trained LogisticRegression model
        """
        logger.info("Training Logistic Regression...")
        start_time = datetime.now()
        
        # Base model
        base_model = LogisticRegression(
            random_state=self.config.random_state,
            max_iter=1000
        )
        
        # Hyperparameter tuning
        if self.config.enable_hyperparameter_tuning:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced']
            }
            
            best_model = self._tune_hyperparameters(
                base_model, param_grid, "Logistic Regression"
            )
        else:
            best_model = base_model
            best_model.fit(self.X_train_, self.y_train_)
        
        # Evaluate model
        performance = self._evaluate_model(best_model, "Logistic Regression", start_time)
        
        # Store results
        self.trained_models_["Logistic Regression"] = best_model
        self.model_performances_["Logistic Regression"] = performance
        
        logger.info(f"Logistic Regression - ROC AUC: {performance.roc_auc:.4f}")
        return best_model
    
    def train_xgboost_classifier(self) -> Optional[xgb.XGBClassifier]:
        """
        Train XGBoost classifier with hyperparameter tuning.
        
        Returns:
            Trained XGBClassifier model or None if XGBoost not available
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping...")
            return None
        
        logger.info("Training XGBoost Classifier...")
        start_time = datetime.now()
        
        # Base model
        base_model = xgb.XGBClassifier(
            random_state=self.config.random_state,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Hyperparameter tuning
        if self.config.enable_hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 1.1, 1.2]
            }
            
            best_model = self._tune_hyperparameters(
                base_model, param_grid, "XGBoost"
            )
        else:
            best_model = base_model
            best_model.fit(self.X_train_, self.y_train_)
        
        # Evaluate model
        performance = self._evaluate_model(best_model, "XGBoost", start_time)
        
        # Store results
        self.trained_models_["XGBoost"] = best_model
        self.model_performances_["XGBoost"] = performance
        
        logger.info(f"XGBoost - ROC AUC: {performance.roc_auc:.4f}")
        return best_model
    
    def train_random_forest(self) -> RandomForestClassifier:
        """
        Train Random Forest classifier with hyperparameter tuning.
        
        Returns:
            Trained RandomForestClassifier model
        """
        logger.info("Training Random Forest...")
        start_time = datetime.now()
        
        # Base model
        base_model = RandomForestClassifier(
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        # Hyperparameter tuning
        if self.config.enable_hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }
            
            best_model = self._tune_hyperparameters(
                base_model, param_grid, "Random Forest"
            )
        else:
            best_model = base_model
            best_model.fit(self.X_train_, self.y_train_)
        
        # Evaluate model
        performance = self._evaluate_model(best_model, "Random Forest", start_time)
        
        # Store results
        self.trained_models_["Random Forest"] = best_model
        self.model_performances_["Random Forest"] = performance
        
        logger.info(f"Random Forest - ROC AUC: {performance.roc_auc:.4f}")
        return best_model
    
    def train_svm_classifier(self) -> SVC:
        """
        Train Support Vector Machine classifier.
        
        Returns:
            Trained SVC model
        """
        logger.info("Training SVM Classifier...")
        start_time = datetime.now()
        
        # Base model
        base_model = SVC(
            random_state=self.config.random_state,
            probability=True  # Enable probability estimates
        )
        
        # Hyperparameter tuning (lighter for SVM due to computational cost)
        if self.config.enable_hyperparameter_tuning:
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'class_weight': [None, 'balanced']
            }
            
            # Use fewer iterations for SVM due to computational cost
            best_model = self._tune_hyperparameters(
                base_model, param_grid, "SVM", n_iter=min(20, self.config.n_iter)
            )
        else:
            best_model = base_model
            best_model.fit(self.X_train_, self.y_train_)
        
        # Evaluate model
        performance = self._evaluate_model(best_model, "SVM", start_time)
        
        # Store results
        self.trained_models_["SVM"] = best_model
        self.model_performances_["SVM"] = performance
        
        logger.info(f"SVM - ROC AUC: {performance.roc_auc:.4f}")
        return best_model
    
    def train_gradient_boosting(self) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting classifier.
        
        Returns:
            Trained GradientBoostingClassifier model
        """
        logger.info("Training Gradient Boosting...")
        start_time = datetime.now()
        
        # Base model
        base_model = GradientBoostingClassifier(
            random_state=self.config.random_state
        )
        
        # Hyperparameter tuning
        if self.config.enable_hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            best_model = self._tune_hyperparameters(
                base_model, param_grid, "Gradient Boosting"
            )
        else:
            best_model = base_model
            best_model.fit(self.X_train_, self.y_train_)
        
        # Evaluate model
        performance = self._evaluate_model(best_model, "Gradient Boosting", start_time)
        
        # Store results
        self.trained_models_["Gradient Boosting"] = best_model
        self.model_performances_["Gradient Boosting"] = performance
        
        logger.info(f"Gradient Boosting - ROC AUC: {performance.roc_auc:.4f}")
        return best_model
    
    def train_neural_network(self) -> MLPClassifier:
        """
        Train Multi-layer Perceptron neural network.
        
        Returns:
            Trained MLPClassifier model
        """
        logger.info("Training Neural Network...")
        start_time = datetime.now()
        
        # Base model
        base_model = MLPClassifier(
            random_state=self.config.random_state,
            max_iter=300
        )
        
        # Hyperparameter tuning
        if self.config.enable_hyperparameter_tuning:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50), (100, 50, 25)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
            
            best_model = self._tune_hyperparameters(
                base_model, param_grid, "Neural Network"
            )
        else:
            best_model = base_model
            best_model.fit(self.X_train_, self.y_train_)
        
        # Evaluate model
        performance = self._evaluate_model(best_model, "Neural Network", start_time)
        
        # Store results
        self.trained_models_["Neural Network"] = best_model
        self.model_performances_["Neural Network"] = performance
        
        logger.info(f"Neural Network - ROC AUC: {performance.roc_auc:.4f}")
        return best_model
    
    def train_ensemble_models(self) -> VotingClassifier:
        """
        Train ensemble models using voting and stacking approaches.
        
        Returns:
            Trained ensemble model
        """
        logger.info("Training Ensemble Models...")
        start_time = datetime.now()
        
        if len(self.trained_models_) < 2:
            logger.warning("Need at least 2 models for ensemble. Skipping...")
            return None
        
        # Select best performing models for ensemble
        model_scores = {name: perf.roc_auc for name, perf in self.model_performances_.items()}
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top 3-5 models for ensemble
        top_models = sorted_models[:min(5, len(sorted_models))]
        logger.info(f"Using top {len(top_models)} models for ensemble: {[name for name, _ in top_models]}")
        
        # Create voting classifier
        voting_models = [(name, self.trained_models_[name]) for name, _ in top_models]
        
        # Hard and soft voting ensembles
        ensemble_models = {}
        
        # Soft voting (uses probabilities)
        soft_ensemble = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        soft_ensemble.fit(self.X_train_, self.y_train_)
        ensemble_models["Soft Voting"] = soft_ensemble
        
        # Hard voting (uses predictions)
        hard_ensemble = VotingClassifier(
            estimators=voting_models,
            voting='hard'
        )
        hard_ensemble.fit(self.X_train_, self.y_train_)
        ensemble_models["Hard Voting"] = hard_ensemble
        
        # Evaluate ensemble models
        best_ensemble = None
        best_ensemble_score = 0
        
        for ensemble_name, ensemble_model in ensemble_models.items():
            performance = self._evaluate_model(ensemble_model, ensemble_name, start_time)
            
            if performance.roc_auc > best_ensemble_score:
                best_ensemble = ensemble_model
                best_ensemble_score = performance.roc_auc
                best_ensemble_name = ensemble_name
            
            self.trained_models_[ensemble_name] = ensemble_model
            self.model_performances_[ensemble_name] = performance
            
            logger.info(f"{ensemble_name} - ROC AUC: {performance.roc_auc:.4f}")
        
        # Store best ensemble
        self.ensemble_model_ = best_ensemble
        
        logger.info(f"Best ensemble: {best_ensemble_name} with ROC AUC: {best_ensemble_score:.4f}")
        return best_ensemble
    
    def _tune_hyperparameters(self, model: Any, param_grid: Dict[str, List], 
                             model_name: str, n_iter: Optional[int] = None) -> Any:
        """
        Tune hyperparameters using configured method.
        
        Args:
            model: Base model to tune
            param_grid: Parameter grid for tuning
            model_name: Name of the model for logging
            n_iter: Number of iterations (for RandomizedSearchCV)
            
        Returns:
            Best tuned model
        """
        logger.info(f"Tuning hyperparameters for {model_name} using {self.config.tuning_method} method...")
        
        cv = StratifiedKFold(n_splits=self.config.tuning_cv_folds, shuffle=True, 
                            random_state=self.config.random_state)
        
        try:
            if self.config.tuning_method == 'grid':
                search = GridSearchCV(
                    model, param_grid,
                    cv=cv,
                    scoring=self.config.scoring_metric,
                    n_jobs=-1,
                    verbose=0
                )
            elif self.config.tuning_method == 'random':
                n_iter_param = n_iter or self.config.n_iter
                search = RandomizedSearchCV(
                    model, param_grid,
                    n_iter=n_iter_param,
                    cv=cv,
                    scoring=self.config.scoring_metric,
                    n_jobs=-1,
                    random_state=self.config.random_state,
                    verbose=0
                )
            elif self.config.tuning_method == 'bayesian' and OPTUNA_AVAILABLE:
                # Optuna-based Bayesian optimization
                best_model = self._bayesian_optimization(model, param_grid, model_name, cv)
                return best_model
            else:
                # Fallback to random search
                search = RandomizedSearchCV(
                    model, param_grid,
                    n_iter=self.config.n_iter,
                    cv=cv,
                    scoring=self.config.scoring_metric,
                    n_jobs=-1,
                    random_state=self.config.random_state,
                    verbose=0
                )
            
            search.fit(self.X_train_, self.y_train_)
            
            # Store hyperparameter results
            self.hyperparameter_results_[model_name] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
            
            logger.info(f"{model_name} best score: {search.best_score_:.4f}")
            logger.info(f"{model_name} best params: {search.best_params_}")
            
            return search.best_estimator_
            
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed for {model_name}: {e}")
            logger.info(f"Training {model_name} with default parameters...")
            model.fit(self.X_train_, self.y_train_)
            return model
    
    def _bayesian_optimization(self, model: Any, param_grid: Dict[str, List], 
                              model_name: str, cv: Any) -> Any:
        """Bayesian hyperparameter optimization using Optuna"""
        def objective(trial):
            params = {}
            for param, values in param_grid.items():
                if isinstance(values[0], int):
                    params[param] = trial.suggest_int(param, min(values), max(values))
                elif isinstance(values[0], float):
                    params[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    params[param] = trial.suggest_categorical(param, values)
            
            model_copy = model.__class__(**params)
            scores = cross_val_score(model_copy, self.X_train_, self.y_train_, 
                                   cv=cv, scoring=self.config.scoring_metric)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_iter)
        
        # Train best model
        best_model = model.__class__(**study.best_params)
        best_model.fit(self.X_train_, self.y_train_)
        
        # Store results
        self.hyperparameter_results_[model_name] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
        
        return best_model
    
    def _evaluate_model(self, model: Any, model_name: str, start_time: datetime) -> ModelPerformance:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model to evaluate
            model_name: Name of the model
            start_time: Training start time
            
        Returns:
            ModelPerformance object with metrics
        """
        # Predictions
        pred_start = datetime.now()
        y_pred = model.predict(self.X_test_)
        y_pred_proba = model.predict_proba(self.X_test_)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        prediction_time = (datetime.now() - pred_start).total_seconds()
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test_, y_pred)
        precision = precision_score(self.y_test_, y_pred, average='binary')
        recall = recall_score(self.y_test_, y_pred, average='binary')
        f1 = f1_score(self.y_test_, y_pred, average='binary')
        roc_auc = roc_auc_score(self.y_test_, y_pred_proba)
        log_loss_score = log_loss(self.y_test_, y_pred_proba)
        avg_precision = average_precision_score(self.y_test_, y_pred_proba)
        
        # Cross-validation scores
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                           random_state=self.config.random_state)
        cv_scores = cross_val_score(model, self.X_train_, self.y_train_, 
                                  cv=cv, scoring=self.config.scoring_metric)
        
        # Training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelPerformance(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            log_loss=log_loss_score,
            average_precision=avg_precision,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            training_time=training_time,
            prediction_time=prediction_time
        )
    
    def _select_best_model(self) -> None:
        """Select the best performing model based on ROC AUC"""
        if not self.model_performances_:
            logger.warning("No models trained, cannot select best model")
            return
        
        best_score = 0
        best_name = None
        
        for name, performance in self.model_performances_.items():
            if performance.roc_auc > best_score:
                best_score = performance.roc_auc
                best_name = name
        
        if best_name:
            self.best_model_ = (best_name, self.trained_models_[best_name])
            logger.info(f"Best model selected: {best_name} with ROC AUC: {best_score:.4f}")
    
    def _create_training_report(self, total_time: float) -> None:
        """Create comprehensive training report"""
        performances = list(self.model_performances_.values())
        best_model_name = self.best_model_[0] if self.best_model_ else "None"
        best_model_score = self.best_model_[1] if self.best_model_ else 0
        
        # Find ensemble performance
        ensemble_performance = None
        for name, perf in self.model_performances_.items():
            if 'Voting' in name or 'Ensemble' in name:
                ensemble_performance = perf
                break
        
        self.training_report_ = ModelTrainingReport(
            total_models_trained=len(self.trained_models_),
            best_model_name=best_model_name,
            best_model_score=best_model_score,
            model_performances=performances,
            ensemble_performance=ensemble_performance,
            hyperparameter_results=self.hyperparameter_results_,
            training_dataset_info={
                'train_shape': self.X_train_.shape,
                'test_shape': self.X_test_.shape,
                'features': self.X_train_.columns.tolist(),
                'target_distribution': self.y_train_.value_counts().to_dict()
            },
            total_training_time=total_time,
            timestamp=datetime.now().isoformat()
        )
    
    def hyperparameter_tuning(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run hyperparameter tuning for specific models.
        
        Args:
            model_names: List of model names to tune (None for all)
            
        Returns:
            Dictionary of tuning results
        """
        if not self.trained_models_:
            logger.error("No models trained. Train models first.")
            return {}
        
        models_to_tune = model_names or list(self.trained_models_.keys())
        tuning_results = {}
        
        for model_name in models_to_tune:
            if model_name in self.trained_models_:
                logger.info(f"Re-tuning hyperparameters for {model_name}...")
                
                # Get base model and parameter grid based on model type
                model = self.trained_models_[model_name]
                param_grid = self._get_param_grid_for_model(model_name)
                
                if param_grid:
                    tuned_model = self._tune_hyperparameters(model, param_grid, model_name)
                    self.trained_models_[model_name] = tuned_model
                    
                    # Re-evaluate
                    start_time = datetime.now()
                    performance = self._evaluate_model(tuned_model, model_name, start_time)
                    self.model_performances_[model_name] = performance
                    
                    tuning_results[model_name] = self.hyperparameter_results_.get(model_name, {})
        
        # Update best model selection
        self._select_best_model()
        
        return tuning_results
    
    def _get_param_grid_for_model(self, model_name: str) -> Dict[str, List]:
        """Get parameter grid for specific model type"""
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        return param_grids.get(model_name, {})
    
    def save_trained_models(self, custom_path: Optional[str] = None) -> Dict[str, str]:
        """
        Save all trained models to disk.
        
        Args:
            custom_path: Custom save path (optional)
            
        Returns:
            Dictionary mapping model names to file paths
        """
        save_path = Path(custom_path) if custom_path else Path(self.config.model_save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Saving {len(self.trained_models_)} trained models...")
        
        for model_name, model in self.trained_models_.items():
            # Clean filename
            filename = model_name.replace(' ', '_').lower()
            file_path = save_path / f"{filename}_{timestamp}.pkl"
            
            try:
                # Save model using joblib
                joblib.dump(model, file_path)
                saved_paths[model_name] = str(file_path)
                logger.info(f"Saved {model_name} to {file_path}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")
        
        # Save training report
        if self.training_report_:
            report_path = save_path / f"training_report_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(asdict(self.training_report_), f, indent=2, default=str)
            saved_paths['training_report'] = str(report_path)
            logger.info(f"Saved training report to {report_path}")
        
        # Save model performances
        performance_path = save_path / f"model_performances_{timestamp}.json"
        performances_dict = {name: asdict(perf) for name, perf in self.model_performances_.items()}
        with open(performance_path, 'w') as f:
            json.dump(performances_dict, f, indent=2, default=str)
        saved_paths['performances'] = str(performance_path)
        
        logger.info(f"All models saved successfully to {save_path}")
        return saved_paths
    
    def load_trained_model(self, model_path: str, model_name: Optional[str] = None) -> Any:
        """Load a trained model from disk"""
        try:
            model = joblib.load(model_path)
            name = model_name or Path(model_path).stem
            self.trained_models_[name] = model
            logger.info(f"Loaded model {name} from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions using trained model"""
        if not self.trained_models_:
            raise ValueError("No trained models available")
        
        # Use best model if not specified
        if model_name is None:
            if self.best_model_:
                model = self.best_model_[1]
                model_name = self.best_model_[0]
            else:
                model_name = list(self.trained_models_.keys())[0]
                model = self.trained_models_[model_name]
        else:
            if model_name not in self.trained_models_:
                raise ValueError(f"Model {model_name} not found")
            model = self.trained_models_[model_name]
        
        logger.info(f"Making predictions using {model_name}")
        return model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Make probability predictions using trained model"""
        if not self.trained_models_:
            raise ValueError("No trained models available")
        
        # Use best model if not specified
        if model_name is None:
            if self.best_model_:
                model = self.best_model_[1]
                model_name = self.best_model_[0]
            else:
                model_name = list(self.trained_models_.keys())[0]
                model = self.trained_models_[model_name]
        else:
            if model_name not in self.trained_models_:
                raise ValueError(f"Model {model_name} not found")
            model = self.trained_models_[model_name]
        
        if not hasattr(model, 'predict_proba'):
            logger.warning(f"Model {model_name} does not support probability prediction")
            return self.predict(X, model_name)
        
        logger.info(f"Making probability predictions using {model_name}")
        return model.predict_proba(X)
    
    def get_model_performance(self, model_name: Optional[str] = None) -> Union[ModelPerformance, Dict[str, ModelPerformance]]:
        """Get performance metrics for models"""
        if model_name:
            return self.model_performances_.get(model_name)
        return self.model_performances_
    
    def get_training_report(self) -> Optional[ModelTrainingReport]:
        """Get comprehensive training report"""
        return self.training_report_
    
    def get_best_model(self) -> Optional[Tuple[str, Any]]:
        """Get the best performing model"""
        return self.best_model_


# Convenience functions
def train_hr_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series,
                   config: Optional[ModelTrainingConfig] = None) -> Tuple[Dict[str, Any], ModelTrainer]:
    """Quick function to train HR models with default settings"""
    trainer = ModelTrainer(config)
    trained_models = trainer.fit(X_train, X_test, y_train, y_test)
    return trained_models, trainer


def quick_model_comparison(X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    """Quick comparison of model performances"""
    config = ModelTrainingConfig(enable_hyperparameter_tuning=False)
    trainer = ModelTrainer(config)
    trainer.fit(X_train, X_test, y_train, y_test)
    
    performances = {}
    for name, perf in trainer.model_performances_.items():
        performances[name] = perf.roc_auc
    
    return performances


if __name__ == "__main__":
    # Test the model trainer
    print("🧪 Testing ModelTrainer...")
    
    # This would test with actual data
    # from src.data_processing.data_loader import load_hr_data
    # from src.data_processing.preprocessor import preprocess_hr_data
    # 
    # data, _ = load_hr_data("data/synthetic/hr_employees.csv")
    # X, y, preprocessor = preprocess_hr_data(data)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 
    # trainer = ModelTrainer()
    # trained_models = trainer.fit(X_train, X_test, y_train, y_test)
    # 
    # report = trainer.get_training_report()
    # print(f"Best model: {report.best_model_name} ({report.best_model_score:.4f})")
    
    print("✅ ModelTrainer test completed!")
