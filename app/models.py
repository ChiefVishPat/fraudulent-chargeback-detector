"""
ML Models module for fraud detection.
Implements RandomForest, LightGBM, and XGBoost with proper handling for imbalanced data.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score,
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, log_loss, brier_score_loss
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
# Import optional ML libraries with fallbacks
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False
import matplotlib.pyplot as plt
import seaborn as sns
import structlog

logger = structlog.get_logger()


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class FraudDetectionModel:
    """Base class for fraud detection models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.best_threshold = 0.5
        self.metrics = {}
        self.feature_importances = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train the model - to be implemented by subclasses."""
        raise NotImplementedError
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Make predictions with optimal threshold."""
        proba = self.predict_proba(X)
        threshold = threshold or self.best_threshold
        return (proba >= threshold).astype(int)
    
    def find_optimal_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Find optimal threshold for fraud detection with business-focused approach."""
        logger.info("Finding optimal threshold for fraud detection", model=self.model_name)
        
        try:
            y_proba = self.predict_proba(X_val)
            
            # Use a simpler approach that's more robust to array dimension issues
            # Test a range of thresholds from 0.05 to 0.5 (fraud detection focused)
            thresholds_to_test = np.arange(0.05, 0.51, 0.01)
            
            best_threshold = 0.1
            best_score = 0
            best_recall = 0
            best_precision = 0
            best_f1 = 0
            
            for threshold in thresholds_to_test:
                y_pred = (y_proba >= threshold).astype(int)
                
                # Calculate metrics
                tp = np.sum((y_pred == 1) & (y_val == 1))
                fp = np.sum((y_pred == 1) & (y_val == 0))
                fn = np.sum((y_pred == 0) & (y_val == 1))
                
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0
                    
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0
                    
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
                
                # For fraud detection, prioritize recall (catching fraud) with reasonable precision
                # Require at least 10% precision and prioritize recall
                if precision >= 0.10 and recall >= 0.75:
                    # Score combines high recall with reasonable precision
                    score = 0.7 * recall + 0.3 * precision
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                        best_recall = recall
                        best_precision = precision
                        best_f1 = f1
            
            # If no threshold meets our criteria, find the one with best recall at 5% precision
            if best_score == 0:
                for threshold in thresholds_to_test:
                    y_pred = (y_proba >= threshold).astype(int)
                    
                    tp = np.sum((y_pred == 1) & (y_val == 1))
                    fp = np.sum((y_pred == 1) & (y_val == 0))
                    fn = np.sum((y_pred == 0) & (y_val == 1))
                    
                    if tp + fp > 0:
                        precision = tp / (tp + fp)
                    else:
                        precision = 0
                        
                    if tp + fn > 0:
                        recall = tp / (tp + fn)
                    else:
                        recall = 0
                        
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0
                    
                    if precision >= 0.05 and recall > best_recall:
                        best_recall = recall
                        best_precision = precision
                        best_f1 = f1
                        best_threshold = threshold
            
            # Final fallback
            if best_threshold == 0.1 and best_recall == 0:
                best_threshold = 0.2  # Conservative default for fraud detection
                y_pred = (y_proba >= best_threshold).astype(int)
                tp = np.sum((y_pred == 1) & (y_val == 1))
                fp = np.sum((y_pred == 1) & (y_val == 0))
                fn = np.sum((y_pred == 0) & (y_val == 1))
                
                best_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                best_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                best_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall) if (best_precision + best_recall) > 0 else 0
            
            logger.info("Fraud detection threshold found",
                       threshold=best_threshold,
                       f1_score=best_f1,
                       precision=best_precision,
                       recall=best_recall,
                       strategy="fraud_detection_grid_search")
            
            self.best_threshold = best_threshold
            return best_threshold
            
        except Exception as e:
            logger.error("Error in threshold optimization, using default", error=str(e))
            self.best_threshold = 0.2
            return 0.2
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, 
                X_train: pd.DataFrame = None, y_train: pd.Series = None) -> Dict:
        """Comprehensive model evaluation with imbalanced data metrics."""
        logger.info("Evaluating model with imbalanced data metrics", model=self.model_name)
        
        # Predictions
        y_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)
        
        # Core metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Imbalanced data specific metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)  # Matthews Correlation Coefficient
        
        # Probability calibration metrics
        try:
            log_loss_score = log_loss(y_test, y_proba)
            brier_score = brier_score_loss(y_test, y_proba)
        except:
            log_loss_score = float('inf')
            brier_score = float('inf')
        
        # Confusion matrix and derived metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics for imbalanced data
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Geometric mean of sensitivity and specificity
        g_mean = np.sqrt(recall * specificity) if (recall * specificity) >= 0 else 0
        
        # Cross-validation score if training data provided
        cv_scores = None
        if X_train is not None and y_train is not None:
            try:
                cv_scores = cross_val_score(
                    self.model, X_train, y_train, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='average_precision'
                )
            except:
                cv_scores = None
        
        metrics = {
            # Primary metrics (most important for imbalanced data)
            'pr_auc': float(pr_auc),  # Most important for imbalanced data
            'roc_auc': float(roc_auc),
            'f1_score': float(f1),
            'balanced_accuracy': float(balanced_acc),
            'mcc': float(mcc),  # Matthews Correlation Coefficient
            
            # Classification metrics
            'precision': float(precision),
            'recall': float(recall),  # Sensitivity
            'specificity': float(specificity),
            'npv': float(npv),
            'g_mean': float(g_mean),
            
            # Error rates
            'fpr': float(fpr),  # False Positive Rate
            'fnr': float(fnr),  # False Negative Rate
            
            # Probability calibration
            'log_loss': float(log_loss_score),
            'brier_score': float(brier_score),
            
            # Threshold and confusion matrix
            'best_threshold': convert_numpy_types(self.best_threshold),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            },
            
            # Additional reports
            'classification_report': convert_numpy_types(
                classification_report(y_test, y_pred, output_dict=True)
            ),
            'cv_scores': convert_numpy_types(cv_scores) if cv_scores is not None else None
        }
        
        self.metrics = metrics
        
        logger.info("Model evaluation completed",
                   model=self.model_name,
                   pr_auc=pr_auc,
                   roc_auc=roc_auc,
                   f1_score=f1,
                   balanced_accuracy=balanced_acc,
                   mcc=mcc)
        
        return metrics
    
    def plot_curves(self, X_test: pd.DataFrame, y_test: pd.Series, save_dir: str):
        """Generate and save ROC and PR curves in organized plots subdirectory."""
        y_proba = self.predict_proba(X_test)
        
        # Create organized plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        plt.figure(figsize=(15, 5))
        
        # ROC Curve
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, label=f'{self.model_name} (AUC = {self.metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, label=f'{self.model_name} (AUC = {self.metrics["pr_auc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix
        plt.subplot(1, 3, 3)
        cm = np.array([[self.metrics['confusion_matrix']['tn'], self.metrics['confusion_matrix']['fp']],
                      [self.metrics['confusion_matrix']['fn'], self.metrics['confusion_matrix']['tp']]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Fraud', 'Fraud'],
                   yticklabels=['Not Fraud', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.tight_layout()
        
        # Save plots in organized structure
        plot_path = os.path.join(plots_dir, f'{self.model_name.lower()}_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Plots saved", model=self.model_name, plot_path=plot_path)
    
    def save_model(self, save_dir: str):
        """Save model and artifacts in organized models subdirectory."""
        # Create organized models directory
        models_dir = os.path.join(save_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model with clean naming (no timestamp in filename)
        model_path = os.path.join(models_dir, f'{self.model_name.lower()}.joblib')
        joblib.dump({
            'model': self.model,
            'best_threshold': self.best_threshold,
            'feature_importances': self.feature_importances,
            'model_name': self.model_name,
            'run_id': self.run_id,
            'saved_timestamp': datetime.now().isoformat()
        }, model_path)
        
        logger.info("Model saved", model=self.model_name, path=model_path)
        return model_path
    
    def load_model(self, model_path: str):
        """Load saved model."""
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.best_threshold = artifacts['best_threshold']
        self.feature_importances = artifacts.get('feature_importances')
        self.run_id = artifacts.get('run_id', 'loaded')
        
        logger.info("Model loaded", model=self.model_name, path=model_path)


class RandomForestFraudModel(FraudDetectionModel):
    """Random Forest model for fraud detection."""
    
    def __init__(self):
        super().__init__("RandomForest")
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train Random Forest model."""
        logger.info("Training Random Forest model")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Feature importances
        self.feature_importances = dict(zip(X_train.columns, self.model.feature_importances_))
        
        # Find optimal threshold
        self.find_optimal_threshold(X_val, y_val)
        
        logger.info("Random Forest training completed")
        return self.feature_importances


class LightGBMFraudModel(FraudDetectionModel):
    """LightGBM model for fraud detection."""
    
    def __init__(self):
        super().__init__("LightGBM")
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train LightGBM model."""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not available. Install with: brew install libomp && uv pip install lightgbm")
            
        logger.info("Training LightGBM model")
        
        # Calculate scale_pos_weight for imbalanced data
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count
        
        # Use either is_unbalance OR scale_pos_weight, not both
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            class_weight='balanced'  # Use class_weight instead
        )
        
        # Fit with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Feature importances
        self.feature_importances = dict(zip(X_train.columns, self.model.feature_importances_))
        
        # Find optimal threshold
        self.find_optimal_threshold(X_val, y_val)
        
        logger.info("LightGBM training completed", 
                   best_iteration=self.model.best_iteration_)
        return self.feature_importances


class XGBoostFraudModel(FraudDetectionModel):
    """XGBoost model for fraud detection."""
    
    def __init__(self):
        super().__init__("XGBoost")
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train XGBoost model."""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not available. Install with: uv pip install xgboost")
            
        logger.info("Training XGBoost model")
        
        # Calculate scale_pos_weight for imbalanced data
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=50,
            n_jobs=-1
        )
        
        # Fit with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Feature importances
        self.feature_importances = dict(zip(X_train.columns, self.model.feature_importances_))
        
        # Find optimal threshold
        self.find_optimal_threshold(X_val, y_val)
        
        logger.info("XGBoost training completed",
                   best_iteration=self.model.best_iteration)
        return self.feature_importances


class ModelEnsemble:
    """Ensemble of fraud detection models."""
    
    def __init__(self):
        self.models = {'RandomForest': RandomForestFraudModel()}
        
        # Add optional models if libraries are available
        if HAS_LIGHTGBM:
            self.models['LightGBM'] = LightGBMFraudModel()
        else:
            logger.warning("LightGBM not available, skipping from ensemble")
            
        if HAS_XGBOOST:
            self.models['XGBoost'] = XGBoostFraudModel()
        else:
            logger.warning("XGBoost not available, skipping from ensemble")
        self.best_model_name = None
        self.ensemble_results = {}
        
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series,
                  use_resampled_data: bool = True,
                  X_train_resampled: pd.DataFrame = None,
                  y_train_resampled: pd.Series = None) -> Dict:
        """Train all models and select the best one with imbalanced data handling."""
        logger.info("Training model ensemble with imbalanced data handling",
                   use_resampled_data=use_resampled_data)
        
        results = {}
        
        # Use resampled data for training if available and requested
        train_X = X_train_resampled if (use_resampled_data and X_train_resampled is not None) else X_train
        train_y = y_train_resampled if (use_resampled_data and y_train_resampled is not None) else y_train
        
        logger.info("Training data info",
                   original_samples=len(X_train),
                   training_samples=len(train_X),
                   resampled=use_resampled_data and X_train_resampled is not None)
        
        for name, model in self.models.items():
            logger.info("Training model", model=name)
            
            try:
                # Train model
                feature_importances = model.train(train_X, train_y, X_val, y_val)
                
                # Evaluate model (always use original unmodified test set)
                metrics = model.evaluate(X_test, y_test, X_train, y_train)
                
                # Store metrics in the model for ensemble weighting
                model.metrics = metrics
                
                results[name] = {
                    'metrics': metrics,
                    'feature_importances': feature_importances,
                    'model': model
                }
                
                logger.info("Model training completed",
                           model=name,
                           pr_auc=metrics['pr_auc'],
                           f1_score=metrics['f1_score'],
                           balanced_accuracy=metrics['balanced_accuracy'],
                           mcc=metrics['mcc'])
                
            except Exception as e:
                logger.error("Model training failed", model=name, error=str(e))
                continue
        
        # Select best model based on composite score of PR AUC and MCC
        if results:
            def composite_score(metrics):
                # Weighted combination of PR AUC (70%) and MCC (30%)
                pr_auc = metrics['pr_auc']
                mcc = (metrics['mcc'] + 1) / 2  # Normalize MCC from [-1,1] to [0,1]
                return 0.7 * pr_auc + 0.3 * mcc
            
            best_model_name = max(results.keys(), 
                                key=lambda x: composite_score(results[x]['metrics']))
            self.best_model_name = best_model_name
            
            best_metrics = results[best_model_name]['metrics']
            logger.info("Best model selected",
                       model=best_model_name,
                       pr_auc=best_metrics['pr_auc'],
                       mcc=best_metrics['mcc'],
                       composite_score=composite_score(best_metrics))
        
        # Convert all results to JSON-serializable format
        self.ensemble_results = convert_numpy_types(results)
        return self.ensemble_results
    
    def get_best_model(self) -> FraudDetectionModel:
        """Get the best performing model."""
        if self.best_model_name is None:
            raise ValueError("Models not trained yet")
        return self.models[self.best_model_name]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble of all trained models."""
        if not self.models:
            raise ValueError("No models trained yet")
            
        # Get predictions from all trained models
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model.model is not None:  # Only use trained models
                pred = model.predict_proba(X)
                predictions.append(pred)
                
                # Weight models based on their performance (PR AUC if available)
                if hasattr(model, 'metrics') and model.metrics and 'pr_auc' in model.metrics:
                    weight = model.metrics['pr_auc']
                else:
                    weight = 1.0  # Default weight if no metrics
                weights.append(weight)
        
        if not predictions:
            raise ValueError("No trained models available for prediction")
        
        # Weighted average ensemble
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        # Weighted average of predictions
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def load_models(self, models_dir: str) -> bool:
        """Load models from a directory."""
        try:
            loaded_models = {}
            
            # Try to load each model type
            for model_name in self.models.keys():
                model_file = os.path.join(models_dir, f"{model_name.lower()}.joblib")
                
                if os.path.exists(model_file):
                    logger.info(f"Loading model {model_name} from {model_file}")
                    self.models[model_name].load_model(model_file)
                    loaded_models[model_name] = self.models[model_name]
                else:
                    logger.warning(f"Model file not found: {model_file}")
            
            if not loaded_models:
                logger.error("No models could be loaded")
                return False
            
            # Find the best model based on saved metrics or default to first loaded
            # For now, we'll default to XGBoost if available, otherwise first loaded
            if 'XGBoost' in loaded_models:
                self.best_model_name = 'XGBoost'
            elif 'LightGBM' in loaded_models:
                self.best_model_name = 'LightGBM'
            elif 'RandomForest' in loaded_models:
                self.best_model_name = 'RandomForest'
            else:
                self.best_model_name = list(loaded_models.keys())[0]
            
            logger.info(f"Models loaded successfully, best model: {self.best_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Make predictions using the best model."""
        best_model = self.get_best_model()
        return best_model.predict(X, threshold)
    
    def save_all_models(self, save_dir: str) -> Dict[str, str]:
        """Save all trained models in organized structure."""
        model_paths = {}
        
        for name, model in self.models.items():
            if model.model is not None:  # Only save trained models
                model_path = model.save_model(save_dir)  # save_dir is the run directory
                model_paths[name] = model_path
        
        return model_paths
    
    def save_results(self, save_dir: str, run_id: str):
        """Save ensemble results and metrics in organized structure."""
        # Create organized run directory structure
        run_dir = os.path.join(save_dir, 'runs', run_id)
        metrics_dir = os.path.join(run_dir, 'metrics')
        logs_dir = os.path.join(run_dir, 'logs')
        
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save metrics in organized metrics subdirectory
        metrics_summary = {}
        for name, result in self.ensemble_results.items():
            metrics_summary[name] = result['metrics']
        
        with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
            json.dump(convert_numpy_types(metrics_summary), f, indent=2)
        
        # Save feature importances
        feature_importances = {}
        for name, result in self.ensemble_results.items():
            feature_importances[name] = result['feature_importances']
        
        with open(os.path.join(metrics_dir, 'feature_importances.json'), 'w') as f:
            json.dump(convert_numpy_types(feature_importances), f, indent=2)
        
        # Save classification reports
        classification_reports = {}
        for name, result in self.ensemble_results.items():
            classification_reports[name] = result['metrics']['classification_report']
        
        with open(os.path.join(metrics_dir, 'classification_reports.json'), 'w') as f:
            json.dump(convert_numpy_types(classification_reports), f, indent=2)
        
        # Save training summary
        training_summary = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'models_trained': list(self.ensemble_results.keys()),
            'best_model_metrics': self.ensemble_results[self.best_model_name]['metrics'] if self.best_model_name else None
        }
        
        with open(os.path.join(run_dir, 'training_summary.json'), 'w') as f:
            json.dump(convert_numpy_types(training_summary), f, indent=2)
        
        logger.info("Ensemble results saved in organized structure", run_dir=run_dir)