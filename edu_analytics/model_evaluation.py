# edu_analytics/model_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
import logging

logger = logging.getLogger(__name__)

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_type: str,
    model_name: str = "Model"
) -> Dict:
    """
    Evaluate a trained model on test data
    
    Parameters:
    -----------
    model : trained model
        The model to evaluate
    X_test : DataFrame
        Test feature data
    y_test : Series
        Test target data
    target_type : str
        Type of target ('categorical', 'numeric', 'time')
    model_name : str
        Name of the model for reporting
        
    Returns:
    --------
    Dictionary with evaluation metrics
    """
    results = {'model_name': model_name}
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    if target_type == 'categorical':
        # Classification metrics
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred, average='weighted')
        results['recall'] = recall_score(y_test, y_pred, average='weighted')
        results['f1'] = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm
        
        # ROC curve and AUC (for binary classification)
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            results['roc_auc'] = auc(fpr, tpr)
            results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    
    else:  # 'numeric' or 'time'
        # Regression metrics
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['r2'] = r2_score(y_test, y_pred)
        
        # Additional metrics
        results['explained_variance'] = 1 - (np.var(y_test - y_pred) / np.var(y_test))
        results['median_absolute_error'] = np.median(np.abs(y_test - y_pred))
    
    logger.info(f"Evaluation results for {model_name}:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    return results

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    target_mapping: Optional[Dict] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix with proper class labels
    
    Parameters:
    -----------
    cm : ndarray
        Confusion matrix
    class_names : List[str], optional
        List of class names
    target_mapping : Dict, optional
        Mapping from original categories to encoded values
    title : str
        Plot title
    
    Returns:
    --------
    Matplotlib Figure
    """
    # Use provided class names, derive from mapping, or use default indices
    if class_names is None:
        if target_mapping:
            # Create reverse mapping (encoded value -> original category)
            reverse_mapping = {v: k for k, v in target_mapping.items()}
            class_names = [reverse_mapping.get(i, str(i)) for i in range(cm.shape[0])]
        else:
            class_names = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    return fig

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    title: str = "Receiver Operating Characteristic"
) -> plt.Figure:
    """
    Plot ROC curve
    
    Parameters:
    -----------
    fpr : ndarray
        False positive rates
    tpr : ndarray
        True positive rates
    roc_auc : float
        Area under ROC curve
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    return fig

def plot_regression_results(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted"
) -> plt.Figure:
    """
    Plot actual vs predicted values for regression
    
    Parameters:
    -----------
    y_test : ndarray
        Actual values
    y_pred : ndarray
        Predicted values
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    
    # Add stats text
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    stats_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}"
    plt.figtext(0.05, 0.05, stats_text, fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    return fig

def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    title: str = "Feature Importance"
) -> Optional[plt.Figure]:
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : trained model
        The model with feature_importances_ attribute
    feature_names : List[str]
        List of feature names
    top_n : int
        Number of top features to display
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib Figure or None if feature importance not available
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model doesn't have feature_importances_ attribute")
        return None
    
    # Get feature importance
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top features
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title(title)
    
    return fig

def plot_learning_curves(
    history: Dict,
    title: str = "Learning Curves"
) -> plt.Figure:
    """
    Plot learning curves for neural networks
    
    Parameters:
    -----------
    history : Dict
        Training history from Keras model
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Metrics plot
    metrics = [key for key in history.keys() if key not in ['loss', 'val_loss']]
    
    if metrics:
        metric = metrics[0]  # Use first metric
        ax2.plot(history[metric], label=f'Training {metric}')
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax2.plot(history[val_metric], label=f'Validation {metric}')
        ax2.set_title(f'Model {metric}')
        ax2.set_ylabel(metric)
        ax2.set_xlabel('Epoch')
        ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def compare_models(
    evaluation_results: List[Dict],
    target_type: str,
    title: str = "Model Comparison"
) -> plt.Figure:
    """
    Compare multiple models visually
    
    Parameters:
    -----------
    evaluation_results : List[Dict]
        List of evaluation result dictionaries
    target_type : str
        Type of target ('categorical', 'numeric', 'time')
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib Figure
    """
    # Extract model names and metrics
    model_names = [result['model_name'] for result in evaluation_results]
    
    if target_type == 'categorical':
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_values = {
            metric: [result[metric] for result in evaluation_results]
            for metric in metrics
        }
    else:  # 'numeric' or 'time'
        metrics = ['r2', 'rmse', 'mae']
        metric_values = {
            metric: [result[metric] for result in evaluation_results]
            for metric in metrics
        }
    
    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(model_names, metric_values[metric])
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1.0 if metric != 'rmse' and metric != 'mae' else None)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_error_distribution(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Error Distribution"
) -> plt.Figure:
    """
    Plot distribution of prediction errors
    
    Parameters:
    -----------
    y_test : ndarray
        Actual values
    y_pred : ndarray
        Predicted values
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib Figure
    """
    errors = y_test - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(errors, kde=True, ax=ax)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Prediction Error')
    
    # Add stats text
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    stats_text = f"Mean Error: {mean_error:.4f}\nStd Dev: {std_error:.4f}"
    plt.figtext(0.05, 0.95, stats_text, fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    return fig

def create_model_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_type: str,
    model_name: str = "Model"
) -> Dict:
    """
    Create a comprehensive model evaluation report
    
    Parameters:
    -----------
    model : trained model
        The model to evaluate
    X_test : DataFrame
        Test feature data
    y_test : Series
        Test target data
    target_type : str
        Type of target ('categorical', 'numeric', 'time')
    model_name : str
        Name of the model for reporting
        
    Returns:
    --------
    Dictionary with evaluation metrics and plots
    """
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, target_type, model_name)
    
    # Create dictionary to store plots
    plots = {}
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    if target_type == 'categorical':
        # Confusion matrix
        cm = results['confusion_matrix']
        plots['confusion_matrix'] = plot_confusion_matrix(cm, title=f"{model_name} Confusion Matrix")
        
        # ROC curve for binary classification
        if 'roc_curve' in results:
            fpr = results['roc_curve']['fpr']
            tpr = results['roc_curve']['tpr']
            roc_auc = results['roc_auc']
            plots['roc_curve'] = plot_roc_curve(fpr, tpr, roc_auc, title=f"{model_name} ROC Curve")
    
    else:  # 'numeric' or 'time'
        # Actual vs Predicted
        plots['regression_plot'] = plot_regression_results(y_test, y_pred, title=f"{model_name} Actual vs Predicted")
        
        # Error distribution
        plots['error_distribution'] = plot_error_distribution(y_test, y_pred, title=f"{model_name} Error Distribution")
    
    # Feature importance if available
    if hasattr(model, 'feature_importances_'):
        plots['feature_importance'] = plot_feature_importance(model, X_test.columns, title=f"{model_name} Feature Importance")
    
    # Add plots to results
    results['plots'] = plots
    
    return results

def calculate_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """
    Calculate feature importance using permutation importance method.
    Works with any model type that has a predict method.
    
    Parameters:
    -----------
    model : trained model
        The model to evaluate
    X : DataFrame
        Feature data
    y : Series
        Target data
    n_repeats : int
        Number of times to permute each feature
    random_state : int
        Random seed
        
    Returns:
    --------
    DataFrame with feature importance scores
    """
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    result = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats, 
        random_state=random_state
    )
    
    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': result.importances_mean,
        'std': result.importances_std
    })
    
    return importance_df.sort_values('importance', ascending=False)