import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    # Add new metrics for imbalanced data
    balanced_accuracy_score, average_precision_score,
    precision_recall_fscore_support
)
import logging
logger = logging.getLogger(__name__)

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_type: str,
    model_name: str = "Model",
    normalize_confusion_matrix: bool = True,  # New parameter for normalized confusion matrix
    threshold: float = 0.5  # New parameter for custom threshold
) -> Dict:
    """
    Evaluate a trained model on test data with enhanced metrics for imbalanced data
    
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
    normalize_confusion_matrix : bool
        Whether to normalize the confusion matrix (better for imbalanced data)
    threshold : float
        Custom threshold for binary classification (default: 0.5)
        
    Returns:
    --------
    Dictionary with evaluation metrics
    """
    results = {'model_name': model_name}
    
    # Make predictions
    if target_type == 'categorical' and hasattr(model, 'predict_proba'):
        # For binary classification, apply custom threshold
        if len(np.unique(y_test)) == 2:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
            # Store the threshold used
            results['threshold_used'] = threshold
        else:
            # For multiclass, use standard prediction
            y_pred = model.predict(X_test)
    else:
        # For regression or models without predict_proba
        y_pred = model.predict(X_test)
    
    if target_type == 'categorical':
        # Classification metrics
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred, average='weighted')
        results['recall'] = recall_score(y_test, y_pred, average='weighted')
        results['f1'] = f1_score(y_test, y_pred, average='weighted')
        
        # Add balanced accuracy - better for imbalanced data
        results['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
        
        # Per-class precision, recall, and F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        # Store per-class metrics
        results['per_class_metrics'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm
        
        # Normalized confusion matrix (important for imbalanced data)
        if normalize_confusion_matrix:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            results['confusion_matrix_normalized'] = cm_normalized
            
        # ROC curve and AUC (for binary classification)
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            results['roc_auc'] = auc(fpr, tpr)
            results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
            
            # Add Precision-Recall curve and AUC (better for imbalanced data)
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
            results['pr_auc'] = average_precision_score(y_test, y_pred_proba)
            results['pr_curve'] = {
                'precision': precision_curve, 
                'recall': recall_curve, 
                'thresholds': pr_thresholds
            }
            
            # Calculate imbalance metrics
            class_counts = np.bincount(y_test)
            if len(class_counts) >= 2:  # Make sure we have at least 2 classes
                minority_class = np.argmin(class_counts)
                majority_class = np.argmax(class_counts)
                imbalance_ratio = class_counts[minority_class] / class_counts[majority_class]
                results['imbalance_ratio'] = imbalance_ratio
                results['minority_class'] = int(minority_class)
                results['majority_class'] = int(majority_class)
                
                # Add specific metrics for minority class
                minority_precision = precision[minority_class]
                minority_recall = recall[minority_class]
                minority_f1 = f1[minority_class]
                
                results['minority_metrics'] = {
                    'precision': minority_precision,
                    'recall': minority_recall,
                    'f1': minority_f1
                }
                
                # Add severity of imbalance classification
                if imbalance_ratio < 0.05:
                    results['imbalance_severity'] = 'Severe'
                elif imbalance_ratio < 0.2:
                    results['imbalance_severity'] = 'Moderate'
                else:
                    results['imbalance_severity'] = 'Mild'
    else:  # 'numeric' or 'time'
        # Regression metrics
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['r2'] = r2_score(y_test, y_pred)
        
        # Additional metrics
        results['explained_variance'] = 1 - (np.var(y_test - y_pred) / np.var(y_test))
        results['median_absolute_error'] = np.median(np.abs(y_test - y_pred))
        
        # Add error distribution statistics
        errors = y_test - y_pred
        results['error_mean'] = errors.mean()
        results['error_std'] = errors.std()
        results['error_skew'] = pd.Series(errors).skew()
        
    logger.info(f"Evaluation results for {model_name}:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    return results

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    target_mapping: Optional[Dict] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,  # New parameter for normalization
    colormap: str = "Blues"  # New parameter for customizing colormap
) -> plt.Figure:
    """
    Plot confusion matrix with proper class labels, optimized for imbalanced data
    
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
    normalize : bool
        Whether to normalize the confusion matrix by row (true class)
    colormap : str
        Colormap to use for the plot
        
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

    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'  # Use percentage format
    else:
        cm_display = cm
        fmt = 'd'    # Use integer format
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with adjusted parameters for better visibility
    sns.heatmap(
        cm_display, 
        annot=True, 
        fmt=fmt, 
        cmap=colormap, 
        xticklabels=class_names, 
        yticklabels=class_names,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 10}
    )
    
    # Add appropriate labels and title
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Customize title based on normalization
    if normalize:
        plt.title(f"{title} (Normalized)", fontsize=14)
    else:
        plt.title(title, fontsize=14)
    
    # Add counts as a footer if normalized
    if normalize:
        footer = f"Total samples: {np.sum(cm)}"
        class_counts = cm.sum(axis=1)
        for i, count in enumerate(class_counts):
            footer += f"\n{class_names[i]}: {count} ({count/np.sum(cm):.1%})"
        plt.figtext(0.5, 0.01, footer, ha="center", fontsize=10, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        plt.subplots_adjust(bottom=0.2)
    
    return fig

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    title: str = "Receiver Operating Characteristic",
    pr_curve: bool = True,  # Add option to include precision-recall curve
    precision: Optional[np.ndarray] = None,
    recall: Optional[np.ndarray] = None,
    pr_auc: Optional[float] = None
) -> plt.Figure:
    """
    Plot ROC curve and optionally PR curve (better for imbalanced data)
    
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
    pr_curve : bool
        Whether to include precision-recall curve (recommended for imbalanced data)
    precision : ndarray, optional
        Precision values for PR curve
    recall : ndarray, optional
        Recall values for PR curve
    pr_auc : float, optional
        Area under PR curve
        
    Returns:
    --------
    Matplotlib Figure
    """
    if pr_curve and precision is not None and recall is not None:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot ROC curve
        ax1.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")
        
        # Plot Precision-Recall curve
        ax2.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})' if pr_auc else 'PR curve')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        # Add baseline (random classifier) line for PR curve - depends on class imbalance
        if 'imbalance_ratio' in locals() and imbalance_ratio is not None:
            minority_class_ratio = imbalance_ratio / (1 + imbalance_ratio)
            ax2.axhline(y=minority_class_ratio, color='r', linestyle='--', 
                       label=f'Baseline ({minority_class_ratio:.2f})')
        ax2.legend(loc="lower left")
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
    else:
        # Just plot ROC curve
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

def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    average_precision: float,
    pos_class_ratio: float = None,
    title: str = "Precision-Recall Curve"
) -> plt.Figure:
    """
    Plot precision-recall curve, particularly useful for imbalanced data
    
    Parameters:
    -----------
    precision : ndarray
        Precision values
    recall : ndarray
        Recall values
    average_precision : float
        Average precision score
    pos_class_ratio : float, optional
        Positive class ratio (for baseline)
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot PR curve
    ax.plot(recall, precision, label=f'AP = {average_precision:.2f}')
    
    # Add baseline if positive class ratio is provided
    if pos_class_ratio is not None:
        ax.axhline(y=pos_class_ratio, color='r', linestyle='--', 
                  label=f'Baseline ({pos_class_ratio:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    
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
    title: str = "Model Comparison",
    focus_on_minority: bool = True  # New parameter to focus on minority class metrics
) -> plt.Figure:
    """
    Compare multiple models visually, with focus on metrics for imbalanced data
    
    Parameters:
    -----------
    evaluation_results : List[Dict]
        List of evaluation result dictionaries
    target_type : str
        Type of target ('categorical', 'numeric', 'time')
    title : str
        Plot title
    focus_on_minority : bool
        Whether to focus on minority class metrics for imbalanced data
        
    Returns:
    --------
    Matplotlib Figure
    """
    # Extract model names and metrics
    model_names = [result['model_name'] for result in evaluation_results]
    
    if target_type == 'categorical':
        # Check if we have minority class metrics and should focus on them
        has_minority_metrics = all('minority_metrics' in result for result in evaluation_results)
        
        if focus_on_minority and has_minority_metrics:
            # Use minority class metrics for comparison
            metrics = ['balanced_accuracy', 'pr_auc']
            if 'minority_metrics' in evaluation_results[0]:
                metrics.extend(['minority_precision', 'minority_recall', 'minority_f1'])
            
            # Extract metrics from results
            metric_values = {}
            for metric in metrics:
                if metric.startswith('minority_'):
                    # Extract from nested dictionary
                    base_metric = metric.replace('minority_', '')
                    metric_values[metric] = [
                        result['minority_metrics'][base_metric] 
                        if 'minority_metrics' in result else np.nan 
                        for result in evaluation_results
                    ]
                else:
                    # Extract from top level
                    metric_values[metric] = [
                        result.get(metric, np.nan) for result in evaluation_results
                    ]
        else:
            # Use standard metrics
            metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']
            if any('pr_auc' in result for result in evaluation_results):
                metrics.append('pr_auc')
                
            metric_values = {
                metric: [result.get(metric, np.nan) for result in evaluation_results]
                for metric in metrics
            }
    else:  # 'numeric' or 'time'
        metrics = ['r2', 'rmse', 'mae']
        metric_values = {
            metric: [result[metric] for result in evaluation_results]
            for metric in metrics
        }
    
    # Create figure with right size for the number of metrics
    fig_width = max(12, len(metrics) * 4)
    fig, axes = plt.subplots(1, len(metrics), figsize=(fig_width, 5))
    
    # Ensure axes is a list even if we have only one metric
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        # Use a diverging color palette for better visualization
        palette = sns.color_palette("RdYlGn", len(model_names))
        # For RMSE and MAE, lower is better so we'll reverse the palette
        if metric in ['rmse', 'mae']:
            palette = palette[::-1]
            
        # Create the bar plot
        bars = ax.bar(model_names, metric_values[metric], color=palette)
        ax.set_title(metric.upper() if not metric.startswith('minority_') 
                    else f"{metric.replace('minority_', '')} (Minority Class)")
        
        # Set y-axis limits appropriately
        if metric in ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'pr_auc']:
            ax.set_ylim(0, 1.0)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        # Rotate x-axis labels for readability
        ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add a descriptive title
    if focus_on_minority and has_minority_metrics:
        plt.suptitle(f"{title} (Focus on Minority Class Performance)", fontsize=16)
    else:
        plt.suptitle(title, fontsize=16)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the suptitle
    
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
    model_name: str = "Model",
    threshold: float = 0.5  # New parameter for custom threshold
) -> Dict:
    """
    Create a comprehensive model evaluation report with enhanced metrics for imbalanced data
    
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
    threshold : float
        Custom threshold for binary classification
        
    Returns:
    --------
    Dictionary with evaluation metrics and plots
    """
    # Evaluate model with potentially custom threshold
    results = evaluate_model(
        model, X_test, y_test, target_type, model_name, 
        normalize_confusion_matrix=True, threshold=threshold
    )
    
    # Create dictionary to store plots
    plots = {}
    
    # Make predictions - respecting custom threshold for binary classification
    if target_type == 'categorical' and hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    if target_type == 'categorical':
        # Confusion matrix - both normalized and raw
        cm = results['confusion_matrix']
        plots['confusion_matrix'] = plot_confusion_matrix(
            cm, title=f"{model_name} Confusion Matrix", normalize=False
        )
        plots['confusion_matrix_normalized'] = plot_confusion_matrix(
            cm, title=f"{model_name} Normalized Confusion Matrix", normalize=True
        )
        
        # ROC and PR curves for binary classification
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            # If we have stored ROC data in results
            if 'roc_curve' in results and 'pr_curve' in results:
                fpr = results['roc_curve']['fpr']
                tpr = results['roc_curve']['tpr']
                roc_auc = results['roc_auc']
                precision = results['pr_curve']['precision']
                recall = results['pr_curve']['recall']
                pr_auc = results['pr_auc']
                
                # Get class imbalance data if available
                if 'imbalance_ratio' in results:
                    imbalance_ratio = results['imbalance_ratio']
                    minority_class = results['minority_class']
                    pos_class_ratio = np.mean(y_test == minority_class)
                else:
                    pos_class_ratio = np.mean(y_test)
                
                # Create combined plot with both ROC and PR curves
                plots['roc_pr_curves'] = plot_roc_curve(
                    fpr, tpr, roc_auc, 
                    title=f"{model_name} ROC & PR Curves",
                    pr_curve=True, 
                    precision=precision,
                    recall=recall,
                    pr_auc=pr_auc
                )
                
                # Also create a dedicated PR curve plot
                plots['pr_curve'] = plot_precision_recall_curve(
                    precision, recall, pr_auc, 
                    pos_class_ratio=pos_class_ratio,
                    title=f"{model_name} Precision-Recall Curve"
                )
    else:  # 'numeric' or 'time'
        # Actual vs Predicted
        plots['regression_plot'] = plot_regression_results(y_test, y_pred, title=f"{model_name} Actual vs Predicted")
        # Error distribution
        plots['error_distribution'] = plot_error_distribution(y_test, y_pred, title=f"{model_name} Error Distribution")
    
    # Feature importance if available
    if hasattr(model, 'feature_importances_'):
        plots['feature_importance'] = plot_feature_importance(model, X_test.columns, title=f"{model_name} Feature Importance")
    elif 'permutation_importance' in results:
        # Create a plot from permutation importance if available
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': results['permutation_importance']['importances_mean']
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
        plt.title(f"{model_name} Feature Importance (Permutation)")
        plots['permutation_importance'] = fig
    
    # Add plots to results
    results['plots'] = plots
    return results

def calculate_permutation_importance(
    model: Any, 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_repeats: int = 10, 
    random_state: int = 42,
    scoring: str = None  # Added parameter for different scoring metrics
) -> pd.DataFrame:
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
    scoring : str, optional
        Scoring metric to use (e.g., 'precision', 'recall', 'f1' for classification,
        'neg_mean_squared_error' for regression)
        
    Returns:
    --------
    DataFrame with feature importance scores
    """
    from sklearn.inspection import permutation_importance
    
    # For imbalanced classification, we might want to use a different scoring metric
    if scoring is None:
        # Infer appropriate scoring based on y
        if len(np.unique(y)) <= 2:
            # For binary classification, use balanced accuracy or f1
            class_counts = np.bincount(y)
            imbalance_ratio = np.min(class_counts) / np.max(class_counts)
            
            if imbalance_ratio < 0.2:
                # For imbalanced data
                scoring = 'f1'
            else:
                # For more balanced data
                scoring = 'balanced_accuracy'
        elif len(np.unique(y)) > 2:
            # For multiclass, use balanced accuracy
            scoring = 'balanced_accuracy'
        else:
            # For regression, use neg_mean_squared_error
            scoring = 'neg_mean_squared_error'
    
    # Handle neural networks and other non-sklearn models
    if hasattr(model, 'predict') and not hasattr(model, 'fit'):
        # Create a wrapper for models that don't have a sklearn-compatible fit method
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
                # For binary classification models that return probabilities
                self.is_binary_classifier = False
                if hasattr(model, 'predict_proba'):
                    try:
                        # Check if it returns probabilities with shape (n_samples, 2)
                        if model.predict_proba(X.iloc[:1]).shape[1] == 2:
                            self.is_binary_classifier = True
                    except:
                        pass
                        
                # Try to infer number of classes for multiclass classifiers
                self.classes_ = np.unique(y)
                
            def predict(self, X):
                return self.model.predict(X)
                
            def predict_proba(self, X):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)
                else:
                    # Fallback for models without predict_proba
                    raise NotImplementedError("Model doesn't support predict_proba")
                    
            def fit(self, X, y):
                # Dummy method for sklearn compatibility
                return self
                
        # Use the wrapper
        wrapped_model = ModelWrapper(model)
    else:
        # Use the model directly
        wrapped_model = model
    
    # Calculate permutation importance
    try:
        result = permutation_importance(
            wrapped_model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring
        )
        
        # Create DataFrame with results
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        })
        
        return importance_df.sort_values('importance', ascending=False)
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {str(e)}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['feature', 'importance', 'std'])

def find_optimal_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric: str = 'f1',
    n_thresholds: int = 100
) -> Tuple[float, Dict, plt.Figure]:
    """
    Find the optimal classification threshold for imbalanced binary classification
    
    Parameters:
    -----------
    model : trained model
        The model that supports predict_proba
    X_test : DataFrame
        Test feature data
    y_test : Series
        Test target data
    metric : str
        Metric to optimize: 'f1', 'precision', 'recall', 'balanced_accuracy', 'specificity'
    n_thresholds : int
        Number of thresholds to evaluate
        
    Returns:
    --------
    Tuple containing:
    - optimal_threshold: The best threshold value
    - metrics_at_threshold: Dictionary of metrics at the optimal threshold
    - fig: Matplotlib figure showing threshold vs. metrics
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, 
        accuracy_score, confusion_matrix
    )
    
    if not hasattr(model, 'predict_proba'):
        logger.error("Model doesn't support predict_proba method")
        return 0.5, {}, None
    
    # Generate probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate threshold values to test
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    # Initialize results
    results = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate basic metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate confusion matrix for specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (recall + specificity) / 2
        
        # Calculate G-mean (geometric mean of sensitivity and specificity)
        g_mean = np.sqrt(recall * specificity) if (recall * specificity) > 0 else 0
        
        # Store all metrics
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'balanced_accuracy': balanced_acc,
            'g_mean': g_mean
        })
    
    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(results)
    
    # Find optimal threshold based on selected metric
    if metric == 'f1':
        optimal_idx = df_results['f1'].idxmax()
    elif metric == 'precision':
        optimal_idx = df_results['precision'].idxmax()
    elif metric == 'recall':
        optimal_idx = df_results['recall'].idxmax()
    elif metric == 'specificity':
        optimal_idx = df_results['specificity'].idxmax()
    elif metric == 'balanced_accuracy':
        optimal_idx = df_results['balanced_accuracy'].idxmax()
    elif metric == 'g_mean':
        optimal_idx = df_results['g_mean'].idxmax()
    else:
        # Default to F1
        optimal_idx = df_results['f1'].idxmax()
    
    optimal_threshold = df_results.loc[optimal_idx, 'threshold']
    
    # Get metrics at optimal threshold
    metrics_at_threshold = df_results.loc[optimal_idx].to_dict()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot all metrics vs threshold
    ax1.plot(df_results['threshold'], df_results['precision'], label='Precision')
    ax1.plot(df_results['threshold'], df_results['recall'], label='Recall')
    ax1.plot(df_results['threshold'], df_results['f1'], label='F1')
    ax1.plot(df_results['threshold'], df_results['balanced_accuracy'], label='Balanced Accuracy')
    ax1.plot(df_results['threshold'], df_results['specificity'], label='Specificity')
    ax1.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs. Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot precision-recall curve
    ax2.plot(df_results['recall'], df_results['precision'])
    ax2.scatter(metrics_at_threshold['recall'], metrics_at_threshold['precision'], 
               color='red', s=100, label=f'Threshold: {optimal_threshold:.2f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(f'Threshold Optimization for {metric.capitalize()}', fontsize=16)
    plt.tight_layout()
    
    return optimal_threshold, metrics_at_threshold, fig