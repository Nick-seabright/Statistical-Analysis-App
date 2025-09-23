# edu_analytics/utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import io
import base64
from datetime import datetime
import logging
import os
import pathlib
import streamlit as st
from datetime import datetime

logger = logging.getLogger(__name__)

def get_save_path(filename, subdir=None):
    """
    Get the full path to save a file in the user-specified directory
    
    Parameters:
    -----------
    filename : str
        Name of the file to save
    subdir : str, optional
        Subdirectory within the main save directory (e.g., 'reports', 'models')
        
    Returns:
    --------
    str : Full path to save the file
    """
    # Get the base directory from session state
    if 'save_directory' not in st.session_state:
        # Default to user's Documents folder if not set
        base_dir = os.path.join(os.path.expanduser("~"), "Documents", "StatisticalAnalysis")
    else:
        base_dir = st.session_state.save_directory
    
    # Create the directory if it doesn't exist
    try:
        if subdir:
            save_dir = os.path.join(base_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = base_dir
            os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        st.warning(f"Could not create directory {save_dir}: {str(e)}")
        # Fall back to base directory
        save_dir = base_dir
    
    # Return the full path
    return os.path.join(save_dir, filename)

def save_file(content, filename, subdir=None):
    """
    Save a file to the user-specified directory
    
    Parameters:
    -----------
    content : bytes or str
        Content to save
    filename : str
        Name of the file
    subdir : str, optional
        Subdirectory within the main save directory
        
    Returns:
    --------
    tuple : (success, message, path)
    """
    try:
        # Get full path
        path = get_save_path(filename, subdir)
        
        # Determine mode based on content type
        mode = 'wb' if isinstance(content, bytes) else 'w'
        
        # Save the file
        with open(path, mode) as f:
            f.write(content)
        
        return True, f"File saved successfully to {path}", path
    except Exception as e:
        return False, f"Error saving file: {str(e)}", None

def get_timestamped_filename(base_name, extension):
    """
    Create a filename with a timestamp
    
    Parameters:
    -----------
    base_name : str
        Base name for the file
    extension : str
        File extension (without the dot)
        
    Returns:
    --------
    str : Filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

# Updated code for utils.py - Enhanced interpret_prediction
def interpret_prediction(
    prediction: float,
    target_type: str,
    target_mapping: Optional[Dict] = None
) -> str:
    """
    Interpret model predictions based on target variable type
    
    Parameters:
    -----------
    prediction : float or int
        Raw model prediction (could be a numeric value or class index)
    target_type : str
        Type of target ('categorical', 'numeric', 'time')
    target_mapping : Dict, optional
        Mapping dictionary for categorical targets (original category -> encoded value)
    
    Returns:
    --------
    str : Human-readable prediction
    """
    if target_type == 'numeric':
        # For numeric targets, just return the value
        return f"{prediction:.2f}"
    elif target_type == 'categorical':
        # For categorical targets, convert back to original label
        if target_mapping:
            # Create reverse mapping (encoded value -> original category)
            reverse_mapping = {v: k for k, v in target_mapping.items()}
            
            # Handle different types of predictions
            if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
                # For array-like predictions, process each value
                return [reverse_mapping.get(int(p) if isinstance(p, float) else p, str(p)) 
                        for p in prediction]
            else:
                # For single value
                pred_key = int(prediction) if isinstance(prediction, float) else prediction
                return reverse_mapping.get(pred_key, str(prediction))
        else:
            # If no mapping available, return as string
            return str(prediction)
    elif target_type == 'time':
        # For time targets, convert minutes back to time format
        from .time_analysis import minutes_to_time_string
        return minutes_to_time_string(prediction)
    else:
        # Unknown target type
        return str(prediction)

def preprocess_input_data(
    input_data: Dict[str, Any],
    data_types: Dict[str, str]
) -> Dict[str, Any]:
    """
    Preprocess input data based on data types
    
    Parameters:
    -----------
    input_data : Dict
        Dictionary of input data (feature: value)
    data_types : Dict
        Dictionary of data types for each feature
        
    Returns:
    --------
    Dict : Preprocessed input data
    """
    processed_data = {}
    
    for feature, value in input_data.items():
        if feature not in data_types:
            processed_data[feature] = value
            continue
        
        dtype = data_types[feature]
        
        if dtype == 'datetime':
            # Convert to datetime and then to numeric (days since epoch)
            try:
                dt = pd.to_datetime(value)
                processed_data[feature] = (dt - pd.Timestamp('1970-01-01')).total_seconds() / (24*3600)
            except:
                processed_data[feature] = value
        
        elif dtype == 'time':
            # Convert time to minutes
            from .time_analysis import convert_time_to_minutes
            processed_data[feature] = convert_time_to_minutes(value)
        
        elif dtype == 'boolean':
            # Convert boolean to 0/1
            processed_data[feature] = 1 if value else 0
        
        elif dtype in ['integer', 'float']:
            # Convert to numeric
            try:
                processed_data[feature] = float(value)
            except:
                processed_data[feature] = value
        
        else:  # categorical
            processed_data[feature] = value
    
    return processed_data

def create_input_widget(
    feature_name: str,
    data_type: str,
    sample_values: Optional[pd.Series] = None
) -> Any:
    """
    Create appropriate input widget based on data type
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature
    data_type : str
        Type of data ('datetime', 'time', 'integer', etc.)
    sample_values : Series, optional
        Sample values for the feature for better defaults
        
    Returns:
    --------
    Input widget object (placeholder in this utility file)
    """
    # This is just a placeholder function since actual widget creation
    # will be handled by the Streamlit app. In a real implementation,
    # this would return Streamlit widget objects.
    return {
        'feature_name': feature_name,
        'data_type': data_type,
        'sample_values': sample_values.describe().to_dict() if sample_values is not None else None
    }

def plot_to_base64(fig: plt.Figure) -> str:
    """
    Convert matplotlib figure to base64 encoded string
    
    Parameters:
    -----------
    fig : matplotlib Figure
        Figure to convert
        
    Returns:
    --------
    str : Base64 encoded string
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode to base64
    encoded = base64.b64encode(image_png).decode('utf-8')
    
    return encoded

def dataframe_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """
    Convert DataFrame to HTML with styling
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame to convert
    max_rows : int
        Maximum number of rows to include
        
    Returns:
    --------
    str : HTML representation of the DataFrame
    """
    # Truncate DataFrame if needed
    if len(df) > max_rows:
        df_display = df.head(max_rows)
        truncated = True
    else:
        df_display = df
        truncated = False
    
    # Convert to HTML with styling
    html = df_display.style.set_properties(**{
        'border': '1px solid #ddd',
        'padding': '8px',
        'text-align': 'right'
    }).to_html()
    
    # Add note about truncation if needed
    if truncated:
        html += f"<p><em>Note: Table truncated to {max_rows} rows out of {len(df)} total rows.</em></p>"
    
    return html

def format_time_duration(seconds: float) -> str:
    """
    Format seconds into human-readable time duration
    
    Parameters:
    -----------
    seconds : float
        Duration in seconds
        
    Returns:
    --------
    str : Formatted time string (e.g., "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes:.0f}m {seconds:.0f}s"
    
    hours, minutes = divmod(minutes, 60)
    return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp into standard format
    
    Parameters:
    -----------
    timestamp : datetime, optional
        Timestamp to format (default: current time)
        
    Returns:
    --------
    str : Formatted timestamp
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def set_plotting_style() -> None:
    """
    Set consistent style for all plots
    """
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def create_report_header(title: str, subtitle: Optional[str] = None) -> str:
    """
    Create HTML header for reports
    
    Parameters:
    -----------
    title : str
        Report title
    subtitle : str, optional
        Report subtitle
        
    Returns:
    --------
    str : HTML header
    """
    header = f"""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #1E88E5; margin-bottom: 10px;">{title}</h1>
    """
    
    if subtitle:
        header += f'<h3 style="color: #555; font-weight: normal;">{subtitle}</h3>'
    
    header += f"""
        <p style="color: #777;">Generated on: {format_timestamp()}</p>
    </div>
    <hr>
    """
    
    return header

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numeric columns in a DataFrame
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    List[str] : List of numeric column names
    """
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of categorical columns in a DataFrame
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    List[str] : List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def memory_usage_summary(df: pd.DataFrame) -> str:
    """
    Create a summary of DataFrame memory usage
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    str : Memory usage summary
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    # Format total memory in appropriate units
    if total_memory < 1024:
        total_memory_str = f"{total_memory} bytes"
    elif total_memory < 1024**2:
        total_memory_str = f"{total_memory/1024:.2f} KB"
    elif total_memory < 1024**3:
        total_memory_str = f"{total_memory/(1024**2):.2f} MB"
    else:
        total_memory_str = f"{total_memory/(1024**3):.2f} GB"
    
    # Get memory usage by type
    memory_by_type = df.dtypes.map(lambda x: str(x)).value_counts()
    memory_by_type = memory_by_type.to_dict()
    
    summary = f"Total memory usage: {total_memory_str}\n"
    summary += "Memory usage by data type:\n"
    
    for dtype, count in memory_by_type.items():
        summary += f"  {dtype}: {count} columns\n"
    
    return summary

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for each feature
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
        
    Returns:
    --------
    DataFrame : VIF for each feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # VIF is only applicable to numeric features
    X_numeric = X.select_dtypes(include=['int64', 'float64'])
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_numeric.columns
    vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
    
    return vif_data.sort_values("VIF", ascending=False)

def get_timestamp_filename(prefix: str, extension: str) -> str:
    """
    Generate a filename with timestamp
    
    Parameters:
    -----------
    prefix : str
        Filename prefix
    extension : str
        File extension
        
    Returns:
    --------
    str : Filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def create_html_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> str:
    """
    Create HTML table from list of dictionaries
    
    Parameters:
    -----------
    data : List[Dict]
        List of dictionaries with data for each row
    headers : List[str], optional
        List of headers (default: keys from first dictionary)
        
    Returns:
    --------
    str : HTML table
    """
    if not data:
        return "<p>No data available</p>"
    
    # If headers not provided, use keys from first dictionary
    if headers is None:
        headers = list(data[0].keys())
    
    # Create HTML table
    html = "<table style='width:100%; border-collapse: collapse;'>"
    
    # Add headers
    html += "<thead><tr>"
    for header in headers:
        html += f"<th style='border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;'>{header}</th>"
    html += "</tr></thead>"
    
    # Add rows
    html += "<tbody>"
    for row in data:
        html += "<tr>"
        for header in headers:
            value = row.get(header, "")
            html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{value}</td>"
        html += "</tr>"
    html += "</tbody>"
    
    html += "</table>"
    
    return html

def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a data array
    
    Parameters:
    -----------
    data : ndarray
        Data array
    confidence : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    Tuple[float, float] : Lower and upper bounds of confidence interval
    """
    from scipy import stats
    
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return mean - interval, mean + interval

def get_original_category_name(feature_name, encoded_value):
    """
    Convert an encoded categorical value back to its original category name
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature
    encoded_value : int
        Encoded value (0, 1, 2, etc.)
    
    Returns:
    --------
    str : Original category name or the encoded value as string if mapping not found
    """
    # Try to get the mapping from session state
    if 'categorical_mappings' in st.session_state and feature_name in st.session_state.categorical_mappings:
        # Get the reverse mapping (encoded → original)
        reverse_mapping = st.session_state.categorical_mappings[feature_name]['reverse']
        # Convert the encoded value to int to ensure proper lookup
        try:
            encoded_int = int(encoded_value) if isinstance(encoded_value, (int, float)) else encoded_value
            # Return the original category name if found, otherwise return the encoded value as string
            return reverse_mapping.get(encoded_int, str(encoded_value))
        except:
            return str(encoded_value)
    else:
        # If no mapping found, return the encoded value as string
        return str(encoded_value)

def get_original_category_names(
    encoded_values: Union[List[int], np.ndarray, int],
    target_mapping: Optional[Dict] = None
) -> Union[List[str], str]:
    """
    Convert encoded categorical values back to original category names
    
    Parameters:
    -----------
    encoded_values : array-like or int
        Encoded values (0, 1, 2, etc.) or a single encoded value
    target_mapping : Dict, optional
        Mapping from original categories to encoded values
        
    Returns:
    --------
    List of original category names or a single category name
    """
    # If no mapping is provided, just convert to strings
    if target_mapping is None:
        if isinstance(encoded_values, (list, np.ndarray)):
            return [str(val) for val in encoded_values]
        else:
            return str(encoded_values)
    
    # Create reverse mapping (encoded value -> original category)
    reverse_mapping = {v: k for k, v in target_mapping.items()}
    
    # Handle both single values and collections
    if isinstance(encoded_values, (list, np.ndarray)):
        return [reverse_mapping.get(val, str(val)) for val in encoded_values]
    else:
        return reverse_mapping.get(encoded_values, str(encoded_values))

def calculate_class_weights(y):
    """
    Calculate balanced class weights for imbalanced datasets
    
    Parameters:
    -----------
    y : array-like
        Target labels
        
    Returns:
    --------
    Dict : Dictionary of class weights {class_label: weight}
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {c: w for c, w in zip(classes, weights)}
    
    return class_weights


def detect_class_imbalance(y):
    """
    Detect and analyze class imbalance in the target variable
    
    Parameters:
    -----------
    y : array-like
        Target variable (categorical)
        
    Returns:
    --------
    Dict : Dictionary with imbalance metrics
    """
    import pandas as pd
    import numpy as np
    
    # Count classes
    counts = pd.Series(y).value_counts()
    total = len(y)
    
    # Calculate metrics
    class_ratios = counts / counts.max()
    min_class = counts.idxmin()
    max_class = counts.idxmax()
    imbalance_ratio = counts.min() / counts.max()
    
    # Determine imbalance severity
    if imbalance_ratio < 0.01:
        severity = "Extreme imbalance"
        suggested_methods = ["SMOTE", "ADASYN", "Class weights", "Ensemble methods"]
    elif imbalance_ratio < 0.1:
        severity = "Strong imbalance"
        suggested_methods = ["SMOTE", "Class weights", "Ensemble methods"]
    elif imbalance_ratio < 0.25:
        severity = "Moderate imbalance"
        suggested_methods = ["Class weights", "Random oversampling"]
    else:
        severity = "Mild or no imbalance"
        suggested_methods = ["Standard methods"]
    
    # Create result dictionary
    result = {
        'class_counts': counts.to_dict(),
        'class_percentages': (counts / total * 100).to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'min_class': min_class,
        'max_class': max_class,
        'severity': severity,
        'suggested_methods': suggested_methods,
        'is_imbalanced': imbalance_ratio < 0.25  # Boolean flag indicating if dataset is imbalanced
    }
    
    return result


def optimize_threshold(y_true, y_pred_proba, metric='f1', beta=1.0):
    """
    Find the optimal classification threshold for imbalanced data
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred_proba : array-like
        Predicted probabilities for the positive class
    metric : str
        Metric to optimize ('f1', 'fbeta', 'geometric_mean', 'balanced_accuracy')
    beta : float
        Beta value for F-beta score (only used if metric='fbeta')
        
    Returns:
    --------
    Dict : Dictionary with optimal threshold and scores
    """
    import numpy as np
    from sklearn.metrics import (
        f1_score, fbeta_score, recall_score, precision_score, 
        balanced_accuracy_score, roc_curve
    )
    
    # Create thresholds to evaluate
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # Calculate metrics for each threshold
    scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate selected metric
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'fbeta':
            score = fbeta_score(y_true, y_pred, beta=beta)
        elif metric == 'geometric_mean':
            sensitivity = recall_score(y_true, y_pred)
            specificity = recall_score(1 - y_true, 1 - y_pred)
            score = np.sqrt(sensitivity * specificity)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred)  # Default to F1
            
        # Calculate additional metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        
        scores.append({
            'threshold': threshold,
            'score': score,
            'precision': precision,
            'recall': recall
        })
    
    # Find threshold with best score
    scores_df = pd.DataFrame(scores)
    best_idx = scores_df['score'].idxmax()
    optimal = scores_df.loc[best_idx].to_dict()
    
    # Also calculate threshold using ROC curve (Youden's J statistic)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    j_optimal_idx = np.argmax(j_scores)
    j_optimal_threshold = roc_thresholds[j_optimal_idx]
    
    return {
        'optimal_threshold': optimal['threshold'],
        'optimal_score': optimal['score'],
        'optimal_precision': optimal['precision'],
        'optimal_recall': optimal['recall'],
        'roc_optimal_threshold': j_optimal_threshold,
        'metric': metric,
        'thresholds': thresholds,
        'scores': scores_df.to_dict('records')
    }


def create_imbalanced_performance_chart(results, threshold_results=None):
    """
    Create a visualization of model performance on imbalanced data
    
    Parameters:
    -----------
    results : Dict
        Dictionary of evaluation metrics from evaluate_model()
    threshold_results : Dict, optional
        Results from optimize_threshold() function
        
    Returns:
    --------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Per-class metrics
    ax1 = axes[0]
    if 'classification_report' in results:
        report = results['classification_report']
        
        # Extract class metrics
        class_metrics = {}
        for key, value in report.items():
            if key not in ['accuracy', 'macro avg', 'weighted avg']:
                class_metrics[key] = {
                    'precision': value['precision'],
                    'recall': value['recall'],
                    'f1-score': value['f1-score'],
                    'support': value['support']
                }
        
        # Create dataframe
        metrics_df = pd.DataFrame(class_metrics).T
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'class'})
        metrics_df = pd.melt(metrics_df, id_vars=['class', 'support'], 
                           var_name='metric', value_name='value')
        
        # Create grouped bar chart
        sns.barplot(x='class', y='value', hue='metric', data=metrics_df, ax=ax1)
        
        # Add support as text
        for i, cls in enumerate(class_metrics.keys()):
            support = class_metrics[cls]['support']
            ax1.text(i, 0.05, f"n={support}", ha='center', fontsize=9)
            
        ax1.set_title('Per-Class Performance Metrics')
        ax1.set_ylim(0, 1.0)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Score')
        
    # Plot 2: Threshold impact (if available)
    ax2 = axes[1]
    if threshold_results is not None:
        thresholds = threshold_results['thresholds']
        scores_df = pd.DataFrame(threshold_results['scores'])
        
        # Plot metrics vs threshold
        ax2.plot(scores_df['threshold'], scores_df['precision'], label='Precision')
        ax2.plot(scores_df['threshold'], scores_df['recall'], label='Recall')
        ax2.plot(scores_df['threshold'], scores_df['score'], label=f"{threshold_results['metric']} Score")
        
        # Add optimal threshold line
        opt_threshold = threshold_results['optimal_threshold']
        ax2.axvline(opt_threshold, color='red', linestyle='--', 
                  label=f'Optimal threshold: {opt_threshold:.2f}')
        
        ax2.set_title('Impact of Classification Threshold')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.legend()
    else:
        # If threshold results not available, show confusion matrix
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title('Confusion Matrix')
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    return fig


def sample_imbalanced_data(X, y, method='smote', random_state=42):
    """
    Apply resampling techniques to handle imbalanced data
    
    Parameters:
    -----------
    X : DataFrame or array-like
        Feature matrix
    y : Series or array-like
        Target variable
    method : str
        Resampling method ('smote', 'adasyn', 'random_over', 'random_under', 'smoteenn', 'smotetomek')
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple : (X_resampled, y_resampled)
    """
    # Import necessary libraries
    try:
        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN, SMOTETomek
    except ImportError:
        raise ImportError("imbalanced-learn package is required. Install it with: pip install imbalanced-learn")
    
    # Apply the selected resampling method
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=random_state)
    elif method == 'random_over':
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'random_under':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smoteenn':
        sampler = SMOTEENN(random_state=random_state)
    elif method == 'smotetomek':
        sampler = SMOTETomek(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    # Perform resampling
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # If X was a DataFrame, convert X_resampled back to DataFrame with same column names
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    
    # If y was a Series, convert y_resampled back to Series
    if isinstance(y, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y.name)
    
    return X_resampled, y_resampled


def plot_imbalance_summary(y):
    """
    Create a visual summary of class imbalance
    
    Parameters:
    -----------
    y : array-like
        Target variable
        
    Returns:
    --------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get imbalance metrics
    imbalance_info = detect_class_imbalance(y)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Class distribution
    counts = pd.Series(imbalance_info['class_counts'])
    ax1.bar(counts.index.astype(str), counts.values)
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    
    # Add count and percentage labels
    for i, (cls, count) in enumerate(counts.items()):
        percentage = imbalance_info['class_percentages'][cls]
        ax1.text(i, count/2, f"{count}\n({percentage:.1f}%)", 
                ha='center', va='center', fontsize=10)
    
    # Plot 2: Imbalance ratio visualization
    ax2.barh(['Imbalance Ratio'], [imbalance_info['imbalance_ratio']])
    ax2.axvline(0.25, color='red', linestyle='--', label='Moderate Imbalance (0.25)')
    ax2.axvline(0.1, color='orange', linestyle='--', label='Strong Imbalance (0.1)')
    ax2.axvline(0.01, color='darkred', linestyle='--', label='Extreme Imbalance (0.01)')
    ax2.set_title('Imbalance Severity')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Imbalance Ratio (min/max)')
    ax2.legend(loc='upper right')
    
    # Add annotations
    plt.figtext(0.5, 0.01, 
               f"Severity: {imbalance_info['severity']}\nSuggested methods: {', '.join(imbalance_info['suggested_methods'])}",
               ha='center', fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig

# Add these functions to edu_analytics/utils.py

def create_neural_network_wrapper(model, threshold=0.5):
    """
    Create a wrapper class for neural network models that works well with Streamlit
    
    Parameters:
    -----------
    model : tf.keras.Model
        The trained neural network model
    threshold : float
        Decision threshold for binary classification
        
    Returns:
    --------
    object : A model wrapper that can be safely stored in session state
    """
    class NeuralNetworkWrapper:
        def __init__(self, model, threshold=0.5):
            self.model = model
            self.threshold = threshold
            # Store model architecture and weights separately to avoid TF session issues
            self.model_config = model.get_config() if hasattr(model, 'get_config') else None
            self.weights = model.get_weights()
            self.is_binary = False
            
            # Try to determine if it's a binary classifier
            try:
                output_shape = model.layers[-1].output_shape
                if isinstance(output_shape, tuple) and output_shape[-1] == 1:
                    self.is_binary = True
                elif hasattr(model, 'output_shape') and model.output_shape[-1] == 1:
                    self.is_binary = True
            except:
                # If we can't determine, assume it's not binary
                pass
        
        def predict(self, X):
            """Make predictions with threshold application"""
            raw_preds = self.model.predict(X)
            
            # Handle binary classification
            if self.is_binary or len(raw_preds.shape) == 1 or raw_preds.shape[1] == 1:
                # Ensure we have the right shape
                if len(raw_preds.shape) > 1:
                    raw_preds = raw_preds.flatten()
                return (raw_preds > self.threshold).astype(int)
            
            # Handle multiclass
            return np.argmax(raw_preds, axis=1)
        
        def predict_proba(self, X):
            """Return class probabilities"""
            raw_preds = self.model.predict(X)
            
            # For binary classification, return probabilities for both classes
            if self.is_binary or len(raw_preds.shape) == 1 or raw_preds.shape[1] == 1:
                if len(raw_preds.shape) > 1:
                    raw_preds = raw_preds.flatten()
                return np.vstack([1 - raw_preds, raw_preds]).T
            
            # For multiclass, return raw predictions
            return raw_preds
        
        def evaluate(self, X, y):
            """Evaluate model performance"""
            return self.model.evaluate(X, y, verbose=0)
        
        def set_threshold(self, new_threshold):
            """Update the decision threshold"""
            self.threshold = new_threshold
            return self
    
    return NeuralNetworkWrapper(model, threshold)


def calculate_permutation_importance_safe(model, X, y, n_repeats=5, max_samples=1000, random_state=42):
    """
    Calculate permutation importance safely for any model type including neural networks
    
    Parameters:
    -----------
    model : model object
        Trained model with predict method
    X : DataFrame
        Feature data
    y : Series or array
        Target data
    n_repeats : int
        Number of times to permute each feature
    max_samples : int
        Maximum number of samples to use (to prevent memory issues)
    random_state : int
        Random seed
        
    Returns:
    --------
    Dict : Dictionary with importance results
    """
    try:
        from sklearn.inspection import permutation_importance
        
        # Limit sample size to prevent memory issues
        if len(X) > max_samples:
            # Randomly sample data
            idx = np.random.RandomState(random_state).choice(
                np.arange(len(X)), size=max_samples, replace=False)
            X_sample = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
            y_sample = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
        else:
            X_sample = X
            y_sample = y
            
        # For neural network models, create a simple wrapper
        if 'tensorflow' in str(type(model)).lower() or 'keras' in str(type(model)).lower():
            class SimpleModelWrapper:
                def __init__(self, model):
                    self.model = model
                
                def predict(self, X):
                    preds = self.model.predict(X)
                    # Handle binary classification
                    if len(preds.shape) == 1 or preds.shape[1] == 1:
                        return (preds > 0.5).astype(int).flatten()
                    # Handle multi-class
                    return np.argmax(preds, axis=1)
                    
                def fit(self, *args, **kwargs):
                    # Dummy method for sklearn compatibility
                    return self
            
            model_for_perm = SimpleModelWrapper(model)
        else:
            model_for_perm = model
            
        # Calculate permutation importance with appropriate scoring
        if len(np.unique(y_sample)) <= 2:  # Binary classification
            scoring = 'balanced_accuracy'
        elif len(np.unique(y_sample)) > 2:  # Multi-class classification
            scoring = 'balanced_accuracy'
        else:  # Regression
            scoring = 'neg_mean_squared_error'
            
        # Run permutation importance calculation
        result = permutation_importance(
            model_for_perm, X_sample, y_sample,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring
        )
        
        # Create a dictionary of results
        importance_results = {
            'importances_mean': result.importances_mean,
            'importances_std': result.importances_std,
            'feature_names': X.columns.tolist(),
            'success': True
        }
        
        return importance_results
        
    except Exception as e:
        # Return error information if calculation fails
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def evaluate_model_with_threshold(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a model with a custom threshold for binary classification
    
    Parameters:
    -----------
    model : model object
        Trained model with predict_proba method
    X_test : DataFrame
        Test feature data
    y_test : Series or array
        Test target data
    threshold : float
        Decision threshold
        
    Returns:
    --------
    Dict : Dictionary with evaluation metrics
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, balanced_accuracy_score
    )
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        # Handle different formats of probability outputs
        if y_proba.shape[1] == 2:
            # Standard scikit-learn format with probabilities for both classes
            y_prob_positive = y_proba[:, 1]
        else:
            # Single column of probabilities
            y_prob_positive = y_proba.flatten()
    else:
        # For models without predict_proba, try to get raw predictions
        try:
            y_prob_positive = model.predict(X_test).flatten()
        except:
            # If that fails too, return error
            return {'error': 'Model does not support probability predictions'}
    
    # Apply threshold
    y_pred = (y_prob_positive >= threshold).astype(int)
    
    # Calculate metrics
    try:
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        try:
            roc_auc = roc_auc_score(y_test, y_prob_positive)
        except:
            roc_auc = None
        
        # Return all metrics
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'threshold': threshold,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }