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

def interpret_prediction(
    prediction: float, 
    target_type: str,
    target_mapping: Optional[Dict] = None
) -> str:
    """
    Interpret model predictions based on target variable type
    
    Parameters:
    -----------
    prediction : float
        Raw model prediction
    target_type : str
        Type of target ('categorical', 'numeric', 'time')
    target_mapping : Dict, optional
        Mapping dictionary for categorical targets
        
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
            # We need to reverse the mapping
            reverse_mapping = {v: k for k, v in target_mapping.items()}
            return reverse_mapping.get(int(prediction) if isinstance(prediction, float) else prediction, str(prediction))
        else:
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