# edu_analytics/time_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import logging

logger = logging.getLogger(__name__)

def convert_time_to_minutes(time_str: Union[str, float, int, None]) -> Optional[float]:
    """
    Convert time string (HH:MM:SS or MM:SS) to minutes
    
    Parameters:
    -----------
    time_str : str, float, int, or None
        Time string to convert
        
    Returns:
    --------
    float : Time in minutes or None if conversion fails
    """
    try:
        # If already numeric, return as is
        if isinstance(time_str, (int, float)):
            return float(time_str)
        
        # Handle None or NaN
        if time_str is None or pd.isna(time_str):
            return None
        
        # Convert string to components
        parts = str(time_str).split(':')
        
        if len(parts) == 2:  # MM:SS
            return float(parts[0]) + float(parts[1]) / 60
        elif len(parts) == 3:  # HH:MM:SS
            return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 60
        else:
            # Try to convert directly to float
            return float(time_str)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert time value: {time_str}")
        return None

def minutes_to_time_string(minutes: Union[float, int, None]) -> Optional[str]:
    """
    Convert minutes to formatted time string (HH:MM:SS or MM:SS)
    
    Parameters:
    -----------
    minutes : float, int, or None
        Minutes to convert
        
    Returns:
    --------
    str : Formatted time string or None if conversion fails
    """
    try:
        if minutes is None or pd.isna(minutes):
            return None
        
        # Convert to float to ensure proper calculation
        minutes = float(minutes)
        
        # Extract components
        hours = int(minutes / 60)
        remaining_minutes = int(minutes % 60)
        seconds = int((minutes * 60) % 60)
        
        if hours > 0:
            return f"{hours}:{remaining_minutes:02d}:{seconds:02d}"
        else:
            return f"{remaining_minutes}:{seconds:02d}"
    except (ValueError, TypeError):
        logger.warning(f"Could not convert minutes to time string: {minutes}")
        return None

def analyze_time_columns(df: pd.DataFrame, time_columns: List[str]) -> Dict:
    """
    Analyze time columns in a dataframe
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe containing time columns
    time_columns : List[str]
        List of column names containing time data
        
    Returns:
    --------
    Dict with analysis results
    """
    results = {}
    
    for column in time_columns:
        # Convert to minutes for analysis
        minutes_series = df[column].apply(convert_time_to_minutes)
        
        # Calculate statistics
        stats = {
            'min': minutes_series.min(),
            'max': minutes_series.max(),
            'mean': minutes_series.mean(),
            'median': minutes_series.median(),
            'std': minutes_series.std(),
            'missing': minutes_series.isna().sum(),
            'missing_pct': minutes_series.isna().mean() * 100
        }
        
        # Convert statistics back to time format for display
        formatted_stats = {
            'min': minutes_to_time_string(stats['min']),
            'max': minutes_to_time_string(stats['max']),
            'mean': minutes_to_time_string(stats['mean']),
            'median': minutes_to_time_string(stats['median']),
            'std': stats['std'],
            'missing': stats['missing'],
            'missing_pct': stats['missing_pct']
        }
        
        # Store both numeric and formatted stats
        results[column] = {
            'numeric_stats': stats,
            'formatted_stats': formatted_stats,
            'minutes_series': minutes_series
        }
    
    return results

def visualize_time_distribution(
    minutes_series: pd.Series,
    column_name: str,
    bins: int = 20
) -> plt.Figure:
    """
    Visualize the distribution of a time variable
    
    Parameters:
    -----------
    minutes_series : Series
        Series of time values in minutes
    column_name : str
        Name of the time column
    bins : int
        Number of histogram bins
        
    Returns:
    --------
    matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    sns.histplot(minutes_series.dropna(), bins=bins, kde=True, ax=ax1)
    ax1.set_title(f"Distribution of {column_name}")
    ax1.set_xlabel("Time (minutes)")
    
    # Add mean and median lines
    mean_time = minutes_series.mean()
    median_time = minutes_series.median()
    
    ax1.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {minutes_to_time_string(mean_time)}')
    ax1.axvline(median_time, color='green', linestyle='--', label=f'Median: {minutes_to_time_string(median_time)}')
    ax1.legend()
    
    # Box plot
    sns.boxplot(y=minutes_series.dropna(), ax=ax2)
    ax2.set_title(f"Box Plot of {column_name}")
    ax2.set_ylabel("Time (minutes)")
    
    # Add formatted time labels to y-axis
    y_ticks = ax2.get_yticks()
    y_tick_labels = [minutes_to_time_string(y) for y in y_ticks]
    ax2.set_yticklabels(y_tick_labels)
    
    plt.tight_layout()
    
    return fig

def analyze_time_by_category(
    df: pd.DataFrame,
    time_column: str,
    category_column: str
) -> Tuple[Dict, plt.Figure]:
    """
    Analyze time variable grouped by a categorical variable
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe containing the data
    time_column : str
        Name of the time column
    category_column : str
        Name of the categorical column
        
    Returns:
    --------
    Tuple containing:
    - Dict with analysis results
    - matplotlib Figure
    """
    # Convert time to minutes
    minutes_series = df[time_column].apply(convert_time_to_minutes)
    
    # Create a copy of the dataframe with converted time
    df_copy = df.copy()
    df_copy[f'{time_column}_minutes'] = minutes_series
    
    # Group by category and calculate statistics
    grouped_stats = df_copy.groupby(category_column)[f'{time_column}_minutes'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    
    # Sort by mean time
    grouped_stats = grouped_stats.sort_values('mean')
    
    # Format times for display
    for stat in ['mean', 'median', 'min', 'max']:
        grouped_stats[f'{stat}_formatted'] = grouped_stats[stat].apply(minutes_to_time_string)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of means
    sns.barplot(x=category_column, y='mean', data=grouped_stats, ax=ax1)
    ax1.set_title(f"Mean {time_column} by {category_column}")
    ax1.set_ylabel("Time (minutes)")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add formatted time labels
    for i, row in grouped_stats.iterrows():
        ax1.text(i, row['mean'], row['mean_formatted'], ha='center', va='bottom')
    
    # Box plot
    sns.boxplot(x=category_column, y=f'{time_column}_minutes', data=df_copy, ax=ax2, order=grouped_stats[category_column])
    ax2.set_title(f"Distribution of {time_column} by {category_column}")
    ax2.set_ylabel("Time (minutes)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return grouped_stats.to_dict('records'), fig

def analyze_time_target(
    X: pd.DataFrame,
    y_original: pd.Series,
    feature_names: List[str]
) -> plt.Figure:
    """
    Special analysis for time-based target variables
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y_original : Series
        Original time target variable
    feature_names : List[str]
        List of feature names
        
    Returns:
    --------
    matplotlib Figure
    """
    # Convert times to minutes for analysis
    y_minutes = y_original.apply(convert_time_to_minutes)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Distribution of times
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(y_minutes.dropna(), bins=30, kde=True, ax=ax1)
    ax1.set_title('Distribution of Target Times')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Frequency')
    
    # Add lines for mean and median
    mean_time = y_minutes.mean()
    median_time = y_minutes.median()
    ax1.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {minutes_to_time_string(mean_time)}')
    ax1.axvline(median_time, color='green', linestyle='--', label=f'Median: {minutes_to_time_string(median_time)}')
    ax1.legend()
    
    # Find top correlating features
    numeric_X = X.select_dtypes(include=['number'])
    correlations = numeric_X.apply(lambda x: x.corr(y_minutes) if len(x.dropna()) > 0 else np.nan)
    top_features = correlations.abs().sort_values(ascending=False).head(3).index.tolist()
    
    # Plot relationships between top features and target
    for i, feature in enumerate(top_features):
        ax = fig.add_subplot(gs[i // 2, 1 + (i % 2)])
        
        # Scatter plot
        sns.regplot(x=X[feature], y=y_minutes, scatter_kws={'alpha': 0.5}, ax=ax)
        ax.set_title(f'{feature} vs Target Time')
        ax.set_xlabel(feature)
        ax.set_ylabel('Time (minutes)')
        
        # Add correlation info
        corr = correlations[feature]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
               fontsize=10, va='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    
    return fig

def time_binning_analysis(
    df: pd.DataFrame,
    time_column: str,
    target_column: str,
    n_bins: int = 10
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Analyze the relationship between binned time intervals and a target variable
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe containing the data
    time_column : str
        Name of the time column
    target_column : str
        Name of the target column
    n_bins : int
        Number of time bins to create
        
    Returns:
    --------
    Tuple containing:
    - DataFrame with binning results
    - matplotlib Figure
    """
    # Convert time to minutes
    minutes_series = df[time_column].apply(convert_time_to_minutes)
    
    # Create a copy of the dataframe with converted time
    df_copy = df.copy()
    df_copy[f'{time_column}_minutes'] = minutes_series
    
    # Create bins
    bins = pd.qcut(df_copy[f'{time_column}_minutes'].dropna(), n_bins, duplicates='drop')
    df_copy[f'{time_column}_bin'] = pd.qcut(df_copy[f'{time_column}_minutes'].dropna(), n_bins, duplicates='drop')
    
    # Calculate target rate or mean for each bin
    if df_copy[target_column].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df_copy[target_column]):
        # For numeric target, calculate mean
        bin_stats = df_copy.groupby(f'{time_column}_bin')[target_column].agg(['mean', 'count']).reset_index()
        target_metric = 'mean'
        y_label = f'Mean {target_column}'
    else:
        # For categorical target, calculate rate of most frequent value
        most_common = df_copy[target_column].value_counts().index[0]
        bin_stats = df_copy.groupby(f'{time_column}_bin').agg({
            target_column: lambda x: (x == most_common).mean(),
            f'{time_column}_minutes': 'count'
        }).reset_index()
        bin_stats.columns = [f'{time_column}_bin', 'rate', 'count']
        target_metric = 'rate'
        y_label = f'Rate of {most_common}'
    
    # Format bin ranges
    bin_stats['bin_start'] = bin_stats[f'{time_column}_bin'].apply(lambda x: minutes_to_time_string(x.left))
    bin_stats['bin_end'] = bin_stats[f'{time_column}_bin'].apply(lambda x: minutes_to_time_string(x.right))
    bin_stats['bin_label'] = bin_stats.apply(lambda x: f"{x['bin_start']} - {x['bin_end']}", axis=1)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Line plot
    ax1.plot(range(len(bin_stats)), bin_stats[target_metric], marker='o')
    ax1.set_title(f'{y_label} by {time_column} Bin')
    ax1.set_xticks(range(len(bin_stats)))
    ax1.set_xticklabels(bin_stats['bin_label'], rotation=45, ha='right')
    ax1.set_ylabel(y_label)
    
    # Bar plot with counts
    ax2.bar(range(len(bin_stats)), bin_stats['count'], alpha=0.7)
    ax2.set_title(f'Sample Count by {time_column} Bin')
    ax2.set_xticks(range(len(bin_stats)))
    ax2.set_xticklabels(bin_stats['bin_label'], rotation=45, ha='right')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    
    return bin_stats, fig

def analyze_optimal_time_thresholds(
    df: pd.DataFrame,
    time_column: str,
    target_column: str,
    n_thresholds: int = 50
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Find optimal time thresholds for predicting a target variable
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe containing the data
    time_column : str
        Name of the time column
    target_column : str
        Name of the target column
    n_thresholds : int
        Number of thresholds to test
        
    Returns:
    --------
    Tuple containing:
    - DataFrame with threshold results
    - matplotlib Figure
    """
    # Convert time to minutes
    minutes_series = df[time_column].apply(convert_time_to_minutes)
    
    # Create a copy of the dataframe with converted time
    df_copy = df.copy()
    df_copy[f'{time_column}_minutes'] = minutes_series
    
    # Determine if target is categorical or numeric
    is_categorical = not pd.api.types.is_numeric_dtype(df_copy[target_column])
    
    if is_categorical:
        # For categorical target, convert to binary if needed
        if df_copy[target_column].nunique() > 2:
            # Use most common value as positive class
            most_common = df_copy[target_column].value_counts().index[0]
            df_copy['target_binary'] = (df_copy[target_column] == most_common).astype(int)
            target_col = 'target_binary'
            target_value = most_common
        else:
            # Already binary
            df_copy['target_binary'] = df_copy[target_column].astype('category').cat.codes
            target_col = 'target_binary'
            target_value = df_copy[target_column].value_counts().index[0]
    else:
        # For numeric target, use median as threshold
        median = df_copy[target_column].median()
        df_copy['target_binary'] = (df_copy[target_column] > median).astype(int)
        target_col = 'target_binary'
        target_value = f">{median}"
    
    # Generate thresholds to test
    time_min = df_copy[f'{time_column}_minutes'].min()
    time_max = df_copy[f'{time_column}_minutes'].max()
    thresholds = np.linspace(time_min, time_max, n_thresholds)
    
    # Calculate target rates for each threshold
    results = []
    for threshold in thresholds:
        # For time variables, often lower is better, so use < comparison
        below_threshold = df_copy[df_copy[f'{time_column}_minutes'] < threshold]
        above_threshold = df_copy[df_copy[f'{time_column}_minutes'] >= threshold]
        
        below_rate = below_threshold[target_col].mean() if len(below_threshold) > 0 else np.nan
        above_rate = above_threshold[target_col].mean() if len(above_threshold) > 0 else np.nan
        
        difference = abs(below_rate - above_rate) if not np.isnan(below_rate) and not np.isnan(above_rate) else 0
        
        # Calculate sample sizes and percentages
        n_below = len(below_threshold)
        n_above = len(above_threshold)
        pct_below = n_below / len(df_copy) if len(df_copy) > 0 else 0
        
        results.append({
            'threshold_minutes': threshold,
            'threshold': minutes_to_time_string(threshold),
            'below_rate': below_rate,
            'above_rate': above_rate,
            'difference': difference,
            'n_below': n_below,
            'n_above': n_above,
            'pct_below': pct_below
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold (maximizing difference)
    optimal_idx = results_df['difference'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold_minutes']
    optimal_time = results_df.loc[optimal_idx, 'threshold']
    optimal_diff = results_df.loc[optimal_idx, 'difference']
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot rates by threshold
    ax1.plot(results_df['threshold_minutes'], results_df['below_rate'], label=f'Rate below threshold')
    ax1.plot(results_df['threshold_minutes'], results_df['above_rate'], label=f'Rate above threshold')
    ax1.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_time}')
    ax1.set_title(f'{target_value} Rate by {time_column} Threshold')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel(f'{target_value} Rate')
    ax1.legend()
    
    # Add x-axis labels as time strings
    x_ticks = ax1.get_xticks()
    x_tick_labels = [minutes_to_time_string(x) for x in x_ticks]
    ax1.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    
    # Plot differences by threshold
    ax2.plot(results_df['threshold_minutes'], results_df['difference'])
    ax2.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_time}')
    ax2.set_title(f'Difference in {target_value} Rate')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Absolute Difference')
    ax2.legend()
    
    # Add x-axis labels as time strings
    x_ticks = ax2.get_xticks()
    x_tick_labels = [minutes_to_time_string(x) for x in x_ticks]
    ax2.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    
    # Add annotation about optimal threshold
    optimal_below_rate = results_df.loc[optimal_idx, 'below_rate']
    optimal_above_rate = results_df.loc[optimal_idx, 'above_rate']
    
    annotation = (f"Optimal threshold: {optimal_time}\n"
                 f"Rate below: {optimal_below_rate:.2%}\n"
                 f"Rate above: {optimal_above_rate:.2%}\n"
                 f"Difference: {optimal_diff:.2%}")
    
    plt.figtext(0.5, 0.01, annotation, ha='center', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    return results_df, fig