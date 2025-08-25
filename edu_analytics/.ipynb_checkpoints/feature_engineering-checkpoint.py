# edu_analytics/feature_engineering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
import logging

logger = logging.getLogger(__name__)

def analyze_correlations(df: pd.DataFrame, target_column: str) -> Union[pd.Series, pd.DataFrame]:
    """
    Perform and visualize correlation analysis, handling categorical target variables
    
    Parameters:
    -----------
    df : DataFrame
        The dataset for correlation analysis
    target_column : str
        The target column name
        
    Returns:
    --------
    Correlation results for the target column
    """
    # Create a copy to avoid modifying the original dataframe
    df_corr = df.copy()
    
    # Check if target column is categorical/non-numeric
    if df_corr[target_column].dtype == 'object' or df_corr[target_column].dtype.name == 'category':
        logger.info(f"Note: Target column '{target_column}' is categorical. Converting to numeric for correlation analysis.")
        
        # Create a temporary numeric version of the target for correlation analysis
        unique_values = df_corr[target_column].unique()
        if len(unique_values) == 2:
            # Binary classification - convert to 0/1
            positive_class = sorted(unique_values)[1]
            df_corr['target_numeric'] = (df_corr[target_column] == positive_class).astype(int)
            logger.info(f"Converted '{target_column}' to binary: {dict(zip(unique_values, [0, 1] if sorted(unique_values)[0] == unique_values[0] else [1, 0]))}")
            target_for_corr = 'target_numeric'
        else:
            # Multi-class - skip correlation with target as it's not meaningful
            logger.info(f"Skipping target correlation as '{target_column}' has multiple classes: {unique_values}")
            target_for_corr = None
    else:
        # Target is already numeric
        target_for_corr = target_column
    
    # Calculate correlation matrix for numeric columns only
    numeric_cols = df_corr.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for correlation analysis.")
        return None
    
    correlation_matrix = df_corr[numeric_cols].corr()
    
    # Target correlation analysis (if applicable)
    if target_for_corr and target_for_corr in correlation_matrix.columns:
        target_correlations = correlation_matrix[target_for_corr].sort_values(ascending=False)
        return target_correlations
    
    return correlation_matrix

def select_features(
    X: pd.DataFrame, 
    y: pd.Series, 
    target_type: str, 
    n_features: int = 10, 
    method: str = 'f_test'
) -> pd.DataFrame:
    """
    Select most important features based on statistical tests
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    target_type : str
        Type of target ('categorical' or 'numeric')
    n_features : int
        Number of features to select
    method : str
        Method for feature selection ('f_test', 'mutual_info')
        
    Returns:
    --------
    DataFrame with selected features
    """
    n_features = min(n_features, X.shape[1])
    
    if target_type == 'categorical':
        if method == 'f_test':
            selector = SelectKBest(f_classif, k=n_features)
        else:  # mutual_info
            selector = SelectKBest(mutual_info_classif, k=n_features)
    else:  # numeric
        if method == 'f_test':
            selector = SelectKBest(f_regression, k=n_features)
        else:  # mutual_info
            selector = SelectKBest(mutual_info_regression, k=n_features)
    
    X_new = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    
    # Create a dataframe with scores
    scores = selector.scores_
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': scores
    })
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    # Log selected features
    logger.info(f"Selected {len(selected_features)} features using {method} for {target_type} target")
    logger.info(f"Top features: {', '.join(selected_features[:5])}")
    
    return X[selected_features], feature_scores

def create_polynomial_features(
    X: pd.DataFrame, 
    features: List[str] = None, 
    degree: int = 2
) -> pd.DataFrame:
    """
    Create polynomial features for specified columns
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    features : List[str]
        List of features to transform, if None all numeric features are used
    degree : int
        Polynomial degree
        
    Returns:
    --------
    DataFrame with original and polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    # If no features specified, use all numeric features
    if features is None:
        features = X.select_dtypes(include=['number']).columns.tolist()
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(X[features])
    
    # Create feature names
    feature_names = poly.get_feature_names_out(features)
    
    # Create dataframe with polynomial features
    poly_df = pd.DataFrame(poly_features, columns=feature_names)
    
    # Remove the original features which will be duplicated
    poly_df = poly_df.iloc[:, len(features):]
    
    # Combine with original dataframe
    result = pd.concat([X.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)
    
    logger.info(f"Created {poly_df.shape[1]} polynomial features of degree {degree}")
    
    return result

def create_interaction_features(
    X: pd.DataFrame, 
    features: List[str] = None
) -> pd.DataFrame:
    """
    Create interaction features between specified columns
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    features : List[str]
        List of features to create interactions for, if None all numeric features are used
        
    Returns:
    --------
    DataFrame with original and interaction features
    """
    # If no features specified, use all numeric features
    if features is None:
        features = X.select_dtypes(include=['number']).columns.tolist()
    
    # Create interaction features
    interactions = {}
    for i, feat1 in enumerate(features):
        for feat2 in features[i+1:]:
            interaction_name = f"{feat1}_x_{feat2}"
            interactions[interaction_name] = X[feat1] * X[feat2]
    
    # Create dataframe with interaction features
    if interactions:
        interactions_df = pd.DataFrame(interactions)
        result = pd.concat([X.reset_index(drop=True), interactions_df.reset_index(drop=True)], axis=1)
        
        logger.info(f"Created {len(interactions)} interaction features")
        
        return result
    
    return X

def create_time_features(
    df: pd.DataFrame, 
    datetime_column: str
) -> pd.DataFrame:
    """
    Create time-based features from a datetime column
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    datetime_column : str
        Name of the datetime column
        
    Returns:
    --------
    DataFrame with additional time features
    """
    # Ensure datetime column is datetime type
    df = df.copy()
    if df[datetime_column].dtype != 'datetime64[ns]':
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    
    # Create time features
    df[f'{datetime_column}_year'] = df[datetime_column].dt.year
    df[f'{datetime_column}_month'] = df[datetime_column].dt.month
    df[f'{datetime_column}_day'] = df[datetime_column].dt.day
    df[f'{datetime_column}_dayofweek'] = df[datetime_column].dt.dayofweek
    df[f'{datetime_column}_hour'] = df[datetime_column].dt.hour
    df[f'{datetime_column}_minute'] = df[datetime_column].dt.minute
    
    # Create season feature
    month = df[datetime_column].dt.month
    df[f'{datetime_column}_season'] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        [0, 1, 2, 3]  # 0=Winter, 1=Spring, 2=Summer, 3=Fall
    )
    
    # Create is_weekend feature
    df[f'{datetime_column}_is_weekend'] = (df[datetime_column].dt.dayofweek >= 5).astype(int)
    
    logger.info(f"Created time features from {datetime_column}")
    
    return df

def visualize_feature_importance(
    importance_df: pd.DataFrame, 
    top_n: int = 15,
    title: str = "Feature Importance"
) -> plt.Figure:
    """
    Visualize feature importance
    
    Parameters:
    -----------
    importance_df : DataFrame
        DataFrame with columns 'Feature' and 'Score' or 'Importance'
    top_n : int
        Number of top features to display
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib Figure
    """
    # Check if we have 'Score' or 'Importance' column
    if 'Score' in importance_df.columns:
        score_col = 'Score'
    elif 'Importance' in importance_df.columns:
        score_col = 'Importance'
    else:
        raise ValueError("DataFrame must have either 'Score' or 'Importance' column")
    
    # Sort and select top features
    sorted_df = importance_df.sort_values(score_col, ascending=True).tail(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_df.plot(kind='barh', x='Feature', y=score_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    
    return fig