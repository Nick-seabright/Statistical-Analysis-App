# streamlit_app/components/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_correlation_heatmap(data, figsize=(10, 8)):
    """Create a correlation heatmap"""
    # Get numeric columns
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    
    # Create correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title('Correlation Matrix')
    
    return fig

def create_feature_importance_plot(importance_df, top_n=15, figsize=(10, 6)):
    """Create a feature importance plot"""
    # Sort by importance
    sorted_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    sorted_df.plot(kind='barh', x='feature', y='importance', ax=ax)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    return fig

def create_distribution_plot(data, column, hue=None, figsize=(10, 6)):
    """Create a distribution plot"""
    fig, ax = plt.subplots(figsize=figsize)
    
    if data[column].dtype in ['int64', 'float64']:
        # Numeric feature - histogram
        sns.histplot(data=data, x=column, hue=hue, kde=True, ax=ax)
    else:
        # Categorical feature - bar chart
        if hue is None:
            # Simple bar chart
            sns.countplot(data=data, x=column, ax=ax)
        else:
            # Stacked bar chart
            pd.crosstab(data[column], data[hue]).plot(kind='bar', stacked=True, ax=ax)
    
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig