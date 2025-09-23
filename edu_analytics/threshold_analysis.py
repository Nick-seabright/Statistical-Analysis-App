import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from .time_analysis import convert_time_to_minutes, minutes_to_time_string
import logging
logger = logging.getLogger(__name__)

def analyze_decision_boundaries(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None
) -> Dict:
    """
    Analyze decision boundaries and thresholds for each variable
    Parameters:
    -----------
    df : DataFrame
        The dataset to analyze
    target_column : str
        The target column name
    feature_columns : List[str], optional
        List of features to analyze, if None, all numeric columns are used
    Returns:
    --------
    Dictionary containing analysis results
    """
    # Create copy of dataframe
    df_analysis = df.copy()
    
    # Detect target type
    target_type = detect_target_type(df_analysis[target_column])
    
    # For visualization, we need a binary target or a numerical target
    if target_type == 'categorical':
        unique_values = np.unique(df_analysis[target_column])
        if len(unique_values) == 2:
            # Binary classification - encode as 0/1
            try:
                df_analysis['target_numeric'] = pd.Categorical(df_analysis[target_column]).codes
            except:
                # If categorical encoding fails, try direct conversion
                df_analysis['target_numeric'] = df_analysis[target_column].astype(int)
            target_column_analysis = 'target_numeric'
            # Get original class names for display
            class_names = unique_values
        else:
            # Multi-class - we'll analyze each class vs the rest
            logger.info(f"Target has {len(unique_values)} classes. Analyzing for each class vs. rest...")
            # Create one-hot encoding for visualization
            for value in unique_values:
                df_analysis[f'target_{value}'] = (df_analysis[target_column] == value).astype(int)
            # For initial analysis, use the first class
            target_column_analysis = f'target_{unique_values[0]}'
            class_names = [f'Not {unique_values[0]}', unique_values[0]]
    elif target_type in ['numeric', 'time']:
        # For regression, we'll bin the target into high/low around the median
        median = df_analysis[target_column].median()
        df_analysis['target_numeric'] = (df_analysis[target_column] > median).astype(int)
        target_column_analysis = 'target_numeric'
        if target_type == 'time':
            time_median = minutes_to_time_string(median)
            class_names = [f'≤ {time_median}', f'> {time_median}']
        else:
            class_names = [f'≤ {median:.2f}', f'> {median:.2f}']
    else:
        # Unsupported target type
        raise ValueError(f"Unsupported target type: {target_type}")
    
    # If no feature columns specified, use all numeric columns except target
    if feature_columns is None:
        feature_columns = df_analysis.select_dtypes(include=['int64', 'float64']).columns
        feature_columns = [col for col in feature_columns
                          if col != target_column and not col.startswith('target_')]
    
    # Analysis results dictionary
    results = {
        'target_column': target_column,
        'target_type': target_type,
        'class_names': class_names,
        'features': {}
    }
    
    logger.info(f"=== Single Variable Analysis for {target_type.capitalize()} Target ===")
    
    # Analyze each feature
    for feature in feature_columns:
        logger.info(f"\nAnalyzing {feature}:")
        
        # First, check the data type
        feature_type = detect_feature_type(df_analysis[feature])
        logger.info(f"Feature type: {feature_type}")
        
        # Initialize variables with default values
        is_categorical = False
        is_time_variable = False
        mapping = None
        
        # Handle different column types
        if feature_type == 'time':
            logger.info("Converting time format data to minutes...")
            df_analysis[f'{feature}_minutes'] = df_analysis[feature].apply(convert_time_to_minutes)
            analysis_feature = f'{feature}_minutes'
            is_time_variable = True
        elif feature_type == 'categorical':
            logger.info(f"{feature} is categorical. Converting to numeric for analysis...")
            # Use label encoding for analysis
            try:
                df_analysis[f'{feature}_encoded'] = pd.Categorical(df_analysis[feature]).codes
                analysis_feature = f'{feature}_encoded'
                is_categorical = True
            except:
                # If categorical encoding fails, use original feature
                logger.warning(f"Categorical encoding failed for {feature}. Using original feature.")
                analysis_feature = feature
        else:
            analysis_feature = feature
        
        # Calculate basic statistics
        basic_stats = df_analysis[analysis_feature].describe()
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90]
        
        # For time variables, use negative values for percentile calculation
        # so that lower times are considered "better"
        if is_time_variable:
            thresholds = np.percentile(-df_analysis[analysis_feature].dropna(), percentiles)
            thresholds = -thresholds  # Convert back to positive values
        else:
            thresholds = np.percentile(df_analysis[analysis_feature].dropna(), percentiles)
        
        # Calculate target rates at different thresholds
        threshold_results = []
        for percentile, threshold in zip(percentiles, thresholds):
            if is_time_variable:
                # For time variables, switch the comparison operators
                above_threshold = df_analysis[df_analysis[analysis_feature] < threshold][target_column_analysis].mean()
                below_threshold = df_analysis[df_analysis[analysis_feature] >= threshold][target_column_analysis].mean()
                threshold_display = minutes_to_time_string(threshold)
                logger.info(f"\nAt {percentile}th percentile (threshold = {threshold_display}):")
                logger.info(f"Percentage of '{class_names[1]}' when {feature} is below threshold (faster times): {above_threshold:.2%}")
                logger.info(f"Percentage of '{class_names[1]}' when {feature} is above threshold (slower times): {below_threshold:.2%}")
            else:
                # For other features, use standard comparison
                if is_categorical:
                    # For categorical, show the original category name if possible
                    try:
                        threshold_display = f"{threshold} ({df[feature].unique()[int(threshold)]})"
                    except:
                        threshold_display = f"{threshold}"
                else:
                    threshold_display = f"{threshold:.2f}"
                above_threshold = df_analysis[df_analysis[analysis_feature] > threshold][target_column_analysis].mean()
                below_threshold = df_analysis[df_analysis[analysis_feature] <= threshold][target_column_analysis].mean()
                logger.info(f"\nAt {percentile}th percentile (threshold = {threshold_display}):")
                logger.info(f"Percentage of '{class_names[1]}' when {feature} is above threshold: {above_threshold:.2%}")
                logger.info(f"Percentage of '{class_names[1]}' when {feature} is below threshold: {below_threshold:.2%}")
            
            difference = abs(above_threshold - below_threshold)
            logger.info(f"Difference: {difference:.2%}")
            
            threshold_results.append({
                'percentile': percentile,
                'threshold': threshold,
                'threshold_display': threshold_display,
                'above_rate': above_threshold,
                'below_rate': below_threshold,
                'difference': difference
            })
        
        # Find optimal splitting point
        best_threshold = None
        best_difference = 0
        best_above_rate = 0
        best_below_rate = 0
        
        threshold_range = np.linspace(
            df_analysis[analysis_feature].min(),
            df_analysis[analysis_feature].max(),
            num=100
        )
        
        for threshold in threshold_range:
            if is_time_variable:
                # For time variables, switch comparison operators
                above_threshold = df_analysis[df_analysis[analysis_feature] < threshold][target_column_analysis].mean()
                below_threshold = df_analysis[df_analysis[analysis_feature] >= threshold][target_column_analysis].mean()
            else:
                above_threshold = df_analysis[df_analysis[analysis_feature] > threshold][target_column_analysis].mean()
                below_threshold = df_analysis[df_analysis[analysis_feature] <= threshold][target_column_analysis].mean()
            
            difference = abs(above_threshold - below_threshold)
            if difference > best_difference:
                best_difference = difference
                best_threshold = threshold
                best_above_rate = above_threshold
                best_below_rate = below_threshold
        
        # Format optimal threshold for display
        if is_time_variable:
            optimal_threshold_display = minutes_to_time_string(best_threshold)
        elif is_categorical:
            try:
                optimal_threshold_display = f"{best_threshold} ({df[feature].unique()[int(best_threshold)]})"
            except:
                optimal_threshold_display = f"{best_threshold}"
        else:
            optimal_threshold_display = f"{best_threshold:.2f}"
        
        # Display results
        logger.info(f"\nOptimal splitting point: {optimal_threshold_display}")
        if is_time_variable:
            logger.info(f"Percentage of '{class_names[1]}' when {feature} is below threshold (faster times): {best_above_rate:.2%}")
            logger.info(f"Percentage of '{class_names[1]}' when {feature} is above threshold (slower times): {best_below_rate:.2%}")
        else:
            logger.info(f"Percentage of '{class_names[1]}' when {feature} is above threshold: {best_above_rate:.2%}")
            logger.info(f"Percentage of '{class_names[1]}' when {feature} is below threshold: {best_below_rate:.2%}")
        
        # Store results for this feature
        results['features'][feature] = {
            'feature_type': feature_type,
            'is_time_variable': is_time_variable,
            'is_categorical': is_categorical,
            'basic_stats': basic_stats.to_dict(),
            'threshold_results': threshold_results,
            'optimal_threshold': best_threshold,
            'optimal_threshold_display': optimal_threshold_display,
            'optimal_difference': best_difference,
            'optimal_above_rate': best_above_rate,
            'optimal_below_rate': best_below_rate
        }
        
        # Create dedicated visualization for single feature threshold
        if len(feature_columns) <= 5:  # Only show plots if there aren't too many features
            # Distribution plot
            fig = plt.figure(figsize=(12, 5))
            
            # For categorical variables, plot a bar chart instead of a histogram
            if is_categorical and not is_time_variable:
                plt.subplot(1, 2, 1)
                # Count occurrences for each category
                category_counts = df_analysis.groupby([feature, target_column_analysis]).size().unstack(fill_value=0)
                # Sort by total count
                category_counts['total'] = category_counts.sum(axis=1)
                category_counts = category_counts.sort_values('total', ascending=False).head(10)
                category_counts = category_counts.drop('total', axis=1)
                # Plot stacked bar chart
                category_counts.plot(kind='bar', stacked=True, ax=plt.gca())
                plt.title(f'Top 10 Categories of {feature} by Target')
                plt.legend(class_names)
                plt.xticks(rotation=45, ha='right')
                
                plt.subplot(1, 2, 2)
                # Calculate and plot the target rate for each category
                target_rates = df_analysis.groupby(feature)[target_column_analysis].mean().sort_values(ascending=False).head(10)
                target_rates.plot(kind='bar')
                plt.title(f'{class_names[1]} Rate by Top 10 {feature} Categories')
                plt.ylabel(f'{class_names[1]} Rate')
                plt.axhline(y=df_analysis[target_column_analysis].mean(), color='r', linestyle='--',
                           label=f'Overall Average: {df_analysis[target_column_analysis].mean():.2%}')
                plt.legend()
                plt.xticks(rotation=45, ha='right')
            else:
                # Regular numeric feature or time variable - use histograms
                plt.subplot(1, 2, 1)
                # Plot distribution by target
                for i in [0, 1]:
                    subset = df_analysis[df_analysis[target_column_analysis] == i][analysis_feature]
                    plt.hist(subset, alpha=0.5, label=class_names[i], bins=30)
                plt.axvline(x=best_threshold, color='r', linestyle='--', label='Best split')
                # For time variables, format the x-axis
                if is_time_variable:
                    plt.title(f'Distribution of {feature} (minutes) by Target')
                    plt.xlabel('Time (minutes)')
                else:
                    plt.title(f'Distribution of {feature} by Target')
                plt.legend()
                
                # Plot 2: Target rate by percentile
                plt.subplot(1, 2, 2)
                percentiles = np.linspace(0, 100, 20)
                if is_time_variable:
                    # For time variables, use negative values to reverse the percentile order
                    # This way, 90th percentile will be the fastest times
                    thresholds = np.percentile(-df_analysis[analysis_feature].dropna(), percentiles)
                    thresholds = -thresholds  # Convert back to positive values
                    target_rates = []
                    for threshold in thresholds:
                        # Use < for time variables (faster times are better)
                        rate = df_analysis[df_analysis[analysis_feature] < threshold][target_column_analysis].mean()
                        target_rates.append(rate)
                    plt.plot(percentiles, target_rates)
                    plt.title(f'{class_names[1]} Rate by Percentile of {feature}')
                    plt.xlabel('Percentile')
                    plt.ylabel(f'{class_names[1]} Rate')
                else:
                    # Original logic for non-time variables
                    thresholds = np.percentile(df_analysis[analysis_feature].dropna(), percentiles)
                    target_rates = []
                    for threshold in thresholds:
                        rate = df_analysis[df_analysis[analysis_feature] > threshold][target_column_analysis].mean()
                        target_rates.append(rate)
                    plt.plot(percentiles, target_rates)
                    plt.title(f'{class_names[1]} Rate by Percentile of {feature}')
                    plt.xlabel('Percentile')
                    plt.ylabel(f'{class_names[1]} Rate')
            
            plt.tight_layout()
            plt.show()
            
        # Add single feature threshold analysis graph
        if len(feature_columns) == 1:
            # Create dedicated visualization for single feature threshold
            feature = feature_columns[0]
            feature_result = results['features'][feature]
            
            # Create a new figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Get threshold and results
            best_threshold = feature_result['optimal_threshold']
            above_rate = feature_result['optimal_above_rate']
            below_rate = feature_result['optimal_below_rate']
            
            # Get the analysis feature name
            is_time = feature_result['is_time_variable']
            analysis_feature = f'{feature}_minutes' if is_time else feature
            
            # Plot 1: Distribution by target
            for i, class_name in enumerate(class_names):
                subset = df_analysis[df_analysis[target_column_analysis] == i][analysis_feature]
                ax1.hist(subset, alpha=0.5, label=class_name, bins=30)
            
            # Add threshold line
            ax1.axvline(x=best_threshold, color='r', linestyle='--', 
                       label=f'Optimal threshold: {feature_result["optimal_threshold_display"]}')
            ax1.set_title(f'Distribution of {feature} by {target_column}')
            ax1.legend()
            
            # Plot 2: Target rate by threshold
            # Generate x-axis points
            feature_min = df_analysis[analysis_feature].min()
            feature_max = df_analysis[analysis_feature].max()
            thresholds = np.linspace(feature_min, feature_max, 100)
            
            # Calculate rates at each threshold
            above_rates = []
            below_rates = []
            
            for threshold in thresholds:
                if is_time:
                    # For time variables, "above" means faster (smaller values)
                    above_mask = df_analysis[analysis_feature] < threshold
                    below_mask = df_analysis[analysis_feature] >= threshold
                else:
                    above_mask = df_analysis[analysis_feature] > threshold
                    below_mask = df_analysis[analysis_feature] <= threshold
                    
                above_rate_at_threshold = df_analysis[above_mask][target_column_analysis].mean() if above_mask.sum() > 0 else np.nan
                below_rate_at_threshold = df_analysis[below_mask][target_column_analysis].mean() if below_mask.sum() > 0 else np.nan
                
                above_rates.append(above_rate_at_threshold)
                below_rates.append(below_rate_at_threshold)
            
            # Plot rates
            if is_time:
                ax2.plot(thresholds, above_rates, 
                        label=f'Percentage of "{class_names[1]}" when {feature} is below threshold', 
                        color='blue')
                ax2.plot(thresholds, below_rates, 
                        label=f'Percentage of "{class_names[1]}" when {feature} is above threshold', 
                        color='green')
            else:
                ax2.plot(thresholds, above_rates, 
                        label=f'Percentage of "{class_names[1]}" when {feature} is above threshold', 
                        color='blue')
                ax2.plot(thresholds, below_rates, 
                        label=f'Percentage of "{class_names[1]}" when {feature} is below threshold', 
                        color='green')
            
            # Add optimal threshold line
            ax2.axvline(x=best_threshold, color='r', linestyle='--')
            
            # Annotate the optimal threshold
            if is_time:
                annotation_text = (
                    f'Optimal threshold: {feature_result["optimal_threshold_display"]}\n'
                    f'Percentage of "{class_names[1]}" when below threshold: {above_rate:.2%}\n'
                    f'Percentage of "{class_names[1]}" when above threshold: {below_rate:.2%}\n'
                    f'Difference: {abs(above_rate - below_rate):.2%}'
                )
            else:
                annotation_text = (
                    f'Optimal threshold: {feature_result["optimal_threshold_display"]}\n'
                    f'Percentage of "{class_names[1]}" when above threshold: {above_rate:.2%}\n'
                    f'Percentage of "{class_names[1]}" when below threshold: {below_rate:.2%}\n'
                    f'Difference: {abs(above_rate - below_rate):.2%}'
                )
                
            ax2.annotate(
                annotation_text,
                xy=(best_threshold, max(above_rate, below_rate)),
                xytext=(best_threshold + (feature_max - feature_min) * 0.1, 
                       max(above_rates + below_rates) * 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
            )
            
            ax2.set_title(f'Target Rate by {feature} Threshold')
            ax2.set_xlabel(feature)
            ax2.set_ylabel(f'Percentage of "{class_names[1]}"')
            ax2.legend()
            
            plt.tight_layout()
            results['single_feature_plot'] = fig
    
    # If we have exactly 2 features, analyze their combination
    if len(feature_columns) == 2:
        feature1, feature2 = feature_columns
        results['feature_combination'] = analyze_feature_combination(
            df_analysis, feature1, feature2, target_column_analysis, class_names
        )
    
    return results

def detect_target_type(series: pd.Series) -> str:
    """
    Detect the type of a target variable
    Parameters:
    -----------
    series : Series
        The target variable series
    Returns:
    --------
    str : Detected type ('categorical', 'numeric', or 'time')
    """
    # Check if it's time data
    if series.dtype == 'object':
        time_pattern = r'^\d{1,2}:\d{2}(:\d{2})?$'
        if series.astype(str).str.match(time_pattern).any():
            return 'time'
    
    # Check if it's categorical
    if series.dtype == 'object' or series.dtype.name == 'category' or series.nunique() < 10:
        return 'categorical'
    
    # Default to numeric
    return 'numeric'

def detect_feature_type(series: pd.Series) -> str:
    """
    Detect the type of a feature
    Parameters:
    -----------
    series : Series
        The feature series
    Returns:
    --------
    str : Detected type ('categorical', 'numeric', or 'time')
    """
    # Check if it's time data
    if series.dtype == 'object':
        time_pattern = r'^\d{1,2}:\d{2}(:\d{2})?$'
        if series.astype(str).str.match(time_pattern).any():
            return 'time'
    
    # Check if it's categorical
    if series.dtype == 'object' or series.dtype.name == 'category':
        return 'categorical'
    
    # Check if it's numeric but with few unique values (treat as categorical)
    if series.nunique() < 10 and series.nunique() / len(series) < 0.05:
        return 'categorical'
    
    # Default to numeric
    return 'numeric'

def analyze_feature_combination(
    df: pd.DataFrame,
    feature1: str,
    feature2: str,
    target_column: str,
    class_names: List[str]
) -> Dict:
    """
    Analyze the combination of two features
    Parameters:
    -----------
    df : DataFrame
        Dataframe with data for analysis
    feature1 : str
        First feature name
    feature2 : str
        Second feature name
    target_column : str
        Target column name
    class_names : List[str]
        Names of the target classes
    Returns:
    --------
    Dict with analysis results
    """
    logger.info(f"\nAnalyzing combination: {feature1} vs {feature2}")
    
    # Get the correct column names for analysis
    is_time1 = 'time' in detect_feature_type(df[feature1])
    is_time2 = 'time' in detect_feature_type(df[feature2])
    analysis_feature1 = f'{feature1}_minutes' if is_time1 else feature1
    analysis_feature2 = f'{feature2}_minutes' if is_time2 else feature2
    
    # Calculate medians
    median1 = df[analysis_feature1].median()
    median2 = df[analysis_feature2].median()
    
    # Format medians for display
    if is_time1:
        median1_display = minutes_to_time_string(median1)
    else:
        median1_display = f"{median1:.2f}"
    
    if is_time2:
        median2_display = minutes_to_time_string(median2)
    else:
        median2_display = f"{median2:.2f}"
    
    # Define quadrants based on whether each variable is a time variable
    quadrants = {
        'Q1': (
            (df[analysis_feature1] < median1 if is_time1 else df[analysis_feature1] > median1) &
            (df[analysis_feature2] < median2 if is_time2 else df[analysis_feature2] > median2)
        ),
        'Q2': (
            (df[analysis_feature1] >= median1 if is_time1 else df[analysis_feature1] <= median1) &
            (df[analysis_feature2] < median2 if is_time2 else df[analysis_feature2] > median2)
        ),
        'Q3': (
            (df[analysis_feature1] >= median1 if is_time1 else df[analysis_feature1] <= median1) &
            (df[analysis_feature2] >= median2 if is_time2 else df[analysis_feature2] <= median2)
        ),
        'Q4': (
            (df[analysis_feature1] < median1 if is_time1 else df[analysis_feature1] > median1) &
            (df[analysis_feature2] >= median2 if is_time2 else df[analysis_feature2] <= median2)
        )
    }
    
    # Calculate target rate for each quadrant
    quadrant_results = {}
    for quadrant, mask in quadrants.items():
        target_rate = df[mask][target_column].mean()
        sample_size = mask.sum()
        sample_pct = sample_size / len(df)
        quadrant_results[quadrant] = {
            'target_rate': target_rate,
            'sample_size': sample_size,
            'sample_pct': sample_pct
        }
        logger.info(f"{quadrant}: Percentage of '{class_names[1]}' = {target_rate:.2%}, Samples = {sample_size} ({sample_pct:.1%})")
    
    # Create visualization
    fig = plt.figure(figsize=(10, 8))
    
    # Determine if target is categorical for coloring
    is_categorical = len(class_names) <= 10  # Use discrete colors for categorical targets with few classes
    
    # Scatter plot colored by target
    if is_categorical:
        # Use a discrete colormap for categorical data
        unique_classes = np.unique(df[target_column])
        n_classes = len(unique_classes)
        
        # Create a discrete colormap with distinct colors
        cmap = plt.cm.get_cmap('tab10', n_classes)  # 'tab10' has distinct colors
        
        # Create scatter plot with discrete colors
        scatter = plt.scatter(
            df[analysis_feature1],
            df[analysis_feature2],
            c=df[target_column],
            cmap=cmap,
            alpha=0.6
        )
        
        # Create a custom legend with class names
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), 
                   markersize=10, label=class_names[i] if i < len(class_names) else f"Class {i}")
            for i in range(n_classes)
        ]
        plt.legend(handles=legend_elements, title="Classes")
    else:
        # Continuous colormap for numeric targets
        scatter = plt.scatter(
            df[analysis_feature1],
            df[analysis_feature2],
            c=df[target_column],
            cmap='coolwarm',
            alpha=0.6
        )
        plt.colorbar(scatter, label=f'{class_names[1]} Rate')
    
    # Invert axes for time variables if needed
    if is_time1:
        plt.gca().invert_xaxis()
    if is_time2:
        plt.gca().invert_yaxis()
    
    # Add axis labels without value judgments
    plt.xlabel(f"{feature1}")
    plt.ylabel(f"{feature2}")
    plt.title(f'Decision Boundary: {feature1} vs {feature2}')
    
    # Add median lines
    plt.axvline(x=median1, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=median2, color='r', linestyle='--', alpha=0.3)
    
    # Add quadrant labels with target rates
    label_coords = {
        'Q1': (df[analysis_feature1].min() + (median1 - df[analysis_feature1].min()) * 0.5, 
               df[analysis_feature2].max() - (df[analysis_feature2].max() - median2) * 0.5),
        'Q2': (df[analysis_feature1].max() - (df[analysis_feature1].max() - median1) * 0.5, 
               df[analysis_feature2].max() - (df[analysis_feature2].max() - median2) * 0.5),
        'Q3': (df[analysis_feature1].max() - (df[analysis_feature1].max() - median1) * 0.5, 
               df[analysis_feature2].min() + (median2 - df[analysis_feature2].min()) * 0.5),
        'Q4': (df[analysis_feature1].min() + (median1 - df[analysis_feature1].min()) * 0.5, 
               df[analysis_feature2].min() + (median2 - df[analysis_feature2].min()) * 0.5)
    }
    
    for quadrant, coords in label_coords.items():
        rate = quadrant_results[quadrant]['target_rate']
        size = quadrant_results[quadrant]['sample_size']
        plt.text(
            coords[0],
            coords[1],
            f"{quadrant}: {rate:.1%}\n(n={size})",
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.tight_layout()
    
    # Return results
    # Return results
    return {
        'feature1': feature1,
        'feature2': feature2,
        'is_time1': is_time1,
        'is_time2': is_time2,
        'median1': median1,
        'median2': median2,
        'median1_display': median1_display,
        'median2_display': median2_display,
        'quadrant_results': quadrant_results,
        'plot': fig
    }

def analyze_custom_threshold_combination(
    df: pd.DataFrame,
    feature1: str,
    threshold1: Union[float, str],
    feature2: str,
    threshold2: Union[float, str],
    target_column: str
) -> pd.DataFrame:
    """
    Analyze specific threshold combinations for two features
    Parameters:
    -----------
    df : DataFrame
        The dataset for analysis
    feature1 : str
        First feature name
    threshold1 : float or str
        Threshold for first feature
    feature2 : str
        Second feature name
    threshold2 : float or str
        Threshold for second feature
    target_column : str
        Target column name
    Returns:
    --------
    DataFrame with results for each combination
    """
    # Create a copy to avoid modifying the original dataframe
    df_analysis = df.copy()
    
    # Detect target type
    target_type = detect_target_type(df_analysis[target_column])
    
    # Process target to get binary variable for analysis
    if target_type == 'categorical':
        unique_values = np.unique(df_analysis[target_column])
        if len(unique_values) == 2:
            # Binary classification - use as is
            df_analysis['target_binary'] = pd.Categorical(df_analysis[target_column]).codes
            target_column_analysis = 'target_binary'
            class_names = unique_values
        else:
            # Multi-class - analyze first class vs rest
            logger.info(f"Target has {len(unique_values)} classes. Analyzing first class vs. rest.")
            first_class = unique_values[0]
            df_analysis['target_binary'] = (df_analysis[target_column] == first_class).astype(int)
            target_column_analysis = 'target_binary'
            class_names = [f'Not {first_class}', first_class]
    else:  # numeric or time
        # For numeric, binarize around median
        median = df_analysis[target_column].median()
        df_analysis['target_binary'] = (df_analysis[target_column] > median).astype(int)
        target_column_analysis = 'target_binary'
        if target_type == 'time':
            time_median = minutes_to_time_string(median)
            class_names = [f'≤ {time_median}', f'> {time_median}']
        else:
            class_names = [f'≤ {median:.2f}', f'> {median:.2f}']
    
    # Check if features are time variables
    is_time1 = 'time' in detect_feature_type(df_analysis[feature1])
    is_time2 = 'time' in detect_feature_type(df_analysis[feature2])
    
    # Convert time features if needed
    if is_time1:
        df_analysis[f'{feature1}_minutes'] = df_analysis[feature1].apply(convert_time_to_minutes)
        feature1_analysis = f'{feature1}_minutes'
        threshold1 = convert_time_to_minutes(threshold1) if isinstance(threshold1, str) else threshold1
    else:
        feature1_analysis = feature1
    
    if is_time2:
        df_analysis[f'{feature2}_minutes'] = df_analysis[feature2].apply(convert_time_to_minutes)
        feature2_analysis = f'{feature2}_minutes'
        threshold2 = convert_time_to_minutes(threshold2) if isinstance(threshold2, str) else threshold2
    else:
        feature2_analysis = feature2
    
    # Define conditions based on time variables
    conditions = {
        'Above Both': (
            (df_analysis[feature1_analysis] < threshold1 if is_time1 else df_analysis[feature1_analysis] > threshold1) &
            (df_analysis[feature2_analysis] < threshold2 if is_time2 else df_analysis[feature2_analysis] > threshold2)
        ),
        'Above 1, Below 2': (
            (df_analysis[feature1_analysis] < threshold1 if is_time1 else df_analysis[feature1_analysis] > threshold1) &
            (df_analysis[feature2_analysis] >= threshold2 if is_time2 else df_analysis[feature2_analysis] <= threshold2)
        ),
        'Below 1, Above 2': (
            (df_analysis[feature1_analysis] >= threshold1 if is_time1 else df_analysis[feature1_analysis] <= threshold1) &
            (df_analysis[feature2_analysis] < threshold2 if is_time2 else df_analysis[feature2_analysis] > threshold2)
        ),
        'Below Both': (
            (df_analysis[feature1_analysis] >= threshold1 if is_time1 else df_analysis[feature1_analysis] <= threshold1) &
            (df_analysis[feature2_analysis] >= threshold2 if is_time2 else df_analysis[feature2_analysis] <= threshold2)
        )
    }
    
    # Calculate results for each condition
    results = {}
    for condition_name, condition in conditions.items():
        subset = df_analysis[condition]
        results[condition_name] = {
            'count': len(subset),
            'target_1_rate': subset[target_column_analysis].mean() if len(subset) > 0 else np.nan,
            'percentage_of_total': len(subset) / len(df_analysis) if len(df_analysis) > 0 else 0
        }
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results).T
    
    # Log results
    logger.info(f"\nCustom Threshold Analysis Results:")
    logger.info(f"Feature 1: {feature1}, Threshold: {threshold1}")
    logger.info(f"Feature 2: {feature2}, Threshold: {threshold2}")
    logger.info("\nResults by quadrant:")
    for condition, row in results_df.iterrows():
        logger.info(f"{condition}: Percentage of '{class_names[1]}' = {row['target_1_rate']:.2%}, "
                   f"Samples = {row['count']} ({row['percentage_of_total']:.1%})")
    
    return results_df

def find_optimal_thresholds(
    df: pd.DataFrame,
    feature: str,
    target_column: str,
    n_thresholds: int = 100
) -> pd.DataFrame:
    """
    Find optimal thresholds for a feature to predict a target variable
    Parameters:
    -----------
    df : DataFrame
        Dataset for analysis
    feature : str
        Feature name
    target_column : str
        Target column name
    n_thresholds : int
        Number of thresholds to test
    Returns:
    --------
    DataFrame with threshold testing results
    """
    # Create a copy to avoid modifying the original dataframe
    df_analysis = df.copy()
    
    # Detect feature type
    feature_type = detect_feature_type(df_analysis[feature])
    is_time = feature_type == 'time'
    
    # Process feature if needed
    if is_time:
        df_analysis[f'{feature}_minutes'] = df_analysis[feature].apply(convert_time_to_minutes)
        analysis_feature = f'{feature}_minutes'
    else:
        analysis_feature = feature
    
    # Detect target type
    target_type = detect_target_type(df_analysis[target_column])
    
    # Process target to get binary variable for analysis
    if target_type == 'categorical':
        unique_values = np.unique(df_analysis[target_column])
        if len(unique_values) == 2:
            # Binary classification - use as is
            df_analysis['target_binary'] = pd.Categorical(df_analysis[target_column]).codes
            target_column_analysis = 'target_binary'
        else:
            # Multi-class - analyze first class vs rest
            first_class = unique_values[0]
            df_analysis['target_binary'] = (df_analysis[target_column] == first_class).astype(int)
            target_column_analysis = 'target_binary'
    else:  # numeric or time
        # For numeric, binarize around median
        median = df_analysis[target_column].median()
        df_analysis['target_binary'] = (df_analysis[target_column] > median).astype(int)
        target_column_analysis = 'target_binary'
    
    # Generate thresholds to test
    feature_min = df_analysis[analysis_feature].min()
    feature_max = df_analysis[analysis_feature].max()
    thresholds = np.linspace(feature_min, feature_max, n_thresholds)
    
    # Test each threshold
    results = []
    for threshold in thresholds:
        if is_time:
            # For time features, lower is typically better
            above_threshold_mask = df_analysis[analysis_feature] < threshold
            below_threshold_mask = df_analysis[analysis_feature] >= threshold
        else:
            # For regular features, higher is typically better
            above_threshold_mask = df_analysis[analysis_feature] > threshold
            below_threshold_mask = df_analysis[analysis_feature] <= threshold
        
        # Calculate target rates
        above_rate = df_analysis[above_threshold_mask][target_column_analysis].mean() if above_threshold_mask.sum() > 0 else np.nan
        below_rate = df_analysis[below_threshold_mask][target_column_analysis].mean() if below_threshold_mask.sum() > 0 else np.nan
        
        # Calculate separation (difference in rates)
        difference = abs(above_rate - below_rate) if not np.isnan(above_rate) and not np.isnan(below_rate) else 0
        
        # Format threshold for display
        if is_time:
            threshold_display = minutes_to_time_string(threshold)
        else:
            threshold_display = f"{threshold:.2f}"
        
        # Store results
        results.append({
            'threshold': threshold,
            'threshold_display': threshold_display,
            'above_rate': above_rate,
            'below_rate': below_rate,
            'difference': difference,
            'above_count': above_threshold_mask.sum(),
            'below_count': below_threshold_mask.sum()
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold (maximizing difference)
    if len(results_df) > 0:
        best_idx = results_df['difference'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_threshold_display = results_df.loc[best_idx, 'threshold_display']
        best_difference = results_df.loc[best_idx, 'difference']
        best_above_rate = results_df.loc[best_idx, 'above_rate']
        best_below_rate = results_df.loc[best_idx, 'below_rate']
        
        logger.info(f"\nOptimal threshold for {feature}: {best_threshold_display}")
        if is_time:
            logger.info(f"Percentage of target when below threshold (faster times): {best_above_rate:.2%}")
            logger.info(f"Percentage of target when above threshold (slower times): {best_below_rate:.2%}")
        else:
            logger.info(f"Percentage of target when above threshold: {best_above_rate:.2%}")
            logger.info(f"Percentage of target when below threshold: {best_below_rate:.2%}")
        logger.info(f"Difference: {best_difference:.2%}")
        
        # Add optimal threshold info to results
        results_df['is_optimal'] = (results_df.index == best_idx)
    
    return results_df