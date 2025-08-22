import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
import statsmodels.api as sm
from statsmodels.formula.api import ols

def perform_t_test(
    data: pd.DataFrame, 
    feature: str, 
    target: str, 
    alpha: float = 0.05,
    equal_var: bool = False
) -> Dict:
    """
    Perform t-test to compare means of a numeric feature between two groups.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset containing features and target
    feature : str
        The numerical feature to analyze
    target : str
        The binary categorical target variable
    alpha : float
        Significance level (default: 0.05)
    equal_var : bool
        Whether to assume equal variances (default: False)
        
    Returns:
    --------
    Dict containing test results
    """
    # Validate that target is binary
    unique_targets = data[target].unique()
    if len(unique_targets) != 2:
        raise ValueError(f"Target variable must be binary. Found {len(unique_targets)} unique values.")
    
    # Split data by target
    group1 = data[data[target] == unique_targets[0]][feature].dropna()
    group2 = data[data[target] == unique_targets[1]][feature].dropna()
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    
    # Prepare results
    results = {
        'feature': feature,
        'target': target,
        'groups': unique_targets.tolist(),
        'group1_mean': group1.mean(),
        'group2_mean': group2.mean(),
        'mean_difference': group2.mean() - group1.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'test_type': 'Student\'s t-test' if equal_var else 'Welch\'s t-test',
        'group1_n': len(group1),
        'group2_n': len(group2),
        'group1_std': group1.std(),
        'group2_std': group2.std(),
    }
    
    return results

def visualize_t_test(results: Dict) -> plt.Figure:
    """
    Create visualization for t-test results.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing t-test results
        
    Returns:
    --------
    matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Means with error bars
    labels = results['groups']
    means = [results['group1_mean'], results['group2_mean']]
    stds = [results['group1_std'], results['group2_std']]
    
    ax1.bar(labels, means, yerr=stds, alpha=0.7, capsize=10)
    ax1.set_ylabel(results['feature'])
    ax1.set_title(f"Mean of {results['feature']} by {results['target']}")
    
    # Add significance asterisk if applicable
    if results['significant']:
        max_val = max(means) + max(stds) + 0.1 * max(means)
        ax1.text(0.5, max_val, '*', ha='center', va='bottom', fontsize=20)
        ax1.plot([0, 1], [max_val*0.95, max_val*0.95], 'k-', linewidth=1)
    
    # Plot 2: Distribution comparison
    x = np.linspace(
        min(results['group1_mean'] - 3*results['group1_std'], 
            results['group2_mean'] - 3*results['group2_std']),
        max(results['group1_mean'] + 3*results['group1_std'], 
            results['group2_mean'] + 3*results['group2_std']),
        1000
    )
    
    # Plot normal distributions
    y1 = stats.norm.pdf(x, results['group1_mean'], results['group1_std'])
    y2 = stats.norm.pdf(x, results['group2_mean'], results['group2_std'])
    
    ax2.plot(x, y1, label=labels[0])
    ax2.plot(x, y2, label=labels[1])
    ax2.fill_between(x, y1, 0, alpha=0.3)
    ax2.fill_between(x, y2, 0, alpha=0.3)
    ax2.set_title('Distribution Comparison')
    ax2.legend()
    
    # Add p-value annotation
    p_value_text = f"p-value: {results['p_value']:.4f}"
    if results['significant']:
        p_value_text += " (significant)"
    else:
        p_value_text += " (not significant)"
    
    plt.figtext(0.5, 0.01, p_value_text, ha='center', fontsize=12)
    plt.figtext(0.5, 0.04, f"Test type: {results['test_type']}", ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def perform_chi_square(
    data: pd.DataFrame,
    feature: str,
    target: str,
    alpha: float = 0.05
) -> Dict:
    """
    Perform chi-square test of independence for categorical variables.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset containing features and target
    feature : str
        The categorical feature to analyze
    target : str
        The categorical target variable
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    Dict containing test results
    """
    # Create contingency table
    contingency_table = pd.crosstab(data[feature], data[target])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    
    # Prepare results
    results = {
        'feature': feature,
        'target': target,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'significant': p_value < alpha,
        'alpha': alpha,
        'contingency_table': contingency_table,
        'expected': expected,
        'cramers_v': cramers_v,
        'effect_size': interpret_cramers_v(cramers_v)
    }
    
    return results

def interpret_cramers_v(v: float) -> str:
    """Interpret Cramer's V effect size"""
    if v < 0.1:
        return "Negligible"
    elif v < 0.2:
        return "Weak"
    elif v < 0.3:
        return "Moderate"
    elif v < 0.4:
        return "Strong"
    else:
        return "Very strong"

def visualize_chi_square(results: Dict) -> plt.Figure:
    """
    Create visualization for chi-square test results.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing chi-square test results
        
    Returns:
    --------
    matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Heatmap of observed frequencies
    sns.heatmap(results['contingency_table'], annot=True, fmt='d', cmap='YlGnBu', ax=ax1)
    ax1.set_title('Observed Frequencies')
    
    # Plot 2: Mosaic plot
    from statsmodels.graphics.mosaicplot import mosaic
    ct = results['contingency_table']
    
    # Convert to format needed by mosaic plot
    props = {}
    if results['significant']:
        # Color significant cells
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if (results['contingency_table'].iloc[i, j] - results['expected'][i, j])**2 / results['expected'][i, j] > 3.84:  # Chi-square critical value for df=1, alpha=0.05
                    key = (ct.index[i], ct.columns[j])
                    props[key] = {'facecolor': 'salmon'}
    
    mosaic(results['contingency_table'].stack().reset_index().values, 
           ax=ax2, 
           properties=props)
    ax2.set_title('Mosaic Plot')
    
    # Add test results as text
    plt.figtext(0.5, 0.01, f"Chi-square: {results['chi2']:.2f}, p-value: {results['p_value']:.4f}", ha='center')
    sig_text = "Significant association" if results['significant'] else "No significant association"
    plt.figtext(0.5, 0.04, f"{sig_text} (α={results['alpha']})", ha='center')
    plt.figtext(0.5, 0.07, f"Cramer's V: {results['cramers_v']:.3f} ({results['effect_size']} effect size)", ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def perform_anova(
    data: pd.DataFrame,
    feature: str,
    group: str,
    alpha: float = 0.05,
    post_hoc: bool = True
) -> Dict:
    """
    Perform one-way ANOVA test.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset containing features and groups
    feature : str
        The numerical feature to analyze
    group : str
        The categorical grouping variable
    alpha : float
        Significance level (default: 0.05)
    post_hoc : bool
        Whether to perform Tukey's HSD post-hoc test (default: True)
        
    Returns:
    --------
    Dict containing test results
    """
    # Validate data
    groups = data[group].unique()
    if len(groups) < 3:
        raise ValueError(f"ANOVA requires at least 3 groups. Found {len(groups)}.")
    
    # Create formula and fit the model
    formula = f"{feature} ~ C({group})"
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Store group statistics
    group_stats = data.groupby(group)[feature].agg(['mean', 'std', 'count']).reset_index()
    
    # Prepare results
    results = {
        'feature': feature,
        'group': group,
        'f_statistic': anova_table['F'][0],
        'p_value': anova_table['PR(>F)'][0],
        'significant': anova_table['PR(>F)'][0] < alpha,
        'alpha': alpha,
        'df_between': anova_table['df'][0],
        'df_within': anova_table['df'][1],
        'ms_between': anova_table['sum_sq'][0] / anova_table['df'][0],
        'ms_within': anova_table['sum_sq'][1] / anova_table['df'][1],
        'ss_between': anova_table['sum_sq'][0],
        'ss_within': anova_table['sum_sq'][1],
        'ss_total': anova_table['sum_sq'][0] + anova_table['sum_sq'][1],
        'group_stats': group_stats,
        'model': model,
        'post_hoc': None
    }
    
    # Calculate effect size (Eta-squared)
    results['eta_squared'] = results['ss_between'] / results['ss_total']
    results['effect_size'] = interpret_eta_squared(results['eta_squared'])
    
    # Perform post-hoc test if requested and ANOVA is significant
    if post_hoc and results['significant']:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(data[feature], data[group], alpha=alpha)
        results['post_hoc'] = tukey
    
    return results

def interpret_eta_squared(eta_squared: float) -> str:
    """Interpret Eta-squared effect size"""
    if eta_squared < 0.01:
        return "Negligible"
    elif eta_squared < 0.06:
        return "Small"
    elif eta_squared < 0.14:
        return "Medium"
    else:
        return "Large"

def visualize_anova(results: Dict) -> plt.Figure:
    """
    Create visualization for ANOVA test results.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing ANOVA test results
        
    Returns:
    --------
    matplotlib Figure
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Group means with error bars
    ax1 = fig.add_subplot(gs[0, 0])
    
    group_stats = results['group_stats']
    x = range(len(group_stats))
    means = group_stats['mean'].values
    stds = group_stats['std'].values
    
    ax1.bar(x, means, yerr=stds, alpha=0.7, capsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_stats[results['group']].values, rotation=45, ha='right')
    ax1.set_ylabel(results['feature'])
    ax1.set_title(f"Mean of {results['feature']} by {results['group']}")
    
    # Plot 2: Box plot
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(x=results['group'], y=results['feature'], data=results['model'].model.data, ax=ax2)
    ax2.set_title('Distribution by Group')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Violin plot
    ax3 = fig.add_subplot(gs[1, 0])
    sns.violinplot(x=results['group'], y=results['feature'], data=results['model'].model.data, ax=ax3)
    ax3.set_title('Density by Group')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Post-hoc test results if available
    ax4 = fig.add_subplot(gs[1, 1])
    
    if results['post_hoc'] is not None:
        # Extract data from post-hoc test
        posthoc_data = pd.DataFrame(data=results['post_hoc']._results_table.data[1:], 
                                   columns=results['post_hoc']._results_table.data[0])
        
        # Convert reject to numeric for coloring
        posthoc_data['reject_numeric'] = posthoc_data['reject'].astype(int)
        
        # Create heatmap for pairwise comparisons
        group1 = [str(x) for x in posthoc_data['group1']]
        group2 = [str(x) for x in posthoc_data['group2']]
        
        # Create a square matrix for the heatmap
        unique_groups = sorted(set(group1 + group2))
        n_groups = len(unique_groups)
        
        # Initialize matrix with NaNs
        matrix = np.full((n_groups, n_groups), np.nan)
        
        # Fill in p-values
        for _, row in posthoc_data.iterrows():
            i = unique_groups.index(str(row['group1']))
            j = unique_groups.index(str(row['group2']))
            matrix[i, j] = row['p-adj']
            matrix[j, i] = row['p-adj']  # Mirror
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        
        # Plot heatmap
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=0.0, 
                   center=results['alpha'], annot=True, fmt='.3f',
                   square=True, linewidths=.5, cbar_kws={"shrink": .5},
                   ax=ax4, xticklabels=unique_groups, yticklabels=unique_groups)
        
        ax4.set_title('Tukey HSD p-values (pairwise)')
    else:
        ax4.text(0.5, 0.5, 'Post-hoc test not performed\n(ANOVA not significant or not requested)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Post-hoc Test')
        ax4.axis('off')
    
    # Add overall ANOVA results
    plt.figtext(0.5, 0.01, f"F({results['df_between']:.0f}, {results['df_within']:.0f}) = {results['f_statistic']:.2f}, p-value: {results['p_value']:.4f}", 
               ha='center', fontsize=12)
    
    sig_text = "Significant differences between groups" if results['significant'] else "No significant differences between groups"
    plt.figtext(0.5, 0.04, f"{sig_text} (α={results['alpha']})", ha='center')
    
    plt.figtext(0.5, 0.07, f"Eta-squared: {results['eta_squared']:.3f} ({results['effect_size']} effect size)", 
               ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def multi_group_analysis(
    data: pd.DataFrame,
    feature: str,
    groups: List[str],
    alpha: float = 0.05
) -> Dict:
    """
    Analyze a numerical feature across multiple grouping variables.
    This function automatically selects the appropriate test based on the number of groups.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset containing features and groups
    feature : str
        The numerical feature to analyze
    groups : List[str]
        List of categorical grouping variables to analyze
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    Dict containing test results for each group
    """
    results = {}
    
    for group in groups:
        # Get number of unique values in the group
        unique_values = data[group].nunique()
        
        if unique_values == 2:
            # For binary groups, use t-test
            results[group] = perform_t_test(data, feature, group, alpha)
        elif unique_values >= 3:
            # For 3+ groups, use ANOVA
            try:
                results[group] = perform_anova(data, feature, group, alpha)
            except Exception as e:
                print(f"Error performing ANOVA for {group}: {str(e)}")
                results[group] = {"error": str(e)}
        else:
            # Not enough groups for analysis
            results[group] = {"error": f"Group '{group}' has only {unique_values} unique value(s), at least 2 required."}
    
    return results

def categorical_association_analysis(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    alpha: float = 0.05
) -> Dict:
    """
    Analyze associations between categorical features and a target variable.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset containing features and target
    target : str
        The categorical target variable
    features : List[str]
        List of categorical features to analyze
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    Dict containing chi-square test results for each feature
    """
    results = {}
    
    for feature in features:
        try:
            results[feature] = perform_chi_square(data, feature, target, alpha)
        except Exception as e:
            print(f"Error performing chi-square test for {feature}: {str(e)}")
            results[feature] = {"error": str(e)}
    
    # Create summary dataframe
    summary = []
    for feature, result in results.items():
        if "error" not in result:
            summary.append({
                'feature': feature,
                'chi2': result['chi2'],
                'p_value': result['p_value'],
                'significant': result['significant'],
                'cramers_v': result['cramers_v'],
                'effect_size': result['effect_size']
            })
    
    results['summary'] = pd.DataFrame(summary).sort_values('cramers_v', ascending=False)
    
    return results

def numerical_correlation_analysis(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    alpha: float = 0.05
) -> Dict:
    """
    Analyze correlations between numerical features and a target variable.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset containing features and target
    target : str
        The numerical target variable
    features : List[str]
        List of numerical features to analyze
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    Dict containing correlation results for each feature
    """
    results = {}
    
    for feature in features:
        try:
            # Calculate Pearson correlation
            corr, p_value = stats.pearsonr(data[feature], data[target])
            
            results[feature] = {
                'feature': feature,
                'target': target,
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < alpha,
                'alpha': alpha,
                'r_squared': corr**2,
                'effect_size': interpret_correlation(corr)
            }
        except Exception as e:
            print(f"Error calculating correlation for {feature}: {str(e)}")
            results[feature] = {"error": str(e)}
    
    # Create summary dataframe
    summary = []
    for feature, result in results.items():
        if "error" not in result:
            summary.append({
                'feature': feature,
                'correlation': result['correlation'],
                'p_value': result['p_value'],
                'significant': result['significant'],
                'r_squared': result['r_squared'],
                'effect_size': result['effect_size']
            })
    
    results['summary'] = pd.DataFrame(summary).sort_values('correlation', ascending=False, key=abs)
    
    return results

def interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient effect size"""
    r_abs = abs(r)
    if r_abs < 0.1:
        return "Negligible"
    elif r_abs < 0.3:
        return "Weak"
    elif r_abs < 0.5:
        return "Moderate"
    elif r_abs < 0.7:
        return "Strong"
    else:
        return "Very strong"

def visualize_correlation_analysis(results: Dict) -> plt.Figure:
    """
    Create visualization for correlation analysis results.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing correlation analysis results
        
    Returns:
    --------
    matplotlib Figure
    """
    # Extract data from results
    features = [result['feature'] for feature, result in results.items() if feature != 'summary']
    correlations = [result['correlation'] for feature, result in results.items() if feature != 'summary']
    p_values = [result['p_value'] for feature, result in results.items() if feature != 'summary']
    significant = [result['significant'] for feature, result in results.items() if feature != 'summary']
    
    # Sort by absolute correlation
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    features = [features[i] for i in sorted_indices]
    correlations = [correlations[i] for i in sorted_indices]
    p_values = [p_values[i] for i in sorted_indices]
    significant = [significant[i] for i in sorted_indices]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot 1: Correlation bar chart
    colors = ['#1e88e5' if sig else '#d1d1d1' for sig in significant]
    ax1.barh(features, correlations, color=colors)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(f"Correlation with {results[features[0]]['target']}")
    ax1.set_xlabel('Pearson Correlation Coefficient')
    
    # Add correlation values as text
    for i, v in enumerate(correlations):
        ax1.text(v + np.sign(v)*0.01, i, f"{v:.3f}", va='center')
    
    # Plot 2: Scatter plots for top 4 features
    if len(features) > 0:
        num_plots = min(4, len(features))
        for i in range(num_plots):
            ax = ax2 if num_plots == 1 else ax2.flatten()[i] if num_plots == 4 else ax2[i]
            
            feature = features[i]
            result = results[feature]
            
            # Create scatter plot
            x = result['feature']
            y = result['target']
            
            sns.regplot(x=data[x], y=data[y], ax=ax, scatter_kws={'alpha': 0.5})
            ax.set_title(f"{x} vs {y} (r={result['correlation']:.3f})")
            
            # Add correlation text
            text = f"r = {result['correlation']:.3f}\np = {result['p_value']:.3e}\n"
            text += f"Effect: {result['effect_size']}"
            
            # Position text based on correlation direction
            if result['correlation'] < 0:
                ax.text(0.95, 0.95, text, transform=ax.transAxes, 
                       ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.1))
            else:
                ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                       ha='left', va='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    
    return fig