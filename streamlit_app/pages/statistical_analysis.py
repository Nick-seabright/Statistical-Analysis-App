import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback

# Add the parent directory to path if running this file directly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from edu_analytics.statistical_tests import (
    perform_t_test, visualize_t_test,
    perform_chi_square, visualize_chi_square,
    perform_anova, visualize_anova,
    perform_correlation, visualize_correlation,
    numerical_correlation_analysis, visualize_correlation_analysis
)

def show_statistical_analysis():
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    # Check if data is processed
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("Please process your data first.")
        return
    
    st.markdown("<div class='subheader'>Statistical Analysis</div>", unsafe_allow_html=True)
    
    # Get data from session state
    data = st.session_state.data
    target_column = st.session_state.processed_data['target_column']
    selected_features = st.session_state.processed_data['selected_features']
    target_type = st.session_state.target_type
    
    # Create tabs for different statistical tests
    tab1, tab2, tab3, tab4 = st.tabs(["T-Tests", "Chi-Square Tests", "ANOVA Tests", "Correlation Analysis"])
    
    with tab1:
        st.markdown("<div class='subheader'>T-Test Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Compare means of numeric features between two groups.</div>", unsafe_allow_html=True)
        
        # For T-tests we need:
        # 1. A numeric feature
        # 2. A binary categorical grouping variable
        
        # Get numeric features
        numeric_features = [col for col in selected_features
                          if data[col].dtype in ['int64', 'float64']
                          or st.session_state.data_types.get(col) in ['integer', 'float', 'numeric']]
        
        # Get binary categorical features
        binary_features = [col for col in data.columns
                         if data[col].nunique() == 2]
        
        # Check if we have the required types of features
        if not numeric_features:
            st.warning("T-test requires numeric features. No numeric features found in your dataset.")
        elif not binary_features:
            st.warning("T-test requires binary categorical variables for grouping. No binary variables found in your dataset.")
        else:
            # Select feature and grouping variable
            numeric_feature = st.selectbox("Select numeric feature", numeric_features, key="ttest_feature")
            
            # For grouping, we can use the target if it's binary, or another binary feature
            if target_type == 'categorical' and data[target_column].nunique() == 2:
                default_group = target_column
            else:
                default_group = binary_features[0] if binary_features else None
                
            # Safety check before creating selectbox
            if default_group and binary_features:
                # Find the index of default_group in binary_features
                default_index = 0
                if default_group in binary_features:
                    default_index = binary_features.index(default_group)
                    
                grouping_variable = st.selectbox(
                    "Select grouping variable (must be binary)",
                    binary_features,
                    index=default_index,
                    key="ttest_group"
                )
                
                # Perform t-test
                if st.button("Perform T-Test", key="run_ttest"):
                    try:
                        with st.spinner("Performing T-test..."):
                            # Perform t-test
                            ttest_results = perform_t_test(
                                data=data,
                                feature=numeric_feature,
                                target=grouping_variable,
                                alpha=0.05,
                                equal_var=False  # Use Welch's t-test by default
                            )
                            
                            # Display results
                            st.markdown("### T-Test Results")
                            
                            # Key metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    label=f"Mean for {ttest_results['groups'][0]}",
                                    value=f"{ttest_results['group1_mean']:.3f}"
                                )
                            with col2:
                                st.metric(
                                    label=f"Mean for {ttest_results['groups'][1]}",
                                    value=f"{ttest_results['group2_mean']:.3f}"
                                )
                            with col3:
                                st.metric(
                                    label="Mean Difference",
                                    value=f"{ttest_results['mean_difference']:.3f}",
                                    delta=f"p={ttest_results['p_value']:.4f}"
                                )
                            
                            # Test details
                            st.markdown("**Test Details:**")
                            st.write(f"t-statistic: {ttest_results['t_statistic']:.4f}")
                            st.write(f"p-value: {ttest_results['p_value']:.4f}")
                            st.write(f"Degrees of freedom: {ttest_results['group1_n'] + ttest_results['group2_n'] - 2}")
                            
                            # Interpretation
                            if ttest_results['significant']:
                                st.success(f"The difference is statistically significant (p < {ttest_results['alpha']}).")
                            else:
                                st.info(f"The difference is not statistically significant (p > {ttest_results['alpha']}).")
                            
                            # Visualize the results
                            fig = visualize_t_test(ttest_results)
                            st.pyplot(fig)
                            plt.close(fig)  # Close the figure to prevent interference
                            
                            # Store the results for the report
                            if 'statistical_tests' not in st.session_state.report_data:
                                st.session_state.report_data['statistical_tests'] = {}
                            test_key = f"ttest_{numeric_feature}_by_{grouping_variable}"
                            st.session_state.report_data['statistical_tests'][test_key] = {
                                'type': 't-test',
                                'results': ttest_results,
                                'description': f"T-test comparing {numeric_feature} means between {grouping_variable} groups"
                            }
                    except Exception as e:
                        st.error(f"Error performing t-test: {str(e)}")
                        st.code(traceback.format_exc())
            else:
                st.warning("No binary categorical variables found for grouping. T-test requires a binary grouping variable.")
    
    with tab2:
        st.markdown("<div class='subheader'>Chi-Square Test Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Test for association between categorical variables.</div>", unsafe_allow_html=True)
        
        # For Chi-square tests we need:
        # 1. Two categorical variables
        
        # Get categorical features
        categorical_features = [col for col in data.columns
                              if data[col].dtype == 'object'
                              or st.session_state.data_types.get(col) in ['categorical', 'boolean']]
        
        # Handle the case when there are not enough categorical features
        if len(categorical_features) < 2:
            st.warning("Chi-square test requires at least two categorical variables. Not enough categorical variables found in your dataset.")
        else:
            # Select feature and target
            categorical_feature = st.selectbox("Select categorical feature", categorical_features, key="chisq_feature")
            
            # Get remaining categorical features (excluding the one already selected)
            remaining_categorical = [col for col in categorical_features if col != categorical_feature]
            
            if not remaining_categorical:
                st.warning("Need at least one more categorical variable for chi-square test.")
            else:
                # Set default target - either the target column if categorical, or the first available categorical feature
                if target_type == 'categorical' and target_column in remaining_categorical:
                    default_index = remaining_categorical.index(target_column)
                else:
                    default_index = 0
                
                # Select the second categorical variable with proper index handling
                categorical_target = st.selectbox(
                    "Select categorical target",
                    remaining_categorical,
                    index=default_index,
                    key="chisq_target"
                )
                
                # Perform chi-square test
                if st.button("Perform Chi-Square Test", key="run_chisq"):
                    try:
                        with st.spinner("Performing Chi-Square test..."):
                            # Perform chi-square test
                            chisq_results = perform_chi_square(
                                data=data,
                                feature=categorical_feature,
                                target=categorical_target,
                                alpha=0.05
                            )
                            
                            # Display results
                            st.markdown("### Chi-Square Test Results")
                            
                            # Key metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    label="Chi-Square Statistic",
                                    value=f"{chisq_results['chi2']:.3f}"
                                )
                            with col2:
                                st.metric(
                                    label="p-value",
                                    value=f"{chisq_results['p_value']:.4f}"
                                )
                            with col3:
                                st.metric(
                                    label="Cramer's V",
                                    value=f"{chisq_results['cramers_v']:.3f}",
                                    delta=chisq_results['effect_size']
                                )
                            
                            # Interpretation
                            if chisq_results['significant']:
                                st.success(f"There is a statistically significant association between {categorical_feature} and {categorical_target} (p < {chisq_results['alpha']}).")
                                st.write(f"The effect size is {chisq_results['effect_size'].lower()} (Cramer's V = {chisq_results['cramers_v']:.3f}).")
                            else:
                                st.info(f"There is no statistically significant association between {categorical_feature} and {categorical_target} (p > {chisq_results['alpha']}).")
                            
                            # Visualize the results
                            fig = visualize_chi_square(chisq_results)
                            st.pyplot(fig)
                            plt.close(fig)  # Close the figure to prevent interference
                            
                            # Display contingency table
                            st.markdown("### Contingency Table")
                            st.dataframe(chisq_results['contingency_table'])
                            
                            # Store the results for the report
                            if 'statistical_tests' not in st.session_state.report_data:
                                st.session_state.report_data['statistical_tests'] = {}
                            test_key = f"chisq_{categorical_feature}_by_{categorical_target}"
                            st.session_state.report_data['statistical_tests'][test_key] = {
                                'type': 'chi-square',
                                'results': chisq_results,
                                'description': f"Chi-square test of association between {categorical_feature} and {categorical_target}"
                            }
                    except Exception as e:
                        st.error(f"Error performing chi-square test: {str(e)}")
                        st.code(traceback.format_exc())
    
    with tab3:
        st.markdown("<div class='subheader'>ANOVA Test Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Compare means of a numeric feature across multiple groups.</div>", unsafe_allow_html=True)
        
        # For ANOVA tests we need:
        # 1. A numeric feature
        # 2. A categorical grouping variable with 3+ groups
        
        # Get numeric features
        numeric_features = [col for col in selected_features
                          if data[col].dtype in ['int64', 'float64']
                          or st.session_state.data_types.get(col) in ['integer', 'float', 'numeric']]
        
        # Get categorical features with 3+ groups
        multi_cat_features = [col for col in data.columns
                            if data[col].nunique() >= 3
                            and (data[col].dtype == 'object'
                                or st.session_state.data_types.get(col) in ['categorical'])]
        
        # Check if we have the required types of features
        if not numeric_features:
            st.warning("ANOVA requires numeric features. No numeric features found in your dataset.")
        elif not multi_cat_features:
            st.warning("ANOVA requires categorical features with 3+ groups. No such features found in your dataset.")
        else:
            # Select feature and grouping variable
            numeric_feature = st.selectbox("Select numeric feature for ANOVA", numeric_features, key="anova_numeric")
            grouping_variable = st.selectbox(
                "Select grouping variable (3+ categories)",
                multi_cat_features,
                key="anova_group"
            )
            
            # Option for post-hoc test
            post_hoc = st.checkbox("Perform post-hoc Tukey HSD test", value=True)
            
            # Perform ANOVA
            if st.button("Perform ANOVA Test", key="run_anova"):
                try:
                    with st.spinner("Performing ANOVA test..."):
                        # Perform ANOVA
                        anova_results = perform_anova(
                            data=data,
                            feature=numeric_feature,
                            group=grouping_variable,
                            alpha=0.05,
                            post_hoc=post_hoc
                        )
                        
                        # Display results
                        st.markdown("### ANOVA Results")
                        
                        # Key metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                label="F-statistic",
                                value=f"{anova_results['f_statistic']:.3f}"
                            )
                        with col2:
                            st.metric(
                                label="p-value",
                                value=f"{anova_results['p_value']:.4f}"
                            )
                        with col3:
                            st.metric(
                                label="Eta-squared",
                                value=f"{anova_results['eta_squared']:.3f}",
                                delta=anova_results['effect_size']
                            )
                        
                        # Interpretation
                        if anova_results['significant']:
                            st.success(f"There are statistically significant differences in {numeric_feature} between {grouping_variable} groups (p < {anova_results['alpha']}).")
                            st.write(f"The effect size is {anova_results['effect_size'].lower()} (Eta-squared = {anova_results['eta_squared']:.3f}).")
                        else:
                            st.info(f"There are no statistically significant differences in {numeric_feature} between {grouping_variable} groups (p > {anova_results['alpha']}).")
                        
                        # Visualize the results
                        fig = visualize_anova(anova_results)
                        st.pyplot(fig)
                        plt.close(fig)  # Close the figure to prevent interference
                        
                        # Display group statistics
                        st.markdown("### Group Statistics")
                        st.dataframe(anova_results['group_stats'])
                        
                        # Display post-hoc results if performed
                        if anova_results['post_hoc'] is not None:
                            st.markdown("### Post-hoc Tukey HSD Results")
                            st.text(str(anova_results['post_hoc']))
                        
                        # Store the results for the report
                        if 'statistical_tests' not in st.session_state.report_data:
                            st.session_state.report_data['statistical_tests'] = {}
                        test_key = f"anova_{numeric_feature}_by_{grouping_variable}"
                        st.session_state.report_data['statistical_tests'][test_key] = {
                            'type': 'anova',
                            'results': anova_results,
                            'description': f"ANOVA test comparing {numeric_feature} means across {grouping_variable} groups"
                        }
                except Exception as e:
                    st.error(f"Error performing ANOVA test: {str(e)}")
                    st.code(traceback.format_exc())
            else:
                if not numeric_features:
                    st.warning("No numeric features found. ANOVA requires a numeric feature.")
                elif not multi_cat_features:
                    st.warning("No categorical features with 3+ groups found. ANOVA requires a categorical grouping variable with at least 3 groups.")
    
    with tab4:
        st.markdown("<div class='subheader'>Correlation Analysis</div>", unsafe_allow_html=True)
        if target_type in ['numeric', 'time']:
            st.markdown("<div class='info-text'>Analyze correlations between numeric features and the target variable.</div>", unsafe_allow_html=True)
            
            # Get numeric features
            numeric_features = [col for col in selected_features
                              if data[col].dtype in ['int64', 'float64']
                              or st.session_state.data_types.get(col) in ['integer', 'float', 'numeric']]
            
            if numeric_features:
                # Select features for correlation analysis
                selected_corr_features = st.multiselect(
                    "Select numeric features for correlation analysis",
                    numeric_features,
                    default=numeric_features[:5] if len(numeric_features) > 5 else numeric_features
                )
                
                if selected_corr_features:
                    # Perform correlation analysis
                    if st.button("Perform Correlation Analysis", key="run_corr"):
                        try:
                            with st.spinner("Analyzing correlations..."):
                                # Perform correlation analysis
                                corr_results = numerical_correlation_analysis(
                                    data=data,
                                    target=target_column,
                                    features=selected_corr_features,
                                    alpha=0.05
                                )
                                
                                # Display summary table
                                st.markdown("### Correlation Summary")
                                st.dataframe(corr_results['summary'])
                                
                                # Add clear separation
                                st.markdown("---")
                                
                                # SECTION 1: Bar chart of correlations - in its own container
                                st.markdown("### Correlation Coefficients")
                                
                                # Extract data for visualization
                                features = [feat for feat in selected_corr_features if "error" not in corr_results[feat]]
                                correlations = [corr_results[feat]['correlation'] for feat in features]
                                significant = [corr_results[feat]['significant'] for feat in features]
                                
                                # Sort by absolute correlation
                                sorted_indices = np.argsort(np.abs(correlations))[::-1]
                                features = [features[i] for i in sorted_indices]
                                correlations = [correlations[i] for i in sorted_indices]
                                significant = [significant[i] for i in sorted_indices]
                                
                                # Create and display the bar chart
                                fig1 = plt.figure(figsize=(10, 6))
                                ax1 = fig1.add_subplot(111)
                                
                                colors = ['#1e88e5' if sig else '#d1d1d1' for sig in significant]
                                ax1.barh(features, correlations, color=colors)
                                ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                ax1.set_title(f"Correlation with {target_column}")
                                ax1.set_xlabel('Pearson Correlation Coefficient')
                                
                                # Add correlation values as text
                                for i, v in enumerate(correlations):
                                    ax1.text(v + np.sign(v)*0.01, i, f"{v:.3f}", va='center')
                                
                                plt.tight_layout()
                                st.pyplot(fig1)
                                plt.close(fig1)  # Close the figure to prevent interference
                                
                                # Add spacing between visualizations
                                st.markdown("---")
                                
                                # SECTION 2: Scatter plots - in separate containers
                                st.markdown("### Detailed Correlations")
                                
                                # Extract data for visualization
                                features = [feat for feat in selected_corr_features if "error" not in corr_results[feat]]
                                if not features:
                                    st.info("No valid features for correlation visualization.")
                                else:
                                    # Sort by absolute correlation
                                    correlations = [corr_results[feat]['correlation'] for feat in features]
                                    sorted_indices = np.argsort(np.abs(correlations))[::-1]
                                    features = [features[i] for i in sorted_indices]
                                    
                                    # Pagination system for features
                                    plots_per_page = 4  # Show 4 plots per page
                                    num_features = len(features)
                                    num_pages = (num_features + plots_per_page - 1) // plots_per_page  # Ceiling division
                                    
                                    if num_pages > 1:
                                        page_number = st.selectbox(
                                            f"Page (showing 4 plots per page, {num_features} total features)",
                                            options=list(range(1, num_pages + 1)),
                                            format_func=lambda x: f"Page {x} of {num_pages}"
                                        )
                                        start_idx = (page_number - 1) * plots_per_page
                                        end_idx = min(start_idx + plots_per_page, num_features)
                                        current_features = features[start_idx:end_idx]
                                        st.info(f"Showing correlations {start_idx+1}-{end_idx} of {num_features} total")
                                    else:
                                        current_features = features
                                    
                                    # Create a 2x2 grid for the current page using Streamlit columns
                                    # First row
                                    if len(current_features) > 0:
                                        col1, col2 = st.columns(2)
                                        
                                        # First plot (top-left)
                                        with col1:
                                            feature = current_features[0]
                                            result = corr_results[feature]
                                            
                                            fig = plt.figure(figsize=(6, 5))
                                            ax = fig.add_subplot(111)
                                            
                                            sns.regplot(
                                                x=result['data'][feature], 
                                                y=result['data'][target_column], 
                                                ax=ax, 
                                                scatter_kws={'alpha': 0.5}
                                            )
                                            
                                            ax.set_title(f"{feature} vs {target_column}\nr={result['correlation']:.3f}")
                                            
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
                                            
                                            plt.tight_layout(pad=2.0)
                                            st.pyplot(fig)
                                            plt.close(fig)
                                        
                                        # Second plot (top-right)
                                        if len(current_features) > 1:
                                            with col2:
                                                feature = current_features[1]
                                                result = corr_results[feature]
                                                
                                                fig = plt.figure(figsize=(6, 5))
                                                ax = fig.add_subplot(111)
                                                
                                                sns.regplot(
                                                    x=result['data'][feature], 
                                                    y=result['data'][target_column], 
                                                    ax=ax, 
                                                    scatter_kws={'alpha': 0.5}
                                                )
                                                
                                                ax.set_title(f"{feature} vs {target_column}\nr={result['correlation']:.3f}")
                                                
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
                                                
                                                plt.tight_layout(pad=2.0)
                                                st.pyplot(fig)
                                                plt.close(fig)
                                    
                                    # Second row
                                    if len(current_features) > 2:
                                        col3, col4 = st.columns(2)
                                        
                                        # Third plot (bottom-left)
                                        with col3:
                                            feature = current_features[2]
                                            result = corr_results[feature]
                                            
                                            fig = plt.figure(figsize=(6, 5))
                                            ax = fig.add_subplot(111)
                                            
                                            sns.regplot(
                                                x=result['data'][feature], 
                                                y=result['data'][target_column], 
                                                ax=ax, 
                                                scatter_kws={'alpha': 0.5}
                                            )
                                            
                                            ax.set_title(f"{feature} vs {target_column}\nr={result['correlation']:.3f}")
                                            
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
                                            
                                            plt.tight_layout(pad=2.0)
                                            st.pyplot(fig)
                                            plt.close(fig)
                                        
                                        # Fourth plot (bottom-right)
                                        if len(current_features) > 3:
                                            with col4:
                                                feature = current_features[3]
                                                result = corr_results[feature]
                                                
                                                fig = plt.figure(figsize=(6, 5))
                                                ax = fig.add_subplot(111)
                                                
                                                sns.regplot(
                                                    x=result['data'][feature], 
                                                    y=result['data'][target_column], 
                                                    ax=ax, 
                                                    scatter_kws={'alpha': 0.5}
                                                )
                                                
                                                ax.set_title(f"{feature} vs {target_column}\nr={result['correlation']:.3f}")
                                                
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
                                                
                                                plt.tight_layout(pad=2.0)
                                                st.pyplot(fig)
                                                plt.close(fig)
                                
                                # Add additional space before next section
                                st.markdown("---")
                                
                                # Show individual correlations
                                st.markdown("### Detailed Correlation Results")
                                for feature in selected_corr_features:
                                    if "error" not in corr_results[feature]:
                                        result = corr_results[feature]
                                        st.markdown(f"#### Correlation with {feature}")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric(
                                                label=f"Correlation",
                                                value=f"{result['correlation']:.3f}"
                                            )
                                        with col2:
                                            st.metric(
                                                label="p-value",
                                                value=f"{result['p_value']:.4f}"
                                            )
                                        with col3:
                                            st.metric(
                                                label="R²",
                                                value=f"{result['r_squared']:.3f}",
                                                delta=result['effect_size']
                                            )
                                        
                                        # Interpretation
                                        if result['significant']:
                                            st.success(f"There is a statistically significant correlation between {feature} and {target_column} (p < {result['alpha']}).")
                                            st.write(f"The effect size is {result['effect_size'].lower()} (r = {result['correlation']:.3f}, R² = {result['r_squared']:.3f}).")
                                        else:
                                            st.info(f"There is no statistically significant correlation between {feature} and {target_column} (p > {result['alpha']}).")
                                        
                                        st.markdown("---")
                                
                                # Store the results for the report
                                if 'statistical_tests' not in st.session_state.report_data:
                                    st.session_state.report_data['statistical_tests'] = {}
                                st.session_state.report_data['statistical_tests']['correlation_analysis'] = {
                                    'type': 'correlation',
                                    'results': corr_results,
                                    'description': f"Correlation analysis between numeric features and {target_column}"
                                }
                        except Exception as e:
                            st.error(f"Error performing correlation analysis: {str(e)}")
                            st.code(traceback.format_exc())
                else:
                    st.warning("Please select at least one feature for correlation analysis.")
            else:
                st.warning("No numeric features found. Correlation analysis requires numeric features.")
        else:
            st.markdown("<div class='info-text'>For categorical targets, correlation analysis is not directly applicable. Consider using T-Tests or ANOVA instead.</div>", unsafe_allow_html=True)
            
            # Offer alternative analysis for categorical targets
            st.markdown("### Alternative: Feature Importance Analysis")
            # Get numeric features
            numeric_features = [col for col in selected_features
                              if data[col].dtype in ['int64', 'float64']
                              or st.session_state.data_types.get(col) in ['integer', 'float', 'numeric']]
            
            if numeric_features:
                # Allow users to select which features to include
                use_all_numeric = st.checkbox("Use all numeric features", value=True)
                
                if use_all_numeric:
                    features_for_analysis = numeric_features
                else:
                    # Let the user select specific numeric features
                    features_for_analysis = st.multiselect(
                        "Select numeric features for importance analysis",
                        options=numeric_features,
                        default=numeric_features[:min(10, len(numeric_features))]  # Default to first 10 features
                    )
                    
                    # Show warning if no features selected
                    if not features_for_analysis:
                        st.warning("Please select at least one feature for importance analysis.")
                        return
                
                # Display the number of selected features
                st.info(f"Using {len(features_for_analysis)} features for importance analysis.")
                
                # Calculate feature importance for categorical target
                if st.button("Analyze Feature Importance", key="run_importance"):
                    try:
                        with st.spinner("Analyzing feature importance..."):
                            # Prepare data for analysis
                            X_subset = data[features_for_analysis].copy()  # Use filtered features
                            y_subset = data[target_column].copy()
                            
                            # Proper handling of missing values
                            # 1. Remove rows with NaN in target
                            valid_target_mask = ~y_subset.isna()
                            X_subset = X_subset[valid_target_mask]
                            y_subset = y_subset[valid_target_mask]
                            
                            # 2. Fill missing values in features with their mean
                            X_subset = X_subset.fillna(X_subset.mean())
                            
                            # 3. Check if we have any columns that are still all NaN (if mean didn't work)
                            all_nan_cols = X_subset.columns[X_subset.isna().all()].tolist()
                            if all_nan_cols:
                                X_subset = X_subset.drop(columns=all_nan_cols)
                                st.warning(f"Dropped columns with all NaN values: {', '.join(all_nan_cols)}")
                            
                            # 4. Check if there are any remaining NaNs
                            if X_subset.isna().any().any():
                                # Fill any remaining NaNs with 0
                                X_subset = X_subset.fillna(0)
                                st.warning("Some NaN values were filled with 0 as a fallback.")
                            
                            # Train a Random Forest to get feature importance
                            from sklearn.ensemble import RandomForestClassifier
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_subset, y_subset)
                            
                            # Get feature importance
                            importances = model.feature_importances_
                            feature_imp = pd.DataFrame({
                                'Feature': numeric_features,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False)
                            
                            # Display results
                            st.markdown("### Feature Importance Results")
                            st.dataframe(feature_imp)
                            
                            # Create a new figure with good sizing
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111)
                            
                            # Plot the bar chart
                            feature_imp.plot(kind='bar', x='Feature', y='Importance', ax=ax)
                            ax.set_title('Feature Importance for Predicting ' + target_column)
                            ax.set_ylabel('Importance Score')
                            ax.set_xlabel('')
                            
                            # Rotate x-axis labels for better readability
                            plt.xticks(rotation=45, ha='right')
                            
                            # Add more space at the bottom for labels
                            plt.subplots_adjust(bottom=0.25)
                            
                            # Ensure tight layout
                            plt.tight_layout()
                            
                            # Display the plot
                            st.pyplot(fig)
                            
                            # Clear the matplotlib figure
                            plt.close(fig)
                            
                            # Store the results for the report
                            if 'statistical_tests' not in st.session_state.report_data:
                                st.session_state.report_data['statistical_tests'] = {}
                            st.session_state.report_data['statistical_tests']['feature_importance'] = {
                                'type': 'feature_importance',
                                'results': feature_imp,
                                'description': f"Feature importance analysis for predicting {target_column}"
                            }
                    except Exception as e:
                        st.error(f"Error analyzing feature importance: {str(e)}")
                        st.code(traceback.format_exc())
            else:
                st.warning("No numeric features found. Feature importance analysis requires numeric features.")

if __name__ == "__main__":
    show_statistical_analysis()