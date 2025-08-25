# streamlit_app/pages/threshold_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the parent directory to path if running this file directly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from edu_analytics.threshold_analysis import analyze_decision_boundaries, analyze_custom_threshold_combination
from edu_analytics.time_analysis import convert_time_to_minutes, minutes_to_time_string

def show_threshold_analysis():
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    
    # Check if data is processed
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("Please process your data first.")
        return
    
    st.markdown("<div class='subheader'>Threshold Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Analyze how different feature thresholds affect the target variable.</div>", unsafe_allow_html=True)
    
    # Get data from session state
    data = st.session_state.data
    target_column = st.session_state.processed_data['target_column']
    selected_features = st.session_state.processed_data['selected_features']
    target_type = st.session_state.target_type
    
    # This analysis works best with:
    # 1. A binary categorical target OR
    # 2. A numeric target that we can binarize around its median
    
    # For features, we need numeric features to find thresholds
    numeric_features = [col for col in selected_features 
                       if data[col].dtype in ['int64', 'float64'] 
                       or st.session_state.data_types.get(col) in ['integer', 'float', 'numeric', 'time']]
    
    # Check if we have the right data types
    if not numeric_features:
        st.warning("Threshold analysis requires numeric features. No numeric features found in your dataset.")
        return
    
    if target_type not in ['categorical', 'numeric', 'time']:
        st.warning(f"Threshold analysis is not applicable for target type '{target_type}'.")
        return
    
    # Create tabs for different analysis approaches
    tab1, tab2 = st.tabs(["Single Feature Thresholds", "Feature Combinations"])
    
    with tab1:
        st.markdown("<div class='subheader'>Single Feature Threshold Analysis</div>", unsafe_allow_html=True)
        
        # Select feature for analysis
        feature_for_threshold = st.selectbox(
            "Select feature for threshold analysis",
            numeric_features
        )
        
        # Run analysis button
        if st.button("Analyze Thresholds", key="run_threshold"):
            try:
                with st.spinner("Analyzing thresholds..."):
                    # Use the analyze_decision_boundaries function but limit to one feature
                    results = analyze_decision_boundaries(
                        df=data,
                        target_column=target_column,
                        feature_columns=[feature_for_threshold]
                    )
                    
                    # Store the analysis in the report data
                    if 'threshold_analysis' not in st.session_state.report_data:
                        st.session_state.report_data['threshold_analysis'] = {}
                    
                    st.session_state.report_data['threshold_analysis'][feature_for_threshold] = {
                        'type': 'single_feature',
                        'feature': feature_for_threshold,
                        'target': target_column,
                        'results': results,
                        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Show results summary after analysis
                    feature_result = results['features'][feature_for_threshold]
                    
                    st.success(f"Threshold analysis completed for {feature_for_threshold}")
                    
                    # Display optimal threshold information
                    st.markdown("### Optimal Threshold")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Optimal Threshold",
                            value=feature_result['optimal_threshold_display']
                        )
                    
                    if feature_result['is_time_variable']:
                        # For time variables, "below threshold" means faster times which is typically better
                        with col2:
                            st.metric(
                                label="Rate Below Threshold",
                                value=f"{feature_result['optimal_above_rate']:.2%}"
                            )
                        
                        with col3:
                            st.metric(
                                label="Rate Above Threshold",
                                value=f"{feature_result['optimal_below_rate']:.2%}"
                            )
                    else:
                        # For regular numeric features, "above threshold" is typically better
                        with col2:
                            st.metric(
                                label="Rate Above Threshold",
                                value=f"{feature_result['optimal_above_rate']:.2%}"
                            )
                        
                        with col3:
                            st.metric(
                                label="Rate Below Threshold",
                                value=f"{feature_result['optimal_below_rate']:.2%}"
                            )
                    
                    # Display the difference
                    st.metric(
                        label="Difference in Rates",
                        value=f"{feature_result['optimal_difference']:.2%}"
                    )
                    
            except Exception as e:
                st.error(f"Error analyzing thresholds: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    with tab2:
        st.markdown("<div class='subheader'>Feature Combination Analysis</div>", unsafe_allow_html=True)
        
        # Select two features for analysis
        if len(numeric_features) < 2:
            st.warning("You need at least two numeric features for combination analysis.")
        else:
            feature1 = st.selectbox(
                "Select first feature",
                numeric_features,
                key="combo_feature1"
            )
            
            feature2 = st.selectbox(
                "Select second feature",
                [f for f in numeric_features if f != feature1],
                key="combo_feature2"
            )
            
            # Run analysis button
            if st.button("Analyze Feature Combination", key="run_combo"):
                try:
                    with st.spinner("Analyzing feature combination..."):
                        # Use the analyze_decision_boundaries function with two features
                        results = analyze_decision_boundaries(
                            df=data,
                            target_column=target_column,
                            feature_columns=[feature1, feature2]
                        )
                        
                        # Store the analysis in the report data
                        if 'threshold_analysis' not in st.session_state.report_data:
                            st.session_state.report_data['threshold_analysis'] = {}
                        
                        combo_key = f"{feature1}_X_{feature2}"
                        st.session_state.report_data['threshold_analysis'][combo_key] = {
                            'type': 'feature_combination',
                            'features': [feature1, feature2],
                            'target': target_column,
                            'results': results,
                            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.success(f"Combination analysis completed for {feature1} and {feature2}")
                        
                        # Display feature combination results if available
                        if 'feature_combination' in results:
                            # Display quadrant analysis summary
                            st.markdown("### Quadrant Analysis")
                            
                            quadrant_results = results['feature_combination']['quadrant_results']
                            
                            # Create a summary table
                            quadrant_df = pd.DataFrame({
                                'Quadrant': list(quadrant_results.keys()),
                                'Target Rate': [r['target_rate'] for r in quadrant_results.values()],
                                'Sample Size': [r['sample_size'] for r in quadrant_results.values()],
                                'Sample %': [f"{r['sample_pct']:.1%}" for r in quadrant_results.values()]
                            })
                            
                            st.dataframe(quadrant_df)
                            
                            # Create a custom visualization to summarize the quadrants
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Plot quadrants as a heatmap
                            quadrant_positions = {
                                'Q1': (0, 1),
                                'Q2': (1, 1),
                                'Q3': (1, 0),
                                'Q4': (0, 0)
                            }
                            
                            # Create a 2x2 matrix for the heatmap
                            heatmap_data = np.zeros((2, 2))
                            for quadrant, pos in quadrant_positions.items():
                                heatmap_data[pos[1], pos[0]] = quadrant_results[quadrant]['target_rate']
                            
                            # Plot heatmap
                            sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='YlGnBu', ax=ax)
                            
                            # Set labels
                            is_time1 = results['feature_combination']['is_time1']
                            is_time2 = results['feature_combination']['is_time2']
                            
                            # Customize labels based on whether the features are time variables
                            ax.set_xlabel(f"{feature1} {'(Lower is better)' if is_time1 else '(Higher is better)'}")
                            ax.set_ylabel(f"{feature2} {'(Lower is better)' if is_time2 else '(Higher is better)'}")
                            
                            # Set x and y tick labels
                            ax.set_xticks([0.5, 1.5])
                            ax.set_yticks([0.5, 1.5])
                            
                            if is_time1:
                                ax.set_xticklabels([f"< {results['feature_combination']['median1_display']}", 
                                                   f"≥ {results['feature_combination']['median1_display']}"])
                            else:
                                ax.set_xticklabels([f"≤ {results['feature_combination']['median1_display']}", 
                                                   f"> {results['feature_combination']['median1_display']}"])
                            
                            if is_time2:
                                ax.set_yticklabels([f"≥ {results['feature_combination']['median2_display']}", 
                                                   f"< {results['feature_combination']['median2_display']}"])
                            else:
                                ax.set_yticklabels([f"≤ {results['feature_combination']['median2_display']}", 
                                                   f"> {results['feature_combination']['median2_display']}"])
                            
                            plt.title(f"Target Rate by {feature1} and {feature2} Quadrants")
                            st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error analyzing feature combination: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Custom threshold analysis
            st.markdown("<div class='subheader'>Custom Threshold Analysis</div>", unsafe_allow_html=True)
            st.markdown("<div class='info-text'>Test specific threshold values for your features.</div>", unsafe_allow_html=True)
            
            # Get feature statistics for help with threshold selection
            feature1_min = data[feature1].min()
            feature1_max = data[feature1].max()
            feature1_median = data[feature1].median()
            
            feature2_min = data[feature2].min()
            feature2_max = data[feature2].max()
            feature2_median = data[feature2].median()
            
            # Display feature statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{feature1}** statistics:")
                st.write(f"Min: {feature1_min}")
                st.write(f"Median: {feature1_median}")
                st.write(f"Max: {feature1_max}")
                
                threshold1 = st.number_input(
                    f"Threshold for {feature1}",
                    min_value=float(feature1_min),
                    max_value=float(feature1_max),
                    value=float(feature1_median)
                )
            
            with col2:
                st.write(f"**{feature2}** statistics:")
                st.write(f"Min: {feature2_min}")
                st.write(f"Median: {feature2_median}")
                st.write(f"Max: {feature2_max}")
                
                threshold2 = st.number_input(
                    f"Threshold for {feature2}",
                    min_value=float(feature2_min),
                    max_value=float(feature2_max),
                    value=float(feature2_median)
                )
            
            # Analyze custom thresholds
            if st.button("Analyze Custom Thresholds", key="run_custom"):
                try:
                    with st.spinner("Analyzing custom thresholds..."):
                        # Use the analyze_custom_threshold_combination function
                        results = analyze_custom_threshold_combination(
                            df=data,
                            feature1=feature1,
                            threshold1=threshold1,
                            feature2=feature2,
                            threshold2=threshold2,
                            target_column=target_column
                        )
                        
                        # Display results
                        st.markdown("### Custom Threshold Analysis Results")
                        st.dataframe(results)
                        
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Get target for coloring
                        if target_type == 'categorical':
                            # Convert categorical target to numeric for coloring
                            if data[target_column].dtype == 'object' or data[target_column].dtype.name == 'category':
                                # Map categorical values to numbers
                                unique_values = data[target_column].unique()
                                target_mapping = {val: i for i, val in enumerate(unique_values)}
                                target_for_plot = data[target_column].map(target_mapping)
                                # Create a colormap with discrete colors for categories
                                cmap = plt.cm.get_cmap('coolwarm', len(unique_values))
                            else:
                                # If it's already numeric, use as is
                                target_for_plot = data[target_column]
                                unique_values = np.unique(target_for_plot)
                                cmap = 'coolwarm'
                        else:
                            # For numeric or time target, binarize around the median
                            target_median = data[target_column].median()
                            target_for_plot = (data[target_column] > target_median).astype(int)
                            unique_values = [0, 1]
                            cmap = 'coolwarm'
                        
                        # Check if features are time variables
                        is_time1 = st.session_state.data_types.get(feature1) == 'time'
                        is_time2 = st.session_state.data_types.get(feature2) == 'time'
                        
                        # Get data for plotting
                        x_data = data[feature1]
                        y_data = data[feature2]
                        
                        # For time variables, convert to minutes for plotting
                        if is_time1:
                            x_data = x_data.apply(convert_time_to_minutes)
                        if is_time2:
                            y_data = y_data.apply(convert_time_to_minutes)
                        
                        # Create scatter plot
                        scatter = plt.scatter(
                            x_data,
                            y_data,
                            c=target_for_plot,
                            cmap=cmap,
                            alpha=0.6
                        )
                        
                        # Add threshold lines
                        threshold1_plot = convert_time_to_minutes(threshold1) if is_time1 else threshold1
                        threshold2_plot = convert_time_to_minutes(threshold2) if is_time2 else threshold2
                        plt.axvline(x=threshold1_plot, color='r', linestyle='--', alpha=0.7)
                        plt.axhline(y=threshold2_plot, color='r', linestyle='--', alpha=0.7)
                        
                        # Add quadrant labels
                        plt.text(
                            x_data.min() + (threshold1_plot - x_data.min()) * 0.5,
                            threshold2_plot + (y_data.max() - threshold2_plot) * 0.5,
                            f"Above 1, Below 2\n{results.loc['Above 1, Below 2', 'target_1_rate']:.1%}",
                            ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                        )
                        plt.text(
                            threshold1_plot + (x_data.max() - threshold1_plot) * 0.5,
                            threshold2_plot + (y_data.max() - threshold2_plot) * 0.5,
                            f"Above Both\n{results.loc['Above Both', 'target_1_rate']:.1%}",
                            ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                        )
                        plt.text(
                            threshold1_plot + (x_data.max() - threshold1_plot) * 0.5,
                            y_data.min() + (threshold2_plot - y_data.min()) * 0.5,
                            f"Below 1, Above 2\n{results.loc['Below 1, Above 2', 'target_1_rate']:.1%}",
                            ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                        )
                        plt.text(
                            x_data.min() + (threshold1_plot - x_data.min()) * 0.5,
                            y_data.min() + (threshold2_plot - y_data.min()) * 0.5,
                            f"Below Both\n{results.loc['Below Both', 'target_1_rate']:.1%}",
                            ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                        )
                        
                        # Create a legend for categorical targets
                        if target_type == 'categorical' and len(unique_values) <= 10:  # Only for a reasonable number of categories
                            # Create legend handles
                            from matplotlib.lines import Line2D
                            
                            # If we have a mapping, use it, otherwise use values directly
                            if 'target_mapping' in locals():
                                legend_elements = [
                                    Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=plt.cm.get_cmap(cmap)(target_mapping[val]/len(unique_values)) 
                                              if hasattr(cmap, '__call__') else plt.cm.get_cmap(cmap)(i/len(unique_values)), 
                                           markersize=10, label=str(val))
                                    for i, val in enumerate(unique_values)
                                ]
                            else:
                                legend_elements = [
                                    Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=plt.cm.get_cmap(cmap)(i/len(unique_values)), 
                                           markersize=10, label=str(val))
                                    for i, val in enumerate(unique_values)
                                ]
                            
                            # Add the legend to the plot - positioned in the upper right corner
                            plt.legend(handles=legend_elements, title=target_column, loc='upper right')
                        
                        # Add labels and title
                        plt.colorbar(scatter, label='Target')
                        plt.xlabel(feature1)
                        plt.ylabel(feature2)
                        plt.title(f'Custom Threshold Analysis: {feature1} vs {feature2}')
                        plt.tight_layout()
                        
                        # Display the plot
                        st.pyplot(fig)
                        plt.close(fig)  # Close the figure to prevent interference
                        
                        # Store the analysis in the report data
                        if 'threshold_analysis' not in st.session_state.report_data:
                            st.session_state.report_data['threshold_analysis'] = {}
                        custom_key = f"custom_{feature1}_{threshold1}_X_{feature2}_{threshold2}"
                        st.session_state.report_data['threshold_analysis'][custom_key] = {
                            'type': 'custom_threshold',
                            'feature1': feature1,
                            'threshold1': threshold1,
                            'feature2': feature2,
                            'threshold2': threshold2,
                            'results': results,
                            'target': target_column,
                            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                except Exception as e:
                    st.error(f"Error analyzing custom thresholds: {str(e)}")
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    show_threshold_analysis()