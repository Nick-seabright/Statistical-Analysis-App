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
        
        # Explanation of threshold analysis
        with st.expander("What is threshold analysis?", expanded=False):
            st.markdown("""
            **Threshold analysis** helps you find the optimal value of a feature that best separates your target classes.
            
            For example, if your target is "Pass/Fail" and your feature is "Study Hours":
            - Does studying more than 3 hours result in a higher pass rate?
            - Is there a specific cutoff point where the pass rate significantly changes?
            
            This analysis helps identify these critical threshold values and quantifies their impact.
            """)
        
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
                    
                    # Display rates with clearer labels
                    if feature_result['is_time_variable']:
                        with col2:
                            st.metric(
                                label=f"Percentage of '{results['class_names'][1]}' when {feature_for_threshold} is below threshold",
                                value=f"{feature_result['optimal_above_rate']:.2%}"
                            )
                        with col3:
                            st.metric(
                                label=f"Percentage of '{results['class_names'][1]}' when {feature_for_threshold} is above threshold",
                                value=f"{feature_result['optimal_below_rate']:.2%}"
                            )
                    else:
                        with col2:
                            st.metric(
                                label=f"Percentage of '{results['class_names'][1]}' when {feature_for_threshold} is above threshold",
                                value=f"{feature_result['optimal_above_rate']:.2%}"
                            )
                        with col3:
                            st.metric(
                                label=f"Percentage of '{results['class_names'][1]}' when {feature_for_threshold} is below threshold",
                                value=f"{feature_result['optimal_below_rate']:.2%}"
                            )
                    
                    # Display the difference
                    st.metric(
                        label="Difference in Rates",
                        value=f"{feature_result['optimal_difference']:.2%}"
                    )
                    
                    # Display the single feature threshold plot if available
                    if 'single_feature_plot' in results:
                        st.pyplot(results['single_feature_plot'])
                    
                    # Add explanation about what these rates mean
                    st.markdown("### Understanding the Results")
                    st.markdown(f"""
                    **What do these rates mean?**
                    
                    These rates show the percentage of observations that belong to the '{results['class_names'][1]}' class 
                    when the feature value is above or below the threshold.
                    
                    For example, with an optimal threshold of **{feature_result['optimal_threshold_display']}**:
                    
                    - When {feature_for_threshold} is {"below" if feature_result['is_time_variable'] else "above"} the threshold, 
                      {feature_result['optimal_above_rate']:.1%} of observations are in the '{results['class_names'][1]}' class
                      
                    - When {feature_for_threshold} is {"above" if feature_result['is_time_variable'] else "below"} the threshold, 
                      {feature_result['optimal_below_rate']:.1%} of observations are in the '{results['class_names'][1]}' class
                    
                    The optimal threshold creates the largest difference ({feature_result['optimal_difference']:.1%}) between these rates,
                    making it the most informative cut-off point for decision making.
                    """)
                    
            except Exception as e:
                st.error(f"Error analyzing thresholds: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    with tab2:
        st.markdown("<div class='subheader'>Feature Combination Analysis</div>", unsafe_allow_html=True)
        
        # Explanation of feature combination analysis
        with st.expander("What is feature combination analysis?", expanded=False):
            st.markdown("""
            **Feature combination analysis** examines how pairs of features work together to separate target classes.
            
            This analysis divides your data into four quadrants based on whether each feature is above or below its median value.
            It then calculates the percentage of the target class in each quadrant, helping you identify powerful feature interactions.
            
            For example, you might discover that students who study more than 3 hours AND attend more than 80% of classes 
            have a 95% pass rate, while those below both thresholds have only a 30% pass rate.
            """)
        
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
                            
                            # Create a summary table
                            quadrant_results = results['feature_combination']['quadrant_results']
                            quadrant_df = pd.DataFrame({
                                'Quadrant': list(quadrant_results.keys()),
                                'Target Rate': [r['target_rate'] for r in quadrant_results.values()],
                                'Sample Size': [r['sample_size'] for r in quadrant_results.values()],
                                'Sample %': [f"{r['sample_pct']:.1%}" for r in quadrant_results.values()]
                            })
                            
                            # Format target rate as percentage
                            quadrant_df['Target Rate'] = quadrant_df['Target Rate'].apply(lambda x: f"{x:.1%}")
                            
                            st.dataframe(quadrant_df)
                            
                            # Get information about features
                            is_time1 = results['feature_combination']['is_time1']
                            is_time2 = results['feature_combination']['is_time2']
                            median1_display = results['feature_combination']['median1_display']
                            median2_display = results['feature_combination']['median2_display']
                            
                            # Display the scatterplot
                            if 'plot' in results['feature_combination']:
                                st.pyplot(results['feature_combination']['plot'])
                            
                            # Add explanation about quadrants
                            st.markdown("### Understanding the Quadrants")
                            st.markdown(f"""
                            The analysis divides data into four quadrants based on whether each feature is above or below the median:
                            
                            - **Q1** (top-right): {feature1} is {"below" if is_time1 else "above"} {median1_display} and {feature2} is {"below" if is_time2 else "above"} {median2_display}
                            - **Q2** (top-left): {feature1} is {"above" if is_time1 else "below"} {median1_display} and {feature2} is {"below" if is_time2 else "above"} {median2_display}
                            - **Q3** (bottom-left): {feature1} is {"above" if is_time1 else "below"} {median1_display} and {feature2} is {"above" if is_time2 else "below"} {median2_display}
                            - **Q4** (bottom-right): {feature1} is {"below" if is_time1 else "above"} {median1_display} and {feature2} is {"above" if is_time2 else "below"} {median2_display}
                            
                            The percentages in each quadrant show how often the target class '{results['class_names'][1]}' appears in that segment of your data.
                            
                            This helps you identify which combination of feature values is most strongly associated with your target.
                            """)
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
                        
                        # Convert target_1_rate to percentage format for display
                        display_results = results.copy()
                        display_results['target_1_rate'] = display_results['target_1_rate'].apply(lambda x: f"{x:.1%}")
                        display_results['percentage_of_total'] = display_results['percentage_of_total'].apply(lambda x: f"{x:.1%}")
                        display_results.columns = ['Count', 'Target Rate', 'Percentage of Total']
                        st.dataframe(display_results)
                        
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
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
                        
                        # Get target type for coloring
                        if target_type == 'categorical':
                            # For categorical targets, use discrete colors
                            unique_values = np.unique(data[target_column])
                            n_classes = len(unique_values)
                            
                            if n_classes <= 10:  # Only use discrete colors for a reasonable number of classes
                                # Create a discrete colormap
                                cmap = plt.cm.get_cmap('tab10', n_classes)
                                
                                # Create scatter plot with discrete colors
                                scatter = plt.scatter(
                                    x_data,
                                    y_data,
                                    c=pd.Categorical(data[target_column]).codes,
                                    cmap=cmap,
                                    alpha=0.6
                                )
                                
                                # Create a custom legend
                                from matplotlib.lines import Line2D
                                legend_elements = [
                                    Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), 
                                           markersize=10, label=str(val))
                                    for i, val in enumerate(unique_values)
                                ]
                                plt.legend(handles=legend_elements, title=target_column)
                            else:
                                # Too many categories, use a simpler approach
                                scatter = plt.scatter(
                                    x_data,
                                    y_data,
                                    c=pd.Categorical(data[target_column]).codes,
                                    cmap='viridis',
                                    alpha=0.6
                                )
                                plt.colorbar(scatter, label=target_column)
                        else:
                            # For numeric targets, use continuous colormap
                            scatter = plt.scatter(
                                x_data,
                                y_data,
                                c=data[target_column],
                                cmap='coolwarm',
                                alpha=0.6
                            )
                            plt.colorbar(scatter, label=target_column)
                        
                        # Add threshold lines
                        threshold1_plot = convert_time_to_minutes(threshold1) if is_time1 else threshold1
                        threshold2_plot = convert_time_to_minutes(threshold2) if is_time2 else threshold2
                        plt.axvline(x=threshold1_plot, color='r', linestyle='--', alpha=0.7)
                        plt.axhline(y=threshold2_plot, color='r', linestyle='--', alpha=0.7)
                        
                        # Get target classes from results
                        target_1_label = str(unique_values[1]) if target_type == 'categorical' and len(unique_values) > 1 else "Target"
                        
                        # Add quadrant labels with target rates
                        quadrant_coords = {
                            'Above Both': (
                                np.mean([threshold1_plot, np.max(x_data)]), 
                                np.mean([threshold2_plot, np.max(y_data)])
                            ),
                            'Above 1, Below 2': (
                                np.mean([threshold1_plot, np.max(x_data)]), 
                                np.mean([threshold2_plot, np.min(y_data)])
                            ),
                            'Below 1, Above 2': (
                                np.mean([threshold1_plot, np.min(x_data)]), 
                                np.mean([threshold2_plot, np.max(y_data)])
                            ),
                            'Below Both': (
                                np.mean([threshold1_plot, np.min(x_data)]), 
                                np.mean([threshold2_plot, np.min(y_data)])
                            )
                        }
                        
                        for quadrant, coords in quadrant_coords.items():
                            rate = results.loc[quadrant, 'target_1_rate']
                            count = results.loc[quadrant, 'count']
                            plt.text(
                                coords[0],
                                coords[1],
                                f"{quadrant}\n{target_1_label} Rate: {rate:.1%}\n(n={count})",
                                ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                            )
                        
                        # Add axis labels and title
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
                        
                        # Add explanation
                        st.markdown("### Understanding the Results")
                        st.markdown(f"""
                        This analysis divides your data into four segments based on your custom thresholds:
                        
                        - **Above Both**: {feature1} > {threshold1} and {feature2} > {threshold2}
                        - **Above 1, Below 2**: {feature1} > {threshold1} and {feature2} ≤ {threshold2}
                        - **Below 1, Above 2**: {feature1} ≤ {threshold1} and {feature2} > {threshold2}
                        - **Below Both**: {feature1} ≤ {threshold1} and {feature2} ≤ {threshold2}
                        
                        For time variables, "above" means slower times and "below" means faster times.
                        
                        The "Target Rate" column shows the percentage of observations in each segment that belong 
                        to the target class. This helps you understand which combination of feature values is most 
                        strongly associated with your target.
                        """)
                        
                except Exception as e:
                    st.error(f"Error analyzing custom thresholds: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    show_threshold_analysis()