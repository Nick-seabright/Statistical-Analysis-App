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
        st.markdown("""
        This analysis finds the optimal threshold for a feature that best separates your target classes. 
        It works by testing different threshold values and measuring how the target distribution differs 
        on either side of the threshold.
        """)
        
        # Configuration options
        with st.expander("Analysis Settings", expanded=False):
            min_group_size = st.slider(
                "Minimum group size (% of total data)", 
                min_value=0.05, 
                max_value=0.3, 
                value=0.1, 
                step=0.05,
                help="Each group must contain at least this percentage of the total data. Higher values ensure more balanced splits."
            )
            
            quality_explanation = st.checkbox("Show quality metrics explanation", value=False)
            if quality_explanation:
                st.info("""
                **Split Quality Metrics:**
                
                - **Good**: Strong separation between groups with well-balanced group sizes
                - **Fair**: Moderate separation or somewhat imbalanced groups
                - **Poor**: Weak separation or highly imbalanced groups
                - **None**: No valid threshold found that meets minimum group size requirements
                
                The quality score considers both the difference in target rates and how balanced the groups are.
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
                        feature_columns=[feature_for_threshold],
                        min_group_size=min_group_size
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
                    
                    # Check if a valid threshold was found
                    if feature_result.get('threshold_quality') == "none" or feature_result.get('optimal_threshold') is None:
                        st.warning(f"No good threshold found for {feature_for_threshold}. The feature may not have a clear split point that meets the minimum group size requirement of {min_group_size:.0%}.")
                    else:
                        quality_label = {
                            "good": "💚 Good",
                            "fair": "🟡 Fair",
                            "poor": "🔴 Poor"
                        }.get(feature_result.get('threshold_quality', "poor"), "⚠️ Unknown")
                        
                        st.success(f"Threshold analysis completed for {feature_for_threshold}")
                        
                        col0, col1, col2 = st.columns([1, 1, 1])
                        # Display threshold quality
                        with col0:
                            st.metric(
                                label="Split Quality",
                                value=quality_label
                            )
                        
                        # Display optimal threshold
                        with col1:
                            st.metric(
                                label="Optimal Threshold",
                                value=feature_result['optimal_threshold_display']
                            )
                        
                        # Display the difference
                        with col2:
                            st.metric(
                                label="Difference in Target Rates",
                                value=f"{feature_result['optimal_difference']:.2%}"
                            )
                        
                        # Show group sizes
                        above_count = feature_result.get('optimal_above_count', 0)
                        below_count = feature_result.get('optimal_below_count', 0)
                        total_count = above_count + below_count
                        
                        st.markdown("### Group Sizes")
                        col4, col5 = st.columns(2)
                        with col4:
                            if feature_result['is_time_variable']:
                                st.metric(
                                    label=f"Group below threshold (faster times)",
                                    value=f"{above_count} samples",
                                    delta=f"{above_count/total_count:.1%} of data"
                                )
                            else:
                                st.metric(
                                    label=f"Group above threshold",
                                    value=f"{above_count} samples",
                                    delta=f"{above_count/total_count:.1%} of data"
                                )
                        
                        with col5:
                            if feature_result['is_time_variable']:
                                st.metric(
                                    label=f"Group above threshold (slower times)",
                                    value=f"{below_count} samples",
                                    delta=f"{below_count/total_count:.1%} of data"
                                )
                            else:
                                st.metric(
                                    label=f"Group below threshold",
                                    value=f"{below_count} samples",
                                    delta=f"{below_count/total_count:.1%} of data"
                                )
                        
                        # Get class names
                        class_names = results.get('class_names', ['Class 0', 'Class 1'])
                        
                        st.markdown("### Target Rates by Group")
                        # Display rate metrics with more context
                        col6, col7 = st.columns(2)
                        if feature_result['is_time_variable']:
                            with col6:
                                st.metric(
                                    label=f"Percentage of '{class_names[1]}' in faster group",
                                    value=f"{feature_result['optimal_above_rate']:.2%}"
                                )
                            with col7:
                                st.metric(
                                    label=f"Percentage of '{class_names[1]}' in slower group",
                                    value=f"{feature_result['optimal_below_rate']:.2%}"
                                )
                        else:
                            with col6:
                                st.metric(
                                    label=f"Percentage of '{class_names[1]}' in above-threshold group",
                                    value=f"{feature_result['optimal_above_rate']:.2%}"
                                )
                            with col7:
                                st.metric(
                                    label=f"Percentage of '{class_names[1]}' in below-threshold group",
                                    value=f"{feature_result['optimal_below_rate']:.2%}"
                                )
                        
                        # Display the analysis plots if they're returned by the function
                        if 'plots' in results and feature_for_threshold in results['plots']:
                            st.markdown("### Threshold Analysis Visualization")
                            st.pyplot(results['plots'][feature_for_threshold])
                            
                            st.markdown("### Interpreting the Threshold Analysis")
                            st.markdown(f"""
                            The graph above shows:
                            
                            1. **Left**: Distribution of {feature_for_threshold} for each target class
                            2. **Right**: How the target rate changes at different threshold values
                            
                            The red vertical line indicates the optimal threshold of **{feature_result['optimal_threshold_display']}**, 
                            which maximizes the difference in target rates while maintaining balanced group sizes.
                            
                            This analysis helps you understand how {feature_for_threshold} relates to the target variable and 
                            where the most significant change in the relationship occurs.
                            """)
                        
                        # Add explanation about what these rates mean
                        with st.expander("What do these rates mean?", expanded=False):
                            st.markdown(f"""
                            **Target Rate** represents the percentage of observations that belong to the positive class '{class_names[1]}'.
                            
                            For example, if your target is "{class_names[0]}/{class_names[1]}" and the positive class is "{class_names[1]}":
                            
                            - **Rate in Group 1**: The percentage of observations classified as "{class_names[1]}" when the feature is {"below" if feature_result['is_time_variable'] else "above"} the threshold
                            - **Rate in Group 2**: The percentage of observations classified as "{class_names[1]}" when the feature is {"above" if feature_result['is_time_variable'] else "below"} the threshold
                            
                            The optimal threshold is the value that creates the largest difference between these rates while maintaining adequately sized groups,
                            helping you identify the most informative cut-off point for decision making.
                            
                            **Quality Score** considers both:
                            1. How different the target rates are between groups
                            2. How balanced the group sizes are (avoiding tiny groups)
                            """)
            except Exception as e:
                st.error(f"Error analyzing thresholds: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    with tab2:
        st.markdown("<div class='subheader'>Feature Combination Analysis</div>", unsafe_allow_html=True)
        st.markdown("""
        This analysis examines how two features work together to separate your target classes. 
        It identifies optimal thresholds for each feature and analyzes how the target variable 
        is distributed across the resulting quadrants.
        """)
        
        # Select two features for analysis
        if len(numeric_features) < 2:
            st.warning("You need at least two numeric features for combination analysis.")
        else:
            # Configuration options
            with st.expander("Analysis Settings", expanded=False):
                min_group_size = st.slider(
                    "Minimum group size (% of total data)", 
                    min_value=0.05, 
                    max_value=0.3, 
                    value=0.1, 
                    step=0.05,
                    help="Each quadrant must contain at least this percentage of the total data.",
                    key="combo_min_group"
                )
            
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
                            feature_columns=[feature1, feature2],
                            min_group_size=min_group_size
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
                            # First check overall quality
                            feature1_quality = results['features'][feature1].get('threshold_quality', 'none')
                            feature2_quality = results['features'][feature2].get('threshold_quality', 'none')
                            
                            if feature1_quality == "none" or feature2_quality == "none":
                                st.warning(f"One or both features do not have a good threshold that meets the minimum group size requirement of {min_group_size:.0%}.")
                            
                            # Display quadrant analysis summary
                            st.markdown("### Quadrant Analysis")
                            
                            # Display optimal thresholds
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    label=f"Optimal Threshold for {feature1}",
                                    value=results['features'][feature1]['optimal_threshold_display']
                                )
                            with col2:
                                st.metric(
                                    label=f"Optimal Threshold for {feature2}",
                                    value=results['features'][feature2]['optimal_threshold_display']
                                )
                            
                            # Display quadrant analysis summary
                            quadrant_results = results['feature_combination']['quadrant_results']
                            
                            # Create a summary table
                            quadrant_df = pd.DataFrame({
                                'Quadrant': list(quadrant_results.keys()),
                                'Target Rate': [r['target_rate'] for r in quadrant_results.values()],
                                'Sample Size': [r['sample_size'] for r in quadrant_results.values()],
                                'Sample %': [f"{r['sample_pct']:.1%}" for r in quadrant_results.values()]
                            })
                            
                            # Sort by target rate for better understanding
                            quadrant_df = quadrant_df.sort_values('Target Rate', ascending=False)
                            
                            st.dataframe(quadrant_df)
                            
                            # Get class names
                            class_names = results.get('class_names', ['Class 0', 'Class 1'])
                            
                            # Create a more informative interpretation
                            st.markdown("### Interpretation")
                            
                            # Find the best and worst quadrants
                            best_quadrant = quadrant_df.iloc[0]['Quadrant']
                            best_rate = quadrant_df.iloc[0]['Target Rate']
                            worst_quadrant = quadrant_df.iloc[-1]['Quadrant']
                            worst_rate = quadrant_df.iloc[-1]['Target Rate']
                            
                            # Get information about the features
                            is_time1 = results['feature_combination']['is_time1']
                            is_time2 = results['feature_combination']['is_time2']
                            
                            # Format description of the best quadrant
                            if best_quadrant == 'Q1':
                                best_quadrant_desc = f"when {feature1} is {'below' if is_time1 else 'above'} threshold AND {feature2} is {'below' if is_time2 else 'above'} threshold"
                            elif best_quadrant == 'Q2':
                                best_quadrant_desc = f"when {feature1} is {'above' if is_time1 else 'below'} threshold AND {feature2} is {'below' if is_time2 else 'above'} threshold"
                            elif best_quadrant == 'Q3':
                                best_quadrant_desc = f"when {feature1} is {'above' if is_time1 else 'below'} threshold AND {feature2} is {'above' if is_time2 else 'below'} threshold"
                            else:  # Q4
                                best_quadrant_desc = f"when {feature1} is {'below' if is_time1 else 'above'} threshold AND {feature2} is {'above' if is_time2 else 'below'} threshold"
                            
                            st.markdown(f"""
                            The highest rate of **{class_names[1]}** ({best_rate:.1%}) occurs {best_quadrant_desc}.
                            
                            The lowest rate of **{class_names[1]}** ({worst_rate:.1%}) occurs in the opposite quadrant.
                            
                            This suggests that the combination of these two features creates more separation than either feature alone.
                            """)
                            
                            # Display the visualization
                            if 'plots' in results and 'feature_combination' in results['plots']:
                                st.markdown("### Visualization")
                                st.pyplot(results['plots']['feature_combination'])
                                
                                st.markdown("""
                                The plot above shows how the target variable is distributed across different 
                                combinations of the two features. Each point represents an observation, colored 
                                by its target value. The red lines indicate the optimal thresholds for each feature.
                                
                                The percentage shown in each quadrant is the rate of the positive class in that region.
                                """)
                            
                            # Add explanation about quadrants
                            with st.expander("What do the quadrants mean?", expanded=False):
                                # Create a visual representation of the quadrants
                                st.markdown(f"""
                                The four quadrants represent different combinations of feature values:
                                
                                - **Q1**: {feature1} is {'below' if is_time1 else 'above'} threshold AND {feature2} is {'below' if is_time2 else 'above'} threshold
                                - **Q2**: {feature1} is {'above' if is_time1 else 'below'} threshold AND {feature2} is {'below' if is_time2 else 'above'} threshold
                                - **Q3**: {feature1} is {'above' if is_time1 else 'below'} threshold AND {feature2} is {'above' if is_time2 else 'below'} threshold
                                - **Q4**: {feature1} is {'below' if is_time1 else 'above'} threshold AND {feature2} is {'above' if is_time2 else 'below'} threshold
                                
                                The "Target Rate" for each quadrant represents the percentage of observations in that quadrant 
                                that belong to the class "{class_names[1]}".
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
                        st.dataframe(results)
                        
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
                        
                        # Get target for coloring
                        if target_type == 'categorical':
                            # For categorical target, use a discrete colormap
                            unique_classes = np.unique(data[target_column])
                            n_classes = len(unique_classes)
                            
                            # Create a discrete colormap with distinct colors
                            cmap = plt.cm.get_cmap('tab10', n_classes)
                            
                            # Create scatter plot with discrete colors
                            scatter = plt.scatter(
                                x_data,
                                y_data,
                                c=data[target_column].astype('category').cat.codes,
                                cmap=cmap,
                                alpha=0.6
                            )
                            
                            # Create a custom legend with class names
                            from matplotlib.lines import Line2D
                            
                            # Get class names if available
                            if 'class_names' in results:
                                class_names = results['class_names']
                            else:
                                class_names = [str(val) for val in unique_classes]
                            
                            legend_elements = [
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), 
                                      markersize=10, label=class_names[i] if i < len(class_names) else f"Class {i}")
                                for i in range(n_classes)
                            ]
                            plt.legend(handles=legend_elements, title="Classes")
                        else:
                            # For numeric target, use continuous colormap
                            scatter = plt.scatter(
                                x_data,
                                y_data,
                                c=data[target_column],
                                cmap='coolwarm',
                                alpha=0.6
                            )
                            plt.colorbar(scatter, label='Target Value')
                        
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
                        
                        # Add labels and title
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
                        
                        # Add explanation of quadrants
                        with st.expander("Understanding the Quadrants", expanded=True):
                            st.markdown(f"""
                            ### What do the quadrants mean?
                            
                            The plot is divided into four quadrants based on your selected thresholds:
                            
                            - **Above Both**: {feature1} > {threshold1} AND {feature2} > {threshold2}
                            - **Above 1, Below 2**: {feature1} > {threshold1} AND {feature2} ≤ {threshold2}
                            - **Below 1, Above 2**: {feature1} ≤ {threshold1} AND {feature2} > {threshold2}
                            - **Below Both**: {feature1} ≤ {threshold1} AND {feature2} ≤ {threshold2}
                            
                            The percentage shown in each quadrant represents the rate of the target class in that region.
                            
                            ### Group Sizes
                            
                            - Above Both: {results.loc['Above Both', 'count']} samples ({results.loc['Above Both', 'percentage_of_total']:.1%} of data)
                            - Above 1, Below 2: {results.loc['Above 1, Below 2', 'count']} samples ({results.loc['Above 1, Below 2', 'percentage_of_total']:.1%} of data)
                            - Below 1, Above 2: {results.loc['Below 1, Above 2', 'count']} samples ({results.loc['Below 1, Above 2', 'percentage_of_total']:.1%} of data)
                            - Below Both: {results.loc['Below Both', 'count']} samples ({results.loc['Below Both', 'percentage_of_total']:.1%} of data)
                            
                            This analysis helps you understand how different combinations of these features relate to your target variable.
                            """)
                            
                            # Show recommendations if there's a clear best quadrant
                            rates = [results.loc[q, 'target_1_rate'] for q in results.index]
                            max_rate = max(rates)
                            min_rate = min(rates)
                            
                            if max_rate - min_rate > 0.2:  # If there's a substantial difference
                                best_quadrant = results.index[rates.index(max_rate)]
                                st.markdown("### Recommendation")
                                st.markdown(f"""
                                Based on this analysis, the "{best_quadrant}" quadrant shows the strongest relationship 
                                with the target, with a target rate of {max_rate:.1%}. 
                                
                                This suggests that this combination of feature values might be particularly important 
                                for predicting your target variable.
                                """)
                            
                except Exception as e:
                    st.error(f"Error analyzing custom thresholds: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    show_threshold_analysis()