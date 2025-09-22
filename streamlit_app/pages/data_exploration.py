# streamlit_app/pages/data_exploration.py
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

from edu_analytics.feature_engineering import analyze_correlations
from edu_analytics.utils import set_plotting_style

def show_data_exploration():
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    
    # Check if data is processed
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("Please process your data first.")
        return
    
    st.markdown("<div class='subheader'>Data Exploration</div>", unsafe_allow_html=True)
    
    # Get data from session state
    data = st.session_state.data
    X = st.session_state.processed_data['X']
    y = st.session_state.processed_data['y']
    target_column = st.session_state.processed_data['target_column']
    selected_features = st.session_state.processed_data['selected_features']
    target_type = st.session_state.target_type
    
    # Create tabs for different exploration views
    tab1, tab2, tab3, tab4 = st.tabs(["Summary Statistics", "Correlations", "Distributions", "Feature Analysis"])
    
    with tab1:
        st.markdown("<div class='subheader'>Summary Statistics</div>", unsafe_allow_html=True)
        
        # Display summary statistics for numeric columns
        numeric_cols = data[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            st.dataframe(data[numeric_cols].describe())
        else:
            st.info("No numeric features to display summary statistics.")
        
        # Display value counts for categorical columns (top 5 categories)
        categorical_cols = [col for col in selected_features if col not in numeric_cols]
        if categorical_cols:
            st.markdown("<div class='subheader'>Categorical Features</div>", unsafe_allow_html=True)
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                st.write(f"**{col}** - Top Categories:")
                st.dataframe(data[col].value_counts().head(10))
        
        # Missing values analysis
        st.markdown("<div class='subheader'>Missing Values</div>", unsafe_allow_html=True)
        missing_data = data[selected_features + [target_column]].isna().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        missing_data['Missing Percentage'] = (missing_data['Missing Values'] / len(data) * 100).round(2)
        missing_data = missing_data.sort_values('Missing Values', ascending=False)
        
        # Only show columns with missing values
        missing_data_filtered = missing_data[missing_data['Missing Values'] > 0]
        if not missing_data_filtered.empty:
            st.dataframe(missing_data_filtered)
            
            # Plot missing values
            if len(missing_data_filtered) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Missing Percentage', y='Column', data=missing_data_filtered, ax=ax)
                plt.title('Missing Values by Column')
                plt.xlabel('Missing Percentage (%)')
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No missing values found in the dataset.")
    
    with tab2:
        st.markdown("<div class='subheader'>Correlation Analysis</div>", unsafe_allow_html=True)
        # Show correlation matrix for numeric features
        numeric_data = data[selected_features + [target_column]].select_dtypes(include=['int64', 'float64'])
        if numeric_data.shape[1] > 1:
            # Check number of features for appropriate sizing
            num_features = numeric_data.shape[1]
            
            # Dynamic figure sizing based on number of features
            figsize = (max(10, num_features * 0.8), max(8, num_features * 0.7))
            
            # Create a more readable correlation heatmap
            fig, ax = plt.subplots(figsize=figsize)
            corr_matrix = numeric_data.corr()
            
            # Create the heatmap with adjusted font size and rotation
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt='.2f', 
                ax=ax,
                # Adjust font size based on number of features
                annot_kws={"size": max(7, 12 - 0.3 * num_features)},
                # Ensure the colorbar is properly sized
                cbar_kws={"shrink": min(1.0, 0.8 if num_features > 15 else 0.9)}
            )
            
            # Adjust tick labels
            plt.xticks(
                rotation=45 if num_features < 20 else 90, 
                ha='right' if num_features < 20 else 'center',
                fontsize=max(6, 10 - 0.2 * num_features)
            )
            plt.yticks(
                rotation=0,
                fontsize=max(6, 10 - 0.2 * num_features)
            )
            
            # Ensure layout is properly adjusted
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show top correlated features with target in a more readable format
            if target_column in corr_matrix.columns:
                st.markdown("<div class='subheader'>Top Correlations with Target</div>", unsafe_allow_html=True)
                
                # Get target correlations and sort
                target_corrs = corr_matrix[target_column].drop(target_column).sort_values(ascending=False)
                
                # Create a more readable bar chart
                fig, ax = plt.subplots(figsize=(10, max(6, len(target_corrs) * 0.4)))
                
                # Plot horizontal bars for better label readability
                target_corrs.plot(kind='barh', ax=ax)
                plt.title(f'Feature Correlations with {target_column}')
                plt.ylabel('')  # Remove y-label as it's redundant
                plt.xlabel('Correlation Coefficient')
                
                # Add value labels to the bars
                for i, v in enumerate(target_corrs):
                    ax.text(v + (0.01 if v >= 0 else -0.06), i, f"{v:.2f}", 
                            va='center', fontsize=max(8, 10 - 0.1 * len(target_corrs)))
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display top positive and negative correlations in a cleaner format
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top Positive Correlations**")
                    # Format the display of correlations
                    positive_corrs = target_corrs.head(min(5, len(target_corrs)))
                    if not positive_corrs.empty:
                        pos_df = pd.DataFrame({
                            'Feature': positive_corrs.index,
                            'Correlation': positive_corrs.values
                        })
                        pos_df['Correlation'] = pos_df['Correlation'].round(3)
                        st.dataframe(pos_df.set_index('Feature'), use_container_width=True)
                    else:
                        st.info("No positive correlations found.")
                        
                with col2:
                    st.markdown("**Top Negative Correlations**")
                    # Format the display of correlations
                    negative_corrs = target_corrs.tail(min(5, len(target_corrs)))
                    if not negative_corrs.empty:
                        neg_df = pd.DataFrame({
                            'Feature': negative_corrs.index,
                            'Correlation': negative_corrs.values
                        })
                        neg_df['Correlation'] = neg_df['Correlation'].round(3)
                        st.dataframe(neg_df.set_index('Feature'), use_container_width=True)
                    else:
                        st.info("No negative correlations found.")
        else:
            st.info("Not enough numeric features for correlation analysis.")
    
    with tab3:
        st.markdown("<div class='subheader'>Data Distributions</div>", unsafe_allow_html=True)
        # Feature selection for distribution plots
        dist_feature = st.selectbox("Select feature for distribution", selected_features)
        # Plot distribution based on feature type
        if data[dist_feature].dtype in ['int64', 'float64']:
            # Numeric feature - histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=data, x=dist_feature, kde=True, ax=ax)
            plt.title(f'Distribution of {dist_feature}')
            plt.tight_layout()  # Ensure layout is properly adjusted
            st.pyplot(fig)
            
            # Distribution by target if target is categorical and not too many categories
            if target_type == 'categorical' and data[target_column].nunique() <= 5:
                fig, ax = plt.subplots(figsize=(10, 6))
                # Get original category names if mapping exists
                if hasattr(st.session_state, 'target_mapping') and st.session_state.target_mapping:
                    target_mapping = st.session_state.target_mapping
                    reverse_mapping = {v: k for k, v in target_mapping.items()}
                    # Create a temporary column with original category names for plotting
                    data_temp = data.copy()
                    if target_column in data_temp.columns:
                        data_temp['target_display'] = data_temp[target_column].map(
                            lambda x: reverse_mapping.get(x, str(x))
                        )
                        hue_column = 'target_display'
                    else:
                        hue_column = target_column
                else:
                    hue_column = target_column
                    
                sns.histplot(data=data_temp if 'data_temp' in locals() else data, 
                            x=dist_feature, hue=hue_column, kde=True, ax=ax)
                plt.title(f'Distribution of {dist_feature} by {target_column}')
                plt.legend(title=target_column)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            # Categorical feature - bar chart with improved readability
            # Count the number of categories
            category_counts = data[dist_feature].value_counts().sort_values(ascending=False)
            num_categories = len(category_counts)
            
            # Adjust figure size based on number of categories
            fig_height = min(10, max(6, num_categories * 0.4))
            fig, ax = plt.subplots(figsize=(10, fig_height))
            
            # Limit to top 20 categories if there are too many
            if num_categories > 20:
                st.info(f"Displaying top 20 out of {num_categories} categories for readability.")
                category_counts = category_counts.head(20)
            
            # Plot horizontal bar chart for better label visibility
            category_counts.plot(kind='barh', ax=ax)
            plt.title(f'Distribution of {dist_feature}')
            plt.xlabel('Count')
            plt.ylabel('')  # Feature name is in the title
            
            # Adjust font size based on number of categories
            plt.yticks(fontsize=max(8, 10 - 0.2 * len(category_counts)))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Distribution by target if target is categorical
            if target_type == 'categorical' and data[target_column].nunique() <= 5:
                # Create cross-tabulation
                cross_tab = pd.crosstab(
                    data[dist_feature],
                    data[target_column],
                    normalize='index'
                )
                
                # Limit to top categories for readability
                if cross_tab.shape[0] > 15:
                    st.info(f"Displaying top 15 out of {cross_tab.shape[0]} categories for readability.")
                    cross_tab = cross_tab.head(15)
                
                # Adjust figure size based on number of categories
                fig_height = min(10, max(6, cross_tab.shape[0] * 0.4))
                fig, ax = plt.subplots(figsize=(10, fig_height))
                
                # Use original target names if mapping exists
                if hasattr(st.session_state, 'target_mapping') and st.session_state.target_mapping:
                    target_mapping = st.session_state.target_mapping
                    reverse_mapping = {v: k for k, v in target_mapping.items()}
                    # Rename columns to original category names
                    cross_tab.columns = [reverse_mapping.get(col, str(col)) for col in cross_tab.columns]
                
                cross_tab.plot(kind='barh', stacked=True, ax=ax)
                plt.title(f'{target_column} Distribution by {dist_feature}')
                plt.xlabel('Proportion')
                plt.ylabel('')  # Feature name is in the title
                
                # Adjust font size based on number of categories
                plt.yticks(fontsize=max(8, 10 - 0.2 * cross_tab.shape[0]))
                
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab4:
        st.markdown("<div class='subheader'>Feature Analysis</div>", unsafe_allow_html=True)
        
        # Select feature for detailed analysis
        feature_for_analysis = st.selectbox("Select feature for detailed analysis", selected_features, key="feature_analysis")
        
        # Display basic stats
        st.write(f"**Feature:** {feature_for_analysis}")
        st.write(f"**Type:** {st.session_state.data_types.get(feature_for_analysis, 'Unknown')}")
        
        # Feature specific analysis based on type
        feature_type = st.session_state.data_types.get(feature_for_analysis)
        
        if feature_type in ['integer', 'float', 'numeric']:
            # Numeric feature analysis
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Statistics:**")
                st.write(f"Mean: {data[feature_for_analysis].mean():.2f}")
                st.write(f"Median: {data[feature_for_analysis].median():.2f}")
                st.write(f"Std Dev: {data[feature_for_analysis].std():.2f}")
                st.write(f"Min: {data[feature_for_analysis].min():.2f}")
                st.write(f"Max: {data[feature_for_analysis].max():.2f}")
            
            with col2:
                # Relationship with target
                if target_type == 'categorical':
                    # Group by target and show means
                    target_means = data.groupby(target_column)[feature_for_analysis].mean().sort_values()
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    target_means.plot(kind='bar', ax=ax)
                    plt.title(f'Mean {feature_for_analysis} by {target_column}')
                    plt.ylabel(feature_for_analysis)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    # Scatter plot with target
                    fig, ax = plt.subplots(figsize=(8, 4))
                    plt.scatter(data[feature_for_analysis], data[target_column], alpha=0.5)
                    plt.title(f'{feature_for_analysis} vs {target_column}')
                    plt.xlabel(feature_for_analysis)
                    plt.ylabel(target_column)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # More visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Box plot
            sns.boxplot(y=data[feature_for_analysis], ax=ax1)
            ax1.set_title(f'Box Plot of {feature_for_analysis}')
            
            # Box plot by target if categorical with not too many categories
            if target_type == 'categorical' and data[target_column].nunique() <= 5:
                sns.boxplot(x=data[target_column], y=data[feature_for_analysis], ax=ax2)
                ax2.set_title(f'Box Plot of {feature_for_analysis} by {target_column}')
            else:
                # QQ plot for normality check
                from scipy import stats
                stats.probplot(data[feature_for_analysis].dropna(), plot=ax2)
                ax2.set_title('Q-Q Plot (Normality Check)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        elif feature_type in ['categorical', 'boolean']:
            # Categorical feature analysis
            value_counts = data[feature_for_analysis].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Statistics:**")
                st.write(f"Unique values: {data[feature_for_analysis].nunique()}")
                st.write(f"Most common: {data[feature_for_analysis].mode()[0]} ({value_counts.iloc[0]} occurrences)")
                st.write(f"Missing values: {data[feature_for_analysis].isna().sum()}")
                
            with col2:
                # Pie chart for category distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                value_counts.head(6).plot(kind='pie', autopct='%1.1f%%', ax=ax)
                plt.title(f'Distribution of {feature_for_analysis}')
                plt.ylabel('')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Stacked bar chart by target
            if target_type == 'categorical' and data[target_column].nunique() <= 5:
                cross_tab = pd.crosstab(data[feature_for_analysis], data[target_column])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                cross_tab.plot(kind='bar', stacked=True, ax=ax)
                plt.title(f'{target_column} Distribution by {feature_for_analysis}')
                plt.xlabel(feature_for_analysis)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Also add percentage view
                cross_tab_pct = pd.crosstab(
                    data[feature_for_analysis], 
                    data[target_column], 
                    normalize='index'
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                cross_tab_pct.plot(kind='bar', stacked=True, ax=ax)
                plt.title(f'{target_column} Distribution (%) by {feature_for_analysis}')
                plt.xlabel(feature_for_analysis)
                plt.ylabel('Percentage')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        
        elif feature_type == 'time':
            # Time feature analysis
            # Import time conversion functions
            from edu_analytics.time_analysis import convert_time_to_minutes, minutes_to_time_string
            
            # Convert time strings to minutes for analysis
            time_in_minutes = data[feature_for_analysis].apply(convert_time_to_minutes)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Statistics:**")
                # Calculate statistics on the numeric representation
                mean_minutes = time_in_minutes.mean()
                median_minutes = time_in_minutes.median()
                min_minutes = time_in_minutes.min()
                max_minutes = time_in_minutes.max()
                
                # Convert back to time format for display
                mean_time = minutes_to_time_string(mean_minutes)
                median_time = minutes_to_time_string(median_minutes)
                min_time = minutes_to_time_string(min_minutes)
                max_time = minutes_to_time_string(max_minutes)
                
                st.write(f"Mean: {mean_time}")
                st.write(f"Median: {median_time}")
                st.write(f"Min: {min_time}")
                st.write(f"Max: {max_time}")
                st.write(f"Missing values: {data[feature_for_analysis].isna().sum()}")
            
            with col2:
                # Histogram of time distribution
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(time_in_minutes.dropna(), kde=True, ax=ax)
                plt.title(f'Distribution of {feature_for_analysis} (in minutes)')
                plt.xlabel('Time (minutes)')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Box plot by target if categorical
            if target_type == 'categorical' and data[target_column].nunique() <= 5:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=data[target_column], y=time_in_minutes, ax=ax)
                plt.title(f'{feature_for_analysis} by {target_column}')
                plt.ylabel('Time (minutes)')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Also show means by target
                means_by_target = data.groupby(target_column)[feature_for_analysis].apply(
                    lambda x: x.apply(convert_time_to_minutes).mean()
                ).sort_values()
                
                st.write("**Mean time by target:**")
                for target_val, mean_time_val in means_by_target.items():
                    formatted_time = minutes_to_time_string(mean_time_val)
                    st.write(f"{target_val}: {formatted_time}")
        
        elif feature_type == 'datetime':
            # Date feature analysis
            date_series = pd.to_datetime(data[feature_for_analysis], errors='coerce')
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Statistics:**")
                st.write(f"Earliest date: {date_series.min().strftime('%Y-%m-%d')}")
                st.write(f"Latest date: {date_series.max().strftime('%Y-%m-%d')}")
                st.write(f"Range: {(date_series.max() - date_series.min()).days} days")
                st.write(f"Missing values: {date_series.isna().sum()}")
            
            with col2:
                # Time series plot - count by month
                date_counts = date_series.dt.to_period('M').value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                date_counts.plot(kind='line', ax=ax)
                plt.title(f'Distribution of {feature_for_analysis} by Month')
                plt.xlabel('Month')
                plt.ylabel('Count')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Extract useful date components
            data_copy = data.copy()
            data_copy[f'{feature_for_analysis}_year'] = date_series.dt.year
            data_copy[f'{feature_for_analysis}_month'] = date_series.dt.month
            data_copy[f'{feature_for_analysis}_day'] = date_series.dt.day
            data_copy[f'{feature_for_analysis}_dayofweek'] = date_series.dt.dayofweek
            
            # Show target relationship with month
            if target_type == 'categorical':
                fig, ax = plt.subplots(figsize=(10, 6))
                monthly_target = data_copy.groupby(f'{feature_for_analysis}_month')[target_column].mean()
                monthly_target.plot(kind='bar', ax=ax)
                plt.title(f'{target_column} Rate by Month')
                plt.xlabel('Month')
                plt.ylabel(f'Average {target_column}')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Show target relationship with day of week
            if target_type == 'categorical':
                fig, ax = plt.subplots(figsize=(10, 6))
                dow_target = data_copy.groupby(f'{feature_for_analysis}_dayofweek')[target_column].mean()
                dow_target.plot(kind='bar', ax=ax)
                plt.title(f'{target_column} Rate by Day of Week')
                plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
                plt.ylabel(f'Average {target_column}')
                plt.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    show_data_exploration()