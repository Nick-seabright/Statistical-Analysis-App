# streamlit_app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import base64
from datetime import datetime
import pickle

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our package
from edu_analytics.data_processing import prepare_data, detect_target_type
from edu_analytics.feature_engineering import analyze_correlations
from edu_analytics.model_training import train_models
from edu_analytics.model_evaluation import evaluate_models
from edu_analytics.statistical_tests import (
    perform_t_test, visualize_t_test,
    perform_chi_square, visualize_chi_square,
    perform_anova, visualize_anova,
    multi_group_analysis, categorical_association_analysis,
    numerical_correlation_analysis, visualize_correlation_analysis
)
# streamlit_app/app.py (continued)
from edu_analytics.threshold_analysis import analyze_decision_boundaries
from edu_analytics.time_analysis import analyze_time_target
from edu_analytics.utils import (
    convert_time_to_minutes,
    interpret_prediction,
    create_model_parameter_widgets
)

# Set page configuration
st.set_page_config(
    page_title="Statistical Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    css = """
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subheader {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .info-text {
            font-size: 1rem;
            color: #555;
        }
        .stButton button {
            background-color: #1E88E5;
            color: white;
        }
        .important-metric {
            font-size: 1.2rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .report-section {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            margin-bottom: 1rem;
        }
        .report-header {
            text-align: center;
            font-size: 1.8rem;
            margin-bottom: 1rem;
            color: #333;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'data_types' not in st.session_state:
        st.session_state.data_types = None
    if 'target_type' not in st.session_state:
        st.session_state.target_type = None
    if 'target_mapping' not in st.session_state:
        st.session_state.target_mapping = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'categorical_encoders' not in st.session_state:
        st.session_state.categorical_encoders = {}
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    if 'report_data' not in st.session_state:
        st.session_state.report_data = {}

# Main layout function
def main():
    # Load CSS and initialize session state
    load_css()
    init_session_state()
    
    # Create sidebar
    create_sidebar()
    
    # Main app title
    st.markdown("<div class='main-header'>Statistical Analytics Dashboard</div>", unsafe_allow_html=True)
    
    # If no data is loaded, show the data upload section
    if st.session_state.data is None:
        data_upload_section()
    else:
        # Show different sections based on sidebar selection
        if st.session_state.current_section == "data_exploration":
            data_exploration_section()
        elif st.session_state.current_section == "statistical_analysis":
            statistical_analysis_section()
        elif st.session_state.current_section == "threshold_analysis":
            threshold_analysis_section()
        elif st.session_state.current_section == "model_training":
            model_training_section()
        elif st.session_state.current_section == "predictions":
            predictions_section()
        elif st.session_state.current_section == "report_generation":
            report_generation_section()

# Sidebar creation
def create_sidebar():
    with st.sidebar:
        st.image("streamlit_app/assets/logo.png", width=150)
        st.title("Navigation")
        
        # Initialize current section in session state if not exists
        if 'current_section' not in st.session_state:
            st.session_state.current_section = "data_exploration"
        
        # Navigation buttons
        if st.button("📊 Data Exploration", key="nav_data"):
            st.session_state.current_section = "data_exploration"
        
        if st.button("📈 Statistical Analysis", key="nav_stats"):
            st.session_state.current_section = "statistical_analysis"
            
        if st.button("🎯 Threshold Analysis", key="nav_threshold"):
            st.session_state.current_section = "threshold_analysis"
            
        if st.button("🧠 Model Training", key="nav_models"):
            st.session_state.current_section = "model_training"
            
        if st.button("🔮 Make Predictions", key="nav_predict"):
            st.session_state.current_section = "predictions"
            
        if st.button("📝 Generate Report", key="nav_report"):
            st.session_state.current_section = "report_generation"
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset All", key="nav_reset"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
        
        # Display dataset info if data is loaded
        if st.session_state.data is not None:
            st.markdown("---")
            st.subheader("Dataset Info")
            st.write(f"Rows: {st.session_state.data.shape[0]}")
            st.write(f"Columns: {st.session_state.data.shape[1]}")
            if st.session_state.target_type:
                st.write(f"Target Type: {st.session_state.target_type}")
                
        # Add helpful resources
        st.markdown("---")
        st.subheader("Resources")
        st.markdown("[GitHub Repository](https://github.com/Nick-seabright/Statistical-Analytics-App)")
        st.markdown("[Submit Issue](https://github.com/yourusername/statistical-analytics/issues)")

# Data upload section
def data_upload_section():
    st.markdown("<div class='subheader'>Upload Your Data</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Upload a CSV file containing your statistical data.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            # Show data preview
            st.markdown("<div class='subheader'>Data Preview</div>", unsafe_allow_html=True)
            st.dataframe(df.head(10))
            
            # Data summary
            st.markdown("<div class='subheader'>Data Summary</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Rows: {df.shape[0]}")
                st.write(f"Columns: {df.shape[1]}")
                st.write(f"Missing values: {df.isna().sum().sum()}")
            
            with col2:
                # Get numeric and categorical columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                st.write(f"Numeric columns: {len(numeric_cols)}")
                st.write(f"Categorical columns: {len(categorical_cols)}")
            
            # Analyze data types for all columns
            st.markdown("<div class='subheader'>Data Types</div>", unsafe_allow_html=True)
            data_types = {}
            for column in df.columns:
                from edu_analytics.data_processing import infer_and_validate_data_type
                data_types[column] = infer_and_validate_data_type(df[column])
            
            # Display inferred data types
            data_types_df = pd.DataFrame({
                'Column': list(data_types.keys()),
                'Inferred Type': list(data_types.values())
            })
            st.dataframe(data_types_df)
            st.session_state.data_types = data_types
            
            # Target selection
            st.markdown("<div class='subheader'>Select Target Variable</div>", unsafe_allow_html=True)
            target_column = st.selectbox("Choose your target variable", df.columns)
            
            # Feature selection
            st.markdown("<div class='subheader'>Select Features</div>", unsafe_allow_html=True)
            st.markdown("<div class='info-text'>Select the features you want to use for analysis.</div>", unsafe_allow_html=True)
            
            all_features = st.checkbox("Select All Features", value=True)
            
            if all_features:
                selected_features = [col for col in df.columns if col != target_column]
            else:
                selected_features = st.multiselect(
                    "Choose your features",
                    [col for col in df.columns if col != target_column],
                    default=[col for col in df.columns if col != target_column][:5]  # Default select first 5 features
                )
            
            # Process data button
            if st.button("Process Data", key="process_data"):
                with st.spinner("Processing data..."):
                    try:
                        # Process the data
                        X, y, categorical_encoders, target_type, target_mapping, scaler, original_target = prepare_data(
                            df, target_column, selected_features, data_types
                        )
                        
                        # Save processed data in session state
                        st.session_state.processed_data = {
                            'X': X,
                            'y': y,
                            'target_column': target_column,
                            'selected_features': selected_features,
                            'original_target': original_target
                        }
                        
                        # Save important metadata
                        st.session_state.target_type = target_type
                        st.session_state.target_mapping = target_mapping
                        st.session_state.scaler = scaler
                        st.session_state.categorical_encoders = categorical_encoders
                        
                        # Show success message
                        st.success(f"Data processed successfully! Target '{target_column}' detected as {target_type} type.")
                        
                        # Navigate to data exploration
                        st.session_state.current_section = "data_exploration"
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Data exploration section
def data_exploration_section():
    if st.session_state.processed_data is None:
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
    
    with tab2:
        st.markdown("<div class='subheader'>Correlation Analysis</div>", unsafe_allow_html=True)
        
        # Show correlation matrix for numeric features
        numeric_data = data[selected_features + [target_column]].select_dtypes(include=['int64', 'float64'])
        if numeric_data.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = numeric_data.corr()
            import seaborn as sns
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
            
            # Show top correlated features with target
            if target_column in corr_matrix.columns:
                st.markdown("<div class='subheader'>Top Correlations with Target</div>", unsafe_allow_html=True)
                target_corrs = corr_matrix[target_column].drop(target_column).sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                target_corrs.plot(kind='bar', ax=ax)
                plt.title(f'Feature Correlations with {target_column}')
                plt.ylabel('Correlation Coefficient')
                plt.tight_layout()
                st.pyplot(fig)
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
            st.pyplot(fig)
            
            # Distribution by target if target is categorical and not too many categories
            if target_type == 'categorical' and data[target_column].nunique() <= 5:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=data, x=dist_feature, hue=target_column, kde=True, ax=ax)
                plt.title(f'Distribution of {dist_feature} by {target_column}')
                st.pyplot(fig)
        else:
            # Categorical feature - bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts = data[dist_feature].value_counts().sort_values(ascending=False).head(15)
            value_counts.plot(kind='bar', ax=ax)
            plt.title(f'Distribution of {dist_feature}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Distribution by target if target is categorical
            if target_type == 'categorical' and data[target_column].nunique() <= 5:
                # Create cross-tabulation
                cross_tab = pd.crosstab(
                    data[dist_feature], 
                    data[target_column],
                    normalize='index'
                ).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                cross_tab.plot(kind='bar', stacked=True, ax=ax)
                plt.title(f'{target_column} Distribution by {dist_feature}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab4:
        st.markdown("<div class='subheader'>Feature Analysis</div>", unsafe_allow_html=True)
        
        # Select feature for detailed analysis
        feature_for_analysis = st.selectbox("Select feature for detailed analysis", selected_features)
        
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
                mean_time = interpret_prediction(mean_minutes, 'time')
                median_time = interpret_prediction(median_minutes, 'time')
                min_time = interpret_prediction(min_minutes, 'time')
                max_time = interpret_prediction(max_minutes, 'time')
                
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
                    formatted_time = interpret_prediction(mean_time_val, 'time')
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

# Statistical analysis section
def statistical_analysis_section():
    if st.session_state.processed_data is None:
        st.warning("Please process your data first.")
        return
    
    st.markdown("<div class='subheader'>Statistical Analysis</div>", unsafe_allow_html=True)
    
    # Get data from session state
    data = st.session_state.data
    X = st.session_state.processed_data['X']
    y = st.session_state.processed_data['y']
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
        
        # Select feature and grouping variable
        numeric_feature = st.selectbox("Select numeric feature", numeric_features)
        
        # For grouping, we can use the target if it's binary, or another binary feature
        if target_type == 'categorical' and data[target_column].nunique() == 2:
            default_group = target_column
        else:
            default_group = binary_features[0] if binary_features else None
        
        if default_group:
            grouping_variable = st.selectbox(
                "Select grouping variable (must be binary)",
                binary_features,
                index=binary_features.index(default_group) if default_group in binary_features else 0
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
        
        # Select feature and target
        if target_type == 'categorical':
            default_target = target_column
        else:
            default_target = categorical_features[0] if categorical_features else None
        
        if categorical_features and default_target:
            categorical_feature = st.selectbox("Select categorical feature", categorical_features)
            
            categorical_target = st.selectbox(
                "Select categorical target",
                [col for col in categorical_features if col != categorical_feature],
                index=categorical_features.index(default_target) if default_target in categorical_features and default_target != categorical_feature else 0
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
        else:
            st.warning("Not enough categorical variables found. Chi-square test requires two categorical variables.")
    
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
        
        # Select feature and grouping variable
        if numeric_features and multi_cat_features:
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
                                
                                # Visualize correlations
                                fig = visualize_correlation_analysis(corr_results)
                                st.pyplot(fig)
                                
                                # Show individual correlations
                                st.markdown("### Detailed Correlation Results")
                                
                                for feature in selected_corr_features:
                                    if "error" not in corr_results[feature]:
                                        result = corr_results[feature]
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric(
                                                label=f"Correlation with {feature}",
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
                # Calculate feature importance for categorical target
                if st.button("Analyze Feature Importance", key="run_importance"):
                    try:
                        with st.spinner("Analyzing feature importance..."):
                            # Prepare data for analysis
                            X_subset = data[numeric_features].copy()
                            y_subset = data[target_column].copy()
                            
                            # Handle missing values
                            X_subset = X_subset.fillna(X_subset.mean())
                            
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
                            
                            # Visualize feature importance
                            fig, ax = plt.subplots(figsize=(10, 6))
                            feature_imp.plot(kind='bar', x='Feature', y='Importance', ax=ax)
                            plt.title('Feature Importance for Predicting ' + target_column)
                            plt.ylabel('Importance Score')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
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
            else:
                st.warning("No numeric features found. Feature importance analysis requires numeric features.")

# Threshold analysis section
def threshold_analysis_section():
    if st.session_state.processed_data is None:
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
                    analyze_decision_boundaries(
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
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
            except Exception as e:
                st.error(f"Error analyzing thresholds: {str(e)}")
    
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
                        analyze_decision_boundaries(
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
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                except Exception as e:
                    st.error(f"Error analyzing feature combination: {str(e)}")
            
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
                        scatter = plt.scatter(
                            data[feature1],
                            data[feature2],
                            c=data[target_column] if target_type == 'categorical' else 
                              (data[target_column] > data[target_column].median()),
                            cmap='coolwarm',
                            alpha=0.6
                        )
                        
                        # Add threshold lines
                        plt.axvline(x=threshold1, color='r', linestyle='--', alpha=0.7)
                        plt.axhline(y=threshold2, color='r', linestyle='--', alpha=0.7)
                        
                        # Add quadrant labels
                        plt.text(
                            data[feature1].min(), 
                            threshold2, 
                            f"Above 1, Below 2\n{results.loc['Above 1, Below 2', 'target_1 rate']:.1%}",
                            verticalalignment='bottom'
                        )
                        plt.text(
                            threshold1, 
                            data[feature2].max(), 
                            f"Above Both\n{results.loc['Above Both', 'target_1 rate']:.1%}",
                            horizontalalignment='left',
                            verticalalignment='top'
                        )
                        plt.text(
                            data[feature1].max(), 
                            threshold2, 
                            f"Below 1, Above 2\n{results.loc['Below 1, Above 2', 'target_1 rate']:.1%}",
                            horizontalalignment='right',
                            verticalalignment='bottom'
                        )
                        plt.text(
                            threshold1, 
                            data[feature2].min(), 
                            f"Below Both\n{results.loc['Below Both', 'target_1 rate']:.1%}",
                            horizontalalignment='left'
                        )
                        
                        plt.colorbar(scatter)
                        plt.xlabel(feature1)
                        plt.ylabel(feature2)
                        plt.title(f'Custom Threshold Analysis: {feature1} vs {feature2}')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
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
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                except Exception as e:
                    st.error(f"Error analyzing custom thresholds: {str(e)}")

# Model training section
def model_training_section():
    if st.session_state.processed_data is None:
        st.warning("Please process your data first.")
        return
    
    st.markdown("<div class='subheader'>Model Training</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Train machine learning models to predict your target variable.</div>", unsafe_allow_html=True)
    
    # Get data from session state
    X = st.session_state.processed_data['X']
    y = st.session_state.processed_data['y']
    target_column = st.session_state.processed_data['target_column']
    selected_features = st.session_state.processed_data['selected_features']
    target_type = st.session_state.target_type
    
    # Create tabs for model training options
    tab1, tab2, tab3 = st.tabs(["Basic Models", "Advanced Configuration", "Model Evaluation"])
    
    with tab1:
        st.markdown("<div class='subheader'>Train Basic Models</div>", unsafe_allow_html=True)
        
        # Select models to train
        st.write("Select models to train:")
        
        # Model selection based on target type
        if target_type == 'categorical':
            train_rf = st.checkbox("Random Forest Classifier", value=True)
            train_svm = st.checkbox("Support Vector Machine", value=False)
            train_xgb = st.checkbox("XGBoost Classifier", value=True)
            train_nn = st.checkbox("Neural Network", value=False)
            
            # Train test split options
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
            
            # Training button
            if st.button("Train Models", key="train_basic"):
                try:
                    with st.spinner("Training models..."):
                        # List of models to train
                        models_to_train = []
                        if train_rf:
                            models_to_train.append(('Random Forest', 'rf'))
                        if train_svm:
                            models_to_train.append(('SVM', 'svm'))
                        if train_xgb:
                            models_to_train.append(('XGBoost', 'xgb'))
                        if train_nn:
                            models_to_train.append(('Neural Network', 'nn'))
                        
                        if not models_to_train:
                            st.warning("Please select at least one model to train.")
                            return
                        
                        # Train models using our function
                        trained_models, evaluation_results, feature_importance = train_models(
                            X=X,
                            y=y,
                            models=models_to_train,
                            target_type=target_type,
                            test_size=test_size,
                            random_state=42
                        )
                        
                        # Store trained models in session state
                        st.session_state.models = trained_models
                        st.session_state.model_evaluation = evaluation_results
                        st.session_state.feature_importance = feature_importance
                        
                        # Show success message
                        st.success(f"Successfully trained {len(trained_models)} models!")
                        
                        # Show evaluation results
                        st.markdown("### Model Performance")
                        st.dataframe(evaluation_results)
                        
                        # Show feature importance
                        if feature_importance is not None:
                            st.markdown("### Feature Importance")
                            
                            # Convert to DataFrame if it's a dictionary
                            if isinstance(feature_importance, dict):
                                # Take the first model's feature importance for display
                                model_name = list(feature_importance.keys())[0]
                                importance_df = feature_importance[model_name]
                                st.write(f"Feature importance from {model_name}:")
                            else:
                                importance_df = feature_importance
                            
                            # Display feature importance
                            st.dataframe(importance_df)
                            
                            # Plot feature importance
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importance_df.sort_values('importance', ascending=True).plot(
                                kind='barh', x='feature', y='importance', ax=ax)
                            plt.title('Feature Importance')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Store results in report data
                        st.session_state.report_data['model_training'] = {
                            'models_trained': [name for name, _ in models_to_train],
                            'evaluation_results': evaluation_results,
                            'feature_importance': feature_importance,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
        
        elif target_type in ['numeric', 'time']:
            train_rf = st.checkbox("Random Forest Regressor", value=True)
            train_svr = st.checkbox("Support Vector Regressor", value=False)
            train_xgb = st.checkbox("XGBoost Regressor", value=True)
            train_nn = st.checkbox("Neural Network Regressor", value=False)
            
            # Train test split options
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
            
            # Training button
            if st.button("Train Models", key="train_basic_reg"):
                try:
                    with st.spinner("Training regression models..."):
                        # List of models to train
                        models_to_train = []
                        if train_rf:
                            models_to_train.append(('Random Forest', 'rf'))
                        if train_svr:
                            models_to_train.append(('SVR', 'svr'))
                        if train_xgb:
                            models_to_train.append(('XGBoost', 'xgb'))
                        if train_nn:
                            models_to_train.append(('Neural Network', 'nn'))
                        
                        if not models_to_train:
                            st.warning("Please select at least one model to train.")
                            return
                        
                        # Train models using our function
                        trained_models, evaluation_results, feature_importance = train_models(
                            X=X,
                            y=y,
                            models=models_to_train,
                            target_type=target_type,
                            test_size=test_size,
                            random_state=42
                        )
                        
                        # Store trained models in session state
                        st.session_state.models = trained_models
                        st.session_state.model_evaluation = evaluation_results
                        st.session_state.feature_importance = feature_importance
                        
                        # Show success message
                        st.success(f"Successfully trained {len(trained_models)} models!")
                        
                        # Show evaluation results
                        st.markdown("### Model Performance")
                        st.dataframe(evaluation_results)
                        
                        # Show feature importance
                        if feature_importance is not None:
                            st.markdown("### Feature Importance")
                            
                            # Convert to DataFrame if it's a dictionary
                            if isinstance(feature_importance, dict):
                                # Take the first model's feature importance for display
                                model_name = list(feature_importance.keys())[0]
                                importance_df = feature_importance[model_name]
                                st.write(f"Feature importance from {model_name}:")
                            else:
                                importance_df = feature_importance
                            
                            # Display feature importance
                            st.dataframe(importance_df)
                            
                            # Plot feature importance
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importance_df.sort_values('importance', ascending=True).plot(
                                kind='barh', x='feature', y='importance', ax=ax)
                            plt.title('Feature Importance')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Store results in report data
                        st.session_state.report_data['model_training'] = {
                            'models_trained': [name for name, _ in models_to_train],
                            'evaluation_results': evaluation_results,
                            'feature_importance': feature_importance,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                except Exception as e:
                    st.error(f"Error training regression models: {str(e)}")
        
        else:
            st.warning(f"Model training is not supported for target type: {target_type}")
    
    with tab2:
        st.markdown("<div class='subheader'>Advanced Model Configuration</div>", unsafe_allow_html=True)
        
        # Model selection for advanced configuration
        model_type = st.selectbox(
            "Select model to configure",
            ["Random Forest", "XGBoost", "Neural Network", "SVM/SVR"]
        )
        
        # Configure hyperparameters based on model type
        if model_type == "Random Forest":
            st.markdown("### Random Forest Configuration")
            
            if target_type == 'categorical':
                # Random Forest Classifier params
                n_estimators = st.slider("Number of trees", 10, 500, 100, 10)
                max_depth = st.slider("Maximum tree depth", 2, 50, 10, 1)
                min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1)
                class_weight = st.selectbox("Class weights", ["balanced", "balanced_subsample", "None"])
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options")
                
                if show_advanced:
                    min_samples_leaf = st.slider("Minimum samples per leaf", 1, 20, 1, 1)
                    criterion = st.selectbox("Split criterion", ["gini", "entropy"])
                    max_features = st.selectbox("Max features", ["sqrt", "log2", "None"])
                else:
                    min_samples_leaf = 1
                    criterion = "gini"
                    max_features = "sqrt"
                
                # Convert "None" string to None
                if class_weight == "None":
                    class_weight = None
                
                if max_features == "None":
                    max_features = None
                
                # Training button
                if st.button("Train Custom Random Forest", key="train_custom_rf"):
                    try:
                        with st.spinner("Training custom Random Forest..."):
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.model_selection import train_test_split
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Create and train model
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                criterion=criterion,
                                max_features=max_features,
                                class_weight=class_weight,
                                random_state=42
                            )
                            
                            model.fit(X_train, y_train)
                            
                            # Evaluate model
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                            
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Custom Random Forest trained with accuracy: {accuracy:.4f}")
                            
                            st.markdown("### Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # Feature importance
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            
                            # Create DataFrame for importance
                            importance_df = pd.DataFrame({
                                'feature': [X.columns[i] for i in indices],
                                'importance': [importances[i] for i in indices]
                            })
                            
                            st.markdown("### Feature Importance")
                            st.dataframe(importance_df)
                            
                            # Plot feature importance
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importance_df.sort_values('importance', ascending=True).tail(15).plot(
                                kind='barh', x='feature', y='importance', ax=ax)
                            plt.title('Feature Importance (Top 15)')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom Random Forest"
                            st.session_state.models[model_name] = model
                            
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            
                            st.session_state.report_data['custom_models'][model_name] = {
                                'accuracy': accuracy,
                                'report': report,
                                'feature_importance': importance_df,
                                'params': {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'criterion': criterion,
                                    'max_features': max_features,
                                    'class_weight': class_weight
                                },
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                    except Exception as e:
                        st.error(f"Error training custom Random Forest: {str(e)}")
            
            else:  # Regression
                # Random Forest Regressor params
                n_estimators = st.slider("Number of trees", 10, 500, 100, 10)
                max_depth = st.slider("Maximum tree depth", 2, 50, 10, 1)
                min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1)
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options")
                
                if show_advanced:
                    min_samples_leaf = st.slider("Minimum samples per leaf", 1, 20, 1, 1)
                    criterion = st.selectbox("Split criterion", ["squared_error", "absolute_error", "poisson"])
                    max_features = st.selectbox("Max features", ["sqrt", "log2", "None"])
                else:
                    min_samples_leaf = 1
                    criterion = "squared_error"
                    max_features = "sqrt"
                
                # Convert "None" string to None
                if max_features == "None":
                    max_features = None
                
                # Training button
                if st.button("Train Custom Random Forest", key="train_custom_rf_reg"):
                    try:
                        with st.spinner("Training custom Random Forest Regressor..."):
                            from sklearn.ensemble import RandomForestRegressor
                            from sklearn.model_selection import train_test_split
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Create and train model
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                criterion=criterion,
                                max_features=max_features,
                                random_state=42
                            )
                            
                            model.fit(X_train, y_train)
                            
                            # Evaluate model
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Custom Random Forest Regressor trained with R² Score: {r2:.4f}")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("MSE", f"{mse:.4f}")
                            col2.metric("RMSE", f"{rmse:.4f}")
                            col3.metric("MAE", f"{mae:.4f}")
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plt.scatter(y_test, y_pred, alpha=0.5)
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            plt.title('Actual vs Predicted')
                            plt.xlabel('Actual')
                            plt.ylabel('Predicted')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Feature importance
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            
                            # Create DataFrame for importance
                            importance_df = pd.DataFrame({
                                'feature': [X.columns[i] for i in indices],
                                'importance': [importances[i] for i in indices]
                            })
                            
                            st.markdown("### Feature Importance")
                            st.dataframe(importance_df)
                            
                            # Plot feature importance
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importance_df.sort_values('importance', ascending=True).tail(15).plot(
                                kind='barh', x='feature', y='importance', ax=ax)
                            plt.title('Feature Importance (Top 15)')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom Random Forest Regressor"
                            st.session_state.models[model_name] = model
                            
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            
                            st.session_state.report_data['custom_models'][model_name] = {
                                'mse': mse,
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2,
                                'feature_importance': importance_df,
                                'params': {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'criterion': criterion,
                                    'max_features': max_features
                                },
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                    except Exception as e:
                        st.error(f"Error training custom Random Forest Regressor: {str(e)}")
        
        elif model_type == "XGBoost":
            st.markdown("### XGBoost Configuration")
            
            if target_type == 'categorical':
                # XGBoost Classifier params
                n_estimators = st.slider("Number of trees", 10, 500, 100, 10, key="xgb_n_est")
                max_depth = st.slider("Maximum tree depth", 2, 20, 6, 1, key="xgb_depth")
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01, key="xgb_lr")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="xgb_adv")
                
                if show_advanced:
                    subsample = st.slider("Subsample ratio", 0.5, 1.0, 0.8, 0.1)
                    colsample_bytree = st.slider("Column sample by tree", 0.5, 1.0, 0.8, 0.1)
                    gamma = st.slider("Minimum loss reduction (gamma)", 0.0, 5.0, 0.0, 0.1)
                    reg_alpha = st.slider("L1 regularization (alpha)", 0.0, 5.0, 0.0, 0.1)
                    reg_lambda = st.slider("L2 regularization (lambda)", 0.0, 5.0, 1.0, 0.1)
                else:
                    subsample = 0.8
                    colsample_bytree = 0.8
                    gamma = 0
                    reg_alpha = 0
                    reg_lambda = 1
                
                # Training button
                if st.button("Train Custom XGBoost", key="train_custom_xgb"):
                    try:
                        with st.spinner("Training custom XGBoost Classifier..."):
                            import xgboost as xgb
                            from sklearn.model_selection import train_test_split
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Create and train model
                            model = xgb.XGBClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                gamma=gamma,
                                reg_alpha=reg_alpha,
                                reg_lambda=reg_lambda,
                                random_state=42
                            )
                            
                            model.fit(X_train, y_train)
                            
                            # Evaluate model
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                            
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Custom XGBoost trained with accuracy: {accuracy:.4f}")
                            
                            st.markdown("### Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # Feature importance
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            
                            # Create DataFrame for importance
                            importance_df = pd.DataFrame({
                                'feature': [X.columns[i] for i in indices],
                                'importance': [importances[i] for i in indices]
                            })
                            
                            st.markdown("### Feature Importance")
                            st.dataframe(importance_df)
                            
                            # Plot feature importance
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importance_df.sort_values('importance', ascending=True).tail(15).plot(
                                kind='barh', x='feature', y='importance', ax=ax)
                            plt.title('Feature Importance (Top 15)')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom XGBoost"
                            st.session_state.models[model_name] = model
                            
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            
                            st.session_state.report_data['custom_models'][model_name] = {
                                'accuracy': accuracy,
                                'report': report,
                                'feature_importance': importance_df,
                                'params': {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'learning_rate': learning_rate,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample_bytree,
                                    'gamma': gamma,
                                    'reg_alpha': reg_alpha,
                                    'reg_lambda': reg_lambda
                                },
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                    except Exception as e:
                        st.error(f"Error training custom XGBoost: {str(e)}")
            
            else:  # Regression
                # XGBoost Regressor params
                n_estimators = st.slider("Number of trees", 10, 500, 100, 10, key="xgb_reg_n_est")
                max_depth = st.slider("Maximum tree depth", 2, 20, 6, 1, key="xgb_reg_depth")
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01, key="xgb_reg_lr")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="xgb_reg_adv")
                
                if show_advanced:
                    subsample = st.slider("Subsample ratio", 0.5, 1.0, 0.8, 0.1, key="xgb_reg_sub")
                    colsample_bytree = st.slider("Column sample by tree", 0.5, 1.0, 0.8, 0.1, key="xgb_reg_col")
                    gamma = st.slider("Minimum loss reduction (gamma)", 0.0, 5.0, 0.0, 0.1, key="xgb_reg_gamma")
                    reg_alpha = st.slider("L1 regularization (alpha)", 0.0, 5.0, 0.0, 0.1, key="xgb_reg_alpha")
                    reg_lambda = st.slider("L2 regularization (lambda)", 0.0, 5.0, 1.0, 0.1, key="xgb_reg_lambda")
                else:
                    subsample = 0.8
                    colsample_bytree = 0.8
                    gamma = 0
                    reg_alpha = 0
                    reg_lambda = 1
                
                # Training button
                if st.button("Train Custom XGBoost", key="train_custom_xgb_reg"):
                    try:
                        with st.spinner("Training custom XGBoost Regressor..."):
                            import xgboost as xgb
                            from sklearn.model_selection import train_test_split
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Create and train model
                            model = xgb.XGBRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                gamma=gamma,
                                reg_alpha=reg_alpha,
                                reg_lambda=reg_lambda,
                                random_state=42
                            )
                            
                            model.fit(X_train, y_train)
                            
                            # Evaluate model
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Custom XGBoost Regressor trained with R² Score: {r2:.4f}")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("MSE", f"{mse:.4f}")
                            col2.metric("RMSE", f"{rmse:.4f}")
                            col3.metric("MAE", f"{mae:.4f}")
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plt.scatter(y_test, y_pred, alpha=0.5)
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            plt.title('Actual vs Predicted')
                            plt.xlabel('Actual')
                            plt.ylabel('Predicted')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Feature importance
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            
                            # Create DataFrame for importance
                            importance_df = pd.DataFrame({
                                'feature': [X.columns[i] for i in indices],
                                'importance': [importances[i] for i in indices]
                            })
                            
                            st.markdown("### Feature Importance")
                            st.dataframe(importance_df)
                            
                            # Plot feature importance
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importance_df.sort_values('importance', ascending=True).tail(15).plot(
                                kind='barh', x='feature', y='importance', ax=ax)
                            plt.title('Feature Importance (Top 15)')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom XGBoost Regressor"
                            st.session_state.models[model_name] = model
                            
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            
                            st.session_state.report_data['custom_models'][model_name] = {
                                'mse': mse,
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2,
                                'feature_importance': importance_df,
                                'params': {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'learning_rate': learning_rate,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample_bytree,
                                    'gamma': gamma,
                                    'reg_alpha': reg_alpha,
                                    'reg_lambda': reg_lambda
                                },
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                    except Exception as e:
                        st.error(f"Error training custom XGBoost Regressor: {str(e)}")

        elif model_type == "Neural Network":
            st.markdown("### Neural Network Configuration")
            
            if target_type == 'categorical':
                # Neural Network Classifier params
                n_layers = st.slider("Number of hidden layers", 1, 5, 2, 1)
                layer_sizes = []
                
                for i in range(n_layers):
                    layer_sizes.append(st.slider(f"Neurons in layer {i+1}", 8, 256, 64, 8))
                
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.1)
                learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
                batch_size = st.slider("Batch size", 8, 128, 32, 8)
                epochs = st.slider("Epochs", 10, 200, 50, 10)
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="nn_adv")
                
                if show_advanced:
                    activation = st.selectbox("Activation function", ["relu", "tanh", "sigmoid", "elu"])
                    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop", "adagrad"])
                    use_batch_norm = st.checkbox("Use Batch Normalization", value=True)
                    early_stopping = st.checkbox("Use Early Stopping", value=True)
                    patience = st.slider("Early stopping patience", 5, 30, 10, 1)
                else:
                    activation = "relu"
                    optimizer = "adam"
                    use_batch_norm = True
                    early_stopping = True
                    patience = 10
                
                # Training button
                if st.button("Train Custom Neural Network", key="train_custom_nn"):
                    try:
                        with st.spinner("Training custom Neural Network Classifier..."):
                            import tensorflow as tf
                            from tensorflow.keras.models import Sequential
                            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
                            from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
                            from tensorflow.keras.callbacks import EarlyStopping
                            from sklearn.model_selection import train_test_split
                            
                            # Set random seed for reproducibility
                            tf.random.set_seed(42)
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Get number of classes
                            n_classes = len(np.unique(y))
                            
                            # Create model
                            model = Sequential()
                            
                            # Input layer
                            model.add(Dense(layer_sizes[0], activation=activation, input_shape=(X.shape[1],)))
                            if use_batch_norm:
                                model.add(BatchNormalization())
                            model.add(Dropout(dropout_rate))
                            
                            # Hidden layers
                            for i in range(1, n_layers):
                                model.add(Dense(layer_sizes[i], activation=activation))
                                if use_batch_norm:
                                    model.add(BatchNormalization())
                                model.add(Dropout(dropout_rate))
                            
                            # Output layer
                            if n_classes == 2:  # Binary classification
                                model.add(Dense(1, activation='sigmoid'))
                                loss = 'binary_crossentropy'
                            else:  # Multi-class classification
                                model.add(Dense(n_classes, activation='softmax'))
                                loss = 'sparse_categorical_crossentropy'
                            
                            # Select optimizer
                            if optimizer == "adam":
                                opt = Adam(learning_rate=learning_rate)
                            elif optimizer == "sgd":
                                opt = SGD(learning_rate=learning_rate)
                            elif optimizer == "rmsprop":
                                opt = RMSprop(learning_rate=learning_rate)
                            else:
                                opt = Adagrad(learning_rate=learning_rate)
                            
                            # Compile model
                            model.compile(
                                optimizer=opt,
                                loss=loss,
                                metrics=['accuracy']
                            )
                            
                            # Callbacks
                            callbacks = []
                            if early_stopping:
                                es = EarlyStopping(
                                    monitor='val_loss',
                                    patience=patience,
                                    restore_best_weights=True
                                )
                                callbacks.append(es)
                            
                            # Train model
                            history = model.fit(
                                X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                callbacks=callbacks,
                                verbose=0
                            )
                            
                            # Evaluate model
                            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
                            y_pred = (model.predict(X_test) > 0.5).astype('int32') if n_classes == 2 else np.argmax(model.predict(X_test), axis=1)
                            
                            from sklearn.metrics import classification_report, confusion_matrix
                            
                            report = classification_report(y_test, y_pred, output_dict=True)
                            
                            # Display results
                            st.success(f"Custom Neural Network trained with accuracy: {accuracy:.4f}")
                            
                            # Plot training history
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                            
                            # Loss plot
                            ax1.plot(history.history['loss'], label='Training Loss')
                            ax1.plot(history.history['val_loss'], label='Validation Loss')
                            ax1.set_title('Model Loss')
                            ax1.set_ylabel('Loss')
                            ax1.set_xlabel('Epoch')
                            ax1.legend()
                            
                            # Accuracy plot
                            ax2.plot(history.history['accuracy'], label='Training Accuracy')
                            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                            ax2.set_title('Model Accuracy')
                            ax2.set_ylabel('Accuracy')
                            ax2.set_xlabel('Epoch')
                            ax2.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Classification report
                            st.markdown("### Classification Report")
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom Neural Network"
                            st.session_state.models[model_name] = model
                            
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            
                            st.session_state.report_data['custom_models'][model_name] = {
                                'accuracy': accuracy,
                                'report': report,
                                'params': {
                                    'n_layers': n_layers,
                                    'layer_sizes': layer_sizes,
                                    'dropout_rate': dropout_rate,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'epochs': epochs,
                                    'activation': activation,
                                    'optimizer': optimizer,
                                    'use_batch_norm': use_batch_norm
                                },
                                'history': {
                                    'loss': history.history['loss'],
                                    'val_loss': history.history['val_loss'],
                                    'accuracy': history.history['accuracy'],
                                    'val_accuracy': history.history['val_accuracy']
                                },
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                    except Exception as e:
                        st.error(f"Error training custom Neural Network: {str(e)}")
            
            else:  # Regression
                # Neural Network Regressor params
                n_layers = st.slider("Number of hidden layers", 1, 5, 2, 1, key="nn_reg_layers")
                layer_sizes = []
                
                for i in range(n_layers):
                    layer_sizes.append(st.slider(f"Neurons in layer {i+1}", 8, 256, 64, 8, key=f"nn_reg_l{i}"))
                
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.1, key="nn_reg_drop")
                learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="nn_reg_lr")
                batch_size = st.slider("Batch size", 8, 128, 32, 8, key="nn_reg_batch")
                epochs = st.slider("Epochs", 10, 200, 50, 10, key="nn_reg_epochs")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="nn_reg_adv")
                
                if show_advanced:
                    activation = st.selectbox("Activation function", ["relu", "tanh", "sigmoid", "elu"], key="nn_reg_act")
                    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop", "adagrad"], key="nn_reg_opt")
                    use_batch_norm = st.checkbox("Use Batch Normalization", value=True, key="nn_reg_bn")
                    early_stopping = st.checkbox("Use Early Stopping", value=True, key="nn_reg_es")
                    patience = st.slider("Early stopping patience", 5, 30, 10, 1, key="nn_reg_pat")
                else:
                    activation = "relu"
                    optimizer = "adam"
                    use_batch_norm = True
                    early_stopping = True
                    patience = 10
                
                # Training button
                if st.button("Train Custom Neural Network", key="train_custom_nn_reg"):
                    try:
                        with st.spinner("Training custom Neural Network Regressor..."):
                            import tensorflow as tf
                            from tensorflow.keras.models import Sequential
                            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
                            from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
                            from tensorflow.keras.callbacks import EarlyStopping
                            from sklearn.model_selection import train_test_split
                            
                            # Set random seed for reproducibility
                            tf.random.set_seed(42)
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Create model
                            model = Sequential()
                            
                            # Input layer
                            model.add(Dense(layer_sizes[0], activation=activation, input_shape=(X.shape[1],)))
                            if use_batch_norm:
                                model.add(BatchNormalization())
                            model.add(Dropout(dropout_rate))
                            
                            # Hidden layers
                            for i in range(1, n_layers):
                                model.add(Dense(layer_sizes[i], activation=activation))
                                if use_batch_norm:
                                    model.add(BatchNormalization())
                                model.add(Dropout(dropout_rate))
                            
                            # Output layer for regression
                            model.add(Dense(1))
                            
                            # Select optimizer
                            if optimizer == "adam":
                                opt = Adam(learning_rate=learning_rate)
                            elif optimizer == "sgd":
                                opt = SGD(learning_rate=learning_rate)
                            elif optimizer == "rmsprop":
                                opt = RMSprop(learning_rate=learning_rate)
                            else:
                                opt = Adagrad(learning_rate=learning_rate)
                            
                            # Compile model
                            model.compile(
                                optimizer=opt,
                                loss='mse',
                                metrics=['mae']
                            )
                            
                            # Callbacks
                            callbacks = []
                            if early_stopping:
                                es = EarlyStopping(
                                    monitor='val_loss',
                                    patience=patience,
                                    restore_best_weights=True
                                )
                                callbacks.append(es)
                            
                            # Train model
                            history = model.fit(
                                X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                callbacks=callbacks,
                                verbose=0
                            )
                            
                            # Evaluate model
                            y_pred = model.predict(X_test).flatten()
                            
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Custom Neural Network Regressor trained with R² Score: {r2:.4f}")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("MSE", f"{mse:.4f}")
                            col2.metric("RMSE", f"{rmse:.4f}")
                            col3.metric("MAE", f"{mae:.4f}")
                            
                            # Plot training history
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                            
                            # Loss plot
                            ax1.plot(history.history['loss'], label='Training Loss')
                            ax1.plot(history.history['val_loss'], label='Validation Loss')
                            ax1.set_title('Model Loss (MSE)')
                            ax1.set_ylabel('Loss')
                            ax1.set_xlabel('Epoch')
                            ax1.legend()
                            
                            # MAE plot
                            ax2.plot(history.history['mae'], label='Training MAE')
                            ax2.plot(history.history['val_mae'], label='Validation MAE')
                            ax2.set_title('Model MAE')
                            ax2.set_ylabel('MAE')
                            ax2.set_xlabel('Epoch')
                            ax2.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plt.scatter(y_test, y_pred, alpha=0.5)
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            plt.title('Actual vs Predicted')
                            plt.xlabel('Actual')
                            plt.ylabel('Predicted')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Error distribution
                            fig, ax = plt.subplots(figsize=(10, 6))
                            errors = y_test - y_pred
                            plt.hist(errors, bins=30)
                            plt.title('Error Distribution')
                            plt.xlabel('Prediction Error')
                            plt.ylabel('Frequency')
                            plt.axvline(x=0, color='r', linestyle='--')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom Neural Network Regressor"
                            st.session_state.models[model_name] = model
                            
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            
                            st.session_state.report_data['custom_models'][model_name] = {
                                'mse': mse,
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2,
                                'params': {
                                    'n_layers': n_layers,
                                    'layer_sizes': layer_sizes,
                                    'dropout_rate': dropout_rate,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'epochs': epochs,
                                    'activation': activation,
                                    'optimizer': optimizer,
                                    'use_batch_norm': use_batch_norm
                                },
                                'history': {
                                    'loss': history.history['loss'],
                                    'val_loss': history.history['val_loss'],
                                    'mae': history.history['mae'],
                                    'val_mae': history.history['val_mae']
                                },
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                    except Exception as e:
                        st.error(f"Error training custom Neural Network Regressor: {str(e)}")
        
        elif model_type == "SVM/SVR":
            st.markdown("### SVM Configuration")
            
            if target_type == 'categorical':
                # SVM params
                C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0, 0.1)
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="svm_adv")
                
                if show_advanced:
                    if kernel in ["rbf", "poly", "sigmoid"]:
                        gamma = st.selectbox("Kernel coefficient (gamma)", ["scale", "auto", "value"])
                        if gamma == "value":
                            gamma_value = st.slider("Gamma value", 0.001, 1.0, 0.1, 0.001, format="%.3f")
                            gamma = gamma_value
                    
                    if kernel == "poly":
                        degree = st.slider("Polynomial degree", 2, 10, 3, 1)
                    else:
                        degree = 3
                    
                    class_weight = st.selectbox("Class weights", ["balanced", "None"])
                    probability = st.checkbox("Enable probability estimates", value=True)
                else:
                    gamma = "scale"
                    degree = 3
                    class_weight = "balanced"
                    probability = True
                
                # Convert "None" string to None
                if class_weight == "None":
                    class_weight = None
                
                # Training button
                if st.button("Train Custom SVM", key="train_custom_svm"):
                    try:
                        with st.spinner("Training custom SVM..."):
                            from sklearn.svm import SVC
                            from sklearn.model_selection import train_test_split
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Create and train model
                            model = SVC(
                                C=C,
                                kernel=kernel,
                                gamma=gamma,
                                degree=degree,
                                class_weight=class_weight,
                                probability=probability,
                                random_state=42
                            )
                            
                            model.fit(X_train, y_train)
                            
                            # Evaluate model
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                            
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Custom SVM trained with accuracy: {accuracy:.4f}")
                            
                            st.markdown("### Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom SVM"
                            st.session_state.models[model_name] = model
                            
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            
                            st.session_state.report_data['custom_models'][model_name] = {
                                'accuracy': accuracy,
                                'report': report,
                                'params': {
                                    'C': C,
                                    'kernel': kernel,
                                    'gamma': gamma,
                                    'degree': degree,
                                    'class_weight': class_weight,
                                    'probability': probability
                                },
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                    except Exception as e:
                        st.error(f"Error training custom SVM: {str(e)}")
            
            else:  # Regression
                # SVR params
                C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0, 0.1, key="svr_c")
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svr_kernel")
                epsilon = st.slider("Epsilon in the epsilon-SVR model", 0.01, 1.0, 0.1, 0.01)
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="svr_adv")
                
                if show_advanced:
                    if kernel in ["rbf", "poly", "sigmoid"]:
                        gamma = st.selectbox("Kernel coefficient (gamma)", ["scale", "auto", "value"], key="svr_gamma")
                        if gamma == "value":
                            gamma_value = st.slider("Gamma value", 0.001, 1.0, 0.1, 0.001, format="%.3f", key="svr_gamma_val")
                            gamma = gamma_value
                    
                    if kernel == "poly":
                        degree = st.slider("Polynomial degree", 2, 10, 3, 1, key="svr_degree")
                    else:
                        degree = 3
                else:
                    gamma = "scale"
                    degree = 3
                
                # Training button
                if st.button("Train Custom SVR", key="train_custom_svr"):
                    try:
                        with st.spinner("Training custom SVR..."):
                            from sklearn.svm import SVR
                            from sklearn.model_selection import train_test_split
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Create and train model
                            model = SVR(
                                C=C,
                                kernel=kernel,
                                gamma=gamma,
                                degree=degree,
                                epsilon=epsilon
                            )
                            
                            model.fit(X_train, y_train)
                            
                            # Evaluate model
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Custom SVR trained with R² Score: {r2:.4f}")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("MSE", f"{mse:.4f}")
                            col2.metric("RMSE", f"{rmse:.4f}")
                            col3.metric("MAE", f"{mae:.4f}")
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plt.scatter(y_test, y_pred, alpha=0.5)
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            plt.title('Actual vs Predicted')
                            plt.xlabel('Actual')
                            plt.ylabel('Predicted')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Error distribution
                            fig, ax = plt.subplots(figsize=(10, 6))
                            errors = y_test - y_pred
                            plt.hist(errors, bins=30)
                            plt.title('Error Distribution')
                            plt.xlabel('Prediction Error')
                            plt.ylabel('Frequency')
                            plt.axvline(x=0, color='r', linestyle='--')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom SVR"
                            st.session_state.models[model_name] = model
                            
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            
                            st.session_state.report_data['custom_models'][model_name] = {
                                'mse': mse,
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2,
                                'params': {
                                    'C': C,
                                    'kernel': kernel,
                                    'gamma': gamma,
                                    'degree': degree,
                                    'epsilon': epsilon
                                },
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                    except Exception as e:
                        st.error(f"Error training custom SVR: {str(e)}")
    
    with tab3:
        st.markdown("<div class='subheader'>Model Evaluation</div>", unsafe_allow_html=True)
        
        # Check if models exist
        if 'models' not in st.session_state or not st.session_state.models:
            st.warning("No trained models available for evaluation. Please train models first.")
            return
        
        # Model selection for evaluation
        model_names = list(st.session_state.models.keys())
        selected_models = st.multiselect(
            "Select models to evaluate",
            model_names,
            default=model_names[:2] if len(model_names) > 1 else model_names
        )
        
        if not selected_models:
            st.warning("Please select at least one model for evaluation.")
            return
        
        # Evaluation button
        if st.button("Evaluate Models", key="evaluate_models"):
            try:
                with st.spinner("Evaluating models..."):
                    from sklearn.model_selection import train_test_split, cross_val_score
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Evaluate selected models
                    results = {}
                    
                    for model_name in selected_models:
                        model = st.session_state.models[model_name]
                        
                        # Cross-validation score
                        if target_type == 'categorical':
                            cv_scores = cross_val_score(model, X, y, cv=5)
                            cv_metric = "Accuracy"
                        else:
                            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                            cv_metric = "R² Score"
                        
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                        
                        # Store results
                        results[model_name] = {
                            f"CV {cv_metric}": cv_mean,
                            f"CV {cv_metric} Std": cv_std,
                            "CV Scores": cv_scores
                        }
                    
                    # Display results
                    st.markdown("### Cross-Validation Results")
                    
                    # Create comparison DataFrame
                    comparison_data = {}
                    for model_name, result in results.items():
                        comparison_data[model_name] = {
                            f"CV {cv_metric}": f"{result[f'CV {cv_metric}']:.4f} ± {result[f'CV {cv_metric} Std']:.4f}"
                        }
                    
                    comparison_df = pd.DataFrame(comparison_data).transpose()
                    st.dataframe(comparison_df)
                    
                    # Plot comparison
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    x = np.arange(len(selected_models))
                    cv_means = [results[model][f"CV {cv_metric}"] for model in selected_models]
                    cv_stds = [results[model][f"CV {cv_metric} Std"] for model in selected_models]
                    
                    # Bar plot with error bars
                    plt.bar(x, cv_means, yerr=cv_stds, alpha=0.7, capsize=10)
                    plt.xticks(x, selected_models, rotation=45, ha='right')
                    plt.ylabel(f"Cross-Validation {cv_metric}")
                    plt.title(f"Model Comparison - {cv_metric}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Individual model evaluations
                    st.markdown("### Individual Model Evaluations")
                    
                    for model_name in selected_models:
                        st.markdown(f"#### {model_name}")
                        model = st.session_state.models[model_name]
                        
                        if target_type == 'categorical':
                            # Classification evaluation
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
                            
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Display metrics
                            st.metric("Accuracy", f"{accuracy:.4f}")
                            
                            # Classification report
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(8, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # ROC curve for binary classification
                            if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                                try:
                                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                    roc_auc = auc(fpr, tpr)
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                                    plt.plot([0, 1], [0, 1], 'k--')
                                    plt.xlim([0.0, 1.0])
                                    plt.ylim([0.0, 1.05])
                                    plt.xlabel('False Positive Rate')
                                    plt.ylabel('True Positive Rate')
                                    plt.title('Receiver Operating Characteristic')
                                    plt.legend(loc="lower right")
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"Could not generate ROC curve: {str(e)}")
                            
                        else:
                            # Regression evaluation
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("MSE", f"{mse:.4f}")
                            col2.metric("RMSE", f"{rmse:.4f}")
                            col3.metric("MAE", f"{mae:.4f}")
                            col4.metric("R² Score", f"{r2:.4f}")
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(8, 6))
                            plt.scatter(y_test, y_pred, alpha=0.5)
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            plt.title('Actual vs Predicted')
                            plt.xlabel('Actual')
                            plt.ylabel('Predicted')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Error distribution
                            fig, ax = plt.subplots(figsize=(8, 6))
                            errors = y_test - y_pred
                            plt.hist(errors, bins=30)
                            plt.title('Error Distribution')
                            plt.xlabel('Prediction Error')
                            plt.ylabel('Frequency')
                            plt.axvline(x=0, color='r', linestyle='--')
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Store evaluation results
                    st.session_state.report_data['model_evaluation'] = {
                        'models_evaluated': selected_models,
                        'cross_validation_results': comparison_df.to_dict(),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
            except Exception as e:
                st.error(f"Error evaluating models: {str(e)}")

# Predictions section
def predictions_section():
    if st.session_state.processed_data is None or 'models' not in st.session_state or not st.session_state.models:
        st.warning("Please process your data and train models first.")
        return
    
    st.markdown("<div class='subheader'>Make Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Use trained models to make predictions on new data.</div>", unsafe_allow_html=True)
    
    # Get data from session state
    target_column = st.session_state.processed_data['target_column']
    selected_features = st.session_state.processed_data['selected_features']
    target_type = st.session_state.target_type
    target_mapping = st.session_state.target_mapping
    data_types = st.session_state.data_types
    
    # Create tabs for different prediction options
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Predictions"])
    
    with tab1:
        st.markdown("<div class='subheader'>Make a Single Prediction</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Enter values for each feature to get a prediction.</div>", unsafe_allow_html=True)
        
        # Select model for prediction
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select model for prediction", model_names)
        
        # Create input widgets for features
        input_values = {}
        
        for feature in selected_features:
            feature_type = data_types.get(feature, 'numeric')
            
            if feature_type in ['integer', 'float', 'numeric']:
                input_values[feature] = st.number_input(f"{feature}", value=0.0, key=f"pred_{feature}")
            elif feature_type == 'boolean':
                input_values[feature] = st.checkbox(f"{feature}", value=False, key=f"pred_{feature}")
            elif feature_type == 'categorical':
                # If we have categorical encoders, use them
                if hasattr(st.session_state, 'categorical_encoders') and feature in st.session_state.categorical_encoders:
                    encoder = st.session_state.categorical_encoders[feature]
                    options = list(encoder.classes_)
                    selected = st.selectbox(f"{feature}", options, key=f"pred_{feature}")
                    # Will encode later when making prediction
                    input_values[feature] = selected
                else:
                    input_values[feature] = st.text_input(f"{feature}", value="", key=f"pred_{feature}")
            elif feature_type == 'time':
                input_values[feature] = st.text_input(f"{feature} (format: MM:SS or HH:MM:SS)", value="00:00", key=f"pred_{feature}")
            elif feature_type == 'datetime':
                input_values[feature] = st.date_input(f"{feature}", value=datetime.now(), key=f"pred_{feature}")
            else:
                input_values[feature] = st.text_input(f"{feature}", value="", key=f"pred_{feature}")
        
        # Prediction button
        if st.button("Make Prediction", key="make_pred"):
            try:
                with st.spinner("Making prediction..."):
                    # Get selected model
                    model = st.session_state.models[selected_model]
                    
                    # Preprocess input data
                    input_processed = {}
                    
                    for feature, value in input_values.items():
                        feature_type = data_types.get(feature, 'numeric')
                        
                        if feature_type == 'time':
                            # Convert time to minutes
                            input_processed[feature] = convert_time_to_minutes(value)
                        elif feature_type == 'categorical':
                            # Handle categorical encoding
                            if hasattr(st.session_state, 'categorical_encoders') and feature in st.session_state.categorical_encoders:
                                encoder = st.session_state.categorical_encoders[feature]
                                try:
                                    input_processed[feature] = encoder.transform([str(value)])[0]
                                except:
                                    st.error(f"Invalid value for {feature}. Must be one of: {', '.join(encoder.classes_)}")
                                    return
                            else:
                                input_processed[feature] = value
                        elif feature_type == 'datetime':
                            # Convert to string for now
                            input_processed[feature] = value.strftime("%Y-%m-%d")
                        elif feature_type == 'boolean':
                            input_processed[feature] = 1 if value else 0
                        else:
                            input_processed[feature] = value
                    
                    # Create input array
                    input_df = pd.DataFrame([input_processed])
                    
                    # Scale the input data
                    if hasattr(st.session_state, 'scaler') and st.session_state.scaler is not None:
                        scaler = st.session_state.scaler
                        input_scaled = scaler.transform(input_df)
                    else:
                        input_scaled = input_df.values
                    
                    # Make prediction
                    if target_type == 'categorical':
                        prediction = model.predict(input_scaled)[0]
                        
                        # Map prediction back to original value if we have a mapping
                        if target_mapping:
                            reverse_mapping = {v: k for k, v in target_mapping.items()}
                            prediction_label = reverse_mapping.get(prediction, str(prediction))
                        else:
                            prediction_label = str(prediction)
                        
                        # Get probabilities if available
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(input_scaled)[0]
                            
                            # Display prediction
                            st.success(f"Prediction: {prediction_label}")
                            
                            # Display probabilities
                            st.markdown("### Prediction Probabilities")
                            
                            prob_data = []
                            for i, prob in enumerate(probabilities):
                                if target_mapping:
                                    class_label = reverse_mapping.get(i, str(i))
                                else:
                                    class_label = str(i)
                                
                                prob_data.append({
                                    'Class': class_label,
                                    'Probability': prob
                                })
                            
                            prob_df = pd.DataFrame(prob_data)
                            st.dataframe(prob_df)
                            
                            # Bar chart of probabilities
                            fig, ax = plt.subplots(figsize=(10, 5))
                            prob_df.sort_values('Probability', ascending=False).plot(
                                kind='bar', x='Class', y='Probability', ax=ax)
                            plt.ylabel('Probability')
                            plt.title('Prediction Probabilities')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            # Just display the prediction
                            st.success(f"Prediction: {prediction_label}")
                    
                    else:  # Numeric or time prediction
                        prediction = model.predict(input_scaled)[0]
                        
                        if target_type == 'time':
                            # Convert prediction to time format
                            prediction_formatted = interpret_prediction(prediction, 'time')
                            st.success(f"Prediction: {prediction_formatted}")
                            
                            # Also show in minutes
                            st.info(f"Predicted time in minutes: {prediction:.2f}")
                        else:
                            st.success(f"Prediction: {prediction:.4f}")
                    
                    # Store prediction in report data
                    if 'predictions' not in st.session_state.report_data:
                        st.session_state.report_data['predictions'] = []
                    
                    st.session_state.report_data['predictions'].append({
                        'model': selected_model,
                        'input': input_processed,
                        'prediction': prediction_label if target_type == 'categorical' else 
                                     prediction_formatted if target_type == 'time' else 
                                     prediction,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown("<div class='subheader'>Batch Predictions</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Upload a CSV file with new data for batch predictions.</div>", unsafe_allow_html=True)
        
        # Select model for batch prediction
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select model for batch prediction", model_names, key="batch_model")
        
        # File uploader for batch predictions
        uploaded_file = st.file_uploader("Upload CSV with new data", type="csv", key="batch_upload")
        
        if uploaded_file is not None:
            try:
                # Load the data
                batch_df = pd.read_csv(uploaded_file)
                
                # Display data preview
                st.markdown("### Data Preview")
                st.dataframe(batch_df.head())
                
                # Check if we have all required features
                missing_features = [feature for feature in selected_features if feature not in batch_df.columns]
                
                if missing_features:
                    st.error(f"Missing required features in uploaded data: {', '.join(missing_features)}")
                    return
                
                # Batch prediction button
                if st.button("Make Batch Predictions", key="batch_pred"):
                    try:
                        with st.spinner("Making batch predictions..."):
                            # Get selected model
                            model = st.session_state.models[selected_model]
                            
                            # Preprocess batch data
                            batch_processed = batch_df.copy()
                            
                            # Process each feature
                            for feature in selected_features:
                                feature_type = data_types.get(feature, 'numeric')
                                
                                if feature_type == 'time':
                                    # Convert time to minutes
                                    batch_processed[feature] = batch_processed[feature].apply(convert_time_to_minutes)
                                elif feature_type == 'categorical':
                                    # Handle categorical encoding
                                    if hasattr(st.session_state, 'categorical_encoders') and feature in st.session_state.categorical_encoders:
                                        encoder = st.session_state.categorical_encoders[feature]
                                        try:
                                            batch_processed[feature] = encoder.transform(batch_processed[feature].astype(str))
                                        except:
                                            st.error(f"Invalid values for {feature}. Must be one of: {', '.join(encoder.classes_)}")
                                            return
                                elif feature_type == 'datetime':
                                    # Convert to numeric (days since a reference date)
                                    batch_processed[feature] = pd.to_datetime(batch_processed[feature])
                                    batch_processed[feature] = (batch_processed[feature] - batch_processed[feature].min()).dt.total_seconds() / (24 * 3600)
                                elif feature_type == 'boolean':
                                    batch_processed[feature] = batch_processed[feature].astype(int)
                            
                            # Extract only the needed features in the right order
                            X_batch = batch_processed[selected_features]
                            
                            # Scale the data
                            if hasattr(st.session_state, 'scaler') and st.session_state.scaler is not None:
                                scaler = st.session_state.scaler
                                X_batch_scaled = scaler.transform(X_batch)
                            else:
                                X_batch_scaled = X_batch.values
                            
                            # Make predictions
                            if target_type == 'categorical':
                                predictions = model.predict(X_batch_scaled)
                                
                                # Map predictions back to original values if we have a mapping
                                if target_mapping:
                                    reverse_mapping = {v: k for k, v in target_mapping.items()}
                                    prediction_labels = [reverse_mapping.get(p, str(p)) for p in predictions]
                                else:
                                    prediction_labels = [str(p) for p in predictions]
                                
                                # Get probabilities if available
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(X_batch_scaled)
                                    
                                    # Create probability columns
                                    class_names = []
                                    for i in range(probabilities.shape[1]):
                                        if target_mapping:
                                            class_name = reverse_mapping.get(i, str(i))
                                        else:
                                            class_name = str(i)
                                        class_names.append(class_name)
                                        batch_df[f'Prob_{class_name}'] = probabilities[:, i]
                                
                                # Add prediction column
                                batch_df['Prediction'] = prediction_labels
                                
                            else:  # Numeric or time prediction
                                predictions = model.predict(X_batch_scaled)
                                
                                if target_type == 'time':
                                    # Add both formatted time and minutes
                                    batch_df['Prediction_Minutes'] = predictions
                                    batch_df['Prediction'] = [interpret_prediction(p, 'time') for p in predictions]
                                else:
                                    batch_df['Prediction'] = predictions
                            
                            # Display results
                            st.markdown("### Prediction Results")
                            st.dataframe(batch_df)
                            
                            # Download button for results
                            csv = batch_df.to_csv(index=False).encode('utf-8')
                            
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime='text/csv',
                            )
                            
                            # Visualize predictions
                            if target_type == 'categorical':
                                # Distribution of predictions
                                fig, ax = plt.subplots(figsize=(10, 6))
                                prediction_counts = batch_df['Prediction'].value_counts()
                                prediction_counts.plot(kind='bar', ax=ax)
                                plt.title('Distribution of Predictions')
                                plt.ylabel('Count')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # If we have probabilities, show distribution of confidence
                                if hasattr(model, 'predict_proba'):
                                    # Get max probability for each prediction (confidence)
                                    if 'Prob_' in batch_df.columns[0]:  # Check if we have probability columns
                                        prob_cols = [c for c in batch_df.columns if c.startswith('Prob_')]
                                        batch_df['Confidence'] = batch_df[prob_cols].max(axis=1)
                                        
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        plt.hist(batch_df['Confidence'], bins=20)
                                        plt.title('Distribution of Prediction Confidence')
                                        plt.xlabel('Confidence')
                                        plt.ylabel('Count')
                                        plt.tight_layout()
                                        st.pyplot(fig)
                            
                            else:  # Numeric or time prediction
                                # Distribution of predictions
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                if target_type == 'time':
                                    plt.hist(batch_df['Prediction_Minutes'], bins=20)
                                    plt.title('Distribution of Predicted Times')
                                    plt.xlabel('Time (minutes)')
                                else:
                                    plt.hist(batch_df['Prediction'], bins=20)
                                    plt.title('Distribution of Predictions')
                                    plt.xlabel('Predicted Value')
                                
                                plt.ylabel('Count')
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            # Store batch prediction info in report data
                            if 'batch_predictions' not in st.session_state.report_data:
                                st.session_state.report_data['batch_predictions'] = []
                            
                            st.session_state.report_data['batch_predictions'].append({
                                'model': selected_model,
                                'records_processed': len(batch_df),
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                    except Exception as e:
                        st.error(f"Error making batch predictions: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"Error loading batch data: {str(e)}")

# Report generation section
def report_generation_section():
    st.markdown("<div class='subheader'>Report Generation</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Generate a comprehensive report of your analysis.</div>", unsafe_allow_html=True)
    
    # Check if we have enough data for a report
    if 'report_data' not in st.session_state:
        st.warning("No analysis data available for report generation. Please perform some analysis first.")
        return
    
    # Report options
    st.markdown("### Report Options")
    
    # Get available sections
    available_sections = []
    
    if 'data' in st.session_state and st.session_state.data is not None:
        available_sections.append("Dataset Overview")
    
    if 'statistical_tests' in st.session_state.report_data:
        available_sections.append("Statistical Analysis")
    
    if 'threshold_analysis' in st.session_state.report_data:
        available_sections.append("Threshold Analysis")
    
    if 'model_training' in st.session_state.report_data:
        available_sections.append("Model Training")
    
    if 'model_evaluation' in st.session_state.report_data:
        available_sections.append("Model Evaluation")
    
    if 'predictions' in st.session_state.report_data or 'batch_predictions' in st.session_state.report_data:
        available_sections.append("Predictions")
    
    # Allow user to select sections to include
    selected_sections = st.multiselect(
        "Select sections to include in the report",
        available_sections,
        default=available_sections
    )
    
    # Report metadata
    st.markdown("### Report Metadata")
    report_title = st.text_input("Report Title", value="Statistical Analytics Report")
    author = st.text_input("Author", value="")
    
    # Generate report button
    if st.button("Generate Report", key="gen_report"):
        if not selected_sections:
            st.warning("Please select at least one section to include in the report.")
            return
        
        try:
            with st.spinner("Generating report..."):
                # Create report content
                report_html = generate_html_report(
                    title=report_title,
                    author=author,
                    sections=selected_sections,
                    report_data=st.session_state.report_data,
                    data=st.session_state.data if 'data' in st.session_state else None,
                    target_type=st.session_state.target_type if 'target_type' in st.session_state else None,
                    target_column=st.session_state.processed_data['target_column'] if 'processed_data' in st.session_state else None
                )
                
                # Convert HTML to PDF
                pdf_report = convert_html_to_pdf(report_html)
                
                # Provide download link
                st.download_button(
                    label="Download Report (PDF)",
                    data=pdf_report,
                    file_name=f"Statistical_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime='application/pdf',
                )
                
                # Also allow downloading HTML version
                st.download_button(
                    label="Download Report (HTML)",
                    data=report_html,
                    file_name=f"Statistical_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime='text/html',
                )
                
                # Show preview
                st.markdown("### Report Preview")
                st.components.v1.html(report_html, height=600, scrolling=True)
                
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Helper function to generate HTML report
def generate_html_report(title, author, sections, report_data, data=None, target_type=None, target_column=None):
    """Generate an HTML report with the given sections and data"""
    
    # Get current date and time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start HTML content
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3, h4 {{
                color: #1E88E5;
                margin-top: 24px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 1px solid #ddd;
            }}
            .metadata {{
                color: #666;
                font-style: italic;
                text-align: center;
                margin-bottom: 30px;
            }}
            .section {{
                margin-bottom: 40px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 8px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .img-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .img-container img {{
                max-width: 100%;
                height: auto;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 0.9em;
            }}
            .highlight {{
                background-color: #e3f2fd;
                padding: 2px 5px;
                border-radius: 3px;
            }}
            .metric {{
                font-weight: bold;
                color: #1E88E5;
            }}
            .warning {{
                color: #f57c00;
                font-weight: bold;
            }}
            .chart {{
                width: 100%;
                max-width: 800px;
                margin: 20px auto;
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
        </div>
        
        <div class="metadata">
            <p>Generated on: {now}</p>
            {f'<p>Author: {author}</p>' if author else ''}
        </div>
    """
    
    # Add content for each selected section
    for section in sections:
        if section == "Dataset Overview" and data is not None:
            html += generate_dataset_overview_section(data, target_type, target_column)
        
        elif section == "Statistical Analysis" and 'statistical_tests' in report_data:
            html += generate_statistical_analysis_section(report_data['statistical_tests'])
        
        elif section == "Threshold Analysis" and 'threshold_analysis' in report_data:
            html += generate_threshold_analysis_section(report_data['threshold_analysis'])
        
        elif section == "Model Training" and 'model_training' in report_data:
            html += generate_model_training_section(report_data['model_training'])
        
        elif section == "Model Evaluation" and 'model_evaluation' in report_data:
            html += generate_model_evaluation_section(report_data['model_evaluation'])
        
        elif section == "Predictions" and ('predictions' in report_data or 'batch_predictions' in report_data):
            html += generate_predictions_section(report_data)
    
    # Add footer
    html += f"""
        <div class="footer">
            <p>Generated using Statistical Analytics Tool</p>
            <p>© {datetime.now().year} Statistical Analytics</p>
        </div>
    </body>
    </html>
    """
    
    return html

# Helper function to convert HTML to PDF
def convert_html_to_pdf(html_content):
    """Convert HTML content to PDF"""
    # For simplicity, we'll just return the HTML as bytes
    # In a real app, you'd use a library like weasyprint or a service like wkhtmltopdf
    try:
        # Try to import weasyprint if available
        from weasyprint import HTML
        
        # Convert HTML to PDF
        pdf_bytes = HTML(string=html_content).write_pdf()
        return pdf_bytes
    except ImportError:
        # If weasyprint is not available, return HTML bytes instead
        st.warning("WeasyPrint not available. Downloading HTML file instead.")
        return html_content.encode('utf-8')

# Helper function to generate dataset overview section
def generate_dataset_overview_section(data, target_type, target_column):
    """Generate HTML for dataset overview section"""
    
    # Basic dataset stats
    rows = data.shape[0]
    cols = data.shape[1]
    missing_values = data.isna().sum().sum()
    missing_pct = (missing_values / (rows * cols)) * 100
    
    # Column types breakdown
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    html = f"""
        <div class="section">
            <h2>Dataset Overview</h2>
            
            <h3>Basic Statistics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Number of Rows</td>
                    <td>{rows}</td>
                </tr>
                <tr>
                    <td>Number of Columns</td>
                    <td>{cols}</td>
                </tr>
                <tr>
                    <td>Missing Values</td>
                    <td>{missing_values} ({missing_pct:.2f}%)</td>
                </tr>
                <tr>
                    <td>Numeric Columns</td>
                    <td>{len(numeric_cols)}</td>
                </tr>
                <tr>
                    <td>Categorical Columns</td>
                    <td>{len(categorical_cols)}</td>
                </tr>
                <tr>
                    <td>Target Column</td>
                    <td>{target_column} ({target_type})</td>
                </tr>
            </table>
            
            <h3>Column Information</h3>
            <table>
                <tr>
                    <th>Column Name</th>
                    <th>Type</th>
                    <th>Non-Null Count</th>
                    <th>Unique Values</th>
                </tr>
    """
    
    # Add row for each column
    for col in data.columns:
        col_type = data[col].dtype
        non_null = data[col].count()
        unique = data[col].nunique()
        html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null} ({(non_null/rows)*100:.1f}%)</td>
                    <td>{unique}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    return html

# Helper function to generate statistical analysis section
def generate_statistical_analysis_section(statistical_tests):
    """Generate HTML for statistical analysis section"""
    
    if not statistical_tests:
        return ""
    
    html = f"""
        <div class="section">
            <h2>Statistical Analysis</h2>
    """
    
    # Process each type of test
    for test_key, test_data in statistical_tests.items():
        if test_key == 'summary':
            continue
            
        test_type = test_data.get('type', 'unknown')
        
        if test_type == 't-test':
            html += generate_ttest_html(test_key, test_data)
        elif test_type == 'chi-square':
            html += generate_chisquare_html(test_key, test_data)
        elif test_type == 'anova':
            html += generate_anova_html(test_key, test_data)
        elif test_type == 'correlation':
            html += generate_correlation_html(test_key, test_data)
        elif test_type == 'feature_importance':
            html += generate_feature_importance_html(test_key, test_data)
    
    html += """
        </div>
    """
    
    return html

# Helper functions for each test type
def generate_ttest_html(test_key, test_data):
    """Generate HTML for t-test results"""
    results = test_data.get('results', {})
    description = test_data.get('description', 'T-test Analysis')
    
    html = f"""
        <h3>{description}</h3>
        <p>
            Comparing means of <span class="highlight">{results.get('feature', '')}</span> 
            between groups of <span class="highlight">{results.get('target', '')}</span>.
        </p>
        
        <table>
            <tr>
                <th>Group</th>
                <th>Mean</th>
                <th>Standard Deviation</th>
                <th>Count</th>
            </tr>
            <tr>
                <td>{results.get('groups', ['Group 1', 'Group 2'])[0]}</td>
                <td>{results.get('group1_mean', 0):.4f}</td>
                <td>{results.get('group1_std', 0):.4f}</td>
                <td>{results.get('group1_n', 0)}</td>
            </tr>
            <tr>
                <td>{results.get('groups', ['Group 1', 'Group 2'])[1]}</td>
                <td>{results.get('group2_mean', 0):.4f}</td>
                <td>{results.get('group2_std', 0):.4f}</td>
                <td>{results.get('group2_n', 0)}</td>
            </tr>
        </table>
        
        <h4>Test Results</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Mean Difference</td>
                <td class="metric">{results.get('mean_difference', 0):.4f}</td>
            </tr>
            <tr>
                <td>t-statistic</td>
                <td>{results.get('t_statistic', 0):.4f}</td>
            </tr>
            <tr>
                <td>p-value</td>
                <td class="metric">{results.get('p_value', 0):.4f}</td>
            </tr>
            <tr>
                <td>Test Type</td>
                <td>{results.get('test_type', 'Unknown')}</td>
            </tr>
            <tr>
                <td>Significance</td>
                <td>{f'<span class="highlight">Significant</span>' if results.get('significant', False) else 'Not significant'}</td>
            </tr>
        </table>
        
        <p>
            <strong>Interpretation:</strong> 
            {f'There is a statistically significant difference in the means of {results.get("feature", "")} between the two groups (p < {results.get("alpha", 0.05)}).' 
            if results.get('significant', False) 
            else f'There is no statistically significant difference in the means of {results.get("feature", "")} between the two groups (p > {results.get("alpha", 0.05)}).'}
        </p>
    """
    
    return html

def generate_chisquare_html(test_key, test_data):
    """Generate HTML for chi-square test results"""
    results = test_data.get('results', {})
    description = test_data.get('description', 'Chi-Square Analysis')
    
    html = f"""
        <h3>{description}</h3>
        <p>
            Testing association between <span class="highlight">{results.get('feature', '')}</span> 
            and <span class="highlight">{results.get('target', '')}</span>.
        </p>
        
        <h4>Test Results</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Chi-square Statistic</td>
                <td class="metric">{results.get('chi2', 0):.4f}</td>
            </tr>
            <tr>
                <td>p-value</td>
                <td class="metric">{results.get('p_value', 0):.4f}</td>
            </tr>
            <tr>
                <td>Degrees of Freedom</td>
                <td>{results.get('dof', 0)}</td>
            </tr>
            <tr>
                <td>Cramer's V</td>
                <td class="metric">{results.get('cramers_v', 0):.4f}</td>
            </tr>
            <tr>
                <td>Effect Size</td>
                <td>{results.get('effect_size', 'Unknown')}</td>
            </tr>
            <tr>
                <td>Significance</td>
                <td>{f'<span class="highlight">Significant</span>' if results.get('significant', False) else 'Not significant'}</td>
            </tr>
        </table>
        
        <p>
            <strong>Interpretation:</strong> 
            {f'There is a statistically significant association between {results.get("feature", "")} and {results.get("target", "")} (p < {results.get("alpha", 0.05)}). The effect size is {results.get("effect_size", "unknown").lower()} (Cramer\'s V = {results.get("cramers_v", 0):.4f}).' 
            if results.get('significant', False) 
            else f'There is no statistically significant association between {results.get("feature", "")} and {results.get("target", "")} (p > {results.get("alpha", 0.05)}).'}
        </p>
    """
    
    return html

def generate_anova_html(test_key, test_data):
    """Generate HTML for ANOVA test results"""
    results = test_data.get('results', {})
    description = test_data.get('description', 'ANOVA Analysis')
    
    html = f"""
        <h3>{description}</h3>
        <p>
            Comparing means of <span class="highlight">{results.get('feature', '')}</span> 
            across groups of <span class="highlight">{results.get('group', '')}</span>.
        </p>
        
        <h4>Test Results</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>F-statistic</td>
                <td class="metric">{results.get('f_statistic', 0):.4f}</td>
            </tr>
            <tr>
                <td>p-value</td>
                <td class="metric">{results.get('p_value', 0):.4f}</td>
            </tr>
            <tr>
                <td>Degrees of Freedom (between, within)</td>
                <td>{results.get('df_between', 0)}, {results.get('df_within', 0)}</td>
            </tr>
            <tr>
                <td>Eta-squared</td>
                <td class="metric">{results.get('eta_squared', 0):.4f}</td>
            </tr>
            <tr>
                <td>Effect Size</td>
                <td>{results.get('effect_size', 'Unknown')}</td>
            </tr>
            <tr>
                <td>Significance</td>
                <td>{f'<span class="highlight">Significant</span>' if results.get('significant', False) else 'Not significant'}</td>
            </tr>
        </table>
        
        <p>
            <strong>Interpretation:</strong> 
            {f'There are statistically significant differences in the means of {results.get("feature", "")} across {results.get("group", "")} groups (p < {results.get("alpha", 0.05)}). The effect size is {results.get("effect_size", "unknown").lower()} (Eta-squared = {results.get("eta_squared", 0):.4f}).' 
            if results.get('significant', False) 
            else f'There are no statistically significant differences in the means of {results.get("feature", "")} across {results.get("group", "")} groups (p > {results.get("alpha", 0.05)}).'}
        </p>
    """
    
    # Add group statistics if available
    if 'group_stats' in results:
        group_stats = results['group_stats']
        
        html += """
            <h4>Group Statistics</h4>
            <table>
                <tr>
                    <th>Group</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Count</th>
                </tr>
        """
        
        for _, row in group_stats.iterrows():
            html += f"""
                <tr>
                    <td>{row[results.get('group', 'Group')]}</td>
                    <td>{row['mean']:.4f}</td>
                    <td>{row['std']:.4f}</td>
                    <td>{row['count']}</td>
                </tr>
            """
        
        html += """
            </table>
        """
    
    return html

def generate_correlation_html(test_key, test_data):
    """Generate HTML for correlation analysis results"""
    description = test_data.get('description', 'Correlation Analysis')
    
    html = f"""
        <h3>{description}</h3>
    """
    
    # If we have a summary, show it first
    if 'summary' in test_data:
        summary = test_data['summary']
        
        html += """
            <h4>Correlation Summary</h4>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Correlation</th>
                    <th>p-value</th>
                    <th>R-squared</th>
                    <th>Effect Size</th>
                    <th>Significance</th>
                </tr>
        """
        
        # Sort by absolute correlation value
        sorted_summary = summary.sort_values('correlation', key=abs, ascending=False)
        
        for _, row in sorted_summary.iterrows():
            html += f"""
                <tr>
                    <td>{row['feature']}</td>
                    <td class="metric">{row['correlation']:.4f}</td>
                    <td>{row['p_value']:.4f}</td>
                    <td>{row['r_squared']:.4f}</td>
                    <td>{row['effect_size']}</td>
                    <td>{'<span class="highlight">Significant</span>' if row['significant'] else 'Not significant'}</td>
                </tr>
            """
        
        html += """
            </table>
        """
    
    return html

def generate_feature_importance_html(test_key, test_data):
    """Generate HTML for feature importance analysis results"""
    description = test_data.get('description', 'Feature Importance Analysis')
    results = test_data.get('results', pd.DataFrame())
    
    html = f"""
        <h3>{description}</h3>
        
        <h4>Feature Importance Ranking</h4>
        <table>
            <tr>
                <th>Feature</th>
                <th>Importance</th>
            </tr>
    """
    
    # Sort by importance if it's a DataFrame
    if isinstance(results, pd.DataFrame):
        sorted_results = results.sort_values('Importance', ascending=False)
        
        for _, row in sorted_results.iterrows():
            html += f"""
                <tr>
                    <td>{row['Feature']}</td>
                    <td class="metric">{row['Importance']:.4f}</td>
                </tr>
            """
    
    html += """
        </table>
        
        <p>
            <strong>Interpretation:</strong> 
            The table above ranks features by their importance in predicting the target variable.
            Features with higher importance scores have a stronger influence on the predictions.
        </p>
    """
    
    return html

# Helper function to generate threshold analysis section
def generate_threshold_analysis_section(threshold_analysis):
    """Generate HTML for threshold analysis section"""
    
    if not threshold_analysis:
        return ""
    
    html = f"""
        <div class="section">
            <h2>Threshold Analysis</h2>
            <p>
                This section analyzes how different feature thresholds affect the target variable, 
                helping identify important decision boundaries in your data.
            </p>
    """
    
    # Process each threshold analysis
    for analysis_key, analysis_data in threshold_analysis.items():
        analysis_type = analysis_data.get('type', 'unknown')
        
        if analysis_type == 'single_feature':
            html += f"""
                <h3>Single Feature Threshold Analysis</h3>
                <p>
                    Analysis of how different thresholds for 
                    <span class="highlight">{analysis_data.get('feature', '')}</span> 
                    affect <span class="highlight">{analysis_data.get('target', '')}</span>.
                </p>
                <p>
                    <em>This analysis was performed on {analysis_data.get('timestamp', '')}.</em>
                </p>
            """
        
        elif analysis_type == 'feature_combination':
            features = analysis_data.get('features', [])
            html += f"""
                <h3>Feature Combination Analysis</h3>
                <p>
                    Analysis of how combinations of 
                    <span class="highlight">{features[0] if len(features) > 0 else ''}</span> and
                    <span class="highlight">{features[1] if len(features) > 1 else ''}</span> 
                    affect <span class="highlight">{analysis_data.get('target', '')}</span>.
                </p>
                <p>
                    <em>This analysis was performed on {analysis_data.get('timestamp', '')}.</em>
                </p>
            """
        
        elif analysis_type == 'custom_threshold':
            html += f"""
                <h3>Custom Threshold Analysis</h3>
                <p>
                    Analysis of how specific thresholds for 
                    <span class="highlight">{analysis_data.get('feature1', '')}</span> 
                    ({analysis_data.get('threshold1', '')}) and
                    <span class="highlight">{analysis_data.get('feature2', '')}</span> 
                    ({analysis_data.get('threshold2', '')}) affect 
                    <span class="highlight">{analysis_data.get('target', '')}</span>.
                </p>
            """
            
            # If we have results, add them
            if 'results' in analysis_data:
                results = analysis_data['results']
                
                html += """
                    <h4>Quadrant Analysis</h4>
                    <table>
                        <tr>
                            <th>Quadrant</th>
                            <th>Count</th>
                            <th>Target Rate</th>
                            <th>% of Total</th>
                        </tr>
                """
                
                for idx, row in results.iterrows():
                    html += f"""
                        <tr>
                            <td>{idx}</td>
                            <td>{row.get('count', 0)}</td>
                            <td class="metric">{row.get('target_1 rate', 0):.2%}</td>
                            <td>{row.get('percentage_of_total', 0):.2%}</td>
                        </tr>
                    """
                
                html += """
                    </table>
                """
            
            html += f"""
                <p>
                    <em>This analysis was performed on {analysis_data.get('timestamp', '')}.</em>
                </p>
            """
    
    html += """
        </div>
    """
    
    return html

# Helper function to generate model training section
def generate_model_training_section(model_training):
    """Generate HTML for model training section"""
    
    if not model_training:
        return ""
    
    models_trained = model_training.get('models_trained', [])
    evaluation_results = model_training.get('evaluation_results', pd.DataFrame())
    feature_importance = model_training.get('feature_importance', None)
    
    html = f"""
        <div class="section">
            <h2>Model Training</h2>
            
            <h3>Models Trained</h3>
            <p>
                The following models were trained:
                <ul>
    """
    
    for model in models_trained:
        html += f"""
                    <li>{model}</li>
        """
    
    html += """
                </ul>
            </p>
            
            <h3>Model Performance</h3>
    """
    
    # Add evaluation results if available
    if isinstance(evaluation_results, pd.DataFrame) and not evaluation_results.empty:
        html += """
            <table>
                <tr>
                    <th>Model</th>
        """
        
        # Add column headers
        for col in evaluation_results.columns:
            html += f"""
                    <th>{col}</th>
            """
        
        html += """
                </tr>
        """
        
        # Add rows for each model
        for idx, row in evaluation_results.iterrows():
            html += f"""
                <tr>
                    <td>{idx}</td>
            """
            
            for col in evaluation_results.columns:
                # Format metric value
                value = row[col]
                if isinstance(value, (int, float)):
                    html += f"""
                        <td class="metric">{value:.4f}</td>
                    """
                else:
                    html += f"""
                        <td>{value}</td>
                    """
            
            html += """
                </tr>
            """
        
        html += """
            </table>
        """
    
    # Add feature importance if available
    if feature_importance is not None:
        html += """
            <h3>Feature Importance</h3>
        """
        
        if isinstance(feature_importance, dict):
            # If it's a dictionary of feature importances by model
            for model_name, importance_df in feature_importance.items():
                html += f"""
                    <h4>Feature Importance from {model_name}</h4>
                """
                
                if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
                    # Sort by importance
                    sorted_importance = importance_df.sort_values('importance', ascending=False)
                    
                    html += """
                        <table>
                            <tr>
                                <th>Feature</th>
                                <th>Importance</th>
                            </tr>
                    """
                    
                    for _, row in sorted_importance.iterrows():
                        html += f"""
                            <tr>
                                <td>{row.get('feature', '')}</td>
                                <td class="metric">{row.get('importance', 0):.4f}</td>
                            </tr>
                        """
                    
                    html += """
                        </table>
                    """
        
        elif isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            # Sort by importance
            sorted_importance = feature_importance.sort_values('importance', ascending=False)
            
            html += """
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
            """
            
            for _, row in sorted_importance.iterrows():
                html += f"""
                    <tr>
                        <td>{row.get('feature', '')}</td>
                        <td class="metric">{row.get('importance', 0):.4f}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
    
    html += f"""
            <p>
                <em>Models were trained on {model_training.get('timestamp', '')}.</em>
            </p>
        </div>
    """
    
    return html

# Helper function to generate model evaluation section
def generate_model_evaluation_section(model_evaluation):
    """Generate HTML for model evaluation section"""
    
    if not model_evaluation:
        return ""
    
    models_evaluated = model_evaluation.get('models_evaluated', [])
    cv_results = model_evaluation.get('cross_validation_results', {})
    
    html = f"""
        <div class="section">
            <h2>Model Evaluation</h2>
            
            <h3>Models Evaluated</h3>
            <p>
                The following models were evaluated:
                <ul>
    """
    
    for model in models_evaluated:
        html += f"""
                    <li>{model}</li>
        """
    
    html += """
                </ul>
            </p>
            
            <h3>Cross-Validation Results</h3>
    """
    
    # Add cross-validation results if available
    if cv_results:
        html += """
            <table>
                <tr>
                    <th>Model</th>
        """
        
        # Get column names from first model
        first_model = list(cv_results.keys())[0] if cv_results else None
        if first_model:
            columns = list(cv_results[first_model].keys())
            
            # Add column headers
            for col in columns:
                html += f"""
                    <th>{col}</th>
                """
        
            html += """
                </tr>
            """
            
            # Add rows for each model
            for model, results in cv_results.items():
                html += f"""
                    <tr>
                        <td>{model}</td>
                """
                
                for col in columns:
                    html += f"""
                        <td class="metric">{results.get(col, '')}</td>
                    """
                
                html += """
                    </tr>
                """
        
        html += """
            </table>
        """
    
    html += f"""
            <p>
                <em>Evaluation was performed on {model_evaluation.get('timestamp', '')}.</em>
            </p>
        </div>
    """
    
    return html

# Helper function to generate predictions section
def generate_predictions_section(report_data):
    """Generate HTML for predictions section"""
    
    html = f"""
        <div class="section">
            <h2>Predictions</h2>
    """
    
    # Add single predictions if available
    if 'predictions' in report_data and report_data['predictions']:
        html += """
            <h3>Individual Predictions</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Prediction</th>
                    <th>Timestamp</th>
                </tr>
        """
        
        for prediction in report_data['predictions']:
            html += f"""
                <tr>
                    <td>{prediction.get('model', '')}</td>
                    <td class="metric">{prediction.get('prediction', '')}</td>
                    <td>{prediction.get('timestamp', '')}</td>
                </tr>
            """
        
        html += """
            </table>
        """
    
    # Add batch predictions if available
    if 'batch_predictions' in report_data and report_data['batch_predictions']:
        html += """
            <h3>Batch Predictions</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Records Processed</th>
                    <th>Timestamp</th>
                </tr>
        """
        
        for batch in report_data['batch_predictions']:
            html += f"""
                <tr>
                    <td>{batch.get('model', '')}</td>
                    <td>{batch.get('records_processed', 0)}</td>
                    <td>{batch.get('timestamp', '')}</td>
                </tr>
            """
        
        html += """
            </table>
        """
    
    html += """
        </div>
    """
    
    return html

# Run the main app
if __name__ == "__main__":
    main()