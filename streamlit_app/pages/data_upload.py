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

def show_data_upload():
    # Import all required functions at the beginning of the function
    from edu_analytics.data_processing import infer_and_validate_data_type, prepare_data
    
    st.markdown("<div class='subheader'>Upload Your Data</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>Upload a CSV file containing your data for analysis.</div>", unsafe_allow_html=True)
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
                        X, y, categorical_encoders, target_type, target_mapping, scaler, original_target, categorical_mappings = prepare_data(
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
                        
                        # Initialize report_data if it doesn't exist
                        if 'report_data' not in st.session_state:
                            st.session_state.report_data = {
                                'statistical_tests': {},
                                'threshold_analysis': {},
                                'model_training': {},
                                'model_evaluation': {},
                                'predictions': [],
                                'batch_predictions': []
                            }
                        
                        # Show success message
                        st.success(f"Data processed successfully! Target '{target_column}' detected as {target_type} type.")
                        
                        # Navigate to data exploration
                        st.session_state.current_section = "data_exploration"
                        st.rerun()  # Changed from st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        st.code(traceback.format_exc())
                        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.code(traceback.format_exc())
            
    else:
        # Show sample data option when no file is uploaded
        st.markdown("### Or Use Sample Data")
        sample_data_option = st.selectbox(
            "Select a sample dataset",
            ["None", "Student Performance", "Employee Attrition", "Housing Prices"]
        )
        
        # When loading sample data, don't automatically process it, just load it into session state
        if sample_data_option != "None" and st.button("Load Sample Data"):
            try:
                # Generate sample data directly without trying to load files
                df = None
                
                if sample_data_option == "Student Performance":
                    # Create student performance demo data
                    df = pd.DataFrame({
                        'student_id': range(1, 101),
                        'hours_studied': np.random.randint(1, 10, 100),
                        'attendance_pct': np.random.randint(60, 100, 100),
                        'previous_gpa': np.random.uniform(2.0, 4.0, 100).round(2),
                        'final_grade': np.random.randint(50, 100, 100)
                    })
                    suggested_target = "final_grade"
                    
                elif sample_data_option == "Employee Attrition":
                    # Create employee attrition demo data
                    df = pd.DataFrame({
                        'employee_id': range(1, 101),
                        'age': np.random.randint(22, 60, 100),
                        'salary': np.random.randint(30000, 120000, 100),
                        'years_at_company': np.random.randint(0, 20, 100),
                        'satisfaction_score': np.random.randint(1, 10, 100),
                        'left_company': np.random.choice(['Yes', 'No'], 100, p=[0.3, 0.7])
                    })
                    suggested_target = "left_company"
                    
                elif sample_data_option == "Housing Prices":
                    # Create housing prices demo data
                    df = pd.DataFrame({
                        'house_id': range(1, 101),
                        'square_feet': np.random.randint(800, 4000, 100),
                        'bedrooms': np.random.randint(1, 6, 100),
                        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], 100),
                        'age_years': np.random.randint(0, 50, 100),
                        'price': np.random.randint(100000, 1000000, 100)
                    })
                    suggested_target = "price"
                    
                else:
                    st.error(f"Unknown sample data option: {sample_data_option}")
                    return
                
                # Store in session state
                st.session_state.data = df
                
                # Store suggested target in session state for persistence
                st.session_state.suggested_target = suggested_target
                
                # Analyze data types for all columns
                data_types = {}
                for column in df.columns:
                    data_types[column] = infer_and_validate_data_type(df[column])
                    
                # Store data types in session state
                st.session_state.data_types = data_types
                
                # Display success message and suggestion
                st.success(f"Sample {sample_data_option} data loaded successfully!")
                st.info(f"Suggested target variable for this dataset: '{suggested_target}'")
                
                # Show data preview
                st.markdown("<div class='subheader'>Data Preview</div>", unsafe_allow_html=True)
                st.dataframe(df.head(10))
                
                # Force a rerun to ensure the selectbox is displayed with the new data
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating sample data: {str(e)}")
                st.code(traceback.format_exc())
        
        # Add this section outside the if condition to ensure it's always displayed when data exists
        if 'data' in st.session_state and st.session_state.data is not None:
            df = st.session_state.data
            suggested_target = st.session_state.get('suggested_target', df.columns[-1])
            
            # Target selection - use the suggested target as the default
            st.markdown("<div class='subheader'>Select Target Variable</div>", unsafe_allow_html=True)
            target_column = st.selectbox(
                "Choose your target variable", 
                df.columns, 
                index=df.columns.get_loc(suggested_target) if suggested_target in df.columns else 0
            )
            
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
            
            # Process data button - use a consistent key across all paths
            if st.button("Process Data", key="process_all_data"):
                with st.spinner("Processing data..."):
                    try:
                        data_types = st.session_state.data_types
                        
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
                        st.session_state.categorical_mappings = categorical_mappings 
                        
                        # Initialize report_data if it doesn't exist
                        if 'report_data' not in st.session_state:
                            st.session_state.report_data = {
                                'statistical_tests': {},
                                'threshold_analysis': {},
                                'model_training': {},
                                'model_evaluation': {},
                                'predictions': [],
                                'batch_predictions': []
                            }
                        
                        # Show success message
                        st.success(f"Data processed successfully! Target '{target_column}' detected as {target_type} type.")
                        
                        # Navigate to data exploration
                        st.session_state.current_section = "data_exploration"
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    show_data_upload()