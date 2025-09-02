# streamlit_app/pages/predictions.py
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

from edu_analytics.utils import interpret_prediction, preprocess_input_data

def show_predictions():
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    
    # Check if data is processed
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("Please process your data first.")
        return
    
    # Check if models are trained
    if 'models' not in st.session_state or not st.session_state.models:
        st.warning("Please train models first.")
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
                input_values[feature] = st.date_input(f"{feature}", value=pd.Timestamp.now(), key=f"pred_{feature}")
            else:
                input_values[feature] = st.text_input(f"{feature}", value="", key=f"pred_{feature}")
        
        # Prediction button
        if st.button("Make Prediction", key="make_pred"):
            try:
                with st.spinner("Making prediction..."):
                    # Get selected model
                    model = st.session_state.models[selected_model]
                    
                    # Preprocess input data using the utility function
                    input_processed = preprocess_input_data(input_values, data_types)
                    
                    # Create input DataFrame with proper column names
                    input_df = pd.DataFrame([input_processed], columns=selected_features)
                    
                    # Apply categorical encoding
                    if hasattr(st.session_state, 'categorical_encoders') and st.session_state.categorical_encoders:
                        for feature, encoder in st.session_state.categorical_encoders.items():
                            if feature in input_df.columns:
                                try:
                                    # Convert to string to ensure compatibility
                                    input_df[feature] = encoder.transform(input_df[feature].astype(str))
                                except Exception as e:
                                    st.error(f"Error encoding {feature}: {str(e)}")
                                    st.info(f"Valid values for {feature} are: {', '.join(encoder.classes_)}")
                                    return
                    
                    # Make sure all values are numeric before scaling
                    for col in input_df.columns:
                        if not pd.api.types.is_numeric_dtype(input_df[col]):
                            st.error(f"Column {col} contains non-numeric values: {input_df[col].iloc[0]}")
                            st.info(f"This might happen if categorical encoding failed. Check your input values.")
                            return
                    
                    # Scale the input data if a scaler is available
                    if hasattr(st.session_state, 'scaler') and st.session_state.scaler is not None:
                        scaler = st.session_state.scaler
                        try:
                            input_scaled = scaler.transform(input_df)
                        except Exception as e:
                            st.error(f"Error during scaling: {str(e)}")
                            st.info("Please ensure all input values match the expected data types.")
                            return
                    else:
                        input_scaled = input_df.values
                    
                    # Make prediction
                    if target_type == 'categorical':
                        prediction = model.predict(input_scaled)[0]
                        
                        # Map prediction back to original value if we have a mapping
                        prediction_label = interpret_prediction(prediction, target_type, target_mapping)
                        
                        # Get probabilities if available
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(input_scaled)[0]
                            
                            # Display prediction
                            st.success(f"Prediction: {prediction_label}")
                            
                            # Display probabilities
                            st.markdown("### Prediction Probabilities")
                            
                            prob_data = []
                            for i, prob in enumerate(probabilities):
                                class_label = interpret_prediction(i, target_type, target_mapping)
                                
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
                        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
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
                                    from edu_analytics.time_analysis import convert_time_to_minutes
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
                                prediction_labels = [interpret_prediction(p, target_type, target_mapping) for p in predictions]
                                
                                # Get probabilities if available
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(X_batch_scaled)
                                    
                                    # Create probability columns
                                    for i in range(probabilities.shape[1]):
                                        class_label = interpret_prediction(i, target_type, target_mapping)
                                        batch_df[f'Prob_{class_label}'] = probabilities[:, i]
                                
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
                                file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
                                    prob_cols = [c for c in batch_df.columns if c.startswith('Prob_')]
                                    if prob_cols:
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
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                    except Exception as e:
                        st.error(f"Error making batch predictions: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"Error loading batch data: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    show_predictions()