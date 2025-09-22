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
    target_mapping = st.session_state.target_mapping if 'target_mapping' in st.session_state else None
    data_types = st.session_state.data_types
    
    # Create tabs for different prediction options
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Predictions"])
    
    with tab1:
        st.markdown("<div class='subheader'>Make a Single Prediction</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-text'>Enter values for each feature to get a prediction.</div>", unsafe_allow_html=True)
        
        # Select model for prediction
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select model for prediction", model_names)
        
        # Get selected model
        model = st.session_state.models[selected_model]
        
        # Add threshold adjustment for classification models
        threshold = 0.5  # Default threshold
        if target_type == 'categorical':
            # Check if it's a binary classification problem
            is_binary = False
            if target_mapping and len(target_mapping) == 2:
                is_binary = True
            elif hasattr(st.session_state, 'processed_data') and 'original_target' in st.session_state.processed_data:
                # Check from original target data
                unique_values = st.session_state.processed_data['original_target'].nunique()
                is_binary = unique_values == 2
                
            # Show threshold adjustment for binary classification with probability support
            if is_binary and hasattr(model, 'predict_proba'):
                # Get the target class names for better UI
                class_names = ["Negative", "Positive"]  # Default names
                if target_mapping:
                    # Extract class names from mapping (sorted by encoded value)
                    reverse_mapping = {v: k for k, v in target_mapping.items()}
                    class_names = [reverse_mapping.get(0, "Negative"), 
                                  reverse_mapping.get(1, "Positive")]
                
                st.write("#### Prediction Threshold")
                st.write(f"Adjust the threshold to balance between predicting '{class_names[0]}' and '{class_names[1]}'.")
                st.write("Lower values increase the likelihood of predicting the positive class.")
                
                # Show threshold slider with the positive class name
                threshold = st.slider(
                    f"Probability threshold for '{class_names[1]}'",
                    min_value=0.01,
                    max_value=0.99,
                    value=0.5,
                    step=0.01,
                    help=f"Lower values will increase the number of '{class_names[1]}' predictions. Higher values will increase '{class_names[0]}' predictions."
                )
                
                # Add explanation about threshold adjustment
                if threshold != 0.5:
                    if threshold < 0.5:
                        st.info(f"Using a threshold of {threshold:.2f} means the model will be more likely to predict '{class_names[1]}'. This is useful when false negatives are more costly than false positives.")
                    else:
                        st.info(f"Using a threshold of {threshold:.2f} means the model will be more likely to predict '{class_names[0]}'. This is useful when false positives are more costly than false negatives.")
        
        # Create input widgets for features
        st.write("#### Input Features")
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
                        # For binary classification with probability support, apply custom threshold
                        if hasattr(model, 'predict_proba') and threshold != 0.5:
                            probabilities = model.predict_proba(input_scaled)[0]
                            prediction = 1 if probabilities[1] > threshold else 0
                        else:
                            prediction = model.predict(input_scaled)[0]
                        
                        # Map prediction back to original value if we have a mapping
                        prediction_label = interpret_prediction(prediction, target_type, target_mapping)
                        
                        # Create a results container with styled output
                        st.markdown("### Prediction Result")
                        
                        # Get probabilities if available
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(input_scaled)[0]
                            
                            # Create a better probability display
                            st.markdown("#### Prediction Probabilities")
                            
                            # Display the prediction with confidence
                            if len(probabilities) == 2:  # Binary classification
                                # Get the positive class probability
                                pos_prob = probabilities[1]
                                
                                # Create a metric with the prediction and confidence
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        label="Predicted Class",
                                        value=prediction_label
                                    )
                                with col2:
                                    st.metric(
                                        label="Confidence",
                                        value=f"{max(probabilities):.2%}"
                                    )
                                
                                # Create a gauge-like progress bar for the positive class probability
                                st.markdown("#### Probability Distribution")
                                st.progress(pos_prob)
                                st.write(f"Probability of '{prediction_label}': **{pos_prob:.2%}**")
                                
                                # Add threshold information
                                if threshold != 0.5:
                                    st.write(f"*Using custom threshold: {threshold:.2f}*")
                            else:  # Multi-class classification
                                # Create a DataFrame for all probabilities
                                prob_data = []
                                for i, prob in enumerate(probabilities):
                                    # Convert encoded class to original category name
                                    if target_mapping:
                                        reverse_mapping = {v: k for k, v in target_mapping.items()}
                                        class_label = reverse_mapping.get(i, str(i))
                                    else:
                                        class_label = str(i)
                                    prob_data.append({
                                        'Class': class_label,
                                        'Probability': prob
                                    })
                                
                                # Display the prediction with confidence
                                st.metric(
                                    label="Predicted Class",
                                    value=prediction_label,
                                    delta=f"Confidence: {max(probabilities):.2%}"
                                )
                                
                                # Create a probability DataFrame and sort by probability
                                prob_df = pd.DataFrame(prob_data)
                                prob_df = prob_df.sort_values('Probability', ascending=False)
                                
                                # Display as a dataframe
                                st.dataframe(prob_df)
                                
                                # Bar chart of probabilities
                                fig, ax = plt.subplots(figsize=(10, 5))
                                prob_df.plot(
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
                    
                    # Determine what to store based on prediction type
                    prediction_to_store = prediction_label if target_type == 'categorical' else \
                                         prediction_formatted if target_type == 'time' else \
                                         prediction
                    
                    # Store additional details for categorical predictions
                    prediction_details = {}
                    if target_type == 'categorical' and hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(input_scaled)[0]
                        prediction_details['probabilities'] = probabilities.tolist()
                        prediction_details['threshold'] = threshold
                    
                    st.session_state.report_data['predictions'].append({
                        'model': selected_model,
                        'input': input_processed,
                        'prediction': prediction_to_store,
                        'details': prediction_details,
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
        
        # Get selected model
        model = st.session_state.models[selected_model]
        
        # Add threshold adjustment for batch predictions (only for binary classification)
        batch_threshold = 0.5  # Default threshold
        if target_type == 'categorical':
            # Check if it's a binary classification problem
            is_binary = False
            if target_mapping and len(target_mapping) == 2:
                is_binary = True
            elif hasattr(st.session_state, 'processed_data') and 'original_target' in st.session_state.processed_data:
                # Check from original target data
                unique_values = st.session_state.processed_data['original_target'].nunique()
                is_binary = unique_values == 2
                
            # Show threshold adjustment for binary classification with probability support
            if is_binary and hasattr(model, 'predict_proba'):
                # Get the target class names for better UI
                class_names = ["Negative", "Positive"]  # Default names
                if target_mapping:
                    # Extract class names from mapping (sorted by encoded value)
                    reverse_mapping = {v: k for k, v in target_mapping.items()}
                    class_names = [reverse_mapping.get(0, "Negative"), 
                                  reverse_mapping.get(1, "Positive")]
                
                st.write("#### Prediction Threshold")
                st.write(f"Adjust the threshold to balance between predicting '{class_names[0]}' and '{class_names[1]}'.")
                
                # Show threshold slider with the positive class name
                batch_threshold = st.slider(
                    f"Probability threshold for '{class_names[1]}'",
                    min_value=0.01,
                    max_value=0.99,
                    value=0.5,
                    step=0.01,
                    key="batch_threshold",
                    help=f"Lower values will increase the number of '{class_names[1]}' predictions."
                )
                
                # Add explanation about threshold adjustment
                if batch_threshold != 0.5:
                    if batch_threshold < 0.5:
                        st.info(f"Using a threshold of {batch_threshold:.2f} means the model will be more likely to predict '{class_names[1]}'. This is useful for imbalanced data where the positive class is rare.")
                    else:
                        st.info(f"Using a threshold of {batch_threshold:.2f} means the model will be more likely to predict '{class_names[0]}'. This reduces false positives.")
        
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
                
                # Add option to view target distribution if target column exists in batch data
                if target_column in batch_df.columns:
                    if st.checkbox("Show target distribution in uploaded data", value=False):
                        st.write("#### Target Distribution in Uploaded Data")
                        
                        if batch_df[target_column].dtype == 'object' or batch_df[target_column].nunique() < 10:
                            # Categorical target - show value counts
                            target_counts = batch_df[target_column].value_counts()
                            
                            # Calculate percentages
                            target_percentages = batch_df[target_column].value_counts(normalize=True) * 100
                            
                            # Create a DataFrame with counts and percentages
                            target_stats = pd.DataFrame({
                                'Count': target_counts,
                                'Percentage': target_percentages
                            })
                            
                            # Display as table
                            st.dataframe(target_stats)
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.bar(target_stats.index, target_stats['Percentage'])
                            ax.set_ylabel('Percentage (%)')
                            ax.set_title(f'Distribution of {target_column}')
                            # Rotate x labels if there are many categories
                            if len(target_stats) > 3:
                                plt.xticks(rotation=45, ha='right')
                            # Add percentage labels on bars
                            for i, v in enumerate(target_stats['Percentage']):
                                ax.text(i, v + 1, f"{v:.1f}%", ha='center')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show imbalance warning if appropriate
                            if len(target_stats) >= 2:
                                min_pct = target_stats['Percentage'].min()
                                if min_pct < 10:  # Less than 10% is minority class
                                    st.warning(f"Your data is imbalanced. The minority class represents only {min_pct:.1f}% of the data.")
                                    st.info(f"Using a threshold adjustment (currently {batch_threshold}) can help improve predictions for the minority class.")
                        else:
                            # Numeric target - show histogram
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.hist(batch_df[target_column], bins=20)
                            ax.set_xlabel(target_column)
                            ax.set_ylabel('Frequency')
                            ax.set_title(f'Distribution of {target_column}')
                            st.pyplot(fig)
                
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
                                        except Exception as e:
                                            st.error(f"Invalid values for {feature}. Must be one of: {', '.join(encoder.classes_)}")
                                            st.write(f"Error details: {str(e)}")
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
                                # For classification with custom threshold
                                if hasattr(model, 'predict_proba') and batch_threshold != 0.5:
                                    probabilities = model.predict_proba(X_batch_scaled)
                                    # Apply custom threshold for binary classification
                                    if probabilities.shape[1] == 2:
                                        predictions = (probabilities[:, 1] > batch_threshold).astype(int)
                                    else:
                                        # For multiclass, just take the highest probability class
                                        predictions = np.argmax(probabilities, axis=1)
                                else:
                                    # Standard prediction
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
                                
                                # Add a confidence column (max probability)
                                if hasattr(model, 'predict_proba'):
                                    batch_df['Confidence'] = np.max(probabilities, axis=1)
                                
                                # Compare with actual values if target column exists in the data
                                if target_column in batch_df.columns:
                                    # Add a column to indicate if prediction matches actual
                                    batch_df['Correct'] = (batch_df['Prediction'] == batch_df[target_column])
                                    
                                    # Calculate and display accuracy metrics
                                    accuracy = batch_df['Correct'].mean()
                                    st.metric("Prediction Accuracy", f"{accuracy:.2%}")
                                    
                                    # Calculate class-specific metrics for imbalanced data
                                    if target_type == 'categorical':
                                        from sklearn.metrics import classification_report
                                        
                                        # Convert target to the same format as predictions if needed
                                        if hasattr(st.session_state, 'categorical_encoders') and target_column in st.session_state.categorical_encoders:
                                            target_encoder = st.session_state.categorical_encoders[target_column]
                                            y_true = target_encoder.transform(batch_df[target_column].astype(str))
                                        else:
                                            y_true = batch_df[target_column]
                                        
                                        # Generate classification report
                                        report = classification_report(y_true, predictions, output_dict=True)
                                        report_df = pd.DataFrame(report).transpose()
                                        
                                        st.write("#### Classification Report")
                                        st.dataframe(report_df)
                                        
                                        # Highlight metrics important for imbalanced data
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            if 'macro avg' in report:
                                                st.metric("Macro Avg Precision", f"{report['macro avg']['precision']:.2f}")
                                        with col2:
                                            if 'macro avg' in report:
                                                st.metric("Macro Avg Recall", f"{report['macro avg']['recall']:.2f}")
                                        with col3:
                                            if 'macro avg' in report:
                                                st.metric("Macro Avg F1", f"{report['macro avg']['f1-score']:.2f}")
                            
                            else:  # Numeric or time prediction
                                predictions = model.predict(X_batch_scaled)
                                if target_type == 'time':
                                    # Add both formatted time and minutes
                                    batch_df['Prediction_Minutes'] = predictions
                                    batch_df['Prediction'] = [interpret_prediction(p, 'time') for p in predictions]
                                else:
                                    batch_df['Prediction'] = predictions
                                
                                # Compare with actual values if target column exists in the data
                                if target_column in batch_df.columns:
                                    # Calculate error metrics
                                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                                    
                                    # For time targets, convert actual values to minutes
                                    if target_type == 'time':
                                        from edu_analytics.time_analysis import convert_time_to_minutes
                                        actual_values = batch_df[target_column].apply(convert_time_to_minutes)
                                        batch_df['Error_Minutes'] = batch_df['Prediction_Minutes'] - actual_values
                                        mse = mean_squared_error(actual_values, batch_df['Prediction_Minutes'])
                                        mae = mean_absolute_error(actual_values, batch_df['Prediction_Minutes'])
                                        r2 = r2_score(actual_values, batch_df['Prediction_Minutes'])
                                    else:
                                        # For numeric targets
                                        batch_df['Error'] = batch_df['Prediction'] - batch_df[target_column]
                                        mse = mean_squared_error(batch_df[target_column], batch_df['Prediction'])
                                        mae = mean_absolute_error(batch_df[target_column], batch_df['Prediction'])
                                        r2 = r2_score(batch_df[target_column], batch_df['Prediction'])
                                    
                                    # Display error metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Mean Squared Error", f"{mse:.4f}")
                                    with col2:
                                        st.metric("Mean Absolute Error", f"{mae:.4f}")
                                    with col3:
                                        st.metric("R² Score", f"{r2:.4f}")
                            
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
                                st.write("#### Prediction Distribution")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                prediction_counts = batch_df['Prediction'].value_counts(normalize=True) * 100
                                prediction_counts.plot(kind='bar', ax=ax)
                                plt.title('Distribution of Predictions')
                                plt.ylabel('Percentage (%)')
                                plt.xticks(rotation=45, ha='right')
                                # Add percentage labels
                                for i, v in enumerate(prediction_counts):
                                    ax.text(i, v + 1, f"{v:.1f}%", ha='center')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # If we have probabilities, show distribution of confidence
                                if 'Confidence' in batch_df.columns:
                                    st.write("#### Prediction Confidence Distribution")
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    plt.hist(batch_df['Confidence'], bins=20)
                                    plt.title('Distribution of Prediction Confidence')
                                    plt.xlabel('Confidence')
                                    plt.ylabel('Count')
                                    plt.axvline(x=batch_threshold, color='r', linestyle='--', 
                                               label=f'Threshold: {batch_threshold}')
                                    plt.legend()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Show confidence by prediction class
                                    st.write("#### Confidence by Predicted Class")
                                    
                                    # Group by prediction and calculate mean confidence
                                    confidence_by_class = batch_df.groupby('Prediction')['Confidence'].agg(['mean', 'std', 'count']).reset_index()
                                    
                                    # Create a more attractive display
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sns.barplot(x='Prediction', y='mean', data=confidence_by_class, ax=ax)
                                    
                                    # Add error bars for standard deviation
                                    for i, row in confidence_by_class.iterrows():
                                        ax.errorbar(i, row['mean'], yerr=row['std'], fmt='o', color='black')
                                    
                                    # Add count annotations
                                    for i, row in confidence_by_class.iterrows():
                                        ax.text(i, row['mean'] + 0.03, f"n={row['count']}", ha='center')
                                    
                                    plt.title('Average Confidence by Predicted Class')
                                    plt.ylabel('Confidence')
                                    plt.ylim(0, 1.1)  # Ensure enough room for annotations
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # Compare with actual values if available
                                if target_column in batch_df.columns:
                                    st.write("#### Prediction vs Actual")
                                    
                                    # Create a confusion matrix
                                    from sklearn.metrics import confusion_matrix
                                    
                                    # Get unique classes (combining both predicted and actual)
                                    all_classes = sorted(set(batch_df['Prediction'].unique()) | set(batch_df[target_column].unique()))
                                    
                                    # Calculate confusion matrix
                                    cm = confusion_matrix(
                                        batch_df[target_column], 
                                        batch_df['Prediction'],
                                        labels=all_classes
                                    )
                                    
                                    # Create a heatmap visualization
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                               xticklabels=all_classes, 
                                               yticklabels=all_classes)
                                    plt.title('Confusion Matrix')
                                    plt.ylabel('True Label')
                                    plt.xlabel('Predicted Label')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # ROC curve for binary classification
                                    if len(all_classes) == 2 and 'Prob_' in batch_df.columns.str.join(''):
                                        try:
                                            st.write("#### ROC Curve")
                                            from sklearn.metrics import roc_curve, auc
                                            
                                            # Find the positive class name (assuming second class is positive)
                                            positive_class = all_classes[1] if len(all_classes) > 1 else all_classes[0]
                                            
                                            # Get the probability column for the positive class
                                            prob_col = f'Prob_{positive_class}'
                                            if prob_col not in batch_df.columns:
                                                # Try to find any probability column
                                                prob_cols = [col for col in batch_df.columns if col.startswith('Prob_')]
                                                if prob_cols:
                                                    prob_col = prob_cols[0]
                                            
                                            # Create binary target (1 for positive class, 0 for others)
                                            y_true_binary = (batch_df[target_column] == positive_class).astype(int)
                                            
                                            # Calculate ROC curve
                                            fpr, tpr, thresholds = roc_curve(y_true_binary, batch_df[prob_col])
                                            roc_auc = auc(fpr, tpr)
                                            
                                            # Plot ROC curve
                                            fig, ax = plt.subplots(figsize=(8, 8))
                                            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                                                    label=f'ROC curve (area = {roc_auc:.2f})')
                                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                            
                                            # Mark the current threshold
                                            current_idx = (np.abs(thresholds - batch_threshold)).argmin()
                                            plt.plot(fpr[current_idx], tpr[current_idx], 'ro', 
                                                    label=f'Current threshold: {batch_threshold:.2f}')
                                            
                                            plt.xlim([0.0, 1.0])
                                            plt.ylim([0.0, 1.05])
                                            plt.xlabel('False Positive Rate')
                                            plt.ylabel('True Positive Rate')
                                            plt.title('Receiver Operating Characteristic')
                                            plt.legend(loc="lower right")
                                            st.pyplot(fig)
                                            
                                            # Add precision-recall curve (better for imbalanced data)
                                            st.write("#### Precision-Recall Curve")
                                            from sklearn.metrics import precision_recall_curve, average_precision_score
                                            
                                            # Calculate precision-recall curve
                                            precision, recall, pr_thresholds = precision_recall_curve(
                                                y_true_binary, batch_df[prob_col])
                                            avg_precision = average_precision_score(y_true_binary, batch_df[prob_col])
                                            
                                            # Plot precision-recall curve
                                            fig, ax = plt.subplots(figsize=(8, 8))
                                            plt.plot(recall, precision, color='blue', lw=2, 
                                                    label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
                                            
                                            # Mark the current threshold on PR curve
                                            # Find the closest threshold value
                                            if len(pr_thresholds) > 0:
                                                current_pr_idx = min(range(len(pr_thresholds)), 
                                                                   key=lambda i: abs(pr_thresholds[i] - batch_threshold))
                                                if current_pr_idx < len(precision) and current_pr_idx < len(recall):
                                                    plt.plot(recall[current_pr_idx], precision[current_pr_idx], 'ro',
                                                            label=f'Current threshold: {batch_threshold:.2f}')
                                            
                                            # Add baseline for reference (the frequency of positive class)
                                            baseline = y_true_binary.mean()
                                            plt.plot([0, 1], [baseline, baseline], color='navy', 
                                                    linestyle='--', label=f'Baseline (positive rate: {baseline:.2f})')
                                            
                                            plt.xlim([0.0, 1.0])
                                            plt.ylim([0.0, 1.05])
                                            plt.xlabel('Recall')
                                            plt.ylabel('Precision')
                                            plt.title('Precision-Recall Curve (Better for Imbalanced Data)')
                                            plt.legend(loc="lower left")
                                            st.pyplot(fig)
                                            
                                            # Provide threshold optimization guidance
                                            st.write("#### Threshold Optimization")
                                            
                                            # Create a dataframe with different threshold values
                                            import numpy as np
                                            threshold_values = np.linspace(0.1, 0.9, 9)
                                            threshold_results = []
                                            
                                            for thresh in threshold_values:
                                                # Apply threshold
                                                pred_at_thresh = (batch_df[prob_col] >= thresh).astype(int)
                                                
                                                # Calculate metrics
                                                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                                                
                                                try:
                                                    precision = precision_score(y_true_binary, pred_at_thresh)
                                                    recall = recall_score(y_true_binary, pred_at_thresh)
                                                    f1 = f1_score(y_true_binary, pred_at_thresh)
                                                    accuracy = accuracy_score(y_true_binary, pred_at_thresh)
                                                    
                                                    threshold_results.append({
                                                        'Threshold': thresh,
                                                        'Precision': precision,
                                                        'Recall': recall,
                                                        'F1 Score': f1,
                                                        'Accuracy': accuracy
                                                    })
                                                except:
                                                    continue
                                            
                                            # Convert to DataFrame
                                            if threshold_results:
                                                thresh_df = pd.DataFrame(threshold_results)
                                                
                                                # Display the table
                                                st.dataframe(thresh_df.style.highlight_max(subset=['F1 Score']))
                                                
                                                # Plot metrics across thresholds
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                plt.plot(thresh_df['Threshold'], thresh_df['Precision'], 
                                                        label='Precision', marker='o')
                                                plt.plot(thresh_df['Threshold'], thresh_df['Recall'], 
                                                        label='Recall', marker='s')
                                                plt.plot(thresh_df['Threshold'], thresh_df['F1 Score'], 
                                                        label='F1 Score', marker='^')
                                                plt.plot(thresh_df['Threshold'], thresh_df['Accuracy'], 
                                                        label='Accuracy', marker='x')
                                                
                                                # Add vertical line for current threshold
                                                plt.axvline(x=batch_threshold, color='r', linestyle='--', 
                                                           label=f'Current threshold: {batch_threshold}')
                                                
                                                plt.xlabel('Threshold')
                                                plt.ylabel('Score')
                                                plt.title('Performance Metrics at Different Thresholds')
                                                plt.legend()
                                                plt.grid(True, alpha=0.3)
                                                st.pyplot(fig)
                                                
                                                # Find optimal threshold based on F1 score
                                                best_f1_idx = thresh_df['F1 Score'].idxmax()
                                                best_f1_threshold = thresh_df.loc[best_f1_idx, 'Threshold']
                                                
                                                st.info(f"Based on F1 score, the optimal threshold would be: {best_f1_threshold:.2f}")
                                                
                                                # Suggest user adjustments based on class imbalance
                                                minority_pct = y_true_binary.mean() * 100
                                                if minority_pct < 10:
                                                    st.warning(f"Your data is highly imbalanced ({minority_pct:.1f}% positive samples). Consider using a lower threshold to improve recall.")
                                                    
                                                    # Find threshold with best recall that maintains reasonable precision
                                                    good_thresholds = thresh_df[thresh_df['Precision'] >= 0.5]
                                                    if not good_thresholds.empty:
                                                        best_recall_idx = good_thresholds['Recall'].idxmax()
                                                        recall_oriented_threshold = thresh_df.loc[best_recall_idx, 'Threshold']
                                                        st.success(f"For better detection of the minority class while maintaining precision ≥ 0.5, try threshold: {recall_oriented_threshold:.2f}")
                                        except Exception as e:
                                            st.error(f"Error generating ROC curve: {str(e)}")
                            
                            else:  # Numeric or time prediction
                                # Distribution of predictions
                                st.write("#### Prediction Distribution")
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
                                
                                # Compare with actual values if available
                                if target_column in batch_df.columns:
                                    st.write("#### Prediction vs Actual")
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    if target_type == 'time':
                                        # Convert actual times to minutes
                                        from edu_analytics.time_analysis import convert_time_to_minutes
                                        actual_minutes = batch_df[target_column].apply(convert_time_to_minutes)
                                        
                                        plt.scatter(actual_minutes, batch_df['Prediction_Minutes'], alpha=0.5)
                                        plt.plot([actual_minutes.min(), actual_minutes.max()], 
                                                [actual_minutes.min(), actual_minutes.max()], 
                                                'r--')
                                        plt.xlabel('Actual Time (minutes)')
                                        plt.ylabel('Predicted Time (minutes)')
                                    else:
                                        plt.scatter(batch_df[target_column], batch_df['Prediction'], alpha=0.5)
                                        plt.plot([batch_df[target_column].min(), batch_df[target_column].max()], 
                                                [batch_df[target_column].min(), batch_df[target_column].max()], 
                                                'r--')
                                        plt.xlabel('Actual')
                                        plt.ylabel('Predicted')
                                    
                                    plt.title('Actual vs Predicted Values')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Error distribution
                                    st.write("#### Error Distribution")
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    if target_type == 'time':
                                        plt.hist(batch_df['Error_Minutes'], bins=20)
                                        plt.title('Distribution of Prediction Errors')
                                        plt.xlabel('Error (minutes)')
                                    else:
                                        plt.hist(batch_df['Error'], bins=20)
                                        plt.title('Distribution of Prediction Errors')
                                        plt.xlabel('Error')
                                    plt.ylabel('Count')
                                    plt.axvline(x=0, color='r', linestyle='--')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            
                            # Store batch prediction info in report data
                            if 'batch_predictions' not in st.session_state.report_data:
                                st.session_state.report_data['batch_predictions'] = []
                            
                            # Store additional details for classifier with threshold adjustment
                            prediction_details = {}
                            if target_type == 'categorical' and hasattr(model, 'predict_proba') and batch_threshold != 0.5:
                                prediction_details['threshold'] = batch_threshold
                                
                                # Store performance metrics if target column exists
                                if target_column in batch_df.columns:
                                    prediction_details['accuracy'] = accuracy
                                    prediction_details['report'] = report
                            
                            st.session_state.report_data['batch_predictions'].append({
                                'model': selected_model,
                                'records_processed': len(batch_df),
                                'details': prediction_details,
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