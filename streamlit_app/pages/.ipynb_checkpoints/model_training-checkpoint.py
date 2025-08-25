# streamlit_app/pages/model_training.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from edu_analytics.utils import save_file, get_timestamped_filename
import pickle

# Add the parent directory to path if running this file directly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from edu_analytics.model_training import train_models
from edu_analytics.model_evaluation import evaluate_model, plot_confusion_matrix, plot_feature_importance

def show_model_training():
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    
    # Check if data is processed
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
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
                            importance_df.sort_values('importance', ascending=True).tail(15).plot(
                                kind='barh', x='feature', y='importance', ax=ax)
                            plt.title('Feature Importance (Top 15)')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Store results in report data
                        st.session_state.report_data['model_training'] = {
                            'models_trained': [name for name, _ in models_to_train],
                            'evaluation_results': evaluation_results,
                            'feature_importance': feature_importance,
                            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
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
                            importance_df.sort_values('importance', ascending=True).tail(15).plot(
                                kind='barh', x='feature', y='importance', ax=ax)
                            plt.title('Feature Importance (Top 15)')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Store results in report data
                        st.session_state.report_data['model_training'] = {
                            'models_trained': [name for name, _ in models_to_train],
                            'evaluation_results': evaluation_results,
                            'feature_importance': feature_importance,
                            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                except Exception as e:
                    st.error(f"Error training regression models: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        else:
            st.warning(f"Model training is not supported for target type: {target_type}")
    
    with tab2:
        st.markdown("<div class='subheader'>Advanced Model Configuration</div>", unsafe_allow_html=True)
        
        # This would include hyperparameter tuning interfaces
        st.write("Advanced model configuration and hyperparameter tuning will be implemented in a future update.")
        
        # For now, we'll show a simpler interface for customizing a Random Forest model
        if target_type == 'categorical':
            st.markdown("### Random Forest Classifier Configuration")
            
            n_estimators = st.slider("Number of trees", 10, 500, 100, 10, key="rf_trees")
            max_depth = st.slider("Maximum tree depth", 2, 50, 10, 1, key="rf_depth")
            min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1, key="rf_split")
            
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
                            random_state=42
                        )
                        
                        model.fit(X_train, y_train)
                        
                        # Store the model
                        if 'models' not in st.session_state:
                            st.session_state.models = {}
                        
                        st.session_state.models['Custom RF'] = model
                        
                        # Evaluate the model
                        from sklearn.metrics import accuracy_score, classification_report
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        st.success(f"Custom Random Forest trained with accuracy: {accuracy:.4f}")
                        
                        # Feature importance
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        importance_df = pd.DataFrame({
                            'feature': [X.columns[i] for i in indices],
                            'importance': [importances[i] for i in indices]
                        })
                        
                        # Show feature importance
                        st.markdown("### Feature Importance")
                        st.dataframe(importance_df)
                        
                        # Plot feature importance
                        fig = plot_feature_importance(model, X.columns, title="Feature Importance")
                        st.pyplot(fig)
                        
                        # Plot confusion matrix
                        st.markdown("### Confusion Matrix")
                        fig = plot_confusion_matrix(confusion_matrix=None, model=model, X_test=X_test, y_test=y_test)
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error training custom Random Forest: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        elif target_type in ['numeric', 'time']:
            st.markdown("### Random Forest Regressor Configuration")
            
            n_estimators = st.slider("Number of trees", 10, 500, 100, 10, key="rfr_trees")
            max_depth = st.slider("Maximum tree depth", 2, 50, 10, 1, key="rfr_depth")
            min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1, key="rfr_split")
            
            if st.button("Train Custom Random Forest Regressor", key="train_custom_rfr"):
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
                            random_state=42
                        )
                        
                        model.fit(X_train, y_train)
                        
                        # Store the model
                        if 'models' not in st.session_state:
                            st.session_state.models = {}
                        
                        st.session_state.models['Custom RF Regressor'] = model
                        
                        # Evaluate the model
                        from sklearn.metrics import mean_squared_error, r2_score
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        st.success(f"Custom Random Forest Regressor trained with R² Score: {r2:.4f}")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("RMSE", f"{rmse:.4f}")
                        col2.metric("R² Score", f"{r2:.4f}")
                        
                        # Feature importance
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        importance_df = pd.DataFrame({
                            'feature': [X.columns[i] for i in indices],
                            'importance': [importances[i] for i in indices]
                        })
                        
                        # Show feature importance
                        st.markdown("### Feature Importance")
                        st.dataframe(importance_df)
                        
                        # Plot feature importance
                        fig = plot_feature_importance(model, X.columns, title="Feature Importance")
                        st.pyplot(fig)
                        
                        # Plot actual vs predicted
                        st.markdown("### Actual vs Predicted")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.scatter(y_test, y_pred, alpha=0.5)
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        plt.xlabel("Actual")
                        plt.ylabel("Predicted")
                        plt.title("Actual vs Predicted Values")
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error training custom Random Forest Regressor: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab3:
        st.markdown("<div class='subheader'>Model Evaluation</div>", unsafe_allow_html=True)
        
        # Check if models exist
        if 'models' not in st.session_state or not st.session_state.models:
            st.warning("No trained models available for evaluation. Please train models first.")
            return
        
        # Model selection for evaluation
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox(
            "Select model to evaluate",
            model_names
        )
        
        if st.button("Evaluate Model", key="evaluate_model"):
            try:
                with st.spinner("Evaluating model..."):
                    # Get model
                    model = st.session_state.models[selected_model]
                    
                    # Split data for evaluation
                    from sklearn.model_selection import train_test_split
                    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Evaluate model
                    if target_type == 'categorical':
                        # Classification evaluation
                        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                        
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Display metrics
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        
                        # Classification report
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.markdown("### Classification Report")
                        st.dataframe(report_df)
                        
                        # Confusion matrix
                        st.markdown("### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig = plot_confusion_matrix(cm)
                        st.pyplot(fig)
                        
                        # Feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("### Feature Importance")
                            fig = plot_feature_importance(model, X.columns)
                            st.pyplot(fig)
                        
                    else:  # Regression
                        # Regression evaluation
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("RMSE", f"{rmse:.4f}")
                        col2.metric("MAE", f"{mae:.4f}")
                        col3.metric("R² Score", f"{r2:.4f}")
                        
                        # Plot actual vs predicted
                        st.markdown("### Actual vs Predicted")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.scatter(y_test, y_pred, alpha=0.5)
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        plt.xlabel("Actual")
                        plt.ylabel("Predicted")
                        plt.title("Actual vs Predicted Values")
                        st.pyplot(fig)
                        
                        # Feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("### Feature Importance")
                            fig = plot_feature_importance(model, X.columns)
                            st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error evaluating model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

if st.button("Save Model", key="save_model"):
    try:
        with st.spinner("Saving model..."):
            # Serialize the model
            model_bytes = pickle.dumps(model)
            
            # Generate filename
            model_filename = get_timestamped_filename(f"{selected_model}_model", "pkl")
            
            # Save model to user-specified directory
            success, message, path = save_file(model_bytes, model_filename, "models")
            
            if success:
                st.success(f"Model saved: {path}")
            else:
                st.warning(message)
                
                # Provide download as backup
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")

# streamlit_app/pages/model_training.py (add at the end)

# Import the file browser
from streamlit_app.components.file_browser import file_browser
import os
import pickle

# Add a section to view saved models
st.markdown("---")
st.subheader("Load Saved Models")

# Get models directory
if 'save_directory' in st.session_state:
    models_dir = os.path.join(st.session_state.save_directory, "models")
    
    # Check if directory exists, create if it doesn't
    if not os.path.exists(models_dir):
        try:
            os.makedirs(models_dir, exist_ok=True)
            st.info(f"Created models directory: {models_dir}")
        except Exception as e:
            st.warning(f"Could not create models directory: {str(e)}")
    
    # Use the file browser component
    selected_model_file = file_browser(models_dir, "pkl")
    
    if selected_model_file:
        st.write(f"Selected model: {os.path.basename(selected_model_file)}")
        
        # Offer option to load the model
        if st.button("Load Model"):
            try:
                with open(selected_model_file, "rb") as f:
                    loaded_model = pickle.load(f)
                
                # Store in session state
                model_name = os.path.basename(selected_model_file).split('.')[0]
                if 'models' not in st.session_state:
                    st.session_state.models = {}
                
                st.session_state.models[model_name] = loaded_model
                st.success(f"Model '{model_name}' loaded successfully!")
                
                # Provide model info if available
                st.write("Model type:", type(loaded_model).__name__)
                
                # If it's a sklearn model, show more details
                if hasattr(loaded_model, 'get_params'):
                    st.write("Model parameters:", loaded_model.get_params())
                
            except Exception as e:
                st.error(f"Could not load model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
else:
    st.info("Please set a save directory in the sidebar to view saved models.")

if __name__ == "__main__":
    show_model_training()