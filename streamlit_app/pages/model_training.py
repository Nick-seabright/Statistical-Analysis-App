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
                            
                            # Plot feature importance with error handling for column names
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Check column names and handle variations
                            if 'importance' in importance_df.columns and 'feature' in importance_df.columns:
                                # Standard column names
                                importance_col = 'importance'
                                feature_col = 'feature'
                            elif 'Importance' in importance_df.columns and 'Feature' in importance_df.columns:
                                # Capitalized column names
                                importance_col = 'Importance'
                                feature_col = 'Feature'
                            elif len(importance_df.columns) >= 2:
                                # Assume first column is feature name and second is importance
                                feature_col = importance_df.columns[0]
                                importance_col = importance_df.columns[1]
                                st.info(f"Using columns: {feature_col} (feature) and {importance_col} (importance)")
                            else:
                                # Not enough columns or unrecognized format
                                st.warning("Feature importance DataFrame has an unexpected format. Cannot plot.")
                                feature_col = None
                                importance_col = None
                            
                            # Plot if we have valid columns
                            if feature_col is not None and importance_col is not None:
                                try:
                                    # Sort and plot
                                    sorted_df = importance_df.sort_values(importance_col, ascending=True).tail(15)
                                    sorted_df.plot(kind='barh', x=feature_col, y=importance_col, ax=ax)
                                    plt.title('Feature Importance (Top 15)')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)  # Close the figure to prevent interference
                                except Exception as e:
                                    st.error(f"Error plotting feature importance: {str(e)}")
                                    st.code(f"DataFrame columns: {importance_df.columns.tolist()}")
                                    st.code(f"DataFrame sample:\n{importance_df.head().to_string()}")
                        
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
        
        # Create tabs for different model types
        model_tabs = st.tabs(["Random Forest", "XGBoost", "Neural Network", "SVM/SVR"])
        
        with model_tabs[0]:
            # RANDOM FOREST CONFIGURATION
            st.markdown("### Random Forest Configuration")
            if target_type == 'categorical':
                # Random Forest Classifier params
                n_estimators = st.slider("Number of trees", 10, 500, 100, 10, key="rf_trees")
                max_depth = st.slider("Maximum tree depth", 2, 50, 10, 1, key="rf_depth")
                min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1, key="rf_split")
                class_weight = st.selectbox("Class weights", ["balanced", "balanced_subsample", "None"], key="rf_class_weight")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="rf_adv")
                if show_advanced:
                    min_samples_leaf = st.slider("Minimum samples per leaf", 1, 20, 1, 1, key="rf_leaf")
                    criterion = st.selectbox("Split criterion", ["gini", "entropy"], key="rf_criterion")
                    max_features = st.selectbox("Max features", ["sqrt", "log2", "None"], key="rf_features")
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
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                    except Exception as e:
                        st.error(f"Error training custom Random Forest: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:  # Regression
                # Random Forest Regressor params
                n_estimators = st.slider("Number of trees", 10, 500, 100, 10, key="rfr_trees")
                max_depth = st.slider("Maximum tree depth", 2, 50, 10, 1, key="rfr_depth")
                min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1, key="rfr_split")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="rfr_adv")
                if show_advanced:
                    min_samples_leaf = st.slider("Minimum samples per leaf", 1, 20, 1, 1, key="rfr_leaf")
                    criterion = st.selectbox("Split criterion", ["squared_error", "absolute_error", "poisson"], key="rfr_criterion")
                    max_features = st.selectbox("Max features", ["sqrt", "log2", "None"], key="rfr_features")
                else:
                    min_samples_leaf = 1
                    criterion = "squared_error"
                    max_features = "sqrt"
                
                # Convert "None" string to None
                if max_features == "None":
                    max_features = None
                
                # Training button
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
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                    except Exception as e:
                        st.error(f"Error training custom Random Forest Regressor: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
        with model_tabs[1]:
            # XGBOOST CONFIGURATION
            st.markdown("### XGBoost Configuration")
            if target_type == 'categorical':
                # XGBoost Classifier params
                n_estimators = st.slider("Number of trees", 10, 500, 100, 10, key="xgb_n_est")
                max_depth = st.slider("Maximum tree depth", 2, 20, 6, 1, key="xgb_depth")
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01, key="xgb_lr")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="xgb_adv")
                if show_advanced:
                    subsample = st.slider("Subsample ratio", 0.5, 1.0, 0.8, 0.1, key="xgb_sub")
                    colsample_bytree = st.slider("Column sample by tree", 0.5, 1.0, 0.8, 0.1, key="xgb_col")
                    gamma = st.slider("Minimum loss reduction (gamma)", 0.0, 5.0, 0.0, 0.1, key="xgb_gamma")
                    reg_alpha = st.slider("L1 regularization (alpha)", 0.0, 5.0, 0.0, 0.1, key="xgb_alpha")
                    reg_lambda = st.slider("L2 regularization (lambda)", 0.0, 5.0, 1.0, 0.1, key="xgb_lambda")
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
                            
                            # Display results (continued)
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
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                    except Exception as e:
                        st.error(f"Error training custom XGBoost: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:  # Regression
                # XGBoost Regressor params
                n_estimators = st.slider("Number of trees", 10, 500, 100, 10, key="xgbr_n_est")
                max_depth = st.slider("Maximum tree depth", 2, 20, 6, 1, key="xgbr_depth")
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01, key="xgbr_lr")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="xgbr_adv")
                if show_advanced:
                    subsample = st.slider("Subsample ratio", 0.5, 1.0, 0.8, 0.1, key="xgbr_sub")
                    colsample_bytree = st.slider("Column sample by tree", 0.5, 1.0, 0.8, 0.1, key="xgbr_col")
                    gamma = st.slider("Minimum loss reduction (gamma)", 0.0, 5.0, 0.0, 0.1, key="xgbr_gamma")
                    reg_alpha = st.slider("L1 regularization (alpha)", 0.0, 5.0, 0.0, 0.1, key="xgbr_alpha")
                    reg_lambda = st.slider("L2 regularization (lambda)", 0.0, 5.0, 1.0, 0.1, key="xgbr_lambda")
                else:
                    subsample = 0.8
                    colsample_bytree = 0.8
                    gamma = 0
                    reg_alpha = 0
                    reg_lambda = 1
                
                # Training button
                if st.button("Train Custom XGBoost Regressor", key="train_custom_xgbr"):
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
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                    except Exception as e:
                        st.error(f"Error training custom XGBoost Regressor: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
        with model_tabs[2]:
            # NEURAL NETWORK CONFIGURATION
            st.markdown("### Neural Network Configuration")
            if target_type == 'categorical':
                # Neural Network Classifier params
                n_layers = st.slider("Number of hidden layers", 1, 5, 2, 1, key="nn_layers")
                layer_sizes = []
                for i in range(n_layers):
                    layer_sizes.append(st.slider(f"Neurons in layer {i+1}", 8, 256, 64, 8, key=f"nn_l{i}"))
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.1, key="nn_dropout")
                learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="nn_lr")
                batch_size = st.slider("Batch size", 8, 128, 32, 8, key="nn_batch")
                epochs = st.slider("Epochs", 10, 200, 50, 10, key="nn_epochs")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="nn_adv")
                if show_advanced:
                    activation = st.selectbox("Activation function", ["relu", "tanh", "sigmoid", "elu"], key="nn_act")
                    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop", "adagrad"], key="nn_opt")
                    use_batch_norm = st.checkbox("Use Batch Normalization", value=True, key="nn_bn")
                    early_stopping = st.checkbox("Use Early Stopping", value=True, key="nn_es")
                    patience = st.slider("Early stopping patience", 5, 30, 10, 1, key="nn_patience")
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
                            if n_classes == 2:
                                y_pred = (model.predict(X_test) > 0.5).astype('int32').flatten()
                            else:
                                y_pred = np.argmax(model.predict(X_test), axis=1)
                                
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
                            
                            # Calculate permutation importance
                            if X_test.shape[0] > 1000:
                                # If dataset is large, sample for faster calculation
                                sample_size = min(1000, X_test.shape[0])
                                X_sample = X_test.sample(sample_size, random_state=42)
                                y_sample = y_test.iloc[X_sample.index]
                            else:
                                X_sample = X_test
                                y_sample = y_test
                                
                            try:
                                # Use a wrapper to make the keras model compatible with sklearn
                                from sklearn.inspection import permutation_importance
                                
                                # Create a wrapper for Keras model to use with permutation importance
                                class KerasClassifierWrapper:
                                    def __init__(self, model):
                                        self.model = model
                                        self.classes_ = np.unique(y_train)
                                    
                                    def predict(self, X):
                                        if len(self.classes_) == 2:
                                            return (self.model.predict(X) > 0.5).astype('int32').flatten()
                                        else:
                                            return np.argmax(self.model.predict(X), axis=1)
                                
                                # Create wrapper
                                wrapper_model = KerasClassifierWrapper(model)
                                
                                # Calculate permutation importance
                                perm_importance = permutation_importance(
                                    wrapper_model, X_sample, y_sample, 
                                    n_repeats=5, random_state=42, n_jobs=-1
                                )
                                
                                # Create DataFrame for importance scores
                                importance_df = pd.DataFrame({
                                    'feature': X.columns,
                                    'importance': perm_importance.importances_mean,
                                    'std': perm_importance.importances_std
                                }).sort_values('importance', ascending=False)
                                
                                # Display feature importance
                                st.markdown("### Feature Importance (Permutation Method)")
                                st.dataframe(importance_df)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                importance_df.head(15).sort_values('importance', ascending=True).plot(
                                    kind='barh', x='feature', y='importance', xerr='std', ax=ax)
                                plt.title('Feature Importance (Permutation Method)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            except Exception as e:
                                st.info(f"Could not calculate permutation importance: {str(e)}")
                                st.info("This is normal for large datasets or complex models.")
                                importance_df = pd.DataFrame(columns=['feature', 'importance', 'std'])
                            
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
                                'feature_importance': importance_df if not importance_df.empty else None,
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                    except Exception as e:
                        st.error(f"Error training custom Neural Network: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:  # Regression
                # Neural Network Regressor params
                n_layers = st.slider("Number of hidden layers", 1, 5, 2, 1, key="nnr_layers")
                layer_sizes = []
                for i in range(n_layers):
                    layer_sizes.append(st.slider(f"Neurons in layer {i+1}", 8, 256, 64, 8, key=f"nnr_l{i}"))
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.1, key="nnr_dropout")
                learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="nnr_lr")
                batch_size = st.slider("Batch size", 8, 128, 32, 8, key="nnr_batch")
                epochs = st.slider("Epochs", 10, 200, 50, 10, key="nnr_epochs")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="nnr_adv")
                if show_advanced:
                    activation = st.selectbox("Activation function", ["relu", "tanh", "sigmoid", "elu"], key="nnr_act")
                    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop", "adagrad"], key="nnr_opt")
                    use_batch_norm = st.checkbox("Use Batch Normalization", value=True, key="nnr_bn")
                    early_stopping = st.checkbox("Use Early Stopping", value=True, key="nnr_es")
                    patience = st.slider("Early stopping patience", 5, 30, 10, 1, key="nnr_patience")
                else:
                    activation = "relu"
                    optimizer = "adam"
                    use_batch_norm = True
                    early_stopping = True
                    patience = 10
                
                # Training button
                if st.button("Train Custom Neural Network Regressor", key="train_custom_nnr"):
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
                            
                            # Calculate permutation importance
                            try:
                                # Use a wrapper to make the keras model compatible with sklearn
                                from sklearn.inspection import permutation_importance
                                
                                # Create a wrapper for Keras model to use with permutation importance
                                class KerasRegressorWrapper:
                                    def __init__(self, model):
                                        self.model = model
                                    
                                    def predict(self, X):
                                        return self.model.predict(X).flatten()
                                
                                # Create wrapper
                                wrapper_model = KerasRegressorWrapper(model)
                                
                                # Calculate permutation importance
                                perm_importance = permutation_importance(
                                    wrapper_model, X_test, y_test, 
                                    n_repeats=5, random_state=42, n_jobs=-1
                                )
                                
                                # Create DataFrame for importance scores
                                importance_df = pd.DataFrame({
                                    'feature': X.columns,
                                    'importance': perm_importance.importances_mean,
                                    'std': perm_importance.importances_std
                                }).sort_values('importance', ascending=False)
                                
                                # Display feature importance
                                st.markdown("### Feature Importance (Permutation Method)")
                                st.dataframe(importance_df)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                importance_df.head(15).sort_values('importance', ascending=True).plot(
                                    kind='barh', x='feature', y='importance', xerr='std', ax=ax)
                                plt.title('Feature Importance (Permutation Method)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            except Exception as e:
                                st.info(f"Could not calculate permutation importance: {str(e)}")
                                st.info("This is normal for large datasets or complex models.")
                                importance_df = pd.DataFrame(columns=['feature', 'importance', 'std'])
                            
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
                                'feature_importance': importance_df if not importance_df.empty else None,
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                    except Exception as e:
                        st.error(f"Error training custom Neural Network Regressor: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
        with model_tabs[3]:
            # SVM/SVR CONFIGURATION
            st.markdown("### SVM Configuration")
            if target_type == 'categorical':
                # SVM params
                C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0, 0.1, key="svm_c")
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="svm_adv")
                if show_advanced:
                    if kernel in ["rbf", "poly", "sigmoid"]:
                        gamma = st.selectbox("Kernel coefficient (gamma)", ["scale", "auto", "value"], key="svm_gamma")
                        if gamma == "value":
                            gamma_value = st.slider("Gamma value", 0.001, 1.0, 0.1, 0.001, format="%.3f", key="svm_gamma_val")
                            gamma = gamma_value
                    else:
                        gamma = "scale"
                    
                    if kernel == "poly":
                        degree = st.slider("Polynomial degree", 2, 10, 3, 1, key="svm_degree")
                    else:
                        degree = 3
                    
                    class_weight = st.selectbox("Class weights", ["balanced", "None"], key="svm_class_weight")
                    probability = st.checkbox("Enable probability estimates", value=True, key="svm_prob")
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
                            
                            # Calculate permutation importance
                            try:
                                from sklearn.inspection import permutation_importance
                                
                                # Calculate permutation importance
                                perm_importance = permutation_importance(
                                    model, X_test, y_test, 
                                    n_repeats=5, random_state=42, n_jobs=-1
                                )
                                
                                # Create DataFrame for importance scores
                                importance_df = pd.DataFrame({
                                    'feature': X.columns,
                                    'importance': perm_importance.importances_mean,
                                    'std': perm_importance.importances_std
                                }).sort_values('importance', ascending=False)
                                
                                # Display feature importance
                                st.markdown("### Feature Importance (Permutation Method)")
                                st.dataframe(importance_df)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                importance_df.head(15).sort_values('importance', ascending=True).plot(
                                    kind='barh', x='feature', y='importance', xerr='std', ax=ax)
                                plt.title('Feature Importance (Permutation Method)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            except Exception as e:
                                st.info(f"Could not calculate permutation importance: {str(e)}")
                                importance_df = pd.DataFrame(columns=['feature', 'importance', 'std'])
                            
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
                                'feature_importance': importance_df if not importance_df.empty else None,
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                    except Exception as e:
                        st.error(f"Error training custom SVM: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:  # Regression
                # SVR params
                C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0, 0.1, key="svr_c")
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svr_kernel")
                epsilon = st.slider("Epsilon in the epsilon-SVR model", 0.01, 1.0, 0.1, 0.01, key="svr_epsilon")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="svr_adv")
                if show_advanced:
                    if kernel in ["rbf", "poly", "sigmoid"]:
                        gamma = st.selectbox("Kernel coefficient (gamma)", ["scale", "auto", "value"], key="svr_gamma")
                        if gamma == "value":
                            gamma_value = st.slider("Gamma value", 0.001, 1.0, 0.1, 0.001, format="%.3f", key="svr_gamma_val")
                            gamma = gamma_value
                    else:
                        gamma = "scale"
                    
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
                            
                            # Calculate permutation importance
                            try:
                                from sklearn.inspection import permutation_importance
                                
                                # Calculate permutation importance
                                perm_importance = permutation_importance(
                                    model, X_test, y_test, 
                                    n_repeats=5, random_state=42, n_jobs=-1
                                )
                                
                                # Create DataFrame for importance scores
                                importance_df = pd.DataFrame({
                                    'feature': X.columns,
                                    'importance': perm_importance.importances_mean,
                                    'std': perm_importance.importances_std
                                }).sort_values('importance', ascending=False)
                                
                                # Display feature importance
                                st.markdown("### Feature Importance (Permutation Method)")
                                st.dataframe(importance_df)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                importance_df.head(15).sort_values('importance', ascending=True).plot(
                                    kind='barh', x='feature', y='importance', xerr='std', ax=ax)
                                plt.title('Feature Importance (Permutation Method)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            except Exception as e:
                                st.info(f"Could not calculate permutation importance: {str(e)}")
                                importance_df = pd.DataFrame(columns=['feature', 'importance', 'std'])
                            
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
                                'feature_importance': importance_df if not importance_df.empty else None,
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                    except Exception as e:
                        st.error(f"Error training custom SVR: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with tab3:
        st.markdown("<div class='subheader'>Model Evaluation</div>", unsafe_allow_html=True)
        
        # Add explanation of metrics
        with st.expander("📊 Understanding Model Evaluation Metrics", expanded=True):
            st.markdown("""
            ### Classification Metrics
            - **Accuracy**: The proportion of correct predictions among the total number of predictions (both true positives and true negatives). *Higher is better*.
            - **Precision**: The proportion of true positives among all positive predictions. Measures how many of the predicted positives are actually positive. *Higher is better*.
            - **Recall (Sensitivity)**: The proportion of true positives among all actual positives. Measures how many of the actual positives were correctly identified. *Higher is better*.
            - **F1 Score**: The harmonic mean of precision and recall. Provides a balance between precision and recall. *Higher is better*.
            - **CV Accuracy**: Cross-validation accuracy, the average accuracy across multiple train-test splits. More robust than a single accuracy score. *Higher is better*.
            - **CV Std**: Standard deviation of cross-validation accuracy. Indicates consistency of model performance. *Lower is better*.
            
            ### Regression Metrics
            - **MSE (Mean Squared Error)**: Average of squared differences between predicted and actual values. Penalizes larger errors more. *Lower is better*.
            - **RMSE (Root Mean Squared Error)**: Square root of MSE. In the same units as the target variable. *Lower is better*.
            - **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values. Less sensitive to outliers than MSE. *Lower is better*.
            - **R² Score**: Proportion of variance in the target variable that is predictable from the features. Ranges from 0 to 1 (can be negative in bad models). *Higher is better*.
            - **CV R²**: Cross-validation R² score, the average R² across multiple train-test splits. *Higher is better*.
            - **CV Std**: Standard deviation of cross-validation R² scores. *Lower is better*.
            
            ### Feature Importance
            - **Feature Importance**: Indicates how useful each feature is in the prediction. For tree-based models, this is calculated directly from the model structure.
            - **Permutation Importance**: For models without built-in feature importance, this measures how much model performance decreases when a feature is randomly shuffled.
            """)
        
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
        
        # In the model evaluation section of tab3 where the error occurs (around line 441)
        if st.button("Evaluate Model", key="evaluate_model"):
            try:
                with st.spinner("Evaluating model..."):
                    # Get model
                    model = st.session_state.models[selected_model]
                    # Split data for evaluation
                    from sklearn.model_selection import train_test_split
                    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Check if this is a neural network model (TensorFlow/Keras)
                    is_neural_network = 'keras' in str(type(model)).lower()
                    
                    # Evaluate model
                    if target_type == 'categorical':
                        # Classification evaluation
                        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
                        
                        # Calculate number of classes once, before using it
                        n_classes = len(np.unique(y))
                        
                        # Handle predictions based on model type
                        if is_neural_network:
                            # For neural networks, we need to convert predictions to classes
                            if n_classes == 2:  # Binary classification
                                raw_predictions = model.predict(X_test)
                                y_pred = (raw_predictions > 0.5).astype(int).flatten()
                            else:  # Multi-class classification
                                raw_predictions = model.predict(X_test)
                                y_pred = np.argmax(raw_predictions, axis=1)
                        else:
                            # Standard models
                            y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Display metrics
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        
                        # Classification report
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.markdown("### Classification Report")
                        st.dataframe(report_df)
                        
                        # Plot confusion matrix
                        st.markdown("### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                        plt.title('Confusion Matrix')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        st.pyplot(fig)
                        
                        # ROC curve for binary classification
                        if n_classes == 2 and (hasattr(model, 'predict_proba') or is_neural_network):
                            st.markdown("### ROC Curve")
                            
                            # Get probability scores
                            if is_neural_network:
                                y_pred_proba = model.predict(X_test).flatten()
                            else:
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                            
                            # Calculate ROC curve and AUC
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            # Plot ROC curve
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
                        
                        # Feature importance (tree-based models) or permutation importance (other models)
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("### Feature Importance")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            plt.bar(range(len(indices[:15])), importances[indices[:15]])
                            plt.xticks(range(len(indices[:15])), [X.columns[i] for i in indices[:15]], rotation=90)
                            plt.title('Feature Importance')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            # For non-tree models, calculate permutation importance
                            st.markdown("### Feature Importance (Permutation Method)")
                            
                            try:
                                from sklearn.inspection import permutation_importance
                                
                                # Create wrapper for neural network models
                                if is_neural_network:
                                    class KerasClassifierWrapper:
                                        def __init__(self, model):
                                            self.model = model
                                            self.classes_ = np.unique(y)
                                        
                                        def predict(self, X):
                                            if len(self.classes_) == 2:
                                                return (self.model.predict(X) > 0.5).astype('int32').flatten()
                                            else:
                                                return np.argmax(self.model.predict(X), axis=1)
                                    
                                    model_for_perm = KerasClassifierWrapper(model)
                                else:
                                    model_for_perm = model
                                    
                                # Calculate permutation importance
                                perm_importance = permutation_importance(
                                    model_for_perm, X_test, y_test, 
                                    n_repeats=5, random_state=42, n_jobs=-1
                                )
                                
                                # Create DataFrame for importance scores
                                importance_df = pd.DataFrame({
                                    'feature': X.columns,
                                    'importance': perm_importance.importances_mean,
                                    'std': perm_importance.importances_std
                                }).sort_values('importance', ascending=False)
                                
                                # Display importance table
                                st.dataframe(importance_df)
                                
                                # Plot importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                importance_df.head(15).sort_values('importance', ascending=True).plot(
                                    kind='barh', x='feature', y='importance', xerr='std', ax=ax)
                                plt.title('Feature Importance (Permutation Method)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            except Exception as e:
                                st.info(f"Could not calculate permutation importance: {str(e)}")
                    
                    else:  # Regression
                        # Regression evaluation
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        
                        # Handle predictions based on model type
                        if is_neural_network:
                            y_pred = model.predict(X_test).flatten()
                        else:
                            y_pred = model.predict(X_test)
                        
                        # Calculate metrics
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
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plt.scatter(y_test, y_pred, alpha=0.5)
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        plt.xlabel("Actual")
                        plt.ylabel("Predicted")
                        plt.title("Actual vs Predicted Values")
                        st.pyplot(fig)
                        
                        # Error distribution
                        st.markdown("### Error Distribution")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        errors = y_test - y_pred
                        plt.hist(errors, bins=30)
                        plt.title('Error Distribution')
                        plt.xlabel('Prediction Error')
                        plt.ylabel('Frequency')
                        plt.axvline(x=0, color='r', linestyle='--')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Feature importance (tree-based models) or permutation importance (other models)
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("### Feature Importance")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            plt.bar(range(len(indices[:15])), importances[indices[:15]])
                            plt.xticks(range(len(indices[:15])), [X.columns[i] for i in indices[:15]], rotation=90)
                            plt.title('Feature Importance')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            # For non-tree models, calculate permutation importance
                            st.markdown("### Feature Importance (Permutation Method)")
                            
                            try:
                                from sklearn.inspection import permutation_importance
                                
                                # Create wrapper for neural network models
                                if is_neural_network:
                                    class KerasRegressorWrapper:
                                        def __init__(self, model):
                                            self.model = model
                                        
                                        def predict(self, X):
                                            return self.model.predict(X).flatten()
                                    
                                    model_for_perm = KerasRegressorWrapper(model)
                                else:
                                    model_for_perm = model
                                    
                                # Calculate permutation importance
                                perm_importance = permutation_importance(
                                    model_for_perm, X_test, y_test, 
                                    n_repeats=5, random_state=42, n_jobs=-1
                                )
                                
                                # Create DataFrame for importance scores
                                importance_df = pd.DataFrame({
                                    'feature': X.columns,
                                    'importance': perm_importance.importances_mean,
                                    'std': perm_importance.importances_std
                                }).sort_values('importance', ascending=False)
                                
                                # Display importance table
                                st.dataframe(importance_df)
                                
                                # Plot importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                importance_df.head(15).sort_values('importance', ascending=True).plot(
                                    kind='barh', x='feature', y='importance', xerr='std', ax=ax)
                                plt.title('Feature Importance (Permutation Method)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            except Exception as e:
                                st.info(f"Could not calculate permutation importance: {str(e)}")
                                
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