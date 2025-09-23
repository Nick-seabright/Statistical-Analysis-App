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
    target_mapping = st.session_state.target_mapping if 'target_mapping' in st.session_state else None
    
    # Create tabs for model training options
    tab1, tab2, tab3 = st.tabs(["Basic Models", "Advanced Configuration", "Model Evaluation"])
    with tab1:
        st.markdown("<div class='subheader'>Train Basic Models</div>", unsafe_allow_html=True)
        # Select models to train
        st.write("Select models to train:")
        # Model selection based on target type
        if target_type == 'categorical':
            # Check if data is imbalanced
            class_counts = pd.Series(y).value_counts()
            imbalance_ratio = class_counts.min() / class_counts.max()
            
            # Show class distribution metrics
            st.markdown("### Class Distribution")
            class_dist_df = pd.DataFrame({
                'Class': class_counts.index,
                'Count': class_counts.values,
                'Percentage': (class_counts.values / len(y) * 100).round(2)
            })
            st.dataframe(class_dist_df)
            
            # Show imbalance warning and options if significant imbalance exists
            if imbalance_ratio < 0.2:
                st.warning(f"Your data is imbalanced. The minority class represents only {imbalance_ratio:.1%} of the majority class.")
                
                handle_imbalance = st.checkbox("Handle class imbalance", value=True)
                
                if handle_imbalance:
                    imbalance_method = st.selectbox(
                        "Select method to handle imbalance",
                        options=["class_weights", "smote", "adasyn", "random_over", "random_under"],
                        help="""
                        class_weights: Adjust model to pay more attention to minority class
                        smote: Generate synthetic samples for minority class
                        adasyn: Adaptive Synthetic Sampling
                        random_over: Random oversampling of minority class
                        random_under: Random undersampling of majority class
                        """
                    )
                    
                    # Display information about the selected method
                    if imbalance_method == "class_weights":
                        st.info("Class weights will be automatically calculated based on class frequencies.")
                    elif imbalance_method == "smote":
                        st.info("SMOTE (Synthetic Minority Over-sampling Technique) will generate synthetic examples for the minority class based on nearest neighbors.")
                    elif imbalance_method == "adasyn":
                        st.info("ADASYN (Adaptive Synthetic Sampling) will generate more synthetic examples for minority instances that are harder to learn.")
                    elif imbalance_method == "random_over":
                        st.info("Random oversampling will duplicate examples from the minority class to balance the dataset.")
                    elif imbalance_method == "random_under":
                        st.info("Random undersampling will remove examples from the majority class to balance the dataset.")
                else:
                    imbalance_method = None
            else:
                handle_imbalance = False
                imbalance_method = None
            
            # Select standard models
            train_rf = st.checkbox("Random Forest Classifier", value=True)
            train_svm = st.checkbox("Support Vector Machine", value=False)
            train_xgb = st.checkbox("XGBoost Classifier", value=True)
            train_nn = st.checkbox("Neural Network", value=False)
            
            # Add specialized models for imbalanced data
            if handle_imbalance:
                st.markdown("### Specialized Models for Imbalanced Data")
                train_balanced_rf = st.checkbox("Balanced Random Forest", value=False, 
                                        help="Variant of Random Forest designed for imbalanced data")
                train_easy_ensemble = st.checkbox("EasyEnsemble", value=False,
                                        help="Ensemble method that's effective with imbalanced data")
            else:
                train_balanced_rf = False
                train_easy_ensemble = False
            
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
                        
                        # Add specialized models for imbalanced data if selected
                        if train_balanced_rf:
                            models_to_train.append(('Balanced Random Forest', 'balanced_rf'))
                        if train_easy_ensemble:
                            models_to_train.append(('EasyEnsemble', 'easy_ensemble'))
                            
                        if not models_to_train:
                            st.warning("Please select at least one model to train.")
                            return
                        
                        # Train models using our function with imbalance handling
                        trained_models, evaluation_results, feature_importance = train_models(
                            X=X,
                            y=y,
                            models=models_to_train,
                            target_type=target_type,
                            test_size=test_size,
                            random_state=42,
                            handle_imbalance=handle_imbalance,
                            imbalance_method=imbalance_method
                        )
                        
                        # Store trained models in session state
                        st.session_state.models = trained_models
                        st.session_state.model_evaluation = evaluation_results
                        st.session_state.feature_importance = feature_importance
                        
                        # Show success message
                        st.success(f"Successfully trained {len(trained_models)} models!")
                        
                        # Show evaluation results
                        st.markdown("### Model Performance")
                        
                        # Add explanations for metrics that are especially important for imbalanced data
                        if handle_imbalance:
                            st.markdown("""
                            #### Key Metrics for Imbalanced Data:
                            - **Balanced Accuracy**: Accuracy that accounts for class imbalance
                            - **Precision**: How many selected items are relevant (TP/(TP+FP))
                            - **Recall**: How many relevant items are selected (TP/(TP+FN))
                            - **F1 Score**: Harmonic mean of precision and recall
                            - **PR-AUC**: Area under the Precision-Recall curve (better than ROC-AUC for imbalanced data)
                            """)
                            
                        st.dataframe(evaluation_results)
                        
                        # If data is imbalanced, add per-class metrics view
                        if handle_imbalance and 'classification_reports' in st.session_state.report_data['model_training']:
                            st.markdown("### Per-Class Performance")
                            selected_model = st.selectbox(
                                "Select model to view detailed metrics",
                                options=list(trained_models.keys())
                            )
                            
                            if selected_model in st.session_state.report_data['model_training']['classification_reports']:
                                report = st.session_state.report_data['model_training']['classification_reports'][selected_model]
                                # Extract per-class metrics from the report
                                classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
                                
                                # Create dataframe of class metrics
                                metrics = ['precision', 'recall', 'f1-score', 'support']
                                class_metrics = []
                                
                                for cls in classes:
                                    row = {'class': cls}
                                    for metric in metrics:
                                        row[metric] = report[cls][metric]
                                    class_metrics.append(row)
                                
                                class_metrics_df = pd.DataFrame(class_metrics)
                                st.dataframe(class_metrics_df)
                                
                                # Plot class-specific metrics
                                fig, ax = plt.subplots(figsize=(10, 6))
                                bar_width = 0.25
                                index = np.arange(len(classes))
                                
                                ax.bar(index - bar_width, class_metrics_df['precision'], bar_width, label='Precision')
                                ax.bar(index, class_metrics_df['recall'], bar_width, label='Recall')
                                ax.bar(index + bar_width, class_metrics_df['f1-score'], bar_width, label='F1 Score')
                                
                                ax.set_xlabel('Class')
                                ax.set_ylabel('Score')
                                ax.set_title('Metrics by Class')
                                ax.set_xticks(index)
                                ax.set_xticklabels(class_metrics_df['class'])
                                ax.legend()
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
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
                            plt.title('Feature Importance')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Store results in report data
                        st.session_state.report_data['model_training'] = {
                            'models_trained': [name for name, _ in models_to_train],
                            'evaluation_results': evaluation_results,
                            'feature_importance': feature_importance,
                            'handle_imbalance': handle_imbalance,
                            'imbalance_method': imbalance_method if handle_imbalance else None,
                            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # If we have classification reports, store them too
                        if hasattr(st.session_state, 'classification_reports'):
                            st.session_state.report_data['model_training']['classification_reports'] = (
                                st.session_state.classification_reports
                            )
                        
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
                            plt.title('Feature Importance')
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
                
                # Class weights configuration
                class_weight_options = ["None", "balanced", "balanced_subsample", "custom"]
                class_weight_choice = st.selectbox(
                    "Class weights", 
                    class_weight_options, 
                    index=1 if 'handle_imbalance' in locals() and handle_imbalance else 0, 
                    key="rf_class_weight_choice"
                )
                
                # If custom class weights selected, show input fields
                if class_weight_choice == "custom":
                    st.markdown("#### Custom Class Weights")
                    class_weight = {}
                    
                    # If we have class labels available, use them for better UX
                    if target_mapping:
                        reverse_mapping = {v: k for k, v in target_mapping.items()}
                        for class_idx in range(len(reverse_mapping)):
                            original_label = reverse_mapping.get(class_idx, f"Class {class_idx}")
                            weight = st.number_input(
                                f"Weight for {original_label}", 
                                min_value=0.1, 
                                max_value=100.0, 
                                value=1.0 if class_idx == 0 else 5.0,  # Default higher weight for minority class
                                step=0.1,
                                key=f"rf_class_weight_{class_idx}"
                            )
                            class_weight[class_idx] = weight
                    else:
                        # If no mapping available, use generic class names
                        num_classes = len(pd.Series(y).unique())
                        for class_idx in range(num_classes):
                            weight = st.number_input(
                                f"Weight for Class {class_idx}", 
                                min_value=0.1, 
                                max_value=100.0, 
                                value=1.0 if class_idx == 0 else 5.0,
                                step=0.1,
                                key=f"rf_class_weight_{class_idx}"
                            )
                            class_weight[class_idx] = weight
                
                elif class_weight_choice == "None":
                    class_weight = None
                else:
                    class_weight = class_weight_choice
                
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
                            
                            # Handle imbalanced data if requested
                            if 'handle_imbalance' in locals() and handle_imbalance and imbalance_method != "class_weights":
                                # Import the appropriate resampling method
                                if imbalance_method == "smote":
                                    from imblearn.over_sampling import SMOTE
                                    resampler = SMOTE(random_state=42)
                                elif imbalance_method == "adasyn":
                                    from imblearn.over_sampling import ADASYN
                                    resampler = ADASYN(random_state=42)
                                elif imbalance_method == "random_over":
                                    from imblearn.over_sampling import RandomOverSampler
                                    resampler = RandomOverSampler(random_state=42)
                                elif imbalance_method == "random_under":
                                    from imblearn.under_sampling import RandomUnderSampler
                                    resampler = RandomUnderSampler(random_state=42)
                                
                                # Apply resampling
                                X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
                                
                                # Show resampling results
                                original_class_counts = pd.Series(y_train).value_counts()
                                resampled_class_counts = pd.Series(y_train_resampled).value_counts()
                                
                                st.info(f"Resampling changed class distribution from {dict(original_class_counts)} to {dict(resampled_class_counts)}")
                                
                                # Use resampled data for training
                                X_train = X_train_resampled
                                y_train = y_train_resampled
                            
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
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            balanced_acc = balanced_accuracy_score(y_test, y_pred)
                            # Display results
                            st.success(f"Custom Random Forest trained with accuracy: {accuracy:.4f}, balanced accuracy: {balanced_acc:.4f}")
                            st.markdown("### Classification Report")
                            # Generate and display classification report
                            report = classification_report(y_test, y_pred, output_dict=True)
                            # If we have target mapping, use it to convert encoded class labels to original names
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                # Create a new report with original category names
                                new_report = {}
                                for key, val in report.items():
                                    if key.isdigit() or (isinstance(key, (int, float)) and int(key) == key):
                                        # This is a class label - convert it
                                        new_key = reverse_mapping.get(int(key), str(key))
                                        new_report[new_key] = val
                                    else:
                                        # This is a metric like 'accuracy', 'macro avg', etc.
                                        new_report[key] = val
                                report = new_report
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            # Get class names for the confusion matrix
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                class_names = [reverse_mapping.get(i, str(i)) for i in range(len(np.unique(y)))]
                                # Plot confusion matrix with original class names
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=class_names, yticklabels=class_names)
                            else:
                                # Default behavior without mapping
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # Plot ROC curve for binary classification
                            if len(np.unique(y_test)) == 2:
                                from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
                                # ROC curve
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                roc_auc = auc(fpr, tpr)
                                
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # ROC curve
                                ax1.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                                ax1.plot([0, 1], [0, 1], 'k--')
                                ax1.set_xlim([0.0, 1.0])
                                ax1.set_ylim([0.0, 1.05])
                                ax1.set_xlabel('False Positive Rate')
                                ax1.set_ylabel('True Positive Rate')
                                ax1.set_title('Receiver Operating Characteristic (ROC)')
                                ax1.legend(loc="lower right")
                                
                                # Precision-Recall curve (better for imbalanced data)
                                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                                pr_auc = average_precision_score(y_test, y_pred_proba)
                                
                                ax2.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
                                ax2.set_xlim([0.0, 1.0])
                                ax2.set_ylim([0.0, 1.05])
                                ax2.set_xlabel('Recall')
                                ax2.set_ylabel('Precision')
                                ax2.set_title('Precision-Recall Curve (Better for Imbalanced Data)')
                                ax2.legend(loc="lower left")
                                
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
                            model_name = "Custom Random Forest"
                            st.session_state.models[model_name] = model
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            st.session_state.report_data['custom_models'][model_name] = {
                                'accuracy': accuracy,
                                'balanced_accuracy': balanced_acc,
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
                
                # Handle imbalanced data
                if 'handle_imbalance' in locals() and handle_imbalance:
                    st.markdown("### Imbalanced Data Handling")
                    
                    # Scale positive weight option (specific to XGBoost)
                    scale_pos_weight = st.number_input(
                        "Scale Positive Weight", 
                        min_value=1.0, 
                        max_value=100.0, 
                        value=len(y) / (2 * np.sum(y)) if target_type == 'categorical' else 1.0,
                        help="Balancing of positive and negative weights. Useful for unbalanced classes."
                    )
                    
                    # Add explanation for XGBoost-specific imbalance parameters
                    st.info("""
                        **Scale Positive Weight**: XGBoost-specific parameter for handling imbalanced data.
                        A higher value gives more weight to the minority class.
                        The default value above is calculated as (total_instances / (2 * positive_instances)),
                        which is a good starting point for imbalanced data.
                    """)
                else:
                    scale_pos_weight = 1.0
                
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
                            
                            # Handle imbalanced data if requested
                            if 'handle_imbalance' in locals() and handle_imbalance and imbalance_method not in ["class_weights", None]:
                                # Import the appropriate resampling method
                                if imbalance_method == "smote":
                                    from imblearn.over_sampling import SMOTE
                                    resampler = SMOTE(random_state=42)
                                elif imbalance_method == "adasyn":
                                    from imblearn.over_sampling import ADASYN
                                    resampler = ADASYN(random_state=42)
                                elif imbalance_method == "random_over":
                                    from imblearn.over_sampling import RandomOverSampler
                                    resampler = RandomOverSampler(random_state=42)
                                elif imbalance_method == "random_under":
                                    from imblearn.under_sampling import RandomUnderSampler
                                    resampler = RandomUnderSampler(random_state=42)
                                
                                # Apply resampling
                                X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
                                
                                # Show resampling results
                                original_class_counts = pd.Series(y_train).value_counts()
                                resampled_class_counts = pd.Series(y_train_resampled).value_counts()
                                
                                st.info(f"Resampling changed class distribution from {dict(original_class_counts)} to {dict(resampled_class_counts)}")
                                
                                # Use resampled data for training
                                X_train = X_train_resampled
                                y_train = y_train_resampled
                            
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
                                scale_pos_weight=scale_pos_weight,
                                random_state=42
                            )
                            model.fit(X_train, y_train)
                            # Evaluate model
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
                            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            balanced_acc = balanced_accuracy_score(y_test, y_pred)
                            # Display results
                            st.success(f"Custom XGBoost trained with accuracy: {accuracy:.4f}, balanced accuracy: {balanced_acc:.4f}")
                            st.markdown("### Classification Report")
                            # Generate and display classification report with original class names
                            report = classification_report(y_test, y_pred, output_dict=True)
                            # If we have target mapping, use it to convert encoded class labels to original names
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                # Create a new report with original category names
                                new_report = {}
                                for key, val in report.items():
                                    if key.isdigit() or (isinstance(key, (int, float)) and int(key) == key):
                                        # This is a class label - convert it
                                        new_key = reverse_mapping.get(int(key), str(key))
                                        new_report[new_key] = val
                                    else:
                                        # This is a metric like 'accuracy', 'macro avg', etc.
                                        new_report[key] = val
                                report = new_report
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            # Get class names for the confusion matrix
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                class_names = [reverse_mapping.get(i, str(i)) for i in range(len(np.unique(y)))]
                                # Plot confusion matrix with original class names
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=class_names, yticklabels=class_names)
                            else:
                                # Default behavior without mapping
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # For binary classification, plot ROC and PR curves
                            if len(np.unique(y_test)) == 2:
                                # Get probabilities
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                                
                                # Create a figure with two subplots
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # ROC curve
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                roc_auc = auc(fpr, tpr)
                                
                                ax1.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                                ax1.plot([0, 1], [0, 1], 'k--')
                                ax1.set_xlim([0.0, 1.0])
                                ax1.set_ylim([0.0, 1.05])
                                ax1.set_xlabel('False Positive Rate')
                                ax1.set_ylabel('True Positive Rate')
                                ax1.set_title('Receiver Operating Characteristic (ROC)')
                                ax1.legend(loc="lower right")
                                
                                # Precision-Recall curve (better for imbalanced data)
                                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                                pr_auc = average_precision_score(y_test, y_pred_proba)
                                
                                ax2.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
                                ax2.set_xlim([0.0, 1.0])
                                ax2.set_ylim([0.0, 1.05])
                                ax2.set_xlabel('Recall')
                                ax2.set_ylabel('Precision')
                                ax2.set_title('Precision-Recall Curve (Better for Imbalanced Data)')
                                ax2.legend(loc="lower left")
                                
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
                            model_name = "Custom XGBoost"
                            st.session_state.models[model_name] = model
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            st.session_state.report_data['custom_models'][model_name] = {
                                'accuracy': accuracy,
                                'balanced_accuracy': balanced_acc,
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
                                    'reg_lambda': reg_lambda,
                                    'scale_pos_weight': scale_pos_weight
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
                        
        with model_tabs[3]:  # SVM/SVR tab
            # SVM/SVR CONFIGURATION
            st.markdown("### SVM/SVR Configuration")
            if target_type == 'categorical':
                # SVM Classifier params
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
                C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0, 0.1, key="svm_c")
                
                # Handle imbalanced data
                if 'handle_imbalance' in locals() and handle_imbalance:
                    st.markdown("### Imbalanced Data Handling")
                    class_weight_options = ["None", "balanced", "custom"]
                    class_weight_choice = st.selectbox(
                        "Class weights",
                        class_weight_options,
                        index=1 if handle_imbalance else 0,
                        key="svm_class_weight_choice"
                    )
                    
                    # If custom class weights selected, show input fields
                    if class_weight_choice == "custom":
                        st.markdown("#### Custom Class Weights")
                        class_weight = {}
                        # If we have class labels available, use them for better UX
                        if target_mapping:
                            reverse_mapping = {v: k for k, v in target_mapping.items()}
                            for class_idx in range(len(reverse_mapping)):
                                original_label = reverse_mapping.get(class_idx, f"Class {class_idx}")
                                weight = st.number_input(
                                    f"Weight for {original_label}",
                                    min_value=0.1,
                                    max_value=100.0,
                                    value=1.0 if class_idx == 0 else 5.0,  # Default higher weight for minority class
                                    step=0.1,
                                    key=f"svm_class_weight_{class_idx}"
                                )
                                class_weight[class_idx] = weight
                        else:
                            # If no mapping available, use generic class names
                            num_classes = len(pd.Series(y).unique())
                            for class_idx in range(num_classes):
                                weight = st.number_input(
                                    f"Weight for Class {class_idx}",
                                    min_value=0.1,
                                    max_value=100.0,
                                    value=1.0 if class_idx == 0 else 5.0,
                                    step=0.1,
                                    key=f"svm_class_weight_{class_idx}"
                                )
                                class_weight[class_idx] = weight
                    elif class_weight_choice == "None":
                        class_weight = None
                    else:
                        class_weight = class_weight_choice
                else:
                    class_weight = None
                    
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="svm_adv")
                if show_advanced:
                    gamma_options = ["scale", "auto", "custom"]
                    gamma_choice = st.selectbox("Gamma", gamma_options, key="svm_gamma_choice")
                    
                    if gamma_choice == "custom":
                        gamma = st.slider("Custom gamma value", 0.001, 10.0, 0.1, 0.001, key="svm_gamma_custom")
                    else:
                        gamma = gamma_choice
                        
                    degree = st.slider("Polynomial degree", 2, 10, 3, 1, key="svm_degree") if kernel == "poly" else 3
                    coef0 = st.slider("Coefficient (coef0)", 0.0, 10.0, 0.0, 0.1, key="svm_coef0") if kernel in ["poly", "sigmoid"] else 0.0
                    
                    probability = st.checkbox("Enable probability estimates", value=True, key="svm_prob")
                    shrinking = st.checkbox("Use shrinking heuristic", value=True, key="svm_shrinking")
                    
                    decision_function_shape = st.selectbox("Decision function shape", ["ovr", "ovo"], key="svm_decision")
                    cache_size = st.slider("Cache size (MB)", 100, 2000, 200, 100, key="svm_cache")
                    max_iter = st.slider("Maximum iterations", -1, 10000, -1, 100, key="svm_max_iter")
                    tol = st.slider("Tolerance", 0.00001, 0.001, 0.0001, 0.00001, format="%.5f", key="svm_tol")
                    
                    class_balance_option = st.radio(
                        "Class balance strategy",
                        ["Default", "Custom threshold"],
                        key="svm_balance_strategy"
                    )
                    
                    if class_balance_option == "Custom threshold" and len(np.unique(y)) == 2:
                        custom_threshold = st.slider(
                            "Decision threshold", 
                            0.0, 1.0, 0.5, 0.05, 
                            help="Adjusts the decision boundary to favor recall or precision",
                            key="svm_threshold"
                        )
                    else:
                        custom_threshold = 0.5
                else:
                    gamma = "scale"
                    degree = 3
                    coef0 = 0.0
                    probability = True
                    shrinking = True
                    decision_function_shape = "ovr"
                    cache_size = 200
                    max_iter = -1
                    tol = 0.0001
                    custom_threshold = 0.5
                    
                # Training button
                if st.button("Train Custom SVM", key="train_custom_svm"):
                    try:
                        with st.spinner("Training custom SVM..."):
                            from sklearn.svm import SVC
                            from sklearn.model_selection import train_test_split
                            from sklearn.preprocessing import StandardScaler
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Scale features (important for SVM)
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Handle imbalanced data if requested
                            if 'handle_imbalance' in locals() and handle_imbalance and imbalance_method != "class_weights":
                                # Import the appropriate resampling method
                                if imbalance_method == "smote":
                                    from imblearn.over_sampling import SMOTE
                                    resampler = SMOTE(random_state=42)
                                elif imbalance_method == "adasyn":
                                    from imblearn.over_sampling import ADASYN
                                    resampler = ADASYN(random_state=42)
                                elif imbalance_method == "random_over":
                                    from imblearn.over_sampling import RandomOverSampler
                                    resampler = RandomOverSampler(random_state=42)
                                elif imbalance_method == "random_under":
                                    from imblearn.under_sampling import RandomUnderSampler
                                    resampler = RandomUnderSampler(random_state=42)
                                
                                # Apply resampling
                                X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train)
                                
                                # Show resampling results
                                original_class_counts = pd.Series(y_train).value_counts()
                                resampled_class_counts = pd.Series(y_train_resampled).value_counts()
                                st.info(f"Resampling changed class distribution from {dict(original_class_counts)} to {dict(resampled_class_counts)}")
                                
                                # Use resampled data for training
                                X_train_scaled = X_train_resampled
                                y_train = y_train_resampled
                            
                            # Create and train model
                            model = SVC(
                                C=C,
                                kernel=kernel,
                                degree=degree,
                                gamma=gamma,
                                coef0=coef0,
                                shrinking=shrinking,
                                probability=probability,
                                tol=tol,
                                cache_size=cache_size,
                                class_weight=class_weight,
                                max_iter=max_iter,
                                decision_function_shape=decision_function_shape,
                                random_state=42
                            )
                            
                            model.fit(X_train_scaled, y_train)
                            
                            # Evaluate model
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
                            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
                            
                            # Get predictions
                            if probability and custom_threshold != 0.5 and len(np.unique(y)) == 2:
                                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                                y_pred = (y_pred_proba > custom_threshold).astype(int)
                                st.info(f"Using custom threshold: {custom_threshold}")
                            else:
                                y_pred = model.predict(X_test_scaled)
                                if probability and len(np.unique(y)) == 2:
                                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                                else:
                                    y_pred_proba = None
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            balanced_acc = balanced_accuracy_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Custom SVM trained with accuracy: {accuracy:.4f}, balanced accuracy: {balanced_acc:.4f}")
                            
                            # Display classification report with original class names
                            st.markdown("### Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            
                            # If we have target mapping, use it to convert encoded class labels to original names
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                
                                # Create a new report with original category names
                                new_report = {}
                                for key, val in report.items():
                                    if key.isdigit() or (isinstance(key, (int, float)) and int(key) == key):
                                        # This is a class label - convert it
                                        new_key = reverse_mapping.get(int(key), str(key))
                                        new_report[new_key] = val
                                    else:
                                        # This is a metric like 'accuracy', 'macro avg', etc.
                                        new_report[key] = val
                                report = new_report
                            
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Plot confusion matrix
                            st.markdown("### Confusion Matrix")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            
                            # Get class names for the confusion matrix
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                class_names = [reverse_mapping.get(i, str(i)) for i in range(len(np.unique(y)))]
                                
                                # Plot confusion matrix with original class names
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=class_names, yticklabels=class_names)
                            else:
                                # Default behavior without mapping
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # For binary classification, plot ROC and PR curves
                            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                                st.markdown("### ROC and Precision-Recall Curves")
                                
                                # Create a figure with two subplots
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # ROC curve
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                roc_auc = auc(fpr, tpr)
                                
                                ax1.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                                ax1.plot([0, 1], [0, 1], 'k--')
                                ax1.set_xlim([0.0, 1.0])
                                ax1.set_ylim([0.0, 1.05])
                                ax1.set_xlabel('False Positive Rate')
                                ax1.set_ylabel('True Positive Rate')
                                ax1.set_title('Receiver Operating Characteristic (ROC)')
                                ax1.legend(loc="lower right")
                                
                                # Precision-Recall curve (better for imbalanced data)
                                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                                pr_auc = average_precision_score(y_test, y_pred_proba)
                                
                                ax2.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
                                ax2.set_xlim([0.0, 1.0])
                                ax2.set_ylim([0.0, 1.05])
                                ax2.set_xlabel('Recall')
                                ax2.set_ylabel('Precision')
                                ax2.set_title('Precision-Recall Curve (Better for Imbalanced Data)')
                                ax2.legend(loc="lower left")
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Decision boundary visualization for 2D data
                                if X.shape[1] == 2:
                                    st.markdown("### Decision Boundary")
                                    
                                    # Create a mesh grid
                                    h = 0.02  # Step size
                                    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                                    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                                    
                                    # Scale mesh grid
                                    mesh_points = np.c_[xx.ravel(), yy.ravel()]
                                    mesh_points_scaled = scaler.transform(mesh_points)
                                    
                                    # Get predictions on mesh
                                    if probability and custom_threshold != 0.5:
                                        Z = model.predict_proba(mesh_points_scaled)[:, 1]
                                        Z = (Z > custom_threshold).astype(int)
                                    else:
                                        Z = model.predict(mesh_points_scaled)
                                    
                                    Z = Z.reshape(xx.shape)
        
                                    # Plot decision boundary
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    # Plot the decision boundary
                                    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
                                    
                                    # Plot the training points
                                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
                                    
                                    # Add legend
                                    if target_mapping:
                                        reverse_mapping = {v: k for k, v in target_mapping.items()}
                                        class_names = [reverse_mapping.get(i, str(i)) for i in sorted(np.unique(y))]
                                        legend = ax.legend(handles=scatter.legend_elements()[0], labels=class_names, 
                                                         loc="upper right", title="Classes")
                                    else:
                                        legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
                                    
                                    ax.add_artist(legend)
                                    
                                    ax.set_xlim(xx.min(), xx.max())
                                    ax.set_ylim(yy.min(), yy.max())
                                    ax.set_xlabel(X.columns[0])
                                    ax.set_ylabel(X.columns[1])
                                    ax.set_title(f'SVM Decision Boundary (kernel: {kernel}, C: {C})')
                                    st.pyplot(fig)
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom SVM"
                            st.session_state.models[model_name] = {
                                'model': model,
                                'scaler': scaler,  # Important to store the scaler with the model
                                'threshold': custom_threshold if len(np.unique(y)) == 2 and probability else None
                            }
                            
                            # Store in report data
                            if 'custom_models' not in st.session_state.report_data:
                                st.session_state.report_data['custom_models'] = {}
                            
                            st.session_state.report_data['custom_models'][model_name] = {
                                'accuracy': accuracy,
                                'balanced_accuracy': balanced_acc,
                                'report': report,
                                'params': {
                                    'C': C,
                                    'kernel': kernel,
                                    'gamma': gamma,
                                    'degree': degree,
                                    'coef0': coef0,
                                    'class_weight': class_weight,
                                    'probability': probability,
                                    'shrinking': shrinking,
                                    'decision_function_shape': decision_function_shape,
                                    'custom_threshold': custom_threshold if len(np.unique(y)) == 2 and probability else None
                                },
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Add download button for model
                            import pickle
                            model_pickle = pickle.dumps({'model': model, 'scaler': scaler, 'threshold': custom_threshold})
                            st.download_button(
                                label="Download Model",
                                data=model_pickle,
                                file_name="svm_model.pkl",
                                mime="application/octet-stream"
                            )
                            
                    except Exception as e:
                        st.error(f"Error training custom SVM: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            else:  # Regression (numeric or time)
                # SVR params
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svr_kernel")
                C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0, 0.1, key="svr_c")
                epsilon = st.slider("Epsilon in the epsilon-SVR model", 0.01, 1.0, 0.1, 0.01, key="svr_epsilon")
                
                # Advanced options toggle
                show_advanced = st.checkbox("Show advanced options", key="svr_adv")
                if show_advanced:
                    gamma_options = ["scale", "auto", "custom"]
                    gamma_choice = st.selectbox("Gamma", gamma_options, key="svr_gamma_choice")
                    
                    if gamma_choice == "custom":
                        gamma = st.slider("Custom gamma value", 0.001, 10.0, 0.1, 0.001, key="svr_gamma_custom")
                    else:
                        gamma = gamma_choice
                        
                    degree = st.slider("Polynomial degree", 2, 10, 3, 1, key="svr_degree") if kernel == "poly" else 3
                    coef0 = st.slider("Coefficient (coef0)", 0.0, 10.0, 0.0, 0.1, key="svr_coef0") if kernel in ["poly", "sigmoid"] else 0.0
                    
                    shrinking = st.checkbox("Use shrinking heuristic", value=True, key="svr_shrinking")
                    
                    cache_size = st.slider("Cache size (MB)", 100, 2000, 200, 100, key="svr_cache")
                    max_iter = st.slider("Maximum iterations", -1, 10000, -1, 100, key="svr_max_iter")
                    tol = st.slider("Tolerance", 0.00001, 0.001, 0.0001, 0.00001, format="%.5f", key="svr_tol")
                else:
                    gamma = "scale"
                    degree = 3
                    coef0 = 0.0
                    shrinking = True
                    cache_size = 200
                    max_iter = -1
                    tol = 0.0001
                    
                # Training button
                if st.button("Train Custom SVR", key="train_custom_svr"):
                    try:
                        with st.spinner("Training custom SVR..."):
                            from sklearn.svm import SVR
                            from sklearn.model_selection import train_test_split
                            from sklearn.preprocessing import StandardScaler
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Scale features (important for SVR)
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Create and train model
                            model = SVR(
                                C=C,
                                kernel=kernel,
                                degree=degree,
                                gamma=gamma,
                                coef0=coef0,
                                epsilon=epsilon,
                                shrinking=shrinking,
                                tol=tol,
                                cache_size=cache_size,
                                max_iter=max_iter
                            )
                            
                            model.fit(X_train_scaled, y_train)
                            
                            # Evaluate model
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            y_pred = model.predict(X_test_scaled)
                            
                            # Calculate metrics
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
                            st.markdown("### Actual vs Predicted Values")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            plt.scatter(y_test, y_pred, alpha=0.5)
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            
                            plt.title('Actual vs Predicted')
                            plt.xlabel('Actual')
                            plt.ylabel('Predicted')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Error analysis
                            st.markdown("### Error Analysis")
                            
                            # Calculate residuals
                            residuals = y_test - y_pred
                            
                            # Plot residuals
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                            
                            # Residuals vs Predicted
                            ax1.scatter(y_pred, residuals, alpha=0.5)
                            ax1.axhline(y=0, color='r', linestyle='--')
                            ax1.set_title('Residuals vs Predicted')
                            ax1.set_xlabel('Predicted Values')
                            ax1.set_ylabel('Residuals')
                            
                            # Histogram of residuals
                            ax2.hist(residuals, bins=20, alpha=0.7)
                            ax2.axvline(x=0, color='r', linestyle='--')
                            ax2.set_title('Residual Distribution')
                            ax2.set_xlabel('Residual Value')
                            ax2.set_ylabel('Frequency')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # If 2D data, visualize the SVR model
                            if X.shape[1] == 2:
                                st.markdown("### 3D Visualization")
                                
                                # Create mesh grid
                                x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                                y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
                                
                                # Scale mesh grid
                                mesh_points = np.c_[xx.ravel(), yy.ravel()]
                                mesh_points_scaled = scaler.transform(mesh_points)
                                
                                # Predict on mesh grid
                                zz = model.predict(mesh_points_scaled).reshape(xx.shape)
                                
                                # Create 3D plot
                                fig = plt.figure(figsize=(10, 8))
                                ax = fig.add_subplot(111, projection='3d')
                                
                                # Plot surface
                                surf = ax.plot_surface(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.7, linewidth=0, antialiased=True)
                                
                                # Plot training points
                                ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], y_train, c='k', s=20, alpha=0.5)
                                
                                # Labels and title
                                ax.set_xlabel(X.columns[0])
                                ax.set_ylabel(X.columns[1])
                                ax.set_zlabel('Target')
                                ax.set_title(f'SVR Model (kernel: {kernel}, C: {C}, epsilon: {epsilon})')
                                
                                # Add colorbar
                                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                                
                                st.pyplot(fig)
                            
                            # Permutation feature importance
                            st.markdown("### Feature Importance (Permutation)")
                            
                            try:
                                from sklearn.inspection import permutation_importance
                                
                                # Calculate permutation importance
                                result = permutation_importance(
                                    model, X_test_scaled, y_test, 
                                    n_repeats=10, random_state=42, n_jobs=-1
                                )
                                
                                # Create DataFrame
                                importance_df = pd.DataFrame({
                                    'feature': X.columns,
                                    'importance': result.importances_mean,
                                    'std': result.importances_std
                                }).sort_values('importance', ascending=False)
                                
                                st.dataframe(importance_df)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                importance_df.sort_values('importance', ascending=True).plot(
                                    kind='barh', x='feature', y='importance', xerr='std', ax=ax
                                )
                                
                                plt.title('Feature Importance (Permutation)')
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                            except Exception as imp_err:
                                st.warning(f"Could not calculate permutation importance: {str(imp_err)}")
                            
                            # Store model
                            if 'models' not in st.session_state:
                                st.session_state.models = {}
                            
                            model_name = "Custom SVR"
                            st.session_state.models[model_name] = {
                                'model': model,
                                'scaler': scaler  # Important to store the scaler with the model
                            }
                            
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
                                    'coef0': coef0,
                                    'epsilon': epsilon,
                                    'shrinking': shrinking
                                },
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Add download button for model
                            import pickle
                            model_pickle = pickle.dumps({'model': model, 'scaler': scaler})
                            st.download_button(
                                label="Download Model",
                                data=model_pickle,
                                file_name="svr_model.pkl",
                                mime="application/octet-stream"
                            )
                            
                    except Exception as e:
                        st.error(f"Error training custom SVR: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        with model_tabs[2]:  # Neural Network tab
            # NEURAL NETWORK CONFIGURATION
            st.markdown("### Neural Network Configuration")
            
            # Common parameters for both classification and regression
            st.markdown("#### Network Architecture")
            
            # Number of layers and neurons
            num_hidden_layers = st.slider("Number of hidden layers", 1, 5, 2, 1, key="nn_layers")
            
            # Create inputs for each layer
            hidden_layers = []
            for i in range(num_hidden_layers):
                neurons = st.slider(f"Neurons in hidden layer {i+1}", 4, 256, 
                                  64 if i == 0 else max(4, 64 // (2**i)), 4, key=f"nn_layer_{i}")
                hidden_layers.append(neurons)
            
            # Activation functions
            activation_options = ["relu", "tanh", "sigmoid", "elu", "selu"]
            hidden_activation = st.selectbox("Hidden layer activation", activation_options, index=0, key="nn_hidden_act")
            
            if target_type == 'categorical':
                # For classification, output activation depends on number of classes
                if len(np.unique(y)) == 2:
                    output_activation = "sigmoid"  # Binary classification
                    st.info("Using sigmoid activation for binary classification output")
                else:
                    output_activation = "softmax"  # Multi-class classification
                    st.info("Using softmax activation for multi-class classification output")
            else:
                # For regression
                output_activation = "linear"
                st.info("Using linear activation for regression output")
            
            # Learning parameters
            st.markdown("#### Training Parameters")
            
            learning_rate = st.select_slider(
                "Learning rate",
                options=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                value=0.001,
                key="nn_lr"
            )
            
            batch_size = st.select_slider(
                "Batch size",
                options=[8, 16, 32, 64, 128, 256],
                value=32,
                key="nn_batch"
            )
            
            epochs = st.slider("Maximum epochs", 10, 300, 100, 10, key="nn_epochs")
            patience = st.slider("Early stopping patience", 3, 50, 10, 1, 
                               help="Number of epochs with no improvement after which training will stop", 
                               key="nn_patience")
            
            # Regularization options
            st.markdown("#### Regularization")
            
            dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.05, 
                                   help="Fraction of neurons to randomly disable during training", 
                                   key="nn_dropout")
            
            use_batch_norm = st.checkbox("Use Batch Normalization", value=True,
                                       help="Normalize layer inputs to improve training stability",
                                       key="nn_batch_norm")
            
            regularization_options = ["None", "L1", "L2", "L1L2"]
            regularization_type = st.selectbox("Regularization type", regularization_options, index=1, key="nn_reg_type")
            
            if regularization_type != "None":
                reg_strength = st.select_slider(
                    "Regularization strength",
                    options=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                    value=0.001,
                    key="nn_reg_strength"
                )
            else:
                reg_strength = 0.0
            
            # Imbalanced data handling for classification
            if target_type == 'categorical' and 'handle_imbalance' in locals() and handle_imbalance:
                st.markdown("#### Imbalanced Data Handling")
                
                class_weight_strategy = st.selectbox(
                    "Class weight strategy",
                    ["balanced", "custom", "focal_loss"],
                    help="""
                    balanced: Automatically calculate weights inversely proportional to class frequencies
                    custom: Manually specify weights for each class
                    focal_loss: Use focal loss (reduces weight of easy examples)
                    """,
                    key="nn_class_weight"
                )
                
                if class_weight_strategy == "custom":
                    # Custom class weights input
                    st.markdown("##### Custom Class Weights")
                    class_weights_dict = {}
                    
                    # If we have class labels available, use them for better UX
                    if target_mapping:
                        reverse_mapping = {v: k for k, v in target_mapping.items()}
                        for class_idx in range(len(reverse_mapping)):
                            original_label = reverse_mapping.get(class_idx, f"Class {class_idx}")
                            weight = st.number_input(
                                f"Weight for {original_label}",
                                min_value=0.1,
                                max_value=100.0,
                                value=1.0 if class_idx == 0 else 5.0,  # Default higher weight for minority class
                                step=0.1,
                                key=f"nn_class_weight_{class_idx}"
                            )
                            class_weights_dict[class_idx] = weight
                    else:
                        # If no mapping available, use generic class names
                        num_classes = len(pd.Series(y).unique())
                        for class_idx in range(num_classes):
                            weight = st.number_input(
                                f"Weight for Class {class_idx}",
                                min_value=0.1,
                                max_value=100.0,
                                value=1.0 if class_idx == 0 else 5.0,
                                step=0.1,
                                key=f"nn_class_weight_{class_idx}"
                            )
                            class_weights_dict[class_idx] = weight
                
                elif class_weight_strategy == "focal_loss":
                    st.info("""
                    Focal Loss reduces the loss contribution from easy examples and focuses on difficult ones.
                    It's particularly effective for imbalanced datasets.
                    """)
                    
                    # Parameters for focal loss
                    focal_gamma = st.slider("Focal Loss gamma", 0.5, 5.0, 2.0, 0.1,
                                          help="Higher values focus more on difficult examples",
                                          key="nn_focal_gamma")
                    focal_alpha = st.slider("Focal Loss alpha (class 1 weight)", 0.1, 0.9, 0.25, 0.05,
                                          help="Weight for the positive class (higher values focus more on minority class)",
                                          key="nn_focal_alpha")
            
            # Advanced options
            show_advanced = st.checkbox("Show advanced options", key="nn_adv")
            if show_advanced:
                st.markdown("#### Advanced Options")
                
                # Optimizer selection
                optimizer_options = ["adam", "sgd", "rmsprop", "adagrad", "adadelta"]
                optimizer_choice = st.selectbox("Optimizer", optimizer_options, index=0, key="nn_optimizer")
                
                if optimizer_choice == "sgd":
                    momentum = st.slider("Momentum", 0.0, 0.99, 0.9, 0.01, key="nn_momentum")
                    nesterov = st.checkbox("Use Nesterov momentum", value=True, key="nn_nesterov")
                
                # Learning rate scheduler
                use_lr_scheduler = st.checkbox("Use learning rate scheduler", value=False, key="nn_use_lr")
                if use_lr_scheduler:
                    lr_scheduler_options = ["reduce_on_plateau", "exponential_decay", "step_decay"]
                    lr_scheduler = st.selectbox("Scheduler type", lr_scheduler_options, key="nn_lr_scheduler")
                    
                    if lr_scheduler == "reduce_on_plateau":
                        lr_factor = st.slider("Reduction factor", 0.1, 0.9, 0.5, 0.1, key="nn_lr_factor")
                        lr_patience = st.slider("LR patience", 1, 10, 3, 1, key="nn_lr_patience")
                    elif lr_scheduler == "exponential_decay":
                        lr_decay = st.slider("Decay rate", 0.5, 0.99, 0.9, 0.01, key="nn_lr_decay")
                    elif lr_scheduler == "step_decay":
                        lr_drop = st.slider("Drop rate", 0.1, 0.9, 0.5, 0.1, key="nn_lr_drop")
                        lr_epochs_drop = st.slider("Epochs between drops", 1, 20, 10, 1, key="nn_lr_epochs_drop")
                
                # Monitor options
                monitor_metric = st.selectbox(
                    "Monitoring metric for early stopping",
                    ["val_loss", "val_accuracy", "val_mae"] if target_type != 'categorical' else ["val_loss", "val_accuracy"],
                    key="nn_monitor"
                )
                
                # Validation split
                validation_split = st.slider("Validation split", 0.1, 0.3, 0.2, 0.05, key="nn_val_split")
            else:
                # Default values for advanced options
                optimizer_choice = "adam"
                momentum = 0.9
                nesterov = True
                use_lr_scheduler = False
                lr_scheduler = "reduce_on_plateau"
                lr_factor = 0.5
                lr_patience = 3
                lr_decay = 0.9
                lr_drop = 0.5
                lr_epochs_drop = 10
                monitor_metric = "val_loss"
                validation_split = 0.2
            
            # Training button
            if st.button("Train Custom Neural Network", key="train_custom_nn"):
                try:
                    with st.spinner("Training custom Neural Network..."):
                        import tensorflow as tf
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
                        from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
                        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                        from tensorflow.keras.regularizers import l1, l2, l1_l2
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import StandardScaler
                        
                        # Check for GPU availability
                        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
                        if gpu_available:
                            st.info("Training on GPU")
                        else:
                            st.info("Training on CPU")
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Handle imbalanced data if requested
                        if target_type == 'categorical' and 'handle_imbalance' in locals() and handle_imbalance:
                            if 'imbalance_method' in locals() and imbalance_method in ["smote", "adasyn", "random_over", "random_under"]:
                                # Import the appropriate resampling method
                                if imbalance_method == "smote":
                                    from imblearn.over_sampling import SMOTE
                                    resampler = SMOTE(random_state=42)
                                elif imbalance_method == "adasyn":
                                    from imblearn.over_sampling import ADASYN
                                    resampler = ADASYN(random_state=42)
                                elif imbalance_method == "random_over":
                                    from imblearn.over_sampling import RandomOverSampler
                                    resampler = RandomOverSampler(random_state=42)
                                elif imbalance_method == "random_under":
                                    from imblearn.under_sampling import RandomUnderSampler
                                    resampler = RandomUnderSampler(random_state=42)
                                
                                # Apply resampling
                                X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train)
                                
                                # Show resampling results
                                original_class_counts = pd.Series(y_train).value_counts()
                                resampled_class_counts = pd.Series(y_train_resampled).value_counts()
                                st.info(f"Resampling changed class distribution from {dict(original_class_counts)} to {dict(resampled_class_counts)}")
                                
                                # Use resampled data for training
                                X_train_scaled = X_train_resampled
                                y_train = y_train_resampled
                        
                        # Set regularizer based on selection
                        if regularization_type == "L1":
                            regularizer = l1(reg_strength)
                        elif regularization_type == "L2":
                            regularizer = l2(reg_strength)
                        elif regularization_type == "L1L2":
                            regularizer = l1_l2(l1=reg_strength, l2=reg_strength)
                        else:
                            regularizer = None
                        
                        # Set optimizer
                        if optimizer_choice == "adam":
                            optimizer = Adam(learning_rate=learning_rate)
                        elif optimizer_choice == "sgd":
                            optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
                        elif optimizer_choice == "rmsprop":
                            optimizer = RMSprop(learning_rate=learning_rate)
                        elif optimizer_choice == "adagrad":
                            optimizer = Adagrad(learning_rate=learning_rate)
                        elif optimizer_choice == "adadelta":
                            optimizer = Adadelta(learning_rate=learning_rate)
                        
                        # Create the model
                        model = Sequential()
                        
                        # Input layer
                        model.add(Dense(
                            hidden_layers[0], 
                            activation=hidden_activation, 
                            kernel_regularizer=regularizer,
                            input_shape=(X_train_scaled.shape[1],)
                        ))
                        
                        if use_batch_norm:
                            model.add(BatchNormalization())
                        
                        if dropout_rate > 0:
                            model.add(Dropout(dropout_rate))
                        
                        # Hidden layers
                        for i in range(1, len(hidden_layers)):
                            model.add(Dense(
                                hidden_layers[i], 
                                activation=hidden_activation,
                                kernel_regularizer=regularizer
                            ))
                            
                            if use_batch_norm:
                                model.add(BatchNormalization())
                            
                            if dropout_rate > 0:
                                model.add(Dropout(dropout_rate))
                        
                        # Output layer
                        if target_type == 'categorical':
                            if len(np.unique(y_train)) == 2:
                                # Binary classification
                                model.add(Dense(1, activation=output_activation))
                                loss = 'binary_crossentropy'
                            else:
                                # Multi-class classification
                                model.add(Dense(len(np.unique(y_train)), activation=output_activation))
                                loss = 'sparse_categorical_crossentropy'
                        else:
                            # Regression
                            model.add(Dense(1, activation=output_activation))
                            loss = 'mse'
                        
                        # For focal loss (binary classification with imbalanced data)
                        if (target_type == 'categorical' and len(np.unique(y_train)) == 2 and 
                            'class_weight_strategy' in locals() and class_weight_strategy == "focal_loss"):
                            
                            # Define focal loss function
                            def binary_focal_loss(gamma=2.0, alpha=0.25):
                                """
                                Binary form of focal loss.
                                
                                FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
                                where p_t is the model's predicted probability for the positive class.
                                
                                Parameters:
                                    gamma: focusing parameter for difficult examples (higher values focus more on hard examples)
                                    alpha: balancing parameter for class imbalance (higher values give more weight to minority class)
                                """
                                gamma = tf.constant(gamma, dtype=tf.float32)
                                alpha = tf.constant(alpha, dtype=tf.float32)
                                
                                def binary_focal_loss_fixed(y_true, y_pred):
                                    y_true = tf.cast(y_true, tf.float32)
                                    # Define epsilon to avoid numerical instability issues
                                    epsilon = tf.keras.backend.epsilon()
                                    # Clip predictions to avoid log(0) errors
                                    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
                                    
                                    # Calculate the focal loss
                                    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
                                    focal_weight = tf.pow(1 - p_t, gamma)
                                    
                                    # Apply alpha weighting
                                    alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
                                    
                                    # Calculate cross entropy
                                    cross_entropy = -tf.math.log(p_t)
                                    
                                    # Calculate focal loss
                                    loss = alpha_factor * focal_weight * cross_entropy
                                    return tf.reduce_mean(loss)
                                
                                return binary_focal_loss_fixed
                            
                            # Use focal loss
                            loss = binary_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
                            st.info(f"Using focal loss with gamma={focal_gamma}, alpha={focal_alpha}")
                        
                        # Set up class weights for imbalanced data
                        if (target_type == 'categorical' and 'class_weight_strategy' in locals() and 
                            class_weight_strategy in ["balanced", "custom"]):
                            
                            if class_weight_strategy == "balanced":
                                # Calculate balanced class weights
                                classes = np.unique(y_train)
                                # Compute weights inversely proportional to class frequencies
                                class_counts = np.bincount(y_train.astype(int))
                                total_samples = np.sum(class_counts)
                                class_weights_dict = {i: total_samples / (len(classes) * count) 
                                                   for i, count in enumerate(class_counts)}
                                st.info(f"Using balanced class weights: {class_weights_dict}")
                            
                            # Note: custom class weights are already set by the UI
                        else:
                            class_weights_dict = None
                        
                        # Compile the model
                        model.compile(
                            optimizer=optimizer,
                            loss=loss,
                            metrics=['accuracy'] if target_type == 'categorical' else ['mae', 'mse']
                        )
                        
                        # Display model summary
                        st.markdown("### Model Architecture")
                        model_summary = []
                        model.summary(print_fn=lambda x: model_summary.append(x))
                        st.code('\n'.join(model_summary))
                        
                        # Set up callbacks
                        callbacks = []
                        
                        # Early stopping
                        early_stopping = EarlyStopping(
                            monitor=monitor_metric,
                            patience=patience,
                            restore_best_weights=True,
                            verbose=1
                        )
                        callbacks.append(early_stopping)
                        
                        # Learning rate scheduler
                        if use_lr_scheduler:
                            if lr_scheduler == "reduce_on_plateau":
                                lr_scheduler_cb = ReduceLROnPlateau(
                                    monitor=monitor_metric,
                                    factor=lr_factor,
                                    patience=lr_patience,
                                    verbose=1,
                                    min_delta=0.0001,
                                    min_lr=0.00001
                                )
                                callbacks.append(lr_scheduler_cb)
                            elif lr_scheduler == "exponential_decay":
                                # Create a callback that adjusts the learning rate exponentially
                                def exponential_decay(epoch):
                                    return learning_rate * (lr_decay ** epoch)
                                
                                lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(exponential_decay)
                                callbacks.append(lr_scheduler_cb)
                            elif lr_scheduler == "step_decay":
                                # Create a callback that adjusts the learning rate in steps
                                def step_decay(epoch):
                                    initial_lrate = learning_rate
                                    drop = lr_drop
                                    epochs_drop = lr_epochs_drop
                                    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
                                    return lrate
                                
                                lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(step_decay)
                                callbacks.append(lr_scheduler_cb)
                        
                        # Add progress bar (custom callback for Streamlit)
                        progress_bar = st.progress(0)
                        epoch_status = st.empty()
                        
                        class ProgressBarCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / epochs
                                progress_bar.progress(min(progress, 1.0))
                                metrics_text = ", ".join(f"{k}: {v:.4f}" for k, v in logs.items())
                                epoch_status.text(f"Epoch {epoch+1}/{epochs} - {metrics_text}")
                        
                        callbacks.append(ProgressBarCallback())
                        
                        # Train the model
                        history = model.fit(
                            X_train_scaled, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            verbose=0,
                            callbacks=callbacks,
                            class_weight=class_weights_dict
                        )
                        
                        # Reset progress display
                        progress_bar.empty()
                        epoch_status.empty()
                        
                        # Show training history
                        st.markdown("### Training History")
                        
                        # Convert history to DataFrame for easier plotting
                        history_df = pd.DataFrame(history.history)
                        
                        # Determine which metrics to plot
                        if target_type == 'categorical':
                            # Classification metrics
                            metric_list = ['loss', 'accuracy']
                            if 'val_loss' in history_df.columns:
                                metric_list.extend(['val_loss', 'val_accuracy'])
                        else:
                            # Regression metrics
                            metric_list = ['loss', 'mae', 'mse']
                            if 'val_loss' in history_df.columns:
                                metric_list.extend(['val_loss', 'val_mae', 'val_mse'])
                        
                        # Create subplots based on available metrics
                        num_plots = len(metric_list) // 2 + len(metric_list) % 2
                        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
                        
                        # Ensure axes is always a list-like object
                        if num_plots == 1:
                            axes = [axes]
                        
                        # Plot each metric
                        for i, metric in enumerate(metric_list):
                            ax_idx = i // 2
                            ax = axes[ax_idx]
                            
                            if 'val_' in metric:
                                continue  # Skip validation metrics as they'll be plotted with their training counterparts
                            
                            # Plot training metric
                            ax.plot(history_df.index, history_df[metric], label=f'Training {metric}')
                            
                            # Plot validation metric if available
                            val_metric = f'val_{metric}'
                            if val_metric in history_df.columns:
                                ax.plot(history_df.index, history_df[val_metric], label=f'Validation {metric}')
                            
                            ax.set_title(f'{metric.capitalize()} over epochs')
                            ax.set_xlabel('Epoch')
                            ax.set_ylabel(metric.capitalize())
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Evaluate the model
                        if target_type == 'categorical':
                            # Classification evaluation
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
                            
                            # Get predictions based on model type
                            if len(np.unique(y_train)) == 2:
                                # Binary classification
                                y_pred_proba = model.predict(X_test_scaled).flatten()
                                y_pred = (y_pred_proba > 0.5).astype('int32')
                            else:
                                # Multi-class classification
                                y_pred_proba = model.predict(X_test_scaled)
                                y_pred = np.argmax(y_pred_proba, axis=1)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            balanced_acc = balanced_accuracy_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Neural Network trained with accuracy: {accuracy:.4f}, balanced accuracy: {balanced_acc:.4f}")
                            
                            # Classification report
                            st.markdown("### Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            
                            # If we have target mapping, use it to convert encoded class labels to original names
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                
                                # Create a new report with original category names
                                new_report = {}
                                for key, val in report.items():
                                    if key.isdigit() or (isinstance(key, (int, float)) and int(key) == key):
                                        # This is a class label - convert it
                                        new_key = reverse_mapping.get(int(key), str(key))
                                        new_report[new_key] = val
                                    else:
                                        # This is a metric like 'accuracy', 'macro avg', etc.
                                        new_report[key] = val
                                report = new_report
                            
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Confusion matrix
                            st.markdown("### Confusion Matrix")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            
                            # Get class names for the confusion matrix
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                class_names = [reverse_mapping.get(i, str(i)) for i in range(len(np.unique(y)))]
                                
                                # Plot confusion matrix with original class names
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=class_names, yticklabels=class_names)
                            else:
                                # Default behavior without mapping
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # For binary classification, plot ROC and PR curves
                            if len(np.unique(y_test)) == 2:
                                from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
                                
                                st.markdown("### ROC and Precision-Recall Curves")
                                
                                # Create a figure with two subplots
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # ROC curve
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                roc_auc = auc(fpr, tpr)
                                
                                ax1.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                                ax1.plot([0, 1], [0, 1], 'k--')
                                ax1.set_xlim([0.0, 1.0])
                                ax1.set_ylim([0.0, 1.05])
                                ax1.set_xlabel('False Positive Rate')
                                ax1.set_ylabel('True Positive Rate')
                                ax1.set_title('Receiver Operating Characteristic (ROC)')
                                ax1.legend(loc="lower right")
                                
                                # Precision-Recall curve (better for imbalanced data)
                                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                                pr_auc = average_precision_score(y_test, y_pred_proba)
                                
                                ax2.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
                                ax2.set_xlim([0.0, 1.0])
                                ax2.set_ylim([0.0, 1.05])
                                ax2.set_xlabel('Recall')
                                ax2.set_ylabel('Precision')
                                ax2.set_title('Precision-Recall Curve (Better for Imbalanced Data)')
                                ax2.legend(loc="lower left")
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                        
                        else:  # Regression
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            # Get predictions
                            y_pred = model.predict(X_test_scaled).flatten()
                            
                            # Calculate metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Display results
                            st.success(f"Neural Network Regressor trained with R² Score: {r2:.4f}")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("MSE", f"{mse:.4f}")
                            col2.metric("RMSE", f"{rmse:.4f}")
                            col3.metric("MAE", f"{mae:.4f}")
                            
                            # Plot actual vs predicted
                            st.markdown("### Actual vs Predicted Values")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            plt.scatter(y_test, y_pred, alpha=0.5)
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            
                            plt.title('Actual vs Predicted')
                            plt.xlabel('Actual')
                            plt.ylabel('Predicted')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Error analysis
                            st.markdown("### Error Analysis")
                            
                            # Calculate residuals
                            residuals = y_test - y_pred
                            
                            # Plot residuals
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                            
                            # Residuals vs Predicted
                            ax1.scatter(y_pred, residuals, alpha=0.5)
                            ax1.axhline(y=0, color='r', linestyle='--')
                            ax1.set_title('Residuals vs Predicted')
                            ax1.set_xlabel('Predicted Values')
                            ax1.set_ylabel('Residuals')
                            
                            # Histogram of residuals
                            ax2.hist(residuals, bins=20, alpha=0.7)
                            ax2.axvline(x=0, color='r', linestyle='--')
                            ax2.set_title('Residual Distribution')
                            ax2.set_xlabel('Residual Value')
                            ax2.set_ylabel('Frequency')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Feature importance through permutation importance
                        st.markdown("### Feature Importance (Permutation)")
                        
                        try:
                            from sklearn.inspection import permutation_importance
                            
                            # Create a wrapper for the keras model
                            class KerasModelWrapper:
                                def __init__(self, model, is_binary=False):
                                    self.model = model
                                    self.is_binary = is_binary
                                    
                                def predict(self, X):
                                    preds = self.model.predict(X)
                                    if self.is_binary:
                                        return preds.flatten()
                                    return preds
                            
                            # Determine if binary classification
                            is_binary = target_type == 'categorical' and len(np.unique(y)) == 2
                            
                            # Create the wrapper
                            model_wrapper = KerasModelWrapper(model, is_binary=is_binary)
                            
                            # Calculate permutation importance
                            scoring = 'balanced_accuracy' if target_type == 'categorical' else 'r2'
                            result = permutation_importance(
                                model_wrapper, X_test_scaled, y_test, 
                                n_repeats=10, random_state=42, n_jobs=-1,
                                scoring=scoring
                            )
                            
                            # Create DataFrame
                            importance_df = pd.DataFrame({
                                'feature': X.columns,
                                'importance': result.importances_mean,
                                'std': result.importances_std
                            }).sort_values('importance', ascending=False)
                            
                            st.dataframe(importance_df)
                            
                            # Plot feature importance
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            importance_df.sort_values('importance', ascending=True).plot(
                                kind='barh', x='feature', y='importance', xerr='std', ax=ax
                            )
                            
                            plt.title('Feature Importance (Permutation)')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                        except Exception as imp_err:
                            st.warning(f"Could not calculate permutation importance: {str(imp_err)}")
                        
                        # Store model in session state
                        if 'models' not in st.session_state:
                            st.session_state.models = {}
                        
                        model_name = "Custom Neural Network"
                        
                        # For neural networks, we need to store the model and preprocessing info
                        model_info = {
                            'model': model,
                            'scaler': scaler,
                            'type': 'classifier' if target_type == 'categorical' else 'regressor',
                            'binary': len(np.unique(y)) == 2 if target_type == 'categorical' else False
                        }
                        
                        st.session_state.models[model_name] = model_info
                        
                        # Store in report data
                        if 'custom_models' not in st.session_state.report_data:
                            st.session_state.report_data['custom_models'] = {}
                        
                        if target_type == 'categorical':
                            st.session_state.report_data['custom_models'][model_name] = {
                                'accuracy': accuracy,
                                'balanced_accuracy': balanced_acc,
                                'report': report,
                                'params': {
                                    'hidden_layers': hidden_layers,
                                    'activation': hidden_activation,
                                    'dropout_rate': dropout_rate,
                                    'batch_normalization': use_batch_norm,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'epochs_trained': len(history.history['loss']),
                                    'regularization': {
                                        'type': regularization_type,
                                        'strength': reg_strength
                                    },
                                    'class_weight_strategy': class_weight_strategy if 'class_weight_strategy' in locals() else None
                                },
                                'history': history.history,
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        else:
                            st.session_state.report_data['custom_models'][model_name] = {
                                'mse': mse,
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2,
                                'params': {
                                    'hidden_layers': hidden_layers,
                                    'activation': hidden_activation,
                                    'dropout_rate': dropout_rate,
                                    'batch_normalization': use_batch_norm,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'epochs_trained': len(history.history['loss']),
                                    'regularization': {
                                        'type': regularization_type,
                                        'strength': reg_strength
                                    }
                                },
                                'history': history.history,
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        
                        # Add option to save model
                        st.markdown("### Save Model")
                        
                        save_format = st.selectbox(
                            "Save format", 
                            ["Keras H5", "TensorFlow SavedModel"],
                            help="H5 is more compact, SavedModel is more complete"
                        )
                        
                        save_path = st.text_input("Model filename", f"custom_nn_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                        
                        if st.button("Save Model"):
                            try:
                                if save_format == "Keras H5":
                                    if not save_path.endswith('.h5'):
                                        save_path += '.h5'
                                    model.save(save_path)
                                    
                                    # Create a downloadable model file
                                    with open(save_path, 'rb') as f:
                                        model_bytes = f.read()
                                    
                                    st.download_button(
                                        label="Download Model",
                                        data=model_bytes,
                                        file_name=os.path.basename(save_path),
                                        mime="application/octet-stream"
                                    )
                                else:  # TensorFlow SavedModel
                                    if not save_path.endswith('/'):
                                        save_path += '/'
                                    model.save(save_path)
                                    
                                    # Create a zip of the saved model directory
                                    import shutil
                                    zip_path = save_path.rstrip('/') + '.zip'
                                    shutil.make_archive(save_path.rstrip('/'), 'zip', save_path)
                                    
                                    # Create a downloadable zip file
                                    with open(zip_path, 'rb') as f:
                                        model_bytes = f.read()
                                    
                                    st.download_button(
                                        label="Download Model",
                                        data=model_bytes,
                                        file_name=os.path.basename(zip_path),
                                        mime="application/zip"
                                    )
                                
                                st.success(f"Model saved successfully to {save_path}")
                            except Exception as save_err:
                                st.error(f"Error saving model: {str(save_err)}")
                        
                        # Add option to visualize network architecture
                        if st.button("Visualize Network Architecture"):
                            try:
                                # Check if tensorflow.keras.utils.plot_model is available
                                from tensorflow.keras.utils import plot_model
                                import io
                                from PIL import Image
                                
                                # Create a temporary file for the model visualization
                                buffer = io.BytesIO()
                                plot_model(
                                    model, 
                                    to_file=buffer, 
                                    show_shapes=True, 
                                    show_layer_names=True,
                                    rankdir='TB'
                                )
                                
                                # Display the image
                                buffer.seek(0)
                                image = Image.open(buffer)
                                st.image(image, caption="Neural Network Architecture")
                            except ImportError:
                                st.warning("Network visualization requires additional dependencies: pydot and graphviz")
                                st.info("Install with: pip install pydot graphviz")
                            except Exception as viz_err:
                                st.error(f"Error visualizing network: {str(viz_err)}")
                        
                except Exception as e:
                    st.error(f"Error training Neural Network: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    with tab3:
        st.markdown("<div class='subheader'>Model Evaluation</div>", unsafe_allow_html=True)
        
        # Initialize evaluation state in session_state if not present
        if 'evaluation_state' not in st.session_state:
            st.session_state.evaluation_state = {
                'threshold_applied': False,
                'threshold_value': 0.5,
                'permutation_calculated': False,
                'selected_model': None,
                'evaluation_performed': False
            }
        
        # Add explanation of metrics
        with st.expander("📊 Understanding Model Evaluation Metrics", expanded=True):
            st.markdown("""
            ### Classification Metrics
            - **Accuracy**: The proportion of correct predictions among the total number of predictions (both true positives and true negatives). _Higher is better_.
            - **Balanced Accuracy**: Average of recall obtained on each class. Better metric for imbalanced datasets. _Higher is better_.
            - **Precision**: The proportion of true positives among all positive predictions. Measures how many of the predicted positives are actually positive. _Higher is better_.
            - **Recall (Sensitivity)**: The proportion of true positives among all actual positives. Measures how many of the actual positives were correctly identified. _Higher is better_.
            - **F1 Score**: The harmonic mean of precision and recall. Provides a balance between precision and recall. _Higher is better_.
            - **PR-AUC**: Area Under the Precision-Recall Curve. Better than ROC-AUC for imbalanced datasets. _Higher is better_.
            - **ROC-AUC**: Area Under the Receiver Operating Characteristic curve. Measures the ability to distinguish between classes. _Higher is better_.
            - **CV Accuracy**: Cross-validation accuracy, the average accuracy across multiple train-test splits. More robust than a single accuracy score. _Higher is better_.
            - **CV Std**: Standard deviation of cross-validation accuracy. Indicates consistency of model performance. _Lower is better_.
            ### For Imbalanced Data
            - Focus on **Balanced Accuracy**, **Recall**, and **PR-AUC** rather than standard accuracy
            - Monitor class-specific metrics to ensure minority class is being predicted properly
            ### Regression Metrics
            - **MSE (Mean Squared Error)**: Average of squared differences between predicted and actual values. Penalizes larger errors more. _Lower is better_.
            - **RMSE (Root Mean Squared Error)**: Square root of MSE. In the same units as the target variable. _Lower is better_.
            - **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values. Less sensitive to outliers than MSE. _Lower is better_.
            - **R² Score**: Proportion of variance in the target variable that is predictable from the features. Ranges from 0 to 1 (can be negative in bad models). _Higher is better_.
            - **CV R²**: Cross-validation R² score, the average R² across multiple train-test splits. _Higher is better_.
            - **CV Std**: Standard deviation of cross-validation R² scores. _Lower is better_.
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
        
        # Use the stored model if already evaluated, otherwise default to first model
        default_index = 0
        if st.session_state.evaluation_state['selected_model'] in model_names:
            default_index = model_names.index(st.session_state.evaluation_state['selected_model'])
        
        selected_model = st.selectbox(
            "Select model to evaluate",
            model_names,
            index=default_index
        )
        
        # Store selected model name in state
        st.session_state.evaluation_state['selected_model'] = selected_model
        
        # Get data from session state
        X = st.session_state.processed_data['X']
        y = st.session_state.processed_data['y']
        target_column = st.session_state.processed_data['target_column']
        target_type = st.session_state.target_type
        target_mapping = st.session_state.target_mapping if 'target_mapping' in st.session_state else None
        
        # Threshold adjustment for binary classification
        custom_threshold = None
        if target_type == 'categorical' and len(np.unique(y)) == 2:
            st.markdown("### Prediction Threshold Adjustment")
            st.info("For imbalanced datasets, adjusting the prediction threshold can improve minority class detection.")
            adjust_threshold = st.checkbox("Adjust prediction threshold", value=False)
            if adjust_threshold:
                custom_threshold = st.slider(
                    "Prediction threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.evaluation_state['threshold_value'],
                    step=0.05,
                    help="Lower values increase sensitivity (more minority class predictions)"
                )
                # Store threshold value in state
                st.session_state.evaluation_state['threshold_value'] = custom_threshold
        
        # Evaluate button with unique key
        evaluate_col1, evaluate_col2 = st.columns([1, 3])
        evaluate_button = evaluate_col1.button("Evaluate Model", key="evaluate_model_button")
        
        # Container for evaluation results
        evaluation_results_container = st.container()
        
        # Check if evaluation should be performed
        if evaluate_button or st.session_state.evaluation_state['evaluation_performed']:
            # Set state flag to indicate evaluation has been performed
            if evaluate_button:
                st.session_state.evaluation_state['evaluation_performed'] = True
                # Reset other flags when new evaluation is requested
                st.session_state.evaluation_state['threshold_applied'] = False
                st.session_state.evaluation_state['permutation_calculated'] = False
            
            with evaluation_results_container:
                try:
                    with st.spinner("Evaluating model..."):
                        # Get model
                        model = st.session_state.models[selected_model]
                        
                        # Check if model is a dictionary (for SVM and NN models that include scaler)
                        if isinstance(model, dict) and 'model' in model:
                            actual_model = model['model']
                            scaler = model.get('scaler', None)
                        else:
                            actual_model = model
                            scaler = None
                        
                        # Split data for evaluation
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Apply scaling if available
                        if scaler is not None:
                            X_test_transformed = scaler.transform(X_test)
                        else:
                            X_test_transformed = X_test
                        
                        # Check if this is a neural network model (TensorFlow/Keras)
                        is_neural_network = 'keras' in str(type(actual_model)).lower()
                        
                        # Evaluate model based on target type
                        if target_type == 'categorical':
                            # Classification evaluation
                            from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                                        roc_curve, auc, precision_recall_curve, average_precision_score,
                                                        precision_score, recall_score, f1_score, balanced_accuracy_score)
                            
                            # Calculate number of classes
                            n_classes = len(np.unique(y))
                            
                            # Store in session state for other functions
                            st.session_state.evaluation_state['n_classes'] = n_classes
                            
                            # Handle predictions based on model type
                            if is_neural_network:
                                # For neural networks
                                if n_classes == 2:  # Binary classification
                                    raw_predictions = actual_model.predict(X_test_transformed)
                                    y_pred_proba = raw_predictions.flatten()
                                    threshold_to_use = custom_threshold if custom_threshold is not None else 0.5
                                    y_pred = (y_pred_proba > threshold_to_use).astype(int)
                                    
                                    # Store predictions and probabilities in session state
                                    st.session_state.evaluation_state['y_test'] = y_test
                                    st.session_state.evaluation_state['y_pred'] = y_pred
                                    st.session_state.evaluation_state['y_pred_proba'] = y_pred_proba
                                    
                                    if custom_threshold is not None:
                                        st.info(f"Using custom threshold: {threshold_to_use}")
                                else:  # Multi-class classification
                                    raw_predictions = actual_model.predict(X_test_transformed)
                                    y_pred = np.argmax(raw_predictions, axis=1)
                                    y_pred_proba = None  # Not applicable for multi-class
                                    
                                    # Store predictions in session state
                                    st.session_state.evaluation_state['y_test'] = y_test
                                    st.session_state.evaluation_state['y_pred'] = y_pred
                            elif hasattr(actual_model, 'predict_proba') and custom_threshold is not None and n_classes == 2:
                                # For models with probability estimates and custom threshold
                                y_pred_proba = actual_model.predict_proba(X_test_transformed)[:, 1]
                                y_pred = (y_pred_proba > custom_threshold).astype(int)
                                
                                # Store predictions and probabilities in session state
                                st.session_state.evaluation_state['y_test'] = y_test
                                st.session_state.evaluation_state['y_pred'] = y_pred
                                st.session_state.evaluation_state['y_pred_proba'] = y_pred_proba
                                
                                st.info(f"Using custom threshold: {custom_threshold}")
                            else:
                                # Standard models
                                y_pred = actual_model.predict(X_test_transformed)
                                
                                # Store predictions in session state
                                st.session_state.evaluation_state['y_test'] = y_test
                                st.session_state.evaluation_state['y_pred'] = y_pred
                                
                                if hasattr(actual_model, 'predict_proba') and n_classes == 2:
                                    y_pred_proba = actual_model.predict_proba(X_test_transformed)[:, 1]
                                    # Store probabilities in session state
                                    st.session_state.evaluation_state['y_pred_proba'] = y_pred_proba
                                else:
                                    y_pred_proba = None
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            balanced_acc = balanced_accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            
                            # Store metrics in session state
                            st.session_state.evaluation_state['metrics'] = {
                                'accuracy': accuracy,
                                'balanced_acc': balanced_acc,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            }
                            
                            # Display metrics
                            st.markdown("### Overall Metrics")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Accuracy", f"{accuracy:.4f}")
                            col1.metric("Balanced Accuracy", f"{balanced_acc:.4f}")
                            col2.metric("Precision", f"{precision:.4f}")
                            col2.metric("Recall", f"{recall:.4f}")
                            col3.metric("F1 Score", f"{f1:.4f}")
                            
                            # Calculate and display class-specific metrics
                            st.markdown("### Class-Specific Metrics")
                            
                            # Generate classification report
                            report = classification_report(y_test, y_pred, output_dict=True)
                            
                            # Convert report to DataFrame with class names
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                
                                # Create a new report with original category names
                                new_report = {}
                                for key, val in report.items():
                                    if key.isdigit() or (isinstance(key, (int, float)) and key == int(key)):
                                        # This is a class label - convert it
                                        new_key = reverse_mapping.get(int(key), str(key))
                                        new_report[new_key] = val
                                    else:
                                        # This is a metric like 'accuracy', 'macro avg', etc.
                                        new_report[key] = val
                                report = new_report
                            
                            # Store report in session state
                            st.session_state.evaluation_state['report'] = report
                            
                            # Convert to DataFrame for display
                            report_df = pd.DataFrame(report).transpose()
                            
                            # Display report
                            st.dataframe(report_df)
                            
                            # Plot confusion matrix
                            st.markdown("### Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            
                            # Store confusion matrix in session state
                            st.session_state.evaluation_state['confusion_matrix'] = cm
                            
                            # Get class names for the confusion matrix
                            if target_mapping:
                                # Create reverse mapping (encoded value -> original category)
                                reverse_mapping = {v: k for k, v in target_mapping.items()}
                                class_names = [reverse_mapping.get(i, str(i)) for i in range(len(np.unique(y)))]
                                
                                # Store class names in session state
                                st.session_state.evaluation_state['class_names'] = class_names
                                
                                # Plot confusion matrix with original class names
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=class_names, yticklabels=class_names,
                                          cmap='Blues')
                            else:
                                st.session_state.evaluation_state['class_names'] = None
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
                            
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # For binary classification, add ROC and PR curves
                            if n_classes == 2 and y_pred_proba is not None:
                                st.markdown("### ROC and Precision-Recall Curves")
                                
                                # Create two columns for the plots
                                col1, col2 = st.columns(2)
                                
                                # ROC curve
                                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                                roc_auc = auc(fpr, tpr)
                                
                                # Store ROC data in session state
                                st.session_state.evaluation_state['roc_data'] = {
                                    'fpr': fpr,
                                    'tpr': tpr,
                                    'thresholds': thresholds,
                                    'roc_auc': roc_auc
                                }
                                
                                fig1, ax1 = plt.subplots(figsize=(8, 6))
                                ax1.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
                                ax1.plot([0, 1], [0, 1], 'k--')
                                
                                # If custom threshold was used, add point on ROC curve
                                if custom_threshold is not None:
                                    # Find index closest to custom threshold
                                    thresh_idx = np.argmin(np.abs(thresholds - custom_threshold))
                                    ax1.plot(fpr[thresh_idx], tpr[thresh_idx], 'ro', markersize=8,
                                            label=f'Threshold = {custom_threshold:.2f}')
                                
                                ax1.set_xlim([0.0, 1.0])
                                ax1.set_ylim([0.0, 1.05])
                                ax1.set_xlabel('False Positive Rate')
                                ax1.set_ylabel('True Positive Rate')
                                ax1.set_title('Receiver Operating Characteristic (ROC)')
                                ax1.legend(loc="lower right")
                                col1.pyplot(fig1)
                                
                                # Precision-Recall curve (better for imbalanced data)
                                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
                                pr_auc = average_precision_score(y_test, y_pred_proba)
                                
                                # Store PR data in session state
                                st.session_state.evaluation_state['pr_data'] = {
                                    'precision': precision_curve,
                                    'recall': recall_curve,
                                    'pr_thresholds': pr_thresholds,
                                    'pr_auc': pr_auc
                                }
                                
                                fig2, ax2 = plt.subplots(figsize=(8, 6))
                                ax2.plot(recall_curve, precision_curve, label=f'PR curve (area = {pr_auc:.3f})')
                                
                                # Add baseline (no skill) line
                                no_skill = len(y_test[y_test == 1]) / len(y_test)
                                ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
                                
                                ax2.set_xlim([0.0, 1.0])
                                ax2.set_ylim([0.0, 1.05])
                                ax2.set_xlabel('Recall')
                                ax2.set_ylabel('Precision')
                                ax2.set_title('Precision-Recall Curve (Better for Imbalanced Data)')
                                ax2.legend(loc="best")
                                col2.pyplot(fig2)
                                
                                # Threshold optimization curve for binary classification
                                st.markdown("### Threshold Optimization")
                                st.info("This visualization helps find the optimal threshold for your specific needs. "
                                      "Move the threshold to balance precision and recall.")
                                
                                # Create dataframe of threshold metrics
                                thresh_metrics = []
                                
                                # Use fewer thresholds for efficiency
                                eval_thresholds = np.linspace(0.05, 0.95, 19)
                                
                                for t in eval_thresholds:
                                    y_pred_t = (y_pred_proba > t).astype(int)
                                    precision_t = precision_score(y_test, y_pred_t, zero_division=0)
                                    recall_t = recall_score(y_test, y_pred_t)
                                    f1_t = 2 * (precision_t * recall_t) / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
                                    accuracy_t = accuracy_score(y_test, y_pred_t)
                                    thresh_metrics.append({
                                        'threshold': t,
                                        'precision': precision_t,
                                        'recall': recall_t,
                                        'f1': f1_t,
                                        'accuracy': accuracy_t
                                    })
                                
                                thresh_df = pd.DataFrame(thresh_metrics)
                                
                                # Store threshold metrics in session state
                                st.session_state.evaluation_state['threshold_metrics'] = thresh_df
                                
                                # Plot threshold metrics
                                fig3, ax3 = plt.subplots(figsize=(10, 6))
                                ax3.plot(thresh_df['threshold'], thresh_df['precision'], label='Precision')
                                ax3.plot(thresh_df['threshold'], thresh_df['recall'], label='Recall')
                                ax3.plot(thresh_df['threshold'], thresh_df['f1'], label='F1 Score')
                                ax3.plot(thresh_df['threshold'], thresh_df['accuracy'], label='Accuracy')
                                
                                # Add vertical line for current/default threshold
                                if custom_threshold is not None:
                                    ax3.axvline(x=custom_threshold, color='r', linestyle='--',
                                              label=f'Custom threshold ({custom_threshold:.2f})')
                                else:
                                    ax3.axvline(x=0.5, color='k', linestyle='--',
                                              label='Default threshold (0.5)')
                                
                                # Find threshold with maximum F1 score
                                best_f1_threshold = thresh_df.loc[thresh_df['f1'].idxmax(), 'threshold']
                                ax3.axvline(x=best_f1_threshold, color='g', linestyle='--',
                                          label=f'Best F1 threshold ({best_f1_threshold:.2f})')
                                
                                ax3.set_xlim([0.0, 1.0])
                                ax3.set_ylim([0.0, 1.05])
                                ax3.set_xlabel('Threshold')
                                ax3.set_ylabel('Score')
                                ax3.set_title('Metrics at Different Thresholds')
                                ax3.legend(loc="best")
                                ax3.grid(True, alpha=0.3)
                                st.pyplot(fig3)
                                
                                # Store best F1 threshold in session state
                                st.session_state.evaluation_state['best_f1_threshold'] = best_f1_threshold
                                
                                # Recommended threshold based on class imbalance
                                class_distribution = pd.Series(y_test).value_counts(normalize=True)
                                minority_class_pct = class_distribution.min()
                                
                                st.markdown("### Threshold Recommendations")
                                st.info(f"Class distribution: {dict(pd.Series(y_test).value_counts())}")
                                
                                recommendations = [
                                    {
                                        "name": "Balanced (Default)",
                                        "threshold": 0.5,
                                        "description": "Standard threshold, assumes equal class importance."
                                    },
                                    {
                                        "name": "Maximum F1",
                                        "threshold": best_f1_threshold,
                                        "description": "Optimizes the balance between precision and recall."
                                    },
                                    {
                                        "name": "Class-Balanced",
                                        "threshold": 1 - minority_class_pct,
                                        "description": "Adjusts for class imbalance, increasing minority class recall."
                                    },
                                ]
                                
                                # Store recommendations in session state
                                st.session_state.evaluation_state['threshold_recommendations'] = recommendations
                                
                                # Display recommendations
                                for rec in recommendations:
                                    col1, col2 = st.columns([1, 3])
                                    col1.metric(rec["name"], f"{rec['threshold']:.2f}")
                                    col2.write(rec["description"])
                                
                                # Add option to apply new threshold
                                st.markdown("### Apply New Threshold")
                                new_threshold = st.slider(
                                    "Select threshold to apply",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=custom_threshold if custom_threshold is not None else 0.5,
                                    step=0.05,
                                    key="threshold_slider"
                                )
                                
                                # Use a container for the button and results
                                threshold_button_col, _ = st.columns([1, 3])
                                apply_threshold = threshold_button_col.button("Apply New Threshold", key="apply_new_threshold_btn")
                                
                                # Container for threshold results
                                threshold_results = st.container()
                                
                                # Update state if button is clicked
                                if apply_threshold:
                                    st.session_state.evaluation_state['threshold_applied'] = True
                                    st.session_state.evaluation_state['applied_threshold'] = new_threshold
                                
                                # Show results if threshold has been applied
                                if st.session_state.evaluation_state['threshold_applied']:
                                    applied_threshold = st.session_state.evaluation_state['applied_threshold']
                                    
                                    with threshold_results:
                                        # Get data from session state
                                        y_test = st.session_state.evaluation_state['y_test']
                                        y_pred_proba = st.session_state.evaluation_state['y_pred_proba']
                                        
                                        # Calculate new predictions and metrics
                                        y_pred_new = (y_pred_proba > applied_threshold).astype(int)
                                        accuracy_new = accuracy_score(y_test, y_pred_new)
                                        precision_new = precision_score(y_test, y_pred_new, zero_division=0)
                                        recall_new = recall_score(y_test, y_pred_new)
                                        f1_new = f1_score(y_test, y_pred_new)
                                        
                                        # Get original metrics
                                        metrics = st.session_state.evaluation_state['metrics']
                                        
                                        # Show new metrics with deltas
                                        st.success(f"Applied threshold: {applied_threshold:.2f}")
                                        col1, col2, col3, col4 = st.columns(4)
                                        col1.metric("Accuracy", f"{accuracy_new:.4f}",
                                                  f"{accuracy_new - metrics['accuracy']:.4f}")
                                        col2.metric("Precision", f"{precision_new:.4f}",
                                                  f"{precision_new - metrics['precision']:.4f}")
                                        col3.metric("Recall", f"{recall_new:.4f}",
                                                  f"{recall_new - metrics['recall']:.4f}")
                                        col4.metric("F1 Score", f"{f1_new:.4f}",
                                                  f"{f1_new - metrics['f1']:.4f}")
                                        
                                        # Update confusion matrix
                                        cm_new = confusion_matrix(y_test, y_pred_new)
                                        fig4, ax4 = plt.subplots(figsize=(8, 6))
                                        
                                        # Get class names if available
                                        class_names = st.session_state.evaluation_state.get('class_names', None)
                                        
                                        if class_names:
                                            sns.heatmap(cm_new, annot=True, fmt='d', ax=ax4,
                                                      xticklabels=class_names, yticklabels=class_names,
                                                      cmap='Blues')
                                        else:
                                            sns.heatmap(cm_new, annot=True, fmt='d', ax=ax4, cmap='Blues')
                                        
                                        plt.title(f'Confusion Matrix (Threshold = {applied_threshold:.2f})')
                                        plt.ylabel('True Label')
                                        plt.xlabel('Predicted Label')
                                        st.pyplot(fig4)
                            
                            # Feature importance section
                            st.markdown("### Feature Importance")
                            
                            if hasattr(actual_model, 'feature_importances_'):
                                # For tree-based models
                                importances = actual_model.feature_importances_
                                indices = np.argsort(importances)[::-1]
                                
                                # Create DataFrame for importance
                                importance_df = pd.DataFrame({
                                    'feature': [X.columns[i] for i in indices],
                                    'importance': [importances[i] for i in indices]
                                })
                                
                                # Store in session state
                                st.session_state.evaluation_state['feature_importance'] = importance_df
                                
                                # Display feature importance
                                st.dataframe(importance_df)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                importance_df.sort_values('importance', ascending=True).tail(15).plot(
                                    kind='barh', x='feature', y='importance', ax=ax)
                                plt.title('Feature Importance (Top 15)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                # For models without feature_importances_ attribute
                                st.info("Feature importance is not directly available for this model type. "
                                      "Consider using permutation importance for model interpretation.")
                                
                                # Permutation importance button in a column to prevent full-width
                                perm_col1, _ = st.columns([1, 3])
                                perm_button = perm_col1.button("Calculate Permutation Importance", key="calc_perm_importance_btn")
                                
                                # Update state if button is clicked
                                if perm_button:
                                    st.session_state.evaluation_state['permutation_calculated'] = True
                                
                                # Container for permutation results
                                perm_results = st.container()
                                
                                # Show permutation importance if calculated
                                if st.session_state.evaluation_state['permutation_calculated']:
                                    with perm_results:
                                        with st.spinner("Calculating permutation importance..."):
                                            try:
                                                from sklearn.inspection import permutation_importance
                                                
                                                # Create wrapper for neural network if needed
                                                if is_neural_network:
                                                    class ModelWrapper:
                                                        def __init__(self, model):
                                                            self.model = model
                                                        
                                                        def predict(self, X):
                                                            if n_classes == 2:
                                                                return (self.model.predict(X) > 0.5).astype(int).flatten()
                                                            else:
                                                                return np.argmax(self.model.predict(X), axis=1)
                                                    
                                                    model_for_perm = ModelWrapper(actual_model)
                                                else:
                                                    model_for_perm = actual_model
                                                
                                                # Calculate permutation importance
                                                scoring = 'balanced_accuracy' if n_classes == 2 else 'accuracy'
                                                
                                                # If we have class imbalance, adjust scoring
                                                class_counts = np.bincount(y)
                                                if len(class_counts) > 1:
                                                    imbalance_ratio = np.min(class_counts) / np.max(class_counts)
                                                    if imbalance_ratio < 0.2:  # Significant imbalance
                                                        scoring = 'balanced_accuracy'
                                                
                                                r = permutation_importance(
                                                    model_for_perm, X_test_transformed, y_test,
                                                    n_repeats=10,
                                                    random_state=42,
                                                    scoring=scoring
                                                )
                                                
                                                # Create DataFrame for importance
                                                perm_importance_df = pd.DataFrame({
                                                    'feature': X.columns,
                                                    'importance': r.importances_mean,
                                                    'std': r.importances_std
                                                }).sort_values('importance', ascending=False)
                                                
                                                # Store in session state
                                                st.session_state.evaluation_state['permutation_importance'] = perm_importance_df
                                                
                                                # Display permutation importance
                                                st.subheader("Permutation Feature Importance")
                                                st.dataframe(perm_importance_df)
                                                
                                                # Plot permutation importance
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                perm_importance_df.head(15).sort_values('importance').plot(
                                                    kind='barh', x='feature', y='importance', xerr='std', ax=ax)
                                                plt.title('Permutation Feature Importance (Top 15)')
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                
                                                # Add explanation
                                                st.markdown("""
                                                **Permutation Importance** measures how much the model performance decreases when a feature is randomly shuffled.
                                                Higher values indicate more important features. Features with negative importance can be noise or might be interacting with other features.
                                                """)
                                            except Exception as e:
                                                st.error(f"Error calculating permutation importance: {str(e)}")
                                                st.code(traceback.format_exc())
                        else:  # Regression evaluation
                            # Regression evaluation
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            # Handle predictions based on model type
                            if is_neural_network:
                                y_pred = actual_model.predict(X_test_transformed).flatten()
                            else:
                                y_pred = actual_model.predict(X_test_transformed)
                            
                            # Store predictions in session state
                            st.session_state.evaluation_state['y_test'] = y_test
                            st.session_state.evaluation_state['y_pred'] = y_pred
                            
                            # Calculate metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Store metrics in session state
                            st.session_state.evaluation_state['metrics'] = {
                                'mse': mse,
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2
                            }
                            
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
                            
                            # Store errors in session state
                            st.session_state.evaluation_state['errors'] = errors
                            
                            plt.hist(errors, bins=30)
                            plt.title('Error Distribution')
                            plt.xlabel('Prediction Error')
                            plt.ylabel('Frequency')
                            plt.axvline(x=0, color='r', linestyle='--')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Feature importance (tree-based models) or permutation importance (other models)
                            st.markdown("### Feature Importance")
                            
                            if hasattr(actual_model, 'feature_importances_'):
                                # For tree-based models
                                importances = actual_model.feature_importances_
                                indices = np.argsort(importances)[::-1]
                                
                                # Create DataFrame for importance
                                importance_df = pd.DataFrame({
                                    'feature': [X.columns[i] for i in indices],
                                    'importance': [importances[i] for i in indices]
                                })
                                
                                # Store in session state
                                st.session_state.evaluation_state['feature_importance'] = importance_df
                                
                                # Display feature importance
                                st.dataframe(importance_df)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                importance_df.sort_values('importance', ascending=True).tail(15).plot(
                                    kind='barh', x='feature', y='importance', ax=ax)
                                plt.title('Feature Importance (Top 15)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                # For models without feature_importances_ attribute
                                st.info("Feature importance is not directly available for this model type. "
                                      "Consider using permutation importance for model interpretation.")
                                
                                # Permutation importance button in a column to prevent full-width
                                perm_col1, _ = st.columns([1, 3])
                                perm_button = perm_col1.button("Calculate Permutation Importance", key="calc_perm_importance_reg_btn")
                                
                                # Update state if button is clicked
                                if perm_button:
                                    st.session_state.evaluation_state['permutation_calculated'] = True
                                
                                # Container for permutation results
                                perm_results = st.container()
                                
                                # Show permutation importance if calculated
                                if st.session_state.evaluation_state['permutation_calculated']:
                                    with perm_results:
                                        with st.spinner("Calculating permutation importance..."):
                                            try:
                                                from sklearn.inspection import permutation_importance
                                                
                                                # Create wrapper for neural network if needed
                                                if is_neural_network:
                                                    class KerasRegressorWrapper:
                                                        def __init__(self, model):
                                                            self.model = model
                                                        
                                                        def predict(self, X):
                                                            return self.model.predict(X).flatten()
                                                    
                                                    model_for_perm = KerasRegressorWrapper(actual_model)
                                                else:
                                                    model_for_perm = actual_model
                                                
                                                # Calculate permutation importance
                                                r = permutation_importance(
                                                    model_for_perm, X_test_transformed, y_test,
                                                    n_repeats=10,
                                                    random_state=42,
                                                    scoring='r2'
                                                )
                                                
                                                # Create DataFrame for importance
                                                perm_importance_df = pd.DataFrame({
                                                    'feature': X.columns,
                                                    'importance': r.importances_mean,
                                                    'std': r.importances_std
                                                }).sort_values('importance', ascending=False)
                                                
                                                # Store in session state
                                                st.session_state.evaluation_state['permutation_importance'] = perm_importance_df
                                                
                                                # Display permutation importance
                                                st.subheader("Permutation Feature Importance")
                                                st.dataframe(perm_importance_df)
                                                
                                                # Plot permutation importance
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                perm_importance_df.head(15).sort_values('importance').plot(
                                                    kind='barh', x='feature', y='importance', xerr='std', ax=ax)
                                                plt.title('Permutation Feature Importance (Top 15)')
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                            except Exception as e:
                                                st.error(f"Error calculating permutation importance: {str(e)}")
                                                st.code(traceback.format_exc())
                except Exception as e:
                    st.error(f"Error evaluating model: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Add cross-validation section
        st.markdown("---")
        st.markdown("### Cross-Validation Analysis")
        
        cv_col1, _ = st.columns([1, 3])
        run_cv = cv_col1.button("Run Cross-Validation", key="run_cv_btn")
        
        # Initialize CV state in session_state if not present
        if 'cv_state' not in st.session_state:
            st.session_state.cv_state = {
                'performed': False,
                'results': None,
                'cv_scores': None
            }
        
        # Update state if button is clicked
        if run_cv:
            st.session_state.cv_state['performed'] = True
        
        # Container for CV results
        cv_results = st.container()
        
        # Show CV results if performed
        if st.session_state.cv_state['performed']:
            with cv_results:
                try:
                    with st.spinner("Running cross-validation..."):
                        # Get model
                        model = st.session_state.models[selected_model]
                        
                        # Check if model is a dictionary (for SVM and NN models that include scaler)
                        if isinstance(model, dict) and 'model' in model:
                            actual_model = model['model']
                            scaler = model.get('scaler', None)
                        else:
                            actual_model = model
                            scaler = None
                        
                        # Set up cross-validation
                        n_folds = st.slider("Number of folds", min_value=3, max_value=10, value=5, step=1)
                        
                        # Choose appropriate cross-validation strategy
                        if target_type == 'categorical':
                            from sklearn.model_selection import StratifiedKFold, cross_validate
                            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                            
                            # Select scoring metrics based on class distribution
                            class_counts = pd.Series(y).value_counts()
                            imbalance_ratio = class_counts.min() / class_counts.max()
                            
                            if imbalance_ratio < 0.2:  # Significantly imbalanced
                                scoring = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                                primary_metric = 'balanced_accuracy'
                            else:  # Relatively balanced
                                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                                primary_metric = 'accuracy'
                        else:  # Regression
                            from sklearn.model_selection import KFold, cross_validate
                            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
                            primary_metric = 'r2'
                        
                        # If scaler is provided, we need to handle scaling in CV
                        if scaler is not None:
                            # For models requiring scaling, we'll use a Pipeline
                            from sklearn.pipeline import Pipeline
                            from sklearn.preprocessing import StandardScaler
                            pipeline = Pipeline([
                                ('scaler', StandardScaler()),
                                ('model', actual_model)
                            ])
                            model_to_cv = pipeline
                        else:
                            model_to_cv = actual_model
                        
                        # Create wrapper for neural network models if needed
                        if is_neural_network:
                            # This is more complex for neural networks
                            # For simplicity, let's just show a message
                            st.warning("Cross-validation for neural networks is not directly supported in this interface. " +
                                     "Consider using the K-fold validation functionality from Keras or TensorFlow.")
                            return
                        
                        # Run cross-validation
                        cv_results = cross_validate(
                            model_to_cv, X, y, 
                            cv=cv,
                            scoring=scoring,
                            return_train_score=True
                        )
                        
                        # Store CV results in session state
                        st.session_state.cv_state['results'] = cv_results
                        st.session_state.cv_state['cv_scores'] = {
                            metric: {
                                'mean': cv_results[f'test_{metric}'].mean(),
                                'std': cv_results[f'test_{metric}'].std(),
                                'values': cv_results[f'test_{metric}']
                            }
                            for metric in scoring
                        }
                        
                        # Display CV results
                        st.subheader("Cross-Validation Results")
                        
                        # Create a table of results
                        cv_df = pd.DataFrame({
                            'Metric': [metric for metric in scoring],
                            'Mean': [cv_results[f'test_{metric}'].mean() for metric in scoring],
                            'Std Dev': [cv_results[f'test_{metric}'].std() for metric in scoring],
                            'Min': [cv_results[f'test_{metric}'].min() for metric in scoring],
                            'Max': [cv_results[f'test_{metric}'].max() for metric in scoring]
                        })
                        
                        # Format negative metrics
                        for metric in scoring:
                            if metric.startswith('neg_'):
                                # Convert negative metrics to positive
                                cv_df.loc[cv_df['Metric'] == metric, 'Mean'] = -cv_df.loc[cv_df['Metric'] == metric, 'Mean']
                                cv_df.loc[cv_df['Metric'] == metric, 'Min'] = -cv_df.loc[cv_df['Metric'] == metric, 'Max']
                                cv_df.loc[cv_df['Metric'] == metric, 'Max'] = -cv_df.loc[cv_df['Metric'] == metric, 'Min']
                                
                                # Rename metrics for display
                                metric_map = {
                                    'neg_mean_squared_error': 'MSE',
                                    'neg_mean_absolute_error': 'MAE',
                                    'neg_root_mean_squared_error': 'RMSE',
                                    'neg_median_absolute_error': 'Median AE',
                                    'neg_log_loss': 'Log Loss'
                                }
                                new_name = metric_map.get(metric, metric)
                                cv_df.loc[cv_df['Metric'] == metric, 'Metric'] = new_name
                        
                        # Display the table
                        st.dataframe(cv_df.set_index('Metric'))
                        
                        # Display results of primary metric
                        primary_scores = cv_results[f'test_{primary_metric}']
                        st.subheader(f"{primary_metric.replace('_', ' ').title()} Scores")
                        
                        # Plot CV scores
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Bar plot of CV scores
                        ax.bar(range(len(primary_scores)), primary_scores, alpha=0.7)
                        ax.axhline(y=primary_scores.mean(), color='r', linestyle='-', 
                                  label=f'Mean = {primary_scores.mean():.4f}')
                        ax.set_xlabel('Fold')
                        ax.set_ylabel(primary_metric.replace('_', ' ').title())
                        ax.set_title(f'Cross-Validation {primary_metric.replace("_", " ").title()} Scores')
                        ax.set_xticks(range(len(primary_scores)))
                        ax.set_xticklabels([f'Fold {i+1}' for i in range(len(primary_scores))])
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Plot comparison of metrics
                        if len(scoring) > 1:
                            st.subheader("Metrics Comparison")
                            
                            # Create data for plotting
                            metrics_to_plot = []
                            for metric in scoring:
                                # Skip negative metrics as we've already converted them
                                if metric.startswith('neg_'):
                                    continue
                                
                                metrics_to_plot.append({
                                    'metric': metric.replace('_', ' ').title(),
                                    'mean': cv_results[f'test_{metric}'].mean(),
                                    'std': cv_results[f'test_{metric}'].std()
                                })
                            
                            # Plot metrics comparison
                            if metrics_to_plot:
                                metrics_df = pd.DataFrame(metrics_to_plot)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                metrics_df.plot(kind='bar', x='metric', y='mean', yerr='std', ax=ax)
                                ax.set_ylabel('Score')
                                ax.set_title('Cross-Validation Metrics Comparison')
                                plt.tight_layout()
                                st.pyplot(fig)
                        
                        # Display training vs test performance
                        st.subheader("Training vs Validation Performance")
                        
                        # Create data for plotting
                        train_test_data = []
                        for metric in scoring:
                            # Skip negative metrics for simplicity
                            if metric.startswith('neg_'):
                                continue
                            
                            train_mean = cv_results[f'train_{metric}'].mean()
                            test_mean = cv_results[f'test_{metric}'].mean()
                            
                            train_test_data.append({
                                'metric': metric.replace('_', ' ').title(),
                                'train': train_mean,
                                'test': test_mean
                            })
                        
                        # Plot training vs test performance
                        if train_test_data:
                            train_test_df = pd.DataFrame(train_test_data)
                            
                            # Calculate overfitting
                            train_test_df['diff'] = train_test_df['train'] - train_test_df['test']
                            train_test_df['overfitting'] = train_test_df['diff'] / train_test_df['train'] * 100
                            
                            # Plot comparison
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Bar plot showing training and test metrics
                            bar_width = 0.35
                            indices = np.arange(len(train_test_df))
                            
                            train_bars = ax.bar(indices - bar_width/2, train_test_df['train'], 
                                              bar_width, label='Training')
                            test_bars = ax.bar(indices + bar_width/2, train_test_df['test'], 
                                             bar_width, label='Validation')
                            
                            ax.set_xlabel('Metric')
                            ax.set_ylabel('Score')
                            ax.set_title('Training vs Validation Performance')
                            ax.set_xticks(indices)
                            ax.set_xticklabels(train_test_df['metric'])
                            ax.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Add overfitting analysis
                            st.subheader("Overfitting Analysis")
                            
                            # Display overfitting percentages
                            overfit_df = train_test_df[['metric', 'train', 'test', 'overfitting']]
                            overfit_df.columns = ['Metric', 'Training Score', 'Validation Score', 'Overfitting (%)']
                            st.dataframe(overfit_df.set_index('Metric'))
                            
                            # Interpretation
                            mean_overfit = overfit_df['Overfitting (%)'].mean()
                            
                            if mean_overfit > 10:
                                st.warning(f"Average overfitting: {mean_overfit:.2f}%. Consider using more regularization or more training data.")
                            elif mean_overfit < 3:
                                st.success(f"Average overfitting: {mean_overfit:.2f}%. The model generalizes well.")
                            else:
                                st.info(f"Average overfitting: {mean_overfit:.2f}%. The model shows reasonable generalization.")
                        
                        # Learning curve analysis
                        st.subheader("Learning Curve Analysis")
                        
                        # Add button to generate learning curve
                        lc_col1, _ = st.columns([1, 3])
                        gen_learning_curve = lc_col1.button("Generate Learning Curve", key="gen_learning_curve_btn")
                        
                        # Initialize learning curve state
                        if 'learning_curve_generated' not in st.session_state.cv_state:
                            st.session_state.cv_state['learning_curve_generated'] = False
                        
                        # Update state if button is clicked
                        if gen_learning_curve:
                            st.session_state.cv_state['learning_curve_generated'] = True
                        
                        # Container for learning curve results
                        lc_results = st.container()
                        
                        # Show learning curve if generated
                        if st.session_state.cv_state['learning_curve_generated']:
                            with lc_results:
                                with st.spinner("Generating learning curve..."):
                                    from sklearn.model_selection import learning_curve
                                    
                                    # Generate learning curve
                                    train_sizes, train_scores, test_scores = learning_curve(
                                        model_to_cv, X, y,
                                        cv=cv,
                                        n_jobs=-1,
                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                        scoring=primary_metric
                                    )
                                    
                                    # Calculate means and standard deviations
                                    train_mean = np.mean(train_scores, axis=1)
                                    train_std = np.std(train_scores, axis=1)
                                    test_mean = np.mean(test_scores, axis=1)
                                    test_std = np.std(test_scores, axis=1)
                                    
                                    # Store in session state
                                    st.session_state.cv_state['learning_curve'] = {
                                        'train_sizes': train_sizes,
                                        'train_mean': train_mean,
                                        'train_std': train_std,
                                        'test_mean': test_mean,
                                        'test_std': test_std
                                    }
                                    
                                    # Plot learning curve
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
                                    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
                                    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
                                    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
                                    ax.set_xlabel('Training examples')
                                    ax.set_ylabel(primary_metric.replace('_', ' ').title())
                                    ax.set_title('Learning Curve')
                                    ax.legend(loc='best')
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                                    
                                    # Learning curve interpretation
                                    gap = train_mean[-1] - test_mean[-1]
                                    slope = (test_mean[-1] - test_mean[0]) / (train_sizes[-1] - train_sizes[0])
                                    
                                    st.markdown("### Learning Curve Interpretation")
                                    
                                    # Analyze gap between training and validation scores
                                    if gap > 0.1:
                                        st.warning(f"High variance (overfitting): The gap between training and validation scores is {gap:.4f}.")
                                        st.markdown("**Recommendations:**")
                                        st.markdown("- Use more regularization")
                                        st.markdown("- Simplify the model")
                                        st.markdown("- Collect more training data")
                                    elif train_mean[-1] < 0.7 and test_mean[-1] < 0.7:
                                        st.warning(f"High bias (underfitting): Both training ({train_mean[-1]:.4f}) and validation ({test_mean[-1]:.4f}) scores are low.")
                                        st.markdown("**Recommendations:**")
                                        st.markdown("- Use a more complex model")
                                        st.markdown("- Add more features")
                                        st.markdown("- Reduce regularization")
                                    else:
                                        st.success(f"Good balance: Training score {train_mean[-1]:.4f}, validation score {test_mean[-1]:.4f}.")
                                    
                                    # Analyze validation curve slope
                                    if slope > 0.01:
                                        st.info(f"The model performance is still improving with more data (slope: {slope:.4f}).")
                                        st.markdown("**Recommendation:** Collect more training data if possible.")
                                    else:
                                        st.info(f"Adding more data may not significantly improve performance (slope: {slope:.4f}).")
                                        st.markdown("**Recommendation:** Focus on feature engineering or model architecture rather than collecting more data.")
                        
                except Exception as e:
                    st.error(f"Error running cross-validation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
        # Add section for saving model evaluation results
        st.markdown("---")
        st.subheader("Save Evaluation Results")
        
        if st.button("Save Evaluation Report", key="save_eval_report"):
            try:
                # Get timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create report dictionary
                evaluation_report = {
                    "model_name": selected_model,
                    "timestamp": timestamp,
                    "metrics": st.session_state.evaluation_state.get('metrics', {}),
                    "confusion_matrix": st.session_state.evaluation_state.get('confusion_matrix', None),
                    "feature_importance": st.session_state.evaluation_state.get('feature_importance', None),
                    "permutation_importance": st.session_state.evaluation_state.get('permutation_importance', None)
                }
                
                # Add classification-specific metrics
                if target_type == 'categorical':
                    evaluation_report.update({
                        "classification_report": st.session_state.evaluation_state.get('report', None),
                        "roc_data": st.session_state.evaluation_state.get('roc_data', None),
                        "pr_data": st.session_state.evaluation_state.get('pr_data', None),
                        "threshold_metrics": st.session_state.evaluation_state.get('threshold_metrics', None)
                    })
                
                # Add cross-validation results if available
                if st.session_state.cv_state.get('performed', False):
                    evaluation_report["cv_scores"] = st.session_state.cv_state.get('cv_scores', None)
                    evaluation_report["learning_curve"] = st.session_state.cv_state.get('learning_curve', None)
                
                # Convert to JSON
                import json
                
                # Handle NumPy arrays and other non-serializable objects
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, pd.DataFrame):
                            return obj.to_dict(orient='records')
                        if np.isscalar(obj) and np.isnan(obj):
                            return None
                        return super().default(obj)
                
                # Generate report JSON
                report_json = json.dumps(evaluation_report, cls=NumpyEncoder, indent=2)
                
                # Create filename
                report_filename = f"model_evaluation_{selected_model}_{timestamp}.json"
                
                # Download button
                st.download_button(
                    label="Download Evaluation Report",
                    data=report_json,
                    file_name=report_filename,
                    mime="application/json"
                )
                
                # If there's a save directory set, also save there
                if 'save_directory' in st.session_state:
                    try:
                        import os
                        reports_dir = os.path.join(st.session_state.save_directory, "reports")
                        os.makedirs(reports_dir, exist_ok=True)
                        
                        report_path = os.path.join(reports_dir, report_filename)
                        with open(report_path, 'w') as f:
                            f.write(report_json)
                        
                        st.success(f"Report saved to {report_path}")
                    except Exception as save_err:
                        st.warning(f"Could not save to directory: {str(save_err)}")
            except Exception as e:
                st.error(f"Error saving evaluation report: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Save model button
if st.button("Save Model", key="save_model"):
    try:
        # Get selected model - need to check if we're in tab3 first
        if 'selected_model' in locals():
            model_to_save = st.session_state.models[selected_model]
            model_name_to_save = selected_model
        else:
            # If not in tab3, take the first model from the session state
            model_name_to_save = list(st.session_state.models.keys())[0]
            model_to_save = st.session_state.models[model_name_to_save]
        with st.spinner("Saving model..."):
            # Serialize the model
            model_bytes = pickle.dumps(model_to_save)
            # Generate filename
            model_filename = get_timestamped_filename(f"{model_name_to_save}_model", "pkl")
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
        import traceback
        st.code(traceback.format_exc())

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