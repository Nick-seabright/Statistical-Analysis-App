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
                            
                            # Handle different possible types of feature_importance
                            if isinstance(feature_importance, dict) and feature_importance:
                                # Dictionary with at least one entry
                                model_name = list(feature_importance.keys())[0]
                                importance_df = feature_importance[model_name]
                                st.write(f"Feature importance from {model_name}:")
                                
                                # Display feature importance
                                st.dataframe(importance_df)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                importance_df.sort_values('importance', ascending=True).tail(15).plot(
                                    kind='barh', x='feature', y='importance', ax=ax)
                                plt.title('Feature Importance')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                            elif isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
                                # It's already a DataFrame
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
                                
                            else:
                                # No valid feature importance data
                                st.info("Feature importance not available for the trained models. Feature importance is only available for tree-based models like Random Forest or XGBoost.")
                        
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
                        
        # Continue with other model tabs (Neural Network, SVM/SVR) - similar changes for imbalanced data handling

    with tab3:
        st.markdown("<div class='subheader'>Model Evaluation</div>", unsafe_allow_html=True)
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
        selected_model = st.selectbox(
            "Select model to evaluate",
            model_names
        )
        
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
                    value=0.5,
                    step=0.05,
                    help="Lower values increase sensitivity (more minority class predictions)"
                )
        
        # In the model evaluation section
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
                        from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                                                    roc_curve, auc, precision_recall_curve, average_precision_score,
                                                    precision_score, recall_score, f1_score, balanced_accuracy_score)
                        # Calculate number of classes once, before using it
                        n_classes = len(np.unique(y))
                        # Handle predictions based on model type
                        if is_neural_network:
                            # For neural networks, we need to convert predictions to classes
                            if n_classes == 2:  # Binary classification
                                raw_predictions = model.predict(X_test)
                                y_pred_proba = raw_predictions.flatten()
                                
                                if custom_threshold is not None:
                                    y_pred = (y_pred_proba > custom_threshold).astype(int)
                                    st.info(f"Using custom threshold: {custom_threshold}")
                                else:
                                    y_pred = (y_pred_proba > 0.5).astype(int)
                            else:  # Multi-class classification
                                raw_predictions = model.predict(X_test)
                                y_pred = np.argmax(raw_predictions, axis=1)
                        elif hasattr(model, 'predict_proba') and custom_threshold is not None:
                            # For models with probability estimates and custom threshold
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            y_pred = (y_pred_proba > custom_threshold).astype(int)
                            st.info(f"Using custom threshold: {custom_threshold}")
                        else:
                            # Standard models
                            y_pred = model.predict(X_test)
                            if hasattr(model, 'predict_proba'):
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                            else:
                                y_pred_proba = None
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        balanced_acc = balanced_accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
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
                        
                        # Convert to DataFrame for display
                        report_df = pd.DataFrame(report).transpose()
                        
                        # Display report
                        st.dataframe(report_df)
                        
                        # Plot confusion matrix
                        st.markdown("### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        
                        # Get class names for the confusion matrix
                        if target_mapping:
                            # Create reverse mapping (encoded value -> original category)
                            reverse_mapping = {v: k for k, v in target_mapping.items()}
                            class_names = [reverse_mapping.get(i, str(i)) for i in range(len(np.unique(y)))]
                            
                            # Plot confusion matrix with original class names
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=class_names, yticklabels=class_names,
                                       cmap='Blues')
                        else:
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
                            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
                            pr_auc = average_precision_score(y_test, y_pred_proba)
                            
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
                            for t in np.linspace(0, 1, 100):
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
                            
                            # Display recommendations
                            for rec in recommendations:
                                col1, col2 = st.columns([1, 3])
                                col1.metric(rec["name"], f"{rec['threshold']:.2f}")
                                col2.write(rec["description"])
                            
                            # Add option to apply new threshold
                            new_threshold = st.slider(
                                "Apply new threshold and reevaluate",
                                min_value=0.0,
                                max_value=1.0,
                                value=custom_threshold if custom_threshold is not None else 0.5,
                                step=0.05
                            )
                            
                            if st.button("Apply New Threshold", key="apply_threshold"):
                                # Recalculate metrics with new threshold
                                y_pred_new = (y_pred_proba > new_threshold).astype(int)
                                accuracy_new = accuracy_score(y_test, y_pred_new)
                                precision_new = precision_score(y_test, y_pred_new)
                                recall_new = recall_score(y_test, y_pred_new)
                                f1_new = f1_score(y_test, y_pred_new)
                                
                                # Show new metrics
                                st.success(f"Applied threshold: {new_threshold:.2f}")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", f"{accuracy_new:.4f}", 
                                         f"{accuracy_new - accuracy:.4f}")
                                col2.metric("Precision", f"{precision_new:.4f}", 
                                         f"{precision_new - precision:.4f}")
                                col3.metric("Recall", f"{recall_new:.4f}", 
                                         f"{recall_new - recall:.4f}")
                                col4.metric("F1 Score", f"{f1_new:.4f}", 
                                         f"{f1_new - f1:.4f}")
                                
                                # Update confusion matrix
                                cm_new = confusion_matrix(y_test, y_pred_new)
                                fig4, ax4 = plt.subplots(figsize=(8, 6))
                                if target_mapping:
                                    sns.heatmap(cm_new, annot=True, fmt='d', ax=ax4, 
                                              xticklabels=class_names, yticklabels=class_names,
                                              cmap='Blues')
                                else:
                                    sns.heatmap(cm_new, annot=True, fmt='d', ax=ax4, cmap='Blues')
                                plt.title(f'Confusion Matrix (Threshold = {new_threshold:.2f})')
                                plt.ylabel('True Label')
                                plt.xlabel('Predicted Label')
                                st.pyplot(fig4)
                        
                        # Feature importance
                        st.markdown("### Feature Importance")
                        if hasattr(model, 'feature_importances_'):
                            # For tree-based models
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            # Create DataFrame for importance
                            importance_df = pd.DataFrame({
                                'feature': [X.columns[i] for i in indices],
                                'importance': [importances[i] for i in indices]
                            })
                            
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
                            
                            if st.button("Calculate Permutation Importance"):
                                with st.spinner("Calculating permutation importance..."):
                                    try:
                                        from sklearn.inspection import permutation_importance
                                        
                                        # Create wrapper for neural network if needed
                                        if is_neural_network:
                                            class ModelWrapper:
                                                def __init__(self, model):
                                                    self.model = model
                                                
                                                def predict(self, X):
                                                    if len(np.unique(y)) == 2:
                                                        return (self.model.predict(X) > 0.5).astype(int).flatten()
                                                    else:
                                                        return np.argmax(self.model.predict(X), axis=1)
                                            
                                            model_for_perm = ModelWrapper(model)
                                        else:
                                            model_for_perm = model
                                        
                                        # Calculate permutation importance - using balanced accuracy for imbalanced data
                                        if n_classes == 2:
                                            scoring = 'balanced_accuracy'
                                        else:
                                            scoring = 'balanced_accuracy' if class_distribution.min() < 0.2 else 'accuracy'
                                        
                                        r = permutation_importance(
                                            model_for_perm, X_test, y_test, 
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
                                        
                                        # Display permutation importance
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
                            # For non-tree models, offer permutation importance
                            st.info("Feature importance is not directly available for this model type. "
                                   "Consider using permutation importance for model interpretation.")
                            
                            if st.button("Calculate Permutation Importance"):
                                with st.spinner("Calculating permutation importance..."):
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
                                        r = permutation_importance(
                                            model_for_perm, X_test, y_test,
                                            n_repeats=10,
                                            random_state=42
                                        )
                                        # Create DataFrame for importance scores
                                        importance_df = pd.DataFrame({
                                            'feature': X.columns,
                                            'importance': r.importances_mean,
                                            'std': r.importances_std
                                        }).sort_values('importance', ascending=False)
                                        # Display importance table
                                        st.dataframe(importance_df)
                                        # Plot importance
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        importance_df.head(15).sort_values('importance').plot(
                                            kind='barh', x='feature', y='importance', xerr='std', ax=ax)
                                        plt.title('Feature Importance (Permutation Method)')
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                    except Exception as e:
                                        st.error(f"Could not calculate permutation importance: {str(e)}")
            except Exception as e:
                st.error(f"Error evaluating model: {str(e)}")
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