import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging

# Add imports for imbalanced data handling
from sklearn.utils.class_weight import compute_class_weight
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logging.warning("imbalanced-learn package not found. Some imbalanced data handling methods will not be available.")

logger = logging.getLogger(__name__)

def create_neural_network_for_target(
    target_type: str,
    input_shape: Tuple[int, ...],
    num_classes: Optional[int] = None,
    layers: List[int] = [64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    class_weights: Optional[Dict] = None  # New parameter for class weights
) -> tf.keras.Model:
    """
    Create an appropriate neural network architecture based on target type
    Parameters:
    -----------
    target_type : str
        Type of target ('categorical', 'numeric', 'time')
    input_shape : Tuple
        Shape of input data
    num_classes : int, optional
        Number of classes for classification tasks
    layers : List[int]
        List of neuron counts for hidden layers
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    class_weights : Dict, optional
        Class weights for handling imbalanced data
    Returns:
    --------
    Compiled Keras model
    """
    model = Sequential()
    # Input layer
    model.add(Dense(layers[0], activation='relu', input_shape=(input_shape[0],)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Output layer depends on target type
    if target_type == 'categorical':
        if num_classes == 2:
            # Binary classification
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            # Multi-class classification
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
    else:
        # Regression (numeric or time)
        model.add(Dense(1, activation='linear'))
        loss = 'mse'
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    metrics = ['accuracy'] if target_type == 'categorical' else ['mae', 'mse']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: List[Tuple[str, str]],
    target_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
    handle_imbalance: bool = False,
    imbalance_method: str = 'class_weights'
) -> Tuple[Dict, pd.DataFrame, Any]:
    """
    Train multiple machine learning models
    Parameters:
    -----------
    X : DataFrame
        Feature data
    y : Series
        Target data
    models : List[Tuple[str, str]]
        List of (model_name, model_type) tuples
    target_type : str
        Type of target ('categorical', 'numeric', 'time')
    test_size : float
        Size of test set
    random_state : int
        Random seed
    handle_imbalance : bool
        Whether to handle class imbalance for classification
    imbalance_method : str
        Method to handle imbalance ('class_weights', 'smote', 'adasyn', 'random_over', 'random_under')
    Returns:
    --------
    Tuple containing:
    - Dictionary of trained models
    - DataFrame with evaluation results
    - Feature importance (if available)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if target_type == 'categorical' else None
    )
    
    # Check for class imbalance and handle it if requested
    if handle_imbalance and target_type == 'categorical':
        # Check class distribution
        class_counts = pd.Series(y_train).value_counts()
        class_ratio = class_counts.min() / class_counts.max()
        
        # Apply resampling if significant imbalance exists
        if class_ratio < 0.2:  # Less than 20% ratio between minority and majority
            logger.info(f"Detected class imbalance (ratio: {class_ratio:.3f}). Applying {imbalance_method}.")
            
            # Apply sampling techniques if requested
            if imbalance_method == 'smote':
                try:
                    from imblearn.over_sampling import SMOTE
                    sampler = SMOTE(random_state=random_state)
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                    logger.info(f"Applied SMOTE. New class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                except Exception as e:
                    logger.error(f"Error applying SMOTE: {str(e)}. Continuing with original data.")
                    
            elif imbalance_method == 'adasyn':
                try:
                    from imblearn.over_sampling import ADASYN
                    sampler = ADASYN(random_state=random_state)
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                    logger.info(f"Applied ADASYN. New class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                except Exception as e:
                    logger.error(f"Error applying ADASYN: {str(e)}. Continuing with original data.")
                    
            elif imbalance_method == 'random_over':
                try:
                    from imblearn.over_sampling import RandomOverSampler
                    sampler = RandomOverSampler(random_state=random_state)
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                    logger.info(f"Applied Random Oversampling. New class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                except Exception as e:
                    logger.error(f"Error applying Random Oversampling: {str(e)}. Continuing with original data.")
                    
            elif imbalance_method == 'random_under':
                try:
                    from imblearn.under_sampling import RandomUnderSampler
                    sampler = RandomUnderSampler(random_state=random_state)
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                    logger.info(f"Applied Random Undersampling. New class distribution: {pd.Series(y_train).value_counts().to_dict()}")
                except Exception as e:
                    logger.error(f"Error applying Random Undersampling: {str(e)}. Continuing with original data.")
    
    # Dictionary to store trained models
    trained_models = {}
    # Dictionary to store evaluation results
    evaluation_results = {}
    # Dictionary to store feature importance
    feature_importance = {}
    
    # Train and evaluate each model
    for model_name, model_type in models:
        logger.info(f"Training {model_name} ({model_type})...")
        try:
            # Initialize model based on type and target type
            if target_type == 'categorical':
                # Calculate class weights if requested
                class_weight_dict = None
                if handle_imbalance and imbalance_method == 'class_weights':
                    try:
                        from sklearn.utils.class_weight import compute_class_weight
                        classes = np.unique(y_train)
                        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                        class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
                        logger.info(f"Computed class weights: {class_weight_dict}")
                    except Exception as e:
                        logger.error(f"Error computing class weights: {str(e)}. Continuing without class weights.")
                        
                # Initialize models with class weights where supported
                if model_type == 'rf':
                    model = RandomForestClassifier(
                        n_estimators=100, 
                        random_state=random_state,
                        class_weight=class_weight_dict
                    )
                elif model_type == 'svm':
                    model = SVC(
                        probability=True, 
                        random_state=random_state,
                        class_weight=class_weight_dict
                    )
                elif model_type == 'xgb':
                    # XGBoost handles class weights differently
                    model = xgb.XGBClassifier(random_state=random_state)
                    # If class weights are computed, apply them using scale_pos_weight
                    if class_weight_dict and len(class_weight_dict) == 2:
                        # For binary classification, use scale_pos_weight
                        keys = list(class_weight_dict.keys())
                        scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]
                        model.set_params(scale_pos_weight=scale_pos_weight)
                        logger.info(f"Set XGBoost scale_pos_weight to {scale_pos_weight}")
                elif model_type == 'nn':
                    num_classes = len(np.unique(y))
                    model = create_neural_network_for_target(
                        target_type='categorical',
                        input_shape=(X.shape[1],),
                        num_classes=num_classes
                    )
                elif model_type == 'balanced_rf':
                    try:
                        from imblearn.ensemble import BalancedRandomForestClassifier
                        model = BalancedRandomForestClassifier(
                            n_estimators=100,
                            random_state=random_state
                        )
                    except ImportError:
                        logger.error("BalancedRandomForestClassifier not available. Please install imbalanced-learn.")
                        continue
                elif model_type == 'easy_ensemble':
                    try:
                        from imblearn.ensemble import EasyEnsembleClassifier
                        model = EasyEnsembleClassifier(
                            n_estimators=10,
                            random_state=random_state
                        )
                    except ImportError:
                        logger.error("EasyEnsembleClassifier not available. Please install imbalanced-learn.")
                        continue
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
            else:  # 'numeric' or 'time'
                if model_type == 'rf':
                    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                elif model_type == 'svm' or model_type == 'svr':
                    model = SVR()
                elif model_type == 'xgb':
                    model = xgb.XGBRegressor(random_state=random_state)
                elif model_type == 'nn':
                    model = create_neural_network_for_target(
                        target_type='numeric',
                        input_shape=(X.shape[1],)
                    )
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                    
            # Train model
            if model_type == 'nn':
                # For neural networks, use early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                # Fit neural network
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
            else:
                # For other models, use standard fit
                model.fit(X_train, y_train)
                
            # Store trained model
            trained_models[model_name] = model
            
            # Evaluate model
            if target_type == 'categorical':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
                # Make predictions - handle neural networks differently
                if model_type == 'nn':
                    # For neural networks, convert probabilities to class predictions
                    if len(np.unique(y)) == 2:  # Binary classification
                        y_pred = (model.predict(X_test) > 0.5).astype('int32').flatten()
                    else:  # Multi-class classification
                        y_pred = np.argmax(model.predict(X_test), axis=1)
                else:
                    # Standard models
                    y_pred = model.predict(X_test)
                    
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                balanced_acc = balanced_accuracy_score(y_test, y_pred)
                
                # Store metrics
                evaluation_results[model_name] = {
                    'Accuracy': accuracy,
                    'Balanced Accuracy': balanced_acc,  # Added for imbalanced data
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }
                
                # Calculate per-class metrics for minority class
                unique_classes = np.unique(y_test)
                if len(unique_classes) == 2:  # Binary classification
                    # Find the minority class
                    class_counts = pd.Series(y_test).value_counts()
                    minority_class = class_counts.idxmin()
                    # Calculate metrics for minority class
                    minority_precision = precision_score(y_test, y_pred, pos_label=minority_class, average='binary')
                    minority_recall = recall_score(y_test, y_pred, pos_label=minority_class, average='binary')
                    minority_f1 = f1_score(y_test, y_pred, pos_label=minority_class, average='binary')
                    # Add to evaluation results
                    evaluation_results[model_name][f'Minority Precision'] = minority_precision
                    evaluation_results[model_name][f'Minority Recall'] = minority_recall
                    evaluation_results[model_name][f'Minority F1'] = minority_f1
                
                # Cross-validation score (skip for neural networks)
                if model_type != 'nn':
                    cv_scores = cross_val_score(model, X, y, cv=5)
                    evaluation_results[model_name]['CV Accuracy'] = cv_scores.mean()
                    evaluation_results[model_name]['CV Std'] = cv_scores.std()
            else:  # 'numeric' or 'time'
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                # Make predictions - handle neural networks consistently
                if model_type == 'nn':
                    y_pred = model.predict(X_test).flatten()  # Flatten for consistent dimensions
                else:
                    y_pred = model.predict(X_test)
                    
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store metrics
                evaluation_results[model_name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R² Score': r2
                }
                
                # Cross-validation score (skip for neural networks)
                if model_type != 'nn':
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    evaluation_results[model_name]['CV R²'] = cv_scores.mean()
                    evaluation_results[model_name]['CV Std'] = cv_scores.std()
                    
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                # For Random Forest and XGBoost
                importances = model.feature_importances_
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                feature_importance[model_name] = importance_df
            elif model_type in ['svm', 'svr']:
                # For SVM models, calculate permutation importance instead
                try:
                    from sklearn.inspection import permutation_importance
                    # Use a small sample for efficiency
                    sample_size = min(500, X_test.shape[0])
                    if sample_size < X_test.shape[0]:
                        X_sample = X_test.iloc[:sample_size]
                        y_sample = y_test.iloc[:sample_size]
                    else:
                        X_sample = X_test
                        y_sample = y_test
                        
                    perm_importance = permutation_importance(
                        model, X_sample, y_sample, n_repeats=5, random_state=random_state, n_jobs=-1
                    )
                    # Create DataFrame with permutation importance
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': perm_importance.importances_mean
                    }).sort_values('importance', ascending=False)
                    feature_importance[model_name] = importance_df
                    logger.info(f"Calculated permutation importance for {model_name}")
                except Exception as e:
                    logger.warning(f"Could not calculate permutation importance for {model_name}: {str(e)}")
                    # Create a dummy importance DataFrame to avoid errors
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': np.zeros(len(X.columns))
                    })
                    feature_importance[model_name] = importance_df
            # For neural networks, try permutation importance if computational resources allow
            elif model_type == 'nn':
                try:
                    # For neural networks, we need a wrapper class for sklearn compatibility
                    class KerasWrapper:
                        def __init__(self, model, binary=False):
                            self.model = model
                            self.binary = binary
                            
                        def predict(self, X):
                            if self.binary:
                                return (self.model.predict(X) > 0.5).astype('int32').flatten()
                            elif target_type == 'categorical':
                                return np.argmax(self.model.predict(X), axis=1)
                            else:
                                return self.model.predict(X).flatten()
                    
                    # Limit sample size for computational efficiency
                    sample_size = min(200, X_test.shape[0])
                    if sample_size < X_test.shape[0]:
                        X_sample = X_test.iloc[:sample_size]
                        y_sample = y_test.iloc[:sample_size]
                    else:
                        X_sample = X_test
                        y_sample = y_test
                    
                    # Create wrapped model
                    is_binary = target_type == 'categorical' and len(np.unique(y)) == 2
                    wrapped_model = KerasWrapper(model, binary=is_binary)
                    
                    # Calculate permutation importance
                    from sklearn.inspection import permutation_importance
                    perm_importance = permutation_importance(
                        wrapped_model, X_sample, y_sample, 
                        n_repeats=3,  # Fewer repeats for neural networks due to computational cost
                        random_state=random_state,
                        n_jobs=-1
                    )
                    
                    # Create DataFrame with permutation importance
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': perm_importance.importances_mean
                    }).sort_values('importance', ascending=False)
                    feature_importance[model_name] = importance_df
                    logger.info(f"Calculated permutation importance for neural network")
                except Exception as e:
                    logger.warning(f"Could not calculate permutation importance for neural network: {str(e)}")
                    # Create a dummy importance DataFrame
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': np.zeros(len(X.columns))
                    })
                    feature_importance[model_name] = importance_df
                    
            logger.info(f"Finished training {model_name}")
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
            
    # Convert evaluation results to DataFrame
    evaluation_df = pd.DataFrame(evaluation_results).T
    
    # If no models were successfully trained, provide empty results
    if not trained_models:
        logger.warning("No models were successfully trained")
        return {}, pd.DataFrame(), {}
        
    return trained_models, evaluation_df, feature_importance
                    
def hyperparameter_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    target_type: str,
    param_grid: Dict[str, List],
    cv: int = 5,
    random_state: int = 42,
    handle_imbalance: bool = False,   # New parameter
    imbalance_method: str = 'auto',   # New parameter
    scoring: str = None               # New parameter for flexible scoring
) -> Tuple[Any, Dict]:
    """
    Perform hyperparameter tuning using grid search with support for imbalanced data
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    y : Series
        Target data
    model_type : str
        Type of model ('rf', 'svm', 'xgb')
    target_type : str
        Type of target ('categorical', 'numeric', 'time')
    param_grid : Dict
        Dictionary of parameters to tune
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed
    handle_imbalance : bool
        Whether to apply techniques for handling imbalanced data
    imbalance_method : str
        Method to use for handling imbalance
    scoring : str
        Scoring metric to use for evaluation
        
    Returns:
    --------
    Tuple containing:
    - Best model
    - Dictionary with best parameters and scores
    """
    from sklearn.model_selection import GridSearchCV
    
    # Check for class imbalance if target is categorical
    imbalance_ratio = None
    class_weights_dict = None
    if target_type == 'categorical':
        class_counts = np.bincount(y)
        if len(class_counts) >= 2:
            minority_count = np.min(class_counts)
            majority_count = np.max(class_counts)
            imbalance_ratio = minority_count / majority_count
            
            # Log information about class imbalance
            logger.info(f"Class distribution: {class_counts}")
            logger.info(f"Imbalance ratio (minority/majority): {imbalance_ratio:.3f}")
    
    # Select appropriate scoring metric for imbalanced data
    if scoring is None:
        if target_type == 'categorical':
            if imbalance_ratio and imbalance_ratio < 0.2:
                # For imbalanced classification, use balanced accuracy or f1
                if len(np.unique(y)) == 2:
                    scoring = 'f1'  # Binary classification
                else:
                    scoring = 'balanced_accuracy'  # Multiclass
            else:
                # For more balanced classification
                scoring = 'accuracy'
        else:
            # For regression
            scoring = 'r2'
    
    # Handle class weights for imbalanced data
    if handle_imbalance and target_type == 'categorical' and imbalance_ratio and imbalance_ratio < 0.5:
        # Calculate class weights
        if imbalance_method in ['class_weight', 'auto']:
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weights_dict = {cls: weight for cls, weight in zip(classes, weights)}
            logger.info(f"Computed class weights: {class_weights_dict}")
            
            # Add class_weight parameter to param_grid if model supports it
            if model_type in ['rf', 'svm']:
                # Add class_weight to param_grid if not already present
                if 'class_weight' not in param_grid:
                    param_grid['class_weight'] = [None, 'balanced']
                    if class_weights_dict:
                        param_grid['class_weight'].append(class_weights_dict)
            
            # For XGBoost, handle scale_pos_weight for binary classification
            elif model_type == 'xgb' and len(np.unique(y)) == 2:
                if 'scale_pos_weight' not in param_grid and len(class_weights_dict) == 2:
                    scale_pos_weight = class_weights_dict[0] / class_weights_dict[1]
                    param_grid['scale_pos_weight'] = [1, scale_pos_weight]
    
    # Initialize model based on type and target type
    if target_type == 'categorical':
        if model_type == 'rf':
            model = RandomForestClassifier(random_state=random_state)
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=random_state)
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(random_state=random_state)
        # Add specialized models for imbalanced data
        elif model_type == 'balanced_rf' and IMBLEARN_AVAILABLE:
            model = BalancedRandomForestClassifier(random_state=random_state)
        elif model_type == 'easy_ensemble' and IMBLEARN_AVAILABLE:
            model = EasyEnsembleClassifier(random_state=random_state)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    else:  # 'numeric' or 'time'
        if model_type == 'rf':
            model = RandomForestRegressor(random_state=random_state)
        elif model_type in ['svm', 'svr']:
            model = SVR()
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(random_state=random_state)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    # For imbalanced data, use specialized CV strategies if available
    if handle_imbalance and target_type == 'categorical' and imbalance_ratio and imbalance_ratio < 0.2:
        try:
            # Use stratified k-fold for classification
            from sklearn.model_selection import StratifiedKFold
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        except ImportError:
            logger.warning("StratifiedKFold not available, using standard CV.")
            cv_strategy = cv
    else:
        cv_strategy = cv
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Apply resampling before fitting if needed
    if handle_imbalance and target_type == 'categorical' and imbalance_ratio and imbalance_ratio < 0.2:
        if IMBLEARN_AVAILABLE and imbalance_method in ['smote', 'adasyn', 'random_over', 'random_under']:
            try:
                logger.info(f"Applying {imbalance_method} before grid search...")
                if imbalance_method == 'smote':
                    sampler = SMOTE(random_state=random_state)
                elif imbalance_method == 'adasyn':
                    sampler = ADASYN(random_state=random_state)
                elif imbalance_method == 'random_over':
                    sampler = RandomOverSampler(random_state=random_state)
                elif imbalance_method == 'random_under':
                    sampler = RandomUnderSampler(random_state=random_state)
                
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                logger.info(f"Resampled data shape: {X_resampled.shape}")
                
                # Fit with resampled data
                grid_search.fit(X_resampled, y_resampled)
            except Exception as e:
                logger.error(f"Error applying resampling: {str(e)}")
                logger.warning("Falling back to original data.")
                grid_search.fit(X, y)
        else:
            # Fit with original data
            grid_search.fit(X, y)
    else:
        # Fit with original data
        grid_search.fit(X, y)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Create results dictionary
    results = {
        'best_params': best_params,
        'best_score': grid_search.best_score_,
        'scoring_metric': scoring,
        'cv_results': grid_search.cv_results_
    }
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score ({scoring}): {grid_search.best_score_:.4f}")
    
    # For imbalanced data, also evaluate on minority class if binary classification
    if target_type == 'categorical' and len(np.unique(y)) == 2 and imbalance_ratio and imbalance_ratio < 0.2:
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import precision_recall_fscore_support
            
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )
            
            # Fit best model on training data
            best_model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average=None
            )
            
            # Determine minority class
            minority_class_idx = np.argmin(np.bincount(y))
            
            # Store minority class metrics
            results['minority_class_metrics'] = {
                'precision': precision[minority_class_idx],
                'recall': recall[minority_class_idx],
                'f1': f1[minority_class_idx],
                'support': support[minority_class_idx],
                'class_idx': minority_class_idx
            }
            
            logger.info(f"Minority class ({minority_class_idx}) metrics:")
            logger.info(f"  Precision: {precision[minority_class_idx]:.4f}")
            logger.info(f"  Recall: {recall[minority_class_idx]:.4f}")
            logger.info(f"  F1: {f1[minority_class_idx]:.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate minority class metrics: {str(e)}")
    
    return best_model, results

def find_optimal_threshold(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
    metric: str = 'f1'
) -> Tuple[float, Dict]:
    """
    Find the optimal classification threshold for binary classification with imbalanced data
    
    Parameters:
    -----------
    model : trained model
        The model that supports predict_proba
    X : DataFrame
        Feature data
    y : Series
        Target data
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed
    metric : str
        Metric to optimize: 'f1', 'precision', 'recall', 'balanced_accuracy', 'g_mean'
        
    Returns:
    --------
    Tuple containing:
    - optimal_threshold: The best threshold value
    - results: Dictionary with metrics at different thresholds
    """
    if not hasattr(model, 'predict_proba'):
        logger.error("Model doesn't support predict_proba method")
        return 0.5, {}
    
    # Use stratified k-fold for imbalanced data
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Track results for each threshold across all folds
    thresholds = np.linspace(0.05, 0.95, 19)  # 0.05, 0.10, 0.15, ..., 0.95
    results = {t: {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'balanced_accuracy': [], 'g_mean': []} 
               for t in thresholds}
    
    # Perform cross-validation
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get probabilities
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Test each threshold
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate confusion matrix values for specificity
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (recall + specificity) / 2
            
            # Calculate G-mean (geometric mean of sensitivity and specificity)
            g_mean = np.sqrt(recall * specificity) if (recall * specificity) > 0 else 0
            
            # Store metrics
            results[threshold]['precision'].append(precision)
            results[threshold]['recall'].append(recall)
            results[threshold]['f1'].append(f1)
            results[threshold]['accuracy'].append(accuracy)
            results[threshold]['balanced_accuracy'].append(balanced_acc)
            results[threshold]['g_mean'].append(g_mean)
    
    # Calculate average metrics across folds
    avg_results = {}
    for threshold in thresholds:
        avg_results[threshold] = {
            'precision': np.mean(results[threshold]['precision']),
            'recall': np.mean(results[threshold]['recall']),
            'f1': np.mean(results[threshold]['f1']),
            'accuracy': np.mean(results[threshold]['accuracy']),
            'balanced_accuracy': np.mean(results[threshold]['balanced_accuracy']),
            'g_mean': np.mean(results[threshold]['g_mean'])
        }
    
    # Find optimal threshold based on selected metric
    if metric == 'precision':
        best_threshold = max(avg_results.items(), key=lambda x: x[1]['precision'])[0]
    elif metric == 'recall':
        best_threshold = max(avg_results.items(), key=lambda x: x[1]['recall'])[0]
    elif metric == 'f1':
        best_threshold = max(avg_results.items(), key=lambda x: x[1]['f1'])[0]
    elif metric == 'balanced_accuracy':
        best_threshold = max(avg_results.items(), key=lambda x: x[1]['balanced_accuracy'])[0]
    elif metric == 'g_mean':
        best_threshold = max(avg_results.items(), key=lambda x: x[1]['g_mean'])[0]
    else:
        # Default to F1
        best_threshold = max(avg_results.items(), key=lambda x: x[1]['f1'])[0]
    
    # Create final results dictionary
    final_results = {
        'optimal_threshold': best_threshold,
        'optimal_metric_value': avg_results[best_threshold][metric],
        'metric': metric,
        'threshold_results': avg_results
    }
    
    logger.info(f"Optimal threshold: {best_threshold:.2f} with {metric}: {avg_results[best_threshold][metric]:.4f}")
    
    return best_threshold, final_results

def save_model(model, filename: str) -> None:
    """
    Save a trained model to file
    Parameters:
    -----------
    model : trained model
        The model to save
    filename : str
        Path to save the model
    """
    import joblib
    import os
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Save model
    if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        model.save(filename)
    else:
        joblib.dump(model, filename)
    logger.info(f"Model saved to {filename}")

def load_model(filename: str) -> Any:
    """
    Load a saved model from file
    Parameters:
    -----------
    filename : str
        Path to the model file
    Returns:
    --------
    Loaded model
    """
    import joblib
    import os
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found: {filename}")
    # Load model
    if filename.endswith(('.h5', '.keras')):
        model = tf.keras.models.load_model(filename)
    else:
        model = joblib.load(filename)
    logger.info(f"Model loaded from {filename}")
    return model

# In edu_analytics/model_evaluation.py
def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    title: str = "Feature Importance"
) -> Optional[plt.Figure]:
    """
    Plot feature importance for models
    """
    if hasattr(model, 'feature_importances_'):
        # Use built-in feature importance
        importances = model.feature_importances_
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    else:
        # Try calculating permutation importance
        try:
            from sklearn.inspection import permutation_importance
            # Use a small subset for performance reasons
            X_sample = X_test[:min(500, len(X_test))]
            y_sample = y_test[:min(500, len(y_test))]
            
            perm_importance = permutation_importance(
                model, X_sample, y_sample, n_repeats=5, random_state=42
            )
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': perm_importance.importances_mean
            }).sort_values('Importance', ascending=False)
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {str(e)}")
            return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title(title)
    return fig