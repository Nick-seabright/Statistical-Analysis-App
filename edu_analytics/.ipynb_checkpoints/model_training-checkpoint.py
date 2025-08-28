# edu_analytics/model_training.py

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

logger = logging.getLogger(__name__)

def create_neural_network_for_target(
    target_type: str, 
    input_shape: Tuple[int, ...], 
    num_classes: Optional[int] = None,
    layers: List[int] = [64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
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
    random_state: int = 42
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
    Returns:
    --------
    Tuple containing:
    - Dictionary of trained models
    - DataFrame with evaluation results
    - Feature importance (if available)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
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
                if model_type == 'rf':
                    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                elif model_type == 'svm':
                    model = SVC(probability=True, random_state=random_state)
                elif model_type == 'xgb':
                    model = xgb.XGBClassifier(random_state=random_state)
                elif model_type == 'nn':
                    num_classes = len(np.unique(y))
                    model = create_neural_network_for_target(
                        target_type='categorical',
                        input_shape=(X.shape[1],),
                        num_classes=num_classes
                    )
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
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
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
                
                # Store metrics
                evaluation_results[model_name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }
                
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
            
            logger.info(f"Finished training {model_name}")
        
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Convert evaluation results to DataFrame
    evaluation_df = pd.DataFrame(evaluation_results).T
    
    return trained_models, evaluation_df, feature_importance

def hyperparameter_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    target_type: str,
    param_grid: Dict[str, List],
    cv: int = 5,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Perform hyperparameter tuning using grid search
    
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
        
    Returns:
    --------
    Tuple containing:
    - Best model
    - Dictionary with best parameters and scores
    """
    from sklearn.model_selection import GridSearchCV
    
    # Initialize model based on type and target type
    if target_type == 'categorical':
        if model_type == 'rf':
            model = RandomForestClassifier(random_state=random_state)
            scoring = 'accuracy'
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=random_state)
            scoring = 'accuracy'
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(random_state=random_state)
            scoring = 'accuracy'
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    else:  # 'numeric' or 'time'
        if model_type == 'rf':
            model = RandomForestRegressor(random_state=random_state)
            scoring = 'r2'
        elif model_type == 'svm' or model_type == 'svr':
            model = SVR()
            scoring = 'r2'
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(random_state=random_state)
            scoring = 'r2'
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Create results dictionary
    results = {
        'best_params': best_params,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_model, results

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