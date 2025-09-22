import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from various file formats.
    Parameters:
    -----------
    file_path : str
        Path to the data file
    Returns:
    --------
    DataFrame containing the loaded data
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def infer_and_validate_data_type(column: pd.Series) -> str:
    """
    Infer and validate data type of a column
    Parameters:
    -----------
    column : Series
        The data column to analyze
    Returns:
    --------
    str : Inferred data type
    """
    # First check if already numeric
    if pd.api.types.is_numeric_dtype(column):
        if pd.api.types.is_integer_dtype(column):
            return 'integer'
        return 'float'
    # Try to convert to numeric
    numeric_series = pd.to_numeric(column, errors='coerce')
    # If we can successfully convert to numeric with minimal NaNs, it's numeric
    if numeric_series.notna().mean() > 0.9:  # Less than 10% NaNs after conversion
        if numeric_series.dropna().apply(lambda x: x.is_integer() if not pd.isna(x) else True).all():
            return 'integer'
        return 'float'
    # Check for time format (HH:MM:SS or MM:SS)
    if column.dtype == 'object' and column.astype(str).str.match(r'^\d{1,2}:\d{2}(:\d{2})?$').any():
        return 'time'
    # Check for date format
    try:
        # Try common date formats
        date_patterns = [
            '%m/%d/%Y',        # MM/DD/YYYY
            '%d/%m/%Y',        # DD/MM/YYYY
            '%Y-%m-%d',        # YYYY-MM-DD
            '%Y/%m/%d',        # YYYY/MM/DD
            '%m-%d-%Y',        # MM-DD-YYYY
            '%d-%m-%Y',        # DD-MM-YYYY
            '%B %d, %Y',       # Month Day, Year
            '%d %B %Y'         # Day Month Year
        ]
        for pattern in date_patterns:
            try:
                # Try to parse using each pattern
                date_series = pd.to_datetime(column, format=pattern, errors='coerce')
                if date_series.notna().mean() > 0.7:  # More than 70% successful conversions
                    return 'datetime'
            except:
                continue
        # If specific patterns don't work, try the flexible parser
        date_series = pd.to_datetime(column, errors='coerce')
        if date_series.notna().mean() > 0.7:
            return 'datetime'
    except:
        pass
    # Check if boolean
    if set(column.dropna().unique()) <= {True, False, 0, 1, '0', '1', 'True', 'False', 'yes', 'no', 'Yes', 'No', 'YES', 'NO'}:
        return 'boolean'
    # Default to categorical
    return 'categorical'

def detect_target_type(series: pd.Series) -> Tuple[str, pd.Series, Optional[Dict]]:
    """
    Detect and validate the type of the target variable
    Parameters:
    -----------
    series : Series
        The target variable series
    Returns:
    --------
    Tuple containing:
    - type_name: 'numeric', 'categorical', or 'time'
    - processed_series: Processed version of the series
    - mapping: Dictionary mapping (for categorical variables) or None
    """
    # Check for and handle NaN values
    if series.isna().any():
        logger.warning(f"Target contains {series.isna().sum()} missing values which will be removed for analysis.")
        series = series.dropna()
    # First try to convert to numeric
    numeric_series = pd.to_numeric(series, errors='coerce')
    # If we can successfully convert to numeric with minimal NaNs, it's numeric
    if numeric_series.notna().mean() > 0.9:  # Less than 10% NaNs after conversion
        logger.info("Target detected as numeric variable")
        return 'numeric', numeric_series, None
    # Check if it's a time variable
    if series.dtype == 'object' and series.astype(str).str.match(r'^\d{1,2}:\d{2}(:\d{2})?$').any():
        logger.info("Target detected as time variable")
        # Convert time to minutes for modeling
        from .time_analysis import convert_time_to_minutes
        processed_series = series.apply(convert_time_to_minutes)
        # Create a mapping between original times and numeric values for interpretation
        mapping = {original: converted for original, converted in
                 zip(series.dropna().unique(), processed_series.dropna().unique())}
        return 'time', processed_series, mapping
    # Check if it's a date variable
    try:
        date_series = pd.to_datetime(series, errors='coerce')
        if date_series.notna().mean() > 0.9:  # Less than 10% NaNs after conversion
            logger.info("Target detected as date variable")
            # Convert to days since a reference date (e.g. oldest date in the series)
            reference_date = date_series.min()
            days_since_ref = (date_series - reference_date).dt.total_seconds() / (24 * 3600)
            # Create mapping for interpretation
            mapping = {'reference_date': reference_date}
            return 'date', days_since_ref, mapping
    except:
        pass
    # Must be categorical
    logger.info("Target detected as categorical variable")
    # Encode categorical variable
    le = LabelEncoder()
    processed_series = le.fit_transform(series.astype(str))
    # Create mapping for interpretation
    mapping = {label: idx for idx, label in enumerate(le.classes_)}
    
    # Check for class imbalance in categorical target
    class_counts = pd.Series(processed_series).value_counts()
    imbalance_ratio = class_counts.min() / class_counts.max()
    if imbalance_ratio < 0.25:
        if imbalance_ratio < 0.01:
            severity = "extreme"
        elif imbalance_ratio < 0.1:
            severity = "strong"
        else:
            severity = "moderate"
        
        logger.warning(f"Detected {severity} class imbalance in target variable. "
                      f"Imbalance ratio: {imbalance_ratio:.3f}. "
                      f"Consider using techniques to handle imbalanced data during model training.")
    
    return 'categorical', processed_series, mapping

def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    selected_features: List[str] = None,
    data_types: Dict[str, str] = None,
    handle_imbalance: bool = False,
    imbalance_method: str = None
) -> Tuple:
    """
    Prepare data for analysis
    Parameters:
    -----------
    df : DataFrame
        The dataset to prepare
    target_column : str
        Name of the target column
    selected_features : List[str], optional
        List of features to include, if None, all columns except target are used
    data_types : Dict[str, str], optional
        Dictionary of data types for each column, if None, types are inferred
    handle_imbalance : bool, optional
        Whether to handle class imbalance (for categorical targets)
    imbalance_method : str, optional
        Method to use for handling imbalance ('smote', 'adasyn', 'random_over', 'random_under', 'class_weights')
    Returns:
    --------
    Tuple containing:
    - X: Feature DataFrame (preprocessed)
    - y: Target Series (preprocessed)
    - categorical_encoders: Dictionary of label encoders for categorical features
    - target_type: Type of target variable ('categorical', 'numeric', 'time', etc.)
    - target_mapping: Mapping dictionary for target (if categorical)
    - scaler: StandardScaler used for feature scaling
    - original_target: Original target Series before processing
    - categorical_mappings: Mappings for categorical features
    - class_weights: Class weights for imbalanced data (if applicable)
    """
    try:
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        # Check for missing values
        missing_values = df_processed.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
            # Fill numeric columns with median, categorical with mode
            for column in df_processed.columns:
                if df_processed[column].dtype in ['int64', 'float64']:
                    df_processed[column].fillna(df_processed[column].median(), inplace=True)
                else:
                    df_processed[column].fillna(df_processed[column].mode()[0], inplace=True)
        # If no selected features provided, use all columns except target
        if selected_features is None:
            selected_features = [col for col in df_processed.columns if col != target_column]
        # If no data types provided, infer them
        if data_types is None:
            data_types = {column: infer_and_validate_data_type(df_processed[column])
                         for column in df_processed.columns}
        # Process target variable
        target_type, processed_target, target_mapping = detect_target_type(df_processed[target_column])
        # Store the original target for reference
        original_target = df_processed[target_column].copy()
        # Replace with processed version
        df_processed[target_column] = processed_target
        # Process features
        categorical_encoders = {}
        time_columns = []
        for feature in selected_features:
            feature_type = data_types.get(feature)
            # Process time features
            if feature_type == 'time':
                from .time_analysis import convert_time_to_minutes
                df_processed[feature] = df_processed[feature].apply(convert_time_to_minutes)
                time_columns.append(feature)
            # Process categorical features
            elif feature_type == 'categorical':
                le = LabelEncoder()
                df_processed[feature] = le.fit_transform(df_processed[feature].astype(str))
                categorical_encoders[feature] = le
            # Process boolean features - convert to 0/1
            elif feature_type == 'boolean':
                # Handle various forms of True/False values
                bool_map = {
                    'true': 1, 'yes': 1, 'y': 1, '1': 1, 1: 1, True: 1,
                    'false': 0, 'no': 0, 'n': 0, '0': 0, 0: 0, False: 0
                }
                # First convert to lowercase strings to standardize
                df_processed[feature] = df_processed[feature].astype(str).str.lower()
                # Then map to 0/1
                df_processed[feature] = df_processed[feature].map(bool_map)
                # Fill any unmapped values with 0
                df_processed[feature] = df_processed[feature].fillna(0).astype(int)
            # Process datetime features
            elif feature_type == 'datetime':
                df_processed[feature] = pd.to_datetime(df_processed[feature], errors='coerce')
                # Convert to days since minimum date
                min_date = df_processed[feature].min()
                df_processed[feature] = (df_processed[feature] - min_date).dt.total_seconds() / (24 * 3600)
        # Prepare X and y
        X = df_processed[selected_features].copy()
        y = df_processed[target_column].copy()
        # Check if X contains any non-numeric data before scaling
        non_numeric_cols = []
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                non_numeric_cols.append(col)
        if non_numeric_cols:
            logger.warning(f"Found non-numeric columns that weren't properly encoded: {non_numeric_cols}")
            # Attempt to convert these columns
            for col in non_numeric_cols:
                try:
                    # Try to convert to numeric
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    # Fill NaN values with column median
                    X[col] = X[col].fillna(X[col].median())
                except Exception as e:
                    logger.error(f"Could not convert column {col} to numeric: {str(e)}")
                    # As a last resort, drop the column
                    logger.warning(f"Dropping column {col} from the feature set")
                    X = X.drop(columns=[col])
        
        # Handle class imbalance if requested and target is categorical
        class_weights = None
        if handle_imbalance and target_type == 'categorical':
            from .utils import detect_class_imbalance, calculate_class_weights, sample_imbalanced_data
            
            # Detect imbalance
            imbalance_info = detect_class_imbalance(y)
            
            # Only apply if actually imbalanced
            if imbalance_info['is_imbalanced']:
                logger.info(f"Handling {imbalance_info['severity']} class imbalance "
                           f"(ratio: {imbalance_info['imbalance_ratio']:.3f}) using {imbalance_method}")
                
                if imbalance_method == 'class_weights':
                    # Calculate class weights for model training
                    class_weights = calculate_class_weights(y)
                    logger.info(f"Calculated class weights: {class_weights}")
                
                elif imbalance_method in ['smote', 'adasyn', 'random_over', 'random_under', 'smoteenn', 'smotetomek']:
                    # Apply resampling
                    X, y = sample_imbalanced_data(X, y, method=imbalance_method)
                    logger.info(f"Resampled data using {imbalance_method}. "
                               f"New class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Print summary of processing
        logger.info("\nData Processing Summary:")
        logger.info(f"Target column: {target_column}")
        logger.info(f"Target type: {target_type}")
        if target_mapping:
            logger.info(f"Target mapping: {target_mapping}")
        logger.info(f"Features processed: {len(selected_features)}")
        logger.info(f"Categorical features encoded: {len(categorical_encoders)}")
        logger.info(f"Time features converted: {len(time_columns)}")
        
        # Store mappings for each categorical feature
        categorical_mappings = {}
        for feature, encoder in categorical_encoders.items():
            if hasattr(encoder, 'classes_'):
                # Create mappings
                forward_mapping = {val: i for i, val in enumerate(encoder.classes_)}
                reverse_mapping = {i: val for i, val in enumerate(encoder.classes_)}
                categorical_mappings[feature] = {
                    'forward': forward_mapping,  # Original → Encoded
                    'reverse': reverse_mapping,  # Encoded → Original
                    'classes': encoder.classes_.tolist()
                }
        
        # If we detected class imbalance, log additional information
        if target_type == 'categorical':
            class_counts = pd.Series(y).value_counts()
            if len(class_counts) > 1:  # Only check imbalance if we have at least 2 classes
                imbalance_ratio = class_counts.min() / class_counts.max()
                if imbalance_ratio < 0.25:
                    logger.warning(f"Class imbalance detected: minority/majority ratio = {imbalance_ratio:.3f}")
                    logger.warning(f"Class distribution: {class_counts.to_dict()}")
                    if handle_imbalance:
                        logger.info(f"Applied {imbalance_method} to address class imbalance")
                    else:
                        logger.warning("No imbalance handling was applied. Consider using imbalance handling techniques during model training.")

        # Return processed data with mappings and additional imbalance handling info
        if handle_imbalance and imbalance_method == 'class_weights':
            return X_scaled, y, categorical_encoders, target_type, target_mapping, scaler, original_target, categorical_mappings, class_weights
        else:
            return X_scaled, y, categorical_encoders, target_type, target_mapping, scaler, original_target, categorical_mappings
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def check_imbalanced_target(y: pd.Series) -> Dict:
    """
    Check if the target variable is imbalanced and provide recommendations
    
    Parameters:
    -----------
    y : Series
        Target variable (categorical)
    
    Returns:
    --------
    Dict with imbalance assessment and recommendations
    """
    # Import utility function
    from .utils import detect_class_imbalance
    
    # Perform imbalance detection
    imbalance_info = detect_class_imbalance(y)
    
    # Log findings
    if imbalance_info['is_imbalanced']:
        logger.warning(f"Class imbalance detected: {imbalance_info['severity']}")
        logger.warning(f"Imbalance ratio: {imbalance_info['imbalance_ratio']:.3f}")
        logger.warning(f"Minority class ({imbalance_info['min_class']}) has only "
                      f"{imbalance_info['class_percentages'][imbalance_info['min_class']]:.2f}% of the data")
        logger.info(f"Recommended techniques: {', '.join(imbalance_info['suggested_methods'])}")
    else:
        logger.info("Target classes are relatively balanced.")
    
    return imbalance_info

def split_data_stratified(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple:
    """
    Split data into train and test sets with stratification for imbalanced targets
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    Tuple of (X_train, X_test, y_train, y_test)
    """
    # For categorical targets, use stratified split to maintain class distribution
    if len(np.unique(y)) <= 10:  # Assuming categorical if <= 10 unique values
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        # For regression or many-class targets, use regular split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def prepare_data_batch(
    new_data: pd.DataFrame,
    features: List[str],
    data_types: Dict[str, str],
    categorical_encoders: Dict,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Prepare new data batch using preprocessing learned from training data
    
    Parameters:
    -----------
    new_data : DataFrame
        New data to preprocess
    features : List[str]
        List of features to include
    data_types : Dict[str, str]
        Dictionary of data types for each feature
    categorical_encoders : Dict
        Dictionary of label encoders for categorical features
    scaler : StandardScaler
        Scaler used for feature scaling
    
    Returns:
    --------
    DataFrame with preprocessed data
    """
    # Create a copy to avoid modifying original data
    df_processed = new_data.copy()
    
    # Process each feature
    for feature in features:
        if feature not in df_processed.columns:
            logger.warning(f"Feature {feature} not found in new data. Skipping.")
            continue
            
        feature_type = data_types.get(feature)
        # Process time features
        if feature_type == 'time':
            from .time_analysis import convert_time_to_minutes
            df_processed[feature] = df_processed[feature].apply(convert_time_to_minutes)
        # Process categorical features
        elif feature_type == 'categorical':
            if feature in categorical_encoders:
                encoder = categorical_encoders[feature]
                # Convert to string to match training data
                df_processed[feature] = df_processed[feature].astype(str)
                # Handle unknown categories
                unknown_categories = set(df_processed[feature].unique()) - set(encoder.classes_)
                if unknown_categories:
                    logger.warning(f"Found unknown categories in {feature}: {unknown_categories}. "
                                  "These will be treated as missing values.")
                    # Replace unknown categories with the most common category
                    most_common = encoder.classes_[0]  # Default to first class
                    for cat in unknown_categories:
                        df_processed.loc[df_processed[feature] == cat, feature] = most_common
                # Transform
                try:
                    df_processed[feature] = encoder.transform(df_processed[feature])
                except Exception as e:
                    logger.error(f"Error encoding {feature}: {str(e)}")
                    # As fallback, use most frequent value
                    most_common_encoded = encoder.transform([encoder.classes_[0]])[0]
                    df_processed[feature] = most_common_encoded
        # Process boolean features
        elif feature_type == 'boolean':
            bool_map = {
                'true': 1, 'yes': 1, 'y': 1, '1': 1, 1: 1, True: 1,
                'false': 0, 'no': 0, 'n': 0, '0': 0, 0: 0, False: 0
            }
            df_processed[feature] = df_processed[feature].astype(str).str.lower()
            df_processed[feature] = df_processed[feature].map(bool_map)
            df_processed[feature] = df_processed[feature].fillna(0).astype(int)
        # Process datetime features
        elif feature_type == 'datetime':
            try:
                df_processed[feature] = pd.to_datetime(df_processed[feature], errors='coerce')
                # Convert to days since Unix epoch for consistency
                df_processed[feature] = (df_processed[feature] - pd.Timestamp('1970-01-01')).dt.total_seconds() / (24 * 3600)
            except Exception as e:
                logger.error(f"Error processing datetime feature {feature}: {str(e)}")
                # As fallback, use median value
                df_processed[feature] = 0
    
    # Extract selected features in the right order
    available_features = [f for f in features if f in df_processed.columns]
    X_new = df_processed[available_features]
    
    # Handle any remaining non-numeric columns
    for col in X_new.columns:
        if not pd.api.types.is_numeric_dtype(X_new[col]):
            logger.warning(f"Column {col} is not numeric after processing. Converting to numeric.")
            X_new[col] = pd.to_numeric(X_new[col], errors='coerce')
            X_new[col] = X_new[col].fillna(0)
    
    # Apply scaling
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=X_new.columns, index=X_new.index)
    
    return X_new_scaled_df