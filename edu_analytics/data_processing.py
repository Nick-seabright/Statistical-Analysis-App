# edu_analytics/data_processing.py

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
    
    return 'categorical', processed_series, mapping

# In edu_analytics/data_processing.py, modify the prepare_data function:

def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    selected_features: List[str] = None,
    data_types: Dict[str, str] = None
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
        
        # Return processed data with mappings
        return X, y, categorical_encoders, target_type, target_mapping, scaler, original_target, categorical_mappings        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise