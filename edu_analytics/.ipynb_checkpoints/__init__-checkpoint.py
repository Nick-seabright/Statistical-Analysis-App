# edu_analytics/__init__.py

# Import main functions to expose at package level
from .data_processing import load_data, prepare_data, detect_target_type, infer_and_validate_data_type
from .statistical_tests import perform_t_test, perform_chi_square, perform_anova, perform_correlation
from .model_training import train_models
from .model_evaluation import evaluate_model
from .threshold_analysis import analyze_decision_boundaries, analyze_custom_threshold_combination
from .time_analysis import analyze_time_target, convert_time_to_minutes
from .feature_engineering import analyze_correlations, select_features

# Package metadata
__version__ = '0.1.0'
__author__ = 'Nicholas Seabright'
__description__ = 'A comprehensive educational analytics and statistical analysis toolkit'