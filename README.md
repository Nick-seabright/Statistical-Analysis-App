# Data Science App 📊

**A Comprehensive Statistical Analysis and Machine Learning Platform**

Created By: Nick Seabright  
Version: 1.0.0  
License: MIT

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Application Workflow](#application-workflow)
- [Module Documentation](#module-documentation)
- [Data Sources](#data-sources)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🎯 Overview

The **Data Science App** is a powerful, user-friendly Streamlit application designed for comprehensive data analysis, statistical testing, and machine learning. Built with education and accessibility in mind, it provides both novice and experienced data scientists with an intuitive interface for exploring datasets, conducting rigorous statistical analyses, and building predictive models.

### Key Highlights

- **End-to-End Analysis**: From data upload to model deployment
- **Statistical Rigor**: Comprehensive hypothesis testing and correlation analysis
- **Machine Learning**: Multiple algorithms with automated hyperparameter tuning
- **Interactive Visualizations**: Dynamic plots and charts for data exploration
- **Report Generation**: Professional, exportable analysis reports
- **No-Code Interface**: Perform complex analyses without writing code
- **Foundry Integration**: Seamless connection to Palantir Foundry datasets

---

## ✨ Features

### 📥 **Data Upload & Management**
- **Multiple Data Sources**
  - CSV file upload
  - Palantir Foundry dataset integration
  - Sample datasets (Student Performance, Employee Attrition, Housing Prices)
- **Automatic Data Type Inference**: Intelligent detection of numeric, categorical, and time-based features
- **Feature Selection**: Choose specific features for analysis
- **Target Variable Detection**: Automatic classification/regression problem detection

### 🧹 **Data Cleaning**
- **Data Quality Overview**
  - Missing value detection and visualization
  - Duplicate record identification
  - Data type validation
- **Missing Value Handling**
  - Drop rows/columns with missing values
  - Fill with mean, median, mode, or custom values
  - Forward/backward fill for time series
- **Data Filtering & Sampling**
  - Filter by column values
  - Random sampling for large datasets
  - Stratified sampling by target variable
- **Column Operations**
  - Rename columns
  - Drop unnecessary columns
  - Reorder columns
- **Data Transformations**
  - Log transformation
  - Square root transformation
  - Standardization
  - Min-max normalization

### 📊 **Data Exploration**
- **Summary Statistics**
  - Descriptive statistics for numeric features
  - Frequency distributions for categorical features
  - Missing value analysis
- **Correlation Analysis**
  - Correlation heatmaps
  - Feature-target correlations
  - Multicollinearity detection
- **Distribution Visualization**
  - Histograms
  - Box plots
  - Violin plots
  - KDE plots
- **Feature Relationships**
  - Scatter plots
  - Pair plots
  - Category comparisons

### 📈 **Statistical Analysis**
- **T-Tests**
  - Independent samples t-test
  - Paired samples t-test
  - One-sample t-test
  - Effect size calculation (Cohen's d)
- **Chi-Square Tests**
  - Chi-square test of independence
  - Goodness of fit test
  - Contingency table analysis
  - Cramér's V effect size
- **ANOVA**
  - One-way ANOVA
  - Post-hoc testing (Tukey HSD)
  - Effect size (eta-squared)
- **Correlation Analysis**
  - Pearson correlation
  - Spearman correlation
  - Kendall's tau
  - Partial correlations

### 🎯 **Threshold Analysis**
- **Decision Boundary Optimization**
  - ROC curve analysis
  - Optimal threshold selection
  - Sensitivity-specificity trade-offs
- **Custom Threshold Testing**
  - User-defined thresholds
  - Multi-feature threshold combinations
  - Performance metric evaluation
- **Visualization**
  - ROC curves
  - Precision-recall curves
  - Confusion matrices

### 🤖 **Supervised Learning**
- **Classification Algorithms**
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - XGBoost Classifier
  - Neural Network (TensorFlow/Keras)
- **Regression Algorithms**
  - Linear Regression
  - Random Forest Regressor
  - Support Vector Regression (SVR)
  - XGBoost Regressor
  - Neural Network Regressor
- **Model Evaluation**
  - Cross-validation
  - Confusion matrices
  - Classification reports
  - ROC-AUC scores
  - R² scores
  - RMSE, MAE, MAPE
- **Feature Importance**
  - Tree-based importance
  - Permutation importance
  - SHAP values (optional)
- **Hyperparameter Tuning**
  - Grid search
  - Random search
  - Bayesian optimization

### 🔬 **Unsupervised Learning**
- **Clustering**
  - K-Means clustering
  - Hierarchical clustering
  - DBSCAN
  - Gaussian Mixture Models
- **Dimensionality Reduction**
  - Principal Component Analysis (PCA)
  - t-SNE
  - UMAP
- **Anomaly Detection**
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM

### 🔮 **Predictions**
- **Single Predictions**
  - Input individual feature values
  - Get instant predictions
  - Confidence scores
  - Prediction interpretation
- **Batch Predictions**
  - Upload CSV with new data
  - Bulk prediction processing
  - Export results
  - Prediction probabilities

### 📝 **Report Generation**
- **Comprehensive Reports**
  - Executive summary
  - Data overview
  - Statistical test results
  - Model performance metrics
  - Visualizations
  - Methodology documentation
- **Export Formats**
  - HTML
  - PDF
  - Markdown
  - JSON (data export)
- **Customization**
  - Include/exclude sections
  - Custom branding
  - Notes and annotations

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nick-seabright/Statistical-Analytics-App.git
cd Statistical-Analytics-App
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import pandas; import sklearn; print('Installation successful!')"
```

---

## 🎬 Quick Start

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Basic Usage Flow

1. **Upload Data**: Start by uploading a CSV file or loading a sample dataset
2. **Clean Data** (Optional): Handle missing values and prepare your data
3. **Process Data**: Select target variable and features
4. **Explore**: Visualize your data and understand distributions
5. **Analyze**: Run statistical tests
6. **Train Models**: Build and evaluate machine learning models
7. **Predict**: Make predictions on new data
8. **Report**: Generate comprehensive analysis reports

---

## 🔄 Application Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Statistical Analysis App                      │
└─────────────────────────────────────────────────────────────────┘

📥 Data Upload
    ↓
🧹 Data Cleaning (Optional)
    ↓
📊 Data Exploration
    ↓
📈 Statistical Analysis
    ↓
🎯 Threshold Analysis
    ↓
🤖 Supervised Learning
    ↓
🔬 Unsupervised Learning
    ↓
🔮 Make Predictions
    ↓
📝 Generate Report
```

### Typical Use Cases

#### Use Case 1: Quick Analysis (Clean Data)
```
Upload CSV → Process Data → Explore → Analyze → Report
```

#### Use Case 2: Large Foundry Dataset
```
Load Foundry → Clean (Sample 10K) → Process → Analyze → Train Models
```

#### Use Case 3: Messy Data
```
Upload CSV → Clean (Handle Missing, Remove Dupes) → 
Process → Explore → Analyze → Train → Predict
```

#### Use Case 4: Machine Learning Focus
```
Load Data → Process → Train Multiple Models → 
Hyperparameter Tuning → Predictions → Report
```

---

## 📦 Module Documentation

### `edu_analytics/`

Core analytics package containing all analysis functions.

#### `data_processing.py`
```python
"""
Data loading, preprocessing, and transformation functions.

Key Functions:
- load_data(): Load data from various sources
- prepare_data(): Preprocess data for analysis
- detect_target_type(): Identify classification vs regression
- infer_and_validate_data_type(): Automatic type detection
"""
```

#### `statistical_tests.py`
```python
"""
Statistical hypothesis testing and correlation analysis.

Key Functions:
- perform_t_test(): Independent/paired t-tests
- perform_chi_square(): Chi-square tests
- perform_anova(): Analysis of variance
- multi_group_analysis(): Multiple group comparisons
- categorical_association_analysis(): Association tests
- numerical_correlation_analysis(): Correlation studies
"""
```

#### `model_training.py`
```python
"""
Machine learning model training and selection.

Key Functions:
- train_models(): Train multiple ML algorithms
- hyperparameter_tuning(): Optimize model parameters
- cross_validate(): K-fold cross-validation
"""
```

#### `model_evaluation.py`
```python
"""
Model performance evaluation and metrics.

Key Functions:
- evaluate_model(): Comprehensive model evaluation
- classification_metrics(): Accuracy, precision, recall, F1
- regression_metrics(): R², RMSE, MAE, MAPE
- confusion_matrix_plot(): Visualize predictions
"""
```

#### `threshold_analysis.py`
```python
"""
Decision threshold optimization and analysis.

Key Functions:
- analyze_decision_boundaries(): ROC curve analysis
- analyze_custom_threshold_combination(): Multi-feature thresholds
- optimize_threshold(): Find optimal decision point
"""
```

#### `time_analysis.py`
```python
"""
Time-based feature analysis and conversions.

Key Functions:
- analyze_time_target(): Time-based target analysis
- convert_time_to_minutes(): Time format conversions
"""
```

#### `feature_engineering.py`
```python
"""
Feature creation, selection, and analysis.

Key Functions:
- analyze_correlations(): Feature correlation analysis
- select_features(): Automated feature selection
- create_interaction_features(): Feature interactions
"""
```

#### `unsupervised_learning.py`
```python
"""
Clustering, dimensionality reduction, and anomaly detection.

Key Functions:
- perform_clustering(): K-means, hierarchical, DBSCAN
- dimensionality_reduction(): PCA, t-SNE, UMAP
- detect_anomalies(): Isolation forest, LOF
"""
```

#### `utils.py`
```python
"""
Utility functions for visualization and file management.

Key Functions:
- interpret_prediction(): Explain model predictions
- set_plotting_style(): Configure matplotlib style
- create_report_header(): Generate report headers
- save_file(): Save analysis results
- get_timestamped_filename(): Generate unique filenames
"""
```

---

## 🗂️ Data Sources

### 1. CSV Upload
- Supported format: `.csv`
- Encoding: UTF-8 (recommended)
- Size limit: Configured by Streamlit settings

### 2. Palantir Foundry Integration
```python
# Load by RID
dataset_rid = "ri.foundry.main.dataset.xxxxx"

# Or select from registered datasets
datasets = get_registered_datasets()
```

### 3. Sample Datasets

#### Student Performance
- **Description**: Academic performance metrics
- **Target**: Final Grade (Classification/Regression)
- **Features**: Study time, attendance, prior grades, demographics
- **Size**: ~1,000 records

#### Employee Attrition
- **Description**: Employee turnover prediction
- **Target**: Attrition (Binary Classification)
- **Features**: Satisfaction, performance, salary, tenure
- **Size**: ~1,500 records

#### Housing Prices
- **Description**: Real estate price prediction
- **Target**: Sale Price (Regression)
- **Features**: Location, size, amenities, year built
- **Size**: ~500 records

---

## 💡 Usage Examples

### Example 1: Basic Statistical Analysis

```python
# 1. Load your data
import pandas as pd
data = pd.read_csv('your_data.csv')

# 2. Import analytics functions
from edu_analytics import perform_t_test, perform_anova

# 3. Run t-test
result = perform_t_test(
    data=data,
    group_column='treatment',
    value_column='outcome'
)

# 4. Run ANOVA for multiple groups
anova_result = perform_anova(
    data=data,
    group_column='group',
    value_column='score'
)
```

### Example 2: Building a Classification Model

```python
from edu_analytics import prepare_data, train_models, evaluate_model

# 1. Prepare data
X, y, encoders, target_type, mapping, scaler, original_y = prepare_data(
    df=data,
    target_column='outcome',
    feature_columns=['feature1', 'feature2', 'feature3'],
    data_types=None
)

# 2. Train models
models, results = train_models(
    X=X,
    y=y,
    target_type='classification',
    model_list=['logistic_regression', 'random_forest', 'xgboost']
)

# 3. Evaluate best model
best_model = models['random_forest']
evaluation = evaluate_model(
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    target_type='classification'
)
```

### Example 3: Threshold Analysis

```python
from edu_analytics import analyze_decision_boundaries

# Analyze optimal threshold for binary classification
threshold_results = analyze_decision_boundaries(
    data=data,
    target_column='outcome',
    feature_column='risk_score'
)

# Extract optimal threshold
optimal_threshold = threshold_results['optimal_threshold']
print(f"Optimal threshold: {optimal_threshold}")
```

### Example 4: Using the Streamlit App

```bash
# 1. Start the app
streamlit run app.py

# 2. In the browser:
#    - Navigate to "📥 Data Upload"
#    - Click "Browse files" and select your CSV
#    - Review data preview and summary

# 3. Clean data (if needed):
#    - Click "🧹 Data Cleaning"
#    - Handle missing values
#    - Apply filters
#    - Click "Apply Changes"

# 4. Process data:
#    - Select target variable
#    - Choose features
#    - Click "Process Data"

# 5. Explore data:
#    - Click "📊 Data Exploration"
#    - Review summary statistics
#    - Analyze correlations
#    - View distributions

# 6. Train models:
#    - Click "🤖 Supervised Learning"
#    - Select algorithms
#    - Click "Train Models"
#    - Review results

# 7. Make predictions:
#    - Click "🔮 Make Predictions"
#    - Enter feature values
#    - Get predictions

# 8. Generate report:
#    - Click "📝 Generate Report"
#    - Select sections to include
#    - Download report
```

---

## 🏗️ Architecture

### Project Structure

```
Statistical-Analytics-App/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
├── ci.yml                          # CI/CD configuration
│
├── edu_analytics/                  # Core analytics package
│   ├── __init__.py                # Package initialization
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── feature_engineering.py     # Feature creation and selection
│   ├── model_training.py          # ML model training
│   ├── model_evaluation.py        # Model evaluation metrics
│   ├── statistical_tests.py       # Statistical hypothesis testing
│   ├── threshold_analysis.py      # Decision threshold optimization
│   ├── time_analysis.py           # Time-based analysis
│   ├── unsupervised_learning.py   # Clustering and dimensionality reduction
│   └── utils.py                   # Utility functions
│
├── streamlit_app/                  # Streamlit UI components
│   ├── components/                # Reusable UI components
│   │   ├── sidebar.py             # Navigation sidebar
│   │   ├── model_cards.py         # Model display cards
│   │   ├── plots.py               # Visualization components
│   │   └── file_browser.py        # File selection UI
│   │
│   └── pages/                      # Application pages
│       ├── data_upload.py         # Data upload interface
│       ├── data_cleaning.py       # Data cleaning operations
│       ├── data_exploration.py    # Data visualization
│       ├── statistical_analysis.py # Statistical tests UI
│       ├── threshold_analysis.py  # Threshold optimization UI
│       ├── model_training.py      # Model training interface
│       ├── unsupervised_learning.py # Clustering UI
│       ├── predictions.py         # Prediction interface
│       └── report_generation.py   # Report creation
│
└── docs/                           # Documentation
    ├── app_workflow.md            # Application workflow guide
    ├── data_cleaning_guide.md     # Data cleaning tutorial
    └── unsupervised_learning_guide.md # Clustering guide
```

### Technology Stack

- **Frontend**: Streamlit 1.20+
- **Data Processing**: Pandas 1.5+, NumPy 1.20+, Polars 1.0+
- **Machine Learning**: scikit-learn 1.0+, XGBoost 1.5+
- **Deep Learning**: TensorFlow 2.10+, PyTorch 2.0+, Keras 3.0+
- **Statistical Analysis**: SciPy 1.7+, statsmodels 0.13+
- **Visualization**: Matplotlib 3.5+, Seaborn 0.11+, Plotly 5.10+
- **Data Balancing**: imbalanced-learn 0.10+

### Design Patterns

- **Modular Architecture**: Separation of concerns between UI and business logic
- **Session State Management**: Persistent state across page navigation
- **Lazy Loading**: Components loaded on-demand for performance
- **Factory Pattern**: Model creation based on algorithm selection
- **Observer Pattern**: Real-time updates during data processing

---

## 📚 API Reference

### Core Functions

#### `prepare_data()`
```python
def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list,
    data_types: dict = None
) -> tuple:
    """
    Prepare data for analysis by encoding, scaling, and splitting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_column : str
        Name of the target variable column
    feature_columns : list
        List of feature column names
    data_types : dict, optional
        Dictionary mapping column names to data types
    
    Returns:
    --------
    tuple : (X, y, encoders, target_type, mapping, scaler, original_y)
        - X: Feature matrix
        - y: Target vector
        - encoders: Dictionary of fitted encoders
        - target_type: 'classification' or 'regression'
        - mapping: Target label mapping (if categorical)
        - scaler: Fitted feature scaler
        - original_y: Original target values before encoding
    """
```

#### `train_models()`
```python
def train_models(
    X: np.ndarray,
    y: np.ndarray,
    target_type: str,
    model_list: list = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> tuple:
    """
    Train multiple machine learning models.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    target_type : str
        'classification' or 'regression'
    model_list : list, optional
        List of model names to train
    cv_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (models, results)
        - models: Dictionary of trained model objects
        - results: Dictionary of evaluation metrics
    """
```

#### `perform_t_test()`
```python
def perform_t_test(
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
    test_type: str = 'independent',
    alpha: float = 0.05
) -> dict:
    """
    Perform t-test for comparing means.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    group_column : str
        Column containing group labels
    value_column : str
        Column containing values to compare
    test_type : str, default='independent'
        Type of t-test: 'independent', 'paired', or 'one_sample'
    alpha : float, default=0.05
        Significance level
    
    Returns:
    --------
    dict : Test results including statistic, p-value, effect size
    """
```

### Session State Variables

```python
# Data storage
st.session_state.data                # Original loaded data
st.session_state.cleaned_data        # Data after cleaning operations
st.session_state.processed_data      # Processed features and target
st.session_state.data_types          # Inferred data types

# Model storage
st.session_state.models              # Trained models dictionary
st.session_state.target_type         # Classification or regression
st.session_state.target_mapping      # Label encoding mapping
st.session_state.scaler              # Fitted feature scaler
st.session_state.categorical_encoders # Fitted label encoders

# Analysis storage
st.session_state.feature_importance  # Feature importance scores
st.session_state.report_data         # Report generation data

# Navigation
st.session_state.current_section     # Current page identifier
st.session_state.save_directory      # Output directory for files
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/Statistical-Analytics-App.git
cd Statistical-Analytics-App

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .

# Check linting
flake8 .
```

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Write clear, descriptive commit messages
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations
- 🧪 Additional test coverage

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

#### Issue 2: Data Loading Fails
```
Error loading data: 'utf-8' codec can't decode byte...
```
**Solution**: Try different encoding
```python
pd.read_csv('file.csv', encoding='latin-1')
# or
pd.read_csv('file.csv', encoding='cp1252')
```

#### Issue 3: Memory Issues with Large Datasets
```
MemoryError: Unable to allocate array...
```
**Solution**: Use data sampling in cleaning step
```python
# In Data Cleaning page, use sampling
df_sample = df.sample(n=10000, random_state=42)
```

#### Issue 4: Model Training Takes Too Long
**Solution**: 
- Use smaller datasets for initial exploration
- Select fewer models to train
- Reduce cross-validation folds
- Disable hyperparameter tuning

#### Issue 5: Streamlit Session State Issues
```
KeyError: 'data'
```
**Solution**: Click "🔄 Reset All" button in sidebar to reinitialize session state

### Getting Help

- 📧 Email: [support@example.com](mailto:support@example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/Nick-seabright/Statistical-Analytics-App/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Nick-seabright/Statistical-Analytics-App/discussions)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Nicholas Seabright

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Streamlit Team**: For the amazing web framework
- **scikit-learn Contributors**: For comprehensive ML tools
- **Pandas Developers**: For powerful data manipulation
- **Open Source Community**: For inspiration and support

---

## 📊 Project Stats

- **Language**: Python 3.8+
- **Framework**: Streamlit
- **Lines of Code**: ~5,000+
- **Modules**: 9 core modules
- **Pages**: 9 interactive pages
- **Supported Algorithms**: 15+ ML algorithms
- **Statistical Tests**: 10+ hypothesis tests

---

## 🗺️ Roadmap

### Version 1.1 (Q2 2025)
- [ ] Add support for more file formats (Excel, JSON, Parquet)
- [ ] Implement SHAP value explanations
- [ ] Add automated feature engineering
- [ ] Include time series analysis module

### Version 1.2 (Q3 2025)
- [ ] REST API for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment guide (AWS, Azure, GCP)
- [ ] Multi-user authentication

### Version 2.0 (Q4 2025)
- [ ] Real-time data streaming
- [ ] AutoML capabilities
- [ ] Natural language query interface
- [ ] Collaborative analysis features

---

## 📞 Contact

**Nicholas Seabright**

- GitHub: [@Nick-seabright](https://github.com/Nick-seabright)
- Email: nick.seabright@example.com
- LinkedIn: [Nicholas Seabright](https://linkedin.com/in/nicholas-seabright)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=Nick-seabright/Statistical-Analytics-App&type=Date)](https://star-history.com/#Nick-seabright/Statistical-Analytics-App&Date)

---

**Happy Analyzing! 📊✨**

*Built with ❤️ by Nick Seabright*
