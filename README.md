# Statistical Analysis App

## Overview
The Statistical Analysis App is a comprehensive data analysis and machine learning tool built with Streamlit. It provides an intuitive interface for data exploration, statistical testing, threshold analysis, model training, and predictions. The app is designed to make sophisticated statistical and machine learning analyses accessible without requiring programming knowledge.

## Features

### Data Management
- **Data Upload**: Import CSV files with automatic data type detection
- **Sample Datasets**: Access built-in sample datasets for testing and learning
- **Data Processing**: Automatic handling of missing values and data type conversion

### Data Exploration
- **Summary Statistics**: View basic statistics for numeric variables
- **Distribution Analysis**: Visualize data distributions with histograms and box plots
- **Correlation Analysis**: Explore relationships between variables with correlation matrices and scatter plots
- **Categorical Analysis**: Analyze categorical data with bar charts and cross-tabulations

### Statistical Testing
- **T-Tests**: Compare means between two groups
- **Chi-Square Tests**: Analyze relationships between categorical variables
- **ANOVA**: Compare means across multiple groups
- **Correlation Analysis**: Measure and test linear relationships between variables

### Threshold Analysis
- **Single Feature Thresholds**: Find optimal decision boundaries for individual features
- **Feature Combinations**: Analyze how pairs of features interact to affect outcomes
- **Custom Threshold Testing**: Test specific threshold values and visualize results

### Model Training
- **Classification Models**:
  - Random Forest Classifier
  - Support Vector Machine
  - XGBoost Classifier
  - Neural Networks
- **Regression Models**:
  - Random Forest Regressor
  - Support Vector Regressor
  - XGBoost Regressor
  - Neural Network Regressor
- **Model Evaluation**: Comprehensive evaluation metrics with cross-validation
- **Feature Importance**: Visualize which features contribute most to predictions

### Predictions
- **Single Predictions**: Make predictions on individual data points
- **Batch Predictions**: Process multiple predictions by uploading a file
- **Prediction Interpretation**: Understand prediction results with visualizations

### Report Generation
- **Customizable Reports**: Generate PDF and HTML reports with selected analyses
- **Data Visualization**: Include charts and tables in reports
- **Export Options**: Save reports locally or download directly

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/statistical-analysis-app.git
cd statistical-analysis-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- tensorflow
- statsmodels
- scipy
- weasyprint (for PDF report generation)

## Usage

### Starting the App
Run the app with Streamlit:
```bash
streamlit run streamlit_app/app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### Workflow
1. **Upload Data**: Start by uploading a CSV file or selecting a sample dataset
2. **Explore Data**: Examine distributions, correlations, and basic statistics
3. **Statistical Tests**: Perform appropriate statistical tests based on your research questions
4. **Threshold Analysis**: Find optimal decision boundaries for classification problems
5. **Train Models**: Build and compare machine learning models
6. **Make Predictions**: Apply trained models to new data
7. **Generate Reports**: Create comprehensive reports of your findings

## Directory Structure
```
statistical-analysis-app/
├── edu_analytics/               # Core analytics modules
│   ├── __init__.py
│   ├── data_processing.py       # Data preparation and processing
│   ├── feature_engineering.py   # Feature creation and selection
│   ├── model_evaluation.py      # Model assessment tools
│   ├── model_training.py        # Machine learning model implementations
│   ├── statistical_tests.py     # Statistical testing functions
│   ├── threshold_analysis.py    # Decision boundary optimization
│   ├── time_analysis.py         # Time variable handling
│   └── utils.py                 # Utility functions
├── streamlit_app/               # Streamlit interface
│   ├── app.py                   # Main application entry point
│   ├── assets/                  # Static assets (images, etc.)
│   ├── components/              # Reusable UI components
│   │   ├── file_browser.py      # File system navigation
│   │   ├── model_cards.py       # Model information display
│   │   ├── plots.py             # Visualization components
│   │   └── sidebar.py           # Navigation sidebar
│   └── pages/                   # Application pages
│       ├── data_exploration.py  # Data analysis page
│       ├── data_upload.py       # Data import page
│       ├── file_management.py   # File system management
│       ├── model_training.py    # Model building page
│       ├── predictions.py       # Prediction interface
│       ├── report_generation.py # Report creation page
│       ├── statistical_analysis.py # Statistical testing page
│       └── threshold_analysis.py # Threshold optimization page
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Advanced Features

### Time Data Analysis
The app includes specialized support for time data (HH:MM:SS format):
- Automatic detection and conversion of time variables
- Statistical analysis based on time thresholds
- Time-based feature engineering

### Custom Threshold Optimization
Find the exact decision boundaries that maximize class separation:
- Interactive threshold visualization
- Detailed statistical analysis of each threshold
- Multi-feature quadrant analysis for complex patterns

### Model Persistence
Save and load trained models for future use:
- Export models to disk
- Import previously trained models
- Share models with colleagues

## Troubleshooting

### Common Issues
- **Memory Errors**: For large datasets, increase available memory with `--memory.available_memory=4GB` when starting Streamlit
- **Missing Dependencies**: Ensure all packages are installed with `pip install -r requirements.txt`
- **Report Generation Errors**: For PDF generation, verify WeasyPrint is properly installed with all dependencies

### Getting Help
If you encounter problems:
1. Check the console for error messages
2. Verify your data format matches the expected input
3. For complex datasets, try processing in smaller batches

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The scikit-learn team for their excellent machine learning library
- The Streamlit team for making data app creation simple and powerful
- All contributors who have helped improve this application

## Contact
For questions or feedback, please reach out to [your-email@example.com](mailto:your-email@example.com) or open an issue on GitHub.

---

*This application is for educational and research purposes. Always validate statistical findings through appropriate scientific methods.*