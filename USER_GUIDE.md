# Data Science App - Complete User Guide

**Version 1.0.0**  
**Created by: Nick Seabright**

---

## 📘 Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Feature Guide](#feature-guide)
5. [Common Use Cases](#common-use-cases)
6. [Tips & Best Practices](#tips--best-practices)
7. [Troubleshooting](#troubleshooting)
8. [FAQs](#faqs)

---

## 🎯 Introduction

### What is the Data Science App?

The **Data Science App** is a comprehensive, no-code platform for data analysis, statistical testing, and machine learning. Built with Streamlit, it provides an intuitive interface for:

- **Data exploration and visualization**
- **Statistical hypothesis testing**
- **Machine learning model training**
- **Predictive analytics**
- **Report generation**

### Who is this for?

- **Data Analysts**: Perform statistical tests without writing code
- **Data Scientists**: Quickly prototype and compare ML models
- **Researchers**: Conduct rigorous statistical analyses
- **Students**: Learn data science concepts interactively
- **Business Users**: Gain insights from data without technical expertise

### Key Benefits

✅ **No Coding Required** - Fully interactive interface  
✅ **End-to-End Analysis** - From upload to report generation  
✅ **Multiple Data Sources** - CSV, Foundry datasets, or sample data  
✅ **Professional Results** - Publication-ready reports and visualizations  
✅ **Educational** - Learn statistical concepts as you work  

---

## 🚀 Getting Started

### Installation & Launch

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the App**
   ```bash
   streamlit run app.py
   ```

3. **Access in Browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to this URL in your browser

### First Time Setup

When you first launch the app, you'll see:
- **Main navigation sidebar** on the left
- **Welcome screen** in the center
- **Data upload section** ready to go

### Understanding the Interface

#### Navigation Sidebar
Located on the left, contains:
- **📥 Data Upload** - Load your data
- **🧹 Data Cleaning** - Clean and prepare data
- **📊 Data Exploration** - Visualize and explore
- **📈 Statistical Analysis** - Run hypothesis tests
- **🎯 Threshold Analysis** - Optimize decision boundaries
- **🤖 Supervised Learning** - Train ML models
- **🔬 Unsupervised Learning** - Clustering and dimensionality reduction
- **🔮 Make Predictions** - Generate predictions
- **📝 Generate Report** - Create comprehensive reports
- **🔄 Reset All** - Start fresh

#### Dataset Info Panel
When data is loaded, the sidebar displays:
- Number of rows and columns
- Target variable name
- Target variable type (classification/regression)
- Number of selected features
- Option to view all features

---

## 📋 Step-by-Step Workflow

### Basic Workflow

```
1. Upload Data → 2. (Optional) Clean Data → 3. Process Data → 
4. Explore → 5. Analyze → 6. Train Models → 7. Predict → 8. Report
```

---

## 🔧 Feature Guide

### 1️⃣ Data Upload (📥)

#### Upload Options

**Option A: CSV File Upload**
1. Click "Browse files" button
2. Select your CSV file from your computer
3. Data will automatically load and display a preview

**Option B: Palantir Foundry Integration**
1. Select a registered dataset from the dropdown, OR
2. Enter a dataset RID manually
3. Click "Load Dataset"

**Option C: Sample Datasets**
- **Student Performance**: Academic performance data (classification/regression)
- **Employee Attrition**: HR turnover prediction (binary classification)
- **Housing Prices**: Real estate pricing (regression)

#### After Upload

Once data is loaded, you'll see:
- **Data Preview** - First 10 rows of your dataset
- **Data Summary** - Row/column counts, missing values
- **Data Types** - Automatically inferred types for each column
- **Target Selection** - Choose your outcome variable
- **Feature Selection** - Choose which columns to include in analysis

#### Processing Your Data

1. **Select Target Variable**: Choose the column you want to predict or analyze
2. **Select Features**: 
   - Check "Select All Features" for convenience, OR
   - Manually select specific features from the multiselect dropdown
3. **Click "Process Data"**: This will:
   - Encode categorical variables
   - Scale numerical features
   - Detect target type (classification vs regression)
   - Prepare data for analysis

**✅ Success!** You'll see a confirmation message and automatically navigate to Data Exploration.

---

### 2️⃣ Data Cleaning (🧹) - OPTIONAL

**When to use this section:**
- Your dataset has missing values
- You need to filter or sample your data
- Large datasets that need reduction
- Data has duplicates
- Need transformations (log, scaling, etc.)

#### Data Quality Overview

Automatically shows:
- **Missing Values**: Heatmap visualization
- **Duplicates**: Count of duplicate rows
- **Data Types**: Column type breakdown
- **Statistics**: Basic descriptive stats

#### Cleaning Operations

##### A. Handle Missing Values

**Strategy Options:**
- **Drop Rows**: Remove any row with missing values
- **Drop Columns**: Remove columns with missing values
- **Fill with Mean**: For numeric columns
- **Fill with Median**: For numeric columns (robust to outliers)
- **Fill with Mode**: For categorical columns
- **Forward Fill**: Use previous value (time series)
- **Backward Fill**: Use next value (time series)
- **Custom Value**: Enter your own fill value

**How to apply:**
1. Select strategy from dropdown
2. If applicable, choose which columns to apply to
3. Changes preview instantly in "Preview Changes" tab

##### B. Filter and Sample Data

**Filter by Value:**
1. Select a column to filter
2. Choose operator (equals, greater than, less than, contains, etc.)
3. Enter filter value
4. Multiple filters can be added

**Sampling Options:**
- **Random Sample**: Specify number of rows or percentage
- **Stratified Sample**: Maintain target variable distribution
  - Useful for maintaining class balance in classification

**Example:**
```
Filter: Age > 18 AND City = "New York"
Sample: 5000 random rows
```

##### C. Remove Duplicates

1. Click "Remove Duplicates" checkbox
2. Option to keep first or last occurrence
3. Preview shows how many duplicates will be removed

##### D. Column Operations

- **Rename Columns**: Change column names for clarity
- **Drop Columns**: Remove unnecessary columns
- **Reorder Columns**: Organize columns logically

##### E. Data Transformations

**Available Transformations:**
- **Log Transform**: For skewed distributions
- **Square Root**: Moderate transformation
- **Standardization**: Mean=0, Std=1 (z-score)
- **Min-Max Normalization**: Scale to 0-1 range

**How to apply:**
1. Select transformation type
2. Choose columns to transform
3. Preview impact before applying

##### Applying Changes

1. Review all changes in **"Preview Changes"** tab
2. Compare side-by-side: Original vs Cleaned
3. Click **"Apply Changes"** to confirm
4. OR click **"Reset"** to start over

**⚠️ Important**: Once applied, changes update `st.session_state.data`

---

### 3️⃣ Data Exploration (📊)

Explore and visualize your processed data.

#### Tab 1: Summary Statistics

**Numeric Features:**
- Count, Mean, Std, Min, 25%, 50%, 75%, Max
- Automatically generated for all numeric columns

**Categorical Features:**
- Top 10 categories by frequency
- Value counts for each categorical variable

#### Tab 2: Correlations

**Correlation Heatmap:**
- Shows Pearson correlation between all numeric features
- Color-coded: Red (positive), Blue (negative)
- Annotated with correlation coefficients

**Feature-Target Correlations:**
- Bar chart showing which features correlate most with target
- Sorted by absolute correlation strength
- Helps identify important predictors

#### Tab 3: Distributions

**For Numeric Features:**
- **Histograms**: Frequency distribution
- **Box Plots**: Quartiles and outliers
- **Violin Plots**: Distribution shape
- **KDE Plots**: Smooth density curves

**For Categorical Features:**
- **Bar Charts**: Category frequencies
- **Pie Charts**: Proportion visualization

**How to use:**
1. Select a feature from dropdown
2. Choose visualization type
3. Chart updates automatically

#### Tab 4: Feature Analysis

**Scatter Plots:**
- Compare two numeric features
- Color-coded by target variable
- Identifies linear/non-linear relationships

**Pair Plots:**
- Matrix of scatter plots for multiple features
- Shows all pairwise relationships
- Great for exploratory analysis

**Category Comparisons:**
- Box plots grouped by categorical variable
- Compare distributions across groups

---

### 4️⃣ Statistical Analysis (📈)

Conduct rigorous statistical hypothesis tests.

#### T-Test (Compare Two Groups)

**When to use:**
- Comparing means of two groups
- Example: "Do males and females have different test scores?"

**Types Available:**
- **Independent Samples**: Two separate groups
- **Paired Samples**: Same subjects measured twice
- **One-Sample**: Compare sample to known population mean

**How to perform:**
1. Select test type
2. Choose grouping variable (e.g., "Gender")
3. Choose value variable (e.g., "Score")
4. Set significance level (default: 0.05)
5. Click "Run T-Test"

**Results Include:**
- t-statistic
- p-value (with interpretation)
- Effect size (Cohen's d)
- 95% Confidence intervals
- Visualization comparing groups

**Interpretation Example:**
```
p-value = 0.003 < 0.05
→ REJECT null hypothesis
→ There IS a significant difference between groups
Effect size (Cohen's d) = 0.62 (Medium effect)
```

#### Chi-Square Test (Categorical Association)

**When to use:**
- Testing association between two categorical variables
- Example: "Is there a relationship between Gender and Pass/Fail?"

**How to perform:**
1. Select first categorical variable
2. Select second categorical variable
3. Click "Run Chi-Square Test"

**Results Include:**
- Chi-square statistic
- p-value
- Degrees of freedom
- Effect size (Cramér's V)
- Contingency table (observed vs expected)
- Heatmap visualization

**Interpretation:**
- p < 0.05: Variables are associated
- p ≥ 0.05: No significant association
- Cramér's V: 0.1 (small), 0.3 (medium), 0.5 (large)

#### ANOVA (Compare Multiple Groups)

**When to use:**
- Comparing means across 3+ groups
- Example: "Do students from 3 different schools have different scores?"

**How to perform:**
1. Select grouping variable (3+ groups)
2. Select value variable (numeric)
3. Click "Run ANOVA"

**Results Include:**
- F-statistic
- p-value
- Effect size (eta-squared)
- Post-hoc tests (Tukey HSD) - identifies which groups differ
- Box plot visualization comparing all groups

**Post-Hoc Results:**
```
Group A vs Group B: p = 0.001 (Significant difference)
Group A vs Group C: p = 0.234 (No difference)
Group B vs Group C: p = 0.045 (Significant difference)
```

#### Advanced Correlation Analysis

**Multi-Group Analysis:**
- Compare distributions across multiple groups simultaneously
- Statistical tests for each pair
- Visual comparison with box plots

**Categorical Association Matrix:**
- Test all categorical variable pairs
- Heatmap of association strengths
- Identify patterns in categorical data

**Numerical Correlation Studies:**
- Pearson, Spearman, Kendall's tau correlations
- Correlation matrix with significance tests
- Partial correlations (control for confounding)

---

### 5️⃣ Threshold Analysis (🎯)

Optimize decision boundaries for classification problems.

**When to use:**
- After training a classification model
- Need to find optimal cutoff point
- Balancing sensitivity and specificity
- Custom business requirements (e.g., "minimize false negatives")

#### Single Feature Threshold

**Analyze Decision Boundaries:**
1. Select a numeric feature
2. System tests all possible thresholds
3. Generates ROC curve
4. Identifies optimal threshold

**Results Include:**
- **ROC Curve**: True Positive Rate vs False Positive Rate
- **AUC Score**: Area under ROC curve (1.0 = perfect)
- **Optimal Threshold**: Best cutoff point
- **Performance Metrics at Optimal**:
  - Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)
  - Precision
  - F1-Score
  - Accuracy

**Visualization:**
- ROC curve with optimal point marked
- Confusion matrix at optimal threshold
- Distribution of feature values by true class

#### Custom Threshold Testing

**Test Specific Thresholds:**
1. Enter custom threshold value
2. View performance metrics
3. Compare against optimal threshold

**Use Case Example:**
```
Medical diagnosis scenario:
- Standard optimal threshold: 0.5 (70% sensitivity)
- Custom threshold: 0.3 (90% sensitivity)
- Trade-off: More false positives, but catch more true cases
```

#### Multi-Feature Combinations

**Test Multiple Thresholds:**
- Combine conditions on multiple features
- Example: "Feature1 > 5 AND Feature2 < 10"
- Evaluate combined rule performance

---

### 6️⃣ Supervised Learning (🤖)

Train machine learning models to make predictions.

#### Classification Algorithms

**Available Models:**
- **Logistic Regression**: Simple, interpretable baseline
- **Random Forest**: Robust, handles non-linearity
- **Support Vector Machine (SVM)**: Good for complex boundaries
- **XGBoost**: High performance, gradient boosting
- **Neural Network**: Deep learning (TensorFlow/Keras)

#### Regression Algorithms

**Available Models:**
- **Linear Regression**: Simple linear relationships
- **Random Forest**: Non-linear patterns
- **Support Vector Regression (SVR)**: Complex patterns
- **XGBoost**: Gradient boosted trees
- **Neural Network**: Deep learning regressor

#### Training Process

**Step 1: Select Models**
- Check boxes for models you want to train
- Can train multiple models simultaneously
- Recommended: Start with 2-3 models for comparison

**Step 2: Configure Training**
- **Test Size**: Percentage of data for testing (default: 20%)
- **Cross-Validation Folds**: Number of CV folds (default: 5)
- **Random Seed**: For reproducibility (default: 42)

**Step 3: Train Models**
1. Click "Train Models" button
2. Wait for training to complete (progress bar shown)
3. Models are automatically evaluated

#### Results & Evaluation

**For Classification:**

**Metrics Displayed:**
- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, % actually positive
- **Recall (Sensitivity)**: Of actual positives, % correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

**Visualizations:**
- **Confusion Matrix**: True vs predicted classes
- **ROC Curve**: For each model
- **Feature Importance**: Which features matter most
- **Model Comparison**: Side-by-side metrics

**Example Output:**
```
Random Forest:
- Accuracy: 0.87
- Precision: 0.85
- Recall: 0.89
- F1-Score: 0.87
- ROC-AUC: 0.92

Top 3 Important Features:
1. Study_Hours: 0.35
2. Attendance: 0.28
3. Previous_Grade: 0.19
```

**For Regression:**

**Metrics Displayed:**
- **R² Score**: Proportion of variance explained (0-1, higher better)
- **RMSE**: Root Mean Squared Error (same units as target)
- **MAE**: Mean Absolute Error (average error)
- **MAPE**: Mean Absolute Percentage Error (% error)

**Visualizations:**
- **Actual vs Predicted**: Scatter plot
- **Residual Plot**: Errors distribution
- **Feature Importance**: Impact of each feature
- **Model Comparison**: Performance table

#### Feature Importance

**Understanding Feature Importance:**
- Shows which features contribute most to predictions
- Values sum to 1.0 (100%)
- Higher values = more important

**Use Cases:**
- **Feature Selection**: Remove low-importance features
- **Interpretation**: Explain model decisions
- **Domain Validation**: Check if importance matches expectations

**Example:**
```
Feature Importance:
┌─────────────────┬────────────┐
│ Feature         │ Importance │
├─────────────────┼────────────┤
│ Income          │ 0.42       │
│ Credit_Score    │ 0.31       │
│ Loan_Amount     │ 0.18       │
│ Age             │ 0.09       │
└─────────────────┴────────────┘
```

#### Hyperparameter Tuning (Advanced)

**When to use:**
- After initial model training
- To squeeze out extra performance
- For production deployment

**Available Tuning Methods:**
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling (faster)
- **Bayesian Optimization**: Smart search (most efficient)

**Common Hyperparameters:**
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **XGBoost**: learning_rate, max_depth, n_estimators
- **SVM**: C, gamma, kernel
- **Neural Network**: layers, neurons, learning_rate

---

### 7️⃣ Unsupervised Learning (🔬)

Discover patterns without labeled data.

#### Clustering

**When to use:**
- Group similar data points
- Customer segmentation
- Anomaly detection
- Pattern discovery

**Algorithms Available:**

**K-Means Clustering:**
- Fast and simple
- Need to specify number of clusters (k)
- Works well for spherical clusters

**How to use:**
1. Select features for clustering
2. Choose number of clusters (2-10 recommended)
3. Click "Run K-Means"
4. View cluster assignments and visualization

**Results:**
- Cluster centers
- Silhouette score (quality metric)
- Cluster size distribution
- 2D visualization (PCA projection if high-dimensional)

**Hierarchical Clustering:**
- Creates tree structure (dendrogram)
- No need to specify k upfront
- Shows cluster hierarchy

**DBSCAN:**
- Density-based clustering
- Automatically finds number of clusters
- Good for non-spherical clusters
- Identifies outliers as noise

**Gaussian Mixture Models:**
- Probabilistic clustering
- Soft assignments (probability of belonging to each cluster)
- Good for overlapping clusters

#### Dimensionality Reduction

**When to use:**
- Too many features (high-dimensional data)
- Visualization of complex data
- Feature engineering
- Remove noise

**Algorithms:**

**PCA (Principal Component Analysis):**
- Linear method
- Preserves maximum variance
- Fast and interpretable
- Results: Explained variance ratio, component loadings

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
- Non-linear method
- Excellent for visualization
- Preserves local structure
- Good for complex patterns

**UMAP (Uniform Manifold Approximation and Projection):**
- Non-linear method
- Faster than t-SNE
- Preserves global and local structure
- State-of-the-art for visualization

**How to use:**
1. Select dimensionality reduction method
2. Choose number of components (2 or 3 for visualization)
3. Click "Run"
4. View 2D/3D scatter plot

#### Anomaly Detection

**When to use:**
- Fraud detection
- Quality control
- System monitoring
- Outlier removal

**Algorithms:**

**Isolation Forest:**
- Fast and scalable
- Good for high-dimensional data
- Contamination parameter: expected % of outliers

**Local Outlier Factor (LOF):**
- Density-based
- Detects local outliers
- n_neighbors parameter: number of neighbors to consider

**One-Class SVM:**
- Robust
- Kernel-based
- Good for complex patterns

**Results:**
- Anomaly labels (-1 = anomaly, 1 = normal)
- Anomaly scores
- Visualization with anomalies highlighted
- Statistics on detected anomalies

---

### 8️⃣ Make Predictions (🔮)

Use trained models to make predictions on new data.

**Prerequisites:**
- At least one model trained in Supervised Learning section
- Model stored in `st.session_state.models`

#### Single Prediction

**When to use:**
- Predict outcome for one case
- Interactive "what-if" analysis
- Real-time decision support

**How to use:**
1. Select which trained model to use
2. Enter values for each feature:
   - Numeric features: Enter number
   - Categorical features: Select from dropdown
3. Click "Make Prediction"

**Results:**
- **Predicted Value**: The model's prediction
- **Confidence Score**: Probability or confidence (for classification)
- **Interpretation**: Plain-English explanation
- **Feature Values Used**: Summary of inputs

**Example Output:**
```
Prediction: PASS (91.2% confidence)

Interpretation:
Based on your inputs, the model predicts the student will PASS 
with high confidence. Key factors:
- Study Hours (10) - Above average
- Attendance (95%) - Excellent
- Previous Grade (85) - Good performance
```

#### Batch Predictions

**When to use:**
- Predict for many cases at once
- Production deployment
- Bulk scoring

**How to use:**
1. Prepare CSV file with same columns as training features
2. Upload file using file uploader
3. Select model
4. Click "Generate Predictions"

**Results:**
- Original data + predictions appended
- Download button for results CSV
- Summary statistics on predictions
- Distribution of predicted classes/values

**Example:**
```
Input: 500 records
Output: 500 predictions

Predicted Distribution:
- PASS: 342 (68.4%)
- FAIL: 158 (31.6%)

Download: predictions_2024-01-15.csv
```

#### Prediction Interpretation

**For Classification:**
- Predicted class
- Probability for each class
- Decision reasoning (if available)
- Comparison to training data

**For Regression:**
- Predicted value
- Confidence interval
- Comparison to typical range
- Contributing factors

---

### 9️⃣ Generate Report (📝)

Create comprehensive analysis reports.

#### Report Sections

**Automatically Included:**
- **Executive Summary**: Key findings overview
- **Data Overview**: Dataset description and statistics
- **Data Quality**: Missing values, cleaning steps performed
- **Exploratory Analysis**: Key visualizations and insights

**Optional Sections (Select):**
- **Statistical Tests**: All hypothesis test results
- **Threshold Analysis**: ROC curves and optimal thresholds
- **Model Training**: Model comparison and performance
- **Feature Importance**: Which features matter most
- **Predictions**: Prediction results and examples
- **Methodology**: Detailed methods used

#### Customization Options

**Report Settings:**
- **Title**: Custom report title
- **Author**: Your name or organization
- **Date**: Auto-filled or custom
- **Include Visualizations**: Toggle charts on/off
- **Detail Level**: Summary vs Full details

#### Export Formats

**HTML Report:**
- Interactive charts
- Professional formatting
- Easy to share via email
- Open in any browser

**PDF Report:**
- Print-ready
- Fixed layout
- Professional appearance
- Good for formal documentation

**Markdown:**
- Plain text format
- Easy to version control
- Can convert to other formats
- Good for technical audiences

**JSON (Data Export):**
- Machine-readable
- All results and metrics
- For further processing
- Integration with other systems

#### How to Generate

1. Click "📝 Generate Report" in sidebar
2. Select sections to include (checkboxes)
3. Configure report settings
4. Click "Generate Report"
5. Preview report in app
6. Click download button for desired format

**Report Content Example:**
```
─────────────────────────────
 STATISTICAL ANALYSIS REPORT
─────────────────────────────

Date: January 15, 2024
Author: Data Science Team
Dataset: Student Performance

EXECUTIVE SUMMARY
─────────────────
• 1,000 students analyzed
• 85% pass rate observed
• Study hours strongly predict success
• Significant gender difference detected

DATA OVERVIEW
─────────────
Rows: 1,000
Features: 15
Target: Pass/Fail (Binary)

... [continued]
```

---

## 💼 Common Use Cases

### Use Case 1: Academic Research

**Scenario**: Analyzing student performance factors

**Workflow:**
```
1. Upload CSV with student data
2. Explore correlations between study habits and grades
3. Run t-test: Males vs Females performance
4. Run ANOVA: Performance across different schools
5. Train Random Forest to predict outcomes
6. Generate report with all statistical tests
```

**Key Features Used:**
- Statistical Analysis (t-test, ANOVA)
- Data Exploration (correlations)
- Supervised Learning (prediction)
- Report Generation

---

### Use Case 2: Business Analytics

**Scenario**: Customer churn prediction

**Workflow:**
```
1. Load Foundry dataset with customer data
2. Clean: Handle missing values, sample 10K records
3. Explore: Find patterns in churned vs retained customers
4. Train models: XGBoost, Random Forest
5. Optimize threshold: Balance retention efforts vs cost
6. Batch predict: Score all current customers
7. Export predictions for CRM system
```

**Key Features Used:**
- Data Cleaning (sampling, missing values)
- Supervised Learning (classification)
- Threshold Analysis (optimization)
- Batch Predictions

---

### Use Case 3: Medical Research

**Scenario**: Disease diagnosis support

**Workflow:**
```
1. Upload clinical trial data
2. Explore distributions of biomarkers
3. Run statistical tests comparing treatment vs control
4. Train Neural Network for diagnosis prediction
5. Optimize threshold for high sensitivity (catch all cases)
6. Generate detailed report with methodology
```

**Key Features Used:**
- Statistical Analysis (hypothesis testing)
- Threshold Analysis (sensitivity optimization)
- Supervised Learning (classification)
- Report Generation (methodology)

---

### Use Case 4: Marketing Analytics

**Scenario**: Customer segmentation

**Workflow:**
```
1. Load customer behavior data
2. Run K-Means clustering (5 segments)
3. Visualize segments with t-SNE
4. Analyze characteristics of each segment
5. Generate profile report for each cluster
```

**Key Features Used:**
- Unsupervised Learning (K-Means, t-SNE)
- Data Exploration (segment profiling)
- Report Generation

---

### Use Case 5: Quality Control

**Scenario**: Manufacturing defect detection

**Workflow:**
```
1. Upload production data
2. Clean: Remove outliers, handle missing sensors
3. Detect anomalies using Isolation Forest
4. Train model to predict defects
5. Optimize threshold: Minimize false negatives
6. Deploy: Batch predictions on new production runs
```

**Key Features Used:**
- Data Cleaning
- Unsupervised Learning (anomaly detection)
- Supervised Learning
- Threshold Analysis
- Batch Predictions

---

## 💡 Tips & Best Practices

### Data Preparation

✅ **DO:**
- Clean data before processing (handle missing values)
- Remove irrelevant columns
- Check for and handle outliers
- Use stratified sampling for imbalanced datasets
- Verify data types are correct

❌ **DON'T:**
- Process data with many missing values without cleaning
- Include ID columns or timestamps as features
- Use tiny samples (<100 rows) for statistical tests
- Mix training and test data

### Statistical Testing

✅ **DO:**
- Check assumptions (normality, equal variances)
- Use appropriate test for your data type
- Report effect sizes, not just p-values
- Use multiple testing corrections when running many tests
- Visualize your results

❌ **DON'T:**
- Fish for significant results (p-hacking)
- Ignore small effect sizes even if p < 0.05
- Test tiny groups (<20 per group)
- Forget to check assumptions

### Machine Learning

✅ **DO:**
- Start with simple models (Logistic Regression, Linear Regression)
- Use cross-validation
- Check feature importance
- Test on truly unseen data
- Compare multiple models
- Save your best models

❌ **DON'T:**
- Overfit by using too many features with few samples
- Forget to scale features
- Use accuracy alone for imbalanced datasets
- Deploy without validating on test set
- Ignore feature engineering

### Model Selection

**For Classification:**
- **Logistic Regression**: Baseline, interpretable
- **Random Forest**: First complex model to try
- **XGBoost**: Best performance usually
- **Neural Network**: Very large datasets, complex patterns

**For Regression:**
- **Linear Regression**: Baseline, interpretable
- **Random Forest**: Handles non-linearity
- **XGBoost**: Top performance
- **Neural Network**: Complex patterns

### Performance Optimization

**Large Datasets (>100K rows):**
1. Use Data Cleaning → Sample Data
2. Sample 10K-50K rows for initial exploration
3. Train on full data only for final model

**Many Features (>50):**
1. Check correlations, remove redundant features
2. Use feature importance to select top features
3. Consider PCA for dimensionality reduction

**Slow Training:**
1. Reduce cross-validation folds (5 → 3)
2. Reduce hyperparameter search space
3. Use Random Search instead of Grid Search

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Issue: "Please process your data first"

**Cause**: Trying to analyze/train without processing data  
**Solution**: 
1. Go to Data Upload section
2. Select target variable and features
3. Click "Process Data" button

---

#### Issue: "Not enough data for analysis"

**Cause**: Dataset too small for statistical tests or ML  
**Solution**:
- Need minimum 30 rows per group for t-test
- Need minimum 100 rows for ANOVA
- Need minimum 200 rows for ML model training

---

#### Issue: Model training fails with error

**Possible Causes & Solutions:**

**"ValueError: could not convert string to float"**
- Categorical variable not properly encoded
- Solution: Ensure all categorical columns are in feature selection

**"Memory Error"**
- Dataset too large
- Solution: Use Data Cleaning → Sample Data to reduce size

**"No convergence"**
- Neural network training failed
- Solution: Increase max_epochs or change learning_rate

---

#### Issue: Very low model accuracy (<60%)

**Possible Causes:**
- Not enough data
- Wrong features selected
- Target variable has too much noise
- Highly imbalanced classes

**Solutions:**
- Collect more data
- Perform feature engineering
- Try different models (Random Forest, XGBoost)
- For imbalanced data: use stratified sampling

---

#### Issue: Predictions don't match training performance

**Cause**: Overfitting  
**Solutions:**
- Use more cross-validation folds
- Reduce model complexity
- Add more training data
- Check for data leakage (target info in features)

---

#### Issue: Can't upload file

**Possible Causes:**
- File too large (>200MB default Streamlit limit)
- Not CSV format
- Encoding issues

**Solutions:**
- Compress file or sample data before upload
- Convert Excel/JSON to CSV
- Save CSV with UTF-8 encoding

---

#### Issue: Statistical test p-values are all 1.0 or NaN

**Causes:**
- Not enough variance in data
- Groups are too small
- Data type mismatch

**Solutions:**
- Check that you have numeric data for numerical tests
- Ensure groups have different values
- Need at least 10 observations per group

---

## ❓ FAQs

### General Questions

**Q: Do I need to know Python to use this app?**  
A: No! The app is completely no-code. Everything is done through the visual interface.

**Q: Can I use my own data?**  
A: Yes! Upload any CSV file or connect to Foundry datasets.

**Q: Is my data saved or stored?**  
A: Data is only stored in your browser session (RAM). When you close the browser, it's gone.

**Q: Can multiple people use the app simultaneously?**  
A: Yes, each user has their own independent session.

**Q: How do I save my work?**  
A: Use the "Generate Report" feature to export your analysis. Download trained models using the save buttons in Model Training section.

---

### Data Questions

**Q: What file formats are supported?**  
A: Currently only CSV files. Excel files should be converted to CSV first.

**Q: What's the maximum dataset size?**  
A: Depends on your system memory. Recommended: <100K rows for smooth performance. Use sampling for larger datasets.

**Q: Can I use data with missing values?**  
A: Yes! Use the Data Cleaning section to handle missing values before processing.

**Q: Can I analyze time series data?**  
A: Limited support. You can analyze time-based features, but specialized time series forecasting is not included.

---

### Statistical Analysis Questions

**Q: Which statistical test should I use?**  
A: 
- **T-test**: Compare means of 2 groups
- **ANOVA**: Compare means of 3+ groups
- **Chi-square**: Test association between 2 categorical variables
- **Correlation**: Measure linear relationship between 2 numeric variables

**Q: What does p-value mean?**  
A: p-value is the probability of seeing your results if there's actually no effect. 
- p < 0.05: Significant result (reject null hypothesis)
- p ≥ 0.05: Not significant (fail to reject null hypothesis)

**Q: What's a good effect size?**  
A: 
- **Cohen's d**: 0.2 (small), 0.5 (medium), 0.8 (large)
- **Cramér's V**: 0.1 (small), 0.3 (medium), 0.5 (large)
- **Eta-squared**: 0.01 (small), 0.06 (medium), 0.14 (large)

---

### Machine Learning Questions

**Q: How do I know which model to choose?**  
A: Start with simple models (Logistic/Linear Regression). If performance is poor, try Random Forest or XGBoost. Compare multiple models.

**Q: What's a good accuracy score?**  
A: Depends on your problem:
- **>90%**: Excellent (but watch for overfitting)
- **80-90%**: Very good
- **70-80%**: Good
- **<70%**: May need improvement

**Q: My model has 99% accuracy - is that good?**  
A: Maybe! Check if data is imbalanced. If 99% of samples are class A, a dumb model predicting always A gets 99% accuracy. Check precision, recall, and confusion matrix.

**Q: What's the difference between accuracy and F1-score?**  
A:
- **Accuracy**: Overall % correct
- **F1-Score**: Balanced measure of precision and recall, better for imbalanced data

**Q: How many features should I include?**  
A: Rule of thumb: At least 10 samples per feature. More features ≠ better model. Start with important features based on domain knowledge and feature importance.

---

### Technical Questions

**Q: Can I export my trained model?**  
A: Yes! Models are saved in your session. You can download model files from the Model Training section.

**Q: Can I use this app offline?**  
A: Yes, once installed. Just run `streamlit run app.py` locally.

**Q: Does this work on mobile devices?**  
A: The interface works on mobile but is optimized for desktop/laptop screens.

**Q: Can I integrate this with other systems?**  
A: The app can read from Foundry datasets. For other integrations, you can export results and import elsewhere.

---

## 📞 Support & Resources

### Getting Help

- **GitHub Issues**: [Submit an issue](https://github.com/Nick-seabright/Statistical-Analysis-App/issues)
- **Documentation**: See `docs/` folder for detailed guides
- **Email Support**: Contact app creator

### Additional Resources

- **Data Cleaning Guide**: `docs/data_cleaning_guide.md`
- **Workflow Guide**: `docs/app_workflow.md`
- **Unsupervised Learning**: `docs/unsupervised_learning_guide.md`

### Video Tutorials (Coming Soon)

- Getting Started (10 min)
- Statistical Analysis Walkthrough (15 min)
- Building Your First Model (20 min)
- Advanced Features (25 min)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

**Created by**: Nick Seabright  
**Version**: 1.0.0  
**Last Updated**: 2024

---

**Happy Analyzing! 📊🚀**

*For issues, suggestions, or contributions, visit the [GitHub repository](https://github.com/Nick-seabright/Statistical-Analytics-App)*
