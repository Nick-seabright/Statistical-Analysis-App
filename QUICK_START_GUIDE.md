# Data Science App - Quick Start Guide

**Version 1.0.0** | **Created by: Nick Seabright**

---

## 🚀 Get Started in 5 Minutes

### Launch the App

```bash
streamlit run app.py
```

Opens at: `http://localhost:8501`

---

## 📋 Quick Workflow

### Basic Analysis (3 Steps)

```
1. Upload CSV → 2. Process Data → 3. Explore & Analyze
```

### Full ML Pipeline (5 Steps)

```
1. Upload → 2. Clean → 3. Process → 4. Train Models → 5. Predict
```

---

## 🎯 Common Tasks

### Task 1: Run a T-Test

**Goal**: Compare two groups (e.g., Male vs Female scores)

1. **Upload data** → Select CSV file
2. **Process data** → Choose target & features → Click "Process Data"
3. **Statistical Analysis** → Select "T-Test"
4. **Configure**:
   - Grouping variable: `Gender`
   - Value variable: `Score`
5. **Run** → Click "Run T-Test"
6. **Interpret**: Check p-value (< 0.05 = significant difference)

**Time**: ~2 minutes

---

### Task 2: Train a Prediction Model

**Goal**: Predict student pass/fail

1. **Upload data** → Load student dataset
2. **Process data**:
   - Target: `Pass_Fail`
   - Features: Select all relevant columns
3. **Supervised Learning**:
   - Select models: `Random Forest`, `XGBoost`
   - Click "Train Models"
4. **Review results**:
   - Compare accuracy scores
   - Check confusion matrices
   - View feature importance
5. **Save best model** → Click "Save Model"

**Time**: ~3-5 minutes

---

### Task 3: Make Predictions

**Goal**: Predict outcome for new cases

**Single Prediction:**
1. Go to **Make Predictions**
2. Select trained model from dropdown
3. Enter feature values
4. Click "Make Prediction"
5. View result & confidence score

**Batch Prediction:**
1. Prepare CSV with same columns as training data
2. Upload CSV file
3. Select model
4. Click "Generate Predictions"
5. Download results CSV

**Time**: ~1 minute

---

### Task 4: Clean Messy Data

**Goal**: Handle missing values and outliers

1. **Upload data** → Load CSV
2. **Data Cleaning**:
   - **Missing Values**:
     - Strategy: `Fill with Mean` (for numeric)
     - Or: `Drop Rows` (if <5% missing)
   - **Sampling** (if >100K rows):
     - Type: `Random Sample`
     - Size: `10,000 rows`
   - **Preview changes** → Check result
3. **Apply Changes** → Update data
4. **Process Data** → Continue to analysis

**Time**: ~2-3 minutes

---

### Task 5: Generate Report

**Goal**: Create comprehensive analysis report

1. Complete analysis (tests, models, etc.)
2. **Generate Report** section
3. **Select sections**:
   - ✅ Executive Summary
   - ✅ Statistical Tests
   - ✅ Model Results
   - ✅ Visualizations
4. **Configure**:
   - Title: Your report name
   - Author: Your name
5. **Generate** → Click button
6. **Download** → Choose format (PDF/HTML)

**Time**: ~1 minute

---

## 🔑 Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Navigate sidebar | Arrow keys |
| Upload file | Ctrl+O |
| Run current section | Enter |
| Reset app | Ctrl+R (in sidebar) |

---

## 📊 Data Format Requirements

### CSV File Structure

```csv
StudentID,Gender,Age,StudyHours,Attendance,Grade,PassFail
1,Male,18,10,95,85,Pass
2,Female,19,8,88,78,Pass
3,Male,18,5,70,65,Fail
...
```

### Requirements

✅ **Headers**: First row must contain column names  
✅ **Encoding**: UTF-8 recommended  
✅ **Delimiters**: Comma (,) or semicolon (;)  
✅ **Missing values**: Can be blank, "NA", "NaN", or "null"  
✅ **Size**: Recommended < 100K rows for smooth performance

❌ **Avoid**:
- Special characters in column names
- Mixed data types in same column
- Duplicate column names

---

## 🎯 Model Selection Guide

### Classification Problems

**Predict categories** (Pass/Fail, Yes/No, Category A/B/C)

| Model | When to Use | Pros | Cons |
|-------|------------|------|------|
| **Logistic Regression** | Start here! Simple problems | Fast, interpretable | Only linear boundaries |
| **Random Forest** | Complex patterns | Robust, handles non-linearity | Slower, less interpretable |
| **XGBoost** | Best performance | Highest accuracy usually | Requires tuning |
| **Neural Network** | Large data (>10K rows) | Flexible, powerful | Needs lots of data, slow |

### Regression Problems

**Predict numbers** (Price, Score, Amount)

| Model | When to Use | Pros | Cons |
|-------|------------|------|------|
| **Linear Regression** | Start here! Linear relationships | Simple, interpretable | Only linear |
| **Random Forest** | Non-linear patterns | Robust, accurate | Less interpretable |
| **XGBoost** | Best performance | Top accuracy | Slower |
| **Neural Network** | Complex patterns, large data | Very flexible | Needs many samples |

---

## 📈 Statistical Test Selection

### Choose Your Test

```
Is data numeric or categorical?
│
├─ NUMERIC
│  │
│  ├─ Compare 2 groups? → T-TEST
│  ├─ Compare 3+ groups? → ANOVA
│  └─ Measure relationship? → CORRELATION
│
└─ CATEGORICAL
   │
   └─ Test association? → CHI-SQUARE TEST
```

### Test Details

| Test | Purpose | Example Question |
|------|---------|------------------|
| **T-Test** | Compare means of 2 groups | "Do males and females have different scores?" |
| **ANOVA** | Compare means of 3+ groups | "Do students from 3 schools differ in performance?" |
| **Chi-Square** | Test categorical association | "Is gender related to pass/fail status?" |
| **Correlation** | Measure linear relationship | "How strongly do study hours relate to grade?" |

---

## 🎨 Visualization Guide

### Exploration Tab

| Chart Type | Best For | Example Use |
|------------|----------|-------------|
| **Histogram** | Distribution of single variable | Age distribution |
| **Box Plot** | Compare distributions across groups | Scores by gender |
| **Scatter Plot** | Relationship between 2 variables | Study hours vs grade |
| **Heatmap** | Correlations between many variables | Feature correlation matrix |
| **Bar Chart** | Categorical frequencies | Count by category |

---

## ⚡ Performance Tips

### Slow Performance?

**Large Dataset (>50K rows)?**
```
Solution: Use Data Cleaning → Random Sample → 10,000 rows
```

**Many Features (>50)?**
```
Solution: 
1. Remove low-variance features
2. Check correlations, remove duplicates
3. Use feature importance to select top 20
```

**Model Training Slow?**
```
Solution:
1. Reduce CV folds: 5 → 3
2. Start with smaller models (Logistic/Linear first)
3. Sample data for initial experiments
```

---

## ❗ Common Errors & Quick Fixes

### "Please process your data first"
**Fix**: Go to Data Upload → Select target & features → Click "Process Data"

### "Not enough data for analysis"
**Fix**: Need minimum 30 samples per group for statistical tests

### Model accuracy is very low (<60%)
**Fix**: 
1. Check data quality
2. Try different models (Random Forest, XGBoost)
3. Add more features
4. Check for class imbalance

### Can't upload file
**Fix**:
1. Ensure file is CSV format
2. Check file size (<200MB)
3. Save with UTF-8 encoding

### All p-values are 1.0 or NaN
**Fix**:
1. Check groups have different values
2. Ensure enough samples (>10 per group)
3. Verify data types are correct

---

## 🎓 Interpretation Cheat Sheet

### P-Values

| p-value | Interpretation | Decision |
|---------|---------------|----------|
| < 0.001 | Highly significant | *** Strong evidence |
| < 0.01 | Very significant | ** Moderate evidence |
| < 0.05 | Significant | * Some evidence |
| ≥ 0.05 | Not significant | No evidence |

### Effect Sizes

**Cohen's d (T-test)**
- 0.2 = Small effect
- 0.5 = Medium effect
- 0.8 = Large effect

**R² (Regression)**
- 0.25 = Weak model
- 0.50 = Moderate model
- 0.75 = Strong model

**Cramér's V (Chi-square)**
- 0.1 = Small association
- 0.3 = Medium association
- 0.5 = Large association

### Model Metrics

**Classification Accuracy**
- 90-100%: Excellent (check for overfitting!)
- 80-90%: Very Good
- 70-80%: Good
- 60-70%: Fair
- <60%: Poor (needs improvement)

**R² Score (Regression)**
- 0.9-1.0: Excellent
- 0.7-0.9: Good
- 0.5-0.7: Moderate
- <0.5: Poor

---

## 🔖 Quick Reference: App Sections

| Section | Icon | Use For |
|---------|------|---------|
| **Data Upload** | 📥 | Load CSV, Foundry data, or samples |
| **Data Cleaning** | 🧹 | Handle missing values, filter, sample |
| **Data Exploration** | 📊 | Visualize distributions, correlations |
| **Statistical Analysis** | 📈 | T-tests, ANOVA, Chi-square |
| **Threshold Analysis** | 🎯 | Optimize classification cutoffs |
| **Supervised Learning** | 🤖 | Train prediction models |
| **Unsupervised Learning** | 🔬 | Clustering, PCA, anomaly detection |
| **Make Predictions** | 🔮 | Single or batch predictions |
| **Generate Report** | 📝 | Export comprehensive reports |

---

## 📱 Mobile Tips

While the app works on mobile, for best experience:
- Use landscape orientation
- Desktop/laptop recommended for:
  - Model training
  - Complex visualizations
  - Report generation

Mobile works well for:
- Data exploration
- Single predictions
- Statistical summaries

---

## 🎯 Success Checklist

Before starting analysis:
- [ ] Data is in CSV format
- [ ] Column names are clear and unique
- [ ] No completely empty columns
- [ ] Target variable is identified
- [ ] Features make sense for prediction

Before training models:
- [ ] Data is cleaned (missing values handled)
- [ ] Features are selected
- [ ] Target variable is correct
- [ ] Have enough data (>200 rows recommended)
- [ ] Data is processed (encoded and scaled)

Before making predictions:
- [ ] Model is trained and saved
- [ ] New data has same columns as training
- [ ] Values are in same format/range
- [ ] Model performance is acceptable

---

## 💡 Pro Tips

1. **Always start with data exploration** - Understand your data before modeling
2. **Use stratified sampling** for imbalanced datasets
3. **Compare multiple models** - Don't rely on just one
4. **Check feature importance** - Understand what drives predictions
5. **Save your work** - Download reports and models regularly
6. **Start simple** - Logistic/Linear Regression before complex models
7. **Cross-validate** - Don't trust single train/test split
8. **Document your process** - Use Report Generation feature

---

## 📞 Need More Help?

- **Full Guide**: See `USER_GUIDE.md` for detailed instructions
- **Workflow**: See `docs/app_workflow.md` for detailed flow
- **Issues**: [GitHub Issues](https://github.com/Nick-seabright/Statistical-Analysis-App/issues)

---

**🚀 Ready to Start? Open the app and click "📥 Data Upload"!**

*Happy Analyzing! - Nick Seabright*
