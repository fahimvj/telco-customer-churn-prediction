# Telco Customer Churn Prediction: A Machine Learning Approach

**Course:** DAS-601: Basic Machine Learning and Artificial Intelligence for Creative Business Analysis  
**Student:** MD Fahim Shahriar Chowdhury
**Instructor:** [Musabbir Hasan Sammak](https://www.linkedin.com/in/musabbir-sammak/)  
**Institution:** East Delta University  

## Abstract

This project develops a comprehensive machine learning pipeline to predict customer churn in the telecommunications industry using the Kaggle Telco Customer Churn dataset. The study implements and compares 12 different classification algorithms, with CatBoost emerging as the optimal model achieving 83.26% AUC-ROC and 79.46% accuracy. Through systematic exploratory data analysis, feature engineering, and hyperparameter optimization, the final model demonstrates strong discriminative power for identifying customers at risk of churning. The solution provides actionable insights for targeted retention strategies and demonstrates practical application of machine learning in business analytics.

---

## Part I – Problem Statement & Dataset

### Background & Motivation

Customer churn represents one of the most critical challenges in the telecommunications industry, with acquiring new customers costing 5-25 times more than retaining existing ones. In a highly competitive market, telecommunications companies must proactively identify customers at risk of churning to implement targeted retention strategies. Machine learning provides a powerful approach to analyze customer behavior patterns and predict churn likelihood, enabling data-driven decision making for customer retention efforts.

### Objective

The primary objective is to develop a predictive model that accurately identifies customers likely to churn, enabling proactive retention interventions. Key performance indicators include:

- **Primary KPI:** AUC-ROC > 80% for robust model discrimination
- **Secondary KPIs:** Balanced precision and recall for practical deployment
- **Business Impact:** Enable targeted retention campaigns with acceptable false positive rates
- **Expected Outcome:** Provide actionable customer risk scores and feature importance insights

### Scope

This binary classification problem focuses on predicting customer churn (Yes/No) using customer demographic, service, account, and financial data. The analysis encompasses the complete machine learning pipeline from data exploration through model deployment, utilizing 7,043 customer records with 20 predictive features.

### Dataset Details

**Source & License:** [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**License:** CC BY-SA 4.0 (Creative Commons)

| Attribute | Value |
|-----------|-------|
| **Total Records** | 7,043 customers |
| **Features** | 20 predictive features + 1 target |
| **Target Distribution** | 73.5% No Churn, 26.5% Churn |
| **Data Quality** | 11 missing values (0.16%) in TotalCharges |
| **Final Dataset** | 7,032 customers after cleaning |

#### Data Dictionary

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| **customerID** | String | Unique customer identifier | "7590-VHVEG" |
| **gender** | Categorical | Customer gender | Male, Female |
| **SeniorCitizen** | Binary | Senior citizen status | 0, 1 |
| **Partner** | Categorical | Has partner | Yes, No |
| **Dependents** | Categorical | Has dependents | Yes, No |
| **tenure** | Numeric | Months with company | 1-72 |
| **PhoneService** | Categorical | Phone service subscription | Yes, No |
| **MultipleLines** | Categorical | Multiple phone lines | Yes, No, No phone service |
| **InternetService** | Categorical | Internet service type | DSL, Fiber optic, No |
| **OnlineSecurity** | Categorical | Online security service | Yes, No, No internet service |
| **OnlineBackup** | Categorical | Online backup service | Yes, No, No internet service |
| **DeviceProtection** | Categorical | Device protection service | Yes, No, No internet service |
| **TechSupport** | Categorical | Technical support service | Yes, No, No internet service |
| **StreamingTV** | Categorical | TV streaming service | Yes, No, No internet service |
| **StreamingMovies** | Categorical | Movie streaming service | Yes, No, No internet service |
| **Contract** | Categorical | Contract term | Month-to-month, One year, Two year |
| **PaperlessBilling** | Categorical | Paperless billing preference | Yes, No |
| **PaymentMethod** | Categorical | Payment method | Electronic check, Mailed check, Bank transfer, Credit card |
| **MonthlyCharges** | Numeric | Monthly charges in dollars | $18.25 - $118.75 |
| **TotalCharges** | Numeric | Total charges to date | $18.8 - $8684.8 |
| **Churn** | Binary | Target variable | Yes, No |

---

## Part II – Exploratory Data Analysis (EDA)

### Dataset Composition & Metadata

The dataset comprises a balanced mix of feature types enabling comprehensive analysis:

- **Categorical Features:** 16 features (80%)
- **Numeric Features:** 4 features (20%)
- **Data Types:** 17 object, 3 numeric (after preprocessing)
- **Memory Usage:** Optimized through appropriate type conversions
- **No Duplicates:** All 7,043 customerIDs are unique

### Key Findings

**Churn Distribution:**
- Overall churn rate: 26.5% (1,869 churned customers)
- Class imbalance handled through stratified sampling

**Tenure Analysis:**
- Critical risk period: First 5 months (highest churn probability)
- Longer tenure correlates with higher retention rates
- Mean tenure: 32 months for retained vs. 18 months for churned customers

**Financial Patterns:**
- Churned customers pay 23% higher monthly charges ($80.19 vs. $65.02)
- Total charges are lower for churned customers due to shorter tenure
- Month-to-month contracts show highest churn rates (42.7%)

**Service Correlations:**
- Security services significantly reduce churn risk (15.3% vs. 41.8%)
- Fiber optic customers show concerning 40.6% churn rate
- Phone service adoption has minimal impact on churn rates

### Visualization Portfolio

The EDA phase generated 13 professional visualizations saved in `Results/figures/eda/`:

1. **01_gender_distribution.png** - Balanced gender distribution analysis
2. **02_contract_vs_churn.png** - Contract type impact revealing month-to-month risk
3. **03_internet_service_vs_churn.png** - Fiber optic service churn patterns
4. **04_payment_method_distribution.png** - Payment method preferences and churn correlation
5. **05_comprehensive_numeric_analysis.png** - Complete numeric variable distributions
6. **06_monthly_charges_by_churn.png** - Pricing impact on customer retention
7. **07_comprehensive_scatter_regression.png** - Multi-variate relationship analysis
8. **08_bubble_plot_tenure_charges_churn.png** - Three-dimensional churn analysis
9. **09_pairplot_numeric_variables.png** - Numeric variable correlation matrix
10. **10_comprehensive_heatmaps.png** - Categorical-numeric relationship mapping
11. **11_additional_churn_analysis.png** - Comprehensive churn dashboard
12. **12_services_churn_analysis.png** - Service feature impact analysis
13. **13_comprehensive_correlation_analysis.png** - Complete feature correlation study

### Data Quality Issues & Fixes

**Issues Identified:**
1. TotalCharges stored as object type with 11 missing values represented as spaces
2. No duplicate records detected
3. All categorical variables properly encoded

**Solutions Applied:**
1. Converted TotalCharges to numeric, handling errors gracefully
2. Removed 11 rows with missing TotalCharges (0.16% of data)
3. Performed stratified train-test split (80/20) preserving class distribution
4. Saved cleaned dataset for downstream processing

---

## Part III – Feature Engineering

### Engineered Features

The feature engineering process enhanced the original dataset through domain-specific transformations:

**Tenure Segmentation:**
- Created tenure groups (New: 0-12 months, Established: 13-36 months, Loyal: 37+ months)
- Rationale: Customer behavior varies significantly across lifecycle stages

**Contract Encoding:**
- Ordinal encoding for contract duration (Month-to-month=0, One year=1, Two year=2)
- Rationale: Captures progressive commitment levels affecting churn propensity

**Service Bundling Features:**
- Total services count per customer
- Premium services indicator (security, backup, protection)
- Rationale: Service bundling typically reduces churn through increased switching costs

**Financial Metrics:**
- Average monthly charges (TotalCharges / tenure)
- Charge-to-tenure ratio for recent customers
- Rationale: Identifies pricing sensitivity patterns

**Categorical Encoding:**
- One-hot encoding for nominal categories
- Label encoding for ordinal variables
- Binary encoding for Yes/No variables

### Pre-processing Pipeline Summary

```python
# Core preprocessing steps implemented:
1. Missing value imputation (TotalCharges conversion)
2. Feature scaling using StandardScaler for numeric features
3. Categorical encoding (One-hot for nominal, Label for ordinal)
4. Feature selection based on correlation analysis
5. Stratified train-test split preservation
```

**Pipeline Components:**
- **Data Cleaning:** Handled missing values and type conversions
- **Feature Creation:** Domain-specific engineered features
- **Encoding:** Appropriate transformations for ML algorithms
- **Scaling:** Standardization for distance-based algorithms
- **Validation:** Stratified splitting maintaining class balance

---

## Part IV – Model Selection

### Candidate Algorithms

Twelve classification algorithms were systematically evaluated using 5-fold stratified cross-validation:

| Algorithm Category | Model | Rationale |
|-------------------|-------|-----------|
| **Linear Models** | Logistic Regression | Baseline interpretable model |
| **Tree-Based** | Decision Tree | Interpretable non-linear model |
| **Tree-Based** | Random Forest | Ensemble bagging approach |
| **Tree-Based** | Extra Trees | Randomized ensemble variant |
| **Boosting** | XGBoost | Gradient boosting framework |
| **Boosting** | LightGBM | Microsoft's efficient boosting |
| **Boosting** | CatBoost | Categorical feature optimization |
| **Boosting** | Gradient Boosting | Scikit-learn implementation |
| **Boosting** | AdaBoost | Adaptive boosting ensemble |
| **Instance-Based** | K-Nearest Neighbors | Non-parametric approach |
| **Probabilistic** | Naive Bayes | Probabilistic classifier |
| **Kernel-Based** | Support Vector Classifier | Non-linear decision boundaries |

### Hyperparameter Tuning Approach

**Top 3 Models Selected for Optimization:**
1. **Gradient Boosting Classifier** (Baseline AUC-ROC: 0.8390)
2. **CatBoost Classifier** (Baseline AUC-ROC: 0.8356)
3. **AdaBoost Classifier** (Baseline AUC-ROC: 0.8345)

**Optimization Strategy:**
- **Primary Technique:** GridSearchCV for exhaustive parameter search
- **Secondary Technique:** RandomizedSearchCV for efficiency (10 iterations)
- **Cross-Validation:** 5-fold stratified CV for robust evaluation
- **Scoring Metric:** AUC-ROC optimized for class imbalance handling
- **Computational Tracking:** Time monitoring for practical deployment considerations

**Search Spaces Defined:**
```python
# Gradient Boosting Parameters
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# CatBoost Parameters
cb_params = {
    'iterations': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5]
}

# AdaBoost Parameters
ada_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.5, 1.0, 1.5],
    'algorithm': ['SAMME', 'SAMME.R']
}
```

### Final Model Choice & Justification

**Selected Model:** CatBoost Classifier

**Justification Criteria:**
1. **Highest AUC-ROC:** Achieved 84.20% after hyperparameter tuning
2. **Categorical Feature Optimization:** Native handling of categorical variables without extensive preprocessing
3. **Training Efficiency:** Faster convergence compared to other gradient boosting methods
4. **Robustness:** Consistent performance across cross-validation folds
5. **Business Suitability:** Balanced precision-recall trade-off appropriate for churn prediction costs

---

## Part V – Model Evaluation

### Performance Metrics Table

| Metric | Value | Business Interpretation |
|--------|-------|------------------------|
| **Accuracy** | 79.46% | Nearly 4 out of 5 predictions are correct |
| **Precision** | 65.92% | 2 out of 3 predicted churners are actual churners |
| **Recall** | 47.06% | Nearly half of actual churners are correctly identified |
| **F1-Score** | 54.91% | Balanced precision-recall performance |
| **AUC-ROC** | 83.26% | Excellent discriminative power for ranking customers |

### Baseline Model Comparison

Top performing models from 12-algorithm evaluation:

| Rank | Model | AUC-ROC | Accuracy | Precision | Recall |
|------|-------|---------|----------|-----------|--------|
| 1 | **CatBoost** | **83.56%** | **79.09%** | **63.94%** | **49.03%** |
| 2 | Gradient Boosting | 83.90% | 79.16% | 63.92% | 49.50% |
| 3 | AdaBoost | 83.45% | 78.84% | 63.24% | 48.83% |
| 4 | LightGBM | 82.42% | 78.12% | 60.88% | 49.57% |
| 5 | Logistic Regression | 82.17% | 78.88% | 65.36% | 43.75% |

### Confusion Matrix Analysis

The final CatBoost model's confusion matrix reveals:
- **True Negatives:** Strong performance in correctly identifying non-churners
- **True Positives:** Moderate success in identifying actual churners
- **False Positives:** Acceptable rate for marketing campaign costs
- **False Negatives:** Room for improvement in capturing all churners

### Business Interpretation

**Model Strengths:**
- **High AUC-ROC (83.26%):** Excellent ability to rank customers by churn probability
- **Moderate Precision (65.92%):** Acceptable false positive rate for targeted campaigns
- **Balanced Performance:** Reasonable trade-off between precision and recall

**Business Applications:**
- **Customer Scoring:** Rank all customers by churn probability for prioritized outreach
- **Campaign Targeting:** Focus retention efforts on top 20% highest-risk customers
- **Resource Allocation:** Cost-effective deployment with manageable false positive rates
- **Performance Monitoring:** Track model degradation through production metrics

**Limitations:**
- **Recall Opportunity:** Model captures only 47% of actual churners, missing 53%
- **Class Imbalance:** May benefit from cost-sensitive learning approaches
- **Feature Drift:** Requires monitoring for changing customer behavior patterns

---

## Conclusion & Future Work

The developed CatBoost model successfully demonstrates the application of machine learning for customer churn prediction, achieving strong discriminative performance with 83.26% AUC-ROC. The comprehensive pipeline from data exploration through model deployment provides a robust foundation for business deployment. Key insights include the critical importance of contract type, tenure patterns, and service bundling in churn prediction. While the model shows excellent ranking capability, the moderate recall suggests opportunities for cost-sensitive optimization in future iterations.

**Future Work Opportunities:**

• **Enhanced Feature Engineering:** Incorporate temporal patterns, customer interaction history, and competitive market data
• **Cost-Sensitive Learning:** Implement class weights and custom loss functions reflecting business costs of false negatives vs false positives
• **Ensemble Methods:** Develop stacking or blending approaches combining multiple algorithm strengths
• **Real-Time Deployment:** Design streaming prediction pipeline with model monitoring and drift detection
• **Deep Learning Exploration:** Investigate neural networks for complex pattern recognition in larger datasets
• **Causal Inference:** Implement causal modeling to understand intervention effects rather than correlation-based predictions
• **Multi-Objective Optimization:** Balance multiple business objectives including customer lifetime value and retention costs

---

## Repository Usage

### Environment Setup

**Prerequisites:**
- Python 3.8+ (tested with Python 3.13.3)
- 4GB+ RAM (8GB recommended for model training)
- 2GB disk space for data and models

**Installation:**
```bash
# Clone repository
git clone https://github.com/fahimvj/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate.bat

# Activate environment (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduction Instructions

**Complete Pipeline Execution:**
```bash
# Step 1: Data Exploration & Cleaning
jupyter notebook EDA_Analysis/01_dataset_exploration.ipynb

# Step 2: Comprehensive EDA Visualizations
jupyter notebook EDA_Analysis/02_eda_visuals.ipynb

# Step 3: Feature Engineering
jupyter notebook Feature_Engineering/feature_engineering.ipynb

# Step 4: Baseline Model Comparison
jupyter notebook Models/baseline_models.ipynb

# Step 5: Hyperparameter Tuning
jupyter notebook Models/hyperparameter_tuning_final_model_selection.ipynb

# Step 6: Final Model Evaluation
jupyter notebook Models/final_model.ipynb
```

**Automated Execution:**
```bash
# Run complete pipeline with consistent random seed
python -c "
import subprocess
notebooks = [
    'EDA_Analysis/01_dataset_exploration.ipynb',
    'EDA_Analysis/02_eda_visuals.ipynb', 
    'Feature_Engineering/feature_engineering.ipynb',
    'Models/baseline_models.ipynb',
    'Models/hyperparameter_tuning_final_model_selection.ipynb',
    'Models/final_model.ipynb'
]
for nb in notebooks:
    subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', nb])
"
```

**Expected Outputs:**
- **Data:** Cleaned datasets in `Data/interim/` and `Data/output/`
- **Models:** Trained models in `Models/` (*.pkl files)
- **Visualizations:** 13 EDA plots in `Results/figures/eda/`
- **Results:** Performance metrics in `Results/reports/`

---

## References

1. **Dataset Source:** Blastchar. (2018). *Telco Customer Churn*. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn

2. **CatBoost Documentation:** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31.

3. **Scikit-learn:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2857.

4. **Customer Churn Analysis:** Verbeke, W., Martens, D., Mues, C., & Baesens, B. (2011). Building comprehensible customer churn prediction models with advanced rule induction techniques. *Expert Systems with Applications*, 38(3), 2354-2364.

5. **Class Imbalance Handling:** Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority oversampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

---

*This project demonstrates comprehensive application of machine learning methodologies for business analytics in the DAS-601 course at East Delta University.*
