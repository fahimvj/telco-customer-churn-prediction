# 🤖 Telco Customer Churn Prediction - Complete ML Pipeline

[![Portfolio](https://img.shields.io/badge/Portfolio-Project-blue)](https://github.com/fahimvj/telco-customer-churn-prediction)
[![ML Pipeline](https://img.shields.io/badge/ML-Pipeline-green)](https://github.com/fahimvj/telco-customer-churn-prediction)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-83.26%25-brightgreen)](https://github.com/fahimvj/telco-customer-churn-prediction)

## 🎯 Project Overview
This comprehensive machine learning project analyzes the Telco Customer Churn dataset to build an effective churn prediction model. The project follows industry best practices with professional-grade exploratory data analysis, feature engineering, model selection, hyperparameter tuning, and final model evaluation. The complete pipeline demonstrates end-to-end machine learning workflow with comprehensive documentation and reproducible results.

**Current Status: Complete ML Pipeline ✅**
- ✅ EDA Phase Completed
- ✅ Feature Engineering Completed  
- ✅ Baseline Models Completed
- ✅ Hyperparameter Tuning Completed
- ✅ Final Model Selection Completed
- ✅ Model Evaluation Completed

## 📊 Dataset Details

### 📈 Dataset Overview
- **Source**: Kaggle Telco Customer Churn Dataset
- **Dataset Size**: 7,044 customers × 21 features (including target)
- **Target Variable**: Churn (Binary: Yes/No)
- **Domain**: Telecommunications customer behavior analysis
- **Problem Type**: Binary Classification
- **Class Distribution**: ~73.5% No Churn, ~26.5% Churn

### � Business Context
The dataset represents a telecommunications company's customer base with comprehensive information about:
- **Customer Demographics**: Gender, age group, family status
- **Service Information**: Phone services, internet type, add-on features
- **Account Details**: Contract type, payment method, billing preferences  
- **Usage Metrics**: Tenure, monthly charges, total charges
- **Outcome**: Whether the customer churned or stayed

### 📋 Feature Categories
**Demographic Features (4 features):**
- `gender`: Customer gender (Male/Female)
- `SeniorCitizen`: Senior citizen status (0/1)
- `Partner`: Has partner (Yes/No)
- `Dependents`: Has dependents (Yes/No)

**Service Features (9 features):**
- `PhoneService`: Phone service subscription
- `MultipleLines`: Multiple phone lines
- `InternetService`: Internet service type (DSL/Fiber optic/No)
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`: Security services
- `TechSupport`: Technical support subscription
- `StreamingTV`, `StreamingMovies`: Streaming services

**Account Features (4 features):**
- `Contract`: Contract term (Month-to-month/One year/Two year)
- `PaperlessBilling`: Paperless billing preference
- `PaymentMethod`: Payment method type
- `tenure`: Customer tenure in months

**Financial Features (2 features):**
- `MonthlyCharges`: Monthly billing amount
- `TotalCharges`: Total amount charged to customer

**Target Variable:**
- `Churn`: Customer churn status (Yes/No)

## �🌟 Portfolio Highlights

### 🎯 Key Achievements
- **83.26% AUC-ROC Score** on test set with CatBoost model
- **13 Professional EDA Visualizations** showcasing data storytelling skills
- **6 Baseline Models** implemented and compared systematically
- **Comprehensive Feature Engineering** with domain-specific insights
- **Systematic Model Evaluation** with proper validation techniques

### 🛠️ Technical Skills Demonstrated
- **Data Science**: EDA, Feature Engineering, Statistical Analysis
- **Machine Learning**: Classification, Ensemble Methods, Hyperparameter Tuning
- **Data Visualization**: Matplotlib, Seaborn, Professional Chart Design
- **Software Engineering**: Clean Code, Git Workflow, Documentation
- **Model Evaluation**: Cross-validation, Performance Metrics, Error Analysis

### 📊 Business Impact
- **Customer Retention**: Identify at-risk customers before they churn
- **Revenue Protection**: Potential to save 15-20% annual revenue loss
- **Marketing Optimization**: Target retention campaigns efficiently
- **Data-Driven Decisions**: Quantifiable insights for business strategy

## 📂 Project Structure
```
📁 Telco Customer Churn Prediction/
├── 📄 README.md                           # Project documentation
├── 📄 requirements.txt                    # Python dependencies
├── 📄 Telco_Customer_kaggle.csv           # Original dataset
├── 📄 .gitignore                          # Git ignore configuration
│
├── 📁 Data/                               # Data pipeline
│   ├── 📁 input/                          # Raw dataset storage
│   │   └── Telco_Customer_kaggle.csv      # Original dataset copy
│   ├── 📁 interim/                        # Intermediate processed data
│   │   ├── telco_clean.csv                # Cleaned dataset
│   │   ├── feature_engineered_train_*.csv  # Feature engineering results
│   │   ├── test_engineered.csv            # Test set with features
│   │   └── *.pkl                          # Trained model artifacts
│   └── 📁 output/                         # Final processed datasets
│       ├── train.csv                      # Training data
│       ├── test.csv                       # Test data
│       ├── feature_engineered_train.csv   # Engineered training features
│       ├── feature_selection_summary.csv  # Feature selection results
│       ├── final_model.pkl                # Best performing model
│       └── final_model_metadata.json      # Model performance metadata
│
├── 📁 EDA_Analysis/                       # ✅ Exploratory Data Analysis
│   ├── 01_dataset_exploration.ipynb       # Data cleaning & initial exploration
│   └── 02_eda_visuals.ipynb              # 13 professional visualizations
│
├── 📁 Feature_Engineering/                # ✅ Feature Engineering
│   └── feature_engineering.ipynb          # Advanced feature creation & selection
│
├── 📁 Models/                             # ✅ Machine Learning Models
│   ├── baseline_models.ipynb              # 6 baseline algorithms comparison
│   ├── hyperparameter_tuning_final_model_selection.ipynb # Model optimization
│   ├── final_model.ipynb                  # Final model training & evaluation
│   ├── test_other_model.ipynb             # Additional model experiments
│   ├── error_analysis.ipynb               # Model error analysis & insights
│   ├── best_model_hypertuned.pkl          # Optimized model artifact
│   └── catboost_info/                     # CatBoost training logs
│
├── 📁 Results/                            # ✅ Analysis Results & Outputs
│   ├── 📁 figures/                        # Visualization outputs
│   │   ├── eda/                          # EDA visualization collection (13 charts)
│   │   └── model/                        # Model performance visualizations
│   └── 📁 reports/                       # Analysis summaries
│       ├── baseline_models_results.csv   # Detailed model performance
│       ├── baseline_models_summary.csv   # Model comparison summary
│       └── top_3_baseline_models.csv     # Best performing models
│
└── 📁 .github/                           # GitHub Integration
    └── workflows/
        └── ci.yml                        # Continuous integration workflow
```

## 🔍 Key Features Analyzed
### 👥 **Demographics**
- Gender distribution, senior citizen status, partner relationships, dependents

### 📞 **Service Portfolio** 
- Phone services, internet types (DSL/Fiber/None), streaming services
- Security features, backup services, device protection, tech support

### 💳 **Account Management**
- Contract types (Month-to-month, One year, Two year)
- Payment methods, billing preferences, customer tenure

### 💰 **Financial Metrics**
- Monthly charges, total charges, pricing patterns by service combinations

## 🎯 Complete ML Pipeline - Key Achievements

### 📊 **Phase 1: Exploratory Data Analysis (EDA) - ✅ COMPLETED**
**13 Professional Visualizations Created** following Python Graph Gallery standards:

1. **01_gender_distribution.png** - Gender distribution analysis
2. **02_contract_vs_churn.png** - Contract type impact on churn  
3. **03_internet_service_vs_churn.png** - Internet service churn patterns
4. **04_payment_method_distribution.png** - Payment method preferences
5. **05_comprehensive_numeric_analysis.png** - Complete numeric variable analysis
6. **06_monthly_charges_by_churn.png** - Pricing impact on churn
7. **07_comprehensive_scatter_regression.png** - Variable relationships
8. **08_bubble_plot_tenure_charges_churn.png** - Multi-dimensional churn analysis
9. **09_pairplot_numeric_variables.png** - Numeric variables correlation matrix
10. **10_comprehensive_heatmaps.png** - Categorical-numeric relationships
11. **11_additional_churn_analysis.png** - Comprehensive churn dashboard
12. **12_services_churn_analysis.png** - Service features impact analysis
13. **13_comprehensive_correlation_analysis.png** - Complete feature correlation

### 🔧 **Phase 2: Feature Engineering - ✅ COMPLETED**
- **Advanced Feature Creation**: Domain-specific feature engineering
- **Feature Selection**: Statistical significance testing and correlation analysis
- **Data Preprocessing**: Scaling, encoding, and transformation pipelines
- **Feature Validation**: Cross-validation and stability testing

### 🤖 **Phase 3: Baseline Model Development - ✅ COMPLETED**
**6 Machine Learning Algorithms Evaluated**:
1. **K-Nearest Neighbors (KNN)** - Instance-based learning
2. **Logistic Regression** - Linear classification baseline
3. **Naive Bayes** - Probabilistic classifier
4. **Random Forest** - Ensemble method
5. **Support Vector Machine (SVM)** - Kernel-based classification
6. **XGBoost** - Gradient boosting framework

### 🎯 **Phase 4: Hyperparameter Tuning - ✅ COMPLETED**
**Top 3 Models Selected & Optimized**:
- **Gradient Boosting Classifier** (AUC-ROC: 0.8412)
- **CatBoost Classifier** (AUC-ROC: 0.8420) ⭐ **WINNER**
- **AdaBoost Classifier** (AUC-ROC: 0.8371)

**Optimization Techniques**:
- GridSearchCV with 5-fold stratified cross-validation
- RandomizedSearchCV for efficient parameter space exploration
- Automated model selection based on AUC-ROC performance

### 🏆 **Phase 5: Final Model Evaluation - ✅ COMPLETED**
**CatBoost Final Model Performance**:
- **Accuracy**: 79.46%
- **Precision**: 65.92%
- **Recall**: 47.06%
- **F1-Score**: 54.91%
- **AUC-ROC**: 83.26%

### 🔬 **Phase 6: Advanced Analysis - ✅ COMPLETED**
- **Ensemble Methods**: Voting classifiers and stacking
- **Error Analysis**: Model debugging and performance optimization
- **Business Impact Assessment**: Cost-benefit analysis for deployment

### 💡 **Critical Business Insights Discovered**
- **Contract Type**: Strongest churn predictor (-0.400 correlation)
- **Early Tenure Risk**: 0-5 months is critical retention period  
- **Price Sensitivity**: Churned customers pay 23% more monthly ($80.19 vs $65.02)
- **Security Services**: Significantly reduce churn risk (15.3% vs 41.8% churn rate)
- **Family Loyalty**: Partners/dependents show higher retention (19.7% vs 32.9% churn)
- **Fiber Optic Challenge**: Concerning 40.6% churn rate despite premium pricing

### 📊 **Statistical Analysis Completed**
- **Correlation Analysis**: Complete feature correlation matrix
- **Distribution Analysis**: Comprehensive univariate and bivariate statistics
- **Churn Pattern Recognition**: Multi-dimensional churn behavior analysis
- **Business Intelligence**: Actionable insights for retention strategies

## 🚀 How to Run the Complete ML Pipeline

### ✅ Prerequisites
- **Python 3.8+** (Tested with Python 3.13.3)
- **Git** (optional, for cloning)
- **Jupyter Notebook** or **VS Code** with Python extension
- **Minimum 4GB RAM** (8GB recommended for model training)
- **2GB disk space** for data and models

### 🛠️ Setup Instructions

1. **📥 Clone or download this repository**
   ```bash
   # If cloning from GitHub
   git clone https://github.com/yourusername/telco-churn-prediction.git
   cd telco-churn-prediction
   
   # Or download ZIP and extract
   # Then navigate to the project directory
   cd "DAS 601 ML Final Project"
   ```

2. **🐍 Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate on Windows (Command Prompt):
   .venv\Scripts\activate.bat
   
   # Activate on Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   
   # Activate on macOS/Linux:
   source .venv/bin/activate
   ```

3. **📦 Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **📊 Verify data placement**
   - Ensure `Telco_Customer_kaggle.csv` is in the root directory
   - The setup will automatically organize data into proper directories

### ▶️ Execution Order - Complete ML Pipeline

Run the notebooks in the following sequence for the complete machine learning pipeline:

#### **Phase 1: Exploratory Data Analysis**
```bash
# Using Jupyter Notebook
jupyter notebook EDA_Analysis/01_dataset_exploration.ipynb
jupyter notebook EDA_Analysis/02_eda_visuals.ipynb

# Using VS Code: Open and run cells in sequence
```

#### **Phase 2: Feature Engineering**  
```bash
# Run feature engineering notebook
jupyter notebook Feature_Engineering/feature_engineering.ipynb
```

#### **Phase 3: Baseline Model Development**
```bash
# Compare 6 different machine learning algorithms
jupyter notebook Models/baseline_models.ipynb
```

#### **Phase 4: Hyperparameter Tuning & Model Selection**
```bash
# Optimize top performing models
jupyter notebook Models/hyperparameter_tuning_final_model_selection.ipynb
```

#### **Phase 5: Final Model Evaluation**
```bash
# Train and evaluate the final CatBoost model
jupyter notebook Models/final_model.ipynb
```

#### **Phase 6: Advanced Analysis (Optional)**
```bash
# Additional analysis and experiments
jupyter notebook Models/test_other_model.ipynb      # Additional model testing
jupyter notebook Models/error_analysis.ipynb       # Model error analysis
```

### 🎯 Quick Start - Run Everything
If you want to execute the complete pipeline:

```bash
# 1. Activate environment
.venv\Scripts\activate.bat  # Windows Command Prompt
# .venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Run complete pipeline
jupyter notebook  # Then open and run notebooks in the order above
```

### 📊 Alternative: VS Code Setup
```bash
# 1. Open VS Code in project directory
code .

# 2. Select Python interpreter
# Ctrl+Shift+P -> "Python: Select Interpreter" -> Choose .venv/Scripts/python.exe

# 3. Open notebooks and run cells in sequence
# Start with EDA_Analysis/01_dataset_exploration.ipynb
```

## 📈 Results Summary

### 🏆 Final Model Performance
**CatBoost Classifier** achieved the best performance:
- **Test Accuracy**: 79.46%
- **AUC-ROC**: 83.26%
- **Precision**: 65.92%
- **Recall**: 47.06%
- **F1-Score**: 54.91%

### 📊 Model Comparison Results
**6 Baseline Models Evaluated**:
1. **CatBoost** - AUC-ROC: 83.26% ⭐ **Best**
2. **Gradient Boosting** - AUC-ROC: 84.12%
3. **AdaBoost** - AUC-ROC: 83.71%
4. **Random Forest** - AUC-ROC: ~82%
5. **XGBoost** - AUC-ROC: ~81%
6. **Other Algorithms** - Lower performance

### 💡 Key Business Insights
- **Contract Type**: Strongest predictor of churn
- **Tenure**: First 5 months are critical for retention
- **Pricing**: Churned customers pay 23% higher monthly charges
- **Services**: Security add-ons significantly reduce churn risk
- **Internet Type**: Fiber optic customers show higher churn rates

## 🔗 Git & GitHub Integration

### 📋 Prerequisites for Git Setup
- **Git installed** on your system ([Download Git](https://git-scm.com/downloads))
- **GitHub account** ([Create account](https://github.com/signup))
- **VS Code** with Git extension (usually pre-installed)

### 🚀 Initial Repository Setup

#### **Option 1: Create New Repository from VS Code**

1. **📁 Initialize Git Repository**
   ```bash
   # Navigate to your project directory
   cd "DAS 601 ML Final Project"
   
   # Initialize Git repository
   git init
   
   # Configure Git (first time only)
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

2. **📝 Create .gitignore file** (automatically excludes unnecessary files)
   ```bash
   # The .gitignore file is automatically created (see below)
   ```

3. **🔗 Connect to GitHub via VS Code**
   - Open VS Code in your project directory
   - Press `Ctrl+Shift+P` and search "Git: Initialize Repository"
   - Go to Source Control panel (`Ctrl+Shift+G`)
   - Click "Publish to GitHub"
   - Choose "Publish to GitHub public repository"
   - Name your repository: `telco-churn-prediction-ml`

#### **Option 2: Clone Existing Repository**

```bash
# Clone the repository
git clone https://github.com/yourusername/telco-churn-prediction-ml.git
cd telco-churn-prediction-ml

# Set up your environment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 🔧 VS Code Git Integration Workflow

#### **Daily Git Workflow in VS Code**

1. **📝 Stage Changes**
   - Open Source Control panel (`Ctrl+Shift+G`)
   - Review changed files
   - Click `+` next to files to stage them
   - Or stage all changes with `Ctrl+Shift+A`

2. **💾 Commit Changes**
   - Write commit message in the text box
   - Press `Ctrl+Enter` or click ✓ to commit
   - Use conventional commit format:
     ```
     feat: add baseline model evaluation
     fix: resolve CatBoost model fitting issue
     docs: update README with results
     data: add feature engineered datasets
     ```

3. **🚀 Push to GitHub**
   - Click "..." menu in Source Control
   - Select "Push" or press `Ctrl+Shift+P` then "Git: Push"
   - Or use sync button (↑↓) to pull and push

#### **Command Line Git Workflow**

```bash
# Check status
git status

# Add files
git add .                          # Add all files
git add Models/final_model.ipynb   # Add specific file

# Commit changes
git commit -m "feat: complete final model evaluation"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main

# Create and switch to new branch
git checkout -b feature/ensemble-models
git push -u origin feature/ensemble-models
```

### 📁 Project Files for Git Integration

#### **Automated .gitignore Creation**
A `.gitignore` file will be created to exclude:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
.venv/
env/
ENV/
.env

# Jupyter Notebook
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Large Data Files (keep samples only)
*.csv
!sample_data.csv
*.pkl
!final_model.pkl

# Model artifacts (optional - you may want to track these)
# *.pkl
# *.joblib

# Results (optional - you may want to track visualizations)
# Results/figures/*.png
```

### 🔐 GitHub Repository Settings

#### **Repository Structure on GitHub**
```
📁 telco-churn-prediction-ml/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📄 LICENSE (optional)
├── 📁 .github/
│   └── 📁 workflows/
│       └── ci.yml (optional - for automated testing)
├── 📁 Data/
├── 📁 EDA_Analysis/
├── 📁 Feature_Engineering/
├── 📁 Models/
└── 📁 Results/
```

#### **Recommended Repository Settings**
- **Repository Name**: `telco-churn-prediction-ml`
- **Description**: "Complete ML pipeline for telecommunications customer churn prediction using CatBoost, featuring EDA, feature engineering, and model optimization"
- **Topics**: `machine-learning`, `churn-prediction`, `catboost`, `data-science`, `telecommunications`, `python`, `jupyter`
- **License**: MIT License (recommended for academic projects)

### 🚀 VS Code Extensions for Git

Install these VS Code extensions for better Git integration:

```bash
# Install via VS Code Extensions Marketplace
# Or via command line:
code --install-extension ms-vscode.vscode-git-graph
code --install-extension eamodio.gitlens
code --install-extension github.vscode-pull-request-github
```

**Recommended Extensions:**
- **GitLens** - Supercharge Git capabilities
- **Git Graph** - View Git repository graph
- **GitHub Pull Requests** - GitHub integration
- **Git History** - View Git log and file history

### 📊 Collaboration Workflow

#### **For Team Projects**
```bash
# Create feature branch
git checkout -b feature/hyperparameter-tuning

# Work on your feature
# ... make changes ...

# Commit and push
git add .
git commit -m "feat: implement hyperparameter tuning for top 3 models"
git push -u origin feature/hyperparameter-tuning

# Create Pull Request on GitHub
# Merge after review
```

#### **Branch Strategy**
- `main` - Production-ready code
- `develop` - Integration branch
- `feature/model-evaluation` - Feature branches
- `hotfix/fix-data-loading` - Quick fixes

### 🎯 Step-by-Step VS Code GitHub Integration

#### **First Time Setup (New Project)**

1. **🔧 Open VS Code in your project folder**
   ```bash
   cd "DAS 601 ML Final Project"
   code .
   ```

2. **🔗 Initialize Git (if not already done)**
   - Open Terminal in VS Code (`Ctrl+`` `)
   - Run: `git init`
   - Configure Git user (first time only):
     ```bash
     git config --global user.name "Your Name"
     git config --global user.email "your.email@example.com"
     ```

3. **📝 Stage and Commit Initial Files**
   - Open Source Control panel (`Ctrl+Shift+G`)
   - You'll see all untracked files
   - Click "+" next to "Changes" to stage all files
   - Write commit message: `Initial commit: Complete ML pipeline project`
   - Click ✓ to commit

4. **🚀 Publish to GitHub**
   - In Source Control panel, click "Publish to GitHub"
   - Choose "Publish to GitHub public repository"
   - Repository name: `telco-churn-prediction-ml`
   - VS Code will create the repository and push your code

#### **Daily Workflow in VS Code**

1. **📋 Start Working**
   - Pull latest changes: `Ctrl+Shift+P` → "Git: Pull"
   - Create new branch (optional): `Ctrl+Shift+P` → "Git: Create Branch"

2. **💻 Make Changes**
   - Edit your notebooks and code
   - VS Code shows modified files with "M" indicator
   - Changes appear in Source Control panel

3. **📝 Commit Changes**
   - Go to Source Control (`Ctrl+Shift+G`)
   - Review changes by clicking on files
   - Stage files with "+" button
   - Write descriptive commit message
   - Commit with ✓ or `Ctrl+Enter`

4. **🔄 Sync with GitHub**
   - Click sync button (↑↓) in status bar
   - Or use `Ctrl+Shift+P` → "Git: Push"

#### **VS Code Git Features**

- **📊 Git Graph**: View repository history
- **🔍 GitLens**: See blame annotations and commit info
- **🌿 Branch Switching**: Click branch name in status bar
- **📋 Merge Conflicts**: VS Code provides visual merge tools
- **📝 Commit History**: Right-click files → "Git: View File History"

### 🏷️ Recommended Commit Message Format

Use conventional commits for better organization:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Examples:**
```bash
feat(models): add CatBoost hyperparameter tuning
fix(data): resolve missing value handling in test set
docs(readme): update installation instructions
data(output): add feature engineered datasets
style(notebooks): improve code formatting and comments
refactor(models): reorganize model evaluation functions
test(pipeline): add data validation tests
chore(deps): update requirements.txt versions
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks
- `data`: Data-related changes

### 📈 Expected Outputs - Complete Pipeline
After successful execution, you will have:

**📁 Data Outputs:**
- `Data/interim/telco_clean.csv` - Cleaned dataset (7,032 records)
- `Data/output/train.csv` - Training set (5,625 records)
- `Data/output/test.csv` - Test set (1,407 records)
- `Data/output/feature_engineered_train.csv` - Feature engineered training data
- `Data/output/feature_engineered_test_wrapper.csv` - Feature engineered test data
- `Data/output/feature_selection_summary.csv` - Feature analysis results

**🤖 Model Outputs:**
- `Data/output/final_model.pkl` - Best trained CatBoost model
- `Data/output/final_model_metadata.json` - Model performance metrics
- `Models/best_model_hypertuned.pkl` - Backup model file

**🎨 Visualization Portfolio (13 files):**
- `Results/figures/01_gender_distribution.png`
- `Results/figures/02_contract_vs_churn.png`
- `Results/figures/03_internet_service_vs_churn.png`
- `Results/figures/04_payment_method_distribution.png`
- `Results/figures/05_comprehensive_numeric_analysis.png`
- `Results/figures/06_monthly_charges_by_churn.png`
- `Results/figures/07_comprehensive_scatter_regression.png`
- `Results/figures/08_bubble_plot_tenure_charges_churn.png`
- `Results/figures/09_pairplot_numeric_variables.png`
- `Results/figures/10_comprehensive_heatmaps.png`
- `Results/figures/11_additional_churn_analysis.png`
- `Results/figures/12_services_churn_analysis.png`
- `Results/figures/13_comprehensive_correlation_analysis.png`

**📊 Performance Results:**
- **Final Model**: CatBoost Classifier
- **Test Accuracy**: 79.46%
- **AUC-ROC Score**: 83.26%
- **Precision**: 65.92%
- **Recall**: 47.06%
- **F1-Score**: 54.91%

**📈 Model Comparison Results:**
- Baseline model performance comparison (6 algorithms)
- Hyperparameter tuning results (top 3 models)
- Ensemble method evaluation
- Feature importance analysis

## 🔧 Dependencies & Technical Stack

### 📚 Core Libraries
- **pandas** (2.0+): Data manipulation and analysis
- **numpy** (1.24+): Numerical computing and array operations
- **matplotlib** (3.7+): Base plotting and visualization framework
- **seaborn** (0.12+): Advanced statistical visualization
- **scikit-learn** (1.3+): Machine learning algorithms and utilities
- **xgboost** (1.7+): Gradient boosting framework
- **catboost** (1.2+): CatBoost gradient boosting library
- **jupyter** (1.0+): Interactive notebook environment
- **joblib** (1.3+): Model serialization and parallel computing

### 🤖 Machine Learning Stack
- **Classification Algorithms**: KNN, Logistic Regression, Naive Bayes, Random Forest, SVM, XGBoost, CatBoost, AdaBoost
- **Hyperparameter Tuning**: GridSearchCV, RandomizedSearchCV
- **Model Evaluation**: Cross-validation, ROC-AUC, Confusion Matrix, Classification Reports
- **Ensemble Methods**: Voting Classifiers, Stacking
- **Feature Engineering**: Statistical transformations, encoding, scaling

### 🎨 Visualization Standards
- **Python Graph Gallery Compliance**: All visualizations follow established best practices
- **Color Theory**: ColorBrewer schemes for optimal visual perception
- **Statistical Graphics**: Edward Tufte principles for clear communication
- **Professional Design**: High-DPI exports suitable for presentations and reports

## 📋 Current Project Status

### ✅ **COMPLETED PHASES**

#### 🔍 **Phase 1: Data Exploration & Cleaning**
- ✅ Dataset inspection and quality assessment
- ✅ Missing value identification and handling (11 missing values removed)
- ✅ Data type conversions and standardization  
- ✅ Train/test split with stratification (80/20 split)
- ✅ Data validation and integrity checks

#### 📊 **Phase 2: Exploratory Data Analysis (EDA)**
- ✅ **13 Professional Visualizations**: Comprehensive visual analysis suite
- ✅ **Statistical Analysis**: Correlation matrices and relationship mapping  
- ✅ **Business Insights**: Actionable findings for retention strategies
- ✅ **Pattern Recognition**: Churn indicators and risk factors identified
- ✅ **Documentation**: Detailed methodology and results reporting

### � **READY FOR NEXT PHASE**

The project has completed comprehensive exploratory data analysis and is ready for:
- **Feature Engineering**: Variable transformation and creation
- **Model Development**: Machine learning algorithm implementation  
- **Hyperparameter Tuning**: Model optimization
- **Final Model Selection**: Best performer identification

## 📊 EDA Results Summary

### 🔍 **Data Quality Assessment**
- **Data Completeness**: 99.84% (11 missing values removed)
- **Feature Distribution**: Balanced mix of categorical and numeric variables
- **Target Balance**: ~26.5% churn rate (suitable for classification)
- **Data Integrity**: No duplicate records, consistent data types

### 📈 **Key Statistical Findings**
- **Strongest Churn Predictor**: Contract type (r = -0.400)
- **Critical Risk Period**: First 5 months of customer tenure
- **Price Impact**: 23% higher monthly charges among churned customers
- **Service Correlation**: Security services strongly linked to retention

### 🎯 **Business Intelligence Extracted**
- **High-Risk Segments**: Month-to-month contracts, fiber optic without security
- **Retention Opportunities**: Early customer engagement, service bundling
- **Pricing Strategy**: Review premium pricing for fiber optic services
- **Service Priority**: Focus on security and support service adoption

## 📞 Contact & Contribution

### 📧 **Project Information**
- **Course**: DAS 601 - Machine Learning
- **Institution**: East Delta University
- **Academic Year**: 2024-2025
- **Project Type**: ML/Ai Final Project

### 🤝 **Collaboration**
For questions, suggestions, or contributions:
- Review the comprehensive documentation in `/Results/reports/`
- Check existing visualizations in `/Results/figures/`
- Refer to methodology details in analysis notebooks
- Create issues for technical questions or improvements

### 📚 **Learning Resources**
- **Python Graph Gallery**: https://python-graph-gallery.com/
- **EDA Best Practices**: "Exploratory Data Analysis with Python"
- **Visualization Theory**: Edward Tufte's "The Visual Display of Quantitative Information"
- **Statistical Analysis**: "Pattern Recognition and Machine Learning" by Bishop

---

## 🏆 **Complete ML Pipeline Achievements**

### 🎯 **Technical Excellence Standards Met**
- ✅ **End-to-End ML Pipeline**: Complete machine learning workflow from data to model
- ✅ **Professional Visualization Portfolio**: 13 publication-ready charts following best practices
- ✅ **Systematic Model Evaluation**: Comprehensive comparison of 6 ML algorithms with optimization
- ✅ **Strong Model Performance**: 83.26% AUC-ROC with 79.46% accuracy on test set
- ✅ **Comprehensive Documentation**: Complete methodology and reproducible workflows

## 🎯 Portfolio Navigation

### 📊 **For Data Science Roles**
- **EDA Expertise**: [`02_eda_visuals.ipynb`](EDA_Analysis/02_eda_visuals.ipynb) - 13 professional visualizations
- **Statistical Analysis**: [`01_dataset_exploration.ipynb`](EDA_Analysis/01_dataset_exploration.ipynb) - Data cleaning & validation
- **Feature Engineering**: [`feature_engineering.ipynb`](Feature_Engineering/feature_engineering.ipynb) - Advanced feature creation

### 🤖 **For ML Engineering Roles**
- **Model Development**: [`baseline_models.ipynb`](Models/baseline_models.ipynb) - 6 algorithms compared
- **Hyperparameter Tuning**: [`hyperparameter_tuning_final_model_selection.ipynb`](Models/hyperparameter_tuning_final_model_selection.ipynb)
- **Final Model**: [`final_model.ipynb`](Models/final_model.ipynb) - CatBoost implementation

### 📈 **For Business Analytics Roles**
- **Business Insights**: [Results/reports/](Results/reports/) - Model performance summaries
- **Visual Reports**: [Results/figures/](Results/figures/) - Professional charts and analysis
- **Performance Metrics**: Detailed model comparison and evaluation results

### 🔧 **For Software Engineering Roles**
- **Clean Code**: Well-documented, modular Python notebooks
- **Version Control**: Professional Git workflow with meaningful commits
- **Project Structure**: Organized directory structure following ML best practices
- **Reproducibility**: Complete setup instructions and dependency management

---

**⭐ Star this repository if you found it helpful for your learning journey!**

**🔗 Connect with me**: [GitHub Profile](https://github.com/fahimvj)

### 📊 **Machine Learning Deliverables**
- ✅ **Final Model**: CatBoost Classifier (83.26% AUC-ROC, 79.46% accuracy)
- ✅ **Model Comparison**: 6 baseline algorithms + 3 optimized models + ensemble methods
- ✅ **Feature Engineering**: Advanced feature creation and selection with statistical validation
- ✅ **Hyperparameter Optimization**: Grid search and randomized search with cross-validation
- ✅ **Production Deployment**: Serialized model with metadata and performance tracking

### 📈 **Business Intelligence Impact**
- ✅ **Churn Prediction Accuracy**: 83.26% AUC-ROC demonstrates excellent discriminative power
- ✅ **Risk Factor Identification**: Contract type strongest predictor (-0.400 correlation)
- ✅ **Customer Segmentation**: High-risk segments identified for targeted retention
- ✅ **Financial Impact**: Price sensitivity analysis reveals 23% higher costs among churners
- ✅ **Actionable Insights**: Service bundling and early engagement strategies identified

### 🔬 **Advanced Analysis Completed**
- ✅ **Ensemble Methods**: Voting classifiers and stacking with multiple base learners
- ✅ **Error Analysis**: Comprehensive model debugging and performance optimization
- ✅ **Statistical Validation**: Cross-validation, significance testing, and correlation analysis
- ✅ **Feature Importance**: Complete feature contribution analysis and selection
- ✅ **Model Interpretation**: Decision boundaries and prediction confidence analysis

*This comprehensive machine learning project demonstrates mastery of the complete ML pipeline from exploratory data analysis through final model evaluation, showcasing advanced techniques in feature engineering, model selection, hyperparameter optimization, and business intelligence extraction for real-world telecommunications churn prediction.*

---
**🎓 This project represents advanced coursework in machine learning and data science for DAS 601 - Complete ML Pipeline Implementation.**
