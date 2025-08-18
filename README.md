# 🎓 Credit Card Fraud Detection - ML Learning Project

A hands-on machine learning project focused on binary classification of fraudulent credit card transactions. This project explores ensemble methods, imbalanced data handling, feature engineering, and model evaluation using a realistic dataset of credit card transactions.

## 🎯 Project Objectives

This is a **learning project** designed to explore:
- **Binary Classification** on highly imbalanced data (0.5% fraud rate)
- **Ensemble Methods** comparing RandomForest, LightGBM, and XGBoost
- **Feature Engineering** for fraud detection patterns
- **Threshold Optimization** for business-aligned decision making
- **Model Evaluation** with appropriate metrics for imbalanced datasets
- **API Development** with FastAPI for model serving
- **Interactive Visualization** for exploring model predictions

---

## 📊 Dataset Overview

### **Credit Card Fraud Dataset**
- **Source**: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Time Period**: January 2019 - December 2020 (2 years)
- **Training Data**: 1,296,675 transactions
- **Test Data**: 555,719 transactions
- **Fraud Rate**: ~0.5% (highly imbalanced - realistic for fraud detection)

### **Dataset Characteristics**
| Metric | Training | Test |
|--------|----------|------|
| **Total Transactions** | 1,296,675 | 555,719 |
| **Fraudulent** | 7,506 (0.58%) | 2,145 (0.39%) |
| **Legitimate** | 1,289,169 (99.42%) | 553,574 (99.61%) |
| **Unique Customers** | 1,000 | Same customers |
| **Unique Merchants** | 800 | Same merchants |
| **Categories** | 14 merchant categories | Same categories |

### **Raw Features (23 columns)**
1. **Transaction Details**
   - `trans_date_trans_time`: Transaction timestamp
   - `cc_num`: Credit card number (anonymized)
   - `amt`: Transaction amount in USD
   - `trans_num`: Unique transaction identifier
   - `unix_time`: Unix timestamp

2. **Customer Demographics**
   - `first`, `last`: Customer names
   - `gender`: M/F
   - `dob`: Date of birth
   - `job`: Job title
   - `street`: Street address

3. **Location Data**
   - `city`, `state`, `zip`: Customer location
   - `lat`, `long`: Customer coordinates
   - `city_pop`: City population

4. **Merchant Information**
   - `merchant`: Merchant name
   - `category`: Merchant category (14 types)
   - `merch_lat`, `merch_long`: Merchant coordinates

5. **Target Variable**
   - `is_fraud`: Binary target (0 = legitimate, 1 = fraud)

---

## 🔧 Feature Engineering Pipeline

### **Engineered Features (27 total)**

#### **1. Amount-Based Features (Major Fraud Indicators)**
```python
# Log transformation for better distribution
amt_log = log(amt + 1)

# Risk thresholds based on fraud analysis
amt_high_risk = amt > 400      # Median fraud amount
amt_very_high_risk = amt > 900 # 75th percentile fraud

# Amount relative to category average
amt_zscore_by_category = (amt - category_mean) / category_std
```

#### **2. Temporal Features (Critical for Fraud)**
```python
# Extract time components
transaction_hour = hour from trans_date_trans_time
transaction_dow = day_of_week from trans_date_trans_time

# Risk time periods based on fraud patterns
is_late_night = hour in [22, 23, 0, 1, 2, 3]  # High fraud hours
is_peak_fraud_hours = hour in [23, 0, 1, 2]   # Peak fraud times
is_weekend = day_of_week in [5, 6]            # Weekend flag
is_business_hours = hour in [9-17]            # Business hours
```

#### **3. Geographic Features**
```python
# Distance calculation using Haversine formula
distance_to_merchant = haversine_distance(
    customer_lat, customer_long, 
    merchant_lat, merchant_long
)

# Distance-based risk indicators
is_distant_transaction = distance > 100_km    # Unusual distance
is_very_distant = distance > 500_km          # Very suspicious
```

#### **4. Category Risk Features**
```python
# High-risk categories based on fraud analysis
high_risk_categories = [
    'shopping_net',  # 1.76% fraud rate
    'misc_net',      # 1.45% fraud rate  
    'grocery_pos'    # 1.41% fraud rate
]
is_high_risk_category = category in high_risk_categories
```

#### **5. Demographic Features**
```python
# Age calculation and risk segments
customer_age = (transaction_date - dob).years
is_young_customer = age < 25    # Higher risk demographic
is_senior_customer = age > 65   # Different risk profile
```

#### **6. Encoding Categorical Variables**
```python
# Label encoding for categorical features
category_encoded = LabelEncoder(category)
gender_encoded = LabelEncoder(gender)  
state_encoded = LabelEncoder(state)
job_encoded = LabelEncoder(job)
```

### **Data Preprocessing Workflow**

1. **Data Loading & Validation**
   - Load train/test CSV files
   - Validate required columns exist
   - Check for missing values

2. **Feature Engineering**
   - Create all 27 engineered features
   - Handle datetime conversions
   - Calculate geographic distances

3. **Data Splitting**
   - **Training**: 60% (778,005 transactions)
   - **Validation**: 20% (259,335 transactions) 
   - **Test**: 20% (259,335 transactions)
   - **Stratified splits** to maintain fraud ratio

4. **Imbalanced Data Handling**
   - **SMOTE** (Synthetic Minority Oversampling) on training data only
   - Balances classes while preserving validation/test integrity
   - Original test set used for final evaluation

5. **Feature Scaling**
   - **StandardScaler** fitted on training data
   - Applied to validation and test sets
   - Ensures consistent feature ranges for ML algorithms

---

## 🤖 Model Selection & Architecture

### **Ensemble Approach: Why Three Models?**

#### **1. RandomForest** (Baseline)
**Why chosen:**
- Handles categorical and numerical features naturally
- Built-in feature importance rankings
- Robust to outliers and missing data
- Good baseline for ensemble comparison

**Configuration:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # Handles imbalanced data
    random_state=42
)
```

#### **2. LightGBM** (Speed + Performance)
**Why chosen:**
- Gradient boosting with high performance
- Excellent for imbalanced datasets
- Fast training on large datasets
- Advanced regularization techniques

**Configuration:**
```python
LGBMClassifier(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42
)
```

#### **3. XGBoost** (Best Performance)
**Why chosen:**
- State-of-the-art gradient boosting
- Excellent handling of imbalanced data
- Built-in regularization
- Typically best performer on tabular data

**Configuration:**
```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    scale_pos_weight=class_ratio,  # Auto-calculated for imbalance
    random_state=42
)
```

### **Model Selection Strategy**

**Composite Scoring:**
```python
# Weighted combination prioritizing imbalanced data metrics
composite_score = 0.7 * pr_auc + 0.3 * normalized_mcc

# Best model: Highest composite score
best_model = max(models, key=lambda m: composite_score(m.metrics))
```

**Why this approach:**
- **PR AUC (70%)**: Most important for imbalanced classification
- **Matthews Correlation Coefficient (30%)**: Balanced measure considering all confusion matrix elements

---

## 🎯 Threshold Optimization

### **Business-Aligned Approach vs Mathematical Optimization**

**Previous Approach (Problematic):**
- Pure F1-score maximization
- Resulted in very high thresholds (89-93%)
- Missed many fraud cases with medium probability

**Current Approach (Fraud-Detection Focused):**
```python
def find_optimal_threshold(y_true, y_proba):
    # Test range: 0.05 to 0.50 (fraud-detection focused)
    for threshold in np.arange(0.05, 0.51, 0.01):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        # Fraud detection criteria:
        if precision >= 0.10 and recall >= 0.75:
            score = 0.7 * recall + 0.3 * precision
            # Select threshold maximizing this score
```

**Business Logic:**
- **Minimum 75% recall**: Catch most fraud
- **Minimum 10% precision**: Reasonable false positive rate
- **Prioritize recall**: Better to flag legitimate transactions than miss fraud
- **Cap threshold at 50%**: Ensure fraud-detection focus

**Results:**
- **XGBoost**: 89.17% → **10%** threshold
- **LightGBM**: 93.15% → **12%** threshold  
- **RandomForest**: 90.54% → **15%** threshold

---

## 📈 Model Performance & Metrics Explained

### **Current Best Model: XGBoost**

| Metric | Value | Explanation |
|--------|-------|-------------|
| **PR AUC** | 0.905 | Area under Precision-Recall curve - most important for imbalanced data |
| **ROC AUC** | 0.998 | Area under ROC curve - overall classification ability |
| **Precision** | 91.2% | Of flagged transactions, 91.2% are actually fraud |
| **Recall** | 79.2% | Of all fraud transactions, we catch 79.2% |
| **F1 Score** | 0.848 | Harmonic mean of precision and recall |
| **Threshold** | 0.10 | Optimal decision boundary for fraud classification |

### **Understanding Key Metrics for Imbalanced Data**

#### **1. PR AUC (Precision-Recall Area Under Curve) - Most Critical**
**What it measures:** How well the model maintains precision as recall increases
**Why important for fraud:** 
- ROC AUC can be misleadingly optimistic with imbalanced data
- PR AUC directly measures trade-off between catching fraud (recall) and false alarms (precision)
- **0.905 = Excellent**: Much better than random (0.005 baseline)

#### **2. Precision (91.2%)**
**What it measures:** `True Positives / (True Positives + False Positives)`
**Business meaning:** "Of all transactions we flag as fraud, 91.2% actually are fraud"
**Impact:** Low false positive rate = fewer legitimate customers inconvenienced

#### **3. Recall/Sensitivity (79.2%)**
**What it measures:** `True Positives / (True Positives + False Negatives)`
**Business meaning:** "Of all actual fraud transactions, we catch 79.2%"
**Impact:** High recall = better fraud prevention, less financial loss

#### **4. Specificity (99.96%)**
**What it measures:** `True Negatives / (True Negatives + False Positives)`
**Business meaning:** "Of all legitimate transactions, 99.96% are correctly identified as safe"
**Impact:** Excellent customer experience for legitimate users

#### **5. Matthews Correlation Coefficient (0.849)**
**What it measures:** Correlation between predictions and actual values (-1 to +1)
**Why important:** Balanced measure considering all four confusion matrix values
**Interpretation:** 0.849 = Very strong positive correlation

#### **6. Confusion Matrix Analysis**
```
                 Predicted
Actual    |  Safe  | Fraud |
----------|--------|-------|
Safe      | 368401 |  148  |  99.96% Specificity
Fraud     |   401  | 1529  |  79.22% Recall
----------|--------|-------|
         91.22% Precision
```

**Business Impact:**
- **True Negatives (368,401)**: Legitimate transactions correctly processed
- **False Positives (148)**: Legitimate customers inconvenienced  
- **True Positives (1,529)**: Fraud successfully caught
- **False Negatives (401)**: Fraud that slipped through

### **Why These Results Are Good for Learning**

1. **Realistic Performance**: 79% recall is excellent for fraud detection
2. **Balanced Trade-offs**: High precision (91%) with good recall
3. **Business Alignment**: Low false positive rate (0.04%) 
4. **Proper Evaluation**: Using appropriate metrics for imbalanced data

---

## 🏗️ System Architecture

### **Backend (FastAPI)**
- **`main.py`**: Application entry point
- **`app/api.py`**: REST API endpoints for training and prediction
- **`app/models.py`**: ML model implementations and ensemble management
- **`app/data_preprocessing.py`**: Feature engineering and data pipeline

### **Frontend (Web Interface)**
- **`frontend/index.html`**: Interactive dashboard with Alpine.js
- **Model Training Tab**: Configure and monitor training
- **Test Predictions Tab**: Browse CSV data with live predictions
- **Model Metrics Tab**: Compare model performance
- **System Info Tab**: Monitor system status

### **Key API Endpoints**
- **`POST /train`**: Train new models with configurable parameters
- **`POST /predict`**: Get fraud predictions for transactions
- **`GET /models/info`**: System status and model information
- **`GET /test-data/samples`**: Browse test dataset with filtering

---

## 🚀 Getting Started

### **Prerequisites**
- Python 3.12+
- UV package manager
- 8GB+ RAM (for dataset processing)

### **Installation**
```bash
# Clone repository
git clone <repository-url>
cd fraudulent-chargeback-detector

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start the application
uv run main.py
```

### **Usage**
1. **Open Web Interface**: http://localhost:8000
2. **Train Models**: Use "Model Training" tab
3. **Test Predictions**: Browse real transactions in "Test Predictions" tab
4. **Compare Performance**: Review metrics in "Model Metrics" tab

---

## 📚 Learning Outcomes

This project demonstrates:

### **Machine Learning Concepts**
- **Binary classification** on highly imbalanced datasets
- **Ensemble methods** with automatic model selection
- **Feature engineering** based on domain knowledge
- **Hyperparameter tuning** for different algorithms
- **Threshold optimization** for business objectives

### **Data Science Best Practices**
- **Proper train/validation/test splits** maintaining temporal order
- **Stratified sampling** preserving class distributions
- **Feature scaling** and preprocessing pipelines
- **Cross-validation** for robust model evaluation
- **Metric selection** appropriate for imbalanced data

### **Software Engineering**
- **API development** with FastAPI
- **Interactive visualization** with modern web technologies
- **Model serving** and prediction endpoints
- **Error handling** and validation
- **Code organization** and documentation

### **Domain Knowledge**
- **Fraud detection patterns** in transaction data
- **Risk indicators** and feature importance
- **Business constraints** in threshold selection
- **Performance trade-offs** between precision and recall

---

## 🔍 Key Insights Learned

1. **Imbalanced Data Challenges**: Standard accuracy is meaningless (99.5% by predicting all legitimate)
2. **Metric Selection**: PR AUC more informative than ROC AUC for rare events
3. **Threshold Optimization**: Business context crucial - not just mathematical optimization
4. **Feature Engineering**: Domain knowledge significantly improves model performance
5. **Ensemble Benefits**: Combining multiple algorithms provides robustness
6. **Evaluation Complexity**: Multiple metrics needed to understand model behavior

---

## 📁 Project Structure

```
fraud-detection/
├── app/
│   ├── api.py              # FastAPI application
│   ├── models.py           # ML model implementations  
│   └── data_preprocessing.py # Feature engineering
├── frontend/
│   └── index.html          # Web interface
├── fraud-detection/        # CSV datasets
├── results/runs/           # Training results and models
├── main.py                 # Application entry point
├── pyproject.toml          # Dependencies and configuration
└── README.md               # This file
```

---

*This project serves as a comprehensive exploration of machine learning for fraud detection, covering the complete workflow from data preprocessing to model deployment and evaluation.*