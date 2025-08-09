# 🛡️ Fraud Detection System

A comprehensive ML-powered credit card fraud detection system using ensemble methods (RandomForest, LightGBM, XGBoost) with FastAPI backend and interactive web frontend.

## 📋 Project Overview

This system is designed for **credit card fraud detection** and **dispute triage** use cases, helping financial institutions identify potentially fraudulent transactions in real-time. The system focuses on:

- **First-party fraud detection**: Identifying transactions that may be disputed or chargebacks
- **Real-time scoring**: Fast prediction API for transaction scoring
- **Model interpretability**: Feature importance and threshold tuning for business decisions
- **Imbalanced data handling**: Specialized techniques for fraud detection where fraudulent transactions are rare

## 🗂️ Dataset

### Expected Schema

The system expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `TransactionID` | string | Unique transaction identifier |
| `CustomerID` | string | Customer identifier |
| `Amount` | float | Transaction amount (USD) |
| `MerchantCategory` | string | Category of merchant (e.g., Electronics, Grocery) |
| `TransactionTime` | datetime | ISO format timestamp |
| `Location` | string | Transaction location |
| `PaymentMethod` | string | Payment method (Credit Card, Debit Card) |
| `AccountAge` | int | Customer account age in days |
| `PreviousTransactions` | int | Number of previous transactions by customer |
| `IsFraud` | int | Target variable (0 = legitimate, 1 = fraudulent) |

### Data Handling

- **Missing values**: Filled with median for numerical features
- **Categorical encoding**: Label encoding for categorical features
- **Feature scaling**: StandardScaler for numerical features
- **Unknown categories**: Handled gracefully during prediction

## 🔧 Feature Engineering

The system implements comprehensive feature engineering based on fraud detection best practices:

### Temporal Features
- **Hour of day**: `transaction_hour` - Fraud patterns vary by time
- **Day of week**: `transaction_day_of_week` 
- **Weekend indicator**: `is_weekend` - Weekend transactions often riskier
- **Night transactions**: `is_night_transaction` - Late night activity

### Amount-Based Features
- **Log transformation**: `amount_log` - Handles amount skewness
- **Amount bins**: `amount_bins` - Categorical amount ranges
- **High amount flag**: `is_high_amount` - 95th percentile threshold

### Velocity & Behavioral Features
- **Transaction rate**: `transactions_per_day` - Customer activity rate
- **Amount per transaction**: `amount_per_previous_transaction` - Spending pattern
- **Account age ratio**: `amount_to_account_age_ratio` - Risk based on account maturity

### Risk Indicators
- **New account flag**: `is_new_account` - Accounts ≤ 30 days
- **Low transaction history**: `has_few_transactions` - ≤ 5 previous transactions
- **Unknown location**: `is_unknown_location` - Location risk factor

### Categorical Encoding
- **Location encoding**: Numerical encoding of transaction locations
- **Merchant category encoding**: Business type risk profiling
- **Payment method encoding**: Card type risk assessment

## 🤖 Algorithms & Model Selection

### Model Architecture

The system implements an **ensemble approach** with three complementary algorithms:

#### 1. RandomForest (Baseline)
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced'  # Handles class imbalance
)
```
- **Why**: Robust baseline, handles mixed data types well
- **Strengths**: Feature importance, resistant to overfitting
- **Use case**: Stable baseline performance

#### 2. LightGBM (Primary)
```python
LGBMClassifier(
    objective='binary',
    is_unbalance=True,        # Built-in imbalance handling
    scale_pos_weight=ratio,   # Additional weight adjustment
    early_stopping_rounds=50
)
```
- **Why**: Optimized for tabular data and imbalanced datasets
- **Strengths**: Fast training, excellent performance on structured data
- **Use case**: Production model for high-volume scoring

#### 3. XGBoost (Advanced)
```python
XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=ratio,   # Class imbalance handling
    early_stopping_rounds=50
)
```
- **Why**: State-of-the-art gradient boosting performance
- **Strengths**: High accuracy, feature importance
- **Use case**: Maximum performance scenarios

### Model Selection Strategy

1. **Primary Metric**: **Area Under Precision-Recall Curve (AUPRC)**
   - More relevant for imbalanced fraud detection than ROC AUC
   - Focus on precision-recall trade-offs important for business decisions

2. **Secondary Metrics**:
   - **ROC AUC**: Overall discriminative ability
   - **F1 Score**: Balanced precision-recall
   - **Precision/Recall**: Business-specific thresholds

3. **Threshold Optimization**:
   - Custom threshold tuning using validation set
   - Maximizes F1 score by default
   - Configurable for business-specific cost functions

### Handling Class Imbalance

- **Class weights**: `class_weight='balanced'` and `scale_pos_weight`
- **Sampling strategy**: Stratified train/test splits
- **Evaluation focus**: Precision-Recall metrics over accuracy
- **Threshold tuning**: Optimal operating points for business needs

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- UV package manager

### Installation

1. **Clone and setup**:
```bash
git clone <repository-url>
cd fraudulent-chargeback-detector
```

2. **Install dependencies with UV**:
```bash
uv pip install -r requirements.txt
```

3. **Start the application**:
```bash
# Option 1: Direct FastAPI
uv run python -m app.api

# Option 2: Using main entry point
uv run python run.py

# Option 3: Development mode with reload
RELOAD=true uv run python run.py
```

4. **Access the application**:
   - **Web Interface**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

### Environment Variables

```bash
HOST=0.0.0.0          # Server host (default: 0.0.0.0)
PORT=8000             # Server port (default: 8000)  
RELOAD=false          # Enable auto-reload for development
```

## 📖 Usage Guide

### 1. Training Models

**Via Web Interface:**
1. Navigate to the "Model Training" tab
2. Configure test size (default: 0.2) and random state
3. Click "Start Training"
4. Monitor progress in real-time

**Via API:**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"test_size": 0.2, "random_state": 42}'
```

### 2. Making Predictions

**Single Transaction Prediction:**
```python
import requests

transaction = {
    "TransactionID": "TXN123",
    "CustomerID": "CUST456", 
    "Amount": 1500.00,
    "MerchantCategory": "Electronics",
    "TransactionTime": "2024-01-15T23:30:00",
    "Location": "Las Vegas",
    "PaymentMethod": "Credit Card",
    "AccountAge": 15,
    "PreviousTransactions": 1
}

response = requests.post(
    "http://localhost:8000/predict",
    json={"transactions": [transaction]}
)

result = response.json()
print(f"Fraud Probability: {result['predictions'][0]['fraud_probability']:.2%}")
print(f"Risk Level: {result['predictions'][0]['risk_level']}")
```

### 3. Model Performance

**Get Metrics:**
```bash
curl "http://localhost:8000/models/metrics"
```

**Feature Importance:**
```bash
curl "http://localhost:8000/models/feature-importance"
```

## 📊 Model Artifacts

All training artifacts are automatically saved to timestamped directories:

```
results/
├── models/
│   ├── randomforest_20240115_143022.joblib
│   ├── lightgbm_20240115_143022.joblib
│   └── xgboost_20240115_143022.joblib
├── validation/20240115_143022/
│   ├── metrics.json
│   ├── classification_reports.json
│   ├── feature_importances.json
│   ├── randomforest_curves.png
│   ├── lightgbm_curves.png
│   └── xgboost_curves.png
└── logs/
    └── fraud_detection_20240115.log
```

## 🔌 API Reference

### Core Endpoints

- `POST /train` - Train models
- `POST /predict` - Predict fraud for transactions
- `GET /models/metrics` - Get model performance metrics
- `GET /models/info` - Get system information
- `GET /training-status` - Check training progress

### Model Management

- `GET /models/feature-importance` - Get feature importance
- `GET /models/plots/{run_id}/{model}` - Download performance plots
- `GET /download/model/{model}` - Download model artifacts
- `GET /runs` - List all training runs

## 🧪 Development

### Running Tests

```bash
uv run pytest tests/
```

### Code Formatting

```bash
uv run black app/
uv run isort app/
```

### Development Mode

```bash
RELOAD=true uv run python run.py
```

## 📈 Performance Characteristics

### Typical Results

On balanced fraud datasets, expect:

| Model | PR AUC | ROC AUC | F1 Score |
|-------|--------|---------|----------|
| **LightGBM** | 0.85-0.92 | 0.90-0.95 | 0.80-0.88 |
| **XGBoost** | 0.83-0.90 | 0.89-0.94 | 0.78-0.86 |
| **RandomForest** | 0.78-0.85 | 0.85-0.91 | 0.72-0.82 |

### Processing Speed

- **Training**: ~30-60 seconds for 10k transactions
- **Prediction**: <50ms per transaction
- **Batch processing**: ~1000 transactions/second

## 🛠️ Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the project root
cd fraudulent-chargeback-detector
uv run python run.py
```

**Port Already in Use:**
```bash
PORT=8080 uv run python run.py
```

**Missing Dependencies:**
```bash
uv pip install -r requirements.txt
```

### Logging

Check logs for detailed information:
```bash
tail -f results/logs/fraud_detection_$(date +%Y%m%d).log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**🚀 Ready to detect fraud? Start with `uv run python run.py` and visit http://localhost:8000!**