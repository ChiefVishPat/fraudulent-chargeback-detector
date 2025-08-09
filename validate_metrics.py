#!/usr/bin/env python3
"""
Validation script to analyze and verify model performance metrics.
This will help us understand if the "perfect" scores are accurate or suspicious.
"""
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from pathlib import Path

def load_latest_results():
    """Load the most recent training results."""
    results_dir = Path("results/validation")
    if not results_dir.exists():
        print("❌ No results directory found. Train models first.")
        return None
    
    # Get most recent run
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        print("❌ No training runs found. Train models first.")
        return None
    
    latest_run = max(run_dirs, key=lambda x: x.name)
    print(f"📊 Analyzing results from: {latest_run.name}")
    
    # Load metrics
    metrics_file = latest_run / "metrics.json"
    if not metrics_file.exists():
        print("❌ No metrics file found.")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return latest_run, metrics

def analyze_dataset():
    """Analyze the training dataset to understand data distribution."""
    print("\n🔍 DATASET ANALYSIS")
    print("=" * 50)
    
    df = pd.read_csv("Fraud.csv")
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent: {df['IsFraud'].sum()} ({df['IsFraud'].mean():.2%})")
    print(f"Legitimate: {(1-df['IsFraud']).sum()} ({(1-df['IsFraud']).mean():.2%})")
    
    # Check data quality
    print(f"\nFeatures available: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Analyze fraud patterns
    print("\n📈 FRAUD PATTERNS:")
    fraud_df = df[df['IsFraud'] == 1]
    legit_df = df[df['IsFraud'] == 0]
    
    print(f"Average fraud amount: ${fraud_df['Amount'].mean():.2f}")
    print(f"Average legit amount: ${legit_df['Amount'].mean():.2f}")
    
    print(f"Fraud account age: {fraud_df['AccountAge'].mean():.1f} days")
    print(f"Legit account age: {legit_df['AccountAge'].mean():.1f} days")
    
    print(f"Fraud previous txns: {fraud_df['PreviousTransactions'].mean():.1f}")
    print(f"Legit previous txns: {legit_df['PreviousTransactions'].mean():.1f}")
    
    return df

def validate_model_metrics(metrics, df):
    """Deep dive into model metrics to verify accuracy."""
    print("\n🔍 METRICS VALIDATION")
    print("=" * 50)
    
    for model_name, model_metrics in metrics.items():
        print(f"\n📊 {model_name}:")
        print(f"   ROC AUC: {model_metrics['roc_auc']:.4f}")
        print(f"   PR AUC:  {model_metrics['pr_auc']:.4f}")
        print(f"   F1:      {model_metrics['f1_score']:.4f}")
        print(f"   Precision: {model_metrics['precision']:.4f}")
        print(f"   Recall:    {model_metrics['recall']:.4f}")
        
        # Analyze confusion matrix
        cm = model_metrics['confusion_matrix']
        tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']
        
        print(f"   Confusion Matrix:")
        print(f"     TN: {tn}  FP: {fp}")
        print(f"     FN: {fn}  TP: {tp}")
        
        # Calculate derived metrics
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total
        
        if tp + fp > 0:
            precision_calc = tp / (tp + fp)
        else:
            precision_calc = 0
            
        if tp + fn > 0:
            recall_calc = tp / (tp + fn)
        else:
            recall_calc = 0
        
        print(f"   Calculated Accuracy: {accuracy:.4f}")
        print(f"   Calculated Precision: {precision_calc:.4f}")
        print(f"   Calculated Recall: {recall_calc:.4f}")
        
        # Check for suspicious patterns
        if model_metrics['roc_auc'] == 1.0 and model_metrics['pr_auc'] == 1.0:
            print(f"   ⚠️  SUSPICIOUS: Perfect scores may indicate overfitting")
            print(f"   ⚠️  Test set size: {total} (very small!)")
            
        if total < 20:
            print(f"   ⚠️  WARNING: Test set too small ({total} samples) for reliable metrics")

def analyze_data_leakage():
    """Check for potential data leakage issues."""
    print("\n🔍 DATA LEAKAGE ANALYSIS")
    print("=" * 50)
    
    df = pd.read_csv("Fraud.csv")
    
    # Check for patterns that might cause leakage
    fraud_amounts = df[df['IsFraud'] == 1]['Amount'].values
    legit_amounts = df[df['IsFraud'] == 0]['Amount'].values
    
    # Check if fraud amounts are distinctly different
    min_fraud = fraud_amounts.min()
    max_legit = legit_amounts.max()
    
    print(f"Minimum fraud amount: ${min_fraud:.2f}")
    print(f"Maximum legit amount: ${max_legit:.2f}")
    
    if min_fraud > max_legit:
        print("⚠️  POTENTIAL LEAKAGE: All fraud amounts > all legit amounts")
        print("   This makes prediction trivially easy!")
    
    # Check temporal patterns
    df['TransactionTime'] = pd.to_datetime(df['TransactionTime'])
    df['hour'] = df['TransactionTime'].dt.hour
    
    fraud_hours = df[df['IsFraud'] == 1]['hour'].values
    legit_hours = df[df['IsFraud'] == 0]['hour'].values
    
    print(f"\nFraud transaction hours: {sorted(fraud_hours)}")
    print(f"Legit transaction hours: {sorted(set(legit_hours))}")
    
    # Check account age patterns
    fraud_ages = df[df['IsFraud'] == 1]['AccountAge'].values
    legit_ages = df[df['IsFraud'] == 0]['AccountAge'].values
    
    print(f"\nFraud account ages: {sorted(fraud_ages)}")
    print(f"Legit account age range: {legit_ages.min()}-{legit_ages.max()}")
    
    if fraud_ages.max() < legit_ages.min():
        print("⚠️  POTENTIAL LEAKAGE: All fraud accounts newer than legit accounts")

def provide_recommendations():
    """Provide recommendations for improving the model."""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    print("1. 📊 DATASET SIZE: Current dataset (40 samples) is too small")
    print("   - Minimum recommended: 1000+ samples")
    print("   - For production: 10,000+ samples")
    
    print("\n2. ⚖️ DATA BALANCE: Need more realistic fraud/legit ratio")
    print("   - Current: ~30% fraud (unrealistic)")
    print("   - Real world: ~0.1-2% fraud")
    
    print("\n3. 🔄 CROSS-VALIDATION: Use k-fold CV instead of single train/test")
    print("   - Current: Single 80/20 split")
    print("   - Better: 5-fold or 10-fold CV")
    
    print("\n4. 📈 MORE FEATURES: Add temporal and behavioral features")
    print("   - Time since last transaction")
    print("   - Velocity features (txn rate)")
    print("   - Geographical features")
    
    print("\n5. 🎯 THRESHOLD TUNING: Optimize for business metrics")
    print("   - Current: F1 optimization")
    print("   - Consider: Cost-weighted optimization")

def main():
    """Main validation function."""
    print("🔍 FRAUD DETECTION MODEL VALIDATION")
    print("=" * 60)
    
    # Load results
    result = load_latest_results()
    if result is None:
        return
    
    latest_run, metrics = result
    
    # Analyze dataset
    df = analyze_dataset()
    
    # Validate metrics
    validate_model_metrics(metrics, df)
    
    # Check for data leakage
    analyze_data_leakage()
    
    # Provide recommendations
    provide_recommendations()
    
    print("\n✅ VALIDATION COMPLETE")
    print("Check the analysis above to understand your model's performance.")

if __name__ == "__main__":
    main()