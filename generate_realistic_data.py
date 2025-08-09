#!/usr/bin/env python3
"""
Generate a realistic fraud detection dataset without data leakage.
This creates a more challenging and realistic dataset for proper model evaluation.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_realistic_fraud_data(num_samples=1000, fraud_rate=0.02):
    """Generate realistic fraud detection dataset."""
    
    num_fraud = int(num_samples * fraud_rate)
    num_legit = num_samples - num_fraud
    
    print(f"Generating {num_samples} transactions:")
    print(f"  - Legitimate: {num_legit} ({(1-fraud_rate)*100:.1f}%)")
    print(f"  - Fraudulent: {num_fraud} ({fraud_rate*100:.1f}%)")
    
    transactions = []
    
    # Generate legitimate transactions
    for i in range(num_legit):
        # Normal account ages (mostly older, established accounts)
        account_age = max(30, int(np.random.gamma(5, 100)))  # Skewed towards older accounts
        
        # Previous transactions based on account age
        avg_txn_per_month = np.random.uniform(5, 30)
        prev_txns = max(0, int((account_age / 30) * avg_txn_per_month * np.random.uniform(0.7, 1.3)))
        
        # Amount distribution (realistic spending patterns)
        if np.random.random() < 0.7:  # 70% small transactions
            amount = np.random.gamma(2, 25)  # Small purchases
        elif np.random.random() < 0.9:  # 20% medium transactions  
            amount = np.random.gamma(3, 100)  # Medium purchases
        else:  # 10% large transactions
            amount = np.random.gamma(2, 300)  # Large purchases
        
        amount = round(max(1.0, amount), 2)
        
        # Transaction time (normal business hours mostly)
        if np.random.random() < 0.8:  # 80% during business hours
            hour = np.random.choice(range(8, 22))
        else:  # 20% off-hours
            hour = np.random.choice(list(range(0, 8)) + list(range(22, 24)))
            
        # Random date in the last 30 days
        base_date = datetime(2024, 1, 15)
        random_days = int(np.random.randint(0, 30))
        transaction_time = base_date + timedelta(days=random_days, hours=int(hour), 
                                               minutes=int(np.random.randint(0, 60)))
        
        # Merchant categories (realistic distribution)
        merchant_categories = [
            'Grocery', 'Gas Station', 'Restaurant', 'Pharmacy', 'Coffee Shop',
            'Retail', 'Online', 'Utilities', 'Entertainment', 'Healthcare'
        ]
        merchant_category = np.random.choice(merchant_categories, 
                                           p=[0.25, 0.15, 0.2, 0.1, 0.05, 0.1, 0.05, 0.03, 0.04, 0.03])
        
        # Locations (realistic distribution)
        locations = [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
            'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte'
        ]
        location = np.random.choice(locations)
        
        # Payment methods
        payment_method = np.random.choice(['Credit Card', 'Debit Card'], p=[0.6, 0.4])
        
        transactions.append({
            'TransactionID': f'TXN{i+1:06d}',
            'CustomerID': f'CUST{np.random.randint(1, num_samples//3):05d}',  # Allow repeat customers
            'Amount': amount,
            'MerchantCategory': merchant_category,
            'TransactionTime': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Location': location,
            'PaymentMethod': payment_method,
            'AccountAge': account_age,
            'PreviousTransactions': prev_txns,
            'IsFraud': 0
        })
    
    # Generate fraudulent transactions (with realistic but elevated risk factors)
    for i in range(num_fraud):
        # Fraud accounts tend to be newer, but not exclusively
        if np.random.random() < 0.6:  # 60% newer accounts
            account_age = max(1, int(np.random.exponential(30)))
        else:  # 40% established accounts (account takeover)
            account_age = max(30, int(np.random.gamma(3, 80)))
        
        # Previous transactions
        if account_age < 60:  # New accounts have fewer transactions
            prev_txns = max(0, int(np.random.poisson(3)))
        else:  # Older accounts (account takeover)
            avg_txn_per_month = np.random.uniform(10, 40)
            prev_txns = max(0, int((account_age / 30) * avg_txn_per_month * np.random.uniform(0.5, 1.2)))
        
        # Fraud amounts - slightly higher tendency but with significant overlap
        if np.random.random() < 0.4:  # 40% small amounts (testing cards)
            amount = np.random.gamma(2, 20)
        elif np.random.random() < 0.7:  # 30% medium amounts
            amount = np.random.gamma(3, 150)  # Slightly higher than legit
        else:  # 30% large amounts
            amount = np.random.gamma(2, 400)  # Higher amounts
        
        amount = round(max(1.0, amount), 2)
        
        # Fraud time patterns (slightly more at night, but not exclusively)
        if np.random.random() < 0.4:  # 40% during off-hours (higher than legit)
            hour = np.random.choice(list(range(0, 6)) + list(range(23, 24)))
        else:  # 60% during business hours
            hour = np.random.choice(range(6, 23))
            
        # Random date
        base_date = datetime(2024, 1, 15)
        random_days = int(np.random.randint(0, 30))
        transaction_time = base_date + timedelta(days=random_days, hours=int(hour), 
                                               minutes=int(np.random.randint(0, 60)))
        
        # Merchant categories (slightly different distribution for fraud)
        merchant_categories = [
            'Electronics', 'Jewelry', 'Online', 'Travel', 'Luxury',
            'Grocery', 'Gas Station', 'Restaurant', 'Retail', 'Entertainment'
        ]
        merchant_category = np.random.choice(merchant_categories,
                                           p=[0.15, 0.1, 0.2, 0.1, 0.05, 0.15, 0.1, 0.05, 0.07, 0.03])
        
        # Locations (include some 'Unknown' but not exclusively)
        locations = [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Miami', 'Las Vegas', 'Unknown', 'International', 'Online'
        ]
        location = np.random.choice(locations, 
                                  p=[0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
        
        # Payment methods (fraud slightly favors credit cards)
        payment_method = np.random.choice(['Credit Card', 'Debit Card'], p=[0.75, 0.25])
        
        transactions.append({
            'TransactionID': f'TXN{num_legit + i + 1:06d}',
            'CustomerID': f'CUST{np.random.randint(1, num_samples//3):05d}',
            'Amount': amount,
            'MerchantCategory': merchant_category,
            'TransactionTime': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Location': location,
            'PaymentMethod': payment_method,
            'AccountAge': account_age,
            'PreviousTransactions': prev_txns,
            'IsFraud': 1
        })
    
    # Shuffle transactions
    random.shuffle(transactions)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Verify no obvious leakage
    fraud_df = df[df['IsFraud'] == 1]
    legit_df = df[df['IsFraud'] == 0]
    
    print(f"\nData Quality Check:")
    print(f"Fraud amount range: ${fraud_df['Amount'].min():.2f} - ${fraud_df['Amount'].max():.2f}")
    print(f"Legit amount range: ${legit_df['Amount'].min():.2f} - ${legit_df['Amount'].max():.2f}")
    print(f"Amount overlap: {(fraud_df['Amount'].min() < legit_df['Amount'].max()) and (legit_df['Amount'].min() < fraud_df['Amount'].max())}")
    
    print(f"Fraud account age range: {fraud_df['AccountAge'].min()} - {fraud_df['AccountAge'].max()} days")
    print(f"Legit account age range: {legit_df['AccountAge'].min()} - {legit_df['AccountAge'].max()} days")
    print(f"Age overlap: {(fraud_df['AccountAge'].min() < legit_df['AccountAge'].max()) and (legit_df['AccountAge'].min() < fraud_df['AccountAge'].max())}")
    
    return df

def main():
    """Generate and save realistic fraud dataset."""
    print("🔨 GENERATING REALISTIC FRAUD DATASET")
    print("=" * 50)
    
    # Generate dataset
    df = generate_realistic_fraud_data(num_samples=1000, fraud_rate=0.02)
    
    # Save to CSV
    df.to_csv('Fraud.csv', index=False)
    print(f"\n✅ Saved {len(df)} transactions to Fraud.csv")
    
    # Show sample data
    print(f"\nSample fraud transactions:")
    print(df[df['IsFraud'] == 1].head(3)[['Amount', 'MerchantCategory', 'AccountAge', 'PreviousTransactions']])
    
    print(f"\nSample legitimate transactions:")
    print(df[df['IsFraud'] == 0].head(3)[['Amount', 'MerchantCategory', 'AccountAge', 'PreviousTransactions']])

if __name__ == "__main__":
    main()