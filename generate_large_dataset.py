#!/usr/bin/env python3
"""
Generate a large, realistic fraud detection dataset for proper evaluation.
Creates 100K+ samples with proper class imbalance handling.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_large_fraud_dataset(num_samples=100000, fraud_rate=0.005):
    """Generate large realistic fraud detection dataset."""
    
    num_fraud = int(num_samples * fraud_rate)
    num_legit = num_samples - num_fraud
    
    print(f"🔨 GENERATING LARGE FRAUD DATASET")
    print(f"📊 Total samples: {num_samples:,}")
    print(f"  - Legitimate: {num_legit:,} ({(1-fraud_rate)*100:.2f}%)")
    print(f"  - Fraudulent: {num_fraud:,} ({fraud_rate*100:.3f}%)")
    print()
    
    transactions = []
    
    # Generate legitimate transactions
    print("🔄 Generating legitimate transactions...")
    for i in tqdm(range(num_legit), desc="Legit transactions"):
        # Realistic account ages - mostly established accounts
        account_age = max(30, int(np.random.gamma(8, 50)))  # Skewed towards older accounts
        
        # Previous transactions based on account age and activity level
        activity_level = np.random.gamma(2, 1)  # Some customers more active than others
        avg_txn_per_month = np.clip(activity_level * 15, 1, 100)
        months = account_age / 30.44  # Average days per month
        prev_txns = max(0, int(months * avg_txn_per_month * np.random.uniform(0.5, 1.5)))
        
        # Realistic amount distribution
        customer_segment = np.random.choice(['low_spender', 'medium_spender', 'high_spender'], 
                                          p=[0.6, 0.3, 0.1])
        
        if customer_segment == 'low_spender':
            amount = np.random.gamma(2, 15)  # $30 average
        elif customer_segment == 'medium_spender':
            amount = np.random.gamma(3, 50)  # $150 average
        else:  # high_spender
            amount = np.random.gamma(2, 200)  # $400 average
            
        amount = round(max(1.0, amount), 2)
        
        # Transaction time - realistic daily patterns
        if np.random.random() < 0.85:  # 85% during reasonable hours
            hour_probs = [0.02, 0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.18,  # 6-13 (morning/lunch)
                         0.15, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01]  # 14-22 (afternoon/evening)
            # Normalize probabilities
            hour_probs = np.array(hour_probs) / np.sum(hour_probs)
            hour = np.random.choice(range(6, 23), p=hour_probs)
        else:  # 15% off-hours
            hour = np.random.choice(list(range(0, 6)) + list(range(23, 24)))
            
        # Random date in the last 90 days
        base_date = datetime(2024, 1, 15)
        random_days = int(np.random.randint(0, 90))
        transaction_time = base_date + timedelta(
            days=random_days, 
            hours=int(hour), 
            minutes=int(np.random.randint(0, 60))
        )
        
        # Merchant categories with realistic distribution
        merchant_categories = [
            'Grocery', 'Gas Station', 'Restaurant', 'Pharmacy', 'Coffee Shop',
            'Retail', 'Online', 'Utilities', 'Entertainment', 'Healthcare',
            'ATM', 'Department Store', 'Fast Food', 'Subscription', 'Transportation'
        ]
        merchant_weights = [0.20, 0.12, 0.15, 0.08, 0.05, 0.10, 0.08, 0.04, 0.06, 0.04, 
                          0.03, 0.02, 0.02, 0.02, 0.03]
        # Normalize weights to sum to 1
        merchant_weights = np.array(merchant_weights)
        merchant_weights = merchant_weights / merchant_weights.sum()
        merchant_category = np.random.choice(merchant_categories, p=merchant_weights)
        
        # Locations with realistic US distribution
        locations = [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
            'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
            'Seattle', 'Denver', 'Boston', 'Nashville', 'Baltimore',
            'Portland', 'Las Vegas', 'Detroit', 'Memphis', 'Louisville'
        ]
        location_weights = [0.08, 0.07, 0.05, 0.04, 0.03] + [0.03] * 10 + [0.02] * 10
        # Normalize weights to sum to 1
        location_weights = np.array(location_weights)
        location_weights = location_weights / location_weights.sum()
        location = np.random.choice(locations, p=location_weights)
        
        # Payment methods
        payment_method = np.random.choice(['Credit Card', 'Debit Card'], p=[0.65, 0.35])
        
        # Customer ID - allow repeat customers (some more active than others)
        if np.random.random() < 0.3:  # 30% are repeat customers
            customer_id = f'CUST{np.random.randint(1, num_samples//20):06d}'  # Frequent customers
        else:
            customer_id = f'CUST{np.random.randint(1, num_samples//2):06d}'   # Occasional customers
        
        transactions.append({
            'TransactionID': f'TXN{i+1:07d}',
            'CustomerID': customer_id,
            'Amount': amount,
            'MerchantCategory': merchant_category,
            'TransactionTime': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Location': location,
            'PaymentMethod': payment_method,
            'AccountAge': account_age,
            'PreviousTransactions': prev_txns,
            'IsFraud': 0
        })
    
    # Generate fraudulent transactions with realistic risk factors
    print("🚨 Generating fraudulent transactions...")
    for i in tqdm(range(num_fraud), desc="Fraud transactions"):
        fraud_type = np.random.choice(['new_account', 'account_takeover', 'card_testing'], 
                                    p=[0.4, 0.35, 0.25])
        
        if fraud_type == 'new_account':
            # New account fraud - very new accounts, low transaction history
            account_age = max(1, int(np.random.exponential(10)))  # Very new accounts
            prev_txns = max(0, int(np.random.poisson(2)))  # Very few transactions
            
            # Amounts - often testing small amounts first, then larger
            if np.random.random() < 0.3:  # 30% small test amounts
                amount = np.random.uniform(1, 50)
            else:  # 70% larger amounts
                amount = np.random.gamma(2, 200)  # Higher amounts
                
        elif fraud_type == 'account_takeover':
            # Account takeover - established account, sudden change in pattern
            account_age = max(180, int(np.random.gamma(5, 100)))  # Older accounts
            # Normal previous transaction count for age
            months = account_age / 30.44
            prev_txns = max(10, int(months * np.random.uniform(10, 50)))
            
            # Amounts similar to high spenders but at unusual times/locations
            amount = np.random.gamma(3, 150)
            
        else:  # card_testing
            # Card testing - various account ages, small amounts
            account_age = max(30, int(np.random.gamma(3, 60)))
            months = account_age / 30.44
            prev_txns = max(0, int(months * np.random.uniform(5, 30)))
            
            # Small test amounts
            amount = np.random.gamma(1.5, 10)  # Small amounts for testing
        
        amount = round(max(1.0, amount), 2)
        
        # Time patterns - fraud slightly more likely at night but not exclusively
        if np.random.random() < 0.35:  # 35% during off-hours (vs 15% for legit)
            hour = np.random.choice(list(range(0, 6)) + list(range(23, 24)))
        else:  # 65% during business hours
            hour = np.random.choice(range(6, 23))
            
        # Random date
        base_date = datetime(2024, 1, 15)
        random_days = int(np.random.randint(0, 90))
        transaction_time = base_date + timedelta(
            days=random_days, 
            hours=int(hour), 
            minutes=int(np.random.randint(0, 60))
        )
        
        # Merchant categories for fraud - different distribution
        if fraud_type == 'card_testing':
            fraud_merchants = ['Gas Station', 'Fast Food', 'ATM', 'Online', 'Retail']
            fraud_weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        else:
            fraud_merchants = ['Electronics', 'Jewelry', 'Online', 'Luxury', 'Travel',
                             'Department Store', 'Retail', 'Entertainment']
            fraud_weights = [0.20, 0.15, 0.15, 0.10, 0.15, 0.10, 0.10, 0.05]
        
        # Normalize weights
        fraud_weights = np.array(fraud_weights)
        fraud_weights = fraud_weights / fraud_weights.sum()
        merchant_category = np.random.choice(fraud_merchants, p=fraud_weights)
        
        # Locations - include riskier locations but not exclusively
        risky_locations = ['Las Vegas', 'Miami', 'Los Angeles', 'New York', 'International',
                          'Unknown', 'Detroit', 'Memphis', 'Portland', 'Online']
        location_weights = [0.15, 0.12, 0.12, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.07]
        # Normalize weights
        location_weights = np.array(location_weights)
        location_weights = location_weights / location_weights.sum()
        location = np.random.choice(risky_locations, p=location_weights)
        
        # Payment methods - fraud slightly favors credit cards
        payment_method = np.random.choice(['Credit Card', 'Debit Card'], p=[0.80, 0.20])
        
        # Customer ID
        customer_id = f'CUST{np.random.randint(1, num_samples//2):06d}'
        
        transactions.append({
            'TransactionID': f'TXN{num_legit + i + 1:07d}',
            'CustomerID': customer_id,
            'Amount': amount,
            'MerchantCategory': merchant_category,
            'TransactionTime': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Location': location,
            'PaymentMethod': payment_method,
            'AccountAge': account_age,
            'PreviousTransactions': prev_txns,
            'IsFraud': 1
        })
    
    # Shuffle all transactions to avoid temporal clustering
    print("🔀 Shuffling transactions...")
    random.shuffle(transactions)
    
    # Create DataFrame
    print("📊 Creating DataFrame...")
    df = pd.DataFrame(transactions)
    
    # Verify data quality
    fraud_df = df[df['IsFraud'] == 1]
    legit_df = df[df['IsFraud'] == 0]
    
    print(f"\n✅ DATA QUALITY VERIFICATION:")
    print(f"Total samples: {len(df):,}")
    print(f"Fraud samples: {len(fraud_df):,} ({len(fraud_df)/len(df)*100:.3f}%)")
    print(f"Legitimate samples: {len(legit_df):,} ({len(legit_df)/len(df)*100:.3f}%)")
    print()
    print(f"Fraud amount range: ${fraud_df['Amount'].min():.2f} - ${fraud_df['Amount'].max():.2f}")
    print(f"Legit amount range: ${legit_df['Amount'].min():.2f} - ${legit_df['Amount'].max():.2f}")
    print(f"Amount overlap: {(fraud_df['Amount'].min() < legit_df['Amount'].max()) and (legit_df['Amount'].min() < fraud_df['Amount'].max())}")
    print()
    print(f"Fraud account age range: {fraud_df['AccountAge'].min()} - {fraud_df['AccountAge'].max()} days")
    print(f"Legit account age range: {legit_df['AccountAge'].min()} - {legit_df['AccountAge'].max()} days") 
    print(f"Age overlap: {(fraud_df['AccountAge'].min() < legit_df['AccountAge'].max()) and (legit_df['AccountAge'].min() < fraud_df['AccountAge'].max())}")
    
    return df

def main():
    """Generate and save large realistic fraud dataset."""
    # Generate 100K samples with 0.5% fraud rate (realistic for financial institutions)
    df = generate_large_fraud_dataset(num_samples=100000, fraud_rate=0.005)
    
    # Save to CSV
    print(f"\n💾 Saving dataset...")
    df.to_csv('Fraud.csv', index=False)
    print(f"✅ Saved {len(df):,} transactions to Fraud.csv")
    print(f"📁 File size: ~{len(df) * 150 / 1024 / 1024:.1f} MB")
    
    # Show sample distributions
    fraud_df = df[df['IsFraud'] == 1]
    legit_df = df[df['IsFraud'] == 0]
    
    print(f"\n📈 SAMPLE STATISTICS:")
    print(f"Average fraud amount: ${fraud_df['Amount'].mean():.2f}")
    print(f"Average legit amount: ${legit_df['Amount'].mean():.2f}")
    print(f"Fraud account age median: {fraud_df['AccountAge'].median():.0f} days")
    print(f"Legit account age median: {legit_df['AccountAge'].median():.0f} days")

if __name__ == "__main__":
    main()