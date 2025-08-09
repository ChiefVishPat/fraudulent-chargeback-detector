"""
Data preprocessing module for fraud detection.
Handles feature engineering, encoding, and data preparation.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter
import structlog

logger = structlog.get_logger()


class FraudDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate the fraud dataset."""
        logger.info("Loading fraud dataset", file_path=file_path)
        
        try:
            df = pd.read_csv(file_path)
            logger.info("Dataset loaded successfully", shape=df.shape)
            
            # Validate required columns
            required_columns = [
                'TransactionID', 'CustomerID', 'Amount', 'MerchantCategory',
                'TransactionTime', 'Location', 'PaymentMethod', 'AccountAge',
                'PreviousTransactions', 'IsFraud'
            ]
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            return df
            
        except Exception as e:
            logger.error("Failed to load dataset", error=str(e))
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from the fraud dataset.
        
        Feature Engineering Rationale:
        1. Temporal features: Hour of day, day of week (fraud patterns vary by time)
        2. Amount features: Log transform, amount bins (large amounts more suspicious)
        3. Velocity features: Transaction rate per customer
        4. Risk ratios: Amount vs account age, transactions per day
        5. Categorical encoding: Location, merchant category, payment method
        """
        logger.info("Starting feature engineering")
        df_processed = df.copy()
        
        # 1. Temporal Features
        df_processed['TransactionTime'] = pd.to_datetime(df_processed['TransactionTime'])
        df_processed['transaction_hour'] = df_processed['TransactionTime'].dt.hour
        df_processed['transaction_day_of_week'] = df_processed['TransactionTime'].dt.dayofweek
        df_processed['is_weekend'] = df_processed['transaction_day_of_week'].isin([5, 6]).astype(int)
        df_processed['is_night_transaction'] = ((df_processed['transaction_hour'] >= 22) | 
                                              (df_processed['transaction_hour'] <= 6)).astype(int)
        
        # 2. Amount Features
        df_processed['amount_log'] = np.log1p(df_processed['Amount'])
        df_processed['amount_bins'] = pd.cut(df_processed['Amount'], 
                                           bins=[0, 50, 200, 500, 1000, float('inf')], 
                                           labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 3. Account and Velocity Features
        df_processed['account_age_days'] = df_processed['AccountAge']
        df_processed['transactions_per_day'] = df_processed['PreviousTransactions'] / np.maximum(df_processed['AccountAge'], 1)
        df_processed['amount_per_previous_transaction'] = df_processed['Amount'] / np.maximum(df_processed['PreviousTransactions'], 1)
        
        # 4. Risk Ratios
        df_processed['amount_to_account_age_ratio'] = df_processed['Amount'] / np.maximum(df_processed['AccountAge'], 1)
        df_processed['is_new_account'] = (df_processed['AccountAge'] <= 30).astype(int)
        df_processed['is_high_amount'] = (df_processed['Amount'] > df_processed['Amount'].quantile(0.95)).astype(int)
        df_processed['has_few_transactions'] = (df_processed['PreviousTransactions'] <= 5).astype(int)
        
        # 5. Location Features
        df_processed['is_unknown_location'] = (df_processed['Location'] == 'Unknown').astype(int)
        
        # 6. Encode Categorical Features
        categorical_features = ['MerchantCategory', 'Location', 'PaymentMethod']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df_processed[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df_processed[feature].astype(str))
            else:
                # Handle unseen categories during prediction
                known_categories = self.label_encoders[feature].classes_
                df_processed[feature] = df_processed[feature].astype(str)
                mask = df_processed[feature].isin(known_categories)
                df_processed.loc[~mask, feature] = known_categories[0]  # Default to first category
                df_processed[f'{feature}_encoded'] = self.label_encoders[feature].transform(df_processed[feature])
        
        logger.info("Feature engineering completed", new_features_count=len(df_processed.columns) - len(df.columns))
        return df_processed
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> tuple:
        """Prepare final feature matrix and target variable."""
        
        # Select feature columns (exclude ID columns and target)
        feature_columns = [
            'Amount', 'amount_log', 'amount_bins', 'AccountAge', 'PreviousTransactions',
            'transaction_hour', 'transaction_day_of_week', 'is_weekend', 'is_night_transaction',
            'transactions_per_day', 'amount_per_previous_transaction', 'amount_to_account_age_ratio',
            'is_new_account', 'is_high_amount', 'has_few_transactions', 'is_unknown_location',
            'MerchantCategory_encoded', 'Location_encoded', 'PaymentMethod_encoded'
        ]
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        X = df[feature_columns].copy()
        y = df['IsFraud'] if 'IsFraud' in df.columns else None
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Scaler fitted and features scaled")
        else:
            X_scaled = self.scaler.transform(X)
            logger.info("Features scaled using existing scaler")
        
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        
        return X_scaled, y
    
    def split_data_three_way(self, X: pd.DataFrame, y: pd.Series, 
                            train_size: float = 0.6, val_size: float = 0.2, 
                            test_size: float = 0.2, random_state: int = 42):
        """Split data into train/validation/test sets with proper stratification for imbalanced data."""
        
        # Validate split sizes
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError(f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}")
            
        logger.info("Splitting data with stratification for imbalanced classes",
                   train_size=train_size, val_size=val_size, test_size=test_size, 
                   random_state=random_state)
        
        # First split: train vs (val + test)
        temp_test_size = val_size + test_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        # Second split: validation vs test
        val_ratio = val_size / temp_test_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state + 1,
            stratify=y_temp, shuffle=True
        )
        
        # Log class distributions
        fraud_rate_train = y_train.mean()
        fraud_rate_val = y_val.mean()
        fraud_rate_test = y_test.mean()
        
        train_counter = Counter(y_train)
        val_counter = Counter(y_val)
        test_counter = Counter(y_test)
        
        logger.info("Stratified data split completed",
                   train_samples=len(X_train),
                   val_samples=len(X_val), 
                   test_samples=len(X_test),
                   fraud_rate_train=round(fraud_rate_train, 4),
                   fraud_rate_val=round(fraud_rate_val, 4),
                   fraud_rate_test=round(fraud_rate_test, 4),
                   train_class_dist=dict(train_counter),
                   val_class_dist=dict(val_counter),
                   test_class_dist=dict(test_counter))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 42):
        """Legacy two-way split for backward compatibility."""
        logger.info("Splitting data (two-way split)", test_size=test_size, random_state=random_state)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        fraud_rate_train = y_train.mean()
        fraud_rate_test = y_test.mean()
        
        logger.info("Data split completed",
                   train_samples=len(X_train),
                   test_samples=len(X_test),
                   fraud_rate_train=round(fraud_rate_train, 4),
                   fraud_rate_test=round(fraud_rate_test, 4))
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote_resampling(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              sampling_strategy: str = 'auto', random_state: int = 42):
        """Apply SMOTE resampling to handle imbalanced classes."""
        logger.info("Applying SMOTE resampling for imbalanced data",
                   original_class_distribution=dict(Counter(y_train)),
                   sampling_strategy=sampling_strategy)
        
        # Use SMOTEENN (SMOTE + Edited Nearest Neighbors) for better results
        smote_enn = SMOTEENN(
            smote=SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5),
            enn=EditedNearestNeighbours(sampling_strategy='majority'),
            random_state=random_state
        )
        
        try:
            X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
            
            # Convert back to DataFrame with original column names
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_resampled = pd.Series(y_resampled, name=y_train.name)
            
            logger.info("SMOTE resampling completed successfully",
                       new_class_distribution=dict(Counter(y_resampled)),
                       samples_added=len(X_resampled) - len(X_train))
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning("SMOTE resampling failed, using original data", error=str(e))
            return X_train, y_train
    
    def get_stratified_cv_splits(self, X: pd.DataFrame, y: pd.Series, 
                                n_splits: int = 5, random_state: int = 42):
        """Get stratified cross-validation splits for imbalanced data."""
        logger.info("Creating stratified CV splits", n_splits=n_splits, 
                   class_distribution=dict(Counter(y)))
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return skf.split(X, y)
    
    def process_pipeline(self, file_path: str, train_size: float = 0.6, val_size: float = 0.2, 
                        test_size: float = 0.2, random_state: int = 42, use_smote: bool = True,
                        smote_strategy: str = 'auto'):
        """Complete data processing pipeline with proper imbalanced data handling."""
        logger.info("Starting complete data processing pipeline with imbalanced data handling",
                   train_size=train_size, val_size=val_size, test_size=test_size,
                   use_smote=use_smote, smote_strategy=smote_strategy)
        
        # Load and engineer features
        df = self.load_data(file_path)
        logger.info("Dataset class distribution", 
                   class_counts=dict(df['IsFraud'].value_counts()),
                   fraud_rate=df['IsFraud'].mean())
        
        df_processed = self.engineer_features(df)
        
        # Prepare features and target
        X, y = self.prepare_features(df_processed, fit_scaler=True)
        
        # Three-way stratified split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_three_way(
            X, y, train_size=train_size, val_size=val_size, test_size=test_size, 
            random_state=random_state
        )
        
        # Apply SMOTE to training data only (never to validation/test data)
        X_train_resampled, y_train_resampled = X_train, y_train
        if use_smote:
            X_train_resampled, y_train_resampled = self.apply_smote_resampling(
                X_train, y_train, sampling_strategy=smote_strategy, random_state=random_state
            )
        
        logger.info("Data processing pipeline completed successfully")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_resampled': X_train_resampled,
            'y_train_resampled': y_train_resampled,
            'feature_columns': self.feature_columns,
            'raw_data': df,
            'processed_data': df_processed,
            'cv_splits': list(self.get_stratified_cv_splits(X_train, y_train, random_state=random_state))
        }
    
    def process_single_transaction(self, transaction_data: dict) -> pd.DataFrame:
        """Process a single transaction for prediction."""
        logger.info("Processing single transaction for prediction")
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Engineer features
        df_processed = self.engineer_features(df)
        
        # Prepare features
        X, _ = self.prepare_features(df_processed, fit_scaler=False)
        
        return X