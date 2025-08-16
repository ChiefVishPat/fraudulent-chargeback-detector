"""
Data preprocessing module for credit card fraud detection.
Handles feature engineering, encoding, and data preparation for the new dataset.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import structlog

logger = structlog.get_logger()

class FraudDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, train_path: str = "fraud-detection/fraudTrain_cleaned.csv", 
                  test_path: str = "fraud-detection/fraudTest_cleaned.csv") -> tuple:
        """Load and validate the credit card fraud dataset."""
        logger.info("Loading credit card fraud dataset", train_path=train_path, test_path=test_path)
        
        try:
            # Load training and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info("Dataset loaded successfully", 
                       train_shape=train_df.shape, 
                       test_shape=test_df.shape)
            
            # Validate required columns
            required_columns = [
                'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
                'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
                'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time',
                'merch_lat', 'merch_long', 'is_fraud'
            ]
            
            missing_columns = set(required_columns) - set(train_df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Drop unnamed index column if present
            if 'Unnamed: 0' in train_df.columns:
                train_df = train_df.drop('Unnamed: 0', axis=1)
            if 'Unnamed: 0' in test_df.columns:
                test_df = test_df.drop('Unnamed: 0', axis=1)
                
            return train_df, test_df
            
        except Exception as e:
            logger.error("Failed to load dataset", error=str(e))
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer fraud-focused features from the credit card transaction data.
        Based on analysis: fraud has higher amounts, late night hours, specific categories.
        """
        logger.info("Engineering fraud-focused features", shape=df.shape)
        
        df = df.copy()
        
        # Convert transaction datetime
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])
        
        # === AMOUNT-BASED FEATURES (Major fraud indicator) ===
        # High amounts are strong fraud indicators (avg fraud $531 vs legit $67)
        df['amt_high_risk'] = (df['amt'] > 400).astype(int)  # Above median fraud amount
        df['amt_very_high_risk'] = (df['amt'] > 900).astype(int)  # Above 75th percentile fraud
        df['amt_log'] = np.log1p(df['amt'])  # Log transformation for better distribution
        
        # Amount relative to category (some categories have higher typical amounts)
        category_amt_stats = df.groupby('category')['amt'].agg(['mean', 'std']).fillna(0)
        df['amt_zscore_by_category'] = df.apply(
            lambda row: (row['amt'] - category_amt_stats.loc[row['category'], 'mean']) / 
                       max(category_amt_stats.loc[row['category'], 'std'], 1), axis=1
        )
        
        # === TEMPORAL FEATURES (Critical fraud patterns) ===
        df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
        df['transaction_dow'] = df['trans_date_trans_time'].dt.dayofweek
        df['is_weekend'] = (df['transaction_dow'] >= 5).astype(int)
        
        # Late night fraud pattern (22:00-23:00, 0:00-3:00 are high risk)
        df['is_late_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 3)).astype(int)
        df['is_peak_fraud_hours'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 3)).astype(int)
        
        # Business vs non-business hours
        df['is_business_hours'] = ((df['transaction_hour'] >= 9) & (df['transaction_hour'] <= 17)).astype(int)
        
        # === CATEGORY RISK FEATURES ===
        # Based on analysis: shopping_net (1.76%), misc_net (1.45%), grocery_pos (1.41%) are highest risk
        high_risk_categories = ['shopping_net', 'misc_net', 'grocery_pos']
        df['is_high_risk_category'] = df['category'].isin(high_risk_categories).astype(int)
        
        # === DEMOGRAPHIC FEATURES ===
        # Calculate age at transaction with bounds checking
        df['customer_age'] = (df['trans_date_trans_time'] - df['dob']).dt.days / 365.25
        df['customer_age'] = df['customer_age'].clip(0, 120)
        df['customer_age'] = df['customer_age'].fillna(df['customer_age'].median())
        
        # Age risk categories (young customers might be higher risk)
        df['is_young_customer'] = (df['customer_age'] < 25).astype(int)
        df['is_senior_customer'] = (df['customer_age'] > 65).astype(int)
        
        # Gender encoding
        df['gender_encoded'] = df['gender'].map({'M': 1, 'F': 0})
        
        # === GEOGRAPHICAL FEATURES ===
        # Distance between customer and merchant
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate the great circle distance between two points on the earth"""
            R = 6371  # Radius of the earth in km
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c
        
        # Clip coordinates to valid ranges
        df['lat'] = df['lat'].clip(-90, 90)
        df['long'] = df['long'].clip(-180, 180)
        df['merch_lat'] = df['merch_lat'].clip(-90, 90)
        df['merch_long'] = df['merch_long'].clip(-180, 180)
        
        df['distance_to_merchant'] = haversine_distance(
            df['lat'], df['long'], df['merch_lat'], df['merch_long']
        )
        
        # Distance risk categories
        df['is_distant_transaction'] = (df['distance_to_merchant'] > 100).astype(int)  # > 100km is suspicious
        df['is_very_distant'] = (df['distance_to_merchant'] > 500).astype(int)  # > 500km is very suspicious
        
        # === ENCODING CATEGORICAL FEATURES ===
        categorical_features = ['category', 'state', 'job']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
            else:
                # Handle unseen categories in test data
                known_categories = self.label_encoders[feature].classes_
                df[feature] = df[feature].astype(str)
                unknown_mask = ~df[feature].isin(known_categories)
                df.loc[unknown_mask, feature] = known_categories[0]  # Default to first class
                df[f'{feature}_encoded'] = self.label_encoders[feature].transform(df[feature])
        
        # === FEATURE SELECTION ===
        # Fraud-focused feature set based on analysis
        feature_columns = [
            # Core original features
            'amt', 'amt_log', 'amt_high_risk', 'amt_very_high_risk', 'amt_zscore_by_category',
            
            # Geographic features
            'lat', 'long', 'merch_lat', 'merch_long', 'city_pop',
            'distance_to_merchant', 'is_distant_transaction', 'is_very_distant',
            
            # Temporal features (critical for fraud)
            'unix_time', 'transaction_hour', 'transaction_dow', 'is_weekend',
            'is_late_night', 'is_peak_fraud_hours', 'is_business_hours',
            
            # Demographic features  
            'customer_age', 'is_young_customer', 'is_senior_customer', 'gender_encoded',
            
            # Category risk features
            'is_high_risk_category', 'category_encoded',
            
            # Other categorical encodings
            'state_encoded', 'job_encoded'
        ]
        
        # Filter out any features that don't exist
        feature_columns = [col for col in feature_columns if col in df.columns]
        self.feature_columns = feature_columns
        
        logger.info("Fraud-focused feature engineering completed", 
                   features_created=len(feature_columns),
                   final_shape=df.shape)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> tuple:
        """Prepare features and target for modeling."""
        
        # Get features and target
        X = df[self.feature_columns].copy()
        y = df['is_fraud'].copy()
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                # Convert categorical/object columns to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle missing values with column-specific strategies
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # Replace inf and -inf with NaN first, then fill with median
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                median_val = X[col].median()
                if pd.isna(median_val):
                    # If median is also NaN, use 0
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
            else:
                # For any remaining non-numeric columns
                X[col] = X[col].fillna(0)
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        logger.info("Features prepared", 
                   feature_count=len(self.feature_columns),
                   fraud_rate=y.mean())
        
        return X_scaled, y
    
    def split_data_three_way(self, X, y, train_size=0.6, val_size=0.2, test_size=0.2, 
                           random_state=42):
        """Split data into train/validation/test sets with stratification."""
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        logger.info("Data split completed",
                   train_size=len(X_train), 
                   val_size=len(X_val),
                   test_size=len(X_test),
                   train_fraud_rate=y_train.mean(),
                   val_fraud_rate=y_val.mean(),
                   test_fraud_rate=y_test.mean())
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_smote_resampling(self, X_train, y_train, strategy='auto', random_state=42):
        """Apply SMOTE resampling to training data."""
        logger.info("Applying SMOTE resampling", 
                   original_samples=len(X_train),
                   original_fraud_rate=y_train.mean())
        
        smote = SMOTE(sampling_strategy=strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info("SMOTE resampling completed",
                   new_samples=len(X_resampled),
                   new_fraud_rate=y_resampled.mean())
        
        return X_resampled, y_resampled
    
    def process_pipeline(self, train_path="fraud-detection/fraudTrain_cleaned.csv",
                        test_path="fraud-detection/fraudTest_cleaned.csv",
                        train_size=0.6, val_size=0.2, test_size=0.2,
                        use_smote=True, smote_strategy='auto',
                        random_state=42):
        """Complete data processing pipeline."""
        
        logger.info("Starting data processing pipeline")
        
        # Load data
        train_df, test_df = self.load_data(train_path, test_path)
        
        # Combine for consistent feature engineering
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Engineer features
        combined_df = self.engineer_features(combined_df)
        
        # Prepare features
        X, y = self.prepare_features(combined_df, fit_scaler=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_three_way(
            X, y, train_size, val_size, test_size, random_state
        )
        
        # Apply SMOTE if requested
        X_train_resampled, y_train_resampled = None, None
        if use_smote:
            X_train_resampled, y_train_resampled = self.apply_smote_resampling(
                X_train, y_train, smote_strategy, random_state
            )
        
        result = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_resampled': X_train_resampled,
            'y_train_resampled': y_train_resampled,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        
        logger.info("Data processing pipeline completed",
                   total_features=len(self.feature_columns))
        
        return result
    
    def process_single_transaction(self, transaction_dict: dict):
        """Process a single transaction for prediction."""
        # Create DataFrame from single transaction
        df = pd.DataFrame([transaction_dict])
        
        # Add dummy is_fraud column for feature engineering (not used in prediction)
        df['is_fraud'] = 0
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Get features only (no target variable for prediction)
        X = df[self.feature_columns].copy()
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                # Convert categorical/object columns to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle missing values with column-specific strategies
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # Replace inf and -inf with NaN first, then fill with median
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                median_val = X[col].median()
                if pd.isna(median_val):
                    # If median is also NaN, use 0
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
            else:
                # For any remaining non-numeric columns
                X[col] = X[col].fillna(0)
        
        # Scale features using existing fitted scaler
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        return X_scaled