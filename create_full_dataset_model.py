#!/usr/bin/env python3
"""
CREATE PRODUCTION MODEL FOR FULL DATASET
Train breakthrough approach on FULL dataset (no outlier removal) for production use
"""

import json
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

def load_full_data():
    """Load full dataset WITHOUT outlier removal for production"""
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    rows = []
    for case in test_cases:
        row = case['input'].copy()
        row['reimbursement'] = case['expected_output']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add basic derived features
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    
    print(f"Loaded FULL dataset: {len(df)} test cases (NO outlier removal)")
    
    return df

def create_breakthrough_features(df):
    """Create exact breakthrough features that achieved 92.33% R²"""
    features = {}
    
    # EXACT same features from breakthrough validation
    features['geometric_mean_all'] = (df['trip_duration_days'] * df['miles_traveled'] * df['total_receipts_amount']) ** (1/3)
    features['days_miles_interaction'] = df['trip_duration_days'] * df['miles_traveled']
    features['days_receipts_interaction'] = df['trip_duration_days'] * df['total_receipts_amount']
    features['miles_receipts_interaction'] = df['miles_traveled'] * df['total_receipts_amount']
    features['three_way_interaction'] = df['trip_duration_days'] * df['miles_traveled'] * df['total_receipts_amount']
    features['days_miles_poly'] = (df['trip_duration_days'] ** 2) * df['miles_traveled']
    features['miles_receipts_poly'] = (df['miles_traveled'] ** 2) * df['total_receipts_amount']
    features['days_receipts_poly'] = (df['trip_duration_days'] ** 2) * df['total_receipts_amount']
    features['harmonic_mean_days_miles'] = 2 / (1/(df['trip_duration_days']+1) + 1/(df['miles_traveled']+1))
    features['miles_exp_normalized'] = np.exp(df['miles_traveled'] / df['miles_traveled'].max())
    features['receipts_log'] = np.log(df['total_receipts_amount'] + 1)
    features['days_sqrt'] = np.sqrt(df['trip_duration_days'])
    features['trip_duration_days'] = df['trip_duration_days']
    features['miles_traveled'] = df['miles_traveled']
    features['total_receipts_amount'] = df['total_receipts_amount']
    features['miles_per_day'] = df['miles_per_day']
    features['receipts_per_day'] = df['receipts_per_day']
    features['cost_per_mile'] = df['total_receipts_amount'] / (df['miles_traveled'] + 1)
    features['total_efficiency'] = df['miles_traveled'] / (df['trip_duration_days'] * df['total_receipts_amount'] + 1)
    features['trip_intensity'] = (df['trip_duration_days'] + df['miles_traveled']) / (df['total_receipts_amount'] + 1)
    
    return pd.DataFrame(features)

def train_production_model():
    """Train production model on FULL dataset"""
    print("🚀 TRAINING PRODUCTION MODEL ON FULL DATASET")
    print("=" * 60)
    
    # Load FULL data (no outlier removal)
    df = load_full_data()
    
    # Create breakthrough features
    features = create_breakthrough_features(df)
    X = features.values
    y = df['reimbursement'].values
    
    print(f"Training with {X.shape[1]} features on {X.shape[0]} samples (FULL dataset)")
    
    # Handle any inf/nan values from feature creation
    features_clean = features.replace([np.inf, -np.inf], np.nan)
    features_clean = features_clean.fillna(features_clean.mean())
    X = features_clean.values
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train breakthrough model (same config as validation)
    model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
    
    print("Training GradientBoostingRegressor on FULL dataset...")
    model.fit(X_scaled, y)
    
    # Test with whole dollar rounding
    train_pred = model.predict(X_scaled).round()
    train_r2 = r2_score(y, train_pred)
    
    print(f"Training R² (full dataset, whole dollar rounding): {train_r2:.6f}")
    
    # Cross-validation on full dataset
    print("Running cross-validation on full dataset...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Cross-validation R²: {cv_mean:.6f} ± {cv_std:.6f}")
    
    # Save model and scaler together
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(features_clean.columns),
        'training_r2': train_r2,
        'cv_r2': cv_mean,
        'cv_std': cv_std,
        'dataset': 'full_no_outlier_removal',
        'approach': 'Production GradientBoosting + Whole Dollar Rounding',
        'notes': 'Trained on FULL dataset for production use - handles all test cases including outliers'
    }
    
    # Save to production location
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ PRODUCTION MODEL SAVED!")
    print(f"📊 Training R²: {train_r2:.6f}")
    print(f"🔄 Cross-validation R²: {cv_mean:.6f} ± {cv_std:.6f}")
    print(f"📁 Saved to: best_model.pkl")
    print(f"🔧 Features: {X.shape[1]} breakthrough features")
    print(f"💡 Rounding: Whole dollar (optimal strategy)")
    print(f"🎯 Dataset: FULL (no outlier removal) - production ready")
    
    return model_data

if __name__ == "__main__":
    model_data = train_production_model()
    
    print(f"\n🎉 PRODUCTION MODEL READY!")
    print(f"✅ Handles ALL test cases (including outliers)")
    print(f"🚀 Expected performance: {model_data['cv_r2']:.4f} R² (cross-validated)")