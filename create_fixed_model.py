#!/usr/bin/env python3
"""
CREATE FIXED PRODUCTION MODEL
Train with consistent feature computation for training and prediction
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
    """Load full dataset for production"""
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    rows = []
    for case in test_cases:
        row = case['input'].copy()
        row['reimbursement'] = case['expected_output']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    
    print(f"Loaded full dataset: {len(df)} test cases")
    
    return df

def create_breakthrough_features_fixed(df, training_stats):
    """Create breakthrough features with FIXED computation (consistent training/prediction)"""
    features = {}
    
    # BREAKTHROUGH FEATURE SET with FIXED computation
    
    # 1. Geometric mean (top breakthrough feature)
    features['geometric_mean_all'] = (df['trip_duration_days'] * df['miles_traveled'] * df['total_receipts_amount']) ** (1/3)
    
    # 2. Key interaction terms
    features['days_miles_interaction'] = df['trip_duration_days'] * df['miles_traveled']
    features['days_receipts_interaction'] = df['trip_duration_days'] * df['total_receipts_amount']
    features['miles_receipts_interaction'] = df['miles_traveled'] * df['total_receipts_amount']
    features['three_way_interaction'] = df['trip_duration_days'] * df['miles_traveled'] * df['total_receipts_amount']
    
    # 3. Polynomial features
    features['days_miles_poly'] = (df['trip_duration_days'] ** 2) * df['miles_traveled']
    features['miles_receipts_poly'] = (df['miles_traveled'] ** 2) * df['total_receipts_amount']
    features['days_receipts_poly'] = (df['trip_duration_days'] ** 2) * df['total_receipts_amount']
    
    # 4. Harmonic mean
    features['harmonic_mean_days_miles'] = 2 / (1/(df['trip_duration_days']+1) + 1/(df['miles_traveled']+1))
    
    # 5. Advanced transformations (FIXED: use consistent max values)
    features['miles_exp_normalized'] = np.exp(df['miles_traveled'] / training_stats['miles_traveled_max'])
    features['receipts_log'] = np.log(df['total_receipts_amount'] + 1)
    features['days_sqrt'] = np.sqrt(df['trip_duration_days'])
    
    # 6. Original features and key derived
    features['trip_duration_days'] = df['trip_duration_days']
    features['miles_traveled'] = df['miles_traveled']
    features['total_receipts_amount'] = df['total_receipts_amount']
    features['miles_per_day'] = df['miles_per_day']
    features['receipts_per_day'] = df['receipts_per_day']
    
    # 7. Efficiency metrics
    features['cost_per_mile'] = df['total_receipts_amount'] / (df['miles_traveled'] + 1)
    features['total_efficiency'] = df['miles_traveled'] / (df['trip_duration_days'] * df['total_receipts_amount'] + 1)
    features['trip_intensity'] = (df['trip_duration_days'] + df['miles_traveled']) / (df['total_receipts_amount'] + 1)
    
    # Handle any inf/nan values
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(0)
    
    return feature_df

def train_fixed_production_model():
    """Train production model with FIXED feature computation"""
    print("🚀 TRAINING FIXED PRODUCTION MODEL")
    print("=" * 60)
    
    # Load full data
    df = load_full_data()
    
    # Calculate training statistics FIRST
    training_stats = {
        'miles_traveled_max': df['miles_traveled'].max(),
        'trip_duration_days_max': df['trip_duration_days'].max(),
        'total_receipts_amount_max': df['total_receipts_amount'].max()
    }
    
    print(f"Training statistics:")
    for key, value in training_stats.items():
        print(f"  {key}: {value}")
    
    # Create features using the consistent function
    features = create_breakthrough_features_fixed(df, training_stats)
    X = features.values
    y = df['reimbursement'].values
    
    print(f"Training with {X.shape[1]} features on {X.shape[0]} samples")
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train breakthrough model
    model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
    
    print("Training GradientBoostingRegressor...")
    model.fit(X_scaled, y)
    
    # Test with whole dollar rounding
    train_pred = model.predict(X_scaled).round()
    train_r2 = r2_score(y, train_pred)
    
    print(f"Training R² (whole dollar rounding): {train_r2:.6f}")
    
    # Cross-validation
    print("Running cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Cross-validation R²: {cv_mean:.6f} ± {cv_std:.6f}")
    
    # Save model, scaler, and training stats together
    model_data = {
        'model': model,
        'scaler': scaler,
        'training_stats': training_stats,
        'feature_names': list(features.columns),
        'training_r2': train_r2,
        'cv_r2': cv_mean,
        'cv_std': cv_std,
        'approach': 'FIXED Production GradientBoosting + Whole Dollar Rounding',
        'notes': 'FIXED feature computation for consistent training/prediction'
    }
    
    # Save model and training stats
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    with open('training_stats.pkl', 'wb') as f:
        pickle.dump(training_stats, f)
    
    print(f"\n✅ FIXED PRODUCTION MODEL SAVED!")
    print(f"📊 Training R²: {train_r2:.6f}")
    print(f"🔄 Cross-validation R²: {cv_mean:.6f} ± {cv_std:.6f}")
    print(f"📁 Model: best_model.pkl")
    print(f"📁 Stats: training_stats.pkl")
    print(f"🔧 Features: {X.shape[1]} with FIXED computation")
    print(f"💡 Rounding: Whole dollar")
    
    # Test feature computation consistency
    print(f"\n🧪 TESTING FEATURE CONSISTENCY...")
    
    # Test on first case
    test_case = {'trip_duration_days': 3, 'miles_traveled': 114, 'total_receipts_amount': 2.16}
    single_df = pd.DataFrame([test_case])
    single_df['miles_per_day'] = single_df['miles_traveled'] / single_df['trip_duration_days']
    single_df['receipts_per_day'] = single_df['total_receipts_amount'] / single_df['trip_duration_days']
    
    # Create features for single case using same function
    single_features = create_breakthrough_features_fixed(single_df, training_stats)
    
    # Compare miles_exp_normalized feature
    training_value = features.loc[0, 'miles_exp_normalized']  # From full dataset
    single_value = single_features.loc[0, 'miles_exp_normalized']  # From single case
    
    print(f"Feature consistency test (miles_exp_normalized):")
    print(f"  Training computation: {training_value:.6f}")
    print(f"  Single case computation: {single_value:.6f}")
    print(f"  Difference: {abs(training_value - single_value):.6f}")
    
    if abs(training_value - single_value) < 0.001:
        print("✅ Feature computation is CONSISTENT!")
    else:
        print("❌ Feature computation is still INCONSISTENT!")
    
    return model_data

if __name__ == "__main__":
    model_data = train_fixed_production_model()
    
    print(f"\n🎉 FIXED PRODUCTION MODEL READY!")
    print(f"🔧 Features computed consistently between training and prediction")
    print(f"🚀 Expected performance: {model_data['cv_r2']:.4f} R² (cross-validated)")