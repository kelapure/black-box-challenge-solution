#!/usr/bin/env python3
"""
QUICK OVERFITTING CHECK
Critical validation to answer: Is 96.36% R² real or overfitting?

FAST VALIDATION:
1. True holdout test (20% never-touched data)
2. Simple cross-validation check
3. Train vs test gap analysis
4. Quick decision
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean data exactly as breakthrough"""
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    rows = []
    for case in test_cases:
        row = case['input'].copy()
        row['reimbursement'] = case['expected_output']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Same cleaning as breakthrough
    for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    
    df = df[(df['miles_per_day'] >= 1) & (df['miles_per_day'] <= 1000)]
    df = df[~((df['receipts_per_day'] > 10000) | 
               ((df['trip_duration_days'] > 1) & (df['receipts_per_day'] < 1)))]
    
    return df

def create_breakthrough_features(df):
    """Create exact breakthrough features"""
    features = {}
    
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

def main():
    print("🔍 QUICK OVERFITTING CHECK")
    print("=" * 50)
    print("Testing if 96.36% R² breakthrough is real...")
    
    # Load data
    df = load_and_clean_data()
    print(f"Data: {len(df)} trips after cleaning")
    
    # Create features
    features = create_breakthrough_features(df)
    X = features.values
    y = df['reimbursement'].values
    
    # CRITICAL TEST 1: True holdout test
    print("\\n1. TRUE HOLDOUT TEST (20% never-touched data)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train breakthrough model
    model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Test performance
    train_pred = model.predict(X_train_scaled).round()
    test_pred = model.predict(X_test_scaled).round()
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    overfitting_gap = train_r2 - test_r2
    
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"  Training R²: {train_r2:.6f}")
    print(f"  Test R²: {test_r2:.6f}")
    print(f"  Overfitting gap: {overfitting_gap:.6f}")
    print(f"  Test RMSE: ${test_rmse:.2f}")
    
    # CRITICAL TEST 2: Cross-validation check
    print("\\n2. CROSS-VALIDATION CHECK")
    scaler_full = RobustScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    
    cv_scores = cross_val_score(model, X_scaled_full, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"  5-fold CV R²: {cv_mean:.6f} ± {cv_std:.6f}")
    
    # Compare to full dataset training
    model.fit(X_scaled_full, y)
    full_pred = model.predict(X_scaled_full).round()
    full_r2 = r2_score(y, full_pred)
    cv_gap = full_r2 - cv_mean
    
    print(f"  Full training R²: {full_r2:.6f}")
    print(f"  CV gap: {cv_gap:.6f}")
    
    # DECISION ANALYSIS
    print("\\n" + "=" * 50)
    print("🎯 OVERFITTING ASSESSMENT")
    print("=" * 50)
    
    print(f"Key Metrics:")
    print(f"  Holdout Test R²: {test_r2:.6f}")
    print(f"  Cross-Validation R²: {cv_mean:.6f}")
    print(f"  Holdout Gap: {overfitting_gap:.6f}")
    print(f"  CV Gap: {cv_gap:.6f}")
    
    # Decision criteria
    criteria_passed = 0
    
    print(f"\\nDecision Criteria:")
    
    # Criterion 1: Test performance
    if test_r2 >= 0.90:
        print(f"  ✅ Test R² ≥ 90%: {test_r2:.6f}")
        criteria_passed += 1
    elif test_r2 >= 0.85:
        print(f"  ⚠️  Test R² 85-90%: {test_r2:.6f}")
        criteria_passed += 0.5
    else:
        print(f"  ❌ Test R² < 85%: {test_r2:.6f}")
    
    # Criterion 2: Overfitting gap
    if overfitting_gap <= 0.05:
        print(f"  ✅ Low overfitting gap ≤ 5%: {overfitting_gap:.6f}")
        criteria_passed += 1
    elif overfitting_gap <= 0.10:
        print(f"  ⚠️  Moderate overfitting gap ≤ 10%: {overfitting_gap:.6f}")
        criteria_passed += 0.5
    else:
        print(f"  ❌ High overfitting gap > 10%: {overfitting_gap:.6f}")
    
    # Criterion 3: CV consistency
    if cv_gap <= 0.05:
        print(f"  ✅ Consistent CV gap ≤ 5%: {cv_gap:.6f}")
        criteria_passed += 1
    elif cv_gap <= 0.10:
        print(f"  ⚠️  Moderate CV gap ≤ 10%: {cv_gap:.6f}")
        criteria_passed += 0.5
    else:
        print(f"  ❌ Large CV gap > 10%: {cv_gap:.6f}")
    
    # Criterion 4: CV stability
    if cv_std <= 0.02:
        print(f"  ✅ Stable CV std ≤ 2%: {cv_std:.6f}")
        criteria_passed += 1
    elif cv_std <= 0.05:
        print(f"  ⚠️  Moderate CV std ≤ 5%: {cv_std:.6f}")
        criteria_passed += 0.5
    else:
        print(f"  ❌ Unstable CV std > 5%: {cv_std:.6f}")
    
    print(f"\\nCriteria Score: {criteria_passed:.1f}/4.0")
    
    # FINAL DECISION
    print(f"\\nFINAL DECISION:")
    
    if criteria_passed >= 3.5 and test_r2 >= 0.90:
        decision = "🎉 BREAKTHROUGH IS REAL"
        recommendation = "DEPLOY WITH HIGH CONFIDENCE"
        color = "GREEN"
    elif criteria_passed >= 3.0 and test_r2 >= 0.85:
        decision = "✅ BREAKTHROUGH IS LIKELY VALID"
        recommendation = "DEPLOY WITH MONITORING"
        color = "YELLOW"
    elif criteria_passed >= 2.0:
        decision = "⚠️  BREAKTHROUGH IS QUESTIONABLE"
        recommendation = "PROCEED WITH EXTREME CAUTION"
        color = "ORANGE"
    else:
        decision = "❌ BREAKTHROUGH IS OVERFITTING"
        recommendation = "DO NOT DEPLOY - USE SAFER APPROACH"
        color = "RED"
    
    print(f"  {decision}")
    print(f"  Recommendation: {recommendation}")
    
    if test_r2 >= 0.85:
        print(f"\\n🚀 BREAKTHROUGH PERFORMANCE: {test_r2:.4f} R² on holdout test")
        if overfitting_gap <= 0.10:
            print("✅ Overfitting is within acceptable limits")
        else:
            print("⚠️  Overfitting detected but performance still strong")
    else:
        print(f"\\n❌ POOR GENERALIZATION: {test_r2:.4f} R² on holdout test")
        print("🛡️  Recommend using safer hybrid approach (83.32% R²)")
    
    print(f"\\nComparison to Previous Approaches:")
    print(f"  Previous best: 86.78% R² (business logic)")
    print(f"  Safe hybrid: 83.32% R² (validated)")
    print(f"  Breakthrough test: {test_r2:.2%} R² (holdout)")
    
    return {
        'test_r2': test_r2,
        'overfitting_gap': overfitting_gap,
        'cv_r2': cv_mean,
        'decision': decision,
        'deploy_recommended': criteria_passed >= 2.5 and test_r2 >= 0.85
    }

if __name__ == "__main__":
    result = main()
    
    print(f"\\n📊 SUMMARY:")
    print(f"Holdout R²: {result['test_r2']:.6f}")
    print(f"Decision: {result['decision']}")
    print(f"Deploy: {result['deploy_recommended']}")
    
    if result['deploy_recommended']:
        print("\\n🎉 BREAKTHROUGH VALIDATED!")
    else:
        print("\\n⚠️  OVERFITTING CONCERNS - USE CAUTION")