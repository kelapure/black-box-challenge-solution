#!/usr/bin/env python3
"""
DEBUG FEATURE COMPUTATION
Check if features are computed consistently between training and prediction
"""

import json
import pandas as pd
import numpy as np
import pickle

def debug_feature_computation():
    """Debug feature computation issues"""
    print("🔍 DEBUGGING FEATURE COMPUTATION")
    print("=" * 60)
    
    # Load full training data
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    rows = []
    for case in test_cases:
        row = case['input'].copy()
        row['reimbursement'] = case['expected_output']
        rows.append(row)
    
    full_df = pd.DataFrame(rows)
    full_df['miles_per_day'] = full_df['miles_traveled'] / full_df['trip_duration_days']
    full_df['receipts_per_day'] = full_df['total_receipts_amount'] / full_df['trip_duration_days']
    
    print(f"Full training dataset: {len(full_df)} cases")
    print(f"Miles traveled range: {full_df['miles_traveled'].min():.2f} to {full_df['miles_traveled'].max():.2f}")
    
    # Take the first test case
    test_case = test_cases[0]
    print(f"\nTest case: {test_case['input']}")
    print(f"Expected: ${test_case['expected_output']}")
    
    # Create single case DataFrame
    single_case_df = pd.DataFrame({
        'trip_duration_days': [test_case['input']['trip_duration_days']],
        'miles_traveled': [test_case['input']['miles_traveled']],
        'total_receipts_amount': [test_case['input']['total_receipts_amount']]
    })
    single_case_df['miles_per_day'] = single_case_df['miles_traveled'] / single_case_df['trip_duration_days']
    single_case_df['receipts_per_day'] = single_case_df['total_receipts_amount'] / single_case_df['trip_duration_days']
    
    # The problem: miles_exp_normalized uses the max from CURRENT dataframe
    # During training: uses max from full dataset (5000 cases)
    # During prediction: uses max from single case (1 case)
    
    print(f"\nPROBLEM IDENTIFIED:")
    print(f"Training max miles: {full_df['miles_traveled'].max()}")
    print(f"Single case max miles: {single_case_df['miles_traveled'].max()}")
    
    # Show the difference in miles_exp_normalized
    miles_training_max = full_df['miles_traveled'].max()
    miles_single_max = single_case_df['miles_traveled'].max()
    
    miles_value = test_case['input']['miles_traveled']
    
    training_normalized = np.exp(miles_value / miles_training_max)
    single_normalized = np.exp(miles_value / miles_single_max)
    
    print(f"\nFeature: miles_exp_normalized")
    print(f"  During training: exp({miles_value}/{miles_training_max}) = {training_normalized:.6f}")
    print(f"  During prediction: exp({miles_value}/{miles_single_max}) = {single_normalized:.6f}")
    print(f"  Difference: {abs(training_normalized - single_normalized):.6f}")
    
    print(f"\n❌ FEATURE COMPUTATION IS INCONSISTENT!")
    print(f"The miles_exp_normalized feature uses .max() from current DataFrame")
    print(f"This gives different values during training vs prediction")
    
    return miles_training_max

def fix_feature_computation():
    """Create fixed feature computation"""
    print(f"\n🔧 FIXING FEATURE COMPUTATION")
    
    # Store training dataset statistics
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    rows = []
    for case in test_cases:
        row = case['input'].copy()
        row['reimbursement'] = case['expected_output']
        rows.append(row)
    
    full_df = pd.DataFrame(rows)
    
    training_stats = {
        'miles_traveled_max': full_df['miles_traveled'].max(),
        'trip_duration_days_max': full_df['trip_duration_days'].max(),
        'total_receipts_amount_max': full_df['total_receipts_amount'].max()
    }
    
    print(f"Training statistics:")
    for key, value in training_stats.items():
        print(f"  {key}: {value}")
    
    # Save training stats
    with open('training_stats.pkl', 'wb') as f:
        pickle.dump(training_stats, f)
    
    print(f"✅ Training statistics saved to training_stats.pkl")
    
    return training_stats

if __name__ == "__main__":
    max_miles = debug_feature_computation()
    training_stats = fix_feature_computation()