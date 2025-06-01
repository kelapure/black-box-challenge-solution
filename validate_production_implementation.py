#!/usr/bin/env python3
"""
VALIDATE PRODUCTION IMPLEMENTATION
Test the updated calculate_reimbursement_final.py against test cases
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import subprocess
import sys

def test_production_implementation():
    """Test the production implementation against test cases"""
    print("🧪 VALIDATING PRODUCTION IMPLEMENTATION")
    print("=" * 60)
    
    # Load test cases
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    print(f"Testing against {len(test_cases)} test cases...")
    
    # Test a sample of cases to validate
    sample_size = min(100, len(test_cases))
    test_sample = test_cases[:sample_size]
    
    predictions = []
    actual = []
    
    for i, case in enumerate(test_sample):
        if i % 20 == 0:
            print(f"  Testing case {i+1}/{sample_size}...")
            
        try:
            # Call the production script
            result = subprocess.run([
                'python', 'calculate_reimbursement_final.py',
                str(case['input']['trip_duration_days']),
                str(case['input']['miles_traveled']),
                str(case['input']['total_receipts_amount'])
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                prediction = float(result.stdout.strip())
                predictions.append(prediction)
                actual.append(case['expected_output'])
            else:
                print(f"    Error in case {i+1}: {result.stderr}")
                break
                
        except Exception as e:
            print(f"    Exception in case {i+1}: {e}")
            break
    
    if len(predictions) == len(actual):
        # Calculate performance metrics
        r2 = r2_score(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        # Exact match analysis
        exact_matches = sum(1 for a, p in zip(actual, predictions) if abs(a - p) < 0.01)
        exact_match_rate = exact_matches / len(actual) * 100
        
        within_1_dollar = sum(1 for a, p in zip(actual, predictions) if abs(a - p) <= 1.0)
        within_5_dollars = sum(1 for a, p in zip(actual, predictions) if abs(a - p) <= 5.0)
        
        print(f"\n✅ PRODUCTION VALIDATION RESULTS:")
        print(f"📊 Sample R²: {r2:.6f}")
        print(f"📊 Sample RMSE: ${rmse:.2f}")
        print(f"📊 Exact matches: {exact_matches}/{len(actual)} ({exact_match_rate:.2f}%)")
        print(f"📊 Within $1: {within_1_dollar}/{len(actual)} ({within_1_dollar/len(actual)*100:.1f}%)")
        print(f"📊 Within $5: {within_5_dollars}/{len(actual)} ({within_5_dollars/len(actual)*100:.1f}%)")
        
        if r2 >= 0.90:
            print(f"\n🎉 EXCELLENT PERFORMANCE! R² ≥ 90%")
        elif r2 >= 0.85:
            print(f"\n🚀 GOOD PERFORMANCE! R² ≥ 85%")
        else:
            print(f"\n⚠️  LOWER PERFORMANCE: R² = {r2:.6f}")
        
        # Test some specific examples
        print(f"\n🔍 SAMPLE PREDICTIONS:")
        for i in range(min(5, len(actual))):
            print(f"  Case {i+1}: Actual=${actual[i]:.2f}, Predicted=${predictions[i]:.2f}, Error=${abs(actual[i]-predictions[i]):.2f}")
        
        return r2, rmse, exact_match_rate
    else:
        print(f"\n❌ VALIDATION FAILED: Only {len(predictions)} predictions out of {len(actual)} test cases")
        return None, None, None

if __name__ == "__main__":
    r2, rmse, exact_rate = test_production_implementation()
    
    if r2 is not None:
        print(f"\n🎯 PRODUCTION VALIDATION SUMMARY:")
        print(f"R²: {r2:.6f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"Exact Match Rate: {exact_rate:.2f}%")
        
        if r2 >= 0.85:
            print(f"\n✅ PRODUCTION IMPLEMENTATION VALIDATED!")
            print(f"Ready for deployment with breakthrough performance.")
        else:
            print(f"\n⚠️  PERFORMANCE BELOW TARGET")
            print(f"Consider further optimization.")
    else:
        print(f"\n❌ VALIDATION FAILED")
        sys.exit(1)