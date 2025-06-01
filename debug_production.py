#!/usr/bin/env python3
"""
DEBUG PRODUCTION IMPLEMENTATION
"""

import json
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score

def debug_implementation():
    """Debug the production implementation"""
    print("🔍 DEBUGGING PRODUCTION IMPLEMENTATION")
    print("=" * 60)
    
    # Load test cases
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Take first case
    case = test_cases[0]
    print(f"Test case: {case['input']}")
    print(f"Expected: ${case['expected_output']}")
    
    # Test the function directly
    from calculate_reimbursement_final import calculate_reimbursement, create_breakthrough_features
    
    result = calculate_reimbursement(
        case['input']['trip_duration_days'],
        case['input']['miles_traveled'],
        case['input']['total_receipts_amount']
    )
    print(f"Production result: ${result}")
    print(f"Error: ${abs(result - case['expected_output']):.2f}")
    
    # Test feature creation
    input_data = pd.DataFrame({
        'trip_duration_days': [case['input']['trip_duration_days']],
        'miles_traveled': [case['input']['miles_traveled']],
        'total_receipts_amount': [case['input']['total_receipts_amount']]
    })
    
    features = create_breakthrough_features(input_data)
    print(f"\nFeatures created: {features.shape}")
    print(f"Feature names: {list(features.columns)}")
    
    # Check model loading
    try:
        with open('best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"\nModel loaded successfully")
        print(f"Model type: {type(model_data['model'])}")
        print(f"Scaler type: {type(model_data['scaler'])}")
        
        # Test model prediction directly
        X_scaled = model_data['scaler'].transform(features.values)
        raw_pred = model_data['model'].predict(X_scaled)[0]
        rounded_pred = round(raw_pred)
        
        print(f"Raw prediction: {raw_pred}")
        print(f"Rounded prediction: {rounded_pred}")
        
    except Exception as e:
        print(f"Model loading error: {e}")
    
    # Test on multiple cases
    print(f"\n🧪 TESTING MULTIPLE CASES:")
    
    sample_cases = test_cases[:10]
    for i, case in enumerate(sample_cases):
        result = calculate_reimbursement(
            case['input']['trip_duration_days'],
            case['input']['miles_traveled'],
            case['input']['total_receipts_amount']
        )
        error = abs(result - case['expected_output'])
        print(f"Case {i+1}: Expected=${case['expected_output']:.2f}, Got=${result:.2f}, Error=${error:.2f}")

if __name__ == "__main__":
    debug_implementation()