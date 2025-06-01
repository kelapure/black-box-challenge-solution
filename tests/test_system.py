#!/usr/bin/env python3
"""
System Tests
Comprehensive testing of the breakthrough reimbursement system
"""

import sys
import os
import json
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from features import BreakthroughFeatureEngine
from model import BreakthroughModel
from predictor import ReimbursementPredictor


def test_feature_engine():
    """Test the feature engineering pipeline"""
    print("🧪 Testing Feature Engine...")
    
    # Create test data
    test_data = pd.DataFrame({
        'trip_duration_days': [3, 5, 7],
        'miles_traveled': [150, 400, 800],
        'total_receipts_amount': [100, 500, 1200]
    })
    
    # Test feature creation
    engine = BreakthroughFeatureEngine()
    features = engine.create_breakthrough_features(test_data)
    
    # Validate
    assert features.shape[0] == 3, "Wrong number of rows"
    assert features.shape[1] == 20, "Wrong number of features"
    assert 'geometric_mean_all' in features.columns, "Missing key feature"
    assert not features.isnull().any().any(), "Contains NaN values"
    
    print("✅ Feature Engine tests passed")
    return True


def test_model_training():
    """Test model training pipeline"""
    print("🧪 Testing Model Training...")
    
    # Test with small subset
    model = BreakthroughModel()
    
    # Create minimal test data
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Use small subset for testing
    subset_cases = test_cases[:100]
    subset_path = 'test_subset.json'
    
    with open(subset_path, 'w') as f:
        json.dump(subset_cases, f)
    
    try:
        # Train on subset
        results = model.train(subset_path)
        
        # Validate results
        assert 'training_r2' in results, "Missing training R²"
        assert 'cv_r2' in results, "Missing CV R²"
        assert results['training_r2'] > 0, "Invalid training R²"
        
        # Test prediction
        prediction = model.predict(3, 150, 100)
        assert isinstance(prediction, (int, float)), "Invalid prediction type"
        assert prediction > 0, "Invalid prediction value"
        
        print("✅ Model Training tests passed")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(subset_path):
            os.remove(subset_path)


def test_predictor():
    """Test the predictor interface"""
    print("🧪 Testing Predictor Interface...")
    
    predictor = ReimbursementPredictor()
    
    # Test basic prediction
    result = predictor.predict(3, 150, 100)
    assert isinstance(result, (int, float)), "Invalid result type"
    assert result > 0, "Invalid result value"
    
    # Test input validation
    try:
        predictor.predict(-1, 150, 100)
        assert False, "Should have raised ValueError for negative days"
    except ValueError:
        pass  # Expected
    
    try:
        predictor.predict(3, -10, 100)
        assert False, "Should have raised ValueError for negative miles"
    except ValueError:
        pass  # Expected
    
    print("✅ Predictor tests passed")
    return True


def test_production_interface():
    """Test the production command-line interface"""
    print("🧪 Testing Production Interface...")
    
    try:
        # Test command line interface
        result = subprocess.run([
            'python3', 'calculate_reimbursement_final.py', '3', '150', '100'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            prediction = float(result.stdout.strip())
            assert prediction > 0, "Invalid prediction from CLI"
            print("✅ Production Interface tests passed")
            return True
        else:
            print(f"❌ CLI failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Production Interface test failed: {e}")
        return False


def test_performance_validation():
    """Test performance on sample data"""
    print("🧪 Testing Performance Validation...")
    
    # Load test cases
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Test on small sample
    sample_size = min(50, len(test_cases))
    sample_cases = test_cases[:sample_size]
    
    predictor = ReimbursementPredictor()
    
    predictions = []
    actuals = []
    
    for case in sample_cases:
        try:
            pred = predictor.predict(
                case['input']['trip_duration_days'],
                case['input']['miles_traveled'],
                case['input']['total_receipts_amount']
            )
            predictions.append(pred)
            actuals.append(case['expected_output'])
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return False
    
    if len(predictions) == len(actuals):
        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        print(f"📊 Sample Performance: R² = {r2:.4f}, RMSE = ${rmse:.2f}")
        
        if r2 >= 0.80:  # Lower threshold for small sample
            print("✅ Performance Validation tests passed")
            return True
        else:
            print(f"❌ Performance below threshold: {r2:.4f}")
            return False
    else:
        print("❌ Prediction count mismatch")
        return False


def run_all_tests():
    """Run all system tests"""
    print("🚀 RUNNING BREAKTHROUGH SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("Feature Engine", test_feature_engine),
        ("Model Training", test_model_training),
        ("Predictor Interface", test_predictor),
        ("Production Interface", test_production_interface),
        ("Performance Validation", test_performance_validation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
        
        print()  # Add spacing
    
    # Summary
    print("=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    passed_count = sum(results.values())
    print(f"\nOverall: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("🎉 ALL TESTS PASSED! System ready for production.")
        return True
    else:
        print("⚠️  Some tests failed. Review before deployment.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)