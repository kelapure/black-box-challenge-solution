#!/usr/bin/env python3
"""
Linear Coefficient Discovery System
Black Box Challenge - Deterministic Foundation Building

This module systematically tests various linear formulas to discover the optimal
coefficients for the basic deterministic model:
reimbursement = base + rate_per_day * days + rate_per_mile * miles + receipt_rate * receipts

Uses the pattern discovery results as a starting point and explores nearby
coefficient spaces to find the best-fitting linear model.
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import itertools
import warnings
warnings.filterwarnings('ignore')

class LinearCoefficientDiscovery:
    """
    Systematically discovers optimal linear coefficients for the reimbursement formula
    """
    
    def __init__(self, data_path='test_cases.json', patterns_path='discovered_patterns.json'):
        """Initialize with test cases data and existing pattern discoveries"""
        self.data_path = data_path
        self.patterns_path = patterns_path
        self.df = None
        self.patterns = None
        self.results = {}
        self.best_model = None
        self.load_data()
        
    def load_data(self):
        """Load test cases and existing pattern discoveries"""
        # Load test cases
        with open(self.data_path, 'r') as f:
            test_cases = json.load(f)
        
        rows = []
        for case in test_cases:
            row = case['input'].copy()
            row['reimbursement'] = case['expected_output']
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        print(f"Loaded {len(self.df)} test cases")
        
        # Load existing patterns
        try:
            with open(self.patterns_path, 'r') as f:
                self.patterns = json.load(f)
            print("Loaded existing pattern discoveries")
        except FileNotFoundError:
            print("No existing patterns found, will discover from scratch")
            self.patterns = {}
            
    def test_base_linear_model(self):
        """Test the basic 3-parameter linear model"""
        print("\n=== BASE LINEAR MODEL TEST ===")
        
        features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        X = self.df[features]
        y = self.df['reimbursement']
        
        # Fit basic model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"Base Linear Model Results:")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  CV R² (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"  Coefficients:")
        for feature, coef in zip(features, model.coef_):
            print(f"    {feature}: {coef:.6f}")
        print(f"  Intercept: {model.intercept_:.6f}")
        
        self.results['base_linear'] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'coefficients': dict(zip(features, model.coef_)),
            'intercept': model.intercept_,
            'model': model
        }
        
        return model
        
    def explore_coefficient_space(self, base_model):
        """Explore coefficient space around the base model"""
        print("\n=== COEFFICIENT SPACE EXPLORATION ===")
        
        base_coefs = base_model.coef_
        base_intercept = base_model.intercept_
        
        # Define search ranges around base coefficients
        day_range = np.arange(base_coefs[0] * 0.8, base_coefs[0] * 1.2, 1.0)
        mile_range = np.arange(base_coefs[1] * 0.8, base_coefs[1] * 1.2, 0.01)
        receipt_range = np.arange(base_coefs[2] * 0.8, base_coefs[2] * 1.2, 0.01)
        
        print(f"Exploring coefficient ranges:")
        print(f"  Days: {day_range[0]:.2f} to {day_range[-1]:.2f}")
        print(f"  Miles: {mile_range[0]:.3f} to {mile_range[-1]:.3f}")
        print(f"  Receipts: {receipt_range[0]:.3f} to {receipt_range[-1]:.3f}")
        
        best_r2 = -1
        best_params = None
        best_model = None
        
        features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        X = self.df[features]
        y = self.df['reimbursement']
        
        # Sample coefficient space (full search would be too expensive)
        n_samples = 1000
        day_samples = np.random.choice(day_range, n_samples)
        mile_samples = np.random.choice(mile_range, n_samples)
        receipt_samples = np.random.choice(receipt_range, n_samples)
        
        print(f"Testing {n_samples} coefficient combinations...")
        
        for i, (day_coef, mile_coef, receipt_coef) in enumerate(zip(day_samples, mile_samples, receipt_samples)):
            if i % 200 == 0:
                print(f"  Progress: {i}/{n_samples}")
                
            # Test manual model with these coefficients
            y_pred = (base_intercept + 
                     day_coef * self.df['trip_duration_days'] +
                     mile_coef * self.df['miles_traveled'] +
                     receipt_coef * self.df['total_receipts_amount'])
            
            r2 = r2_score(y, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = {
                    'trip_duration_days': day_coef,
                    'miles_traveled': mile_coef,
                    'total_receipts_amount': receipt_coef,
                    'intercept': base_intercept
                }
        
        print(f"\nBest coefficient combination found:")
        print(f"  R²: {best_r2:.6f}")
        print(f"  Days coefficient: {best_params['trip_duration_days']:.6f}")
        print(f"  Miles coefficient: {best_params['miles_traveled']:.6f}")
        print(f"  Receipts coefficient: {best_params['total_receipts_amount']:.6f}")
        print(f"  Intercept: {best_params['intercept']:.6f}")
        
        self.results['coefficient_exploration'] = {
            'best_r2': best_r2,
            'best_params': best_params,
            'search_samples': n_samples
        }
        
        return best_params
        
    def test_intercept_variations(self, base_model):
        """Test different intercept values while keeping coefficients fixed"""
        print("\n=== INTERCEPT VARIATION TEST ===")
        
        features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        X = self.df[features]
        y = self.df['reimbursement']
        
        # Use base model coefficients but vary intercept
        base_coefs = base_model.coef_
        base_intercept = base_model.intercept_
        
        # Test intercepts in range around base
        intercept_range = np.arange(base_intercept * 0.5, base_intercept * 1.5, 10.0)
        
        best_r2 = -1
        best_intercept = None
        
        print(f"Testing {len(intercept_range)} intercept values...")
        
        for intercept in intercept_range:
            y_pred = (intercept + 
                     base_coefs[0] * self.df['trip_duration_days'] +
                     base_coefs[1] * self.df['miles_traveled'] +
                     base_coefs[2] * self.df['total_receipts_amount'])
            
            r2 = r2_score(y, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_intercept = intercept
        
        print(f"Best intercept: {best_intercept:.2f} (R² = {best_r2:.6f})")
        print(f"Original intercept: {base_intercept:.2f}")
        print(f"Improvement: {best_r2 - base_model.score(X, y):.6f}")
        
        self.results['intercept_optimization'] = {
            'best_intercept': best_intercept,
            'best_r2': best_r2,
            'original_intercept': base_intercept,
            'improvement': best_r2 - base_model.score(X, y)
        }
        
        return best_intercept
        
    def test_alternative_formulations(self):
        """Test alternative linear formulations"""
        print("\n=== ALTERNATIVE FORMULATIONS TEST ===")
        
        # Test with derived features
        self.df['miles_per_day'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        self.df['receipts_per_day'] = self.df['total_receipts_amount'] / self.df['trip_duration_days']
        self.df['receipts_per_mile'] = self.df['total_receipts_amount'] / (self.df['miles_traveled'] + 1e-8)
        
        formulations = [
            {
                'name': 'Base + Miles/Day',
                'features': ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day']
            },
            {
                'name': 'Base + Receipts/Day',
                'features': ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'receipts_per_day']
            },
            {
                'name': 'Per-Day Model',
                'features': ['trip_duration_days', 'miles_per_day', 'receipts_per_day']
            },
            {
                'name': 'Efficiency Model',
                'features': ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'receipts_per_mile']
            }
        ]
        
        y = self.df['reimbursement']
        alternative_results = {}
        
        for formulation in formulations:
            print(f"\nTesting {formulation['name']}:")
            
            X = self.df[formulation['features']]
            
            model = LinearRegression()
            model.fit(X, y)
            
            r2 = model.score(X, y)
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  CV R² (5-fold): {cv_mean:.4f}")
            print(f"  Coefficients:")
            for feature, coef in zip(formulation['features'], model.coef_):
                print(f"    {feature}: {coef:.6f}")
            
            alternative_results[formulation['name']] = {
                'r2': r2,
                'rmse': rmse,
                'cv_mean': cv_mean,
                'coefficients': dict(zip(formulation['features'], model.coef_)),
                'intercept': model.intercept_,
                'features': formulation['features']
            }
        
        self.results['alternative_formulations'] = alternative_results
        
        # Find best alternative
        best_alt = max(alternative_results.items(), key=lambda x: x[1]['r2'])
        print(f"\nBest alternative formulation: {best_alt[0]} (R² = {best_alt[1]['r2']:.4f})")
        
        return alternative_results
        
    def build_optimized_model(self):
        """Build the final optimized linear model based on all discoveries"""
        print("\n=== BUILDING OPTIMIZED MODEL ===")
        
        # Use best parameters from coefficient exploration if available
        if 'coefficient_exploration' in self.results:
            best_params = self.results['coefficient_exploration']['best_params']
            
            # Create manual model with optimized coefficients
            features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
            
            def optimized_predict(days, miles, receipts):
                return (best_params['intercept'] +
                       best_params['trip_duration_days'] * days +
                       best_params['miles_traveled'] * miles +
                       best_params['total_receipts_amount'] * receipts)
            
            # Test optimized model
            y_pred = [optimized_predict(row['trip_duration_days'], 
                                      row['miles_traveled'], 
                                      row['total_receipts_amount']) 
                     for _, row in self.df.iterrows()]
            
            y = self.df['reimbursement']
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            print(f"Optimized Linear Model Performance:")
            print(f"  R²: {r2:.6f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAE: ${mae:.2f}")
            print(f"\nOptimized Formula:")
            print(f"  reimbursement = {best_params['intercept']:.2f}")
            print(f"                + {best_params['trip_duration_days']:.4f} * days")
            print(f"                + {best_params['miles_traveled']:.6f} * miles")
            print(f"                + {best_params['total_receipts_amount']:.6f} * receipts")
            
            self.results['optimized_model'] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'formula': best_params,
                'predict_function': optimized_predict
            }
            
            self.best_model = optimized_predict
            
        else:
            print("No coefficient exploration results available, using base model")
            self.best_model = self.results['base_linear']['model']
            
        return self.best_model
        
    def validate_against_patterns(self):
        """Validate discovered coefficients against known patterns"""
        print("\n=== PATTERN VALIDATION ===")
        
        if not self.patterns:
            print("No existing patterns to validate against")
            return
            
        if 'multi_linear' in self.patterns:
            pattern_ml = self.patterns['multi_linear']
            
            print("Comparing with pattern discovery results:")
            print(f"Pattern Discovery R²: {pattern_ml['r2']:.4f}")
            if 'optimized_model' in self.results:
                print(f"Optimized Model R²: {self.results['optimized_model']['r2']:.4f}")
            if 'base_linear' in self.results:
                print(f"Base Linear R²: {self.results['base_linear']['r2']:.4f}")
            
            print(f"\nCoefficient Comparison:")
            print(f"                   Pattern     Base Linear    Optimized")
            
            if 'base_linear' in self.results and 'optimized_model' in self.results:
                base_coefs = self.results['base_linear']['coefficients']
                opt_params = self.results['optimized_model']['formula']
                
                for feature in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']:
                    pattern_val = pattern_ml['coefficients'][feature]
                    base_val = base_coefs[feature]
                    opt_val = opt_params[feature]
                    print(f"{feature:15s}: {pattern_val:8.4f}    {base_val:8.4f}    {opt_val:8.4f}")
                    
    def analyze_residuals(self):
        """Analyze residuals to identify systematic errors"""
        print("\n=== RESIDUAL ANALYSIS ===")
        
        if 'optimized_model' in self.results:
            predict_func = self.results['optimized_model']['predict_function']
            
            y_pred = [predict_func(row['trip_duration_days'], 
                                 row['miles_traveled'], 
                                 row['total_receipts_amount']) 
                     for _, row in self.df.iterrows()]
        elif 'base_linear' in self.results:
            model = self.results['base_linear']['model']
            features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
            X = self.df[features]
            y_pred = model.predict(X)
        else:
            print("No model available for residual analysis")
            return
            
        y_true = self.df['reimbursement']
        residuals = y_true - y_pred
        
        print(f"Residual Statistics:")
        print(f"  Mean: ${residuals.mean():.2f}")
        print(f"  Std: ${residuals.std():.2f}")
        print(f"  Min: ${residuals.min():.2f}")
        print(f"  Max: ${residuals.max():.2f}")
        
        # Analyze residuals by input ranges
        print(f"\nResidual Analysis by Input Ranges:")
        
        # Days
        for days_range in [(1, 3), (4, 6), (7, 10), (11, float('inf'))]:
            mask = ((self.df['trip_duration_days'] >= days_range[0]) & 
                   (self.df['trip_duration_days'] <= days_range[1]))
            if mask.sum() > 0:
                range_residuals = residuals[mask]
                print(f"  Days {days_range[0]}-{days_range[1]}: "
                      f"mean=${range_residuals.mean():.2f}, std=${range_residuals.std():.2f} (n={mask.sum()})")
        
        # Miles
        for miles_range in [(0, 100), (101, 300), (301, 600), (601, float('inf'))]:
            mask = ((self.df['miles_traveled'] >= miles_range[0]) & 
                   (self.df['miles_traveled'] <= miles_range[1]))
            if mask.sum() > 0:
                range_residuals = residuals[mask]
                print(f"  Miles {miles_range[0]}-{miles_range[1]}: "
                      f"mean=${range_residuals.mean():.2f}, std=${range_residuals.std():.2f} (n={mask.sum()})")
        
        self.results['residual_analysis'] = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'residuals': residuals.tolist()
        }
        
        return residuals
        
    def run_full_discovery(self):
        """Run complete linear coefficient discovery process"""
        print("🔍 STARTING LINEAR COEFFICIENT DISCOVERY")
        print("=" * 60)
        
        # Test base linear model
        base_model = self.test_base_linear_model()
        
        # Explore coefficient space
        self.explore_coefficient_space(base_model)
        
        # Test intercept variations
        self.test_intercept_variations(base_model)
        
        # Test alternative formulations
        self.test_alternative_formulations()
        
        # Build optimized model
        self.build_optimized_model()
        
        # Validate against existing patterns
        self.validate_against_patterns()
        
        # Analyze residuals
        self.analyze_residuals()
        
        print("\n" + "=" * 60)
        print("🎯 LINEAR COEFFICIENT DISCOVERY COMPLETE")
        
        return self.results
        
    def save_results(self, output_path='linear_coefficients.json'):
        """Save discovered coefficients and results"""
        # Convert any non-serializable objects
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_value = {}
                for k, v in value.items():
                    if k == 'model':
                        continue  # Skip model objects
                    elif k == 'predict_function':
                        continue  # Skip function objects
                    else:
                        serializable_value[k] = v
                serializable_results[key] = serializable_value
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n💾 Linear coefficient results saved to {output_path}")

if __name__ == "__main__":
    # Run linear coefficient discovery
    discovery = LinearCoefficientDiscovery()
    results = discovery.run_full_discovery()
    discovery.save_results()
    
    print("\n📋 SUMMARY OF KEY FINDINGS:")
    print("-" * 40)
    
    if 'optimized_model' in results:
        opt = results['optimized_model']
        formula = opt['formula']
        print(f"Optimized Linear Model (R² = {opt['r2']:.4f}):")
        print(f"  Formula: ${formula['intercept']:.2f}")
        print(f"         + ${formula['trip_duration_days']:.2f} × days")
        print(f"         + ${formula['miles_traveled']:.4f} × miles")
        print(f"         + ${formula['total_receipts_amount']:.4f} × receipts")
        print(f"  RMSE: ${opt['rmse']:.2f}")
        print(f"  MAE: ${opt['mae']:.2f}")
    elif 'base_linear' in results:
        base = results['base_linear']
        print(f"Base Linear Model (R² = {base['r2']:.4f}):")
        print(f"  RMSE: ${base['rmse']:.2f}")
        print(f"  Cross-validation: {base['cv_mean']:.4f} ± {base['cv_std']:.4f}")