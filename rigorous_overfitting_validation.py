#!/usr/bin/env python3
"""
RIGOROUS OVERFITTING VALIDATION
Black Box Challenge - Critical Validation Before Production

CRITICAL QUESTION: Is our 96.36% R² breakthrough real or overfitting like Task 2.5?

Task 2.5 disaster: 100% training → 0.1% test (catastrophic overfitting)
Current breakthrough: 96.4% training, 91.9% CV → Need rigorous validation

VALIDATION STRATEGY:
1. True holdout test set (20% never-touched data)
2. Multiple cross-validation strategies
3. Feature importance analysis for data leakage
4. Learning curves analysis
5. Temporal validation patterns
6. Bootstrap validation
7. Compare multiple random seeds

DECISION CRITERIA:
- If test R² > 90%: Breakthrough is REAL
- If test R² 85-90%: Breakthrough is MODERATE (still good)
- If test R² < 85%: Overfitting detected (use safe hybrid)
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, 
    learning_curve, validation_curve
)
from sklearn.preprocessing import RobustScaler
# import matplotlib.pyplot as plt  # Removed - not available
import warnings
warnings.filterwarnings('ignore')

class RigorousOverfittingValidation:
    """
    Comprehensive validation to detect overfitting in breakthrough approach
    """
    
    def __init__(self, data_path='test_cases.json'):
        """Initialize with test cases"""
        self.data_path = data_path
        self.df = None
        self.validation_results = {}
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and prepare data exactly as in breakthrough approach"""
        with open(self.data_path, 'r') as f:
            test_cases = json.load(f)
        
        rows = []
        for case in test_cases:
            row = case['input'].copy()
            row['reimbursement'] = case['expected_output']
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        original_size = len(self.df)
        
        print(f"Loaded {len(self.df)} test cases")
        
        # Apply EXACT same cleaning as breakthrough approach
        for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        # Remove impossible combinations
        self.df['miles_per_day'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        self.df['receipts_per_day'] = self.df['total_receipts_amount'] / self.df['trip_duration_days']
        
        self.df = self.df[(self.df['miles_per_day'] >= 1) & (self.df['miles_per_day'] <= 1000)]
        self.df = self.df[~((self.df['receipts_per_day'] > 10000) | 
                           ((self.df['trip_duration_days'] > 1) & (self.df['receipts_per_day'] < 1)))]
        
        print(f"After cleaning: {len(self.df)} trips ({original_size - len(self.df)} removed)")
        
        # Create EXACT same features as breakthrough
        self.create_breakthrough_features()
        
    def create_breakthrough_features(self):
        """Create exact same features as breakthrough approach"""
        features = {}
        
        # Exact same features from breakthrough
        features['geometric_mean_all'] = (self.df['trip_duration_days'] * self.df['miles_traveled'] * self.df['total_receipts_amount']) ** (1/3)
        features['days_miles_interaction'] = self.df['trip_duration_days'] * self.df['miles_traveled']
        features['days_receipts_interaction'] = self.df['trip_duration_days'] * self.df['total_receipts_amount']
        features['miles_receipts_interaction'] = self.df['miles_traveled'] * self.df['total_receipts_amount']
        features['three_way_interaction'] = self.df['trip_duration_days'] * self.df['miles_traveled'] * self.df['total_receipts_amount']
        features['days_miles_poly'] = (self.df['trip_duration_days'] ** 2) * self.df['miles_traveled']
        features['miles_receipts_poly'] = (self.df['miles_traveled'] ** 2) * self.df['total_receipts_amount']
        features['days_receipts_poly'] = (self.df['trip_duration_days'] ** 2) * self.df['total_receipts_amount']
        features['harmonic_mean_days_miles'] = 2 / (1/(self.df['trip_duration_days']+1) + 1/(self.df['miles_traveled']+1))
        features['miles_exp_normalized'] = np.exp(self.df['miles_traveled'] / self.df['miles_traveled'].max())
        features['receipts_log'] = np.log(self.df['total_receipts_amount'] + 1)
        features['days_sqrt'] = np.sqrt(self.df['trip_duration_days'])
        features['trip_duration_days'] = self.df['trip_duration_days']
        features['miles_traveled'] = self.df['miles_traveled']
        features['total_receipts_amount'] = self.df['total_receipts_amount']
        features['miles_per_day'] = self.df['miles_per_day']
        features['receipts_per_day'] = self.df['receipts_per_day']
        features['cost_per_mile'] = self.df['total_receipts_amount'] / (self.df['miles_traveled'] + 1)
        features['total_efficiency'] = self.df['miles_traveled'] / (self.df['trip_duration_days'] * self.df['total_receipts_amount'] + 1)
        features['trip_intensity'] = (self.df['trip_duration_days'] + self.df['miles_traveled']) / (self.df['total_receipts_amount'] + 1)
        
        self.feature_matrix = pd.DataFrame(features)
        print(f"Created {len(features)} features (exact match to breakthrough)")
        
    def create_true_holdout_test_set(self):
        """Create true holdout test set that was NEVER used in development"""
        print("\\n=== CREATING TRUE HOLDOUT TEST SET ===")
        
        # Split data into train/validation (80%) and TRUE TEST (20%)
        # Use stratified split to ensure representative test set
        X = self.feature_matrix.values
        y = self.df['reimbursement'].values
        
        # Split with multiple random states to test stability
        test_results = []
        
        for random_state in [42, 123, 456, 789, 999]:
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state, shuffle=True
            )
            
            print(f"\\nRandom state {random_state}:")
            print(f"  Train/Val: {len(X_trainval)} samples")
            print(f"  Test: {len(X_test)} samples")
            
            # Scale features
            scaler = RobustScaler()
            X_trainval_scaled = scaler.fit_transform(X_trainval)
            X_test_scaled = scaler.transform(X_test)
            
            # Train breakthrough model on train/val data
            model = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5, 
                random_state=42
            )
            model.fit(X_trainval_scaled, y_trainval)
            
            # Test on TRUE holdout
            test_pred = model.predict(X_test_scaled)
            test_pred_rounded = test_pred.round()  # Whole dollar rounding from breakthrough
            
            test_r2 = r2_score(y_test, test_pred_rounded)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred_rounded))
            
            # Training performance for comparison
            train_pred = model.predict(X_trainval_scaled)
            train_pred_rounded = train_pred.round()
            train_r2 = r2_score(y_trainval, train_pred_rounded)
            
            overfitting_gap = train_r2 - test_r2
            
            print(f"  Train R²: {train_r2:.6f}")
            print(f"  Test R²: {test_r2:.6f}")
            print(f"  Overfitting gap: {overfitting_gap:.6f}")
            print(f"  Test RMSE: ${test_rmse:.2f}")
            
            test_results.append({
                'random_state': random_state,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'overfitting_gap': overfitting_gap,
                'test_rmse': test_rmse
            })
        
        # Analyze stability across random states
        test_r2_values = [result['test_r2'] for result in test_results]
        overfitting_gaps = [result['overfitting_gap'] for result in test_results]
        
        mean_test_r2 = np.mean(test_r2_values)
        std_test_r2 = np.std(test_r2_values)
        mean_overfitting = np.mean(overfitting_gaps)
        std_overfitting = np.std(overfitting_gaps)
        
        print(f"\\n📊 HOLDOUT TEST SUMMARY:")
        print(f"Test R² across 5 random splits: {mean_test_r2:.6f} ± {std_test_r2:.6f}")
        print(f"Overfitting gap: {mean_overfitting:.6f} ± {std_overfitting:.6f}")
        
        self.validation_results['holdout_test'] = {
            'individual_results': test_results,
            'mean_test_r2': mean_test_r2,
            'std_test_r2': std_test_r2,
            'mean_overfitting': mean_overfitting,
            'std_overfitting': std_overfitting
        }
        
        return mean_test_r2, mean_overfitting
        
    def deep_cross_validation_analysis(self):
        """Comprehensive cross-validation analysis"""
        print("\\n=== DEEP CROSS-VALIDATION ANALYSIS ===")
        
        X = self.feature_matrix.values
        y = self.df['reimbursement'].values
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test multiple CV strategies
        cv_strategies = [
            ('5-fold', KFold(n_splits=5, shuffle=True, random_state=42)),
            ('10-fold', KFold(n_splits=10, shuffle=True, random_state=42)),
            ('5-fold_alt', KFold(n_splits=5, shuffle=True, random_state=123)),
            ('10-fold_alt', KFold(n_splits=10, shuffle=True, random_state=123))
        ]
        
        model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
        
        cv_results = {}
        
        for cv_name, cv_strategy in cv_strategies:
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_strategy, scoring='r2')
            
            mean_cv = cv_scores.mean()
            std_cv = cv_scores.std()
            
            print(f"{cv_name}: CV R² = {mean_cv:.6f} ± {std_cv:.6f}")
            
            cv_results[cv_name] = {
                'mean': mean_cv,
                'std': std_cv,
                'scores': cv_scores.tolist()
            }
        
        self.validation_results['cross_validation'] = cv_results
        
        # Calculate overall CV stability
        all_means = [results['mean'] for results in cv_results.values()]
        cv_stability = np.std(all_means)
        
        print(f"\\nCV Stability (std of means): {cv_stability:.6f}")
        print("(Lower is better - indicates consistent performance)")
        
        return np.mean(all_means), cv_stability
        
    def learning_curve_analysis(self):
        """Analyze learning curves to detect overfitting patterns"""
        print("\\n=== LEARNING CURVE ANALYSIS ===")
        
        X = self.feature_matrix.values
        y = self.df['reimbursement'].values
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
        
        # Test different training set sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_scaled, y, 
            train_sizes=train_sizes,
            cv=5,
            scoring='r2',
            random_state=42
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        print("Learning curve analysis:")
        print("Training Size | Train R² (±std) | Val R² (±std) | Gap")
        print("-" * 55)
        
        for i, size in enumerate(train_sizes_abs):
            gap = train_mean[i] - val_mean[i]
            print(f"{size:11.0f} | {train_mean[i]:.6f} (±{train_std[i]:.4f}) | {val_mean[i]:.6f} (±{val_std[i]:.4f}) | {gap:.4f}")
        
        # Check for overfitting patterns
        final_gap = train_mean[-1] - val_mean[-1]
        gap_trend = train_mean[-1] - train_mean[-3]  # Is training score still increasing?
        
        print(f"\\nFinal overfitting gap: {final_gap:.6f}")
        print(f"Training score trend (last 3 sizes): {gap_trend:.6f}")
        
        self.validation_results['learning_curve'] = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'final_gap': final_gap,
            'gap_trend': gap_trend
        }
        
        return final_gap, gap_trend
        
    def feature_importance_analysis(self):
        """Analyze feature importance for potential data leakage"""
        print("\\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        X = self.feature_matrix.values
        y = self.df['reimbursement'].values
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Get feature importances
        importances = model.feature_importances_
        feature_names = list(self.feature_matrix.columns)
        
        # Sort by importance
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 most important features:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
        # Check for suspicious patterns
        top_5_importance = sum([imp for _, imp in feature_importance[:5]])
        total_importance = sum([imp for _, imp in feature_importance])
        
        print(f"\\nTop 5 features account for: {top_5_importance/total_importance*100:.1f}% of importance")
        
        # Flag potential data leakage concerns
        leakage_flags = []
        for feature, importance in feature_importance[:5]:
            if importance > 0.3:  # Single feature dominates
                leakage_flags.append(f"High dominance: {feature} ({importance:.3f})")
        
        if leakage_flags:
            print("\\n⚠️  Potential data leakage concerns:")
            for flag in leakage_flags:
                print(f"  - {flag}")
        else:
            print("\\n✅ No obvious data leakage patterns detected")
        
        self.validation_results['feature_importance'] = {
            'feature_importance': feature_importance,
            'top_5_share': top_5_importance/total_importance,
            'leakage_flags': leakage_flags
        }
        
        return leakage_flags
        
    def bootstrap_validation(self):
        """Bootstrap validation for robust performance estimation"""
        print("\\n=== BOOTSTRAP VALIDATION ===")
        
        X = self.feature_matrix.values
        y = self.df['reimbursement'].values
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_bootstrap = 50
        bootstrap_scores = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
            X_boot = X_scaled[indices]
            y_boot = y[indices]
            
            # Out-of-bag samples
            oob_indices = np.setdiff1d(np.arange(len(X_scaled)), indices)
            if len(oob_indices) == 0:
                continue
                
            X_oob = X_scaled[oob_indices]
            y_oob = y[oob_indices]
            
            # Train and test
            model = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5, 
                random_state=i
            )
            model.fit(X_boot, y_boot)
            
            oob_pred = model.predict(X_oob).round()
            oob_score = r2_score(y_oob, oob_pred)
            bootstrap_scores.append(oob_score)
        
        mean_bootstrap = np.mean(bootstrap_scores)
        std_bootstrap = np.std(bootstrap_scores)
        
        print(f"Bootstrap validation (n={len(bootstrap_scores)}):")
        print(f"  Mean R²: {mean_bootstrap:.6f} ± {std_bootstrap:.6f}")
        print(f"  95% CI: [{mean_bootstrap - 1.96*std_bootstrap:.6f}, {mean_bootstrap + 1.96*std_bootstrap:.6f}]")
        
        self.validation_results['bootstrap'] = {
            'mean': mean_bootstrap,
            'std': std_bootstrap,
            'scores': bootstrap_scores
        }
        
        return mean_bootstrap, std_bootstrap
        
    def final_overfitting_assessment(self):
        """Final assessment of overfitting risk"""
        print("\\n" + "=" * 70)
        print("🔍 FINAL OVERFITTING ASSESSMENT")
        print("=" * 70)
        
        # Collect all validation metrics
        holdout_r2 = self.validation_results['holdout_test']['mean_test_r2']
        holdout_gap = self.validation_results['holdout_test']['mean_overfitting']
        
        cv_results = self.validation_results['cross_validation']
        mean_cv_r2 = np.mean([result['mean'] for result in cv_results.values()])
        
        learning_gap = self.validation_results['learning_curve']['final_gap']
        
        bootstrap_r2 = self.validation_results['bootstrap']['mean']
        bootstrap_std = self.validation_results['bootstrap']['std']
        
        leakage_flags = self.validation_results['feature_importance']['leakage_flags']
        
        print(f"VALIDATION SUMMARY:")
        print(f"  Holdout Test R²: {holdout_r2:.6f} (±{self.validation_results['holdout_test']['std_test_r2']:.6f})")
        print(f"  Cross-Validation R²: {mean_cv_r2:.6f}")
        print(f"  Bootstrap R²: {bootstrap_r2:.6f} (±{bootstrap_std:.6f})")
        print(f"  Holdout Overfitting Gap: {holdout_gap:.6f}")
        print(f"  Learning Curve Gap: {learning_gap:.6f}")
        print(f"  Data Leakage Flags: {len(leakage_flags)}")
        
        # DECISION CRITERIA
        print(f"\\nDECISION ANALYSIS:")
        
        # Criterion 1: Holdout test performance
        if holdout_r2 >= 0.90:
            holdout_verdict = "✅ EXCELLENT (≥90%)"
        elif holdout_r2 >= 0.85:
            holdout_verdict = "⚠️  GOOD (85-90%)"
        else:
            holdout_verdict = "❌ POOR (<85% - likely overfitting)"
            
        # Criterion 2: Overfitting gaps
        if holdout_gap <= 0.05 and learning_gap <= 0.05:
            overfitting_verdict = "✅ LOW OVERFITTING (<5%)"
        elif holdout_gap <= 0.10 and learning_gap <= 0.10:
            overfitting_verdict = "⚠️  MODERATE OVERFITTING (5-10%)"
        else:
            overfitting_verdict = "❌ HIGH OVERFITTING (>10%)"
            
        # Criterion 3: Consistency
        holdout_std = self.validation_results['holdout_test']['std_test_r2']
        if holdout_std <= 0.02 and bootstrap_std <= 0.02:
            consistency_verdict = "✅ HIGHLY CONSISTENT"
        elif holdout_std <= 0.05 and bootstrap_std <= 0.05:
            consistency_verdict = "⚠️  MODERATELY CONSISTENT"
        else:
            consistency_verdict = "❌ INCONSISTENT"
            
        # Criterion 4: Data leakage
        if len(leakage_flags) == 0:
            leakage_verdict = "✅ NO LEAKAGE DETECTED"
        else:
            leakage_verdict = f"⚠️  {len(leakage_flags)} POTENTIAL LEAKAGE FLAGS"
        
        print(f"  1. Holdout Performance: {holdout_verdict}")
        print(f"  2. Overfitting Level: {overfitting_verdict}")
        print(f"  3. Consistency: {consistency_verdict}")
        print(f"  4. Data Leakage: {leakage_verdict}")
        
        # FINAL DECISION
        good_criteria = 0
        if holdout_r2 >= 0.85: good_criteria += 1
        if holdout_gap <= 0.10: good_criteria += 1
        if holdout_std <= 0.05: good_criteria += 1
        if len(leakage_flags) == 0: good_criteria += 1
        
        print(f"\\nFINAL DECISION ({good_criteria}/4 criteria passed):")
        
        if good_criteria >= 3 and holdout_r2 >= 0.90:
            decision = "🎉 BREAKTHROUGH IS REAL - DEPLOY WITH CONFIDENCE"
            confidence = "HIGH"
        elif good_criteria >= 3 and holdout_r2 >= 0.85:
            decision = "✅ BREAKTHROUGH IS VALID - DEPLOY WITH MONITORING"
            confidence = "MEDIUM"
        elif good_criteria >= 2:
            decision = "⚠️  BREAKTHROUGH IS QUESTIONABLE - PROCEED WITH CAUTION"
            confidence = "LOW"
        else:
            decision = "❌ BREAKTHROUGH IS OVERFITTING - DO NOT DEPLOY"
            confidence = "VERY LOW"
        
        print(f"  {decision}")
        print(f"  Confidence Level: {confidence}")
        
        if "REAL" in decision or "VALID" in decision:
            print(f"\\n🚀 RECOMMENDED: Use breakthrough approach (Holdout R²: {holdout_r2:.6f})")
        else:
            print(f"\\n🛡️  RECOMMENDED: Use safe hybrid approach instead")
            
        return {
            'holdout_r2': holdout_r2,
            'decision': decision,
            'confidence': confidence,
            'good_criteria': good_criteria,
            'deploy_recommended': "REAL" in decision or "VALID" in decision
        }
        
    def run_rigorous_validation(self):
        """Run complete rigorous validation"""
        print("🔍 STARTING RIGOROUS OVERFITTING VALIDATION")
        print("=" * 70)
        print("CRITICAL: Validating if 96.36% R² breakthrough is real or overfitting")
        
        # Run all validation tests
        holdout_r2, holdout_gap = self.create_true_holdout_test_set()
        cv_r2, cv_stability = self.deep_cross_validation_analysis()
        learning_gap, gap_trend = self.learning_curve_analysis()
        leakage_flags = self.feature_importance_analysis()
        bootstrap_r2, bootstrap_std = self.bootstrap_validation()
        
        # Final assessment
        final_decision = self.final_overfitting_assessment()
        
        print("\\n" + "=" * 70)
        print("🎯 RIGOROUS VALIDATION COMPLETE")
        
        return final_decision

if __name__ == "__main__":
    # Run rigorous overfitting validation
    validator = RigorousOverfittingValidation()
    decision = validator.run_rigorous_validation()
    
    print(f"\\n📋 VALIDATION SUMMARY:")
    print(f"Holdout Test R²: {decision['holdout_r2']:.6f}")
    print(f"Final Decision: {decision['decision']}")
    print(f"Confidence: {decision['confidence']}")
    print(f"Deploy Recommended: {decision['deploy_recommended']}")
    
    if decision['deploy_recommended']:
        print("\\n🎉 BREAKTHROUGH VALIDATED - READY FOR PRODUCTION!")
    else:
        print("\\n⚠️  OVERFITTING DETECTED - BREAKTHROUGH NOT RELIABLE")