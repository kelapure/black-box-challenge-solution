#!/usr/bin/env python3
"""
Minimal Hybrid Approach for 90%+ Accuracy
Black Box Challenge - Final Solution

Pure business logic has plateaued at 83.10% R². This minimal hybrid approach
combines our interpretable business rule foundation with a heavily constrained
ML component to capture the final residual patterns.

Current: 83.10% R² → Target: 90%+ R² (Gap: 6.90%)

Strategy:
1. Use business rules as the interpretable foundation (83.10% R²)
2. Apply minimal ML residual corrector (max 3 features, heavy regularization)
3. Implement strict overfitting prevention (learned from Task 2.5 failure)
4. Maintain interpretability through business rule dominance
5. Use cross-validation to ensure generalization

Constraints:
- Maximum 3 features in ML component
- Heavy regularization (Ridge/Lasso with high alpha)
- Business rules provide 90%+ of prediction
- ML provides small residual corrections only
- Extensive cross-validation to prevent overfitting
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MinimalHybridApproach:
    """
    Implements minimal hybrid approach: business rules + constrained ML
    """
    
    def __init__(self, data_path='test_cases.json'):
        """Initialize with test cases"""
        self.data_path = data_path
        self.df = None
        self.ml_model = None
        self.feature_scaler = None
        self.hybrid_results = {}
        self.cv_results = {}
        self.load_data()
        
    def load_data(self):
        """Load test cases and apply current best business logic system"""
        # Load test cases
        with open(self.data_path, 'r') as f:
            test_cases = json.load(f)
        
        rows = []
        for case in test_cases:
            row = case['input'].copy()
            row['reimbursement'] = case['expected_output']
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        self.df['miles_per_day'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        self.df['receipts_per_day'] = self.df['total_receipts_amount'] / self.df['trip_duration_days']
        
        print(f"Loaded {len(self.df)} test cases")
        
        # Apply current best business logic system (83.10% R²)
        self.apply_business_logic_foundation()
        
    def apply_business_logic_foundation(self):
        """Apply the best business logic system as foundation"""
        # Linear baseline (interpretable foundation)
        baseline = {
            'intercept': 318.40,
            'trip_duration_days': 71.28,
            'miles_traveled': 0.794100,
            'total_receipts_amount': 0.290366
        }
        
        self.df['business_logic_baseline'] = (
            baseline['intercept'] +
            baseline['trip_duration_days'] * self.df['trip_duration_days'] +
            baseline['miles_traveled'] * self.df['miles_traveled'] +
            baseline['total_receipts_amount'] * self.df['total_receipts_amount']
        )
        
        # Apply all discovered business rules
        self.df['business_rule_adjustment'] = 0
        
        # Core business rules (interpretable and validated)
        mask = self.df['trip_duration_days'].isin([5, 6])
        self.df.loc[mask, 'business_rule_adjustment'] += -34.57
        
        mask = self.df['total_receipts_amount'] > 2025
        self.df.loc[mask, 'business_rule_adjustment'] -= 99.48
        
        mask = ((self.df['trip_duration_days'] >= 8) &
                (self.df['miles_traveled'] >= 900) &
                (self.df['total_receipts_amount'] >= 1200))
        self.df.loc[mask, 'business_rule_adjustment'] += -1.62
        
        mask = self.df['miles_traveled'] >= 550
        self.df.loc[mask, 'business_rule_adjustment'] += -41.02
        
        mask = self.df['total_receipts_amount'] < 103
        self.df.loc[mask, 'business_rule_adjustment'] -= 299.03
        
        mask = (self.df['miles_per_day'] >= 94) & (self.df['miles_per_day'] < 180)
        self.df.loc[mask, 'business_rule_adjustment'] += -27.84
        
        mask = (self.df['miles_per_day'] >= 185) & (self.df['miles_per_day'] < 300)
        self.df.loc[mask, 'business_rule_adjustment'] += -41.29
        
        mask = self.df['miles_per_day'] < 102
        self.df.loc[mask, 'business_rule_adjustment'] += 47.70
        
        mask = ((self.df['trip_duration_days'] >= 7) &
                (self.df['receipts_per_day'] > 178))
        self.df.loc[mask, 'business_rule_adjustment'] -= 89.41
        
        # Apply conservative tier effects (most impactful only)
        self.apply_key_tier_effects()
        
        # Apply conservative ratio effects
        self.apply_key_ratio_effects()
        
        # Calculate business logic prediction and residuals
        self.df['business_logic_prediction'] = self.df['business_logic_baseline'] + self.df['business_rule_adjustment']
        self.df['business_logic_prediction'] = (self.df['business_logic_prediction'] * 4).round() / 4
        self.df['residual_for_ml'] = self.df['reimbursement'] - self.df['business_logic_prediction']
        
        business_r2 = r2_score(self.df['reimbursement'], self.df['business_logic_prediction'])
        business_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['business_logic_prediction']))
        
        print(f"Business logic foundation: R²={business_r2:.4f}, RMSE=${business_rmse:.2f}")
        print(f"Gap to 90% target: {0.90 - business_r2:.4f}")
        print(f"ML component must improve by: {0.90 - business_r2:.4f} R² points")
        
    def apply_key_tier_effects(self):
        """Apply only the most impactful tier effects"""
        # Top miles tier effects only
        miles_effects = [
            ((0, 50), 143.44),
            ((100, 200), 72.23),
            ((500, 800), -54.10)
        ]
        
        for (min_miles, max_miles), effect in miles_effects:
            mask = (self.df['miles_traveled'] >= min_miles) & (self.df['miles_traveled'] < max_miles)
            self.df.loc[mask, 'business_rule_adjustment'] += effect * 0.2  # Conservative
        
        # Top receipt tier effects only
        receipt_effects = [
            ((100, 300), -219.74),
            ((1000, 1500), 158.42)
        ]
        
        for (min_receipts, max_receipts), effect in receipt_effects:
            mask = (self.df['total_receipts_amount'] >= min_receipts) & (self.df['total_receipts_amount'] < max_receipts)
            self.df.loc[mask, 'business_rule_adjustment'] += effect * 0.15  # Conservative
            
    def apply_key_ratio_effects(self):
        """Apply only the most impactful ratio effects"""
        # Calculate key ratios
        self.df['miles_per_dollar'] = self.df['miles_traveled'] / (self.df['total_receipts_amount'] + 1)
        
        # High efficiency spending effect
        mask = (self.df['receipts_per_day'] >= 200) & (self.df['receipts_per_day'] < 250)
        self.df.loc[mask, 'business_rule_adjustment'] += 89.00 * 0.05  # Very conservative
        
        # High miles per dollar efficiency
        mask = (self.df['miles_per_dollar'] >= 1.0) & (self.df['miles_per_dollar'] < 2.0)
        self.df.loc[mask, 'business_rule_adjustment'] += 132.16 * 0.03  # Very conservative
        
    def create_minimal_ml_features(self):
        """Create minimal set of ML features (max 3) for residual correction"""
        print("\\n=== CREATING MINIMAL ML FEATURES ===")
        
        # Test potential features for correlation with residuals
        potential_features = {}
        
        # 1. Non-linear transformations of core variables
        potential_features['sqrt_miles'] = np.sqrt(self.df['miles_traveled'])
        potential_features['log_receipts'] = np.log(self.df['total_receipts_amount'] + 1)
        potential_features['days_squared'] = self.df['trip_duration_days'] ** 2
        
        # 2. Interaction terms
        potential_features['days_miles_interaction'] = self.df['trip_duration_days'] * self.df['miles_traveled']
        potential_features['miles_receipts_interaction'] = self.df['miles_traveled'] * self.df['total_receipts_amount']
        
        # 3. Ratio features
        potential_features['efficiency_ratio'] = self.df['miles_traveled'] / (self.df['trip_duration_days'] * self.df['total_receipts_amount'] + 1)
        potential_features['cost_intensity'] = self.df['total_receipts_amount'] / (self.df['miles_traveled'] + 1)
        
        # 4. Complex patterns that business rules can't capture
        potential_features['trip_complexity'] = (self.df['trip_duration_days'] * self.df['miles_traveled']) / (self.df['total_receipts_amount'] + 100)
        potential_features['spending_efficiency'] = self.df['total_receipts_amount'] / (self.df['trip_duration_days'] ** 2 + 1)
        
        # Calculate correlations with ML residuals
        correlations = []
        for feature_name, feature_values in potential_features.items():
            correlation = np.corrcoef(feature_values, self.df['residual_for_ml'])[0, 1]
            correlations.append((abs(correlation), feature_name, correlation, feature_values))
        
        # Sort by absolute correlation
        correlations.sort(reverse=True)
        
        print("Feature correlations with business logic residuals:")
        for abs_corr, name, corr, values in correlations[:10]:
            print(f"  {name}: {corr:.4f}")
        
        # Select top 3 features (constraint: max 3 features)
        selected_features = {}
        for i in range(min(3, len(correlations))):
            if correlations[i][0] > 0.05:  # Minimum meaningful correlation
                abs_corr, name, corr, values = correlations[i]
                selected_features[name] = values
                print(f"\\nSelected feature {i+1}: {name} (correlation: {corr:.4f})")
        
        if len(selected_features) == 0:
            print("\\n⚠️  No features with meaningful correlation found!")
            print("Using basic interaction terms as fallback...")
            selected_features = {
                'days_miles_interaction': self.df['trip_duration_days'] * self.df['miles_traveled'],
                'sqrt_miles': np.sqrt(self.df['miles_traveled']),
                'log_receipts': np.log(self.df['total_receipts_amount'] + 1)
            }
        
        return selected_features
        
    def implement_constrained_ml_residual_corrector(self):
        """Implement heavily constrained ML component for residual correction"""
        print("\\n=== IMPLEMENTING CONSTRAINED ML RESIDUAL CORRECTOR ===")
        
        # Get minimal features
        ml_features = self.create_minimal_ml_features()
        
        # Create feature matrix
        feature_names = list(ml_features.keys())
        X = np.column_stack([ml_features[name] for name in feature_names])
        y = self.df['residual_for_ml'].values
        
        print(f"\\nML component: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"Features: {feature_names}")
        
        # Standardize features to prevent any single feature from dominating
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Test multiple heavily regularized models with cross-validation
        models_to_test = [
            ('Ridge_high_reg', Ridge(alpha=100.0)),      # Heavy regularization
            ('Ridge_ultra_reg', Ridge(alpha=500.0)),     # Ultra heavy regularization
            ('Lasso_high_reg', Lasso(alpha=10.0)),       # Heavy L1 regularization
            ('Lasso_ultra_reg', Lasso(alpha=50.0)),      # Ultra heavy L1 regularization
            ('ElasticNet_balanced', ElasticNet(alpha=20.0, l1_ratio=0.5))  # Balanced regularization
        ]
        
        print("\\nTesting heavily regularized ML models with 5-fold CV:")
        
        best_model = None
        best_cv_score = -np.inf
        best_model_name = ""
        
        # Use 5-fold cross-validation to prevent overfitting
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models_to_test:
            cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()
            
            print(f"  {model_name}: CV R² = {mean_cv_score:.6f} ± {std_cv_score:.6f}")
            
            # Check for overfitting indicators
            model.fit(X_scaled, y)
            train_r2 = model.score(X_scaled, y)
            overfitting_gap = train_r2 - mean_cv_score
            
            print(f"    Train R² = {train_r2:.6f}, CV R² = {mean_cv_score:.6f}, Gap = {overfitting_gap:.6f}")
            
            # Select model with best CV score and low overfitting
            if mean_cv_score > best_cv_score and overfitting_gap < 0.05:  # Max 5% overfitting gap
                best_cv_score = mean_cv_score
                best_model = model
                best_model_name = model_name
        
        if best_model is None:
            print("\\n⚠️  All models show overfitting! Using most conservative Ridge model...")
            best_model = Ridge(alpha=1000.0)  # Extremely conservative
            best_model.fit(X_scaled, y)
            best_model_name = "Ridge_emergency_conservative"
        
        print(f"\\nSelected model: {best_model_name}")
        print(f"Best CV R²: {best_cv_score:.6f}")
        
        # Store the selected model
        self.ml_model = best_model
        
        # Calculate ML residual corrections
        ml_residual_corrections = self.ml_model.predict(X_scaled)
        
        # Apply conservative scaling to ML corrections (prevent ML from dominating)
        ml_scaling_factor = 0.3  # ML can contribute max 30% of the correction
        ml_residual_corrections *= ml_scaling_factor
        
        print(f"\\nML residual correction stats:")
        print(f"  Mean correction: ${ml_residual_corrections.mean():.2f}")
        print(f"  Std correction: ${ml_residual_corrections.std():.2f}")
        print(f"  Max correction: ${ml_residual_corrections.max():.2f}")
        print(f"  Min correction: ${ml_residual_corrections.min():.2f}")
        print(f"  Scaling factor applied: {ml_scaling_factor}")
        
        # Store ML features and corrections
        self.df['ml_residual_correction'] = ml_residual_corrections
        self.ml_feature_names = feature_names
        
        # Store CV results for validation
        self.cv_results = {
            'best_model_name': best_model_name,
            'best_cv_score': best_cv_score,
            'feature_names': feature_names,
            'scaling_factor': ml_scaling_factor
        }
        
        return ml_residual_corrections
        
    def create_final_hybrid_predictions(self):
        """Create final hybrid predictions combining business logic + ML"""
        print("\\n=== CREATING FINAL HYBRID PREDICTIONS ===")
        
        # Get ML residual corrections
        ml_corrections = self.implement_constrained_ml_residual_corrector()
        
        # Combine business logic foundation with ML residual corrections
        hybrid_pred = self.df['business_logic_prediction'] + self.df['ml_residual_correction']
        
        # Apply quarter rounding (maintaining the business system's rounding)
        hybrid_pred_rounded = (hybrid_pred * 4).round() / 4
        
        # Calculate performance metrics
        business_r2 = r2_score(self.df['reimbursement'], self.df['business_logic_prediction'])
        hybrid_r2 = r2_score(self.df['reimbursement'], hybrid_pred_rounded)
        
        business_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['business_logic_prediction']))
        hybrid_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], hybrid_pred_rounded))
        
        business_mae = mean_absolute_error(self.df['reimbursement'], self.df['business_logic_prediction'])
        hybrid_mae = mean_absolute_error(self.df['reimbursement'], hybrid_pred_rounded)
        
        print(f"\\n=== HYBRID APPROACH RESULTS ===")
        print(f"Business Logic:  R²={business_r2:.6f}, RMSE=${business_rmse:.2f}, MAE=${business_mae:.2f}")
        print(f"Hybrid System:   R²={hybrid_r2:.6f}, RMSE=${hybrid_rmse:.2f}, MAE=${hybrid_mae:.2f}")
        print(f"ML Contribution: ΔR²={hybrid_r2-business_r2:.6f}, ΔRMSE=${business_rmse-hybrid_rmse:.2f}, ΔMAE=${business_mae-hybrid_mae:.2f}")
        
        # Calculate ML contribution percentage
        total_prediction_variance = np.var(hybrid_pred_rounded)
        business_prediction_variance = np.var(self.df['business_logic_prediction'])
        ml_contribution_pct = (1 - business_prediction_variance / total_prediction_variance) * 100
        
        print(f"\\nModel Composition:")
        print(f"  Business Logic Dominance: {100 - ml_contribution_pct:.1f}%")
        print(f"  ML Component Contribution: {ml_contribution_pct:.1f}%")
        
        # Exact match analysis
        exact_matches = (np.abs(self.df['reimbursement'] - hybrid_pred_rounded) < 0.01).sum()
        exact_match_rate = exact_matches / len(self.df) * 100
        
        within_1_dollar = (np.abs(self.df['reimbursement'] - hybrid_pred_rounded) <= 1.0).sum()
        within_5_dollars = (np.abs(self.df['reimbursement'] - hybrid_pred_rounded) <= 5.0).sum()
        within_10_dollars = (np.abs(self.df['reimbursement'] - hybrid_pred_rounded) <= 10.0).sum()
        
        print(f"\\nAccuracy Analysis:")
        print(f"  Exact matches: {exact_matches} ({exact_match_rate:.2f}%)")
        print(f"  Within $1: {within_1_dollar} ({within_1_dollar/len(self.df)*100:.1f}%)")
        print(f"  Within $5: {within_5_dollars} ({within_5_dollars/len(self.df)*100:.1f}%)")
        print(f"  Within $10: {within_10_dollars} ({within_10_dollars/len(self.df)*100:.1f}%)")
        
        # Check target achievement
        target_r2 = 0.90
        if hybrid_r2 >= target_r2:
            print(f"\\n🎯 SUCCESS! Achieved {hybrid_r2:.4f} R² (≥90% target)")
            print("🏆 Hybrid approach successfully bridges the gap!")
        else:
            remaining_gap = target_r2 - hybrid_r2
            print(f"\\n⚠️  Still {remaining_gap:.4f} gap remaining to reach 90% target")
            
        # Store final predictions
        self.df['hybrid_prediction'] = hybrid_pred_rounded
        
        return {
            'hybrid_r2': hybrid_r2,
            'hybrid_rmse': hybrid_rmse,
            'hybrid_mae': hybrid_mae,
            'business_r2': business_r2,
            'ml_improvement': hybrid_r2 - business_r2,
            'ml_contribution_pct': ml_contribution_pct,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_match_rate,
            'within_tolerances': {
                'within_1_dollar': within_1_dollar,
                'within_5_dollars': within_5_dollars,
                'within_10_dollars': within_10_dollars
            },
            'target_achieved': hybrid_r2 >= target_r2
        }
        
    def validate_hybrid_system(self):
        """Validate hybrid system to ensure no overfitting"""
        print("\\n=== VALIDATING HYBRID SYSTEM ===")
        
        # Cross-validation on the full hybrid system
        print("\\nPerforming 5-fold cross-validation on hybrid system:")
        
        # Prepare features
        ml_features = {}
        for name in self.ml_feature_names:
            if name == 'sqrt_miles':
                ml_features[name] = np.sqrt(self.df['miles_traveled'])
            elif name == 'log_receipts':
                ml_features[name] = np.log(self.df['total_receipts_amount'] + 1)
            elif name == 'days_squared':
                ml_features[name] = self.df['trip_duration_days'] ** 2
            elif name == 'days_miles_interaction':
                ml_features[name] = self.df['trip_duration_days'] * self.df['miles_traveled']
            elif name == 'miles_receipts_interaction':
                ml_features[name] = self.df['miles_traveled'] * self.df['total_receipts_amount']
            elif name == 'efficiency_ratio':
                ml_features[name] = self.df['miles_traveled'] / (self.df['trip_duration_days'] * self.df['total_receipts_amount'] + 1)
            elif name == 'cost_intensity':
                ml_features[name] = self.df['total_receipts_amount'] / (self.df['miles_traveled'] + 1)
            elif name == 'trip_complexity':
                ml_features[name] = (self.df['trip_duration_days'] * self.df['miles_traveled']) / (self.df['total_receipts_amount'] + 100)
            elif name == 'spending_efficiency':
                ml_features[name] = self.df['total_receipts_amount'] / (self.df['trip_duration_days'] ** 2 + 1)
        
        X = np.column_stack([ml_features[name] for name in self.ml_feature_names])
        y_residuals = self.df['residual_for_ml'].values
        
        # 5-fold cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_residuals[train_idx], y_residuals[val_idx]
            
            # Business logic predictions (same for all folds)
            business_train = self.df['business_logic_prediction'].iloc[train_idx]
            business_val = self.df['business_logic_prediction'].iloc[val_idx]
            
            # Train ML component
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Use same model type as selected
            if hasattr(self.ml_model, 'alpha'):
                model = type(self.ml_model)(alpha=self.ml_model.alpha)
            else:
                model = type(self.ml_model)()
            
            model.fit(X_train_scaled, y_train)
            
            # Predict ML corrections
            ml_corrections_val = model.predict(X_val_scaled) * 0.3  # Same scaling factor
            
            # Create hybrid predictions
            hybrid_val = business_val + ml_corrections_val
            hybrid_val_rounded = (hybrid_val * 4).round() / 4
            
            # Calculate R² for this fold
            actual_val = self.df['reimbursement'].iloc[val_idx]
            fold_r2 = r2_score(actual_val, hybrid_val_rounded)
            cv_scores.append(fold_r2)
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Full dataset performance
        full_r2 = r2_score(self.df['reimbursement'], self.df['hybrid_prediction'])
        
        # Check for overfitting
        overfitting_gap = full_r2 - cv_mean
        
        print(f"Cross-validation results:")
        print(f"  CV R² mean: {cv_mean:.6f} ± {cv_std:.6f}")
        print(f"  Full R²: {full_r2:.6f}")
        print(f"  Overfitting gap: {overfitting_gap:.6f}")
        
        if overfitting_gap < 0.02:  # Less than 2% gap
            print("✅ Validation passed: No significant overfitting detected")
            validation_status = "PASSED"
        elif overfitting_gap < 0.05:  # Less than 5% gap
            print("⚠️  Validation warning: Minor overfitting detected")
            validation_status = "WARNING"
        else:
            print("❌ Validation failed: Significant overfitting detected")
            validation_status = "FAILED"
        
        self.cv_results.update({
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'full_r2': full_r2,
            'overfitting_gap': overfitting_gap,
            'validation_status': validation_status
        })
        
        return validation_status == "PASSED" or validation_status == "WARNING"
        
    def run_minimal_hybrid_approach(self):
        """Run complete minimal hybrid approach"""
        print("🔍 STARTING MINIMAL HYBRID APPROACH FOR 90%+ ACCURACY")
        print("=" * 70)
        print("STRATEGY: Interpretable business logic foundation + constrained ML residual correction")
        
        # Create hybrid system
        results = self.create_final_hybrid_predictions()
        
        # Validate system
        validation_passed = self.validate_hybrid_system()
        
        print("\\n" + "=" * 70)
        print("🎯 MINIMAL HYBRID APPROACH COMPLETE")
        
        return {
            'performance_results': results,
            'cv_results': self.cv_results,
            'validation_passed': validation_passed,
            'final_r2': results['hybrid_r2'],
            'target_achieved': results['target_achieved'],
            'interpretability_maintained': results['ml_contribution_pct'] < 40  # Business logic dominates
        }
        
    def save_hybrid_results(self, output_path='minimal_hybrid_results.json'):
        """Save minimal hybrid approach results"""
        results = {
            'hybrid_results': self.hybrid_results,
            'cv_results': self.cv_results,
            'approach_summary': {
                'business_logic_dominance': True,
                'ml_features_count': len(self.ml_feature_names) if hasattr(self, 'ml_feature_names') else 0,
                'ml_feature_names': self.ml_feature_names if hasattr(self, 'ml_feature_names') else [],
                'heavy_regularization': True,
                'cross_validation': True
            }
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results = convert_numpy_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\\n💾 Minimal hybrid results saved to {output_path}")

if __name__ == "__main__":
    # Run minimal hybrid approach
    hybrid_system = MinimalHybridApproach()
    results = hybrid_system.run_minimal_hybrid_approach()
    hybrid_system.save_hybrid_results()
    
    print(f"\\n📋 MINIMAL HYBRID APPROACH SUMMARY:")
    print(f"Final R²: {results['final_r2']:.6f}")
    print(f"90% target achieved: {results['target_achieved']}")
    print(f"Interpretability maintained: {results['interpretability_maintained']}")
    print(f"Validation passed: {results['validation_passed']}")
    
    if results['target_achieved']:
        print("🎉 SUCCESS: 90%+ accuracy achieved with minimal hybrid approach!")
        print("🏆 Mission accomplished - interpretable foundation + minimal ML enhancement!")
    else:
        remaining = 0.90 - results['final_r2']
        print(f"⚠️  Remaining gap: {remaining:.4f} to reach 90%")
        print("💡 Consider: Final production implementation with current best system")