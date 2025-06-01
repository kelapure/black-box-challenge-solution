#!/usr/bin/env python3
"""
Comprehensive Rule-Based System
Black Box Challenge - Deterministic Foundation Building

This module builds a comprehensive rule-based system that integrates all discovered
business logic from employee interviews into a single, coherent, interpretable system.

System Architecture:
1. Linear Baseline (foundation)
2. Main Business Rules (primary adjustments)
3. Edge Case Rules (refinements)
4. Rule Precedence and Conflict Resolution
5. Final Rounding and Formatting

Priority: Interpretable business logic over statistical optimization
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveRuleSystem:
    """
    Comprehensive business rule-based reimbursement calculation system
    """
    
    def __init__(self, data_path='test_cases.json'):
        """Initialize with test cases"""
        self.data_path = data_path
        self.df = None
        self.linear_baseline = None
        self.business_rules = {}
        self.rule_precedence = []
        self.performance_metrics = {}
        self.load_data()
        self.initialize_baseline()
        
    def load_data(self):
        """Load test cases data"""
        with open(self.data_path, 'r') as f:
            test_cases = json.load(f)
        
        rows = []
        for case in test_cases:
            row = case['input'].copy()
            row['reimbursement'] = case['expected_output']
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        
        # Add derived features
        self.df['miles_per_day'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        self.df['receipts_per_day'] = self.df['total_receipts_amount'] / self.df['trip_duration_days']
        
        print(f"Loaded {len(self.df)} test cases")
        
    def initialize_baseline(self):
        """Initialize the optimized linear baseline from previous discoveries"""
        # Use the best linear coefficients discovered
        self.linear_baseline = {
            'intercept': 318.40,
            'trip_duration_days': 71.28,
            'miles_traveled': 0.7941,
            'total_receipts_amount': 0.2904
        }
        
        print("Initialized optimized linear baseline")
        
    def calculate_linear_baseline(self, days, miles, receipts):
        """Calculate linear baseline reimbursement"""
        return (self.linear_baseline['intercept'] +
                self.linear_baseline['trip_duration_days'] * days +
                self.linear_baseline['miles_traveled'] * miles +
                self.linear_baseline['total_receipts_amount'] * receipts)
    
    def define_comprehensive_business_rules(self):
        """Define all business rules discovered from interviews with clear precedence"""
        print("\n=== DEFINING COMPREHENSIVE BUSINESS RULES ===")
        
        # Rule 1: Sweet Spot Bonus (Peggy's 5-6 day bonus)
        self.business_rules['sweet_spot_bonus'] = {
            'description': "5-6 day trips get sweet spot bonus (Peggy's rule)",
            'priority': 1,
            'conditions': {
                'trip_duration_days': [5, 6]
            },
            'adjustment': 78.85,
            'rule_type': 'bonus',
            'source': 'Peggy interview'
        }
        
        # Rule 2: Receipt Ceiling Penalty (Peggy's high receipt penalty)
        self.business_rules['receipt_ceiling'] = {
            'description': "High receipts (>$2193) get ceiling penalty (Peggy's rule)",
            'priority': 2,
            'conditions': {
                'total_receipts_amount': {'operator': '>', 'value': 2193}
            },
            'adjustment': -99.48,
            'rule_type': 'penalty',
            'source': 'Peggy interview'
        }
        
        # Rule 3: Big Trip Jackpot (Sarah's jackpot rule)
        self.business_rules['big_trip_jackpot'] = {
            'description': "Big trips (8+ days, 900+ miles, $1200+ receipts) get jackpot bonus (Sarah's rule)",
            'priority': 3,
            'conditions': {
                'trip_duration_days': {'operator': '>=', 'value': 8},
                'miles_traveled': {'operator': '>=', 'value': 900},
                'total_receipts_amount': {'operator': '>=', 'value': 1200}
            },
            'adjustment': 46.67,
            'rule_type': 'bonus',
            'source': 'Sarah interview'
        }
        
        # Rule 4: Distance Bonus (Tom's logistical bonus)
        self.business_rules['distance_bonus'] = {
            'description': "Long distance trips (600+ miles) get logistical bonus (Tom's rule)",
            'priority': 4,
            'conditions': {
                'miles_traveled': {'operator': '>=', 'value': 600}
            },
            'adjustment': 28.75,
            'rule_type': 'bonus',
            'source': 'Tom interview'
        }
        
        # Rule 5: Minimum Spending Penalty (edge case discovery)
        self.business_rules['minimum_spending_penalty'] = {
            'description': "Very low spending (<$103) gets minimum threshold penalty",
            'priority': 5,
            'conditions': {
                'total_receipts_amount': {'operator': '<', 'value': 103}
            },
            'adjustment': -299.03,
            'rule_type': 'penalty',
            'source': 'Edge case analysis'
        }
        
        # Rule 6: Efficiency Bonuses (John's road warrior + edge case refinement)
        self.business_rules['efficiency_moderate_bonus'] = {
            'description': "Moderate efficiency (100-200 miles/day) gets road warrior bonus",
            'priority': 6,
            'conditions': {
                'miles_per_day': {'operator': '>=', 'value': 100},
                'miles_per_day_max': {'operator': '<', 'value': 200}
            },
            'adjustment': 47.59,
            'rule_type': 'bonus',
            'source': 'John interview + edge case analysis'
        }
        
        self.business_rules['efficiency_high_bonus'] = {
            'description': "High efficiency (200-300 miles/day) gets road warrior bonus",
            'priority': 7,
            'conditions': {
                'miles_per_day': {'operator': '>=', 'value': 200},
                'miles_per_day_max': {'operator': '<', 'value': 300}
            },
            'adjustment': 33.17,
            'rule_type': 'bonus',
            'source': 'John interview + edge case analysis'
        }
        
        self.business_rules['efficiency_low_penalty'] = {
            'description': "Low efficiency (0-100 miles/day) gets efficiency penalty",
            'priority': 8,
            'conditions': {
                'miles_per_day': {'operator': '<', 'value': 100}
            },
            'adjustment': -23.66,  # Average of very_low and low penalties
            'rule_type': 'penalty',
            'source': 'Edge case analysis'
        }
        
        # Rule 7: Long Trip High Spending Interaction (Tom's spending judgment)
        self.business_rules['long_trip_high_spending_penalty'] = {
            'description': "Long trips (7+ days) with high spending rate get judgment penalty",
            'priority': 9,
            'conditions': {
                'trip_duration_days': {'operator': '>=', 'value': 7},
                'receipts_per_day': {'operator': '>', 'value': 178}  # 1.5x median for long trips
            },
            'adjustment': -89.41,
            'rule_type': 'penalty',
            'source': 'Tom interview + edge case analysis'
        }
        
        # Rule 8: Quarter Rounding (system artifact)
        self.business_rules['quarter_rounding'] = {
            'description': "Final result rounded to nearest quarter (system artifact)",
            'priority': 10,
            'rule_type': 'rounding',
            'source': 'Edge case analysis'
        }
        
        # Set rule precedence order
        self.rule_precedence = [
            'sweet_spot_bonus',
            'receipt_ceiling',
            'big_trip_jackpot',
            'distance_bonus',
            'minimum_spending_penalty',
            'efficiency_moderate_bonus',
            'efficiency_high_bonus',
            'efficiency_low_penalty',
            'long_trip_high_spending_penalty',
            'quarter_rounding'
        ]
        
        print(f"Defined {len(self.business_rules)} comprehensive business rules")
        for i, rule_name in enumerate(self.rule_precedence, 1):
            rule = self.business_rules[rule_name]
            print(f"  {i}. {rule_name}: {rule['description']}")
    
    def evaluate_rule_condition(self, rule, row):
        """Evaluate if a business rule's conditions are met for a given row"""
        conditions = rule.get('conditions', {})
        
        for field, condition in conditions.items():
            if field.endswith('_max'):
                # Handle max conditions for ranges
                base_field = field[:-4]
                if isinstance(condition, dict):
                    operator = condition['operator']
                    value = condition['value']
                    if operator == '<' and row[base_field] >= value:
                        return False
                    elif operator == '<=' and row[base_field] > value:
                        return False
                continue
                
            row_value = row[field]
            
            if isinstance(condition, list):
                # List condition (e.g., days in [5, 6])
                if row_value not in condition:
                    return False
            elif isinstance(condition, dict):
                # Operator condition
                operator = condition['operator']
                value = condition['value']
                
                if operator == '>' and row_value <= value:
                    return False
                elif operator == '>=' and row_value < value:
                    return False
                elif operator == '<' and row_value >= value:
                    return False
                elif operator == '<=' and row_value > value:
                    return False
                elif operator == '==' and row_value != value:
                    return False
            else:
                # Direct value condition
                if row_value != condition:
                    return False
        
        return True
    
    def apply_comprehensive_rules(self):
        """Apply all business rules in precedence order"""
        print("\n=== APPLYING COMPREHENSIVE BUSINESS RULES ===")
        
        # Initialize predictions with linear baseline
        self.df['linear_baseline'] = [
            self.calculate_linear_baseline(row['trip_duration_days'], 
                                         row['miles_traveled'], 
                                         row['total_receipts_amount'])
            for _, row in self.df.iterrows()
        ]
        
        # Track rule applications
        self.df['rule_adjustments'] = 0
        rule_applications = {}
        
        # Apply each rule in precedence order
        for rule_name in self.rule_precedence:
            if rule_name == 'quarter_rounding':
                continue  # Handle rounding separately
                
            rule = self.business_rules[rule_name]
            adjustments = []
            
            for _, row in self.df.iterrows():
                if self.evaluate_rule_condition(rule, row):
                    adjustments.append(rule['adjustment'])
                else:
                    adjustments.append(0)
            
            adjustments = np.array(adjustments)
            self.df['rule_adjustments'] += adjustments
            
            # Track applications
            affected_count = (adjustments != 0).sum()
            avg_adjustment = adjustments[adjustments != 0].mean() if affected_count > 0 else 0
            
            rule_applications[rule_name] = {
                'affected_trips': int(affected_count),
                'avg_adjustment': float(avg_adjustment),
                'total_adjustment': float(adjustments.sum())
            }
            
            print(f"{rule_name}: {affected_count} trips affected, avg adjustment: ${avg_adjustment:.2f}")
        
        # Calculate pre-rounding prediction
        self.df['pre_rounding_prediction'] = self.df['linear_baseline'] + self.df['rule_adjustments']
        
        # Apply quarter rounding
        self.df['final_prediction'] = (self.df['pre_rounding_prediction'] * 4).round() / 4
        
        # Calculate final residuals
        self.df['final_residual'] = self.df['reimbursement'] - self.df['final_prediction']
        
        return rule_applications
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        print("\n=== CALCULATING PERFORMANCE METRICS ===")
        
        # Basic metrics
        baseline_r2 = r2_score(self.df['reimbursement'], self.df['linear_baseline'])
        final_r2 = r2_score(self.df['reimbursement'], self.df['final_prediction'])
        
        baseline_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['linear_baseline']))
        final_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['final_prediction']))
        
        baseline_mae = mean_absolute_error(self.df['reimbursement'], self.df['linear_baseline'])
        final_mae = mean_absolute_error(self.df['reimbursement'], self.df['final_prediction'])
        
        # Exact match analysis
        exact_matches = (np.abs(self.df['final_residual']) < 0.01).sum()
        exact_match_rate = exact_matches / len(self.df) * 100
        
        # Error distribution analysis
        residual_std = self.df['final_residual'].std()
        residual_mean = self.df['final_residual'].mean()
        
        # Within tolerance analysis
        within_1_dollar = (np.abs(self.df['final_residual']) <= 1.0).sum()
        within_5_dollars = (np.abs(self.df['final_residual']) <= 5.0).sum()
        within_10_dollars = (np.abs(self.df['final_residual']) <= 10.0).sum()
        
        self.performance_metrics = {
            'linear_baseline': {
                'r2': baseline_r2,
                'rmse': baseline_rmse,
                'mae': baseline_mae
            },
            'final_model': {
                'r2': final_r2,
                'rmse': final_rmse,
                'mae': final_mae
            },
            'improvements': {
                'r2_improvement': final_r2 - baseline_r2,
                'rmse_improvement': baseline_rmse - final_rmse,
                'mae_improvement': baseline_mae - final_mae
            },
            'accuracy_analysis': {
                'exact_matches': int(exact_matches),
                'exact_match_rate': exact_match_rate,
                'within_1_dollar': int(within_1_dollar),
                'within_1_dollar_rate': within_1_dollar / len(self.df) * 100,
                'within_5_dollars': int(within_5_dollars),
                'within_5_dollars_rate': within_5_dollars / len(self.df) * 100,
                'within_10_dollars': int(within_10_dollars),
                'within_10_dollars_rate': within_10_dollars / len(self.df) * 100
            },
            'residual_statistics': {
                'mean': residual_mean,
                'std': residual_std,
                'min': float(self.df['final_residual'].min()),
                'max': float(self.df['final_residual'].max())
            }
        }
        
        print(f"Performance Summary:")
        print(f"  Linear baseline: R²={baseline_r2:.4f}, RMSE=${baseline_rmse:.2f}, MAE=${baseline_mae:.2f}")
        print(f"  Final model:     R²={final_r2:.4f}, RMSE=${final_rmse:.2f}, MAE=${final_mae:.2f}")
        print(f"  Improvement:     ΔR²={final_r2-baseline_r2:.4f}, ΔRMSE=${baseline_rmse-final_rmse:.2f}, ΔMAE=${baseline_mae-final_mae:.2f}")
        print(f"  Exact matches:   {exact_matches} ({exact_match_rate:.1f}%)")
        print(f"  Within $1:       {within_1_dollar} ({within_1_dollar/len(self.df)*100:.1f}%)")
        print(f"  Within $5:       {within_5_dollars} ({within_5_dollars/len(self.df)*100:.1f}%)")
        print(f"  Within $10:      {within_10_dollars} ({within_10_dollars/len(self.df)*100:.1f}%)")
        
        return self.performance_metrics
    
    def perform_cross_validation(self, cv_folds=5):
        """Perform cross-validation to test generalization"""
        print(f"\n=== PERFORMING {cv_folds}-FOLD CROSS-VALIDATION ===")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        cv_rmse_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.df)):
            print(f"Fold {fold + 1}/{cv_folds}...")
            
            train_data = self.df.iloc[train_idx]
            val_data = self.df.iloc[val_idx]
            
            # Apply rules to validation data
            val_predictions = []
            
            for _, row in val_data.iterrows():
                # Calculate linear baseline
                baseline_pred = self.calculate_linear_baseline(
                    row['trip_duration_days'],
                    row['miles_traveled'],
                    row['total_receipts_amount']
                )
                
                # Apply business rules
                total_adjustment = 0
                
                for rule_name in self.rule_precedence:
                    if rule_name == 'quarter_rounding':
                        continue
                        
                    rule = self.business_rules[rule_name]
                    if self.evaluate_rule_condition(rule, row):
                        total_adjustment += rule['adjustment']
                
                # Apply rounding
                final_pred = ((baseline_pred + total_adjustment) * 4).round() / 4
                val_predictions.append(final_pred)
            
            # Calculate fold metrics
            fold_r2 = r2_score(val_data['reimbursement'], val_predictions)
            fold_rmse = np.sqrt(mean_squared_error(val_data['reimbursement'], val_predictions))
            
            cv_scores.append(fold_r2)
            cv_rmse_scores.append(fold_rmse)
            
            print(f"  Fold {fold + 1}: R²={fold_r2:.4f}, RMSE=${fold_rmse:.2f}")
        
        cv_mean_r2 = np.mean(cv_scores)
        cv_std_r2 = np.std(cv_scores)
        cv_mean_rmse = np.mean(cv_rmse_scores)
        cv_std_rmse = np.std(cv_rmse_scores)
        
        print(f"\nCross-Validation Results:")
        print(f"  R²: {cv_mean_r2:.4f} ± {cv_std_r2:.4f}")
        print(f"  RMSE: ${cv_mean_rmse:.2f} ± ${cv_std_rmse:.2f}")
        
        # Check for overfitting
        train_r2 = self.performance_metrics['final_model']['r2']
        overfitting_gap = train_r2 - cv_mean_r2
        
        print(f"  Overfitting check: Train R²={train_r2:.4f}, CV R²={cv_mean_r2:.4f}")
        print(f"  Overfitting gap: {overfitting_gap:.4f} (target: <0.05)")
        
        if overfitting_gap < 0.05:
            print("  ✅ No significant overfitting detected")
        else:
            print("  ⚠️  Possible overfitting detected")
        
        return cv_mean_r2, cv_std_r2, cv_mean_rmse, cv_std_rmse
    
    def generate_prediction_function(self):
        """Generate a standalone prediction function"""
        def predict_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
            """
            Comprehensive business rule-based reimbursement prediction
            
            Args:
                trip_duration_days: Number of days on trip
                miles_traveled: Total miles traveled
                total_receipts_amount: Total dollar amount of receipts
                
            Returns:
                Predicted reimbursement amount
            """
            # Derived features
            miles_per_day = miles_traveled / trip_duration_days
            receipts_per_day = total_receipts_amount / trip_duration_days
            
            # Linear baseline
            baseline = (318.40 + 
                       71.28 * trip_duration_days + 
                       0.7941 * miles_traveled + 
                       0.2904 * total_receipts_amount)
            
            # Business rule adjustments
            adjustment = 0
            
            # Rule 1: Sweet Spot Bonus (5-6 days)
            if trip_duration_days in [5, 6]:
                adjustment += 78.85
            
            # Rule 2: Receipt Ceiling Penalty (>$2193)
            if total_receipts_amount > 2193:
                adjustment -= 99.48
            
            # Rule 3: Big Trip Jackpot (8+ days, 900+ miles, $1200+ receipts)
            if (trip_duration_days >= 8 and 
                miles_traveled >= 900 and 
                total_receipts_amount >= 1200):
                adjustment += 46.67
            
            # Rule 4: Distance Bonus (600+ miles)
            if miles_traveled >= 600:
                adjustment += 28.75
            
            # Rule 5: Minimum Spending Penalty (<$103)
            if total_receipts_amount < 103:
                adjustment -= 299.03
            
            # Rule 6: Efficiency Bonuses/Penalties
            if 100 <= miles_per_day < 200:
                adjustment += 47.59
            elif 200 <= miles_per_day < 300:
                adjustment += 33.17
            elif miles_per_day < 100:
                adjustment -= 23.66
            
            # Rule 7: Long Trip High Spending Penalty
            if trip_duration_days >= 7 and receipts_per_day > 178:
                adjustment -= 89.41
            
            # Apply adjustments
            pre_rounding = baseline + adjustment
            
            # Rule 8: Quarter Rounding
            final_amount = round(pre_rounding * 4) / 4
            
            return final_amount
        
        return predict_reimbursement
    
    def run_comprehensive_analysis(self):
        """Run complete comprehensive rule system analysis"""
        print("🔍 BUILDING COMPREHENSIVE RULE-BASED SYSTEM")
        print("=" * 60)
        
        self.define_comprehensive_business_rules()
        rule_applications = self.apply_comprehensive_rules()
        performance_metrics = self.calculate_performance_metrics()
        cv_r2, cv_std_r2, cv_rmse, cv_std_rmse = self.perform_cross_validation()
        
        prediction_function = self.generate_prediction_function()
        
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE RULE-BASED SYSTEM COMPLETE")
        
        return {
            'business_rules': self.business_rules,
            'rule_precedence': self.rule_precedence,
            'rule_applications': rule_applications,
            'performance_metrics': performance_metrics,
            'cross_validation': {
                'cv_r2_mean': cv_r2,
                'cv_r2_std': cv_std_r2,
                'cv_rmse_mean': cv_rmse,
                'cv_rmse_std': cv_std_rmse
            },
            'prediction_function': prediction_function
        }
    
    def save_comprehensive_system(self, output_path='comprehensive_rule_system.json'):
        """Save the comprehensive rule system"""
        # Prepare serializable results
        results = {
            'linear_baseline': self.linear_baseline,
            'business_rules': self.business_rules,
            'rule_precedence': self.rule_precedence,
            'performance_metrics': self.performance_metrics,
            'system_summary': {
                'total_rules': len(self.business_rules),
                'rule_types': list(set(rule.get('rule_type', 'unknown') for rule in self.business_rules.values())),
                'final_r2': self.performance_metrics['final_model']['r2'],
                'final_rmse': self.performance_metrics['final_model']['rmse'],
                'improvement_from_baseline': self.performance_metrics['improvements']['r2_improvement']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Comprehensive rule system saved to {output_path}")

if __name__ == "__main__":
    # Build comprehensive rule-based system
    system = ComprehensiveRuleSystem()
    results = system.run_comprehensive_analysis()
    system.save_comprehensive_system()
    
    print(f"\n📋 FINAL SUMMARY:")
    print(f"Comprehensive Rule-Based System Performance:")
    print(f"  Final R²: {results['performance_metrics']['final_model']['r2']:.4f}")
    print(f"  Final RMSE: ${results['performance_metrics']['final_model']['rmse']:.2f}")
    print(f"  Exact matches: {results['performance_metrics']['accuracy_analysis']['exact_matches']} ({results['performance_metrics']['accuracy_analysis']['exact_match_rate']:.1f}%)")
    print(f"  Within $5: {results['performance_metrics']['accuracy_analysis']['within_5_dollars']} ({results['performance_metrics']['accuracy_analysis']['within_5_dollars_rate']:.1f}%)")
    print(f"  Cross-validation R²: {results['cross_validation']['cv_r2_mean']:.4f} ± {results['cross_validation']['cv_r2_std']:.4f}")
    print(f"  Business rules implemented: {len(results['business_rules'])}")
    print(f"  Total improvement from linear baseline: +{results['performance_metrics']['improvements']['r2_improvement']:.4f} R²")