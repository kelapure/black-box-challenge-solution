#!/usr/bin/env python3
"""
Precision Parameter Tuning for 95%+ Accuracy
Black Box Challenge - Final Precision Enhancement

This module performs precision parameter tuning to close the final 7.53% gap
from current 87.47% R² to the target 95%+ accuracy. Based on complex business
rule discovery, we now fine-tune all coefficients, thresholds, and bonus
amounts using exact residual patterns.

Strategy:
1. Precision coefficient optimization using residual gradients
2. Threshold fine-tuning for maximum separation
3. Bonus amount calibration based on exact residual analysis
4. Variable coefficient implementation for context-dependent rates
5. Micro-adjustment optimization for remaining systematic errors
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PrecisionParameterTuner:
    """
    Fine-tunes all parameters for maximum precision using residual analysis
    """
    
    def __init__(self, data_path='test_cases.json', complex_rules_path='complex_business_rules.json'):
        """Initialize with test cases and complex business rules"""
        self.data_path = data_path
        self.complex_rules_path = complex_rules_path
        self.df = None
        self.complex_rules = None
        self.optimized_parameters = {}
        self.precision_improvements = {}
        self.load_data()
        
    def load_data(self):
        """Load test cases and complex business rules"""
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
        
        # Load complex business rules
        try:
            with open(self.complex_rules_path, 'r') as f:
                self.complex_rules = json.load(f)
            print("Loaded complex business rules")
        except FileNotFoundError:
            print("No complex rules found")
            self.complex_rules = {}
            
        # Apply current comprehensive system to get baseline residuals
        self.apply_current_comprehensive_system()
        
    def apply_current_comprehensive_system(self):
        """Apply current comprehensive system with all discovered rules"""
        # Start with optimized linear baseline
        baseline = {
            'intercept': 318.40,
            'trip_duration_days': 71.28,
            'miles_traveled': 0.7941,
            'total_receipts_amount': 0.2904
        }
        
        # Calculate linear baseline
        self.df['linear_baseline'] = (
            baseline['intercept'] +
            baseline['trip_duration_days'] * self.df['trip_duration_days'] +
            baseline['miles_traveled'] * self.df['miles_traveled'] +
            baseline['total_receipts_amount'] * self.df['total_receipts_amount']
        )
        
        # Apply all current business rules
        self.df['business_rule_adjustment'] = 0
        
        # Current rules from previous discovery
        mask = self.df['trip_duration_days'].isin([5, 6])
        self.df.loc[mask, 'business_rule_adjustment'] += 78.85
        
        mask = self.df['total_receipts_amount'] > 2193
        self.df.loc[mask, 'business_rule_adjustment'] -= 99.48
        
        mask = ((self.df['trip_duration_days'] >= 8) &
                (self.df['miles_traveled'] >= 900) &
                (self.df['total_receipts_amount'] >= 1200))
        self.df.loc[mask, 'business_rule_adjustment'] += 46.67
        
        mask = self.df['miles_traveled'] >= 600
        self.df.loc[mask, 'business_rule_adjustment'] += 28.75
        
        mask = self.df['total_receipts_amount'] < 103
        self.df.loc[mask, 'business_rule_adjustment'] -= 299.03
        
        mask = (self.df['miles_per_day'] >= 100) & (self.df['miles_per_day'] < 200)
        self.df.loc[mask, 'business_rule_adjustment'] += 47.59
        
        mask = (self.df['miles_per_day'] >= 200) & (self.df['miles_per_day'] < 300)
        self.df.loc[mask, 'business_rule_adjustment'] += 33.17
        
        mask = self.df['miles_per_day'] < 100
        self.df.loc[mask, 'business_rule_adjustment'] -= 23.66
        
        mask = ((self.df['trip_duration_days'] >= 7) &
                (self.df['receipts_per_day'] > 178))
        self.df.loc[mask, 'business_rule_adjustment'] -= 89.41
        
        # Calculate current predictions and residuals
        self.df['pre_rounding_prediction'] = self.df['linear_baseline'] + self.df['business_rule_adjustment']
        self.df['current_prediction'] = (self.df['pre_rounding_prediction'] * 4).round() / 4
        self.df['residual'] = self.df['reimbursement'] - self.df['current_prediction']
        
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        current_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction']))
        
        print(f"Current comprehensive system: R²={current_r2:.4f}, RMSE=${current_rmse:.2f}")
        print(f"Gap to 95% target: {0.95 - current_r2:.4f}")
        
    def optimize_linear_coefficients_precision(self):
        """Optimize linear coefficients using gradient-based residual analysis"""
        print("\n=== OPTIMIZING LINEAR COEFFICIENTS FOR PRECISION ===")
        
        def objective_function(coeffs):
            """Objective function for coefficient optimization"""
            intercept, days_coef, miles_coef, receipts_coef = coeffs
            
            # Calculate predictions with new coefficients
            linear_pred = (intercept + 
                          days_coef * self.df['trip_duration_days'] +
                          miles_coef * self.df['miles_traveled'] +
                          receipts_coef * self.df['total_receipts_amount'])
            
            # Add business rule adjustments
            total_pred = linear_pred + self.df['business_rule_adjustment']
            
            # Apply quarter rounding
            final_pred = (total_pred * 4).round() / 4
            
            # Calculate negative R² (for minimization)
            r2 = r2_score(self.df['reimbursement'], final_pred)
            return -r2
        
        # Current coefficients as starting point
        initial_coeffs = [318.40, 71.28, 0.7941, 0.2904]
        
        # Define bounds (allow ±20% variation)
        bounds = [
            (250, 400),     # intercept
            (55, 90),       # days coefficient  
            (0.6, 1.0),     # miles coefficient
            (0.2, 0.4)      # receipts coefficient
        ]
        
        # Optimize coefficients
        print("Optimizing linear coefficients...")
        result = minimize(objective_function, initial_coeffs, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            opt_intercept, opt_days, opt_miles, opt_receipts = result.x
            
            print(f"Optimization successful!")
            print(f"Original coefficients:")
            print(f"  Intercept: ${initial_coeffs[0]:.2f} → ${opt_intercept:.2f}")
            print(f"  Days: ${initial_coeffs[1]:.4f} → ${opt_days:.4f}")
            print(f"  Miles: ${initial_coeffs[2]:.6f} → ${opt_miles:.6f}")
            print(f"  Receipts: ${initial_coeffs[3]:.6f} → ${opt_receipts:.6f}")
            
            # Test improvement
            original_r2 = -objective_function(initial_coeffs)
            optimized_r2 = -result.fun
            improvement = optimized_r2 - original_r2
            
            print(f"R² improvement: {improvement:.6f} ({improvement*100:.4f}%)")
            
            self.optimized_parameters['linear_coefficients'] = {
                'intercept': opt_intercept,
                'trip_duration_days': opt_days,
                'miles_traveled': opt_miles,
                'total_receipts_amount': opt_receipts,
                'improvement': improvement
            }
            
            # Update baseline with optimized coefficients
            self.df['optimized_linear_baseline'] = (
                opt_intercept +
                opt_days * self.df['trip_duration_days'] +
                opt_miles * self.df['miles_traveled'] +
                opt_receipts * self.df['total_receipts_amount']
            )
            
        else:
            print("Coefficient optimization failed")
            self.df['optimized_linear_baseline'] = self.df['linear_baseline']
            
    def optimize_business_rule_thresholds(self):
        """Optimize thresholds for business rules using residual separation analysis"""
        print("\n=== OPTIMIZING BUSINESS RULE THRESHOLDS ===")
        
        threshold_optimizations = {}
        
        # 1. Optimize receipt ceiling threshold
        print("\n1. Receipt ceiling threshold optimization:")
        current_threshold = 2193
        
        # Test thresholds around current value
        test_thresholds = np.arange(2000, 2400, 25)
        best_threshold = current_threshold
        best_separation = 0
        
        for threshold in test_thresholds:
            below = self.df[self.df['total_receipts_amount'] <= threshold]
            above = self.df[self.df['total_receipts_amount'] > threshold]
            
            if len(below) > 100 and len(above) > 50:
                below_residual = below['residual'].mean()
                above_residual = above['residual'].mean()
                separation = abs(above_residual - below_residual)
                
                if separation > best_separation:
                    best_separation = separation
                    best_threshold = threshold
        
        print(f"  Current threshold: ${current_threshold}")
        print(f"  Optimized threshold: ${best_threshold}")
        print(f"  Separation improvement: {best_separation:.2f}")
        
        threshold_optimizations['receipt_ceiling'] = {
            'original': current_threshold,
            'optimized': best_threshold,
            'separation': best_separation
        }
        
        # 2. Optimize distance bonus threshold
        print("\n2. Distance bonus threshold optimization:")
        current_threshold = 600
        
        test_thresholds = np.arange(550, 700, 25)
        best_threshold = current_threshold
        best_separation = 0
        
        for threshold in test_thresholds:
            below = self.df[self.df['miles_traveled'] < threshold]
            above = self.df[self.df['miles_traveled'] >= threshold]
            
            if len(below) > 100 and len(above) > 100:
                below_residual = below['residual'].mean()
                above_residual = above['residual'].mean()
                separation = abs(above_residual - below_residual)
                
                if separation > best_separation:
                    best_separation = separation
                    best_threshold = threshold
        
        print(f"  Current threshold: {current_threshold} miles")
        print(f"  Optimized threshold: {best_threshold} miles")
        print(f"  Separation improvement: {best_separation:.2f}")
        
        threshold_optimizations['distance_bonus'] = {
            'original': current_threshold,
            'optimized': best_threshold,
            'separation': best_separation
        }
        
        # 3. Optimize efficiency thresholds
        print("\n3. Efficiency threshold optimization:")
        
        # Test efficiency boundaries
        efficiency_tests = [
            ('low_efficiency_upper', 100, np.arange(90, 110, 2)),
            ('medium_efficiency_lower', 100, np.arange(90, 110, 2)),
            ('medium_efficiency_upper', 200, np.arange(180, 220, 5)),
            ('high_efficiency_lower', 200, np.arange(180, 220, 5))
        ]
        
        for boundary_name, current_val, test_range in efficiency_tests:
            best_threshold = current_val
            best_separation = 0
            
            for threshold in test_range:
                if 'lower' in boundary_name:
                    below = self.df[self.df['miles_per_day'] < threshold]
                    above = self.df[self.df['miles_per_day'] >= threshold]
                else:
                    below = self.df[self.df['miles_per_day'] <= threshold]
                    above = self.df[self.df['miles_per_day'] > threshold]
                
                if len(below) > 50 and len(above) > 50:
                    below_residual = below['residual'].mean()
                    above_residual = above['residual'].mean()
                    separation = abs(above_residual - below_residual)
                    
                    if separation > best_separation:
                        best_separation = separation
                        best_threshold = threshold
            
            print(f"  {boundary_name}: {current_val} → {best_threshold} (separation: {best_separation:.2f})")
            
            threshold_optimizations[boundary_name] = {
                'original': current_val,
                'optimized': best_threshold,
                'separation': best_separation
            }
        
        self.optimized_parameters['thresholds'] = threshold_optimizations
        
    def optimize_bonus_amounts_precision(self):
        """Optimize bonus amounts using exact residual analysis"""
        print("\n=== OPTIMIZING BONUS AMOUNTS FOR PRECISION ===")
        
        bonus_optimizations = {}
        
        # 1. Sweet spot bonus optimization
        print("\n1. Sweet spot bonus optimization:")
        sweet_spot_mask = self.df['trip_duration_days'].isin([5, 6])
        other_mask = ~sweet_spot_mask
        
        if sweet_spot_mask.sum() > 0 and other_mask.sum() > 0:
            sweet_spot_residual = self.df[sweet_spot_mask]['residual'].mean()
            other_residual = self.df[other_mask]['residual'].mean()
            optimal_bonus = sweet_spot_residual - other_residual
            
            current_bonus = 78.85
            print(f"  Current bonus: ${current_bonus:.2f}")
            print(f"  Optimal bonus: ${optimal_bonus:.2f}")
            print(f"  Adjustment: ${optimal_bonus - current_bonus:.2f}")
            
            bonus_optimizations['sweet_spot'] = {
                'original': current_bonus,
                'optimized': optimal_bonus,
                'adjustment': optimal_bonus - current_bonus
            }
        
        # 2. Distance bonus optimization
        print("\n2. Distance bonus optimization:")
        distance_mask = self.df['miles_traveled'] >= 600
        no_distance_mask = ~distance_mask
        
        if distance_mask.sum() > 0 and no_distance_mask.sum() > 0:
            distance_residual = self.df[distance_mask]['residual'].mean()
            no_distance_residual = self.df[no_distance_mask]['residual'].mean()
            optimal_bonus = distance_residual - no_distance_residual
            
            current_bonus = 28.75
            print(f"  Current bonus: ${current_bonus:.2f}")
            print(f"  Optimal bonus: ${optimal_bonus:.2f}")
            print(f"  Adjustment: ${optimal_bonus - current_bonus:.2f}")
            
            bonus_optimizations['distance_bonus'] = {
                'original': current_bonus,
                'optimized': optimal_bonus,
                'adjustment': optimal_bonus - current_bonus
            }
        
        # 3. Big trip jackpot optimization
        print("\n3. Big trip jackpot optimization:")
        jackpot_mask = ((self.df['trip_duration_days'] >= 8) &
                       (self.df['miles_traveled'] >= 900) &
                       (self.df['total_receipts_amount'] >= 1200))
        non_jackpot_mask = ~jackpot_mask
        
        if jackpot_mask.sum() > 0 and non_jackpot_mask.sum() > 0:
            jackpot_residual = self.df[jackpot_mask]['residual'].mean()
            non_jackpot_residual = self.df[non_jackpot_mask]['residual'].mean()
            optimal_bonus = jackpot_residual - non_jackpot_residual
            
            current_bonus = 46.67
            print(f"  Current bonus: ${current_bonus:.2f}")
            print(f"  Optimal bonus: ${optimal_bonus:.2f}")
            print(f"  Adjustment: ${optimal_bonus - current_bonus:.2f}")
            
            bonus_optimizations['big_trip_jackpot'] = {
                'original': current_bonus,
                'optimized': optimal_bonus,
                'adjustment': optimal_bonus - current_bonus
            }
        
        # 4. Efficiency bonus optimizations
        print("\n4. Efficiency bonus optimizations:")
        
        efficiency_categories = [
            ('medium_efficiency', 100, 200, 47.59),
            ('high_efficiency', 200, 300, 33.17),
            ('low_efficiency_penalty', 0, 100, -23.66)
        ]
        
        for cat_name, min_eff, max_eff, current_adjustment in efficiency_categories:
            if max_eff == 300:
                cat_mask = (self.df['miles_per_day'] >= min_eff) & (self.df['miles_per_day'] < 300)
            elif min_eff == 0:
                cat_mask = self.df['miles_per_day'] < max_eff
            else:
                cat_mask = (self.df['miles_per_day'] >= min_eff) & (self.df['miles_per_day'] < max_eff)
            
            other_mask = ~cat_mask
            
            if cat_mask.sum() > 50 and other_mask.sum() > 50:
                cat_residual = self.df[cat_mask]['residual'].mean()
                other_residual = self.df[other_mask]['residual'].mean()
                optimal_adjustment = cat_residual - other_residual
                
                print(f"  {cat_name}: ${current_adjustment:.2f} → ${optimal_adjustment:.2f} (Δ${optimal_adjustment - current_adjustment:.2f})")
                
                bonus_optimizations[cat_name] = {
                    'original': current_adjustment,
                    'optimized': optimal_adjustment,
                    'adjustment': optimal_adjustment - current_adjustment
                }
        
        self.optimized_parameters['bonus_amounts'] = bonus_optimizations
        
    def implement_tier_system_refinements(self):
        """Implement tier system refinements from complex rule discovery"""
        print("\n=== IMPLEMENTING TIER SYSTEM REFINEMENTS ===")
        
        tier_adjustments = {}
        
        if 'tier_systems' in self.complex_rules:
            tier_systems = self.complex_rules['tier_systems']
            
            # 1. Miles tier system
            if 'miles_tier_system' in tier_systems:
                print("\n1. Implementing miles tier system:")
                miles_tiers = tier_systems['miles_tier_system']
                
                self.df['miles_tier_adjustment'] = 0
                
                for tier_name, tier_data in miles_tiers.items():
                    min_miles, max_miles = tier_data['range']
                    effect = tier_data['effect']
                    
                    if max_miles == float('inf'):
                        mask = self.df['miles_traveled'] >= min_miles
                    else:
                        mask = (self.df['miles_traveled'] >= min_miles) & (self.df['miles_traveled'] < max_miles)
                    
                    self.df.loc[mask, 'miles_tier_adjustment'] += effect
                    
                    print(f"  {tier_name}: {mask.sum()} trips, ${effect:.2f} adjustment")
                
                tier_adjustments['miles_tier_system'] = miles_tiers
            
            # 2. Receipt tier system
            if 'receipt_tier_system' in tier_systems:
                print("\n2. Implementing receipt tier system:")
                receipt_tiers = tier_systems['receipt_tier_system']
                
                self.df['receipt_tier_adjustment'] = 0
                
                for tier_name, tier_data in receipt_tiers.items():
                    min_receipts, max_receipts = tier_data['range']
                    effect = tier_data['effect']
                    
                    if max_receipts == float('inf'):
                        mask = self.df['total_receipts_amount'] >= min_receipts
                    else:
                        mask = (self.df['total_receipts_amount'] >= min_receipts) & (self.df['total_receipts_amount'] < max_receipts)
                    
                    self.df.loc[mask, 'receipt_tier_adjustment'] += effect
                    
                    print(f"  {tier_name}: {mask.sum()} trips, ${effect:.2f} adjustment")
                
                tier_adjustments['receipt_tier_system'] = receipt_tiers
            
            # 3. Days tier system
            if 'days_tier_system' in tier_systems:
                print("\n3. Implementing days tier system:")
                days_tiers = tier_systems['days_tier_system']
                
                self.df['days_tier_adjustment'] = 0
                
                for tier_name, tier_data in days_tiers.items():
                    days = tier_data['days']
                    effect = tier_data['effect']
                    
                    mask = self.df['trip_duration_days'] == days
                    self.df.loc[mask, 'days_tier_adjustment'] += effect
                    
                    print(f"  {days} days: {mask.sum()} trips, ${effect:.2f} adjustment")
                
                tier_adjustments['days_tier_system'] = days_tiers
        
        self.optimized_parameters['tier_systems'] = tier_adjustments
        
    def test_precision_optimized_system(self):
        """Test the complete precision-optimized system"""
        print("\n=== TESTING PRECISION-OPTIMIZED SYSTEM ===")
        
        # Start with optimized linear baseline
        if 'linear_coefficients' in self.optimized_parameters:
            base_pred = self.df['optimized_linear_baseline']
        else:
            base_pred = self.df['linear_baseline']
        
        # Add optimized business rule adjustments
        total_adjustment = self.df['business_rule_adjustment'].copy()
        
        # Apply optimized bonus amounts
        if 'bonus_amounts' in self.optimized_parameters:
            bonus_opts = self.optimized_parameters['bonus_amounts']
            
            # Adjust sweet spot bonus
            if 'sweet_spot' in bonus_opts:
                mask = self.df['trip_duration_days'].isin([5, 6])
                adjustment_diff = bonus_opts['sweet_spot']['adjustment']
                total_adjustment.loc[mask] += adjustment_diff
            
            # Adjust distance bonus
            if 'distance_bonus' in bonus_opts:
                mask = self.df['miles_traveled'] >= 600
                adjustment_diff = bonus_opts['distance_bonus']['adjustment']
                total_adjustment.loc[mask] += adjustment_diff
            
            # Adjust efficiency bonuses
            for eff_type in ['medium_efficiency', 'high_efficiency', 'low_efficiency_penalty']:
                if eff_type in bonus_opts:
                    adjustment_diff = bonus_opts[eff_type]['adjustment']
                    
                    if eff_type == 'medium_efficiency':
                        mask = (self.df['miles_per_day'] >= 100) & (self.df['miles_per_day'] < 200)
                    elif eff_type == 'high_efficiency':
                        mask = (self.df['miles_per_day'] >= 200) & (self.df['miles_per_day'] < 300)
                    else:  # low_efficiency_penalty
                        mask = self.df['miles_per_day'] < 100
                    
                    total_adjustment.loc[mask] += adjustment_diff
        
        # Add tier system adjustments
        if hasattr(self.df, 'miles_tier_adjustment'):
            total_adjustment += self.df['miles_tier_adjustment']
        if hasattr(self.df, 'receipt_tier_adjustment'):
            total_adjustment += self.df['receipt_tier_adjustment'] 
        if hasattr(self.df, 'days_tier_adjustment'):
            total_adjustment += self.df['days_tier_adjustment']
        
        # Calculate final precision-optimized predictions
        self.df['precision_prediction'] = base_pred + total_adjustment
        self.df['precision_prediction'] = (self.df['precision_prediction'] * 4).round() / 4  # Quarter rounding
        
        # Calculate performance metrics
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        precision_r2 = r2_score(self.df['reimbursement'], self.df['precision_prediction'])
        
        current_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction']))
        precision_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['precision_prediction']))
        
        current_mae = mean_absolute_error(self.df['reimbursement'], self.df['current_prediction'])
        precision_mae = mean_absolute_error(self.df['reimbursement'], self.df['precision_prediction'])
        
        # Exact match analysis
        exact_matches = (np.abs(self.df['reimbursement'] - self.df['precision_prediction']) < 0.01).sum()
        exact_match_rate = exact_matches / len(self.df) * 100
        
        within_1_dollar = (np.abs(self.df['reimbursement'] - self.df['precision_prediction']) <= 1.0).sum()
        within_5_dollars = (np.abs(self.df['reimbursement'] - self.df['precision_prediction']) <= 5.0).sum()
        within_10_dollars = (np.abs(self.df['reimbursement'] - self.df['precision_prediction']) <= 10.0).sum()
        
        print(f"\n=== PRECISION OPTIMIZATION RESULTS ===")
        print(f"Current system:    R²={current_r2:.6f}, RMSE=${current_rmse:.2f}, MAE=${current_mae:.2f}")
        print(f"Precision system:  R²={precision_r2:.6f}, RMSE=${precision_rmse:.2f}, MAE=${precision_mae:.2f}")
        print(f"Improvement:       ΔR²={precision_r2-current_r2:.6f}, ΔRMSE=${current_rmse-precision_rmse:.2f}, ΔMAE=${current_mae-precision_mae:.2f}")
        
        print(f"\nAccuracy Analysis:")
        print(f"  Exact matches: {exact_matches} ({exact_match_rate:.2f}%)")
        print(f"  Within $1: {within_1_dollar} ({within_1_dollar/len(self.df)*100:.1f}%)")
        print(f"  Within $5: {within_5_dollars} ({within_5_dollars/len(self.df)*100:.1f}%)")
        print(f"  Within $10: {within_10_dollars} ({within_10_dollars/len(self.df)*100:.1f}%)")
        
        target_r2 = 0.95
        if precision_r2 >= target_r2:
            print(f"\n🎯 SUCCESS! Achieved {precision_r2:.4f} R² (≥95% target)")
        else:
            remaining_gap = target_r2 - precision_r2
            print(f"\n⚠️  Still {remaining_gap:.4f} gap remaining to reach 95% target")
        
        return {
            'precision_r2': precision_r2,
            'precision_rmse': precision_rmse,
            'precision_mae': precision_mae,
            'improvement': precision_r2 - current_r2,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_match_rate,
            'within_tolerances': {
                'within_1_dollar': within_1_dollar,
                'within_5_dollars': within_5_dollars,
                'within_10_dollars': within_10_dollars
            }
        }
        
    def run_precision_parameter_tuning(self):
        """Run complete precision parameter tuning process"""
        print("🔍 STARTING PRECISION PARAMETER TUNING FOR 95%+ ACCURACY")
        print("=" * 70)
        
        self.optimize_linear_coefficients_precision()
        self.optimize_business_rule_thresholds()
        self.optimize_bonus_amounts_precision()
        self.implement_tier_system_refinements()
        results = self.test_precision_optimized_system()
        
        print("\n" + "=" * 70)
        print("🎯 PRECISION PARAMETER TUNING COMPLETE")
        
        return {
            'optimized_parameters': self.optimized_parameters,
            'performance_results': results,
            'final_r2': results['precision_r2'],
            'target_achieved': results['precision_r2'] >= 0.95
        }
        
    def save_precision_optimization(self, output_path='precision_optimization_results.json'):
        """Save precision optimization results"""
        results = {
            'optimized_parameters': self.optimized_parameters,
            'precision_improvements': self.precision_improvements,
            'optimization_summary': {
                'total_parameter_categories': len(self.optimized_parameters),
                'final_performance': {
                    'r2': float(r2_score(self.df['reimbursement'], self.df['precision_prediction'])),
                    'rmse': float(np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['precision_prediction']))),
                    'mae': float(mean_absolute_error(self.df['reimbursement'], self.df['precision_prediction']))
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Precision optimization results saved to {output_path}")

if __name__ == "__main__":
    # Run precision parameter tuning
    tuner = PrecisionParameterTuner()
    results = tuner.run_precision_parameter_tuning()
    tuner.save_precision_optimization()
    
    print(f"\n📋 PRECISION TUNING SUMMARY:")
    print(f"Final R²: {results['final_r2']:.6f}")
    print(f"Target achieved: {results['target_achieved']}")
    print(f"Optimization categories: {len(results['optimized_parameters'])}")
    
    if results['target_achieved']:
        print("🎉 SUCCESS: 95%+ accuracy achieved!")
    else:
        remaining = 0.95 - results['final_r2']
        print(f"⚠️  Remaining gap: {remaining:.4f} to reach 95%")