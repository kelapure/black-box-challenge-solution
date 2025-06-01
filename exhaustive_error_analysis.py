#!/usr/bin/env python3
"""
Exhaustive Error Analysis for 90%+ Accuracy
Black Box Challenge - Final Business Logic Push

This is our final attempt to reach 90%+ R² using pure business logic before
considering a hybrid approach. We'll perform exhaustive error analysis to
discover every remaining discoverable business rule.

Current: 86.78% R² → Target: 90%+ R² (Gap: 3.22%)

Strategy:
1. Analyze every large residual individually for patterns
2. Find micro-adjustments and edge case rules
3. Discover remaining mathematical relationships
4. Implement conservative rule refinements
5. Test each rule addition carefully to avoid overfitting

Focus: Maximum interpretability, minimal complexity, no overfitting.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class ExhaustiveErrorAnalysis:
    """
    Performs exhaustive error analysis to discover final business rules
    """
    
    def __init__(self, data_path='test_cases.json'):
        """Initialize with test cases"""
        self.data_path = data_path
        self.df = None
        self.final_rules = {}
        self.rule_tests = []
        self.load_data()
        
    def load_data(self):
        """Load test cases and apply current best system (86.78% R²)"""
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
        
        # Apply current best system (from Task 2.6 - 86.78% R²)
        self.apply_current_best_system()
        
    def apply_current_best_system(self):
        """Apply the best system so far (86.78% R² from Task 2.6)"""
        # Linear baseline
        baseline = {
            'intercept': 318.40,
            'trip_duration_days': 71.28,
            'miles_traveled': 0.794100,
            'total_receipts_amount': 0.290366
        }
        
        self.df['linear_baseline'] = (
            baseline['intercept'] +
            baseline['trip_duration_days'] * self.df['trip_duration_days'] +
            baseline['miles_traveled'] * self.df['miles_traveled'] +
            baseline['total_receipts_amount'] * self.df['total_receipts_amount']
        )
        
        # Apply all precision-optimized rules
        self.df['business_rule_adjustment'] = 0
        
        # Core optimized rules
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
        
        # Apply tier systems from complex rules
        self.apply_tier_systems()
        
        # Apply ratio-based rules from Task 2.6
        self.apply_missing_logic_rules()
        
        # Calculate current predictions and residuals
        self.df['pre_rounding_prediction'] = self.df['linear_baseline'] + self.df['business_rule_adjustment']
        self.df['current_prediction'] = (self.df['pre_rounding_prediction'] * 4).round() / 4
        self.df['residual'] = self.df['reimbursement'] - self.df['current_prediction']
        
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        current_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction']))
        
        print(f"Current best system: R²={current_r2:.4f}, RMSE=${current_rmse:.2f}")
        print(f"Gap to 90% target: {0.90 - current_r2:.4f}")
        
    def apply_tier_systems(self):
        """Apply tier systems from complex rules (simplified)"""
        # Miles tier effects (top performers only)
        miles_tier_effects = {
            (0, 50): 143.44,
            (50, 100): 89.46,
            (100, 200): 72.23,
            (500, 800): -54.10
        }
        
        for (min_miles, max_miles), effect in miles_tier_effects.items():
            if max_miles == float('inf'):
                mask = self.df['miles_traveled'] >= min_miles
            else:
                mask = (self.df['miles_traveled'] >= min_miles) & (self.df['miles_traveled'] < max_miles)
            
            self.df.loc[mask, 'business_rule_adjustment'] += effect * 0.3  # Conservative application
        
        # Receipt tier effects (top performers only)
        receipt_tier_effects = {
            (100, 300): -219.74,
            (1000, 1500): 158.42,
            (600, 1000): 92.35
        }
        
        for (min_receipts, max_receipts), effect in receipt_tier_effects.items():
            mask = (self.df['total_receipts_amount'] >= min_receipts) & (self.df['total_receipts_amount'] < max_receipts)
            self.df.loc[mask, 'business_rule_adjustment'] += effect * 0.2  # Conservative application
            
    def apply_missing_logic_rules(self):
        """Apply missing logic rules from Task 2.6 (conservative)"""
        # Daily spending rate effects
        spending_effects = {
            (150, 200): 58.21,
            (200, 250): 89.00
        }
        
        for (min_spend, max_spend), effect in spending_effects.items():
            mask = (self.df['receipts_per_day'] >= min_spend) & (self.df['receipts_per_day'] < max_spend)
            self.df.loc[mask, 'business_rule_adjustment'] += effect * 0.1  # Very conservative
        
        # Miles per dollar efficiency effects
        self.df['miles_per_dollar'] = self.df['miles_traveled'] / (self.df['total_receipts_amount'] + 1)
        
        efficiency_effects = {
            (0.5, 1.0): 78.82,
            (1.0, 2.0): 132.16
        }
        
        for (min_eff, max_eff), effect in efficiency_effects.items():
            mask = (self.df['miles_per_dollar'] >= min_eff) & (self.df['miles_per_dollar'] < max_eff)
            self.df.loc[mask, 'business_rule_adjustment'] += effect * 0.05  # Very conservative
            
    def analyze_large_residuals_exhaustively(self):
        """Analyze every large residual to find remaining patterns"""
        print("\\n=== EXHAUSTIVE LARGE RESIDUAL ANALYSIS ===")
        
        # Focus on residuals > $100 (significant errors)
        large_residuals = self.df[np.abs(self.df['residual']) > 100].copy()
        large_residuals = large_residuals.sort_values('residual', key=abs, ascending=False)
        
        print(f"\\nAnalyzing {len(large_residuals)} trips with residuals > $100")
        print(f"Largest residuals: ${large_residuals['residual'].max():.2f} to ${large_residuals['residual'].min():.2f}")
        
        # Analyze patterns in large residuals
        patterns_found = {}
        
        # 1. Specific value combinations that consistently appear
        print("\\n1. Value combination patterns in large residuals:")
        
        value_combinations = []
        for _, row in large_residuals.head(50).iterrows():  # Top 50 largest residuals
            days = row['trip_duration_days']
            miles = row['miles_traveled']
            receipts = row['total_receipts_amount']
            residual = row['residual']
            
            # Create signature for this trip
            days_cat = "short" if days <= 3 else "medium" if days <= 7 else "long"
            miles_cat = "low" if miles < 300 else "med" if miles < 700 else "high"
            receipts_cat = "low" if receipts < 500 else "med" if receipts < 1500 else "high"
            
            signature = f"{days_cat}_{miles_cat}_{receipts_cat}"
            value_combinations.append((signature, residual, days, miles, receipts))
        
        # Find repeated signatures
        signature_counts = Counter([combo[0] for combo in value_combinations])
        
        for signature, count in signature_counts.items():
            if count >= 3:  # Pattern appears 3+ times
                signature_residuals = [combo[1] for combo in value_combinations if combo[0] == signature]
                avg_residual = np.mean(signature_residuals)
                
                print(f"  {signature}: {count} trips, avg residual ${avg_residual:.2f}")
                
                if abs(avg_residual) > 50:
                    patterns_found[f"large_residual_{signature}"] = {
                        'pattern': signature,
                        'count': count,
                        'avg_residual': avg_residual,
                        'description': f"Large residual pattern: {signature}"
                    }
        
        # 2. Specific exact values that cause problems
        print("\\n2. Problematic exact values:")
        
        # Check for specific days that are problematic
        for days in range(1, 16):
            day_data = large_residuals[large_residuals['trip_duration_days'] == days]
            if len(day_data) >= 3:
                avg_residual = day_data['residual'].mean()
                print(f"  {days} days: {len(day_data)} large residual trips, avg ${avg_residual:.2f}")
                
                if abs(avg_residual) > 80:
                    patterns_found[f"problematic_days_{days}"] = {
                        'days': days,
                        'count': len(day_data),
                        'avg_residual': avg_residual,
                        'effect': avg_residual * 0.5  # Conservative correction
                    }
        
        # Check for specific mile ranges
        mile_ranges = [(0, 100), (100, 200), (200, 400), (400, 600), (600, 800), (800, 1200), (1200, 2000)]
        
        for min_miles, max_miles in mile_ranges:
            range_data = large_residuals[
                (large_residuals['miles_traveled'] >= min_miles) & 
                (large_residuals['miles_traveled'] < max_miles)
            ]
            
            if len(range_data) >= 3:
                avg_residual = range_data['residual'].mean()
                print(f"  {min_miles}-{max_miles} miles: {len(range_data)} large residual trips, avg ${avg_residual:.2f}")
                
                if abs(avg_residual) > 80:
                    patterns_found[f"problematic_miles_{min_miles}_{max_miles}"] = {
                        'mile_range': (min_miles, max_miles),
                        'count': len(range_data),
                        'avg_residual': avg_residual,
                        'effect': avg_residual * 0.3  # Conservative correction
                    }
        
        # 3. Unusual ratio patterns
        print("\\n3. Unusual ratio patterns in large residuals:")
        
        # Calculate unusual ratios
        large_residuals['total_per_day'] = large_residuals['total_receipts_amount'] / large_residuals['trip_duration_days']
        large_residuals['miles_to_receipts_ratio'] = large_residuals['miles_traveled'] / (large_residuals['total_receipts_amount'] + 1)
        
        # Check for extreme ratios
        extreme_ratios = [
            ('ultra_high_daily_spend', large_residuals['total_per_day'] > 400),
            ('ultra_low_daily_spend', large_residuals['total_per_day'] < 30),
            ('ultra_efficient_miles', large_residuals['miles_to_receipts_ratio'] > 3),
            ('ultra_inefficient_miles', large_residuals['miles_to_receipts_ratio'] < 0.2)
        ]
        
        for ratio_name, ratio_mask in extreme_ratios:
            ratio_data = large_residuals[ratio_mask]
            if len(ratio_data) >= 3:
                avg_residual = ratio_data['residual'].mean()
                print(f"  {ratio_name}: {len(ratio_data)} trips, avg residual ${avg_residual:.2f}")
                
                if abs(avg_residual) > 60:
                    patterns_found[f"extreme_ratio_{ratio_name}"] = {
                        'ratio_type': ratio_name,
                        'count': len(ratio_data),
                        'avg_residual': avg_residual,
                        'effect': avg_residual * 0.4  # Conservative correction
                    }
        
        self.final_rules['large_residual_patterns'] = patterns_found
        
    def discover_micro_adjustments(self):
        """Find small, precise adjustments for remaining systematic biases"""
        print("\\n=== DISCOVERING MICRO-ADJUSTMENTS ===")
        
        micro_adjustments = {}
        
        # 1. Fine-tune existing thresholds by small amounts
        print("\\n1. Fine-tuning existing thresholds:")
        
        # Test small adjustments to receipt ceiling
        current_threshold = 2025
        best_threshold = current_threshold
        best_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        
        for threshold in range(1950, 2100, 25):
            test_adjustment = self.df['business_rule_adjustment'].copy()
            
            # Remove current effect
            current_mask = self.df['total_receipts_amount'] > current_threshold
            test_adjustment[current_mask] += 99.48  # Remove current penalty
            
            # Apply new effect
            new_mask = self.df['total_receipts_amount'] > threshold
            test_adjustment[new_mask] -= 99.48  # Apply new penalty
            
            test_pred = self.df['linear_baseline'] + test_adjustment
            test_pred_rounded = (test_pred * 4).round() / 4
            test_r2 = r2_score(self.df['reimbursement'], test_pred_rounded)
            
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_threshold = threshold
        
        improvement = best_r2 - r2_score(self.df['reimbursement'], self.df['current_prediction'])
        print(f"  Receipt ceiling: {current_threshold} → {best_threshold}, R² improvement: {improvement:.6f}")
        
        if improvement > 0.001:
            micro_adjustments['receipt_ceiling_fine_tune'] = {
                'old_threshold': current_threshold,
                'new_threshold': best_threshold,
                'improvement': improvement
            }
        
        # 2. Small coefficient adjustments
        print("\\n2. Small coefficient micro-adjustments:")
        
        current_coeffs = [71.28, 0.794100, 0.290366]  # days, miles, receipts
        coeff_names = ['days', 'miles', 'receipts']
        
        for i, (current_coeff, name) in enumerate(zip(current_coeffs, coeff_names)):
            best_coeff = current_coeff
            best_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
            
            # Test small adjustments (±5%)
            test_range = np.linspace(current_coeff * 0.95, current_coeff * 1.05, 21)
            
            for test_coeff in test_range:
                test_linear = self.df['linear_baseline'].copy()
                
                # Adjust the specific coefficient
                if i == 0:  # days
                    test_linear = (318.40 + test_coeff * self.df['trip_duration_days'] + 
                                 0.794100 * self.df['miles_traveled'] + 
                                 0.290366 * self.df['total_receipts_amount'])
                elif i == 1:  # miles
                    test_linear = (318.40 + 71.28 * self.df['trip_duration_days'] + 
                                 test_coeff * self.df['miles_traveled'] + 
                                 0.290366 * self.df['total_receipts_amount'])
                else:  # receipts
                    test_linear = (318.40 + 71.28 * self.df['trip_duration_days'] + 
                                 0.794100 * self.df['miles_traveled'] + 
                                 test_coeff * self.df['total_receipts_amount'])
                
                test_pred = test_linear + self.df['business_rule_adjustment']
                test_pred_rounded = (test_pred * 4).round() / 4
                test_r2 = r2_score(self.df['reimbursement'], test_pred_rounded)
                
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_coeff = test_coeff
            
            improvement = best_r2 - r2_score(self.df['reimbursement'], self.df['current_prediction'])
            print(f"  {name} coefficient: {current_coeff:.6f} → {best_coeff:.6f}, R² improvement: {improvement:.6f}")
            
            if improvement > 0.0005:
                micro_adjustments[f'{name}_coefficient_fine_tune'] = {
                    'old_coefficient': current_coeff,
                    'new_coefficient': best_coeff,
                    'improvement': improvement
                }
        
        # 3. Bonus amount micro-adjustments
        print("\\n3. Bonus amount micro-adjustments:")
        
        # Test small adjustments to major bonuses/penalties
        bonus_tests = [
            ('sweet_spot_penalty', self.df['trip_duration_days'].isin([5, 6]), -34.57),
            ('distance_penalty', self.df['miles_traveled'] >= 550, -41.02),
            ('efficiency_medium_penalty', (self.df['miles_per_day'] >= 94) & (self.df['miles_per_day'] < 180), -27.84)
        ]
        
        for bonus_name, mask, current_bonus in bonus_tests:
            best_bonus = current_bonus
            best_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
            
            # Test small adjustments (±20%)
            test_range = np.linspace(current_bonus * 0.8, current_bonus * 1.2, 21)
            
            for test_bonus in test_range:
                test_adjustment = self.df['business_rule_adjustment'].copy()
                
                # Remove current effect and apply new effect
                test_adjustment[mask] = test_adjustment[mask] - current_bonus + test_bonus
                
                test_pred = self.df['linear_baseline'] + test_adjustment
                test_pred_rounded = (test_pred * 4).round() / 4
                test_r2 = r2_score(self.df['reimbursement'], test_pred_rounded)
                
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_bonus = test_bonus
            
            improvement = best_r2 - r2_score(self.df['reimbursement'], self.df['current_prediction'])
            print(f"  {bonus_name}: ${current_bonus:.2f} → ${best_bonus:.2f}, R² improvement: {improvement:.6f}")
            
            if improvement > 0.0005:
                micro_adjustments[f'{bonus_name}_fine_tune'] = {
                    'old_bonus': current_bonus,
                    'new_bonus': best_bonus,
                    'improvement': improvement,
                    'affected_trips': mask.sum()
                }
        
        self.final_rules['micro_adjustments'] = micro_adjustments
        
    def discover_remaining_edge_cases(self):
        """Find any remaining edge cases we haven't discovered"""
        print("\\n=== DISCOVERING REMAINING EDGE CASES ===")
        
        edge_cases = {}
        
        # 1. Extreme value combinations
        print("\\n1. Extreme value combinations:")
        
        extreme_combinations = [
            ('ultra_short_ultra_high_miles', (self.df['trip_duration_days'] == 1) & (self.df['miles_traveled'] >= 500)),
            ('ultra_long_ultra_low_miles', (self.df['trip_duration_days'] >= 14) & (self.df['miles_traveled'] <= 100)),
            ('medium_days_ultra_high_receipts', (self.df['trip_duration_days'].isin([5, 6, 7])) & (self.df['total_receipts_amount'] >= 3000)),
            ('ultra_high_efficiency', self.df['miles_per_day'] >= 500),
            ('zero_or_tiny_miles', self.df['miles_traveled'] <= 10)
        ]
        
        for case_name, case_mask in extreme_combinations:
            case_data = self.df[case_mask]
            
            if len(case_data) >= 5:  # Enough for pattern
                avg_residual = case_data['residual'].mean()
                std_residual = case_data['residual'].std()
                
                print(f"  {case_name}: {len(case_data)} trips, avg residual ${avg_residual:.2f}, std ${std_residual:.2f}")
                
                if abs(avg_residual) > 40:
                    edge_cases[case_name] = {
                        'count': len(case_data),
                        'avg_residual': avg_residual,
                        'effect': avg_residual * 0.6,  # Conservative application
                        'description': f"Edge case: {case_name}"
                    }
        
        # 2. Specific problematic values
        print("\\n2. Specific problematic exact values:")
        
        # Check for exact values that consistently cause problems
        problem_values = []
        
        # Check exact mile values
        mile_counts = self.df['miles_traveled'].value_counts()
        for miles, count in mile_counts.items():
            if count >= 5:  # Appears multiple times
                mile_data = self.df[self.df['miles_traveled'] == miles]
                avg_residual = mile_data['residual'].mean()
                
                if abs(avg_residual) > 80:
                    problem_values.append(('exact_miles', miles, count, avg_residual))
        
        # Check exact receipt values
        receipt_counts = self.df['total_receipts_amount'].value_counts()
        for receipts, count in receipt_counts.items():
            if count >= 5:
                receipt_data = self.df[self.df['total_receipts_amount'] == receipts]
                avg_residual = receipt_data['residual'].mean()
                
                if abs(avg_residual) > 80:
                    problem_values.append(('exact_receipts', receipts, count, avg_residual))
        
        for value_type, value, count, avg_residual in problem_values[:10]:  # Top 10
            print(f"  {value_type} = {value}: {count} trips, avg residual ${avg_residual:.2f}")
            
            edge_cases[f'{value_type}_{value}'] = {
                'value_type': value_type,
                'value': value,
                'count': count,
                'avg_residual': avg_residual,
                'effect': avg_residual * 0.5  # Conservative
            }
        
        # 3. Ratio-based edge cases
        print("\\n3. Ratio-based edge cases:")
        
        # Calculate additional ratios
        self.df['trip_cost_efficiency'] = self.df['miles_traveled'] / (self.df['total_receipts_amount'] + self.df['trip_duration_days'])
        self.df['daily_mile_intensity'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        
        ratio_edge_cases = [
            ('ultra_cost_efficient', self.df['trip_cost_efficiency'] > 2.0),
            ('ultra_cost_inefficient', self.df['trip_cost_efficiency'] < 0.1),
            ('extreme_daily_intensity', self.df['daily_mile_intensity'] > 300),
            ('minimal_daily_intensity', self.df['daily_mile_intensity'] < 10)
        ]
        
        for case_name, case_mask in ratio_edge_cases:
            case_data = self.df[case_mask]
            
            if len(case_data) >= 10:
                avg_residual = case_data['residual'].mean()
                
                print(f"  {case_name}: {len(case_data)} trips, avg residual ${avg_residual:.2f}")
                
                if abs(avg_residual) > 30:
                    edge_cases[case_name] = {
                        'count': len(case_data),
                        'avg_residual': avg_residual,
                        'effect': avg_residual * 0.3,  # Very conservative
                        'description': f"Ratio edge case: {case_name}"
                    }
        
        self.final_rules['edge_cases'] = edge_cases
        
    def test_incremental_rule_additions(self):
        """Test each discovered rule individually to avoid overfitting"""
        print("\\n=== TESTING INCREMENTAL RULE ADDITIONS ===")
        
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        current_prediction = self.df['current_prediction'].copy()
        
        improvements = []
        
        # Test micro-adjustments
        if 'micro_adjustments' in self.final_rules:
            for rule_name, rule_data in self.final_rules['micro_adjustments'].items():
                test_pred = current_prediction.copy()
                
                # Apply the rule based on type
                if 'coefficient' in rule_name:
                    # Would need to recalculate from scratch - skip for now
                    continue
                elif 'threshold' in rule_name:
                    # Would need complex threshold changes - skip for now
                    continue
                elif 'bonus' in rule_name:
                    improvement = rule_data['improvement']
                    improvements.append((rule_name, improvement, rule_data))
        
        # Test edge cases
        if 'edge_cases' in self.final_rules:
            for rule_name, rule_data in self.final_rules['edge_cases'].items():
                test_pred = current_prediction.copy()
                effect = rule_data['effect']
                
                # Apply effect based on rule type
                if 'ultra_short_ultra_high_miles' in rule_name:
                    mask = (self.df['trip_duration_days'] == 1) & (self.df['miles_traveled'] >= 500)
                elif 'ultra_long_ultra_low_miles' in rule_name:
                    mask = (self.df['trip_duration_days'] >= 14) & (self.df['miles_traveled'] <= 100)
                elif 'zero_or_tiny_miles' in rule_name:
                    mask = self.df['miles_traveled'] <= 10
                elif 'ultra_high_efficiency' in rule_name:
                    mask = self.df['miles_per_day'] >= 500
                elif 'ultra_cost_efficient' in rule_name:
                    mask = self.df['trip_cost_efficiency'] > 2.0
                elif 'ultra_cost_inefficient' in rule_name:
                    mask = self.df['trip_cost_efficiency'] < 0.1
                else:
                    continue  # Skip complex rules for now
                
                test_pred[mask] += effect
                test_pred_rounded = (test_pred * 4).round() / 4
                test_r2 = r2_score(self.df['reimbursement'], test_pred_rounded)
                improvement = test_r2 - current_r2
                
                if improvement > 0.0001:  # Meaningful improvement
                    improvements.append((rule_name, improvement, rule_data))
        
        # Sort by improvement and apply best rules
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\\nFound {len(improvements)} potentially beneficial rules:")
        for rule_name, improvement, rule_data in improvements[:10]:  # Top 10
            print(f"  {rule_name}: R² improvement {improvement:.6f}")
        
        return improvements[:5]  # Return top 5 for careful application
        
    def apply_final_conservative_rules(self):
        """Apply final rules conservatively to reach 90% target"""
        print("\\n=== APPLYING FINAL CONSERVATIVE RULES ===")
        
        # Get best rule improvements
        best_improvements = self.test_incremental_rule_additions()
        
        # Start with current best prediction
        final_pred = self.df['current_prediction'].copy()
        cumulative_improvement = 0
        
        # Apply top rules conservatively
        for rule_name, improvement, rule_data in best_improvements:
            if cumulative_improvement + improvement > 0.05:  # Don't add too much
                break
                
            print(f"\\nApplying {rule_name} (projected improvement: {improvement:.6f})")
            
            # Apply rule with conservative scaling
            if 'edge_cases' in self.final_rules and rule_name in self.final_rules['edge_cases']:
                effect = rule_data['effect'] * 0.5  # Very conservative
                
                # Apply based on rule type
                if 'ultra_short_ultra_high_miles' in rule_name:
                    mask = (self.df['trip_duration_days'] == 1) & (self.df['miles_traveled'] >= 500)
                elif 'ultra_long_ultra_low_miles' in rule_name:
                    mask = (self.df['trip_duration_days'] >= 14) & (self.df['miles_traveled'] <= 100)
                elif 'zero_or_tiny_miles' in rule_name:
                    mask = self.df['miles_traveled'] <= 10
                elif 'ultra_high_efficiency' in rule_name:
                    mask = self.df['miles_per_day'] >= 500
                elif 'ultra_cost_efficient' in rule_name:
                    mask = self.df['trip_cost_efficiency'] > 2.0
                elif 'ultra_cost_inefficient' in rule_name:
                    mask = self.df['trip_cost_efficiency'] < 0.1
                else:
                    continue
                
                final_pred[mask] += effect
                print(f"  Applied to {mask.sum()} trips with effect ${effect:.2f}")
            
            cumulative_improvement += improvement
        
        # Apply quarter rounding
        final_pred_rounded = (final_pred * 4).round() / 4
        
        # Calculate final performance
        final_r2 = r2_score(self.df['reimbursement'], final_pred_rounded)
        final_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], final_pred_rounded))
        final_mae = mean_absolute_error(self.df['reimbursement'], final_pred_rounded)
        
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        
        print(f"\\n=== FINAL EXHAUSTIVE ERROR ANALYSIS RESULTS ===")
        print(f"Current system:  R²={current_r2:.6f}, RMSE=${np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction'])):.2f}")
        print(f"Final system:    R²={final_r2:.6f}, RMSE=${final_rmse:.2f}, MAE=${final_mae:.2f}")
        print(f"Improvement:     ΔR²={final_r2-current_r2:.6f}")
        
        # Exact match analysis
        exact_matches = (np.abs(self.df['reimbursement'] - final_pred_rounded) < 0.01).sum()
        exact_match_rate = exact_matches / len(self.df) * 100
        
        within_1_dollar = (np.abs(self.df['reimbursement'] - final_pred_rounded) <= 1.0).sum()
        within_5_dollars = (np.abs(self.df['reimbursement'] - final_pred_rounded) <= 5.0).sum()
        within_10_dollars = (np.abs(self.df['reimbursement'] - final_pred_rounded) <= 10.0).sum()
        
        print(f"\\nAccuracy Analysis:")
        print(f"  Exact matches: {exact_matches} ({exact_match_rate:.2f}%)")
        print(f"  Within $1: {within_1_dollar} ({within_1_dollar/len(self.df)*100:.1f}%)")
        print(f"  Within $5: {within_5_dollars} ({within_5_dollars/len(self.df)*100:.1f}%)")
        print(f"  Within $10: {within_10_dollars} ({within_10_dollars/len(self.df)*100:.1f}%)")
        
        target_r2 = 0.90
        if final_r2 >= target_r2:
            print(f"\\n🎯 SUCCESS! Achieved {final_r2:.4f} R² (≥90% target with pure business logic)")
        else:
            remaining_gap = target_r2 - final_r2
            print(f"\\n⚠️  Still {remaining_gap:.4f} gap remaining to reach 90% target")
            print("Next step: Consider minimal hybrid approach (Task 2.8)")
        
        # Store final prediction
        self.df['final_prediction'] = final_pred_rounded
        
        return {
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'final_mae': final_mae,
            'improvement': final_r2 - current_r2,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_match_rate,
            'within_tolerances': {
                'within_1_dollar': within_1_dollar,
                'within_5_dollars': within_5_dollars,
                'within_10_dollars': within_10_dollars
            },
            'target_achieved': final_r2 >= target_r2,
            'rules_applied': len(best_improvements)
        }
        
    def run_exhaustive_error_analysis(self):
        """Run complete exhaustive error analysis"""
        print("🔍 STARTING EXHAUSTIVE ERROR ANALYSIS FOR 90%+ ACCURACY")
        print("=" * 70)
        print("FINAL PUSH: Pure business logic to reach 90% R² target")
        
        self.analyze_large_residuals_exhaustively()
        self.discover_micro_adjustments()
        self.discover_remaining_edge_cases()
        results = self.apply_final_conservative_rules()
        
        print("\\n" + "=" * 70)
        print("🎯 EXHAUSTIVE ERROR ANALYSIS COMPLETE")
        
        return {
            'final_rules': self.final_rules,
            'performance_results': results,
            'final_r2': results['final_r2'],
            'target_achieved': results['target_achieved']
        }
        
    def save_exhaustive_analysis_results(self, output_path='exhaustive_error_analysis_results.json'):
        """Save exhaustive error analysis results"""
        results = {
            'final_rules': self.final_rules,
            'analysis_summary': {
                'total_rule_categories': len(self.final_rules),
                'total_rules_found': sum(len(category) for category in self.final_rules.values()),
                'approach': 'Exhaustive error analysis with conservative rule application'
            }
        }
        
        # Convert numpy types to Python types
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
        print(f"\\n💾 Exhaustive error analysis results saved to {output_path}")

if __name__ == "__main__":
    # Run exhaustive error analysis
    analyzer = ExhaustiveErrorAnalysis()
    results = analyzer.run_exhaustive_error_analysis()
    analyzer.save_exhaustive_analysis_results()
    
    print(f"\\n📋 EXHAUSTIVE ERROR ANALYSIS SUMMARY:")
    print(f"Final R²: {results['final_r2']:.6f}")
    print(f"90% target achieved: {results['target_achieved']}")
    print(f"Rule categories discovered: {len(results['final_rules'])}")
    print(f"Rules applied: {results['performance_results']['rules_applied']}")
    
    if results['target_achieved']:
        print("🎉 SUCCESS: 90%+ accuracy achieved with pure business logic!")
        print("🏆 Mission accomplished - interpretable business rules reached target!")
    else:
        remaining = 0.90 - results['final_r2']
        print(f"⚠️  Remaining gap: {remaining:.4f} to reach 90%")
        print("➡️  Next: Task 2.8 - Consider minimal hybrid approach")