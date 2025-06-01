#!/usr/bin/env python3
"""
Calibrated Business Rules Implementation
Black Box Challenge - Deterministic Foundation Building

This module implements business rules with careful calibration to avoid overcorrection.
Uses percentage-based adjustments and residual-guided calibration.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class CalibratedBusinessRules:
    """
    Implements carefully calibrated business rules based on interview insights
    """
    
    def __init__(self, data_path='test_cases.json', linear_coeffs_path='linear_coefficients.json'):
        """Initialize with test cases and linear model baseline"""
        self.data_path = data_path
        self.linear_coeffs_path = linear_coeffs_path
        self.df = None
        self.linear_baseline = None
        self.business_rules = {}
        self.load_data()
        
    def load_data(self):
        """Load test cases and linear baseline model"""
        # Load test cases
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
        
        # Load linear baseline coefficients
        try:
            with open(self.linear_coeffs_path, 'r') as f:
                linear_results = json.load(f)
            
            if 'optimized_model' in linear_results:
                self.linear_baseline = linear_results['optimized_model']['formula']
            elif 'base_linear' in linear_results:
                coeffs = linear_results['base_linear']['coefficients']
                self.linear_baseline = {
                    'intercept': linear_results['base_linear']['intercept'],
                    'trip_duration_days': coeffs['trip_duration_days'],
                    'miles_traveled': coeffs['miles_traveled'],
                    'total_receipts_amount': coeffs['total_receipts_amount']
                }
            
            print("Loaded linear baseline model")
            
        except FileNotFoundError:
            # Fallback to pattern discovery results
            self.linear_baseline = {
                'intercept': 318.40,
                'trip_duration_days': 70.36,
                'miles_traveled': 0.818,
                'total_receipts_amount': 0.288
            }
    
    def calculate_linear_baseline(self, days, miles, receipts):
        """Calculate linear baseline reimbursement"""
        return (self.linear_baseline['intercept'] +
                self.linear_baseline['trip_duration_days'] * days +
                self.linear_baseline['miles_traveled'] * miles +
                self.linear_baseline['total_receipts_amount'] * receipts)
    
    def add_baseline_predictions(self):
        """Add linear baseline predictions to dataframe"""
        self.df['linear_baseline'] = [
            self.calculate_linear_baseline(row['trip_duration_days'], 
                                         row['miles_traveled'], 
                                         row['total_receipts_amount'])
            for _, row in self.df.iterrows()
        ]
        
        self.df['baseline_residual'] = self.df['reimbursement'] - self.df['linear_baseline']
        
        baseline_r2 = r2_score(self.df['reimbursement'], self.df['linear_baseline'])
        print(f"Linear baseline R²: {baseline_r2:.4f}")
        
    def implement_calibrated_business_rules(self):
        """Implement business rules with careful calibration based on residual analysis"""
        print("\n=== IMPLEMENTING CALIBRATED BUSINESS RULES ===")
        
        # Rule 1: John's Mileage Diminishing Returns
        # Instead of flat penalty, use percentage reduction on excess miles
        print("\n1. Mileage Diminishing Returns (John's Rule):")
        high_miles = self.df[self.df['miles_traveled'] > 100]
        low_miles = self.df[self.df['miles_traveled'] <= 100]
        
        if len(high_miles) > 0:
            # Calculate actual diminishing return rate from residuals
            residual_diff = high_miles['baseline_residual'].mean() - low_miles['baseline_residual'].mean()
            # Scale to percentage of excess miles above 100
            avg_excess_miles = (high_miles['miles_traveled'] - 100).mean()
            diminishing_rate = residual_diff / avg_excess_miles if avg_excess_miles > 0 else 0
            
            print(f"  Residual difference: ${residual_diff:.2f}")
            print(f"  Average excess miles: {avg_excess_miles:.1f}")
            print(f"  Diminishing rate: ${diminishing_rate:.4f} per excess mile")
            
            self.business_rules['mileage_diminishing_returns'] = {
                'threshold': 100,
                'rate_per_excess_mile': float(diminishing_rate),
                'description': 'Diminishing returns on miles over 100'
            }
        
        # Rule 2: Peggy's Sweet Spot Bonus (5-6 days)
        print("\n2. Sweet Spot Bonus (Peggy's Rule):")
        sweet_spot = self.df[self.df['trip_duration_days'].isin([5, 6])]
        others = self.df[~self.df['trip_duration_days'].isin([5, 6])]
        
        if len(sweet_spot) > 0:
            sweet_spot_bonus = sweet_spot['baseline_residual'].mean() - others['baseline_residual'].mean()
            # Cap the bonus at reasonable percentage of baseline
            avg_sweet_spot_baseline = sweet_spot['linear_baseline'].mean()
            bonus_percentage = (sweet_spot_bonus / avg_sweet_spot_baseline) * 100
            
            # Limit to reasonable percentage
            if bonus_percentage > 10:  # Cap at 10%
                sweet_spot_bonus = avg_sweet_spot_baseline * 0.10
                bonus_percentage = 10.0
                
            print(f"  Raw bonus: ${sweet_spot['baseline_residual'].mean() - others['baseline_residual'].mean():.2f}")
            print(f"  Calibrated bonus: ${sweet_spot_bonus:.2f} ({bonus_percentage:.1f}%)")
            
            self.business_rules['sweet_spot_bonus'] = {
                'days': [5, 6],
                'bonus_amount': float(sweet_spot_bonus),
                'bonus_percentage': float(bonus_percentage),
                'description': '5-6 day trips get sweet spot bonus'
            }
        
        # Rule 3: Receipt Ceiling Effect (Peggy's Rule)
        print("\n3. Receipt Ceiling Effect (Peggy's Rule):")
        receipt_90th = np.percentile(self.df['total_receipts_amount'], 90)
        high_receipts = self.df[self.df['total_receipts_amount'] > receipt_90th]
        normal_receipts = self.df[self.df['total_receipts_amount'] <= receipt_90th]
        
        if len(high_receipts) > 0:
            # Use percentage penalty instead of flat amount
            ceiling_penalty_pct = 0.05  # 5% penalty for high receipts
            avg_high_receipt_baseline = high_receipts['linear_baseline'].mean()
            ceiling_penalty = avg_high_receipt_baseline * ceiling_penalty_pct
            
            print(f"  High receipt threshold: ${receipt_90th:.0f}")
            print(f"  Residual difference: ${high_receipts['baseline_residual'].mean() - normal_receipts['baseline_residual'].mean():.2f}")
            print(f"  Calibrated penalty: ${ceiling_penalty:.2f} ({ceiling_penalty_pct*100:.1f}%)")
            
            self.business_rules['receipt_ceiling'] = {
                'threshold': float(receipt_90th),
                'penalty_percentage': ceiling_penalty_pct,
                'penalty_amount': float(ceiling_penalty),
                'description': 'High receipts (>90th percentile) get ceiling penalty'
            }
        
        # Rule 4: Sarah's Big Trip Jackpot
        print("\n4. Big Trip Jackpot (Sarah's Rule):")
        jackpot_criteria = (
            (self.df['trip_duration_days'] >= 8) &
            (self.df['miles_traveled'] >= 900) &
            (self.df['total_receipts_amount'] >= 1200)
        )
        
        jackpot_trips = self.df[jackpot_criteria]
        non_jackpot = self.df[~jackpot_criteria]
        
        if len(jackpot_trips) > 0:
            jackpot_bonus = jackpot_trips['baseline_residual'].mean() - non_jackpot['baseline_residual'].mean()
            avg_jackpot_baseline = jackpot_trips['linear_baseline'].mean()
            jackpot_percentage = (jackpot_bonus / avg_jackpot_baseline) * 100
            
            print(f"  Qualifying trips: {len(jackpot_trips)}")
            print(f"  Bonus: ${jackpot_bonus:.2f} ({jackpot_percentage:.1f}%)")
            
            self.business_rules['big_trip_jackpot'] = {
                'criteria': {
                    'min_days': 8,
                    'min_miles': 900,
                    'min_receipts': 1200
                },
                'bonus_amount': float(jackpot_bonus),
                'bonus_percentage': float(jackpot_percentage),
                'description': 'Big trips (8+ days, 900+ miles, $1200+ receipts) get jackpot bonus'
            }
        
        # Rule 5: Distance Bonus (Tom's Rule)
        print("\n5. Distance Bonus (Tom's Rule):")
        distance_threshold = 600
        long_distance = self.df[self.df['miles_traveled'] >= distance_threshold]
        short_distance = self.df[self.df['miles_traveled'] < distance_threshold]
        
        if len(long_distance) > 0:
            distance_bonus = long_distance['baseline_residual'].mean() - short_distance['baseline_residual'].mean()
            # Cap at reasonable amount
            if distance_bonus > 100:
                distance_bonus = 100
                
            print(f"  Long distance trips ({distance_threshold}+ miles): {len(long_distance)}")
            print(f"  Bonus: ${distance_bonus:.2f}")
            
            self.business_rules['distance_bonus'] = {
                'threshold': distance_threshold,
                'bonus_amount': float(distance_bonus),
                'description': 'Long distance trips get logistical bonus'
            }
    
    def apply_calibrated_rules(self):
        """Apply all calibrated business rules to calculate final predictions"""
        print("\n=== APPLYING CALIBRATED RULES ===")
        
        self.df['business_rule_adjustment'] = 0
        
        for rule_name, rule in self.business_rules.items():
            adjustment = np.zeros(len(self.df))
            
            if rule_name == 'mileage_diminishing_returns':
                # Apply diminishing returns to excess miles only
                excess_miles = np.maximum(0, self.df['miles_traveled'] - rule['threshold'])
                adjustment = excess_miles * rule['rate_per_excess_mile']
                affected = (excess_miles > 0).sum()
                
            elif rule_name == 'sweet_spot_bonus':
                mask = self.df['trip_duration_days'].isin(rule['days'])
                adjustment[mask] = rule['bonus_amount']
                affected = mask.sum()
                
            elif rule_name == 'receipt_ceiling':
                mask = self.df['total_receipts_amount'] > rule['threshold']
                adjustment[mask] = -rule['penalty_amount']
                affected = mask.sum()
                
            elif rule_name == 'big_trip_jackpot':
                criteria = rule['criteria']
                mask = ((self.df['trip_duration_days'] >= criteria['min_days']) &
                       (self.df['miles_traveled'] >= criteria['min_miles']) &
                       (self.df['total_receipts_amount'] >= criteria['min_receipts']))
                adjustment[mask] = rule['bonus_amount']
                affected = mask.sum()
                
            elif rule_name == 'distance_bonus':
                mask = self.df['miles_traveled'] >= rule['threshold']
                adjustment[mask] = rule['bonus_amount']
                affected = mask.sum()
            
            self.df['business_rule_adjustment'] += adjustment
            
            avg_adj = adjustment[adjustment != 0].mean() if (adjustment != 0).sum() > 0 else 0
            print(f"{rule_name}: {affected} trips affected, avg adjustment: ${avg_adj:.2f}")
        
        # Calculate final prediction
        self.df['final_prediction'] = self.df['linear_baseline'] + self.df['business_rule_adjustment']
        self.df['final_residual'] = self.df['reimbursement'] - self.df['final_prediction']
        
        # Performance metrics
        baseline_r2 = r2_score(self.df['reimbursement'], self.df['linear_baseline'])
        final_r2 = r2_score(self.df['reimbursement'], self.df['final_prediction'])
        
        baseline_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['linear_baseline']))
        final_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['final_prediction']))
        
        baseline_mae = mean_absolute_error(self.df['reimbursement'], self.df['linear_baseline'])
        final_mae = mean_absolute_error(self.df['reimbursement'], self.df['final_prediction'])
        
        print(f"\n=== PERFORMANCE COMPARISON ===")
        print(f"Linear baseline: R²={baseline_r2:.4f}, RMSE=${baseline_rmse:.2f}, MAE=${baseline_mae:.2f}")
        print(f"With rules:      R²={final_r2:.4f}, RMSE=${final_rmse:.2f}, MAE=${final_mae:.2f}")
        print(f"Improvement:     ΔR²={final_r2-baseline_r2:.4f}, ΔRMSE=${baseline_rmse-final_rmse:.2f}, ΔMAE=${baseline_mae-final_mae:.2f}")
        
        return final_r2, final_rmse, final_mae
    
    def analyze_rule_effectiveness(self):
        """Analyze which rules are most effective"""
        print(f"\n=== RULE EFFECTIVENESS ANALYSIS ===")
        
        # Test each rule individually
        individual_improvements = {}
        
        for test_rule in self.business_rules.keys():
            # Apply only this rule
            test_adjustment = np.zeros(len(self.df))
            
            rule = self.business_rules[test_rule]
            
            if test_rule == 'mileage_diminishing_returns':
                excess_miles = np.maximum(0, self.df['miles_traveled'] - rule['threshold'])
                test_adjustment = excess_miles * rule['rate_per_excess_mile']
                
            elif test_rule == 'sweet_spot_bonus':
                mask = self.df['trip_duration_days'].isin(rule['days'])
                test_adjustment[mask] = rule['bonus_amount']
                
            elif test_rule == 'receipt_ceiling':
                mask = self.df['total_receipts_amount'] > rule['threshold']
                test_adjustment[mask] = -rule['penalty_amount']
                
            elif test_rule == 'big_trip_jackpot':
                criteria = rule['criteria']
                mask = ((self.df['trip_duration_days'] >= criteria['min_days']) &
                       (self.df['miles_traveled'] >= criteria['min_miles']) &
                       (self.df['total_receipts_amount'] >= criteria['min_receipts']))
                test_adjustment[mask] = rule['bonus_amount']
                
            elif test_rule == 'distance_bonus':
                mask = self.df['miles_traveled'] >= rule['threshold']
                test_adjustment[mask] = rule['bonus_amount']
            
            # Calculate R² with only this rule
            test_prediction = self.df['linear_baseline'] + test_adjustment
            baseline_r2 = r2_score(self.df['reimbursement'], self.df['linear_baseline'])
            test_r2 = r2_score(self.df['reimbursement'], test_prediction)
            improvement = test_r2 - baseline_r2
            
            individual_improvements[test_rule] = improvement
            print(f"{test_rule}: ΔR² = {improvement:.4f}")
        
        # Rank rules by effectiveness
        sorted_rules = sorted(individual_improvements.items(), key=lambda x: x[1], reverse=True)
        print(f"\nRule Ranking (by R² improvement):")
        for i, (rule_name, improvement) in enumerate(sorted_rules, 1):
            print(f"  {i}. {rule_name}: +{improvement:.4f}")
        
        return individual_improvements
    
    def run_calibrated_analysis(self):
        """Run complete calibrated business rules analysis"""
        print("🔍 STARTING CALIBRATED BUSINESS RULES ANALYSIS")
        print("=" * 60)
        
        self.add_baseline_predictions()
        self.implement_calibrated_business_rules()
        final_r2, final_rmse, final_mae = self.apply_calibrated_rules()
        individual_improvements = self.analyze_rule_effectiveness()
        
        print("\n" + "=" * 60)
        print("🎯 CALIBRATED BUSINESS RULES ANALYSIS COMPLETE")
        
        return self.business_rules, final_r2, final_rmse
    
    def save_calibrated_rules(self, output_path='calibrated_business_rules.json'):
        """Save calibrated business rules"""
        results = {
            'business_rules': self.business_rules,
            'linear_baseline': self.linear_baseline,
            'model_performance': {
                'final_r2': float(r2_score(self.df['reimbursement'], self.df['final_prediction'])),
                'final_rmse': float(np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['final_prediction']))),
                'final_mae': float(mean_absolute_error(self.df['reimbursement'], self.df['final_prediction']))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Calibrated business rules saved to {output_path}")

if __name__ == "__main__":
    # Run calibrated business rules analysis
    analyzer = CalibratedBusinessRules()
    business_rules, final_r2, final_rmse = analyzer.run_calibrated_analysis()
    analyzer.save_calibrated_rules()
    
    print(f"\n📋 SUMMARY:")
    print(f"Final model R²: {final_r2:.4f}")
    print(f"Final model RMSE: ${final_rmse:.2f}")
    print(f"Number of business rules: {len(business_rules)}")