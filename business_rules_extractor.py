#!/usr/bin/env python3
"""
Business Rules Extractor
Black Box Challenge - Deterministic Foundation Building

This module extracts and implements specific business rules from employee interviews,
building on the linear baseline to create interpretable deterministic logic.

Key Interview Rules to Extract:
- John: Mileage diminishing returns (30-40% drop after ~100 miles), road warrior bonus
- Peggy: 5-6 day sweet spot bonus, receipt ceiling effects, small receipt penalties  
- Sarah: Big trip jackpot (8+ days + 900+ miles + $1200+ receipts = 25% bonus)
- Tom: Distance bonuses, seasonal variations, spending rate penalties
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class BusinessRulesExtractor:
    """
    Extracts and implements business rules from employee interviews
    """
    
    def __init__(self, data_path='test_cases.json', linear_coeffs_path='linear_coefficients.json'):
        """Initialize with test cases and linear model baseline"""
        self.data_path = data_path
        self.linear_coeffs_path = linear_coeffs_path
        self.df = None
        self.linear_baseline = None
        self.business_rules = {}
        self.rule_impacts = {}
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
            print("No linear coefficients found, using pattern discovery baseline")
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
        print(f"Mean residual: ${self.df['baseline_residual'].mean():.2f}")
        print(f"Residual std: ${self.df['baseline_residual'].std():.2f}")
    
    def extract_john_mileage_rules(self):
        """
        Extract John's mileage rules:
        - Diminishing returns after ~100 miles (30-40% rate drop)
        - Road warrior bonus for high miles/day ratios
        """
        print("\n=== JOHN'S MILEAGE RULES ===")
        
        # Test mileage breakpoint at 100 miles
        low_miles = self.df[self.df['miles_traveled'] <= 100]
        high_miles = self.df[self.df['miles_traveled'] > 100]
        
        print(f"Mileage Analysis:")
        print(f"  ≤100 miles: {len(low_miles)} trips, avg residual: ${low_miles['baseline_residual'].mean():.2f}")
        print(f"  >100 miles: {len(high_miles)} trips, avg residual: ${high_miles['baseline_residual'].mean():.2f}")
        
        # Test if high-mile trips have systematic negative residuals (diminishing returns)
        if len(high_miles) > 0 and high_miles['baseline_residual'].mean() < low_miles['baseline_residual'].mean():
            miles_penalty = abs(high_miles['baseline_residual'].mean() - low_miles['baseline_residual'].mean())
            print(f"  Mileage penalty detected: ${miles_penalty:.2f} for trips >100 miles")
            
            self.business_rules['mileage_diminishing_returns'] = {
                'threshold': 100,
                'penalty': miles_penalty,
                'applies_to': 'miles_over_threshold'
            }
        
        # Test road warrior bonus (high miles/day)
        efficiency_threshold = 200  # miles per day
        high_efficiency = self.df[self.df['miles_per_day'] > efficiency_threshold]
        low_efficiency = self.df[self.df['miles_per_day'] <= efficiency_threshold]
        
        if len(high_efficiency) > 10 and len(low_efficiency) > 10:
            print(f"\nRoad Warrior Analysis (>{efficiency_threshold} miles/day):")
            print(f"  High efficiency: {len(high_efficiency)} trips, avg residual: ${high_efficiency['baseline_residual'].mean():.2f}")
            print(f"  Low efficiency: {len(low_efficiency)} trips, avg residual: ${low_efficiency['baseline_residual'].mean():.2f}")
            
            if high_efficiency['baseline_residual'].mean() > low_efficiency['baseline_residual'].mean():
                road_warrior_bonus = high_efficiency['baseline_residual'].mean() - low_efficiency['baseline_residual'].mean()
                print(f"  Road warrior bonus detected: ${road_warrior_bonus:.2f}")
                
                self.business_rules['road_warrior_bonus'] = {
                    'threshold': efficiency_threshold,
                    'bonus': road_warrior_bonus,
                    'applies_to': 'miles_per_day_over_threshold'
                }
    
    def extract_peggy_sweet_spot_rules(self):
        """
        Extract Peggy's rules:
        - 5-6 day sweet spot bonus
        - Receipt ceiling effects  
        - Small receipt penalties
        """
        print("\n=== PEGGY'S SWEET SPOT RULES ===")
        
        # Test 5-6 day sweet spot
        sweet_spot_days = self.df[self.df['trip_duration_days'].isin([5, 6])]
        other_days = self.df[~self.df['trip_duration_days'].isin([5, 6])]
        
        print(f"Sweet Spot Analysis (5-6 days):")
        print(f"  Sweet spot trips: {len(sweet_spot_days)}, avg residual: ${sweet_spot_days['baseline_residual'].mean():.2f}")
        print(f"  Other trips: {len(other_days)}, avg residual: ${other_days['baseline_residual'].mean():.2f}")
        
        if len(sweet_spot_days) > 0 and sweet_spot_days['baseline_residual'].mean() > other_days['baseline_residual'].mean():
            sweet_spot_bonus = sweet_spot_days['baseline_residual'].mean() - other_days['baseline_residual'].mean()
            print(f"  Sweet spot bonus detected: ${sweet_spot_bonus:.2f}")
            
            self.business_rules['sweet_spot_bonus'] = {
                'days': [5, 6],
                'bonus': sweet_spot_bonus,
                'applies_to': 'trip_duration_in_range'
            }
        
        # Test receipt ceiling effect (high receipts get diminishing returns)
        receipt_90th = np.percentile(self.df['total_receipts_amount'], 90)
        high_receipts = self.df[self.df['total_receipts_amount'] > receipt_90th]
        low_receipts = self.df[self.df['total_receipts_amount'] <= receipt_90th]
        
        print(f"\nReceipt Ceiling Analysis (>${receipt_90th:.0f}+):")
        print(f"  High receipts: {len(high_receipts)} trips, avg residual: ${high_receipts['baseline_residual'].mean():.2f}")
        print(f"  Low receipts: {len(low_receipts)} trips, avg residual: ${low_receipts['baseline_residual'].mean():.2f}")
        
        if len(high_receipts) > 0 and high_receipts['baseline_residual'].mean() < low_receipts['baseline_residual'].mean():
            receipt_penalty = abs(high_receipts['baseline_residual'].mean() - low_receipts['baseline_residual'].mean())
            print(f"  Receipt ceiling penalty detected: ${receipt_penalty:.2f}")
            
            self.business_rules['receipt_ceiling'] = {
                'threshold': receipt_90th,
                'penalty': receipt_penalty,
                'applies_to': 'receipts_over_threshold'
            }
        
        # Test small receipt penalty
        small_receipt_threshold = np.percentile(self.df['total_receipts_amount'], 10)
        small_receipts = self.df[self.df['total_receipts_amount'] < small_receipt_threshold]
        normal_receipts = self.df[self.df['total_receipts_amount'] >= small_receipt_threshold]
        
        print(f"\nSmall Receipt Analysis (<${small_receipt_threshold:.0f}):")
        print(f"  Small receipts: {len(small_receipts)} trips, avg residual: ${small_receipts['baseline_residual'].mean():.2f}")
        print(f"  Normal receipts: {len(normal_receipts)} trips, avg residual: ${normal_receipts['baseline_residual'].mean():.2f}")
        
        if len(small_receipts) > 0 and small_receipts['baseline_residual'].mean() < normal_receipts['baseline_residual'].mean():
            small_receipt_penalty = abs(small_receipts['baseline_residual'].mean() - normal_receipts['baseline_residual'].mean())
            print(f"  Small receipt penalty detected: ${small_receipt_penalty:.2f}")
            
            self.business_rules['small_receipt_penalty'] = {
                'threshold': small_receipt_threshold,
                'penalty': small_receipt_penalty,
                'applies_to': 'receipts_under_threshold'
            }
    
    def extract_sarah_jackpot_rules(self):
        """
        Extract Sarah's big trip jackpot rule:
        - 8+ days + 900+ miles + $1200+ receipts = 25% bonus
        """
        print("\n=== SARAH'S JACKPOT RULES ===")
        
        # Test exact jackpot criteria from interview
        jackpot_criteria = (
            (self.df['trip_duration_days'] >= 8) &
            (self.df['miles_traveled'] >= 900) &
            (self.df['total_receipts_amount'] >= 1200)
        )
        
        jackpot_trips = self.df[jackpot_criteria]
        non_jackpot_trips = self.df[~jackpot_criteria]
        
        print(f"Big Trip Jackpot Analysis (8+ days, 900+ miles, $1200+ receipts):")
        print(f"  Jackpot trips: {len(jackpot_trips)}")
        
        if len(jackpot_trips) > 0:
            print(f"  Jackpot avg residual: ${jackpot_trips['baseline_residual'].mean():.2f}")
            print(f"  Non-jackpot avg residual: ${non_jackpot_trips['baseline_residual'].mean():.2f}")
            
            jackpot_bonus = jackpot_trips['baseline_residual'].mean() - non_jackpot_trips['baseline_residual'].mean()
            print(f"  Jackpot bonus detected: ${jackpot_bonus:.2f}")
            
            # Calculate as percentage of baseline amount
            avg_jackpot_baseline = jackpot_trips['linear_baseline'].mean()
            jackpot_percentage = (jackpot_bonus / avg_jackpot_baseline) * 100
            print(f"  Jackpot bonus as percentage: {jackpot_percentage:.1f}%")
            
            self.business_rules['big_trip_jackpot'] = {
                'criteria': {
                    'min_days': 8,
                    'min_miles': 900,
                    'min_receipts': 1200
                },
                'bonus': jackpot_bonus,
                'bonus_percentage': jackpot_percentage,
                'applies_to': 'all_criteria_met'
            }
        else:
            print("  No trips found matching jackpot criteria")
    
    def extract_tom_distance_rules(self):
        """
        Extract Tom's rules:
        - Distance bonuses for "logistically annoying" distances
        - Seasonal budget variations
        """
        print("\n=== TOM'S DISTANCE RULES ===")
        
        # Test distance bonus for long trips (600+ miles based on Tom's examples)
        distance_threshold = 600
        long_distance = self.df[self.df['miles_traveled'] >= distance_threshold]
        short_distance = self.df[self.df['miles_traveled'] < distance_threshold]
        
        print(f"Distance Bonus Analysis ({distance_threshold}+ miles):")
        print(f"  Long distance: {len(long_distance)} trips, avg residual: ${long_distance['baseline_residual'].mean():.2f}")
        print(f"  Short distance: {len(short_distance)} trips, avg residual: ${short_distance['baseline_residual'].mean():.2f}")
        
        if len(long_distance) > 0 and long_distance['baseline_residual'].mean() > short_distance['baseline_residual'].mean():
            distance_bonus = long_distance['baseline_residual'].mean() - short_distance['baseline_residual'].mean()
            print(f"  Distance bonus detected: ${distance_bonus:.2f}")
            
            self.business_rules['distance_bonus'] = {
                'threshold': distance_threshold,
                'bonus': distance_bonus,
                'applies_to': 'miles_over_threshold'
            }
        
        # Test spending rate penalty (high spending per day on long trips)
        long_trips = self.df[self.df['trip_duration_days'] >= 7]
        if len(long_trips) > 20:
            # Split by spending rate
            median_spending_rate = long_trips['receipts_per_day'].median()
            high_spending = long_trips[long_trips['receipts_per_day'] > median_spending_rate]
            low_spending = long_trips[long_trips['receipts_per_day'] <= median_spending_rate]
            
            print(f"\nSpending Rate Analysis (long trips, ${median_spending_rate:.0f}/day threshold):")
            print(f"  High spending rate: {len(high_spending)} trips, avg residual: ${high_spending['baseline_residual'].mean():.2f}")
            print(f"  Low spending rate: {len(low_spending)} trips, avg residual: ${low_spending['baseline_residual'].mean():.2f}")
            
            if len(high_spending) > 0 and high_spending['baseline_residual'].mean() < low_spending['baseline_residual'].mean():
                spending_penalty = abs(high_spending['baseline_residual'].mean() - low_spending['baseline_residual'].mean())
                print(f"  High spending penalty detected: ${spending_penalty:.2f}")
                
                self.business_rules['high_spending_penalty'] = {
                    'min_days': 7,
                    'spending_threshold': median_spending_rate,
                    'penalty': spending_penalty,
                    'applies_to': 'high_spending_on_long_trips'
                }
    
    def test_business_rules_impact(self):
        """Test the impact of applying all discovered business rules"""
        print("\n=== BUSINESS RULES IMPACT ANALYSIS ===")
        
        # Apply all business rules to calculate adjustments
        self.df['business_rule_adjustment'] = 0
        
        for rule_name, rule in self.business_rules.items():
            adjustment = np.zeros(len(self.df))
            
            if rule_name == 'mileage_diminishing_returns':
                mask = self.df['miles_traveled'] > rule['threshold']
                excess_miles = np.maximum(0, self.df['miles_traveled'] - rule['threshold'])
                adjustment[mask] = -rule['penalty'] * (excess_miles[mask] / 100)  # Scale by excess miles
                
            elif rule_name == 'road_warrior_bonus':
                mask = self.df['miles_per_day'] > rule['threshold']
                adjustment[mask] = rule['bonus']
                
            elif rule_name == 'sweet_spot_bonus':
                mask = self.df['trip_duration_days'].isin(rule['days'])
                adjustment[mask] = rule['bonus']
                
            elif rule_name == 'receipt_ceiling':
                mask = self.df['total_receipts_amount'] > rule['threshold']
                adjustment[mask] = -rule['penalty']
                
            elif rule_name == 'small_receipt_penalty':
                mask = self.df['total_receipts_amount'] < rule['threshold']
                adjustment[mask] = -rule['penalty']
                
            elif rule_name == 'big_trip_jackpot':
                criteria = rule['criteria']
                mask = ((self.df['trip_duration_days'] >= criteria['min_days']) &
                       (self.df['miles_traveled'] >= criteria['min_miles']) &
                       (self.df['total_receipts_amount'] >= criteria['min_receipts']))
                adjustment[mask] = rule['bonus']
                
            elif rule_name == 'distance_bonus':
                mask = self.df['miles_traveled'] >= rule['threshold']
                adjustment[mask] = rule['bonus']
                
            elif rule_name == 'high_spending_penalty':
                mask = ((self.df['trip_duration_days'] >= rule['min_days']) &
                       (self.df['receipts_per_day'] > rule['spending_threshold']))
                adjustment[mask] = -rule['penalty']
            
            self.df['business_rule_adjustment'] += adjustment
            
            # Track individual rule impact
            self.rule_impacts[rule_name] = {
                'affected_trips': int(mask.sum()),
                'avg_adjustment': float(adjustment[mask].mean() if mask.sum() > 0 else 0),
                'total_adjustment': float(adjustment.sum())
            }
            
            print(f"{rule_name}: {mask.sum()} trips affected, avg adjustment: ${adjustment[mask].mean():.2f}" if mask.sum() > 0 else f"{rule_name}: No trips affected")
        
        # Calculate combined prediction
        self.df['combined_prediction'] = self.df['linear_baseline'] + self.df['business_rule_adjustment']
        self.df['combined_residual'] = self.df['reimbursement'] - self.df['combined_prediction']
        
        # Performance comparison
        baseline_r2 = r2_score(self.df['reimbursement'], self.df['linear_baseline'])
        combined_r2 = r2_score(self.df['reimbursement'], self.df['combined_prediction'])
        
        baseline_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['linear_baseline']))
        combined_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['combined_prediction']))
        
        print(f"\nPerformance Comparison:")
        print(f"  Linear baseline R²: {baseline_r2:.4f}, RMSE: ${baseline_rmse:.2f}")
        print(f"  Combined model R²: {combined_r2:.4f}, RMSE: ${combined_rmse:.2f}")
        print(f"  Improvement: {combined_r2 - baseline_r2:.4f} R² points, ${baseline_rmse - combined_rmse:.2f} RMSE reduction")
        
        return combined_r2, combined_rmse
    
    def run_full_extraction(self):
        """Run complete business rules extraction process"""
        print("🔍 STARTING BUSINESS RULES EXTRACTION")
        print("=" * 60)
        
        # Add baseline predictions
        self.add_baseline_predictions()
        
        # Extract rules from each interview
        self.extract_john_mileage_rules()
        self.extract_peggy_sweet_spot_rules()
        self.extract_sarah_jackpot_rules()
        self.extract_tom_distance_rules()
        
        # Test combined impact
        combined_r2, combined_rmse = self.test_business_rules_impact()
        
        print("\n" + "=" * 60)
        print("🎯 BUSINESS RULES EXTRACTION COMPLETE")
        
        return self.business_rules, self.rule_impacts
    
    def save_business_rules(self, output_path='extracted_business_rules.json'):
        """Save extracted business rules and analysis"""
        results = {
            'business_rules': self.business_rules,
            'rule_impacts': self.rule_impacts,
            'linear_baseline': self.linear_baseline,
            'extraction_summary': {
                'total_rules_extracted': len(self.business_rules),
                'total_affected_trips': sum(impact['affected_trips'] for impact in self.rule_impacts.values()),
                'avg_rule_adjustment': np.mean([abs(impact['avg_adjustment']) for impact in self.rule_impacts.values() if impact['avg_adjustment'] != 0])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Business rules saved to {output_path}")

if __name__ == "__main__":
    # Run business rules extraction
    extractor = BusinessRulesExtractor()
    business_rules, rule_impacts = extractor.run_full_extraction()
    extractor.save_business_rules()
    
    print("\n📋 SUMMARY OF EXTRACTED RULES:")
    print("-" * 40)
    
    for rule_name, rule in business_rules.items():
        impact = rule_impacts[rule_name]
        print(f"{rule_name}:")
        print(f"  Affected trips: {impact['affected_trips']}")
        print(f"  Avg adjustment: ${impact['avg_adjustment']:.2f}")
        
        if 'bonus' in rule:
            print(f"  Bonus: ${rule['bonus']:.2f}")
        if 'penalty' in rule:
            print(f"  Penalty: ${rule['penalty']:.2f}")
        if 'threshold' in rule:
            print(f"  Threshold: {rule['threshold']}")
        print()