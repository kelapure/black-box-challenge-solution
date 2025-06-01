#!/usr/bin/env python3
"""
Edge Case Business Logic Miner
Black Box Challenge - Deterministic Foundation Building

This module mines additional business logic from interview edge cases and exceptions
that weren't captured in the main rules. Focuses on:

1. Rounding rules and decimal artifacts (Tom's $1.02 example)
2. Seasonal/quarterly budget variations (Tom & Sarah's observations)
3. Department-specific logic patterns
4. Travel day calculation edge cases (John's Sunday night/Tuesday morning example)
5. Small trip penalties and minimum thresholds
6. Complex interaction effects between rules
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class EdgeCaseBusinessLogicMiner:
    """
    Mines additional business logic from interview edge cases and exceptions
    """
    
    def __init__(self, data_path='test_cases.json', rules_path='calibrated_business_rules.json'):
        """Initialize with test cases and existing business rules"""
        self.data_path = data_path
        self.rules_path = rules_path
        self.df = None
        self.existing_rules = None
        self.edge_case_rules = {}
        self.load_data()
        
    def load_data(self):
        """Load test cases and existing business rules"""
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
        
        # Load existing business rules
        try:
            with open(self.rules_path, 'r') as f:
                rules_data = json.load(f)
            self.existing_rules = rules_data
            print("Loaded existing business rules")
        except FileNotFoundError:
            print("No existing rules found")
            self.existing_rules = {}
            
    def apply_existing_rules(self):
        """Apply existing business rules to get baseline predictions"""
        if 'linear_baseline' not in self.existing_rules:
            print("No linear baseline found, using fallback")
            return
            
        baseline = self.existing_rules['linear_baseline']
        
        # Calculate linear baseline
        self.df['linear_baseline'] = (
            baseline['intercept'] +
            baseline['trip_duration_days'] * self.df['trip_duration_days'] +
            baseline['miles_traveled'] * self.df['miles_traveled'] +
            baseline['total_receipts_amount'] * self.df['total_receipts_amount']
        )
        
        # Apply business rules
        self.df['business_rule_adjustment'] = 0
        
        if 'business_rules' in self.existing_rules:
            for rule_name, rule in self.existing_rules['business_rules'].items():
                adjustment = np.zeros(len(self.df))
                
                if rule_name == 'mileage_diminishing_returns' and 'rate_per_excess_mile' in rule:
                    excess_miles = np.maximum(0, self.df['miles_traveled'] - rule['threshold'])
                    adjustment = excess_miles * rule['rate_per_excess_mile']
                    
                elif rule_name == 'sweet_spot_bonus' and 'bonus_amount' in rule:
                    mask = self.df['trip_duration_days'].isin(rule['days'])
                    adjustment[mask] = rule['bonus_amount']
                    
                elif rule_name == 'receipt_ceiling' and 'penalty_amount' in rule:
                    mask = self.df['total_receipts_amount'] > rule['threshold']
                    adjustment[mask] = -rule['penalty_amount']
                    
                elif rule_name == 'big_trip_jackpot' and 'bonus_amount' in rule:
                    criteria = rule['criteria']
                    mask = ((self.df['trip_duration_days'] >= criteria['min_days']) &
                           (self.df['miles_traveled'] >= criteria['min_miles']) &
                           (self.df['total_receipts_amount'] >= criteria['min_receipts']))
                    adjustment[mask] = rule['bonus_amount']
                    
                elif rule_name == 'distance_bonus' and 'bonus_amount' in rule:
                    mask = self.df['miles_traveled'] >= rule['threshold']
                    adjustment[mask] = rule['bonus_amount']
                
                self.df['business_rule_adjustment'] += adjustment
        
        self.df['current_prediction'] = self.df['linear_baseline'] + self.df['business_rule_adjustment']
        self.df['current_residual'] = self.df['reimbursement'] - self.df['current_prediction']
        
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        print(f"Current model R²: {current_r2:.4f}")
        
    def mine_rounding_and_decimal_rules(self):
        """
        Mine rounding rules based on Tom's observation:
        '$199.99 receipt -> $1.02 more than expected' suggests rounding artifacts
        """
        print("\n=== MINING ROUNDING AND DECIMAL RULES ===")
        
        # Look for patterns in decimal places of reimbursements
        self.df['reimbursement_cents'] = (self.df['reimbursement'] * 100) % 100
        self.df['prediction_cents'] = (self.df['current_prediction'] * 100) % 100
        
        # Analyze common decimal endings
        common_endings = self.df['reimbursement_cents'].value_counts().head(10)
        print("Most common reimbursement decimal endings:")
        for cents, count in common_endings.items():
            print(f"  ${cents/100:.2f}: {count} occurrences ({count/len(self.df)*100:.1f}%)")
        
        # Test for rounding to nearest quarter, dollar, etc.
        quarter_rounded = ((self.df['reimbursement'] * 4).round() / 4)
        dollar_rounded = self.df['reimbursement'].round()
        
        quarter_matches = (np.abs(self.df['reimbursement'] - quarter_rounded) < 0.01).sum()
        dollar_matches = (np.abs(self.df['reimbursement'] - dollar_rounded) < 0.01).sum()
        
        print(f"\nRounding analysis:")
        print(f"  Quarter rounding matches: {quarter_matches} ({quarter_matches/len(self.df)*100:.1f}%)")
        print(f"  Dollar rounding matches: {dollar_matches} ({dollar_matches/len(self.df)*100:.1f}%)")
        
        # Look for systematic rounding patterns in residuals
        self.df['residual_rounded_quarter'] = ((self.df['current_residual'] * 4).round() / 4)
        self.df['residual_rounded_dollar'] = self.df['current_residual'].round()
        
        # Test if applying rounding improves predictions
        rounded_quarter_pred = self.df['current_prediction'] + self.df['residual_rounded_quarter']
        rounded_dollar_pred = self.df['current_prediction'] + self.df['residual_rounded_dollar']
        
        quarter_r2 = r2_score(self.df['reimbursement'], rounded_quarter_pred)
        dollar_r2 = r2_score(self.df['reimbursement'], rounded_dollar_pred)
        
        print(f"  Quarter rounding R²: {quarter_r2:.4f}")
        print(f"  Dollar rounding R²: {dollar_r2:.4f}")
        
        # Implement best rounding rule if it improves performance
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        
        if quarter_r2 > current_r2 and quarter_r2 > dollar_r2:
            self.edge_case_rules['quarter_rounding'] = {
                'description': 'Round final result to nearest quarter',
                'improvement': quarter_r2 - current_r2,
                'rule_type': 'rounding'
            }
            print(f"  Quarter rounding rule added: +{quarter_r2 - current_r2:.4f} R²")
            
        elif dollar_r2 > current_r2:
            self.edge_case_rules['dollar_rounding'] = {
                'description': 'Round final result to nearest dollar',
                'improvement': dollar_r2 - current_r2,
                'rule_type': 'rounding'
            }
            print(f"  Dollar rounding rule added: +{dollar_r2 - current_r2:.4f} R²")
    
    def mine_seasonal_budget_variations(self):
        """
        Mine seasonal/quarterly budget variations based on Tom & Sarah's observations:
        'End of quarter? More generous. Middle of the fiscal year? Tighter.'
        """
        print("\n=== MINING SEASONAL BUDGET VARIATIONS ===")
        
        # Since we don't have actual dates, we'll simulate quarterly effects
        # by analyzing if there are systematic patterns in the data that could represent timing
        
        # Look for systematic residual patterns that could indicate timing effects
        # Group by position in dataset (proxy for chronological ordering)
        self.df['data_position'] = range(len(self.df))
        self.df['quarter_proxy'] = (self.df['data_position'] // 1250) % 4  # Divide into 4 quarters
        
        # Analyze residuals by quarter proxy
        quarterly_residuals = self.df.groupby('quarter_proxy')['current_residual'].agg(['mean', 'std', 'count'])
        
        print("Residual analysis by quarter proxy:")
        for quarter, stats in quarterly_residuals.iterrows():
            print(f"  Quarter {quarter}: mean=${stats['mean']:.2f}, std=${stats['std']:.2f}, n={stats['count']}")
        
        # Test if any quarter shows significantly different patterns
        quarter_effects = {}
        base_mean = self.df['current_residual'].mean()
        
        for quarter in range(4):
            quarter_data = self.df[self.df['quarter_proxy'] == quarter]
            if len(quarter_data) > 100:  # Ensure enough data
                quarter_mean = quarter_data['current_residual'].mean()
                effect = quarter_mean - base_mean
                
                if abs(effect) > 10:  # Significant effect threshold
                    quarter_effects[f'quarter_{quarter}'] = {
                        'adjustment': effect,
                        'affected_trips': len(quarter_data),
                        'description': f'Quarter {quarter} budget adjustment'
                    }
                    print(f"  Quarter {quarter} effect detected: ${effect:.2f}")
        
        if quarter_effects:
            self.edge_case_rules['seasonal_variations'] = quarter_effects
    
    def mine_minimum_trip_thresholds(self):
        """
        Mine minimum trip thresholds based on interview insights:
        - Very short trips discouraged
        - Very small expenses penalized
        """
        print("\n=== MINING MINIMUM TRIP THRESHOLDS ===")
        
        # Analyze very short trips (1-2 days)
        short_trips = self.df[self.df['trip_duration_days'] <= 2]
        normal_trips = self.df[self.df['trip_duration_days'] > 2]
        
        if len(short_trips) > 10 and len(normal_trips) > 10:
            short_residual = short_trips['current_residual'].mean()
            normal_residual = normal_trips['current_residual'].mean()
            
            print(f"Short trip analysis (≤2 days):")
            print(f"  Short trips: {len(short_trips)}, avg residual: ${short_residual:.2f}")
            print(f"  Normal trips: {len(normal_trips)}, avg residual: ${normal_residual:.2f}")
            
            if short_residual < normal_residual - 20:  # Significant penalty
                penalty = abs(short_residual - normal_residual)
                self.edge_case_rules['short_trip_penalty'] = {
                    'max_days': 2,
                    'penalty': penalty,
                    'description': 'Very short trips get minimum threshold penalty',
                    'affected_trips': len(short_trips)
                }
                print(f"  Short trip penalty detected: ${penalty:.2f}")
        
        # Analyze very low spending trips
        low_spending_threshold = np.percentile(self.df['total_receipts_amount'], 5)
        low_spending = self.df[self.df['total_receipts_amount'] < low_spending_threshold]
        normal_spending = self.df[self.df['total_receipts_amount'] >= low_spending_threshold]
        
        if len(low_spending) > 10:
            low_residual = low_spending['current_residual'].mean()
            normal_residual = normal_spending['current_residual'].mean()
            
            print(f"\nLow spending analysis (<${low_spending_threshold:.0f}):")
            print(f"  Low spending: {len(low_spending)}, avg residual: ${low_residual:.2f}")
            print(f"  Normal spending: {len(normal_spending)}, avg residual: ${normal_residual:.2f}")
            
            if low_residual < normal_residual - 20:
                penalty = abs(low_residual - normal_residual)
                self.edge_case_rules['minimum_spending_penalty'] = {
                    'threshold': low_spending_threshold,
                    'penalty': penalty,
                    'description': 'Very low spending trips get minimum threshold penalty',
                    'affected_trips': len(low_spending)
                }
                print(f"  Minimum spending penalty detected: ${penalty:.2f}")
    
    def mine_travel_efficiency_bonuses(self):
        """
        Mine travel efficiency bonuses based on interview insights:
        - System rewards 'actual work' vs 'just being somewhere else'
        - High miles/day gets 'road warrior bonus'
        """
        print("\n=== MINING TRAVEL EFFICIENCY BONUSES ===")
        
        # Analyze different efficiency categories
        efficiency_ranges = [
            (0, 50, 'very_low'),
            (50, 100, 'low'),
            (100, 200, 'moderate'),
            (200, 300, 'high'),
            (300, float('inf'), 'very_high')
        ]
        
        efficiency_effects = {}
        base_residual = self.df['current_residual'].mean()
        
        for min_eff, max_eff, category in efficiency_ranges:
            if max_eff == float('inf'):
                mask = self.df['miles_per_day'] >= min_eff
            else:
                mask = (self.df['miles_per_day'] >= min_eff) & (self.df['miles_per_day'] < max_eff)
            
            category_data = self.df[mask]
            
            if len(category_data) > 50:  # Ensure enough data
                category_residual = category_data['current_residual'].mean()
                effect = category_residual - base_residual
                
                print(f"Efficiency {category} ({min_eff}-{max_eff} miles/day): {len(category_data)} trips, effect: ${effect:.2f}")
                
                if abs(effect) > 15:  # Significant effect
                    efficiency_effects[category] = {
                        'min_efficiency': min_eff,
                        'max_efficiency': max_eff if max_eff != float('inf') else None,
                        'adjustment': effect,
                        'affected_trips': len(category_data),
                        'description': f'Efficiency bonus/penalty for {category} efficiency'
                    }
        
        if efficiency_effects:
            self.edge_case_rules['efficiency_adjustments'] = efficiency_effects
    
    def mine_complex_interaction_rules(self):
        """
        Mine complex interaction rules between different factors
        """
        print("\n=== MINING COMPLEX INTERACTION RULES ===")
        
        # Test interaction between trip length and spending rate
        long_trips = self.df[self.df['trip_duration_days'] >= 7]
        
        if len(long_trips) > 100:
            median_spending_rate = long_trips['receipts_per_day'].median()
            
            # Long trips with high spending
            high_spend_long = long_trips[long_trips['receipts_per_day'] > median_spending_rate * 1.5]
            low_spend_long = long_trips[long_trips['receipts_per_day'] <= median_spending_rate]
            
            if len(high_spend_long) > 10 and len(low_spend_long) > 10:
                high_residual = high_spend_long['current_residual'].mean()
                low_residual = low_spend_long['current_residual'].mean()
                
                print(f"Long trip spending interaction:")
                print(f"  High spending long trips: {len(high_spend_long)}, residual: ${high_residual:.2f}")
                print(f"  Low spending long trips: {len(low_spend_long)}, residual: ${low_residual:.2f}")
                
                if abs(high_residual - low_residual) > 30:
                    effect = high_residual - low_residual
                    self.edge_case_rules['long_trip_spending_interaction'] = {
                        'min_days': 7,
                        'high_spending_multiplier': 1.5,
                        'effect': effect,
                        'description': 'Interaction between long trips and high spending rates',
                        'affected_trips': len(high_spend_long)
                    }
                    print(f"  Long trip + high spending interaction detected: ${effect:.2f}")
    
    def test_edge_case_rules_impact(self):
        """Test the impact of applying all discovered edge case rules"""
        print("\n=== TESTING EDGE CASE RULES IMPACT ===")
        
        if not self.edge_case_rules:
            print("No edge case rules discovered")
            return
        
        # Apply edge case rules
        self.df['edge_case_adjustment'] = 0
        
        for rule_name, rule in self.edge_case_rules.items():
            adjustment = np.zeros(len(self.df))
            
            if rule_name == 'short_trip_penalty':
                mask = self.df['trip_duration_days'] <= rule['max_days']
                adjustment[mask] = -rule['penalty']
                
            elif rule_name == 'minimum_spending_penalty':
                mask = self.df['total_receipts_amount'] < rule['threshold']
                adjustment[mask] = -rule['penalty']
                
            elif rule_name == 'efficiency_adjustments':
                for category, effect_rule in rule.items():
                    if effect_rule['max_efficiency']:
                        mask = ((self.df['miles_per_day'] >= effect_rule['min_efficiency']) &
                               (self.df['miles_per_day'] < effect_rule['max_efficiency']))
                    else:
                        mask = self.df['miles_per_day'] >= effect_rule['min_efficiency']
                    
                    adjustment[mask] += effect_rule['adjustment']
                    
            elif rule_name == 'long_trip_spending_interaction':
                mask = ((self.df['trip_duration_days'] >= rule['min_days']) &
                       (self.df['receipts_per_day'] > 
                        self.df[self.df['trip_duration_days'] >= rule['min_days']]['receipts_per_day'].median() * rule['high_spending_multiplier']))
                adjustment[mask] = rule['effect']
            
            self.df['edge_case_adjustment'] += adjustment
            
            affected = (adjustment != 0).sum()
            avg_adj = adjustment[adjustment != 0].mean() if affected > 0 else 0
            print(f"{rule_name}: {affected} trips affected, avg adjustment: ${avg_adj:.2f}")
        
        # Calculate performance with edge case rules
        self.df['enhanced_prediction'] = self.df['current_prediction'] + self.df['edge_case_adjustment']
        
        # Apply rounding if detected
        if 'quarter_rounding' in self.edge_case_rules:
            self.df['enhanced_prediction'] = (self.df['enhanced_prediction'] * 4).round() / 4
        elif 'dollar_rounding' in self.edge_case_rules:
            self.df['enhanced_prediction'] = self.df['enhanced_prediction'].round()
        
        # Performance comparison
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        enhanced_r2 = r2_score(self.df['reimbursement'], self.df['enhanced_prediction'])
        
        current_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction']))
        enhanced_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['enhanced_prediction']))
        
        print(f"\n=== PERFORMANCE COMPARISON ===")
        print(f"Current model: R²={current_r2:.4f}, RMSE=${current_rmse:.2f}")
        print(f"With edge cases: R²={enhanced_r2:.4f}, RMSE=${enhanced_rmse:.2f}")
        print(f"Improvement: ΔR²={enhanced_r2-current_r2:.4f}, ΔRMSE=${current_rmse-enhanced_rmse:.2f}")
        
        return enhanced_r2, enhanced_rmse
    
    def run_edge_case_mining(self):
        """Run complete edge case business logic mining"""
        print("🔍 STARTING EDGE CASE BUSINESS LOGIC MINING")
        print("=" * 60)
        
        self.apply_existing_rules()
        self.mine_rounding_and_decimal_rules()
        self.mine_seasonal_budget_variations()
        self.mine_minimum_trip_thresholds()
        self.mine_travel_efficiency_bonuses()
        self.mine_complex_interaction_rules()
        
        if self.edge_case_rules:
            enhanced_r2, enhanced_rmse = self.test_edge_case_rules_impact()
        else:
            enhanced_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
            enhanced_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction']))
        
        print("\n" + "=" * 60)
        print("🎯 EDGE CASE BUSINESS LOGIC MINING COMPLETE")
        
        return self.edge_case_rules, enhanced_r2, enhanced_rmse
    
    def save_edge_case_rules(self, output_path='edge_case_business_rules.json'):
        """Save discovered edge case rules"""
        results = {
            'edge_case_rules': self.edge_case_rules,
            'existing_rules': self.existing_rules,
            'mining_summary': {
                'total_edge_rules': len(self.edge_case_rules),
                'rules_discovered': list(self.edge_case_rules.keys())
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Edge case rules saved to {output_path}")

if __name__ == "__main__":
    # Run edge case business logic mining
    miner = EdgeCaseBusinessLogicMiner()
    edge_rules, enhanced_r2, enhanced_rmse = miner.run_edge_case_mining()
    miner.save_edge_case_rules()
    
    print(f"\n📋 SUMMARY:")
    print(f"Enhanced model R²: {enhanced_r2:.4f}")
    print(f"Enhanced model RMSE: ${enhanced_rmse:.2f}")
    print(f"Edge case rules discovered: {len(edge_rules)}")
    
    for rule_name, rule in edge_rules.items():
        print(f"  {rule_name}: {rule.get('description', 'Complex rule')}")