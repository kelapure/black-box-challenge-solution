#!/usr/bin/env python3
"""
Complex Business Rule Discovery for 95%+ Accuracy
Black Box Challenge - Precision Enhancement

This module discovers complex business rules including:
1. Interaction effects between multiple variables
2. Conditional logic with nested rules  
3. Edge case refinements for extreme scenarios
4. Multi-threshold tier systems
5. Context-dependent adjustments

Based on residual analysis showing systematic biases, we implement sophisticated
but interpretable business logic to achieve production-level accuracy.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

class ComplexBusinessRuleDiscovery:
    """
    Discovers complex, multi-factor business rules for precision enhancement
    """
    
    def __init__(self, data_path='test_cases.json'):
        """Initialize with test cases"""
        self.data_path = data_path
        self.df = None
        self.complex_rules = {}
        self.interaction_rules = {}
        self.tier_systems = {}
        self.conditional_rules = {}
        self.load_data()
        
    def load_data(self):
        """Load test cases and apply current system"""
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
        
        # Apply current comprehensive system to get residuals
        self.apply_current_system()
        
    def apply_current_system(self):
        """Apply current comprehensive rule system"""
        # Linear baseline with optimized coefficients
        baseline = {
            'intercept': 318.40,
            'trip_duration_days': 71.28,
            'miles_traveled': 0.7941,
            'total_receipts_amount': 0.2904
        }
        
        self.df['linear_baseline'] = (
            baseline['intercept'] +
            baseline['trip_duration_days'] * self.df['trip_duration_days'] +
            baseline['miles_traveled'] * self.df['miles_traveled'] +
            baseline['total_receipts_amount'] * self.df['total_receipts_amount']
        )
        
        # Apply existing business rules
        self.df['business_rule_adjustment'] = 0
        
        # Sweet spot bonus (5-6 days)
        mask = self.df['trip_duration_days'].isin([5, 6])
        self.df.loc[mask, 'business_rule_adjustment'] += 78.85
        
        # Receipt ceiling penalty (>$2193)
        mask = self.df['total_receipts_amount'] > 2193
        self.df.loc[mask, 'business_rule_adjustment'] -= 99.48
        
        # Big trip jackpot (8+ days, 900+ miles, $1200+ receipts)
        mask = ((self.df['trip_duration_days'] >= 8) &
                (self.df['miles_traveled'] >= 900) &
                (self.df['total_receipts_amount'] >= 1200))
        self.df.loc[mask, 'business_rule_adjustment'] += 46.67
        
        # Distance bonus (600+ miles)
        mask = self.df['miles_traveled'] >= 600
        self.df.loc[mask, 'business_rule_adjustment'] += 28.75
        
        # Minimum spending penalty (<$103)
        mask = self.df['total_receipts_amount'] < 103
        self.df.loc[mask, 'business_rule_adjustment'] -= 299.03
        
        # Efficiency bonuses/penalties
        mask = (self.df['miles_per_day'] >= 100) & (self.df['miles_per_day'] < 200)
        self.df.loc[mask, 'business_rule_adjustment'] += 47.59
        
        mask = (self.df['miles_per_day'] >= 200) & (self.df['miles_per_day'] < 300)
        self.df.loc[mask, 'business_rule_adjustment'] += 33.17
        
        mask = self.df['miles_per_day'] < 100
        self.df.loc[mask, 'business_rule_adjustment'] -= 23.66
        
        # Long trip high spending penalty
        mask = ((self.df['trip_duration_days'] >= 7) &
                (self.df['receipts_per_day'] > 178))
        self.df.loc[mask, 'business_rule_adjustment'] -= 89.41
        
        # Calculate predictions and residuals
        self.df['pre_rounding_prediction'] = self.df['linear_baseline'] + self.df['business_rule_adjustment']
        self.df['current_prediction'] = (self.df['pre_rounding_prediction'] * 4).round() / 4
        self.df['residual'] = self.df['reimbursement'] - self.df['current_prediction']
        
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        print(f"Current system R²: {current_r2:.4f}")
        print(f"Need to improve by: {0.95 - current_r2:.4f} points to reach 95%")
        
    def discover_multi_threshold_tier_systems(self):
        """Discover sophisticated tier systems with multiple thresholds"""
        print("\n=== DISCOVERING MULTI-THRESHOLD TIER SYSTEMS ===")
        
        # 1. Miles-based tier system (addressing major systematic bias)
        print("\n1. Miles-based tier system:")
        mile_tiers = [
            (0, 50, 'ultra_short'),
            (50, 100, 'short'),
            (100, 200, 'medium_short'),
            (200, 300, 'medium'),
            (300, 500, 'medium_long'),
            (500, 800, 'long'),
            (800, 1200, 'very_long'),
            (1200, float('inf'), 'ultra_long')
        ]
        
        mile_tier_effects = {}
        base_residual = self.df['residual'].mean()
        
        for min_miles, max_miles, tier_name in mile_tiers:
            if max_miles == float('inf'):
                mask = self.df['miles_traveled'] >= min_miles
                tier_desc = f"{min_miles}+"
            else:
                mask = (self.df['miles_traveled'] >= min_miles) & (self.df['miles_traveled'] < max_miles)
                tier_desc = f"{min_miles}-{max_miles}"
            
            tier_data = self.df[mask]
            
            if len(tier_data) >= 30:
                tier_residual = tier_data['residual'].mean()
                tier_effect = tier_residual - base_residual
                
                print(f"  {tier_name} ({tier_desc} miles): effect ${tier_effect:.2f}, n={len(tier_data)}")
                
                if abs(tier_effect) > 10:
                    mile_tier_effects[tier_name] = {
                        'range': (min_miles, max_miles),
                        'effect': tier_effect,
                        'count': len(tier_data),
                        'description': f"Miles tier {tier_desc}"
                    }
        
        self.tier_systems['miles_tier_system'] = mile_tier_effects
        
        # 2. Receipt-based tier system (addressing receipt bias patterns)
        print("\n2. Receipt-based tier system:")
        receipt_tiers = [
            (0, 100, 'minimal'),
            (100, 300, 'low'),
            (300, 600, 'moderate'),
            (600, 1000, 'standard'),
            (1000, 1500, 'high'),
            (1500, 2000, 'premium'),
            (2000, 2500, 'luxury'),
            (2500, float('inf'), 'ultra_luxury')
        ]
        
        receipt_tier_effects = {}
        
        for min_receipts, max_receipts, tier_name in receipt_tiers:
            if max_receipts == float('inf'):
                mask = self.df['total_receipts_amount'] >= min_receipts
                tier_desc = f"${min_receipts}+"
            else:
                mask = (self.df['total_receipts_amount'] >= min_receipts) & (self.df['total_receipts_amount'] < max_receipts)
                tier_desc = f"${min_receipts}-{max_receipts}"
            
            tier_data = self.df[mask]
            
            if len(tier_data) >= 30:
                tier_residual = tier_data['residual'].mean()
                tier_effect = tier_residual - base_residual
                
                print(f"  {tier_name} ({tier_desc}): effect ${tier_effect:.2f}, n={len(tier_data)}")
                
                if abs(tier_effect) > 10:
                    receipt_tier_effects[tier_name] = {
                        'range': (min_receipts, max_receipts),
                        'effect': tier_effect,
                        'count': len(tier_data),
                        'description': f"Receipt tier {tier_desc}"
                    }
        
        self.tier_systems['receipt_tier_system'] = receipt_tier_effects
        
        # 3. Days-based tier system (addressing day-specific biases)
        print("\n3. Days-based tier system:")
        day_tier_effects = {}
        
        for days in range(1, 16):
            day_data = self.df[self.df['trip_duration_days'] == days]
            
            if len(day_data) >= 30:
                day_residual = day_data['residual'].mean()
                day_effect = day_residual - base_residual
                
                print(f"  {days} days: effect ${day_effect:.2f}, n={len(day_data)}")
                
                if abs(day_effect) > 15:
                    day_tier_effects[f'day_{days}'] = {
                        'days': days,
                        'effect': day_effect,
                        'count': len(day_data),
                        'description': f"{days}-day trip adjustment"
                    }
        
        self.tier_systems['days_tier_system'] = day_tier_effects
        
    def discover_complex_interaction_effects(self):
        """Discover complex multi-variable interaction effects"""
        print("\n=== DISCOVERING COMPLEX INTERACTION EFFECTS ===")
        
        # 1. Days × Miles × Receipts three-way interaction
        print("\n1. Three-way interaction: Days × Miles × Receipts")
        
        # Create categorical bins for interaction analysis
        self.df['days_cat'] = pd.cut(self.df['trip_duration_days'], 
                                    bins=[0, 3, 6, 10, float('inf')], 
                                    labels=['Short', 'Medium', 'Long', 'VeryLong'])
        
        self.df['miles_cat'] = pd.cut(self.df['miles_traveled'], 
                                     bins=[0, 300, 600, 1000, float('inf')], 
                                     labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        self.df['receipts_cat'] = pd.cut(self.df['total_receipts_amount'], 
                                        bins=[0, 500, 1200, 2000, float('inf')], 
                                        labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        three_way_interactions = {}
        base_residual = self.df['residual'].mean()
        
        for days_cat in ['Short', 'Medium', 'Long', 'VeryLong']:
            for miles_cat in ['Low', 'Medium', 'High', 'VeryHigh']:
                for receipts_cat in ['Low', 'Medium', 'High', 'VeryHigh']:
                    mask = ((self.df['days_cat'] == days_cat) & 
                           (self.df['miles_cat'] == miles_cat) & 
                           (self.df['receipts_cat'] == receipts_cat))
                    
                    interaction_data = self.df[mask]
                    
                    if len(interaction_data) >= 20:
                        interaction_residual = interaction_data['residual'].mean()
                        interaction_effect = interaction_residual - base_residual
                        
                        if abs(interaction_effect) > 30:
                            combination = f"{days_cat}Days_{miles_cat}Miles_{receipts_cat}Receipts"
                            
                            print(f"  {combination}: effect ${interaction_effect:.2f}, n={len(interaction_data)}")
                            
                            three_way_interactions[combination] = {
                                'days_category': days_cat,
                                'miles_category': miles_cat,
                                'receipts_category': receipts_cat,
                                'effect': interaction_effect,
                                'count': len(interaction_data),
                                'description': f"Three-way interaction: {combination}"
                            }
        
        self.interaction_rules['three_way_interactions'] = three_way_interactions
        
        # 2. Efficiency × Spending Rate interaction
        print("\n2. Efficiency × Spending Rate interaction:")
        
        self.df['efficiency_cat'] = pd.cut(self.df['miles_per_day'], 
                                          bins=[0, 75, 150, 250, float('inf')], 
                                          labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        self.df['spending_rate_cat'] = pd.cut(self.df['receipts_per_day'], 
                                             bins=[0, 100, 200, 300, float('inf')], 
                                             labels=['Frugal', 'Moderate', 'High', 'Luxury'])
        
        efficiency_spending_interactions = {}
        
        for eff_cat in ['Low', 'Medium', 'High', 'VeryHigh']:
            for spend_cat in ['Frugal', 'Moderate', 'High', 'Luxury']:
                mask = ((self.df['efficiency_cat'] == eff_cat) & 
                       (self.df['spending_rate_cat'] == spend_cat))
                
                interaction_data = self.df[mask]
                
                if len(interaction_data) >= 25:
                    interaction_residual = interaction_data['residual'].mean()
                    interaction_effect = interaction_residual - base_residual
                    
                    combination = f"{eff_cat}Efficiency_{spend_cat}Spending"
                    
                    print(f"  {combination}: effect ${interaction_effect:.2f}, n={len(interaction_data)}")
                    
                    if abs(interaction_effect) > 25:
                        efficiency_spending_interactions[combination] = {
                            'efficiency_category': eff_cat,
                            'spending_category': spend_cat,
                            'effect': interaction_effect,
                            'count': len(interaction_data),
                            'description': f"Efficiency-spending interaction: {combination}"
                        }
        
        self.interaction_rules['efficiency_spending_interactions'] = efficiency_spending_interactions
        
    def discover_conditional_nested_rules(self):
        """Discover conditional rules with nested logic"""
        print("\n=== DISCOVERING CONDITIONAL NESTED RULES ===")
        
        conditional_rules = {}
        
        # 1. Conditional spending rules based on trip length
        print("\n1. Conditional spending rules by trip length:")
        
        # Short trips (1-3 days) - different spending tolerance
        short_trips = self.df[self.df['trip_duration_days'] <= 3]
        if len(short_trips) > 100:
            short_low_spend = short_trips[short_trips['receipts_per_day'] < 100]
            short_high_spend = short_trips[short_trips['receipts_per_day'] >= 200]
            
            if len(short_low_spend) > 10 and len(short_high_spend) > 10:
                low_effect = short_low_spend['residual'].mean()
                high_effect = short_high_spend['residual'].mean()
                
                print(f"  Short trips, low spending: ${low_effect:.2f} (n={len(short_low_spend)})")
                print(f"  Short trips, high spending: ${high_effect:.2f} (n={len(short_high_spend)})")
                
                conditional_rules['short_trip_spending'] = {
                    'condition': 'trip_duration_days <= 3',
                    'nested_rules': {
                        'low_spending': {'threshold': 100, 'effect': low_effect},
                        'high_spending': {'threshold': 200, 'effect': high_effect}
                    },
                    'description': 'Short trips have different spending tolerance'
                }
        
        # Long trips (8+ days) - efficiency expectations
        long_trips = self.df[self.df['trip_duration_days'] >= 8]
        if len(long_trips) > 100:
            long_low_efficiency = long_trips[long_trips['miles_per_day'] < 75]
            long_high_efficiency = long_trips[long_trips['miles_per_day'] >= 150]
            
            if len(long_low_efficiency) > 10 and len(long_high_efficiency) > 10:
                low_eff_effect = long_low_efficiency['residual'].mean()
                high_eff_effect = long_high_efficiency['residual'].mean()
                
                print(f"  Long trips, low efficiency: ${low_eff_effect:.2f} (n={len(long_low_efficiency)})")
                print(f"  Long trips, high efficiency: ${high_eff_effect:.2f} (n={len(long_high_efficiency)})")
                
                conditional_rules['long_trip_efficiency'] = {
                    'condition': 'trip_duration_days >= 8',
                    'nested_rules': {
                        'low_efficiency': {'threshold': 75, 'effect': low_eff_effect},
                        'high_efficiency': {'threshold': 150, 'effect': high_eff_effect}
                    },
                    'description': 'Long trips have efficiency expectations'
                }
        
        # 2. Conditional mileage rules based on receipts
        print("\n2. Conditional mileage rules by receipt level:")
        
        # High receipt trips - mileage expectations differ
        high_receipt_trips = self.df[self.df['total_receipts_amount'] >= 1500]
        if len(high_receipt_trips) > 100:
            high_receipt_short_miles = high_receipt_trips[high_receipt_trips['miles_traveled'] < 300]
            high_receipt_long_miles = high_receipt_trips[high_receipt_trips['miles_traveled'] >= 800]
            
            if len(high_receipt_short_miles) > 10 and len(high_receipt_long_miles) > 10:
                short_miles_effect = high_receipt_short_miles['residual'].mean()
                long_miles_effect = high_receipt_long_miles['residual'].mean()
                
                print(f"  High receipts, short miles: ${short_miles_effect:.2f} (n={len(high_receipt_short_miles)})")
                print(f"  High receipts, long miles: ${long_miles_effect:.2f} (n={len(high_receipt_long_miles)})")
                
                conditional_rules['high_receipt_mileage'] = {
                    'condition': 'total_receipts_amount >= 1500',
                    'nested_rules': {
                        'short_miles': {'threshold': 300, 'effect': short_miles_effect},
                        'long_miles': {'threshold': 800, 'effect': long_miles_effect}
                    },
                    'description': 'High receipt trips have mileage expectations'
                }
        
        self.conditional_rules = conditional_rules
        
    def discover_edge_case_refinements(self):
        """Discover refinements for extreme edge cases"""
        print("\n=== DISCOVERING EDGE CASE REFINEMENTS ===")
        
        edge_cases = {}
        
        # 1. Ultra-short trips (1 day) with various characteristics
        print("\n1. Ultra-short trip refinements:")
        one_day_trips = self.df[self.df['trip_duration_days'] == 1]
        
        if len(one_day_trips) > 50:
            # Ultra-short high miles (likely one-day business blitz)
            one_day_high_miles = one_day_trips[one_day_trips['miles_traveled'] >= 200]
            one_day_low_miles = one_day_trips[one_day_trips['miles_traveled'] < 50]
            
            if len(one_day_high_miles) > 5:
                high_miles_effect = one_day_high_miles['residual'].mean()
                print(f"  1-day high miles (≥200): ${high_miles_effect:.2f} (n={len(one_day_high_miles)})")
                
                edge_cases['one_day_high_miles'] = {
                    'condition': 'trip_duration_days == 1 AND miles_traveled >= 200',
                    'effect': high_miles_effect,
                    'count': len(one_day_high_miles),
                    'description': 'One-day business blitz pattern'
                }
            
            if len(one_day_low_miles) > 5:
                low_miles_effect = one_day_low_miles['residual'].mean()
                print(f"  1-day low miles (<50): ${low_miles_effect:.2f} (n={len(one_day_low_miles)})")
                
                edge_cases['one_day_low_miles'] = {
                    'condition': 'trip_duration_days == 1 AND miles_traveled < 50',
                    'effect': low_miles_effect,
                    'count': len(one_day_low_miles),
                    'description': 'One-day local meeting pattern'
                }
        
        # 2. Ultra-long trips (12+ days) 
        print("\n2. Ultra-long trip refinements:")
        ultra_long_trips = self.df[self.df['trip_duration_days'] >= 12]
        
        if len(ultra_long_trips) > 30:
            ultra_long_avg = ultra_long_trips['residual'].mean()
            print(f"  12+ day trips: ${ultra_long_avg:.2f} (n={len(ultra_long_trips)})")
            
            if abs(ultra_long_avg) > 20:
                edge_cases['ultra_long_trips'] = {
                    'condition': 'trip_duration_days >= 12',
                    'effect': ultra_long_avg,
                    'count': len(ultra_long_trips),
                    'description': 'Ultra-long trip adjustment'
                }
        
        # 3. Ultra-high mileage (1500+ miles)
        print("\n3. Ultra-high mileage refinements:")
        ultra_high_miles = self.df[self.df['miles_traveled'] >= 1500]
        
        if len(ultra_high_miles) > 20:
            ultra_miles_avg = ultra_high_miles['residual'].mean()
            print(f"  1500+ mile trips: ${ultra_miles_avg:.2f} (n={len(ultra_high_miles)})")
            
            if abs(ultra_miles_avg) > 25:
                edge_cases['ultra_high_mileage'] = {
                    'condition': 'miles_traveled >= 1500',
                    'effect': ultra_miles_avg,
                    'count': len(ultra_high_miles),
                    'description': 'Ultra-high mileage adjustment'
                }
        
        # 4. Ultra-high receipts (3000+ dollars)
        print("\n4. Ultra-high receipt refinements:")
        ultra_high_receipts = self.df[self.df['total_receipts_amount'] >= 3000]
        
        if len(ultra_high_receipts) > 15:
            ultra_receipts_avg = ultra_high_receipts['residual'].mean()
            print(f"  $3000+ receipt trips: ${ultra_receipts_avg:.2f} (n={len(ultra_high_receipts)})")
            
            if abs(ultra_receipts_avg) > 30:
                edge_cases['ultra_high_receipts'] = {
                    'condition': 'total_receipts_amount >= 3000',
                    'effect': ultra_receipts_avg,
                    'count': len(ultra_high_receipts),
                    'description': 'Ultra-high receipt adjustment'
                }
        
        # 5. Extreme efficiency cases
        print("\n5. Extreme efficiency refinements:")
        ultra_low_efficiency = self.df[self.df['miles_per_day'] < 20]
        ultra_high_efficiency = self.df[self.df['miles_per_day'] >= 400]
        
        if len(ultra_low_efficiency) > 15:
            ultra_low_eff_avg = ultra_low_efficiency['residual'].mean()
            print(f"  Ultra-low efficiency (<20 mi/day): ${ultra_low_eff_avg:.2f} (n={len(ultra_low_efficiency)})")
            
            edge_cases['ultra_low_efficiency'] = {
                'condition': 'miles_per_day < 20',
                'effect': ultra_low_eff_avg,
                'count': len(ultra_low_efficiency),
                'description': 'Ultra-low efficiency penalty'
            }
        
        if len(ultra_high_efficiency) > 10:
            ultra_high_eff_avg = ultra_high_efficiency['residual'].mean()
            print(f"  Ultra-high efficiency (≥400 mi/day): ${ultra_high_eff_avg:.2f} (n={len(ultra_high_efficiency)})")
            
            edge_cases['ultra_high_efficiency'] = {
                'condition': 'miles_per_day >= 400',
                'effect': ultra_high_eff_avg,
                'count': len(ultra_high_efficiency),
                'description': 'Ultra-high efficiency bonus'
            }
        
        self.complex_rules['edge_case_refinements'] = edge_cases
        
    def estimate_complex_rules_impact(self):
        """Estimate the potential impact of implementing all discovered complex rules"""
        print("\n=== ESTIMATING COMPLEX RULES IMPACT ===")
        
        total_rules = 0
        total_affected_trips = 0
        estimated_variance_reduction = 0
        
        # Count tier system rules
        for tier_system, tiers in self.tier_systems.items():
            tier_count = len(tiers)
            tier_trips = sum(tier['count'] for tier in tiers.values())
            total_rules += tier_count
            total_affected_trips += tier_trips
            
            print(f"{tier_system}: {tier_count} tiers, {tier_trips} trips affected")
        
        # Count interaction rules
        for interaction_type, interactions in self.interaction_rules.items():
            interaction_count = len(interactions)
            interaction_trips = sum(interaction['count'] for interaction in interactions.values())
            total_rules += interaction_count
            total_affected_trips += interaction_trips
            
            print(f"{interaction_type}: {interaction_count} interactions, {interaction_trips} trips affected")
        
        # Count conditional rules
        conditional_count = len(self.conditional_rules)
        if conditional_count > 0:
            total_rules += conditional_count
            print(f"Conditional rules: {conditional_count} rules")
        
        # Count edge case rules
        if 'edge_case_refinements' in self.complex_rules:
            edge_case_count = len(self.complex_rules['edge_case_refinements'])
            edge_case_trips = sum(edge['count'] for edge in self.complex_rules['edge_case_refinements'].values())
            total_rules += edge_case_count
            total_affected_trips += edge_case_trips
            
            print(f"Edge case refinements: {edge_case_count} rules, {edge_case_trips} trips affected")
        
        # Estimate variance reduction potential
        current_residual_variance = np.var(self.df['residual'])
        large_residual_trips = len(self.df[np.abs(self.df['residual']) > 100])
        
        # Conservative estimate: complex rules could address 60% of large residuals
        potential_improvement = 0.6 * (large_residual_trips / len(self.df))
        estimated_r2_gain = potential_improvement * 0.1147  # Remaining gap to 95%
        
        print(f"\nComplex Rules Summary:")
        print(f"  Total new rules: {total_rules}")
        print(f"  Total trips affected: {total_affected_trips} ({total_affected_trips/len(self.df)*100:.1f}%)")
        print(f"  Large residual trips: {large_residual_trips} ({large_residual_trips/len(self.df)*100:.1f}%)")
        print(f"  Estimated R² gain: {estimated_r2_gain:.4f}")
        print(f"  Projected R² with complex rules: {0.8353 + estimated_r2_gain:.4f}")
        
        if 0.8353 + estimated_r2_gain >= 0.92:
            print(f"  🎯 PROMISING: Complex rules may get us close to 95% target!")
        else:
            print(f"  ⚠️  Complex rules alone may not reach 95% - need additional precision tuning")
        
        return {
            'total_rules': total_rules,
            'total_affected_trips': total_affected_trips,
            'estimated_r2_gain': estimated_r2_gain,
            'projected_r2': 0.8353 + estimated_r2_gain
        }
        
    def run_complex_rule_discovery(self):
        """Run complete complex business rule discovery"""
        print("🔍 STARTING COMPLEX BUSINESS RULE DISCOVERY FOR 95%+ ACCURACY")
        print("=" * 75)
        
        self.discover_multi_threshold_tier_systems()
        self.discover_complex_interaction_effects()
        self.discover_conditional_nested_rules()
        self.discover_edge_case_refinements()
        impact_estimate = self.estimate_complex_rules_impact()
        
        print("\n" + "=" * 75)
        print("🎯 COMPLEX BUSINESS RULE DISCOVERY COMPLETE")
        
        return {
            'tier_systems': self.tier_systems,
            'interaction_rules': self.interaction_rules,
            'conditional_rules': self.conditional_rules,
            'complex_rules': self.complex_rules,
            'impact_estimate': impact_estimate
        }
        
    def save_complex_rules(self, output_path='complex_business_rules.json'):
        """Save all discovered complex rules"""
        results = {
            'tier_systems': self.tier_systems,
            'interaction_rules': self.interaction_rules,
            'conditional_rules': self.conditional_rules,
            'complex_rules': self.complex_rules,
            'discovery_summary': {
                'total_tier_systems': len(self.tier_systems),
                'total_interaction_types': len(self.interaction_rules),
                'total_conditional_rules': len(self.conditional_rules),
                'total_edge_cases': len(self.complex_rules.get('edge_case_refinements', {})),
                'complexity_level': 'High - sophisticated multi-factor business logic'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Complex business rules saved to {output_path}")

if __name__ == "__main__":
    # Run complex business rule discovery
    discoverer = ComplexBusinessRuleDiscovery()
    results = discoverer.run_complex_rule_discovery()
    discoverer.save_complex_rules()
    
    print(f"\n📋 COMPLEX RULES DISCOVERY SUMMARY:")
    print(f"Tier systems: {len(results['tier_systems'])}")
    print(f"Interaction rule types: {len(results['interaction_rules'])}")
    print(f"Conditional rules: {len(results['conditional_rules'])}")
    print(f"Edge case refinements: {len(results['complex_rules'].get('edge_case_refinements', {}))}")
    print(f"Total new rules: {results['impact_estimate']['total_rules']}")
    print(f"Projected R² with complex rules: {results['impact_estimate']['projected_r2']:.4f}")