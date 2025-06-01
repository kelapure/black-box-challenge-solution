#!/usr/bin/env python3
"""
Missing Business Logic Discovery for 95%+ Accuracy
Black Box Challenge - Final Business Rule Categories

After Task 2.5's failure (variable coefficients caused overfitting), this module
returns to simple, interpretable business logic to find the missing 9.28% gap
from 85.72% R² to reach 95% target.

Focus Areas:
1. Department-specific reimbursement patterns
2. Timing effects (quarterly, seasonal, fiscal year patterns)
3. Cumulative trip patterns (frequent traveler effects)
4. Geographic clustering effects
5. Sequential trip dependencies
6. Simple ratio-based rules we haven't discovered yet

Strategy: Keep it simple, interpretable, and focused on business logic.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MissingBusinessLogicDiscovery:
    """
    Discovers missing simple business logic categories for final accuracy push
    """
    
    def __init__(self, data_path='test_cases.json'):
        """Initialize with test cases"""
        self.data_path = data_path
        self.df = None
        self.missing_rules = {}
        self.load_data()
        
    def load_data(self):
        """Load test cases and apply current best system (precision-optimized from Task 2.4)"""
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
        
        # Apply best system so far (precision-optimized from Task 2.4 - 85.72% R²)
        self.apply_best_current_system()
        
    def apply_best_current_system(self):
        """Apply the best system so far (precision-optimized from Task 2.4)"""
        # Optimized linear baseline
        baseline = {
            'intercept': 318.40,
            'trip_duration_days': 71.28,
            'miles_traveled': 0.794100,
            'total_receipts_amount': 0.290366
        }
        
        # Calculate linear baseline
        self.df['linear_baseline'] = (
            baseline['intercept'] +
            baseline['trip_duration_days'] * self.df['trip_duration_days'] +
            baseline['miles_traveled'] * self.df['miles_traveled'] +
            baseline['total_receipts_amount'] * self.df['total_receipts_amount']
        )
        
        # Apply precision-optimized business rules and tier systems
        self.df['business_rule_adjustment'] = 0
        
        # Apply all the rules from Task 2.4 precision optimization
        self.apply_precision_optimized_rules()
        self.apply_tier_system_from_complex_rules()
        
        # Calculate current predictions and residuals
        self.df['pre_rounding_prediction'] = self.df['linear_baseline'] + self.df['business_rule_adjustment']
        self.df['current_prediction'] = (self.df['pre_rounding_prediction'] * 4).round() / 4
        self.df['residual'] = self.df['reimbursement'] - self.df['current_prediction']
        
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        current_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction']))
        
        print(f"Current best system: R²={current_r2:.4f}, RMSE=${current_rmse:.2f}")
        print(f"Gap to 95% target: {0.95 - current_r2:.4f}")
        
    def apply_precision_optimized_rules(self):
        """Apply all precision-optimized rules from Task 2.4"""
        # Optimized sweet spot (now penalty)
        mask = self.df['trip_duration_days'].isin([5, 6])
        self.df.loc[mask, 'business_rule_adjustment'] += -34.57
        
        # Optimized receipt ceiling (threshold $2025)
        mask = self.df['total_receipts_amount'] > 2025
        self.df.loc[mask, 'business_rule_adjustment'] -= 99.48
        
        # Optimized big trip jackpot (now minimal)
        mask = ((self.df['trip_duration_days'] >= 8) &
                (self.df['miles_traveled'] >= 900) &
                (self.df['total_receipts_amount'] >= 1200))
        self.df.loc[mask, 'business_rule_adjustment'] += -1.62
        
        # Optimized distance bonus (threshold 550, now penalty)
        mask = self.df['miles_traveled'] >= 550
        self.df.loc[mask, 'business_rule_adjustment'] += -41.02
        
        # Minimum spending penalty
        mask = self.df['total_receipts_amount'] < 103
        self.df.loc[mask, 'business_rule_adjustment'] -= 299.03
        
        # Optimized efficiency adjustments
        mask = (self.df['miles_per_day'] >= 94) & (self.df['miles_per_day'] < 180)
        self.df.loc[mask, 'business_rule_adjustment'] += -27.84
        
        mask = (self.df['miles_per_day'] >= 185) & (self.df['miles_per_day'] < 300)
        self.df.loc[mask, 'business_rule_adjustment'] += -41.29
        
        mask = self.df['miles_per_day'] < 102
        self.df.loc[mask, 'business_rule_adjustment'] += 47.70
        
        # Long trip high spending penalty
        mask = ((self.df['trip_duration_days'] >= 7) &
                (self.df['receipts_per_day'] > 178))
        self.df.loc[mask, 'business_rule_adjustment'] -= 89.41
        
    def apply_tier_system_from_complex_rules(self):
        """Apply tier system from complex rules discovery (Task 2.3)"""
        # Load complex rules if available
        try:
            with open('complex_business_rules.json', 'r') as f:
                complex_rules = json.load(f)
                
            if 'tier_systems' in complex_rules:
                tier_systems = complex_rules['tier_systems']
                
                # Miles tier system
                if 'miles_tier_system' in tier_systems:
                    miles_tiers = tier_systems['miles_tier_system']
                    for tier_name, tier_data in miles_tiers.items():
                        min_miles, max_miles = tier_data['range']
                        effect = tier_data['effect']
                        
                        if max_miles == float('inf'):
                            mask = self.df['miles_traveled'] >= min_miles
                        else:
                            mask = (self.df['miles_traveled'] >= min_miles) & (self.df['miles_traveled'] < max_miles)
                        
                        self.df.loc[mask, 'business_rule_adjustment'] += effect
                
                # Receipt tier system
                if 'receipt_tier_system' in tier_systems:
                    receipt_tiers = tier_systems['receipt_tier_system']
                    for tier_name, tier_data in receipt_tiers.items():
                        min_receipts, max_receipts = tier_data['range']
                        effect = tier_data['effect']
                        
                        if max_receipts == float('inf'):
                            mask = self.df['total_receipts_amount'] >= min_receipts
                        else:
                            mask = (self.df['total_receipts_amount'] >= min_receipts) & (self.df['total_receipts_amount'] < max_receipts)
                        
                        self.df.loc[mask, 'business_rule_adjustment'] += effect
                
                # Days tier system
                if 'days_tier_system' in tier_systems:
                    days_tiers = tier_systems['days_tier_system']
                    for tier_name, tier_data in days_tiers.items():
                        days = tier_data['days']
                        effect = tier_data['effect']
                        
                        mask = self.df['trip_duration_days'] == days
                        self.df.loc[mask, 'business_rule_adjustment'] += effect
                        
        except FileNotFoundError:
            print("Complex rules file not found, skipping tier systems")
            
    def discover_simple_ratio_based_rules(self):
        """Discover simple ratio-based business rules we haven't found yet"""
        print("\\n=== DISCOVERING SIMPLE RATIO-BASED RULES ===")
        
        ratio_rules = {}
        
        # 1. Total spending to days ratio (daily spending rate categories)
        print("\\n1. Daily spending rate categories:")
        daily_spending_thresholds = [50, 100, 150, 200, 250, 300]
        
        for i, threshold in enumerate(daily_spending_thresholds):
            if i == 0:
                mask = self.df['receipts_per_day'] < threshold
                category = f"ultra_frugal_<${threshold}"
            else:
                prev_threshold = daily_spending_thresholds[i-1]
                mask = (self.df['receipts_per_day'] >= prev_threshold) & (self.df['receipts_per_day'] < threshold)
                category = f"spending_${prev_threshold}-{threshold}"
            
            category_data = self.df[mask]
            if len(category_data) >= 30:
                avg_residual = category_data['residual'].mean()
                std_residual = category_data['residual'].std()
                
                print(f"  {category}: {len(category_data)} trips, avg residual ${avg_residual:.2f}")
                
                if abs(avg_residual) > 25:
                    ratio_rules[category] = {
                        'threshold': threshold,
                        'effect': avg_residual,
                        'count': len(category_data),
                        'std': std_residual
                    }
        
        # 2. Miles to receipts ratio (efficiency of spending)
        print("\\n2. Miles per dollar spent ratios:")
        self.df['miles_per_dollar'] = self.df['miles_traveled'] / (self.df['total_receipts_amount'] + 1)  # +1 to avoid division by zero
        
        miles_per_dollar_thresholds = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
        
        for i, threshold in enumerate(miles_per_dollar_thresholds):
            if i == 0:
                mask = self.df['miles_per_dollar'] < threshold
                category = f"expensive_travel_<{threshold}mi/$"
            else:
                prev_threshold = miles_per_dollar_thresholds[i-1]
                mask = (self.df['miles_per_dollar'] >= prev_threshold) & (self.df['miles_per_dollar'] < threshold)
                category = f"efficiency_{prev_threshold}-{threshold}mi/$"
            
            category_data = self.df[mask]
            if len(category_data) >= 50:
                avg_residual = category_data['residual'].mean()
                
                print(f"  {category}: {len(category_data)} trips, avg residual ${avg_residual:.2f}")
                
                if abs(avg_residual) > 20:
                    ratio_rules[category] = {
                        'threshold': threshold,
                        'effect': avg_residual,
                        'count': len(category_data)
                    }
        
        # 3. Trip intensity ratio (miles × days combined effect)
        print("\\n3. Trip intensity ratios:")
        self.df['trip_intensity'] = self.df['miles_traveled'] * self.df['trip_duration_days']
        
        intensity_thresholds = [200, 500, 1000, 2000, 5000, 10000]
        
        for i, threshold in enumerate(intensity_thresholds):
            if i == 0:
                mask = self.df['trip_intensity'] < threshold
                category = f"low_intensity_<{threshold}"
            else:
                prev_threshold = intensity_thresholds[i-1]
                mask = (self.df['trip_intensity'] >= prev_threshold) & (self.df['trip_intensity'] < threshold)
                category = f"intensity_{prev_threshold}-{threshold}"
            
            category_data = self.df[mask]
            if len(category_data) >= 40:
                avg_residual = category_data['residual'].mean()
                
                print(f"  {category}: {len(category_data)} trips, avg residual ${avg_residual:.2f}")
                
                if abs(avg_residual) > 30:
                    ratio_rules[category] = {
                        'threshold': threshold,
                        'effect': avg_residual,
                        'count': len(category_data)
                    }
        
        self.missing_rules['ratio_based_rules'] = ratio_rules
        
    def discover_sequential_trip_patterns(self):
        """Discover patterns based on trip sequences and ordering"""
        print("\\n=== DISCOVERING SEQUENTIAL TRIP PATTERNS ===")
        
        sequential_rules = {}
        
        # Assume trip order based on some implicit ordering in the data
        # (In real scenario, we'd have timestamp data)
        self.df['trip_order'] = range(len(self.df))
        
        # 1. Beginning vs end of dataset patterns (proxy for temporal effects)
        print("\\n1. Dataset position patterns (proxy for temporal effects):")
        
        dataset_segments = [
            ('early_trips', self.df['trip_order'] < len(self.df) * 0.2),
            ('mid_early_trips', (self.df['trip_order'] >= len(self.df) * 0.2) & (self.df['trip_order'] < len(self.df) * 0.4)),
            ('middle_trips', (self.df['trip_order'] >= len(self.df) * 0.4) & (self.df['trip_order'] < len(self.df) * 0.6)),
            ('mid_late_trips', (self.df['trip_order'] >= len(self.df) * 0.6) & (self.df['trip_order'] < len(self.df) * 0.8)),
            ('late_trips', self.df['trip_order'] >= len(self.df) * 0.8)
        ]
        
        for segment_name, segment_mask in dataset_segments:
            segment_data = self.df[segment_mask]
            avg_residual = segment_data['residual'].mean()
            
            print(f"  {segment_name}: {len(segment_data)} trips, avg residual ${avg_residual:.2f}")
            
            if abs(avg_residual) > 15:
                sequential_rules[segment_name] = {
                    'effect': avg_residual,
                    'count': len(segment_data),
                    'description': f"Dataset position effect: {segment_name}"
                }
        
        # 2. Trip index modulo patterns (periodic effects)
        print("\\n2. Periodic patterns (modulo effects):")
        
        for modulo in [3, 4, 5, 7, 10]:
            for remainder in range(modulo):
                mask = (self.df['trip_order'] % modulo) == remainder
                segment_data = self.df[mask]
                
                if len(segment_data) >= 100:
                    avg_residual = segment_data['residual'].mean()
                    
                    if abs(avg_residual) > 20:
                        pattern_name = f"modulo_{modulo}_remainder_{remainder}"
                        print(f"  {pattern_name}: {len(segment_data)} trips, avg residual ${avg_residual:.2f}")
                        
                        sequential_rules[pattern_name] = {
                            'modulo': modulo,
                            'remainder': remainder,
                            'effect': avg_residual,
                            'count': len(segment_data),
                            'description': f"Every {modulo} trips, position {remainder}"
                        }
        
        self.missing_rules['sequential_patterns'] = sequential_rules
        
    def discover_cumulative_trip_effects(self):
        """Discover cumulative trip effects (frequent traveler patterns)"""
        print("\\n=== DISCOVERING CUMULATIVE TRIP EFFECTS ===")
        
        cumulative_rules = {}
        
        # 1. Trip frequency simulation (based on similar characteristics)
        print("\\n1. Simulated frequent traveler patterns:")
        
        # Group trips by similar characteristics as proxy for same employee
        self.df['employee_proxy'] = (
            (self.df['trip_duration_days'] // 2) * 1000 +
            (self.df['miles_traveled'] // 100) * 10 +
            (self.df['total_receipts_amount'] // 200)
        )
        
        employee_trip_counts = self.df['employee_proxy'].value_counts()
        frequent_travelers = employee_trip_counts[employee_trip_counts >= 3].index
        
        if len(frequent_travelers) > 0:
            frequent_mask = self.df['employee_proxy'].isin(frequent_travelers)
            frequent_data = self.df[frequent_mask]
            infrequent_data = self.df[~frequent_mask]
            
            frequent_avg = frequent_data['residual'].mean()
            infrequent_avg = infrequent_data['residual'].mean()
            
            print(f"  Frequent travelers (3+ similar trips): {len(frequent_data)} trips, avg residual ${frequent_avg:.2f}")
            print(f"  Infrequent travelers: {len(infrequent_data)} trips, avg residual ${infrequent_avg:.2f}")
            
            if abs(frequent_avg - infrequent_avg) > 20:
                cumulative_rules['frequent_traveler_effect'] = {
                    'frequent_effect': frequent_avg,
                    'infrequent_effect': infrequent_avg,
                    'difference': frequent_avg - infrequent_avg,
                    'frequent_count': len(frequent_data),
                    'infrequent_count': len(infrequent_data)
                }
        
        # 2. Round number effects (psychological patterns)
        print("\\n2. Round number psychological effects:")
        
        round_number_patterns = [
            ('round_days', self.df['trip_duration_days'] % 5 == 0),
            ('round_miles_100s', self.df['miles_traveled'] % 100 == 0),
            ('round_miles_50s', self.df['miles_traveled'] % 50 == 0),
            ('round_receipts_100s', self.df['total_receipts_amount'] % 100 == 0),
            ('round_receipts_50s', self.df['total_receipts_amount'] % 50 == 0)
        ]
        
        for pattern_name, pattern_mask in round_number_patterns:
            round_data = self.df[pattern_mask]
            non_round_data = self.df[~pattern_mask]
            
            if len(round_data) >= 50 and len(non_round_data) >= 50:
                round_avg = round_data['residual'].mean()
                non_round_avg = non_round_data['residual'].mean()
                difference = round_avg - non_round_avg
                
                print(f"  {pattern_name}: round {len(round_data)} trips (${round_avg:.2f}), non-round {len(non_round_data)} trips (${non_round_avg:.2f}), diff ${difference:.2f}")
                
                if abs(difference) > 15:
                    cumulative_rules[pattern_name] = {
                        'round_effect': round_avg,
                        'non_round_effect': non_round_avg,
                        'difference': difference,
                        'round_count': len(round_data),
                        'non_round_count': len(non_round_data)
                    }
        
        self.missing_rules['cumulative_effects'] = cumulative_rules
        
    def discover_geographic_clustering_effects(self):
        """Discover geographic clustering effects (similar distances/locations)"""
        print("\\n=== DISCOVERING GEOGRAPHIC CLUSTERING EFFECTS ===")
        
        geographic_rules = {}
        
        # 1. Distance clustering (trips to similar distances may be same locations)
        print("\\n1. Distance clustering patterns:")
        
        # Group by similar distances (±25 miles)
        distance_clusters = defaultdict(list)
        
        for i, row in self.df.iterrows():
            miles = row['miles_traveled']
            cluster_key = round(miles / 25) * 25  # Round to nearest 25 miles
            distance_clusters[cluster_key].append(i)
        
        for cluster_distance, trip_indices in distance_clusters.items():
            if len(trip_indices) >= 20:  # Significant cluster
                cluster_data = self.df.iloc[trip_indices]
                avg_residual = cluster_data['residual'].mean()
                std_residual = cluster_data['residual'].std()
                
                print(f"  ~{cluster_distance} mile trips: {len(trip_indices)} trips, avg residual ${avg_residual:.2f}")
                
                if abs(avg_residual) > 25:
                    geographic_rules[f'distance_cluster_{cluster_distance}'] = {
                        'distance': cluster_distance,
                        'effect': avg_residual,
                        'count': len(trip_indices),
                        'std': std_residual
                    }
        
        # 2. Common trip duration + distance combinations (route patterns)
        print("\\n2. Common route patterns (days + distance combinations):")
        
        route_patterns = defaultdict(list)
        
        for i, row in self.df.iterrows():
            days = row['trip_duration_days']
            miles_category = "short" if row['miles_traveled'] < 300 else "medium" if row['miles_traveled'] < 700 else "long"
            pattern_key = f"{days}day_{miles_category}"
            route_patterns[pattern_key].append(i)
        
        for pattern_name, trip_indices in route_patterns.items():
            if len(trip_indices) >= 30:
                pattern_data = self.df.iloc[trip_indices]
                avg_residual = pattern_data['residual'].mean()
                
                print(f"  {pattern_name}: {len(trip_indices)} trips, avg residual ${avg_residual:.2f}")
                
                if abs(avg_residual) > 20:
                    geographic_rules[pattern_name] = {
                        'effect': avg_residual,
                        'count': len(trip_indices),
                        'description': f"Route pattern: {pattern_name}"
                    }
        
        self.missing_rules['geographic_clustering'] = geographic_rules
        
    def discover_mathematical_relationship_gaps(self):
        """Discover mathematical relationships we may have missed"""
        print("\\n=== DISCOVERING MATHEMATICAL RELATIONSHIP GAPS ===")
        
        math_rules = {}
        
        # 1. Non-linear transformations we haven't tried
        print("\\n1. Non-linear transformation patterns:")
        
        # Square root effects
        self.df['sqrt_days'] = np.sqrt(self.df['trip_duration_days'])
        self.df['sqrt_miles'] = np.sqrt(self.df['miles_traveled'])
        self.df['sqrt_receipts'] = np.sqrt(self.df['total_receipts_amount'])
        
        # Log effects (safe log with +1)
        self.df['log_days'] = np.log(self.df['trip_duration_days'] + 1)
        self.df['log_miles'] = np.log(self.df['miles_traveled'] + 1)
        self.df['log_receipts'] = np.log(self.df['total_receipts_amount'] + 1)
        
        # Test correlation with residuals
        transformations = [
            ('sqrt_days', 'Square root of days'),
            ('sqrt_miles', 'Square root of miles'),
            ('sqrt_receipts', 'Square root of receipts'),
            ('log_days', 'Log of days'),
            ('log_miles', 'Log of miles'),
            ('log_receipts', 'Log of receipts')
        ]
        
        for transform_col, description in transformations:
            correlation = self.df[transform_col].corr(self.df['residual'])
            
            print(f"  {description}: correlation with residual = {correlation:.4f}")
            
            if abs(correlation) > 0.1:
                # Find optimal coefficient for this transformation
                from sklearn.linear_model import LinearRegression
                X = self.df[[transform_col]].values
                y = self.df['residual'].values
                
                model = LinearRegression()
                model.fit(X, y)
                coef = model.coef_[0]
                
                # Test improvement
                transform_adjustment = coef * self.df[transform_col]
                test_pred = self.df['current_prediction'] + transform_adjustment
                test_pred_rounded = (test_pred * 4).round() / 4
                
                transform_r2 = r2_score(self.df['reimbursement'], test_pred_rounded)
                current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
                improvement = transform_r2 - current_r2
                
                print(f"    Coefficient: {coef:.6f}, R² improvement: {improvement:.6f}")
                
                if improvement > 0.001:
                    math_rules[transform_col] = {
                        'coefficient': coef,
                        'improvement': improvement,
                        'correlation': correlation,
                        'description': description
                    }
        
        # 2. Ratio-based mathematical patterns
        print("\\n2. Advanced ratio patterns:")
        
        # Efficiency ratios
        ratios = [
            ('total_efficiency', self.df['miles_traveled'] / (self.df['trip_duration_days'] * self.df['total_receipts_amount'] + 1)),
            ('cost_per_mile', self.df['total_receipts_amount'] / (self.df['miles_traveled'] + 1)),
            ('miles_squared_per_day', (self.df['miles_traveled'] ** 2) / self.df['trip_duration_days']),
            ('receipts_squared_per_day', (self.df['total_receipts_amount'] ** 2) / self.df['trip_duration_days'])
        ]
        
        for ratio_name, ratio_values in ratios:
            correlation = ratio_values.corr(self.df['residual'])
            
            print(f"  {ratio_name}: correlation with residual = {correlation:.4f}")
            
            if abs(correlation) > 0.08:
                # Find optimal coefficient
                from sklearn.linear_model import LinearRegression
                X = ratio_values.values.reshape(-1, 1)
                y = self.df['residual'].values
                
                model = LinearRegression()
                model.fit(X, y)
                coef = model.coef_[0]
                
                # Test improvement
                ratio_adjustment = coef * ratio_values
                test_pred = self.df['current_prediction'] + ratio_adjustment
                test_pred_rounded = (test_pred * 4).round() / 4
                
                ratio_r2 = r2_score(self.df['reimbursement'], test_pred_rounded)
                current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
                improvement = ratio_r2 - current_r2
                
                print(f"    Coefficient: {coef:.8f}, R² improvement: {improvement:.6f}")
                
                if improvement > 0.001:
                    math_rules[ratio_name] = {
                        'coefficient': coef,
                        'improvement': improvement,
                        'correlation': correlation,
                        'description': f"Advanced ratio: {ratio_name}"
                    }
        
        self.missing_rules['mathematical_relationships'] = math_rules
        
    def test_missing_business_logic_impact(self):
        """Test the combined impact of all discovered missing business logic"""
        print("\\n=== TESTING MISSING BUSINESS LOGIC IMPACT ===")
        
        # Start with current predictions
        enhanced_pred = self.df['current_prediction'].copy()
        total_improvements = []
        
        # Apply ratio-based rules
        if 'ratio_based_rules' in self.missing_rules:
            print("\\nApplying ratio-based rules:")
            ratio_rules = self.missing_rules['ratio_based_rules']
            
            for rule_name, rule_data in ratio_rules.items():
                if 'ultra_frugal' in rule_name:
                    mask = self.df['receipts_per_day'] < rule_data['threshold']
                elif 'spending_' in rule_name:
                    # Parse threshold range
                    range_parts = rule_name.split('_')[1].split('-')
                    min_threshold = int(range_parts[0].replace('$', ''))
                    max_threshold = int(range_parts[1])
                    mask = (self.df['receipts_per_day'] >= min_threshold) & (self.df['receipts_per_day'] < max_threshold)
                elif 'expensive_travel' in rule_name:
                    mask = self.df['miles_per_dollar'] < rule_data['threshold']
                elif 'efficiency_' in rule_name:
                    # Parse efficiency range
                    range_parts = rule_name.split('_')[1].split('-')
                    min_threshold = float(range_parts[0])
                    max_threshold = float(range_parts[1].replace('mi/$', ''))
                    mask = (self.df['miles_per_dollar'] >= min_threshold) & (self.df['miles_per_dollar'] < max_threshold)
                elif 'intensity_' in rule_name:
                    # Parse intensity range
                    range_parts = rule_name.split('_')[1].split('-')
                    min_threshold = int(range_parts[0])
                    max_threshold = int(range_parts[1])
                    mask = (self.df['trip_intensity'] >= min_threshold) & (self.df['trip_intensity'] < max_threshold)
                elif 'low_intensity' in rule_name:
                    mask = self.df['trip_intensity'] < rule_data['threshold']
                else:
                    continue
                
                enhanced_pred[mask] += rule_data['effect'] * 0.3  # Conservative application
                print(f"  Applied {rule_name}: {mask.sum()} trips affected")
        
        # Apply mathematical relationships
        if 'mathematical_relationships' in self.missing_rules:
            print("\\nApplying mathematical relationships:")
            math_rules = self.missing_rules['mathematical_relationships']
            
            for transform_name, transform_data in math_rules.items():
                coef = transform_data['coefficient']
                improvement = transform_data['improvement']
                
                if transform_name in self.df.columns:
                    adjustment = coef * self.df[transform_name] * 0.5  # Conservative application
                    enhanced_pred += adjustment
                    print(f"  Applied {transform_name}: coefficient {coef:.6f}")
                elif transform_name in ['total_efficiency', 'cost_per_mile', 'miles_squared_per_day', 'receipts_squared_per_day']:
                    # Recalculate the ratio
                    if transform_name == 'total_efficiency':
                        ratio_values = self.df['miles_traveled'] / (self.df['trip_duration_days'] * self.df['total_receipts_amount'] + 1)
                    elif transform_name == 'cost_per_mile':
                        ratio_values = self.df['total_receipts_amount'] / (self.df['miles_traveled'] + 1)
                    elif transform_name == 'miles_squared_per_day':
                        ratio_values = (self.df['miles_traveled'] ** 2) / self.df['trip_duration_days']
                    elif transform_name == 'receipts_squared_per_day':
                        ratio_values = (self.df['total_receipts_amount'] ** 2) / self.df['trip_duration_days']
                    
                    adjustment = coef * ratio_values * 0.5  # Conservative application
                    enhanced_pred += adjustment
                    print(f"  Applied {transform_name}: coefficient {coef:.8f}")
        
        # Apply quarter rounding
        enhanced_pred_rounded = (enhanced_pred * 4).round() / 4
        
        # Calculate performance metrics
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        enhanced_r2 = r2_score(self.df['reimbursement'], enhanced_pred_rounded)
        
        current_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction']))
        enhanced_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], enhanced_pred_rounded))
        
        current_mae = mean_absolute_error(self.df['reimbursement'], self.df['current_prediction'])
        enhanced_mae = mean_absolute_error(self.df['reimbursement'], enhanced_pred_rounded)
        
        # Exact match analysis
        exact_matches = (np.abs(self.df['reimbursement'] - enhanced_pred_rounded) < 0.01).sum()
        exact_match_rate = exact_matches / len(self.df) * 100
        
        within_1_dollar = (np.abs(self.df['reimbursement'] - enhanced_pred_rounded) <= 1.0).sum()
        within_5_dollars = (np.abs(self.df['reimbursement'] - enhanced_pred_rounded) <= 5.0).sum()
        within_10_dollars = (np.abs(self.df['reimbursement'] - enhanced_pred_rounded) <= 10.0).sum()
        
        print(f"\\n=== MISSING BUSINESS LOGIC ENHANCEMENT RESULTS ===")
        print(f"Current system:    R²={current_r2:.6f}, RMSE=${current_rmse:.2f}, MAE=${current_mae:.2f}")
        print(f"Enhanced system:   R²={enhanced_r2:.6f}, RMSE=${enhanced_rmse:.2f}, MAE=${enhanced_mae:.2f}")
        print(f"Improvement:       ΔR²={enhanced_r2-current_r2:.6f}, ΔRMSE=${current_rmse-enhanced_rmse:.2f}, ΔMAE=${current_mae-enhanced_mae:.2f}")
        
        print(f"\\nAccuracy Analysis:")
        print(f"  Exact matches: {exact_matches} ({exact_match_rate:.2f}%)")
        print(f"  Within $1: {within_1_dollar} ({within_1_dollar/len(self.df)*100:.1f}%)")
        print(f"  Within $5: {within_5_dollars} ({within_5_dollars/len(self.df)*100:.1f}%)")
        print(f"  Within $10: {within_10_dollars} ({within_10_dollars/len(self.df)*100:.1f}%)")
        
        target_r2 = 0.95
        if enhanced_r2 >= target_r2:
            print(f"\\n🎯 SUCCESS! Achieved {enhanced_r2:.4f} R² (≥95% target)")
        else:
            remaining_gap = target_r2 - enhanced_r2
            print(f"\\n⚠️  Still {remaining_gap:.4f} gap remaining to reach 95% target")
        
        return {
            'enhanced_r2': enhanced_r2,
            'enhanced_rmse': enhanced_rmse,
            'enhanced_mae': enhanced_mae,
            'improvement': enhanced_r2 - current_r2,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_match_rate,
            'within_tolerances': {
                'within_1_dollar': within_1_dollar,
                'within_5_dollars': within_5_dollars,
                'within_10_dollars': within_10_dollars
            },
            'target_achieved': enhanced_r2 >= target_r2
        }
        
    def run_missing_logic_discovery(self):
        """Run complete missing business logic discovery"""
        print("🔍 STARTING MISSING BUSINESS LOGIC DISCOVERY FOR 95%+ ACCURACY")
        print("=" * 70)
        
        self.discover_simple_ratio_based_rules()
        self.discover_sequential_trip_patterns()
        self.discover_cumulative_trip_effects()
        self.discover_geographic_clustering_effects()
        self.discover_mathematical_relationship_gaps()
        results = self.test_missing_business_logic_impact()
        
        print("\\n" + "=" * 70)
        print("🎯 MISSING BUSINESS LOGIC DISCOVERY COMPLETE")
        
        return {
            'missing_rules': self.missing_rules,
            'performance_results': results,
            'final_r2': results['enhanced_r2'],
            'target_achieved': results['target_achieved']
        }
        
    def save_missing_logic_results(self, output_path='missing_business_logic_results.json'):
        """Save missing business logic discovery results"""
        results = {
            'missing_rules': self.missing_rules,
            'discovery_summary': {
                'total_rule_categories': len(self.missing_rules),
                'total_rules_found': sum(len(category) for category in self.missing_rules.values()),
                'approach': 'Simple, interpretable business logic patterns'
            }
        }
        
        # Convert numpy types to Python types for JSON serialization
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
        print(f"\\n💾 Missing business logic results saved to {output_path}")

if __name__ == "__main__":
    # Run missing business logic discovery
    discoverer = MissingBusinessLogicDiscovery()
    results = discoverer.run_missing_logic_discovery()
    discoverer.save_missing_logic_results()
    
    print(f"\\n📋 MISSING BUSINESS LOGIC DISCOVERY SUMMARY:")
    print(f"Final R²: {results['final_r2']:.6f}")
    print(f"Target achieved: {results['target_achieved']}")
    print(f"Rule categories discovered: {len(results['missing_rules'])}")
    print(f"Total rules found: {sum(len(category) for category in results['missing_rules'].values())}")
    
    if results['target_achieved']:
        print("🎉 SUCCESS: 95%+ accuracy achieved with missing business logic!")
    else:
        remaining = 0.95 - results['final_r2']
        print(f"⚠️  Remaining gap: {remaining:.4f} to reach 95%")