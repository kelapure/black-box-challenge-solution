#!/usr/bin/env python3
"""
Deep Residual Analysis for High-Precision Business Logic Discovery
Black Box Challenge - Targeting 95%+ Accuracy

This module performs exhaustive residual analysis to identify missing business logic
patterns that explain the remaining 16.47% variance in our current 83.53% model.

Strategy:
1. Systematic analysis of large residuals (>$100, >$200, >$500)
2. Pattern discovery in residual distributions by input combinations
3. Identification of systematic biases indicating missing rules
4. Discovery of complex interaction effects not yet captured
5. Quantitative extraction of precise thresholds and coefficients from data
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
# Removed matplotlib/seaborn dependencies to avoid import issues
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DeepResidualAnalyzer:
    """
    Performs deep residual analysis to identify missing business logic patterns
    """
    
    def __init__(self, data_path='test_cases.json', rule_system_path='comprehensive_rule_system.json'):
        """Initialize with test cases and current rule system"""
        self.data_path = data_path
        self.rule_system_path = rule_system_path
        self.df = None
        self.current_rules = None
        self.missing_patterns = {}
        self.precision_opportunities = {}
        self.load_data()
        
    def load_data(self):
        """Load test cases and current rule system"""
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
        
        # Load current rule system and apply it
        try:
            with open(self.rule_system_path, 'r') as f:
                self.current_rules = json.load(f)
            print("Loaded current rule system")
        except FileNotFoundError:
            print("No rule system found, will use basic linear baseline")
            self.current_rules = None
            
        self.apply_current_system()
        
    def apply_current_system(self):
        """Apply current comprehensive rule system to get predictions and residuals"""
        # Use the optimized linear baseline
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
        
        # Apply business rules (replicating comprehensive system)
        self.df['business_rule_adjustment'] = 0
        
        # Rule 1: Sweet Spot Bonus (5-6 days)
        mask = self.df['trip_duration_days'].isin([5, 6])
        self.df.loc[mask, 'business_rule_adjustment'] += 78.85
        
        # Rule 2: Receipt Ceiling Penalty (>$2193)
        mask = self.df['total_receipts_amount'] > 2193
        self.df.loc[mask, 'business_rule_adjustment'] -= 99.48
        
        # Rule 3: Big Trip Jackpot (8+ days, 900+ miles, $1200+ receipts)
        mask = ((self.df['trip_duration_days'] >= 8) &
                (self.df['miles_traveled'] >= 900) &
                (self.df['total_receipts_amount'] >= 1200))
        self.df.loc[mask, 'business_rule_adjustment'] += 46.67
        
        # Rule 4: Distance Bonus (600+ miles)
        mask = self.df['miles_traveled'] >= 600
        self.df.loc[mask, 'business_rule_adjustment'] += 28.75
        
        # Rule 5: Minimum Spending Penalty (<$103)
        mask = self.df['total_receipts_amount'] < 103
        self.df.loc[mask, 'business_rule_adjustment'] -= 299.03
        
        # Rule 6: Efficiency Bonuses/Penalties
        mask = (self.df['miles_per_day'] >= 100) & (self.df['miles_per_day'] < 200)
        self.df.loc[mask, 'business_rule_adjustment'] += 47.59
        
        mask = (self.df['miles_per_day'] >= 200) & (self.df['miles_per_day'] < 300)
        self.df.loc[mask, 'business_rule_adjustment'] += 33.17
        
        mask = self.df['miles_per_day'] < 100
        self.df.loc[mask, 'business_rule_adjustment'] -= 23.66
        
        # Rule 7: Long Trip High Spending Penalty
        mask = ((self.df['trip_duration_days'] >= 7) &
                (self.df['receipts_per_day'] > 178))
        self.df.loc[mask, 'business_rule_adjustment'] -= 89.41
        
        # Calculate predictions with quarter rounding
        self.df['pre_rounding_prediction'] = self.df['linear_baseline'] + self.df['business_rule_adjustment']
        self.df['current_prediction'] = (self.df['pre_rounding_prediction'] * 4).round() / 4
        self.df['residual'] = self.df['reimbursement'] - self.df['current_prediction']
        
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        current_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction']))
        
        print(f"Current system performance: R² = {current_r2:.4f}, RMSE = ${current_rmse:.2f}")
        print(f"Target improvement needed: {0.95 - current_r2:.4f} R² points to reach 95%")
        
    def analyze_large_residuals(self):
        """Analyze cases with large residuals to identify missing patterns"""
        print("\n=== ANALYZING LARGE RESIDUALS ===")
        
        # Define residual thresholds
        residual_thresholds = [50, 100, 200, 500]
        
        for threshold in residual_thresholds:
            large_residuals = self.df[np.abs(self.df['residual']) > threshold]
            
            if len(large_residuals) == 0:
                continue
                
            print(f"\nResiduals > ${threshold}:")
            print(f"  Count: {len(large_residuals)} ({len(large_residuals)/len(self.df)*100:.1f}%)")
            print(f"  Mean absolute residual: ${np.abs(large_residuals['residual']).mean():.2f}")
            print(f"  Max residual: ${large_residuals['residual'].max():.2f}")
            print(f"  Min residual: ${large_residuals['residual'].min():.2f}")
            
            # Analyze patterns in large residuals
            print(f"  Trip duration distribution:")
            duration_dist = large_residuals['trip_duration_days'].value_counts().sort_index()
            for days, count in duration_dist.head(10).items():
                pct = count / len(large_residuals) * 100
                print(f"    {days} days: {count} trips ({pct:.1f}%)")
            
            print(f"  Miles traveled ranges:")
            mile_ranges = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, float('inf'))]
            for min_m, max_m in mile_ranges:
                if max_m == float('inf'):
                    mask = large_residuals['miles_traveled'] >= min_m
                    range_str = f"{min_m}+"
                else:
                    mask = (large_residuals['miles_traveled'] >= min_m) & (large_residuals['miles_traveled'] < max_m)
                    range_str = f"{min_m}-{max_m}"
                
                count = mask.sum()
                if count > 0:
                    pct = count / len(large_residuals) * 100
                    avg_residual = large_residuals[mask]['residual'].mean()
                    print(f"    {range_str} miles: {count} trips ({pct:.1f}%), avg residual: ${avg_residual:.2f}")
    
    def discover_systematic_biases(self):
        """Discover systematic biases indicating missing business rules"""
        print("\n=== DISCOVERING SYSTEMATIC BIASES ===")
        
        # Analyze residuals by input value ranges to find systematic patterns
        
        # 1. Days-based systematic patterns
        print("\n1. Days-based patterns:")
        days_analysis = self.df.groupby('trip_duration_days')['residual'].agg(['mean', 'std', 'count'])
        
        for days in range(1, 15):
            if days in days_analysis.index and days_analysis.loc[days, 'count'] >= 10:
                mean_res = days_analysis.loc[days, 'mean']
                count = days_analysis.loc[days, 'count']
                if abs(mean_res) > 20:  # Significant bias
                    print(f"  {days} days: avg residual ${mean_res:.2f} (n={count}) - SYSTEMATIC BIAS")
        
        # 2. Miles-based systematic patterns
        print("\n2. Miles-based patterns:")
        self.df['miles_bin'] = pd.cut(self.df['miles_traveled'], 
                                     bins=[0, 50, 100, 200, 300, 500, 800, 1200, float('inf')],
                                     labels=['0-50', '50-100', '100-200', '200-300', '300-500', '500-800', '800-1200', '1200+'])
        
        miles_analysis = self.df.groupby('miles_bin')['residual'].agg(['mean', 'std', 'count'])
        
        for bin_name, stats in miles_analysis.iterrows():
            if stats['count'] >= 20 and abs(stats['mean']) > 15:
                print(f"  {bin_name} miles: avg residual ${stats['mean']:.2f} (n={stats['count']}) - SYSTEMATIC BIAS")
        
        # 3. Receipts-based systematic patterns
        print("\n3. Receipts-based patterns:")
        self.df['receipts_bin'] = pd.cut(self.df['total_receipts_amount'],
                                       bins=[0, 100, 300, 600, 1000, 1500, 2000, 3000, float('inf')],
                                       labels=['0-100', '100-300', '300-600', '600-1000', '1000-1500', '1500-2000', '2000-3000', '3000+'])
        
        receipts_analysis = self.df.groupby('receipts_bin')['residual'].agg(['mean', 'std', 'count'])
        
        for bin_name, stats in receipts_analysis.iterrows():
            if stats['count'] >= 20 and abs(stats['mean']) > 15:
                print(f"  {bin_name} receipts: avg residual ${stats['mean']:.2f} (n={stats['count']}) - SYSTEMATIC BIAS")
        
        # 4. Efficiency-based patterns (refined)
        print("\n4. Efficiency-based patterns (refined):")
        self.df['efficiency_bin'] = pd.cut(self.df['miles_per_day'],
                                         bins=[0, 25, 50, 75, 100, 150, 200, 300, 500, float('inf')],
                                         labels=['0-25', '25-50', '50-75', '75-100', '100-150', '150-200', '200-300', '300-500', '500+'])
        
        efficiency_analysis = self.df.groupby('efficiency_bin')['residual'].agg(['mean', 'std', 'count'])
        
        for bin_name, stats in efficiency_analysis.iterrows():
            if stats['count'] >= 10 and abs(stats['mean']) > 15:
                print(f"  {bin_name} miles/day: avg residual ${stats['mean']:.2f} (n={stats['count']}) - SYSTEMATIC BIAS")
    
    def discover_interaction_effects(self):
        """Discover complex interaction effects between variables"""
        print("\n=== DISCOVERING INTERACTION EFFECTS ===")
        
        # 1. Days × Miles interaction
        print("\n1. Days × Miles interaction patterns:")
        
        # Create interaction categories
        self.df['days_category'] = pd.cut(self.df['trip_duration_days'], bins=[0, 3, 6, 10, float('inf')], labels=['Short', 'Medium', 'Long', 'VeryLong'])
        self.df['miles_category'] = pd.cut(self.df['miles_traveled'], bins=[0, 200, 600, 1200, float('inf')], labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        interaction_analysis = self.df.groupby(['days_category', 'miles_category'])['residual'].agg(['mean', 'count'])
        
        for (days_cat, miles_cat), stats in interaction_analysis.iterrows():
            if stats['count'] >= 15 and abs(stats['mean']) > 25:
                print(f"  {days_cat} days × {miles_cat} miles: avg residual ${stats['mean']:.2f} (n={stats['count']}) - INTERACTION EFFECT")
        
        # 2. Days × Receipts interaction
        print("\n2. Days × Receipts interaction patterns:")
        self.df['receipts_category'] = pd.cut(self.df['total_receipts_amount'], bins=[0, 500, 1200, 2000, float('inf')], labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        interaction_analysis = self.df.groupby(['days_category', 'receipts_category'])['residual'].agg(['mean', 'count'])
        
        for (days_cat, receipts_cat), stats in interaction_analysis.iterrows():
            if stats['count'] >= 15 and abs(stats['mean']) > 25:
                print(f"  {days_cat} days × {receipts_cat} receipts: avg residual ${stats['mean']:.2f} (n={stats['count']}) - INTERACTION EFFECT")
        
        # 3. Miles × Receipts interaction
        print("\n3. Miles × Receipts interaction patterns:")
        
        interaction_analysis = self.df.groupby(['miles_category', 'receipts_category'])['residual'].agg(['mean', 'count'])
        
        for (miles_cat, receipts_cat), stats in interaction_analysis.iterrows():
            if stats['count'] >= 15 and abs(stats['mean']) > 25:
                print(f"  {miles_cat} miles × {receipts_cat} receipts: avg residual ${stats['mean']:.2f} (n={stats['count']}) - INTERACTION EFFECT")
    
    def discover_missing_thresholds(self):
        """Discover missing threshold effects through changepoint analysis"""
        print("\n=== DISCOVERING MISSING THRESHOLDS ===")
        
        # 1. Days threshold analysis
        print("\n1. Days threshold analysis:")
        days_residuals = self.df.groupby('trip_duration_days')['residual'].mean()
        
        # Look for sudden changes in residual patterns
        for days in range(1, 20):
            if days in days_residuals.index and (days + 1) in days_residuals.index:
                change = days_residuals[days + 1] - days_residuals[days]
                if abs(change) > 30:  # Significant threshold effect
                    print(f"  Day {days} → {days+1}: residual change ${change:.2f} - POTENTIAL THRESHOLD")
        
        # 2. Miles threshold analysis
        print("\n2. Miles threshold analysis:")
        
        # Test specific thresholds mentioned in interviews or found in patterns
        test_thresholds = [50, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 900, 1000, 1200]
        
        for threshold in test_thresholds:
            below = self.df[self.df['miles_traveled'] <= threshold]
            above = self.df[self.df['miles_traveled'] > threshold]
            
            if len(below) >= 50 and len(above) >= 50:
                below_mean = below['residual'].mean()
                above_mean = above['residual'].mean()
                difference = above_mean - below_mean
                
                if abs(difference) > 20:
                    print(f"  Miles threshold {threshold}: below=${below_mean:.2f}, above=${above_mean:.2f}, diff=${difference:.2f} - THRESHOLD EFFECT")
        
        # 3. Receipts threshold analysis
        print("\n3. Receipts threshold analysis:")
        
        test_thresholds = [50, 100, 200, 300, 500, 800, 1000, 1200, 1500, 2000, 2500, 3000]
        
        for threshold in test_thresholds:
            below = self.df[self.df['total_receipts_amount'] <= threshold]
            above = self.df[self.df['total_receipts_amount'] > threshold]
            
            if len(below) >= 50 and len(above) >= 50:
                below_mean = below['residual'].mean()
                above_mean = above['residual'].mean()
                difference = above_mean - below_mean
                
                if abs(difference) > 20:
                    print(f"  Receipts threshold ${threshold}: below=${below_mean:.2f}, above=${above_mean:.2f}, diff=${difference:.2f} - THRESHOLD EFFECT")
    
    def discover_precision_coefficient_adjustments(self):
        """Discover precise coefficient adjustments needed for higher accuracy"""
        print("\n=== DISCOVERING PRECISION COEFFICIENT ADJUSTMENTS ===")
        
        # Analyze residuals vs each input variable to find coefficient adjustments
        
        # 1. Days coefficient adjustment
        days_correlation = self.df['residual'].corr(self.df['trip_duration_days'])
        print(f"\n1. Days coefficient adjustment:")
        print(f"  Current coefficient: $71.28")
        print(f"  Residual correlation with days: {days_correlation:.4f}")
        
        if abs(days_correlation) > 0.05:
            # Calculate suggested adjustment
            residual_per_day = self.df['residual'].sum() / self.df['trip_duration_days'].sum()
            suggested_adjustment = residual_per_day * days_correlation * 10  # Scale factor
            new_coefficient = 71.28 + suggested_adjustment
            print(f"  Suggested adjustment: {suggested_adjustment:+.4f}")
            print(f"  New coefficient: ${new_coefficient:.4f}")
        
        # 2. Miles coefficient adjustment
        miles_correlation = self.df['residual'].corr(self.df['miles_traveled'])
        print(f"\n2. Miles coefficient adjustment:")
        print(f"  Current coefficient: $0.7941")
        print(f"  Residual correlation with miles: {miles_correlation:.4f}")
        
        if abs(miles_correlation) > 0.05:
            residual_per_mile = self.df['residual'].sum() / self.df['miles_traveled'].sum()
            suggested_adjustment = residual_per_mile * miles_correlation * 10
            new_coefficient = 0.7941 + suggested_adjustment
            print(f"  Suggested adjustment: {suggested_adjustment:+.6f}")
            print(f"  New coefficient: ${new_coefficient:.6f}")
        
        # 3. Receipts coefficient adjustment
        receipts_correlation = self.df['residual'].corr(self.df['total_receipts_amount'])
        print(f"\n3. Receipts coefficient adjustment:")
        print(f"  Current coefficient: $0.2904")
        print(f"  Residual correlation with receipts: {receipts_correlation:.4f}")
        
        if abs(receipts_correlation) > 0.05:
            residual_per_receipt = self.df['residual'].sum() / self.df['total_receipts_amount'].sum()
            suggested_adjustment = residual_per_receipt * receipts_correlation * 10
            new_coefficient = 0.2904 + suggested_adjustment
            print(f"  Suggested adjustment: {suggested_adjustment:+.6f}")
            print(f"  New coefficient: ${new_coefficient:.6f}")
    
    def identify_outlier_patterns(self):
        """Identify patterns in extreme outliers that may indicate special cases"""
        print("\n=== IDENTIFYING OUTLIER PATTERNS ===")
        
        # Find extreme outliers (beyond 2 standard deviations)
        residual_std = self.df['residual'].std()
        residual_mean = self.df['residual'].mean()
        
        outlier_threshold = 2 * residual_std
        outliers = self.df[np.abs(self.df['residual'] - residual_mean) > outlier_threshold]
        
        print(f"Extreme outliers (>{outlier_threshold:.0f} from mean):")
        print(f"  Count: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)")
        
        if len(outliers) > 0:
            # Analyze extreme positive outliers
            positive_outliers = outliers[outliers['residual'] > 0]
            print(f"\nPositive outliers (system under-predicts): {len(positive_outliers)}")
            
            if len(positive_outliers) > 0:
                print(f"  Avg days: {positive_outliers['trip_duration_days'].mean():.1f}")
                print(f"  Avg miles: {positive_outliers['miles_traveled'].mean():.1f}")
                print(f"  Avg receipts: ${positive_outliers['total_receipts_amount'].mean():.0f}")
                print(f"  Avg efficiency: {positive_outliers['miles_per_day'].mean():.1f}")
                print(f"  Avg residual: ${positive_outliers['residual'].mean():.2f}")
            
            # Analyze extreme negative outliers
            negative_outliers = outliers[outliers['residual'] < 0]
            print(f"\nNegative outliers (system over-predicts): {len(negative_outliers)}")
            
            if len(negative_outliers) > 0:
                print(f"  Avg days: {negative_outliers['trip_duration_days'].mean():.1f}")
                print(f"  Avg miles: {negative_outliers['miles_traveled'].mean():.1f}")
                print(f"  Avg receipts: ${negative_outliers['total_receipts_amount'].mean():.0f}")
                print(f"  Avg efficiency: {negative_outliers['miles_per_day'].mean():.1f}")
                print(f"  Avg residual: ${negative_outliers['residual'].mean():.2f}")
    
    def quantify_improvement_potential(self):
        """Quantify the potential R² improvement from addressing identified patterns"""
        print("\n=== QUANTIFYING IMPROVEMENT POTENTIAL ===")
        
        current_r2 = r2_score(self.df['reimbursement'], self.df['current_prediction'])
        total_variance = np.var(self.df['reimbursement'])
        unexplained_variance = np.var(self.df['residual'])
        
        print(f"Current model:")
        print(f"  R²: {current_r2:.4f}")
        print(f"  Unexplained variance: ${unexplained_variance:.0f}")
        print(f"  RMSE: ${np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction'])):.2f}")
        
        # Estimate potential improvement from systematic biases
        large_residuals = self.df[np.abs(self.df['residual']) > 100]
        if len(large_residuals) > 0:
            large_residual_variance = np.var(large_residuals['residual'])
            potential_r2_gain = large_residual_variance / total_variance
            
            print(f"\nImprovement potential from large residuals (>${100}+):")
            print(f"  Large residuals: {len(large_residuals)} ({len(large_residuals)/len(self.df)*100:.1f}%)")
            print(f"  Large residual variance: ${large_residual_variance:.0f}")
            print(f"  Potential R² gain: {potential_r2_gain:.4f}")
            print(f"  Target R² if addressed: {current_r2 + potential_r2_gain:.4f}")
        
        # Calculate how much improvement we need
        target_r2 = 0.95
        needed_improvement = target_r2 - current_r2
        needed_variance_reduction = needed_improvement * total_variance
        
        print(f"\nTarget achievement analysis:")
        print(f"  Target R²: {target_r2:.4f}")
        print(f"  Current R²: {current_r2:.4f}")
        print(f"  Needed improvement: {needed_improvement:.4f}")
        print(f"  Needed variance reduction: ${needed_variance_reduction:.0f}")
        print(f"  Current unexplained variance: ${unexplained_variance:.0f}")
        print(f"  Required variance reduction: {(needed_variance_reduction/unexplained_variance)*100:.1f}%")
    
    def run_deep_residual_analysis(self):
        """Run complete deep residual analysis"""
        print("🔍 STARTING DEEP RESIDUAL ANALYSIS FOR 95%+ ACCURACY")
        print("=" * 70)
        
        self.analyze_large_residuals()
        self.discover_systematic_biases()
        self.discover_interaction_effects()
        self.discover_missing_thresholds()
        self.discover_precision_coefficient_adjustments()
        self.identify_outlier_patterns()
        self.quantify_improvement_potential()
        
        print("\n" + "=" * 70)
        print("🎯 DEEP RESIDUAL ANALYSIS COMPLETE")
        
        return {
            'missing_patterns': self.missing_patterns,
            'precision_opportunities': self.precision_opportunities,
            'current_performance': {
                'r2': r2_score(self.df['reimbursement'], self.df['current_prediction']),
                'rmse': np.sqrt(mean_squared_error(self.df['reimbursement'], self.df['current_prediction']))
            }
        }
    
    def save_analysis(self, output_path='deep_residual_analysis.json'):
        """Save deep residual analysis results"""
        # Create summary statistics for saving
        results = {
            'analysis_summary': {
                'total_cases': len(self.df),
                'large_residuals_100': len(self.df[np.abs(self.df['residual']) > 100]),
                'large_residuals_200': len(self.df[np.abs(self.df['residual']) > 200]),
                'large_residuals_500': len(self.df[np.abs(self.df['residual']) > 500]),
                'max_positive_residual': float(self.df['residual'].max()),
                'max_negative_residual': float(self.df['residual'].min()),
                'residual_std': float(self.df['residual'].std()),
                'current_r2': float(r2_score(self.df['reimbursement'], self.df['current_prediction'])),
                'target_r2': 0.95,
                'improvement_needed': float(0.95 - r2_score(self.df['reimbursement'], self.df['current_prediction']))
            },
            'systematic_patterns': {
                'days_patterns': self.df.groupby('trip_duration_days')['residual'].mean().to_dict(),
                'efficiency_patterns': self.df.groupby('efficiency_bin')['residual'].mean().to_dict() if hasattr(self.df, 'efficiency_bin') else {},
            },
            'missing_patterns': self.missing_patterns,
            'precision_opportunities': self.precision_opportunities
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Deep residual analysis saved to {output_path}")

if __name__ == "__main__":
    # Run deep residual analysis
    analyzer = DeepResidualAnalyzer()
    results = analyzer.run_deep_residual_analysis()
    analyzer.save_analysis()
    
    print(f"\n📋 CRITICAL FINDINGS SUMMARY:")
    print(f"Current R²: {results['current_performance']['r2']:.4f}")
    print(f"Target R²: 0.9500")
    print(f"Gap to close: {0.95 - results['current_performance']['r2']:.4f}")
    print(f"Current RMSE: ${results['current_performance']['rmse']:.2f}")
    print(f"Systematic patterns identified for precision enhancement")