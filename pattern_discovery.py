#!/usr/bin/env python3
"""
Pattern Discovery Module
Black Box Challenge - Deterministic Foundation Building

This module analyzes the test cases data to identify mathematical relationships
between inputs (trip_duration_days, miles_traveled, total_receipts_amount)
and outputs (reimbursement amounts).

Based on employee interviews, we're looking for:
1. Per diem patterns (daily rates)
2. Mileage patterns (with diminishing returns)
3. Receipt patterns (with ceilings)
4. Efficiency bonuses (miles/day ratios)
5. Sweet spot bonuses (5-6 day trips)
6. Big trip jackpots (high days + miles + receipts)
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class PatternDiscovery:
    """
    Analyzes data patterns based on employee interview insights
    """
    
    def __init__(self, data_path='test_cases.json'):
        """Initialize with test cases data"""
        self.data_path = data_path
        self.df = None
        self.patterns = {}
        self.load_data()
        
    def load_data(self):
        """Load and prepare the test cases data"""
        with open(self.data_path, 'r') as f:
            test_cases = json.load(f)
        
        # Convert to DataFrame
        rows = []
        for case in test_cases:
            row = case['input'].copy()
            row['reimbursement'] = case['expected_output']
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        
        # Add derived features for analysis
        self.df['miles_per_day'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        self.df['receipts_per_day'] = self.df['total_receipts_amount'] / self.df['trip_duration_days']
        self.df['receipts_per_mile'] = self.df['total_receipts_amount'] / (self.df['miles_traveled'] + 1e-8)
        
        print(f"Loaded {len(self.df)} test cases")
        print(f"Columns: {list(self.df.columns)}")
        
    def analyze_basic_patterns(self):
        """Analyze basic statistical patterns and correlations"""
        print("\n=== BASIC PATTERN ANALYSIS ===")
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        # Correlations with reimbursement
        correlations = self.df.corr()['reimbursement'].sort_values(ascending=False)
        print(f"\nCorrelations with reimbursement:")
        for col, corr in correlations.items():
            if col != 'reimbursement':
                print(f"  {col}: {corr:.3f}")
        
        self.patterns['correlations'] = correlations.to_dict()
        
    def analyze_per_diem_patterns(self):
        """
        Analyze per diem patterns based on John's interview:
        - Base amount per day
        - Consistent daily logic
        """
        print("\n=== PER DIEM PATTERN ANALYSIS ===")
        
        # Calculate reimbursement per day
        self.df['reimbursement_per_day'] = self.df['reimbursement'] / self.df['trip_duration_days']
        
        print(f"Reimbursement per day statistics:")
        print(self.df['reimbursement_per_day'].describe())
        
        # Test linear relationship with days
        X = self.df[['trip_duration_days']].values
        y = self.df['reimbursement'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        print(f"\nLinear model (reimbursement ~ days):")
        print(f"  Coefficient: ${model.coef_[0]:.2f} per day")
        print(f"  Intercept: ${model.intercept_:.2f}")
        print(f"  R²: {model.score(X, y):.3f}")
        
        self.patterns['per_diem'] = {
            'daily_rate_coefficient': model.coef_[0],
            'base_amount': model.intercept_,
            'r2_score': model.score(X, y)
        }
        
    def analyze_mileage_patterns(self):
        """
        Analyze mileage patterns based on John's interview:
        - Diminishing returns after certain point
        - Per-mile rate drops by 30-40% after threshold
        """
        print("\n=== MILEAGE PATTERN ANALYSIS ===")
        
        # Test different mileage thresholds mentioned in interviews
        thresholds = [100, 120, 200, 300, 500]
        
        for threshold in thresholds:
            low_miles = self.df[self.df['miles_traveled'] <= threshold]
            high_miles = self.df[self.df['miles_traveled'] > threshold]
            
            if len(low_miles) > 10 and len(high_miles) > 10:
                low_rate = (low_miles['reimbursement'] / low_miles['miles_traveled']).mean()
                high_rate = (high_miles['reimbursement'] / high_miles['miles_traveled']).mean()
                reduction = (low_rate - high_rate) / low_rate * 100
                
                print(f"\nMileage threshold {threshold}:")
                print(f"  Rate ≤{threshold} miles: ${low_rate:.3f}/mile")
                print(f"  Rate >{threshold} miles: ${high_rate:.3f}/mile")
                print(f"  Reduction: {reduction:.1f}%")
        
        # Test piecewise linear regression
        self.analyze_piecewise_mileage()
        
    def analyze_piecewise_mileage(self):
        """Test piecewise linear mileage patterns"""
        # Try different breakpoints
        breakpoints = [100, 150, 200, 250, 300]
        best_r2 = -1
        best_breakpoint = None
        
        for breakpoint in breakpoints:
            # Create piecewise features
            self.df[f'miles_low_{breakpoint}'] = np.minimum(self.df['miles_traveled'], breakpoint)
            self.df[f'miles_high_{breakpoint}'] = np.maximum(0, self.df['miles_traveled'] - breakpoint)
            
            # Fit model
            X = self.df[[f'miles_low_{breakpoint}', f'miles_high_{breakpoint}']]
            y = self.df['reimbursement']
            
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            
            if r2 > best_r2:
                best_r2 = r2
                best_breakpoint = breakpoint
                
            print(f"Breakpoint {breakpoint}: R² = {r2:.3f}")
        
        print(f"\nBest mileage breakpoint: {best_breakpoint} miles (R² = {best_r2:.3f})")
        
        self.patterns['mileage'] = {
            'best_breakpoint': best_breakpoint,
            'best_r2': best_r2
        }
        
    def analyze_receipt_patterns(self):
        """
        Analyze receipt patterns based on Peggy's interview:
        - Ceiling effect after certain amount
        - Small amounts discouraged
        """
        print("\n=== RECEIPT PATTERN ANALYSIS ===")
        
        # Test for ceiling effects
        receipt_percentiles = np.percentile(self.df['total_receipts_amount'], [10, 25, 50, 75, 90, 95])
        
        for percentile in [75, 90, 95]:
            threshold = np.percentile(self.df['total_receipts_amount'], percentile)
            
            low_receipts = self.df[self.df['total_receipts_amount'] <= threshold]
            high_receipts = self.df[self.df['total_receipts_amount'] > threshold]
            
            if len(high_receipts) > 5:
                low_rate = (low_receipts['reimbursement'] / low_receipts['total_receipts_amount']).mean()
                high_rate = (high_receipts['reimbursement'] / high_receipts['total_receipts_amount']).mean()
                
                print(f"\nReceipt threshold {percentile}th percentile (${threshold:.0f}):")
                print(f"  Rate ≤${threshold:.0f}: {low_rate:.3f}")
                print(f"  Rate >${threshold:.0f}: {high_rate:.3f}")
        
        # Test polynomial relationship
        X = self.df[['total_receipts_amount']]
        y = self.df['reimbursement']
        
        # Linear
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_r2 = linear_model.score(X, y)
        
        # Quadratic
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        poly_r2 = poly_model.score(X_poly, y)
        
        print(f"\nReceipt relationship:")
        print(f"  Linear R²: {linear_r2:.3f}")
        print(f"  Quadratic R²: {poly_r2:.3f}")
        
        self.patterns['receipts'] = {
            'linear_r2': linear_r2,
            'quadratic_r2': poly_r2,
            'ceiling_threshold': np.percentile(self.df['total_receipts_amount'], 90)
        }
        
    def analyze_efficiency_bonus(self):
        """
        Analyze efficiency bonus based on John and Sarah's interviews:
        - "Road warrior bonus" for high miles/day
        - System rewards intensity
        """
        print("\n=== EFFICIENCY BONUS ANALYSIS ===")
        
        # Test different efficiency thresholds
        efficiency_thresholds = [100, 150, 200, 250, 300]
        
        for threshold in efficiency_thresholds:
            high_efficiency = self.df[self.df['miles_per_day'] > threshold]
            low_efficiency = self.df[self.df['miles_per_day'] <= threshold]
            
            if len(high_efficiency) > 10 and len(low_efficiency) > 10:
                high_avg = high_efficiency['reimbursement'].mean()
                low_avg = low_efficiency['reimbursement'].mean()
                bonus = (high_avg - low_avg) / low_avg * 100
                
                print(f"\nEfficiency threshold {threshold} miles/day:")
                print(f"  High efficiency avg: ${high_avg:.2f}")
                print(f"  Low efficiency avg: ${low_avg:.2f}")
                print(f"  Bonus: {bonus:.1f}%")
        
        self.patterns['efficiency'] = {
            'miles_per_day_correlation': self.df['miles_per_day'].corr(self.df['reimbursement'])
        }
        
    def analyze_sweet_spot_bonus(self):
        """
        Analyze sweet spot bonus based on Peggy's interview:
        - 5-6 day trips get extra reward
        """
        print("\n=== SWEET SPOT BONUS ANALYSIS ===")
        
        # Group by trip duration
        duration_groups = self.df.groupby('trip_duration_days')['reimbursement'].agg(['mean', 'count'])
        
        print("\nAverage reimbursement by trip duration:")
        for days, stats in duration_groups.iterrows():
            if stats['count'] >= 5:  # Only show groups with enough data
                print(f"  {days} days: ${stats['mean']:.2f} (n={stats['count']})")
        
        # Test 5-6 day sweet spot
        sweet_spot = self.df[self.df['trip_duration_days'].isin([5, 6])]
        other_trips = self.df[~self.df['trip_duration_days'].isin([5, 6])]
        
        if len(sweet_spot) > 0:
            sweet_avg = sweet_spot['reimbursement'].mean()
            other_avg = other_trips['reimbursement'].mean()
            bonus = (sweet_avg - other_avg) / other_avg * 100
            
            print(f"\nSweet spot (5-6 days) analysis:")
            print(f"  Sweet spot avg: ${sweet_avg:.2f}")
            print(f"  Other trips avg: ${other_avg:.2f}")
            print(f"  Bonus: {bonus:.1f}%")
            
            self.patterns['sweet_spot'] = {
                'bonus_percentage': bonus,
                'sweet_spot_avg': sweet_avg,
                'other_avg': other_avg
            }
        
    def analyze_big_trip_jackpot(self):
        """
        Analyze big trip jackpot based on Sarah's interview:
        - 8+ days + 900+ miles + 1200+ receipts = 25% bonus
        """
        print("\n=== BIG TRIP JACKPOT ANALYSIS ===")
        
        # Test the specific combination mentioned
        jackpot_criteria = (
            (self.df['trip_duration_days'] >= 8) &
            (self.df['miles_traveled'] >= 900) &
            (self.df['total_receipts_amount'] >= 1200)
        )
        
        jackpot_trips = self.df[jackpot_criteria]
        other_trips = self.df[~jackpot_criteria]
        
        if len(jackpot_trips) > 0:
            jackpot_avg = jackpot_trips['reimbursement'].mean()
            other_avg = other_trips['reimbursement'].mean()
            bonus = (jackpot_avg - other_avg) / other_avg * 100
            
            print(f"\nBig trip jackpot (8+ days, 900+ miles, $1200+ receipts):")
            print(f"  Jackpot trips: {len(jackpot_trips)}")
            print(f"  Jackpot avg: ${jackpot_avg:.2f}")
            print(f"  Other avg: ${other_avg:.2f}")
            print(f"  Bonus: {bonus:.1f}%")
            
            self.patterns['big_trip_jackpot'] = {
                'bonus_percentage': bonus,
                'qualifying_trips': len(jackpot_trips),
                'jackpot_avg': jackpot_avg
            }
        else:
            print("No trips found matching big trip jackpot criteria")
            
    def test_multi_linear_model(self):
        """Test basic multi-linear regression model"""
        print("\n=== MULTI-LINEAR MODEL TEST ===")
        
        # Test simple model
        features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        X = self.df[features]
        y = self.df['reimbursement']
        
        model = LinearRegression()
        model.fit(X, y)
        
        print(f"Multi-linear model (R² = {model.score(X, y):.3f}):")
        for feature, coef in zip(features, model.coef_):
            print(f"  {feature}: {coef:.3f}")
        print(f"  Intercept: {model.intercept_:.3f}")
        
        self.patterns['multi_linear'] = {
            'r2': model.score(X, y),
            'coefficients': dict(zip(features, model.coef_)),
            'intercept': model.intercept_
        }
        
    def run_full_analysis(self):
        """Run all pattern discovery analyses"""
        print("🔍 STARTING PATTERN DISCOVERY ANALYSIS")
        print("=" * 50)
        
        self.analyze_basic_patterns()
        self.analyze_per_diem_patterns()
        self.analyze_mileage_patterns()
        self.analyze_receipt_patterns()
        self.analyze_efficiency_bonus()
        self.analyze_sweet_spot_bonus()
        self.analyze_big_trip_jackpot()
        self.test_multi_linear_model()
        
        print("\n" + "=" * 50)
        print("🎯 PATTERN DISCOVERY COMPLETE")
        
        return self.patterns
        
    def save_patterns(self, output_path='discovered_patterns.json'):
        """Save discovered patterns to file"""
        with open(output_path, 'w') as f:
            json.dump(self.patterns, f, indent=2)
        print(f"\n💾 Patterns saved to {output_path}")

if __name__ == "__main__":
    # Run pattern discovery
    discovery = PatternDiscovery()
    patterns = discovery.run_full_analysis()
    discovery.save_patterns()
    
    print("\n📋 SUMMARY OF KEY FINDINGS:")
    print("-" * 30)
    
    if 'multi_linear' in patterns:
        ml = patterns['multi_linear']
        print(f"Multi-linear R²: {ml['r2']:.3f}")
        print(f"Daily rate: ${ml['coefficients']['trip_duration_days']:.2f}")
        print(f"Mile rate: ${ml['coefficients']['miles_traveled']:.3f}")
        print(f"Receipt rate: {ml['coefficients']['total_receipts_amount']:.3f}")
    
    if 'sweet_spot' in patterns and patterns['sweet_spot']['bonus_percentage'] > 5:
        print(f"Sweet spot bonus: {patterns['sweet_spot']['bonus_percentage']:.1f}%")
    
    if 'big_trip_jackpot' in patterns:
        print(f"Big trip jackpot: {patterns['big_trip_jackpot']['bonus_percentage']:.1f}%")