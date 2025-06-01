#!/usr/bin/env python3
"""
Advanced Interview Mining for High-Precision Business Logic
Black Box Challenge - Targeting 95%+ Accuracy

This module performs exhaustive mining of employee interviews to extract every
quantitative detail, threshold, percentage, and condition mentioned but not yet
implemented. Based on deep residual analysis showing systematic biases, we need
to find the missing business logic buried in interview details.

Focus Areas:
1. Re-examine every interview for specific numbers, percentages, thresholds
2. Extract conditional logic patterns ("if this, then that")
3. Identify department-specific, timing-specific, and contextual rules
4. Map interview insights to discovered residual patterns
5. Quantify every vague statement into testable business rules
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import re
import warnings
warnings.filterwarnings('ignore')

class AdvancedInterviewMiner:
    """
    Performs exhaustive mining of interview transcripts for precision business logic
    """
    
    def __init__(self, data_path='test_cases.json', residual_analysis_path='deep_residual_analysis.json'):
        """Initialize with test cases and residual analysis results"""
        self.data_path = data_path
        self.residual_analysis_path = residual_analysis_path
        self.df = None
        self.residual_patterns = None
        self.interview_insights = {}
        self.extracted_rules = {}
        self.load_data()
        
    def load_data(self):
        """Load test cases and residual analysis"""
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
        
        # Load residual analysis if available
        try:
            with open(self.residual_analysis_path, 'r') as f:
                self.residual_patterns = json.load(f)
            print("Loaded residual analysis patterns")
        except FileNotFoundError:
            print("No residual analysis found")
            
    def apply_current_system_with_residuals(self):
        """Apply current system and calculate residuals for rule discovery"""
        # Use optimized baseline
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
        
        # Apply existing business rules
        self.df['business_rule_adjustment'] = 0
        
        # Current rules (from comprehensive system)
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
        
        # Calculate final predictions and residuals
        self.df['pre_rounding_prediction'] = self.df['linear_baseline'] + self.df['business_rule_adjustment']
        self.df['current_prediction'] = (self.df['pre_rounding_prediction'] * 4).round() / 4
        self.df['residual'] = self.df['reimbursement'] - self.df['current_prediction']
        
        print(f"Current system R²: {r2_score(self.df['reimbursement'], self.df['current_prediction']):.4f}")
        
    def extract_john_quantitative_details(self):
        """Extract every quantitative detail from John's interview"""
        print("\n=== EXTRACTING JOHN'S QUANTITATIVE DETAILS ===")
        
        john_insights = {
            # Specific numbers mentioned
            'dayton_columbus_distance': 90,  # "Take, for example, folks driving from Dayton to Columbus—what is that, 90 miles?"
            'indianapolis_cincinnati_distance': 120,  # "But if someone drove from Indianapolis to Cincinnati, which is a bit further—maybe 110, 120?"
            'mileage_rate_drop_percentage': 35,  # "the per-mile rate past a certain point dropped by 30 to 40 percent, give or take"
            'nashville_distance': 300,  # "someone drove from HQ to Nashville—like 300 miles or something"
            'chicago_rental_miles': 40,  # "someone who flew to Chicago and rented a car for 40 miles of driving"
            'marcus_lightning_miles': 400,  # "He'd hit three cities in two days, rack up like 400 miles"
            'marcus_lightning_days': 2,
            'milwaukee_training_miles': 20,  # "he'd take a week-long \"training\" in Milwaukee—barely left the hotel, maybe 20 miles total driving"
            'milwaukee_training_days': 7,
            'excel_simulator_accuracy': 12,  # "He got close—like within $12 on average"
            'omaha_days': 6,  # "a 6-day trip to Omaha"
            'akron_days': 2   # "a 2-day trip to Akron"
        }
        
        # Test specific mileage breakpoints mentioned in John's interview
        print("Testing John's specific mileage scenarios:")
        
        # Test 90-mile vs 120-mile scenarios
        miles_90 = self.df[self.df['miles_traveled'].between(85, 95)]
        miles_120 = self.df[self.df['miles_traveled'].between(115, 125)]
        
        if len(miles_90) > 0 and len(miles_120) > 0:
            avg_90 = miles_90['residual'].mean()
            avg_120 = miles_120['residual'].mean()
            print(f"  90-mile trips: avg residual ${avg_90:.2f} (n={len(miles_90)})")
            print(f"  120-mile trips: avg residual ${avg_120:.2f} (n={len(miles_120)})")
            print(f"  Difference: ${avg_120 - avg_90:.2f}")
            
            if abs(avg_120 - avg_90) > 10:
                self.extracted_rules['john_90_120_mile_difference'] = {
                    'miles_90_bias': avg_90,
                    'miles_120_bias': avg_120,
                    'difference': avg_120 - avg_90,
                    'source': 'John interview - Dayton/Columbus vs Indianapolis/Cincinnati example'
                }
        
        # Test Marcus lightning trip pattern (400 miles in 2 days = 200 miles/day)
        lightning_trips = self.df[
            (self.df['trip_duration_days'] == 2) & 
            (self.df['miles_traveled'].between(350, 450))
        ]
        
        if len(lightning_trips) > 0:
            lightning_avg = lightning_trips['residual'].mean()
            print(f"  Lightning trips (Marcus pattern): avg residual ${lightning_avg:.2f} (n={len(lightning_trips)})")
            
            if abs(lightning_avg) > 15:
                self.extracted_rules['marcus_lightning_pattern'] = {
                    'pattern': '2 days, 350-450 miles (high intensity)',
                    'residual_bias': lightning_avg,
                    'trips_affected': len(lightning_trips),
                    'source': 'John interview - Marcus lightning road trips'
                }
        
        # Test Milwaukee training pattern (7 days, ~20 miles = very low efficiency)
        milwaukee_trips = self.df[
            (self.df['trip_duration_days'] >= 6) & 
            (self.df['miles_traveled'] <= 30)
        ]
        
        if len(milwaukee_trips) > 0:
            milwaukee_avg = milwaukee_trips['residual'].mean()
            print(f"  Milwaukee pattern (long stay, low miles): avg residual ${milwaukee_avg:.2f} (n={len(milwaukee_trips)})")
            
            if abs(milwaukee_avg) > 15:
                self.extracted_rules['milwaukee_training_pattern'] = {
                    'pattern': '6+ days, ≤30 miles (minimal travel)',
                    'residual_bias': milwaukee_avg,
                    'trips_affected': len(milwaukee_trips),
                    'source': 'John interview - Milwaukee week-long training example'
                }
        
        self.interview_insights['john'] = john_insights
        
    def extract_peggy_quantitative_details(self):
        """Extract every quantitative detail from Peggy's interview"""
        print("\n=== EXTRACTING PEGGY'S QUANTITATIVE DETAILS ===")
        
        peggy_insights = {
            # Specific numbers and contexts
            'myrtle_beach_year_started': 1998,  # "We've been going every summer since '98"
            'cleveland_nights': 3,  # "I once expensed three nights in Cleveland"
            'panera_portion': 0.5,  # "a half-eaten Panera sandwich"
            'san_jose_amount': 2400,  # "$2,400 expense report after a week in San Jose"
            'san_jose_duration': 7,  # "week in San Jose"
            'akron_nights': 2,  # "just stayed in Akron for a couple nights"
            'charlotte_nights': 5,  # "five nights" in Charlotte
            'charlotte_walking': True,  # "lots of walking, my plantar fasciitis was flaring"
            'denver_days': 10,  # "ten-day training session in Denver"
            'denver_daily_spending': 180,  # "probably averaging $180 a day in expenses"
            'phoenix_daily_spending': 60,  # "maybe $60 a day"
            'kyle_receipts': 8.50,  # "Kyle from procurement turned in like $8.50 in receipts"
            'kyle_duration': 1,  # "one-day workshop"
            'sweet_spot_days': [5, 6],  # "around day five? Maybe day six?"
            'myrtle_humidity_month': 'May'  # "Myrtle Beach in May"
        }
        
        # Test specific scenarios mentioned by Peggy
        print("Testing Peggy's specific scenarios:")
        
        # Test 3-night Cleveland scenario with minimal receipts
        cleveland_pattern = self.df[
            (self.df['trip_duration_days'] == 3) & 
            (self.df['total_receipts_amount'] < 100)
        ]
        
        if len(cleveland_pattern) > 0:
            cleveland_avg = cleveland_pattern['residual'].mean()
            print(f"  Cleveland pattern (3 days, low receipts): avg residual ${cleveland_avg:.2f} (n={len(cleveland_pattern)})")
            
            if abs(cleveland_avg) > 20:
                self.extracted_rules['cleveland_pattern'] = {
                    'pattern': '3 days, <$100 receipts',
                    'residual_bias': cleveland_avg,
                    'trips_affected': len(cleveland_pattern),
                    'source': 'Peggy interview - Cleveland 3-night minimal expense example'
                }
        
        # Test San Jose high-spending week pattern
        san_jose_pattern = self.df[
            (self.df['trip_duration_days'] == 7) & 
            (self.df['total_receipts_amount'].between(2200, 2600))
        ]
        
        if len(san_jose_pattern) > 0:
            san_jose_avg = san_jose_pattern['residual'].mean()
            print(f"  San Jose pattern (7 days, ~$2400): avg residual ${san_jose_avg:.2f} (n={len(san_jose_pattern)})")
            
            if abs(san_jose_avg) > 20:
                self.extracted_rules['san_jose_pattern'] = {
                    'pattern': '7 days, $2200-2600 receipts',
                    'residual_bias': san_jose_avg,
                    'trips_affected': len(san_jose_pattern),
                    'source': 'Peggy interview - San Jose conference week example'
                }
        
        # Test Denver 10-day high spending vs Phoenix frugal pattern
        denver_pattern = self.df[
            (self.df['trip_duration_days'] == 10) & 
            (self.df['receipts_per_day'].between(170, 190))
        ]
        
        phoenix_pattern = self.df[
            (self.df['trip_duration_days'].between(8, 12)) & 
            (self.df['receipts_per_day'].between(50, 70))
        ]
        
        if len(denver_pattern) > 0:
            denver_avg = denver_pattern['residual'].mean()
            print(f"  Denver pattern (10 days, $180/day): avg residual ${denver_avg:.2f} (n={len(denver_pattern)})")
            
        if len(phoenix_pattern) > 0:
            phoenix_avg = phoenix_pattern['residual'].mean()
            print(f"  Phoenix pattern (long trip, $60/day): avg residual ${phoenix_avg:.2f} (n={len(phoenix_pattern)})")
            
            if len(denver_pattern) > 0 and abs(denver_avg - phoenix_avg) > 30:
                self.extracted_rules['denver_vs_phoenix_spending'] = {
                    'denver_pattern': '10 days, $170-190/day',
                    'phoenix_pattern': '8-12 days, $50-70/day',
                    'denver_bias': denver_avg,
                    'phoenix_bias': phoenix_avg,
                    'spending_penalty_difference': denver_avg - phoenix_avg,
                    'source': 'Peggy interview - Denver vs Phoenix spending comparison'
                }
        
        # Test Kyle's minimal receipt pattern
        kyle_pattern = self.df[
            (self.df['trip_duration_days'] == 1) & 
            (self.df['total_receipts_amount'] < 10)
        ]
        
        if len(kyle_pattern) > 0:
            kyle_avg = kyle_pattern['residual'].mean()
            print(f"  Kyle pattern (1 day, <$10): avg residual ${kyle_avg:.2f} (n={len(kyle_pattern)})")
            
            if abs(kyle_avg) > 30:
                self.extracted_rules['kyle_minimal_pattern'] = {
                    'pattern': '1 day, <$10 receipts',
                    'residual_bias': kyle_avg,
                    'trips_affected': len(kyle_pattern),
                    'source': 'Peggy interview - Kyle procurement workshop example'
                }
        
        self.interview_insights['peggy'] = peggy_insights
        
    def extract_tom_quantitative_details(self):
        """Extract every quantitative detail from Tom's interview"""
        print("\n=== EXTRACTING TOM'S QUANTITATIVE DETAILS ===")
        
        tom_insights = {
            # Specific numbers and contexts
            'schnauzer_litter_count': 6,  # "six miniature schnauzers"
            'schnauzer_colors': {'silver': 4, 'pepper_salt': 2},
            'breeding_start_year': 2009,  # "Been doing it since '09"
            'seattle_chicago_days': 9,  # "The move took like nine days start to finish"
            'tampa_director_days': 5,  # "five-day trip we did for a director flying in from Tampa"
            'little_rock_miles': 600,  # "from Little Rock—drove in, actually. 600-something miles"
            'st_louis_miles': 250,  # "another hire from St. Louis, about 250 miles away"
            'candidate_tour_miles': 700,  # "both drove about 700 miles"
            'candidate_tour_days': 3,  # "both stayed three days"
            'dinner_receipt': 199.99,  # "candidate dinner receipt that came in at... I want to say something like $199.99"
            'weird_adjustment': 1.02,  # "And the reimbursement came out weirdly specific. Like $1.02 more than expected"
            'timing_sensitivity': True,  # "the same exact trip submitted in different quarters would come back with slightly different numbers"
        }
        
        # Test specific scenarios mentioned by Tom
        print("Testing Tom's specific scenarios:")
        
        # Test 9-day relocation vs 5-day director trip
        nine_day_moves = self.df[self.df['trip_duration_days'] == 9]
        five_day_trips = self.df[self.df['trip_duration_days'] == 5]
        
        if len(nine_day_moves) > 0 and len(five_day_trips) > 0:
            nine_day_avg = nine_day_moves['residual'].mean()
            five_day_avg = five_day_trips['residual'].mean()
            print(f"  9-day moves: avg residual ${nine_day_avg:.2f} (n={len(nine_day_moves)})")
            print(f"  5-day trips: avg residual ${five_day_avg:.2f} (n={len(five_day_trips)})")
            
            # Tom mentioned 9-day relocation got less than 5-day trip
            if nine_day_avg < five_day_avg - 20:
                self.extracted_rules['tom_long_relocation_penalty'] = {
                    'nine_day_bias': nine_day_avg,
                    'five_day_bias': five_day_avg,
                    'long_move_penalty': five_day_avg - nine_day_avg,
                    'source': 'Tom interview - 9-day Seattle-Chicago relocation vs 5-day Tampa director'
                }
        
        # Test Little Rock (600 miles) vs St. Louis (250 miles) distance effect
        long_distance_hires = self.df[self.df['miles_traveled'].between(550, 650)]
        medium_distance_hires = self.df[self.df['miles_traveled'].between(200, 300)]
        
        if len(long_distance_hires) > 0 and len(medium_distance_hires) > 0:
            long_avg = long_distance_hires['residual'].mean()
            medium_avg = medium_distance_hires['residual'].mean()
            print(f"  Long distance hires (550-650 miles): avg residual ${long_avg:.2f} (n={len(long_distance_hires)})")
            print(f"  Medium distance hires (200-300 miles): avg residual ${medium_avg:.2f} (n={len(medium_distance_hires)})")
            
            if long_avg > medium_avg + 15:
                self.extracted_rules['tom_distance_hiring_bonus'] = {
                    'long_distance_bias': long_avg,
                    'medium_distance_bias': medium_avg,
                    'distance_bonus': long_avg - medium_avg,
                    'source': 'Tom interview - Little Rock vs St. Louis hiring distance comparison'
                }
        
        # Test 700-mile, 3-day candidate scenarios with different activity levels
        candidate_pattern = self.df[
            (self.df['trip_duration_days'] == 3) & 
            (self.df['miles_traveled'].between(650, 750))
        ]
        
        if len(candidate_pattern) > 10:
            # Split by efficiency (proxy for "whirlwind" vs "leisurely")
            high_efficiency = candidate_pattern[candidate_pattern['miles_per_day'] > 200]
            low_efficiency = candidate_pattern[candidate_pattern['miles_per_day'] <= 200]
            
            if len(high_efficiency) > 0 and len(low_efficiency) > 0:
                high_avg = high_efficiency['residual'].mean()
                low_avg = low_efficiency['residual'].mean()
                print(f"  Whirlwind candidates (>200 mi/day): avg residual ${high_avg:.2f} (n={len(high_efficiency)})")
                print(f"  Leisurely candidates (≤200 mi/day): avg residual ${low_avg:.2f} (n={len(low_efficiency)})")
                
                if high_avg > low_avg + 20:
                    self.extracted_rules['tom_candidate_activity_bonus'] = {
                        'whirlwind_bias': high_avg,
                        'leisurely_bias': low_avg,
                        'activity_bonus': high_avg - low_avg,
                        'source': 'Tom interview - whirlwind vs leisurely candidate tour comparison'
                    }
        
        # Test $199.99 dinner receipt rounding effect
        near_200_receipts = self.df[self.df['total_receipts_amount'].between(195, 205)]
        
        if len(near_200_receipts) > 0:
            near_200_avg = near_200_receipts['residual'].mean()
            print(f"  Near-$200 receipts: avg residual ${near_200_avg:.2f} (n={len(near_200_receipts)})")
            
            # Look for the $1.02 adjustment pattern Tom mentioned
            if abs(near_200_avg - 1.02) < 0.5:
                self.extracted_rules['tom_199_rounding_artifact'] = {
                    'near_200_bias': near_200_avg,
                    'expected_adjustment': 1.02,
                    'source': 'Tom interview - $199.99 dinner receipt rounding anomaly'
                }
        
        self.interview_insights['tom'] = tom_insights
        
    def extract_sarah_quantitative_details(self):
        """Extract every quantitative detail from Sarah's interview"""
        print("\n=== EXTRACTING SARAH'S QUANTITATIVE DETAILS ===")
        
        sarah_insights = {
            # Specific numbers and contexts
            'coordination_people': 40,  # "I coordinate travel for about 40 people"
            'coordination_states': 5,  # "across five states"
            'sprint_trip_days': 2,  # "fly in, hit three locations in two days"
            'sprint_trip_locations': 3,
            'sprint_bonus_percentage': 20,  # "20% higher than you'd expect"
            'phoenix_week_days': 7,  # "Phoenix office for a week-long training"
            'sales_director_cities': 4,  # "sales director do a four-city tour"
            'sales_director_days': 3,  # "800 miles in three days"
            'sales_director_miles': 800,
            'sales_director_daily_spending': 250,  # "Probably $250 a day in expenses"
            'modest_director_daily_spending': 80,  # "maybe $80 a day"
            'cleveland_days': 2,  # "two-day trip to Cleveland"
            'cleveland_receipts': 150,  # "turned in like $150 in receipts"
            'janet_days': 8,  # "eight-day trip"
            'janet_miles': 900,  # "900 miles"
            'janet_receipts': 1200,  # "$1,200 in expenses"
            'janet_bonus_percentage': 25,  # "25% higher than anyone expected"
            'end_of_quarter_generosity': True,  # "End of quarter? More generous"
            'mid_fiscal_year_tightness': True   # "Middle of the fiscal year? Tighter"
        }
        
        # Test specific scenarios mentioned by Sarah
        print("Testing Sarah's specific scenarios:")
        
        # Test sprint trip pattern (2 days, 3 locations, high miles)
        # Estimate 3 locations = roughly 200+ miles per day
        sprint_pattern = self.df[
            (self.df['trip_duration_days'] == 2) & 
            (self.df['miles_per_day'] >= 200)
        ]
        
        if len(sprint_pattern) > 0:
            sprint_avg = sprint_pattern['residual'].mean()
            print(f"  Sprint trips (2 days, 200+ mi/day): avg residual ${sprint_avg:.2f} (n={len(sprint_pattern)})")
            
            if sprint_avg > 15:  # Sarah mentioned 20% higher
                self.extracted_rules['sarah_sprint_trip_bonus'] = {
                    'pattern': '2 days, ≥200 miles/day (high intensity)',
                    'residual_bias': sprint_avg,
                    'trips_affected': len(sprint_pattern),
                    'expected_bonus_pct': 20,
                    'source': 'Sarah interview - sprint trip pattern with 20% bonus'
                }
        
        # Test Phoenix camping trip pattern (7 days, low miles)
        phoenix_pattern = self.df[
            (self.df['trip_duration_days'] == 7) & 
            (self.df['miles_per_day'] < 50)
        ]
        
        if len(phoenix_pattern) > 0:
            phoenix_avg = phoenix_pattern['residual'].mean()
            print(f"  Phoenix camping (7 days, <50 mi/day): avg residual ${phoenix_avg:.2f} (n={len(phoenix_pattern)})")
            
            if phoenix_avg < -15:
                self.extracted_rules['sarah_phoenix_camping_penalty'] = {
                    'pattern': '7 days, <50 miles/day (minimal activity)',
                    'residual_bias': phoenix_avg,
                    'trips_affected': len(phoenix_pattern),
                    'source': 'Sarah interview - Phoenix week-long training with minimal travel'
                }
        
        # Test sales director high spending vs modest spending pattern
        four_city_high_spend = self.df[
            (self.df['trip_duration_days'] == 3) & 
            (self.df['miles_traveled'].between(750, 850)) &
            (self.df['receipts_per_day'].between(240, 260))
        ]
        
        similar_trip_modest_spend = self.df[
            (self.df['trip_duration_days'] == 3) & 
            (self.df['miles_traveled'].between(750, 850)) &
            (self.df['receipts_per_day'].between(70, 90))
        ]
        
        if len(four_city_high_spend) > 0 and len(similar_trip_modest_spend) > 0:
            high_spend_avg = four_city_high_spend['residual'].mean()
            modest_spend_avg = similar_trip_modest_spend['residual'].mean()
            print(f"  4-city high spend ($250/day): avg residual ${high_spend_avg:.2f} (n={len(four_city_high_spend)})")
            print(f"  Similar trip modest spend ($80/day): avg residual ${modest_spend_avg:.2f} (n={len(similar_trip_modest_spend)})")
            
            if modest_spend_avg > high_spend_avg + 30:
                self.extracted_rules['sarah_spending_level_judgment'] = {
                    'high_spend_bias': high_spend_avg,
                    'modest_spend_bias': modest_spend_avg,
                    'spending_penalty': high_spend_avg - modest_spend_avg,
                    'source': 'Sarah interview - sales director high vs modest spending comparison'
                }
        
        # Test Cleveland minimal trip pattern
        cleveland_minimal = self.df[
            (self.df['trip_duration_days'] == 2) & 
            (self.df['total_receipts_amount'].between(140, 160)) &
            (self.df['miles_traveled'] < 100)
        ]
        
        if len(cleveland_minimal) > 0:
            cleveland_avg = cleveland_minimal['residual'].mean()
            print(f"  Cleveland minimal (2 days, ~$150, low miles): avg residual ${cleveland_avg:.2f} (n={len(cleveland_minimal)})")
            
            if cleveland_avg < -20:
                self.extracted_rules['sarah_cleveland_minimal_penalty'] = {
                    'pattern': '2 days, ~$150 receipts, <100 miles',
                    'residual_bias': cleveland_avg,
                    'trips_affected': len(cleveland_minimal),
                    'source': 'Sarah interview - Cleveland minimal trip example'
                }
        
        # Test Janet's exact jackpot pattern (already implemented, but verify)
        janet_exact = self.df[
            (self.df['trip_duration_days'] == 8) & 
            (self.df['miles_traveled'] == 900) &
            (self.df['total_receipts_amount'] == 1200)
        ]
        
        if len(janet_exact) > 0:
            janet_avg = janet_exact['residual'].mean()
            print(f"  Janet exact pattern (8 days, 900 miles, $1200): avg residual ${janet_avg:.2f} (n={len(janet_exact)})")
        
        self.interview_insights['sarah'] = sarah_insights
        
    def extract_conditional_logic_patterns(self):
        """Extract conditional logic patterns from interview statements"""
        print("\n=== EXTRACTING CONDITIONAL LOGIC PATTERNS ===")
        
        conditional_rules = {}
        
        # 1. John's conditional: "If you're on a long trip and you're living it up every night, the system gets... judgmental"
        # Already partially implemented, but refine the conditions
        long_trip_high_lifestyle = self.df[
            (self.df['trip_duration_days'] >= 6) &  # Long trip
            (self.df['receipts_per_day'] > self.df['receipts_per_day'].quantile(0.75))  # High lifestyle
        ]
        
        if len(long_trip_high_lifestyle) > 50:
            lifestyle_avg = long_trip_high_lifestyle['residual'].mean()
            print(f"Long trip + high lifestyle: avg residual ${lifestyle_avg:.2f} (n={len(long_trip_high_lifestyle)})")
            
            if lifestyle_avg < -25:
                conditional_rules['long_trip_lifestyle_penalty'] = {
                    'condition': 'trip_days >= 6 AND receipts_per_day > 75th percentile',
                    'effect': lifestyle_avg,
                    'affected_trips': len(long_trip_high_lifestyle),
                    'source': 'John interview conditional logic'
                }
        
        # 2. Peggy's conditional: "If you're buying steak dinners three nights in a row, the system just puts up a wall"
        # Look for consecutive high-spending patterns (proxy: high per-day spending)
        high_daily_luxury = self.df[
            (self.df['trip_duration_days'] >= 3) &
            (self.df['receipts_per_day'] > 200)  # Steak dinner level
        ]
        
        if len(high_daily_luxury) > 20:
            luxury_avg = high_daily_luxury['residual'].mean()
            print(f"High daily luxury (≥3 days, >$200/day): avg residual ${luxury_avg:.2f} (n={len(high_daily_luxury)})")
            
            if luxury_avg < -30:
                conditional_rules['luxury_spending_wall'] = {
                    'condition': 'trip_days >= 3 AND receipts_per_day > $200',
                    'effect': luxury_avg,
                    'affected_trips': len(high_daily_luxury),
                    'source': 'Peggy interview - steak dinner wall logic'
                }
        
        # 3. Tom's conditional: "Once it's far enough that it probably required a flight, rental car, or serious planning"
        # Test logistical complexity threshold
        complex_logistics = self.df[
            (self.df['miles_traveled'] >= 500) |  # Probably required flight
            ((self.df['trip_duration_days'] >= 5) & (self.df['miles_traveled'] >= 300))  # Serious planning
        ]
        
        simple_logistics = self.df[
            (self.df['miles_traveled'] < 300) & 
            (self.df['trip_duration_days'] <= 3)
        ]
        
        if len(complex_logistics) > 100 and len(simple_logistics) > 100:
            complex_avg = complex_logistics['residual'].mean()
            simple_avg = simple_logistics['residual'].mean()
            print(f"Complex logistics: avg residual ${complex_avg:.2f} (n={len(complex_logistics)})")
            print(f"Simple logistics: avg residual ${simple_avg:.2f} (n={len(simple_logistics)})")
            
            if complex_avg > simple_avg + 15:
                conditional_rules['logistics_complexity_bonus'] = {
                    'condition': 'miles >= 500 OR (days >= 5 AND miles >= 300)',
                    'complex_effect': complex_avg,
                    'simple_effect': simple_avg,
                    'bonus': complex_avg - simple_avg,
                    'source': 'Tom interview - logistical complexity threshold'
                }
        
        # 4. Sarah's conditional: "If you can hit all the right notes—long trip, high mileage, high expenses—the system gives you this massive bonus"
        # Test triple-condition jackpot (more refined than current implementation)
        triple_jackpot = self.df[
            (self.df['trip_duration_days'] >= 7) &  # Long trip
            (self.df['miles_traveled'] >= 800) &   # High mileage  
            (self.df['total_receipts_amount'] >= 1000)  # High expenses
        ]
        
        if len(triple_jackpot) > 30:
            jackpot_avg = triple_jackpot['residual'].mean()
            print(f"Triple jackpot (7+ days, 800+ miles, $1000+ receipts): avg residual ${jackpot_avg:.2f} (n={len(triple_jackpot)})")
            
            if jackpot_avg > 25:
                conditional_rules['triple_condition_jackpot'] = {
                    'condition': 'days >= 7 AND miles >= 800 AND receipts >= $1000',
                    'effect': jackpot_avg,
                    'affected_trips': len(triple_jackpot),
                    'source': 'Sarah interview - triple condition massive bonus'
                }
        
        self.extracted_rules['conditional_logic'] = conditional_rules
        
    def quantify_vague_statements(self):
        """Convert vague interview statements into specific, testable rules"""
        print("\n=== QUANTIFYING VAGUE STATEMENTS ===")
        
        vague_quantifications = {}
        
        # 1. John: "road warrior bonus" - quantify what makes a road warrior
        # Test different efficiency levels to find the precise threshold
        efficiency_levels = [150, 175, 200, 225, 250, 275, 300]
        
        print("Road warrior efficiency analysis:")
        for threshold in efficiency_levels:
            road_warriors = self.df[self.df['miles_per_day'] >= threshold]
            regular_travelers = self.df[self.df['miles_per_day'] < threshold]
            
            if len(road_warriors) >= 50 and len(regular_travelers) >= 50:
                warrior_avg = road_warriors['residual'].mean()
                regular_avg = regular_travelers['residual'].mean()
                bonus = warrior_avg - regular_avg
                
                print(f"  {threshold}+ mi/day: bonus ${bonus:.2f} (n={len(road_warriors)})")
                
                if bonus > 20 and threshold not in [200]:  # Find additional thresholds beyond current 200
                    vague_quantifications[f'road_warrior_{threshold}'] = {
                        'threshold': threshold,
                        'bonus': bonus,
                        'affected_trips': len(road_warriors),
                        'source': 'John interview - road warrior bonus quantification'
                    }
        
        # 2. Peggy: "sweet spot" - find the precise days that get bonuses
        print("\nSweet spot day analysis:")
        for days in range(3, 9):
            day_trips = self.df[self.df['trip_duration_days'] == days]
            other_trips = self.df[self.df['trip_duration_days'] != days]
            
            if len(day_trips) >= 50:
                day_avg = day_trips['residual'].mean()
                other_avg = other_trips['residual'].mean()
                bonus = day_avg - other_avg
                
                print(f"  {days} days: bonus ${bonus:.2f} (n={len(day_trips)})")
                
                if bonus > 15 and days not in [5, 6]:  # Find additional sweet spots
                    vague_quantifications[f'sweet_spot_day_{days}'] = {
                        'days': days,
                        'bonus': bonus,
                        'affected_trips': len(day_trips),
                        'source': 'Peggy interview - sweet spot quantification'
                    }
        
        # 3. Tom: "logistically annoying" distances - find precise distance thresholds
        print("\nLogistical annoyance distance analysis:")
        distance_thresholds = [400, 500, 600, 700, 800, 900, 1000]
        
        for threshold in distance_thresholds:
            annoying_distances = self.df[self.df['miles_traveled'] >= threshold]
            easy_distances = self.df[self.df['miles_traveled'] < threshold]
            
            if len(annoying_distances) >= 100 and len(easy_distances) >= 100:
                annoying_avg = annoying_distances['residual'].mean()
                easy_avg = easy_distances['residual'].mean()
                bonus = annoying_avg - easy_avg
                
                print(f"  {threshold}+ miles: bonus ${bonus:.2f} (n={len(annoying_distances)})")
                
                if bonus > 10 and threshold not in [600]:  # Find additional thresholds
                    vague_quantifications[f'logistics_threshold_{threshold}'] = {
                        'threshold': threshold,
                        'bonus': bonus,
                        'affected_trips': len(annoying_distances),
                        'source': 'Tom interview - logistical annoyance quantification'
                    }
        
        # 4. Sarah: "system respects commitment" - quantify commitment levels
        print("\nCommitment level analysis:")
        
        # Define commitment as combination of days, miles, and receipts
        commitment_scores = (
            self.df['trip_duration_days'] / self.df['trip_duration_days'].max() * 0.4 +
            self.df['miles_traveled'] / self.df['miles_traveled'].max() * 0.3 +
            self.df['total_receipts_amount'] / self.df['total_receipts_amount'].max() * 0.3
        )
        
        high_commitment = self.df[commitment_scores >= commitment_scores.quantile(0.9)]
        low_commitment = self.df[commitment_scores <= commitment_scores.quantile(0.1)]
        
        if len(high_commitment) > 0 and len(low_commitment) > 0:
            high_avg = high_commitment['residual'].mean()
            low_avg = low_commitment['residual'].mean()
            commitment_bonus = high_avg - low_avg
            
            print(f"  High commitment (top 10%): avg residual ${high_avg:.2f} (n={len(high_commitment)})")
            print(f"  Low commitment (bottom 10%): avg residual ${low_avg:.2f} (n={len(low_commitment)})")
            print(f"  Commitment bonus: ${commitment_bonus:.2f}")
            
            if commitment_bonus > 30:
                vague_quantifications['commitment_respect'] = {
                    'high_commitment_threshold': commitment_scores.quantile(0.9),
                    'bonus': commitment_bonus,
                    'affected_trips': len(high_commitment),
                    'source': 'Sarah interview - system respects commitment'
                }
        
        self.extracted_rules['vague_quantifications'] = vague_quantifications
        
    def run_advanced_interview_mining(self):
        """Run complete advanced interview mining process"""
        print("🔍 STARTING ADVANCED INTERVIEW MINING FOR PRECISION RULES")
        print("=" * 70)
        
        self.apply_current_system_with_residuals()
        self.extract_john_quantitative_details()
        self.extract_peggy_quantitative_details()
        self.extract_tom_quantitative_details()
        self.extract_sarah_quantitative_details()
        self.extract_conditional_logic_patterns()
        self.quantify_vague_statements()
        
        print("\n" + "=" * 70)
        print("🎯 ADVANCED INTERVIEW MINING COMPLETE")
        
        return {
            'interview_insights': self.interview_insights,
            'extracted_rules': self.extracted_rules,
            'total_new_rules': sum(len(v) if isinstance(v, dict) else 1 for v in self.extracted_rules.values())
        }
        
    def save_extracted_rules(self, output_path='advanced_interview_rules.json'):
        """Save all extracted rules and insights"""
        results = {
            'interview_insights': self.interview_insights,
            'extracted_rules': self.extracted_rules,
            'extraction_summary': {
                'total_rule_categories': len(self.extracted_rules),
                'total_quantitative_insights': sum(len(insights) for insights in self.interview_insights.values()),
                'precision_improvement_potential': 'High - specific scenarios and thresholds identified'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Advanced interview rules saved to {output_path}")

if __name__ == "__main__":
    # Run advanced interview mining
    miner = AdvancedInterviewMiner()
    results = miner.run_advanced_interview_mining()
    miner.save_extracted_rules()
    
    print(f"\n📋 ADVANCED MINING SUMMARY:")
    print(f"Interview insights extracted: {sum(len(insights) for insights in results['interview_insights'].values())}")
    print(f"New rule categories: {len(results['extracted_rules'])}")
    print(f"Total new precision rules: {results['total_new_rules']}")
    print(f"Ready for implementation in high-precision system")