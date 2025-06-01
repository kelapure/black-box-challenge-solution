#!/usr/bin/env python3
"""
Deterministic Rule Engine
Black Box Challenge - Deterministic Foundation Building

This module implements the DeterministicRuleEngine class that encapsulates
all business logic discovered from employee interviews into a clean,
production-ready, interpretable system.

Design Principles:
1. Pure business rule logic (no statistical black boxes)
2. Clear precedence ordering
3. Full interpretability and auditability
4. Modular rule system for easy maintenance
5. Comprehensive logging and explanation capabilities
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

class RuleType(Enum):
    """Types of business rules"""
    BONUS = "bonus"
    PENALTY = "penalty" 
    ADJUSTMENT = "adjustment"
    ROUNDING = "rounding"

@dataclass
class RuleApplication:
    """Record of a rule being applied"""
    rule_name: str
    rule_description: str
    condition_met: bool
    adjustment_amount: float
    explanation: str

class DeterministicRuleEngine:
    """
    Production-ready deterministic rule engine implementing all business logic
    discovered from employee interviews
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the deterministic rule engine
        
        Args:
            verbose: Enable detailed logging and explanations
        """
        self.verbose = verbose
        self.linear_baseline = self._initialize_linear_baseline()
        self.business_rules = self._initialize_business_rules()
        self.rule_precedence = self._initialize_rule_precedence()
        self.rule_applications_log = []
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        
    def _initialize_linear_baseline(self) -> Dict[str, float]:
        """Initialize the linear baseline coefficients"""
        return {
            'intercept': 318.40,
            'trip_duration_days': 71.28,
            'miles_traveled': 0.7941,
            'total_receipts_amount': 0.2904,
            'description': 'Linear baseline: base amount + daily rate + mileage rate + receipt rate'
        }
    
    def _initialize_business_rules(self) -> Dict[str, Dict]:
        """Initialize all business rules discovered from interviews"""
        return {
            'sweet_spot_bonus': {
                'description': "5-6 day trips get sweet spot bonus (Peggy's rule)",
                'rule_type': RuleType.BONUS,
                'priority': 1,
                'source': 'Peggy interview - optimal trip length',
                'adjustment': 78.85,
                'conditions': {
                    'trip_duration_days': {'operator': 'in', 'values': [5, 6]}
                },
                'explanation': 'Medium-length trips hit the sweet spot for productivity vs overhead'
            },
            
            'receipt_ceiling': {
                'description': "High receipts (>$2193) get ceiling penalty (Peggy's rule)",
                'rule_type': RuleType.PENALTY,
                'priority': 2,
                'source': 'Peggy interview - discourages excessive spending',
                'adjustment': -99.48,
                'conditions': {
                    'total_receipts_amount': {'operator': '>', 'value': 2193}
                },
                'explanation': 'System caps reimbursement to discourage lavish spending'
            },
            
            'big_trip_jackpot': {
                'description': "Big trips (8+ days, 900+ miles, $1200+ receipts) get jackpot bonus (Sarah's rule)",
                'rule_type': RuleType.BONUS,
                'priority': 3,
                'source': 'Sarah interview - rewards major business initiatives',
                'adjustment': 46.67,
                'conditions': {
                    'trip_duration_days': {'operator': '>=', 'value': 8},
                    'miles_traveled': {'operator': '>=', 'value': 900},
                    'total_receipts_amount': {'operator': '>=', 'value': 1200}
                },
                'explanation': 'Major business trips with high commitment get special recognition'
            },
            
            'distance_bonus': {
                'description': "Long distance trips (600+ miles) get logistical bonus (Tom's rule)",
                'rule_type': RuleType.BONUS,
                'priority': 4,
                'source': 'Tom interview - compensates for logistical complexity',
                'adjustment': 28.75,
                'conditions': {
                    'miles_traveled': {'operator': '>=', 'value': 600}
                },
                'explanation': 'Long-distance travel involves additional planning and logistics'
            },
            
            'minimum_spending_penalty': {
                'description': "Very low spending (<$103) gets minimum threshold penalty",
                'rule_type': RuleType.PENALTY,
                'priority': 5,
                'source': 'Edge case analysis - discourages trivial expense claims',
                'adjustment': -299.03,
                'conditions': {
                    'total_receipts_amount': {'operator': '<', 'value': 103}
                },
                'explanation': 'Minimal expenses suggest trip may not justify reimbursement'
            },
            
            'efficiency_moderate_bonus': {
                'description': "Moderate efficiency (100-200 miles/day) gets road warrior bonus",
                'rule_type': RuleType.BONUS,
                'priority': 6,
                'source': 'John interview - rewards productive travel patterns',
                'adjustment': 47.59,
                'conditions': {
                    'miles_per_day': {'operator': '>=', 'value': 100},
                    'miles_per_day_max': {'operator': '<', 'value': 200}
                },
                'explanation': 'Moderate efficiency shows productive travel without excessive rushing'
            },
            
            'efficiency_high_bonus': {
                'description': "High efficiency (200-300 miles/day) gets road warrior bonus",
                'rule_type': RuleType.BONUS,
                'priority': 7,
                'source': 'John interview - rewards high-intensity travel',
                'adjustment': 33.17,
                'conditions': {
                    'miles_per_day': {'operator': '>=', 'value': 200},
                    'miles_per_day_max': {'operator': '<', 'value': 300}
                },
                'explanation': 'High efficiency shows intensive business activity'
            },
            
            'efficiency_low_penalty': {
                'description': "Low efficiency (0-100 miles/day) gets efficiency penalty",
                'rule_type': RuleType.PENALTY,
                'priority': 8,
                'source': 'John interview - discourages unproductive travel',
                'adjustment': -23.66,
                'conditions': {
                    'miles_per_day': {'operator': '<', 'value': 100}
                },
                'explanation': 'Low efficiency suggests minimal business activity'
            },
            
            'long_trip_high_spending_penalty': {
                'description': "Long trips (7+ days) with high spending rate get judgment penalty",
                'rule_type': RuleType.PENALTY,
                'priority': 9,
                'source': 'Tom interview - discourages extended luxury travel',
                'adjustment': -89.41,
                'conditions': {
                    'trip_duration_days': {'operator': '>=', 'value': 7},
                    'receipts_per_day': {'operator': '>', 'value': 178}
                },
                'explanation': 'Extended trips with high daily spending suggest excessive lifestyle'
            },
            
            'quarter_rounding': {
                'description': "Final result rounded to nearest quarter (system artifact)",
                'rule_type': RuleType.ROUNDING,
                'priority': 10,
                'source': 'Edge case analysis - legacy system rounding behavior',
                'explanation': 'System rounds to quarter dollars for practical financial handling'
            }
        }
    
    def _initialize_rule_precedence(self) -> List[str]:
        """Initialize rule precedence order"""
        return [
            'sweet_spot_bonus',
            'receipt_ceiling', 
            'big_trip_jackpot',
            'distance_bonus',
            'minimum_spending_penalty',
            'efficiency_moderate_bonus',
            'efficiency_high_bonus',
            'efficiency_low_penalty',
            'long_trip_high_spending_penalty',
            'quarter_rounding'
        ]
    
    def _calculate_derived_features(self, trip_duration_days: int, miles_traveled: float, 
                                  total_receipts_amount: float) -> Dict[str, float]:
        """Calculate derived features needed for rule evaluation"""
        return {
            'miles_per_day': miles_traveled / trip_duration_days,
            'receipts_per_day': total_receipts_amount / trip_duration_days
        }
    
    def _evaluate_condition(self, value: float, condition: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        operator = condition['operator']
        
        if operator == '>':
            return value > condition['value']
        elif operator == '>=':
            return value >= condition['value']
        elif operator == '<':
            return value < condition['value']
        elif operator == '<=':
            return value <= condition['value']
        elif operator == '==':
            return value == condition['value']
        elif operator == 'in':
            return value in condition['values']
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def _evaluate_rule_conditions(self, rule: Dict, trip_data: Dict[str, float]) -> bool:
        """Evaluate all conditions for a rule"""
        conditions = rule.get('conditions', {})
        
        for field, condition in conditions.items():
            # Handle max conditions for ranges
            if field.endswith('_max'):
                base_field = field[:-4]
                if not self._evaluate_condition(trip_data[base_field], condition):
                    return False
                continue
            
            # Regular condition evaluation
            if field not in trip_data:
                return False
                
            if not self._evaluate_condition(trip_data[field], condition):
                return False
        
        return True
    
    def _calculate_linear_baseline(self, trip_duration_days: int, miles_traveled: float,
                                 total_receipts_amount: float) -> float:
        """Calculate the linear baseline reimbursement"""
        baseline = self.linear_baseline
        return (baseline['intercept'] +
                baseline['trip_duration_days'] * trip_duration_days +
                baseline['miles_traveled'] * miles_traveled +
                baseline['total_receipts_amount'] * total_receipts_amount)
    
    def _apply_quarter_rounding(self, amount: float) -> float:
        """Apply quarter rounding to final amount"""
        return round(amount * 4) / 4
    
    def predict_reimbursement(self, trip_duration_days: int, miles_traveled: float,
                            total_receipts_amount: float, 
                            return_explanation: bool = False) -> float | Tuple[float, List[RuleApplication]]:
        """
        Predict reimbursement amount using deterministic business rules
        
        Args:
            trip_duration_days: Number of days on trip
            miles_traveled: Total miles traveled  
            total_receipts_amount: Total dollar amount of receipts
            return_explanation: If True, return detailed explanation of rule applications
            
        Returns:
            Predicted reimbursement amount, optionally with explanation
        """
        # Clear previous applications if verbose
        if self.verbose:
            self.rule_applications_log = []
            self.logger.info(f"Processing trip: {trip_duration_days} days, {miles_traveled} miles, ${total_receipts_amount} receipts")
        
        # Calculate linear baseline
        baseline_amount = self._calculate_linear_baseline(
            trip_duration_days, miles_traveled, total_receipts_amount
        )
        
        if self.verbose:
            self.logger.info(f"Linear baseline: ${baseline_amount:.2f}")
        
        # Prepare trip data for rule evaluation
        derived_features = self._calculate_derived_features(
            trip_duration_days, miles_traveled, total_receipts_amount
        )
        
        trip_data = {
            'trip_duration_days': trip_duration_days,
            'miles_traveled': miles_traveled,
            'total_receipts_amount': total_receipts_amount,
            **derived_features
        }
        
        # Apply business rules in precedence order
        total_adjustment = 0
        rule_applications = []
        
        for rule_name in self.rule_precedence:
            if rule_name == 'quarter_rounding':
                continue  # Handle rounding separately
                
            rule = self.business_rules[rule_name]
            condition_met = self._evaluate_rule_conditions(rule, trip_data)
            
            adjustment = rule['adjustment'] if condition_met else 0
            total_adjustment += adjustment
            
            # Log rule application
            application = RuleApplication(
                rule_name=rule_name,
                rule_description=rule['description'],
                condition_met=condition_met,
                adjustment_amount=adjustment,
                explanation=rule['explanation'] if condition_met else f"Conditions not met for {rule['description']}"
            )
            
            rule_applications.append(application)
            
            if self.verbose and condition_met:
                self.logger.info(f"Applied {rule_name}: ${adjustment:.2f} - {rule['explanation']}")
        
        # Calculate pre-rounding amount
        pre_rounding_amount = baseline_amount + total_adjustment
        
        # Apply quarter rounding
        final_amount = self._apply_quarter_rounding(pre_rounding_amount)
        
        if self.verbose:
            self.logger.info(f"Pre-rounding: ${pre_rounding_amount:.2f}")
            self.logger.info(f"Final amount: ${final_amount:.2f}")
        
        # Store applications for potential audit
        self.rule_applications_log = rule_applications
        
        if return_explanation:
            return final_amount, rule_applications
        else:
            return final_amount
    
    def explain_prediction(self, trip_duration_days: int, miles_traveled: float,
                         total_receipts_amount: float) -> Dict[str, Any]:
        """
        Provide detailed explanation of how a reimbursement was calculated
        
        Args:
            trip_duration_days: Number of days on trip
            miles_traveled: Total miles traveled
            total_receipts_amount: Total dollar amount of receipts
            
        Returns:
            Detailed explanation dictionary
        """
        final_amount, rule_applications = self.predict_reimbursement(
            trip_duration_days, miles_traveled, total_receipts_amount, 
            return_explanation=True
        )
        
        # Calculate components
        baseline_amount = self._calculate_linear_baseline(
            trip_duration_days, miles_traveled, total_receipts_amount
        )
        
        total_adjustments = sum(app.adjustment_amount for app in rule_applications)
        pre_rounding = baseline_amount + total_adjustments
        rounding_adjustment = final_amount - pre_rounding
        
        # Build explanation
        explanation = {
            'trip_details': {
                'duration_days': trip_duration_days,
                'miles_traveled': miles_traveled,
                'total_receipts': total_receipts_amount,
                'miles_per_day': miles_traveled / trip_duration_days,
                'receipts_per_day': total_receipts_amount / trip_duration_days
            },
            'calculation_breakdown': {
                'linear_baseline': {
                    'amount': baseline_amount,
                    'formula': f"${self.linear_baseline['intercept']:.2f} + ${self.linear_baseline['trip_duration_days']:.2f}×{trip_duration_days} + ${self.linear_baseline['miles_traveled']:.4f}×{miles_traveled} + ${self.linear_baseline['total_receipts_amount']:.4f}×{total_receipts_amount}",
                    'description': self.linear_baseline['description']
                },
                'business_rule_adjustments': [
                    {
                        'rule_name': app.rule_name,
                        'description': app.rule_description,
                        'applied': app.condition_met,
                        'adjustment': app.adjustment_amount,
                        'explanation': app.explanation
                    }
                    for app in rule_applications if app.rule_name != 'quarter_rounding'
                ],
                'total_adjustments': total_adjustments,
                'pre_rounding_amount': pre_rounding,
                'rounding_adjustment': rounding_adjustment,
                'final_amount': final_amount
            },
            'applied_rules_summary': [
                app.rule_name for app in rule_applications 
                if app.condition_met and app.rule_name != 'quarter_rounding'
            ],
            'business_logic_rationale': [
                app.explanation for app in rule_applications 
                if app.condition_met and app.rule_name != 'quarter_rounding'
            ]
        }
        
        return explanation
    
    def batch_predict(self, trip_data: List[Tuple[int, float, float]]) -> List[float]:
        """
        Predict reimbursements for multiple trips
        
        Args:
            trip_data: List of (trip_duration_days, miles_traveled, total_receipts_amount) tuples
            
        Returns:
            List of predicted reimbursement amounts
        """
        return [
            self.predict_reimbursement(days, miles, receipts)
            for days, miles, receipts in trip_data
        ]
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about the rule system"""
        return {
            'total_rules': len(self.business_rules),
            'rule_types': {
                rule_type.value: sum(1 for rule in self.business_rules.values() 
                                   if rule.get('rule_type') == rule_type)
                for rule_type in RuleType
            },
            'rule_precedence': self.rule_precedence,
            'linear_baseline_coefficients': self.linear_baseline,
            'rule_sources': {
                rule_name: rule['source'] 
                for rule_name, rule in self.business_rules.items()
            }
        }
    
    def validate_rule_consistency(self) -> Dict[str, Any]:
        """Validate that the rule system is internally consistent"""
        issues = []
        warnings = []
        
        # Check for conflicting rules
        bonus_rules = [name for name, rule in self.business_rules.items() 
                      if rule.get('rule_type') == RuleType.BONUS]
        penalty_rules = [name for name, rule in self.business_rules.items() 
                        if rule.get('rule_type') == RuleType.PENALTY]
        
        # Check precedence order matches rule priorities
        expected_order = sorted(
            [(name, rule.get('priority', 999)) for name, rule in self.business_rules.items()],
            key=lambda x: x[1]
        )
        
        if [name for name, _ in expected_order if name != 'quarter_rounding'] != [name for name in self.rule_precedence if name != 'quarter_rounding']:
            issues.append("Rule precedence order does not match priority values")
        
        # Check for reasonable adjustment amounts
        for rule_name, rule in self.business_rules.items():
            if rule.get('rule_type') != RuleType.ROUNDING:
                adjustment = abs(rule.get('adjustment', 0))
                if adjustment > 500:
                    warnings.append(f"Large adjustment amount in {rule_name}: ${adjustment:.2f}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'bonus_rules_count': len(bonus_rules),
            'penalty_rules_count': len(penalty_rules),
            'total_rules': len(self.business_rules)
        }

if __name__ == "__main__":
    # Demonstrate the Deterministic Rule Engine
    print("🎯 DETERMINISTIC RULE ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize engine
    engine = DeterministicRuleEngine(verbose=True)
    
    # Validate rule system
    validation = engine.validate_rule_consistency()
    print(f"\nRule System Validation:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Issues: {validation['issues']}")
    print(f"  Warnings: {validation['warnings']}")
    print(f"  Total rules: {validation['total_rules']}")
    
    # Test cases from different interview scenarios
    test_cases = [
        (3, 90, 200, "Short trip, low efficiency (John's penalty case)"),
        (5, 300, 800, "Sweet spot trip (Peggy's bonus case)"),
        (10, 1200, 1500, "Big trip jackpot (Sarah's bonus case)"),
        (2, 50, 80, "Minimal trip (minimum penalty case)"),
        (7, 800, 2500, "Long trip, high spending (Tom's penalty case)")
    ]
    
    print(f"\n🧪 TEST CASES:")
    print("-" * 40)
    
    for days, miles, receipts, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input: {days} days, {miles} miles, ${receipts} receipts")
        
        amount = engine.predict_reimbursement(days, miles, receipts)
        print(f"  Predicted: ${amount:.2f}")
        
        # Get detailed explanation
        explanation = engine.explain_prediction(days, miles, receipts)
        print(f"  Applied rules: {', '.join(explanation['applied_rules_summary']) if explanation['applied_rules_summary'] else 'None'}")
    
    print(f"\n📊 RULE SYSTEM STATISTICS:")
    stats = engine.get_rule_statistics()
    print(f"  Total rules: {stats['total_rules']}")
    print(f"  Bonus rules: {stats['rule_types']['bonus']}")
    print(f"  Penalty rules: {stats['rule_types']['penalty']}")
    print(f"  Rule sources: {len(set(stats['rule_sources'].values()))} distinct sources")
    
    print("\n" + "=" * 60)
    print("🎯 DETERMINISTIC RULE ENGINE READY FOR PRODUCTION")