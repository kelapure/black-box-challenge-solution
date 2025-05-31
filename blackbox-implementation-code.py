# main.py - Master Implementation File
"""
Black Box Challenge: Legacy Reimbursement System Reverse Engineering
Main implementation following Explore, Plan, Code, Commit workflow
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHASE 1: DATA UNDERSTANDING & EXPLORATION
# ============================================================================

class DataExplorer:
    """Task 1.1-1.3: Data Loading, Validation, and EDA"""
    
    def __init__(self):
        self.data = None
        self.insights = {}
        
    def explore(self):
        """EXPLORE phase: Understand the data"""
        print("=== PHASE 1: DATA EXPLORATION ===")
        print("Loading and analyzing test_cases.json...")
        
    def plan(self):
        """PLAN phase: Design data analysis approach"""
        return {
            'steps': [
                'Load JSON data',
                'Validate data integrity',
                'Statistical analysis',
                'Pattern identification',
                'Document analysis'
            ],
            'metrics': ['completeness', 'distributions', 'correlations']
        }
    
    def code(self, data_path='test_cases.json'):
        """CODE phase: Implement data exploration"""
        # Load data
        with open(data_path, 'r') as f:
            self.data = pd.DataFrame(json.load(f))
        
        # Validate data
        assert all(col in self.data.columns for col in 
                  ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement'])
        
        # Statistical analysis
        self.insights['stats'] = self.data.describe()
        self.insights['correlations'] = self.data.corr()
        
        # Pattern identification
        self.insights['patterns'] = self._identify_patterns()
        
        return self.insights
    
    def _identify_patterns(self):
        """Identify initial patterns in the data"""
        patterns = {}
        
        # Check for linear relationships
        for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']:
            corr = self.data[col].corr(self.data['reimbursement'])
            patterns[f'{col}_correlation'] = corr
        
        # Check for common ratios
        patterns['reimbursement_per_day'] = (self.data['reimbursement'] / 
                                            self.data['trip_duration_days']).describe()
        patterns['reimbursement_per_mile'] = (self.data['reimbursement'] / 
                                             self.data['miles_traveled']).describe()
        
        return patterns
    
    def commit(self):
        """COMMIT phase: Document findings"""
        report = {
            'data_shape': self.data.shape,
            'insights': self.insights,
            'initial_observations': self._generate_observations()
        }
        return report
    
    def _generate_observations(self):
        """Generate human-readable observations"""
        obs = []
        if self.insights['correlations']['reimbursement']['miles_traveled'] > 0.8:
            obs.append("Strong correlation with miles traveled")
        if self.insights['correlations']['reimbursement']['trip_duration_days'] > 0.8:
            obs.append("Strong correlation with trip duration")
        return obs

# ============================================================================
# PHASE 2A: RULE-BASED DISCOVERY
# ============================================================================

class RuleDiscovery:
    """Task 2A.1-2A.4: Rule-based pattern discovery"""
    
    def __init__(self, data):
        self.data = data
        self.rules = []
        
    def explore(self):
        """EXPLORE phase: Understand potential rules"""
        print("\n=== PHASE 2A: RULE DISCOVERY ===")
        print("Exploring potential business rules...")
        
    def plan(self):
        """PLAN phase: Design rule discovery approach"""
        return {
            'rule_types': [
                'Linear combinations',
                'Piecewise functions',
                'Threshold-based rules',
                'Historical business patterns'
            ],
            'methods': [
                'Regression analysis',
                'Decision tree extraction',
                'Segmentation analysis'
            ]
        }
    
    def code(self):
        """CODE phase: Implement rule discovery"""
        self.rules = []
        
        # Test linear combinations
        self._test_linear_rules()
        
        # Test piecewise linear functions
        self._test_piecewise_rules()
        
        # Test business logic patterns
        self._test_business_rules()
        
        # Test threshold rules
        self._test_threshold_rules()
        
        return self.rules
    
    def _test_linear_rules(self):
        """Test for linear combination rules"""
        X = self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
        y = self.data['reimbursement']
        
        # Simple linear regression
        lr = LinearRegression()
        lr.fit(X, y)
        pred = lr.predict(X)
        mae = mean_absolute_error(y, pred)
        
        if mae < 1.0:  # Good linear fit
            self.rules.append({
                'type': 'linear',
                'formula': f"{lr.coef_[0]:.2f}*days + {lr.coef_[1]:.2f}*miles + {lr.coef_[2]:.2f}*receipts + {lr.intercept_:.2f}",
                'coefficients': lr.coef_,
                'intercept': lr.intercept_,
                'mae': mae
            })
    
    def _test_piecewise_rules(self):
        """Test for piecewise linear functions"""
        # Test different breakpoints
        for threshold_col in ['trip_duration_days', 'miles_traveled']:
            for threshold in [5, 7, 10, 100, 200, 500]:
                mask = self.data[threshold_col] <= threshold
                if mask.sum() > 10 and (~mask).sum() > 10:  # Enough samples
                    # Fit separate models
                    rule = self._fit_piecewise(threshold_col, threshold, mask)
                    if rule['improvement'] > 0.1:  # Significant improvement
                        self.rules.append(rule)
    
    def _fit_piecewise(self, col, threshold, mask):
        """Fit piecewise linear model"""
        X = self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
        y = self.data['reimbursement']
        
        # Fit two separate models
        lr1, lr2 = LinearRegression(), LinearRegression()
        lr1.fit(X[mask], y[mask])
        lr2.fit(X[~mask], y[~mask])
        
        # Calculate improvement
        pred = np.where(mask, lr1.predict(X), lr2.predict(X))
        mae = mean_absolute_error(y, pred)
        
        return {
            'type': 'piecewise',
            'condition': f"{col} <= {threshold}",
            'model_low': lr1,
            'model_high': lr2,
            'mae': mae,
            'improvement': 0.1  # Placeholder
        }
    
    def _test_business_rules(self):
        """Test common business reimbursement patterns"""
        # Per diem patterns
        daily_rate_patterns = [50, 75, 100, 125, 150]
        for rate in daily_rate_patterns:
            pred = self.data['trip_duration_days'] * rate
            if mean_absolute_error(self.data['reimbursement'], pred) < 50:
                self.rules.append({
                    'type': 'per_diem',
                    'rate': rate,
                    'formula': f"days * {rate}"
                })
        
        # Mileage reimbursement patterns (common rates from 1960s-2020s)
        mileage_rates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.58]
        for rate in mileage_rates:
            pred = self.data['miles_traveled'] * rate
            if mean_absolute_error(self.data['reimbursement'], pred) < 50:
                self.rules.append({
                    'type': 'mileage',
                    'rate': rate,
                    'formula': f"miles * {rate}"
                })
    
    def _test_threshold_rules(self):
        """Test for threshold-based rules"""
        # Receipt threshold rules
        receipt_thresholds = [0, 25, 50, 100, 200]
        for threshold in receipt_thresholds:
            mask = self.data['total_receipts_amount'] > threshold
            if mask.sum() > 10:
                # Test if different rules apply above/below threshold
                self._analyze_threshold_split(mask, threshold)
    
    def _analyze_threshold_split(self, mask, threshold):
        """Analyze rules above/below threshold"""
        # Placeholder for threshold analysis
        pass
    
    def commit(self):
        """COMMIT phase: Select best rules"""
        # Sort rules by accuracy
        valid_rules = [r for r in self.rules if 'mae' in r]
        best_rules = sorted(valid_rules, key=lambda x: x['mae'])[:5]
        return best_rules

# ============================================================================
# PHASE 2B: MACHINE LEARNING APPROACH
# ============================================================================

class MLModels:
    """Task 2B.1-2B.4: Machine learning models"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.results = {}
        
    def explore(self):
        """EXPLORE phase: Understand ML requirements"""
        print("\n=== PHASE 2B: MACHINE LEARNING ===")
        print("Exploring ML model options...")
        
    def plan(self):
        """PLAN phase: Design ML approach"""
        return {
            'models': [
                'RandomForest',
                'GradientBoosting',
                'Ridge Regression',
                'Neural Network (if needed)'
            ],
            'validation': '5-fold cross-validation',
            'metrics': ['MAE', 'RMSE', 'Exact match rate']
        }
    
    def code(self):
        """CODE phase: Implement ML models"""
        X = self.data[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
        y = self.data['reimbursement']
        
        # Feature engineering
        X_enhanced = self._engineer_features(X)
        
        # Train multiple models
        models = {
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            # Cross-validation
            scores = cross_val_score(model, X_enhanced, y, cv=cv, 
                                   scoring='neg_mean_absolute_error')
            
            # Fit on full data
            model.fit(X_enhanced, y)
            pred = model.predict(X_enhanced)
            
            self.models[name] = model
            self.results[name] = {
                'cv_mae': -scores.mean(),
                'mae': mean_absolute_error(y, pred),
                'exact_matches': np.sum(np.abs(y - pred) <= 0.01) / len(y),
                'close_matches': np.sum(np.abs(y - pred) <= 1.00) / len(y)
            }
        
        return self.results
    
    def _engineer_features(self, X):
        """Create additional features"""
        X_enhanced = X.copy()
        
        # Interaction features
        X_enhanced['days_miles'] = X['trip_duration_days'] * X['miles_traveled']
        X_enhanced['days_receipts'] = X['trip_duration_days'] * X['total_receipts_amount']
        X_enhanced['miles_receipts'] = X['miles_traveled'] * X['total_receipts_amount']
        
        # Ratio features
        X_enhanced['miles_per_day'] = X['miles_traveled'] / (X['trip_duration_days'] + 1)
        X_enhanced['receipts_per_day'] = X['total_receipts_amount'] / (X['trip_duration_days'] + 1)
        X_enhanced['receipts_per_mile'] = X['total_receipts_amount'] / (X['miles_traveled'] + 1)
        
        # Polynomial features (squared terms)
        X_enhanced['days_squared'] = X['trip_duration_days'] ** 2
        X_enhanced['miles_squared'] = X['miles_traveled'] ** 2
        
        return X_enhanced
    
    def commit(self):
        """COMMIT phase: Select best model"""
        best_model_name = min(self.results, key=lambda x: self.results[x]['mae'])
        return {
            'best_model': best_model_name,
            'model': self.models[best_model_name],
            'performance': self.results[best_model_name]
        }

# ============================================================================
# PHASE 3: MODEL EVALUATION & SELECTION
# ============================================================================

class ModelEvaluator:
    """Task 3.1-3.3: Model evaluation and selection"""
    
    def __init__(self, rules, ml_models, data):
        self.rules = rules
        self.ml_models = ml_models
        self.data = data
        self.best_model = None
        
    def explore(self):
        """EXPLORE phase: Understand evaluation needs"""
        print("\n=== PHASE 3: MODEL EVALUATION ===")
        print("Evaluating all models...")
        
    def plan(self):
        """PLAN phase: Design evaluation strategy"""
        return {
            'evaluation_criteria': [
                'Exact match rate (±$0.01)',
                'Close match rate (±$1.00)',
                'Mean Absolute Error',
                'Maximum Error',
                'Interpretability'
            ],
            'selection_method': 'Weighted scoring'
        }
    
    def code(self):
        """CODE phase: Implement evaluation"""
        results = []
        
        # Evaluate rule-based models
        for rule in self.rules:
            if 'mae' in rule:
                results.append({
                    'type': 'rule',
                    'model': rule,
                    'score': self._calculate_score(rule['mae'])
                })
        
        # Evaluate ML models
        for name, performance in self.ml_models.items():
            results.append({
                'type': 'ml',
                'name': name,
                'performance': performance,
                'score': self._calculate_score(performance['mae'])
            })
        
        # Select best model
        self.best_model = min(results, key=lambda x: x['score'])
        return self.best_model
    
    def _calculate_score(self, mae):
        """Calculate weighted score for model selection"""
        # Lower MAE is better
        return mae
    
    def commit(self):
        """COMMIT phase: Finalize model selection"""
        return {
            'selected_model': self.best_model,
            'justification': 'Lowest MAE with good generalization'
        }

# ============================================================================
# PHASE 4: FINAL IMPLEMENTATION
# ============================================================================

class FinalImplementation:
    """Task 4.1-4.3: Final implementation and optimization"""
    
    def __init__(self, best_model):
        self.model = best_model
        self.implementation = None
        
    def explore(self):
        """EXPLORE phase: Understand implementation requirements"""
        print("\n=== PHASE 4: FINAL IMPLEMENTATION ===")
        print("Creating final implementation...")
        
    def plan(self):
        """PLAN phase: Design final implementation"""
        return {
            'components': [
                'Input validation',
                'Feature engineering',
                'Model prediction',
                'Output formatting'
            ],
            'optimizations': [
                'Edge case handling',
                'Numerical stability',
                'Performance optimization'
            ]
        }
    
    def code(self):
        """CODE phase: Implement final solution"""
        # This will be the actual implementation used in run.sh
        implementation = '''
def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Calculate reimbursement amount using discovered model"""
    
    # Input validation
    trip_duration_days = max(0, int(trip_duration_days))
    miles_traveled = max(0, int(miles_traveled))
    total_receipts_amount = max(0, float(total_receipts_amount))
    
    # Apply discovered model (placeholder - will be replaced with actual model)
    # Example: Linear combination discovered through analysis
    reimbursement = (
        50.0 * trip_duration_days +
        0.45 * miles_traveled +
        0.8 * total_receipts_amount +
        25.0  # base amount
    )
    
    # Ensure non-negative and round to 2 decimal places
    reimbursement = max(0, round(reimbursement, 2))
    
    return reimbursement
'''
        self.implementation = implementation
        return implementation
    
    def commit(self):
        """COMMIT phase: Create final deliverable"""
        # Create run.sh script
        run_sh_content = '''#!/bin/bash
# Legacy Reimbursement System Implementation

python3 -c "
import sys

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    # [Implementation will be inserted here based on discovered model]
    trip_duration_days = max(0, int(trip_duration_days))
    miles_traveled = max(0, int(miles_traveled))
    total_receipts_amount = max(0, float(total_receipts_amount))
    
    # Placeholder formula - will be replaced with discovered model
    reimbursement = (
        50.0 * trip_duration_days +
        0.45 * miles_traveled +
        0.8 * total_receipts_amount +
        25.0
    )
    
    return round(max(0, reimbursement), 2)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Error: Requires 3 arguments', file=sys.stderr)
        sys.exit(1)
    
    result = calculate_reimbursement(sys.argv[1], sys.argv[2], sys.argv[3])
    print(result)
"
'''
        return {
            'run_sh': run_sh_content,
            'implementation': self.implementation
        }

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class BlackBoxSolver:
    """Main orchestrator for the black box challenge"""
    
    def __init__(self):
        self.data_explorer = DataExplorer()
        self.rule_discovery = None
        self.ml_models = None
        self.evaluator = None
        self.final_impl = None
        
    def solve(self):
        """Execute the complete solution pipeline"""
        print("Starting Black Box Challenge Solution Pipeline")
        print("=" * 50)
        
        # Phase 1: Data Exploration
        self.data_explorer.explore()
        plan = self.data_explorer.plan()
        insights = self.data_explorer.code()
        report = self.data_explorer.commit()
        
        # Phase 2A: Rule Discovery (can run in parallel with 2B)
        self.rule_discovery = RuleDiscovery(self.data_explorer.data)
        self.rule_discovery.explore()
        rule_plan = self.rule_discovery.plan()
        rules = self.rule_discovery.code()
        best_rules = self.rule_discovery.commit()
        
        # Phase 2B: Machine Learning
        self.ml_models = MLModels(self.data_explorer.data)
        self.ml_models.explore()
        ml_plan = self.ml_models.plan()
        ml_results = self.ml_models.code()
        best_ml = self.ml_models.commit()
        
        # Phase 3: Evaluation
        self.evaluator = ModelEvaluator(best_rules, ml_results, self.data_explorer.data)
        self.evaluator.explore()
        eval_plan = self.evaluator.plan()
        best_model = self.evaluator.code()
        final_selection = self.evaluator.commit()
        
        # Phase 4: Final Implementation
        self.final_impl = FinalImplementation(best_model)
        self.final_impl.explore()
        impl_plan = self.final_impl.plan()
        implementation = self.final_impl.code()
        deliverables = self.final_impl.commit()
        
        print("\n" + "=" * 50)
        print("Solution Pipeline Complete!")
        
        return {
            'insights': report,
            'best_rules': best_rules,
            'best_ml_model': best_ml,
            'final_model': final_selection,
            'deliverables': deliverables
        }

if __name__ == "__main__":
    solver = BlackBoxSolver()
    solution = solver.solve()
    
    # Save run.sh
    with open('run.sh', 'w') as f:
        f.write(solution['deliverables']['run_sh'])
    
    import os
    os.chmod('run.sh', 0o755)
    
    print("\nSolution saved to run.sh")
    print("Run ./eval.sh to test the solution")
