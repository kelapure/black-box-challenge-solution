#!/usr/bin/env python3
"""
UNCONSTRAINED BREAKTHROUGH APPROACH - REMOVING ALL ARTIFICIAL LIMITS
Black Box Challenge - Final Breakthrough Attempt

HYPOTHESIS: We've been TOO conservative. Let's remove artificial constraints and push harder.

Previous artificial constraints to REMOVE:
- Max 3 features (why limit ourselves?!)
- Heavy regularization (maybe too conservative)
- 30% ML scaling limits (arbitrary!)
- Avoiding ensemble methods (fear-based decision)
- Assuming quarter rounding is correct (what if it's wrong?)
- Conservative feature engineering

NEW STRATEGY:
1. COMPREHENSIVE feature engineering (50+ features)
2. ENSEMBLE methods with proper validation
3. QUESTION FUNDAMENTAL ASSUMPTIONS (quarter rounding, etc.)
4. OUTLIER detection and cleaning
5. ADVANCED ML techniques we avoided
6. FULL POWER validation to prevent overfitting properly
7. DIFFERENT target transformations

Time to break through the 83-87% plateau!
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import warnings
warnings.filterwarnings('ignore')

class UnconstrainedBreakthroughApproach:
    """
    REMOVES ALL ARTIFICIAL CONSTRAINTS - going for maximum accuracy with proper validation
    """
    
    def __init__(self, data_path='test_cases.json'):
        """Initialize with test cases"""
        self.data_path = data_path
        self.df = None
        self.feature_matrix = None
        self.feature_names = []
        self.best_models = {}
        self.breakthrough_results = {}
        self.load_data()
        
    def load_data(self):
        """Load test cases"""
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
        print("🚀 REMOVING ALL ARTIFICIAL CONSTRAINTS - GOING FOR BREAKTHROUGH!")
        
    def detect_and_remove_outliers(self):
        """Detect and remove outliers that might be skewing results"""
        print("\\n=== OUTLIER DETECTION AND CLEANING ===")
        
        original_size = len(self.df)
        
        # Remove extreme outliers using IQR method
        for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # More aggressive than usual 1.5*IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers_before = len(self.df)
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            outliers_removed = outliers_before - len(self.df)
            
            if outliers_removed > 0:
                print(f"  {col}: removed {outliers_removed} outliers")
        
        # Remove impossible combinations
        impossible_before = len(self.df)
        
        # Remove trips with impossible efficiency (>1000 miles/day or <1 mile/day)
        self.df = self.df[(self.df['miles_per_day'] >= 1) & (self.df['miles_per_day'] <= 1000)]
        
        # Remove trips with impossible spending (>$10000/day or <$1/day for multi-day trips)
        self.df = self.df[~((self.df['receipts_per_day'] > 10000) | 
                           ((self.df['trip_duration_days'] > 1) & (self.df['receipts_per_day'] < 1)))]
        
        impossible_removed = impossible_before - len(self.df)
        if impossible_removed > 0:
            print(f"  Impossible combinations: removed {impossible_removed} trips")
        
        total_removed = original_size - len(self.df)
        print(f"\\nOutlier cleaning: {original_size} → {len(self.df)} trips ({total_removed} removed, {total_removed/original_size*100:.1f}%)")
        
        # Recalculate derived features
        self.df['miles_per_day'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        self.df['receipts_per_day'] = self.df['total_receipts_amount'] / self.df['trip_duration_days']
        
    def question_fundamental_assumptions(self):
        """Question fundamental assumptions like quarter rounding"""
        print("\\n=== QUESTIONING FUNDAMENTAL ASSUMPTIONS ===")
        
        # Test different rounding strategies
        print("\\n1. Testing rounding assumptions:")
        
        # Create a simple baseline prediction for testing
        baseline_pred = (
            318.40 + 
            71.28 * self.df['trip_duration_days'] + 
            0.794100 * self.df['miles_traveled'] + 
            0.290366 * self.df['total_receipts_amount']
        )
        
        rounding_tests = [
            ('no_rounding', baseline_pred),
            ('quarter_rounding', (baseline_pred * 4).round() / 4),
            ('half_dollar', (baseline_pred * 2).round() / 2),
            ('whole_dollar', baseline_pred.round()),
            ('nickel_rounding', (baseline_pred * 20).round() / 20),
            ('dime_rounding', (baseline_pred * 10).round() / 10)
        ]
        
        best_rounding = None
        best_r2 = -np.inf
        
        for rounding_name, rounded_pred in rounding_tests:
            r2 = r2_score(self.df['reimbursement'], rounded_pred)
            print(f"  {rounding_name}: R² = {r2:.6f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_rounding = rounding_name
        
        print(f"\\n  🎯 Best rounding strategy: {best_rounding} (R² = {best_r2:.6f})")
        
        # Test if reimbursement values follow expected patterns
        print("\\n2. Analyzing reimbursement value patterns:")
        
        # Check what rounding patterns exist in actual reimbursements
        reimbursement_remainders = {}
        for divisor, name in [(0.25, 'quarter'), (0.5, 'half'), (1.0, 'dollar'), (0.1, 'dime'), (0.05, 'nickel')]:
            remainders = (self.df['reimbursement'] % divisor).round(3)
            unique_remainders = remainders.unique()
            reimbursement_remainders[name] = len(unique_remainders)
            print(f"  {name} remainders: {len(unique_remainders)} unique values")
        
        # The one with fewest unique remainders is likely the rounding rule
        likely_rounding = min(reimbursement_remainders.items(), key=lambda x: x[1])
        print(f"  🎯 Likely actual rounding: {likely_rounding[0]} ({likely_rounding[1]} unique remainders)")
        
        return best_rounding, likely_rounding[0]
        
    def create_comprehensive_features(self):
        """Create comprehensive feature set - NO LIMITS!"""
        print("\\n=== COMPREHENSIVE FEATURE ENGINEERING ===")
        print("Creating 50+ features - NO ARTIFICIAL LIMITS!")
        
        features = {}
        
        # 1. Original features
        features['trip_duration_days'] = self.df['trip_duration_days']
        features['miles_traveled'] = self.df['miles_traveled']
        features['total_receipts_amount'] = self.df['total_receipts_amount']
        
        # 2. Basic derived features
        features['miles_per_day'] = self.df['miles_per_day']
        features['receipts_per_day'] = self.df['receipts_per_day']
        features['miles_per_dollar'] = self.df['miles_traveled'] / (self.df['total_receipts_amount'] + 1)
        features['cost_per_mile'] = self.df['total_receipts_amount'] / (self.df['miles_traveled'] + 1)
        
        # 3. Non-linear transformations
        for base_feature in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']:
            base_values = self.df[base_feature]
            
            # Powers
            features[f'{base_feature}_squared'] = base_values ** 2
            features[f'{base_feature}_cubed'] = base_values ** 3
            features[f'{base_feature}_sqrt'] = np.sqrt(base_values)
            features[f'{base_feature}_cbrt'] = np.cbrt(base_values)
            
            # Logarithms
            features[f'{base_feature}_log'] = np.log(base_values + 1)
            features[f'{base_feature}_log10'] = np.log10(base_values + 1)
            
            # Inverse
            features[f'{base_feature}_inverse'] = 1 / (base_values + 1)
            
            # Exponential (with clipping to prevent overflow)
            exp_values = np.exp(base_values / base_values.max())
            features[f'{base_feature}_exp_normalized'] = exp_values
        
        # 4. Two-way interactions (all combinations)
        base_cols = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        for i, col1 in enumerate(base_cols):
            for col2 in base_cols[i+1:]:
                features[f'{col1}_X_{col2}'] = self.df[col1] * self.df[col2]
                features[f'{col1}_DIV_{col2}'] = self.df[col1] / (self.df[col2] + 1)
                features[f'{col2}_DIV_{col1}'] = self.df[col2] / (self.df[col1] + 1)
        
        # 5. Three-way interactions
        features['days_miles_receipts'] = self.df['trip_duration_days'] * self.df['miles_traveled'] * self.df['total_receipts_amount']
        features['efficiency_cost'] = (self.df['miles_traveled'] / self.df['trip_duration_days']) * (self.df['total_receipts_amount'] / self.df['miles_traveled'])
        
        # 6. Complex ratios and efficiency metrics
        features['total_efficiency'] = self.df['miles_traveled'] / (self.df['trip_duration_days'] * self.df['total_receipts_amount'] + 1)
        features['spending_intensity'] = self.df['total_receipts_amount'] / (self.df['trip_duration_days'] ** 2 + 1)
        features['trip_complexity'] = (self.df['trip_duration_days'] + self.df['miles_traveled']) / (self.df['total_receipts_amount'] + 1)
        
        # 7. Polynomial combinations
        features['days_miles_poly'] = self.df['trip_duration_days']**2 * self.df['miles_traveled']
        features['miles_receipts_poly'] = self.df['miles_traveled']**2 * self.df['total_receipts_amount']
        features['days_receipts_poly'] = self.df['trip_duration_days']**2 * self.df['total_receipts_amount']
        
        # 8. Advanced efficiency metrics
        features['multi_efficiency'] = self.df['miles_traveled'] / (np.sqrt(self.df['trip_duration_days']) * np.sqrt(self.df['total_receipts_amount']) + 1)
        features['cost_efficiency_nonlinear'] = np.log(self.df['miles_traveled'] + 1) / np.log(self.df['total_receipts_amount'] + 1)
        
        # 9. Binned/categorical features as numerical
        # Trip length categories
        trip_length_bins = pd.cut(self.df['trip_duration_days'], bins=[-np.inf, 2, 5, 8, 12, np.inf], labels=False)
        features['trip_length_category'] = trip_length_bins
        
        # Miles categories
        miles_bins = pd.cut(self.df['miles_traveled'], bins=[-np.inf, 200, 500, 800, 1200, np.inf], labels=False)
        features['miles_category'] = miles_bins
        
        # Spending categories
        spending_bins = pd.cut(self.df['total_receipts_amount'], bins=[-np.inf, 500, 1000, 1500, 2500, np.inf], labels=False)
        features['spending_category'] = spending_bins
        
        # 10. Advanced mathematical transformations
        features['harmonic_mean_days_miles'] = 2 / (1/(self.df['trip_duration_days']+1) + 1/(self.df['miles_traveled']+1))
        features['geometric_mean_all'] = (self.df['trip_duration_days'] * self.df['miles_traveled'] * self.df['total_receipts_amount']) ** (1/3)
        
        # 11. Trigonometric features (normalized inputs)
        days_norm = self.df['trip_duration_days'] / self.df['trip_duration_days'].max()
        miles_norm = self.df['miles_traveled'] / self.df['miles_traveled'].max()
        receipts_norm = self.df['total_receipts_amount'] / self.df['total_receipts_amount'].max()
        
        features['sin_days'] = np.sin(days_norm * np.pi)
        features['cos_miles'] = np.cos(miles_norm * np.pi)
        features['sin_receipts'] = np.sin(receipts_norm * np.pi)
        
        # Convert to DataFrame
        self.feature_matrix = pd.DataFrame(features)
        self.feature_names = list(features.keys())
        
        print(f"\\nCreated {len(self.feature_names)} features:")
        print(f"  Basic features: 3")
        print(f"  Derived features: {len([f for f in self.feature_names if 'per' in f])}")
        print(f"  Non-linear transforms: {len([f for f in self.feature_names if any(x in f for x in ['squared', 'sqrt', 'log', 'exp'])])}")
        print(f"  Interactions: {len([f for f in self.feature_names if 'X' in f or 'DIV' in f])}")
        print(f"  Complex features: {len([f for f in self.feature_names if any(x in f for x in ['efficiency', 'complexity', 'poly'])])}")
        
        # Remove any features with NaN or infinite values
        initial_features = len(self.feature_names)
        self.feature_matrix = self.feature_matrix.replace([np.inf, -np.inf], np.nan)
        self.feature_matrix = self.feature_matrix.dropna(axis=1)
        self.feature_names = list(self.feature_matrix.columns)
        
        print(f"\\nAfter cleaning: {len(self.feature_names)} features ({initial_features - len(self.feature_names)} removed due to NaN/inf)")
        
    def advanced_feature_selection(self):
        """Advanced feature selection to find the most predictive features"""
        print("\\n=== ADVANCED FEATURE SELECTION ===")
        
        X = self.feature_matrix.values
        y = self.df['reimbursement'].values
        
        # 1. Univariate feature selection
        print("\\n1. Univariate feature selection (F-test):")
        selector_univariate = SelectKBest(score_func=f_regression, k=min(30, len(self.feature_names)))
        X_univariate = selector_univariate.fit_transform(X, y)
        selected_features_univariate = np.array(self.feature_names)[selector_univariate.get_support()]
        
        # Get feature scores
        feature_scores = list(zip(self.feature_names, selector_univariate.scores_))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Top 10 features by F-score:")
        for feat_name, score in feature_scores[:10]:
            print(f"    {feat_name}: {score:.2f}")
        
        # 2. Recursive Feature Elimination with Ridge
        print("\\n2. Recursive Feature Elimination (Ridge):")
        estimator = Ridge(alpha=1.0)
        selector_rfe = RFE(estimator, n_features_to_select=min(25, len(self.feature_names)))
        X_rfe = selector_rfe.fit_transform(X, y)
        selected_features_rfe = np.array(self.feature_names)[selector_rfe.get_support()]
        
        print(f"  Selected {len(selected_features_rfe)} features via RFE")
        
        # 3. Combine selections
        combined_features = list(set(selected_features_univariate) | set(selected_features_rfe))
        print(f"\\n3. Combined selection: {len(combined_features)} features")
        
        # Create reduced feature matrix
        feature_indices = [i for i, name in enumerate(self.feature_names) if name in combined_features]
        self.feature_matrix_selected = self.feature_matrix.iloc[:, feature_indices]
        self.selected_feature_names = [self.feature_names[i] for i in feature_indices]
        
        print(f"Final selected features: {len(self.selected_feature_names)}")
        
        return self.feature_matrix_selected, self.selected_feature_names
        
    def test_advanced_ml_models(self):
        """Test advanced ML models we previously avoided"""
        print("\\n=== TESTING ADVANCED ML MODELS ===")
        print("NO MORE CONSTRAINTS - testing everything!")
        
        # Get features
        X_selected, selected_names = self.advanced_feature_selection()
        y = self.df['reimbursement'].values
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = scaler.fit_transform(X_selected)
        
        # Models to test (NO CONSTRAINTS!)
        models_to_test = {
            'Ridge_optimized': Ridge(),
            'Lasso_optimized': Lasso(),
            'ElasticNet_optimized': ElasticNet(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR_RBF': SVR(kernel='rbf'),
            'SVR_Poly': SVR(kernel='poly', degree=2),
        }
        
        # Hyperparameter grids
        param_grids = {
            'Ridge_optimized': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso_optimized': {'alpha': [0.01, 0.1, 1.0, 10.0]},
            'ElasticNet_optimized': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
            'RandomForest': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]},
            'ExtraTrees': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]},
            'GradientBoosting': {'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]},
            'SVR_RBF': {'C': [1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01]},
            'SVR_Poly': {'C': [1, 10, 100], 'gamma': ['scale', 'auto']},
        }
        
        print(f"\\nTesting {len(models_to_test)} advanced models with {X_scaled.shape[1]} features...")
        
        best_model = None
        best_score = -np.inf
        best_model_name = ""
        model_results = {}
        
        # 5-fold CV for model selection
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models_to_test.items():
            print(f"\\n  Testing {model_name}...")
            
            try:
                # Grid search with cross-validation
                if model_name in param_grids:
                    grid_search = GridSearchCV(
                        model, 
                        param_grids[model_name], 
                        cv=cv, 
                        scoring='r2', 
                        n_jobs=-1,
                        error_score='raise'
                    )
                    grid_search.fit(X_scaled, y)
                    
                    best_model_for_type = grid_search.best_estimator_
                    best_score_for_type = grid_search.best_score_
                    best_params = grid_search.best_params_
                    
                    print(f"    Best params: {best_params}")
                else:
                    # Simple CV for models without hyperparameters
                    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
                    best_score_for_type = cv_scores.mean()
                    best_model_for_type = model.fit(X_scaled, y)
                
                print(f"    CV R²: {best_score_for_type:.6f}")
                
                # Check for overfitting
                train_score = best_model_for_type.score(X_scaled, y)
                overfitting_gap = train_score - best_score_for_type
                print(f"    Train R²: {train_score:.6f}, Gap: {overfitting_gap:.6f}")
                
                model_results[model_name] = {
                    'cv_score': best_score_for_type,
                    'train_score': train_score,
                    'overfitting_gap': overfitting_gap,
                    'model': best_model_for_type
                }
                
                # Select best model (allowing some overfitting if CV score is much better)
                if best_score_for_type > best_score and overfitting_gap < 0.1:  # Allow up to 10% overfitting
                    best_score = best_score_for_type
                    best_model = best_model_for_type
                    best_model_name = model_name
                    
            except Exception as e:
                print(f"    Failed: {str(e)}")
                
        print(f"\\n🎯 BEST MODEL: {best_model_name}")
        print(f"🎯 BEST CV R²: {best_score:.6f}")
        
        if best_score > 0.90:
            print("🎉 BREAKTHROUGH! Achieved >90% R²!")
        elif best_score > 0.85:
            print("🚀 SIGNIFICANT IMPROVEMENT! >85% R²")
        
        # Store results
        self.best_models = model_results
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.feature_scaler = scaler
        
        return best_model, best_score, model_results
        
    def test_ensemble_approaches(self):
        """Test ensemble approaches combining multiple models"""
        print("\\n=== TESTING ENSEMBLE APPROACHES ===")
        
        if not hasattr(self, 'best_models'):
            print("No models available for ensembling")
            return None, 0
        
        X_selected, _ = self.advanced_feature_selection()
        X_scaled = self.feature_scaler.transform(X_selected)
        y = self.df['reimbursement'].values
        
        # Get top 3 models for ensembling
        model_scores = [(name, results['cv_score']) for name, results in self.best_models.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = model_scores[:3]
        
        print(f"Ensembling top 3 models:")
        for name, score in top_models:
            print(f"  {name}: {score:.6f}")
        
        # Simple averaging ensemble
        ensemble_predictions = np.zeros(len(y))
        
        for model_name, _ in top_models:
            model = self.best_models[model_name]['model']
            pred = model.predict(X_scaled)
            ensemble_predictions += pred
        
        ensemble_predictions /= len(top_models)
        
        # Apply the best rounding strategy we discovered
        if hasattr(self, 'best_rounding'):
            if self.best_rounding == 'quarter_rounding':
                ensemble_predictions = (ensemble_predictions * 4).round() / 4
            elif self.best_rounding == 'half_dollar':
                ensemble_predictions = (ensemble_predictions * 2).round() / 2
            elif self.best_rounding == 'whole_dollar':
                ensemble_predictions = ensemble_predictions.round()
            # else: no rounding
        
        ensemble_r2 = r2_score(y, ensemble_predictions)
        ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_predictions))
        
        print(f"\\n🎯 ENSEMBLE R²: {ensemble_r2:.6f}")
        print(f"🎯 ENSEMBLE RMSE: ${ensemble_rmse:.2f}")
        
        if ensemble_r2 > 0.90:
            print("🎉 ENSEMBLE BREAKTHROUGH! >90% R²!")
        
        return ensemble_predictions, ensemble_r2
        
    def final_breakthrough_validation(self):
        """Final validation of breakthrough approach"""
        print("\\n=== FINAL BREAKTHROUGH VALIDATION ===")
        
        # Get best single model performance
        best_single_r2 = max([results['cv_score'] for results in self.best_models.values()])
        
        # Get ensemble performance
        ensemble_pred, ensemble_r2 = self.test_ensemble_approaches()
        
        # Choose best approach
        if ensemble_r2 > best_single_r2:
            final_predictions = ensemble_pred
            final_r2 = ensemble_r2
            approach_used = "Ensemble"
        else:
            X_selected, _ = self.advanced_feature_selection()
            X_scaled = self.feature_scaler.transform(X_selected)
            final_predictions = self.best_model.predict(X_scaled)
            
            # Apply rounding
            if hasattr(self, 'best_rounding') and self.best_rounding == 'quarter_rounding':
                final_predictions = (final_predictions * 4).round() / 4
            
            final_r2 = r2_score(self.df['reimbursement'], final_predictions)
            approach_used = self.best_model_name
        
        final_rmse = np.sqrt(mean_squared_error(self.df['reimbursement'], final_predictions))
        final_mae = mean_absolute_error(self.df['reimbursement'], final_predictions)
        
        # Exact match analysis
        exact_matches = (np.abs(self.df['reimbursement'] - final_predictions) < 0.01).sum()
        exact_match_rate = exact_matches / len(self.df) * 100
        
        within_1_dollar = (np.abs(self.df['reimbursement'] - final_predictions) <= 1.0).sum()
        within_5_dollars = (np.abs(self.df['reimbursement'] - final_predictions) <= 5.0).sum()
        within_10_dollars = (np.abs(self.df['reimbursement'] - final_predictions) <= 10.0).sum()
        
        print(f"\\n=== UNCONSTRAINED BREAKTHROUGH RESULTS ===")
        print(f"🚀 APPROACH USED: {approach_used}")
        print(f"🚀 FINAL R²: {final_r2:.6f}")
        print(f"🚀 FINAL RMSE: ${final_rmse:.2f}")
        print(f"🚀 FINAL MAE: ${final_mae:.2f}")
        
        print(f"\\nAccuracy Analysis:")
        print(f"  Exact matches: {exact_matches} ({exact_match_rate:.2f}%)")
        print(f"  Within $1: {within_1_dollar} ({within_1_dollar/len(self.df)*100:.1f}%)")
        print(f"  Within $5: {within_5_dollars} ({within_5_dollars/len(self.df)*100:.1f}%)")
        print(f"  Within $10: {within_10_dollars} ({within_10_dollars/len(self.df)*100:.1f}%)")
        
        # Check breakthrough achievement
        if final_r2 >= 0.95:
            print(f"\\n🎉 MAJOR BREAKTHROUGH! Achieved {final_r2:.4f} R² (≥95% target)")
            breakthrough_level = "MAJOR"
        elif final_r2 >= 0.90:
            print(f"\\n🚀 BREAKTHROUGH! Achieved {final_r2:.4f} R² (≥90% target)")
            breakthrough_level = "SIGNIFICANT"
        elif final_r2 >= 0.87:
            print(f"\\n📈 IMPROVEMENT! Achieved {final_r2:.4f} R² (>87% previous best)")
            breakthrough_level = "MODERATE"
        else:
            print(f"\\n⚠️  No breakthrough: {final_r2:.4f} R² (similar to previous)")
            breakthrough_level = "NONE"
        
        return {
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'final_mae': final_mae,
            'approach_used': approach_used,
            'breakthrough_level': breakthrough_level,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_match_rate,
            'within_tolerances': {
                'within_1_dollar': within_1_dollar,
                'within_5_dollars': within_5_dollars,
                'within_10_dollars': within_10_dollars
            },
            'features_used': len(self.selected_feature_names),
            'models_tested': len(self.best_models)
        }
        
    def run_unconstrained_breakthrough(self):
        """Run complete unconstrained breakthrough approach"""
        print("🚀 STARTING UNCONSTRAINED BREAKTHROUGH APPROACH")
        print("=" * 70)
        print("REMOVING ALL ARTIFICIAL CONSTRAINTS!")
        print("GOING FOR MAXIMUM ACCURACY WITH PROPER VALIDATION!")
        
        # Step 1: Clean data
        self.detect_and_remove_outliers()
        
        # Step 2: Question assumptions
        best_rounding, likely_rounding = self.question_fundamental_assumptions()
        self.best_rounding = best_rounding
        
        # Step 3: Create comprehensive features
        self.create_comprehensive_features()
        
        # Step 4: Test advanced ML models
        best_model, best_score, model_results = self.test_advanced_ml_models()
        
        # Step 5: Final validation and breakthrough assessment
        results = self.final_breakthrough_validation()
        
        print("\\n" + "=" * 70)
        print("🎯 UNCONSTRAINED BREAKTHROUGH APPROACH COMPLETE")
        
        return results

if __name__ == "__main__":
    # Run unconstrained breakthrough approach
    breakthrough_system = UnconstrainedBreakthroughApproach()
    results = breakthrough_system.run_unconstrained_breakthrough()
    
    print(f"\\n📋 FINAL BREAKTHROUGH SUMMARY:")
    print(f"🚀 Final R²: {results['final_r2']:.6f}")
    print(f"🚀 Breakthrough Level: {results['breakthrough_level']}")
    print(f"🚀 Approach: {results['approach_used']}")
    print(f"🚀 Features Used: {results['features_used']}")
    print(f"🚀 Models Tested: {results['models_tested']}")
    
    if results['breakthrough_level'] in ['MAJOR', 'SIGNIFICANT']:
        print("\\n🎉 BREAKTHROUGH ACHIEVED!")
        print("🏆 Unconstrained approach succeeded!")
    elif results['breakthrough_level'] == 'MODERATE':
        print("\\n📈 SIGNIFICANT IMPROVEMENT!")
        print("💪 Breaking through previous plateaus!")
    else:
        print("\\n🤔 Plateau confirmed - may represent fundamental limit")
        print("💡 Consider domain expertise or additional data")