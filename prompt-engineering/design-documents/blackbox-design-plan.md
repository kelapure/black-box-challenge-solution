# Black Box Challenge: Legacy Reimbursement System Reverse Engineering
## Design & Implementation Plan

### 1. Overall Architecture & Strategy

#### 1.1 Hybrid Approach
- **Primary**: Rule-based reverse engineering through pattern analysis
- **Secondary**: Machine learning models for capturing non-linear patterns
- **Tertiary**: Ensemble approach combining both methods

#### 1.2 Execution Flow
```
1. Data Loading & Validation
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Pattern Discovery
5. Model Development (Parallel)
   - Rule-based models
   - ML models
6. Model Evaluation & Selection
7. Ensemble Creation
8. Final Implementation
```

### 2. Task Decomposition

#### Phase 1: Data Understanding & Exploration
**Task 1.1: Data Loading & Validation**
- Load test_cases.json
- Validate data integrity
- Check for missing values, outliers
- Create data quality report

**Task 1.2: Document Analysis**
- Parse PRD document
- Extract employee interview insights
- Create knowledge base of hints/constraints

**Task 1.3: Exploratory Data Analysis**
- Statistical analysis of inputs/outputs
- Correlation analysis
- Distribution analysis
- Identify potential data clusters

**Task 1.4: Feature Engineering**
- Create derived features (ratios, products, powers)
- Identify potential breakpoints
- Create interaction features

#### Phase 2: Pattern Discovery (Parallel Execution)

**Branch A: Rule-Based Discovery**

**Task 2A.1: Linear Pattern Analysis**
- Test for linear combinations
- Identify coefficients for each input
- Test for piecewise linear functions

**Task 2A.2: Non-Linear Pattern Analysis**
- Test polynomial relationships
- Test logarithmic/exponential patterns
- Test step functions and thresholds

**Task 2A.3: Conditional Logic Discovery**
- Decision tree analysis
- Rule extraction
- Identify business logic patterns

**Task 2A.4: Historical Business Logic Patterns**
- Test for common 1960s business rules:
  - Per diem rates
  - Mileage reimbursement tiers
  - Receipt threshold rules
  - Duration-based multipliers

**Branch B: Machine Learning Approach**

**Task 2B.1: Regression Models**
- Linear Regression with regularization
- Polynomial Regression
- Support Vector Regression
- Random Forest Regression

**Task 2B.2: Neural Network Models**
- Simple feedforward networks
- Networks with different activation functions
- Ensemble of small networks

**Task 2B.3: Gradient Boosting Models**
- XGBoost
- LightGBM
- CatBoost

**Task 2B.4: Symbolic Regression**
- Genetic programming to discover formulas
- Feature importance analysis

#### Phase 3: Model Evaluation & Selection

**Task 3.1: Cross-Validation Setup**
- Create train/validation splits
- Implement k-fold cross-validation
- Stratified sampling based on output ranges

**Task 3.2: Evaluation Metrics**
- Exact match rate (±$0.01)
- Close match rate (±$1.00)
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Maximum error analysis

**Task 3.3: Model Interpretability**
- Extract rules from ML models
- Validate discovered rules against business logic
- Create interpretable model representation

#### Phase 4: Ensemble & Optimization

**Task 4.1: Model Combination**
- Weighted ensemble based on accuracy
- Voting mechanisms for discrete rules
- Stacking approach if beneficial

**Task 4.2: Edge Case Handling**
- Identify and handle outliers
- Implement bounds checking
- Add special case logic

**Task 4.3: Final Optimization**
- Fine-tune parameters
- Optimize for exact matches
- Minimize maximum error

### 3. Implementation Details

#### 3.1 Data Pipeline Architecture
```python
# Main pipeline structure
class ReimbursementPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.rule_engine = RuleEngine()
        self.ml_ensemble = MLEnsemble()
        self.final_model = FinalModel()
    
    def process(self, trip_days, miles, receipts):
        # Pipeline processing logic
        pass
```

#### 3.2 Error Handling & Validation
- Input validation (non-negative values, reasonable ranges)
- Numerical stability checks
- Logging for debugging
- Graceful degradation for edge cases

#### 3.3 Performance Optimization
- Vectorized operations for batch processing
- Caching of intermediate results
- Efficient rule evaluation
- Model serialization for fast loading

### 4. Evaluation Strategy

#### 4.1 Validation Approach
- 80/20 train/test split of the 1,000 cases
- Time-based validation if timestamps available
- Leave-one-out cross-validation for final model

#### 4.2 Success Metrics
- Primary: Exact match rate > 95%
- Secondary: All predictions within ±$1.00
- Tertiary: Average error < $0.10

#### 4.3 Generalization Testing
- Test on synthetic edge cases
- Extrapolation testing
- Robustness to input variations

### 5. Risk Mitigation

#### 5.1 Potential Challenges
- **Overfitting**: Use regularization and cross-validation
- **Hidden business rules**: Systematic rule testing
- **Numerical precision**: Use appropriate data types
- **Legacy quirks**: Test for common programming patterns from 1960s

#### 5.2 Fallback Strategies
- If ML fails: Focus on rule extraction
- If rules are too complex: Use ensemble
- If accuracy is low: Analyze error patterns

### 6. Implementation Workflow

#### Step 1: Initial Setup
```bash
# Create project structure
mkdir -p {data,models,analysis,output}
# Copy test_cases.json to data/
# Create main.py and run.sh
```

#### Step 2: Parallel Task Execution
```python
# Task orchestration
tasks = {
    'eda': explore_data,
    'rules': discover_rules,
    'ml': train_models,
    'ensemble': create_ensemble
}
# Execute tasks concurrently where possible
```

#### Step 3: Iterative Refinement
1. Run initial model
2. Analyze errors
3. Refine based on patterns
4. Repeat until convergence

### 7. Deliverables

1. **run.sh**: Final implementation script
2. **model.pkl**: Serialized model
3. **analysis_report.md**: Detailed findings
4. **rules_discovered.txt**: Extracted business logic
5. **evaluation_results.json**: Performance metrics

### 8. Success Criteria

- ✓ Exact match rate > 95% on test set
- ✓ Generalization to unseen cases
- ✓ Interpretable business rules extracted
- ✓ Execution time < 5 seconds per case
- ✓ No external dependencies

This plan provides a comprehensive approach to reverse-engineering the legacy system with high accuracy while maintaining interpretability and robustness.