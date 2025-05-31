# Claude Code Prompts for Black Box Challenge
## A Step-by-Step Tutorial

Based on a successful implementation that achieved 100% accuracy on a legacy travel reimbursement system reverse-engineering challenge.

---

## Strategic Overview: The EPIC Framework

The proven **EPIC methodology** from the implementation:

- **E**xplore: Understand the problem space and available data
- **P**lan: Design a systematic approach with multiple strategies  
- **I**mplement: Execute both rule-based and ML approaches in parallel
- **C**ommit: Validate, optimize, and deliver the final solution

### Architecture Principles

Based on the successful design plan:

1. **Hybrid Approach**: Combine rule-based discovery with machine learning
2. **Parallel Execution**: Run multiple strategies simultaneously
3. **Iterative Refinement**: Continuously improve based on error analysis
4. **Document Integration**: Leverage any available documentation or hints
5. **Validation Focus**: Aim for 100% exact matches, not just "good enough"

---

## PROMPT 1: Initial Setup and Strategic Planning
**When to use**: At the very beginning, after navigating to the challenge directory

```
I have a black box coding challenge where I need to reverse-engineer a 60-year-old travel reimbursement system. The goal is to discover the exact formula used by analyzing 1,000 historical examples.

I want to use a comprehensive hybrid approach following the proven EPIC methodology. Please help me implement a systematic solution.

First, analyze the available files and create a strategic plan:

1. **EXPLORE PHASE**:
   - Check all files in the directory (test_cases.json, eval.sh, documentation, etc.)
   - Load and analyze test_cases.json structure and patterns
   - Document any hints from PRD or interview files if they exist
   - Perform initial statistical analysis and correlation studies

2. **PLAN PHASE**:
   - Design a multi-pronged approach with parallel execution
   - Plan rule-based discovery (linear, piecewise, business rules)
   - Plan ML model pipeline (feature engineering, multiple algorithms)
   - Design evaluation and model selection criteria
   - Create implementation strategy for final solution

Please start with the exploration phase and show me:
- Directory contents and file analysis
- Data structure and basic statistics  
- Initial correlation patterns
- Any document analysis results
- A detailed execution plan for the remaining phases

Use the TodoWrite tool to track our progress through this comprehensive approach.
```

---

## PROMPT 2: Document Analysis and Business Rule Extraction
**When to use**: When PRD or interview files are found, or when systematic rule discovery is needed

```
Implement comprehensive document analysis to extract business rules and constraints.

Based on the proven DocumentAnalyzer implementation, create a system that:

1. **DOCUMENT PARSING**:
   - Extract business rules from PRD using regex patterns
   - Parse employee interviews for calculation hints
   - Identify numerical values and their contexts
   - Map historical business logic patterns

2. **RULE EXTRACTION PATTERNS**:
   - Per diem rates: "per day", "daily rate", "$X per day"
   - Mileage rates: "per mile", "X cents per mile", "mileage rate"
   - Receipt rules: "receipts required", "X% reimbursement", "over $X"
   - Thresholds: "maximum", "minimum", "cap", "limit"
   - Conditions: "if", "when", "over X days", "first X days"

3. **INTEGRATION WITH DATA**:
   - Test extracted rates against actual data
   - Validate rules using statistical analysis
   - Combine document hints with pattern discovery
   - Generate formula structure suggestions

Implement the DocumentAnalyzer class with these capabilities:
- Pattern matching for business rules (per diem, mileage, receipt thresholds)
- Numerical value extraction and validation
- Context analysis for each found rule
- Integration with data patterns to validate hints
- Formula structure recommendation based on discovered constraints

This will provide validated business logic to guide our model development.
```

---

## PROMPT 3: Systematic Rule-Based Discovery Engine
**When to use**: After initial exploration, to implement comprehensive rule testing

```
Implement the systematic rule discovery engine from the proven implementation.

Create a RuleDiscovery class that tests multiple pattern types systematically:

1. **LINEAR PATTERN ANALYSIS**:
   - Test single-factor formulas with comprehensive rate ranges
   - Per diem: Test rates from $25-$250 in reasonable increments
   - Mileage: Test rates from $0.05-$1.00 with focus on historical rates
   - Receipt percentages: Test 50%-100% reimbursement rates
   - Linear combinations using regression analysis

2. **PIECEWISE FUNCTION TESTING**:
   - Different rates for trip duration tiers (1-3 days, 4-7 days, 8+ days)
   - Mileage rate tiers (0-100, 101-300, 301+ miles)
   - Receipt thresholds with different reimbursement rules
   - Breakpoint detection using decision tree analysis

3. **HISTORICAL BUSINESS PATTERNS**:
   - Test common 1960s-era rates and business rules
   - Government per diem schedules from that time period
   - Standard mileage rates by decade
   - Common rounding rules (nearest $0.25, $0.50, $1.00)

4. **CONDITIONAL LOGIC DISCOVERY**:
   - Minimum/maximum limits testing
   - Base fees + variable components
   - Duration-based multipliers
   - Receipt requirement thresholds

Implement systematic testing that:
- Tests thousands of rule combinations methodically
- Calculates exact match percentages for each rule
- Identifies the top performing patterns
- Handles edge cases and validation
- Provides interpretable rule descriptions

Target: Find rules with >95% exact match rate before moving to ML approaches.
```

---

## PROMPT 4: Advanced Machine Learning Pipeline
**When to use**: In parallel with rule discovery, to capture complex non-linear patterns

```
Implement the comprehensive machine learning pipeline from the proven MLModels class.

Create an advanced ML system to discover complex patterns:

1. **FEATURE ENGINEERING**:
   - Create extensive feature set from basic inputs
   - Interaction terms: days×miles, days×receipts, miles×receipts
   - Polynomial features: squared and cubic terms
   - Ratio features: miles/day, receipts/day, receipts/mile
   - Binned features for threshold detection

2. **MODEL ENSEMBLE**:
   - RandomForest with 100+ trees for pattern capture
   - GradientBoosting for sequential learning
   - Ridge/Lasso regression for linear relationships
   - Decision trees for rule extraction
   - Cross-validation with 5-fold methodology

3. **INTERPRETABILITY EXTRACTION**:
   - Extract decision rules from tree-based models
   - Analyze feature importance rankings
   - Generate explanations for predictions
   - Convert ML patterns back to business rules
   - Validate discovered patterns against domain knowledge

4. **PERFORMANCE EVALUATION**:
   - Calculate exact match rates (±$0.01)
   - Measure close match rates (±$1.00)
   - Track mean absolute error and RMSE
   - Ensure generalization to unseen cases

Implement the MLModels class with:
- Automated feature engineering pipeline
- Multiple model training with proper validation
- Model interpretation and rule extraction
- Performance comparison and selection
- Integration with rule-based findings

Goal: Discover patterns that rule-based methods might miss while maintaining interpretability.
```

---

## PROMPT 5: Intelligent Error Analysis and Model Fusion
**When to use**: To achieve the final push from 95%+ to 100% accuracy

```
Implement sophisticated error analysis and model fusion to achieve 100% accuracy.

Based on the ModelEvaluator approach, create a system that:

1. **COMPREHENSIVE ERROR ANALYSIS**:
   - Identify all cases with prediction errors > $0.01
   - Cluster errors by magnitude, sign, and input characteristics
   - Analyze error patterns across different input ranges
   - Look for systematic biases in predictions
   - Identify outliers vs systematic errors

2. **PATTERN-BASED ERROR CORRECTION**:
   - Test if errors are constant offsets (missing base fee)
   - Check for percentage-based systematic errors
   - Analyze errors by input value ranges (low/medium/high)
   - Test for missing conditional logic
   - Look for rounding rule inconsistencies

3. **MODEL FUSION STRATEGIES**:
   - Combine best rule-based model with best ML model
   - Use ensemble voting for discrete predictions
   - Apply ML model for complex cases, rules for simple cases
   - Create hybrid model with rule-based backbone + ML corrections
   - Use confidence-based model selection

4. **ITERATIVE REFINEMENT**:
   - For each error pattern found, create targeted fix
   - Re-evaluate entire model after each fix
   - Ensure fixes don't break previously correct predictions
   - Continue until zero errors remain
   - Validate against edge cases

Implement error analysis that:
- Systematically analyzes all prediction errors
- Generates targeted fixes for each error pattern
- Combines multiple models intelligently
- Achieves and validates 100% accuracy
- Prepares optimized implementation

Success criteria: Zero prediction errors (±$0.01) on all test cases.
```

---

## PROMPT 6: Production Implementation and Validation
**When to use**: Once 100% accuracy is achieved, to create the final deliverable

```
Create the production-ready implementation with comprehensive validation.

Based on the FinalImplementation class, generate a clean solution:

1. **FINAL IMPLEMENTATION CREATION**:
   - Distill the discovered formula into clean, efficient code
   - Implement input validation and edge case handling
   - Ensure numerical stability and precision
   - Add error handling for invalid inputs
   - Optimize for reliability

2. **COMPREHENSIVE TESTING**:
   - Validate against all 1,000 test cases
   - Test edge cases (zero values, very large values)
   - Verify rounding and precision handling
   - Test with invalid inputs
   - Ensure consistent outputs across multiple runs

3. **DELIVERABLE CREATION**:
   - Generate clean calculate_reimbursement function
   - Create executable run.sh with proper argument handling
   - Add comprehensive input validation
   - Include error messages for debugging
   - Ensure cross-platform compatibility

4. **DOCUMENTATION AND ANALYSIS**:
   - Document the discovered formula and its derivation
   - Explain the business logic behind the solution
   - Provide confidence assessment for generalization
   - Create analysis of the discovery process

Create the final implementation that:
- Implements the discovered formula cleanly
- Ensures 100% accuracy on all test cases
- Handles all edge cases gracefully
- Generates proper run.sh executable
- Provides comprehensive validation

Deliverables: working run.sh, clean implementation, validation results.
```

---

## PROMPT 7: Complete Autonomous Implementation
**When to use**: For fully autonomous execution of the entire pipeline

```
Execute the complete black box reverse-engineering pipeline autonomously using the proven implementation architecture.

Implement the full BlackBoxSolver orchestrator that:

1. **PHASE 1 - EXPLORE**:
   - Use DataExplorer class to load and analyze all available data
   - Perform statistical analysis and pattern identification
   - Extract business rules from any documentation using DocumentAnalyzer
   - Create comprehensive data understanding report

2. **PHASE 2 - PLAN & IMPLEMENT** (Parallel Execution):
   - Branch A: Systematic rule-based discovery using RuleDiscovery class
     * Test linear combinations and business patterns
     * Implement piecewise function detection
     * Discover conditional logic and thresholds
   - Branch B: Advanced ML pipeline using MLModels class
     * Feature engineering and model ensemble
     * Pattern extraction and interpretability
     * Cross-validation and generalization testing

3. **PHASE 3 - OPTIMIZE & INTEGRATE**:
   - Use ModelEvaluator to compare all discovered models
   - Perform comprehensive error analysis
   - Implement model fusion strategies
   - Achieve 100% exact match accuracy

4. **PHASE 4 - COMMIT & DELIVER**:
   - Use FinalImplementation to create production-ready code
   - Generate executable run.sh with validation
   - Test against all cases and edge conditions
   - Document the discovered formula and process

Execute the complete pipeline following the proven architecture:
- Target: 100% exact matches (±$0.01)
- Use TodoWrite tool to track progress through all phases
- Implement parallel processing where beneficial
- Focus on interpretable business rules
- Ensure robust error handling and validation

Begin autonomous execution now. Work systematically through all phases until achieving 100% accuracy and delivering the final run.sh implementation.
```

---

## PROMPT 8: Alternative Streamlined Approach
**When to use**: When you want a more direct approach without full pipeline complexity

```
I have a black box challenge to reverse-engineer a travel reimbursement formula from 1,000 examples in test_cases.json. The goal is 100% exact matches (within ±$0.01).

Please work systematically to:
1. Explore the data and any documentation
2. Test formulas from simple to complex business patterns
3. Analyze errors and refine approaches
4. Continue iterating until 100% exact matches achieved
5. Create the final run.sh implementation

Key strategies based on the proven approach:
- Start with simple business rules (per diem, mileage rates)
- Test common rates from the 1960s era
- Use error analysis to discover missing components
- Apply both rule-based and ML approaches
- Don't stop until achieving 100% exact matches

Work methodically through the problem. Make decisions based on data findings. Use the TodoWrite tool to track progress. Begin now and continue until you achieve 100% exact matches with a working run.sh implementation.
```

---

## Implementation Architecture Reference

### Core Classes from Proven Implementation

#### 1. DataExplorer Class
```python
class DataExplorer:
    """Handles data loading, validation, and initial pattern analysis"""
    def explore(self): # Initial data understanding
    def plan(self): # Design analysis approach  
    def code(self): # Execute statistical analysis
    def commit(self): # Generate insights
```

#### 2. DocumentAnalyzer Class  
```python
class DocumentAnalyzer:
    """Extracts business rules from PRD and interview documents"""
    - Pattern matching for rates, thresholds, conditions
    - Numerical value extraction and context analysis
    - Business rule validation against actual data
    - Formula structure recommendation
```

#### 3. RuleDiscovery Class
```python
class RuleDiscovery:
    """Systematic testing of business rule patterns"""
    - Linear combination testing
    - Piecewise function discovery
    - Historical business pattern validation
    - Threshold and conditional logic detection
```

#### 4. MLModels Class
```python
class MLModels:
    """Advanced machine learning pipeline"""
    - Feature engineering (interactions, ratios, polynomials)
    - Model ensemble (RandomForest, GradientBoosting, etc.)
    - Cross-validation and generalization testing
    - Pattern extraction and interpretability analysis
```

#### 5. ModelEvaluator Class
```python
class ModelEvaluator:
    """Model comparison and selection"""
    - Performance metrics calculation
    - Error pattern analysis
    - Model fusion strategies
    - Final model selection
```

#### 6. FinalImplementation Class
```python
class FinalImplementation:
    """Production-ready solution generation"""
    - Clean code generation
    - Input validation and error handling
    - run.sh script creation
    - Comprehensive testing and validation
```

### Success Patterns from Proven Implementation

#### ✅ What Works:
1. **Start Simple**: Test obvious business rules first
2. **Use Parallel Execution**: Run multiple approaches simultaneously  
3. **Focus on Exact Matches**: Don't settle for "close enough"
4. **Leverage Documentation**: Extract every possible hint
5. **Systematic Error Analysis**: Understand why predictions fail
6. **Validate Thoroughly**: Test edge cases and generalization

#### ❌ What to Avoid:
1. **Jumping to Complex Solutions**: Skip simple rule testing
2. **Ignoring Business Context**: Miss obvious domain patterns
3. **Accepting Partial Accuracy**: Stop before 100% matches
4. **Poor Error Analysis**: Not understanding failure modes
5. **Inadequate Validation**: Not testing edge cases thoroughly

### Historical Context for Reimbursement Systems

#### Common Business Rules:
- Tiered per diem (first X days vs. additional days)
- Mileage tiers (different rates by distance ranges)
- Receipt thresholds (reimbursement required vs. per diem)
- Maximum daily/trip limits
- Rounding rules (nearest $0.25, $0.50, $1.00)
- Base fees + variable components

#### Historical Rate Ranges:
- 1960s Per Diem: $15-$25 per day
- 1960s Mileage: $0.08-$0.12 per mile
- Common receipt reimbursement: 70%-100%

This framework provides everything needed to successfully reverse-engineer black box systems using Claude Code with the proven methodology that achieved 100% accuracy.