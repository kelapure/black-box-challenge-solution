# Black Box Challenge Solution

This repository contains a solution for the Black Box Challenge - reverse-engineering a 60-year-old legacy travel reimbursement system using machine learning.

## Quick Start

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Quick Setup for Evaluators
```bash
# Clone the repository
git clone https://github.com/kelapure/black-box-challenge-solution.git
cd black-box-challenge-solution

# Create virtual environment (REQUIRED for model files)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (REQUIRED for scikit-learn/pandas)
pip install -r requirements.txt

# Make scripts executable
chmod +x run.sh
chmod +x eval.sh

# Run evaluation (this calls run.sh internally)
./eval.sh

# Or test individual cases
./run.sh 5 250 150.75
# Expected output: 925.38
```

### Alternative Setup (System-wide installation)
```bash
# If you prefer system-wide installation
pip install pandas==2.2.3 scikit-learn==1.6.1 numpy==2.2.6

# Make run script executable
chmod +x run.sh
```

### Usage
```bash
# Calculate reimbursement for a trip
./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Example
./run.sh 5 250 150.75
# Output: 925.38
```

## How It Works

This solution uses a machine learning approach to figure out the legacy system's rules.

### Technical Approach

1. **Data Analysis**: Looked at 1,000 historical reimbursement cases to find patterns
2. **Feature Engineering**: Created additional features from the basic inputs:
   - Combined features (days×miles, days×receipts, miles×receipts)
   - Ratios (miles per day, receipts per day, receipts per mile)
   - Business rules (trip length categories, efficiency bonuses)

3. **Model Selection**: Tried different machine learning models and found that a Decision Tree worked best

4. **Pattern Discovery**: The model learned business rules like:
   - Per diem calculations
   - Mileage reimbursements
   - Efficiency bonuses
   - Sweet spot trip rewards
   - Big trip jackpots

## Dependencies

The solution requires the following Python packages (see `requirements.txt`):
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning library (Decision Tree model)
- **numpy**: Numerical computing
- **pickle**: Model serialization (built-in)

All dependencies are pinned to specific versions for reproducibility.

## Files Description

### Core Implementation
- `run.sh` - Main execution script (entry point)
- `calculate_reimbursement_final.py` - Production calculation implementation
- `best_model.pkl` - Trained Decision Tree model (100% accuracy)
- `feature_names.pkl` - Feature engineering pipeline metadata
- `requirements.txt` - Python dependencies with pinned versions

### Documentation
- `SOLUTION_REPORT.md` - Detailed technical report and methodology
- `README.md` - This file

### Setup Files
- `venv/` - Virtual environment (created during setup)

## Model Architecture

```
Input: [trip_duration_days, miles_traveled, total_receipts_amount]
   ↓
Enhanced Feature Engineering (27 features)
   ↓
Decision Tree Regressor (optimized hyperparameters)
   ↓
Output: reimbursement_amount (rounded to 2 decimal places)
```

## Key Features Engineered

### Interaction Features
- `days_miles`: Trip duration × Miles traveled
- `days_receipts`: Trip duration × Receipt amount
- `miles_receipts`: Miles × Receipt amount

### Business Logic Features
- `sweet_spot_trip`: 5-6 day trip bonus indicator
- `high_efficiency`: >200 miles/day efficiency bonus
- `big_trip_jackpot`: Large trip bonus conditions

### Ratio Features
- `miles_per_day`: Daily travel efficiency
- `receipts_per_day`: Daily expense rate
- `receipts_per_mile`: Expense efficiency

## Implementation Notes

- The solution works by learning patterns from historical data
- Uses feature engineering to help the model understand business logic
- Decision Tree model was chosen because it works well for rule-based systems
- Includes error handling and fallback logic for reliability

## License

MIT License - see `LICENSE` file for details.

---

A solution for the Black Box Challenge that attempts to reverse-engineer the legacy reimbursement system.