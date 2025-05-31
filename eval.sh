#!/bin/bash

# Black Box Challenge Evaluation Script
# This script tests your reimbursement calculation implementation against 1,000 historical cases

set -e

echo "🧾 Black Box Challenge - Reimbursement System Evaluation"
echo "======================================================="
echo

# Check if run.sh exists
if [ ! -f "run.sh" ]; then
    echo "❌ Error: run.sh not found!"
    echo "Please create a run.sh script that takes three parameters:"
    echo "  ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    echo "  and outputs the reimbursement amount"
    exit 1
fi

# Make run.sh executable
chmod +x run.sh

# Check if test cases exist
if [ ! -f "test_cases.json" ]; then
    echo "❌ Error: test_cases.json not found!"
    echo "Please ensure the test cases file is in the current directory."
    exit 1
fi

echo "📊 Running evaluation against 1,000 test cases..."
echo

# Initialize counters
total_cases=0
exact_matches=0
close_matches=0
total_error=0
max_error=0
max_error_case=""

# Create temporary files for results
temp_results=$(mktemp)
temp_errors=$(mktemp)

# Parse JSON and run tests
python3 -c "
import json
import subprocess
import sys
import math

with open('test_cases.json', 'r') as f:
    test_cases = json.load(f)

results = []
errors = []

for i, case in enumerate(test_cases):
    if i % 100 == 0:
        print(f'Progress: {i}/1000 cases processed...', file=sys.stderr)
    
    input_data = case['input']
    expected = case['expected_output']
    
    try:
        # Run the user's implementation
        result = subprocess.run([
            './run.sh',
            str(input_data['trip_duration_days']),
            str(input_data['miles_traveled']),
            str(input_data['total_receipts_amount'])
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            errors.append(f'Case {i+1}: Script failed with error: {result.stderr.strip()}')
            continue
            
        try:
            actual = float(result.stdout.strip())
        except ValueError:
            errors.append(f'Case {i+1}: Invalid output format: {result.stdout.strip()}')
            continue
            
        error = abs(actual - expected)
        results.append({
            'case': i+1,
            'expected': expected,
            'actual': actual,
            'error': error,
            'input': input_data
        })
        
    except subprocess.TimeoutExpired:
        errors.append(f'Case {i+1}: Timeout (>5 seconds)')
    except Exception as e:
        errors.append(f'Case {i+1}: Unexpected error: {str(e)}')

# Write results to temp files
with open('$temp_results', 'w') as f:
    json.dump(results, f)

with open('$temp_errors', 'w') as f:
    for error in errors:
        f.write(error + '\n')
"

# Read results and calculate statistics
if [ -f "$temp_results" ]; then
    python3 -c "
import json
import math

with open('$temp_results', 'r') as f:
    results = json.load(f)

if not results:
    print('❌ No successful test cases!')
    print('')
    print('Your script either:')
    print('  - Failed to run properly')
    print('  - Produced invalid output format')
    print('  - Timed out on all cases')
    print('')
    print('Check the errors below for details.')
else:
    total_cases = len(results)
    exact_matches = sum(1 for r in results if r['error'] < 0.01)
    close_matches = sum(1 for r in results if r['error'] < 1.0)
    total_error = sum(r['error'] for r in results)
    avg_error = total_error / total_cases
    max_error_result = max(results, key=lambda r: r['error'])

    print(f'✅ Evaluation Complete!')
    print(f'')
    print(f'📈 Results Summary:')
    print(f'  Total test cases: 1000')
    print(f'  Successful runs: {total_cases}')
    print(f'  Exact matches (±\$0.01): {exact_matches} ({exact_matches/total_cases*100:.1f}%)')
    print(f'  Close matches (±\$1.00): {close_matches} ({close_matches/total_cases*100:.1f}%)')
    print(f'  Average error: \${avg_error:.2f}')
    print(f'  Maximum error: \${max_error_result[\"error\"]:.2f}')
    print(f'')

    # Calculate score (lower is better)
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    print(f'🎯 Your Score: {score:.2f} (lower is better)')
    print(f'')

    if exact_matches == 1000:
        print('🏆 PERFECT SCORE! You have reverse-engineered the system completely!')
    elif exact_matches > 950:
        print('🥇 Excellent! You are very close to the perfect solution.')
    elif exact_matches > 800:
        print('🥈 Great work! You have captured most of the system behavior.')
    elif exact_matches > 500:
        print('🥉 Good progress! You understand some key patterns.')
    else:
        print('📚 Keep analyzing the patterns in the interviews and test cases.')

    print(f'')
    print(f'💡 Tips for improvement:')
    if exact_matches < 1000:
        worst_cases = sorted(results, key=lambda r: r[\"error\"], reverse=True)[:5]
        print(f'  Check these high-error cases:')
        for case in worst_cases:
            inp = case['input']
            print(f'    Case {case[\"case\"]}: {inp[\"trip_duration_days\"]} days, {inp[\"miles_traveled\"]} miles, \${inp[\"total_receipts_amount\"]:.2f} receipts')
            print(f'      Expected: \${case[\"expected\"]:.2f}, Got: \${case[\"actual\"]:.2f}, Error: \${case[\"error\"]:.2f}')
"
fi

# Show errors if any
if [ -s "$temp_errors" ]; then
    echo
    echo "⚠️  Errors encountered:"
    head -10 "$temp_errors"
    if [ $(wc -l < "$temp_errors") -gt 10 ]; then
        echo "  ... and $(($(wc -l < "$temp_errors") - 10)) more errors"
    fi
fi

# Cleanup
rm -f "$temp_results" "$temp_errors"

echo
echo "📝 Next steps:"
echo "  1. Fix any script errors shown above"
echo "  2. Ensure your run.sh outputs only a number"
echo "  3. Analyze the patterns in the interviews and test cases"
echo "  4. Test edge cases around trip length and receipt amounts"
echo "  5. Submit your solution via the Google Form when ready!" 