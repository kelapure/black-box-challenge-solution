You are the worlds best competitive topcoder solving a top coding  black box challenge presented below. You are to create a design and implementation plan for a blackblox coding data science problem. I

I want to encourage deep thinking especially in the planning phase and focus on the accuracy and correctness of the answer. I want the design and implementation to be decomposable into sub-tasks that I am going to ask claude code to execute concurrently and/or sequentially. I want to follow the Explore, plan, code, commit workflow for every task and sub-tasks. The plan should figure out the best execution strategy and programming  and execution style including functional or procedural programming. Data pipelining, data science, statistical programming and any other techniques including training and transformation should be explicitly thought of specifically data to be manipulated.  Take into consideration all the design and architecture things that I may have missed.  Consider edge cases or error conditions. Take into consideration known technical constraints, dependencies, or suggestions. Take into consideration the eval for checking intermediate and final outputs. 

Ask any clarifying questions BEFORE you create to solve the blackbox problem.

-- BLACK BOX CHALLENGE START --
Black Box Challenge: Legacy Reimbursement System
Reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews.
ACME Corp's legacy reimbursement system has been running for 60 years. No one knows how it works, but it's still used daily.
8090 has built them a new system, but ACME Corp is confused by the differences in results. Your mission is to figure out the original business logic so we can explain why ours is different and better.
Your job: create a perfect replica of the legacy system by reverse-engineering its behavior from 1,000 historical input/output examples and employee interviews.
What You Have
Input Parameters
The system takes three inputs:
* trip_duration_days - Number of days spent traveling (integer)
* miles_traveled - Total miles traveled (integer)
* total_receipts_amount - Total dollar amount of receipts (float)
Documentation
* A PRD (Product Requirements Document)
* Employee interviews with system hints
Output
* Single numeric reimbursement amount (float, rounded to 2 decimal places)
Historical Data
* test_cases.json - 1,000 historical input/output examples
Getting Started
1. Analyze the data: Look at test_cases.json to understand patterns
2. Create your implementation:
   * Copy run.sh.template to run.sh
   * Implement your calculation logic
   * Make sure it outputs just the reimbursement amount
3. Test your solution: Run ./eval.sh to see how you're doing
4. Iterate: Use the feedback to improve your algorithm
Implementation Requirements
Your run.sh script must:
* Take exactly 3 parameters: trip_duration_days, miles_traveled, total_receipts_amount
* Output a single number (the reimbursement amount)
* Run in under 5 seconds per test case
* Work without external dependencies (no network calls, databases, etc.)
Example:

./run.sh 5 250 150.75
# Should output something like: 487.25
Evaluation
Run ./eval.sh to test your solution against all 1,000 cases. The script will show:
* Exact matches: Cases within ±$0.01 of the expected output
* Close matches: Cases within ±$1.00 of the expected output
* Average error: Mean absolute difference from expected outputs
* Score: Lower is better (combines accuracy and precision)
Your submission will be tested against 5,000 additional cases that are not included in test_cases.json.

-- BLACK BOX CHALLENGE END --
