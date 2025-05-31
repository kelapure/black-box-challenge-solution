# document_analyzer.py - PRD and Interview Analysis Module
"""
Specialized module for analyzing PRD and employee interview documents
to extract business rules and constraints
"""

import re
import json
from typing import Dict, List, Tuple

class DocumentAnalyzer:
    """Analyze PRD and employee interviews for business logic hints"""
    
    def __init__(self):
        self.prd_content = None
        self.interviews = []
        self.extracted_rules = []
        self.constraints = {}
        
    def explore(self):
        """EXPLORE phase: Understand document structure"""
        print("\n=== DOCUMENT ANALYSIS ===")
        print("Analyzing PRD and employee interviews...")
        
    def plan(self):
        """PLAN phase: Design document analysis approach"""
        return {
            'analysis_targets': [
                'Business rules mentioned in PRD',
                'Numerical values and thresholds',
                'Employee hints about calculations',
                'Historical context clues',
                'Edge cases and exceptions'
            ],
            'extraction_methods': [
                'Pattern matching',
                'Keyword extraction',
                'Numerical value extraction',
                'Rule template matching'
            ]
        }
    
    def code(self, prd_path='prd.txt', interviews_path='interviews.txt'):
        """CODE phase: Implement document analysis"""
        
        # Load documents
        try:
            with open(prd_path, 'r') as f:
                self.prd_content = f.read()
        except:
            print(f"Warning: Could not load {prd_path}")
            self.prd_content = ""
            
        try:
            with open(interviews_path, 'r') as f:
                self.interviews = f.read().split('\n---\n')  # Assuming interviews are separated
        except:
            print(f"Warning: Could not load {interviews_path}")
            self.interviews = []
        
        # Extract business rules
        self._extract_prd_rules()
        self._extract_interview_hints()
        self._extract_numerical_patterns()
        self._identify_constraints()
        
        return {
            'rules': self.extracted_rules,
            'constraints': self.constraints
        }
    
    def _extract_prd_rules(self):
        """Extract business rules from PRD"""
        if not self.prd_content:
            return
            
        # Common patterns in reimbursement PRDs
        patterns = [
            # Per diem patterns
            (r'per\s*(?:diem|day).*?(\$?\d+\.?\d*)', 'per_diem'),
            (r'daily\s*(?:rate|allowance).*?(\$?\d+\.?\d*)', 'per_diem'),
            
            # Mileage patterns
            (r'(?:mileage|per\s*mile).*?(\$?\d+\.?\d*)', 'mileage_rate'),
            (r'(\d+\.?\d*)\s*(?:cents?|¢)\s*per\s*mile', 'mileage_rate_cents'),
            
            # Receipt patterns
            (r'receipt.*?(?:required|needed).*?(?:over|above|exceeding).*?(\$?\d+\.?\d*)', 'receipt_threshold'),
            (r'(?:reimburse|reimbursement).*?(\d+)%?\s*of.*?receipt', 'receipt_percentage'),
            
            # Maximum limits
            (r'(?:maximum|max|cap).*?(\$?\d+\.?\d*)', 'max_limit'),
            (r'(?:up\s*to|not\s*(?:to\s*)?exceed).*?(\$?\d+\.?\d*)', 'max_limit'),
            
            # Minimum thresholds
            (r'(?:minimum|min).*?(\$?\d+\.?\d*)', 'min_threshold'),
            
            # Duration-based rules
            (r'(?:trips?|travel).*?(?:longer|more|over|exceeding).*?(\d+)\s*days?', 'duration_threshold'),
            (r'(?:first|initial).*?(\d+)\s*days?.*?different', 'duration_tier'),
        ]
        
        for pattern, rule_type in patterns:
            matches = re.findall(pattern, self.prd_content, re.IGNORECASE)
            for match in matches:
                value = self._parse_value(match)
                if value is not None:
                    self.extracted_rules.append({
                        'type': rule_type,
                        'value': value,
                        'source': 'PRD',
                        'context': self._get_context(pattern, self.prd_content)
                    })
    
    def _extract_interview_hints(self):
        """Extract hints from employee interviews"""
        for interview in self.interviews:
            if not interview.strip():
                continue
                
            # Common employee descriptions
            hint_patterns = [
                # Calculation descriptions
                (r'(?:calculate|multiply|add).*?(\w+).*?(?:by|times|with).*?(\$?\d+\.?\d*)', 'calculation'),
                (r'(?:it\'s|its|just|simply).*?(\$?\d+\.?\d*).*?(?:per|for\s*each)', 'simple_rate'),
                
                # Rule descriptions
                (r'(?:if|when).*?(?:more|less|over|under).*?(\d+)', 'conditional'),
                (r'(?:different|special|extra).*?(?:rate|rule).*?(\d+)', 'special_case'),
                
                # Historical context
                (r'(?:used\s*to|originally|back\s*then).*?(\$?\d+\.?\d*)', 'historical'),
                (r'(?:changed|updated).*?(?:from|to).*?(\$?\d+\.?\d*)', 'changed_value'),
                
                # Rounding rules
                (r'(?:round|rounding).*?(?:up|down|nearest).*?(\$?\d+\.?\d*)', 'rounding'),
                
                # Exceptions
                (r'(?:except|unless|but\s*not).*?(\w+)', 'exception'),
            ]
            
            for pattern, hint_type in hint_patterns:
                matches = re.findall(pattern, interview, re.IGNORECASE)
                for match in matches:
                    self.extracted_rules.append({
                        'type': f'interview_{hint_type}',
                        'content': match,
                        'source': 'Interview',
                        'full_context': interview[:200]  # First 200 chars for context
                    })
    
    def _extract_numerical_patterns(self):
        """Extract all numerical values mentioned"""
        all_text = self.prd_content + '\n'.join(self.interviews)
        
        # Find all numbers with context
        number_pattern = r'(\$?\d+\.?\d*)'
        numbers = re.findall(number_pattern, all_text)
        
        # Common significant values in reimbursement systems
        significant_values = []
        for num_str in numbers:
            value = self._parse_value(num_str)
            if value is not None:
                # Check if it's a common rate
                if value in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.58]:
                    significant_values.append(('mileage_rate', value))
                elif value in [25, 50, 75, 100, 125, 150, 175, 200]:
                    significant_values.append(('per_diem', value))
                elif value in [5, 7, 10, 14, 21, 30]:
                    significant_values.append(('duration_threshold', value))
        
        self.constraints['significant_values'] = significant_values
    
    def _identify_constraints(self):
        """Identify business constraints from documents"""
        constraints = {
            'has_per_diem': any(r['type'] == 'per_diem' for r in self.extracted_rules),
            'has_mileage': any('mileage' in r['type'] for r in self.extracted_rules),
            'has_receipts': any('receipt' in r['type'] for r in self.extracted_rules),
            'has_tiers': any('tier' in r['type'] or 'threshold' in r['type'] for r in self.extracted_rules),
            'has_maximums': any('max' in r['type'] for r in self.extracted_rules),
        }
        
        self.constraints.update(constraints)
    
    def _parse_value(self, value_str):
        """Parse numerical value from string"""
        try:
            # Remove $ and convert
            clean_str = value_str.replace('$', '').replace(',', '')
            return float(clean_str)
        except:
            return None
    
    def _get_context(self, pattern, text):
        """Get surrounding context for a pattern match"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            return text[start:end]
        return ""
    
    def commit(self):
        """COMMIT phase: Summarize findings"""
        summary = {
            'total_rules_found': len(self.extracted_rules),
            'rule_types': list(set(r['type'] for r in self.extracted_rules)),
            'constraints': self.constraints,
            'key_insights': self._generate_insights()
        }
        
        return summary
    
    def _generate_insights(self):
        """Generate key insights from document analysis"""
        insights = []
        
        # Check for common patterns
        if self.constraints.get('has_per_diem'):
            per_diem_values = [r['value'] for r in self.extracted_rules if r['type'] == 'per_diem']
            if per_diem_values:
                insights.append(f"Per diem rates found: {per_diem_values}")
        
        if self.constraints.get('has_mileage'):
            mileage_values = [r['value'] for r in self.extracted_rules if 'mileage' in r['type']]
            if mileage_values:
                insights.append(f"Mileage rates found: {mileage_values}")
        
        if self.constraints.get('has_tiers'):
            insights.append("System appears to have tiered reimbursement rules")
        
        return insights


class HintIntegrator:
    """Integrate document hints with data analysis"""
    
    def __init__(self, document_analysis, data):
        self.doc_analysis = document_analysis
        self.data = data
        self.integrated_rules = []
        
    def integrate_hints(self):
        """Integrate document hints with data patterns"""
        
        # Test each hint against actual data
        for rule in self.doc_analysis['rules']:
            if rule['type'] == 'per_diem' and 'value' in rule:
                # Test if this per diem rate matches the data
                pred = self.data['trip_duration_days'] * rule['value']
                mae = np.mean(np.abs(self.data['reimbursement'] - pred))
                
                if mae < 100:  # Reasonable fit
                    self.integrated_rules.append({
                        'type': 'per_diem',
                        'rate': rule['value'],
                        'mae': mae,
                        'source': 'document_validated'
                    })
            
            elif rule['type'] == 'mileage_rate' and 'value' in rule:
                # Test mileage rate
                rate = rule['value'] if rule['value'] > 1 else rule['value']  # Handle cents vs dollars
                pred = self.data['miles_traveled'] * rate
                mae = np.mean(np.abs(self.data['reimbursement'] - pred))
                
                if mae < 100:
                    self.integrated_rules.append({
                        'type': 'mileage',
                        'rate': rate,
                        'mae': mae,
                        'source': 'document_validated'
                    })
        
        return self.integrated_rules
    
    def suggest_formula_structure(self):
        """Suggest formula structure based on hints"""
        components = []
        
        if self.doc_analysis['constraints'].get('has_per_diem'):
            components.append("per_diem_rate * trip_duration_days")
        
        if self.doc_analysis['constraints'].get('has_mileage'):
            components.append("mileage_rate * miles_traveled")
        
        if self.doc_analysis['constraints'].get('has_receipts'):
            components.append("receipt_factor * total_receipts_amount")
        
        formula_structure = " + ".join(components) if components else "complex_formula"
        
        return {
            'suggested_structure': formula_structure,
            'components': components,
            'has_conditions': self.doc_analysis['constraints'].get('has_tiers', False)
        }


# Example usage in main pipeline
def analyze_documents(data):
    """Run document analysis as part of the main pipeline"""
    
    # Create analyzer
    analyzer = DocumentAnalyzer()
    
    # Run analysis
    analyzer.explore()
    plan = analyzer.plan()
    results = analyzer.code()
    summary = analyzer.commit()
    
    # Integrate with data
    integrator = HintIntegrator(results, data)
    validated_rules = integrator.integrate_hints()
    formula_suggestion = integrator.suggest_formula_structure()
    
    return {
        'document_summary': summary,
        'validated_rules': validated_rules,
        'formula_suggestion': formula_suggestion
    }
