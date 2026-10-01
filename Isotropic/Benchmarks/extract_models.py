"""
Extract discovered model coefficients for cases where ground truth wasn't recovered
"""

import re

# Cases that didn't recover ground truth (from verification summary)
failed_cases = {
    'O2_RelNoise0pct': ['LarsCV', 'LarsAIC', 'LarsBIC'],
    'O2_RelNoise5pct': ['LassoCV', 'LarsCV', 'LarsAIC', 'OMPCV', 'OMPAIC', 'OMPBIC'],
    'O2_RelNoise10pct': ['LassoCV', 'LarsCV', 'LarsAIC', 'OMPCV', 'OMPAIC'],
    'MR2_RelNoise0pct': ['LarsCV', 'LarsAIC', 'LarsBIC'],
    'MR1O1_RelNoise10pct': ['LassoCV', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC'],
    'MR2O2_RelNoise0pct': ['LarsCV', 'LarsAIC', 'LarsBIC'],
    'MR2O2_RelNoise5pct': ['LarsBIC', 'OMPCV'],
    'MR2O2_RelNoise10pct': ['LassoCV'],
}

# Read the log file
with open('outputs_PSII/log_PSII_test.txt', 'r') as f:
    lines = f.readlines()

# Find models for each case
models_found = {}
current_scenario = None
current_method = None
in_model_section = False
model_terms = {}

for i, line in enumerate(lines):
    # Detect scenario
    if 'Processing Scenario:' in line:
        current_scenario = line.split('Processing Scenario:')[1].strip()
        if current_scenario not in failed_cases:
            current_scenario = None
    
    # Detect method
    if current_scenario and '--- Post-processing Result for:' in line:
        current_method = line.split('for:')[1].strip()
        in_model_section = False
        model_terms = {}
    
    # Detect final model section
    if current_scenario and current_method and f'Final Model ({current_method}):' in line:
        in_model_section = True
        model_terms = {}
        continue
    
    # Extract model terms
    if in_model_section and line.strip() and not line.strip().startswith('Checking') and not line.strip().startswith('Consistency'):
        # Look for model term lines like:     'Ogden: λ^-3': 0.0161 kPa
        match = re.search(r"'(.+?)':\s*([-\d.]+)\s*kPa", line)
        if match:
            term_name = match.group(1)
            coeff_kpa = float(match.group(2))
            model_terms[term_name] = coeff_kpa
        elif 'Calculating P11' in line or 'Metrics:' in line:
            # End of model section
            if current_method in failed_cases.get(current_scenario, []):
                key = (current_scenario, current_method)
                if model_terms:
                    models_found[key] = dict(model_terms)
            in_model_section = False
            model_terms = {}

# Also get R²_min and AvgNRMSE from the verification table
print("Found models for cases that didn't recover ground truth:\n")
for (scenario, method), terms in sorted(models_found.items()):
    print(f"{scenario} - {method}:")
    for term, coeff in terms.items():
        print(f"  {term}: {coeff} kPa = {coeff*1000:.1f} Pa")
    print()

print(f"\nTotal cases found: {len(models_found)}")
print(f"Total cases expected: {sum(len(v) for v in failed_cases.values())}")
