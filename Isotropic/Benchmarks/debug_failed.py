import re

# Read the log
with open('outputs_PSII/log_PSII_test.txt', 'r') as f:
    content = f.read()

# Test case: O2_RelNoise0pct, LarsCV
scenario = "O2_RelNoise0pct"
method = "LarsCV"

search_pattern = f"Final Model \\({method}\\):"

# Find all occurrences
start_pos = 0
found_count = 0
while True:
    match_pos = content.find(search_pattern, start_pos)
    if match_pos == -1:
        break
    
    found_count += 1
    print(f"\n=== Match {found_count} at position {match_pos} ===")
    
    # Check if this is in the right scenario
    prev_section = content[max(0, match_pos-5000):match_pos]
    
    # Test different patterns
    test1 = f"Scenario: {scenario}" in prev_section
    test2 = f"Processing Scenario: {scenario}" in prev_section
    test3 = f"SCENARIO: {scenario}" in prev_section.upper()
    test4 = scenario in prev_section
    
    print(f"  'Scenario: {scenario}' found: {test1}")
    print(f"  'Processing Scenario: {scenario}' found: {test2}")
    print(f"  'SCENARIO: {scenario}' (upper) found: {test3}")
    print(f"  '{scenario}' anywhere found: {test4}")
    
    if test4:
        # Find the actual scenario line
        lines = prev_section.split('\n')
        for line in lines[-20:]:
            if scenario in line:
                print(f"  Actual line: {line[:100]}")
    
    # Extract coefficients
    model_section = content[match_pos:match_pos+800]
    coeff_pattern = r"'([^']+)':\s*([-0-9.e+]+)\s*([km]?Pa)"
    coeffs = re.findall(coeff_pattern, model_section[:200])
    print(f"  Found {len(coeffs)} coefficients: {coeffs}")
    
    start_pos = match_pos + 1

print(f"\n\nTotal matches found: {found_count}")
