import re

with open('outputs_PSII/log_PSII_test.txt', 'r') as f:
    content = f.read()

scenario = "O2_RelNoise0pct"
method = "LarsCV"

# Try the corrected pattern
search_pattern = f"  Final Model ({method}):"

print(f"Searching for: '{search_pattern}'")
print(f"Pattern length: {len(search_pattern)}")

match_pos = content.find(search_pattern)
print(f"\nFirst match at position: {match_pos}")

if match_pos != -1:
    # Show context
    print(f"\nContext before match:")
    print(repr(content[match_pos-50:match_pos]))
    print(f"\nMatch and after:")
    print(repr(content[match_pos:match_pos+100]))
    
    # Check scenario
    prev_section = content[max(0, match_pos-5000):match_pos]
    test3 = f"SCENARIO: {scenario}" in prev_section.upper()
    test4 = scenario in prev_section
    print(f"\n'SCENARIO: {scenario}' (upper) found: {test3}")
    print(f"'{scenario}' anywhere found: {test4}")
else:
    # Try without spaces
    alt_pattern = f"Final Model ({method}):"
    alt_pos = content.find(alt_pattern)
    print(f"\nAlternate pattern (no leading spaces) found at: {alt_pos}")
