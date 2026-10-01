scenario = "O2_RelNoise0pct"
prev_section = "### PROCESSING SCENARIO: O2_RelNoise0pct (Fit UT+PS+EBT P11) ###"

test1 = f"SCENARIO: {scenario}".upper() in prev_section.upper()
test2 = scenario.upper() in prev_section.upper()

print(f"Test 1 (SCENARIO: {{scenario}}.upper() in prev.upper()): {test1}")
print(f"Test 2 (scenario.upper() in prev.upper()): {test2}")

print(f"\nSearching for: '{f'SCENARIO: {scenario}'.upper()}'")
print(f"In: '{prev_section.upper()}'")
