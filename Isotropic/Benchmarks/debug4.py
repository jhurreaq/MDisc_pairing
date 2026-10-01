with open('outputs_PSII/log_PSII_test.txt', 'r') as f:
    content = f.read()

scenario = "O2_RelNoise0pct"
method = "LarsCV"
search_pattern = f"  Final Model ({method}):"

match_pos = content.find(search_pattern)
print(f"Match at position: {match_pos}")

prev_section = content[max(0, match_pos-5000):match_pos]

# Show last few lines
lines = prev_section.split('\n')
print(f"\nLast 10 lines before match:")
for i, line in enumerate(lines[-10:]):
    print(f"  {i}: {repr(line[:80])}")

# Check for scenario
search_target = f"SCENARIO: {scenario}".upper()
print(f"\nSearching for: '{search_target}'")
print(f"In uppercased prev_section: {search_target in prev_section.upper()}")

# Find where scenario appears
if scenario in prev_section:
    scenario_pos = prev_section.rfind(scenario)
    context = prev_section[scenario_pos-50:scenario_pos+100]
    print(f"\nFound '{scenario}' at offset {scenario_pos}:")
    print(repr(context))
