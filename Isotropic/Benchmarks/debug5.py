with open('outputs_PSII/log_PSII_test.txt', 'r') as f:
    content = f.read()

method = "LarsCV"
search_pattern = f"  Final Model ({method}):"

match_pos = content.find(search_pattern)
print(f"Final Model at position: {match_pos}")

# Search backwards for scenario marker
scenario_pattern = "### PROCESSING SCENARIO:"
scenario_pos = content.rfind(scenario_pattern, 0, match_pos)
print(f"Scenario marker at position: {scenario_pos}")
print(f"Distance: {match_pos - scenario_pos} characters")

if scenario_pos != -1:
    # Extract the scenario line
    line_end = content.find('\n', scenario_pos)
    scenario_line = content[scenario_pos:line_end]
    print(f"\nScenario line: {scenario_line}")
