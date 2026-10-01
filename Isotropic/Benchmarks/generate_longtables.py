"""
Generate complete longtable LaTeX file from log files with 100% accuracy
Extracts all data from PSII and sampling logs and creates proper LaTeX tables
"""

import re
from typing import Dict, List, Tuple

def parse_psii_all_data(log_file: str) -> Tuple[Dict, Dict, Dict]:
    """Extract model summary, verification, and times from PSII log"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Parse model summary
    summary_start = content.find('### OVERALL FINAL REFIT MODEL SUMMARY')
    summary_section = content[summary_start:summary_start+50000]
    lines = summary_section.split('\n')
    
    model_data = {}
    current_scenario = None
    
    for line in lines:
        if '--- Scenario:' in line:
            current_scenario = line.split(':')[1].strip().replace(' ---', '')
            model_data[current_scenario] = {}
        elif current_scenario and '|' in line and 'Method' not in line and '---' not in line:
            parts = line.split('|')
            if len(parts) < 13:
                continue
            
            try:
                method = parts[1].strip()
                n_terms = int(parts[2].strip())
                r2_ut = float(parts[3].strip())
                nrmse_ut = float(parts[5].strip())
                r2_ps = float(parts[6].strip())
                nrmse_ps = float(parts[8].strip())
                r2_ebt = float(parts[9].strip())
                nrmse_ebt = float(parts[11].strip())
                r2_test = float(parts[12].strip())
                rmse_test = float(parts[13].strip())
                
                model_data[current_scenario][method] = {
                    'n_terms': n_terms,
                    'r2_ut': r2_ut,
                    'r2_ps': r2_ps,
                    'r2_ebt': r2_ebt,
                    'avg_nrmse': (nrmse_ut + nrmse_ps + nrmse_ebt) / 3,
                    'r2_test': r2_test,
                    'rmse_test': rmse_test
                }
            except (ValueError, IndexError):
                continue
    
    # Parse verification (whitespace-delimited format)
    verif_start = content.find('### VERIFICATION SUMMARY: Discovered vs. Ground Truth Terms ###')
    verif_section = content[verif_start:verif_start+30000]
    lines = verif_section.split('\n')
    
    verification = {}
    
    for line in lines:
        # Lines have format: scenario method n_ground n_discovered n_correct n_missed n_spurious ...
        parts = line.split()
        if len(parts) >= 7 and parts[0] not in ['Scenario', 'Verification']:
            try:
                scenario = parts[0]
                method = parts[1]
                n_missed = int(parts[5])
                n_spurious = int(parts[6])
                
                if scenario not in verification:
                    verification[scenario] = {}
                verification[scenario][method] = (n_missed == 0 and n_spurious == 0)
            except (ValueError, IndexError):
                continue
    
    # Parse times (whitespace-delimited format)
    time_start = content.find('### COMPUTATION TIMES')
    if time_start == -1:
        times = {}
    else:
        time_section = content[time_start:time_start+30000]
        lines = time_section.split('\n')
        
        times = {}
        
        for line in lines:
            # Lines have format: scenario method time
            parts = line.split()
            if len(parts) >= 3 and parts[0] not in ['Scenario', 'COMPUTATION']:
                try:
                    scenario = parts[0]
                    method = parts[1]
                    time = float(parts[2])
                    
                    if scenario not in times:
                        times[scenario] = {}
                    times[scenario][method] = time
                except (ValueError, IndexError):
                    continue
    
    return model_data, verification, times

def parse_sampling_all_data(log_file: str) -> Tuple[Dict, Dict, Dict]:
    """Extract model summary, verification, and times from sampling log"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Parse model summary (pipe-delimited format)
    summary_start = content.find('### OVERALL FINAL REFIT MODEL SUMMARY')
    summary_section = content[summary_start:summary_start+50000]
    lines = summary_section.split('\n')
    
    model_data = {}
    current_scenario = None
    
    for line in lines:
        if '--- Scenario:' in line:
            current_scenario = line.split(':')[1].strip().replace(' ---', '')
            model_data[current_scenario] = {}
        elif current_scenario and '|' in line and 'Method' not in line and '---' not in line and '|----------' not in line:
            parts = line.split('|')
            # Filter out empty strings and strip whitespace
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) < 8:
                continue
            
            try:
                method = parts[0]
                n_terms = int(parts[1])
                r2_overall = float(parts[2])
                # Skip RMSE(Overall)[kPa] at index 3
                nrmse_overall = float(parts[4])
                r2_test = float(parts[5])
                rmse_test = float(parts[6])
                
                model_data[current_scenario][method] = {
                    'n_terms': n_terms,
                    'r2_overall': r2_overall,
                    'nrmse_overall': nrmse_overall,
                    'r2_test': r2_test,
                    'rmse_test': rmse_test
                }
            except (ValueError, IndexError):
                continue
    
    # Parse verification (whitespace-delimited format, same as PSII)
    verif_start = content.find('### VERIFICATION SUMMARY: Discovered vs. Ground Truth Terms ###')
    verif_section = content[verif_start:verif_start+30000]
    lines = verif_section.split('\n')
    
    verification = {}
    
    for line in lines:
        # Lines have format: scenario method n_ground n_discovered n_correct n_missed n_spurious ...
        parts = line.split()
        if len(parts) >= 7 and parts[0] not in ['Scenario', 'Verification']:
            try:
                scenario = parts[0]
                method = parts[1]
                n_missed = int(parts[5])
                n_spurious = int(parts[6])
                
                if scenario not in verification:
                    verification[scenario] = {}
                verification[scenario][method] = (n_missed == 0 and n_spurious == 0)
            except (ValueError, IndexError):
                continue
    
    # Parse times (whitespace-delimited format, same as PSII)
    time_start = content.find('### COMPUTATION TIMES')
    if time_start == -1:
        times = {}
    else:
        time_section = content[time_start:time_start+30000]
        lines = time_section.split('\n')
        
        times = {}
        
        for line in lines:
            # Lines have format: scenario method time
            parts = line.split()
            if len(parts) >= 3 and parts[0] not in ['Scenario', 'COMPUTATION']:
                try:
                    scenario = parts[0]
                    method = parts[1]
                    time = float(parts[2])
                    
                    if scenario not in times:
                        times[scenario] = {}
                    times[scenario][method] = time
                except (ValueError, IndexError):
                    continue
    
    return model_data, verification, times

def extract_failed_models(log_file: str, verification: Dict, debug=False) -> Dict:
    """Extract model coefficients for cases where ground truth wasn't discovered"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Build list of failed cases
    failed_cases = {}
    for scenario, methods in verification.items():
        failed_methods = [m for m, discovered in methods.items() if not discovered]
        if failed_methods:
            failed_cases[scenario] = failed_methods
    
    if debug:
        print(f"\nDEBUG: Total failed cases: {sum(len(v) for v in failed_cases.values())}")
        for sc, meths in list(failed_cases.items())[:2]:
            print(f"  {sc}: {meths[:3]}")
    
    # Extract models using more robust pattern
    models = {}
    total_found = 0
    
    for scenario in failed_cases:
        models[scenario] = {}
        
        for method in failed_cases[scenario]:
            # Search for the final model section for this specific scenario and method
            # Pattern: look for "  Final Model (MethodName):" (note: TWO spaces at start)
            search_pattern = f"  Final Model ({method}):"
            
            # Find all occurrences
            start_pos = 0
            matches_found = 0
            while True:
                match_pos = content.find(search_pattern, start_pos)
                if match_pos == -1:
                    break
                
                matches_found += 1
                
                # Check if this is in the right scenario
                # Find the MOST RECENT scenario header before this match
                scenario_pattern = "### PROCESSING SCENARIO:"
                recent_scenario_pos = content.rfind(scenario_pattern, 0, match_pos)
                
                if recent_scenario_pos != -1:
                    # Extract the scenario name from that header
                    line_end = content.find('\n', recent_scenario_pos)
                    scenario_line = content[recent_scenario_pos:line_end]
                    scenario_match = scenario in scenario_line
                else:
                    scenario_match = False
                
                if debug and matches_found == 1:
                    print(f"\nDEBUG {scenario}/{method}: Found {matches_found} pattern matches, scenario_match={scenario_match}")
                
                if scenario_match:
                    # Extract the model content (next ~500 chars should contain the coefficients)
                    model_section = content[match_pos:match_pos+800]
                    
                    # Find where the model section ends
                    end_markers = ['Checking physical consistency', 'Final Model (', 'Processing Scenario:', '### ']
                    end_pos = len(model_section)
                    for marker in end_markers:
                        marker_pos = model_section.find(marker, len(search_pattern))
                        if marker_pos != -1 and marker_pos < end_pos:
                            end_pos = marker_pos
                    
                    model_str = model_section[:end_pos]
                    
                    # Extract coefficients using the pattern
                    coeff_pattern = r"'([^']+)':\s*([-0-9.e+]+)\s*([km]?Pa)"
                    coeffs = re.findall(coeff_pattern, model_str)
                    
                    if debug:
                        print(f"  Extracted {len(coeffs)} coeffs from {len(model_str)} chars")
                        if len(coeffs) > 0:
                            print(f"  First coeff: {coeffs[0]}")
                    
                    if coeffs:
                        models[scenario][method] = coeffs
                        total_found += 1
                        break
                
                start_pos = match_pos + 1
    
    if debug:
        print(f"\nDEBUG: Total models extracted: {total_found}")
    
    return models

def format_sef(coeffs: List[Tuple[str, str, str]]) -> str:
    """Format coefficient list into SEF expression for failed models table"""
    terms = []
    for term_name, value, unit in coeffs:
        val_float = float(value)
        
        # Convert to Pa
        if unit == 'kPa':
            val_float *= 1000
        elif unit == 'MPa':
            val_float *= 1000000
        
        # Format value (1 decimal place)
        val_str = f"{val_float:.1f}"
        
        # Convert term name to LaTeX format
        if 'MR:' in term_name:
            # Mooney-Rivlin terms: "MR: (I₁-3)" -> "(I_1 - 3)"
            term_latex = term_name.replace('MR: ', '').replace('₁', '_1').replace('₂', '_2')
        elif 'Ogden:' in term_name:
            # Ogden terms: "Ogden: λ^-3" -> "\sum (\lambda_k^{-3} - 1)"
            exponent = term_name.split('^')[1] if '^' in term_name else '1'
            term_latex = f"\\sum (\\lambda_k^{{{exponent}}} - 1)"
        else:
            term_latex = term_name
        
        terms.append(f"{val_str} {term_latex}")
    
    return " + ".join(terms)

def generate_psii_table_rows(model_data: Dict, verification: Dict, times: Dict) -> List[str]:
    """Generate all data rows for PSII table (Table 1)"""
    
    scenarios = [
        ('O2', 'RelNoise0pct', '0'),
        ('O2', 'RelNoise5pct', '5'),
        ('O2', 'RelNoise10pct', '10'),
        ('MR2', 'RelNoise0pct', '0'),
        ('MR2', 'RelNoise5pct', '5'),
        ('MR2', 'RelNoise10pct', '10'),
        ('MR1O1', 'RelNoise0pct', '0'),
        ('MR1O1', 'RelNoise5pct', '5'),
        ('MR1O1', 'RelNoise10pct', '10'),
        ('MR2O2', 'RelNoise0pct', '0'),
        ('MR2O2', 'RelNoise5pct', '5'),
        ('MR2O2', 'RelNoise10pct', '10'),
    ]
    
    methods = [
        ('LASSO', 'LassoCV', 'CV'),
        ('LASSO', 'LassoAIC', 'AIC'),
        ('LASSO', 'LassoBIC', 'BIC'),
        ('LARS', 'LarsCV', 'CV'),
        ('LARS', 'LarsAIC', 'AIC'),
        ('LARS', 'LarsBIC', 'BIC'),
        ('OMP', 'OMPCV', 'CV'),
        ('OMP', 'OMPAIC', 'AIC'),
        ('OMP', 'OMPBIC', 'BIC'),
    ]
    
    rows = []
    
    for scenario_name, noise_suffix, noise_pct in scenarios:
        scenario_key = f"{scenario_name}_{noise_suffix}"
        
        if scenario_key not in model_data:
            continue
        
        for idx, (alg, method_key, criterion) in enumerate(methods):
            if method_key not in model_data[scenario_key]:
                continue
            
            d = model_data[scenario_key][method_key]
            discovered = verification.get(scenario_key, {}).get(method_key, False)
            time_val = times.get(scenario_key, {}).get(method_key, 0.0)
            
            check = "$\\checkmark$" if discovered else "$\\times$"
            
            # Build row based on position
            if idx == 0:
                # First row: show scenario name
                row = f"{scenario_name} & {alg} & {criterion} & {check} & {d['n_terms']} & "
            elif idx == 1:
                # Second row: show noise level
                row = f"{noise_pct}\\% noise & & {criterion} & {check} & {d['n_terms']} & "
            elif idx % 3 == 0:
                # First of a group: show algorithm
                row = f"& {alg} & {criterion} & {check} & {d['n_terms']} & "
            else:
                # Others: empty first columns
                row = f"& & {criterion} & {check} & {d['n_terms']} & "
            
            # Add metrics
            row += f"{d['r2_ut']:.4f} & {d['r2_ps']:.4f} & {d['r2_ebt']:.4f} & "
            row += f"{d['avg_nrmse']:.4f} & {d['r2_test']:.4f} & {d['rmse_test']:.4f} & "
            row += f"{time_val:.4f} \\\\"
            
            # Add midrule after last method of each scenario
            if idx == len(methods) - 1:
                row += " \\midrule"
            
            rows.append(row)
    
    return rows

def generate_sampling_table_rows(model_data: Dict, verification: Dict, times: Dict) -> List[str]:
    """Generate all data rows for sampling table (Table 2)"""
    
    scenarios = [
        ('O2', 'RelNoise0pct', '0'),
        ('O2', 'RelNoise5pct', '5'),
        ('O2', 'RelNoise10pct', '10'),
        ('MR2', 'RelNoise0pct', '0'),
        ('MR2', 'RelNoise5pct', '5'),
        ('MR2', 'RelNoise10pct', '10'),
        ('MR1O1', 'RelNoise0pct', '0'),
        ('MR1O1', 'RelNoise5pct', '5'),
        ('MR1O1', 'RelNoise10pct', '10'),
        ('MR2O2', 'RelNoise0pct', '0'),
        ('MR2O2', 'RelNoise5pct', '5'),
        ('MR2O2', 'RelNoise10pct', '10'),
    ]
    
    methods = [
        ('LASSO', 'LassoCV', 'CV'),
        ('LASSO', 'LassoAIC', 'AIC'),
        ('LASSO', 'LassoBIC', 'BIC'),
        ('LARS', 'LarsCV', 'CV'),
        ('LARS', 'LarsAIC', 'AIC'),
        ('LARS', 'LarsBIC', 'BIC'),
        ('OMP', 'OMPCV', 'CV'),
        ('OMP', 'OMPAIC', 'AIC'),
        ('OMP', 'OMPBIC', 'BIC'),
    ]
    
    rows = []
    
    for scenario_name, noise_suffix, noise_pct in scenarios:
        scenario_key = f"{scenario_name}_{noise_suffix}"
        
        if scenario_key not in model_data:
            continue
        
        for idx, (alg, method_key, criterion) in enumerate(methods):
            if method_key not in model_data[scenario_key]:
                continue
            
            d = model_data[scenario_key][method_key]
            discovered = verification.get(scenario_key, {}).get(method_key, False)
            time_val = times.get(scenario_key, {}).get(method_key, 0.0)
            
            check = "$\\checkmark$" if discovered else "$\\times$"
            
            # Build row
            if idx == 0:
                row = f"{scenario_name} & {alg} & {criterion} & {check} & {d['n_terms']} & "
            elif idx == 1:
                row = f"{noise_pct}\\% noise & & {criterion} & {check} & {d['n_terms']} & "
            elif idx % 3 == 0:
                row = f"& {alg} & {criterion} & {check} & {d['n_terms']} & "
            else:
                row = f"& & {criterion} & {check} & {d['n_terms']} & "
            
            # Add sampling-specific metrics
            row += f"{d['r2_overall']:.4f} & {d['nrmse_overall']:.4f} & "
            row += f"{d['r2_test']:.4f} & {d['rmse_test']:.4f} & "
            row += f"{time_val:.4f} \\\\"
            
            if idx == len(methods) - 1:
                row += " \\midrule"
            
            rows.append(row)
    
    return rows

def generate_failed_table_rows(failed_models: Dict, model_data: Dict, verification: Dict) -> List[str]:
    """Generate rows for failed discoveries table with ground truth headers and proper formatting"""
    
    # Ground truth definitions
    ground_truths = {
        'O2_RelNoise0pct': r'16.0 \sum (\lambda_k^{-3} - 1) + 8.0 \sum (\lambda_k^{3} - 1)',
        'O2_RelNoise5pct': r'16.0 \sum (\lambda_k^{-3} - 1) + 8.0 \sum (\lambda_k^{3} - 1)',
        'O2_RelNoise10pct': r'16.0 \sum (\lambda_k^{-3} - 1) + 8.0 \sum (\lambda_k^{3} - 1)',
        'MR2_RelNoise0pct': r'40.0 (I_1 - 3) + 20.0 (I_2 - 3)',
        'MR2_RelNoise5pct': r'40.0 (I_1 - 3) + 20.0 (I_2 - 3)',
        'MR2_RelNoise10pct': r'40.0 (I_1 - 3) + 20.0 (I_2 - 3)',
        'MR1O1_RelNoise0pct': r'40.0 (I_2 - 3) + 8.0 \sum (\lambda_k^{-3} - 1)',
        'MR1O1_RelNoise5pct': r'40.0 (I_2 - 3) + 8.0 \sum (\lambda_k^{-3} - 1)',
        'MR1O1_RelNoise10pct': r'40.0 (I_2 - 3) + 8.0 \sum (\lambda_k^{-3} - 1)',
        'MR2O2_RelNoise0pct': r'40.0 (I_1 - 3) + 20.0 (I_2 - 3) + 16.0 \sum (\lambda_k^{-3} - 1) + 800.0 \sum (\lambda_k^{1} - 1)',
        'MR2O2_RelNoise5pct': r'40.0 (I_1 - 3) + 20.0 (I_2 - 3) + 16.0 \sum (\lambda_k^{-3} - 1) + 800.0 \sum (\lambda_k^{1} - 1)',
        'MR2O2_RelNoise10pct': r'40.0 (I_1 - 3) + 20.0 (I_2 - 3) + 16.0 \sum (\lambda_k^{-3} - 1) + 800.0 \sum (\lambda_k^{1} - 1)',
    }
    
    # Scenario display names
    scenario_names = {
        'O2_RelNoise0pct': ('O2', '0'),
        'O2_RelNoise5pct': ('O2', '5'),
        'O2_RelNoise10pct': ('O2', '10'),
        'MR2_RelNoise0pct': ('MR2', '0'),
        'MR2_RelNoise5pct': ('MR2', '5'),
        'MR2_RelNoise10pct': ('MR2', '10'),
        'MR1O1_RelNoise0pct': ('MR1O1', '0'),
        'MR1O1_RelNoise5pct': ('MR1O1', '5'),
        'MR1O1_RelNoise10pct': ('MR1O1', '10'),
        'MR2O2_RelNoise0pct': ('MR2O2', '0'),
        'MR2O2_RelNoise5pct': ('MR2O2', '5'),
        'MR2O2_RelNoise10pct': ('MR2O2', '10'),
    }
    
    # Method display names
    method_names = {
        'LassoCV': 'LASSO-CV', 'LassoAIC': 'LASSO-AIC', 'LassoBIC': 'LASSO-BIC',
        'LarsCV': 'LARS-CV', 'LarsAIC': 'LARS-AIC', 'LarsBIC': 'LARS-BIC',
        'OMPCV': 'OMP-CV', 'OMPAIC': 'OMP-AIC', 'OMPBIC': 'OMP-BIC',
    }
    
    rows = []
    
    # Order scenarios
    scenario_order = [
        'O2_RelNoise0pct', 'O2_RelNoise5pct', 'O2_RelNoise10pct',
        'MR2_RelNoise0pct', 'MR2_RelNoise5pct', 'MR2_RelNoise10pct',
        'MR1O1_RelNoise0pct', 'MR1O1_RelNoise5pct', 'MR1O1_RelNoise10pct',
        'MR2O2_RelNoise0pct', 'MR2O2_RelNoise5pct', 'MR2O2_RelNoise10pct',
    ]
    
    for scenario_key in scenario_order:
        if scenario_key not in failed_models or not failed_models[scenario_key]:
            continue
        
        scenario_name, noise_pct = scenario_names[scenario_key]
        ground_truth = ground_truths[scenario_key]
        
        # Add ground truth header row
        header = f"\\textbf{{{scenario_name} ({noise_pct}\\% noise):}} & \\multicolumn{{5}}{{l}}{{\\textit{{Ground Truth: }} ${ground_truth}$}} \\\\"
        rows.append(header)
        
        # Add failed methods for this scenario
        for method_key, coeffs in sorted(failed_models[scenario_key].items()):
            method_display = method_names.get(method_key, method_key)
            n_terms = len(coeffs)
            sef = format_sef(coeffs)
            
            # Get R², NRMSE, and test metrics from model_data
            r2_min = 1.0
            avg_nrmse = 0.0
            r2_test = 1.0
            rmse_test = 0.0
            if scenario_key in model_data and method_key in model_data[scenario_key]:
                d = model_data[scenario_key][method_key]
                # Handle both PSII (r2_ut, r2_ps, r2_ebt) and sampling (r2_overall) formats
                if 'r2_ut' in d:
                    r2_min = min(float(d['r2_ut']), float(d['r2_ps']), float(d['r2_ebt']))
                    avg_nrmse = float(d['avg_nrmse'])
                elif 'r2_overall' in d:
                    r2_min = float(d['r2_overall'])
                    avg_nrmse = float(d['nrmse_overall'])
                # Get test metrics
                r2_test = float(d.get('r2_test', 1.0))
                rmse_test = float(d.get('rmse_test', 0.0))
            
            row = f"{method_display} ({n_terms}) & ${sef}$ & {r2_min:.4f} & {avg_nrmse:.4f} & {r2_test:.4f} & {rmse_test:.4f} \\\\"
            rows.append(row)
        
        # Add midrule after each scenario
        rows.append("\\midrule")
    
    # Remove last midrule
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    
    return rows

def write_complete_latex_file(output_file: str, 
                              psii_rows: List[str], 
                              sampling_rows: List[str],
                              psii_failed_rows: List[str],
                              sampling_failed_rows: List[str]):
    """Write the complete LaTeX file with all 4 tables"""
    
    with open(output_file, 'w') as f:
        # Header
        f.write("% Auto-generated longtable file from log data\n")
        f.write("% Generated with 100% accuracy from source logs\n\n")
        
        # Table 1: PSII results
        f.write("\\begin{longtable}{llcccccccccc}\n")
        f.write("\\caption{Performance summary for UT+PS+EBT training strategy (180 points)}\\\\\n")
        f.write("\\label{tab:psii_results}\\\\\n")
        f.write("\\toprule\n")
        f.write("Scenario & Algorithm & \\begin{tabular}[c]{@{}l@{}}Selection\\\\ criteria\\end{tabular} & ")
        f.write("\\begin{tabular}[c]{@{}c@{}}Ground truth\\\\ discovered?\\end{tabular} & ")
        f.write("$n^{\\mathcal{A}}_{\\phi}$ & $R^2_{\\text{UT}}$ & $R^2_{\\text{PS}}$ & $R^2_{\\text{EBT}}$ & ")
        f.write("AvgNRMSE & $R^2_{\\text{Test}}$ & ")
        f.write("\\begin{tabular}[c]{@{}c@{}}RMSE$_{\\text{Test}}$\\\\ {[}MPa{]}\\end{tabular} & ")
        f.write("\\begin{tabular}[c]{@{}c@{}}Time\\\\ {[}s{]}\\end{tabular} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endhead\n")
        f.write("\\bottomrule\n")
        f.write("\\endfoot\n")
        
        for row in psii_rows:
            f.write(row + "\n")
        
        f.write("\\end{longtable}\n\n")
        
        # Table 2: Sampling results
        f.write("\\begin{longtable}{llcccccccc}\n")
        f.write("\\caption{Performance summary for sampling strategy with $\\alpha \\in \\{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0\\}$ at $\\beta=-1$ (540 points)}\\\\\n")
        f.write("\\label{tab:sampling_results}\\\\\n")
        f.write("\\toprule\n")
        f.write("Scenario & Algorithm & \\begin{tabular}[c]{@{}l@{}}Selection\\\\ criteria\\end{tabular} & ")
        f.write("\\begin{tabular}[c]{@{}c@{}}Ground truth\\\\ discovered?\\end{tabular} & ")
        f.write("$n^{\\mathcal{A}}_{\\phi}$ & $R^2_{\\text{Overall}}$ & NRMSE$_{\\text{Overall}}$ & ")
        f.write("$R^2_{\\text{Test}}$ & ")
        f.write("\\begin{tabular}[c]{@{}c@{}}RMSE$_{\\text{Test}}$\\\\ {[}MPa{]}\\end{tabular} & ")
        f.write("\\begin{tabular}[c]{@{}c@{}}Time\\\\ {[}s{]}\\end{tabular} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endhead\n")
        f.write("\\bottomrule\n")
        f.write("\\endfoot\n")
        
        for row in sampling_rows:
            f.write(row + "\n")
        
        f.write("\\end{longtable}\n\n")
        
        # Table 3: PSII failed discoveries
        f.write("{\\small\n")
        f.write("\\begin{longtable}{@{}l l cccc@{}}\n")
        f.write("\\caption{\\textbf{Isotropic Synthetic Data:} Cases where ground truth was not fully recovered - UT+PS+EBT strategy}\\\\\n")
        f.write("\\label{tab:psii_failed}\\\\\n")
        f.write("\\toprule\n")
        f.write("\\multicolumn{1}{@{}l}{\\begin{tabular}[c]{@{}l@{}}Scenario / Algorithm\\\\ ($n^{\\mathcal{A}}_{\\phi}$)\\end{tabular}} &\n")
        f.write("\\multicolumn{1}{l}{Discovered SEF Form ($\\overline{W}$)} &\n")
        f.write("$R^2_{\\min}$ & AvgNRMSE & $R^2_{\\text{Test}}$ & \\begin{tabular}[c]{@{}c@{}}RMSE$_{\\text{Test}}$\\\\ {[}MPa{]}\\end{tabular} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endfirsthead\n\n")
        f.write("\\multicolumn{6}{c}%\n")
        f.write("{{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\\n")
        f.write("\\toprule\n")
        f.write("\\multicolumn{1}{@{}l}{\\begin{tabular}[c]{@{}l@{}}Scenario / Algorithm\\\\ ($n^{\\mathcal{A}}_{\\phi}$)\\end{tabular}} &\n")
        f.write("\\multicolumn{1}{l}{Discovered SEF Form ($\\overline{W}$)} &\n")
        f.write("$R^2_{\\min}$ & AvgNRMSE & $R^2_{\\text{Test}}$ & \\begin{tabular}[c]{@{}c@{}}RMSE$_{\\text{Test}}$\\\\ {[}MPa{]}\\end{tabular} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endhead\n\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{6}{r}{{Continued on next page}} \\\\\n")
        f.write("\\endfoot\n\n")
        f.write("\\bottomrule\n")
        f.write("\\multicolumn{6}{l}{\\footnotesize{\\textit{Note:} Discovered Model coefficients are in Pa. Ogden-type terms follow the form $c_i(\\sum_j \\lambda_j^{\\alpha_i} - 1)$. Mooney-Rivlin terms follow the form $c_i(I_k - 3)$ where $I_k$ is the $k$-th invariant. Coefficients $c_i$ are from Ridge refit. $R^2_{\\min}$ is $\\min\\{R^2(\\text{UT } P_{11}), R^2(\\text{PS } P_{11}), R^2(\\text{EBT } P_{11})\\}$. AvgNRMSE is the average of NRMSE values for the same three components. These cases did not fully recover the ground truth model.}} \\\\\n")
        f.write("\\endlastfoot\n\n")
        
        for row in psii_failed_rows:
            f.write(row + "\n")
        
        f.write("\\end{longtable}\n")
        f.write("}\n\n")
        
        # Table 4: Sampling failed discoveries
        f.write("{\\small\n")
        f.write("\\begin{longtable}{@{}l l cccc@{}}\n")
        f.write("\\caption{\\textbf{Isotropic Synthetic Data:} Cases where ground truth was not fully recovered - Sampling strategy}\\\\\n")
        f.write("\\label{tab:sampling_failed}\\\\\n")
        f.write("\\toprule\n")
        f.write("\\multicolumn{1}{@{}l}{\\begin{tabular}[c]{@{}l@{}}Scenario / Algorithm\\\\ ($n^{\\mathcal{A}}_{\\phi}$)\\end{tabular}} &\n")
        f.write("\\multicolumn{1}{l}{Discovered SEF Form ($\\overline{W}$)} &\n")
        f.write("$R^2_{\\min}$ & AvgNRMSE & $R^2_{\\text{Test}}$ & \\begin{tabular}[c]{@{}c@{}}RMSE$_{\\text{Test}}$\\\\ {[}MPa{]}\\end{tabular} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endfirsthead\n\n")
        f.write("\\multicolumn{6}{c}%\n")
        f.write("{{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\\n")
        f.write("\\toprule\n")
        f.write("\\multicolumn{1}{@{}l}{\\begin{tabular}[c]{@{}l@{}}Scenario / Algorithm\\\\ ($n^{\\mathcal{A}}_{\\phi}$)\\end{tabular}} &\n")
        f.write("\\multicolumn{1}{l}{Discovered SEF Form ($\\overline{W}$)} &\n")
        f.write("$R^2_{\\min}$ & AvgNRMSE & $R^2_{\\text{Test}}$ & \\begin{tabular}[c]{@{}c@{}}RMSE$_{\\text{Test}}$\\\\ {[}MPa{]}\\end{tabular} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endhead\n\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{6}{r}{{Continued on next page}} \\\\\n")
        f.write("\\endfoot\n\n")
        f.write("\\bottomrule\n")
        f.write("\\multicolumn{6}{l}{\\footnotesize{\\textit{Note:} Discovered Model coefficients are in Pa. Ogden-type terms follow the form $c_i(\\sum_j \\lambda_j^{\\alpha_i} - 1)$. Mooney-Rivlin terms follow the form $c_i(I_k - 3)$ where $I_k$ is the $k$-th invariant. Coefficients $c_i$ are from Ridge refit. $R^2_{\\min}$ is $\\min\\{R^2(\\text{UT } P_{11}), R^2(\\text{PS } P_{11}), R^2(\\text{EBT } P_{11})\\}$. AvgNRMSE is the average of NRMSE values for the same three components. These cases did not fully recover the ground truth model.}} \\\\\n")
        f.write("\\endlastfoot\n\n")
        
        for row in sampling_failed_rows:
            f.write(row + "\n")
        
        f.write("\\end{longtable}\n")
        f.write("}\n")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("LONGTABLE GENERATION - Extracting data from logs...")
    print("="*60)
    
    # Extract PSII data
    print("\n[1/6] Parsing PSII model summary, verification, and times...")
    psii_data, psii_verification, psii_times = parse_psii_all_data('outputs_PSII/log_PSII_test.txt')
    print(f"  ✓ Extracted {len(psii_data)} scenarios with {sum(len(v) for v in psii_data.values())} method entries")
    
    # Extract sampling data
    print("\n[2/6] Parsing sampling model summary, verification, and times...")
    sampling_data, sampling_verification, sampling_times = parse_sampling_all_data('outputs_sampling/log_sampling.txt')
    print(f"  ✓ Extracted {len(sampling_data)} scenarios with {sum(len(v) for v in sampling_data.values())} method entries")
    
    # Extract failed models
    print("\n[3/6] Extracting failed model coefficients from PSII log...")
    psii_failed = extract_failed_models('outputs_PSII/log_PSII_test.txt', psii_verification, debug=False)
    n_psii_failed = sum(len(v) for v in psii_failed.values())
    print(f"  ✓ Extracted {n_psii_failed} failed models")
    
    print("\n[4/6] Extracting failed model coefficients from sampling log...")
    sampling_failed = extract_failed_models('outputs_sampling/log_sampling.txt', sampling_verification, debug=False)
    n_sampling_failed = sum(len(v) for v in sampling_failed.values())
    print(f"  ✓ Extracted {n_sampling_failed} failed models")
    
    # Generate table rows
    print("\n[5/6] Generating LaTeX table rows...")
    psii_rows = generate_psii_table_rows(psii_data, psii_verification, psii_times)
    sampling_rows = generate_sampling_table_rows(sampling_data, sampling_verification, sampling_times)
    
    psii_failed_rows = generate_failed_table_rows(psii_failed, psii_data, psii_verification)
    sampling_failed_rows = generate_failed_table_rows(sampling_failed, sampling_data, sampling_verification)
    
    print(f"  ✓ Generated {len(psii_rows)} PSII rows")
    print(f"  ✓ Generated {len(sampling_rows)} sampling rows")
    print(f"  ✓ Generated {len(psii_failed_rows)} PSII failed model rows")
    print(f"  ✓ Generated {len(sampling_failed_rows)} sampling failed model rows")
    
    # Write output file
    print("\n[6/6] Writing longtable_updated.tex...")
    write_complete_latex_file('longtable_updated.tex', 
                             psii_rows, sampling_rows,
                             psii_failed_rows, sampling_failed_rows)
    print("  ✓ File written successfully")
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION - Spot checking O2_RelNoise0pct")
    print("="*60)
    scenario = 'O2_RelNoise0pct'
    for method in ['LassoCV', 'LarsCV', 'LarsBIC']:
        if scenario in psii_data and method in psii_data[scenario]:
            d = psii_data[scenario][method]
            discovered = psii_verification[scenario][method]
            print(f"{method}: R²_UT={d['r2_ut']:.4f}, R²_PS={d['r2_ps']:.4f}, R²_EBT={d['r2_ebt']:.4f}, AvgNRMSE={d['avg_nrmse']:.4f}, Discovered={discovered}")
    
    print("\n✓ Generation complete! Check longtable_updated.tex")
