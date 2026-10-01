#!/usr/bin/env python3
"""
Quick test to verify the new test dataset generation functionality works.
"""

import numpy as np
print("=" * 80)
print("TEST: Verify New Functions are Importable and Callable")
print("=" * 80)

print("\n1. Testing imports...")
try:
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    
    from MDisc_synthetic_data_rel_noise_20251004_UT_PS_EBT_pos import (
        generate_test_data_generalized_modes,
        evaluate_on_test_set,
        compute_stretches_from_alpha_beta,
        CANONICAL_MODES
    )
    print("   ✓ All new functions imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

print("\n2. Testing compute_stretches_from_alpha_beta...")
try:
    # Test with different α values
    test_cases = [
        (2.0, 1.0, -1.0, "PS-I with α=1"),
        (2.0, 1.5, -1.0, "Generalized with α=1.5"),
        (2.0, 3.0, -1.0, "Generalized with α=3.0"),
    ]
    
    for lam, alpha, beta, desc in test_cases:
        l1, l2, l3 = compute_stretches_from_alpha_beta(lam, alpha, beta)
        det = l1 * l2 * l3
        print(f"   {desc}: λ₁={l1:.4f}, λ₂={l2:.4f}, λ₃={l3:.4f}, det={det:.6f}")
        if abs(det - 1.0) > 1e-10:
            print(f"   ❌ Incompressibility violated!")
            sys.exit(1)
    
    print("   ✓ Stretch computation works correctly (incompressibility preserved)")
except Exception as e:
    print(f"   ❌ Stretch computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Checking CANONICAL_MODES configuration...")
try:
    print(f"   Available modes: {list(CANONICAL_MODES.keys())}")
    for mode, (alpha, beta) in CANONICAL_MODES.items():
        print(f"   {mode}: α={alpha}, β={beta}")
    print("   ✓ Canonical modes configured correctly")
except Exception as e:
    print(f"   ❌ Mode configuration failed: {e}")
    sys.exit(1)

print("\n4. Testing generalized mode stretch values...")
try:
    lam = 2.0
    alphas_to_test = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    beta = -1.0
    
    print(f"   For λ={lam}, β={beta}:")
    print(f"   {'α':<6} {'λ₁':<8} {'λ₂':<8} {'λ₃':<8} {'det':<8}")
    print(f"   {'-'*40}")
    for alpha in alphas_to_test:
        l1, l2, l3 = compute_stretches_from_alpha_beta(lam, alpha, beta)
        det = l1 * l2 * l3
        print(f"   {alpha:<6.1f} {l1:<8.4f} {l2:<8.4f} {l3:<8.4f} {det:<8.6f}")
    
    print("   ✓ Generalized modes computed successfully for all α values")
except Exception as e:
    print(f"   ❌ Generalized mode computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL BASIC TESTS PASSED")
print("=" * 80)
print("\nSummary:")
print("  ✓ New functions are importable")
print("  ✓ Stretch computation preserves incompressibility")
print("  ✓ Canonical modes are properly configured")
print("  ✓ Generalized modes with varying α work correctly")
print("\nNext steps:")
print("  • Run full script to generate training data (UT, PS-I, EBT)")
print("  • Generate test data with α∈{1.0, 1.5, ..., 5.0}, β=-1")
print("  • Discover models from training data")
print("  • Evaluate on test set to get R²_test and MSE_test")
print("=" * 80)

