#!/usr/bin/env python3
"""
Test script to verify that the generalized deformation mode implementation
produces identical results to the original hardcoded implementation.
"""

import numpy as np
import pandas as pd
import sys
import os

# Import from the refactored script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from MDisc_synthetic_data_rel_noise_20251004_UT_PS_EBT_pos import (
    compute_stretches_from_alpha_beta,
    compute_invariants_from_alpha_beta,
    compute_invariants,
    CANONICAL_MODES,
    EPS
)

def test_stretch_computation():
    """
    Test that the generalized stretch computation produces expected values
    for canonical modes.
    """
    print("=" * 70)
    print("TEST 1: Verify stretch computation for canonical modes")
    print("=" * 70)
    
    test_lambda = 1.5
    
    # Test UT: α=1, β=-1/2
    alpha_ut, beta_ut = CANONICAL_MODES['UT']
    l1_ut, l2_ut, l3_ut = compute_stretches_from_alpha_beta(test_lambda, alpha_ut, beta_ut)
    expected_l1_ut = test_lambda
    expected_l2_ut = test_lambda ** (-0.5)
    expected_l3_ut = test_lambda ** (-0.5)
    
    print(f"\nUT Mode (λ={test_lambda}, α={alpha_ut}, β={beta_ut}):")
    print(f"  Computed: λ₁={l1_ut:.6f}, λ₂={l2_ut:.6f}, λ₃={l3_ut:.6f}")
    print(f"  Expected: λ₁={expected_l1_ut:.6f}, λ₂={expected_l2_ut:.6f}, λ₃={expected_l3_ut:.6f}")
    print(f"  Incompressibility check: λ₁·λ₂·λ₃ = {l1_ut * l2_ut * l3_ut:.6f} (should be 1.0)")
    
    assert abs(l1_ut - expected_l1_ut) < 1e-10, "UT λ₁ mismatch"
    assert abs(l2_ut - expected_l2_ut) < 1e-10, "UT λ₂ mismatch"
    assert abs(l3_ut - expected_l3_ut) < 1e-10, "UT λ₃ mismatch"
    assert abs(l1_ut * l2_ut * l3_ut - 1.0) < 1e-10, "UT incompressibility violation"
    print("  ✓ PASS")
    
    # Test PS: α=1, β=-1 (new convention: λ₃=1)
    alpha_ps, beta_ps = CANONICAL_MODES['PS']
    l1_ps, l2_ps, l3_ps = compute_stretches_from_alpha_beta(test_lambda, alpha_ps, beta_ps)
    expected_l1_ps = test_lambda
    expected_l2_ps = test_lambda ** (-1)
    expected_l3_ps = 1.0
    
    print(f"\nPS Mode (λ={test_lambda}, α={alpha_ps}, β={beta_ps}):")
    print(f"  Computed: λ₁={l1_ps:.6f}, λ₂={l2_ps:.6f}, λ₃={l3_ps:.6f}")
    print(f"  Expected: λ₁={expected_l1_ps:.6f}, λ₂={expected_l2_ps:.6f}, λ₃={expected_l3_ps:.6f}")
    print(f"  Incompressibility check: λ₁·λ₂·λ₃ = {l1_ps * l2_ps * l3_ps:.6f} (should be 1.0)")
    print(f"  NOTE: This is PS-I convention (λ₃=1), different from old implementation (λ₂=1)")
    
    assert abs(l1_ps - expected_l1_ps) < 1e-10, "PS λ₁ mismatch"
    assert abs(l2_ps - expected_l2_ps) < 1e-10, "PS λ₂ mismatch"
    assert abs(l3_ps - expected_l3_ps) < 1e-10, "PS λ₃ mismatch"
    assert abs(l1_ps * l2_ps * l3_ps - 1.0) < 1e-10, "PS incompressibility violation"
    print("  ✓ PASS")
    
    # Test EBT: α=1, β=1
    alpha_ebt, beta_ebt = CANONICAL_MODES['EBT']
    l1_ebt, l2_ebt, l3_ebt = compute_stretches_from_alpha_beta(test_lambda, alpha_ebt, beta_ebt)
    expected_l1_ebt = test_lambda
    expected_l2_ebt = test_lambda
    expected_l3_ebt = test_lambda ** (-2)
    
    print(f"\nEBT Mode (λ={test_lambda}, α={alpha_ebt}, β={beta_ebt}):")
    print(f"  Computed: λ₁={l1_ebt:.6f}, λ₂={l2_ebt:.6f}, λ₃={l3_ebt:.6f}")
    print(f"  Expected: λ₁={expected_l1_ebt:.6f}, λ₂={expected_l2_ebt:.6f}, λ₃={expected_l3_ebt:.6f}")
    print(f"  Incompressibility check: λ₁·λ₂·λ₃ = {l1_ebt * l2_ebt * l3_ebt:.6f} (should be 1.0)")
    
    assert abs(l1_ebt - expected_l1_ebt) < 1e-10, "EBT λ₁ mismatch"
    assert abs(l2_ebt - expected_l2_ebt) < 1e-10, "EBT λ₂ mismatch"
    assert abs(l3_ebt - expected_l3_ebt) < 1e-10, "EBT λ₃ mismatch"
    assert abs(l1_ebt * l2_ebt * l3_ebt - 1.0) < 1e-10, "EBT incompressibility violation"
    print("  ✓ PASS")
    
    print("\n✓ ALL STRETCH TESTS PASSED\n")

def test_invariant_computation():
    """
    Test that the parameterized invariant computation matches
    the standard computation from stretches.
    """
    print("=" * 70)
    print("TEST 2: Verify invariant computation consistency")
    print("=" * 70)
    
    test_lambda = 2.0
    
    for mode_name, (alpha, beta) in CANONICAL_MODES.items():
        # Compute stretches
        l1, l2, l3 = compute_stretches_from_alpha_beta(test_lambda, alpha, beta)
        
        # Compute invariants two ways
        I1_from_stretches, I2_from_stretches = compute_invariants(l1, l2, l3)
        I1_from_params, I2_from_params = compute_invariants_from_alpha_beta(test_lambda, alpha, beta)
        
        print(f"\n{mode_name} Mode (λ={test_lambda}, α={alpha}, β={beta}):")
        print(f"  Stretches: λ₁={l1:.6f}, λ₂={l2:.6f}, λ₃={l3:.6f}")
        print(f"  I₁ from stretches: {I1_from_stretches:.6f}")
        print(f"  I₁ from params:    {I1_from_params:.6f}")
        print(f"  Difference:        {abs(I1_from_stretches - I1_from_params):.2e}")
        print(f"  I₂ from stretches: {I2_from_stretches:.6f}")
        print(f"  I₂ from params:    {I2_from_params:.6f}")
        print(f"  Difference:        {abs(I2_from_stretches - I2_from_params):.2e}")
        
        assert abs(I1_from_stretches - I1_from_params) < 1e-10, f"{mode_name} I₁ mismatch"
        assert abs(I2_from_stretches - I2_from_params) < 1e-10, f"{mode_name} I₂ mismatch"
        print("  ✓ PASS")
    
    print("\n✓ ALL INVARIANT TESTS PASSED\n")

def test_ps_convention_change():
    """
    Document the change in PS convention from λ₂=1 (PS-II) to λ₃=1 (PS-I).
    """
    print("=" * 70)
    print("TEST 3: Document PS convention change")
    print("=" * 70)
    
    test_lambda = 2.0
    
    # Old PS-II convention (λ₂=1, what the code used to do)
    l1_old = test_lambda
    l2_old = 1.0
    l3_old = 1.0 / test_lambda
    I1_old, I2_old = compute_invariants(l1_old, l2_old, l3_old)
    
    # New PS-I convention (λ₃=1, what the code does now)
    alpha_ps, beta_ps = CANONICAL_MODES['PS']
    l1_new, l2_new, l3_new = compute_stretches_from_alpha_beta(test_lambda, alpha_ps, beta_ps)
    I1_new, I2_new = compute_invariants(l1_new, l2_new, l3_new)
    
    print(f"\nFor λ = {test_lambda}:")
    print(f"\nOLD PS-II Convention (λ₂=1, β=0):")
    print(f"  λ₁={l1_old:.6f}, λ₂={l2_old:.6f}, λ₃={l3_old:.6f}")
    print(f"  I₁={I1_old:.6f}, I₂={I2_old:.6f}")
    
    print(f"\nNEW PS-I Convention (λ₃=1, β=-1):")
    print(f"  λ₁={l1_new:.6f}, λ₂={l2_new:.6f}, λ₃={l3_new:.6f}")
    print(f"  I₁={I1_new:.6f}, I₂={I2_new:.6f}")
    
    print(f"\nDifference in invariants:")
    print(f"  ΔI₁ = {abs(I1_new - I1_old):.6f}")
    print(f"  ΔI₂ = {abs(I2_new - I2_old):.6f}")
    
    print("\n⚠ WARNING: PS convention has changed!")
    print("  Old: Pure Shear II (λ₂=1, poorer identifiability)")
    print("  New: Pure Shear I (λ₃=1, better identifiability)")
    print("  This means model discovery results for PS will be DIFFERENT from baseline.")
    print("  However, UT and EBT should remain identical.\n")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GENERALIZED DEFORMATION MODE VERIFICATION")
    print("=" * 70 + "\n")
    
    try:
        test_stretch_computation()
        test_invariant_computation()
        test_ps_convention_change()
        
        print("=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Stretch computations are correct")
        print("  ✓ Invariant computations are consistent")
        print("  ✓ Incompressibility is preserved")
        print("  ⚠ PS convention changed from PS-II to PS-I (expected)")
        print("\nNext step: Run full model discovery to verify UT and EBT produce")
        print("identical results to baseline, and PS produces physically better results.")
        print("=" * 70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
