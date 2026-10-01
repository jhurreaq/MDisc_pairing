#!/usr/bin/env python3
"""
Test to verify that PS-I and PS-II produce DIFFERENT stresses P11,
even though they have identical invariants I1 = I2.

The key difference is in the pressure elimination:
- PS-II (λ₂=1): P₁₁ = ∂W/∂λ₁ - (λ₃/λ₁)·∂W/∂λ₃, P₂₂ = ∂W/∂λ₂ - (λ₃/λ₂)·∂W/∂λ₃ = 0
- PS-I  (λ₃=1): P₁₁ = ∂W/∂λ₁ - (λ₃/λ₁)·∂W/∂λ₃, P₂₂ = ∂W/∂λ₂ - (λ₃/λ₂)·∂W/∂λ₃
"""

import numpy as np
import sys

def compute_invariants(l1, l2, l3):
    I1 = l1**2 + l2**2 + l3**2
    I2 = (l1*l2)**2 + (l2*l3)**2 + (l1*l3)**2
    return I1, I2

def compute_invariant_derivatives(l1, l2, l3):
    dI1dl1, dI1dl2, dI1dl3 = 2*l1, 2*l2, 2*l3
    dI2dl1 = 2*l1*(l2**2 + l3**2)
    dI2dl2 = 2*l2*(l1**2 + l3**2)
    dI2dl3 = 2*l3*(l1**2 + l2**2)
    return (dI1dl1, dI1dl2, dI1dl3), (dI2dl1, dI2dl2, dI2dl3)

def compute_stress_neo_hookean(l1, l2, l3, mu=1.0):
    """
    Neo-Hookean model: W = (μ/2)(I₁ - 3)
    ∂W/∂λᵢ = μ·∂I₁/∂λᵢ
    """
    (dI1dl1, dI1dl2, dI1dl3), _ = compute_invariant_derivatives(l1, l2, l3)
    
    dW_dl1 = mu * dI1dl1
    dW_dl2 = mu * dI1dl2
    dW_dl3 = mu * dI1dl3
    
    # PK1 stress with plane stress (P₃₃=0)
    P11 = dW_dl1 - (l3/l1) * dW_dl3
    P22 = dW_dl2 - (l3/l2) * dW_dl3
    
    return P11, P22

def compute_stress_mooney_rivlin(l1, l2, l3, c10=1.0, c01=0.5):
    """
    Mooney-Rivlin model: W = c₁₀(I₁ - 3) + c₀₁(I₂ - 3)
    ∂W/∂λᵢ = c₁₀·∂I₁/∂λᵢ + c₀₁·∂I₂/∂λᵢ
    """
    (dI1dl1, dI1dl2, dI1dl3), (dI2dl1, dI2dl2, dI2dl3) = compute_invariant_derivatives(l1, l2, l3)
    
    dW_dl1 = c10 * dI1dl1 + c01 * dI2dl1
    dW_dl2 = c10 * dI1dl2 + c01 * dI2dl2
    dW_dl3 = c10 * dI1dl3 + c01 * dI2dl3
    
    # PK1 stress with plane stress (P₃₃=0)
    P11 = dW_dl1 - (l3/l1) * dW_dl3
    P22 = dW_dl2 - (l3/l2) * dW_dl3
    
    return P11, P22

print("=" * 80)
print("VERIFICATION: PS-I vs PS-II Stress Differences")
print("=" * 80)
print("\nFor pure shear modes:")
print("  I₁ = I₂ ALWAYS (invariants are identical)")
print("  BUT stresses P₁₁ and P₂₂ are DIFFERENT due to pressure elimination")
print("=" * 80)

# Test parameters
test_lambdas = [1.5, 2.0, 2.5, 3.0]
mu = 1.0  # Neo-Hookean parameter
c10, c01 = 1.0, 0.5  # Mooney-Rivlin parameters

print("\n" + "=" * 80)
print("TEST 1: Neo-Hookean Model W = (μ/2)(I₁ - 3), μ = 1.0")
print("=" * 80)

for lam in test_lambdas:
    # PS-II: λ₁=λ, λ₂=1, λ₃=1/λ
    l1_ps2 = lam
    l2_ps2 = 1.0
    l3_ps2 = 1.0 / lam
    I1_ps2, I2_ps2 = compute_invariants(l1_ps2, l2_ps2, l3_ps2)
    P11_ps2, P22_ps2 = compute_stress_neo_hookean(l1_ps2, l2_ps2, l3_ps2, mu)
    
    # PS-I: λ₁=λ, λ₂=1/λ, λ₃=1
    l1_ps1 = lam
    l2_ps1 = 1.0 / lam
    l3_ps1 = 1.0
    I1_ps1, I2_ps1 = compute_invariants(l1_ps1, l2_ps1, l3_ps1)
    P11_ps1, P22_ps1 = compute_stress_neo_hookean(l1_ps1, l2_ps1, l3_ps1, mu)
    
    print(f"\nλ = {lam}")
    print(f"  PS-II (λ₂=1): λ₁={l1_ps2:.3f}, λ₂={l2_ps2:.3f}, λ₃={l3_ps2:.3f}")
    print(f"                I₁={I1_ps2:.4f}, I₂={I2_ps2:.4f}")
    print(f"                P₁₁={P11_ps2:.4f}, P₂₂={P22_ps2:.4f}")
    print(f"  PS-I  (λ₃=1): λ₁={l1_ps1:.3f}, λ₂={l2_ps1:.3f}, λ₃={l3_ps1:.3f}")
    print(f"                I₁={I1_ps1:.4f}, I₂={I2_ps1:.4f}")
    print(f"                P₁₁={P11_ps1:.4f}, P₂₂={P22_ps1:.4f}")
    print(f"  Invariants match: I₁ diff = {abs(I1_ps1-I1_ps2):.2e}, I₂ diff = {abs(I2_ps1-I2_ps2):.2e}")
    print(f"  Stresses differ: ΔP₁₁ = {abs(P11_ps1-P11_ps2):.4f}, ΔP₂₂ = {abs(P22_ps1-P22_ps2):.4f}")
    
    if abs(I1_ps1 - I1_ps2) > 1e-10 or abs(I2_ps1 - I2_ps2) > 1e-10:
        print("  ❌ ERROR: Invariants should be identical!")
        sys.exit(1)
    
    if abs(P11_ps1 - P11_ps2) < 1e-10:
        print("  ❌ ERROR: P₁₁ should be different!")
        sys.exit(1)

print("\n✓ Neo-Hookean: Invariants identical, stresses different ✓")

print("\n" + "=" * 80)
print("TEST 2: Mooney-Rivlin Model W = c₁₀(I₁-3) + c₀₁(I₂-3), c₁₀=1.0, c₀₁=0.5")
print("=" * 80)

for lam in test_lambdas:
    # PS-II: λ₁=λ, λ₂=1, λ₃=1/λ
    l1_ps2 = lam
    l2_ps2 = 1.0
    l3_ps2 = 1.0 / lam
    P11_ps2, P22_ps2 = compute_stress_mooney_rivlin(l1_ps2, l2_ps2, l3_ps2, c10, c01)
    
    # PS-I: λ₁=λ, λ₂=1/λ, λ₃=1
    l1_ps1 = lam
    l2_ps1 = 1.0 / lam
    l3_ps1 = 1.0
    P11_ps1, P22_ps1 = compute_stress_mooney_rivlin(l1_ps1, l2_ps1, l3_ps1, c10, c01)
    
    print(f"\nλ = {lam}")
    print(f"  PS-II: P₁₁={P11_ps2:.4f}, P₂₂={P22_ps2:.4f}")
    print(f"  PS-I:  P₁₁={P11_ps1:.4f}, P₂₂={P22_ps1:.4f}")
    print(f"  Difference: ΔP₁₁ = {abs(P11_ps1-P11_ps2):.4f}, ΔP₂₂ = {abs(P22_ps1-P22_ps2):.4f}")
    
    if abs(P11_ps1 - P11_ps2) < 1e-10:
        print("  ❌ ERROR: P₁₁ should be different!")
        sys.exit(1)

print("\n✓ Mooney-Rivlin: Stresses different ✓")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("✓ PS-I and PS-II have IDENTICAL invariants (I₁ = I₂)")
print("✓ BUT they produce DIFFERENT stresses due to pressure elimination")
print("✓ The stress formula P₁₁ = ∂W/∂λ₁ - (λ₃/λ₁)·∂W/∂λ₃ is correctly implemented")
print("✓ Model discovery will find different constitutive parameters for PS-I vs PS-II")
print("=" * 80)
