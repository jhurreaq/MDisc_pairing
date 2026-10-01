"""Reference-volume energies and their analytical hyperelastic stresses.

F maps material to spatial coordinates. All fitted observations are perfectly
incompressible. Pressure is a constraint reaction, never a fitted intercept.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class EnergyTerm:
    index: int
    family: str
    label: str
    powers: tuple = (0, 0)
    exponent: float = 0.0
    invariant: Optional[str] = None
    activation: str = "linear"
    enabled: bool = True
    exclusion_reason: Optional[str] = None

    @property
    def term_id(self):
        return self.index + 1

    @property
    def nonlinear(self):
        return self.activation.startswith("exp")


def validate_F(F, *, incompressible=False):
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (3, 3) or not np.all(np.isfinite(F)) or np.any(np.linalg.det(F) <= 0):
        raise ValueError("F must be finite, orientation-preserving 3x3 matrices")
    if incompressible and not np.allclose(np.linalg.det(F), 1, rtol=0, atol=1e-8):
        raise ValueError("Stress model is incompressible: det(F) must equal one")
    return F


def project_pressure(hyperstress, normal):
    """Set n.sigma.n=0; rotated boundary normals preserve frame indifference."""
    normal = np.asarray(normal, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError("Plane-stress normal must be a finite nonzero vector")
    normal = normal / np.linalg.norm(normal)
    pressure = np.einsum('i,...ij,j->...', normal, hyperstress, normal)
    return hyperstress - pressure[..., None, None] * np.eye(3)


class EnergyLibrary:
    def __init__(self, terms, *, pressure_normal=(0, 0, 1)):
        self.terms = tuple(terms)
        self.active_terms = tuple(t for t in self.terms if t.enabled)
        if [t.index for t in self.terms] != list(range(len(self.terms))):
            raise ValueError("Term indices must preserve the complete ordered registry")
        self.pressure_normal = np.asarray(pressure_normal, float)

    def _coefficients(self, coefficients):
        coefficients = np.asarray(coefficients, float)
        if coefficients.shape != (len(self.active_terms),) or not np.isfinite(coefficients).all() or np.any(coefficients < 0):
            raise ValueError('Material amplitudes must be finite, nonnegative, and follow enabled term order')
        return coefficients

    def expand(self, coefficients):
        coefficients = np.asarray(coefficients, float)
        if coefficients.shape != (len(self.active_terms),):
            raise ValueError("Coefficients must follow enabled library order")
        full = np.zeros(len(self.terms))
        full[[t.index for t in self.active_terms]] = coefficients
        return full

    def compact(self, full):
        full = np.asarray(full, float)
        if full.shape != (len(self.terms),):
            raise ValueError("Full coefficient vector must follow original registry numbering")
        if any(full[t.index] != 0 for t in self.terms if not t.enabled):
            raise ValueError("Excluded terms cannot receive nonzero coefficients")
        return full[[t.index for t in self.active_terms]]

    def _term_values(self, F, nonlinear=None, *, with_derivatives=True):
        """Return unit-coefficient energies, dW/dF, and d(dW/dF)/dw1."""
        F = validate_F(F)
        nonlinear = {} if nonlinear is None else nonlinear
        allowed = {t.index for t in self.active_terms if t.nonlinear}
        if set(nonlinear) - allowed:
            raise ValueError('Shape parameters may only refer to enabled exponential terms')
        C = np.swapaxes(F, -1, -2) @ F
        I1 = np.trace(C, axis1=-2, axis2=-1)
        I2 = 0.5 * (I1 ** 2 - np.trace(C @ C, axis1=-2, axis2=-1))
        dI1 = 2 * F
        dI2 = 2 * (I1[..., None, None] * F - F @ C)
        values, gradients, shape_gradients = [], [], []
        svd = None
        for term in self.active_terms:
            if term.family == 'MR':
                i, j = term.powers
                a, b = I1 - 3.0, I2 - 3.0
                value = a ** i * b ** j
                da = i * a ** (i - 1) * b ** j if i else np.zeros_like(a)
                db = j * a ** i * b ** (j - 1) if j else np.zeros_like(b)
                gradient = da[..., None, None] * dI1 + db[..., None, None] * dI2
                shape_gradient = np.zeros_like(F)
            elif term.family == 'Ogden':
                if svd is None:
                    svd = np.linalg.svd(F)
                U, stretches, Vh = svd
                alpha = term.exponent
                value = np.sum(stretches ** alpha, axis=-1) - 3.0
                gradient = (U * (alpha * stretches ** (alpha - 1))[..., None, :]) @ Vh
                shape_gradient = np.zeros_like(F)
            else:
                inv = term.invariant
                if inv == 'I1':
                    q, dq = I1 - 3.0, dI1
                elif inv == 'I2':
                    q, dq = I2 - 3.0, dI2
                elif inv.startswith('I4'):
                    axis = {'I4f': 0, 'I4s': 1, 'I4n': 2}[inv]
                    q = np.maximum(C[..., axis, axis] - 1.0, 0.0)
                    dq = np.zeros_like(F)
                    dq[..., :, axis] = 2 * F[..., :, axis] * (C[..., axis, axis] > 1)[..., None]
                else:
                    a, b = {'I8fs': (0, 1), 'I8fn': (0, 2), 'I8sn': (1, 2)}[inv]
                    q = C[..., a, b]
                    dq = np.zeros_like(F)
                    dq[..., :, a] = F[..., :, b]
                    dq[..., :, b] = F[..., :, a]
                w1 = float(nonlinear.get(term.index, 1.0))
                if not np.isfinite(w1) or w1 <= 0:
                    raise ValueError("Exponential shape parameters must be finite and positive")
                with np.errstate(over='raise', invalid='raise'):
                    if term.activation == 'linear':
                        value, derivative, shape_derivative = q, np.ones_like(q), np.zeros_like(q)
                    elif term.activation == 'square':
                        value, derivative, shape_derivative = q ** 2, 2 * q, np.zeros_like(q)
                    elif term.activation == 'exp_linear':
                        value = np.expm1(w1 * q)
                        derivative = w1 * np.exp(w1 * q)
                        shape_derivative = (1 + w1 * q) * np.exp(w1 * q)
                    elif term.activation == 'exp_square':
                        value = np.expm1(w1 * q ** 2)
                        derivative = 2 * w1 * q * np.exp(w1 * q ** 2)
                        shape_derivative = 2 * q * (1 + w1 * q ** 2) * np.exp(w1 * q ** 2)
                    else:
                        raise ValueError(f"Unknown activation: {term.activation}")
                gradient = derivative[..., None, None] * dq
                shape_gradient = shape_derivative[..., None, None] * dq
            if not np.all(np.isfinite(value)) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError(f"Nonfinite energy or derivative for term {term.term_id}")
            values.append(value)
            if with_derivatives:
                gradients.append(gradient)
                shape_gradients.append(shape_gradient)
        if not with_derivatives:
            return np.stack(values, axis=-1), None, None
        return np.stack(values, axis=-1), np.stack(gradients, axis=-3), np.stack(shape_gradients, axis=-3)

    def energy(self, F, coefficients, nonlinear=None):
        values, _, _ = self._term_values(F, nonlinear, with_derivatives=False)
        return values @ self._coefficients(coefficients)

    def stress(self, F, coefficients, nonlinear=None, *, measure='cauchy', normal=None, pressure=None):
        F = validate_F(F, incompressible=True)
        _, gradients, _ = self._term_values(F, nonlinear)
        derivative = np.einsum('...pij,p->...ij', gradients, self._coefficients(coefficients))
        hyperstress = derivative @ np.swapaxes(F, -1, -2)
        if pressure is not None:
            if normal is not None or not np.isfinite(pressure).all():
                raise ValueError('Supply either a finite pressure or a plane-stress boundary normal')
            sigma = hyperstress - np.asarray(pressure)[..., None, None] * np.eye(3)
        else:
            sigma = project_pressure(hyperstress, self.pressure_normal if normal is None else normal)
        if measure == 'cauchy':
            return sigma
        if measure == 'nominal':
            return sigma @ np.swapaxes(np.linalg.inv(F), -1, -2)
        raise ValueError("Unknown stress measure")

    def design(self, observations, nonlinear=None, *, shape_derivative=False):
        F = observations.F
        _, gradients, shapes = self._term_values(F, nonlinear)
        gradients = shapes if shape_derivative else gradients
        hyper = gradients @ np.swapaxes(F[..., None, :, :], -1, -2)
        stress = project_pressure(hyper, self.pressure_normal)
        if observations.measure == 'nominal':
            stress = stress @ np.swapaxes(np.linalg.inv(F[..., None, :, :]), -1, -2)
        return np.stack([stress[i, :, a, b] for i, (a, b) in enumerate(observations.components)])

    def predict(self, observations, coefficients, nonlinear=None):
        return self.design(observations, nonlinear) @ self._coefficients(coefficients)


def isotropic_library(*, mr_order=3, ogden_exponents=()):
    terms = []
    for power in range(1, mr_order + 1):
        for i in range(power + 1):
            j = power - i
            left = '(I₁-3)' + (f'^{i}' if i > 1 else '') if i else ''
            right = '(I₂-3)' + (f'^{j}' if j > 1 else '') if j else ''
            terms.append(EnergyTerm(len(terms), 'MR', f'MR: {left}{right}', powers=(i, j)))
    for exponent in ogden_exponents:
        if not np.isfinite(exponent) or exponent == 0:
            raise ValueError("Ogden exponents must be finite and nonzero")
        if any(t.family == 'Ogden' and t.exponent == exponent for t in terms):
            continue
        terms.append(EnergyTerm(len(terms), 'Ogden', f'Ogden: λ^{exponent:g}', exponent=float(exponent)))
    if not terms:
        raise ValueError("An energy library must contain terms")
    return EnergyLibrary(terms)


def cardiac_library():
    terms = []
    for inv in ('I1', 'I2', 'I4f', 'I4s', 'I4n', 'I8fs', 'I8fn', 'I8sn'):
        for activation in ('linear', 'exp_linear', 'square', 'exp_square'):
            enabled = inv in ('I1', 'I2') or activation in ('square', 'exp_square')
            index = len(terms)
            shifted = f'({inv}-3)' if inv in ('I1', 'I2') else (f'⟨{inv}-1⟩₊' if inv.startswith('I4') else inv)
            argument = shifted + ('²' if activation in ('square','exp_square') else '')
            expression = f'[exp(w1_{index+1} {argument})-1]' if activation.startswith('exp') else argument
            terms.append(EnergyTerm(index, 'cardiac', f'c{index+1}{expression}',
                                   invariant=inv, activation=activation, enabled=enabled,
                                   exclusion_reason=None if enabled else 'No smooth stress-free anisotropic reference response'))
    return EnergyLibrary(terms, pressure_normal=(0, 1, 0))
