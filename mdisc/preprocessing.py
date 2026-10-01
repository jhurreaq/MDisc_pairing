"""Training-only loss weights and origin-preserving nondimensionalization."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class WeightPolicy:
    balance_modes: bool = True
    normalize_stress: bool = True
    rms_floor_fraction: float = 1e-6

    def __post_init__(self):
        if not (0 < self.rms_floor_fraction <= 1):
            raise ValueError('RMS floor fraction must satisfy 0 < fraction <= 1')

    def fit(self, y, modes):
        y, modes = np.asarray(y, float), np.asarray(modes)
        rms = {mode: float(np.sqrt(np.mean(y[modes == mode] ** 2))) for mode in np.unique(modes)}
        global_rms = float(np.sqrt(np.mean(np.square(list(rms.values())))))
        floor = max(global_rms * self.rms_floor_fraction, np.finfo(float).tiny ** 0.25)
        factors = {}
        for mode, value in rms.items():
            count = int(np.sum(modes == mode))
            balance = len(y) / (len(rms) * count) if self.balance_modes else 1.0
            factors[mode] = balance * (max(global_rms, floor) / max(value, floor)) ** 2 if self.normalize_stress else balance
        weights = np.array([factors[m] for m in modes])
        normalization = float(np.mean(weights))
        return FittedWeights({m: v / normalization for m, v in factors.items()}, rms)

@dataclass(frozen=True)
class FittedWeights:
    factors: dict
    rms: dict

    def transform(self, modes):
        missing = set(modes) - set(self.factors)
        if missing:
            raise ValueError(f"Validation includes modes absent from training: {sorted(missing)}")
        return np.array([self.factors[m] for m in modes], dtype=float)

def validate_weights(weights, n):
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Loss weights must be positive finite values, one per observation")
    return w

@dataclass(frozen=True)
class DesignScaling:
    columns: np.ndarray
    target: float

    @classmethod
    def fit(cls, X, y, weights):
        X, y = np.asarray(X, float), np.asarray(y, float)
        w = validate_weights(weights, len(y))
        scales = np.sqrt(np.sum(w[:, None] * X ** 2, axis=0) / np.sum(w))
        scales[scales <= np.finfo(float).tiny ** 0.25] = 1.0
        target = max(float(np.sqrt(np.sum(w * y ** 2) / np.sum(w))), np.finfo(float).tiny ** 0.25)
        return cls(scales, target)

    def transform(self, X, y, weights):
        sw = np.sqrt(validate_weights(weights, len(y)))
        return X / self.columns * sw[:, None], y / self.target * sw

    def coefficients_to_physical(self, coefficients):
        return coefficients * self.target / self.columns

    def coefficients_to_scaled(self, coefficients):
        return coefficients * self.columns / self.target
