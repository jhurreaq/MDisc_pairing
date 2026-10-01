"""Synthetic ground truths and experiment settings; coefficients are in Pa."""
from dataclasses import dataclass, field
import numpy as np
from mdisc.discovery import DiscoverySettings
from mdisc.configuration import PlotSettings

@dataclass(frozen=True)
class GroundTruthTerm:
    family: str
    coefficient_pa: float
    powers: tuple = (0, 0)
    exponent: float = 0.0

    def __post_init__(self):
        if self.family not in ('MR', 'Ogden') or not np.isfinite(self.coefficient_pa) or self.coefficient_pa < 0:
            raise ValueError('Ground-truth terms require a known family and a finite nonnegative amplitude')
        if self.family == 'MR' and (len(self.powers) != 2 or sum(self.powers) < 1 or
                                  any(not isinstance(p, int) or p < 0 for p in self.powers)):
            raise ValueError('MR powers must be two nonnegative integers with positive total degree')
        if self.family == 'Ogden' and (not np.isfinite(self.exponent) or self.exponent == 0):
            raise ValueError('An Ogden term requires a finite nonzero exponent')

@dataclass(frozen=True)
class GroundTruth:
    name: str
    coefficients_pa: tuple

    @property
    def mr(self):
        return any(term.family == 'MR' for term in self.coefficients_pa)

    @property
    def ogden(self):
        return any(term.family == 'Ogden' for term in self.coefficients_pa)

GROUND_TRUTHS = (
    GroundTruth('O2', (GroundTruthTerm('Ogden',16,exponent=-3), GroundTruthTerm('Ogden',8,exponent=3))),
    GroundTruth('MR2', (GroundTruthTerm('MR',40,powers=(1,0)), GroundTruthTerm('MR',20,powers=(0,1)))),
    GroundTruth('MR1O1', (GroundTruthTerm('MR',40,powers=(0,1)), GroundTruthTerm('Ogden',8,exponent=-3))),
    GroundTruth('MR2O2', (GroundTruthTerm('MR',40,powers=(1,0)), GroundTruthTerm('MR',20,powers=(0,1)),
                         GroundTruthTerm('Ogden',16,exponent=-3), GroundTruthTerm('Ogden',800,exponent=1))),
)

@dataclass(frozen=True)
class Configuration:
    discovery: DiscoverySettings = field(default_factory=DiscoverySettings)
    plot: PlotSettings = field(default_factory=PlotSettings)
    ground_truths: tuple = GROUND_TRUTHS
    noise_levels: tuple = (0.0, 0.05, 0.10)
    points: int = 60
    stretch_range: tuple = (0.6, 5.0)
    mr_order: int = 4
    ogden_exponents: tuple = (-10,-9,-8,-7,-6,-5,-4,-3,-1,1,3,4,5,6,7,8,9,10)

    def __post_init__(self):
        if self.points < self.discovery.folds or not (0 < self.stretch_range[0] < self.stretch_range[1]):
            raise ValueError('Synthetic points must cover the folds and stretches must be positive and ordered')
        if not self.ground_truths or not self.noise_levels or any(not (0 <= n < 1) for n in self.noise_levels):
            raise ValueError('Specify ground truths and noise levels with 0 <= noise < 1')
