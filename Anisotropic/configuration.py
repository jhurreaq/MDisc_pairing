"""Cardiac experiment settings and the published four-term comparison model."""
from dataclasses import dataclass, field
from mdisc.discovery import DiscoverySettings
from mdisc.configuration import PlotSettings

@dataclass(frozen=True)
class Configuration:
    discovery: DiscoverySettings = field(default_factory=DiscoverySettings)
    plot: PlotSettings = field(default_factory=PlotSettings)
    noise_levels: tuple = (0.0, 0.05, 0.10)
    sheet: str = 'Sheet1'
    # Original zero-based indices; exported reports also provide one-based IDs.
    reference_coefficients: tuple = ((6, 5.162), (11, 0.081), (19, 0.315), (23, 0.486))
    reference_nonlinear: tuple = ((11, 21.151), (19, 4.371), (23, 0.508))

    def __post_init__(self):
        if not self.noise_levels or any(not (0 <= n < 1) for n in self.noise_levels):
            raise ValueError('Specify noise levels with 0 <= noise < 1')
