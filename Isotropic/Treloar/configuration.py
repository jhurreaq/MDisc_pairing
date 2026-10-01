"""Explicit fitting/evaluation scopes and actual isotropic library."""
from dataclasses import dataclass, field
from mdisc.discovery import DiscoverySettings
from mdisc.configuration import PlotSettings

@dataclass(frozen=True)
class Configuration:
    discovery: DiscoverySettings = field(default_factory=DiscoverySettings)
    plot: PlotSettings = field(default_factory=PlotSettings)
    fit_datasets: tuple = ('Treloar',)
    mr_order: int = 3
    ogden_exponents: tuple = (-4, -3, -1, 1, 3, 4)

    def __post_init__(self):
        if not self.fit_datasets or len(set(self.fit_datasets)) != len(self.fit_datasets) or set(self.fit_datasets) - {'Treloar','Kawabata'}:
            raise ValueError('Choose a nonempty unique subset of Treloar and Kawabata for fitting')
