"""Select synthetic library families; all mechanics live in mdisc.mechanics."""
from mdisc.mechanics import isotropic_library

def make_library(configuration, ground_truth):
    return isotropic_library(mr_order=configuration.mr_order if ground_truth.mr else 0,
                             ogden_exponents=configuration.ogden_exponents if ground_truth.ogden else ())
