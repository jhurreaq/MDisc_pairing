"""Treloar library selection; mechanics are shared with synthetic discovery."""
from mdisc.mechanics import isotropic_library

def make_library(configuration):
    return isotropic_library(mr_order=configuration.mr_order, ogden_exponents=configuration.ogden_exponents)
