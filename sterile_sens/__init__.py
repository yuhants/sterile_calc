"""Sensitivity projections for sterile-neutrino searches with levitated
nanoparticles, reproducing Carney, Leach & Moore, PRX Quantum 4, 010315 (2023).
"""

from .constants import p_sql_kev
from .detector import Detector, detector_for, DETECTOR_100NM, DETECTOR_3H, DETECTOR_35S
from .isotopes import EC_ISOTOPES, BETA_ISOTOPES, ECIsotope, BetaIsotope, ECBranch
from .montecarlo import simulate_ec, simulate_beta
from .sensitivity import sensitivity_curve, nll_upper_limit, toy_limit, get_isotope
from . import plotting
