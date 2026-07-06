"""Physical constants. Energies/momenta are in keV (c = 1) unless noted."""

import numpy as np

M_E = 510.99895      # electron mass [keV]
ALPHA = 1.0 / 137.035999
N_AVOGADRO = 6.02214076e23

# SI helpers
HBAR_SI = 1.054571817e-34   # J s
C_SI = 2.99792458e8         # m/s
KEV_SI = 1.602176634e-16    # J
KEV_C_SI = KEV_SI / C_SI    # 1 keV/c in kg m/s


def p_sql_kev(mass_g: float, f_trap_hz: float) -> float:
    """Standard-quantum-limit momentum resolution, Eq. (5) of the paper.

    Delta p_SQL = sqrt(hbar * m * omega), returned in keV/c.
    (= 15 keV for a 1 fg sphere in a 100 kHz trap.)
    """
    omega = 2 * np.pi * f_trap_hz
    p_si = np.sqrt(HBAR_SI * (mass_g * 1e-3) * omega)
    return p_si / KEV_C_SI


def electron_momentum(T_kev):
    """Relativistic electron momentum [keV/c] from kinetic energy [keV]."""
    T = np.asarray(T_kev, dtype=float)
    return np.sqrt(T * (T + 2.0 * M_E))
