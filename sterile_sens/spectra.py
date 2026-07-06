"""Beta spectra and heavy-neutrino phase-space factors."""

import numpy as np

from .constants import M_E, ALPHA
from .isotopes import BetaIsotope, ECBranch


def fermi_function(Z_daughter, T_kev):
    """Nonrelativistic Fermi function F(Z, E) for beta- decay."""
    T = np.clip(np.asarray(T_kev, dtype=float), 1e-6, None)
    E = T + M_E
    p = np.sqrt(T * (T + 2 * M_E))
    eta = ALPHA * Z_daughter * E / p
    return 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))


def beta_spectrum(T_kev, iso: BetaIsotope, m_nu=0.0):
    """Unnormalized allowed beta spectrum dN/dT_e for neutrino mass m_nu [keV].

    dN/dT ~ F(Z,E) p_e E_e E_nu sqrt(E_nu^2 - m_nu^2),  E_nu = Q - T.
    """
    T = np.asarray(T_kev, dtype=float)
    E_nu = iso.Q - T
    out = np.zeros_like(T)
    ok = (T >= 0) & (E_nu > m_nu)
    Tn, En = T[ok], E_nu[ok]
    pe = np.sqrt(Tn * (Tn + 2 * M_E))
    Ee = Tn + M_E
    out[ok] = fermi_function(iso.Z_daughter, Tn) * pe * Ee * En * np.sqrt(En**2 - m_nu**2)
    return out


def beta_phase_space_ratio(iso: BetaIsotope, m4):
    """Gamma(m4)/Gamma(0): rate suppression of the heavy-state branch."""
    if m4 >= iso.Q:
        return 0.0
    T = np.linspace(0, iso.Q, 4000)
    return np.trapezoid(beta_spectrum(T, iso, m4), T) / \
        np.trapezoid(beta_spectrum(T, iso, 0.0), T)


def sample_beta_Te(rng, n, iso: BetaIsotope, m4=0.0):
    """Sample n electron kinetic energies from the allowed spectrum."""
    T = np.linspace(0, iso.Q - m4, 4000)
    pdf = beta_spectrum(T, iso, m4)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, T)


def ec_phase_space_ratio(branch: ECBranch, m4):
    """Two-body EC: Gamma ~ p_nu E_nu, so ratio = sqrt(1 - (m4/E_nu)^2)."""
    if m4 >= branch.E_nu:
        return 0.0
    return np.sqrt(1.0 - (m4 / branch.E_nu) ** 2)
